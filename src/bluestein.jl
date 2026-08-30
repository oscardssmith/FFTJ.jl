# Bluestein's algorithm: an exact length-n DFT for any n via a circular
# convolution of length N = nextsmooth(2n-1).  With the chirp
# c[j] = exp(D·iπ j²/n):  X[k] = c[k] · Σ_j (x[j]c[j]) · conj(c[k-j]),
# so X = c .* ifft_N(fft_N(x .* c, padded) .* ĝ) where ĝ is the precomputed
# transform of the wrapped kernel conj(c).  The angle is reduced as j² mod 2n
# in exact integers before cospi/sinpi, so the chirp is accurate for any n.

# Smallest efficient transform length ≥ N: small odd part on a large power of
# two, matching where the engine measures fastest (odd stages need wide
# strides; a 2-part below ~2^6 falls off a cliff).
const BLUESTEIN_ODD = (1, 3, 5, 7, 9, 15, 21)
function nextsmooth(N::Int)
    best = typemax(Int)
    for o in BLUESTEIN_ODD
        k = max(6, Base.top_set_bit(cld(N, o) - 1))
        c = o << k
        c < best && (best = c)
    end
    return best
end

function chirp(::Type{T}, n::Int, D::Int) where {T}
    c = Vector{Complex{T}}(undef, n)
    for j in 0:n-1
        t = widemul(j, j) % (2n)   # exp(iπ·t/n) has period t mod 2n
        θ = D * (Float64(t) / n)
        si, co = sincospi(θ)
        c[j+1] = Complex{T}(co, si)
    end
    return c
end

struct BluesteinPlan{T,D}
    n::Int
    c::Vector{Complex{T}}     # chirp
    ghat::Vector{Complex{T}}  # fft_N of the wrapped kernel conj(c)
    pf::FFTPlan{T,-1}
    pinv::FFTPlan{T,1}
    a::Vector{Complex{T}}     # length-N work buffers
    b::Vector{Complex{T}}
    scale::T
end

# chirp + transformed kernel are plan-independent; share them by (T, n, D)
const _BLUESTEIN_CACHE = Dict{Tuple{DataType,Int,Int},Any}()
function _bluestein_parts(::Type{T}, n::Int, D::Int) where {T}
    r = _cached(_BLUESTEIN_CACHE, (T, n, D)) do
        N = nextsmooth(2n - 1)
        c = chirp(T, n, D)
        g = zeros(Complex{T}, N)
        g[1:n] .= conj.(c)
        g[N-n+2:N] .= conj.(view(c, n:-1:2))   # kernel wraps: g[N-j+1] = conj(c[j+1])
        (N, c, _plan(T, N, -1)(g))
    end
    return r::Tuple{Int,Vector{Complex{T}},Vector{Complex{T}}}
end

function _bluestein(::Type{T}, n::Int, D::Int, scale::T=default_scale(T, n, D);
                    nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat}
    N, c, ghat = _bluestein_parts(T, n, D)
    pf = _plan(T, N, -1; nthreads)
    # pf and pinv only ever run sequentially, so they share scratch planes
    pinv = _plan(T, N, +1; bufs=(pf.are, pf.aim, pf.bre, pf.bim), nthreads)
    return BluesteinPlan{T,D}(n, c, ghat, pf, pinv,
                              Vector{Complex{T}}(undef, N),
                              Vector{Complex{T}}(undef, N), scale)
end

# the elementwise passes, over index ranges so they can be chunked across
# tasks (pure elementwise maps: chunking is trivially bitwise-safe)
function _bs_chirp_in!(a, x, c, n::Int, lo::Int, hi::Int)
    @inbounds for j in lo:min(hi, n)
        a[j] = x[j] * c[j]
    end
    @inbounds for j in max(lo, n + 1):hi
        a[j] = zero(eltype(a))
    end
    return nothing
end
function _bs_ghat!(b, ghat, lo::Int, hi::Int)
    @inbounds @simd for j in lo:hi
        b[j] *= ghat[j]
    end
    return nothing
end
function _bs_chirp_out!(y, c, a, sc, lo::Int, hi::Int)
    @inbounds for k in lo:hi
        y[k] = sc * (c[k] * a[k])
    end
    return nothing
end

function (p::BluesteinPlan{T,D})(y::CVec{T}, x::CVec{T}) where {T,D}
    n = p.n
    length(x) == n || throw(DimensionMismatch("plan length $n, input length $(length(x))"))
    length(y) == n || throw(DimensionMismatch("plan length $n, output length $(length(y))"))
    a, b, c, ghat = p.a, p.b, p.c, p.ghat
    N = length(a)
    nt = p.pf.nt
    if nt > 1
        tforeach((lo, hi) -> _bs_chirp_in!(a, x, c, n, lo, hi), N, nt)
    else
        _bs_chirp_in!(a, x, c, n, 1, N)
    end
    p.pf(b, a)
    if nt > 1
        tforeach((lo, hi) -> _bs_ghat!(b, ghat, lo, hi), N, nt)
    else
        _bs_ghat!(b, ghat, 1, N)
    end
    p.pinv(a, b)
    sc = p.scale
    if nt > 1
        tforeach((lo, hi) -> _bs_chirp_out!(y, c, a, sc, lo, hi), n, nt)
    else
        _bs_chirp_out!(y, c, a, sc, 1, n)
    end
    return y
end

(p::BluesteinPlan{T})(x::Vector{Complex{T}}) where {T} = p(similar(x), x)

plan_nt(p::BluesteinPlan) = p.pf.nt
