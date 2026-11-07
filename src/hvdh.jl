# Length-n DFT by Gaussian resampling (Harvey & van der Hoeven, "Integer
# multiplication in time O(n log n)", §4) onto a smooth t = n + Δ with
# gcd(n, t) = 1 and Δ small. With F_k normalized by 1/k, T P_n F_n = P_t F_t S,
# where S spreads the input onto the t-grid with a narrow Gaussian and T
# resamples the spectrum. Unknown x_ℓ = (F_n u)_{Δℓ mod n} sits at position
# ρℓ (ρ = t/n) of the t-grid, and grid row q holds the equation
#     z_q = (F_t S u)_{Δq mod t} = Σ_ℓ exp(-a(ρℓ - q)²) x_ℓ,   a = πα²,
# for the rows q = [ρℓ] nearest each unknown (other rows are identities).
# Indexed by q, the system is banded (|q' - q| ≤ B) and well conditioned
# however close t/n is to 1, so S's width and the band stay fixed where
# oversampled interpolation's width grows like 1/√(t/n - 1). Its inverse is
# wide, so it is solved by banded LU without pivoting (measured stable), in
# chunks of L rows with H halo rows per side (the inverse decays, so chunks
# are independent to rounding), K chunks across the lanes of a vector. The
# factors depend only on n, so they are tabulated at plan time while the
# table stays cached, and above that rebuilt per call from per-column
# exponentials. The spread writes S u permuted, (S u)_k to position Δk mod t,
# so the transform's output q is (F_t S u)_{Δq} and each chunk reads its rows
# sequentially.

# one group solve's scratch (each task owns one)
struct HvdhScratch{T}
    yr::Vector{T}           # (R + 2B)·K: right-hand side, then solution
    yi::Vector{T}
    uf::Vector{T}           # (R + B)·(B+1)·K: 1/U_ii, U_{i,i+1..i+B}
    ob::Vector{Int}         # R·K: output index of each row (-1: identity row)
    ring::Vector{T}         # 8·3·K: A', G, 1/G of the columns in reach
end

struct HvdhPlan{T,D,w,K,B,P}
    n::Int
    t::Int
    Δ::Int
    H::Int                  # halo rows per side
    L::Int                  # core rows per chunk
    G::Int                  # groups of K chunks
    a::T
    α::Float64              # spreading width (a = πα²)
    rc::NTuple{7,T}         # column-recurrence factors, see _hvdh_step
    rsm::Int                # re-seed the recurrence every rsm + 1 columns
    srs::Int                # re-seed the spread's weights every srs vectors
    p::P                    # length-t transform, unnormalized
    ch::Vector{T}           # ch[d] = exp(-a d²)
    se::Vector{Int}         # per chunk, at its first column (q = start - H - B):
                            # e = 2tℓ' - 2nq + n for the next unknown ℓ' at or after q
    sx::Vector{Int}         # per chunk, at its first row: q mod t and Δℓ' mod n
    so::Vector{Int}
    ure::Vector{T}          # deinterleaved input
    uim::Vector{T}
    v::Vector{T}            # 2t, interleaved: spread input, permuted
    Y::Vector{T}            # 2t, interleaved: its transform, row q holds (F_t S u)_{Δq}
    ws::Vector{HvdhScratch{T}}   # per-task solve scratch
    lt::Vector{T}           # precomputed factors (empty: built per call): per group,
    ut::Vector{T}           # multipliers (R·B·K), uf's rows ((R+B)·(B+1)·K)
    ot::Vector{Int}         # and ob (R·K)
    nt::Int
    scale::T
end

plan_nt(p::HvdhPlan) = p.nt

# spreading width α: the smallest that gives band 2, which pays for its extra
# taps (F64: 12, F32: 6) with a halved halo; larger α measured no faster
_hvdh_α(::Type{T}) where {T} = sizeof(T) ≥ 8 ? 1.45 : 1.0
# smallest t/n - 1 sought; the halo grows like its inverse square root
const _HVDH_θMIN = 1 / 128
const _HVDH_LTARGET = 256
# columns between exact re-seeds of the column recurrence (a power of two);
# A' drifts quadratically, so the error grows past 4 (2.7e-15 at 16 in F64)
const _HVDH_RESEED = 4
# vectors between exact re-seeds of the spread's weights (a power of two)
const _HVDH_SRESEED = 8
# Largest precomputed solve-factor table: lookup beats rebuilding the factors
# per call while it stays cached (measured parity between 6 and 10 MB).
const _HVDH_TABLE_MAX = 8 << 20

# t coprime to n: for odd n the smallest power-of-two-rich length (as
# nextsmooth, without its floor), for even n the smallest odd smooth one
function _hvdh_t(n::Int)
    N = ceil(Int, n * (1 + _HVDH_θMIN))
    t = typemax(Int)
    if isodd(n)
        for o in SMOOTH_ODD
            gcd(o, n) == 1 && (t = min(t, o << Base.top_set_bit(cld(N, o) - 1)))
        end
    else
        for m in N|1:2:2n-1
            factor_smooth(m) !== nothing && gcd(m, n) == 1 && (t = m; break)
        end
    end
    return t - n < n ? t : nothing
end

# Odd n only: even n force an odd t, whose transform is slow. Measured
# crossovers vs ResamplePlan (F64 ~1k; F32 near parity from 13k to 30k,
# winning from 40k).
_use_hvdh(::Type{T}, n::Int) where {T} = isodd(n) && n ≥ (sizeof(T) ≥ 8 ? 1000 : 16384)

# exp for |x| ≲ 10: Cody–Waite reduction by ln 2 and a Taylor polynomial
const _EXP_C64 = ntuple(j -> 1 / factorial(j - 1), 14)
const _EXP_C32 = ntuple(j -> Float32(1 / factorial(j - 1)), 8)
@inline _horner(r, c::Tuple{Any}) = typeof(r)(c[1])
@inline _horner(r, c::Tuple) = muladd(_horner(r, Base.tail(c)), r, typeof(r)(c[1]))
@inline function _vexp(x::Vec{K,T}) where {K,T}
    IT, bias, mb, hi, lo, C = T === Float64 ?
        (Int64, 1023, 52, 6.93147180369123816490e-01, 1.90821492927058770002e-10, _EXP_C64) :
        (Int32, Int32(127), Int32(23), 0.693145751953125f0, 1.428606765330187045f-6, _EXP_C32)
    kf = round(x * T(1.4426950408889634))
    r = muladd(kf, -T(hi), x)
    r = muladd(kf, -T(lo), r)
    k = convert(Vec{K,IT}, kf)
    return _horner(r, C) * reinterpret(Vec{K,T}, (k + bias) << mb)
end

# halo: a solution row depends on right-hand-side rows through a row of the
# inverse, which decays; probe a few windows of the ℓ-indexed system densely,
# including the slowest-decaying ones, around β ≈ ±1/2 (Δℓ ≡ n/2 mod n)
function _hvdh_halo(n, t, a, B, tol)
    ρ = t / n
    c = invmod(t - n, n)
    probes = [round.(Int, range(0, n - 1; length=min(n, 16)));
              [mod(h * c, n) for h in (n ÷ 2, n ÷ 2 + 1)]]
    P = 64
    while true
        H = 0
        for ℓ0 in probes
            m = 2P + 1
            M = zeros(m, m)
            for i in 1:m, j in max(1, i - B - 1):min(m, i + B + 1)
                # row i: grid row q_i = [ρℓ_i]; column j: unknown at ρℓ_j
                ℓi, ℓj = ℓ0 - P + i - 1, ℓ0 - P + j - 1
                M[i, j] = exp(-a * (ρ * ℓj - fld(2t * ℓi + n, 2n))^2)
            end
            e = zeros(m)
            e[P+1] = 1
            r = abs.(transpose(M) \ e)
            mx = maximum(r)
            for j in 1:m
                r[j] > tol * mx && (H = max(H, abs(j - P - 1)))
            end
        end
        H < P - 2B && return ceil(Int, ρ * H) + B
        P *= 2
        P > 8192 && throw(ArgumentError("HvdH halo did not converge for n = $n, t = $t"))
    end
end

# 2tℓ' - 2nq + n for the first unknown ℓ' with ρℓ' ≥ q - 1/2, and ℓ'
function _hvdh_e(n, t, q)
    ℓ = cld(2n * q - n, 2t)
    return 2t * ℓ - 2n * q + n, ℓ
end

function _hvdh(::Type{T}, n::Int, D::Int, scale::T=default_scale(T, n, D);
               K::Int=pick_width(T), reseed::Int=_HVDH_RESEED, Lt::Int=_HVDH_LTARGET,
               sreseed::Int=_HVDH_SRESEED, α::Real=_hvdh_α(T),
               factors::Symbol=:auto, nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat}
    ispow2(reseed) || throw(ArgumentError("reseed must be a power of two (got $reseed)"))
    ispow2(sreseed) || throw(ArgumentError("sreseed must be a power of two (got $sreseed)"))
    t = _hvdh_t(n)
    t === nothing && throw(ArgumentError("no coprime smooth t for n = $n"))
    Tw = promote_type(T, Float64)
    Δ = t - n
    α = Tw(α)
    a = π * α^2
    tol = eps(T) / 4
    # S: Gaussian truncated where it drops below tol
    h = ceil(Int, α * sqrt(log(1 / Tw(tol)) / π))
    w = 2h
    # band: entries exp(-a(d + β')²), |β'| ≤ 1/2, below tol beyond |d| = B
    B = floor(Int, sqrt(log(1 / Tw(tol)) / a) + 1 / 2)
    2B + 1 ≤ 8 || throw(ArgumentError("band $B too wide for the column ring"))
    H = _hvdh_halo(n, t, a, B, Tw(tol))
    G = max(1, cld(t, K * Lt))
    L = cld(t, K * G)
    R = L + 2H
    se, sx, so = (Vector{Int}(undef, G * K) for _ in 1:3)
    for c in 0:G*K-1
        q0 = mod(c * L - H, t)    # first row
        se[c+1] = first(_hvdh_e(n, t, q0 - B))
        sx[c+1] = q0
        so[c+1] = Int(mod(widemul(Δ, last(_hvdh_e(n, t, q0))), n))
    end
    nt = plan_tasks(n, nthreads)
    pt = _anyplan(T, t, D, one(T); nthreads=nt)
    function scratch()
        uf = zeros(T, (R + B) * (B + 1) * K)
        for i in 0:B-1, lane in 1:K   # leading pad rows: identity
            uf[i*(B+1)*K+lane] = one(T)
        end
        return HvdhScratch{T}(zeros(T, (R + 2B) * K), zeros(T, (R + 2B) * K), uf,
                              Vector{Int}(undef, R * K), zeros(T, 8 * 3 * K))
    end
    ws = [scratch() for _ in 1:min(nt, G)]
    uf = ws[1].uf
    δ = Tw(Δ) / n
    rc = T.((exp(2a), exp(-2a), exp(-a), exp(2a * δ), exp(-2a * δ), exp(-a * δ^2),
             exp(-2a * δ^2)))
    factors in (:auto, :table, :fly) ||
        throw(ArgumentError("factors must be :auto, :table or :fly (got $factors)"))
    tbytes = G * (R * B * K * sizeof(T) + length(uf) * sizeof(T) + R * K * sizeof(Int))
    tab = factors === :table || (factors === :auto && tbytes ≤ _HVDH_TABLE_MAX)
    p = HvdhPlan{T,D,w,K,B,typeof(pt)}(
        n, t, Δ, H, L, G, T(a), Float64(α), rc, reseed - 1, sreseed, pt, T[exp(-a * d^2) for d in 1:B],
        se, sx, so, Vector{T}(undef, n + Δ + pick_width(T)), Vector{T}(undef, n + Δ + pick_width(T)),
        Vector{T}(undef, 2t), zeros(T, 2t), ws,
        Vector{T}(undef, tab ? G * R * B * K : 0), Vector{T}(undef, tab ? G * length(uf) : 0),
        Vector{Int}(undef, tab ? G * R * K : 0), nt, scale)
    if tab
        lf = Vector{T}(undef, R * B * K)
        sc = ws[1]
        for g in 0:G-1
            _hvdh_factor!(p, sc, g, lf, uf, sc.ob)
            copyto!(p.lt, g * length(lf) + 1, lf, 1, length(lf))
            copyto!(p.ut, g * length(uf) + 1, uf, 1, length(uf))
            copyto!(p.ot, g * R * K + 1, sc.ob, 1, R * K)
        end
    end
    return p
end

function (p::HvdhPlan{T,D,w})(y::CVec{T}, x::CVec{T}) where {T,D,w}
    n = p.n
    length(x) == n || throw(DimensionMismatch("plan length $n, input length $(length(x))"))
    length(y) == n || throw(DimensionMismatch("plan length $n, output length $(length(y))"))
    nt = p.nt
    _hvdh_scatter!(p.ure, p.uim, x, n, p.Δ, nt)
    _hvdh_spread!(p.v, p.ure, p.uim, n, p.t, p.Δ, Val(w), p.α, p.srs, nt)
    p.p(reinterpret(Complex{T}, p.Y), reinterpret(Complex{T}, p.v))
    GC.@preserve y begin
        py = _rawptr(y)
        if nt > 1
            # groups are independent; each task solves on its own scratch
            tforeach_chunks((ci, lo, hi) -> _hvdh_groups!(p, p.ws[ci], py, lo - 1, hi - 1),
                            p.G, nt)
        else
            _hvdh_groups!(p, p.ws[1], py, 0, p.G - 1)
        end
    end
    return y
end

# a serial copy of p for concurrent use: shares the read-only tables (factor
# tables, seeds, stage chains), owns fresh working buffers
function _hvdh_serial_clone(p::HvdhPlan{T,D,w,K,B,P}) where {T,D,w,K,B,P}
    s0 = p.ws[1]
    sc = HvdhScratch{T}(zero(s0.yr), zero(s0.yi), copy(s0.uf), similar(s0.ob), zero(s0.ring))
    pt = _anyplan(T, p.t, D, one(T); nthreads=1)::P
    return HvdhPlan{T,D,w,K,B,P}(p.n, p.t, p.Δ, p.H, p.L, p.G, p.a, p.α, p.rc, p.rsm, p.srs,
                                 pt, p.ch, p.se, p.sx, p.so, similar(p.ure), similar(p.uim),
                                 similar(p.v), zero(p.Y), [sc], p.lt, p.ut, p.ot, 1, p.scale)
end

function _hvdh_groups!(p::HvdhPlan{T}, sc::HvdhScratch{T}, py::Ptr{T}, glo::Int, ghi::Int) where {T}
    tab = !isempty(p.lt)
    for g in glo:ghi
        tab ? _hvdh_subst!(p, sc, py, g) : _hvdh_group!(p, sc, py, g)
    end
    return nothing
end

(p::HvdhPlan{T})(x::Vector{Complex{T}}) where {T} = p(similar(x), x)

# S u in transform order, v[q] = (S u)_k for q = Δk mod t. Output k sits at
# kn/t = s + δ with kn ≡ -q (mod t), so δ = (t - q)/t ∈ (0, 1], and its tap at
# grid s + o is u_l with lΔ ≡ q + (o - 1)Δ (mod n). Stored permuted,
# y[lΔ mod n] = u_l, each tap is a contiguous run in q, so consecutive
# outputs fill a vector's lanes. Writing r = iΔ + j (j < Δ) gives l = i + jc,
# c = Δ⁻¹ mod n, so an S×S tile of (i, j) reads S runs of x along i and writes
# S runs of y along j, transposed in registers; rows r ≥ ⌊n/Δ⌋Δ and the last
# Δ mod S columns go singly.
function _hvdh_scatter!(yr::Vector{T}, yi::Vector{T}, x, n, Δ, nt::Int=1) where {T}
    S = 8
    c = invmod(Δ, n)
    I = n ÷ Δ
    J8 = Δ - Δ % S
    wrap(l) = l - ifelse(l ≥ n, n, 0)
    GC.@preserve x begin
        px = _rawptr(x)
        # S×S tiles, by column block; tiles write disjoint columns
        if nt > 1
            tforeach((lo, hi) -> _hvdh_tiles!(yr, yi, px, x, n, Δ, c, I, lo - 1, hi - 1),
                     J8 ÷ S, nt)
        else
            _hvdh_tiles!(yr, yi, px, x, n, Δ, c, I, 0, J8 ÷ S - 1)
        end
        l0 = mod(J8 * c, n)
        @inbounds for j in J8:Δ-1     # leftover columns
            l = l0
            for i in 0:I-1
                yr[i*Δ+j+1] = real(x[l+1])
                yi[i*Δ+j+1] = imag(x[l+1])
                l = wrap(l + 1)
            end
            l0 = wrap(l0 + c)
        end
        l = mod(I * Δ * c, n)
        @inbounds for r in I*Δ+1:n      # leftover rows
            yr[r] = real(x[l+1])
            yi[r] = imag(x[l+1])
            l = wrap(l + c)
        end
    end
    @inbounds for j in n+1:length(yr)   # periodic pad
        yr[j] = yr[j-n]
        yi[j] = yi[j-n]
    end
    return nothing
end

# column blocks blo..bhi of _hvdh_scatter!'s tiles; block b's first column
# reads x from l = (bS + j)·c mod n, exactly as advancing from block 0
function _hvdh_tiles!(yr::Vector{T}, yi::Vector{T}, px::Ptr{T}, x, n, Δ, c, I,
                      blo::Int, bhi::Int) where {T}
    S = 8
    wrap(l) = l - ifelse(l ≥ n, n, 0)
    @inbounds for b in blo:bhi
        jb = S * b
        ls = ntuple(j -> Int(mod(widemul(jb + j - 1, c), n)), Val(S))
        for ib in 0:S:I-1
            rs = map(@inline(l -> _hvdh_ldc(px, x, l, n, Val(S))), ls)
            re = vtranspose(map(first, rs))
            im = vtranspose(map(last, rs))
            for i in 1:S
                ib + i ≤ I || break
                r = (ib + i - 1) * Δ + jb + 1
                vstore(re[i], yr, r)
                vstore(im[i], yi, r)
            end
            ls = map(@inline(l -> wrap(l + S)), ls)
        end
    end
    return nothing
end

# x[l+1 : l+S] deinterleaved, cyclically
@inline function _hvdh_ldc(px::Ptr{T}, x, l, n, ::Val{S}) where {T,S}
    if l + S ≤ n
        sz = sizeof(T)
        return deinterleave2(vload(Vec{S,T}, px + 2sz * l), vload(Vec{S,T}, px + sz * (2l + S)))
    end
    return (Vec{S,T}(ntuple(i -> @inbounds(real(x[mod(l + i - 1, n)+1])), Val(S))),
            Vec{S,T}(ntuple(i -> @inbounds(imag(x[mod(l + i - 1, n)+1])), Val(S))))
end

# Weight exp(-b(δ - o)²)/α = E·Gᵒ·exp(-bo²) with E = exp(-bδ²)/α, G = exp(2bδ).
# Outputs go in J rows of q = iΔ + j at a time, V consecutive j per vector:
# tap o of row i reads y[(i + o - 1)Δ + j], so the J rows share their loads,
# and row g's weights follow row 0's by constants and H = exp(2bθδ),
# θ = Δ/t. Along j, δ falls by V/t per vector, so the states advance by
# multiplication, re-seeded exactly every rs vectors. y is padded by Δ + V,
# so a row's taps never wrap.
function _hvdh_spread!(v::Vector{T}, yr::Vector{T}, yi::Vector{T}, n, t, Δ,
                       ::Val{w}, α, rs, nt::Int=1, ::Val{J}=Val(8)) where {T,w,J}
    V = pick_width(T)
    hw = w ÷ 2
    b = π / Float64(α)^2
    θ = Δ / t
    κ = V / t
    fs = (T(exp(-2b * κ^2)), T(exp(-2b * κ)), T(exp(2b * κ)), T(exp(-2b * θ * κ)))
    kc = ntuple(i -> T(exp(-b * (i - hw)^2)), Val(w))    # tap o = i - hw
    gc = (ntuple(g -> T(exp(-b * ((g - 1) * θ)^2)), Val(J)),
          ntuple(g -> T(exp(-2b * (g - 1) * θ)), Val(J)),
          ntuple(g -> T(exp(2b * (g - 1) * θ)), Val(J)))
    # work units: (row block, run of rs vectors along j); each unit starts on
    # a re-seed, so any split reproduces the serial values exactly
    nseg = cld(Δ, rs * V)
    nu = cld(cld(t, Δ), J) * nseg
    if nt > 1
        tforeach((lo, hi) -> _hvdh_spread_units!(v, yr, yi, n, t, Δ, Val(w), α, rs, Val(J),
                                                 b, θ, κ, fs, kc, gc, nseg, lo - 1, hi - 1),
                 nu, nt)
    else
        _hvdh_spread_units!(v, yr, yi, n, t, Δ, Val(w), α, rs, Val(J),
                            b, θ, κ, fs, kc, gc, nseg, 0, nu - 1)
    end
    return nothing
end

function _hvdh_spread_units!(v::Vector{T}, yr::Vector{T}, yi::Vector{T}, n, t, Δ,
                             ::Val{w}, α, rs, ::Val{J}, b, θ, κ, (fE, fG, fGi, fH), kc, gc,
                             nseg, ulo, uhi) where {T,w,J}
    V = pick_width(T)
    VT = Vec{V,T}
    hw = w ÷ 2
    js = Vec{V,Int}(ntuple(j -> j - 1, Val(V)))
    @inbounds for u in ulo:uhi
        ib, seg = divrem(u, nseg)
        i0 = J * ib
        bs = ntuple(m -> mod((i0 + m - 1 - hw) * Δ, n) + 1, Val(J + w - 1))
        lim = ntuple(g -> clamp(t - (i0 + g - 1) * Δ, 0, Δ), Val(J))
        E = G = Gi = R = H = zero(VT)
        j = seg * rs * V
        jend = min(Δ, j + rs * V)
        blk = 0
        while j < jend
            if blk & (rs - 1) == 0
                δ = convert(VT, (t - i0 * Δ - j) - js) * T(1 / t)
                E = _vexp(-T(b) * δ * δ) * T(1 / α)
                G = _vexp(T(2b) * δ)
                Gi = _vexp(-T(2b) * δ)
                R = _vexp(T(2b * κ) * δ - T(b * κ^2))
                H = _vexp(T(2b * θ) * δ)
            end
            _hvdh_sblock!(v, i0 * Δ + j, Δ, lim, yr, yi, bs, j, E, G, Gi, H, kc, gc,
                          Val(J), Val(w))
            E *= R
            R *= fE
            G *= fG
            Gi *= fGi
            H *= fH
            j += V
            blk += 1
        end
    end
    return nothing
end

# J rows × V lanes, row g's outputs to v[q + gΔ ...] (row g, tap o reads
# y[bs[g + o + hw] + j]); lanes at or past lim[g] - j are not stored
@inline function _hvdh_sstore!(v::Vector{T}, q, lv, re::Vec{V,T}, im::Vec{V,T}) where {T,V}
    @inbounds if lv ≥ V
        lo, hi = interleave_chunks(re, im, Val(1))
        vstore(lo, v, 2q + 1)
        vstore(hi, v, 2q + V + 1)
    else
        for s in 0:lv-1
            v[2(q+s)+1] = re[s+1]
            v[2(q+s)+2] = im[s+1]
        end
    end
end
@generated function _hvdh_sblock!(v, q, Δ, lim, yr::Vector{T}, yi::Vector{T}, bs, j,
                                  E::Vec{V,T}, G, Gi, H, kc, gc, ::Val{J}, ::Val{w}) where {T,V,J,w}
    hw = w ÷ 2
    body = Expr[:(F = E)]
    for g in 0:J-1
        re, im = Symbol(:re, g), Symbol(:im, g)
        g > 0 && push!(body, :(F *= H))
        push!(body, quote
            Eg = F * gc[1][$(g + 1)]
            Gg = G * gc[2][$(g + 1)]
            Gig = Gi * gc[3][$(g + 1)]
            $re = Eg * vload(Vec{V,T}, yr, bs[$(g + hw)] + j)
            $im = Eg * vload(Vec{V,T}, yi, bs[$(g + hw)] + j)
        end)
        for (step, os) in ((:Gg, 1:hw), (:Gig, -1:-1:1-hw))
            push!(body, :(P = Eg))
            for o in os
                m = g + o + hw
                push!(body, quote
                    P *= $step
                    W = P * kc[$(o + hw)]
                    $re = muladd(W, vload(Vec{V,T}, yr, bs[$m] + j), $re)
                    $im = muladd(W, vload(Vec{V,T}, yi, bs[$m] + j), $im)
                end)
            end
        end
        push!(body, :(_hvdh_sstore!(v, q + $g * Δ, lim[$(g + 1)] - j, $re, $im)))
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(body...)
        end
        return nothing
    end
end

# Column state at e, with β' = (e - n)/2n and δ = Δ/n:
# (G, 1/G, A', Hh) = (exp(-2aβ'), exp(2aβ'), exp(-aβ'²), exp(-2aβ'δ)), exact
@inline function _hvdh_seed(e::Vec{K,Int}, n, a::T, δ::T) where {K,T}
    VT = Vec{K,T}
    β = convert(VT, e - n) * T(1 / (2n))
    g = _vexp(T(-2) * a * β)
    return (g, one(VT) / g, _vexp(-a * β * β), _vexp(T(-2) * a * δ * β))
end

# advance the state one column: off the unknowns e -= 2n (β' -= 1), on one
# e += 2Δ (β' += δ); each factor is a constant, or the state itself for A'
@inline function _hvdh_step(e::Vec{K,Int}, (g, gi, A, hh), rc::NTuple{7,T},
                            n, Δ) where {K,T}
    VT = Vec{K,T}
    e2a, em2a, ema, e2ad, em2ad, emad2, em2ad2 = rc   # e2ad = exp(2aδ), ...
    hit = e < 2n
    # hit is per lane, so a lane select (which needs one vector branch)
    A *= vifelse(hit, hh * emad2, gi * ema)
    g *= vifelse(hit, VT(em2ad), e2a)
    gi *= vifelse(hit, VT(e2ad), em2a)
    hh *= vifelse(hit, VT(em2ad2), e2ad)
    return vifelse(hit, e + 2Δ, e - 2n), (g, gi, A, hh)
end

# store a column in ring slot (column mod 8): A' (0 off the unknowns), G, 1/G
@inline function _hvdh_col!(ring, slot, e::Vec{K,Int}, (g, gi, A, _), n) where {K}
    o = slot * 3 * K + 1
    @inbounds begin
        vstore(vifelse(e < 2n, A, zero(A)), ring, o)
        vstore(g, ring, o + K)
        vstore(gi, ring, o + 2K)
    end
    return nothing
end

# prime the column ring with group g's columns -B .. B-1 (relative to row 0);
# returns the column state after them
@inline function _hvdh_prime!(p::HvdhPlan{T,D,w,K,B}, ring, g) where {T,D,w,K,B}
    n, Δ = p.n, p.Δ
    e = @inbounds vload(Vec{K,Int}, p.se, g * K + 1)
    cs = _hvdh_seed(e, n, p.a, T(Δ / n))
    for c in -B:B-1
        _hvdh_col!(ring, c & 7, e, cs, n)
        e, cs = _hvdh_step(e, cs, p.rc, n, Δ)
    end
    return e, cs
end

# band row i from the ring (ch[d] = exp(-a d²)), eliminated against rows
# i-B .. i-1 of uf, with its U row stored to uf; returns the advanced column
# state and next output index on, row i's output index (-1: identity row),
# and the multipliers f[k] of rows i-k
@inline function _hvdh_row!(p::HvdhPlan{T,D,w,K,B}, ring, i, e, cs, on, uf, ch) where {T,D,w,K,B}
    n, Δ = p.n, p.Δ
    VT = Vec{K,T}
    S = B + 1   # uf row stride, in vectors
    @inbounds begin
        (i + 2B) & p.rsm == 0 && (cs = _hvdh_seed(e, n, p.a, T(Δ / n)))
        _hvdh_col!(ring, (i + B) & 7, e, cs, n)
        e, cs = _hvdh_step(e, cs, p.rc, n, Δ)
        Aq = vload(VT, ring, (i & 7) * 3 * K + 1)
        hit = Aq > zero(T)
        # r[m + B + 1] = entry at column i + m
        r = ntuple(Val(2B + 1)) do m1
            @inline
            m = m1 - B - 1
            m == 0 && return vifelse(hit, Aq, one(T))
            o = ((i + m) & 7) * 3 * K + 1
            gc = vload(VT, ring, o + (m > 0 ? K : 2K))
            gm = gc
            for _ in 2:abs(m)
                gm *= gc
            end
            vifelse(hit, ch[abs(m)] * vload(VT, ring, o) * gm, zero(T))
        end
        f = ntuple(_ -> zero(VT), Val(B))
        for k in B:-1:1
            rb = (i - k + B) * S * K + 1
            fk = r[B+1-k] * vload(VT, uf, rb)
            f = Base.setindex(f, fk, k)
            for j in 1:B
                r = Base.setindex(r, muladd(-fk, vload(VT, uf, rb + j * K), r[B+1-k+j]),
                                  B + 1 - k + j)
            end
        end
        rb = (i + B) * S * K + 1
        vstore(one(VT) / r[B+1], uf, rb)
        for j in 1:B
            vstore(r[B+1+j], uf, rb + j * K)
        end
    end
    oq = vifelse(hit, on, -1)
    on = vifelse(hit, on + Δ, on)
    return e, cs, vifelse(on ≥ n, on - n, on), oq, f
end

# one group of K chunks, factoring as it goes: each row is built and
# eliminated while its right-hand side is gathered and reduced (the two
# overlap), then the back pass
function _hvdh_group!(p::HvdhPlan{T,D,w,K,B}, sc::HvdhScratch{T}, py::Ptr{T},
                      g::Int) where {T,D,w,K,B}
    t, R = p.t, p.L + 2p.H
    VT, VI = Vec{K,T}, Vec{K,Int}
    yr, yi, ob, uf, ring, Y = sc.yr, sc.yi, sc.ob, sc.uf, sc.ring, p.Y
    ch = ntuple(d -> p.ch[d], Val(B))
    e, cs = _hvdh_prime!(p, ring, g)
    @inbounds begin
        idx = vload(VI, p.sx, g * K + 1)
        on = vload(VI, p.so, g * K + 1)
        for i in 0:R-1
            e, cs, on, oq, f = _hvdh_row!(p, ring, i, e, cs, on, uf, ch)
            vstore(oq, ob, i * K + 1)
            hit = oq ≥ 0
            ar = vifelse(hit, vgather(Y, 2idx + 1), zero(T))
            ai = vifelse(hit, vgather(Y, 2idx + 2), zero(T))
            for k in B:-1:1           # previous rows i-B, ..., i-1
                yo = (i - k + B) * K + 1
                ar = muladd(-f[k], vload(VT, yr, yo), ar)
                ai = muladd(-f[k], vload(VT, yi, yo), ai)
            end
            vstore(ar, yr, (i + B) * K + 1)
            vstore(ai, yi, (i + B) * K + 1)
            idx += 1
            idx = vifelse(idx ≥ t, idx - t, idx)
        end
    end
    _hvdh_back!(p, sc, py, uf, 0, ob, 0)
    return nothing
end

# back-substitute one group's rows from U (uf[uo+1 ...], row stride (B+1)·K)
# and scatter the core rows to their outputs (ob[oo+1 ...], -1: none)
@inline function _hvdh_back!(p::HvdhPlan{T,D,w,K,B}, sc::HvdhScratch{T}, py::Ptr{T},
                             uf, uo, ob, oo) where {T,D,w,K,B}
    n, t, H, L = p.n, p.t, p.H, p.L
    R = L + 2H
    VT, VI = Vec{K,T}, Vec{K,Int}
    sz = sizeof(T)
    yr, yi = sc.yr, sc.yi
    S = B + 1
    sc = p.scale * T(n) / t
    @inbounds for i in R-1:-1:H
        rb = uo + (i + B) * S * K + 1
        yo = (i + B) * K + 1
        xr, xi = vload(VT, yr, yo), vload(VT, yi, yo)
        for j in 1:B
            u = vload(VT, uf, rb + j * K)
            xr = muladd(-u, vload(VT, yr, yo + j * K), xr)
            xi = muladd(-u, vload(VT, yi, yo + j * K), xi)
        end
        iu = vload(VT, uf, rb)
        xr *= iu
        xi *= iu
        vstore(xr, yr, yo)
        vstore(xi, yi, yo)
        if i < H + L
            o = vload(VI, ob, oo + i * K + 1)
            m = o ≥ 0
            pyo = py + max(o, VI(0)) * (2sz)
            vscatter(xr * sc, pyo, m)
            vscatter(xi * sc, pyo + sz, m)
        end
    end
    return nothing
end

# factor group g's band without data (for the tables): multipliers to lf,
# 1/U_ii and U_{i,i+1..i+B} to uf, each row's output index (-1: none) to ob
function _hvdh_factor!(p::HvdhPlan{T,D,w,K,B}, sc::HvdhScratch{T}, g::Int,
                       lf, uf, ob) where {T,D,w,K,B}
    R = p.L + 2p.H
    ch = ntuple(d -> p.ch[d], Val(B))
    e, cs = _hvdh_prime!(p, sc.ring, g)
    @inbounds begin
        on = vload(Vec{K,Int}, p.so, g * K + 1)
        for i in 0:R-1
            e, cs, on, oq, f = _hvdh_row!(p, sc.ring, i, e, cs, on, uf, ch)
            vstore(oq, ob, i * K + 1)
            for k in 1:B
                vstore(f[k], lf, (i * B + B - k) * K + 1)
            end
        end
    end
    return nothing
end

# one group from the precomputed tables: gather the rows, forward-substitute
# with the multipliers, back pass
function _hvdh_subst!(p::HvdhPlan{T,D,w,K,B}, sc::HvdhScratch{T}, py::Ptr{T},
                      g::Int) where {T,D,w,K,B}
    t, R = p.t, p.L + 2p.H
    VT, VI = Vec{K,T}, Vec{K,Int}
    yr, yi, Y, lt, ot = sc.yr, sc.yi, p.Y, p.lt, p.ot
    lo, oo = g * R * B * K, g * R * K
    @inbounds begin
        idx = vload(VI, p.sx, g * K + 1)
        for i in 0:R-1
            hit = vload(VI, ot, oo + i * K + 1) ≥ 0
            ar = vifelse(hit, vgather(Y, 2idx + 1), zero(T))
            ai = vifelse(hit, vgather(Y, 2idx + 2), zero(T))
            for k in B:-1:1
                f = vload(VT, lt, lo + (i * B + B - k) * K + 1)
                yo = (i - k + B) * K + 1
                ar = muladd(-f, vload(VT, yr, yo), ar)
                ai = muladd(-f, vload(VT, yi, yo), ai)
            end
            vstore(ar, yr, (i + B) * K + 1)
            vstore(ai, yi, (i + B) * K + 1)
            idx += 1
            idx = vifelse(idx ≥ t, idx - t, idx)
        end
    end
    _hvdh_back!(p, sc, py, p.ut, g * length(sc.uf), ot, oo)
    return nothing
end

_hvdh_w(::HvdhPlan{T,D,w}) where {T,D,w} = w

# Batched HvdH: length-n transforms of B lanes (row j, lane i at B·j + i), in
# place, as RsPass. The lanes share the spread weights and the band, so the
# weights are a table and the band is factored once, as one cyclic chunk with
# scalar factors, while the lanes are the vector dimension. Spread output q's
# taps are the w input rows from qc - hw (c = Δ⁻¹ mod n): _hvdh_spread!'s
# y[(q + (o-1)Δ) mod n] with y[lΔ mod n] = u_l.
struct HvPass{T}
    n::Int
    t::Int
    w::Int
    nb::Int                 # band
    H::Int                  # halo rows per side
    R::Int                  # chunk rows, t + 2H
    B::Int                  # lanes per row
    C::Int                  # lanes per chunk
    wt::Vector{T}           # t·w spread weights
    l0::Vector{Int}         # each spread output's first input row
    cf::Vector{T}           # per chunk row: multipliers of rows i-nb .. i-1,
                            # 1/U_ii, U_{i,i+1..i+nb}
    oi::Vector{Int}         # per chunk row: output row (-1: identity row)
    q0::Int                 # transform row of chunk row 0
    sc::T                   # n/t
    stages::Vector{Stage{T}}
    ro::Vector{Int}         # w tap offsets, scratch
    are::Vector{T}          # chunk scratch: input (n rows)
    aim::Vector{T}
    pre::Vector{T}          # spread and transform, then back-substitution
    pim::Vector{T}          # (max(t, R + 2nb) rows)
    qre::Vector{T}
    qim::Vector{T}
    zr::Vector{T}           # forward substitution (R + 2nb rows)
    zi::Vector{T}
end

const _HV_PASS_CACHE = Dict{Tuple{DataType,Int,Int},Any}()

function _hv_parts(::Type{T}, n::Int, D::Int) where {T}
    sp = _hvdh(T, n, D, one(T); K=1, Lt=typemax(Int) ÷ 2, factors=:table)
    t, Δ, H, w, nb = sp.t, sp.Δ, sp.H, _hvdh_w(sp), length(sp.ch)
    R = sp.L + 2H
    S = nb + 1
    cf = Vector{T}(undef, R * (2nb + 1))
    for i in 0:R-1
        co = i * (2nb + 1)
        copyto!(cf, co + 1, sp.lt, i * nb + 1, nb)
        copyto!(cf, co + nb + 1, sp.ut, (i + nb) * S + 1, S)
    end
    Tw = promote_type(T, Float64)
    α = Tw(sp.α)
    b = π / α^2
    hw = w ÷ 2
    c = invmod(Δ, n)
    wt = Vector{T}(undef, t * w)
    l0 = Vector{Int}(undef, t)
    for q in 0:t-1
        δ = Tw(t - q) / t
        for o in 1-hw:hw
            wt[q*w+o+hw] = T(exp(-b * (δ - o)^2) / α)
        end
        l0[q+1] = mod(q * c - hw, n)
    end
    return t, w, nb, H, R, wt, l0, cf, copy(sp.ot), sp.sx[1], T(n / t)
end

function _hv_pass(::Type{T}, n::Int, D::Int, B::Int) where {T}
    t, w, nb, H, R, wt, l0, cf, oi, q0, sc =
        _cached(() -> _hv_parts(T, n, D), _HV_PASS_CACHE, (T, n, D))
    W = pick_width(T)
    M = max(t, R + 2nb)
    C = B < 2W ? B : clamp(_RS_CHUNK_MAX ÷ (4 * M * sizeof(T)) ÷ W * W, W, B ÷ W * W)
    cmax = B - (max(1, B ÷ C) - 1) * C   # the last chunk is the widest
    stages, _ = build_stages_cached(T, t, D; allow_small=B == 1)
    pl() = Vector{T}(undef, cmax * M)
    return HvPass{T}(n, t, w, nb, H, R, B, C, wt, l0, cf, oi, q0, sc, stages,
                     Vector{Int}(undef, w),
                     Vector{T}(undef, cmax * n), Vector{T}(undef, cmax * n),
                     pl(), pl(), pl(), pl(),
                     zeros(T, cmax * (R + 2nb)), zeros(T, cmax * (R + 2nb)))
end

function _rs_block!(q::HvPass{T}, xre::Vector{T}, xim::Vector{T}, off::Int,
                    d::Val, tw=nothing) where {T}
    _rs_chunks!(q, xre, xim, off, d, tw, 0, max(1, q.B ÷ q.C) - 1)
    return nothing
end

function _rs_chunks!(q::HvPass{T}, xre::Vector{T}, xim::Vector{T}, off::Int,
                     d::Val, tw, clo::Int, chi::Int) where {T}
    B, C = q.B, q.C
    nc = max(1, B ÷ C)
    for c in clo:chi
        i0 = c * C
        _hv_chunk!(q, xre, xim, off, i0, c == nc - 1 ? B - i0 : C, d, tw)
    end
    return nothing
end

_rs_clone(q::HvPass{T}) where {T} =
    HvPass{T}(q.n, q.t, q.w, q.nb, q.H, q.R, q.B, q.C, q.wt, q.l0, q.cf, q.oi, q.q0, q.sc,
              q.stages, similar(q.ro), similar(q.are), similar(q.aim), similar(q.pre),
              similar(q.pim), similar(q.qre), similar(q.qim), zero(q.zr), zero(q.zi))

function _hv_chunk!(q::HvPass{T}, xre::Vector{T}, xim::Vector{T}, off0::Int, i0::Int,
                    L::Int, d::Val, tw) where {T}
    n, t, w, nb, H, R, B = q.n, q.t, q.w, q.nb, q.H, q.R, q.B
    off = off0 + i0
    are, aim, ro, cf, oi = q.are, q.aim, q.ro, q.cf, q.oi
    @inbounds for j in 0:n-1
        if tw === nothing
            _rows_scale!(are, aim, L * j, xre, xim, off + B * j, one(T), L)
        else
            _rows_scale_tw!(are, aim, L * j, xre, xim, off + B * j, one(T), L, tw, j, i0)
        end
    end
    @inbounds for k in 0:t-1
        r = q.l0[k+1]
        for o in 1:w
            ro[o] = L * r
            r += 1
            r = ifelse(r == n, 0, r)
        end
        _rows_spread!(q.pre, q.pim, L * k, are, aim, q.wt, w * k, ro, w, L)
    end
    rng = 1:L*t
    gre, gim, sre, sim = run_chain!(view(q.pre, rng), view(q.pim, rng), view(q.qre, rng),
                                    view(q.qim, rng), q.stages, t, 1, L, d)
    gre, gim, xr, xi = parent(gre), parent(gim), parent(sre), parent(sim)
    zr, zi = q.zr, q.zi
    fill!(view(zr, 1:L*nb), zero(T))
    fill!(view(zi, 1:L*nb), zero(T))
    k = q.q0
    @inbounds for i in 0:R-1      # forward substitution, rows i-nb .. i-1
        _rows_fwd!(zr, zi, L * (i + nb), gre, gim, L * k, oi[i+1] ≥ 0, cf,
                   i * (2nb + 1), nb, L)
        k += 1
        k = ifelse(k == t, 0, k)
    end
    fill!(view(xr, L*(R+nb)+1:L*(R+2nb)), zero(T))
    fill!(view(xi, L*(R+nb)+1:L*(R+2nb)), zero(T))
    @inbounds for i in R-1:-1:H   # back substitution into the spare planes
        _rows_back!(xr, xi, zr, zi, L * (i + nb), cf, i * (2nb + 1) + nb, nb, L)
        o = oi[i+1]
        i < R - H && o ≥ 0 &&
            _rows_scale!(xre, xim, off + B * o, xr, xi, L * (i + nb), q.sc, L)
    end
    return nothing
end

# f(z, i) over L lanes in windows of typeof(z) (a zero vector, as closures
# don't specialize on a type argument), overlapping at the end, so only for
# out-of-place work
@inline function _hv_lanes(f::F, L, ::Type{T}) where {F,T}
    W = pick_width(T)
    if W > 1 && L ≥ W
        for i in QWin(L, W)
            f(zero(Vec{W,T}), i)
        end
    else
        for i in 1:L
            f(zero(Vec{1,T}), i)
        end
    end
    return nothing
end

# y row at yo = Σ_o wt[wo + o] · x row at ro[o]
@inline function _rows_spread!(yre, yim, yo, xre, xim, wt::Vector{T}, wo, ro, w, L) where {T}
    _hv_lanes(L, T) do z, i
        @inline
        VT = typeof(z)
        @inbounds begin
            ar = ai = z
            for o in 1:w
                c = VT(wt[wo+o])
                ar = muladd(c, vload(VT, xre, ro[o] + i), ar)
                ai = muladd(c, vload(VT, xim, ro[o] + i), ai)
            end
            vstore(ar, yre, yo + i)
            vstore(ai, yim, yo + i)
        end
    end
end

# z row at zo = (hit ? y row at yo : 0) - Σ_m cf[co + m] · z row nb + 1 - m back
@inline function _rows_fwd!(zr, zi, zo, yre, yim, yo, hit, cf::Vector{T}, co, nb, L) where {T}
    _hv_lanes(L, T) do z, i
        @inline
        VT = typeof(z)
        @inbounds begin
            ar = hit ? vload(VT, yre, yo + i) : z
            ai = hit ? vload(VT, yim, yo + i) : z
            for m in 1:nb
                f = VT(cf[co+m])
                zp = zo - L * (nb + 1 - m) + i
                ar = muladd(-f, vload(VT, zr, zp), ar)
                ai = muladd(-f, vload(VT, zi, zp), ai)
            end
            vstore(ar, zr, zo + i)
            vstore(ai, zi, zo + i)
        end
    end
end

# x row at xo = (z row at xo - Σ_j U_j · x row j ahead) / U_ii, with
# cf[co + 1] = 1/U_ii and cf[co + 1 + j] = U_j
@inline function _rows_back!(xr, xi, zr, zi, xo, cf::Vector{T}, co, nb, L) where {T}
    _hv_lanes(L, T) do z, i
        @inline
        VT = typeof(z)
        @inbounds begin
            ar = vload(VT, zr, xo + i)
            ai = vload(VT, zi, xo + i)
            for j in 1:nb
                u = VT(cf[co+1+j])
                ar = muladd(-u, vload(VT, xr, xo + L * j + i), ar)
                ai = muladd(-u, vload(VT, xi, xo + L * j + i), ai)
            end
            iu = VT(cf[co+1])
            vstore(ar * iu, xr, xo + i)
            vstore(ai * iu, xi, xo + i)
        end
    end
end
