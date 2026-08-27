struct Stage{T}
    radix::Int
    small::Bool     # small-stride stage: twiddles expanded per lane
    twr::Vector{T}  # (radix-1) twiddles per butterfly, in access order
    twi::Vector{T}
end

# small-stride path applies when s < W and a full vector spans whole butterflies
use_small(::Type{T}, radix::Int, m::Int, s::Int) where {T} =
    radix == 4 && s < pick_width(T) && m * s ≥ pick_width(T) &&
    (m * s) % pick_width(T) == 0

# factor n = 2^a·3^b·5^c·7^d, or nothing when n is not 7-smooth
function factor_smooth(n::Int)
    a = trailing_zeros(n)
    r = n >> a
    b = 0; while r % 3 == 0; r ÷= 3; b += 1; end
    c = 0; while r % 5 == 0; r ÷= 5; c += 1; end
    d = 0; while r % 7 == 0; r ÷= 7; d += 1; end
    r == 1 || return nothing
    return a, b, c, d
end

# D: sign of the transform exponent (-1 forward, +1 inverse)
struct FFTPlan{T,D}
    n::Int
    fused::Bool  # edge passes fused into first/last stage (needs ≥2 vector-wide stages)
    stages::Vector{Stage{T}}
    are::Vector{T}
    aim::Vector{T}
    bre::Vector{T}
    bim::Vector{T}
    scale::T
end

function make_stage(::Type{T}, radix::Int, n::Int, s::Int, D::Int;
                    allow_small::Bool=true) where {T}
    m = n ÷ radix
    Tw = promote_type(T, Float64)
    small = allow_small && use_small(T, radix, m, s)
    if small
        ms = m * s
        twr = Vector{T}(undef, (radix - 1) * ms)
        twi = Vector{T}(undef, (radix - 1) * ms)
        for j in 1:radix-1, f in 1:ms
            p = (f - 1) ÷ s
            θ = Tw(2 * D * p * j) / n   # angle in units of π: w = exp(iπθ)
            si, co = sincospi(θ)
            twr[(j-1)*ms+f] = T(co)
            twi[(j-1)*ms+f] = T(si)
        end
    else
        twr = Vector{T}(undef, (radix - 1) * m)
        twi = Vector{T}(undef, (radix - 1) * m)
        for p in 0:m-1, j in 1:radix-1
            θ = Tw(2 * D * p * j) / n
            si, co = sincospi(θ)
            twr[(radix-1)*p+j] = T(co)
            twi[(radix-1)*p+j] = T(si)
        end
    end
    return Stage{T}(radix, small, twr, twi)
end

default_scale(::Type{T}, n, D) where {T} = D == -1 ? one(T) : one(T) / n

# The 1-D stage chain for a 7-smooth n: final power-of-two stage f (m == 1,
# twiddle-free, fusable); the rest of the 2-part runs as a radix-4 tower
# first, so the odd-radix stages in between see large W-divisible strides.
# allow_small=false suppresses the small-stride twiddle layout (batched N-d
# passes run every stage at effective stride s·Bi ≥ W, never the small path).
function build_stages(::Type{T}, n::Int, D::Int; allow_small::Bool=true) where {T}
    fac = factor_smooth(n)
    fac === nothing && throw(ArgumentError("build_stages requires a 7-smooth length (got $n)"))
    a, b3, c5, d7 = fac
    f = a == 0 ? 1 : isodd(a) ? (a ≥ 3 ? 8 : 2) : 4
    k4 = (a - trailing_zeros(f)) ÷ 2
    stages = Stage{T}[]
    ncur = n
    s = 1
    for _ in 1:k4
        push!(stages, make_stage(T, 4, ncur, s, D; allow_small))
        s *= 4
        ncur >>>= 2
    end
    for (radix, cnt) in ((7, d7), (5, c5), (3, b3))
        for _ in 1:cnt
            push!(stages, make_stage(T, radix, ncur, s, D; allow_small))
            s *= radix
            ncur ÷= radix
        end
    end
    f > 1 && push!(stages, make_stage(T, f, ncur, s, D; allow_small))
    return stages, f
end

# Immutable plan parts (stage chains with their twiddle tables, Bluestein
# chirps, real-edge twiddles) are shared across plans, FFTW-style; entries
# live for the session and the GC handles the rest. Only per-plan scratch
# buffers are allocated fresh, so repeated planning at one size is cheap —
# DSP-style code plans per call.
const _PLAN_CACHE_LOCK = ReentrantLock()
const _STAGE_CACHE = Dict{Tuple{DataType,Int,Int,Bool},Any}()
function build_stages_cached(::Type{T}, n::Int, D::Int;
                             allow_small::Bool=true) where {T}
    r = lock(_PLAN_CACHE_LOCK) do
        get!(() -> build_stages(T, n, D; allow_small), _STAGE_CACHE,
             (T, n, D, allow_small))
    end
    return r::Tuple{Vector{Stage{T}},Int}
end

function _plan(::Type{T}, n::Int, D::Int,
               scale::T=default_scale(T, n, D)) where {T<:AbstractFloat}
    n ≥ 1 || throw(ArgumentError("transform length must be positive (got $n)"))
    stages, f = build_stages_cached(T, n, D)
    W = pick_width(T)
    slast = f > 1 ? n ÷ f : 0
    fused = W > 1 && length(stages) ≥ 2 &&
            stages[1].radix == 4 && stages[1].small && (n >>> 2) % W == 0 &&
            f > 1 && slast ≥ W && slast % W == 0
    return FFTPlan{T,D}(n, fused, stages,
                        Vector{T}(undef, n), Vector{T}(undef, n),
                        Vector{T}(undef, n), Vector{T}(undef, n), scale)
end

# A smooth length whose odd stages would run at scalar strides (small 2-part)
# is faster as an exact Bluestein over a padded pow2-rich length than as a
# direct chain, once n is large enough to amortize the two transforms.
function _slow_smooth(::Type{T}, n::Int, fac) where {T}
    a, b, c, d = fac
    return b + c + d > 0 && n ≥ 256 && (1 << a) < 4 * pick_width(T)
end

function _anyplan(::Type{T}, n::Int, D::Int,
                  scale::T=default_scale(T, n, D)) where {T<:AbstractFloat}
    n ≥ 1 || throw(ArgumentError("transform length must be positive (got $n)"))
    fac = factor_smooth(n)
    if fac === nothing || _slow_smooth(T, n, fac)
        return _bluestein(T, n, D, scale)
    end
    return _plan(T, n, D, scale)
end
plan_fft(::Type{T}, n::Integer) where {T<:AbstractFloat} = _anyplan(T, Int(n), -1)
plan_ifft(::Type{T}, n::Integer) where {T<:AbstractFloat} = _anyplan(T, Int(n), +1)
# unnormalized inverse (like FFTW's bfft)
plan_bfft(::Type{T}, n::Integer) where {T<:AbstractFloat} =
    _anyplan(T, Int(n), +1, one(T))

@inline function run_stage!(bre, bim, are, aim, ncur, s, st, d::Val)
    if st.radix == 4
        if st.small
            if s == 1
                stage4_small!(bre, bim, are, aim, ncur, Val(1), st.twr, st.twi, d)
            elseif s == 4
                stage4_small!(bre, bim, are, aim, ncur, Val(4), st.twr, st.twi, d)
            else
                stage4!(bre, bim, are, aim, ncur, s, st.twr, st.twi, d)
            end
        else
            stage4!(bre, bim, are, aim, ncur, s, st.twr, st.twi, d)
        end
    elseif st.radix == 8
        stage8_plain!(bre, bim, are, aim, s, d)
    elseif st.radix == 3
        stage3!(bre, bim, are, aim, ncur, s, st.twr, st.twi, d)
    elseif st.radix == 5
        stage5!(bre, bim, are, aim, ncur, s, st.twr, st.twi, d)
    elseif st.radix == 7
        stage7!(bre, bim, are, aim, ncur, s, st.twr, st.twi, d)
    else
        stage2!(bre, bim, are, aim, ncur, s, st.twr, st.twi)
    end
    return nothing
end

# complex vectors accepted by plan execution: anything contiguous with a
# recoverable data pointer (Vector, contiguous views, reinterprets of either —
# used by rfft/irfft and the N-d pencil loops to avoid copies)
const CVec{T} = AbstractVector{Complex{T}}
_rawptr(x::AbstractArray{Complex{T}}) where {T} = convert(Ptr{T}, pointer(x))
_rawptr(x::Base.ReinterpretArray{Complex{T},1,T}) where {T} =
    convert(Ptr{T}, pointer(parent(x)))
_rawptr(x::AbstractVector{T}) where {T<:Real} = convert(Ptr{T}, pointer(x))

function (p::FFTPlan{T,D})(y::CVec{T}, x::CVec{T}) where {T,D}
    n = p.n
    length(x) == n || throw(DimensionMismatch("plan length $n, input length $(length(x))"))
    length(y) == n || throw(DimensionMismatch("plan length $n, output length $(length(y))"))
    are, aim, bre, bim = p.are, p.aim, p.bre, p.bim
    stages = p.stages
    if p.fused
        GC.@preserve x y begin
            px = _rawptr(x)
            py = _rawptr(y)
            st1 = stages[1]
            stage4_first!(are, aim, px, n, st1.twr, st1.twi, Val(D))
            ncur = n >>> 2
            s = 4
            @inbounds for i in 2:length(stages)-1
                st = stages[i]
                run_stage!(bre, bim, are, aim, ncur, s, st, Val(D))
                are, bre = bre, are
                aim, bim = bim, aim
                s *= st.radix
                ncur ÷= st.radix
            end
            lastradix = stages[end].radix
            if lastradix == 4
                stage4_last!(py, are, aim, s, p.scale, Val(D))
            elseif lastradix == 8
                stage8_last!(py, are, aim, s, p.scale, Val(D))
            else
                stage2_last!(py, are, aim, s, p.scale)
            end
        end
        return y
    end
    xs = reinterpret(T, x)
    @inbounds for i in 1:n
        are[i] = xs[2i-1]
        aim[i] = xs[2i]
    end
    ncur = n
    s = 1
    @inbounds for st in p.stages
        run_stage!(bre, bim, are, aim, ncur, s, st, Val(D))
        are, bre = bre, are
        aim, bim = bim, aim
        s *= st.radix
        ncur ÷= st.radix
    end
    scale = p.scale
    ys = reinterpret(T, y)
    @inbounds for i in 1:n
        ys[2i-1] = scale * are[i]
        ys[2i] = scale * aim[i]
    end
    return y
end

(p::FFTPlan{T})(x::Vector{Complex{T}}) where {T} = p(similar(x), x)
