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

# odd primes the direct engine has stage kernels for, largest first (the
# order odd stages run in, after the radix-4 tower); 11/13 go through the
# generic bfp butterfly, 7/5/3 through their handwritten kernels
const ODD_RADICES = (13, 11, 7, 5, 3)

# factor n = 2^a · Π p^c over ODD_RADICES (largest first, matching stage
# order), or nothing when a larger prime remains
function factor_radices(n::Int)
    a = trailing_zeros(n)
    r = n >> a
    ps = Pair{Int,Int}[]
    for p in ODD_RADICES
        c = 0
        while r % p == 0
            r ÷= p
            c += 1
        end
        c > 0 && push!(ps, p => c)
    end
    r == 1 || return nothing
    return a, ps
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
    nt::Int      # task count for execution (1 = serial, zero-alloc)
    scale::T
end

# plan-time task count: thread only when each task gets ≥ MIN_CHUNK elements
# of per-stage work
plan_tasks(work::Int, nthreads::Int) = min(max(nthreads, 1), max(work ÷ MIN_CHUNK, 1))

plan_nt(p::FFTPlan) = p.nt

function make_stage(::Type{T}, radix::Int, n::Int, s::Int, D::Int;
                    allow_small::Bool=true) where {T}
    m = n ÷ radix
    Tw = promote_type(T, Float64)
    small = allow_small && use_small(T, radix, m, s)
    if small
        ms = m * s
        twr = Vector{T}(undef, (radix - 1) * ms)
        twi = Vector{T}(undef, (radix - 1) * ms)
        for j in 1:radix-1, p in 0:m-1
            θ = Tw(2 * D * p * j) / n   # angle in units of π: w = exp(iπθ)
            si, co = sincospi(θ)
            base = (j - 1) * ms + p * s
            for q in 1:s   # twiddle repeated per lane across the stride
                twr[base+q] = T(co)
                twi[base+q] = T(si)
            end
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

# The 1-D stage chain for an n factoring over ODD_RADICES: final power-of-two
# stage f (m == 1, twiddle-free, fusable); the rest of the 2-part runs as a
# radix-4 tower first, so the odd-radix stages in between see large
# W-divisible strides. allow_small=false suppresses the small-stride twiddle
# layout (batched N-d passes run every stage at effective stride s·Bi ≥ W,
# never the small path).
function build_stages(::Type{T}, n::Int, D::Int; allow_small::Bool=true) where {T}
    fac = factor_radices(n)
    fac === nothing &&
        throw(ArgumentError("build_stages requires a length with prime factors ≤ $(ODD_RADICES[1]) (got $n)"))
    a, ps = fac
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
    for (radix, cnt) in ps
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
_cached(f, dict::Dict, key) = lock(() -> get!(f, dict, key), _PLAN_CACHE_LOCK)

const _STAGE_CACHE = Dict{Tuple{DataType,Int,Int,Bool},Any}()
function build_stages_cached(::Type{T}, n::Int, D::Int;
                             allow_small::Bool=true) where {T}
    r = _cached(() -> build_stages(T, n, D; allow_small), _STAGE_CACHE,
                (T, n, D, allow_small))
    return r::Tuple{Vector{Stage{T}},Int}
end

# bufs shares another plan's scratch planes (safe when the two plans never
# execute concurrently, e.g. Bluestein's sequential forward/inverse pair)
function _plan(::Type{T}, n::Int, D::Int, scale::T=default_scale(T, n, D);
               bufs::Union{Nothing,NTuple{4,Vector{T}}}=nothing,
               nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat}
    n ≥ 1 || throw(ArgumentError("transform length must be positive (got $n)"))
    stages, _ = build_stages_cached(T, n, D)
    if bufs === nothing
        bufs = (Vector{T}(undef, n), Vector{T}(undef, n),
                Vector{T}(undef, n), Vector{T}(undef, n))
    end
    return FFTPlan{T,D}(n, pencil_fused(T, n, stages), stages, bufs...,
                        plan_tasks(n, nthreads), scale)
end

# Routing for smooth-but-odd-heavy lengths (measured 2026-08, see the
# threading spec): direct chains beat Bluestein at every smooth size for
# W ≤ 8 (the two padded transforms cost ~5× the butterfly work, and threaded
# q-splits keep odd stages vectorized), so only wide-vector Float32 (W ≥ 16)
# keeps the Bluestein route, for the small odd-heavy sizes where odd stages
# run at scalar strides: crossover measured between 3^8 = 6561 (Bluestein
# 1.4× ahead) and 14175 (direct 1.5× ahead).
function _slow_smooth(::Type{T}, n::Int) where {T}
    W = pick_width(T)
    a = trailing_zeros(n)
    return W ≥ 16 && n >> a > 1 && 256 ≤ n ≤ 8192 && (1 << a) < 4 * W
end

function _anyplan(::Type{T}, n::Int, D::Int, scale::T=default_scale(T, n, D);
                  nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat}
    n ≥ 1 || throw(ArgumentError("transform length must be positive (got $n)"))
    if factor_radices(n) === nothing || _slow_smooth(T, n)
        return _bluestein(T, n, D, scale; nthreads)
    end
    return _smoothplan(T, n, D, scale; nthreads)
end
plan_fft(::Type{T}, n::Integer; nthreads::Integer=Threads.nthreads()) where {T<:AbstractFloat} =
    _anyplan(T, Int(n), -1; nthreads=Int(nthreads))
plan_ifft(::Type{T}, n::Integer; nthreads::Integer=Threads.nthreads()) where {T<:AbstractFloat} =
    _anyplan(T, Int(n), +1; nthreads=Int(nthreads))
# unnormalized inverse (like FFTW's bfft)
plan_bfft(::Type{T}, n::Integer; nthreads::Integer=Threads.nthreads()) where {T<:AbstractFloat} =
    _anyplan(T, Int(n), +1, one(T); nthreads=Int(nthreads))

@inline function run_stage!(bre, bim, are, aim, ncur, s, st, d::Val, nt::Int=1)
    nt > 1 && return run_stage_par!(bre, bim, are, aim, ncur, s, st, d, nt)
    if st.radix == 4
        if st.small
            # small layout only arises at s ∈ {1, 4}: radix-4 tower strides
            # are powers of 4 and use_small requires s < W ≤ 16
            if s == 1
                stage4_small!(bre, bim, are, aim, ncur, Val(1), st.twr, st.twi, d)
            else
                stage4_small!(bre, bim, are, aim, ncur, Val(4), st.twr, st.twi, d)
            end
        else
            stage4!(bre, bim, are, aim, ncur, s, st.twr, st.twi, d)
        end
    elseif st.radix == 8
        stage8_plain!(bre, bim, are, aim, s, d)
    elseif st.radix == 3
        stagep!(bre, bim, are, aim, ncur, s, st.twr, st.twi, Val(3), d)
    elseif st.radix == 5
        stagep!(bre, bim, are, aim, ncur, s, st.twr, st.twi, Val(5), d)
    elseif st.radix == 7
        stagep!(bre, bim, are, aim, ncur, s, st.twr, st.twi, Val(7), d)
    elseif st.radix == 11
        stagep!(bre, bim, are, aim, ncur, s, st.twr, st.twi, Val(11), d)
    elseif st.radix == 13
        stagep!(bre, bim, are, aim, ncur, s, st.twr, st.twi, Val(13), d)
    else
        stage2!(bre, bim, are, aim, ncur, s, st.twr, st.twi)
    end
    return nothing
end

# Parallel stage dispatch: split butterflies (p) when there are enough of
# them, else split the stride lanes (q) into W-aligned chunks. Since
# p·s = n/radix, one of the two is always plentiful for large stages. Chunks
# are disjoint in input and output, and the split never changes which inputs
# any butterfly combines — results are bitwise identical to serial.
function run_stage_par!(bre, bim, are, aim, ncur::Int, s::Int, st, d::Val, nt::Int)
    W = pick_width(eltype(bre))
    radix = st.radix
    if radix == 4 && st.small
        # flat vector-block loop over (ncur>>>2)·s elements (small s ∈ {1, 4})
        if s == 1
            tforeach((lo, hi) -> stage4_small!(bre, bim, are, aim, ncur, Val(1),
                                               st.twr, st.twi, d, lo:hi),
                     ncur >>> 2, nt; align=W)
        else
            tforeach((lo, hi) -> stage4_small!(bre, bim, are, aim, ncur, Val(4),
                                               st.twr, st.twi, d, lo:hi),
                     (ncur >>> 2) * 4, nt; align=W)
        end
    elseif radix == 8
        tforeach((lo, hi) -> stage8_plain!(bre, bim, are, aim, s, d, lo:hi),
                 s, nt; align=W)
    else
        m = ncur ÷ radix
        if m ≥ nt || s < W
            tforeach((lo, hi) -> _stage_pq!(bre, bim, are, aim, ncur, s, st, d,
                                            (lo - 1):(hi - 1), 1:s),
                     m, nt)
        else
            tforeach((lo, hi) -> _stage_pq!(bre, bim, are, aim, ncur, s, st, d,
                                            0:m-1, lo:hi),
                     s, nt; align=W)
        end
    end
    return nothing
end

@inline function _stage_pq!(bre, bim, are, aim, ncur, s, st, d::Val,
                            pr::UnitRange{Int}, qr::UnitRange{Int})
    radix = st.radix
    if radix == 4
        stage4!(bre, bim, are, aim, ncur, s, st.twr, st.twi, d, pr, qr)
    elseif radix == 3
        stagep!(bre, bim, are, aim, ncur, s, st.twr, st.twi, Val(3), d, pr, qr)
    elseif radix == 5
        stagep!(bre, bim, are, aim, ncur, s, st.twr, st.twi, Val(5), d, pr, qr)
    elseif radix == 7
        stagep!(bre, bim, are, aim, ncur, s, st.twr, st.twi, Val(7), d, pr, qr)
    elseif radix == 11
        stagep!(bre, bim, are, aim, ncur, s, st.twr, st.twi, Val(11), d, pr, qr)
    elseif radix == 13
        stagep!(bre, bim, are, aim, ncur, s, st.twr, st.twi, Val(13), d, pr, qr)
    else
        stage2!(bre, bim, are, aim, ncur, s, st.twr, st.twi, pr, qr)
    end
    return nothing
end

# run a stage chain, ping-ponging between the (cre,cim) and (ore,oim) planes;
# effective stride is s·B (B = inner batch). Returns the final plane roles.
@inline function run_chain!(cre, cim, ore, oim, stages, ncur::Int, s::Int,
                            B::Int, d::Val, nt::Int=1)
    @inbounds for st in stages
        run_stage!(ore, oim, cre, cim, ncur, s * B, st, d, nt)
        cre, ore = ore, cre
        cim, oim = oim, cim
        s *= st.radix
        ncur ÷= st.radix
    end
    return cre, cim, ore, oim
end

# fused final stage: twiddle-free power-of-two radix writing interleaved output
@inline function run_last_stage!(py::Ptr{T}, are, aim, s::Int, radix::Int,
                                 scale::T, d::Val, nt::Int=1) where {T}
    if nt > 1
        W = pick_width(T)
        if radix == 4
            tforeach((lo, hi) -> stage4_last!(py, are, aim, s, scale, d, lo:hi),
                     s, nt; align=W)
        elseif radix == 8
            tforeach((lo, hi) -> stage8_last!(py, are, aim, s, scale, d, lo:hi),
                     s, nt; align=W)
        else
            tforeach((lo, hi) -> stage2_last!(py, are, aim, s, scale, lo:hi),
                     s, nt; align=W)
        end
        return nothing
    end
    if radix == 4
        stage4_last!(py, are, aim, s, scale, d)
    elseif radix == 8
        stage8_last!(py, are, aim, s, scale, d)
    else
        stage2_last!(py, are, aim, s, scale)
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
    nt = p.nt
    if p.fused
        GC.@preserve x y begin
            px = _rawptr(x)
            py = _rawptr(y)
            st1 = stages[1]
            if nt > 1
                # let-bound copies: are/aim are reassigned below, and capturing
                # a reassigned variable would box it (allocating on every call,
                # nt == 1 included)
                let yr = are, yi = aim
                    tforeach((lo, hi) -> stage4_first!(yr, yi, px, n, st1.twr,
                                                       st1.twi, Val(D), lo:hi),
                             n >>> 2, nt; align=pick_width(T))
                end
            else
                stage4_first!(are, aim, px, n, st1.twr, st1.twi, Val(D))
            end
            are, aim, bre, bim =
                run_chain!(are, aim, bre, bim, view(stages, 2:length(stages)-1),
                           n >>> 2, 4, 1, Val(D), nt)
            lastradix = stages[end].radix
            run_last_stage!(py, are, aim, n ÷ lastradix, lastradix, p.scale,
                            Val(D), nt)
        end
        return y
    end
    deinterleave_all!(are, aim, x, nt)
    are, aim, bre, bim = run_chain!(are, aim, bre, bim, stages, n, 1, 1, Val(D), nt)
    interleave_all!(y, are, aim, p.scale, nt)
    return y
end

(p::FFTPlan{T})(x::Vector{Complex{T}}) where {T} = p(similar(x), x)
