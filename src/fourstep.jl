# Four-step (six-step) decomposition for large 1-D transforms. Direct chains
# stream the whole array from DRAM once per stage; past the cache sizes that
# is ~log4(n) full-array sweeps. Here n = n1·n2 runs as
#   deinterleave → T → n1 pencil-FFTs(n2) + twiddle → T → n2 pencil-FFTs(n1)
#   → T → interleave
# (~7 sweeps), where every pencil FFT is cache-resident and phases thread
# across pencils with no barriers. With input index j = j1 + n1·j2 and output
# k = n2·k1 + k2:
#   X[n2·k1 + k2] = Σ_{j1} ω_{n1}^{j1 k1} · ω_n^{j1 k2} · Σ_{j2} x[j1 + n1 j2] ω_{n2}^{j2 k2}
# so the twiddle ω_n^{D·j1·k2} sits between the two FFT phases and the final
# transpose delivers natural order.

# minimum length before the extra transpose/twiddle sweeps pay for themselves
const FOURSTEP_MIN = 1 << 21

struct FourStepPlan{T,D}
    n::Int
    n1::Int                     # input-fastest extent (phase-2 pencil length)
    n2::Int                     # phase-1 pencil length
    stages1::Vector{Stage{T}}   # chain for n1
    stages2::Vector{Stage{T}}   # chain for n2
    twr::Vector{T}              # ω_n^{D·j1·k2}, index k2 + n2·j1 + 1
    twi::Vector{T}              #   (contiguous along each phase-1 pencil)
    are::Vector{T}
    aim::Vector{T}
    bre::Vector{T}
    bim::Vector{T}
    nt::Int
    scale::T
end

plan_nt(p::FourStepPlan) = p.nt

# balanced split of n over its prime factors: largest remaining factor goes
# to the currently-smaller side, so n1, n2 ≈ √n and both stay radix-buildable
function _fs_split(n::Int)
    a, ps = factor_radices(n)
    primes = Int[]
    for (p, c) in ps, _ in 1:c
        push!(primes, p)
    end
    for _ in 1:a
        push!(primes, 2)
    end
    n1 = n2 = 1
    for p in primes   # already descending: odd primes descending, then 2s
        if n1 ≤ n2
            n1 *= p
        else
            n2 *= p
        end
    end
    return n1, n2
end

const _FS_TW_CACHE = Dict{Tuple{DataType,Int,Int,Int},Any}()
function _fs_twiddles(::Type{T}, n::Int, n1::Int, D::Int) where {T}
    r = _cached(_FS_TW_CACHE, (T, n, n1, D)) do
        n2 = n ÷ n1
        twr = Vector{T}(undef, n)
        twi = Vector{T}(undef, n)
        for j1 in 0:n1-1, k2 in 0:n2-1
            t = Int((2 * widemul(j1, k2)) % (2n))   # exp(iπ·t/n), t mod 2n exact
            si, co = sincospi(D * (Float64(t) / n))
            twr[j1*n2+k2+1] = T(co)
            twi[j1*n2+k2+1] = T(si)
        end
        (twr, twi)
    end
    return r::Tuple{Vector{T},Vector{T}}
end

function _fourstep(::Type{T}, n::Int, D::Int, scale::T=default_scale(T, n, D);
                   bufs::Union{Nothing,NTuple{4,Vector{T}}}=nothing,
                   nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat}
    n1, n2 = _fs_split(n)
    stages1, _ = build_stages_cached(T, n1, D)
    stages2, _ = build_stages_cached(T, n2, D)
    twr, twi = _fs_twiddles(T, n, n1, D)
    if bufs === nothing
        bufs = (Vector{T}(undef, n), Vector{T}(undef, n),
                Vector{T}(undef, n), Vector{T}(undef, n))
    end
    return FourStepPlan{T,D}(n, n1, n2, stages1, stages2, twr, twi, bufs...,
                             plan_tasks(n, nthreads), scale)
end

# Smooth lengths: four-step only when big AND threaded. Its extra transpose/
# twiddle sweeps pay only when multiple cores contend for memory bandwidth
# (measured 4t: 0.74–0.85× direct at ≥ 2^21; serial: 1.1–1.7× slower — one
# core can't saturate DRAM, so direct's streaming stages are free serially).
function _smoothplan(::Type{T}, n::Int, D::Int, scale::T=default_scale(T, n, D);
                     bufs::Union{Nothing,NTuple{4,Vector{T}}}=nothing,
                     nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat}
    if n ≥ FOURSTEP_MIN && plan_tasks(n, nthreads) > 1
        return _fourstep(T, n, D, scale; bufs, nthreads)
    end
    return _plan(T, n, D, scale; bufs, nthreads)
end

# in-place twiddle multiply of one pencil (length len views), table at toff
function _fs_twiddle!(re::PlaneVec{T}, im::PlaneVec{T},
                      twr::Vector{T}, twi::Vector{T}, toff::Int) where {T}
    len = length(re)
    W = pick_width(T)
    if W > 1 && len ≥ W
        @inbounds for q in QWin(len, W)
            wr = vload(Vec{W,T}, twr, toff + q)
            wi = vload(Vec{W,T}, twi, toff + q)
            r = vload(Vec{W,T}, re, q)
            i = vload(Vec{W,T}, im, q)
            or, oi = cmul(wr, wi, r, i)
            vstore(or, re, q)
            vstore(oi, im, q)
        end
    else
        @inbounds for q in 1:len
            or, oi = cmul(twr[toff+q], twi[toff+q], re[q], im[q])
            re[q] = or
            im[q] = oi
        end
    end
    return nothing
end

# pencils olo:ohi of one FFT phase: per-pencil cache-resident chain, plus the
# fused twiddle post-pass while the pencil is hot (phase 1 only)
function _fs_pencils!(cre, cim, ore, oim, stages, len::Int,
                      twr, twi, olo::Int, ohi::Int, d::Val)
    @inbounds for o in olo-1:ohi-1
        rng = o*len+1:(o+1)*len
        fr, fi, _, _ = run_chain!(view(cre, rng), view(cim, rng),
                                  view(ore, rng), view(oim, rng),
                                  stages, len, 1, 1, d)
        if twr !== nothing
            _fs_twiddle!(fr, fi, twr, twi, o * len)
        end
    end
    return nothing
end

# one phase: B pencils of length len, threaded across pencils (no barriers);
# returns the plane roles with the phase's ping-pong parity applied
function _fs_phase!(cre, cim, ore, oim, stages, len::Int, B::Int,
                    twr, twi, d::Val, nt::Int)
    if nt > 1
        let cre = cre, cim = cim, ore = ore, oim = oim
            tforeach((lo, hi) -> _fs_pencils!(cre, cim, ore, oim, stages, len,
                                              twr, twi, lo, hi, d),
                     B, nt)
        end
    else
        _fs_pencils!(cre, cim, ore, oim, stages, len, twr, twi, 1, B, d)
    end
    return isodd(length(stages)) ? (ore, oim, cre, cim) : (cre, cim, ore, oim)
end

function (p::FourStepPlan{T,D})(y::CVec{T}, x::CVec{T}) where {T,D}
    n = p.n
    length(x) == n || throw(DimensionMismatch("plan length $n, input length $(length(x))"))
    length(y) == n || throw(DimensionMismatch("plan length $n, output length $(length(y))"))
    n1, n2, nt = p.n1, p.n2, p.nt
    cre, cim, ore, oim = p.are, p.aim, p.bre, p.bim
    deinterleave_all!(cre, cim, x, nt)
    # A[j1,j2] (j1 fast) → A'[j2,j1]
    cre, cim, ore, oim = transpose_planes!(ore, oim, cre, cim, n1, n2, nt)
    # FFT over j2 + twiddle ω_n^{D·j1·k2}: n1 pencils of length n2
    cre, cim, ore, oim = _fs_phase!(cre, cim, ore, oim, p.stages2, n2, n1,
                                    p.twr, p.twi, Val(D), nt)
    # B'[k2,j1] → B[j1,k2]
    cre, cim, ore, oim = transpose_planes!(ore, oim, cre, cim, n2, n1, nt)
    # FFT over j1: n2 pencils of length n1 → D[k1,k2]
    cre, cim, ore, oim = _fs_phase!(cre, cim, ore, oim, p.stages1, n1, n2,
                                    nothing, nothing, Val(D), nt)
    # D[k1,k2] → X in natural order (k2 fastest)
    cre, cim, ore, oim = transpose_planes!(ore, oim, cre, cim, n1, n2, nt)
    interleave_all!(y, cre, cim, p.scale, nt)
    return y
end

(p::FourStepPlan{T})(x::Vector{Complex{T}}) where {T} = p(similar(x), x)
