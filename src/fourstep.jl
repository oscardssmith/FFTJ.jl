# Four-step 1-D FFT for out-of-cache lengths n = N1·N2, with j = j1 + N1·j2
# and k = k2 + N2·k1. Pass A transforms strips of S columns j1 over j2 in a
# cache-resident buffer and applies the twiddles ω^{j1·k2} on the way out;
# pass B transforms groups of G columns k2 over j1 and writes natural-order
# output rows directly. Each pass reads and writes the data once. S and G
# are multiples of W; wider strips measured no faster (more nesting, no
# TLB relief), so they default to W.

# A length-N transform of L-lane rows whose buffers would outgrow L2 runs as
# a nested four-step, N = M1·M2 (j = j1 + M1·j2, k = k2 + M2·k1). Its
# callers store input row j at slot j2 + M2·j1, so each length-M2 transform
# is one contiguous block; it scales by ω_N^{j1·k2} (shared by all lanes)
# into slot j1 + P1·k2 and transforms those blocks over j1 in place, leaving
# output row k at slot k1 + P1·k2 for callers to read. Blocks are padded to
# P1 = M1 + pad rows so the twiddle pass's row writes avoid a power-of-two
# stride. Its passes stay in L3. Without nesting the geometry is M1 = P1 = 1,
# M2 = N: both layouts are identity.
struct InnerFourStep{T}
    N::Int
    L::Int
    M1::Int
    M2::Int
    P1::Int                 # padded block stride in rows
    stA::Vector{Stage{T}}   # length-M2 chain, batch L
    stB::Vector{Stage{T}}   # length-M1 chain, batch L
    twr::Vector{T}          # tw[j1 + M1·k2 + 1] = ω_N^{j1·k2}
    twi::Vector{T}
    sre::Vector{T}          # L·M2 ping-pong scratch for the first transforms
    sim::Vector{T}
end

_lanes_geom(::Nothing, N) = (1, N, 1)
_lanes_geom(q::InnerFourStep, N) = (q.M1, q.M2, q.P1)
# rows a lane buffer must hold for a length-N transform
_lanes_rows(::Nothing, N) = N
_lanes_rows(q::InnerFourStep, N) = q.P1 * q.M2

# buffer bytes (two L-lane plane pairs) above which a pass's inner
# transform nests: balanced F64 splits stop fitting L2 near 2^24
const _FS_INNER_MAX = 1 << 19

function _inner_fourstep(::Type{T}, N::Int, L::Int, D::Int, pad::Int) where {T}
    4 * L * N * sizeof(T) ≤ _FS_INNER_MAX && return nothing
    M2 = 1 << (trailing_zeros(N) ÷ 2)
    M2 == 1 && return nothing   # no power-of-two part to nest on
    M1 = N ÷ M2
    Tw = promote_type(T, Float64)
    twr, twi = Vector{T}(undef, N), Vector{T}(undef, N)
    for k2 in 0:M2-1, j1 in 0:M1-1
        s, c = sincospi(Tw(2 * D * j1 * k2) / N)
        twr[j1+M1*k2+1] = T(c)
        twi[j1+M1*k2+1] = T(s)
    end
    return InnerFourStep{T}(N, L, M1, M2, M1 + pad,
                            first(build_stages_cached(T, M2, D; allow_small=false)),
                            first(build_stages_cached(T, M1, D; allow_small=false)),
                            twr, twi, Vector{T}(undef, L * M2), Vector{T}(undef, L * M2))
end

# length-N transform of the L-lane rows in (are, aim) laid out per
# _lanes_geom, using (bre, bim) as scratch; returns the planes holding the
# result
@inline function _lanes_fft!(inner::Nothing, stages, are, aim, bre, bim, N, L, d::Val)
    cre, cim, _, _ = run_chain!(are, aim, bre, bim, stages, N, 1, L, d)
    return cre, cim
end
function _lanes_fft!(q::InnerFourStep{T}, stages, are, aim, bre, bim, N, L, d::Val) where {T}
    W = pick_width(T)
    M1, M2, P1 = q.M1, q.M2, q.P1
    sre, sim = view(q.sre, 1:L*M2), view(q.sim, 1:L*M2)
    @inbounds for j1 in 0:M1-1
        blk = L*M2*j1+1:L*M2*(j1+1)
        cre, cim, _, _ = run_chain!(view(are, blk), view(aim, blk), sre, sim,
                                    q.stA, M2, 1, L, d)
        for k2 in 0:M2-1
            wr = q.twr[j1+M1*k2+1]
            wi = q.twi[j1+M1*k2+1]
            for v in 1:W:L
                yr = vload(Vec{W,T}, cre, L * k2 + v)
                yi = vload(Vec{W,T}, cim, L * k2 + v)
                vstore(muladd(yr, wr, -(yi * wi)), bre, L * (j1 + P1 * k2) + v)
                vstore(muladd(yr, wi, yi * wr), bim, L * (j1 + P1 * k2) + v)
            end
        end
    end
    # the consumed input planes are the ping-pong partner; every block takes
    # the same number of stages, so the result lands in the same plane pair
    @inbounds for k2 in 0:M2-1
        blk = L*P1*k2+1:L*(P1*k2+M1)
        run_chain!(view(bre, blk), view(bim, blk), view(are, blk), view(aim, blk),
                   q.stB, M1, 1, L, d)
    end
    return isodd(length(q.stB)) ? (are, aim) : (bre, bim)
end

# one task's strip buffers, with its own copies of the nested scratch
struct FsScratch{T}
    are::Vector{T}
    aim::Vector{T}
    bre::Vector{T}
    bim::Vector{T}
    innerA::Union{Nothing,InnerFourStep{T}}
    innerB::Union{Nothing,InnerFourStep{T}}
end

_inner_clone(::Nothing) = nothing
_inner_clone(q::InnerFourStep{T}) where {T} =
    InnerFourStep{T}(q.N, q.L, q.M1, q.M2, q.P1, q.stA, q.stB, q.twr, q.twi,
                     similar(q.sre), similar(q.sim))

struct FourStepPlan{T,D}
    n::Int
    N1::Int
    N2::Int
    S::Int                  # pass A strip width (columns j1)
    G::Int                  # pass B group width (columns k2)
    Z1::Int                 # padded row stride of z (≥ N1)
    stA::Vector{Stage{T}}   # length-N2 chain, inner batch S
    stB::Vector{Stage{T}}   # length-N1 chain, inner batch G
    innerA::Union{Nothing,InnerFourStep{T}}   # nested pass-A transform, if any
    innerB::Union{Nothing,InnerFourStep{T}}
    R::Int                  # lane twiddles ω^{k2·l} = b1[k2 % R]·b2[k2 ÷ R]
    b1r::Vector{T}          # b1[S·r + l + 1] = ω^{r·l}
    b1i::Vector{T}
    b2r::Vector{T}          # b2[S·t + l + 1] = ω^{R·t·l}
    b2i::Vector{T}
    K::Int                  # two-level twiddles: ω^e = hi[e ÷ K]·lo[e % K]
    hir::Vector{T}
    hii::Vector{T}
    lor::Vector{T}
    loi::Vector{T}
    zre::Vector{T}          # n-element intermediate planes
    zim::Vector{T}
    ws::Vector{FsScratch{T}}   # per-task strip buffers
    nt::Int
    scale::T
end

plan_nt(p::FourStepPlan) = p.nt

# (N1, N2) with both factors chain-buildable and at least one strip wide, or
# nothing. With N2 = 0, N2 is the divisor nearest t ≈ √n, capped at nflat so
# pass A doesn't nest (one nested pass measured ~20% faster than two) but
# never below half balanced, so pass B's buffers keep growing as √n. A split
# whose factors both tile into width·W strips is preferred within 2× of t;
# otherwise the nearest buildable split, whose last strip/group is ragged.
function _fourstep_split(::Type{T}, n::Int, width::Int, N2::Int=0) where {T}
    W = pick_width(T)
    W > 1 || return nothing
    S = width * W
    if N2 == 0
        r = isqrt(n)
        nflat = _FS_INNER_MAX ÷ (4 * width * W * sizeof(T))
        N2 = _fs_pick(n, S, max(min(r, nflat), r ÷ 2))
        N2 == 0 && return nothing
    end
    N1 = n ÷ N2
    ok = N1 * N2 == n && N1 ≥ S && N2 ≥ S &&
         factor_radices(N1) !== nothing && factor_radices(N2) !== nothing
    return ok ? (N1, N2) : nothing
end

# divisor of n nearest t (ties: the smaller), both factors buildable and ≥ S;
# a tiling one (both factors multiples of S) within 2× of t first, else any
function _fs_pick(n::Int, S::Int, t::Int)
    best = besttile = 0
    dist(d) = abs(log(d / t))
    for q in 1:isqrt(n)
        n % q == 0 || continue
        for d in (q, n ÷ q)
            (d ≥ S && n ÷ d ≥ S) || continue
            (factor_radices(d) === nothing || factor_radices(n ÷ d) === nothing) && continue
            better(b) = b == 0 || dist(d) < dist(b) || (dist(d) == dist(b) && d < b)
            better(best) && (best = d)
            d % S == 0 && (n ÷ d) % S == 0 && dist(d) ≤ log(2) && better(besttile) &&
                (besttile = d)
        end
    end
    return besttile != 0 ? besttile : best
end

# both factors tile into W-wide strips (no ragged strip or group)
_fs_tiled(::Type{T}, n::Int, (N1, N2)) where {T} =
    N1 % pick_width(T) == 0 && N2 % pick_width(T) == 0

# Vec{W,Bool} for W floats of interleaved complex data: complex lanes
# base .. base + W/2 - 1, valid below lim
@inline _cmask(::Val{W}, base::Int, lim::Int) where {W} =
    Vec{W,Int}(ntuple(f -> (f - 1) >> 1, Val(W))) + base < lim

# Smallest complex-data size routed to a tiled four-step, at the measured
# parity with the direct chain. With 1-vector strips: threaded ~1 MB, serial
# ~8 MB (direct 10-20% ahead up to 4 MB). Wider strips (_fs_width > 1) move
# both down: threaded ~0.5 MB (F64 0.99, F32 0.83), serial ~3 MB F64 (0.99;
# 0.80-0.93 from 4 MB) and ~5 MB F32 (4 MB 1.07, 6 MB 0.88).
function _fs_min_bytes(::Type{T}, threaded::Bool, wide::Bool) where {T}
    wide || return threaded ? 1 << 20 : 8 << 20
    threaded && return 1 << 19
    return sizeof(T) ≥ 8 ? 3 << 20 : 5 << 20
end

function _use_fourstep(::Type{T}, n::Int, threaded::Bool) where {T}
    sp = _fourstep_split(T, n, 1)
    (sp !== nothing && _fs_tiled(T, n, sp)) || return false
    wide = _fs_width(T, sp[2], sp[1]) > 1
    return 2 * n * sizeof(T) ≥ _fs_min_bytes(T, threaded, wide)
end

# Smooth n without a tiled split that run a balanced ragged four-step:
# - odd-heavy (2-part < 4W, so a direct chain's odd stages run at scalar
#   strides), serial or threaded: 0.31-0.73× the direct chain from ~2k
#   (F64) / ~1k (F32) elements, also below resampling;
# - F64 with a 2-part of exactly 4W, from 16k: 0.96 at 20k, 0.63-0.65
#   from 300k (F32's 4W = 64 still loses serially, 1.3-2.7×);
# - threaded, any 2-part, from 2^15: two fork-joins beat a direct chain's
#   per-stage ones (0.40-0.93 from 20k F64 / 40k F32).
# With a larger 2-part the serial direct chain's radix-4 tower vectorizes and
# wins (F32 8000: four-step 2.7×).
function _use_ragged(::Type{T}, n::Int, threaded::Bool=false) where {T}
    a2 = 1 << trailing_zeros(n)
    W4 = 4 * pick_width(T)
    ok = (a2 < W4 && n ≥ (sizeof(T) ≥ 8 ? 2048 : 1024)) ||
         (sizeof(T) ≥ 8 && a2 == W4 && n ≥ 1 << 14) ||
         (threaded && n ≥ 1 << 15)
    ok || return false
    sp = _fourstep_split(T, n, 1)
    return sp !== nothing && !_fs_tiled(T, n, sp)
end

# four-step synchronizes twice per call, not per stage, so it threads from
# much smaller sizes than direct chains: one task per this many elements
const _FS_TASK_MIN = 1 << 13

# Strip width in vectors: the smallest giving strided rows of ≥ 256 bytes
# (128-byte rows at a page stride cost 2.3× their traffic; 256 vs 512 bytes
# barely differ), narrowed until the strip buffers (4 planes of S lanes ×
# each pass length) fit the cache budget. N1 > 0: a 1-D split, whose factors
# must also tile at that width; N1 = 0: only pass A's length N2 (2-D rows).
# Measured w = 2 vs 1: 2-D 0.79-0.98 (one noisy 1.15), 1-D 2^20 0.73-0.84.
function _fs_width(::Type{T}, N2::Int, N1::Int=0) where {T}
    W = pick_width(T)
    w = max(1, cld(256, 2 * W * sizeof(T)))
    fits(w) = 4 * w * W * max(N1, N2) * sizeof(T) ≤ _FS_INNER_MAX &&
              (N1 == 0 || (N1 % (w * W) == 0 && N2 % (w * W) == 0))
    while w > 1 && !fits(w)
        w -= 1
    end
    return w
end

# width = 0: split as for width 1, then widen by _fs_width (a wider strip
# never changes the split: halving four-step's N2 cap cost 2^22 1.10-1.13×)
function _fourstep(::Type{T}, n::Int, D::Int, scale::T=default_scale(T, n, D);
                   N2::Int=0, width::Int=0, zpad::Int=0,
                   ipad::Int=1, nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat}
    W = pick_width(T)
    split = _fourstep_split(T, n, max(width, 1), N2)
    split === nothing &&
        throw(ArgumentError("no four-step split of $n tiles into width-$(max(width, 1) * W) strips"))
    N1, N2 = split
    width == 0 && (width = _fs_width(T, N2, N1))
    S = G = width * W
    zpad == 0 && (zpad = S)
    Tw = promote_type(T, Float64)
    ω(e) = sincospi(Tw(2 * D * e) / n)   # (sin, cos) of the angle of ω^e
    R = 1 << (trailing_zeros(N2) ÷ 2)
    nt = cld(N2, R)
    b1r, b1i = Vector{T}(undef, S * R), Vector{T}(undef, S * R)
    b2r, b2i = Vector{T}(undef, S * nt), Vector{T}(undef, S * nt)
    for r in 0:R-1, l in 0:S-1
        s, c = ω(r * l)
        b1r[S*r+l+1] = T(c); b1i[S*r+l+1] = T(s)
    end
    for t in 0:nt-1, l in 0:S-1
        s, c = ω(R * t * l)
        b2r[S*t+l+1] = T(c); b2i[S*t+l+1] = T(s)
    end
    K = isqrt(n - 1) + 1
    nh = cld(n, K)
    hir, hii = Vector{T}(undef, nh), Vector{T}(undef, nh)
    lor, loi = Vector{T}(undef, K), Vector{T}(undef, K)
    for i in 0:nh-1
        s, c = ω(i * K)
        hir[i+1] = T(c); hii[i+1] = T(s)
    end
    for i in 0:K-1
        s, c = ω(i)
        lor[i+1] = T(c); loi[i+1] = T(s)
    end
    stA, _ = build_stages_cached(T, N2, D; allow_small=false)
    stB, _ = build_stages_cached(T, N1, D; allow_small=false)
    zpad % W == 0 || throw(ArgumentError("zpad must be a multiple of $W (got $zpad)"))
    # a ragged last strip writes its spare lanes into z's row padding
    N1 % S == 0 || zpad ≥ S ||
        throw(ArgumentError("zpad must cover a strip ($S) when N1 = $N1 is ragged"))
    Z1 = N1 + zpad
    innerA = _inner_fourstep(T, N2, S, D, ipad)
    innerB = _inner_fourstep(T, N1, G, D, ipad)
    nb = max(S * _lanes_rows(innerA, N2), G * _lanes_rows(innerB, N1))
    na, ng = cld(N1, S), cld(N2, G)
    nt = min(max(nthreads, 1), na, ng, max(1, n ÷ _FS_TASK_MIN))
    ws = [FsScratch{T}(Vector{T}(undef, nb), Vector{T}(undef, nb),
                       Vector{T}(undef, nb), Vector{T}(undef, nb),
                       i == 1 ? innerA : _inner_clone(innerA),
                       i == 1 ? innerB : _inner_clone(innerB)) for i in 1:nt]
    # rows past N2 pad a ragged last group (read, never written; zero)
    nz = Z1 * ng * G
    return FourStepPlan{T,D}(n, N1, N2, S, G, Z1, stA, stB, innerA, innerB,
                             R, b1r, b1i, b2r, b2i, K, hir, hii, lor, loi,
                             zeros(T, nz), zeros(T, nz), ws, nt, scale)
end

# the same algorithm as p on one task (routing can differ by task count, so
# plan_*(...; nthreads=1) need not be)
_serial_copy(p::FFTPlan{T,D}) where {T,D} = _plan(T, p.n, D, p.scale; nthreads=1)
_serial_copy(p::FourStepPlan{T,D}) where {T,D} =
    _fourstep(T, p.n, D, p.scale; N2=p.N2, nthreads=1)

function (p::FourStepPlan{T,D})(y::CVec{T}, x::CVec{T}) where {T,D}
    n = p.n
    length(x) == n || throw(DimensionMismatch("plan length $n, input length $(length(x))"))
    length(y) == n || throw(DimensionMismatch("plan length $n, output length $(length(y))"))
    na, nb = cld(p.N1, p.S), cld(p.N2, p.G)
    GC.@preserve x y begin
        px, py = _rawptr(x), _rawptr(y)
        if p.nt > 1
            # strips (pass A) and groups (pass B) are independent; the
            # barrier between the passes orders z's writes before its reads
            tforeach_chunks((ci, lo, hi) -> _fourstep_a!(p, p.ws[ci], px, lo, hi), na, p.nt)
            tforeach_chunks((ci, lo, hi) -> _fourstep_b!(p, p.ws[ci], py, lo, hi), nb, p.nt)
        else
            _fourstep_a!(p, p.ws[1], px, 1, na)
            _fourstep_b!(p, p.ws[1], py, 1, nb)
        end
    end
    return y
end

(p::FourStepPlan{T})(x::Vector{Complex{T}}) where {T} = p(similar(x), x)

# pass A, strips slo..shi: per strip of S columns j1 = c..c+S-1, gather rows
# j2, transform over j2, twiddle by ω^{j1·k2} = ω^{c·k2}·ω^{l·k2} and store
# into z. Without the twiddle (TW = false) this is the strided pass of a 2-D
# N1×N2 transform (Fs2Plan).
function _fourstep_a!(p::FourStepPlan{T,D}, buf::FsScratch{T}, px::Ptr{T},
                      slo::Int, shi::Int, ::Val{TW}=Val(true)) where {T,D,TW}
    W = pick_width(T)
    sz = sizeof(T)
    N1, N2, S, K = p.N1, p.N2, p.S, p.K
    are, aim, zre, zim = buf.are, buf.aim, p.zre, p.zim
    M1, M2, P1 = _lanes_geom(p.innerA, N2)
    Z1 = p.Z1
    @inbounds for c in S*(slo-1):S:S*shi-1
        # a ragged last strip (columns past N1) masks its loads, so it never
        # reads past x; its spare lanes land in z's row padding
        lim = N1 - c
        # row j2 = a + M1·b goes to slot b + M2·a
        for a in 0:M1-1, b in 0:M2-1, v in 0:W:S-1
            e = c + N1 * (a + M1 * b) + v
            if lim ≥ S
                u = vload(Vec{W,T}, px + sz * 2e)
                w = vload(Vec{W,T}, px + sz * (2e + W))
            else
                u = vload(Vec{W,T}, px + sz * 2e, _cmask(Val(W), v, lim))
                w = vload(Vec{W,T}, px + sz * (2e + W), _cmask(Val(W), v + W ÷ 2, lim))
            end
            r, i = deinterleave2(u, w)
            vstore(r, are, S * (b + M2 * a) + v + 1)
            vstore(i, aim, S * (b + M2 * a) + v + 1)
        end
        cre, cim = _lanes_fft!(buf.innerA, p.stA, are, aim, buf.bre, buf.bim, N2, S, Val(D))
        # output row k2 = b + M2·a sits at slot a + P1·b
        for b in 0:M2-1, a in 0:M1-1
            k2 = b + M2 * a
            slot = a + P1 * b
            if !TW
                for v in 1:W:S
                    vstore(vload(Vec{W,T}, cre, S * slot + v), zre, c + Z1 * k2 + v)
                    vstore(vload(Vec{W,T}, cim, S * slot + v), zim, c + Z1 * k2 + v)
                end
                continue
            end
            hi, lo = divrem(c * k2, K)
            ar = p.hir[hi+1] * p.lor[lo+1] - p.hii[hi+1] * p.loi[lo+1]
            ai = p.hir[hi+1] * p.loi[lo+1] + p.hii[hi+1] * p.lor[lo+1]
            tt, rr = divrem(k2, p.R)
            for v in 1:W:S
                b1r = vload(Vec{W,T}, p.b1r, S * rr + v)
                b1i = vload(Vec{W,T}, p.b1i, S * rr + v)
                b2r = vload(Vec{W,T}, p.b2r, S * tt + v)
                b2i = vload(Vec{W,T}, p.b2i, S * tt + v)
                br = muladd(b1r, b2r, -(b1i * b2i))
                bi = muladd(b1r, b2i, b1i * b2r)
                twr = muladd(ar, br, -(ai * bi))
                twi = muladd(ar, bi, ai * br)
                yr = vload(Vec{W,T}, cre, S * slot + v)
                yi = vload(Vec{W,T}, cim, S * slot + v)
                vstore(muladd(yr, twr, -(yi * twi)), zre, c + Z1 * k2 + v)
                vstore(muladd(yr, twi, yi * twr), zim, c + Z1 * k2 + v)
            end
        end
    end
    return nothing
end

# pass B, groups glo..ghi: per group of G columns k2 = c2..c2+G-1 (one
# contiguous block of z), transpose into lanes, transform over j1, write rows
# k1 of y in natural order
function _fourstep_b!(p::FourStepPlan{T,D}, buf::FsScratch{T}, py::Ptr{T},
                      glo::Int, ghi::Int) where {T,D}
    W = pick_width(T)
    sz = sizeof(T)
    N1, N2, G = p.N1, p.N2, p.G
    are, aim, zre, zim = buf.are, buf.aim, p.zre, p.zim
    sc = Vec{W,T}(p.scale)
    M1, M2, P1 = _lanes_geom(p.innerB, N1)
    Z1 = p.Z1
    @inbounds for c2 in G*(glo-1):G:G*ghi-1
        # a ragged last group reads z's zero rows past N2 and masks its stores
        lim = N2 - c2
        for g0 in 0:W:G-1
            off = Z1 * (c2 + g0)
            # tiles of W rows j1; when W ∤ N1 the last overlaps the previous
            # (same task, identical values)
            for jq in QWin(N1, W)
                j1 = jq - 1
                tr = vtranspose(ntuple(g -> vload(Vec{W,T}, zre, off + Z1 * (g - 1) + j1 + 1), Val(W)))
                ti = vtranspose(ntuple(g -> vload(Vec{W,T}, zim, off + Z1 * (g - 1) + j1 + 1), Val(W)))
                # row j = a + M1·b goes to slot b + M2·a
                b, a = divrem(j1, M1)
                for i in 1:W
                    slot = b + M2 * a
                    vstore(tr[i], are, G * slot + g0 + 1)
                    vstore(ti[i], aim, G * slot + g0 + 1)
                    a += 1
                    a == M1 && (a = 0; b += 1)
                end
            end
        end
        cre, cim = _lanes_fft!(buf.innerB, p.stB, are, aim, buf.bre, buf.bim, N1, G, Val(D))
        # output row k1 = b + M2·a sits at slot a + P1·b
        for b in 0:M2-1, a in 0:M1-1, v in 0:W:G-1
            k1 = b + M2 * a
            slot = a + P1 * b
            r = vload(Vec{W,T}, cre, G * slot + v + 1) * sc
            i = vload(Vec{W,T}, cim, G * slot + v + 1) * sc
            lo, hi = interleave_chunks(r, i, Val(1))
            e = c2 + N2 * k1 + v
            if lim ≥ G
                vstore(lo, py + sz * 2e)
                vstore(hi, py + sz * (2e + W))
            else
                # columns past N2 would be the next output row's
                vstore(lo, py + sz * 2e, _cmask(Val(W), v, lim))
                vstore(hi, py + sz * (2e + W), _cmask(Val(W), v + W ÷ 2, lim))
            end
        end
    end
    return nothing
end

# A 2-D n1×n2 transform (dim 1 contiguous) is a four-step without the
# twiddle: pass A transforms strips of the strided dim 2 straight from the
# interleaved input into z, pass B transforms each contiguous row of z and
# writes interleaved output from its fused final stage. Each pass reads and
# writes memory once, where per-stage strided passes sweep it once per stage.
struct Fs2Plan{T,D}
    fs::FourStepPlan{T,D}      # pass A strips, z and per-task scratch (N1 = n1, N2 = n2)
    rows::Vector{Stage{T}}     # length-n1 chain, small-stride first stage
    scale::T
end

# Smallest complex-data size given the two-pass 2-D path
const _FS2_MIN_BYTES = 1 << 20

# Threaded from 1 MB: two sweeps and syncs instead of one per stage pay when
# threads contend for bandwidth (4t F64: 0.61-0.92 from 256²). Serial from
# 4 MB with whole strips (strip width from _fs_width): 0.81-0.96, but 1.05-1.06
# at 1-2 MB and 1.15 for F32 1000² (32-wide strips, the last 8 used). Only
# while pass A's strips don't nest (the strided dim n2 is fixed, unlike
# four-step's capped N2): 4096² nested 1.02-1.30.
function _use_fs2(::Type{T}, n1::Int, n2::Int, threaded::Bool) where {T}
    bytes = 2 * n1 * n2 * sizeof(T)
    bytes ≥ (threaded ? _FS2_MIN_BYTES : 4_FS2_MIN_BYTES) || return false
    threaded || n1 % (_fs_width(T, n2) * pick_width(T)) == 0 || return false
    4 * pick_width(T) * n2 * sizeof(T) ≤ _FS_INNER_MAX || return false
    _fourstep_split(T, n1 * n2, 1, n2) === nothing && return false
    # pass B's rows end in a fused power-of-two stage writing interleaved output
    f = last(build_stages_cached(T, n1, -1))
    return f in (2, 4, 8) && n1 ÷ f ≥ pick_width(T)
end

function _fs2(::Type{T}, n1::Int, n2::Int, D::Int, scale::T;
              nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat}
    fs = _fourstep(T, n1 * n2, D, one(T); N2=n2, width=_fs_width(T, n2), nthreads)
    rows, _ = build_stages_cached(T, n1, D)
    return Fs2Plan{T,D}(fs, rows, scale)
end

function (p::Fs2Plan{T,D})(y::AbstractArray{Complex{T}}, x::AbstractArray{Complex{T}}) where {T,D}
    fs = p.fs
    na, n2 = cld(fs.N1, fs.S), fs.N2
    GC.@preserve x y begin
        px, py = _rawptr(x), _rawptr(y)
        if fs.nt > 1
            tforeach_chunks((ci, lo, hi) -> _fourstep_a!(fs, fs.ws[ci], px, lo, hi, Val(false)),
                            na, fs.nt)
            tforeach_chunks((ci, lo, hi) -> _fs2_rows!(p, fs.ws[ci], py, lo - 1, hi - 1),
                            n2, fs.nt)
        else
            _fourstep_a!(fs, fs.ws[1], px, 1, na, Val(false))
            _fs2_rows!(p, fs.ws[1], py, 0, n2 - 1)
        end
    end
    return y
end

# pass B, rows rlo..rhi of z: the length-n1 chain in place on the row (with
# the task's scratch as ping-pong partner), its final stage writing y's row
function _fs2_rows!(p::Fs2Plan{T,D}, buf::FsScratch{T}, py::Ptr{T}, rlo::Int, rhi::Int) where {T,D}
    fs = p.fs
    n1, Z1 = fs.N1, fs.Z1
    f = p.rows[end].radix
    chain = view(p.rows, 1:length(p.rows)-1)
    br, bi = view(buf.bre, 1:n1), view(buf.bim, 1:n1)
    sz = sizeof(T)
    @inbounds for k2 in rlo:rhi
        rng = Z1*k2+1:Z1*k2+n1
        cre, cim, _, _ = run_chain!(view(fs.zre, rng), view(fs.zim, rng), br, bi, chain,
                                    n1, 1, 1, Val(D))
        run_last_stage!(py + 2sz * n1 * k2, cre, cim, n1 ÷ f, f, p.scale, Val(D))
    end
    return nothing
end
