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

# (N1, N2) for a 7-smooth n, or nothing when the split doesn't tile into
# width·W strips. With N2 = 0, a balanced power-of-two N2 (the odd part goes
# to N1), capped so pass A doesn't nest: one nested pass measured ~20%
# faster than two. The cap stops at half balanced so pass B's buffers keep
# growing as √n.
function _fourstep_split(::Type{T}, n::Int, width::Int, N2::Int=0) where {T}
    W = pick_width(T)
    fac = factor_smooth(n)
    (W > 1 && fac !== nothing) || return nothing
    if N2 == 0
        nflat = _FS_INNER_MAX ÷ (4 * width * W * sizeof(T))
        Nb = 1 << (fac[1] ÷ 2)
        N2 = max(min(Nb, prevpow(2, max(nflat, W))), Nb ÷ 2)
    end
    N1 = n ÷ N2
    S = width * W
    ok = N1 * N2 == n && N1 % S == 0 && N2 % S == 0
    return ok ? (N1, N2) : nothing
end

# Smallest complex-data size routed to four-step: measured slower at 1 MB
# and faster at 2 MB in both precisions.
const _FS_MIN_BYTES = 3 << 19

_use_fourstep(::Type{T}, n::Int) where {T} =
    2 * n * sizeof(T) ≥ _FS_MIN_BYTES && _fourstep_split(T, n, 1) !== nothing

function _fourstep(::Type{T}, n::Int, D::Int, scale::T=default_scale(T, n, D);
                   N2::Int=0, width::Int=1, zpad::Int=pick_width(T),
                   ipad::Int=1, nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat}
    W = pick_width(T)
    split = _fourstep_split(T, n, width, N2)
    split === nothing &&
        throw(ArgumentError("no four-step split of $n tiles into width-$(width * W) strips"))
    N1, N2 = split
    S = G = width * W
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
    Z1 = N1 + zpad
    innerA = _inner_fourstep(T, N2, S, D, ipad)
    innerB = _inner_fourstep(T, N1, G, D, ipad)
    nb = max(S * _lanes_rows(innerA, N2), G * _lanes_rows(innerB, N1))
    nt = min(plan_tasks(n, nthreads), N1 ÷ S, N2 ÷ G)
    ws = [FsScratch{T}(Vector{T}(undef, nb), Vector{T}(undef, nb),
                       Vector{T}(undef, nb), Vector{T}(undef, nb),
                       i == 1 ? innerA : _inner_clone(innerA),
                       i == 1 ? innerB : _inner_clone(innerB)) for i in 1:nt]
    return FourStepPlan{T,D}(n, N1, N2, S, G, Z1, stA, stB, innerA, innerB,
                             R, b1r, b1i, b2r, b2i, K, hir, hii, lor, loi,
                             Vector{T}(undef, Z1 * N2), Vector{T}(undef, Z1 * N2),
                             ws, nt, scale)
end

function (p::FourStepPlan{T,D})(y::CVec{T}, x::CVec{T}) where {T,D}
    n = p.n
    length(x) == n || throw(DimensionMismatch("plan length $n, input length $(length(x))"))
    length(y) == n || throw(DimensionMismatch("plan length $n, output length $(length(y))"))
    na, nb = p.N1 ÷ p.S, p.N2 ÷ p.G
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
# into z
function _fourstep_a!(p::FourStepPlan{T,D}, buf::FsScratch{T}, px::Ptr{T},
                      slo::Int, shi::Int) where {T,D}
    W = pick_width(T)
    sz = sizeof(T)
    N1, N2, S, K = p.N1, p.N2, p.S, p.K
    are, aim, zre, zim = buf.are, buf.aim, p.zre, p.zim
    M1, M2, P1 = _lanes_geom(p.innerA, N2)
    Z1 = p.Z1
    @inbounds for c in S*(slo-1):S:S*shi-1
        # row j2 = a + M1·b goes to slot b + M2·a
        for a in 0:M1-1, b in 0:M2-1, v in 0:W:S-1
            e = c + N1 * (a + M1 * b) + v
            u = vload(Vec{W,T}, px + sz * 2e)
            w = vload(Vec{W,T}, px + sz * (2e + W))
            r, i = deinterleave2(u, w)
            vstore(r, are, S * (b + M2 * a) + v + 1)
            vstore(i, aim, S * (b + M2 * a) + v + 1)
        end
        cre, cim = _lanes_fft!(buf.innerA, p.stA, are, aim, buf.bre, buf.bim, N2, S, Val(D))
        # output row k2 = b + M2·a sits at slot a + P1·b
        for b in 0:M2-1, a in 0:M1-1
            k2 = b + M2 * a
            slot = a + P1 * b
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
        for g0 in 0:W:G-1
            off = Z1 * (c2 + g0)
            for j1 in 0:W:N1-1
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
            vstore(lo, py + sz * 2e)
            vstore(hi, py + sz * (2e + W))
        end
    end
    return nothing
end
