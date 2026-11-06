# Non-smooth n with a sizable 7-smooth factor: n = p·s with s the largest
# 7-smooth divisor (so gcd(p, s) = 1), one Cooley–Tukey step with N1 = p,
# N2 = s (j = j1 + p·j2, k = k2 + s·k1). The length-s transforms over j2
# are exact, batched over the p contiguous j1; after the twiddles ω^{j1·k2}
# and a transpose to k2-fastest, the length-p transforms over j1 are one
# batched resampling pass with s lanes, whose output rows k1 of s lanes are
# already natural order. Only the length-p part is approximate, and its
# weight table has p·w entries rather than n·w.

struct CompositePlan{T,D}
    n::Int
    p::Int
    s::Int
    stages::Vector{Stage{T}}   # length-s chain, inner batch p
    rs::Union{RsPass{T},HvPass{T}}   # length-p resampling, s lanes
    K::Int                     # two-level twiddles: ω^e = hi[e ÷ K]·lo[e % K]
    hir::Vector{T}
    hii::Vector{T}
    lor::Vector{T}
    loi::Vector{T}
    lvr::Vector{T}             # lane twiddles lv[W·j1 + l + 1] = ω^{j1·l}
    lvi::Vector{T}
    are::Vector{T}
    aim::Vector{T}
    bre::Vector{T}
    bim::Vector{T}
    scale::T
end

# smallest smooth factor, and smallest n per precision, routed here rather
# than to full-length resampling: measured crossovers (F64 loses at 3.2k
# and wins at 6.5k; F32, whose full-length interpolation is ~3.6× cheaper
# per element than F64's, loses at 16k and wins at 40k)
const _COMPOSITE_MIN_S = 16
_composite_min_n(::Type{T}) where {T} = sizeof(T) ≥ 8 ? 4096 : 32768

# smallest p whose pass is batched HvdH rather than oversampled interpolation:
# measured parity from ~200 to ~2k (halo rows dominate tiny p), ~1.2× at 4k–10k
const _COMPOSITE_HV_MINP = 500

function _smooth_part(n::Int)
    s = 1
    for f in (2, 3, 5, 7)
        while n % (s * f) == 0
            s *= f
        end
    end
    return s
end

_use_composite(::Type{T}, n::Int) where {T} =
    n ≥ _composite_min_n(T) && _smooth_part(n) ≥ _COMPOSITE_MIN_S

function _composite(::Type{T}, n::Int, D::Int,
                    scale::T=default_scale(T, n, D)) where {T<:AbstractFloat}
    s = _smooth_part(n)
    p = n ÷ s
    Tw = promote_type(T, Float64)
    K = isqrt(n - 1) + 1
    nh = cld(n, K)
    hir, hii = Vector{T}(undef, nh), Vector{T}(undef, nh)
    lor, loi = Vector{T}(undef, K), Vector{T}(undef, K)
    for i in 0:nh-1
        si, co = sincospi(Tw(2 * D * i * K) / n)
        hir[i+1] = T(co); hii[i+1] = T(si)
    end
    for i in 0:K-1
        si, co = sincospi(Tw(2 * D * i) / n)
        lor[i+1] = T(co); loi[i+1] = T(si)
    end
    W = pick_width(T)
    lvr, lvi = Vector{T}(undef, W * p), Vector{T}(undef, W * p)
    for j1 in 0:p-1, l in 0:W-1
        si, co = sincospi(Tw(2 * D * (j1 * l)) / n)
        lvr[W*j1+l+1] = T(co); lvi[W*j1+l+1] = T(si)
    end
    stages, _ = build_stages_cached(T, s, D; allow_small=false)
    rs = p ≥ _COMPOSITE_HV_MINP ? _hv_pass(T, p, D, s) : _rs_pass(T, p, D, s)
    return CompositePlan{T,D}(n, p, s, stages, rs, K,
                              hir, hii, lor, loi, lvr, lvi,
                              Vector{T}(undef, n), Vector{T}(undef, n),
                              Vector{T}(undef, n), Vector{T}(undef, n), scale)
end

function (q::CompositePlan{T,D})(y::CVec{T}, x::CVec{T}) where {T,D}
    n, p, s = q.n, q.p, q.s
    length(x) == n || throw(DimensionMismatch("plan length $n, input length $(length(x))"))
    length(y) == n || throw(DimensionMismatch("plan length $n, output length $(length(y))"))
    deinterleave_all!(q.are, q.aim, x)
    cre, cim, ore, oim = run_chain!(q.are, q.aim, q.bre, q.bim, q.stages, s, 1, p, Val(D))
    transpose_planes!(ore, oim, cre, cim, p, s)
    _rs_block!(q.rs, ore, oim, 0, Val(D), q)   # twiddles fused into its deconvolution
    interleave_all!(y, ore, oim, q.scale)
    return y
end

(q::CompositePlan{T})(x::Vector{Complex{T}}) where {T} = q(similar(x), x)

# Small smooth part (2 ≤ s < _COMPOSITE_MIN_S), where a pass batched over s
# lanes can't fill a vector: decimate in time, x_r[j] = x[r + s·j], run s
# single-lane HvdH transforms of the odd length p, and combine as
# X[k1 + p·k2] = Σ_r ω_s^{r·k2} (ω_n^{r·k1} Y_r[k1]) — the twiddled Y_r[k1]
# go to k1 + p·r, so the batched length-s chain writes natural order.
struct SplitPlan{T,D,H}
    n::Int
    p::Int
    s::Int
    hv::H                      # length-p HvdH, unnormalized
    stages::Vector{Stage{T}}   # length-s chain, inner batch p
    K::Int                     # two-level twiddles: ω^e = hi[e ÷ K]·lo[e % K]
    hir::Vector{T}
    hii::Vector{T}
    lor::Vector{T}
    loi::Vector{T}
    lvr::Vector{T}             # lane twiddles lv[W·r + l + 1] = ω^{r·l}
    lvi::Vector{T}
    xs::Vector{Complex{T}}     # the s decimated inputs, contiguous
    ys::Vector{Complex{T}}     # its transform
    are::Vector{T}
    aim::Vector{T}
    bre::Vector{T}
    bim::Vector{T}
    scale::T
end

# even n only (odd n run HvdH at full length). Measured vs full-length
# resampling: F64 noisy parity near 4k–12k, wins from 20k (0.52–0.87); F32
# s = 2 loses at 33k and is at parity from 200k, s ≥ 6 wins from 98k.
function _use_split(::Type{T}, n::Int) where {T}
    s = _smooth_part(n)
    return iseven(n) && s < _COMPOSITE_MIN_S && _use_hvdh(T, n ÷ s) &&
           n ≥ (sizeof(T) ≥ 8 ? 16384 : 65536)
end

function _split(::Type{T}, n::Int, D::Int, scale::T=default_scale(T, n, D);
                nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat}
    s = _smooth_part(n)
    p = n ÷ s
    Tw = promote_type(T, Float64)
    K = isqrt(n - 1) + 1
    nh = cld(n, K)
    hir, hii = Vector{T}(undef, nh), Vector{T}(undef, nh)
    lor, loi = Vector{T}(undef, K), Vector{T}(undef, K)
    for i in 0:nh-1
        si, co = sincospi(Tw(2 * D * i * K) / n)
        hir[i+1] = T(co); hii[i+1] = T(si)
    end
    for i in 0:K-1
        si, co = sincospi(Tw(2 * D * i) / n)
        lor[i+1] = T(co); loi[i+1] = T(si)
    end
    W = pick_width(T)
    lvr, lvi = Vector{T}(undef, W * s), Vector{T}(undef, W * s)
    for r in 0:s-1, l in 0:W-1
        si, co = sincospi(Tw(2 * D * (r * l)) / n)
        lvr[W*r+l+1] = T(co); lvi[W*r+l+1] = T(si)
    end
    # threads inside each sub-transform; the decimate/twiddle/combine passes
    # are O(n) and stay serial
    hv = _hvdh(T, p, D, one(T); nthreads)
    stages, _ = build_stages_cached(T, s, D; allow_small=false)
    return SplitPlan{T,D,typeof(hv)}(n, p, s, hv, stages, K, hir, hii, lor, loi, lvr, lvi,
                                     Vector{Complex{T}}(undef, n), Vector{Complex{T}}(undef, p),
                                     Vector{T}(undef, n), Vector{T}(undef, n),
                                     Vector{T}(undef, n), Vector{T}(undef, n), scale)
end

function (q::SplitPlan{T,D})(y::CVec{T}, x::CVec{T}) where {T,D}
    n, p, s = q.n, q.p, q.s
    length(x) == n || throw(DimensionMismatch("plan length $n, input length $(length(x))"))
    length(y) == n || throw(DimensionMismatch("plan length $n, output length $(length(y))"))
    xs, ys, are, aim = q.xs, q.ys, q.are, q.aim
    # one sequential pass over x into the s decimated sequences
    @inbounds for j in 0:p-1, r in 0:s-1
        xs[r*p+j+1] = x[s*j+r+1]
    end
    @inbounds for r in 0:s-1
        q.hv(ys, view(xs, r*p+1:(r+1)*p))
        _split_twiddle!(are, aim, p * r, ys, q, r)
    end
    cre, cim, _, _ = run_chain!(are, aim, q.bre, q.bim, q.stages, s, 1, p, Val(D))
    interleave_all!(y, cre, cim, q.scale)
    return y
end

(q::SplitPlan{T})(x::Vector{Complex{T}}) where {T} = q(similar(x), x)

plan_nt(q::SplitPlan) = plan_nt(q.hv)

# planes at o + k1 ← ω^{r·k1}·ys[k1], per W-lane window as
# ω^{r·k1₀}·ω^{r·l}; windows may overlap (out of place)
function _split_twiddle!(are::Vector{T}, aim::Vector{T}, o::Int, ys::Vector{Complex{T}},
                         q::SplitPlan{T}, r::Int) where {T}
    p = q.p
    W = pick_width(T)
    sz = sizeof(T)
    @inbounds if W > 1 && p ≥ W
        lvr = vload(Vec{W,T}, q.lvr, W * r + 1)
        lvi = vload(Vec{W,T}, q.lvi, W * r + 1)
        GC.@preserve ys begin
            py = _rawptr(ys)
            for k in QWin(p, W)
                ar, ai = _tw2(q, r * (k - 1))
                twr = muladd(ar, lvr, -(ai * lvi))
                twi = muladd(ar, lvi, ai * lvr)
                zr, zi = deinterleave2(vload(Vec{W,T}, py + 2sz * (k - 1)),
                                       vload(Vec{W,T}, py + sz * (2(k - 1) + W)))
                vstore(muladd(zr, twr, -(zi * twi)), are, o + k)
                vstore(muladd(zr, twi, zi * twr), aim, o + k)
            end
        end
    else
        for k1 in 0:p-1
            wr, wi = _tw2(q, r * k1)
            z = ys[k1+1]
            are[o+k1+1] = real(z) * wr - imag(z) * wi
            aim[o+k1+1] = real(z) * wi + imag(z) * wr
        end
    end
    return nothing
end

@inline function _tw2(q::Union{CompositePlan,SplitPlan}, e::Int)
    hi, lo = divrem(e, q.K)
    @inbounds return (q.hir[hi+1] * q.lor[lo+1] - q.hii[hi+1] * q.loi[lo+1],
                      q.hir[hi+1] * q.loi[lo+1] + q.hii[hi+1] * q.lor[lo+1])
end

# the resampling pass's deconvolution with the twiddles folded in: after the
# transpose, row j1 lane k2 = lane0 + i - 1 takes c·ω^{j1·k2}, built per
# W-lane window as (c·ω^{j1·k2₀})·ω^{j1·l}; windows may overlap (out of place)
@inline function _rows_scale_tw!(yre, yim, yo, xre, xim, xo, c::T, L,
                                 q::CompositePlan{T}, j1::Int, lane0::Int) where {T}
    W = pick_width(T)
    @inbounds if W > 1 && L ≥ W
        lvr = vload(Vec{W,T}, q.lvr, W * j1 + 1)
        lvi = vload(Vec{W,T}, q.lvi, W * j1 + 1)
        for i in QWin(L, W)
            ar, ai = _tw2(q, j1 * (lane0 + i - 1))
            twr = muladd(c * ar, lvr, -((c * ai) * lvi))
            twi = muladd(c * ar, lvi, (c * ai) * lvr)
            xr = vload(Vec{W,T}, xre, xo + i)
            xi = vload(Vec{W,T}, xim, xo + i)
            vstore(muladd(xr, twr, -(xi * twi)), yre, yo + i)
            vstore(muladd(xr, twi, xi * twr), yim, yo + i)
        end
    else
        for i in 1:L
            wr, wi = _tw2(q, j1 * (lane0 + i - 1))
            xr, xi = xre[xo+i], xim[xo+i]
            yre[yo+i] = c * (xr * wr - xi * wi)
            yim[yo+i] = c * (xr * wi + xi * wr)
        end
    end
    return nothing
end
