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
    rs::RsPass{T}              # length-p resampling, s lanes
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
    return CompositePlan{T,D}(n, p, s, stages, _rs_pass(T, p, D, s), K,
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

@inline function _tw2(q::CompositePlan, e::Int)
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
