# Stockham autosort passes. Each stage transforms x -> y out-of-place with the
# reordering permutation absorbed into the output indexing, so there is no
# bit-reversal step. n is the current sub-transform length, s the number of
# interleaved sub-transforms (butterfly stride); s*n == total length at every
# stage. Twiddles are precomputed in access order (radix-1 per butterfly).

# W-wide windows covering 1:s; when W ∤ s the last window overlaps the previous
# one, which is safe: within a butterfly p, lanes are independent and recomputed
# lanes store identical values. Requires s ≥ W.
struct QWin
    s::Int
    W::Int
end
@inline function Base.iterate(qw::QWin, q::Int=1)
    q > qw.s && return nothing
    q + qw.W - 1 ≤ qw.s ? (q, q + qw.W) : (qw.s - qw.W + 1, qw.s + 1)
end
Base.IteratorSize(::Type{QWin}) = Base.SizeUnknown()
Base.eltype(::Type{QWin}) = Int

function stage4!(yre::Vector{T}, yim::Vector{T}, xre::Vector{T}, xim::Vector{T},
                 n::Int, s::Int, twr::Vector{T}, twi::Vector{T}, d::Val{D}) where {T,D}
    m = n >>> 2
    ms = m * s
    W = pick_width(T)
    if W > 1 && s ≥ W
        @inbounds for p in 0:m-1
            w1r = Vec{W,T}(twr[3p+1]); w1i = Vec{W,T}(twi[3p+1])
            w2r = Vec{W,T}(twr[3p+2]); w2i = Vec{W,T}(twi[3p+2])
            w3r = Vec{W,T}(twr[3p+3]); w3i = Vec{W,T}(twi[3p+3])
            xb = s * p
            yb = 4 * s * p
            for q in QWin(s, W)
                ar = vload(Vec{W,T}, xre, xb + q);        ai = vload(Vec{W,T}, xim, xb + q)
                br = vload(Vec{W,T}, xre, xb + ms + q);   bi = vload(Vec{W,T}, xim, xb + ms + q)
                cr = vload(Vec{W,T}, xre, xb + 2ms + q);  ci = vload(Vec{W,T}, xim, xb + 2ms + q)
                dr = vload(Vec{W,T}, xre, xb + 3ms + q);  di = vload(Vec{W,T}, xim, xb + 3ms + q)
                y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i =
                    bf4(ar, ai, br, bi, cr, ci, dr, di,
                        w1r, w1i, w2r, w2i, w3r, w3i, d)
                vstore(y0r, yre, yb + q);      vstore(y0i, yim, yb + q)
                vstore(y1r, yre, yb + s + q);  vstore(y1i, yim, yb + s + q)
                vstore(y2r, yre, yb + 2s + q); vstore(y2i, yim, yb + 2s + q)
                vstore(y3r, yre, yb + 3s + q); vstore(y3i, yim, yb + 3s + q)
            end
        end
    else
        @inbounds for p in 0:m-1
            w1r = twr[3p+1]; w1i = twi[3p+1]
            w2r = twr[3p+2]; w2i = twi[3p+2]
            w3r = twr[3p+3]; w3i = twi[3p+3]
            xb = s * p
            yb = 4 * s * p
            for q in 1:s
                y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i =
                    bf4(xre[xb+q], xim[xb+q],
                        xre[xb+ms+q], xim[xb+ms+q],
                        xre[xb+2ms+q], xim[xb+2ms+q],
                        xre[xb+3ms+q], xim[xb+3ms+q],
                        w1r, w1i, w2r, w2i, w3r, w3i, d)
                yre[yb+q] = y0r;      yim[yb+q] = y0i
                yre[yb+s+q] = y1r;    yim[yb+s+q] = y1i
                yre[yb+2s+q] = y2r;   yim[yb+2s+q] = y2i
                yre[yb+3s+q] = y3r;   yim[yb+3s+q] = y3i
            end
        end
    end
    return nothing
end

# Small-stride radix-4 stage (s < W): vectorize across W÷s butterfly groups by
# loading W consecutive flat elements from each quarter; the strided output
# permutation becomes a static register interleave. Twiddles are pre-expanded
# per lane: twr/twi hold w1, w2, w3 each over the m*s flat indices.
function stage4_small!(yre::Vector{T}, yim::Vector{T}, xre::Vector{T}, xim::Vector{T},
                       n::Int, ::Val{S}, twr::Vector{T}, twi::Vector{T},
                       d::Val{D}) where {T,S,D}
    W = pick_width(T)
    m = n >>> 2
    ms = m * S
    @inbounds for f in 1:W:ms
        ar = vload(Vec{W,T}, xre, f);         ai = vload(Vec{W,T}, xim, f)
        br = vload(Vec{W,T}, xre, ms + f);    bi = vload(Vec{W,T}, xim, ms + f)
        cr = vload(Vec{W,T}, xre, 2ms + f);   ci = vload(Vec{W,T}, xim, 2ms + f)
        dr = vload(Vec{W,T}, xre, 3ms + f);   di = vload(Vec{W,T}, xim, 3ms + f)
        w1r = vload(Vec{W,T}, twr, f);        w1i = vload(Vec{W,T}, twi, f)
        w2r = vload(Vec{W,T}, twr, ms + f);   w2i = vload(Vec{W,T}, twi, ms + f)
        w3r = vload(Vec{W,T}, twr, 2ms + f);  w3i = vload(Vec{W,T}, twi, 2ms + f)
        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i =
            bf4(ar, ai, br, bi, cr, ci, dr, di,
                w1r, w1i, w2r, w2i, w3r, w3i, d)
        o0r, o1r, o2r, o3r = interleave4(y0r, y1r, y2r, y3r, Val(S))
        o0i, o1i, o2i, o3i = interleave4(y0i, y1i, y2i, y3i, Val(S))
        base = 4 * (f - 1)
        vstore(o0r, yre, base + 1);      vstore(o0i, yim, base + 1)
        vstore(o1r, yre, base + W + 1);  vstore(o1i, yim, base + W + 1)
        vstore(o2r, yre, base + 2W + 1); vstore(o2i, yim, base + 2W + 1)
        vstore(o3r, yre, base + 3W + 1); vstore(o3i, yim, base + 3W + 1)
    end
    return nothing
end

# First stage (radix 4, s == 1) reading directly from the interleaved complex
# input: two vector loads + a deinterleave shuffle replace the copy pass.
# Twiddles use the expanded small-stage layout.
function stage4_first!(yre::Vector{T}, yim::Vector{T}, px::Ptr{T}, n::Int,
                       twr::Vector{T}, twi::Vector{T}, d::Val{D}) where {T,D}
    W = pick_width(T)
    m = n >>> 2
    sz = sizeof(T)
    @inbounds for f in 1:W:m
        b0 = 2 * (f - 1)
        u = vload(Vec{W,T}, px + sz * b0);            v = vload(Vec{W,T}, px + sz * (b0 + W))
        ar, ai = deinterleave2(u, v)
        u = vload(Vec{W,T}, px + sz * (b0 + 2m));     v = vload(Vec{W,T}, px + sz * (b0 + 2m + W))
        br, bi = deinterleave2(u, v)
        u = vload(Vec{W,T}, px + sz * (b0 + 4m));     v = vload(Vec{W,T}, px + sz * (b0 + 4m + W))
        cr, ci = deinterleave2(u, v)
        u = vload(Vec{W,T}, px + sz * (b0 + 6m));     v = vload(Vec{W,T}, px + sz * (b0 + 6m + W))
        dr, di = deinterleave2(u, v)
        w1r = vload(Vec{W,T}, twr, f);       w1i = vload(Vec{W,T}, twi, f)
        w2r = vload(Vec{W,T}, twr, m + f);   w2i = vload(Vec{W,T}, twi, m + f)
        w3r = vload(Vec{W,T}, twr, 2m + f);  w3i = vload(Vec{W,T}, twi, 2m + f)
        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i =
            bf4(ar, ai, br, bi, cr, ci, dr, di,
                w1r, w1i, w2r, w2i, w3r, w3i, d)
        o0r, o1r, o2r, o3r = interleave4(y0r, y1r, y2r, y3r, Val(1))
        o0i, o1i, o2i, o3i = interleave4(y0i, y1i, y2i, y3i, Val(1))
        base = 4 * (f - 1)
        vstore(o0r, yre, base + 1);      vstore(o0i, yim, base + 1)
        vstore(o1r, yre, base + W + 1);  vstore(o1i, yim, base + W + 1)
        vstore(o2r, yre, base + 2W + 1); vstore(o2i, yim, base + 2W + 1)
        vstore(o3r, yre, base + 3W + 1); vstore(o3i, yim, base + 3W + 1)
    end
    return nothing
end

# Last stage (m == 1, so twiddle-free) writing scaled interleaved complex
# output directly: butterfly, scale, re/im interleave shuffle, store.
function stage4_last!(py::Ptr{T}, xre::Vector{T}, xim::Vector{T}, s::Int,
                      scale::T, d::Val{D}) where {T,D}
    W = pick_width(T)
    sc = Vec{W,T}(scale)
    sz = sizeof(T)
    @inbounds for q in 1:W:s
        ar = vload(Vec{W,T}, xre, q);       ai = vload(Vec{W,T}, xim, q)
        br = vload(Vec{W,T}, xre, s + q);   bi = vload(Vec{W,T}, xim, s + q)
        cr = vload(Vec{W,T}, xre, 2s + q);  ci = vload(Vec{W,T}, xim, 2s + q)
        dr = vload(Vec{W,T}, xre, 3s + q);  di = vload(Vec{W,T}, xim, 3s + q)
        y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i =
            bf4_core(ar, ai, br, bi, cr, ci, dr, di, d)
        lo, hi = interleave_chunks(y0r * sc, y0i * sc, Val(1))
        vstore(lo, py + sz * 2 * (q - 1)); vstore(hi, py + sz * (2 * (q - 1) + W))
        lo, hi = interleave_chunks(y1r * sc, y1i * sc, Val(1))
        vstore(lo, py + sz * 2 * (s + q - 1)); vstore(hi, py + sz * (2 * (s + q - 1) + W))
        lo, hi = interleave_chunks(y2r * sc, y2i * sc, Val(1))
        vstore(lo, py + sz * 2 * (2s + q - 1)); vstore(hi, py + sz * (2 * (2s + q - 1) + W))
        lo, hi = interleave_chunks(y3r * sc, y3i * sc, Val(1))
        vstore(lo, py + sz * 2 * (3s + q - 1)); vstore(hi, py + sz * (2 * (3s + q - 1) + W))
    end
    return nothing
end

# Radix-8 final stage (m == 1, twiddle-free), fused interleaved output. Used
# for odd log2 n so the extra radix-2 pass disappears.
function stage8_last!(py::Ptr{T}, xre::Vector{T}, xim::Vector{T}, s::Int,
                      scale::T, d::Val{D}) where {T,D}
    W = pick_width(T)
    sc = Vec{W,T}(scale)
    c = Vec{W,T}(T(sqrt(0.5)))
    sz = sizeof(T)
    @inbounds for q in 1:W:s
        x0r = vload(Vec{W,T}, xre, q);       x0i = vload(Vec{W,T}, xim, q)
        x1r = vload(Vec{W,T}, xre, s + q);   x1i = vload(Vec{W,T}, xim, s + q)
        x2r = vload(Vec{W,T}, xre, 2s + q);  x2i = vload(Vec{W,T}, xim, 2s + q)
        x3r = vload(Vec{W,T}, xre, 3s + q);  x3i = vload(Vec{W,T}, xim, 3s + q)
        x4r = vload(Vec{W,T}, xre, 4s + q);  x4i = vload(Vec{W,T}, xim, 4s + q)
        x5r = vload(Vec{W,T}, xre, 5s + q);  x5i = vload(Vec{W,T}, xim, 5s + q)
        x6r = vload(Vec{W,T}, xre, 6s + q);  x6i = vload(Vec{W,T}, xim, 6s + q)
        x7r = vload(Vec{W,T}, xre, 7s + q);  x7i = vload(Vec{W,T}, xim, 7s + q)
        ys = bf8_core(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                      x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i, c, d)
        Base.Cartesian.@nexprs 8 j -> begin
            lo, hi = interleave_chunks(ys[2j-1] * sc, ys[2j] * sc, Val(1))
            vstore(lo, py + sz * 2 * ((j - 1) * s + q - 1))
            vstore(hi, py + sz * (2 * ((j - 1) * s + q - 1) + W))
        end
    end
    return nothing
end

# Radix-8 final stage writing split planes (unfused fallback); butterfly
# partners at stride s, outputs land at the same offsets.
function stage8_plain!(yre::Vector{T}, yim::Vector{T}, xre::Vector{T}, xim::Vector{T},
                       s::Int, d::Val{D}) where {T,D}
    W = pick_width(T)
    if W > 1 && s ≥ W
        c = Vec{W,T}(T(sqrt(0.5)))
        @inbounds for q in QWin(s, W)
            Base.Cartesian.@nexprs 8 j -> begin
                xr_j = vload(Vec{W,T}, xre, (j - 1) * s + q)
                xi_j = vload(Vec{W,T}, xim, (j - 1) * s + q)
            end
            ys = bf8_core(xr_1, xi_1, xr_2, xi_2, xr_3, xi_3, xr_4, xi_4,
                          xr_5, xi_5, xr_6, xi_6, xr_7, xi_7, xr_8, xi_8, c, d)
            Base.Cartesian.@nexprs 8 j -> begin
                vstore(ys[2j-1], yre, (j - 1) * s + q)
                vstore(ys[2j], yim, (j - 1) * s + q)
            end
        end
    else
        c = T(sqrt(0.5))
        @inbounds for q in 1:s
            Base.Cartesian.@nexprs 8 j -> begin
                xr_j = xre[(j-1)*s+q]
                xi_j = xim[(j-1)*s+q]
            end
            ys = bf8_core(xr_1, xi_1, xr_2, xi_2, xr_3, xi_3, xr_4, xi_4,
                          xr_5, xi_5, xr_6, xi_6, xr_7, xi_7, xr_8, xi_8, c, d)
            Base.Cartesian.@nexprs 8 j -> begin
                yre[(j-1)*s+q] = ys[2j-1]
                yim[(j-1)*s+q] = ys[2j]
            end
        end
    end
    return nothing
end

# radix-2 variant of the fused last stage (odd log2 n)
function stage2_last!(py::Ptr{T}, xre::Vector{T}, xim::Vector{T}, s::Int,
                      scale::T) where {T}
    W = pick_width(T)
    sc = Vec{W,T}(scale)
    sz = sizeof(T)
    @inbounds for q in 1:W:s
        ar = vload(Vec{W,T}, xre, q);      ai = vload(Vec{W,T}, xim, q)
        br = vload(Vec{W,T}, xre, s + q);  bi = vload(Vec{W,T}, xim, s + q)
        lo, hi = interleave_chunks((ar + br) * sc, (ai + bi) * sc, Val(1))
        vstore(lo, py + sz * 2 * (q - 1)); vstore(hi, py + sz * (2 * (q - 1) + W))
        lo, hi = interleave_chunks((ar - br) * sc, (ai - bi) * sc, Val(1))
        vstore(lo, py + sz * 2 * (s + q - 1)); vstore(hi, py + sz * (2 * (s + q - 1) + W))
    end
    return nothing
end

# Odd-radix stages. These run after the radix-4 tower, so s is large for all
# but tiny n (scalar fallback otherwise).
function stage3!(yre::Vector{T}, yim::Vector{T}, xre::Vector{T}, xim::Vector{T},
                 n::Int, s::Int, twr::Vector{T}, twi::Vector{T}, ::Val{D}) where {T,D}
    m = n ÷ 3
    ms = m * s
    k = T(D * sqrt(0.75))
    nh = T(-0.5)
    W = pick_width(T)
    if W > 1 && s ≥ W
        kv = Vec{W,T}(k); nhv = Vec{W,T}(nh)
        @inbounds for p in 0:m-1
            w1r = Vec{W,T}(twr[2p+1]); w1i = Vec{W,T}(twi[2p+1])
            w2r = Vec{W,T}(twr[2p+2]); w2i = Vec{W,T}(twi[2p+2])
            xb = s * p; yb = 3 * s * p
            for q in QWin(s, W)
                ys = bf3(vload(Vec{W,T}, xre, xb + q), vload(Vec{W,T}, xim, xb + q),
                         vload(Vec{W,T}, xre, xb + ms + q), vload(Vec{W,T}, xim, xb + ms + q),
                         vload(Vec{W,T}, xre, xb + 2ms + q), vload(Vec{W,T}, xim, xb + 2ms + q),
                         kv, nhv)
                vstore(ys[1], yre, yb + q); vstore(ys[2], yim, yb + q)
                o1r, o1i = cmul(w1r, w1i, ys[3], ys[4])
                vstore(o1r, yre, yb + s + q); vstore(o1i, yim, yb + s + q)
                o2r, o2i = cmul(w2r, w2i, ys[5], ys[6])
                vstore(o2r, yre, yb + 2s + q); vstore(o2i, yim, yb + 2s + q)
            end
        end
    else
        @inbounds for p in 0:m-1
            w1r = twr[2p+1]; w1i = twi[2p+1]
            w2r = twr[2p+2]; w2i = twi[2p+2]
            xb = s * p; yb = 3 * s * p
            for q in 1:s
                ys = bf3(xre[xb+q], xim[xb+q], xre[xb+ms+q], xim[xb+ms+q],
                         xre[xb+2ms+q], xim[xb+2ms+q], k, nh)
                yre[yb+q] = ys[1]; yim[yb+q] = ys[2]
                o1r, o1i = cmul(w1r, w1i, ys[3], ys[4])
                yre[yb+s+q] = o1r; yim[yb+s+q] = o1i
                o2r, o2i = cmul(w2r, w2i, ys[5], ys[6])
                yre[yb+2s+q] = o2r; yim[yb+2s+q] = o2i
            end
        end
    end
    return nothing
end

function stage5!(yre::Vector{T}, yim::Vector{T}, xre::Vector{T}, xim::Vector{T},
                 n::Int, s::Int, twr::Vector{T}, twi::Vector{T}, ::Val{D}) where {T,D}
    m = n ÷ 5
    ms = m * s
    c1 = T(cospi(0.4)); c2 = T(cospi(0.8))
    s1 = T(D * sinpi(0.4)); s2 = T(D * sinpi(0.8))
    W = pick_width(T)
    if W > 1 && s ≥ W
        c1v = Vec{W,T}(c1); c2v = Vec{W,T}(c2); s1v = Vec{W,T}(s1); s2v = Vec{W,T}(s2)
        @inbounds for p in 0:m-1
            Base.Cartesian.@nexprs 4 j -> begin
                wr_j = Vec{W,T}(twr[4p+j]); wi_j = Vec{W,T}(twi[4p+j])
            end
            xb = s * p; yb = 5 * s * p
            for q in QWin(s, W)
                Base.Cartesian.@nexprs 5 j -> begin
                    xr_j = vload(Vec{W,T}, xre, xb + (j - 1) * ms + q)
                    xi_j = vload(Vec{W,T}, xim, xb + (j - 1) * ms + q)
                end
                ys = bf5(xr_1, xi_1, xr_2, xi_2, xr_3, xi_3, xr_4, xi_4, xr_5, xi_5,
                         c1v, c2v, s1v, s2v)
                vstore(ys[1], yre, yb + q); vstore(ys[2], yim, yb + q)
                Base.Cartesian.@nexprs 4 j -> begin
                    or_j, oi_j = cmul(wr_j, wi_j, ys[2j+1], ys[2j+2])
                    vstore(or_j, yre, yb + j * s + q); vstore(oi_j, yim, yb + j * s + q)
                end
            end
        end
    else
        @inbounds for p in 0:m-1
            Base.Cartesian.@nexprs 4 j -> begin
                wr_j = twr[4p+j]; wi_j = twi[4p+j]
            end
            xb = s * p; yb = 5 * s * p
            for q in 1:s
                Base.Cartesian.@nexprs 5 j -> begin
                    xr_j = xre[xb+(j-1)*ms+q]
                    xi_j = xim[xb+(j-1)*ms+q]
                end
                ys = bf5(xr_1, xi_1, xr_2, xi_2, xr_3, xi_3, xr_4, xi_4, xr_5, xi_5,
                         c1, c2, s1, s2)
                yre[yb+q] = ys[1]; yim[yb+q] = ys[2]
                Base.Cartesian.@nexprs 4 j -> begin
                    or_j, oi_j = cmul(wr_j, wi_j, ys[2j+1], ys[2j+2])
                    yre[yb+j*s+q] = or_j; yim[yb+j*s+q] = oi_j
                end
            end
        end
    end
    return nothing
end

bf7_consts(::Type{T}, D::Int) where {T} =
    (ntuple(i -> T(cospi(2 * (((i - 1) % 3 + 1) * ((i - 1) ÷ 3 + 1)) / 7)), 9),
     ntuple(i -> T(D * sinpi(2 * (((i - 1) % 3 + 1) * ((i - 1) ÷ 3 + 1)) / 7)), 9))

function stage7!(yre::Vector{T}, yim::Vector{T}, xre::Vector{T}, xim::Vector{T},
                 n::Int, s::Int, twr::Vector{T}, twi::Vector{T}, ::Val{D}) where {T,D}
    m = n ÷ 7
    ms = m * s
    cc, ss = bf7_consts(T, D)
    W = pick_width(T)
    if W > 1 && s ≥ W
        ccv = map(Vec{W,T}, cc); ssv = map(Vec{W,T}, ss)
        @inbounds for p in 0:m-1
            Base.Cartesian.@nexprs 6 j -> begin
                wr_j = Vec{W,T}(twr[6p+j]); wi_j = Vec{W,T}(twi[6p+j])
            end
            xb = s * p; yb = 7 * s * p
            for q in QWin(s, W)
                Base.Cartesian.@nexprs 7 j -> begin
                    xr_j = vload(Vec{W,T}, xre, xb + (j - 1) * ms + q)
                    xi_j = vload(Vec{W,T}, xim, xb + (j - 1) * ms + q)
                end
                ys = bf7(xr_1, xi_1, xr_2, xi_2, xr_3, xi_3, xr_4, xi_4,
                         xr_5, xi_5, xr_6, xi_6, xr_7, xi_7, ccv, ssv)
                vstore(ys[1], yre, yb + q); vstore(ys[2], yim, yb + q)
                Base.Cartesian.@nexprs 6 j -> begin
                    or_j, oi_j = cmul(wr_j, wi_j, ys[2j+1], ys[2j+2])
                    vstore(or_j, yre, yb + j * s + q); vstore(oi_j, yim, yb + j * s + q)
                end
            end
        end
    else
        @inbounds for p in 0:m-1
            Base.Cartesian.@nexprs 6 j -> begin
                wr_j = twr[6p+j]; wi_j = twi[6p+j]
            end
            xb = s * p; yb = 7 * s * p
            for q in 1:s
                Base.Cartesian.@nexprs 7 j -> begin
                    xr_j = xre[xb+(j-1)*ms+q]
                    xi_j = xim[xb+(j-1)*ms+q]
                end
                ys = bf7(xr_1, xi_1, xr_2, xi_2, xr_3, xi_3, xr_4, xi_4,
                         xr_5, xi_5, xr_6, xi_6, xr_7, xi_7, cc, ss)
                yre[yb+q] = ys[1]; yim[yb+q] = ys[2]
                Base.Cartesian.@nexprs 6 j -> begin
                    or_j, oi_j = cmul(wr_j, wi_j, ys[2j+1], ys[2j+2])
                    yre[yb+j*s+q] = or_j; yim[yb+j*s+q] = oi_j
                end
            end
        end
    end
    return nothing
end

function stage2!(yre::Vector{T}, yim::Vector{T}, xre::Vector{T}, xim::Vector{T},
                 n::Int, s::Int, twr::Vector{T}, twi::Vector{T}) where {T}
    m = n >>> 1
    ms = m * s
    W = pick_width(T)
    if W > 1 && s ≥ W
        @inbounds for p in 0:m-1
            wr = Vec{W,T}(twr[p+1]); wi = Vec{W,T}(twi[p+1])
            xb = s * p
            yb = 2 * s * p
            for q in QWin(s, W)
                ar = vload(Vec{W,T}, xre, xb + q);      ai = vload(Vec{W,T}, xim, xb + q)
                br = vload(Vec{W,T}, xre, xb + ms + q); bi = vload(Vec{W,T}, xim, xb + ms + q)
                y0r, y0i, y1r, y1i = bf2(ar, ai, br, bi, wr, wi)
                vstore(y0r, yre, yb + q);     vstore(y0i, yim, yb + q)
                vstore(y1r, yre, yb + s + q); vstore(y1i, yim, yb + s + q)
            end
        end
    else
        @inbounds for p in 0:m-1
            wr = twr[p+1]; wi = twi[p+1]
            xb = s * p
            yb = 2 * s * p
            for q in 1:s
                y0r, y0i, y1r, y1i =
                    bf2(xre[xb+q], xim[xb+q], xre[xb+ms+q], xim[xb+ms+q], wr, wi)
                yre[yb+q] = y0r;    yim[yb+q] = y0i
                yre[yb+s+q] = y1r;  yim[yb+s+q] = y1i
            end
        end
    end
    return nothing
end
