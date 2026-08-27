# Radix butterfly kernels on split re/im values. Generic over scalar T and
# SIMD.Vec, so the same kernel serves the scalar and vectorized stage loops.
# D is the sign of the exponent: -1 forward, +1 inverse.

# contiguous plane storage: a Vector or a contiguous view of one (the offset
# mechanism for batched N-d passes)
const PlaneVec{T} = Union{Vector{T},SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}

# host vector register width in bits (SVE and other scalable extensions are
# not detectable here, so aarch64 assumes 128-bit NEON)
function _vector_bits()
    Sys.ARCH === :x86_64 || return 128
    CPUID = Base.BinaryPlatforms.CPUID
    CPUID.test_cpu_feature(CPUID.JL_X86_avx512f) && return 512
    CPUID.test_cpu_feature(CPUID.JL_X86_avx) && return 256
    return 128
end
const VECTOR_BITS = _vector_bits()

# native SIMD width used to vectorize stage loops (in elements of T)
pick_width(::Type{T}) where {T<:Union{Float32,Float64}} = VECTOR_BITS ÷ (8 * sizeof(T))
pick_width(::Type) = 1

@inline function cmul(wr, wi, xr, xi)
    muladd(wr, xr, -wi * xi), muladd(wr, xi, wi * xr)
end

# Interleave S-sized chunks of u and v: the sequence u0 v0 u1 v1 ... (chunks of
# S lanes) split back into two W-wide vectors.
@generated function interleave_chunks(u::Vec{W,T}, v::Vec{W,T}, ::Val{S}) where {W,T,S}
    G = W ÷ S
    idx(gblk, off) = (iseven(gblk) ? 0 : W) + (gblk >> 1) * S + off
    lomask = ntuple(i -> idx((i - 1) ÷ S, (i - 1) % S), W)
    himask = ntuple(i -> idx((i - 1) ÷ S + G, (i - 1) % S), W)
    quote
        $(Expr(:meta, :inline))
        (shufflevector(u, v, Val($lomask)), shufflevector(u, v, Val($himask)))
    end
end

# From y0..y3 each holding W÷S chunks of S lanes (one chunk per butterfly
# group), produce the output order y0_g y1_g y2_g y3_g per group g.
@inline function interleave4(y0::Vec{W,T}, y1::Vec{W,T}, y2::Vec{W,T}, y3::Vec{W,T},
                             ::Val{S}) where {W,T,S}
    a0, a1 = interleave_chunks(y0, y1, Val(S))
    b0, b1 = interleave_chunks(y2, y3, Val(S))
    o0, o1 = interleave_chunks(a0, b0, Val(2S))
    o2, o3 = interleave_chunks(a1, b1, Val(2S))
    return o0, o1, o2, o3
end

# W×W register transpose: log2(W) rounds of lane interleaves. Round r pairs
# vector i with i+W/2 and emits (lo, hi) in order, which after all rounds
# yields the transposed tile.
@generated function vtranspose(vs::NTuple{W,Vec{W,T}}) where {W,T}
    syms = [Symbol(:v, 0, :_, i) for i in 1:W]
    body = [:($(syms[i]) = vs[$i]) for i in 1:W]
    rounds = trailing_zeros(W)
    for r in 1:rounds
        prev = syms
        syms = [Symbol(:v, r, :_, i) for i in 1:W]
        for i in 1:(W >> 1)
            push!(body, :(($(syms[2i-1]), $(syms[2i])) =
                interleave_chunks($(prev[i]), $(prev[i+(W>>1)]), Val(1))))
        end
    end
    quote
        $(Expr(:meta, :inline))
        $(body...)
        tuple($(syms...))
    end
end

@generated function vreverse(v::Vec{W,T}) where {W,T}
    mask = ntuple(i -> W - i, W)
    quote
        $(Expr(:meta, :inline))
        shufflevector(v, Val($mask))
    end
end

# split 2W interleaved lanes (re,im pairs) into re and im vectors
@generated function deinterleave2(u::Vec{W,T}, v::Vec{W,T}) where {W,T}
    even = ntuple(i -> 2i - 2, W)
    odd = ntuple(i -> 2i - 1, W)
    quote
        $(Expr(:meta, :inline))
        (shufflevector(u, v, Val($even)), shufflevector(u, v, Val($odd)))
    end
end

# y0 = a + b; y1 = w * (a - b)
@inline function bf2(ar, ai, br, bi, wr, wi)
    y0r = ar + br
    y0i = ai + bi
    y1r, y1i = cmul(wr, wi, ar - br, ai - bi)
    return y0r, y0i, y1r, y1i
end

# Radix-4 DIF butterfly, pre-twiddle:
#   y0 = (a+c) + (b+d)
#   t1 = (a-c) + D*i*(b-d)
#   t2 = (a+c) - (b+d)
#   t3 = (a-c) - D*i*(b-d)
@inline function bf4_core(ar, ai, br, bi, cr, ci, dr, di, ::Val{D}) where {D}
    apcr = ar + cr; apci = ai + ci
    amcr = ar - cr; amci = ai - ci
    bpdr = br + dr; bpdi = bi + di
    bmdr = br - dr; bmdi = bi - di
    y0r = apcr + bpdr
    y0i = apci + bpdi
    if D == -1
        t1r = amcr + bmdi; t1i = amci - bmdr
        t3r = amcr - bmdi; t3i = amci + bmdr
    else
        t1r = amcr - bmdi; t1i = amci + bmdr
        t3r = amcr + bmdi; t3i = amci - bmdr
    end
    return y0r, y0i, t1r, t1i, apcr - bpdr, apci - bpdi, t3r, t3i
end

# Radix-8 DIF butterfly (pre-twiddle): two DFT4s over even/odd inputs,
# combined with w8^k rotations (c = √2/2). Bins in order y0..y7.
@inline function bf8_core(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                          x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i,
                          c, d::Val{D}) where {D}
    e0r, e0i, e1r, e1i, e2r, e2i, e3r, e3i =
        bf4_core(x0r, x0i, x2r, x2i, x4r, x4i, x6r, x6i, d)
    o0r, o0i, o1r, o1i, o2r, o2i, o3r, o3i =
        bf4_core(x1r, x1i, x3r, x3i, x5r, x5i, x7r, x7i, d)
    if D == -1
        t1r = c * (o1r + o1i); t1i = c * (o1i - o1r)   # w8 = c(1-i)
        t2r = o2i;             t2i = -o2r              # -i
        t3r = c * (o3i - o3r); t3i = -c * (o3r + o3i)  # w8³ = c(-1-i)
    else
        t1r = c * (o1r - o1i); t1i = c * (o1i + o1r)   # w8 = c(1+i)
        t2r = -o2i;            t2i = o2r               # i
        t3r = -c * (o3r + o3i); t3i = c * (o3r - o3i)  # w8³ = c(-1+i)
    end
    return (e0r + o0r, e0i + o0i, e1r + t1r, e1i + t1i,
            e2r + t2r, e2i + t2i, e3r + t3r, e3i + t3i,
            e0r - o0r, e0i - o0i, e1r - t1r, e1i - t1i,
            e2r - t2r, e2i - t2i, e3r - t3r, e3i - t3i)
end

# Radix-3 DFT: X1/X2 = (a - t/2) ± i·k(b-c), t = b+c, k = D·√3/2 (direction
# folded into k by the caller). nh is -1/2 in the working type.
@inline function bf3(ar, ai, br, bi, cr, ci, k, nh)
    tr = br + cr; ti = bi + ci
    sr = k * (br - cr); si = k * (bi - ci)
    ur = muladd(tr, nh, ar); ui = muladd(ti, nh, ai)
    return (ar + tr, ai + ti, ur - si, ui + sr, ur + si, ui - sr)
end

# Radix-5 DFT via symmetric pairing: c1/c2 = cos(2π/5), cos(4π/5);
# s1/s2 = D·sin(2π/5), D·sin(4π/5).
@inline function bf5(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i, x4r, x4i,
                     c1, c2, s1, s2)
    t1r = x1r + x4r; t1i = x1i + x4i
    t2r = x2r + x3r; t2i = x2i + x3i
    t3r = x1r - x4r; t3i = x1i - x4i
    t4r = x2r - x3r; t4i = x2i - x3i
    y0r = x0r + (t1r + t2r); y0i = x0i + (t1i + t2i)
    u1r = muladd(c1, t1r, muladd(c2, t2r, x0r))
    u1i = muladd(c1, t1i, muladd(c2, t2i, x0i))
    u2r = muladd(c2, t1r, muladd(c1, t2r, x0r))
    u2i = muladd(c2, t1i, muladd(c1, t2i, x0i))
    v1r = muladd(s1, t3r, s2 * t4r); v1i = muladd(s1, t3i, s2 * t4i)
    v2r = muladd(s2, t3r, -(s1 * t4r)); v2i = muladd(s2, t3i, -(s1 * t4i))
    return (y0r, y0i,
            u1r - v1i, u1i + v1r,
            u2r - v2i, u2i + v2r,
            u2r + v2i, u2i - v2r,
            u1r + v1i, u1i - v1r)
end

# Radix-7 DFT via symmetric pairing: cc[3(k-1)+j] = cos(2πjk/7),
# ss[3(k-1)+j] = D·sin(2πjk/7), for output pair k = 1..3 and input pair j = 1..3.
@inline function bf7(x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
                     x4r, x4i, x5r, x5i, x6r, x6i, cc, ss)
    t1r = x1r + x6r; t1i = x1i + x6i; d1r = x1r - x6r; d1i = x1i - x6i
    t2r = x2r + x5r; t2i = x2i + x5i; d2r = x2r - x5r; d2i = x2i - x5i
    t3r = x3r + x4r; t3i = x3i + x4i; d3r = x3r - x4r; d3i = x3i - x4i
    y0r = x0r + (t1r + t2r + t3r); y0i = x0i + (t1i + t2i + t3i)
    u1r = muladd(cc[1], t1r, muladd(cc[2], t2r, muladd(cc[3], t3r, x0r)))
    u1i = muladd(cc[1], t1i, muladd(cc[2], t2i, muladd(cc[3], t3i, x0i)))
    u2r = muladd(cc[4], t1r, muladd(cc[5], t2r, muladd(cc[6], t3r, x0r)))
    u2i = muladd(cc[4], t1i, muladd(cc[5], t2i, muladd(cc[6], t3i, x0i)))
    u3r = muladd(cc[7], t1r, muladd(cc[8], t2r, muladd(cc[9], t3r, x0r)))
    u3i = muladd(cc[7], t1i, muladd(cc[8], t2i, muladd(cc[9], t3i, x0i)))
    v1r = muladd(ss[1], d1r, muladd(ss[2], d2r, ss[3] * d3r))
    v1i = muladd(ss[1], d1i, muladd(ss[2], d2i, ss[3] * d3i))
    v2r = muladd(ss[4], d1r, muladd(ss[5], d2r, ss[6] * d3r))
    v2i = muladd(ss[4], d1i, muladd(ss[5], d2i, ss[6] * d3i))
    v3r = muladd(ss[7], d1r, muladd(ss[8], d2r, ss[9] * d3r))
    v3i = muladd(ss[7], d1i, muladd(ss[8], d2i, ss[9] * d3i))
    return (y0r, y0i,
            u1r - v1i, u1i + v1r,
            u2r - v2i, u2i + v2r,
            u3r - v3i, u3i + v3r,
            u3r + v3i, u3i - v3r,
            u2r + v2i, u2i - v2r,
            u1r + v1i, u1i - v1r)
end

# full butterfly: t1..t3 twiddled by w1..w3
@inline function bf4(ar, ai, br, bi, cr, ci, dr, di,
                     w1r, w1i, w2r, w2i, w3r, w3i, d::Val)
    y0r, y0i, t1r, t1i, t2r, t2i, t3r, t3i =
        bf4_core(ar, ai, br, bi, cr, ci, dr, di, d)
    y1r, y1i = cmul(w1r, w1i, t1r, t1i)
    y2r, y2i = cmul(w2r, w2i, t2r, t2i)
    y3r, y3i = cmul(w3r, w3i, t3r, t3i)
    return y0r, y0i, y1r, y1i, y2r, y2i, y3r, y3i
end
