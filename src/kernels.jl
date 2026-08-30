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

# Generic odd-prime DFT butterfly via conjugate-symmetric pairing, generated
# for any odd prime P (serves every odd radix; measured equal to the
# previously handwritten radix-3/5/7 kernels). With h = (P-1)/2
# and t_j = x_j + x_{P-j}, d_j = x_j - x_{P-j}:
#   X_0 = x_0 + Σ t_j
#   X_k = u_k + i·v_k,  X_{P-k} = u_k - i·v_k        (bins in natural order)
#   u_k = x_0 + Σ_j cos(2πjk/P)·t_j,   v_k = D·Σ_j sin(2πjk/P)·d_j
# All coefficients are real (h² muladds per bank per plane), ~4× fewer real
# multiplies than a naive complex O(P²) DFT.
# cc/ss coefficient banks indexed (k-1)h + j. Generated so
# the tuples are compile-time constants (a runtime-length ntuple would be
# type-unstable and allocate on every stage call).
@generated function bfp_consts(::Type{T}, ::Val{P}, ::Val{D}) where {T,P,D}
    h = (P - 1) >> 1
    cc = ntuple(i -> T(cospi(2 * (((i - 1) % h + 1) * ((i - 1) ÷ h + 1)) / P)), h * h)
    ss = ntuple(i -> T(D * sinpi(2 * (((i - 1) % h + 1) * ((i - 1) ÷ h + 1)) / P)), h * h)
    return :(($cc, $ss))
end

# broadcast a scalar tuple to Vec lanes; generated because Base's tuple map
# hits its 32-element inference limit at P = 13 (h² = 36) and allocates
@generated function vsplat(::Type{Vec{W,T}}, t::NTuple{N,T}) where {W,T,N}
    quote
        $(Expr(:meta, :inline))
        @inbounds $(Expr(:tuple, (:(Vec{W,T}(t[$i])) for i in 1:N)...))
    end
end

@generated function bfp(xr::NTuple{P,V}, xi::NTuple{P,V},
                        cc::NTuple{HH,S}, ss::NTuple{HH,S}) where {P,V,S,HH}
    h = (P - 1) >> 1
    @assert HH == h * h
    body = Expr[]
    tr = [Symbol(:t, j, :r) for j in 1:h]
    ti = [Symbol(:t, j, :i) for j in 1:h]
    dr = [Symbol(:d, j, :r) for j in 1:h]
    di = [Symbol(:d, j, :i) for j in 1:h]
    for j in 1:h
        push!(body, :($(tr[j]) = xr[$(j + 1)] + xr[$(P + 1 - j)]),
              :($(ti[j]) = xi[$(j + 1)] + xi[$(P + 1 - j)]),
              :($(dr[j]) = xr[$(j + 1)] - xr[$(P + 1 - j)]),
              :($(di[j]) = xi[$(j + 1)] - xi[$(P + 1 - j)]))
    end
    sumr, sumi = tr[1], ti[1]
    for j in 2:h
        sumr = :($sumr + $(tr[j]))
        sumi = :($sumi + $(ti[j]))
    end
    push!(body, :(y0r = xr[1] + $sumr), :(y0i = xi[1] + $sumi))
    for k in 1:h
        er, ei = :(xr[1]), :(xi[1])
        for j in h:-1:1
            c = :(cc[$((k - 1) * h + j)])
            er = :(muladd($c, $(tr[j]), $er))
            ei = :(muladd($c, $(ti[j]), $ei))
        end
        s = :(ss[$((k - 1) * h + h)])
        vr, vi = :($s * $(dr[h])), :($s * $(di[h]))
        for j in h-1:-1:1
            s = :(ss[$((k - 1) * h + j)])
            vr = :(muladd($s, $(dr[j]), $vr))
            vi = :(muladd($s, $(di[j]), $vi))
        end
        push!(body, :($(Symbol(:u, k, :r)) = $er), :($(Symbol(:u, k, :i)) = $ei),
              :($(Symbol(:v, k, :r)) = $vr), :($(Symbol(:v, k, :i)) = $vi))
    end
    outs = Any[:y0r, :y0i]
    for k in 1:h
        push!(outs, :($(Symbol(:u, k, :r)) - $(Symbol(:v, k, :i))),
              :($(Symbol(:u, k, :i)) + $(Symbol(:v, k, :r))))
    end
    for k in h:-1:1
        push!(outs, :($(Symbol(:u, k, :r)) + $(Symbol(:v, k, :i))),
              :($(Symbol(:u, k, :i)) - $(Symbol(:v, k, :r))))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(body...)
            tuple($(outs...))
        end
    end
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
