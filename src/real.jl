# Real transforms. Even n: pack the real input as n/2 complex points by
# reinterpret (no copy), run the half-size complex transform, and split the
# even/odd spectra in an O(n) post-pass with edge twiddles e^{∓2πik/n}.
# Odd n: fall back to a full-length complex transform.

struct RfftPlan{T,P}
    n::Int
    inner::P                  # length n÷2 (even) or n (odd) forward plan
    twr::Vector{T}            # e^{-2πik/n}, k = 0..n÷2 (even case)
    twi::Vector{T}
    buf::Vector{Complex{T}}   # Z (even) / complex input copy (odd)
    buf2::Vector{Complex{T}}  # odd case: complex output
end

struct IrfftPlan{T,P}
    n::Int
    inner::P                  # length n÷2 (even) or n (odd) inverse plan
    pre::T                    # pre-pass pair factor: 1/2 (irfft) or 1 (brfft)
    twr::Vector{T}            # e^{+2πik/n}, k = 0..n÷2-1 (even case)
    twi::Vector{T}
    buf::Vector{Complex{T}}   # Z (even) / full Hermitian spectrum (odd)
    buf2::Vector{Complex{T}}  # odd case: complex output
end

const _RTW_CACHE = Dict{Tuple{DataType,Int,Int,Int},Any}()
function real_twiddles(::Type{T}, n::Int, len::Int, D::Int) where {T}
    r = _cached(_RTW_CACHE, (T, n, len, D)) do
        twr = Vector{T}(undef, len)
        twi = Vector{T}(undef, len)
        for k in 0:len-1
            θ = D * (Float64(2k) / n)
            si, co = sincospi(θ)
            twr[k+1] = T(co)
            twi[k+1] = T(si)
        end
        (twr, twi)
    end
    return r::Tuple{Vector{T},Vector{T}}
end

function plan_rfft(::Type{T}, n::Integer) where {T<:AbstractFloat}
    n = Int(n)
    n ≥ 1 || throw(ArgumentError("transform length must be positive (got $n)"))
    even = iseven(n)
    len = even ? n >>> 1 : n
    inner = _anyplan(T, len, -1)
    twr, twi = even ? real_twiddles(T, n, len + 1, -1) : (T[], T[])
    return RfftPlan{T,typeof(inner)}(n, inner, twr, twi,
                                     Vector{Complex{T}}(undef, len),
                                     Vector{Complex{T}}(undef, even ? 0 : n))
end

# normalized = true gives irfft (1/n folded in); false gives FFTW-style brfft
# (unnormalized: brfft = n·irfft, via an unnormalized inner transform and a
# pre-pass pair factor of 1 instead of 1/2).
function plan_irfft(::Type{T}, n::Integer; normalized::Bool=true) where {T<:AbstractFloat}
    n = Int(n)
    n ≥ 1 || throw(ArgumentError("transform length must be positive (got $n)"))
    pre = normalized ? T(0.5) : one(T)
    even = iseven(n)
    len = even ? n >>> 1 : n
    inner = normalized ? _anyplan(T, len, +1) : _anyplan(T, len, +1, one(T))
    twr, twi = even ? real_twiddles(T, n, len, +1) : (T[], T[])
    return IrfftPlan{T,typeof(inner)}(n, inner, pre, twr, twi,
                                      Vector{Complex{T}}(undef, len),
                                      Vector{Complex{T}}(undef, even ? 0 : n))
end

plan_brfft(::Type{T}, n::Integer) where {T<:AbstractFloat} =
    plan_irfft(T, n; normalized=false)

# y may be a contiguous view (e.g. the leading half of a longer spectrum buffer)
function (p::RfftPlan{T})(y::AbstractVector{Complex{T}}, x::AbstractVector{T}) where {T}
    n = p.n
    length(x) == n || throw(DimensionMismatch("plan length $n, input length $(length(x))"))
    length(y) == n ÷ 2 + 1 ||
        throw(DimensionMismatch("rfft output must have length $(n ÷ 2 + 1), got $(length(y))"))
    if iseven(n)
        h = n >>> 1
        Z = p.buf
        p.inner(Z, reinterpret(Complex{T}, x))
        twr, twi = p.twr, p.twi
        W = pick_width(T)
        @inbounds begin
            z0 = Z[1]
            y[1] = Complex(real(z0) + imag(z0), zero(T))
            y[h+1] = Complex(real(z0) - imag(z0), zero(T))
            half = T(0.5)
            if W > 1 && h - 1 ≥ W
                # k runs over blocks of 1:h-1; the partner block h-k is loaded
                # ascending and lane-reversed. X[k] = xe + w·(-i·xo):
                #   yr = xer + (wr·xoi + wi·xor),  yi = xei + (wi·xoi - wr·xor)
                GC.@preserve Z y begin
                    pz = convert(Ptr{T}, pointer(Z))
                    py = convert(Ptr{T}, pointer(y))
                    sz = sizeof(T)
                    hv = Vec{W,T}(half)
                    for q in QWin(h - 1, W)
                        u = vload(Vec{W,T}, pz + sz * 2q)
                        v = vload(Vec{W,T}, pz + sz * (2q + W))
                        ar, ai = deinterleave2(u, v)
                        boff = h - q - W + 1
                        u = vload(Vec{W,T}, pz + sz * 2boff)
                        v = vload(Vec{W,T}, pz + sz * (2boff + W))
                        br0, bi0 = deinterleave2(u, v)
                        br = vreverse(br0); bi = -vreverse(bi0)
                        xer = (ar + br) * hv; xei = (ai + bi) * hv
                        xor_ = (ar - br) * hv; xoi = (ai - bi) * hv
                        wr = vload(Vec{W,T}, twr, q + 1)
                        wi = vload(Vec{W,T}, twi, q + 1)
                        yr = xer + muladd(wr, xoi, wi * xor_)
                        yi = xei + muladd(wi, xoi, -(wr * xor_))
                        lo, hi = interleave_chunks(yr, yi, Val(1))
                        vstore(lo, py + sz * 2q)
                        vstore(hi, py + sz * (2q + W))
                    end
                end
            else
                for k in 1:h-1
                    a = Z[k+1]
                    b = conj(Z[h-k+1])
                    xe = (a + b) * half
                    xo = (a - b) * half
                    w = Complex(twr[k+1], twi[k+1])
                    y[k+1] = xe + w * Complex(imag(xo), -real(xo))  # xe + w·(-i·xo)
                end
            end
        end
    else
        copyto!(p.buf, x)
        p.inner(p.buf2, p.buf)
        copyto!(y, 1, p.buf2, 1, n ÷ 2 + 1)
    end
    return y
end

function (p::IrfftPlan{T})(y::AbstractVector{T}, X::AbstractVector{Complex{T}}) where {T}
    n = p.n
    length(y) == n || throw(DimensionMismatch("plan length $n, output length $(length(y))"))
    length(X) == n ÷ 2 + 1 ||
        throw(DimensionMismatch("irfft input must have length $(n ÷ 2 + 1), got $(length(X))"))
    if iseven(n)
        h = n >>> 1
        Z = p.buf
        twr, twi = p.twr, p.twi
        half = p.pre
        W = pick_width(T)
        if W > 1 && h ≥ W
            # k blocks over 0:h-1; partner h-k loaded ascending, lane-reversed.
            # Z[k] = xe + i·(w·xd):  Zr = xer - ti,  Zi = xei + tr
            GC.@preserve X Z begin
                px = convert(Ptr{T}, pointer(X))
                pz = convert(Ptr{T}, pointer(Z))
                sz = sizeof(T)
                hv = Vec{W,T}(half)
                @inbounds for q in QWin(h, W)
                    k0 = q - 1
                    u = vload(Vec{W,T}, px + sz * 2k0)
                    v = vload(Vec{W,T}, px + sz * (2k0 + W))
                    ar, ai = deinterleave2(u, v)
                    boff = h - k0 - W + 1
                    u = vload(Vec{W,T}, px + sz * 2boff)
                    v = vload(Vec{W,T}, px + sz * (2boff + W))
                    br0, bi0 = deinterleave2(u, v)
                    br = vreverse(br0); bi = -vreverse(bi0)
                    xer = (ar + br) * hv; xei = (ai + bi) * hv
                    xdr = (ar - br) * hv; xdi = (ai - bi) * hv
                    wr = vload(Vec{W,T}, twr, q)
                    wi = vload(Vec{W,T}, twi, q)
                    tr = muladd(wr, xdr, -(wi * xdi))
                    ti = muladd(wr, xdi, wi * xdr)
                    lo, hi = interleave_chunks(xer - ti, xei + tr, Val(1))
                    vstore(lo, pz + sz * 2k0)
                    vstore(hi, pz + sz * (2k0 + W))
                end
            end
        else
            @inbounds for k in 0:h-1
                a = X[k+1]
                b = conj(X[h-k+1])
                xe = (a + b) * half
                t = Complex(twr[k+1], twi[k+1]) * ((a - b) * half)
                Z[k+1] = xe + Complex(-imag(t), real(t))  # xe + i·t
            end
        end
        p.inner(reinterpret(Complex{T}, y), Z)
    else
        buf = p.buf
        h = n ÷ 2
        @inbounds begin
            for k in 1:h+1
                buf[k] = X[k]
            end
            for k in 1:h
                buf[n-k+1] = conj(X[k+1])
            end
        end
        p.inner(p.buf2, buf)
        @inbounds for j in 1:n
            y[j] = real(p.buf2[j])
        end
    end
    return y
end

(p::RfftPlan{T})(x::Vector{T}) where {T} =
    p(Vector{Complex{T}}(undef, p.n ÷ 2 + 1), x)
(p::IrfftPlan{T})(X::Vector{Complex{T}}) where {T} =
    p(Vector{T}(undef, p.n), X)
