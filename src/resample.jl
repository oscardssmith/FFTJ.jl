# Approximate length-n DFT by resampling onto a smooth length m ≥ σn (a
# type-2 NUFFT at the points k/n): fold the modes to j' ∈ [-⌊n/2⌋, ⌈n/2⌉),
# divide by the kernel's Fourier transform, zero-pad, transform at length m,
# then interpolate the m-grid at u_k = k·m/n with a w-tap
# exponential-of-semicircle kernel ψ(z) = exp(β(√(1-z²)-1)).
# Error ≈ exp(-πw√(1-1/σ)), relative to ‖x‖. This serves every length whose
# own stage chain is unavailable (not 7-smooth) or slow.

# Smallest efficient transform length ≥ N: small odd part on a large power of
# two, matching where the engine measures fastest (odd stages need wide
# strides; a 2-part below ~2^6 falls off a cliff).
const SMOOTH_ODD = (1, 3, 5, 7, 9, 15, 21)
function nextsmooth(N::Int)
    best = typemax(Int)
    for o in SMOOTH_ODD
        k = max(6, Base.top_set_bit(cld(N, o) - 1))
        c = o << k
        c < best && (best = c)
    end
    return best
end

# smallest 7-smooth length ≥ N: batched passes vectorize across lanes, so
# they don't need nextsmooth's long power-of-two part
function nextsmooth7(N::Int)
    m = N
    while factor_smooth(m) === nothing
        m += 1
    end
    return m
end

function _gausslegendre(::Type{Tw}, nq::Int) where {Tw}
    z = zeros(Tw, nq)
    wq = zeros(Tw, nq)
    for i in 1:nq
        x = cospi(Tw(i - 0.25) / (nq + 0.5))
        dp = one(Tw)
        for _ in 1:20
            p0, p1 = one(Tw), x
            for k in 2:nq
                p0, p1 = p1, ((2k - 1) * x * p1 - (k - 1) * p0) / k
            end
            dp = nq * (x * p1 - p0) / (x^2 - 1)
            dx = p1 / dp
            x -= dx
            abs(dx) < 4eps(Tw) && break
        end
        z[i] = x
        wq[i] = 2 / ((1 - x^2) * dp^2)
    end
    return z, wq
end

_es(z, β) = abs(z) ≥ 1 ? zero(z) : exp(β * (sqrt(1 - z^2) - 1))

# Output k reads grid l = q-h+1 .. q+h where k·m = q·n + r; its weights
# depend only on the offset δ = r/n. Two weight sources hand the
# interpolation (first grid index, weight vector) per output:
# a full n×w table, or piecewise polynomials in δ (P pieces of degree d,
# Horner across the taps) with (q, r) stepped from per-block constants.
struct WTable{T}
    wt::Vector{T}        # w weights per output, contiguous
    l0::Vector{Int32}    # each output's first grid index (unwrapped)
end
struct WPoly{T,d}
    C::Vector{T}         # C[(p·(d+1) + c)·w + i + 1]: coefficient c of tap i, piece p
    P::Int
    Pn::Float64          # P / n
    n::Int
    h::Int
    e::Vector{Int}       # e[c+1] = c·m mod n, g[c+1] = ⌊c·m/n⌋ for c = 0..B
    g::Vector{Int}
end

_wstart(::WTable) = nothing
_wstart(::WPoly) = (0, 0)
_wadvance(::WTable, st, c) = st
@inline function _wadvance(ws::WPoly, (q, r), c)
    rr = r + @inbounds ws.e[c+1]
    cy = rr ≥ ws.n
    return (q + @inbounds(ws.g[c+1]) + cy, rr - ifelse(cy, ws.n, 0))
end

# Per-block setup, then the 1-based grid index of output k+j's first tap
# and its weights. The polynomial source computes the block's indices,
# pieces and local variables as B-lane vectors so the per-output work is
# lane extraction plus Horner.
@inline _wprep(ws::WTable, k, st, ::Val{B}) where {B} = k
@inline _wget(ws::WTable{T}, k, j, ::Val{w}) where {T,w} =
    @inbounds (Int(ws.l0[k+j+1]) + 1, vload(Vec{w,T}, ws.wt, (k + j) * w + 1))

@inline function _wprep(ws::WPoly{T,d}, k, (q, r), ::Val{B}) where {T,d,B}
    @inbounds begin
        rv = r + vload(Vec{B,Int}, ws.e, 1)
        cy = rv ≥ ws.n
        rv = vifelse(cy, rv - ws.n, rv)
        qv = q + vload(Vec{B,Int}, ws.g, 1) + vifelse(cy, Vec{B,Int}(1), Vec{B,Int}(0))
    end
    s = convert(Vec{B,Float64}, rv) * ws.Pn          # δ·P ∈ [0, P)
    pcv = min(convert(Vec{B,Int}, s), ws.P - 1)      # truncation; s ≥ 0
    tv = convert(Vec{B,T}, 2 * (s - convert(Vec{B,Float64}, pcv)) - 1)
    return qv - (ws.h - 2), pcv * (d + 1), tv
end
@inline function _wget(ws::WPoly{T,d}, (lv, bv, tv), j, ::Val{w}) where {T,d,w}
    base = bv[j+1] * w + 1
    t = tv[j+1]
    @inbounds begin
        wv = vload(Vec{w,T}, ws.C, base + d * w)
        for c in d-1:-1:0
            wv = muladd(wv, t, vload(Vec{w,T}, ws.C, base + c * w))
        end
    end
    return lv[j+1], wv
end

# whole block of B outputs from k has its windows inside the grid
@inline _wblock_ok(ws::WTable, k, st, B, w, m) =
    @inbounds ws.l0[k+1] ≥ 0 && ws.l0[k+B] + w ≤ m
@inline _wblock_ok(ws::WPoly, k, (q, r), B, w, m) =
    q - ws.h + 1 ≥ 0 && q + @inbounds(ws.g[B]) + 1 - ws.h + 1 + w ≤ m

# Chebyshev interpolants of each tap's weight on each piece, converted to
# monomials in the local variable; d grows until the fit error is ≤ tolw or
# stops halving (coefficient rounding floors it near 25eps(T))
function _fit_weights(::Type{T}, ::Type{Tw}, β, w::Int, P::Int, tolw) where {T,Tw}
    prev, preverr = _fit_weights_d(T, Tw, β, w, P, 1)
    for d in 2:30
        C, err = _fit_weights_d(T, Tw, β, w, P, d)
        err ≤ tolw && return C, d
        err > preverr / 2 && return prev, d - 1
        prev, preverr = C, err
    end
    return prev, 30
end

# degree-d fit and its max abs error, sampled at the T-rounded coefficients
function _fit_weights_d(::Type{T}, ::Type{Tw}, β, w::Int, P::Int, d::Int) where {T,Tw}
    h = w ÷ 2
    tj = [cospi(Tw(j + 0.5) / (d + 1)) for j in 0:d]
    C = zeros(T, P * (d + 1) * w)
    mono = zeros(Tw, d + 1)
    err = zero(Tw)
    for p in 0:P-1, i in 0:w-1
        f(t) = _es(((p + (t + 1) / 2) / P + h - 1 - i) / h, β)
        fj = f.(tj)
        fill!(mono, 0)
        Tkm, Tk = zeros(Tw, d + 1), zeros(Tw, d + 1)   # T₀, T₁ in monomials
        Tkm[1] = 1
        Tk[2] = 1
        for k in 0:d
            a = sum(fj[j+1] * cospi(Tw(k) * (j + 0.5) / (d + 1)) for j in 0:d) * 2 / (d + 1)
            k == 0 && (a /= 2)
            mono .+= a .* (k == 0 ? Tkm : Tk)
            if k ≥ 1   # advance T_k → T_{k+1} = 2t·T_k - T_{k-1}
                Tn = -Tkm
                Tn[2:end] .+= 2 .* Tk[1:end-1]
                Tkm, Tk = Tk, Tn
            end
        end
        for c in 0:d
            mono[c+1] = C[(p*(d+1)+c)*w+i+1] = T(mono[c+1])
        end
        for t in range(-one(Tw), one(Tw); length=4d + 4)
            err = max(err, abs(evalpoly(t, mono) - f(t)))
        end
    end
    return C, err
end

# Largest weight table kept: F64 polynomial weights reach parity near this
# size; F32 ones stay ~10% slower but save the table's memory and setup.
const _RS_TABLE_MAX = 1 << 21

struct ResamplePlan{T,D,w,WS}
    n::Int
    m::Int
    corr::Vector{T}      # scale / Φ̂(j'/m) for input index j
    ws::WS               # interpolation weights
    p::FFTPlan{T,D}      # length-m stage chain and split-plane scratch
    scale::T             # output scale (folded into corr)
end

# Plan-independent parts (m, w, corr, weights), shared by (T, n, D, scale)
# for the default parameters since DSP-style code plans per call.
const _RESAMPLE_CACHE = Dict{Tuple{DataType,Int,Int,Any},Any}()
function _resample(::Type{T}, n::Int, D::Int, scale::T=default_scale(T, n, D);
                   kws...) where {T<:AbstractFloat}
    m, w, corr, ws = isempty(kws) ?
        _cached(() -> _resample_parts(T, n, D, scale), _RESAMPLE_CACHE, (T, n, D, scale)) :
        _resample_parts(T, n, D, scale; kws...)
    return ResamplePlan{T,D,w,typeof(ws)}(n, m, corr, ws,
                                          _plan(T, m, D, one(T); nthreads=1), scale)
end

# tol is the target error relative to ‖x‖; w defaults to the narrowest
# multiple of 4 meeting it at the chosen grid ratio
function _resample_parts(::Type{T}, n::Int, D::Int, scale::T;
                         σ::Real=2, tol::Real=10eps(T), w::Int=0, γ::Real=0.97,
                         weights::Symbol=:auto, P::Int=sizeof(T) ≥ 8 ? 32 : 64,
                         m::Int=nextsmooth(ceil(Int, σ * n))) where {T<:AbstractFloat}
    Tw = promote_type(T, Float64)
    m < typemax(Int32) || throw(ArgumentError("resampled length $m exceeds Int32 indexing"))
    if w == 0
        # oversampling beyond 2 (tiny n padded up to nextsmooth's minimum)
        # measured no gain, so the width never drops below σ = 2's
        w = 4 * cld(ceil(Int, log(1 / Tw(tol)) / (π * sqrt(1 - max(Tw(n) / m, Tw(0.5))))), 4)
    end
    w % 4 == 0 || throw(ArgumentError("w must be a multiple of 4 (got $w)"))
    β = Tw(γ) * π * w * (1 - Tw(n) / (2m))
    h = w ÷ 2
    # Φ̂(ν) = ∫ψ(2u/w)e^{-2πiνu}du over grid units u; ψ is even, so a
    # half-interval Gauss–Legendre cosine sum suffices
    zq, wq = _gausslegendre(Tw, 2w + 20)
    corr = Vector{T}(undef, n)
    hp = cld(n, 2)
    for j in 0:n-1
        jp = j < hp ? j : j - n
        ν = Tw(jp) / m
        Φ = zero(Tw)
        for q in eachindex(zq)
            zq[q] > 0 || continue
            Φ += wq[q] * _es(zq[q], β) * cospi(ν * w * zq[q])
        end
        corr[j+1] = T(scale / (w * Φ))   # (w/2)·2 halves → w
    end
    if weights === :auto
        weights = n * w * sizeof(T) ≤ _RS_TABLE_MAX ? :table : :poly
    end
    if weights === :table
        wt = Vector{T}(undef, n * w)
        l0 = Vector{Int32}(undef, n)
        for k in 0:n-1
            q, r = divrem(widemul(k, m), n)
            l0[k+1] = Int32(q - h + 1)
            δ = Tw(r) / n
            for i in 0:w-1
                wt[k*w+i+1] = T(_es((δ + h - 1 - i) / h, β))
            end
        end
        ws = WTable{T}(wt, l0)
    elseif weights === :poly
        C, d = _fit_weights(T, Tw, β, w, P, Tw(tol) / 16)
        B = max(1, pick_width(T) ÷ 2)
        e = [Int(widemul(c, m) % n) for c in 0:B]
        g = [Int(widemul(c, m) ÷ n) for c in 0:B]
        ws = WPoly{T,d}(C, P, P / n, n, h, e, g)
    else
        throw(ArgumentError("weights must be :table or :poly (got $weights)"))
    end
    return m, w, corr, ws
end

function (p::ResamplePlan{T,D,w})(y::CVec{T}, x::CVec{T}) where {T,D,w}
    n, m = p.n, p.m
    length(x) == n || throw(DimensionMismatch("plan length $n, input length $(length(x))"))
    length(y) == n || throw(DimensionMismatch("plan length $n, output length $(length(y))"))
    f = p.p
    are, aim = f.are, f.aim
    hp = cld(n, 2)
    GC.@preserve x begin
        px = _rawptr(x)
        _deconv!(are, aim, 0, px, 0, p.corr, hp)
        _deconv!(are, aim, m - n + hp, px, hp, p.corr, n - hp)
    end
    @inbounds for j in hp+1:m-n+hp
        are[j] = zero(T); aim[j] = zero(T)
    end
    cre, cim, _, _ = run_chain!(are, aim, f.bre, f.bim, f.stages, m, 1, 1, Val(D))
    GC.@preserve y _interp!(_rawptr(y), cre, cim, p.ws, n, m, Val(w))
    return y
end

(p::ResamplePlan{T})(x::Vector{Complex{T}}) where {T} = p(similar(x), x)

# are/aim[o+1 : o+len] = x[xo+1 : xo+len] .* corr[xo+1 : xo+len], deinterleaved
function _deconv!(are::Vector{T}, aim::Vector{T}, o::Int, px::Ptr{T}, xo::Int,
                  corr::Vector{T}, len::Int) where {T}
    W = pick_width(T)
    sz = sizeof(T)
    if W > 1 && len ≥ W
        @inbounds for c in QWin(len, W)
            u = vload(Vec{W,T}, px + sz * 2 * (xo + c - 1))
            v = vload(Vec{W,T}, px + sz * (2 * (xo + c - 1) + W))
            r, i = deinterleave2(u, v)
            cv = vload(Vec{W,T}, corr, xo + c)
            vstore(r * cv, are, o + c)
            vstore(i * cv, aim, o + c)
        end
    else
        @inbounds for c in 1:len
            are[o+c] = unsafe_load(px, 2(xo + c) - 1) * corr[xo+c]
            aim[o+c] = unsafe_load(px, 2(xo + c)) * corr[xo+c]
        end
    end
    return nothing
end

# Each output is two length-w dot products (re, im). Blocks of V/2 outputs
# reduce together instead of one horizontal sum each: fold every product to
# s = min(w, V) lanes, pack the V rows into s vectors, then halve all
# segments of a vector pair at once per tree level (a blend, a shuffle and
# an add), ending in one vector of interleaved outputs.
_blockable(V, w) = V > 1 && (w % V == 0 || V % w == 0) && ispow2(min(w, V))

# sum of the length-s chunks of v
@generated function _chunksum(v::Vec{w,T}, ::Val{s}) where {w,T,s}
    w == s && return :v
    chunks = [:(shufflevector(v, Val($(Tuple(j*s:j*s+s-1))))) for j in 0:w÷s-1]
    return foldl((a, b) -> :($a + $b), chunks)
end

# tree-reduce s vectors whose length-s segments carry row labels; returns
# the statements and the result with lanes permuted into row order
function _tree_reduce_code(vecs, labels, s::Int, V::Int)
    stmts = Expr[]
    L = s
    while L > 1
        half = L ÷ 2
        nvecs, nlabels = Symbol[], Vector{Int}[]
        for p in 1:2:length(vecs)
            A, B = vecs[p], vecs[p+1]
            ip, iq, lab = Int[], Int[], Int[]
            for l in 0:V-1
                seg, pos = divrem(l, L)
                first = pos < half
                sw = seg * L + (pos + half) % L   # partner lane within the segment
                push!(ip, first ? l : V + l)
                push!(iq, first ? sw : V + sw)
                push!(lab, first ? labels[p][l+1] : labels[p+1][l+1])
            end
            t = gensym(:t)
            push!(stmts, :($t = shufflevector($A, $B, Val($(Tuple(ip)))) +
                                shufflevector($A, $B, Val($(Tuple(iq))))))
            push!(nvecs, t)
            push!(nlabels, lab)
        end
        vecs, labels = nvecs, nlabels
        L = half
    end
    perm = [findfirst(==(i), labels[1]) - 1 for i in 0:V-1]
    res = perm == 0:V-1 ? vecs[1] : :(shufflevector($(vecs[1]), Val($(Tuple(perm)))))
    return stmts, res
end

# V/2 outputs starting at k, all windows inside the grid
@generated function _interp_block!(py::Ptr{T}, gre, gim, ws, k, st,
                                   ::Val{w}, ::Val{V}) where {T,w,V}
    s = min(w, V)
    c = V ÷ s   # rows per packed vector
    body = Expr[:(pr = _wprep(ws, k, st, Val($(V ÷ 2))))]
    rows = Symbol[]
    for j in 0:V÷2-1
        l, wv, re, im = gensym(:l), gensym(:wv), gensym(:re), gensym(:im)
        push!(body, quote
            $l, $wv = _wget(ws, pr, $j, Val($w))
            $re = _chunksum(vload(Vec{$w,T}, gre, $l) * $wv, Val($s))
            $im = _chunksum(vload(Vec{$w,T}, gim, $l) * $wv, Val($s))
        end)
        push!(rows, re, im)
    end
    vecs, labels = Symbol[], Vector{Int}[]
    for g in 0:s-1
        parts = Any[rows[g*c+i+1] for i in 0:c-1]
        len = s
        while length(parts) > 1
            parts = Any[:(shufflevector($(parts[2i-1]), $(parts[2i]), Val($(Tuple(0:2len-1)))))
                        for i in 1:length(parts)÷2]
            len *= 2
        end
        v = gensym(:v)
        push!(body, :($v = $(parts[1])))
        push!(vecs, v)
        push!(labels, [g * c + l ÷ s for l in 0:V-1])
    end
    stmts, res = _tree_reduce_code(vecs, labels, s, V)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $(body...)
            $(stmts...)
            vstore($res, py + $(sizeof(T)) * 2k)
        end
        return nothing
    end
end

@inline function _interp!(py::Ptr{T}, gre, gim, ws, n, m, ::Val{w}) where {T,w}
    V = pick_width(T)
    B = _blockable(V, w) ? V ÷ 2 : 0
    k = 0
    st = _wstart(ws)
    @inbounds while k < n
        # block path when every window in it lies inside the grid
        if B > 0 && k + B ≤ n && _wblock_ok(ws, k, st, B, w, m)
            _interp_block!(py, gre, gim, ws, k, st, Val(w), Val(V))
            st = _wadvance(ws, st, B)
            k += B
            continue
        end
        l1, wv = _wget(ws, _wprep(ws, k, st, Val(1)), 0, Val(w))
        lk = l1 - 1
        if lk ≥ 0 && lk + w ≤ m
            re = sum(vload(Vec{w,T}, gre, lk + 1) * wv)
            im = sum(vload(Vec{w,T}, gim, lk + 1) * wv)
        else   # window wraps around the grid ends
            re = im = zero(T)
            for i in 0:w-1
                l = mod(lk + i, m) + 1
                re += wv[i+1] * gre[l]
                im += wv[i+1] * gim[l]
            end
        end
        unsafe_store!(py, re, 2k + 1)
        unsafe_store!(py, im, 2k + 2)
        st = _wadvance(ws, st, 1)
        k += 1
    end
    return nothing
end

# --- batched resampling along one dimension of an N-d array ---------------
# Rows of B lanes (element (i, j) at i + B·j, B pencils side by side). The
# correction factors, the length-m transform and the interpolation weights
# are shared by every pencil, so each step is a vector op across the lanes:
# deconvolve into a zero-padded m-row block, run the smooth chain batched,
# then form each output row as w weighted grid rows. Lanes go through in
# chunks of C (a multiple of W; the last chunk absorbs the remainder) so the
# chain's working set stays in L2.
struct RsPass{T}
    n::Int
    m::Int
    w::Int
    B::Int                  # lanes per row
    C::Int                  # lanes per chunk
    corr::Vector{T}         # 1 / Φ̂(j'/m) per input row
    wt::Vector{T}           # w weights per output row
    l0::Vector{Int32}       # each output row's first grid row (unwrapped)
    stages::Vector{Stage{T}}
    pre::Vector{T}          # chunk scratch plane pairs, < 2C·m each
    pim::Vector{T}
    qre::Vector{T}
    qim::Vector{T}
end

const _RS_PASS_CACHE = Dict{Tuple{DataType,Int,Int},Any}()

# chunk scratch budget (four planes), as for four-step's inner buffers
const _RS_CHUNK_MAX = 1 << 19

function _rs_pass(::Type{T}, n::Int, D::Int, B::Int) where {T}
    m, w, corr, ws = _cached(_RS_PASS_CACHE, (T, n, D)) do
        _resample_parts(T, n, D, one(T); weights=:table, m=nextsmooth7(2n))
    end
    W = pick_width(T)
    C = B < 2W ? B : clamp(_RS_CHUNK_MAX ÷ (4 * m * sizeof(T)) ÷ W * W, W, B ÷ W * W)
    cmax = B - (max(1, B ÷ C) - 1) * C   # the last chunk is the widest
    stages, _ = build_stages_cached(T, m, D; allow_small=B == 1)
    return RsPass{T}(n, m, w, B, C, corr, ws.wt, ws.l0, stages,
                     Vector{T}(undef, cmax * m), Vector{T}(undef, cmax * m),
                     Vector{T}(undef, cmax * m), Vector{T}(undef, cmax * m))
end

# transform the n rows of B lanes starting after element off of (xre, xim),
# in place, chunk by chunk. tw, when given, supplies per-element complex
# factors (row j, lane i) folded into the deconvolution (_rows_scale_tw!).
function _rs_block!(q::RsPass{T}, xre::Vector{T}, xim::Vector{T}, off::Int,
                    d::Val, tw=nothing) where {T}
    B, C = q.B, q.C
    nc = max(1, B ÷ C)
    for c in 0:nc-1
        i0 = c * C
        _rs_chunk!(q, xre, xim, off, i0, c == nc - 1 ? B - i0 : C, d, tw)
    end
    return nothing
end

function _rs_chunk!(q::RsPass{T}, xre::Vector{T}, xim::Vector{T}, off0::Int, i0::Int,
                    L::Int, d::Val, tw) where {T}
    n, m, w, B = q.n, q.m, q.w, q.B
    off = off0 + i0
    hp = cld(n, 2)
    pre, pim = q.pre, q.pim
    @inbounds for j in 0:n-1
        jj = j < hp ? j : m - n + j
        if tw === nothing
            _rows_scale!(pre, pim, L * jj, xre, xim, off + B * j, q.corr[j+1], L)
        else
            _rows_scale_tw!(pre, pim, L * jj, xre, xim, off + B * j, q.corr[j+1], L,
                            tw, j, i0)
        end
    end
    fill!(view(pre, L*hp+1:L*(m-n+hp)), zero(T))
    fill!(view(pim, L*hp+1:L*(m-n+hp)), zero(T))
    rng = 1:L*m
    gre, gim, _, _ = run_chain!(view(pre, rng), view(pim, rng), view(q.qre, rng),
                                view(q.qim, rng), q.stages, m, 1, L, d)
    @inbounds for k in 0:n-1
        _rows_interp!(xre, xim, off + B * k, gre, gim, q.wt, w * k,
                      Int(q.l0[k+1]), w, m, L)
    end
    return nothing
end

# y row at yo = c · x row at xo (B lanes; offsets are 0-based)
@inline function _rows_scale!(yre, yim, yo, xre, xim, xo, c::T, B) where {T}
    W = pick_width(T)
    if W > 1 && B ≥ W
        cv = Vec{W,T}(c)
        @inbounds for i in QWin(B, W)
            vstore(vload(Vec{W,T}, xre, xo + i) * cv, yre, yo + i)
            vstore(vload(Vec{W,T}, xim, xo + i) * cv, yim, yo + i)
        end
    else
        @inbounds for i in 1:B
            yre[yo+i] = xre[xo+i] * c
            yim[yo+i] = xim[xo+i] * c
        end
    end
    return nothing
end

# y row at yo = Σ_t wt[wo + t] · grid row (l + t) mod m
@inline function _rows_interp!(yre, yim, yo, gre, gim, wt::Vector{T}, wo, l, w, m, B) where {T}
    W = pick_width(T)
    if W > 1 && B ≥ W
        @inbounds for i in QWin(B, W)
            ar = ai = Vec{W,T}(zero(T))
            for t in 0:w-1
                g = l + t
                g += ifelse(g < 0, m, 0)
                g -= ifelse(g ≥ m, m, 0)
                c = Vec{W,T}(wt[wo+t+1])
                ar = muladd(c, vload(Vec{W,T}, gre, B * g + i), ar)
                ai = muladd(c, vload(Vec{W,T}, gim, B * g + i), ai)
            end
            vstore(ar, yre, yo + i)
            vstore(ai, yim, yo + i)
        end
    else
        @inbounds for i in 1:B
            ar = ai = zero(T)
            for t in 0:w-1
                g = l + t
                g += ifelse(g < 0, m, 0)
                g -= ifelse(g ≥ m, m, 0)
                ar = muladd(wt[wo+t+1], gre[B*g+i], ar)
                ai = muladd(wt[wo+t+1], gim[B*g+i], ai)
            end
            yre[yo+i] = ar
            yim[yo+i] = ai
        end
    end
    return nothing
end
