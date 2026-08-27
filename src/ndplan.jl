# N-dimensional plans: per-dim batched-strided Stockham passes over split
# planes. A pass along dim d runs the ordinary stage kernels with effective
# stride s·Bi (Bi = prod(dims[1:d-1])) once per outer block (Bo blocks of
# Bi·nd contiguous elements), so the inner batch rides in the vectorized
# q-loop. Design: docs/superpowers/specs/2026-08-27-fftj-nd-transforms-design.md

struct DimPass{T}
    nd::Int
    Bi::Int
    Bo::Int
    stages::Vector{Stage{T}}   # empty when fallback is used
    bstages::Vector{Stage{T}}  # batched-transposed chain (dim 1 only; may be empty)
    fallback::Union{Nothing,BluesteinPlan{T,-1},BluesteinPlan{T,1}}
end

struct NdPlan{T,D,N}
    dims::NTuple{N,Int}
    len::Int
    passes::Vector{DimPass{T}}
    are::Vector{T}
    aim::Vector{T}
    bre::Vector{T}
    bim::Vector{T}
    pbuf::Vector{Complex{T}}   # pencil gather buffers for non-smooth dims
    pbuf2::Vector{Complex{T}}
    fuse_in::Bool              # first pass's first stage reads interleaved input
    fuse_out::Bool             # last pass's final stage writes interleaved output
    dim1_batched::Bool         # dim 1 runs transposed with pencils as lanes
    scale::T
end

# would a per-pencil chain of length n get the fast fused-edge 1-D treatment?
function pencil_fused(::Type{T}, n::Int, stages::Vector{Stage{T}}) where {T}
    W = pick_width(T)
    isempty(stages) && return false
    f = stages[end].radix
    slast = f in (2, 4, 8) ? n ÷ f : 0
    return W > 1 && length(stages) ≥ 2 &&
           stages[1].radix == 4 && stages[1].small && (n >>> 2) % W == 0 &&
           slast ≥ W && slast % W == 0
end

# dims before first_dim are pure batch (rfft reuses this for dims 2..N)
function _ndplan(::Type{T}, dims::NTuple{N,Int}, D::Int, scale::T;
                 first_dim::Int=1) where {T<:AbstractFloat,N}
    all(≥(1), dims) || throw(ArgumentError("dimensions must be positive (got $dims)"))
    len = prod(dims)
    W = pick_width(T)
    passes = DimPass{T}[]
    maxfb = 0
    for d in first_dim:N
        nd = dims[d]
        nd == 1 && continue
        Bi = prod(dims[1:d-1])
        Bo = prod(dims[d+1:N])
        if factor_smooth(nd) !== nothing
            stages, _ = build_stages(T, nd, D; allow_small=Bi == 1)
            # dim-1 chains that don't earn the fused-edge treatment run
            # transposed, with pencils as vector lanes, on a no-small chain
            bstages = if Bi == 1 && Bo ≥ W && !pencil_fused(T, nd, stages)
                first(build_stages(T, nd, D; allow_small=false))
            else
                Stage{T}[]
            end
            push!(passes, DimPass{T}(nd, Bi, Bo, stages, bstages, nothing))
        else
            push!(passes, DimPass{T}(nd, Bi, Bo, Stage{T}[], Stage{T}[],
                                     _bluestein(T, nd, D, one(T))))
            maxfb = max(maxfb, nd)
        end
    end
    fuse_in = false
    fuse_out = false
    dim1_batched = !isempty(passes) && !isempty(passes[1].bstages)
    if !isempty(passes) && W > 1
        p1 = passes[1]
        if p1.fallback === nothing && !dim1_batched
            st1 = p1.stages[1]
            if p1.Bi == 1
                fuse_in = st1.radix == 4 && st1.small && (p1.nd >>> 2) % W == 0
            else
                fuse_in = st1.radix == 4 && p1.Bi ≥ W
            end
        end
        pl = passes[end]
        if pl.fallback === nothing && !(dim1_batched && length(passes) == 1)
            lst = pl.stages[end]
            fuse_out = lst.radix in (2, 4, 8) &&
                       pl.Bi * (pl.nd ÷ lst.radix) ≥ W
        end
        # a single stage overall can't serve both edges
        total = sum(length(p.stages) for p in passes)
        if fuse_in && fuse_out && total < 2
            fuse_in = false
        end
    end
    return NdPlan{T,D,N}(dims, len, passes,
                         Vector{T}(undef, len), Vector{T}(undef, len),
                         Vector{T}(undef, len), Vector{T}(undef, len),
                         Vector{Complex{T}}(undef, maxfb),
                         Vector{Complex{T}}(undef, maxfb),
                         fuse_in, fuse_out, dim1_batched, scale)
end

# per-pencil gather -> 1-D plan -> scatter, in place at the split-plane level
function _fallback_pass!(cre::Vector{T}, cim::Vector{T}, pass::DimPass{T},
                         buf::Vector{Complex{T}}, buf2::Vector{Complex{T}}) where {T}
    nd, Bi = pass.nd, pass.Bi
    fb = pass.fallback
    bv = view(buf, 1:nd)
    bv2 = view(buf2, 1:nd)
    @inbounds for o in 0:pass.Bo-1, i in 0:Bi-1
        base = o * Bi * nd + i + 1
        for k in 0:nd-1
            bv[k+1] = Complex(cre[base+Bi*k], cim[base+Bi*k])
        end
        fb(bv2, bv)
        for k in 0:nd-1
            z = bv2[k+1]
            cre[base+Bi*k] = real(z)
            cim[base+Bi*k] = imag(z)
        end
    end
    return nothing
end

function _exec_nd!(p::NdPlan{T,D,N}, y::AbstractArray{Complex{T}},
                   x::AbstractArray{Complex{T}}) where {T,D,N}
    len = p.len
    length(x) == len || throw(DimensionMismatch("plan length $len, input length $(length(x))"))
    length(y) == len || throw(DimensionMismatch("plan length $len, output length $(length(y))"))
    np = length(p.passes)
    if np == 0
        sc = p.scale
        @inbounds for c in 1:len
            y[c] = sc * x[c]
        end
        return y
    end
    GC.@preserve x y begin
        px = _rawptr(x)
        py = _rawptr(y)
        cre, cim, ore, oim, stagestart = _nd_input!(p, x, px)
        _nd_core!(p, y, py, cre, cim, ore, oim, stagestart, p.dim1_batched, true)
    end
    return y
end

# input side: convert x into split planes (or consume the first stage doing
# so); returns the current/other plane roles and the first pass's start stage
function _nd_input!(p::NdPlan{T,D,N}, x, px::Ptr{T}) where {T,D,N}
    sz = sizeof(T)
    cre, cim, ore, oim = p.are, p.aim, p.bre, p.bim
    stagestart = 1
    @inbounds begin
        if p.dim1_batched
            # dim 1 transposed: pencils become vector lanes, the chain runs as
            # plain wide strided stages, then transpose back
            pass1 = p.passes[1]
            n1, B = pass1.nd, pass1.Bo
            deinterleave_all!(cre, cim, x)
            transpose_plane!(ore, cre, n1, B)
            transpose_plane!(oim, cim, n1, B)
            cre, ore = ore, cre
            cim, oim = oim, cim
            ncur = n1
            s = 1
            for st in pass1.bstages
                run_stage!(ore, oim, cre, cim, ncur, s * B, st, Val(D))
                cre, ore = ore, cre
                cim, oim = oim, cim
                s *= st.radix
                ncur ÷= st.radix
            end
            transpose_plane!(ore, cre, B, n1)
            transpose_plane!(oim, cim, B, n1)
            cre, ore = ore, cre
            cim, oim = oim, cim
        elseif p.fuse_in
            pass1 = p.passes[1]
            st1 = pass1.stages[1]
            block = pass1.Bi * pass1.nd
            for o in 0:pass1.Bo-1
                off = o * block
                yr = view(cre, off+1:off+block)
                yi = view(cim, off+1:off+block)
                if pass1.Bi == 1
                    stage4_first!(yr, yi, px + sz * 2off, pass1.nd,
                                  st1.twr, st1.twi, Val(D))
                else
                    stage4_ilvin!(yr, yi, px + sz * 2off, pass1.nd, pass1.Bi,
                                  st1.twr, st1.twi, Val(D))
                end
            end
            stagestart = 2
        else
            deinterleave_all!(cre, cim, x)
        end
    end
    return cre, cim, ore, oim, stagestart
end

# the strided pass chain, plus (when emit) the fused/converted output. When
# emit=false the plan's reserved fuse_out stage still runs as a plain stage
# and no output conversion happens; returns the final plane roles.
function _nd_core!(p::NdPlan{T,D,N}, y, py::Ptr{T}, cre, cim, ore, oim,
                   stagestart::Int, skip1::Bool, emit::Bool) where {T,D,N}
    np = length(p.passes)
    @inbounds begin
        for (pi, pass) in enumerate(p.passes)
            pi == 1 && skip1 && continue
            if pass.fallback !== nothing
                _fallback_pass!(cre, cim, pass, p.pbuf, p.pbuf2)
                continue
            end
            from = pi == 1 ? stagestart : 1
            to = length(pass.stages)
            pi == np && p.fuse_out && emit && (to -= 1)
            to < from && continue
            ncur0 = pass.nd
            s0 = 1
            for k in 1:from-1
                s0 *= pass.stages[k].radix
                ncur0 ÷= pass.stages[k].radix
            end
            block = pass.Bi * pass.nd
            for o in 0:pass.Bo-1
                rng = o*block+1:(o+1)*block
                ar = view(cre, rng); ai = view(cim, rng)
                br = view(ore, rng); bi = view(oim, rng)
                ncur = ncur0
                s = s0
                for k in from:to
                    st = pass.stages[k]
                    run_stage!(br, bi, ar, ai, ncur, s * pass.Bi, st, Val(D))
                    ar, br = br, ar
                    ai, bi = bi, ai
                    s *= st.radix
                    ncur ÷= st.radix
                end
            end
            if isodd(to - from + 1)
                cre, ore = ore, cre
                cim, oim = oim, cim
            end
        end
        if emit
            if p.fuse_out
                pl = p.passes[end]
                lst = pl.stages[end]
                f = lst.radix
                s_eff = pl.Bi * (pl.nd ÷ f)
                if f == 4
                    stage4_last!(py, cre, cim, s_eff, p.scale, Val(D))
                elseif f == 8
                    stage8_last!(py, cre, cim, s_eff, p.scale, Val(D))
                else
                    stage2_last!(py, cre, cim, s_eff, p.scale)
                end
            else
                interleave_all!(y, cre, cim, p.scale)
            end
        end
    end
    return cre, cim, ore, oim
end

function (p::NdPlan{T,D,N})(y::Array{Complex{T},N}, x::Array{Complex{T},N}) where {T,D,N}
    size(x) == p.dims || throw(DimensionMismatch("plan dims $(p.dims), input $(size(x))"))
    size(y) == p.dims || throw(DimensionMismatch("plan dims $(p.dims), output $(size(y))"))
    _exec_nd!(p, y, x)
    return y
end
(p::NdPlan{T,D,N})(x::Array{Complex{T},N}) where {T,D,N} = p(similar(x), x)

# --- real N-d transforms: 1-D real edge along dim 1, complex passes above ---

# Batched real edge applies when the half-length chain is smooth and the
# batch fills vector lanes. When the per-pencil path would get the fused 1-D
# treatment anyway, the batched path's transpose passes only pay for
# themselves while the working set is cache-resident.
function _real_batched(::Type{T}, n1::Int, B::Int) where {T}
    W = pick_width(T)
    (W > 1 && iseven(n1) && n1 ≥ 2 && B ≥ W) || return false
    h = n1 >>> 1
    factor_smooth(h) === nothing && return false
    stages, _ = build_stages(T, h, -1)
    pencil_fused(T, h, stages) || return true
    # measured crossover: with fused pencils available, the transpose overhead
    # only pays at W = 16 (Float32) and while the working set is L2-resident
    return W ≥ 16 && (h + 1) * B * sizeof(T) ≤ (1 << 19)
end

struct RfftNdPlan{T,N,R}
    rdims::NTuple{N,Int}
    cdims::NTuple{N,Int}
    B::Int                     # batch = prod(dims[2:N])
    r1::R                      # 1-D rfft plan for n1 (twiddle source when batched)
    C::Vector{Complex{T}}      # per-pencil intermediate (empty when batched)
    rest::NdPlan{T,-1,N}       # dims 2..N, dim 1 as batch
    batched::Bool
    bstages::Vector{Stage{T}}  # forward chain for h = n1>>1, batched layout
end

function _ndrfft(::Type{T}, dims::NTuple{N,Int}) where {T<:AbstractFloat,N}
    n1 = dims[1]
    cdims = (n1 ÷ 2 + 1, dims[2:N]...)
    B = prod(dims[2:N])
    batched = _real_batched(T, n1, B)
    bstages = batched ? first(build_stages(T, n1 >>> 1, -1; allow_small=false)) :
              Stage{T}[]
    return RfftNdPlan{T,N,typeof(plan_rfft(T, n1))}(
        dims, cdims, B, plan_rfft(T, n1),
        Vector{Complex{T}}(undef, batched ? 0 : prod(cdims)),
        _ndplan(T, cdims, -1, one(T); first_dim=2), batched, bstages)
end

function (p::RfftNdPlan{T,N})(Y::Array{Complex{T},N}, X::Array{T,N}) where {T,N}
    size(X) == p.rdims || throw(DimensionMismatch("plan dims $(p.rdims), input $(size(X))"))
    size(Y) == p.cdims || throw(DimensionMismatch("rfft output dims must be $(p.cdims), got $(size(Y))"))
    n1 = p.rdims[1]
    h1p = p.cdims[1]
    B = p.B
    r = p.rest
    if p.batched
        # dim-1 real edge entirely in split planes, pencils as lanes: batched
        # half-length transform, across-pencil split/merge, transpose into the
        # rest chain's layout, then the complex passes with fused output.
        h = n1 >>> 1
        cre, cim, ore, oim = r.are, r.aim, r.bre, r.bim
        GC.@preserve X Y begin
            deinterleave_all!(cre, cim, reinterpret(Complex{T}, X))
            transpose_plane!(ore, cre, h, B)
            transpose_plane!(oim, cim, h, B)
            cre, ore = ore, cre
            cim, oim = oim, cim
            ncur = h
            s = 1
            for st in p.bstages
                run_stage!(ore, oim, cre, cim, ncur, s * B, st, Val(-1))
                cre, ore = ore, cre
                cim, oim = oim, cim
                s *= st.radix
                ncur ÷= st.radix
            end
            rfft_post_batched!(ore, oim, cre, cim, B, h, p.r1.twr, p.r1.twi)
            cre, ore = ore, cre
            cim, oim = oim, cim
            transpose_plane!(ore, cre, B, h1p)
            transpose_plane!(oim, cim, B, h1p)
            cre, ore = ore, cre
            cim, oim = oim, cim
            _nd_core!(r, Y, _rawptr(Y), cre, cim, ore, oim, 1, false, true)
        end
    else
        C = p.C
        for o in 0:B-1
            p.r1(view(C, o*h1p+1:(o+1)*h1p), view(X, o*n1+1:(o+1)*n1))
        end
        _exec_nd!(r, Y, C)
    end
    return Y
end
(p::RfftNdPlan{T,N})(X::Array{T,N}) where {T,N} =
    p(Array{Complex{T},N}(undef, p.cdims), X)

struct IrfftNdPlan{T,N,R}
    rdims::NTuple{N,Int}
    cdims::NTuple{N,Int}
    B::Int
    rest::NdPlan{T,1,N}      # dims 2..N backward, dim 1 as batch
    C::Vector{Complex{T}}    # per-pencil intermediate (empty when batched)
    r1::R                    # 1-D irfft (normalized) or brfft plan for n1
    batched::Bool
    bstages::Vector{Stage{T}}  # backward chain for h = n1>>1, batched layout
    bpre::T                  # pre-pass factor with all normalization folded in
end

function _ndirfft(::Type{T}, dims::NTuple{N,Int}; normalized::Bool) where {T<:AbstractFloat,N}
    n1 = dims[1]
    h = n1 >>> 1
    cdims = (n1 ÷ 2 + 1, dims[2:N]...)
    B = prod(dims[2:N])
    restscale = normalized ? one(T) / B : one(T)
    r1 = plan_irfft(T, n1; normalized)
    batched = _real_batched(T, n1, B)
    bstages = batched ? first(build_stages(T, h, +1; allow_small=false)) : Stage{T}[]
    # batched path skips both the rest chain's output scale and the 1-D inner
    # normalization, so fold everything into the pre-pass factor:
    # irfft total = 1/(n1·prod(dims[2:N])) = (1/2)·(1/h)·restscale
    bpre = normalized ? restscale / n1 : one(T)
    return IrfftNdPlan{T,N,typeof(r1)}(
        dims, cdims, B,
        _ndplan(T, cdims, +1, restscale; first_dim=2),
        Vector{Complex{T}}(undef, batched ? 0 : prod(cdims)), r1,
        batched, bstages, bpre)
end

function (p::IrfftNdPlan{T,N})(y::Array{T,N}, X::Array{Complex{T},N}) where {T,N}
    size(X) == p.cdims || throw(DimensionMismatch("irfft input dims must be $(p.cdims), got $(size(X))"))
    size(y) == p.rdims || throw(DimensionMismatch("plan dims $(p.rdims), output $(size(y))"))
    n1 = p.rdims[1]
    h1p = p.cdims[1]
    B = p.B
    r = p.rest
    if p.batched
        h = n1 >>> 1
        GC.@preserve X y begin
            px = _rawptr(X)
            cre, cim, ore, oim, stagestart = _nd_input!(r, X, px)
            cre, cim, ore, oim =
                _nd_core!(r, X, px, cre, cim, ore, oim, stagestart,
                          r.dim1_batched, false)
            transpose_plane!(ore, cre, h1p, B)
            transpose_plane!(oim, cim, h1p, B)
            cre, ore = ore, cre
            cim, oim = oim, cim
            irfft_pre_batched!(ore, oim, cre, cim, B, h, p.r1.twr, p.r1.twi, p.bpre)
            cre, ore = ore, cre
            cim, oim = oim, cim
            ncur = h
            s = 1
            for st in p.bstages
                run_stage!(ore, oim, cre, cim, ncur, s * B, st, Val(1))
                cre, ore = ore, cre
                cim, oim = oim, cim
                s *= st.radix
                ncur ÷= st.radix
            end
            transpose_plane!(ore, cre, B, h)
            transpose_plane!(oim, cim, B, h)
            cre, ore = ore, cre
            cim, oim = oim, cim
            interleave_all!(reinterpret(Complex{T}, y), cre, cim, one(T))
        end
    else
        _exec_nd!(r, p.C, X)
        C = p.C
        for o in 0:B-1
            p.r1(view(y, o*n1+1:(o+1)*n1), view(C, o*h1p+1:(o+1)*h1p))
        end
    end
    return y
end
(p::IrfftNdPlan{T,N})(X::Array{Complex{T},N}) where {T,N} =
    p(Array{T,N}(undef, p.rdims), X)
