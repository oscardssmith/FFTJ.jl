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
    scale::T
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
            push!(passes, DimPass{T}(nd, Bi, Bo, stages, nothing))
        else
            push!(passes, DimPass{T}(nd, Bi, Bo, Stage{T}[],
                                     _bluestein(T, nd, D, one(T))))
            maxfb = max(maxfb, nd)
        end
    end
    fuse_in = false
    fuse_out = false
    if !isempty(passes) && W > 1
        p1 = passes[1]
        if p1.fallback === nothing
            st1 = p1.stages[1]
            if p1.Bi == 1
                fuse_in = st1.radix == 4 && st1.small && (p1.nd >>> 2) % W == 0
            else
                fuse_in = st1.radix == 4 && p1.Bi ≥ W
            end
        end
        pl = passes[end]
        if pl.fallback === nothing
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
                         fuse_in, fuse_out, scale)
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
        sz = sizeof(T)
        cre, cim, ore, oim = p.are, p.aim, p.bre, p.bim
        stagestart = 1
        if p.fuse_in
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
        for (pi, pass) in enumerate(p.passes)
            if pass.fallback !== nothing
                _fallback_pass!(cre, cim, pass, p.pbuf, p.pbuf2)
                continue
            end
            from = pi == 1 ? stagestart : 1
            to = length(pass.stages)
            pi == np && p.fuse_out && (to -= 1)
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
    return y
end

function (p::NdPlan{T,D,N})(y::Array{Complex{T},N}, x::Array{Complex{T},N}) where {T,D,N}
    size(x) == p.dims || throw(DimensionMismatch("plan dims $(p.dims), input $(size(x))"))
    size(y) == p.dims || throw(DimensionMismatch("plan dims $(p.dims), output $(size(y))"))
    _exec_nd!(p, y, x)
    return y
end
(p::NdPlan{T,D,N})(x::Array{Complex{T},N}) where {T,D,N} = p(similar(x), x)

# --- real N-d transforms: 1-D real edge along dim 1, complex passes above ---

struct RfftNdPlan{T,N,R}
    rdims::NTuple{N,Int}
    cdims::NTuple{N,Int}
    r1::R                    # 1-D rfft plan for n1
    C::Vector{Complex{T}}    # (n1÷2+1)·rest intermediate
    rest::NdPlan{T,-1,N}     # dims 2..N, dim 1 as batch
end

function _ndrfft(::Type{T}, dims::NTuple{N,Int}) where {T<:AbstractFloat,N}
    n1 = dims[1]
    cdims = (n1 ÷ 2 + 1, dims[2:N]...)
    return RfftNdPlan{T,N,typeof(plan_rfft(T, n1))}(
        dims, cdims, plan_rfft(T, n1),
        Vector{Complex{T}}(undef, prod(cdims)),
        _ndplan(T, cdims, -1, one(T); first_dim=2))
end

function (p::RfftNdPlan{T,N})(Y::Array{Complex{T},N}, X::Array{T,N}) where {T,N}
    size(X) == p.rdims || throw(DimensionMismatch("plan dims $(p.rdims), input $(size(X))"))
    size(Y) == p.cdims || throw(DimensionMismatch("rfft output dims must be $(p.cdims), got $(size(Y))"))
    n1 = p.rdims[1]
    h1p = p.cdims[1]
    Bo = prod(p.rdims[2:N])
    C = p.C
    for o in 0:Bo-1
        p.r1(view(C, o*h1p+1:(o+1)*h1p), view(X, o*n1+1:(o+1)*n1))
    end
    _exec_nd!(p.rest, Y, C)
    return Y
end
(p::RfftNdPlan{T,N})(X::Array{T,N}) where {T,N} =
    p(Array{Complex{T},N}(undef, p.cdims), X)

struct IrfftNdPlan{T,N,R}
    rdims::NTuple{N,Int}
    cdims::NTuple{N,Int}
    rest::NdPlan{T,1,N}      # dims 2..N backward, dim 1 as batch
    C::Vector{Complex{T}}
    r1::R                    # 1-D irfft (normalized) or brfft plan for n1
end

function _ndirfft(::Type{T}, dims::NTuple{N,Int}; normalized::Bool) where {T<:AbstractFloat,N}
    n1 = dims[1]
    cdims = (n1 ÷ 2 + 1, dims[2:N]...)
    restscale = normalized ? one(T) / prod(dims[2:N]) : one(T)
    r1 = plan_irfft(T, n1; normalized)
    return IrfftNdPlan{T,N,typeof(r1)}(
        dims, cdims,
        _ndplan(T, cdims, +1, restscale; first_dim=2),
        Vector{Complex{T}}(undef, prod(cdims)), r1)
end

function (p::IrfftNdPlan{T,N})(y::Array{T,N}, X::Array{Complex{T},N}) where {T,N}
    size(X) == p.cdims || throw(DimensionMismatch("irfft input dims must be $(p.cdims), got $(size(X))"))
    size(y) == p.rdims || throw(DimensionMismatch("plan dims $(p.rdims), output $(size(y))"))
    n1 = p.rdims[1]
    h1p = p.cdims[1]
    Bo = prod(p.rdims[2:N])
    _exec_nd!(p.rest, p.C, X)
    C = p.C
    for o in 0:Bo-1
        p.r1(view(y, o*n1+1:(o+1)*n1), view(C, o*h1p+1:(o+1)*h1p))
    end
    return y
end
(p::IrfftNdPlan{T,N})(X::Array{Complex{T},N}) where {T,N} =
    p(Array{T,N}(undef, p.rdims), X)
