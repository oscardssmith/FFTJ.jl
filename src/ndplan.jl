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
    # serial clones of fallback for parallel pencil execution (share the cached
    # chirp/ghat/stage chains, own scratch); empty when the pass is serial
    fallbacks::Vector{Any}
end

struct NdPlan{T,D,N}
    dims::NTuple{N,Int}
    len::Int
    passes::Vector{DimPass{T}}
    are::Vector{T}
    aim::Vector{T}
    bre::Vector{T}
    bim::Vector{T}
    pbufs::Vector{Vector{Complex{T}}}  # per-task pencil gather buffers for
    pbufs2::Vector{Vector{Complex{T}}} # non-smooth dims ([1] is the serial one)
    fuse_in::Bool              # first pass's first stage reads interleaved input
    fuse_out::Bool             # last pass's final stage writes interleaved output
    dim1_batched::Bool         # dim 1 runs transposed with pencils as lanes
    nt::Int                    # task count for execution (1 = serial, zero-alloc)
    scale::T
end

plan_nt(p::NdPlan) = p.nt

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
                 first_dim::Int=1,
                 nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat,N}
    all(≥(1), dims) || throw(ArgumentError("dimensions must be positive (got $dims)"))
    len = prod(dims)
    W = pick_width(T)
    nt = plan_tasks(len, nthreads)
    passes = DimPass{T}[]
    maxfb = 0
    for d in first_dim:N
        nd = dims[d]
        nd == 1 && continue
        Bi = prod(dims[1:d-1])
        Bo = prod(dims[d+1:N])
        if factor_radices(nd) !== nothing
            stages, _ = build_stages_cached(T, nd, D; allow_small=Bi == 1)
            # dim-1 chains that don't earn the fused-edge treatment run
            # transposed, with pencils as vector lanes, on a no-small chain
            bstages = if Bi == 1 && Bo ≥ W && !pencil_fused(T, nd, stages)
                first(build_stages_cached(T, nd, D; allow_small=false))
            else
                Stage{T}[]
            end
            push!(passes, DimPass{T}(nd, Bi, Bo, stages, bstages, nothing, Any[]))
        else
            # pencils ≥ nt run parallel on serial clones; fewer pencils run a
            # serial loop over the primary plan (threaded internally)
            fbs = Any[]
            if nt > 1 && Bi * Bo ≥ nt
                for _ in 1:nt
                    push!(fbs, _bluestein(T, nd, D, one(T); nthreads=1))
                end
            end
            push!(passes, DimPass{T}(nd, Bi, Bo, Stage{T}[], Stage{T}[],
                                     _bluestein(T, nd, D, one(T); nthreads), fbs))
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
    npb = any(!isempty(p.fallbacks) for p in passes) ? nt : 1
    return NdPlan{T,D,N}(dims, len, passes,
                         Vector{T}(undef, len), Vector{T}(undef, len),
                         Vector{T}(undef, len), Vector{T}(undef, len),
                         [Vector{Complex{T}}(undef, maxfb) for _ in 1:npb],
                         [Vector{Complex{T}}(undef, maxfb) for _ in 1:npb],
                         fuse_in, fuse_out, dim1_batched, nt, scale)
end

# transpose both planes from (cre,cim) into (ore,oim); returns the roles
# swapped so the transposed data is the caller's new current plane
function transpose_planes!(ore, oim, cre, cim, R::Int, C::Int, nt::Int=1)
    if nt > 1
        W = pick_width(eltype(ore))
        tforeach((lo, hi) -> (transpose_plane!(ore, cre, R, C, lo:hi);
                              transpose_plane!(oim, cim, R, C, lo:hi)),
                 C, nt; align=W)
    else
        transpose_plane!(ore, cre, R, C)
        transpose_plane!(oim, cim, R, C)
    end
    return ore, oim, cre, cim
end

# pencils trange (1-based) of one fallback pass: gather -> 1-D plan -> scatter,
# in place at the split-plane level
function _fallback_range!(cre::Vector{T}, cim::Vector{T}, pass::DimPass{T}, fb,
                          buf::Vector{Complex{T}}, buf2::Vector{Complex{T}},
                          tlo::Int, thi::Int) where {T}
    nd, Bi = pass.nd, pass.Bi
    bv = view(buf, 1:nd)
    bv2 = view(buf2, 1:nd)
    @inbounds for t in tlo-1:thi-1
        o, i = divrem(t, Bi)
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

# per-pencil fallback pass: parallel over pencils when there are enough of
# them (each chunk on its own serial clone + scratch), else a serial loop over
# the primary plan, which threads its own inner transforms
function _fallback_pass!(cre::Vector{T}, cim::Vector{T}, pass::DimPass{T},
                         pbufs::Vector{Vector{Complex{T}}},
                         pbufs2::Vector{Vector{Complex{T}}}, nt::Int) where {T}
    total = pass.Bo * pass.Bi
    if nt > 1 && !isempty(pass.fallbacks) && total ≥ nt
        tforeach_chunks((ci, lo, hi) -> _fallback_range!(cre, cim, pass,
                                                         pass.fallbacks[ci],
                                                         pbufs[ci], pbufs2[ci],
                                                         lo, hi),
                        total, nt)
    else
        _fallback_range!(cre, cim, pass, pass.fallback, pbufs[1], pbufs2[1],
                         1, total)
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
        y .= p.scale .* x
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

# fused-input blocks olo:ohi (1-based); nt > 1 threads within each block
# (used when there are too few blocks to parallelize over)
function _fusein_blocks!(cre::Vector{T}, cim::Vector{T}, px::Ptr{T},
                         pass::DimPass{T}, st1::Stage{T}, d::Val,
                         olo::Int, ohi::Int, nt::Int) where {T}
    sz = sizeof(T)
    block = pass.Bi * pass.nd
    W = pick_width(T)
    @inbounds for o in olo-1:ohi-1
        off = o * block
        yr = view(cre, off+1:off+block)
        yi = view(cim, off+1:off+block)
        pb = px + sz * 2off
        if pass.Bi == 1
            if nt > 1
                let yr = yr, yi = yi, pb = pb
                    tforeach((lo, hi) -> stage4_first!(yr, yi, pb, pass.nd,
                                                       st1.twr, st1.twi, d, lo:hi),
                             pass.nd >>> 2, nt; align=W)
                end
            else
                stage4_first!(yr, yi, pb, pass.nd, st1.twr, st1.twi, d)
            end
        else
            if nt > 1
                let yr = yr, yi = yi, pb = pb
                    tforeach((lo, hi) -> stage4_ilvin!(yr, yi, pb, pass.nd,
                                                       pass.Bi, st1.twr, st1.twi,
                                                       d, (lo - 1):(hi - 1)),
                             pass.nd >>> 2, nt)
                end
            else
                stage4_ilvin!(yr, yi, pb, pass.nd, pass.Bi, st1.twr, st1.twi, d)
            end
        end
    end
    return nothing
end

# input side: convert x into split planes (or consume the first stage doing
# so); returns the current/other plane roles and the first pass's start stage
function _nd_input!(p::NdPlan{T,D,N}, x, px::Ptr{T}) where {T,D,N}
    cre, cim, ore, oim = p.are, p.aim, p.bre, p.bim
    nt = p.nt
    stagestart = 1
    @inbounds begin
        if p.dim1_batched
            # dim 1 transposed: pencils become vector lanes, the chain runs as
            # plain wide strided stages, then transpose back. (Named passb, not
            # pass1: sharing a name with the fuse_in branch's closure-captured
            # local would box it and allocate on every call.)
            passb = p.passes[1]
            n1, B = passb.nd, passb.Bo
            deinterleave_all!(cre, cim, x, nt)
            cre, cim, ore, oim = transpose_planes!(ore, oim, cre, cim, n1, B, nt)
            cre, cim, ore, oim =
                run_chain!(cre, cim, ore, oim, passb.bstages, n1, 1, B, Val(D), nt)
            cre, cim, ore, oim = transpose_planes!(ore, oim, cre, cim, B, n1, nt)
        elseif p.fuse_in
            pass1 = p.passes[1]
            st1 = pass1.stages[1]
            if nt > 1 && pass1.Bo ≥ nt
                # cre/cim are reassigned in the branch above; capture
                # let-bound copies so the closure doesn't box them
                let cr = cre, ci = cim
                    tforeach((lo, hi) -> _fusein_blocks!(cr, ci, px, pass1, st1,
                                                         Val(D), lo, hi, 1),
                             pass1.Bo, nt)
                end
            else
                _fusein_blocks!(cre, cim, px, pass1, st1, Val(D), 1, pass1.Bo, nt)
            end
            stagestart = 2
        else
            deinterleave_all!(cre, cim, x, nt)
        end
    end
    return cre, cim, ore, oim, stagestart
end

# the strided pass chain, plus (when emit) the fused/converted output. When
# emit=false the plan's reserved fuse_out stage still runs as a plain stage
# and no output conversion happens; returns the final plane roles.
# blocks olo:ohi (1-based) of one strided pass; nt > 1 threads within each
# block's chain (used when there are too few blocks to parallelize over)
function _pass_blocks!(cre, cim, ore, oim, chain, ncur0::Int, s0::Int, Bi::Int,
                       block::Int, olo::Int, ohi::Int, d::Val, nt::Int)
    @inbounds for o in olo-1:ohi-1
        rng = o*block+1:(o+1)*block
        run_chain!(view(cre, rng), view(cim, rng),
                   view(ore, rng), view(oim, rng),
                   chain, ncur0, s0, Bi, d, nt)
    end
    return nothing
end

function _nd_core!(p::NdPlan{T,D,N}, y, py::Ptr{T}, cre, cim, ore, oim,
                   stagestart::Int, skip1::Bool, emit::Bool) where {T,D,N}
    np = length(p.passes)
    nt = p.nt
    @inbounds begin
        for (pi, pass) in enumerate(p.passes)
            pi == 1 && skip1 && continue
            if pass.fallback !== nothing
                _fallback_pass!(cre, cim, pass, p.pbufs, p.pbufs2, nt)
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
            chain = view(pass.stages, from:to)
            if nt > 1 && pass.Bo ≥ nt
                # enough independent blocks: parallelize over them, chains
                # serial inside. cre/... are reassigned below, so capture
                # let-bound copies to avoid boxing them.
                let cr = cre, ci = cim, or = ore, oi = oim, chain = chain,
                    ncur0 = ncur0, s0 = s0, Bi = pass.Bi, block = block
                    tforeach((lo, hi) -> _pass_blocks!(cr, ci, or, oi, chain,
                                                       ncur0, s0, Bi, block,
                                                       lo, hi, Val(D), 1),
                             pass.Bo, nt)
                end
            else
                # few blocks (e.g. the last dim, Bo == 1): thread inside the
                # chains instead
                _pass_blocks!(cre, cim, ore, oim, chain, ncur0, s0, pass.Bi,
                              block, 1, pass.Bo, Val(D), nt)
            end
            if isodd(to - from + 1)
                cre, ore = ore, cre
                cim, oim = oim, cim
            end
        end
        if emit
            if p.fuse_out
                pl = p.passes[end]
                f = pl.stages[end].radix
                run_last_stage!(py, cre, cim, pl.Bi * (pl.nd ÷ f), f, p.scale,
                                Val(D), nt)
            else
                interleave_all!(y, cre, cim, p.scale, nt)
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
    factor_radices(h) === nothing && return false
    stages, _ = build_stages_cached(T, h, -1)
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
    r1s::Vector{R}             # serial clones for parallel pencils (may be empty)
    C::Vector{Complex{T}}      # per-pencil intermediate (empty when batched)
    rest::NdPlan{T,-1,N}       # dims 2..N, dim 1 as batch
    batched::Bool
    bstages::Vector{Stage{T}}  # forward chain for h = n1>>1, batched layout
end

plan_nt(p::RfftNdPlan) = p.rest.nt

function _ndrfft(::Type{T}, dims::NTuple{N,Int};
                 nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat,N}
    n1 = dims[1]
    cdims = (n1 ÷ 2 + 1, dims[2:N]...)
    B = prod(dims[2:N])
    batched = _real_batched(T, n1, B)
    bstages = batched ? first(build_stages_cached(T, n1 >>> 1, -1; allow_small=false)) :
              Stage{T}[]
    rest = _ndplan(T, cdims, -1, one(T); first_dim=2, nthreads)
    r1 = plan_rfft(T, n1; nthreads)
    R = typeof(r1)
    r1s = R[]
    if !batched && rest.nt > 1 && B ≥ rest.nt
        for _ in 1:rest.nt
            push!(r1s, plan_rfft(T, n1; nthreads=1))
        end
    end
    return RfftNdPlan{T,N,R}(
        dims, cdims, B, r1, r1s,
        Vector{Complex{T}}(undef, batched ? 0 : prod(cdims)),
        rest, batched, bstages)
end

function (p::RfftNdPlan{T,N})(Y::Array{Complex{T},N}, X::Array{T,N}) where {T,N}
    size(X) == p.rdims || throw(DimensionMismatch("plan dims $(p.rdims), input $(size(X))"))
    size(Y) == p.cdims || throw(DimensionMismatch("rfft output dims must be $(p.cdims), got $(size(Y))"))
    n1 = p.rdims[1]
    h1p = p.cdims[1]
    B = p.B
    r = p.rest
    nt = r.nt
    if p.batched
        # dim-1 real edge entirely in split planes, pencils as lanes: batched
        # half-length transform, across-pencil split/merge, transpose into the
        # rest chain's layout, then the complex passes with fused output.
        h = n1 >>> 1
        cre, cim, ore, oim = r.are, r.aim, r.bre, r.bim
        GC.@preserve X Y begin
            deinterleave_all!(cre, cim, reinterpret(Complex{T}, X), nt)
            cre, cim, ore, oim = transpose_planes!(ore, oim, cre, cim, h, B, nt)
            cre, cim, ore, oim =
                run_chain!(cre, cim, ore, oim, p.bstages, h, 1, B, Val(-1), nt)
            rfft_post_batched!(ore, oim, cre, cim, B, h, p.r1.twr, p.r1.twi, nt)
            cre, cim, ore, oim = transpose_planes!(cre, cim, ore, oim, B, h1p, nt)
            _nd_core!(r, Y, _rawptr(Y), cre, cim, ore, oim, 1, false, true)
        end
    else
        C = p.C
        if nt > 1 && !isempty(p.r1s)
            tforeach_chunks((ci, olo, ohi) -> _real_pencils!(p.r1s[ci], C, X,
                                                             h1p, n1, olo, ohi),
                            B, nt)
        else
            _real_pencils!(p.r1, C, X, h1p, n1, 1, B)
        end
        _exec_nd!(r, Y, C)
    end
    return Y
end

# real-edge pencils olo:ohi (1-based): src rows of srows → dst rows of drows
# through the 1-D plan r1 (rfft or irfft direction)
function _real_pencils!(r1, dst, src, drows::Int, srows::Int, olo::Int, ohi::Int)
    for o in olo-1:ohi-1
        r1(view(dst, o*drows+1:(o+1)*drows), view(src, o*srows+1:(o+1)*srows))
    end
    return nothing
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
    r1s::Vector{R}           # serial clones for parallel pencils (may be empty)
    batched::Bool
    bstages::Vector{Stage{T}}  # backward chain for h = n1>>1, batched layout
    bpre::T                  # pre-pass factor with all normalization folded in
end

plan_nt(p::IrfftNdPlan) = p.rest.nt

function _ndirfft(::Type{T}, dims::NTuple{N,Int}; normalized::Bool,
                  nthreads::Int=Threads.nthreads()) where {T<:AbstractFloat,N}
    n1 = dims[1]
    h = n1 >>> 1
    cdims = (n1 ÷ 2 + 1, dims[2:N]...)
    B = prod(dims[2:N])
    restscale = normalized ? one(T) / B : one(T)
    r1 = plan_irfft(T, n1; normalized, nthreads)
    batched = _real_batched(T, n1, B)
    bstages = batched ? first(build_stages_cached(T, h, +1; allow_small=false)) : Stage{T}[]
    # batched path skips both the rest chain's output scale and the 1-D inner
    # normalization, so fold everything into the pre-pass factor:
    # irfft total = 1/(n1·prod(dims[2:N])) = (1/2)·(1/h)·restscale
    bpre = normalized ? restscale / n1 : one(T)
    rest = _ndplan(T, cdims, +1, restscale; first_dim=2, nthreads)
    R = typeof(r1)
    r1s = R[]
    if !batched && rest.nt > 1 && B ≥ rest.nt
        for _ in 1:rest.nt
            push!(r1s, plan_irfft(T, n1; normalized, nthreads=1))
        end
    end
    return IrfftNdPlan{T,N,R}(
        dims, cdims, B, rest,
        Vector{Complex{T}}(undef, batched ? 0 : prod(cdims)), r1, r1s,
        batched, bstages, bpre)
end

function (p::IrfftNdPlan{T,N})(y::Array{T,N}, X::Array{Complex{T},N}) where {T,N}
    size(X) == p.cdims || throw(DimensionMismatch("irfft input dims must be $(p.cdims), got $(size(X))"))
    size(y) == p.rdims || throw(DimensionMismatch("plan dims $(p.rdims), output $(size(y))"))
    n1 = p.rdims[1]
    h1p = p.cdims[1]
    B = p.B
    r = p.rest
    nt = r.nt
    if p.batched
        h = n1 >>> 1
        GC.@preserve X y begin
            px = _rawptr(X)
            cre, cim, ore, oim, stagestart = _nd_input!(r, X, px)
            cre, cim, ore, oim =
                _nd_core!(r, X, px, cre, cim, ore, oim, stagestart,
                          r.dim1_batched, false)
            cre, cim, ore, oim = transpose_planes!(ore, oim, cre, cim, h1p, B, nt)
            irfft_pre_batched!(ore, oim, cre, cim, B, h, p.r1.twr, p.r1.twi,
                               p.bpre, nt)
            cre, cim, ore, oim =
                run_chain!(ore, oim, cre, cim, p.bstages, h, 1, B, Val(1), nt)
            cre, cim, ore, oim = transpose_planes!(ore, oim, cre, cim, B, h, nt)
            interleave_all!(reinterpret(Complex{T}, y), cre, cim, one(T), nt)
        end
    else
        _exec_nd!(r, p.C, X)
        C = p.C
        if nt > 1 && !isempty(p.r1s)
            tforeach_chunks((ci, olo, ohi) -> _real_pencils!(p.r1s[ci], y, C,
                                                             n1, h1p, olo, ohi),
                            B, nt)
        else
            _real_pencils!(p.r1, y, C, n1, h1p, 1, B)
        end
    end
    return y
end
(p::IrfftNdPlan{T,N})(X::Array{Complex{T},N}) where {T,N} =
    p(Array{T,N}(undef, p.rdims), X)
