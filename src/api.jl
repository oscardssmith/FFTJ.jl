# FFTW-style convenience API: plans from example arrays, mul!/* application,
# inv, in-place plans, unnormalized backward transforms, and one-shot
# functions. Deliberately FFTJ's own functions rather than AbstractFFTs
# generic extensions, so loading FFTJ never changes FFTW's dispatch (or vice
# versa); AbstractFFTs is used only for its backend-neutral utilities
# (fftshift, fftfreq, ...), which are re-exported.

const AnyPlan{T} = Union{FFTPlan{T},BluesteinPlan{T}}

# In-place complex plans. Execution is alias-safe (the input is fully consumed
# into scratch before the output is written), so these wrap ordinary plans.
# Array-based constructors (with FFTW-style region and ignored planner flags)
# live in the N-d section below and cover N == 1.
struct InplacePlan{P}
    p::P
end

plan_fft!(::Type{T}, n::Integer) where {T<:AbstractFloat} = InplacePlan(plan_fft(T, n))
plan_ifft!(::Type{T}, n::Integer) where {T<:AbstractFloat} = InplacePlan(plan_ifft(T, n))
plan_bfft!(::Type{T}, n::Integer) where {T<:AbstractFloat} = InplacePlan(plan_bfft(T, n))

# a plan scaled by a constant, as produced by inv; .p and .scale are public
# (DSP unwraps inv(p).p to get the unnormalized inverse)
struct Scaled{P,T}
    p::P
    scale::T
end

# application: mul!(y, p, x) and p * x
LinearAlgebra.mul!(y::CVec{T}, p::AnyPlan{T}, x::CVec{T}) where {T} = p(y, x)
Base.:*(p::AnyPlan{T}, x::CVec{T}) where {T} = p(x)
LinearAlgebra.mul!(y::AbstractVector{Complex{T}}, p::RfftPlan{T}, x::Vector{T}) where {T} =
    p(y, x)
Base.:*(p::RfftPlan{T}, x::Vector{T}) where {T} = p(x)
LinearAlgebra.mul!(y::Vector{T}, p::IrfftPlan{T}, X::Vector{Complex{T}}) where {T} =
    p(y, X)
Base.:*(p::IrfftPlan{T}, X::Vector{Complex{T}}) where {T} = p(X)
function LinearAlgebra.mul!(y::AbstractArray, ip::InplacePlan, x::AbstractArray)
    ip.p(y, x)
    return y
end
function Base.:*(ip::InplacePlan, x::AbstractArray)
    ip.p(x, x)
    return x
end
function LinearAlgebra.mul!(y, s::Scaled, x)
    mul!(y, s.p, x)
    rmul!(y, s.scale)
    return y
end
Base.:*(s::Scaled, x) = rmul!(s.p * x, s.scale)

Base.size(p::AnyPlan) = (p.n,)
Base.size(p::RfftPlan) = (p.n,)
Base.size(p::IrfftPlan) = (p.n ÷ 2 + 1,)
Base.size(ip::InplacePlan) = size(ip.p)

_dir(::FFTPlan{T,D}) where {T,D} = D
_dir(::BluesteinPlan{T,D}) where {T,D} = D

# inv: forward ↦ scaled unnormalized backward (so inv(p).p is unnormalized,
# matching DSP's inv(plan_fft!(x)).p idiom); normalized inverse ↦ plain
# forward; unnormalized backward ↦ scaled forward.
function Base.inv(p::AnyPlan{T}) where {T}
    n = p.n
    if _dir(p) == -1
        return Scaled(plan_bfft(T, n), one(T) / n)
    elseif p.scale == 1
        return Scaled(plan_fft(T, n), one(T) / n)
    else
        return plan_fft(T, n)
    end
end
function Base.inv(ip::InplacePlan)
    s = inv(ip.p)
    return s isa Scaled ? Scaled(InplacePlan(s.p), s.scale) : InplacePlan(s)
end
# inv(c·P) = P⁻¹ / c
function Base.inv(s::Scaled)
    q = inv(s.p)
    return q isa Scaled ? Scaled(q.p, q.scale / s.scale) : Scaled(q, inv(s.scale))
end

# in-place one-shot transforms (out-of-place one-shots are in the N-d section)
function fft!(x::Array{Complex{T},N}) where {T<:AbstractFloat,N}
    p = plan_fft!(x)
    p * x
    return x
end
function ifft!(x::Array{Complex{T},N}) where {T<:AbstractFloat,N}
    p = plan_ifft!(x)
    p * x
    return x
end
function bfft!(x::Array{Complex{T},N}) where {T<:AbstractFloat,N}
    p = plan_bfft!(x)
    p * x
    return x
end

# --- N-d array methods -------------------------------------------------------

_checkregion_nd(region, N) =
    region == 1:N || region == ntuple(identity, N) || (N == 1 && region == 1) ||
    throw(ArgumentError("FFTJ N-d plans transform all dims (expected region 1:$N, got $region)"))

function plan_fft(x::Array{Complex{T},N}, region=1:N; kws...) where {T<:AbstractFloat,N}
    _checkregion_nd(region, N)
    return N == 1 ? plan_fft(T, length(x)) : _ndplan(T, size(x), -1, one(T))
end
function plan_bfft(x::Array{Complex{T},N}, region=1:N; kws...) where {T<:AbstractFloat,N}
    _checkregion_nd(region, N)
    return N == 1 ? plan_bfft(T, length(x)) : _ndplan(T, size(x), +1, one(T))
end
function plan_ifft(x::Array{Complex{T},N}, region=1:N; kws...) where {T<:AbstractFloat,N}
    _checkregion_nd(region, N)
    return N == 1 ? plan_ifft(T, length(x)) :
           _ndplan(T, size(x), +1, one(T) / length(x))
end
function plan_fft!(x::Array{Complex{T},N}, region=1:N; kws...) where {T<:AbstractFloat,N}
    _checkregion_nd(region, N)
    return InplacePlan(N == 1 ? plan_fft(T, length(x)) : _ndplan(T, size(x), -1, one(T)))
end
function plan_bfft!(x::Array{Complex{T},N}, region=1:N; kws...) where {T<:AbstractFloat,N}
    _checkregion_nd(region, N)
    return InplacePlan(N == 1 ? plan_bfft(T, length(x)) : _ndplan(T, size(x), +1, one(T)))
end
function plan_ifft!(x::Array{Complex{T},N}, region=1:N; kws...) where {T<:AbstractFloat,N}
    _checkregion_nd(region, N)
    return InplacePlan(N == 1 ? plan_ifft(T, length(x)) :
                       _ndplan(T, size(x), +1, one(T) / length(x)))
end
function plan_rfft(x::Array{T,N}, region=1:N; kws...) where {T<:AbstractFloat,N}
    _checkregion_nd(region, N)
    return N == 1 ? plan_rfft(T, length(x)) : _ndrfft(T, size(x))
end
function plan_brfft(X::Array{Complex{T},N}, n1::Integer, region=1:N;
                    kws...) where {T<:AbstractFloat,N}
    _checkregion_nd(region, N)
    size(X, 1) == n1 ÷ 2 + 1 ||
        throw(DimensionMismatch("brfft of leading length $n1 needs $(n1 ÷ 2 + 1) rows, got $(size(X, 1))"))
    return N == 1 ? plan_brfft(T, n1) :
           _ndirfft(T, (Int(n1), size(X)[2:N]...); normalized=false)
end
function plan_irfft(X::Array{Complex{T},N}, n1::Integer, region=1:N;
                    kws...) where {T<:AbstractFloat,N}
    _checkregion_nd(region, N)
    size(X, 1) == n1 ÷ 2 + 1 ||
        throw(DimensionMismatch("irfft of leading length $n1 needs $(n1 ÷ 2 + 1) rows, got $(size(X, 1))"))
    return N == 1 ? plan_irfft(T, n1) :
           _ndirfft(T, (Int(n1), size(X)[2:N]...); normalized=true)
end

LinearAlgebra.mul!(y::Array{Complex{T},N}, p::NdPlan{T,D,N},
                   x::Array{Complex{T},N}) where {T,D,N} = p(y, x)
Base.:*(p::NdPlan{T,D,N}, x::Array{Complex{T},N}) where {T,D,N} = p(x)
LinearAlgebra.mul!(Y::Array{Complex{T},N}, p::RfftNdPlan{T,N},
                   X::Array{T,N}) where {T,N} = p(Y, X)
Base.:*(p::RfftNdPlan{T,N}, X::Array{T,N}) where {T,N} = p(X)
LinearAlgebra.mul!(y::Array{T,N}, p::IrfftNdPlan{T,N},
                   X::Array{Complex{T},N}) where {T,N} = p(y, X)
Base.:*(p::IrfftNdPlan{T,N}, X::Array{Complex{T},N}) where {T,N} = p(X)

Base.size(p::NdPlan) = p.dims
Base.size(p::RfftNdPlan) = p.rdims
Base.size(p::IrfftNdPlan) = p.cdims

function Base.inv(p::NdPlan{T,D,N}) where {T,D,N}
    if D == -1
        return Scaled(_ndplan(T, p.dims, +1, one(T)), one(T) / p.len)
    elseif p.scale == 1
        return Scaled(_ndplan(T, p.dims, -1, one(T)), one(T) / p.len)
    else
        return _ndplan(T, p.dims, -1, one(T))
    end
end

# one-shot N-d transforms
fft(x::Array{Complex{T},N}) where {T<:AbstractFloat,N} = plan_fft(x) * x
ifft(x::Array{Complex{T},N}) where {T<:AbstractFloat,N} = plan_ifft(x) * x
bfft(x::Array{Complex{T},N}) where {T<:AbstractFloat,N} = plan_bfft(x) * x
fft(x::Array{T,N}) where {T<:AbstractFloat,N} = fft(Complex{T}.(x))
rfft(x::Array{T,N}) where {T<:AbstractFloat,N} = plan_rfft(x) * x
irfft(X::Array{Complex{T},N}, n1::Integer) where {T<:AbstractFloat,N} =
    plan_irfft(X, n1) * X
brfft(X::Array{Complex{T},N}, n1::Integer) where {T<:AbstractFloat,N} =
    plan_brfft(X, n1) * X
