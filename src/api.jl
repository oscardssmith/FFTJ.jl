# FFTW-style convenience API: plans from example arrays, mul!/* application,
# inv, in-place plans, unnormalized backward transforms, and one-shot
# functions. Deliberately FFTJ's own functions rather than AbstractFFTs
# generic extensions, so loading FFTJ never changes FFTW's dispatch (or vice
# versa); AbstractFFTs is used only for its backend-neutral utilities
# (fftshift, fftfreq, ...), which are re-exported.

const AnyPlan{T} = Union{FFTPlan{T},BluesteinPlan{T}}

# region: FFTJ is 1-D only for now; accept and validate FFTW-style region args
_checkregion(region) =
    region in (1, (1,), 1:1) ||
    throw(ArgumentError("FFTJ supports 1-D transforms along dimension 1 only (got region=$region)"))

# planner flags (MEASURE/ESTIMATE etc.) are meaningless here; accept and ignore
plan_fft(x::AbstractVector{Complex{T}}, region=1; kws...) where {T<:AbstractFloat} =
    (_checkregion(region); plan_fft(T, length(x)))
plan_ifft(x::AbstractVector{Complex{T}}, region=1; kws...) where {T<:AbstractFloat} =
    (_checkregion(region); plan_ifft(T, length(x)))
plan_bfft(x::AbstractVector{Complex{T}}, region=1; kws...) where {T<:AbstractFloat} =
    (_checkregion(region); plan_bfft(T, length(x)))
plan_rfft(x::AbstractVector{T}, region=1; kws...) where {T<:AbstractFloat} =
    (_checkregion(region); plan_rfft(T, length(x)))
function plan_brfft(X::AbstractVector{Complex{T}}, n::Integer, region=1;
                    kws...) where {T<:AbstractFloat}
    _checkregion(region)
    length(X) == n ÷ 2 + 1 ||
        throw(DimensionMismatch("brfft of length $n needs $(n ÷ 2 + 1) inputs, got $(length(X))"))
    return plan_brfft(T, n)
end
plan_irfft(X::AbstractVector{Complex{T}}, n::Integer, region=1; kws...) where {T<:AbstractFloat} =
    (_checkregion(region); plan_irfft(T, n))

# In-place complex plans. Execution is alias-safe (the input is fully consumed
# into scratch before the output is written), so these wrap ordinary plans.
struct InplacePlan{P}
    p::P
end
plan_fft!(x::AbstractVector{Complex{T}}, region=1; kws...) where {T<:AbstractFloat} =
    (_checkregion(region); InplacePlan(plan_fft(T, length(x))))
plan_ifft!(x::AbstractVector{Complex{T}}, region=1; kws...) where {T<:AbstractFloat} =
    (_checkregion(region); InplacePlan(plan_ifft(T, length(x))))
plan_bfft!(x::AbstractVector{Complex{T}}, region=1; kws...) where {T<:AbstractFloat} =
    (_checkregion(region); InplacePlan(plan_bfft(T, length(x))))

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
function LinearAlgebra.mul!(y::CVec{T}, ip::InplacePlan, x::CVec{T}) where {T}
    ip.p(y, x)
    return y
end
function Base.:*(ip::InplacePlan, x::CVec{T}) where {T}
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

# one-shot transforms
fft(x::AbstractVector{Complex{T}}) where {T<:AbstractFloat} =
    plan_fft(T, length(x))(Vector(x))
fft(x::AbstractVector{T}) where {T<:AbstractFloat} = fft(Complex{T}.(x))
ifft(x::AbstractVector{Complex{T}}) where {T<:AbstractFloat} =
    plan_ifft(T, length(x))(Vector(x))
bfft(x::AbstractVector{Complex{T}}) where {T<:AbstractFloat} =
    plan_bfft(T, length(x))(Vector(x))
rfft(x::AbstractVector{T}) where {T<:AbstractFloat} = plan_rfft(T, length(x))(Vector(x))
irfft(X::AbstractVector{Complex{T}}, n::Integer) where {T<:AbstractFloat} =
    plan_irfft(T, n)(Vector(X))
brfft(X::AbstractVector{Complex{T}}, n::Integer) where {T<:AbstractFloat} =
    plan_brfft(T, n)(Vector(X))
function fft!(x::Vector{Complex{T}}) where {T<:AbstractFloat}
    plan_fft(T, length(x))(x, x)
    return x
end
function ifft!(x::Vector{Complex{T}}) where {T<:AbstractFloat}
    plan_ifft(T, length(x))(x, x)
    return x
end
function bfft!(x::Vector{Complex{T}}) where {T<:AbstractFloat}
    plan_bfft(T, length(x))(x, x)
    return x
end
