module FFTJ

using SIMD
using LinearAlgebra: LinearAlgebra, mul!, rmul!
import AbstractFFTs

# backend-neutral utilities, same names FFTW re-exports
using AbstractFFTs: fftshift, ifftshift, fftfreq, rfftfreq, Frequencies

export plan_fft, plan_ifft, plan_bfft, plan_fft!, plan_ifft!, plan_bfft!,
       plan_rfft, plan_irfft, plan_brfft,
       fft, ifft, bfft, fft!, ifft!, bfft!, rfft, irfft, brfft,
       fftshift, ifftshift, fftfreq, rfftfreq, Frequencies,
       FFTPlan

include("threads.jl")
include("kernels.jl")
include("stockham.jl")
include("plan.jl")
include("resample.jl")
include("real.jl")
include("ndplan.jl")
include("fourstep.jl")
include("sixstep.jl")
include("composite.jl")
include("hvdh.jl")
include("api.jl")

end # module
