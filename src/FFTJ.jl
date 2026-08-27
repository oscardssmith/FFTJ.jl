module FFTJ

using SIMD

export plan_fft, plan_ifft, plan_rfft, plan_irfft, FFTPlan

include("kernels.jl")
include("stockham.jl")
include("plan.jl")
include("bluestein.jl")
include("real.jl")

end # module
