using Test, FFTJ
import FFTW
using LinearAlgebra: norm, mul!
using Random: Xoshiro

@testset "complex fft/ifft vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0xff7)
    @testset "n = 2^$logn" for logn in 0:18
        n = 2^logn
        x = randn(rng, Complex{T}, n)
        ref = FFTW.fft(ComplexF64.(x))

        p = plan_fft(T, n)
        y = p(x)
        @test y isa Vector{Complex{T}}
        @test norm(y .- ref) ≤ 50 * eps(T) * norm(ref)

        pinv = plan_ifft(T, n)
        refinv = FFTW.ifft(ComplexF64.(x))
        @test norm(pinv(x) .- refinv) ≤ 50 * eps(T) * norm(refinv)

        # round trip
        @test norm(pinv(y) .- x) ≤ 50 * eps(T) * norm(x)

        # in-place-into-output form, and it must not allocate
        y2 = similar(x)
        p(y2, x)
        @test y2 == y
        p(y2, x)
        @test (@allocated p(y2, x)) == 0
    end
end

@testset "mixed radix vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0x357)
    ns = [3, 5, 6, 7, 9, 10, 12, 14, 15, 20, 21, 24, 25, 27, 35, 45, 48, 49, 60,
          63, 105, 120, 315, 27 * 32, 3 * 2^8, 9 * 2^8, 5 * 2^8, 7 * 2^8,
          15 * 2^7, 105 * 2^4, 5^3 * 2^6, 7^2 * 2^6, 3^4 * 5 * 7 * 2^5]
    @testset "n = $n" for n in ns
        x = randn(rng, Complex{T}, n)
        ref = FFTW.fft(ComplexF64.(x))
        p = plan_fft(T, n)
        y = p(x)
        @test norm(y .- ref) ≤ 50 * eps(T) * norm(ref)
        pinv = plan_ifft(T, n)
        @test norm(pinv(y) .- x) ≤ 50 * eps(T) * norm(x)
    end
end

@testset "bluestein (non-smooth n) vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0xb1e)
    ns = [11, 13, 17, 22, 26, 33, 39, 97, 101, 121, 169, 331, 997, 1025, 1234,
          2049, 10007]
    @testset "n = $n" for n in ns
        x = randn(rng, Complex{T}, n)
        ref = FFTW.fft(ComplexF64.(x))
        p = plan_fft(T, n)
        y = p(x)
        @test norm(y .- ref) ≤ 100 * eps(T) * norm(ref)
        pinv = plan_ifft(T, n)
        @test norm(pinv(y) .- x) ≤ 100 * eps(T) * norm(x)
        y2 = similar(x)
        p(y2, x)
        p(y2, x)
        @test (@allocated p(y2, x)) == 0
    end
end

@testset "rfft/irfft vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0x4ea1)
    ns = [1, 2, 3, 4, 5, 6, 8, 11, 16, 22, 34, 64, 100, 255, 256, 257, 1024,
          768, 1000, 4096, 2^14, 3 * 2^10, 7 * 2^8]
    @testset "n = $n" for n in ns
        x = randn(rng, T, n)
        ref = FFTW.rfft(Float64.(x))
        p = plan_rfft(T, n)
        y = p(x)
        @test length(y) == n ÷ 2 + 1
        @test norm(y .- ref) ≤ 100 * eps(T) * norm(ref)

        pinv = plan_irfft(T, n)
        xr = pinv(y)
        @test norm(xr .- x) ≤ 100 * eps(T) * norm(x)
        refinv = FFTW.irfft(ComplexF64.(y), n)
        @test norm(pinv(y) .- refinv) ≤ 100 * eps(T) * max(norm(refinv), one(T))

        if n > 4
            y2 = similar(y)
            p(y2, x)
            p(y2, x)
            @test (@allocated p(y2, x)) == 0
            x2 = similar(x)
            pinv(x2, y)
            pinv(x2, y)
            @test (@allocated pinv(x2, y)) == 0
        end
    end
end

@testset "FFTW-compat API ($T, n=$n)" for T in (Float32, Float64), n in (96, 11)
    rng = Xoshiro(0xc0a7)
    tol(ref) = 100 * eps(T) * max(norm(ref), one(T))
    x = randn(rng, Complex{T}, n)
    xr = randn(rng, T, n)

    # array-based planning, * and mul!
    p = plan_fft(x; flags=:ignored)
    @test p * x ≈ FFTW.fft(x) atol = tol(FFTW.fft(x))
    y = similar(x)
    @test mul!(y, p, x) === y
    @test y ≈ FFTW.fft(x) atol = tol(FFTW.fft(x))

    # unnormalized backward
    pb = plan_bfft(x)
    @test pb * x ≈ FFTW.bfft(x) atol = tol(FFTW.bfft(x))

    # in-place: DSP's complex conv idiom `p! * buf; ...; inv(p!).p`
    p! = plan_fft!(x)
    z = copy(x)
    @test (p! * z) === z
    @test z ≈ FFTW.fft(x) atol = tol(FFTW.fft(x))
    ip = inv(p!).p
    @test (ip * z) === z
    @test z ≈ T(n) .* x atol = tol(n .* x)
    ipfull = inv(p!)
    z2 = copy(x)
    p! * z2
    ipfull * z2
    @test z2 ≈ x atol = tol(x)

    # rfft/brfft: DSP's _fftfilt!/dpss idiom
    pr = plan_rfft(xr)
    X = pr * xr
    @test X ≈ FFTW.rfft(xr) atol = tol(FFTW.rfft(xr))
    pbr = plan_brfft(X, n)
    @test pbr * X ≈ FFTW.brfft(X, n) atol = tol(FFTW.brfft(X, n))
    yr = similar(xr)
    @test mul!(yr, pbr, X) === yr
    @test yr ≈ FFTW.brfft(X, n) atol = tol(FFTW.brfft(X, n))

    # hilbert idiom: mul! into a view of a longer buffer
    if iseven(n)
        Xbig = zeros(Complex{T}, n)
        mul!(view(Xbig, 1:(n >> 1) + 1), pr, xr)
        @test Xbig[1:(n >> 1) + 1] ≈ FFTW.rfft(xr) atol = tol(FFTW.rfft(xr))
    end

    # one-shot functions
    @test fft(x) ≈ FFTW.fft(x) atol = tol(FFTW.fft(x))
    @test ifft(x) ≈ FFTW.ifft(x) atol = tol(FFTW.ifft(x))
    @test bfft(x) ≈ FFTW.bfft(x) atol = tol(FFTW.bfft(x))
    @test rfft(xr) ≈ FFTW.rfft(xr) atol = tol(FFTW.rfft(xr))
    @test irfft(X, n) ≈ FFTW.irfft(X, n) atol = tol(FFTW.irfft(X, n))
    @test brfft(X, n) ≈ FFTW.brfft(X, n) atol = tol(FFTW.brfft(X, n))
    @test ifft!(copy(x)) ≈ FFTW.ifft(x) atol = tol(FFTW.ifft(x))
    @test fft!(copy(x)) ≈ FFTW.fft(x) atol = tol(FFTW.fft(x))

    # re-exported utilities
    @test fftshift(fftfreq(4)) == [-0.5, -0.25, 0.0, 0.25]
    @test rfftfreq(n) isa Frequencies

    # region validation
    @test_throws ArgumentError plan_fft(x, 2)
end

@testset "argument checking" begin
    @test_throws ArgumentError plan_fft(Float64, 0)
    p = plan_fft(Float64, 8)
    @test_throws DimensionMismatch p(randn(ComplexF64, 4))
end
