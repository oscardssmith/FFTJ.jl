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

        # in-place-into-output form, and it must not allocate (threaded plans
        # may allocate task objects, so the guarantee is for nthreads=1)
        y2 = similar(x)
        p(y2, x)
        @test y2 == y
        p1 = plan_fft(T, n; nthreads=1)
        p1(y2, x)
        p1(y2, x)
        @test (@allocated p1(y2, x)) == 0
    end
end

@testset "mixed radix vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0x357)
    ns = [3, 5, 6, 7, 9, 10, 12, 14, 15, 20, 21, 24, 25, 27, 35, 45, 48, 49, 60,
          63, 105, 120, 315, 27 * 32, 3 * 2^8, 9 * 2^8, 5 * 2^8, 7 * 2^8,
          15 * 2^7, 105 * 2^4, 5^3 * 2^6, 7^2 * 2^6, 3^4 * 5 * 7 * 2^5,
          # radix-11/13 stages (generic odd-prime butterfly), incl. scalar
          # strides (429), repeated primes, and all radices at once
          11, 13, 22, 26, 429, 352, 416, 1001, 11^2 * 2^4, 13^2 * 2^4,
          2^4 * 3^2 * 11 * 13, 3 * 5 * 7 * 11 * 13 * 2^4]
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

@testset "N-d transforms vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0xd1d)
    tol(ref) = 200 * eps(T) * max(norm(ref), one(T))
    shapes = [(64, 48), (256, 256), (33, 17), (16, 11), (11, 16), (8, 1), (1, 8),
              (128, 3, 5), (16, 16, 16), (12, 10, 14),
              (32, 64), (60, 50), (24, 40, 3)]  # dim-1-batched paths
    @testset "dims = $dims" for dims in shapes
        x = randn(rng, Complex{T}, dims)
        ref = FFTW.fft(ComplexF64.(x))
        p = plan_fft(x)
        y = p * x
        @test y ≈ ref atol = tol(ref)
        z = similar(x)
        @test mul!(z, p, x) === z
        @test z ≈ ref atol = tol(ref)

        # normalized inverse round trip and unnormalized backward
        @test plan_ifft(x) * y ≈ x atol = tol(x)
        @test plan_bfft(x) * y ≈ length(x) .* x atol = tol(length(x) .* x)

        # in-place + inv(p!).p idiom (DSP complex conv)
        p! = plan_fft!(x)
        w = copy(x)
        @test (p! * w) === w
        @test w ≈ ref atol = tol(ref)
        ip = inv(p!).p
        ip * w
        @test w ≈ length(x) .* x atol = tol(length(x) .* x)

        # real transforms
        xr = randn(rng, T, dims)
        rref = FFTW.rfft(Float64.(xr))
        pr = plan_rfft(xr)
        Xr = pr * xr
        @test size(Xr) == size(rref)
        @test Xr ≈ rref atol = tol(rref)
        n1 = dims[1]
        pbr = plan_brfft(Xr, n1)
        bref = FFTW.brfft(ComplexF64.(Xr), n1)
        @test pbr * Xr ≈ bref atol = tol(bref)
        @test plan_irfft(Xr, n1) * Xr ≈ xr atol = tol(xr)

        # allocation-free planned execution (threaded plans may allocate task
        # objects, so the guarantee is for nthreads=1)
        if prod(dims) > 32
            ps = plan_fft(x; nthreads=1)
            mul!(z, ps, x)
            mul!(z, ps, x)
            @test (@allocated mul!(z, ps, x)) == 0
        end
    end
    # one-shots
    A = randn(rng, Complex{T}, (24, 15))
    @test fft(A) ≈ FFTW.fft(ComplexF64.(A)) atol = tol(FFTW.fft(ComplexF64.(A)))
    @test ifft(fft(A)) ≈ A atol = tol(A)
    Ar = randn(rng, T, (20, 12))
    @test rfft(Ar) ≈ FFTW.rfft(Float64.(Ar)) atol = tol(FFTW.rfft(Float64.(Ar)))
    @test irfft(rfft(Ar), 20) ≈ Ar atol = tol(Ar)
    @test fft!(copy(A)) ≈ FFTW.fft(ComplexF64.(A)) atol = tol(FFTW.fft(ComplexF64.(A)))
end

@testset "smooth-size routing" begin
    # Float64 (and any W ≤ 8 host): direct chains beat Bluestein at every
    # smooth size (measured 2026-08), so smooth n never routes to Bluestein
    for n in (405, 2835, 3^8, 3^12, 2 * 3^6 * 5^2 * 7^2)
        @test plan_fft(Float64, n) isa FFTJ.FFTPlan
    end
    # non-smooth still routes to Bluestein
    @test plan_fft(Float64, 10007) isa FFTJ.BluesteinPlan
    @test plan_fft(Float32, 10007) isa FFTJ.BluesteinPlan
    if FFTJ.pick_width(Float32) ≥ 16
        # wide-vector Float32: scalar-stride odd stages keep Bluestein ahead
        # for small odd-heavy sizes, direct wins above the measured crossover
        @test plan_fft(Float32, 729) isa FFTJ.BluesteinPlan
        @test plan_fft(Float32, 3^12) isa FFTJ.FFTPlan
        @test plan_fft(Float32, 2 * 3^6 * 5^2 * 7^2) isa FFTJ.FFTPlan
    end
end

include("threading.jl")

@testset "argument checking" begin
    @test_throws ArgumentError plan_fft(Float64, 0)
    p = plan_fft(Float64, 8)
    @test_throws DimensionMismatch p(randn(ComplexF64, 4))
end
