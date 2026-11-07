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

@testset "resampled (non-smooth n) vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0xb1e)
    # includes one past the weight-table cap, and (F32 only) slow-smooth
    # lengths with a small 2-part
    ns = [17, 19, 23, 34, 38, 51, 97, 101, 289, 331, 361, 997, 1026, 1234,
          2050, 10006, 65542]
    T == Float32 && append!(ns, [3^7, 5^4 * 2, 3^4 * 5^2])
    @testset "n = $n" for n in ns
        x = randn(rng, Complex{T}, n)
        ref = FFTW.fft(ComplexF64.(x))
        # built directly: the larger even sizes route to SplitPlan
        p = FFTJ._resample(T, n, -1)
        FFTJ._use_split(T, n) || FFTJ._use_ragged(T, n) ||
            @test plan_fft(T, n) isa FFTJ.ResamplePlan
        y = p(x)
        @test norm(y .- ref) ≤ 100 * eps(T) * norm(ref)
        pinv = FFTJ._resample(T, n, +1)
        @test norm(pinv(y) .- x) ≤ 100 * eps(T) * norm(x)
        pb = FFTJ._resample(T, n, +1, one(T))
        @test norm(pb(y) .- n .* x) ≤ 100 * eps(T) * n * norm(x)
        y2 = similar(x)
        p(y2, x)
        p(y2, x)
        @test (@allocated p(y2, x)) == 0
    end
end

@testset "split (small smooth × odd non-smooth n) vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0x5b1)
    # s = 2..14 (radix-2/3/4/7 combine chains); the first few are below the
    # routing thresholds and are built directly
    ns = [2 * 1009, 6 * 1009, 14 * 1009, 2 * 10007, 4 * 10007, 12 * 2003,
          2 * 32771, 8 * 16411, 2 * 65537]
    @testset "n = $n" for n in ns
        x = randn(rng, Complex{T}, n)
        ref = FFTW.fft(ComplexF64.(x))
        p = FFTJ._split(T, n, -1; nthreads=1)
        FFTJ._use_split(T, n) && @test plan_fft(T, n) isa FFTJ.SplitPlan
        y = p(x)
        @test norm(y .- ref) ≤ 100 * eps(T) * norm(ref)
        q4 = FFTJ._split(T, n, -1; nthreads=4)   # threaded sub-transforms
        @test q4(x) == FFTJ._serial_copy(q4)(x)
        pinv = FFTJ._split(T, n, +1)
        @test norm(pinv(y) .- x) ≤ 100 * eps(T) * norm(x)
        pb = FFTJ._split(T, n, +1, one(T))
        @test norm(pb(y) .- n .* x) ≤ 100 * eps(T) * n * norm(x)
        y2 = similar(x)
        p(y2, x)
        @test (@allocated p(y2, x)) == 0
    end
    @test any(n -> FFTJ._use_split(T, n), ns)
end

@testset "HvdH resampling vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0x4d4)
    # tiny and even lengths (reachable only directly), halos wider than t,
    # and lengths routed here
    ns = [11, 13, 17, 19, 22, 23, 26, 34, 38, 46, 97, 101, 202, 331, 997, 1001,
          3107, 10007, 20011, 65537]
    @testset "n = $n" for n in ns
        x = randn(rng, Complex{T}, n)
        ref = FFTW.fft(ComplexF64.(x))
        p = FFTJ._hvdh(T, n, -1; nthreads=1)
        y = p(x)
        @test norm(y .- ref) ≤ 50 * eps(T) * norm(ref)
        @test FFTJ._hvdh(T, n, -1; factors=:fly)(x) ≈ y
        # threaded (from 65537 at 4 tasks) is bitwise its serial copy, both
        # factor modes (the inner transform's routing may differ from nthreads=1)
        for f in (:auto, :fly)
            p4 = FFTJ._hvdh(T, n, -1; factors=f, nthreads=4)
            @test p4(x) == FFTJ._serial_copy(p4)(x)
        end
        pinv = FFTJ._hvdh(T, n, +1)
        @test norm(pinv(y) .- x) ≤ 50 * eps(T) * norm(x)
        y2 = similar(x)
        p(y2, x)
        @test (@allocated p(y2, x)) == 0
        FFTJ.factor_radices(n) === nothing &&
            @test plan_fft(T, n) isa (FFTJ._use_hvdh(T, n) && !FFTJ._use_composite(T, n) ?
                                      FFTJ.HvdhPlan : Union{FFTJ.ResamplePlan,FFTJ.CompositePlan})
    end
    # even multiples of 30 have no coprime smooth t
    @test_throws ArgumentError FFTJ._hvdh(T, 390, -1)
    @test plan_bfft(T, 20011) isa FFTJ.HvdhPlan
    xb = randn(rng, Complex{T}, 20011)
    @test norm(plan_bfft(T, 20011)(plan_fft(T, 20011)(xb)) .- 20011 .* xb) ≤
          50 * eps(T) * 20011 * norm(xb)
end

@testset "composite (smooth × non-smooth n) vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0xc0b)
    ns = filter(n -> FFTJ._use_composite(T, n),
                [17 * 1024, 101 * 64, 1009 * 16, 19 * 3072, 17 * 2^13, 101 * 1024,
                 1009 * 64, 10007 * 32, 17 * 19 * 512, 3^4 * 5 * 7 * 101])
    @test length(ns) ≥ 4
    @testset "n = $n" for n in ns
        x = randn(rng, Complex{T}, n)
        ref = FFTW.fft(ComplexF64.(x))
        p = plan_fft(T, n; nthreads=1)
        @test p isa FFTJ.CompositePlan
        y = p(x)
        @test norm(y .- ref) ≤ 100 * eps(T) * norm(ref)
        # threaded (lane chunks over tasks from 2^16 elements) is bitwise serial
        @test plan_fft(T, n; nthreads=4)(x) == y
        pinv = plan_ifft(T, n)
        @test norm(pinv(y) .- x) ≤ 100 * eps(T) * norm(x)
        pb = plan_bfft(T, n)
        @test norm(pb(y) .- n .* x) ≤ 100 * eps(T) * n * norm(x)
        y2 = similar(x)
        p(y2, x)
        @test (@allocated p(y2, x)) == 0
    end
end

@testset "four-step (out-of-cache n) vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0x4573)
    # threaded routing reaches four-step at test-friendly sizes
    ns = filter(n -> FFTJ._use_fourstep(T, n, true),
                [2^17, 2^18, 2^20, 3 * 2^16, 5 * 2^16, 7 * 2^16, 9 * 2^15, 15 * 2^15])
    @test length(ns) ≥ 6
    @testset "n = $n" for n in ns
        x = randn(rng, Complex{T}, n)
        ref = FFTW.fft(ComplexF64.(x))
        p = plan_fft(T, n; nthreads=4)
        @test p isa FFTJ.FourStepPlan
        y = p(x)
        @test norm(y .- ref) ≤ 50 * eps(T) * norm(ref)
        pinv = plan_ifft(T, n; nthreads=4)
        @test norm(pinv(y) .- x) ≤ 50 * eps(T) * norm(x)
        # the serial plan of the same split is bitwise equal and alloc-free
        p1 = FFTJ._fourstep(T, n, -1; nthreads=1)
        y2 = similar(x)
        p1(y2, x)
        @test y2 == y
        @test (@allocated p1(y2, x)) == 0
    end
    # lopsided splits nest one pass's inner transform at a testable size
    W = FFTJ.pick_width(T)
    n = 2^16
    @testset "N2 = $N2" for N2 in (W, n ÷ W)
        x = randn(rng, Complex{T}, n)
        ref = FFTW.fft(ComplexF64.(x))
        for D in (-1, 1)
            p = FFTJ._fourstep(T, n, D; N2)
            @test (p.innerA !== nothing) || (p.innerB !== nothing)
            r = D == -1 ? ref : FFTW.ifft(ComplexF64.(x))
            @test norm(p(x) .- r) ≤ 50 * eps(T) * norm(r)
        end
    end
end

@testset "ragged four-step (odd-heavy n) vs FFTW ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0x7a6)
    # neither factor a multiple of W (ragged strips and groups), 11/13
    # factors, a 2-part below 4W, and a mixed size with a ragged split
    ns = [3^8, 5^6, 7^5, 3^4 * 11 * 13, 13^4, 2 * 3^9, 2835, 3^10 * 2, 1000]
    @testset "n = $n" for n in ns
        sp = FFTJ._fourstep_split(T, n, 1)
        sp === nothing && continue
        x = randn(rng, Complex{T}, n)
        ref = FFTW.fft(ComplexF64.(x))
        p = FFTJ._fourstep(T, n, -1; nthreads=1)
        FFTJ._use_ragged(T, n) && @test plan_fft(T, n) isa FFTJ.FourStepPlan
        y = p(x)
        @test norm(y .- ref) ≤ 50 * eps(T) * norm(ref)
        @test FFTJ._fourstep(T, n, -1; nthreads=4)(x) == y   # bitwise serial
        pinv = FFTJ._fourstep(T, n, +1)
        @test norm(pinv(y) .- x) ≤ 50 * eps(T) * norm(x)
        pb = FFTJ._fourstep(T, n, +1, one(T))
        @test norm(pb(y) .- n .* x) ≤ 50 * eps(T) * n * norm(x)
        y2 = similar(x)
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
    # direct chains for small sizes and a sizable 2-part (11/13 radices too)
    for n in (405, 11 * 1024, 4000)
        @test plan_fft(Float64, n; nthreads=1) isa FFTJ.FFTPlan
    end
    @test plan_fft(Float32, 8000; nthreads=1) isa FFTJ.FFTPlan
    # odd-heavy sizes from ~2k (F64) / ~1k (F32): ragged four-step, serial too
    for n in (2835, 3^8, 3^12, 2 * 3^6 * 5^2 * 7^2, 13^3)
        @test plan_fft(Float64, n; nthreads=1) isa FFTJ.FourStepPlan
    end
    # F64 with a 2-part of exactly 4W from 16k; threaded, any 2-part from 2^15
    @test plan_fft(Float64, 1441440; nthreads=1) isa FFTJ.FourStepPlan
    @test plan_fft(Float64, 12000; nthreads=1) isa FFTJ.FFTPlan
    @test plan_fft(Float32, 40000; nthreads=1) isa FFTJ.FFTPlan
    @test plan_fft(Float32, 40000; nthreads=4) isa FFTJ.FourStepPlan
    # tiled four-step from 1 MB threaded, 8 MB serial
    @test plan_fft(Float64, 2^18; nthreads=1) isa FFTJ.FFTPlan
    @test plan_fft(Float64, 2^19; nthreads=1) isa FFTJ.FourStepPlan
    @test plan_fft(Float64, 2^16; nthreads=4) isa FFTJ.FourStepPlan
    @test plan_fft(Float64, 2^15; nthreads=4) isa FFTJ.FFTPlan
    # non-smooth routes to resampling (large odd n: HvdH)
    @test plan_fft(Float64, 10006) isa FFTJ.ResamplePlan
    @test plan_fft(Float64, 10007) isa FFTJ.HvdhPlan
    @test plan_fft(Float32, 10007) isa FFTJ.ResamplePlan
    # Float32: small odd-heavy sizes resample, larger ones ragged four-step
    @test plan_fft(Float32, 729) isa FFTJ.ResamplePlan
    @test plan_fft(Float32, 2187) isa FFTJ.FourStepPlan
    @test plan_fft(Float32, 3^12) isa FFTJ.FourStepPlan
end

include("threading.jl")

@testset "argument checking" begin
    @test_throws ArgumentError plan_fft(Float64, 0)
    p = plan_fft(Float64, 8)
    @test_throws DimensionMismatch p(randn(ComplexF64, 4))
end
