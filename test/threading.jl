# Threading tests. Runnable standalone (julia -t4 --project=<env with FFTJ+FFTW>)
# or via runtests.jl. Threaded-vs-serial comparisons are bitwise (==): chunk
# boundaries must never change any butterfly's arithmetic.

if !@isdefined(FFTJ)
    using Test, FFTJ
    import FFTW
    using LinearAlgebra: norm, mul!
    using Random: Xoshiro
end

# The same algorithm tree as p with every task count forced to 1 — the
# bitwise reference for threaded plans. (plan_* with nthreads=1 can pick a
# different algorithm: routing legitimately depends on the task count, e.g.
# four-step only pays when threaded.)
serial_like(p::FFTJ.FFTPlan{T,D}) where {T,D} =
    FFTJ._plan(T, p.n, D, p.scale; nthreads=1)
serial_like(p::FFTJ.SixStepPlan{T,D}) where {T,D} =
    FFTJ._sixstep(T, p.n, D, p.scale; nthreads=1)
serial_like(p::Union{FFTJ.FourStepPlan,FFTJ.ResamplePlan,FFTJ.CompositePlan}) = p

# plan types with no threaded execution yet
serial_only(p) = p isa Union{FFTJ.ResamplePlan,FFTJ.CompositePlan} ||
                 (p isa Union{FFTJ.RfftPlan,FFTJ.IrfftPlan} && serial_only(p.inner))
function serial_like(p::FFTJ.RfftPlan{T}) where {T}
    inner = serial_like(p.inner)
    return FFTJ.RfftPlan{T,typeof(inner)}(p.n, inner, p.twr, p.twi,
                                          similar(p.buf), similar(p.buf2))
end
function serial_like(p::FFTJ.IrfftPlan{T}) where {T}
    inner = serial_like(p.inner)
    return FFTJ.IrfftPlan{T,typeof(inner)}(p.n, inner, p.pre, p.twr, p.twi,
                                           similar(p.buf), similar(p.buf2))
end

@testset "tforeach chunking" begin
    lk = ReentrantLock()
    @testset "exact cover: len=$len nt=$nt align=$align" for len in (0, 1, 5, 16, 100, 1000),
                                                             nt in (1, 2, 3, 8),
                                                             align in (1, 4, 16)
        cov = zeros(Int, len)
        FFTJ.tforeach(len, nt; align) do lo, hi
            @lock lk cov[lo:hi] .+= 1
        end
        @test all(==(1), cov)
    end

    # chunk starts sit on align boundaries
    starts = Int[]
    FFTJ.tforeach(100, 3; align=16) do lo, hi
        @lock lk push!(starts, lo)
    end
    @test all(s -> (s - 1) % 16 == 0, starts)

    # chunk boundaries are deterministic
    r1 = Tuple{Int,Int}[]
    r2 = Tuple{Int,Int}[]
    FFTJ.tforeach((lo, hi) -> (@lock lk push!(r1, (lo, hi))), 1000, 4; align=8)
    FFTJ.tforeach((lo, hi) -> (@lock lk push!(r2, (lo, hi))), 1000, 4; align=8)
    @test sort(r1) == sort(r2)

    # nt == 1 runs on the calling task with the full range
    calls = Tuple{Int,Int}[]
    t = Ref{Any}(nothing)
    FFTJ.tforeach((lo, hi) -> (push!(calls, (lo, hi)); t[] = current_task()), 10, 1)
    @test calls == [(1, 10)]
    @test t[] === current_task()

    # errors from worker chunks propagate
    @test_throws Exception FFTJ.tforeach((lo, hi) -> error("boom"), 100, 4)
end

@testset "threaded 1-D complex vs serial ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0x7472)
    # 2^20: fused radix-4 chain; 2^19: radix-8 last stage; 3·2^18: radix-3
    # stage; 3^4·5·7·2^10: all odd radices; 1441440: radix-11/13 stages;
    # 2^22 and 3·2^21: six-step; 1000003: prime → resampling (serial)
    ns = [2^20, 2^19, 3 * 2^18, 3^4 * 5 * 7 * 2^10, 1441440, 2^22, 3 * 2^21,
          1000003]
    @testset "n = $n" for n in ns
        x = randn(rng, Complex{T}, n)
        p1 = plan_fft(T, n; nthreads=1)
        pt = plan_fft(T, n)
        if Threads.nthreads() > 1
            serial_only(pt) ? (@test_broken FFTJ.plan_nt(pt) > 1) :
                              (@test FFTJ.plan_nt(pt) > 1)
        end
        p1s = serial_like(pt)
        yt = pt(x)
        @test yt == p1s(x)            # bitwise-identical to same-algorithm serial
        @test pt(x) == yt             # deterministic across executions
        ref = FFTW.fft(ComplexF64.(x))
        @test norm(yt .- ref) ≤ 100 * eps(T) * norm(ref)
        @test norm(p1(x) .- ref) ≤ 100 * eps(T) * norm(ref)

        pt_inv = plan_ifft(T, n)
        @test pt_inv(yt) == serial_like(pt_inv)(yt)
        @test norm(pt_inv(yt) .- x) ≤ 100 * eps(T) * norm(x)
        ptb = plan_bfft(T, n)
        @test ptb(yt) == serial_like(ptb)(yt)

        # serial execution stays allocation-free at big sizes, for both the
        # public nthreads=1 plan and the threaded plan's algorithm
        y2 = similar(x)
        p1(y2, x)
        p1(y2, x)
        @test (@allocated p1(y2, x)) == 0
        p1s(y2, x)
        p1s(y2, x)
        @test (@allocated p1s(y2, x)) == 0
    end
end

@testset "threaded N-d complex vs serial ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0xd7d)
    # (1024,512): block-parallel passes; (2^16,4): last dim has Bo=1 → intra-
    # stage; (4,2^16): dim-1 batched-transposed chain; (513,255): both dims
    # non-smooth → batched resampling passes; (256,33,17): 3-d mixed
    shapes = [(1024, 512), (2^16, 4), (4, 2^16), (513, 255), (256, 33, 17)]
    @testset "dims = $dims" for dims in shapes
        x = randn(rng, Complex{T}, dims)
        p1 = plan_fft(x; nthreads=1)
        pt = plan_fft(x)
        Threads.nthreads() > 1 && @test FFTJ.plan_nt(pt) > 1
        y1 = p1 * x
        yt = pt * x
        @test yt == y1                    # bitwise-identical to serial
        @test pt * x == yt                # deterministic across executions
        ref = FFTW.fft(ComplexF64.(x))
        @test norm(vec(yt) .- vec(ref)) ≤ 200 * eps(T) * norm(vec(ref))

        @test plan_ifft(x) * yt == plan_ifft(x; nthreads=1) * yt
        @test norm(vec(plan_ifft(x) * yt) .- vec(x)) ≤ 200 * eps(T) * norm(vec(x))
        @test plan_bfft(x) * yt == plan_bfft(x; nthreads=1) * yt

        # in-place plans thread identically
        w = copy(x)
        w1 = copy(x)
        plan_fft!(x) * w
        plan_fft!(x; nthreads=1) * w1
        @test w == w1

        # nthreads=1 N-d plan execution stays allocation-free
        z = similar(x)
        mul!(z, p1, x)
        mul!(z, p1, x)
        @test (@allocated mul!(z, p1, x)) == 0
    end
end

@testset "threaded real transforms vs serial ($T)" for T in (Float32, Float64)
    rng = Xoshiro(0x5ea1)
    # 2^20: even, half-complex fast path; 3·2^19: even mixed-radix half;
    # 1000003: odd → full-length complex resampling fallback (serial)
    @testset "1-D n = $n" for n in (2^20, 3 * 2^19, 1000003)
        x = randn(rng, T, n)
        pr1 = plan_rfft(T, n; nthreads=1)
        prt = plan_rfft(T, n)
        if Threads.nthreads() > 1
            serial_only(prt) ? (@test_broken FFTJ.plan_nt(prt) > 1) :
                               (@test FFTJ.plan_nt(prt) > 1)
        end
        Xt = prt(x)
        @test Xt == serial_like(prt)(x)
        ref = FFTW.rfft(Float64.(x))
        @test norm(Xt .- ref) ≤ 100 * eps(T) * norm(ref)
        @test norm(pr1(x) .- ref) ≤ 100 * eps(T) * norm(ref)

        pit = plan_irfft(T, n)
        @test pit(Xt) == serial_like(pit)(Xt)
        @test norm(pit(Xt) .- x) ≤ 100 * eps(T) * norm(x)
        pbt = plan_brfft(T, n)
        @test pbt(Xt) == serial_like(pbt)(Xt)

        X2 = similar(Xt)
        pr1(X2, x)
        pr1(X2, x)
        @test (@allocated pr1(X2, x)) == 0
    end
    # (1024,512): batched real edge candidate; (511,513): odd n1, non-smooth
    # dims → fallback pencils; (256,33,17): 3-d, per-pencil real edge
    @testset "N-d dims = $dims" for dims in [(1024, 512), (511, 513), (256, 33, 17)]
        xr = randn(rng, T, dims)
        pr1 = plan_rfft(xr; nthreads=1)
        prt = plan_rfft(xr)
        X1 = pr1 * xr
        Xt = prt * xr
        @test Xt == X1
        ref = FFTW.rfft(Float64.(xr))
        @test norm(vec(Xt) .- vec(ref)) ≤ 200 * eps(T) * norm(vec(ref))

        n1 = dims[1]
        @test plan_irfft(Xt, n1) * Xt == plan_irfft(Xt, n1; nthreads=1) * Xt
        @test norm(vec(plan_irfft(Xt, n1) * Xt) .- vec(xr)) ≤
              200 * eps(T) * norm(vec(xr))
        @test plan_brfft(Xt, n1) * Xt == plan_brfft(Xt, n1; nthreads=1) * Xt

        # the non-batched per-pencil path has a pre-existing 32-byte/pencil
        # wrapper allocation (present at serial HEAD); threading must add none
        Z = similar(Xt)
        B = prod(dims[2:end])
        mul!(Z, pr1, xr)
        mul!(Z, pr1, xr)
        @test (@allocated mul!(Z, pr1, xr)) ≤ 32 * B
    end
end

@testset "derived plans inherit threading" begin
    T = Float64
    n = 2^20
    # inv of a serial plan stays serial; inv of a threaded plan stays threaded
    @test FFTJ.plan_nt(inv(plan_fft(T, n; nthreads=1)).p) == 1
    if Threads.nthreads() > 1
        @test FFTJ.plan_nt(inv(plan_fft(T, n)).p) > 1
    end
    # in-place planners accept the kwarg (type- and array-based)
    x = randn(Xoshiro(0xabc), Complex{T}, n)
    p! = plan_fft!(T, n; nthreads=1)
    @test FFTJ.plan_nt(p!.p) == 1
    w = copy(x)
    w1 = copy(x)
    plan_fft!(T, n) * w
    p! * w1
    @test w == w1
    A = randn(Xoshiro(0xabd), Complex{T}, (512, 256))
    @test FFTJ.plan_nt(plan_ifft!(A; nthreads=1).p) == 1
end
