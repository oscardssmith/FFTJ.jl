# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

FFTJ.jl is a pure-Julia FFT library (complex, real, and N-d transforms) with an
FFTW-compatible API. It deliberately does **not** extend the AbstractFFTs plan
interface — all `plan_*`/`fft`/`mul!` methods are FFTJ's own functions, so
loading FFTJ never changes FFTW's dispatch. AbstractFFTs is used only for the
re-exported utilities (`fftshift`, `fftfreq`, ...).

## Commands

```bash
# run the full test suite (FFTW is a test-only dep, used as the reference)
julia --project -e 'using Pkg; Pkg.test()'

# faster iteration: reuse a session / run tests directly against dev deps
julia --project=test -e 'using Pkg; Pkg.develop(path="."); include("test/runtests.jl")'
```

Tests live in `test/runtests.jl` plus `test/threading.jl` (included from it;
also runnable standalone). Run the suite at both `-t 1` and `-t 4` — threading
tests compare threaded results bitwise against serial plans. To run a single
testset, either comment out others temporarily or paste the testset into a
REPL with `using Test, FFTJ; import FFTW`. Every transform is tested against
FFTW with tolerances of 50–200 ulps relative to the reference norm, and
serial (`nthreads=1`) planned execution paths assert `@allocated == 0` — keep
both properties when changing kernels.

## Architecture

Execution operates on **split re/im planes** (separate `Vector{T}` for real
and imaginary parts), not interleaved complex, with de/interleave passes at
the edges (fused into the first/last stage when possible). The type parameter
`D` is the sign of the transform exponent (-1 forward, +1 inverse) everywhere.

Pipeline for a 1-D transform of length n:

- **n factoring over the radix set** (2^a times odd primes in `ODD_RADICES`,
  currently {3,5,7,11,13}): a Stockham autosort chain of `Stage`s built by
  `build_stages` (plan.jl). Order matters: the radix-4 tower runs first so
  odd-radix stages see large SIMD-divisible strides (odd stages run largest
  radix first); the final stage is a twiddle-free power-of-two radix (2/4/8)
  that can be fused with output interleaving. Stages ping-pong between two
  plane pairs via `run_chain!`. All odd radices go through the `@generated`
  conjugate-symmetric-pairing butterfly `bfp` (kernels.jl), measured equal to
  the formerly handwritten 3/5/7 kernels — but only after two traps were
  fixed: `@inbounds` does not propagate into `ntuple` closures (bounds checks
  cost ~20% at radix 5), and tuple `map`/runtime-length `ntuple` allocate
  past inference limits.
- **n with larger prime factors** (or wide-vector-Float32 small odd sizes,
  see `_slow_smooth`): Bluestein's algorithm (bluestein.jl) over a padded
  length from `nextsmooth` (small odd part × large power of two), reusing two
  ordinary `FFTPlan`s.
- **real transforms** (real.jl): even n packs input as n/2 complex points via
  reinterpret (no copy) + O(n) even/odd split post-pass; odd n falls back to a
  full complex transform.
- **N-d** (ndplan.jl): per-dimension batched-strided passes running the same
  stage kernels with effective stride s·Bi; dim-1 pencils that don't earn the
  fused-edge treatment run transposed with pencils as vector lanes. Non-smooth
  dims gather pencils into buffers and run Bluestein per pencil. N-d plans
  transform all dims only (`region` must be `1:N`).

File layout: `kernels.jl` (SIMD butterflies `bf2..bf8`, width selection,
shuffles — generic over scalar and `SIMD.Vec`, so one kernel body serves both
paths) → `stockham.jl` (stage loops: plain, small-stride, fused first/last,
batched) → `plan.jl` (1-D planning, stage chains, `FFTPlan` execution) →
`bluestein.jl` → `real.jl` → `ndplan.jl` → `api.jl` (FFTW-style user API:
array-based planning, `*`/`mul!`, `inv`, in-place `InplacePlan`, `Scaled`,
one-shots, promoting fallbacks).

Key invariants and conventions:

- **Plan caching** (plan.jl): immutable parts (stage chains/twiddles, chirps,
  real-edge twiddles) are cached in session-lifetime `Dict`s behind
  `_PLAN_CACHE_LOCK`; only per-plan scratch buffers are fresh. Repeated
  planning at one size must stay cheap — DSP-style code plans per call.
- **Twiddle accuracy**: angles are computed in units of π (`sincospi`) with
  exact integer reduction (e.g. `j² mod 2n` for the Bluestein chirp) — don't
  replace with `exp(2πi...)` floating-point angle math.
- **SIMD width** is picked at load time from host vector register bits
  (`VECTOR_BITS` / `pick_width` in kernels.jl); the "small-stride" twiddle
  layout and fusion decisions depend on it, so width-dependent logic must
  handle W ∈ {2, 4, 8, 16} and W == 1 (non-Float32/64 scalars).
- Plan execution is **alias-safe** (input fully consumed into scratch before
  output is written) — in-place plans rely on this by wrapping ordinary plans.
- `inv(p).p` must remain the unnormalized backward plan and `Scaled`/`.scale`
  public — DSP.jl unwraps them (`inv(plan_fft!(x)).p` idiom, tested in
  runtests.jl).
- Sharing scratch (`bufs` kwarg of `_plan`) is only safe for plans that never
  execute concurrently, e.g. Bluestein's sequential forward/inverse pair.

Threading (threads.jl; design: docs/superpowers/specs/2026-08-30-fftj-threading-design.md):

- All `plan_*` functions take `nthreads` (default `Threads.nthreads()`); the
  plan stores an effective task count `nt` computed at plan time via
  `plan_tasks` (`MIN_CHUNK` elements of per-stage work per task minimum).
  Everything runs through `tforeach`/`tforeach_chunks` in threads.jl.
- **Bitwise determinism is a hard invariant**: chunking never changes which
  inputs a butterfly combines, so threaded results are `==` serial for every
  task count (tested). Chunk boundaries are W-aligned with the last chunk
  absorbing the tail, so `QWin`'s overlapping tail window never crosses a
  task boundary.
- `nt == 1` execution must take today's exact serial code paths and stay
  zero-allocation. **Closure-boxing trap**: a closure capturing a variable
  that is reassigned *anywhere* in the enclosing function makes Julia box it
  at function entry, allocating on every call including serial ones — always
  `let`-bind plane variables (`cre`/`are`/...) before capturing them, and
  never reuse a captured local's name across branches.
- Parallel split per stage: butterflies (`p`) when `m ≥ nt`, else stride
  lanes (`q`) in W-aligned chunks; N-d passes prefer whole independent blocks
  (`Bo ≥ nt`). Bluestein/rfft pencil loops parallelize over per-task serial
  plan clones stored in the plan (cheap: immutable parts are cached/shared).
- Threaded (default) plans may allocate O(nstages·nt) task objects per call —
  the zero-alloc guarantee applies to `nthreads=1` plans only. The N-d
  non-batched real path has a known pre-existing 32 B/pencil wrapper
  allocation independent of threading.
- Scaling is bandwidth-bound: good while the working set is cache-resident
  (~2–3× on 4 cores up to ~2^20). Above FOURSTEP_MIN (2^21), *threaded*
  smooth plans route to the four-step/six-step `FourStepPlan` (fourstep.jl):
  3 transposes + 2 cache-resident pencil-FFT phases with a fused twiddle,
  measured 0.74–0.85× direct at 4t. Serial plans stay direct — one core
  can't saturate DRAM, so four-step's extra sweeps lose serially (1.1–1.7×).
  Routing therefore depends on the effective task count; the bitwise
  threaded==serial tests compare against a same-algorithm serial plan
  (`serial_like` in test/threading.jl), not `plan_*(...; nthreads=1)`.
