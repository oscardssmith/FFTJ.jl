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
- **four-step** `FourStepPlan` (fourstep.jl): two passes, each reading and
  writing the data once, threaded across strips/groups with per-task scratch
  (two fork-joins per call, so its own task count `_FS_TASK_MIN`). Taken by
  odd-heavy smooth n (2-part < 4W, from ~2k F64 / ~1k F32 elements; F64 with
  a 2-part of 4W from 16k; any non-tiled n from 2^15 when threaded:
  `_use_ragged`) on a balanced split whose last strip/group may be
  ragged (masked loads/stores, z padding), and by power-of-two-tiled splits
  from `_fs_min_bytes` (1 MB threaded, 8 MB serial).
- **n with larger prime factors** (or small odd-heavy Float32 sizes, see
  `_slow_smooth`): a large smooth factor × a non-smooth one runs as
  `CompositePlan` (composite.jl); odd n ≥ ~1k (F64) / ~16k (F32) as
  Harvey–van der Hoeven Gaussian resampling (`HvdhPlan`, hvdh.jl); even n
  with a small smooth part (s < 16) as s single-lane HvdH transforms of the
  odd cofactor plus a radix-s combine (`SplitPlan`, composite.jl); the rest
  as oversampled interpolation onto a smooth length (`ResamplePlan`,
  resample.jl). HvdH threads its permute, spread (in units aligned to its
  recurrence re-seeds), inner transform and group solves (per-task
  `HvdhScratch`); `SplitPlan` gives tasks whole sub-transforms when s ≥ nt,
  else threads inside them. The batched `RsPass`/`HvPass` (N-d non-smooth
  dims, composite) spread their lane chunks over per-task copies
  (`_rs_copies`: shared tables, own scratch); chunking never depends on nt.
  Only `ResamplePlan` is serial; it gets small sizes only (F64 below ~16k,
  F32 below ~250k, where split's p < 16384 cutoff leaves even n).
- **real transforms** (real.jl): even n packs input as n/2 complex points via
  reinterpret (no copy) + O(n) even/odd split post-pass; odd n falls back to a
  full complex transform.
- **N-d** (ndplan.jl): per-dimension batched-strided passes running the same
  stage kernels with effective stride s·Bi; dim-1 pencils that don't earn the
  fused-edge treatment run transposed with pencils as vector lanes. Non-smooth
  dims run a batched resampling pass (`RsPass`) with pencils as lanes. N-d plans
  transform all dims only (`region` must be `1:N`).

File layout: `kernels.jl` (SIMD butterflies `bf2..bf8`, width selection,
shuffles — generic over scalar and `SIMD.Vec`, so one kernel body serves both
paths) → `stockham.jl` (stage loops: plain, small-stride, fused first/last,
batched) → `plan.jl` (1-D planning, stage chains, `FFTPlan` execution) →
`resample.jl` → `real.jl` → `ndplan.jl` → `fourstep.jl` → `hvdh.jl` →
`composite.jl` → `api.jl` (FFTW-style user API:
array-based planning, `*`/`mul!`, `inv`, in-place `InplacePlan`, `Scaled`,
one-shots, promoting fallbacks).

Key invariants and conventions:

- **Plan caching** (plan.jl): immutable parts (stage chains/twiddles, chirps,
  real-edge twiddles) are cached in session-lifetime `Dict`s behind
  `_PLAN_CACHE_LOCK`; only per-plan scratch buffers are fresh. Repeated
  planning at one size must stay cheap — DSP-style code plans per call.
- **Twiddle accuracy**: angles are computed in units of π (`sincospi`) with
  exact integer reduction (e.g. `j·k mod 2n` for four-step twiddles) — don't
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
  (`Bo ≥ nt`). Four-step splits pass A's strips and pass B's groups across
  tasks, each with its own scratch (`FsScratch`).
- Threaded (default) plans may allocate O(nstages·nt) task objects per call —
  the zero-alloc guarantee applies to `nthreads=1` plans only. The N-d
  non-batched real path has a known pre-existing 32 B/pencil wrapper
  allocation independent of threading.
- Scaling is bandwidth-bound: direct chains scale ~2–3× on 4 cores while
  cache-resident, but they synchronize once per stage, so below ~2^16 they
  barely gain. Four-step synchronizes twice per call and takes over from 1 MB
  threaded (odd-heavy sizes from ~2k elements); a three-transpose six-step
  variant lost to it everywhere and was removed.
  The bitwise threaded==serial tests compare against a same-algorithm serial
  plan (`serial_like` in test/threading.jl).
- Serial benchmarks of memory-bound sizes swing badly under background load
  (same-process ratios moved 0.6→1.0 between runs at load ~3); measure
  routing thresholds on a quiet machine.
