# Threading primitive. All parallel execution goes through tforeach so that
# chunk boundaries are a deterministic function of (len, nt, align) — results
# must be bitwise identical to serial execution for every task count.

# Minimum per-task elements of stage work before threading pays; plans compute
# their task count from this at plan time.
const MIN_CHUNK = 1 << 15

# Split 1:len into at most nt contiguous chunks whose starts sit on `align`
# boundaries, run f(ci, lo, hi) (ci = chunk index, for per-task scratch) on
# nt-1 spawned tasks plus the calling task, wait for all. The last chunk
# absorbs any sub-align tail, so every chunk spans ≥ align elements — a
# vectorized chunk never needs QWin's overlapping tail window to reach across
# a task boundary. nt == 1 (or a single chunk) calls f(1, 1, len) with no spawn.
function tforeach_chunks(f, len::Int, nt::Int; align::Int=1)
    len > 0 || return nothing
    units = len ÷ align              # whole align-units of work
    nchunks = min(nt, units)
    if nchunks <= 1
        f(1, 1, len)
        return nothing
    end
    per, rem = divrem(units, nchunks)
    tasks = Vector{Task}(undef, nchunks - 1)
    lo = 1
    for c in 1:nchunks
        u = per + (c <= rem)
        hi = c == nchunks ? len : lo + u * align - 1
        if c < nchunks
            tasks[c] = Threads.@spawn f($c, $lo, $hi)
        else
            f(c, lo, hi)
        end
        lo = hi + 1
    end
    foreach(wait, tasks)
    return nothing
end

tforeach(f, len::Int, nt::Int; align::Int=1) =
    tforeach_chunks((ci, lo, hi) -> f(lo, hi), len, nt; align)
