# ntsync-android

Full userspace replacement of the Linux `ntsync` kernel driver
(`drivers/misc/ntsync.c`) for Android — NT semaphores, recursive/abandoned
mutexes, and auto/manual-reset events with `WAIT_ANY` / `WAIT_ALL` waits and
timeouts. Cross-process: objects live in shared memory, waits use futexes.
Intended to back the ntsync fast path of Proton/Wine's ntdll
(`dlls/ntdll/unix/sync.c`) on Android, where no `/dev/ntsync` device exists.

## Architecture (kernel concepts → NDK primitives)

| Kernel ntsync                        | This library                                    |
|--------------------------------------|-------------------------------------------------|
| global object table (`ntsync_device`)| slot table in a file-backed shared mapping      |
| `dev->wait_all_lock`                 | process-shared **robust** pthread mutex in shm (waiter registration/walks only; object state is lock-free CAS) |
| per-object wait queues + `wake_up_process` | per-object intrusive waiter lists; each waiter sleeps on its own node futex word |
| wake-then-recheck (`try_wake_any/all`) | waiters wake, re-evaluate conditions; wake walk is skipped entirely when the list is empty |
| fd lifetime / close-on-death         | creator-pid tracking + `ntsync_sweep_dead()`    |
| `get_task` owner-death check         | explicit `ntsync_mutex_kill` (same as kernel)   |

The region is initialized on first use; the path is the `ntsync_init(path)`
argument if given, else `$NTSYNC_SHM`, else `$TMPDIR/ntsync_userspace.shm`
(the caller is expected to export `TMPDIR`; Termux does this for every
process, so all Wine processes automatically share the region). First opener
creates and initializes the file under `flock`; later processes just `mmap`
it. Capacity is 16384 objects plus an 8192-entry shared waiter-node pool.

`NTSYNC_SHM` exists for containerized setups (e.g. GameNative) where
wineserver and game processes run with different `TMPDIR`s and would
otherwise not see the same region.

Each slot's state (sem count, event flag, mutex owner/recursion/abandoned)
is packed into a single `u64` mutated with lock-free CAS, so signal
operations and already-signaled waits run without taking the region mutex.
The robust mutex only protects waiter registration and wake walks; if a
process dies mid-operation the next locker gets `EOWNERDEAD` and marks the
mutex consistent.

## Environment variables

- `NTSYNC_SHM` — override the shared-region path.
- `NTSYNC_NO_SIGUSR1_BLOCK=1` — skip the SIGUSR1 deferral around
  region-lock acquisitions. By default the library blocks SIGUSR1 while
  holding the shared region mutex (two `pthread_sigmask` syscalls per
  acquisition): Wine suspends threads with SIGUSR1, and a thread parked
  while holding the mutex would wedge every process on the region. Only
  set this when the host process never installs a SIGUSR1 handler (i.e.
  not under Wine); it measurably reduces lock-path CPU under contention
  (~13% per-op in the multi-process benchmark).
- `NTSYNC_DEBUG=1` — log to stderr, stdout and `$TMPDIR/ntsync_debug.log`
  (the file is the reliable channel for in-game diagnostics); dump
  per-process stats every 10 s (waits, fast-path hits, lock acquisitions,
  wake walks, signal→scheduled wake latency `wake_lat_cnt/us_sum/us_max`);
  emit a "stuck wait" dump with per-object state (including creator pid
  liveness) when a wait stalls.
- Object lifetime: an object is reaped only when its creator **and every
  holder** are dead. `ntsync_retain(handle)` registers the calling process as
  a holder, `ntsync_release` drops it, and `close` does it implicitly. Call
  `retain` when a handle arrives from another process — wineserver relays
  objects, and a relayed handle would otherwise be reaped at the next sweep
  once its creator exits. The kernel driver has no equivalent because a
  duplicated fd keeps the object alive on its own. A slot tracks up to 8
  holder pids; an object relayed wider than that is never reaped
  automatically (a leak, rather than freeing something still in use). Layout
  version 10: an older region file is rejected with `-ESTALE`.
- `NTSYNC_SWEEP_INTERVAL_SEC` — auto-sweep cadence for dead-process
  cleanup, default 30. The kernel driver reaps a dead process's objects
  via fd release; userspace has no such hook, so the library runs the
  `ntsync_sweep_dead()` cleanup on its own, rate-limited region-wide.
  It is triggered only from wait paths (before a waiter sleeps, and on
  wait timeouts) — threads that have slack, so a scan never lands on a
  latency-critical signaler — and slot allocation on a full table always
  sweeps first. `0` disables auto-sweeping (explicit
  `ntsync_sweep_dead()` still works).
- `NTSYNC_SPIN_ITERS` — bounded spin-before-block, default `0` (off).
  When > 0, a waiter whose wait did not succeed on the fast path spins
  up to N iterations on a read-only acquire check before sleeping on the
  futex; a signal landing inside the window is acquired lock-free,
  avoiding a sleep+wake pair (two context switches). The spin is
  adaptive per thread (JVM-style credit: wins grant budget, exhausted
  spins cost double, out-of-credit threads only probe every 8th wait),
  so threads whose waits are genuinely long (e.g. 1 ms vsync polls)
  stop paying the spin tax automatically while short-handoff threads
  (mutex ping-pong) keep winning. Measured on the host: ~3x lower CPU
  and ~98% fewer context switches on a pure handoff benchmark
  (bench_3), ~+5% CPU on a mixed load dominated by long waits
  (bench_4). Device-tune: 100-300 is the sensible range; each iteration
  is a shared-cache-line load (tens of ns under contention).
  `spin_wins`/`spin_exhausted` in the NTSYNC_DEBUG stats show the
  hit rate.
- `NTSYNC_SPIN_TAIL_US` — deadline spin tail, default `0` (off), capped
  at 2000. When a wait has a timeout, the last N microseconds before the
  deadline are spun instead of parked. A futex timeout cannot fire closer
  than the process's timer slack (50 us under bionic), which is what a
  frame-pacing wait is measured against, so this is the difference between
  landing on the frame boundary and landing 60-100 us late. Measured on an
  idle 8 Elite (Android 15, 1 ms frame budget): overshoot p50 106 us ->
  1.1 us, p99 132 us -> 1.4 us. Opt in only for a thread that is pacing
  frames; a 150-200 us tail is the useful range and it burns that much CPU
  per frame while enabled.
  On a loaded mid-tier device (Dimensity 8400, load average ~20, 1 ms
  budget) the same knob gave p50 118 us -> 5 us but p99 670 us -> 490 us
  only: a spin lands on time only if it gets the CPU, so this is a
  quiet-device optimisation of the median, not a fix for loaded-device
  jitter.
  Its real cost, measured with `sim_wait_only_pacing` (one wait per frame,
  nothing else in the loop — the older `sim_vsync_pacing` number was 3x too
  high because that loop spins 150 us itself): 24 -> 80 us of CPU per frame
  on a Snapdragon 8 Gen 2, 7.5 -> 99 us on x86, i.e. about 5 % of a core at
  a 1 ms frame and a 150 us tail. Cheaper alternative for an app that
  controls the process: `prctl(PR_SET_TIMERSLACK, 1)` — 106 us -> 20-56 us
  p50 on the idle device, 118 us -> 64 us on the loaded one, no spin cost.
- `NTSYNC_SPIN_TAIL_BUDGET_PCT` — how much of one core the spin tail may
  spend, default 15, `0` = unlimited. A per-thread token bucket refilled at
  that rate: a thread that spends faster than it refills is throttled within
  a few frames and its waits park normally until it has allowance again. At
  1 kHz a 150 us tail sits right at the 15 % default, so the knob is inert
  there by design; it exists to stop a mistuning, which is one env edit away
  — a tail longer than the frame means the wait never parks at all
  (a 1500 us tail on a 1 ms frame pins a core: 998 us of CPU per frame,
  measured), and the same mistuning at the default budget costs 176 us and
  keeps the parked overshoot.

### When to enable spinning (per-game guide)

Spinning pays off only when a game's waits are **short handoffs** —
a waiter blocks and the signal lands within a few microseconds (mutex
protecting a tiny critical section, fine-grained job/task dispatch).
It is wasted on **rendezvous** waits, where the waiter blocks because
the work genuinely is not ready yet (producer-consumer pacing, vsync
polling) and the signal arrives hundreds of microseconds later.

Measured on device (Snapdragon-class, NTSYNC_DEBUG stats):

| Game | waits/s | wake_walks/ signal_ops | spin win rate | NTSYNC_SPIN_ITERS |
|---|---|---|---|---|
| Persona 5 Royal | 61k avg, 300k burst | 1% | **60-76% in bursts** | **300** |
| Dishonored | 11k | 5% | 3% | 0 |
| Lies of P | 3k | 67% | 1.4% | 0 |
| Hades II | 0.5k | 68% | 0.2% | 0 |

The signal is the **wake_walks ratio**: games where most signals are
no-op chatter (low ratio, waits usually short) tend to benefit; games
where most signals genuinely wake someone (high ratio, waits are
rendezvous) never do. P5R's profile — extreme bursts of hundreds of
thousands of waits/s with signals landing almost immediately — is the
fine-grained job-dispatch pattern spinning was built for.

A P5R control run with spin at 0 showed the identical wait profile
(same burst timing, wake latency, signal volume): spinning only
removes sleeps (25-28% of blocked waits during bursts), it does not
change game behavior. Note Denuvo titles like P5R run background
anti-tamper threads whose chatter-heavy, microsecond-handoff sync
pattern is exactly the spin-friendly quadrant, so Denuvo games are
prime candidates for the 5-minute test below.

The adaptive credit makes a wrong setting cheap rather than harmful
(threads that never win degrade to a 1/8 probe rate, ~0.2% of one
core), but 0 remains the correct default. To decide for a new game:

1. Run 5 minutes of the *busiest* gameplay with
   `NTSYNC_DEBUG=1 NTSYNC_SPIN_ITERS=300`.
2. Check `spin_wins` vs `spin_exhausted` in the dump: keep 300 when
   the win rate is > ~20%; set 0 otherwise.
3. When in doubt, leave it 0 — a bad spin setting gains nothing,
   it only wastes a little battery.

## Layout

- `src/core.rs` — shm object table + futex wait engine (unit-tested on host,
  including a cross-process fork test and `#[ignore]`d perf simulations)
- `src/ffi.rs` — exported C ABI (`ntsync_*` functions)
- `src/lib.rs` — tests and diagnostics glue
- `include/ntsync_user.h` — C header (same struct layout as `<linux/ntsync.h>`)
- `.cargo/config.toml` — `-z max-page-size=16384` linker flags for all Android
  targets (Google Play 16 KB page-size requirement)

## API

The C API mirrors the kernel ioctls one-to-one: objects are identified by
`uint32_t` handles instead of fds, and every function returns `0` on success
or a negative errno (`-EINVAL`, `-EPERM`, `-EOVERFLOW`, `-ETIMEDOUT`,
`-EOWNERDEAD`, `-EFAULT`), exactly like the kernel.

```c
ntsync_init(NULL);                  /* or ntsync_init("/path/to/region") */

uint32_t sem;
ntsync_create_sem(&sem, &(struct ntsync_sem_args){ .count = 0, .max = 4 });

uint32_t objs[] = { sem };
struct ntsync_wait_args wait = {
    .timeout = UINT64_MAX,          /* absolute ns, CLOCK_MONOTONIC */
    .objs = (uintptr_t)objs,
    .count = 1,
    .owner = gettid(),
};
int ret = ntsync_wait_any(&wait);   /* 0, wait.index set; -EOWNERDEAD if
                                       an abandoned mutex was acquired */
uint32_t prev = 1;
ntsync_sem_release(sem, &prev);     /* prev overwritten with old count */
ntsync_close(sem);
```

Handles are plain integers — share them between processes however you like
(shared memory, environment, your server protocol); no fd passing needed.
Handle 0 is never handed out (slot 0 is reserved): it is the "no alert"
sentinel in `ntsync_wait_args.alert`.

### Region layout versioning

The shm filename carries a `.vN` layout-version suffix
(`ntsync_userspace.v7.shm`) so builds with different layouts never open the
same region. A leftover stale file is unlinked and recreated — never
truncated or zeroed in place, which would SIGBUS/corrupt processes from an
older build that still have it mapped (old mappers keep their ghost inode).

### Semantic details (matching the kernel)

- `sem_release` returns `-EOVERFLOW` and leaves the count unchanged if it
  would exceed `max`.
- Mutexes are recursive per owner tid (tids are system-wide, so ownership
  works across processes); `mutex_unlock` returns the previous recursion
  count in `args->count`; `mutex_kill(handle, owner)` marks the mutex
  abandoned (`-EPERM` if caller is not the owner). Acquiring an abandoned
  mutex succeeds and the wait/read call returns `-EOWNERDEAD`.
- Auto-reset events are consumed by one waiter; manual-reset events stay
  signaled until `ntsync_event_reset`; `ntsync_event_pulse` wakes all
  current waiters and leaves the event unsignaled. `ntsync_event_set` /
  `_reset` / `_pulse` store the previous signaled state in `*prev` (if
  non-NULL), like the kernel ioctls.
- Alertable waits follow the kernel contract: if `alert` is nonzero it names
  an event object that aborts the wait, which then returns success with
  `index == count`.
- `timeout == UINT64_MAX` waits indefinitely; otherwise it is an absolute
  ns deadline on `CLOCK_MONOTONIC` (or `CLOCK_REALTIME` with
  `NTSYNC_WAIT_REALTIME`).
- Max 64 objects per wait (`NTSYNC_MAX_WAIT_COUNT`).

### Divergences from the kernel

- Closing an object that other threads/processes are waiting on fails those
  waits with `-EINVAL` (the kernel keeps objects alive via fd references).
- Objects leaked by a crashing process are not freed automatically (no fd
  close hook in userspace). Run `ntsync_sweep_dead()` from a launcher or
  server process after a child exits.
- Waits are not signal-interruptible with `-ERESTARTSYS` semantics; `EINTR`
  wakes are absorbed and the wait resumes.

## Build

Using the build script (mirrors the proton-wine arm64ec script; defaults to
`$HOME/Android/Sdk/ndk/27.3.13750724`, API 28). Builds arm64-v8a and x86_64;
16KB page-size alignment is always on (`.cargo/config.toml`), there is no
option for it:

```sh
./build-scripts/build-android.sh --build      # build both ABIs + verify alignment
./build-scripts/build-android.sh --install    # copy .so (per-ABI) + header to $OUTPUT_DIR
./build-scripts/build-android.sh --clean
# overrides: NDK=... API=... OUTPUT_DIR=...
```

Manually:

```sh
rustup target add aarch64-linux-android armv7-linux-androideabi x86_64-linux-android
NDK=~/Android/Sdk/ndk/<version>/toolchains/llvm/prebuilt/linux-x86_64/bin
CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$NDK/aarch64-linux-android24-clang \
cargo build --release --target aarch64-linux-android
# or: cargo install cargo-ndk && cargo ndk -t arm64-v8a -o jniLibs build --release
```

Verify page-size alignment:

```sh
readelf -lW target/aarch64-linux-android/release/libntsync_android.so | grep LOAD
# alignment column must be 0x4000
```

CI: `.github/workflows/build.yml` runs the host unit tests and builds
arm64-v8a / armeabi-v7a / x86_64 release libraries (API 28), verifies the
16KB alignment, and uploads each `.so` + `ntsync_user.h` as artifacts.

## Integrating with Proton

The crate builds as `cdylib`, `rlib` and `staticlib`, so Wine can also
static-link `libntsync_android.a` if a shared library is inconvenient.

Link `libntsync_android.so` into the ntdll unix-side build (or `dlopen` it)
and route the ntsync calls in `dlls/ntdll/unix/sync.c` to the `ntsync_*`
functions instead of `ioctl(fd, NTSYNC_IOC_*, ...)`. Replace the fd table
with the u32 handles; where Wine passes object fds through wineserver
(`SCM_RIGHTS`), pass the integer handle instead — all processes attached to
the same region path see the same objects. After a process exits, call
`ntsync_sweep_dead()` to reclaim its objects.

## Test

```sh
cargo test   # host-side unit tests, including cross-process fork tests.
             # Serialized: they fork while other test threads may hold the
             # shared region lock, which deadlocks the child.

# perf simulations (signaled-wait latency, signal churn, ping-pong wake
# latency same-process and cross-process, contended mutex):
cargo test --release -- --ignored --nocapture perf
```

## Benchmarks (CPU/memory regression baseline)

`src/bench.rs` simulates a real Wine workload (render thread with frame
pacing, thread-pool workers on job semaphores, contended device mutex,
object create/close churn, alertable APC-style waits,
MsgWaitForMultipleObjects-style multi-waits, cross-process
wineserver↔client ping-pong, a 4-process contention test on the shared
region, and an exec-based variant where each child process independently
opens/mmaps the region like a fresh Wine process, reporting its own shm
RSS) and profiles it with `getrusage` (CPU time,
context switches) and `/proc/self/smaps` (shm file size, mapped RSS, dirty
pages). Use it to validate that an optimization actually reduces CPU or
file-I/O without regressing anything else:

```sh
# 1. Before changing code: record the baseline.
NTSYNC_BENCH_BASELINE=write cargo test --release -- --ignored --nocapture --test-threads=1 bench

# 2. After changing code: run again; every metric is printed next to its
#    baseline value with a delta (REGRESSED = >20% worse).
cargo test --release -- --ignored --nocapture --test-threads=1 bench

# 3. Optionally make regressions fail the run:
NTSYNC_BENCH_STRICT=1 cargo test --release -- --ignored --nocapture --test-threads=1 bench
```

Baselines live in `target/ntsync-bench-baseline.json`; the latest run is
always in `target/ntsync-bench-latest.json`. All metrics are
lower-is-better (`cpu_ns_per_op`, `voluntary_ctxsw`, `shm_rss_kb_*`, ...).
Benches serialize on a global lock because rusage/RSS are process-wide;
always pass `--test-threads=1` for comparable numbers, and compare only
baselines recorded on the same machine.

Current facts this suite pins down: the shm region file is
983,120 B (~960 KiB: 16384 slots x 40 B + 8192 waiter nodes x 40 B), but
thanks to lazy init it starts sparse — ~16 kB mapped RSS at init,
growing only as objects and waiters actually arrive. The signaled-wait
fast path takes zero voluntary context switches (no syscalls).

## License

Copyright (C) 2026 Joshua Tam <297250+joshuatam@users.noreply.github.com>

GNU Lesser General Public License v3.0 — see [LICENSE](LICENSE). LGPL-3.0
incorporates the terms of GPL-3.0 by reference
(<https://www.gnu.org/licenses/gpl-3.0.txt>). This allows linking
`libntsync_android.so` into Wine (LGPL-2.1+) or other programs without
affecting their license, while the library itself stays copyleft.
`include/ntsync_user.h` uses the same struct layout as the kernel's
GPL-2.0-with-syscall-exception uapi header `<linux/ntsync.h>`; note the
different licensing of that upstream header.
