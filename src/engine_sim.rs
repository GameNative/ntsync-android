// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (C) 2026 Joshua Tam <297250+joshuatam@users.noreply.github.com>

//! Engine frame-pacing simulations for the userspace ntsync layer.
//!
//! Each scenario is a load model of how a shipping engine drives the Win32
//! synchronisation primitives that Wine routes to this library, at frame rates
//! a high-end title actually hits on a phone once the frame is cheap
//! (500-1000 fps, i.e. 1-2 ms per frame):
//!
//! - `sim_ue45_frame_graph`: Unreal 4/5. Game thread pushes parallel work into a
//!   mutex-guarded queue behind a semaphore, workers release a completion
//!   semaphore, the frame ends on an auto-reset event set by a render thread.
//!   This is the shape of `FQueuedThreadPool` + `FRunnable` + the game/render
//!   fence.
//! - `sim_unity_job_system`: Unity. One auto-reset event per job, completed by
//!   `JobHandle.Complete` waits, plus the contended "graphics device" mutex that
//!   main thread and workers both take.
//! - `sim_godot_worker_pool`: Godot. `WorkerThreadPool` task group: semaphore
//!   queue in, per-group completion semaphore out, main thread waits for the
//!   whole group.
//! - `sim_vsync_pacing`: the frame-boundary sleep. A wait with a short timeout
//!   on an event that is never set, which is how a title paces to the display
//!   when vsync is emulated rather than blocking in `Present`.
//! - `sim_wineserver_object_churn`: a second process creating and closing
//!   objects at wineserver rates while the game thread signals and waits, which
//!   is the only way the shared region lock is contended in practice.
//!
//! Run them (all take ~1-3 s each, and they must not run concurrently):
//!
//! ```sh
//! cargo test --release -- --ignored --nocapture --test-threads=1 engine
//! ```
//!
//! Reported per scenario: frame period distribution against the target,
//! overruns, and the handoff latency (signal -> waiter running) distribution.

use crate::core::*;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Once};
use std::thread;
use std::time::{Duration, Instant};

static INIT: Once = Once::new();

fn setup() {
    INIT.call_once(|| {
        let path = std::env::temp_dir().join(format!("ntsync-engine-{}.shm", std::process::id()));
        let _ = std::fs::remove_file(&path);
        init(Some(path.to_str().unwrap())).unwrap();
    });
    let _ = sweep_dead();
}

/// Process CPU time (utime+stime), to price a spin.
fn cpu_time() -> Duration {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) };
    let us = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
    us(ru.ru_utime) + us(ru.ru_stime)
}

/// `PR_SET_TIMERSLACK` to 1 ns, if the run asked for it: the cheap way to get
/// a tight timer on bionic (default slack is 50 us), at the price of more timer
/// interrupts instead of a spin.
fn maybe_tighten_timerslack() -> bool {
    if std::env::var("NTSYNC_SIM_TIMERSLACK").as_deref() != Ok("1") {
        return false;
    }
    unsafe { libc::prctl(libc::PR_SET_TIMERSLACK, 1u64, 0, 0, 0) == 0 }
}

fn me() -> u32 {
    std::process::id()
}

/// Latency samples in nanoseconds, reported as quantiles.
struct Hist(Vec<u64>);

impl Hist {
    fn with_capacity(n: usize) -> Self {
        Self(Vec::with_capacity(n))
    }
    fn push(&mut self, d: Duration) {
        self.0.push(d.as_nanos() as u64);
    }
    fn report(&mut self, name: &str) {
        if self.0.is_empty() {
            println!("  {name:<26} (no samples)");
            return;
        }
        self.0.sort_unstable();
        let q = |p: f64| self.0[(((self.0.len() - 1) as f64) * p) as usize];
        println!(
            "  {name:<26} n={:<7} p50={:>9.2}us p90={:>9.2}us p99={:>9.2}us max={:>9.2}us",
            self.0.len(),
            q(0.50) as f64 / 1000.0,
            q(0.90) as f64 / 1000.0,
            q(0.99) as f64 / 1000.0,
            self.0[self.0.len() - 1] as f64 / 1000.0
        );
    }
}

fn handoff_thread(go: u32, acked: u32, stop: Arc<AtomicBool>) -> thread::JoinHandle<Hist> {
    thread::spawn(move || {
        let mut hist = Hist::with_capacity(200_000);
        let mut local = [0u64; 1];
        while !stop.load(Ordering::Relaxed) {
            match wait_any(&[go], me(), Some(Duration::from_millis(50)), 0) {
                WaitOutcome::Signaled { .. } => {
                    let t = Instant::now();
                    let _ = event_set(acked);
                    // The peer measures; nothing to record here.
                    local[0] = t.elapsed().as_nanos() as u64;
                }
                WaitOutcome::Timeout => {}
                _ => break,
            }
        }
        hist.push(Duration::from_nanos(local[0]));
        hist
    })
}

/// Frame accounting shared by the engine scenarios.
struct Frames {
    target: Duration,
    periods: Hist,
    overruns: u32,
    total: u32,
}

impl Frames {
    fn new(fps: u32) -> Self {
        Self {
            target: Duration::from_secs_f64(1.0 / fps as f64),
            periods: Hist::with_capacity(8192),
            overruns: 0,
            total: 0,
        }
    }
    fn end(&mut self, start: Instant) {
        let el = start.elapsed();
        self.periods.push(el);
        self.total += 1;
        if el > self.target {
            self.overruns += 1;
        }
    }
    fn report(&mut self, name: &str, fps: u32) {
        println!(
            "--- {name}: target {fps} fps ({:.0} us/frame), {} frames, {} overruns ({:.2}%)",
            self.target.as_secs_f64() * 1e6,
            self.total,
            self.overruns,
            100.0 * self.overruns as f64 / self.total.max(1) as f64
        );
        self.periods.report("frame period");
    }
}

// ---------------------------------------------------------------- Unreal ---

/// Unreal 4/5 frame graph: parallel work behind a semaphore, completion
/// counted by a second semaphore, frame closed by a render-thread event.
#[test]
#[ignore]
fn sim_ue45_frame_graph() {
    setup();
    const FPS: u32 = 500;
    const TASKS: usize = 8;
    const WORKERS: usize = 3;
    const FRAMES: usize = 400;

    let job_sem = create_semaphore(0, 64).unwrap();
    let done_sem = create_semaphore(0, 64).unwrap();
    let queue_mutex = create_mutex(0, 0).unwrap();
    let render_go = create_event(false, false).unwrap();
    let frame_done = create_event(false, false).unwrap();
    let stop = Arc::new(AtomicBool::new(false));

    // Worker pool: pull a job, do the work, release the completion semaphore.
    let mut workers = Vec::new();
    for _ in 0..WORKERS {
        let stop = stop.clone();
        workers.push(thread::spawn(move || {
            let mut done = 0u64;
            while !stop.load(Ordering::Relaxed) {
                match wait_any(&[job_sem], me(), Some(Duration::from_millis(50)), 0) {
                    WaitOutcome::Signaled { .. } => {}
                    WaitOutcome::Timeout => continue,
                    _ => break,
                }
                // Pop a job under the queue mutex, as FQueuedThreadPool does.
                wait_any(&[queue_mutex], me(), None, 0);
                let _ = mutex_unlock(queue_mutex, me());
                // ~20 us of "work".
                let t = Instant::now();
                while t.elapsed() < Duration::from_micros(20) {
                    std::hint::spin_loop();
                }
                let _ = sem_release(done_sem, 1);
                done += 1;
            }
            done
        }));
    }

    // Render thread: one frame in flight, closed by an auto-reset event.
    let render = {
        let stop = stop.clone();
        thread::spawn(move || {
            let mut frames = 0u64;
            while !stop.load(Ordering::Relaxed) {
                match wait_any(&[render_go], me(), Some(Duration::from_millis(50)), 0) {
                    WaitOutcome::Signaled { .. } => {}
                    WaitOutcome::Timeout => continue,
                    _ => break,
                }
                // Render-side work (submit + GPU wait, ~1 ms) before the
                // frame fence is set: this is what the game thread blocks on.
                let t = Instant::now();
                while t.elapsed() < Duration::from_micros(1_000) {
                    std::hint::spin_loop();
                }
                let _ = event_set(frame_done);
                frames += 1;
            }
            frames
        })
    };

    let mut frames = Frames::new(FPS);
    let budget = Duration::from_secs_f64(1.0 / FPS as f64);
    for _ in 0..FRAMES {
        let start = Instant::now();
        for _ in 0..TASKS {
            wait_any(&[queue_mutex], me(), None, 0);
            let _ = mutex_unlock(queue_mutex, me());
            let _ = sem_release(job_sem, 1);
        }
        for _ in 0..TASKS {
            let _ = wait_any(&[done_sem], me(), None, 0);
        }
        let _ = event_set(render_go);
        // Pace the frame: wait for the render thread inside the remaining
        // budget, as a title does when Present does not block.
        let left = budget.saturating_sub(start.elapsed());
        let _ = wait_any(&[frame_done], me(), Some(left), 0);
        frames.end(start);
    }
    stop.store(true, Ordering::Relaxed);
    let _ = sem_release(job_sem, WORKERS as u32);
    let _ = event_set(render_go);
    frames.report("UE4/UE5 frame graph", FPS);
    for w in workers {
        w.join().unwrap();
    }
    let _ = render.join().unwrap();
    assert_eq!(frames.total as usize, FRAMES);
    let _ = close(job_sem);
    let _ = close(done_sem);
    let _ = close(queue_mutex);
    let _ = close(render_go);
    let _ = close(frame_done);
}

// ------------------------------------------------------------------ Unity ---

/// Unity job system: per-job events, `Complete()` waits, and the contended
/// graphics-device mutex.
#[test]
#[ignore]
fn sim_unity_job_system() {
    setup();
    const FPS: u32 = 500;
    const JOBS: usize = 8;
    const WORKERS: usize = 4;
    const FRAMES: usize = 400;

    let job_sem = create_semaphore(0, 64).unwrap();
    let gfx = create_mutex(0, 0).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let job_events: Vec<u32> = (0..JOBS).map(|_| create_event(false, false).unwrap()).collect();
    let next_job = Arc::new(AtomicU32::new(0));

    let mut workers = Vec::new();
    for _ in 0..WORKERS {
        let stop = stop.clone();
        let events = job_events.clone();
        let next_job = next_job.clone();
        workers.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                match wait_any(&[job_sem], me(), Some(Duration::from_millis(50)), 0) {
                    WaitOutcome::Signaled { .. } => {}
                    WaitOutcome::Timeout => continue,
                    _ => break,
                }
                let idx = next_job.fetch_add(1, Ordering::Relaxed) as usize % events.len();
                // Work, then take the graphics-device mutex to submit, as a
                // job with render commands does.
                let t = Instant::now();
                while t.elapsed() < Duration::from_micros(15) {
                    std::hint::spin_loop();
                }
                let _ = wait_any(&[gfx], me(), None, 0);
                let _ = mutex_unlock(gfx, me());
                let _ = event_set(events[idx]);
            }
        }));
    }

    let mut frames = Frames::new(FPS);
    let budget = Duration::from_secs_f64(1.0 / FPS as f64);
    let mut complete = Hist::with_capacity(8192);
    for _ in 0..FRAMES {
        let start = Instant::now();
        next_job.store(0, Ordering::Relaxed);
        for _ in 0..JOBS {
            let _ = sem_release(job_sem, 1);
        }
        // JobHandle.Complete(): one wait per job, in dispatch order.
        for &event in job_events.iter() {
            let w = Instant::now();
            let _ = wait_any(&[event], me(), Some(Duration::from_millis(50)), 0);
            complete.push(w.elapsed());
        }
        // Main thread also submits under the same device mutex.
        let _ = wait_any(&[gfx], me(), None, 0);
        let _ = mutex_unlock(gfx, me());
        let left = budget.saturating_sub(start.elapsed());
        if !left.is_zero() {
            let _ = wait_any(&[job_events[0]], me(), Some(left), 0);
        }
        frames.end(start);
    }
    stop.store(true, Ordering::Relaxed);
    let _ = sem_release(job_sem, WORKERS as u32);
    frames.report("Unity job system", FPS);
    complete.report("Complete() wait");
    for w in workers {
        w.join().unwrap();
    }
    assert_eq!(frames.total as usize, FRAMES);
    for e in job_events {
        let _ = close(e);
    }
    let _ = close(job_sem);
    let _ = close(gfx);
}

// ------------------------------------------------------------------ Godot ---

/// Godot `WorkerThreadPool`: task group in, group completion semaphore out.
#[test]
#[ignore]
fn sim_godot_worker_pool() {
    setup();
    const FPS: u32 = 500;
    const TASKS: usize = 6;
    const WORKERS: usize = 3;
    const FRAMES: usize = 400;

    let job_sem = create_semaphore(0, 64).unwrap();
    let group_sem = create_semaphore(0, 64).unwrap();
    let queue = create_mutex(0, 0).unwrap();
    let stop = Arc::new(AtomicBool::new(false));

    let mut workers = Vec::new();
    for _ in 0..WORKERS {
        let stop = stop.clone();
        workers.push(thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                match wait_any(&[job_sem], me(), Some(Duration::from_millis(50)), 0) {
                    WaitOutcome::Signaled { .. } => {}
                    WaitOutcome::Timeout => continue,
                    _ => break,
                }
                wait_any(&[queue], me(), None, 0);
                let _ = mutex_unlock(queue, me());
                let t = Instant::now();
                while t.elapsed() < Duration::from_micros(25) {
                    std::hint::spin_loop();
                }
                let _ = sem_release(group_sem, 1);
            }
        }));
    }

    let mut frames = Frames::new(FPS);
    let budget = Duration::from_secs_f64(1.0 / FPS as f64);
    for _ in 0..FRAMES {
        let start = Instant::now();
        for _ in 0..TASKS {
            let _ = wait_any(&[queue], me(), None, 0);
            let _ = mutex_unlock(queue, me());
            let _ = sem_release(job_sem, 1);
        }
        // wait_for_group_task_completion()
        for _ in 0..TASKS {
            let _ = wait_any(&[group_sem], me(), None, 0);
        }
        let left = budget.saturating_sub(start.elapsed());
        if !left.is_zero() {
            let _ = wait_any(&[group_sem], me(), Some(left), 0);
        }
        frames.end(start);
    }
    stop.store(true, Ordering::Relaxed);
    let _ = sem_release(job_sem, WORKERS as u32);
    frames.report("Godot WorkerThreadPool", FPS);
    for w in workers {
        w.join().unwrap();
    }
    assert_eq!(frames.total as usize, FRAMES);
    let _ = close(job_sem);
    let _ = close(group_sem);
    let _ = close(queue);
}

// ------------------------------------------------------- frame boundary ---

/// Pacing to the frame boundary: repeated short-timeout waits on an event that
/// never signals, against `thread::sleep` for reference. Reports overshoot,
/// which is the jitter a player feels as uneven frames.
#[test]
#[ignore]
fn sim_vsync_pacing() {
    setup();
    const FPS: u32 = 1000;
    const FRAMES: usize = 1000;

    let never = create_event(false, false).unwrap();
    let period = Duration::from_secs_f64(1.0 / FPS as f64);
    let tight = maybe_tighten_timerslack();
    let cpu0 = cpu_time();
    let mut overshoot = Hist::with_capacity(FRAMES);
    let mut sleep_overshoot = Hist::with_capacity(FRAMES);
    let mut next = Instant::now();

    for _ in 0..FRAMES {
        next += period;
        let now = Instant::now();
        if next > now {
            let left = next - now;
            let t = Instant::now();
            let _ = wait_any(&[never], me(), Some(left), 0);
            // Overshoot beyond the intended wake point is the jitter.
            overshoot.push(t.elapsed().saturating_sub(left));
        }
    }
    // Reference: the same pacing loop on its own deadline, parking with the
    // libc sleep primitive instead.
    let mut next3 = Instant::now();
    for _ in 0..FRAMES {
        next3 += period;
        let t = Instant::now();
        if next3 > t {
            let left = next3 - t;
            let t = Instant::now();
            thread::sleep(left);
            sleep_overshoot.push(t.elapsed().saturating_sub(left));
        }
    }
    // Same pacing, spinning through the last 150 us and parking only for the
    // remainder: trade a little CPU for a tighter wake point.
    const SPIN_WINDOW: Duration = Duration::from_micros(150);
    let mut spun = Hist::with_capacity(FRAMES);
    let mut next2 = Instant::now();
    for _ in 0..FRAMES {
        next2 += period;
        let now = Instant::now();
        if next2 > now {
            let left = next2 - now;
            let park = left.saturating_sub(SPIN_WINDOW);
            let t = Instant::now();
            if !park.is_zero() {
                let _ = wait_any(&[never], me(), Some(park), 0);
            }
            while Instant::now() < next2 {
                std::hint::spin_loop();
            }
            spun.push(t.elapsed().saturating_sub(left));
        }
    }
    let cpu_per_frame = cpu_time().saturating_sub(cpu0) / FRAMES as u32;
    println!(
        "--- frame-boundary pacing at {FPS} fps ({:.0} us/frame), timerslack tightened: {tight}, \
         cpu/frame {:.2}us",
        period.as_secs_f64() * 1e6,
        cpu_per_frame.as_secs_f64() * 1e6
    );
    overshoot.report("ntsync wait overshoot");
    spun.report("spin(150us)+wait overshoot");
    sleep_overshoot.report("thread::sleep overshoot");
    let _ = close(never);
}

/// One library wait per frame and nothing else, so cpu/frame is the cost of
/// the wait path alone. This is the instrument for the spin-tail knobs: with
/// a tail configured the wait spins its last `NTSYNC_SPIN_TAIL_US`, and
/// `NTSYNC_SPIN_TAIL_BUDGET_PCT` caps how much of a core that may cost, so
/// an A/B over the budget shows the gate binding (cpu/frame falls, overshoot
/// returns to the parked profile).
#[test]
#[ignore]
fn sim_wait_only_pacing() {
    setup();
    const FPS: u32 = 1000;
    const FRAMES: usize = 2000;

    let never = create_event(false, false).unwrap();
    let period = Duration::from_secs_f64(1.0 / FPS as f64);
    let tail = std::env::var("NTSYNC_SPIN_TAIL_US").unwrap_or_else(|_| "off".into());
    let pct = std::env::var("NTSYNC_SPIN_TAIL_BUDGET_PCT").unwrap_or_else(|_| "default".into());

    let cpu0 = cpu_time();
    let mut over = Hist::with_capacity(FRAMES);
    let mut next = Instant::now();
    for _ in 0..FRAMES {
        next += period;
        let now = Instant::now();
        if next > now {
            let left = next - now;
            let t = Instant::now();
            let _ = wait_any(&[never], me(), Some(left), 0);
            over.push(t.elapsed().saturating_sub(left));
        }
    }
    let cpu = cpu_time().saturating_sub(cpu0) / FRAMES as u32;
    println!(
        "--- wait-only pacing at {FPS} fps, tail={tail}us budget={pct}%, \
         cpu/frame {:.2}us",
        cpu.as_secs_f64() * 1e6
    );
    over.report("wait-only overshoot");
    let _ = close(never);
}

// ---------------------------------------------------- cross-process lock ---

/// What wineserver does to the shared region while a game runs: object create
/// and close in another process, which are the operations that take the
/// region lock. The game thread signals and waits throughout; the numbers are
/// what a title would see as frame jitter.
#[test]
#[ignore]
fn sim_wineserver_object_churn() {
    setup();
    const SECS: u64 = 2;
    let ev = create_event(false, false).unwrap();
    let partner = create_event(false, false).unwrap();

    let pid = unsafe { libc::fork() };
    assert!(pid >= 0);
    if pid == 0 {
        // Child == wineserver stand-in: churn objects for the run duration.
        let deadline = Instant::now() + Duration::from_secs(SECS);
        while Instant::now() < deadline {
            for _ in 0..16 {
                if let Ok(h) = create_event(false, false) {
                    let _ = close(h);
                }
            }
        }
        unsafe { libc::_exit(0) }
    }

    // Game thread: one signal plus one short-timeout wait per "frame".
    let mut set_lat = Hist::with_capacity(400_000);
    let mut wait_lat = Hist::with_capacity(400_000);
    let deadline = Instant::now() + Duration::from_secs(SECS);
    while Instant::now() < deadline {
        let t = Instant::now();
        let _ = event_set(ev);
        set_lat.push(t.elapsed());
        let t = Instant::now();
        let _ = wait_any(&[partner], me(), Some(Duration::from_micros(500)), 0);
        wait_lat.push(t.elapsed());
        let _ = event_reset(ev);
    }
    let mut status = 0;
    unsafe { libc::waitpid(pid, &mut status, 0) };
    println!("--- game thread ops while another process creates/closes objects");
    set_lat.report("event_set latency");
    wait_lat.report("wait(500us timeout) latency");
    let _ = close(ev);
    let _ = close(partner);
}

// ------------------------------------------------------- crashed helper ---

/// A helper process died leaving objects behind. The opportunistic sweep runs
/// from a wait path (here: the pacing wait, which is where engines hit it) and
/// reaps them with the region lock held, so the game thread sees the sweep as
/// frame jitter. 500 objects is a modest helper; a launcher or a crashed
/// "wineserver-adjacent" process leaves far more.
#[test]
#[ignore]
fn sim_sweep_hitch() {
    setup();
    const OBJECTS: usize = 500;
    const FRAMES: usize = 1000;

    // Leave the orphans: a dead owner's events.
    let pid = unsafe { libc::fork() };
    assert!(pid >= 0);
    if pid == 0 {
        for _ in 0..OBJECTS {
            let _ = create_event(false, false).unwrap();
        }
        unsafe { libc::_exit(0) }
    }
    let mut status = 0;
    unsafe { libc::waitpid(pid, &mut status, 0) };

    // Pacing wait with a 1 ms timeout, as a 500 fps loop would use, long
    // enough that the 256-call sweep gate fires inside the measured window.
    let never = create_event(false, false).unwrap();
    let mut lat = Hist::with_capacity(FRAMES);
    test_reset_sweep_clock();
    for _ in 0..FRAMES {
        let t = Instant::now();
        let _ = wait_any(&[never], me(), Some(Duration::from_millis(1)), 0);
        lat.push(t.elapsed());
    }
    println!("--- pacing wait while a dead process's {OBJECTS} objects are swept");
    lat.report("wait(1ms timeout)");

    // The reaping pass itself, then a repeat pass over the same table with
    // nothing left to reap: that repeat is what the rate limiter is for.
    let mut orphan2 = 0;
    for _ in 0..OBJECTS {
        if let Ok(h) = create_event(false, false) {
            let _ = close(h);
            orphan2 += 1;
        }
    }
    let _ = orphan2;
    let t = Instant::now();
    let freed = sweep_dead().unwrap_or(0);
    let reap = t.elapsed();
    let t = Instant::now();
    let again = sweep_dead().unwrap_or(0);
    let repeat = t.elapsed();
    println!("  sweep: reaped {freed} in {:?}, repeat pass (nothing dead) {again} in {:?}", reap, repeat);
    let _ = close(never);
}

// --------------------------------------------------------------- handoff ---

/// Signal -> waiter-running cost in isolation, for reference against the
/// engine scenarios above.
#[test]
#[ignore]
fn sim_handoff_cost() {
    setup();
    let go = create_event(false, false).unwrap();
    let acked = create_event(false, false).unwrap();
    let stop = Arc::new(AtomicBool::new(false));
    let waiter = handoff_thread(go, acked, stop.clone());
    let mut hist = Hist::with_capacity(100_000);
    let mut peer = Hist::with_capacity(100_000);
    for _ in 0..20_000 {
        let t = Instant::now();
        let _ = event_set(go);
        hist.push(t.elapsed());
        let t = Instant::now();
        let _ = wait_any(&[acked], me(), Some(Duration::from_millis(50)), 0);
        peer.push(t.elapsed());
    }
    stop.store(true, Ordering::Relaxed);
    let _ = event_set(go);
    let _ = waiter.join();
    println!("--- cross-thread handoff");
    hist.report("set_event -> return");
    peer.report("wake -> waiter running");
    let _ = close(go);
    let _ = close(acked);
}

// ---------------------------------------------------------------- basics ---

/// Reference points for the operations the engine scenarios are built from:
/// uncontended acquire/wait, semaphore ping-pong, contended mutex.
#[test]
#[ignore]
fn sim_primitive_costs() {
    setup();
    const N: usize = 200_000;

    let ev = create_event(false, true).unwrap();
    let mut signal = Hist::with_capacity(N);
    let mut read = Hist::with_capacity(N);
    let mut fastwait = Hist::with_capacity(N);
    for _ in 0..N {
        let t = Instant::now();
        let _ = event_set(ev);
        signal.push(t.elapsed());
        let t = Instant::now();
        let _ = read_event(ev);
        read.push(t.elapsed());
        let t = Instant::now();
        let _ = wait_any(&[ev], me(), Some(Duration::ZERO), 0);
        fastwait.push(t.elapsed());
    }
    // Timeout expiries: the pacing path.
    let never = create_event(false, false).unwrap();
    let mut timeout = Hist::with_capacity(20_000);
    for _ in 0..20_000 {
        let t = Instant::now();
        let _ = wait_any(&[never], me(), Some(Duration::from_micros(200)), 0);
        timeout.push(t.elapsed());
    }
    // Uncontended semaphore acquire/release.
    let sem = create_semaphore(1, 1).unwrap();
    let mut semops = Hist::with_capacity(N);
    for _ in 0..N {
        let t = Instant::now();
        let _ = wait_any(&[sem], me(), Some(Duration::ZERO), 0);
        let _ = sem_release(sem, 1);
        semops.push(t.elapsed());
    }
    // Uncontended mutex acquire/release.
    let m = create_mutex(0, 0).unwrap();
    let mut mutexops = Hist::with_capacity(N);
    for _ in 0..N {
        let t = Instant::now();
        let _ = wait_any(&[m], me(), Some(Duration::ZERO), 0);
        let _ = mutex_unlock(m, me());
        mutexops.push(t.elapsed());
    }
    println!("--- primitive costs (single thread)");
    signal.report("event_set");
    read.report("event_read");
    fastwait.report("wait(signaled) fast path");
    timeout.report("wait(200us timeout) expiry");
    semops.report("sem acquire+release");
    mutexops.report("mutex acquire+release");
    let _ = close(ev);
    let _ = close(never);
    let _ = close(sem);
    let _ = close(m);
}
