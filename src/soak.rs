// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (C) 2026 Joshua Tam <297250+joshuatam@users.noreply.github.com>

//! Soak tests for the dead-owner sweep, run with the interval cranked down so
//! passes happen constantly instead of once per 30 s.
//!
//! ```sh
//! NTSYNC_SWEEP_INTERVAL_SEC=1 cargo test --release -- --ignored --nocapture soak --test-threads=1
//! ```
//!
//! What is being hunted:
//!
//! - a stale reap corrupting a live object's waiter list (freed slot reused
//!   while the sweep was unlocked, then detached against it) — shows up as a
//!   canary object that stops signalling or a wait that never returns;
//! - double-insertion into the node free list (a node freed twice, or freed
//!   while still on a waiter's own chain) — shows up as two waiters handed the
//!   same node, i.e. a wait that hangs or a wake that never lands;
//! - object lifetime after the creator dies: a handle relayed to another
//!   process is reaped out from under the holder, which the kernel driver does
//!   not do. Recorded, not asserted: it is a design difference, and knowing
//!   whether it happens tells us what a relayed handle can be trusted for.
//!
//! The parent stays single-threaded across every `fork`, so a child never
//! inherits a region mutex held by a thread that does not exist in it (the
//! hazard that makes the ordinary suite serial).

use crate::core::*;
use std::sync::Once;
use std::time::{Duration, Instant};

static INIT: Once = Once::new();

fn setup() {
    INIT.call_once(|| {
        let path = std::env::temp_dir().join(format!("ntsync-soak-{}.shm", std::process::id()));
        let _ = std::fs::remove_file(&path);
        init(Some(path.to_str().unwrap())).unwrap();
    });
}

fn me() -> u32 {
    std::process::id()
}

/// Drive enough waits to pass the 256-call sweep gate, with timeouts so each
/// one takes the ETIMEDOUT path where the opportunistic sweep lives.
fn trigger_sweeps(ev: u32, calls: usize) {
    // Force the region-wide interval to look elapsed, so a pass is guaranteed
    // to run inside these calls instead of depending on wall time.
    test_reset_sweep_clock();
    for _ in 0..calls {
        let _ = wait_any(&[ev], me(), Some(Duration::from_millis(1)), 0);
    }
}

/// Canary: a parent-owned set of primitives that must survive every sweep
/// untouched, because their creator (this process) is alive.
fn check_canaries(tag: &str) {
    let ev = create_event(false, false).unwrap();
    let sem = create_semaphore(1, 8).unwrap();
    let mtx = create_mutex(0, 0).unwrap();

    // read_event returns (manual, signaled).
    assert!(event_set(ev).is_ok(), "{tag}: event_set failed");
    assert!(read_event(ev).unwrap().1, "{tag}: event not signaled");
    assert!(event_reset(ev).is_ok(), "{tag}: event_reset failed");
    assert!(!read_event(ev).unwrap().1, "{tag}: event still signaled");

    assert_eq!(sem_release(sem, 2).unwrap(), 1, "{tag}: sem count wrong");
    assert_eq!(read_sem(sem).unwrap().0, 3, "{tag}: sem value wrong");

    assert!(
        matches!(
            wait_any(&[mtx], me(), Some(Duration::ZERO), 0),
            WaitOutcome::Signaled { index: 0, .. }
        ),
        "{tag}: mutex acquire failed"
    );
    assert!(mutex_unlock(mtx, me()).is_ok(), "{tag}: mutex release failed");

    assert!(close(ev) && close(sem) && close(mtx), "{tag}: close failed");
}

/// Holder bookkeeping: retain is idempotent per process and release clears the
/// entry, so a holder cannot inflate its own record into a permanent leak.
#[test]
#[ignore]
fn soak_holder_bookkeeping() {
    setup();
    let h = create_event(false, false).unwrap();
    assert_eq!(test_holders(h).unwrap().1, Vec::<u32>::new());

    assert_eq!(retain(h), Ok(()));
    assert_eq!(test_holders(h).unwrap().1, vec![me()]);
    assert_eq!(retain(h), Ok(()), "retain is idempotent");
    assert_eq!(test_holders(h).unwrap().1, vec![me()], "duplicate holder entry");

    assert_eq!(release(h), Ok(()));
    assert_eq!(test_holders(h).unwrap().1, Vec::<u32>::new());
    assert_eq!(release(h), Ok(()), "release of a non-holder is a no-op");

    // close() also drops the holder entry.
    assert_eq!(retain(h), Ok(()));
    assert!(close(h));
    assert!(test_holders(h).is_none(), "slot freed but still resolvable");
    assert!(retain(h).is_err(), "retain on a freed handle must fail");
}

/// Repeated process death under a fast sweep: the sweep must never damage a
/// live owner's objects, hang, or hand out a corrupted node.
#[test]
#[ignore]
fn soak_process_death_under_sweep() {
    setup();
    const CYCLES: usize = 40;
    const ORPHANS: usize = 64;

    let trigger = create_event(false, false).unwrap();
    let started = Instant::now();
    for cycle in 0..CYCLES {
        check_canaries(&format!("cycle {cycle} pre"));

        let pid = unsafe { libc::fork() };
        assert!(pid >= 0);
        if pid == 0 {
            // Child: leave objects and registered waiter nodes behind, with no
            // cleanup at all, and with threads still blocked at exit.
            let mut orphans = Vec::with_capacity(ORPHANS);
            for _ in 0..ORPHANS {
                match create_event(false, false) {
                    Ok(h) => orphans.push(h),
                    Err(_) => break,
                }
            }
            let blocker = match create_event(false, false) {
                Ok(h) => h,
                Err(_) => unsafe { libc::_exit(0) },
            };
            for _ in 0..2 {
                std::thread::spawn(move || {
                    let _ = wait_any(&[blocker], unsafe { libc::getpid() } as u32, None, 0);
                });
            }
            // Let the waiters register their nodes, then die without unwinding.
            std::thread::sleep(Duration::from_millis(5));
            unsafe { libc::_exit(0) }
        }
        let mut status = 0;
        unsafe { libc::waitpid(pid, &mut status, 0) };

        // Reap them via the opportunistic path, several passes per cycle.
        trigger_sweeps(trigger, 768);

        check_canaries(&format!("cycle {cycle} post"));
        eprintln!("soak: cycle {cycle}/{CYCLES} ok ({:?})", started.elapsed());
    }
    let _ = close(trigger);
}

/// Handle relay across process death, the defect this branch fixes: a handle
/// passed to another process must survive its creator the way a duplicated fd
/// does under the kernel driver. Without `retain` it is reaped at the next
/// pass; with it the object lives until the holder releases or dies.
#[test]
#[ignore]
fn soak_relayed_handle_after_creator_death() {
    setup();
    let trigger = create_event(false, false).unwrap();

    let mut fds = [0i32; 2];
    assert_eq!(unsafe { libc::pipe(fds.as_mut_ptr()) }, 0);
    let (rd, wr) = (fds[0], fds[1]);

    let pid = unsafe { libc::fork() };
    assert!(pid >= 0);
    if pid == 0 {
        // Child: create an object and hand the handle to the parent.
        let h = create_event(false, false).unwrap();
        let bytes = h.to_ne_bytes();
        unsafe {
            libc::write(wr, bytes.as_ptr() as *const _, 4);
            libc::_exit(0)
        }
    }
    unsafe { libc::close(wr) };
    let mut bytes = [0u8; 4];
    let n = unsafe { libc::read(rd, bytes.as_mut_ptr() as *mut _, 4) };
    assert_eq!(n, 4);
    unsafe { libc::close(rd) };
    let mut status = 0;
    unsafe { libc::waitpid(pid, &mut status, 0) };
    let relayed = u32::from_ne_bytes(bytes);

    // Unretained: the sweep is entitled to reap it once the creator is gone.
    println!("soak: relayed handle {relayed:#x} from dead pid {pid}, without retain:");
    println!("  before any sweep:  usable = {}", read_event(relayed).is_ok());
    trigger_sweeps(trigger, 512);
    println!("  after one sweep:   usable = {}", read_event(relayed).is_ok());

    // Retained: now it must outlive the creator until this process lets go.
    let mut fds2 = [0i32; 2];
    assert_eq!(unsafe { libc::pipe(fds2.as_mut_ptr()) }, 0);
    let (rd2, wr2) = (fds2[0], fds2[1]);
    let pid2 = unsafe { libc::fork() };
    assert!(pid2 >= 0);
    if pid2 == 0 {
        let h = create_event(false, false).unwrap();
        unsafe {
            libc::write(wr2, h.to_ne_bytes().as_ptr() as *const _, 4);
            libc::_exit(0)
        }
    }
    unsafe { libc::close(wr2) };
    let mut bytes2 = [0u8; 4];
    assert_eq!(
        unsafe { libc::read(rd2, bytes2.as_mut_ptr() as *mut _, 4) },
        4
    );
    unsafe { libc::close(rd2) };
    let mut status2 = 0;
    unsafe { libc::waitpid(pid2, &mut status2, 0) };
    let kept = u32::from_ne_bytes(bytes2);

    println!("soak: relayed handle {kept:#x} from dead pid {pid2}, with retain:");
    assert_eq!(retain(kept), Ok(()), "retain failed");
    println!("  after retain:      usable = {}", read_event(kept).is_ok());
    trigger_sweeps(trigger, 512);
    assert!(
        read_event(kept).is_ok(),
        "retained object was reaped after its creator died"
    );
    assert!(event_set(kept).is_ok(), "retained object stopped working");
    println!("  after one sweep:   usable = true, signal ok");
    assert!(release(kept).is_ok(), "release failed");
    trigger_sweeps(trigger, 512);
    assert!(
        read_event(kept).is_err(),
        "object still alive after the last holder released it"
    );
    println!("  after release:     usable = false (reaped as expected)");

    let _ = close(trigger);
}
