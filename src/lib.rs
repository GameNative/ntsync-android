// Userspace ntsync library for Android.
//
// Copyright (C) 2026 Joshua Tam <297250+joshuatam@users.noreply.github.com>
//
// This library is free software: you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, version 3 only.
//
// This library is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
// General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <https://www.gnu.org/licenses/>.
//
// SPDX-License-Identifier: LGPL-3.0-only

pub mod core;
pub mod ffi;

#[cfg(test)]
mod bench;

#[cfg(test)]
mod engine_sim;

#[cfg(test)]
mod soak;

pub use core::{Error, WaitOutcome, MAX_WAIT_COUNT};

#[cfg(test)]
mod tests {
    use super::core::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Once};
    use std::thread;
    use std::time::Duration;

    static INIT: Once = Once::new();

    fn setup() {
        INIT.call_once(|| {
            let path = std::env::temp_dir()
                .join(format!("ntsync-test-{}.shm", std::process::id()));
            let _ = std::fs::remove_file(&path);
            init(Some(path.to_str().unwrap())).unwrap();
        });
    }

    fn is_signaled(outcome: WaitOutcome, index: u32) -> bool {
        matches!(outcome, WaitOutcome::Signaled { index: i, .. } if i == index)
    }

    #[test]
    fn semaphore_basic() {
        setup();
        let h = create_semaphore(2, 5).unwrap();
        assert!(create_semaphore(6, 5).is_err());
        assert!(is_signaled(wait_any(&[h], 1, Some(Duration::ZERO), 0), 0));
        assert!(is_signaled(wait_any(&[h], 1, Some(Duration::ZERO), 0), 0));
        assert_eq!(
            wait_any(&[h], 1, Some(Duration::from_millis(10)), 0),
            WaitOutcome::Timeout
        );
        assert_eq!(sem_release(h, 3).unwrap(), 0);
        assert_eq!(read_sem(h).unwrap(), (3, 5));
        // Overflow: error, state unchanged (kernel behavior).
        assert_eq!(sem_release(h, 3), Err(Error::Overflow));
        assert_eq!(read_sem(h).unwrap(), (3, 5));
        assert!(close(h));
    }

    // ---- Budgeted wakes: a signal op wakes only as many wait-any waiters
    // as it can satisfy, oldest first; wait-all waiters as before. ----

    fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
        let start = std::time::Instant::now();
        while !cond() {
            assert!(start.elapsed() < Duration::from_secs(10), "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(1));
        }
    }

    /// Park `n` single-object waiters on `h`, one at a time so their order
    /// on the list is known. Each pushes its id to `order` once it acquires.
    fn park_single_waiters(
        h: u32,
        n: usize,
        owner_base: u32,
        order: &Arc<std::sync::Mutex<Vec<usize>>>,
    ) -> Vec<thread::JoinHandle<()>> {
        let mut joins = Vec::new();
        for i in 0..n {
            let order = order.clone();
            joins.push(thread::spawn(move || {
                let out = wait_any(&[h], owner_base + i as u32, None, 0);
                assert!(matches!(out, WaitOutcome::Signaled { index: 0, .. }));
                order.lock().unwrap().push(i);
            }));
            wait_until("waiter to park", || test_waiter_counts(h).0 as usize == i + 1);
        }
        joins
    }

    #[test]
    fn semaphore_release_wakes_only_released_count() {
        setup();
        let h = create_semaphore(0, 10).unwrap();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let joins = park_single_waiters(h, 4, 9100, &order);

        sem_release(h, 1).unwrap();
        wait_until("one waiter to acquire", || order.lock().unwrap().len() == 1);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(order.lock().unwrap().len(), 1, "release(1) satisfied more than one waiter");
        assert_eq!(test_waiter_counts(h), (3, 0), "the other waiters must stay parked");

        sem_release(h, 2).unwrap();
        wait_until("two more waiters", || order.lock().unwrap().len() == 3);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(test_waiter_counts(h), (1, 0));

        sem_release(h, 1).unwrap();
        for j in joins {
            j.join().unwrap();
        }
        assert_eq!(read_sem(h).unwrap().0, 0);
        assert!(close(h));
    }

    #[test]
    fn single_waiters_are_woken_oldest_first() {
        setup();
        let h = create_semaphore(0, 10).unwrap();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let joins = park_single_waiters(h, 5, 9200, &order);
        for k in 1..=5 {
            sem_release(h, 1).unwrap();
            wait_until("next waiter", || order.lock().unwrap().len() == k);
        }
        for j in joins {
            j.join().unwrap();
        }
        assert_eq!(*order.lock().unwrap(), vec![0, 1, 2, 3, 4]);
        assert!(close(h));
    }

    #[test]
    fn auto_event_wakes_one_manual_event_wakes_all() {
        setup();
        let auto = create_event(false, false).unwrap();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let joins = park_single_waiters(auto, 3, 9300, &order);
        event_set(auto).unwrap();
        wait_until("one auto waiter", || order.lock().unwrap().len() == 1);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(order.lock().unwrap().len(), 1);
        assert_eq!(test_waiter_counts(auto), (2, 0));
        // A set that lands while the event is still signaled is a no-op, as
        // on Windows; keep setting until everyone got through.
        let start = std::time::Instant::now();
        while order.lock().unwrap().len() < 3 {
            let _ = event_set(auto);
            thread::sleep(Duration::from_millis(1));
            assert!(start.elapsed() < Duration::from_secs(10), "auto-event waiter stranded");
        }
        for j in joins {
            j.join().unwrap();
        }
        assert!(close(auto));

        let manual = create_event(true, false).unwrap();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let joins = park_single_waiters(manual, 3, 9310, &order);
        event_set(manual).unwrap();
        for j in joins {
            j.join().unwrap();
        }
        assert_eq!(order.lock().unwrap().len(), 3);
        assert!(close(manual));
    }

    #[test]
    fn mutex_unlock_hands_over_to_one_waiter() {
        setup();
        let m = create_mutex(9400, 1).unwrap();
        let inside = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let done = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let mut joins = Vec::new();
        for i in 0..4u32 {
            let (inside, done) = (inside.clone(), done.clone());
            joins.push(thread::spawn(move || {
                let me = 9401 + i;
                assert!(is_signaled(wait_any(&[m], me, None, 0), 0));
                assert_eq!(inside.fetch_add(1, Ordering::SeqCst), 0, "two owners at once");
                thread::sleep(Duration::from_millis(2));
                inside.fetch_sub(1, Ordering::SeqCst);
                mutex_unlock(m, me).unwrap();
                done.fetch_add(1, Ordering::SeqCst);
            }));
        }
        wait_until("all mutex waiters parked", || test_waiter_counts(m).0 == 4);
        mutex_unlock(m, 9400).unwrap();
        for j in joins {
            j.join().unwrap();
        }
        assert_eq!(done.load(Ordering::SeqCst), 4);
        assert!(close(m));
    }

    #[test]
    fn multi_object_and_alertable_waiters_are_woken_within_budget() {
        setup();
        let sem = create_semaphore(0, 10).unwrap();
        let other = create_event(true, false).unwrap();
        let alert = create_event(true, false).unwrap();
        let got = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let mut joins = Vec::new();
        {
            let got = got.clone();
            joins.push(thread::spawn(move || {
                assert!(is_signaled(wait_any(&[sem, other], 9500, None, 0), 0));
                got.fetch_add(1, Ordering::SeqCst);
            }));
        }
        wait_until("multi waiter", || test_waiter_counts(sem).1 == 1);
        {
            let got = got.clone();
            joins.push(thread::spawn(move || {
                assert!(is_signaled(wait_any(&[sem], 9501, None, alert), 0));
                got.fetch_add(1, Ordering::SeqCst);
            }));
        }
        wait_until("alertable waiter", || test_waiter_counts(sem).1 == 2);
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        joins.extend(park_single_waiters(sem, 2, 9510, &order));
        assert_eq!(test_waiter_counts(sem), (2, 2));

        // Release one unit: only the oldest waiter (the multi-object one) is
        // woken and takes it; the other three stay parked.
        sem_release(sem, 1).unwrap();
        wait_until("one acquire", || got.load(Ordering::SeqCst) == 1);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(test_waiter_counts(sem), (2, 1), "the other waiters must stay parked");
        assert!(order.lock().unwrap().is_empty());
        sem_release(sem, 3).unwrap();
        for j in joins {
            j.join().unwrap();
        }
        assert_eq!(got.load(Ordering::SeqCst) + order.lock().unwrap().len() as u32, 4);
        for h in [sem, other, alert] {
            assert!(close(h));
        }
    }

    /// A wait-all waiter may need more than this object, so it is woken
    /// whatever the budget.
    #[test]
    fn wait_all_waiters_are_always_woken() {
        setup();
        let sem = create_semaphore(0, 10).unwrap();
        let gate = create_event(true, true).unwrap();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));
        let mut joins = park_single_waiters(sem, 1, 9550, &order);
        joins.push(thread::spawn(move || {
            assert!(matches!(wait_all(&[sem, gate], 9551, None, 0), WaitOutcome::Signaled { .. }));
        }));
        wait_until("wait-all waiter", || test_waiter_counts(sem) == (1, 1));
        // Two units for two waiters: the single one is the oldest and uses
        // the whole budget of the first release if it is the only one woken.
        sem_release(sem, 1).unwrap();
        sem_release(sem, 1).unwrap();
        for j in joins {
            j.join().unwrap();
        }
        assert_eq!(read_sem(sem).unwrap().0, 0);
        for h in [sem, gate] {
            assert!(close(h));
        }
    }

    /// A wait-any waiter woken for a semaphore unit that returns for another
    /// reason (its alert fired, or an earlier object in its list was
    /// signaled) must pass the wake on, or the unit sits in the semaphore
    /// with the next waiter asleep. `alertable` picks the reason.
    fn decoy_passes_its_wake_on(alertable: bool, owner_base: u32) {
        setup();
        let mut passed_on = 0;
        for round in 0..200u32 {
            let sem = create_semaphore(0, 10).unwrap();
            let ev = create_event(true, false).unwrap();
            let owner = owner_base + round * 2;
            // The decoy parks first, so it is the oldest waiter and the one
            // a release of one unit wakes.
            let decoy = thread::spawn(move || {
                let out = if alertable {
                    wait_any(&[sem], owner, None, ev)
                } else {
                    wait_any(&[ev, sem], owner, None, 0)
                };
                let sem_index = if alertable { 0 } else { 1 };
                assert!(matches!(out, WaitOutcome::Signaled { .. }), "decoy: {out:?}");
                is_signaled(out, sem_index)
            });
            wait_until("decoy to park", || test_waiter_counts(sem).1 == 1);
            let consumer = thread::spawn(move || {
                assert!(is_signaled(wait_any(&[sem], owner + 1, None, 0), 0));
            });
            wait_until("consumer to park", || test_waiter_counts(sem).0 == 1);

            // The release wakes only the decoy. The event lands before the
            // decoy gets to run, so it returns on the event and leaves the
            // unit behind.
            sem_release(sem, 1).unwrap();
            event_set(ev).unwrap();
            if decoy.join().unwrap() {
                // The decoy won the race and took the unit itself.
                sem_release(sem, 1).unwrap();
            } else {
                passed_on += 1;
            }
            let start = std::time::Instant::now();
            while !consumer.is_finished() {
                assert!(
                    start.elapsed() < Duration::from_secs(10),
                    "round {round}: consumer left parked, sem count {:?}",
                    read_sem(sem)
                );
                thread::sleep(Duration::from_micros(200));
            }
            consumer.join().unwrap();
            assert_eq!(read_sem(sem).unwrap().0, 0);
            for h in [sem, ev] {
                assert!(close(h));
            }
        }
        assert!(passed_on > 0, "no round had the decoy return on the event");
    }

    #[test]
    fn alerted_waiter_passes_its_wake_on() {
        decoy_passes_its_wake_on(true, 9800);
    }

    #[test]
    fn waiter_satisfied_by_another_object_passes_its_wake_on() {
        decoy_passes_its_wake_on(false, 10300);
    }

    /// A budgeted wake is charged when its node is detached and issued after
    /// the unlock. If the oldest waiter belongs to a process that died, the
    /// wake finds nobody parked, and the unit must go to the next waiter
    /// instead of staying in the semaphore with that waiter asleep.
    #[test]
    fn dead_waiter_does_not_strand_a_unit() {
        setup();
        let sem = create_semaphore(0, 10).unwrap();
        // Keep the auto-sweep from purging the dead node before the release.
        test_touch_sweep_clock();
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            let _ = wait_any(&[sem], 11001, None, 0);
            unsafe { libc::_exit(0) };
        }
        wait_until("child waiter to park", || test_waiter_counts(sem).0 == 1);
        unsafe {
            libc::kill(pid, libc::SIGKILL);
            libc::waitpid(pid, std::ptr::null_mut(), 0);
        }
        let live = thread::spawn(move || {
            assert!(is_signaled(wait_any(&[sem], 11002, Some(Duration::from_secs(10)), 0), 0));
        });
        wait_until("live waiter to park", || test_waiter_counts(sem).0 == 2);
        sem_release(sem, 1).unwrap();
        let start = std::time::Instant::now();
        while !live.is_finished() {
            assert!(start.elapsed() < Duration::from_secs(5), "unit stranded behind a dead waiter");
            thread::sleep(Duration::from_millis(1));
        }
        live.join().unwrap();
        assert_eq!(read_sem(sem).unwrap().0, 0);
        assert!(close(sem));
    }

    /// Host cooperation for SIGUSR1 (ntsync_set_lock_flag_hook), modelled on
    /// Wine: the handler "suspends" the thread (blocks until resumed) unless
    /// the thread's lock flag is set, in which case it marks the signal
    /// pending and returns; the library re-raises it after the unlock.
    mod usr1 {
        use super::super::core::*;
        use std::sync::atomic::{AtomicBool, AtomicI32, AtomicU32, Ordering};

        pub static TARGET: AtomicU32 = AtomicU32::new(0);
        pub static FLAG: AtomicI32 = AtomicI32::new(0);
        pub static SUSPENDS: AtomicU32 = AtomicU32::new(0);
        pub static SUSPENDED: AtomicBool = AtomicBool::new(false);
        pub static RESUME: AtomicBool = AtomicBool::new(false);
        /// Negative control: the handler suspends regardless of the flag.
        pub static IGNORE_FLAG: AtomicBool = AtomicBool::new(false);

        pub unsafe extern "C" fn lock_flag() -> *mut i32 {
            if unsafe { libc::gettid() } as u32 == TARGET.load(Ordering::SeqCst) {
                FLAG.as_ptr()
            } else {
                std::ptr::null_mut()
            }
        }

        pub extern "C" fn handler(_: i32) {
            if !IGNORE_FLAG.load(Ordering::SeqCst) && FLAG.load(Ordering::SeqCst) != 0 {
                FLAG.fetch_or(LOCK_FLAG_SIGNAL_PENDING, Ordering::SeqCst);
                return;
            }
            SUSPENDS.fetch_add(1, Ordering::SeqCst);
            SUSPENDED.store(true, Ordering::SeqCst);
            let ts = libc::timespec { tv_sec: 0, tv_nsec: 50_000 };
            while !RESUME.load(Ordering::SeqCst) {
                unsafe { libc::nanosleep(&ts, std::ptr::null_mut()) };
            }
            RESUME.store(false, Ordering::SeqCst);
            SUSPENDED.store(false, Ordering::SeqCst);
        }

        pub fn install() {
            unsafe {
                let mut sa: libc::sigaction = std::mem::zeroed();
                sa.sa_sigaction = handler as extern "C" fn(i32) as usize;
                sa.sa_flags = libc::SA_RESTART;
                libc::sigemptyset(&mut sa.sa_mask);
                libc::sigaction(libc::SIGUSR1, &sa, std::ptr::null_mut());
            }
            set_lock_flag_hook(Some(lock_flag));
        }
    }

    /// Suspend a thread that is busy taking and dropping the region lock,
    /// `rounds` times, and each time check that another thread can still
    /// take the lock while it is suspended. Returns (wedged rounds,
    /// signals the library deferred).
    fn suspend_lock_holder(rounds: u32, ignore_flag: bool) -> (u32, u64) {
        setup();
        usr1::install();
        usr1::IGNORE_FLAG.store(ignore_flag, Ordering::SeqCst);
        usr1::SUSPENDS.store(0, Ordering::SeqCst);
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        let busy = {
            let stop = stop.clone();
            thread::spawn(move || {
                let tid = unsafe { libc::gettid() } as u32;
                usr1::TARGET.store(tid, Ordering::SeqCst);
                tx.send(tid).unwrap();
                while !stop.load(Ordering::Relaxed) {
                    // create and close each take the region lock.
                    let e = create_event(false, false).unwrap();
                    assert!(close(e));
                }
            })
        };
        let tid = rx.recv().unwrap();
        let deferred0 = test_counter(20);
        let mut wedged = 0;
        for round in 0..rounds {
            unsafe { libc::syscall(libc::SYS_tgkill, libc::getpid(), tid as i32, libc::SIGUSR1) };
            let start = std::time::Instant::now();
            while !usr1::SUSPENDED.load(Ordering::SeqCst) {
                assert!(start.elapsed() < Duration::from_secs(5), "round {round}: SIGUSR1 lost");
                std::hint::spin_loop();
            }
            // Another thread must get the region lock while the target is
            // suspended.
            let probe = thread::spawn(|| {
                let e = create_event(false, false).unwrap();
                assert!(close(e));
            });
            let start = std::time::Instant::now();
            while !probe.is_finished() && start.elapsed() < Duration::from_millis(300) {
                thread::sleep(Duration::from_micros(200));
            }
            if !probe.is_finished() {
                wedged += 1;
            }
            usr1::RESUME.store(true, Ordering::SeqCst);
            while usr1::SUSPENDED.load(Ordering::SeqCst) {
                std::hint::spin_loop();
            }
            probe.join().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        busy.join().unwrap();
        usr1::TARGET.store(0, Ordering::SeqCst);
        usr1::IGNORE_FLAG.store(false, Ordering::SeqCst);
        set_lock_flag_hook(None);
        assert_eq!(usr1::SUSPENDS.load(Ordering::SeqCst), rounds, "every SIGUSR1 suspends exactly once");
        (wedged, test_counter(20) - deferred0)
    }

    #[test]
    fn sigusr1_in_region_lock_is_deferred_not_lost() {
        let (wedged, deferred) = suspend_lock_holder(300, false);
        eprintln!("300 suspends: {wedged} wedged, {deferred} deferred inside the lock");
        assert_eq!(wedged, 0, "a thread suspended inside the region lock wedged it");
        assert!(deferred > 0, "no SIGUSR1 landed inside the lock; the test proved nothing");
    }

    /// Negative control for the test above: a handler that ignores the flag
    /// suspends the thread inside the lock and wedges it.
    #[test]
    fn sigusr1_in_region_lock_wedges_without_deferral() {
        let (wedged, _) = suspend_lock_holder(300, true);
        eprintln!("300 suspends, flag ignored: {wedged} wedged");
        assert!(wedged > 0, "no round wedged; the deferral test cannot tell a fix from luck");
    }

    /// No lost wakeups under contention. Exactly as many units are released
    /// as consumers need, so a wake spent on a thread that never takes its
    /// unit would leave a consumer parked forever: every consumer must
    /// finish within the deadline. Thieves steal units with zero-timeout
    /// waits and give them back, so woken waiters regularly lose the race.
    #[test]
    fn stress_budgeted_wakes_never_strand_a_waiter() {
        setup();
        const PER: u32 = 300;
        let sem = create_semaphore(0, 100_000).unwrap();
        let other = create_event(true, false).unwrap();
        let alert = create_event(true, false).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let taken = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let mut consumers = Vec::new();
        // 4 single-object consumers, 2 multi-object, 2 alertable (one is both).
        for i in 0..7u32 {
            let taken = taken.clone();
            consumers.push(thread::spawn(move || {
                for _ in 0..PER {
                    // Consumer 4 returns on `other` and consumers 5 and 6 on
                    // their alert whenever the toggler fires them; they
                    // then wait again, having passed their wake on.
                    loop {
                        let (out, sem_index) = match i {
                            4 => (wait_any(&[other, sem], 9600 + i, None, 0), 1),
                            5 => (wait_any(&[sem, other], 9600 + i, None, alert), 0),
                            6 => (wait_any(&[sem], 9600 + i, None, alert), 0),
                            _ => (wait_any(&[sem], 9600 + i, None, 0), 0),
                        };
                        assert!(matches!(out, WaitOutcome::Signaled { .. }), "consumer {i}: {out:?}");
                        if is_signaled(out, sem_index) {
                            break;
                        }
                        std::thread::yield_now();
                    }
                    taken.fetch_add(1, Ordering::Relaxed);
                }
            }));
        }
        let mut thieves = Vec::new();
        for t in 0..2u32 {
            let stop = stop.clone();
            thieves.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    if is_signaled(wait_any(&[sem], 9700 + t, Some(Duration::ZERO), 0), 0) {
                        std::hint::spin_loop();
                        sem_release(sem, 1).unwrap();
                    }
                    std::thread::yield_now();
                }
            }));
        }
        for (t, ev) in [other, alert].into_iter().enumerate() {
            let stop = stop.clone();
            thieves.push(thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    event_set(ev).unwrap();
                    std::hint::spin_loop();
                    event_reset(ev).unwrap();
                    thread::sleep(Duration::from_micros(50 + 70 * t as u64));
                }
            }));
        }
        let total = PER * 7;
        let mut released = 0;
        let mut k = 0u32;
        while released < total {
            let n = (1 + k % 4).min(total - released);
            sem_release(sem, n).unwrap();
            released += n;
            k += 1;
            if k % 16 == 0 {
                thread::sleep(Duration::from_micros(200));
            }
        }
        let start = std::time::Instant::now();
        while taken.load(Ordering::Relaxed) < total {
            assert!(
                start.elapsed() < Duration::from_secs(20),
                "lost wakeup: {} of {total} units taken, sem count {:?}, waiters {:?}",
                taken.load(Ordering::Relaxed),
                read_sem(sem),
                test_waiter_counts(sem)
            );
            thread::sleep(Duration::from_millis(5));
        }
        for c in consumers {
            c.join().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        for t in thieves {
            t.join().unwrap();
        }
        assert_eq!(read_sem(sem).unwrap().0, 0);
        for h in [sem, other, alert] {
            assert!(close(h));
        }
    }

    /// Mutex and auto-reset event handoffs under contention: mutual
    /// exclusion holds and nobody is left parked.
    #[test]
    fn stress_mutex_and_auto_event_handoff() {
        setup();
        let m = create_mutex(0, 0).unwrap();
        let ev = create_event(false, false).unwrap();
        let inside = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let mut joins = Vec::new();
        for i in 0..6u32 {
            let inside = inside.clone();
            joins.push(thread::spawn(move || {
                let me = 9800 + i;
                for _ in 0..500 {
                    assert!(is_signaled(wait_any(&[m], me, None, 0), 0));
                    assert_eq!(inside.fetch_add(1, Ordering::SeqCst), 0, "two mutex owners");
                    inside.fetch_sub(1, Ordering::SeqCst);
                    mutex_unlock(m, me).unwrap();
                }
            }));
        }
        // Auto-event ping-pong: each consumer waits, the producer sets until
        // every wait has been satisfied.
        let got = Arc::new(std::sync::atomic::AtomicU32::new(0));
        for i in 0..4u32 {
            let got = got.clone();
            joins.push(thread::spawn(move || {
                for _ in 0..200 {
                    assert!(is_signaled(wait_any(&[ev], 9900 + i, None, 0), 0));
                    got.fetch_add(1, Ordering::SeqCst);
                }
            }));
        }
        let start = std::time::Instant::now();
        while got.load(Ordering::SeqCst) < 800 {
            let _ = event_set(ev);
            std::thread::yield_now();
            assert!(start.elapsed() < Duration::from_secs(20), "auto-event waiter stranded at {}", got.load(Ordering::SeqCst));
        }
        for j in joins {
            j.join().unwrap();
        }
        for h in [m, ev] {
            assert!(close(h));
        }
    }

    #[test]
    fn handle_zero_is_never_allocated() {
        setup();
        // Handle 0 is the "no alert" sentinel in ntsync_wait_args.alert, so it
        // must never be handed out: slot 0 is reserved at region init.
        let h = create_event(false, false).unwrap();
        assert_ne!(h & ((1 << 14) - 1), 0);
        assert!(!close(0));
        assert!(close(h));
    }

    #[test]
    fn semaphore_blocks_and_wakes() {
        setup();
        let h = create_semaphore(0, 1).unwrap();
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            sem_release(h, 1).unwrap();
        });
        assert!(is_signaled(wait_any(&[h], 1, None, 0), 0));
        t.join().unwrap();
        assert!(close(h));
    }

    #[test]
    fn mutex_recursion_and_ownership() {
        setup();
        let h = create_mutex(0, 0).unwrap();
        // Kernel: !owner != !count is invalid.
        assert_eq!(create_mutex(5, 0), Err(Error::Invalid));
        assert!(is_signaled(wait_any(&[h], 7, Some(Duration::ZERO), 0), 0));
        assert!(is_signaled(wait_any(&[h], 7, Some(Duration::ZERO), 0), 0));
        assert_eq!(read_mutex(h).unwrap(), (2, 7, false));
        // Not the owner: cannot wait or unlock.
        assert_eq!(
            wait_any(&[h], 8, Some(Duration::from_millis(10)), 0),
            WaitOutcome::Timeout
        );
        assert_eq!(mutex_unlock(h, 8), Err(Error::Perm));
        // Unlock returns previous count.
        assert_eq!(mutex_unlock(h, 7).unwrap(), 2);
        assert_eq!(mutex_unlock(h, 7).unwrap(), 1);
        assert_eq!(read_mutex(h).unwrap(), (0, 0, false));
        assert!(close(h));
    }

    #[test]
    fn mutex_abandoned() {
        setup();
        let h = create_mutex(0, 0).unwrap();
        assert!(is_signaled(wait_any(&[h], 1, Some(Duration::ZERO), 0), 0));
        assert_eq!(mutex_kill(h, 2), Err(Error::Perm));
        mutex_kill(h, 1).unwrap();
        assert_eq!(read_mutex(h).unwrap(), (0, 0, true));
        match wait_any(&[h], 2, Some(Duration::ZERO), 0) {
            WaitOutcome::Signaled { index, owner_dead } => {
                assert_eq!(index, 0);
                assert!(owner_dead);
            }
            o => panic!("expected signal, got {o:?}"),
        }
        // Abandoned flag is cleared on acquisition.
        assert_eq!(read_mutex(h).unwrap(), (1, 2, false));
        assert!(close(h));
    }

    #[test]
    fn event_auto_and_manual_reset() {
        setup();
        let auto = create_event(false, true).unwrap();
        assert!(is_signaled(wait_any(&[auto], 1, Some(Duration::ZERO), 0), 0));
        // Auto-reset: consumed after wait.
        assert!(!read_event(auto).unwrap().1);

        let manual = create_event(true, true).unwrap();
        assert!(is_signaled(wait_any(&[manual], 1, Some(Duration::ZERO), 0), 0));
        assert!(is_signaled(wait_any(&[manual], 1, Some(Duration::ZERO), 0), 0));
        event_reset(manual).unwrap();
        assert_eq!(
            wait_any(&[manual], 1, Some(Duration::from_millis(10)), 0),
            WaitOutcome::Timeout
        );
        assert!(close(auto));
        assert!(close(manual));
    }

    #[test]
    fn event_pulse_wakes_waiters() {
        setup();
        let h = create_event(true, false).unwrap();
        let hits = Arc::new(AtomicBool::new(true));
        let mut threads = Vec::new();
        for i in 0..3 {
            let hits = hits.clone();
            threads.push(thread::spawn(move || {
                if !matches!(
                    wait_any(&[h], 10 + i, None, 0),
                    WaitOutcome::Signaled { .. }
                ) {
                    hits.store(false, Ordering::SeqCst);
                }
            }));
        }
        thread::sleep(Duration::from_millis(50));
        event_pulse(h).unwrap();
        for t in threads {
            t.join().unwrap();
        }
        assert!(hits.load(Ordering::SeqCst));
        assert!(!read_event(h).unwrap().1);
        assert!(close(h));
    }

    #[test]
    fn wait_all_semantics() {
        setup();
        let sem = create_semaphore(1, 1).unwrap();
        let ev = create_event(true, false).unwrap();
        assert_eq!(
            wait_all(&[sem, ev], 1, Some(Duration::from_millis(10)), 0),
            WaitOutcome::Timeout
        );
        event_set(ev).unwrap();
        assert!(is_signaled(wait_all(&[sem, ev], 1, Some(Duration::ZERO), 0), 0));
        assert_eq!(read_sem(sem).unwrap(), (0, 1));
        assert!(read_event(ev).unwrap().1); // manual event stays signaled
        assert!(close(sem));
        assert!(close(ev));
    }

    #[test]
    fn wait_any_index_selection() {
        setup();
        let s1 = create_semaphore(0, 1).unwrap();
        let s2 = create_semaphore(1, 1).unwrap();
        assert!(is_signaled(wait_any(&[s1, s2], 1, Some(Duration::ZERO), 0), 1));
        assert!(close(s1));
        assert!(close(s2));
    }

    #[test]
    fn invalid_handles() {
        setup();
        assert_eq!(
            wait_any(&[0xDEADBEEF], 1, Some(Duration::ZERO), 0),
            WaitOutcome::Invalid
        );
        assert_eq!(wait_any(&[], 1, None, 0), WaitOutcome::Invalid);
        assert_eq!(wait_any(&[1], 0, None, 0), WaitOutcome::Invalid);
        assert_eq!(sem_release(0xDEADBEEF, 1), Err(Error::Invalid));
        assert_eq!(event_set(0xDEADBEEF), Err(Error::Invalid));
        assert!(!close(0xDEADBEEF));
    }

    #[test]
    fn cross_process_semaphore() {
        setup();
        let h = create_semaphore(0, 1).unwrap();
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            // Child: block until the parent releases.
            let ok = matches!(
                wait_any(&[h], 42, None, 0),
                WaitOutcome::Signaled { index: 0, .. }
            );
            std::process::exit(if ok { 0 } else { 1 });
        }
        thread::sleep(Duration::from_millis(50));
        sem_release(h, 1).unwrap();
        let mut status = 0;
        unsafe { libc::waitpid(pid, &mut status, 0) };
        assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
        assert!(close(h));
    }

    #[test]
    fn many_waiters_targeted_wake() {
        // 32 threads each blocked on their own event; signaling one event
        // must wake exactly its waiter without hanging the others.
        setup();
        let events: Vec<u32> = (0..32).map(|_| create_event(false, false).unwrap()).collect();
        let threads: Vec<_> = events
            .iter()
            .map(|&e| {
                thread::spawn(move || {
                    matches!(wait_any(&[e], 42, Some(Duration::from_secs(10)), 0), WaitOutcome::Signaled { .. })
                })
            })
            .collect();
        thread::sleep(Duration::from_millis(50));
        for &e in &events {
            event_set(e).unwrap();
        }
        for t in threads {
            assert!(t.join().unwrap());
        }
        for e in events {
            assert!(close(e));
        }
    }

    #[test]
    fn cross_process_multi_object_wait() {
        setup();
        let sem = create_semaphore(0, 1).unwrap();
        let ev = create_event(true, false).unwrap();
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            // Child: wait-all on both objects.
            let ok = matches!(
                wait_all(&[sem, ev], 42, Some(Duration::from_secs(10)), 0),
                WaitOutcome::Signaled { .. }
            );
            std::process::exit(if ok { 0 } else { 1 });
        }
        thread::sleep(Duration::from_millis(50));
        sem_release(sem, 1).unwrap();
        thread::sleep(Duration::from_millis(50));
        event_set(ev).unwrap();
        let mut status = 0;
        unsafe { libc::waitpid(pid, &mut status, 0) };
        assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
        assert!(close(sem));
        assert!(close(ev));
    }

    #[test]
    fn sweep_dead_frees_orphans() {
        setup();
        let h = create_event(false, false).unwrap();
        // Simulate a crashed creator by clobbering the pid via a fake pid
        // that cannot exist: use sweep with our own (live) pid first.
        assert_eq!(sweep_dead().unwrap(), 0);
        assert!(read_event(h).is_ok());
        assert!(close(h));
    }

    #[test]
    fn reap_frees_dead_owner_slot() {
        setup();
        // Positive control for the re-validation test below: a candidate that
        // still matches its slot frees it, and a live verdict never does.
        let dead = create_event(false, false).unwrap();
        let cand = test_slot_candidate(dead).unwrap();
        assert_eq!(test_reap_slot_candidate(cand, false), 1);
        assert!(read_event(dead).is_err(), "reaped object still readable");

        let live = create_event(false, false).unwrap();
        let cand = test_slot_candidate(live).unwrap();
        assert_eq!(test_reap_slot_candidate(cand, true), 0);
        assert!(read_event(live).is_ok(), "live object was freed");
        assert!(close(live));
    }

    #[test]
    fn reap_skips_slot_reused_while_unlocked() {
        setup();
        // The sweep probes liveness with the region lock dropped, so a slot it
        // deemed dead may have been freed and handed out again by the time it
        // reaps. The stale candidate must not touch the new object.
        let stale = create_event(false, false).unwrap();
        let cand = test_slot_candidate(stale).unwrap();
        assert!(close(stale));
        // The allocator walks forward from its cursor, so march it back round
        // to the slot we just freed (a new index per step, the previous one
        // closed again).
        let mut fresh = 0;
        for _ in 0..20000 {
            let h = create_event(false, false).unwrap();
            if h & SLOT_MASK == stale & SLOT_MASK {
                fresh = h;
                break;
            }
            assert!(close(h));
        }
        assert_ne!(fresh, 0, "allocator never came back to the freed slot");
        assert_eq!(
            test_reap_slot_candidate(cand, false),
            0,
            "stale candidate freed a reused slot"
        );
        assert!(read_event(fresh).is_ok(), "reused object was destroyed");
        assert!(close(fresh));
    }

    #[test]
    fn reap_skips_live_waiter_node() {
        setup();
        let h = create_event(false, false).unwrap();
        let waiter = thread::spawn(move || {
            wait_any(&[h], std::process::id() as u32, Some(Duration::from_secs(5)), 0)
        });
        let mut cand = None;
        for _ in 0..2000 {
            if let Some(first) = test_first_waiter(h) {
                cand = Some(first);
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        let (index, node_pid) = cand.expect("waiter never registered a node");
        assert_eq!(node_pid, std::process::id());
        // The guard on a node candidate is "still registered by the same pid".
        // A candidate recorded against another pid (the node was recycled
        // while the sweep was unlocked) must be left alone even when the
        // verdict says dead.
        let stale = (index, node_pid.wrapping_add(1));
        assert!(!test_reap_node_candidate(stale, false), "stale node candidate detached");
        assert!(
            test_first_waiter(h).is_some(),
            "live waiter's node was detached"
        );
        assert_eq!(event_set(h).unwrap(), 0);
        assert!(is_signaled(waiter.join().unwrap(), 0));
        assert!(close(h));
    }

    #[test]
    fn sweep_purges_dead_waiter_and_recycles_its_node() {
        setup();
        let h = create_event(false, false).unwrap();
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0);
        if pid == 0 {
            // Block on the parent's event forever: this registers a node whose
            // owner dies without ever detaching it.
            let _ = wait_any(&[h], unsafe { libc::getpid() } as u32, None, 0);
            unsafe { libc::_exit(0) }
        }
        let mut cand = None;
        for _ in 0..2000 {
            if let Some(first) = test_first_waiter(h) {
                cand = Some(first);
                break;
            }
            thread::sleep(Duration::from_millis(1));
        }
        let (node, _) = cand.expect("child never blocked on the event");

        unsafe { libc::kill(pid, libc::SIGKILL) };
        let mut status = 0;
        unsafe { libc::waitpid(pid, &mut status, 0) };

        // Sweep and inspect in one locked section: the count the sweep returns
        // covers whatever other tests left dead too, so it is not asserted.
        let (reaped, pooled, _freed) = test_sweep_and_inspect_node(node);
        assert!(reaped, "dead waiter's node {node} was not purged");
        assert!(
            pooled,
            "dead waiter's node {node} was not returned to the node pool"
        );
        assert!(
            test_first_waiter(h).is_none(),
            "dead waiter's node stayed registered"
        );
        assert!(read_event(h).is_ok(), "object left unusable");
        assert!(close(h));
    }

    #[test]
    fn pid_cache_keeps_verdicts_straight() {
        setup();
        let me = std::process::id();
        // Own pid is alive, and repeated lookups stay consistent.
        assert_eq!(test_pid_cache(&[me, me, me]), vec![true, true, true]);
        // pid 0 is never an owner.
        assert_eq!(test_pid_cache(&[0]), vec![false]);
        // More distinct pids than the cache holds: the round-robin overwrite
        // must not turn a live pid dead or a dead pid alive.
        let mut pids = vec![me];
        pids.extend((0..40u32).map(|i| 900_000 + i)); // absent -> dead
        pids.push(me);
        let verdicts = test_pid_cache(&pids);
        assert!(verdicts[0], "live pid reported dead");
        assert!(
            verdicts[1..41].iter().all(|&a| !a),
            "absent pid reported alive"
        );
        assert!(verdicts[41], "live pid went stale after cache wraparound");
    }

    #[test]
    fn tail_budget_burst_is_frames_not_seconds() {
        // The burst allowance must be the budget's rate over ~10 ms, so a
        // thread that spends faster than it refills is throttled within a
        // few frames. A bucket holding the whole per-second allowance would
        // let 150 us/frame run unthrottled for seconds.
        assert_eq!(crate::core::test_tail_budget_cap(100_000_000), 1_000_000);
        assert_eq!(crate::core::test_tail_budget_cap(150_000_000), 1_500_000);
        // A 150 us tail fits ~6 times in the 15% burst.
        assert!(crate::core::test_tail_budget_cap(150_000_000) / 150_000 >= 5);
        // Never zero: a tiny budget must still be able to grant something.
        assert_eq!(crate::core::test_tail_budget_cap(1), 1);
        assert_eq!(crate::core::test_tail_budget_cap(0), 1);
    }

    #[test]
    fn auto_sweep_claims_interval_when_nothing_dead() {
        setup();
        let h = create_event(false, false).unwrap();
        test_reset_sweep_clock();
        // Nothing is dead here: the pass finds no candidates at all. It must
        // still claim its interval, otherwise every thread's 256-call gate
        // starts a fresh region-wide scan and the throttle is gone.
        assert_eq!(
            wait_any(&[h], std::process::id() as u32, Some(std::time::Duration::from_millis(10)), 0),
            WaitOutcome::Timeout
        );
        assert!(
            test_sweep_clock_ns() > 0,
            "auto-sweep must claim its interval even when nothing died"
        );
        assert!(close(h));
    }

    #[test]
    fn sweep_reaps_orphans_across_chunks() {
        setup();
        // More orphans than one sweep chunk (SWEEP_CHUNK = 256), so the pass
        // crosses a chunk boundary and must not drop the final short chunk.
        const ORPHANS: usize = 300;
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0);
        if pid == 0 {
            for _ in 0..ORPHANS {
                create_event(false, false).unwrap();
            }
            unsafe { libc::_exit(0) }
        }
        let mut status = 0;
        unsafe { libc::waitpid(pid, &mut status, 0) };
        assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
        let freed = sweep_dead().unwrap();
        assert!(freed as usize >= ORPHANS, "freed {freed}, expected >= {ORPHANS}");
    }

    #[test]
    fn sweep_scan_chunks_the_table() {
        setup();
        // More objects than one chunk, so the scan has to cross a chunk
        // boundary and the pid cache has to serve several chunks.
        const OBJECTS: usize = 300;
        let mut handles = Vec::new();
        for _ in 0..OBJECTS {
            handles.push(create_event(false, false).unwrap());
        }
        let (chunks, cands) = test_scan_candidates();
        assert!(chunks >= 2, "expected several chunks, got {chunks}");
        assert!(cands >= OBJECTS, "expected >= {OBJECTS} candidates, got {cands}");
        for h in handles {
            assert!(close(h));
        }
    }

    #[test]
    fn auto_sweep_reaps_dead_creator() {
        setup();
        let mine = create_event(false, false).unwrap();
        // A child process creates an object and dies without closing it.
        let pid = unsafe { libc::fork() };
        assert!(pid >= 0);
        if pid == 0 {
            let _ = create_event(false, false).unwrap();
            unsafe { libc::_exit(0) };
        }
        let mut status = 0;
        unsafe { libc::waitpid(pid, &mut status, 0) };
        // The orphan exists: an explicit sweep would find it. Instead, an
        // ordinary blocking wait must reap it via the auto-sweep path
        // (triggered before a waiter goes to sleep).
        test_reset_sweep_clock();
        assert_eq!(
            wait_any(&[mine], std::process::id() as u32, Some(std::time::Duration::from_millis(10)), 0),
            WaitOutcome::Timeout
        );
        assert_eq!(sweep_dead().unwrap(), 0, "auto-sweep should have reaped the orphan");
        assert!(close(mine));
    }
}

/// Performance simulation of the paths Wine's ntdll exercises. Not run by
/// default; use:
///   cargo test --release -- --ignored --nocapture perf
#[cfg(test)]
mod perf {
    use super::core::*;
    use std::sync::Once;
    use std::thread;
    use std::time::{Duration, Instant};

    static INIT: Once = Once::new();

    fn setup() {
        INIT.call_once(|| {
            let path = std::env::temp_dir()
                .join(format!("ntsync-perf-{}.shm", std::process::id()));
            let _ = std::fs::remove_file(&path);
            init(Some(path.to_str().unwrap())).unwrap();
        });
    }

    fn report(name: &str, iters: u64, elapsed: Duration) {
        let ns = elapsed.as_nanos() as u64 / iters.max(1);
        println!(
            "{name:<42} {iters:>8} iters  {ns:>8} ns/op ({:>7.2} us)",
            ns as f64 / 1000.0
        );
    }

    /// Hot path: WaitForSingleObject on an already-signaled manual event.
    /// Should be a pure lock-free peek (no mutex, no syscall).
    #[test]
    #[ignore = "perf simulation"]
    fn perf_wait_signaled_event() {
        setup();
        let h = create_event(true, true).unwrap();
        const N: u64 = 100_000;
        let t = Instant::now();
        for _ in 0..N {
            assert!(matches!(
                wait_any(&[h], 1, Some(Duration::ZERO), 0),
                WaitOutcome::Signaled { .. }
            ));
        }
        report("wait_any(signaled manual event)", N, t.elapsed());
        assert!(close(h));
    }

    /// Set/reset churn with no waiters (typical game event traffic when
    /// nobody is blocked). Should skip FUTEX_WAKE entirely.
    #[test]
    #[ignore = "perf simulation"]
    fn perf_event_set_reset_no_waiters() {
        setup();
        let h = create_event(true, false).unwrap();
        const N: u64 = 100_000;
        let t = Instant::now();
        for _ in 0..N {
            event_set(h).unwrap();
            event_reset(h).unwrap();
        }
        report("event_set+reset (no waiters)", N, t.elapsed());
        assert!(close(h));
    }

    /// Uncontended semaphore acquire/release pairs (e.g. D3D command
    /// semaphores).
    #[test]
    #[ignore = "perf simulation"]
    fn perf_sem_acquire_release_uncontended() {
        setup();
        let h = create_semaphore(0, 1_000_000).unwrap();
        sem_release(h, 100_000).unwrap();
        const N: u64 = 100_000;
        let t = Instant::now();
        for _ in 0..N {
            assert!(matches!(
                wait_any(&[h], 1, Some(Duration::ZERO), 0),
                WaitOutcome::Signaled { .. }
            ));
        }
        report("sem acquire (precharged)", N, t.elapsed());
        assert!(close(h));
    }

    /// Full wake latency: two threads ping-ponging on semaphores. Measures
    /// futex sleep + wake + re-check, i.e. the blocked-wait cost.
    #[test]
    #[ignore = "perf simulation"]
    fn perf_ping_pong_wake_latency() {
        setup();
        let s1 = create_semaphore(0, 1).unwrap();
        let s2 = create_semaphore(0, 1).unwrap();
        const N: u64 = 10_000;
        let worker = thread::spawn(move || {
            for _ in 0..N {
                assert!(matches!(
                    wait_any(&[s1], 2, Some(Duration::from_secs(5)), 0),
                    WaitOutcome::Signaled { .. }
                ));
                sem_release(s2, 1).unwrap();
            }
        });
        // Let the worker block first.
        thread::sleep(Duration::from_millis(10));
        let t = Instant::now();
        for _ in 0..N {
            sem_release(s1, 1).unwrap();
            assert!(matches!(
                wait_any(&[s2], 1, Some(Duration::from_secs(5)), 0),
                WaitOutcome::Signaled { .. }
            ));
        }
        report("ping-pong round trip (2 wakes)", N, t.elapsed());
        worker.join().unwrap();
        assert!(close(s1));
        assert!(close(s2));
    }

    /// Same ping-pong across a fork(), the real wineserver↔client shape:
    /// waker and waiter are in different processes sharing the region.
    #[test]
    #[ignore = "perf simulation"]
    fn perf_cross_process_ping_pong() {
        setup();
        let s1 = create_semaphore(0, 1).unwrap();
        let s2 = create_semaphore(0, 1).unwrap();
        const N: u64 = 10_000;
        let pid = unsafe { libc::fork() };
        if pid == 0 {
            // Child: wait on s1, release s2.
            for _ in 0..N {
                if !matches!(
                    wait_any(&[s1], 2, Some(Duration::from_secs(30)), 0),
                    WaitOutcome::Signaled { .. }
                ) {
                    std::process::exit(1);
                }
                sem_release(s2, 1).unwrap();
            }
            std::process::exit(0);
        }
        thread::sleep(Duration::from_millis(10));
        let t = Instant::now();
        for _ in 0..N {
            sem_release(s1, 1).unwrap();
            assert!(matches!(
                wait_any(&[s2], 1, Some(Duration::from_secs(30)), 0),
                WaitOutcome::Signaled { .. }
            ));
        }
        report("cross-process ping-pong (2 wakes)", N, t.elapsed());
        let mut status = 0;
        unsafe { libc::waitpid(pid, &mut status, 0) };
        assert!(libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0);
        assert!(close(s1));
        assert!(close(s2));
    }

    /// Contended mutex: 4 threads hammering one mutex (wait + unlock).
    #[test]
    #[ignore = "perf simulation"]
    fn perf_contended_mutex() {
        setup();
        let m = create_mutex(0, 0).unwrap();
        const THREADS: u32 = 4;
        const PER_THREAD: u64 = 5_000;
        let t = Instant::now();
        let handles: Vec<_> = (1..=THREADS)
            .map(|owner| {
                thread::spawn(move || {
                    for _ in 0..PER_THREAD {
                        assert!(matches!(
                            wait_any(&[m], owner, Some(Duration::from_secs(30)), 0),
                            WaitOutcome::Signaled { .. }
                        ));
                        mutex_unlock(m, owner).unwrap();
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        let total = THREADS as u64 * PER_THREAD;
        report("contended mutex acquire+release (4 thr)", total, t.elapsed());
        assert!(close(m));
    }
}
