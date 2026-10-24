// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (C) 2026 Joshua Tam <297250+joshuatam@users.noreply.github.com>

//! Userspace reimplementation of the Linux ntsync driver semantics
//! (drivers/misc/ntsync.c): NT semaphores, mutexes, and events.
//!
//! Cross-process design using Android/Linux NDK primitives, tuned for
//! Proton/Wine workloads where dozens of threads hammer sync objects:
//!
//! - All objects live in a fixed-size table in a file-backed shared mapping
//!   (opened by every process at the same path), mirroring the kernel's
//!   global object table.
//! - Each object's mutable state is packed into a single atomic u64, so
//!   try-acquires and signal operations (sem_release, event_set/reset,
//!   mutex_unlock, ...) are lock-free single-word CAS loops. Satisfied
//!   waits complete without taking any lock.
//! - A process-shared *robust* pthread mutex serializes only waiter-list
//!   registration and wake walks (and creation/close). If a process dies
//!   holding it, the next locker gets EOWNERDEAD and marks it consistent.
//! - Wakeups are per-object like the kernel's wait queues: each object keeps
//!   an intrusive list of registered waiters, and each waiter sleeps with
//!   futex(FUTEX_WAIT) on its own word in the shared waiter pool. A state
//!   change walks only that object's list (mirroring wake_up(&obj->q) +
//!   try_wake_any_obj/try_wake_all re-check). Registration and wake walks
//!   both happen under the region lock, and waiters re-check under it
//!   before sleeping, so no wakeup can be missed.
//!
//! Divergences from the kernel: closing an object that other threads are
//! waiting on fails those waits with EINVAL (the kernel keeps the object
//! alive via fd references); WAIT_ALL acquisition is a per-object CAS
//! sequence with rollback rather than a single atomic commit under
//! wait_all_lock; object cleanup after a process crash requires
//! ntsync_sweep_dead().

use std::ffi::CString;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;
use std::time::Duration;

pub const MAX_WAIT_COUNT: usize = 64;

const MAGIC: u64 = 0x6E7473796E635F75; // "ntsyc_u"
const VERSION: u32 = 10;
const SLOT_COUNT: u32 = 16384;
const NODE_COUNT: u32 = 8192;
const INDEX_BITS: u32 = 14; // log2(SLOT_COUNT)
const INDEX_MASK: u32 = (1 << INDEX_BITS) - 1;
const NIL: u32 = u32::MAX;

/// Resolve the region path: explicit argument, else $NTSYNC_SHM, else
/// $TMPDIR/ntsync_userspace.shm (the layout version is inserted before the
/// ".shm" extension: ntsync_userspace.vN.shm).
/// NTSYNC_SHM exists because in containerized
/// setups (e.g. GameNative) wineserver and game processes run with different
/// TMPDIRs and would otherwise mmap different files and never share objects.
fn resolve_path(path: Option<&str>) -> Result<String, i32> {
    if let Some(p) = path {
        return Ok(p.into());
    }
    if let Ok(shm) = std::env::var("NTSYNC_SHM") {
        if !shm.is_empty() {
            return Ok(shm);
        }
    }
    match std::env::var("TMPDIR") {
        Ok(tmp) if !tmp.is_empty() => Ok(format!("{tmp}/ntsync_userspace.shm")),
        _ => Err(libc::EINVAL),
    }
}

const SLOT_FREE: u32 = 0;
const SLOT_USED: u32 = 1;

/// Holder pids a slot tracks. An object relayed to more processes than this
/// gets the overflow bit and is never reaped automatically, which is the safe
/// direction: a leaked object costs memory, a wrongly freed one costs a game.
const MAX_HOLDERS: usize = 8;
const HOLDER_OVERFLOW: u32 = 0x8000_0000;

const TYPE_SEM: u32 = 1;
const TYPE_MUTEX: u32 = 2;
const TYPE_EVENT: u32 = 3;
/// Slot 0 is permanently reserved: handle 0 is the "no alert" sentinel in
/// ntsync_wait_args.alert, so it must never be handed out as a real object.
const TYPE_RESERVED: u32 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// Bad handle, wrong object type, or invalid argument (-EINVAL).
    Invalid,
    /// Caller is not the mutex owner (-EPERM).
    Perm,
    /// Semaphore count would exceed its maximum (-EOVERFLOW).
    Overflow,
    /// Region could not be opened/created/mapped (-errno).
    Init(i32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaitOutcome {
    /// Object at `index` satisfied the wait; `owner_dead` is set when an
    /// abandoned mutex was acquired (kernel: -EOWNERDEAD).
    Signaled { index: u32, owner_dead: bool },
    /// Timed out (kernel: -ETIMEDOUT).
    Timeout,
    /// Bad arguments, or an object was closed mid-wait (kernel: -EINVAL).
    Invalid,
}

/// One object slot in the shared table. `state`/`generation`/`obj_type`/`pid`
/// change only under the region lock; `w0` and `d` are accessed atomically
/// at all times (lock-free on the hot paths).
///
/// w0 packing by type:
///   sem:   hi32 = max,    lo32 = count
///   mutex: hi32 = count (31 bits) | ownerdead<<31, lo32 = owner
///   event: hi32 = manual, lo32 = signaled
#[repr(C)]
#[derive(Clone, Copy)]
struct Slot {
    state: u32,     // SLOT_FREE / SLOT_USED
    generation: u32,
    obj_type: u32,  // TYPE_*
    pid: u32,       // creator pid, for ntsync_sweep_dead
    waiter_head: u32, // head of the per-object waiter list (node index, NIL)
    holders_len: u32, // pids in `holders`; HOLDER_OVERFLOW if it spilled
    w0: u64,        // packed object state (atomic)
    d: u64,         // event: pulse_seq (atomic)
    /// Processes that hold this object besides its creator. wineserver relays
    /// a handle to another process when it hands the object on, and userspace
    /// has no fd-close hook on process death, so the only way to know a holder
    /// is gone is to remember who it was. The sweep reaps an object only when
    /// the creator AND every holder are dead.
    holders: [u32; MAX_HOLDERS],
}

const SLOT_SIZE: usize = std::mem::size_of::<Slot>();

/// One waiter node in the shared pool. A thread waiting on N objects
/// allocates N nodes and registers one on each object's waiter list;
/// it sleeps on nodes[0].seq ("head node"). Signalers detach the node
/// from their object's list, bump the head node's seq, and FUTEX_WAKE it.
/// Accessed only under the region lock, except `seq` during futex waits.
#[repr(C)]
struct WaitNode {
    seq: AtomicU32,     // futex word (only the head node's is waited on)
    registered: u32,    // 1 while linked on `obj`'s waiter list
    obj: u32,           // object slot index this node is registered on
    wait_word: u32,     // head node index whose `seq` is the futex word
    obj_prev: u32,      // per-object list links (doubly linked, O(1) detach)
    obj_next: u32,
    waiter_next: u32,   // next node of the same waiter / free-list link
    pid: u32,           // owner pid, for ntsync_sweep_dead purging
    /// CLOCK_MONOTONIC ns when the head node was last woken (diagnostics;
    /// written by the waker before bumping `seq`, read by the waiter after
    /// futex wake while it still owns the node).
    wake_ts: std::sync::atomic::AtomicU64,
}

/// CLOCK_MONOTONIC in nanoseconds.
fn monotonic_ns() -> u64 {
    let mut ts = unsafe { std::mem::zeroed::<libc::timespec>() };
    unsafe { libc::clock_gettime(libc::CLOCK_MONOTONIC, &mut ts) };
    (ts.tv_sec as u64) * 1_000_000_000 + ts.tv_nsec as u64
}

const NODE_SIZE: usize = std::mem::size_of::<WaitNode>();

#[repr(C)]
struct Header {
    magic: u64,
    version: u32,
    capacity: u32,
    node_capacity: u32,
    /// Free-list head for the waiter node pool (linked via waiter_next).
    node_free_head: u32,
    /// Lazy pool: index of the first never-handed-out node. Nodes below
    /// this cursor are either in use or on the free list; nodes at or above
    /// it have never been touched, so their pages stay untouched (sparse
    /// zero pages) until actually needed. Caller holds the lock.
    node_fresh: u32,
    /// Allocation cursor.
    next_scan: u32,
    /// CLOCK_MONOTONIC ns of the last auto-sweep completion (atomic).
    sweep_ns: u64,
    /// Debug: which thread currently holds `lock` (0 = none). Written after
    /// acquiring, cleared before releasing; only meaningful while locked.
    lock_owner_pid: u32,
    lock_owner_tid: u32,
    lock: libc::pthread_mutex_t,
    // Slot array follows immediately, then the waiter node pool.
}

const HEADER_SIZE: usize = std::mem::size_of::<Header>();

// Bionic exports the robust process-shared pthread mutex API (API 28+) but
// the libc crate does not expose it for Android, and the NDK stub libc.so
// omits both symbols entirely, which breaks consumers linking with
// --no-allow-shlib-undefined. Resolve them via dlsym at runtime instead.
#[cfg(target_os = "android")]
mod robust {
    use std::sync::OnceLock;

    type ConsistentFn = unsafe extern "C" fn(*mut libc::pthread_mutex_t) -> i32;
    type SetrobustFn = unsafe extern "C" fn(*mut libc::pthread_mutexattr_t, i32) -> i32;

    // Fallbacks if the symbols are somehow missing: without robust mutexes the
    // kernel never reports EOWNERDEAD, so no-op consistent/setrobust is safe.
    unsafe extern "C" fn consistent_stub(_: *mut libc::pthread_mutex_t) -> i32 { 0 }
    unsafe extern "C" fn setrobust_stub(_: *mut libc::pthread_mutexattr_t, _: i32) -> i32 { 0 }

    static FUNCS: OnceLock<(ConsistentFn, SetrobustFn)> = OnceLock::new();

    fn funcs() -> (ConsistentFn, SetrobustFn) {
        *FUNCS.get_or_init(|| unsafe {
            let handle = libc::dlopen(b"libc.so\0".as_ptr().cast(), libc::RTLD_NOW);
            let lookup = |name: &[u8]| -> *mut libc::c_void {
                if handle.is_null() { return std::ptr::null_mut(); }
                libc::dlsym(handle, name.as_ptr().cast())
            };
            let consistent = lookup(b"pthread_mutex_consistent\0");
            let setrobust = lookup(b"pthread_mutexattr_setrobust\0");
            (
                if consistent.is_null() { consistent_stub } else { std::mem::transmute::<*mut libc::c_void, ConsistentFn>(consistent) },
                if setrobust.is_null() { setrobust_stub } else { std::mem::transmute::<*mut libc::c_void, SetrobustFn>(setrobust) },
            )
        })
    }

    pub unsafe fn pthread_mutex_consistent(lock: *mut libc::pthread_mutex_t) -> i32 {
        funcs().0(lock)
    }

    pub unsafe fn pthread_mutexattr_setrobust(attr: *mut libc::pthread_mutexattr_t, robust: i32) -> i32 {
        funcs().1(attr, robust)
    }
}
#[cfg(target_os = "android")]
use robust::{pthread_mutex_consistent, pthread_mutexattr_setrobust};

#[cfg(target_os = "android")]
const PTHREAD_PROCESS_SHARED: i32 = 1;
#[cfg(target_os = "android")]
const PTHREAD_MUTEX_ROBUST: i32 = 1;

#[cfg(not(target_os = "android"))]
use libc::{
    pthread_mutex_consistent, pthread_mutexattr_setrobust, PTHREAD_MUTEX_ROBUST,
    PTHREAD_PROCESS_SHARED,
};

fn errno() -> i32 {
    #[cfg(target_os = "android")]
    unsafe {
        *libc::__errno()
    }
    #[cfg(not(target_os = "android"))]
    unsafe {
        *libc::__errno_location()
    }
}

// Lock-free accessors for the packed per-object state words.
#[inline]
fn ld32(p: &u32) -> u32 {
    unsafe { (&*(p as *const u32 as *const AtomicU32)).load(Ordering::Acquire) }
}
#[inline]
fn ld64(p: &u64) -> u64 {
    unsafe { (&*(p as *const u64 as *const AtomicU64)).load(Ordering::Acquire) }
}
#[inline]
fn st64(p: &u64, v: u64) {
    unsafe { (&*(p as *const u64 as *const AtomicU64)).store(v, Ordering::Release) }
}
#[inline]
fn cas64(p: &u64, old: u64, new: u64) -> Result<u64, u64> {
    let r = unsafe {
        (&*(p as *const u64 as *const AtomicU64)).compare_exchange(
            old,
            new,
            Ordering::AcqRel,
            Ordering::Acquire,
        )
    };
    if r.is_err() {
        stat_bump(6); // cas_retries
    }
    r
}

// w0 packing helpers.
#[inline]
fn pack_sem(count: u32, max: u32) -> u64 {
    ((max as u64) << 32) | count as u64
}
#[inline]
fn pack_mutex(owner: u32, count: u32, ownerdead: bool) -> u64 {
    (((count as u64) & 0x7fff_ffff) | ((ownerdead as u64) << 31)) << 32 | owner as u64
}
#[inline]
fn pack_event(signaled: bool, manual: bool) -> u64 {
    ((manual as u64) << 32) | signaled as u64
}

fn nodes_offset() -> usize {
    HEADER_SIZE + SLOT_COUNT as usize * SLOT_SIZE
}

fn region_size() -> usize {
    nodes_offset() + NODE_COUNT as usize * NODE_SIZE
}

struct Region {
    base: *mut u8,
}

unsafe impl Send for Region {}
unsafe impl Sync for Region {}

impl Region {
    fn header(&self) -> &Header {
        unsafe { &*(self.base as *const Header) }
    }

    fn slot_ptr(&self, index: u32) -> *mut Slot {
        unsafe { self.base.add(HEADER_SIZE + index as usize * SLOT_SIZE) as *mut Slot }
    }

    fn node_ptr(&self, index: u32) -> *mut WaitNode {
        unsafe { self.base.add(nodes_offset() + index as usize * NODE_SIZE) as *mut WaitNode }
    }

    /// Get a used slot matching a handle's index+generation.
    /// Lock-free safe: state/generation only change under the region lock
    /// and are read here with acquire loads.
    fn slot(&self, handle: u32) -> Option<*mut Slot> {
        let index = handle & INDEX_MASK;
        let generation = handle >> INDEX_BITS;
        if index >= SLOT_COUNT {
            return None;
        }
        let slot = self.slot_ptr(index);
        let s = unsafe { &*slot };
        if ld32(&s.state) == SLOT_USED && ld32(&s.generation) == generation {
            Some(slot)
        } else {
            None
        }
    }

    fn lock_guard(&self) -> Result<RegionGuard<'_>, i32> {
        let lock = &self.header().lock as *const libc::pthread_mutex_t as *mut _;
        let debug = debug_enabled();
        let mut waited = std::time::Duration::ZERO;
        // Wine suspends threads with SIGUSR1; if it lands while we hold the
        // shared region mutex the holder stays parked and every process
        // (including wineserver) wedges on the lock. Defer it until release.
        // Consumers without a SIGUSR1 handler can set
        // NTSYNC_NO_SIGUSR1_BLOCK=1 to skip the two pthread_sigmask
        // syscalls per lock acquisition.
        let old_mask = if sigusr1_block_enabled() {
            let mut old_mask = unsafe { std::mem::zeroed::<libc::sigset_t>() };
            let mut block = unsafe { std::mem::zeroed::<libc::sigset_t>() };
            unsafe {
                libc::sigemptyset(&mut block);
                libc::sigaddset(&mut block, libc::SIGUSR1);
                libc::pthread_sigmask(libc::SIG_BLOCK, &block, &mut old_mask);
            }
            Some(old_mask)
        } else {
            None
        };
        let mut tries = 0u32;
        loop {
            tries += 1;
            let ret = if debug {
                // Poll with a timeout so a stuck holder can be logged.
                let mut ts = unsafe { std::mem::zeroed::<libc::timespec>() };
                unsafe { libc::clock_gettime(libc::CLOCK_REALTIME, &mut ts) };
                ts.tv_sec += 5;
                unsafe { libc::pthread_mutex_timedlock(lock, &ts) }
            } else {
                unsafe { libc::pthread_mutex_lock(lock) }
            };
            if ret == 0 || ret == libc::EOWNERDEAD {
                stat_bump(4); // lock_acq
                if tries > 1 {
                    stat_bump(5); // lock_contended
                }
                if ret == libc::EOWNERDEAD {
                    // Previous holder died mid-operation; the object states are
                    // still structurally valid because every op is a small,
                    // single-pass mutation.
                    unsafe { pthread_mutex_consistent(lock) };
                }
                if debug {
                    let h = self.header() as *const Header as *mut Header;
                    unsafe {
                        (*h).lock_owner_pid = std::process::id();
                        (*h).lock_owner_tid = current_tid();
                    }
                }
                return Ok(RegionGuard { region: self, old_mask });
            }
            if ret == libc::ETIMEDOUT {
                waited += std::time::Duration::from_secs(5);
                let h = self.header();
                debug_log(&format!(
                    "region lock blocked {:?} (pid {} tid {}): held by pid {} tid {}",
                    waited,
                    std::process::id(),
                    current_tid(),
                    h.lock_owner_pid,
                    h.lock_owner_tid,
                ));
                continue;
            }
            if ret != libc::EINTR {
                if let Some(old_mask) = old_mask {
                    unsafe {
                        libc::pthread_sigmask(libc::SIG_SETMASK, &old_mask, std::ptr::null_mut())
                    };
                }
                return Err(-ret);
            }
        }
    }
}

struct RegionGuard<'a> {
    region: &'a Region,
    old_mask: Option<libc::sigset_t>,
}

impl Drop for RegionGuard<'_> {
    fn drop(&mut self) {
        let h = self.region.header() as *const Header as *mut Header;
        unsafe {
            (*h).lock_owner_pid = 0;
            (*h).lock_owner_tid = 0;
            let lock = &(*h).lock as *const libc::pthread_mutex_t as *mut _;
            libc::pthread_mutex_unlock(lock);
            if let Some(old_mask) = &self.old_mask {
                libc::pthread_sigmask(libc::SIG_SETMASK, old_mask, std::ptr::null_mut());
            }
        }
    }
}

/// FUTEX_WAIT on a waiter node's seq word.
fn futex_wait(addr: *const AtomicU32, expected: u32, timeout: Option<Duration>) -> i32 {
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let ts_ptr = match timeout {
        Some(d) => {
            ts.tv_sec = d.as_secs() as _;
            ts.tv_nsec = d.subsec_nanos() as _;
            &mut ts as *mut _
        }
        None => std::ptr::null_mut(),
    };
    let ret = unsafe {
        libc::syscall(libc::SYS_futex, addr as *const u32, libc::FUTEX_WAIT, expected, ts_ptr, 0, 0)
    };
    if ret == 0 {
        0
    } else {
        -errno()
    }
}

fn futex_wake(addr: *const AtomicU32) {
    // Each head node is slept on by exactly one waiter, so wake one.
    unsafe {
        libc::syscall(libc::SYS_futex, addr as *const u32, libc::FUTEX_WAKE, 1, 0, 0, 0);
    }
}

/// Pop a node from the free list, or hand out a never-before-used node
/// from the lazy pool. Caller holds the lock.
fn node_alloc(r: &Region) -> Option<u32> {
    let head = r.header().node_free_head;
    let idx = if head != NIL {
        let node = r.node_ptr(head);
        let next = unsafe { (*node).waiter_next };
        unsafe { (*(r.base as *mut Header)).node_free_head = next };
        head
    } else {
        // Free list empty: consume a fresh node. Its memory is a sparse
        // zero page (never written since region creation), so seq = 0,
        // registered = 0 — exactly the state a recycled node is reset to.
        let fresh = r.header().node_fresh;
        if fresh >= NODE_COUNT {
            return None;
        }
        unsafe { (*(r.base as *mut Header)).node_fresh = fresh + 1 };
        fresh
    };
    let node = r.node_ptr(idx);
    unsafe {
        (*node).registered = 0;
        (*node).obj = NIL;
        (*node).wait_word = NIL;
        (*node).obj_prev = NIL;
        (*node).obj_next = NIL;
        (*node).waiter_next = NIL;
        (*node).pid = std::process::id();
    }
    Some(idx)
}

/// Return a node to the free list. Caller holds the lock; the node must
/// already be detached from any object list.
fn node_free(r: &Region, idx: u32) {
    let node = r.node_ptr(idx);
    unsafe {
        (*node).waiter_next = r.header().node_free_head;
        (*(r.base as *mut Header)).node_free_head = idx;
    }
}

/// Detach a node from its object's waiter list (O(1), doubly linked).
/// Caller holds the lock.
fn node_detach(r: &Region, idx: u32) {
    let node = r.node_ptr(idx);
    unsafe {
        if (*node).registered == 0 {
            return;
        }
        let obj = (*node).obj;
        let slot = r.slot_ptr(obj);
        if (*node).obj_prev == NIL {
            (*slot).waiter_head = (*node).obj_next;
        } else {
            (*r.node_ptr((*node).obj_prev)).obj_next = (*node).obj_next;
        }
        if (*node).obj_next != NIL {
            (*r.node_ptr((*node).obj_next)).obj_prev = (*node).obj_prev;
        }
        (*node).registered = 0;
        (*node).obj = NIL;
        (*node).obj_prev = NIL;
        (*node).obj_next = NIL;
    }
}

/// Register `node` on object `obj_idx`'s waiter list. Caller holds the lock.
fn node_register(r: &Region, idx: u32, obj_idx: u32, wait_word: u32) {
    let node = r.node_ptr(idx);
    let slot = r.slot_ptr(obj_idx);
    unsafe {
        let old_head = (*slot).waiter_head;
        (*node).registered = 1;
        (*node).obj = obj_idx;
        (*node).wait_word = wait_word;
        (*node).obj_prev = NIL;
        (*node).obj_next = old_head;
        if old_head != NIL {
            (*r.node_ptr(old_head)).obj_prev = idx;
        }
        (*slot).waiter_head = idx;
    }
}

/// Wake every waiter registered on an object (kernel: wake_up(&obj->q)).
/// Caller holds the lock. Each waiter is detached and its head seq bumped;
/// it re-checks conditions when scheduled.
fn wake_object(r: &Region, obj_idx: u32) {
    let debug = debug_enabled();
    let slot = r.slot_ptr(obj_idx);
    let mut idx = unsafe { (*slot).waiter_head };
    while idx != NIL {
        let node = r.node_ptr(idx);
        let (next, word) = unsafe { ((*node).obj_next, (*node).wait_word) };
        node_detach(r, idx);
        let head = r.node_ptr(word);
        unsafe {
            if debug {
                (*head).wake_ts.store(monotonic_ns(), Ordering::Relaxed);
            }
            (*head).seq.fetch_add(1, Ordering::Release);
            futex_wake(&(*head).seq);
        }
        idx = next;
    }
}

/// Lock-free fast check followed by a locked wake walk, used by signal ops.
/// The race with a waiter that registers just after our NIL load is closed
/// by the waiter re-checking acquirability under the region lock before it
/// sleeps (registration and wake walks are mutually exclusive).
fn wake_waiters_if_any(r: &Region, obj_idx: u32) {
    let slot = unsafe { &*r.slot_ptr(obj_idx) };
    if ld32(&slot.waiter_head) == NIL {
        return;
    }
    stat_bump(8); // wake_walks
    if let Ok(_g) = r.lock_guard() {
        wake_object(r, obj_idx);
    }
}

fn current_tid() -> u32 {
    unsafe { libc::syscall(libc::SYS_gettid) as u32 }
}

fn open_and_map(path: &str) -> Result<Region, i32> {
    // Distinct library layouts never share a region file: the version suffix
    // keeps stale files from older builds out of the way. A leftover file
    // with the wrong size/version is unlinked and recreated, never truncated
    // or zeroed in place - in-place mutation would SIGBUS/corrupt any process
    // that still has it mapped (e.g. a wineserver from an older build sharing
    // the path during a version transition).
    let path = match path.rsplit_once('.') {
        // Keep .shm as the final extension; the layout version goes in the
        // middle: ntsync_userspace.v7.shm
        Some((stem, "shm")) => format!("{stem}.v{VERSION}.shm"),
        _ => format!("{path}.v{VERSION}"),
    };
    if debug_enabled() {
        debug_log(&format!("ntsync shm path: {path} (pid {})", std::process::id()));
    }
    let c_path = CString::new(path).map_err(|_| libc::EINVAL)?;
    for _ in 0..4 {
        match unsafe { try_open_and_map(&c_path) } {
            Ok(region) => return Ok(region),
            Err(libc::ESTALE) => unsafe {
                // Stale or corrupt file; remove it (processes that still have
                // it mapped keep their ghost inode, unharmed) and retry.
                libc::unlink(c_path.as_ptr());
            }
            Err(e) => return Err(e),
        }
    }
    Err(libc::ESTALE)
}

/// Returns ESTALE when the existing file does not match this library's
/// layout; the caller then unlinks and retries.
unsafe fn try_open_and_map(c_path: &CString) -> Result<Region, i32> {
    let fd = libc::open(
        c_path.as_ptr(),
        libc::O_RDWR | libc::O_CREAT | libc::O_CLOEXEC,
        0o600,
    );
    if fd < 0 {
        return Err(errno());
    }
    // Serialize initialization across processes.
    if libc::flock(fd, libc::LOCK_EX) != 0 {
        let e = errno();
        libc::close(fd);
        return Err(e);
    }
    let mut st: libc::stat = std::mem::zeroed();
    if libc::fstat(fd, &mut st) != 0 {
        let e = errno();
        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);
        return Err(e);
    }
    if st.st_size != 0 && st.st_size != region_size() as i64 {
        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);
        return Err(libc::ESTALE);
    }
    let need_init = st.st_size == 0;
    if need_init && libc::ftruncate(fd, region_size() as _) != 0 {
        let e = errno();
        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);
        return Err(e);
    }
    let base = libc::mmap(
        std::ptr::null_mut(),
        region_size(),
        libc::PROT_READ | libc::PROT_WRITE,
        libc::MAP_SHARED,
        fd,
        0,
    );
    if base == libc::MAP_FAILED {
        let e = errno();
        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);
        return Err(e);
    }
    let region = Region { base: base as *mut u8 };
    if need_init {
        // No bulk zeroing: a freshly ftruncate'd file reads as zeros, and
        // leaving the pages untouched keeps them sparse - no page-cache
        // allocation, no flash writeback for the ~960 KiB nobody may ever
        // use. Only the header and the reserved slot 0 are written; waiter
        // nodes come from the lazy pool (header.node_fresh), so the node
        // area needs no initialization at all. Zeroed memory gives every
        // slot SLOT_FREE/generation 0 and every node seq 0/registered 0,
        // which is exactly the state the recycling paths reset to.
        let header = &mut *(base as *mut Header);
        let mut attr: libc::pthread_mutexattr_t = std::mem::zeroed();
        libc::pthread_mutexattr_init(&mut attr);
        libc::pthread_mutexattr_setpshared(&mut attr, PTHREAD_PROCESS_SHARED);
        pthread_mutexattr_setrobust(&mut attr, PTHREAD_MUTEX_ROBUST);
        libc::pthread_mutex_init(&mut header.lock, &attr);
        libc::pthread_mutexattr_destroy(&mut attr);
        header.magic = MAGIC;
        header.version = VERSION;
        header.capacity = SLOT_COUNT;
        header.node_capacity = NODE_COUNT;
        header.node_free_head = NIL;
        header.node_fresh = 0;
        header.next_scan = 0;
        header.sweep_ns = 0;
        // Reserve slot 0 (see TYPE_RESERVED). pid 0 keeps sweep_dead away.
        let slot0 = &mut *region.slot_ptr(0);
        slot0.waiter_head = NIL;
        slot0.obj_type = TYPE_RESERVED;
        slot0.pid = 0;
        slot0.state = SLOT_USED;
    } else if region.header().magic != MAGIC
        || region.header().version != VERSION
        || region.header().capacity != SLOT_COUNT
        || region.header().node_capacity != NODE_COUNT
    {
        libc::munmap(base, region_size());
        libc::flock(fd, libc::LOCK_UN);
        libc::close(fd);
        return Err(libc::ESTALE);
    }
    libc::flock(fd, libc::LOCK_UN);
    libc::close(fd);
    Ok(region)
}

static REGION: OnceLock<Result<Region, i32>> = OnceLock::new();

/// Initialize (idempotently) with the shared-memory file at `path`.
/// Called automatically with $TMPDIR/ntsync_userspace.vN.shm if the library
/// is used without an explicit init.
pub fn init(path: Option<&str>) -> Result<(), Error> {
    debug_enabled(); // read NTSYNC_DEBUG and arm the watchdog/stats at startup
    let result = REGION.get_or_init(|| resolve_path(path).and_then(|p| open_and_map(&p)));
    match result {
        Ok(_) => Ok(()),
        Err(e) => Err(Error::Init(*e)),
    }
}

fn region() -> Result<&'static Region, Error> {
    init(None)?;
    match REGION.get() {
        Some(Ok(r)) => Ok(r),
        Some(Err(e)) => Err(Error::Init(*e)),
        None => unreachable!(),
    }
}

/// next_scan high bit: set once the allocation cursor has wrapped around
/// the table. Until then, slots at or above the cursor have never been
/// handed out (sparse zero pages), so sweep_dead must not read them —
/// scanning the whole table would fault ~640 KB of untouched slot pages
/// into RSS and defeat the lazy-init layout.
const SCAN_WRAPPED: u32 = 0x8000_0000;

fn alloc_slot(r: &Region) -> Option<u32> {
    // Caller holds the lock.
    let header = r.header();
    let cursor = header.next_scan & !SCAN_WRAPPED;
    for i in 0..SLOT_COUNT {
        let index = (cursor + i) % SLOT_COUNT;
        let slot = r.slot_ptr(index);
        unsafe {
            if (*slot).state == SLOT_FREE {
                (*slot).waiter_head = NIL;
                let next = (index + 1) % SLOT_COUNT;
                let mut flags = header.next_scan & SCAN_WRAPPED;
                if next == 0 {
                    flags = SCAN_WRAPPED; // cursor wrapped: table fully seen
                }
                (*(r.base as *mut Header)).next_scan = next | flags;
                return Some(((*slot).generation << INDEX_BITS) | index);
            }
        }
    }
    None
}

/// Publish a freshly allocated slot: state fields first, SLOT_USED last
/// (release) so lock-free readers never see a half-initialized object.
unsafe fn slot_publish(r: &Region, handle: u32, obj_type: u32, w0: u64) {
    let slot = r.slot_ptr(handle & INDEX_MASK);
    st64(&(*slot).w0, w0);
    st64(&(*slot).d, 0);
    (*slot).obj_type = obj_type;
    (*slot).pid = libc::getpid() as u32;
    (*slot).holders_len = 0;
    (*slot).holders = [0; MAX_HOLDERS];
    (&*(&(*slot).state as *const u32 as *const AtomicU32)).store(SLOT_USED, Ordering::Release);
}

pub fn create_semaphore(count: u32, max: u32) -> Result<u32, Error> {
    if count > max {
        return Err(Error::Invalid);
    }
    let r = region()?;
    let _g = r.lock_guard().map_err(Error::Init)?;
    let handle = alloc_slot_or_sweep(r).ok_or(Error::Init(libc::ENOMEM))?;
    unsafe { slot_publish(r, handle, TYPE_SEM, pack_sem(count, max)) };
    Ok(handle)
}

pub fn create_mutex(owner: u32, count: u32) -> Result<u32, Error> {
    if (owner == 0) != (count == 0) || count > 0x7fff_ffff {
        return Err(Error::Invalid);
    }
    let r = region()?;
    let _g = r.lock_guard().map_err(Error::Init)?;
    let handle = alloc_slot_or_sweep(r).ok_or(Error::Init(libc::ENOMEM))?;
    unsafe { slot_publish(r, handle, TYPE_MUTEX, pack_mutex(owner, count, false)) };
    Ok(handle)
}

pub fn create_event(manual: bool, signaled: bool) -> Result<u32, Error> {
    let r = region()?;
    let _g = r.lock_guard().map_err(Error::Init)?;
    let handle = alloc_slot_or_sweep(r).ok_or(Error::Init(libc::ENOMEM))?;
    unsafe { slot_publish(r, handle, TYPE_EVENT, pack_event(signaled, manual)) };
    Ok(handle)
}

/// Remove `pid` from a slot's holder list, if present. Caller holds the lock.
unsafe fn holder_remove(slot: *mut Slot, pid: u32) {
    let len = (*slot).holders_len & !HOLDER_OVERFLOW;
    let cap = (len as usize).min(MAX_HOLDERS);
    for i in 0..cap {
        if (*slot).holders[i] == pid {
            let last = cap - 1;
            (*slot).holders[i] = (*slot).holders[last];
            (*slot).holders[last] = 0;
            (*slot).holders_len =
                ((cap - 1) as u32) | ((*slot).holders_len & HOLDER_OVERFLOW);
            return;
        }
    }
}

/// Record that the calling process holds `handle`. Caller holds the lock.
unsafe fn holder_add(slot: *mut Slot, pid: u32) {
    let len = (*slot).holders_len & !HOLDER_OVERFLOW;
    let cap = (len as usize).min(MAX_HOLDERS);
    for i in 0..cap {
        if (*slot).holders[i] == pid {
            return; // already recorded
        }
    }
    if cap == MAX_HOLDERS {
        // Out of room: keep the object alive forever rather than risk freeing
        // it while an untracked holder still uses it.
        (*slot).holders_len |= HOLDER_OVERFLOW;
        return;
    }
    (*slot).holders[cap] = pid;
    (*slot).holders_len =
        ((cap + 1) as u32) | ((*slot).holders_len & HOLDER_OVERFLOW);
}

/// Register the calling process as a holder of `handle`.
///
/// A handle can outlive the process that created it: wineserver relays objects
/// to other processes, and userspace has no fd-close hook, so without this the
/// sweep reaps a relayed object as soon as its creator exits. Call it when a
/// handle arrives from elsewhere; `close` removes the entry again. Idempotent
/// per process.
pub fn retain(handle: u32) -> Result<(), Error> {
    if handle & INDEX_MASK == 0 {
        return Err(Error::Invalid);
    }
    let r = region()?;
    let _g = r.lock_guard().map_err(Error::Init)?;
    let slot = r.slot(handle).ok_or(Error::Invalid)?;
    unsafe { holder_add(slot, libc::getpid() as u32) };
    Ok(())
}

/// Drop the calling process's holder entry for `handle` without closing it.
/// `close` does this itself; this exists for a process that stops using a
/// relayed handle but does not own the object.
pub fn release(handle: u32) -> Result<(), Error> {
    if handle & INDEX_MASK == 0 {
        return Err(Error::Invalid);
    }
    let r = region()?;
    let _g = r.lock_guard().map_err(Error::Init)?;
    let slot = r.slot(handle).ok_or(Error::Invalid)?;
    unsafe { holder_remove(slot, libc::getpid() as u32) };
    Ok(())
}

pub fn close(handle: u32) -> bool {
    if handle & INDEX_MASK == 0 {
        return false; // slot 0 is reserved (TYPE_RESERVED)
    }
    let Ok(r) = region() else { return false };
    let Ok(_g) = r.lock_guard() else { return false };
    let Some(slot) = r.slot(handle) else { return false };
    unsafe {
        // This process no longer holds the object. wineserver normally closes
        // only when the last handle is gone, so the slot is freed below as it
        // always was; the holder entry only exists so a *crashed* holder is
        // not mistaken for a dead creator.
        holder_remove(slot, libc::getpid() as u32);
        // Free first (acquire/release pairs with slot_publish), then wake
        // waiters so they re-validate and fail with -EINVAL (kernel keeps
        // objects alive via fds, so it has no equivalent).
        (&*(&(*slot).state as *const u32 as *const AtomicU32)).store(SLOT_FREE, Ordering::Release);
        (*slot).generation = (*slot).generation.wrapping_add(1);
    }
    wake_object(r, handle & INDEX_MASK);
    true
}

/// Release `count` from a semaphore. Returns the previous count.
/// On overflow the state is left unchanged (kernel: -EOVERFLOW). Lock-free.
pub fn sem_release(handle: u32, count: u32) -> Result<u32, Error> {
    let r = region()?;
    let slot = r.slot(handle).filter(|s| unsafe { ld32(&(**s).obj_type) } == TYPE_SEM).ok_or(Error::Invalid)?;
    let w0 = unsafe { &(*slot).w0 };
    loop {
        let w = ld64(w0);
        let cur = w as u32;
        let max = (w >> 32) as u32;
        stat_bump(7); // signal_ops
        let Some(sum) = cur.checked_add(count).filter(|&s| s <= max) else {
            return Err(Error::Overflow);
        };
        match cas64(w0, w, pack_sem(sum, max)) {
            Ok(_) => {
                wake_waiters_if_any(r, handle & INDEX_MASK);
                return Ok(cur);
            }
            Err(_) => continue,
        }
    }
}

pub fn event_set(handle: u32) -> Result<u32, Error> {
    stat_bump(7);
    let r = region()?;
    let slot = r.slot(handle).filter(|s| unsafe { ld32(&(**s).obj_type) } == TYPE_EVENT).ok_or(Error::Invalid)?;
    let w0 = unsafe { &(*slot).w0 };
    loop {
        let w = ld64(w0);
        let prev = w as u32;
        if prev != 0 {
            return Ok(prev);
        }
        match cas64(w0, w, w | 1) {
            Ok(_) => {
                wake_waiters_if_any(r, handle & INDEX_MASK);
                return Ok(0);
            }
            Err(_) => continue,
        }
    }
}

pub fn event_reset(handle: u32) -> Result<u32, Error> {
    stat_bump(7);
    let r = region()?;
    let slot = r.slot(handle).filter(|s| unsafe { ld32(&(**s).obj_type) } == TYPE_EVENT).ok_or(Error::Invalid)?;
    let w0 = unsafe { &(*slot).w0 };
    loop {
        let w = ld64(w0);
        let prev = w as u32;
        if prev == 0 {
            return Ok(0);
        }
        match cas64(w0, w, w & !1u64) {
            Ok(_) => return Ok(prev),
            Err(_) => continue,
        }
    }
}

pub fn event_pulse(handle: u32) -> Result<u32, Error> {
    // Rare and awkward for the packed-state fast path (two words involved):
    // do it under the region lock, like the kernel's pulse under
    // wait_all_lock.
    let r = region()?;
    let _g = r.lock_guard().map_err(Error::Init)?;
    let slot = r.slot(handle).filter(|s| unsafe { (**s).obj_type } == TYPE_EVENT).ok_or(Error::Invalid)?;
    let prev = unsafe {
        let prev = ld64(&(*slot).w0) as u32;
        // Wake all current waiters, then return to unsignaled.
        let d = ld64(&(*slot).d);
        st64(&(*slot).d, d.wrapping_add(1));
        st64(&(*slot).w0, ld64(&(*slot).w0) & !1u64);
        prev
    };
    wake_object(r, handle & INDEX_MASK);
    Ok(prev)
}

/// Unlock a mutex held by `owner`. Returns the previous recursion count.
/// Lock-free.
pub fn mutex_unlock(handle: u32, owner: u32) -> Result<u32, Error> {
    stat_bump(7);
    if owner == 0 {
        return Err(Error::Invalid);
    }
    let r = region()?;
    let slot = r.slot(handle).filter(|s| unsafe { ld32(&(**s).obj_type) } == TYPE_MUTEX).ok_or(Error::Invalid)?;
    let w0 = unsafe { &(*slot).w0 };
    loop {
        let w = ld64(w0);
        let cur_owner = w as u32;
        let hi = (w >> 32) as u32;
        let count = hi & 0x7fff_ffff;
        let ownerdead = hi >> 31 != 0;
        if cur_owner != owner {
            return Err(Error::Perm);
        }
        let freed = count == 1;
        let new = pack_mutex(if freed { 0 } else { owner }, count - 1, ownerdead);
        match cas64(w0, w, new) {
            Ok(_) => {
                if freed {
                    wake_waiters_if_any(r, handle & INDEX_MASK);
                }
                return Ok(count);
            }
            Err(_) => continue,
        }
    }
}

/// Mark a mutex abandoned and release ownership (kernel: MUTEX_KILL).
/// Lock-free.
pub fn mutex_kill(handle: u32, owner: u32) -> Result<(), Error> {
    stat_bump(7);
    if owner == 0 {
        return Err(Error::Invalid);
    }
    let r = region()?;
    let slot = r.slot(handle).filter(|s| unsafe { ld32(&(**s).obj_type) } == TYPE_MUTEX).ok_or(Error::Invalid)?;
    let w0 = unsafe { &(*slot).w0 };
    loop {
        let w = ld64(w0);
        if w as u32 != owner {
            return Err(Error::Perm);
        }
        match cas64(w0, w, pack_mutex(0, 0, true)) {
            Ok(_) => {
                wake_waiters_if_any(r, handle & INDEX_MASK);
                return Ok(());
            }
            Err(_) => continue,
        }
    }
}

/// Returns (count, max). Lock-free.
pub fn read_sem(handle: u32) -> Result<(u32, u32), Error> {
    let r = region()?;
    let slot = r.slot(handle).filter(|s| unsafe { ld32(&(**s).obj_type) } == TYPE_SEM).ok_or(Error::Invalid)?;
    let w = ld64(unsafe { &(*slot).w0 });
    Ok((w as u32, (w >> 32) as u32))
}

/// Returns (count, owner, owner_dead). Lock-free.
pub fn read_mutex(handle: u32) -> Result<(u32, u32, bool), Error> {
    let r = region()?;
    let slot = r.slot(handle).filter(|s| unsafe { ld32(&(**s).obj_type) } == TYPE_MUTEX).ok_or(Error::Invalid)?;
    let w = ld64(unsafe { &(*slot).w0 });
    let hi = (w >> 32) as u32;
    Ok((hi & 0x7fff_ffff, w as u32, hi >> 31 != 0))
}

/// Returns (manual, signaled). Lock-free.
pub fn read_event(handle: u32) -> Result<(bool, bool), Error> {
    let r = region()?;
    let slot = r.slot(handle).filter(|s| unsafe { ld32(&(**s).obj_type) } == TYPE_EVENT).ok_or(Error::Invalid)?;
    let w = ld64(unsafe { &(*slot).w0 });
    Ok(((w >> 32) != 0, (w as u32) != 0))
}

/// Check whether an object is acquirable ("locked" in kernel terms) without
/// mutating it. Lock-free.
fn is_locked(slot: &Slot, owner: u32, entry_seq: u64) -> bool {
    let w = ld64(&slot.w0);
    match slot.obj_type {
        TYPE_SEM => w as u32 > 0,
        TYPE_MUTEX => {
            let cur_owner = w as u32;
            let count = ((w >> 32) as u32) & 0x7fff_ffff;
            (cur_owner == 0 || cur_owner == owner) && count < 0x7fff_ffff
        }
        TYPE_EVENT => w as u32 != 0 || ld64(&slot.d) != entry_seq,
        _ => false,
    }
}

/// Attempt to acquire an object (lock-free CAS). Returns Some(owner_dead)
/// on success, None if not acquirable.
fn try_acquire(slot: &Slot, owner: u32, entry_seq: u64) -> Option<bool> {
    match slot.obj_type {
        TYPE_SEM => loop {
            let w = ld64(&slot.w0);
            let count = w as u32;
            if count == 0 {
                return None;
            }
            let max = (w >> 32) as u32;
            match cas64(&slot.w0, w, pack_sem(count - 1, max)) {
                Ok(_) => return Some(false),
                Err(_) => continue,
            }
        },
        TYPE_MUTEX => loop {
            let w = ld64(&slot.w0);
            let cur_owner = w as u32;
            let hi = (w >> 32) as u32;
            let count = hi & 0x7fff_ffff;
            let ownerdead = hi >> 31 != 0;
            if count >= 0x7fff_ffff {
                return None;
            }
            let (new, od) = if cur_owner == 0 {
                (pack_mutex(owner, count + 1, false), ownerdead)
            } else if cur_owner == owner {
                (pack_mutex(owner, count + 1, ownerdead), false)
            } else {
                return None;
            };
            match cas64(&slot.w0, w, new) {
                Ok(_) => return Some(od),
                Err(_) => continue,
            }
        },
        TYPE_EVENT => {
            let w = ld64(&slot.w0);
            if w as u32 != 0 {
                if (w >> 32) == 0 {
                    // auto-reset: claim it
                    loop {
                        let w = ld64(&slot.w0);
                        if w as u32 == 0 {
                            // Lost the race; fall through to pulse check.
                            break;
                        }
                        match cas64(&slot.w0, w, w & !1u64) {
                            Ok(_) => return Some(false),
                            Err(_) => continue,
                        }
                    }
                } else {
                    return Some(false);
                }
            }
            if ld64(&slot.d) != entry_seq {
                // Pulsed during the wait; considered satisfied (pulse leaves
                // the event unsignaled, so nothing to reset).
                return Some(false);
            }
            None
        }
        _ => None,
    }
}

/// Undo a try_acquire on `slot` (wait-all rollback). Mutex state is restored
/// wholesale (we own it, nobody else can mutate it); semaphore count is
/// incremented back (concurrent releases must not be lost); an auto-reset
/// event is re-signaled.
fn rollback_acquire(slot: &Slot, owner: u32, owner_dead_was: bool) {
    match slot.obj_type {
        TYPE_SEM => loop {
            let w = ld64(&slot.w0);
            let count = w as u32;
            let max = (w >> 32) as u32;
            if cas64(&slot.w0, w, pack_sem(count + 1, max)).is_ok() {
                return;
            }
        },
        TYPE_MUTEX => loop {
            let w = ld64(&slot.w0);
            let count = ((w >> 32) as u32) & 0x7fff_ffff;
            if count == 1 {
                // We hold the only reference: restore the empty state.
                if cas64(&slot.w0, w, pack_mutex(0, 0, owner_dead_was)).is_ok() {
                    return;
                }
            } else {
                // Recursive self-acquire: just drop one level.
                let od = ((w >> 32) as u32) >> 31 != 0;
                if cas64(&slot.w0, w, pack_mutex(owner, count - 1, od)).is_ok() {
                    return;
                }
            }
        },
        TYPE_EVENT => loop {
            let w = ld64(&slot.w0);
            // Only auto-reset events were mutated (signaled 1 -> 0).
            if (w >> 32) == 0 && (w as u32) == 0 {
                if cas64(&slot.w0, w, w | 1).is_ok() {
                    return;
                }
            } else {
                return;
            }
        },
        _ => {}
    }
}

/// Log to stderr, stdout, and $TMPDIR/ntsync_debug.log. The launcher's
/// logcat only captures its own stdout, so the file is the reliable
/// channel for in-game diagnostics.
fn debug_log(msg: &str) {
    use std::io::Write;
    // Raw writes that never panic: wineserver/game processes may close or
    // redirect stdout/stderr, and println!/eprintln! panic on EBADF, which
    // would silently kill the stats thread.
    let mut line = msg.as_bytes().to_vec();
    line.push(b'\n');
    unsafe {
        let _ = libc::write(2, line.as_ptr() as *const _, line.len());
        let _ = libc::write(1, line.as_ptr() as *const _, line.len());
    }
    if let Some(tmp) = std::env::var_os("TMPDIR") {
        let path = std::path::PathBuf::from(tmp).join("ntsync_debug.log");
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(path) {
            let _ = f.write_all(&line);
        }
    }
}

// ---- Instrumentation (NTSYNC_DEBUG) ----
// Cheap per-process relaxed counters; a background thread dumps them every
// 10s so on-device runs can show where the time actually goes.
mod stats {
    use std::sync::atomic::{AtomicU64, Ordering};

    pub const NAMES: [&str; 15] = [
        "wait_calls",        // total wait_any/wait_all calls
        "wait_fast_ok",      // satisfied without any lock
        "wait_blocked",      // actually slept on a futex
        "futex_sleeps",      // futex_wait syscalls
        "lock_acq",          // region lock acquisitions
        "lock_contended",    // lock acquisitions that looped/EOWNERDEAD
        "cas_retries",       // CAS loop retries (contention on object words)
        "signal_ops",        // release/set/unlock/... calls
        "wake_walks",        // times a signal op found waiters and locked
        "nodes_registered",  // waiter-node registrations
        "wake_lat_cnt",      // wakes with a timestamp sample
        "wake_lat_us_sum",   // total wake latency (signal -> waiter running), us
        "wake_lat_us_max",   // worst wake latency, us
        "spin_wins",         // waits satisfied during the pre-sleep spin
        "spin_exhausted",    // spins that still ended in a futex sleep
    ];
    pub static COUNTERS: [AtomicU64; 15] = [
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
        AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    ];

    #[inline]
    pub fn bump(i: usize) {
        COUNTERS[i].fetch_add(1, Ordering::Relaxed);
    }
    #[inline]
    pub fn add(i: usize, n: u64) {
        COUNTERS[i].fetch_add(n, Ordering::Relaxed);
    }
}
use stats::{add as stat_add, bump as stat_bump};

fn stats_dump_loop() {
    std::thread::spawn(|| {
        let dump = |line: &mut String| {
            line.clear();
            line.push_str(&format!("ntsync stats (pid {}):", std::process::id()));
            for (i, name) in stats::NAMES.iter().enumerate() {
                let v = stats::COUNTERS[i].load(Ordering::Relaxed);
                line.push_str(&format!(" {name}={v}"));
            }
            debug_log(line);
        };
        let mut line = String::new();
        std::thread::sleep(Duration::from_secs(1));
        dump(&mut line);
        loop {
            std::thread::sleep(Duration::from_secs(10));
            dump(&mut line);
        }
    });
}

fn debug_enabled() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| {
        let on = std::env::var_os("NTSYNC_DEBUG")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false);
        if on {
            debug_log(&format!("watchdog armed (pid {})", std::process::id()));
            stats_dump_loop();
        }
        on
    })
}

/// Whether to defer SIGUSR1 around region-lock acquisitions (default on).
/// Only needed when the host process installs a SIGUSR1 handler that can
/// suspend threads (Wine's thread-suspend machinery); everything else can
/// set NTSYNC_NO_SIGUSR1_BLOCK=1 to skip the two pthread_sigmask syscalls
/// per lock acquisition.
fn sigusr1_block_enabled() -> bool {
    static ONCE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ONCE.get_or_init(|| {
        std::env::var_os("NTSYNC_NO_SIGUSR1_BLOCK")
            .map(|v| v.is_empty() || v == "0")
            .unwrap_or(true)
    })
}

/// Dump the state of every object involved in a stuck wait, to distinguish
/// a lost wakeup (object signaled while we sleep) from a never-coming signal.
fn debug_dump_stuck_wait(
    r: &Region,
    handles: &[u32],
    owner: u32,
    alert: u32,
    all: bool,
    start: std::time::Instant,
) {
    let type_name = |t: u32| match t {
        TYPE_EVENT => "event",
        TYPE_MUTEX => "mutex",
        TYPE_SEM => "sem",
        _ => "?",
    };
    debug_log(&format!(
        "stuck wait: {:?} elapsed, {} owner={} alert={:#x}",
        start.elapsed(),
        if all { "all" } else { "any" },
        owner,
        alert,
    ));
    if let Ok(_g) = r.lock_guard() {
        for h in handles.iter().copied().chain((alert != 0).then_some(alert)) {
            match r.slot(h) {
                Some(slot) => unsafe {
                    let s = &*slot;
                    let locked = is_locked(s, owner, ld64(&s.d));
                    debug_log(&format!(
                        "  {:#010x} {} state={} w0={:#x} d={} pid={} pid_alive={} gen={} waiters={} now_signaled={}",
                        h, type_name(s.obj_type), s.state, ld64(&s.w0), ld64(&s.d), s.pid,
                        pid_alive(s.pid), s.generation,
                        s.waiter_head != NIL, locked,
                    ));
                },
                None => debug_log(&format!("  {:#010x} <closed/invalid>", h)),
            }
        }
    }
}

pub fn wait_any(handles: &[u32], owner: u32, timeout: Option<Duration>, alert: u32) -> WaitOutcome {
    wait(handles, owner, timeout, false, alert)
}

pub fn wait_all(handles: &[u32], owner: u32, timeout: Option<Duration>, alert: u32) -> WaitOutcome {
    wait(handles, owner, timeout, true, alert)
}

/// Fixed-capacity stack set of waiter node indices: up to MAX_WAIT_COUNT
/// object nodes plus one alert node. Replaces a per-wait heap Vec.
struct NodeSet {
    idx: [u32; MAX_WAIT_COUNT + 1],
    len: usize,
}

impl NodeSet {
    fn new() -> Self {
        NodeSet {
            idx: [0; MAX_WAIT_COUNT + 1],
            len: 0,
        }
    }
    fn push(&mut self, v: u32) {
        self.idx[self.len] = v;
        self.len += 1;
    }
    fn is_empty(&self) -> bool {
        self.len == 0
    }
    fn clear(&mut self) {
        self.len = 0;
    }
    fn iter(&self) -> std::slice::Iter<'_, u32> {
        self.idx[..self.len].iter()
    }
}

impl std::ops::Index<usize> for NodeSet {
    type Output = u32;
    fn index(&self, i: usize) -> &u32 {
        &self.idx[i]
    }
}

/// Detach and free all nodes of a waiter (caller holds the lock).
fn waiter_cleanup(r: &Region, nodes: &mut NodeSet) {
    for &idx in nodes.iter() {
        node_detach(r, idx);
    }
    for &idx in nodes.iter() {
        node_free(r, idx);
    }
    nodes.clear();
}

/// Try to satisfy the wait right now (lock-free). Some(outcome) = done,
/// None = not satisfiable at this instant. Handles must be pre-validated.
fn try_satisfy(
    r: &Region,
    handles: &[u32],
    owner: u32,
    entry_seqs: &[u64; MAX_WAIT_COUNT],
    all: bool,
) -> Option<WaitOutcome> {
    if !all {
        for (i, h) in handles.iter().enumerate() {
            let slot = unsafe { &*r.slot_ptr(h & INDEX_MASK) };
            if let Some(owner_dead) = try_acquire(slot, owner, entry_seqs[i]) {
                return Some(WaitOutcome::Signaled {
                    index: i as u32,
                    owner_dead,
                });
            }
        }
        None
    } else {
        // Acquire one by one; roll back if any object is busy. Stack array:
        // handles.len() <= MAX_WAIT_COUNT, so no heap allocation. Entries
        // are write-before-read (0..n_acquired), so leave them uninit
        // rather than zeroing 1 KB on every wait-all fast path.
        let mut acquired: [std::mem::MaybeUninit<(usize, bool)>; MAX_WAIT_COUNT] =
            [std::mem::MaybeUninit::uninit(); MAX_WAIT_COUNT];
        let mut n_acquired = 0usize;
        let mut owner_dead = false;
        for (i, h) in handles.iter().enumerate() {
            let slot = unsafe { &*r.slot_ptr(h & INDEX_MASK) };
            match try_acquire(slot, owner, entry_seqs[i]) {
                Some(od) => {
                    owner_dead |= od;
                    acquired[n_acquired].write((i, od));
                    n_acquired += 1;
                }
                None => {
                    for a in &acquired[..n_acquired] {
                        let &(j, od) = unsafe { a.assume_init_ref() };
                        rollback_acquire(
                            unsafe { &*r.slot_ptr(handles[j] & INDEX_MASK) },
                            owner,
                            od,
                        );
                    }
                    return None;
                }
            }
        }
        Some(WaitOutcome::Signaled { index: 0, owner_dead })
    }
}

fn wait(handles: &[u32], owner: u32, timeout: Option<Duration>, all: bool, alert: u32) -> WaitOutcome {
    if owner == 0 || handles.is_empty() || handles.len() > MAX_WAIT_COUNT {
        return WaitOutcome::Invalid;
    }
    // Reject duplicate handles (kernel behavior) without allocating.
    for (i, h) in handles.iter().enumerate() {
        if handles[..i].contains(h) {
            return WaitOutcome::Invalid;
        }
    }
    let Ok(r) = region() else {
        return WaitOutcome::Invalid;
    };
    stat_bump(0); // wait_calls

    let deadline = timeout.map(|d| std::time::Instant::now() + d);
    let debug = debug_enabled();
    let start = std::time::Instant::now();
    let mut last_dump = start;

    let mut entry_seqs = [0u64; MAX_WAIT_COUNT];
    let n_nodes = handles.len() + (alert != 0) as usize;
    let mut nodes = NodeSet::new();

    // ---- Fast path: validate and try to acquire without any lock. ----
    let fast_valid = handles.iter().enumerate().all(|(i, h)| {
        match r.slot(*h) {
            Some(slot) => unsafe {
                let s = &*slot;
                entry_seqs[i] = if s.obj_type == TYPE_EVENT { ld64(&s.d) } else { 0 };
                matches!(s.obj_type, TYPE_SEM | TYPE_MUTEX | TYPE_EVENT)
            },
            None => false,
        }
    }) && (alert == 0
        || matches!(r.slot(alert), Some(s) if unsafe { (*s).obj_type } == TYPE_EVENT));
    if fast_valid {
        if alert != 0 {
            let slot = unsafe { &*r.slot(alert).unwrap() };
            if ld64(&slot.w0) as u32 != 0 {
                return WaitOutcome::Signaled {
                    index: handles.len() as u32,
                    owner_dead: false,
                };
            }
        }
        if let Some(outcome) = try_satisfy(r, handles, owner, &entry_seqs, all) {
            stat_bump(1); // wait_fast_ok
            return outcome;
        }
        if timeout == Some(Duration::ZERO) {
            return WaitOutcome::Timeout;
        }
    }
    // Fall through to the blocking path; the locked section below
    // re-validates authoritatively (a close racing the fast path fails it).

    loop {
        // Take the region lock for the validate/register cycle. Registration
        // and signalers' wake walks are mutually exclusive, and we re-check
        // acquirability under the lock before sleeping, so no wakeup is lost.
        let Ok(_g) = r.lock_guard() else {
            if let Ok(g) = r.lock_guard() {
                waiter_cleanup(r, &mut nodes);
                drop(g);
            }
            return WaitOutcome::Invalid;
        };

        // Authoritative validation (also catches objects closed mid-wait).
        let mut valid = true;
        for (i, h) in handles.iter().enumerate() {
            match r.slot(*h) {
                Some(slot) => unsafe {
                    let s = &*slot;
                    if !matches!(s.obj_type, TYPE_SEM | TYPE_MUTEX | TYPE_EVENT) {
                        valid = false;
                        break;
                    }
                    if nodes.is_empty() && s.obj_type == TYPE_EVENT {
                        entry_seqs[i] = ld64(&s.d);
                    }
                },
                None => {
                    valid = false;
                    break;
                }
            }
        }
        if valid && alert != 0 {
            valid = matches!(r.slot(alert), Some(s) if unsafe { (*s).obj_type } == TYPE_EVENT);
        }
        if !valid {
            waiter_cleanup(r, &mut nodes);
            return WaitOutcome::Invalid;
        }

        // Alertable wait: the alert event wins immediately if signaled.
        // It is only tested, never acquired (wineserver resets it when the
        // APC queue empties) - kernel ntsync_wait_args.alert contract.
        if alert != 0 {
            let slot = r.slot(alert).expect("validated above");
            if ld64(unsafe { &(*slot).w0 }) as u32 != 0 {
                waiter_cleanup(r, &mut nodes);
                return WaitOutcome::Signaled {
                    index: handles.len() as u32,
                    owner_dead: false,
                };
            }
        }

        // Allocate nodes on first block, then (re)register one node per
        // object *before* the acquire attempt below. Signalers change object
        // state with a lock-free CAS and only afterwards walk waiter lists,
        // so registration must precede our check: a state change that landed
        // before the check is seen by it, and one that lands after finds our
        // nodes on the list (registration and wake walks are mutually
        // exclusive under the lock). Check-then-register would race the CAS
        // and lose the wakeup.
        if nodes.is_empty() {
            for _ in 0..n_nodes {
                match node_alloc(r) {
                    Some(idx) => nodes.push(idx),
                    None => {
                        waiter_cleanup(r, &mut nodes);
                        return WaitOutcome::Invalid; // waiter pool exhausted
                    }
                }
            }
        }
        let head = nodes[0];
        for (i, h) in handles.iter().enumerate() {
            if unsafe { (*r.node_ptr(nodes[i])).registered } == 0 {
                node_register(r, nodes[i], h & INDEX_MASK, head);
            }
        }
        if alert != 0 {
            let an = nodes[handles.len()];
            if unsafe { (*r.node_ptr(an)).registered } == 0 {
                node_register(r, an, alert & INDEX_MASK, head);
            }
        }

        // Make the registration globally visible before checking object
        // state: an Acquire load alone lets the registration store linger in
        // the store buffer (StoreLoad reordering), so a signaler's lock-free
        // waiter_head read could still see NIL after its state CAS. The
        // fence pairs with the signaler's CAS-then-read order: if our check
        // observes pre-CAS state, the signaler's read is guaranteed to
        // observe our registration.
        std::sync::atomic::fence(Ordering::SeqCst);

        // Authoritative acquire attempt.
        if let Some(outcome) = try_satisfy(r, handles, owner, &entry_seqs, all) {
            waiter_cleanup(r, &mut nodes);
            return outcome;
        }

        // Timeout check before going to sleep.
        let remaining = match deadline {
            Some(dl) => {
                let now = std::time::Instant::now();
                if now >= dl {
                    waiter_cleanup(r, &mut nodes);
                    return WaitOutcome::Timeout;
                }
                Some(dl - now)
            }
            None => None,
        };

        stat_bump(2); // wait_blocked
        stat_add(9, n_nodes as u64); // nodes_registered
        // Sample the head seq under the lock: any signaler that detaches us
        // after this point bumps the seq, so futex_wait below either blocks
        // or fails immediately with EAGAIN - never a missed wakeup.
        let seq = unsafe { (*r.node_ptr(head)).seq.load(Ordering::Acquire) };
        drop(_g);

        // Bounded spin before sleeping (NTSYNC_SPIN_ITERS, default 0 = off):
        // a signal that lands within the window is acquired lock-free,
        // avoiding a futex sleep + wake pair (two context switches) and the
        // signaler's wake walk. Correctness does not depend on the spin:
        // nodes stay registered, so a signaler either hands the state to our
        // try_satisfy CAS or detaches us and bumps seq, in which case
        // futex_wait below returns EAGAIN and the loop re-attempts. The
        // entry_seqs snapshot is not refreshed during the spin, so a pulse
        // shorter than the window can be missed - identical semantics to a
        // short sleep (woken waiters re-check with the same snapshot).
        let spin = spin_budget();
        if spin > 0 {
            // Reaching the end of the loop without winning means the
            // spin did not pay off - penalize below. The detached-mid-
            // spin break is neutral: a signaler DID target us, we just
            // lost the CAS race for the state; that is a timing race,
            // not evidence that spinning is useless for this thread.
            let mut neutral = false;
            for _ in 0..spin {
                if alert != 0 {
                    let s = r.slot(alert).expect("validated above");
                    if ld64(unsafe { &(*s).w0 }) as u32 != 0 {
                        stat_bump(13); // spin_wins (alert)
                        spin_credit_reward();
                        if let Ok(g) = r.lock_guard() {
                            waiter_cleanup(r, &mut nodes);
                            drop(g);
                        }
                        return WaitOutcome::Signaled {
                            index: handles.len() as u32,
                            owner_dead: false,
                        };
                    }
                }
                // Read-only peek first: a failed try_satisfy still CASes
                // the object words, and N spinners CASing one word turns
                // the cache line into a ping-pong that slows the very
                // owner we are waiting for. Only attempt the real acquire
                // when the load-only check says it could succeed (any
                // object for WAIT_ANY, all for WAIT_ALL).
                let peek = handles.iter().enumerate().any(|(i, h)| {
                    is_locked(unsafe { &*r.slot_ptr(h & INDEX_MASK) }, owner, entry_seqs[i])
                }) && (!all || handles.iter().enumerate().all(|(i, h)| {
                    is_locked(unsafe { &*r.slot_ptr(h & INDEX_MASK) }, owner, entry_seqs[i])
                }));
                if peek {
                    if let Some(outcome) = try_satisfy(r, handles, owner, &entry_seqs, all) {
                        stat_bump(13); // spin_wins
                        if let Ok(g) = r.lock_guard() {
                            waiter_cleanup(r, &mut nodes);
                            drop(g);
                        }
                        spin_credit_reward();
                        return outcome;
                    }
                }
                // A signaler detached us mid-spin: the wake was directed at
                // us but the state went to someone else (or the object was
                // closed). Stop spinning; futex_wait returns EAGAIN and the
                // locked section re-registers or fails validation.
                if unsafe { ld32(&(*r.node_ptr(head)).registered) } == 0 {
                    neutral = true;
                    break;
                }
                std::hint::spin_loop();
            }
            if !neutral {
                stat_bump(14); // spin_exhausted
                spin_credit_penalize();
            }
        }

        // Going to sleep: a thread with slack, so this is the right place
        // for the rate-limited dead-process sweep. Deliberately NOT hooked
        // into the lock-free signal ops, so a scan never lands on a
        // latency-critical signaler (e.g. a render thread's SetEvent).
        auto_sweep_dead(r);

        // With NTSYNC_DEBUG=1, cap each futex wait at 5s so a stuck wait
        // periodically dumps the object states to stderr.
        let slice = if debug {
            Some(remaining.map_or(Duration::from_secs(5), |r| r.min(Duration::from_secs(5))))
        } else {
            remaining
        };

        // Spin tail: a futex timeout cannot fire closer than the process's
        // timer slack, so hand the last `tail` of the wait to a spin instead
        // of parking it. Short waits spin immediately; longer ones park for
        // the part before the tail, and the next iteration lands here to spin
        // out the remainder.
        // Inside the tail, spin only while the thread's allowance lasts; when
        // it is spent the wait parks the whole remainder, so a thread pacing
        // every frame cannot quietly burn a core.
        let mut slice = slice;
        if let (Some(tail), Some(left)) = (spin_tail(), slice) {
            if left > tail {
                // Only shorten the park when this thread will be allowed to
                // spin the tail it is parked short of; otherwise the wait
                // would cost two parks for a tail it never gets.
                if tail_budget_can(tail.as_nanos() as u64) {
                    slice = Some(left - tail);
                }
            } else if tail_budget_take(left.as_nanos() as u64) {
                let word = unsafe { &(*r.node_ptr(head)).seq };
                let dl = std::time::Instant::now() + left;
                while std::time::Instant::now() < dl {
                    if word.load(Ordering::Acquire) != seq {
                        break;
                    }
                    std::hint::spin_loop();
                }
                continue;
            }
        }
        stat_bump(3); // futex_sleeps
        let ret = futex_wait(unsafe { &(*r.node_ptr(head)).seq }, seq, slice);
        if ret == 0 && debug {
            // Woken by a signaler: measure signal-to-running latency.
            let head_node = unsafe { &*r.node_ptr(head) };
            let ts = head_node.wake_ts.load(Ordering::Relaxed);
            if ts != 0 {
                let lat_us = monotonic_ns().saturating_sub(ts) / 1000;
                stat_bump(10); // wake_lat_cnt
                stat_add(11, lat_us); // wake_lat_us_sum
                stats::COUNTERS[12].fetch_max(lat_us, Ordering::Relaxed); // wake_lat_us_max
            }
        }
        match ret {
            0 | libc::EAGAIN => {}
            x if x == libc::ETIMEDOUT => {
                // A signal may have landed just as the deadline expired; do
                // not decide here. Fall through to detach + loop, where the
                // locked section re-attempts the acquire authoritatively
                // before the deadline check returns Timeout.
                //
                // Also a good moment for an opportunistic sweep: a wait that
                // keeps timing out may be blocked on an object whose creator
                // process died (the kernel driver reaps those via fd release;
                // userspace must scan). Rate-limited region-wide.
                auto_sweep_dead(r);
                if debug && deadline.is_none_or(|dl| std::time::Instant::now() < dl) && last_dump.elapsed() >= Duration::from_secs(5) {
                    debug_dump_stuck_wait(r, handles, owner, alert, all, start);
                    last_dump = std::time::Instant::now();
                }
            }
            _ => {} // EINTR and friends: loop and re-check
        }
        // Detach our nodes that are still registered (timeout/EINTR wake);
        // signalers already detached the ones they woke. Peek lock-free
        // first: on the normal signal-wake path nothing is registered, so
        // we skip the robust-mutex + sigmask round entirely. A stale `1`
        // read just means a harmless extra lock; a detached node's `0` is
        // stable (only we re-register our own nodes, after this point).
        let any_registered = nodes
            .iter()
            .any(|&idx| unsafe { ld32(&(*r.node_ptr(idx)).registered) } != 0);
        if any_registered {
            if let Ok(_g) = r.lock_guard() {
                for &idx in nodes.iter() {
                    node_detach(r, idx);
                }
            }
        }
    }
}

fn pid_alive(pid: u32) -> bool {
    if pid == 0 {
        return false;
    }
    let ret = unsafe { libc::kill(pid as i32, 0) };
    !(ret != 0 && errno() == libc::ESRCH)
}

/// A dead-owner reaping candidate found by `SweepScan::next_chunk`, carried
/// across the lock-drop in `auto_sweep_dead` so the liveness probe (a
/// `kill(2)` syscall) runs OUTSIDE the region lock. `generation` is only
/// meaningful for a slot (it detects the slot being freed and reallocated
/// while unlocked).
pub(crate) struct DeadCand {
    node: bool,
    index: u32,
    pid: u32,
    generation: u32,
    /// Holder pids as they were when the candidate was collected (slots only).
    /// The candidate is skipped if they changed before the reap, and the whole
    /// object is kept alive if any of them is still alive.
    holders: [u32; MAX_HOLDERS],
    holders_len: u32,
}

impl DeadCand {
    fn node(index: u32, pid: u32) -> Self {
        Self {
            node: true,
            index,
            pid,
            generation: 0,
            holders: [0; MAX_HOLDERS],
            holders_len: 0,
        }
    }
}

/// Candidates per chunk: bounds both the memory a sweep holds while the
/// region lock is held and the span it spends unlocked probing liveness.
const SWEEP_CHUNK: usize = 256;

/// Cursor over one sweep's candidates. Waiter nodes are visited before
/// object slots, so every dead waiter is detached before a dead object's
/// slot is freed and recycled with an empty waiter list.
struct SweepScan {
    node_hi: u32,
    slot_hi: u32,
    next_node: u32,
    next_slot: u32,
}

impl SweepScan {
    /// Snapshot the ranges that were ever handed out. Caller holds the region
    /// lock; the ranges only grow behind it, so objects created during the
    /// pass are left to the next one.
    fn new(r: &Region) -> Self {
        let header = r.header();
        // Only nodes below node_fresh were ever handed out; the rest have
        // never been touched (registered == 0 by virtue of sparse zero pages).
        // Only slots below the allocation cursor were ever handed out — until
        // the cursor wraps (SCAN_WRAPPED), higher slots are sparse zero pages
        // that must not be faulted in just to discover pid == 0.
        let slot_hi = if header.next_scan & SCAN_WRAPPED != 0 {
            SLOT_COUNT
        } else {
            header.next_scan
        };
        Self {
            node_hi: header.node_fresh.min(NODE_COUNT),
            slot_hi,
            next_node: 0,
            next_slot: 0,
        }
    }

    /// Fill `out` with the next chunk of candidates: nodes and slots whose
    /// owner pid MIGHT be dead. Runs NO syscall, so the caller can drop the
    /// region lock before probing liveness. A full `out` may or may not end
    /// the scan; an empty one always does. Caller holds the region lock.
    fn next_chunk(&mut self, r: &Region, out: &mut Vec<DeadCand>) {
        out.clear();
        while out.len() < SWEEP_CHUNK {
            if self.next_node < self.node_hi {
                let index = self.next_node;
                self.next_node += 1;
                let node = r.node_ptr(index);
                unsafe {
                    if (*node).registered != 0 {
                        out.push(DeadCand::node(index, (*node).pid));
                    }
                }
            } else if self.next_slot < self.slot_hi {
                let index = self.next_slot;
                self.next_slot += 1;
                let slot = r.slot_ptr(index);
                unsafe {
                    if (*slot).state == SLOT_USED && (*slot).pid != 0 {
                        out.push(DeadCand {
                            node: false,
                            index,
                            pid: (*slot).pid,
                            generation: (*slot).generation,
                            holders: (*slot).holders,
                            holders_len: (*slot).holders_len,
                        });
                    }
                }
            } else {
                return;
            }
        }
    }
}

/// Distinct pids one sweep remembers liveness for. A process that owns many
/// objects and waiter nodes is the normal case in a Wine prefix, so caching
/// verdicts keeps a sweep at one `kill(2)` per pid instead of one per
/// candidate. Overflow just probes the pid again.
const PID_CACHE: usize = 32;

/// Stack-local pid liveness cache for one sweep: no allocation, no lock.
struct PidCache {
    pid: [u32; PID_CACHE],
    alive: [bool; PID_CACHE],
    next: usize,
}

impl PidCache {
    fn new() -> Self {
        Self {
            pid: [0; PID_CACHE],
            alive: [false; PID_CACHE],
            next: 0,
        }
    }

    fn alive(&mut self, pid: u32) -> bool {
        if pid == 0 {
            return false; // never an owner; same verdict as pid_alive(0)
        }
        for i in 0..PID_CACHE {
            if self.pid[i] == pid {
                return self.alive[i];
            }
        }
        let alive = pid_alive(pid);
        let i = self.next;
        self.next = (i + 1) % PID_CACHE;
        self.pid[i] = pid;
        self.alive[i] = alive;
        alive
    }
}

/// Whether a candidate's owner is still alive, creator or any holder. A
/// relayed object outlives its creator by design, so a dead creator alone is
/// not enough to reap it.
fn cand_alive(c: &DeadCand, cache: &mut PidCache) -> bool {
    if cache.alive(c.pid) {
        return true;
    }
    let len = (c.holders_len & !HOLDER_OVERFLOW) as usize;
    for &holder in c.holders.iter().take(len.min(MAX_HOLDERS)) {
        if holder != 0 && cache.alive(holder) {
            return true;
        }
    }
    false
}

/// Reap the candidates the caller has already probed: `alive[i]` is the
/// liveness verdict for candidate `i`, computed (possibly outside the lock)
/// by the caller. Each candidate is re-validated under the lock before it is
/// mutated, so a node/slot reused while the lock was dropped is left alone
/// (a reused slot has a new generation; a reused node a different registered
/// pid). Runs NO syscall. Caller holds the region lock. Returns objects freed.
fn reap_dead_locked(r: &Region, cands: &[DeadCand], alive: &[bool]) -> u32 {
    let mut freed = 0;
    for (c, &is_alive) in cands.iter().zip(alive.iter()) {
        if is_alive {
            continue;
        }
        if c.node {
            let node = r.node_ptr(c.index);
            unsafe {
                if (*node).registered != 0 && (*node).pid == c.pid {
                    node_detach(r, c.index);
                    // Detaching only unlinks. A dead waiter never runs its own
                    // cleanup, so nothing else can reach this node: return it
                    // to the pool or the region loses one node per dead waiter
                    // until the table is exhausted and waits start failing.
                    node_free(r, c.index);
                }
            }
        } else {
            let slot = r.slot_ptr(c.index);
            unsafe {
                if (*slot).state == SLOT_USED
                    && (*slot).pid == c.pid
                    && (*slot).generation == c.generation
                    // A holder that registered while the lock was dropped must
                    // not be freed out from under: re-check the snapshot.
                    && (*slot).holders_len == c.holders_len
                    && (*slot).holders == c.holders
                    // Unknown holder set (the list spilled): keep it.
                    && (*slot).holders_len & HOLDER_OVERFLOW == 0
                {
                    (*slot).state = SLOT_FREE;
                    (*slot).generation = (*slot).generation.wrapping_add(1);
                    wake_object(r, c.index);
                    freed += 1;
                }
            }
        }
    }
    freed
}

/// Body of sweep_dead with the region lock already held. The liveness probe
/// runs under the lock here: the caller asked for a sweep, so it may block.
/// Returns the number of objects freed.
fn sweep_dead_locked(r: &Region) -> u32 {
    let mut scan = SweepScan::new(r);
    let mut cands = Vec::new();
    let mut verdicts: Vec<bool> = Vec::new();
    let mut cache = PidCache::new();
    let mut freed = 0;
    loop {
        scan.next_chunk(r, &mut cands);
        if cands.is_empty() {
            break;
        }
        verdicts.clear();
        verdicts.extend(cands.iter().map(|c| cand_alive(c, &mut cache)));
        freed += reap_dead_locked(r, &cands, &verdicts);
    }
    freed
}

/// How often the library runs sweep_dead() opportunistically on its own.
/// The kernel driver reaps a dead process's objects via fd release;
/// userspace has no such hook, so without this a crashed process's
/// objects linger until some caller runs ntsync_sweep_dead().
/// NTSYNC_SWEEP_INTERVAL_SEC overrides the 30 s default; 0 disables.
fn sweep_interval_ns() -> u64 {
    static INTERVAL: OnceLock<u64> = OnceLock::new();
    *INTERVAL.get_or_init(|| {
        let secs = std::env::var("NTSYNC_SWEEP_INTERVAL_SEC")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(30);
        secs * 1_000_000_000
    })
}

/// Max spin iterations per blocking wait. NTSYNC_SPIN_ITERS sets it;
/// default 0 (disabled) so behavior is unchanged unless explicitly
/// opted in. Capped at 100k to bound silly values. Each iteration is a
/// handful of atomic loads (~10-20ns), so e.g. 200 iters costs ~2-4us
/// of CPU in exchange for every avoided sleep/wake pair.
fn spin_max_iters() -> u32 {
    static ITERS: OnceLock<u32> = OnceLock::new();
    *ITERS.get_or_init(|| {
        std::env::var("NTSYNC_SPIN_ITERS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(0)
            .min(100_000)
    })
}

/// Deadline spin tail. When a wait is within this many microseconds of its
/// timeout, spin to the deadline instead of parking the thread: a futex
/// timeout cannot fire closer than the process's timer slack (50 us under
/// bionic, which is what a frame-pacing wait is measured against), so the
/// last stretch is the difference between landing on the frame boundary and
/// landing 60+ us late. NTSYNC_SPIN_TAIL_US sets it; default 0 (disabled)
/// so nothing changes unless a caller opts in. Capped at 2000 us.
fn spin_tail() -> Option<Duration> {
    static TAIL: OnceLock<Option<Duration>> = OnceLock::new();
    *TAIL.get_or_init(|| {
        let us = std::env::var("NTSYNC_SPIN_TAIL_US")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(0)
            .min(2000);
        (us > 0).then(|| Duration::from_micros(us))
    })
}

/// Per-thread spin budget for the tail. A pacing thread wakes every frame, so
/// spinning 150 us of each 1 ms frame is 15% of a core: the allowance is
/// enforced rather than trusted. Once a thread has spent it, its waits park
/// normally until it refills. `NTSYNC_SPIN_TAIL_BUDGET_PCT` sets the
/// percentage of one core the tail may use (default 15, 0 = unlimited). A
/// 150 us tail on a 1 kHz frame costs 15% of a core, so that is the value
/// that sustains the win; lower it to trade the median for CPU.
fn spin_tail_budget_ns() -> u64 {
    static BUDGET: OnceLock<u64> = OnceLock::new();
    *BUDGET.get_or_init(|| {
        let pct = std::env::var("NTSYNC_SPIN_TAIL_BUDGET_PCT")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(15)
            .min(100);
        pct * 10_000_000 // that fraction of a second, in ns
    })
}

/// Largest burst a thread may bank: its allowance for 10 ms of wall time.
/// The bucket must not hold more than a few frames' worth, or a thread
/// spending faster than it refills goes unthrottled for seconds and the
/// rate limit never binds.
#[cfg(test)]
pub(crate) fn test_tail_budget_cap(budget: u64) -> u64 {
    tail_budget_cap(budget)
}

fn tail_budget_cap(budget: u64) -> u64 {
    (budget / 100).max(1)
}

thread_local! {
    /// (tokens, last refill). Starts empty so the first use fills it.
    static TAIL_BUDGET: std::cell::Cell<(u64, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// Charge `want_ns` of spin against this thread's allowance, refilling it by
/// the time elapsed since the last charge. False means the allowance is spent
/// and the caller must park instead of spinning.
fn tail_budget_take(want_ns: u64) -> bool {
    tail_budget(want_ns, true)
}

/// Whether the allowance covers `want_ns`, without charging it. Used to
/// decide the park length before the tail it leads into is spun.
fn tail_budget_can(want_ns: u64) -> bool {
    tail_budget(want_ns, false)
}

fn tail_budget(want_ns: u64, charge: bool) -> bool {
    let budget = spin_tail_budget_ns();
    if budget == 0 {
        return true; // unlimited
    }
    let cap = tail_budget_cap(budget);
    let now = monotonic_ns();
    TAIL_BUDGET.with(|c| {
        let (mut tokens, mut last) = c.get();
        if last == 0 {
            tokens = cap;
            last = now;
        } else {
            let elapsed = now.saturating_sub(last);
            last = now;
            tokens = tokens
                .saturating_add(elapsed.saturating_mul(budget) / 1_000_000_000)
                .min(cap);
        }
        let ok = tokens >= want_ns;
        if ok && charge {
            tokens -= want_ns;
        }
        c.set((tokens, last));
        ok
    })
}

// Adaptive spin credit (per thread): spinning pays off only for waiters
// whose waits are usually satisfied quickly (mutex handoffs); threads
// whose waits routinely time out (e.g. a 1ms vsync poll) would burn CPU
// for nothing. Track outcomes like JVM adaptive spinning: a win adds
// credit, an exhausted spin costs double, and only threads in credit
// spin at all. Starts at 1 so every thread gets one chance to prove it
// wins. Purely thread-local: ~1ns overhead.
thread_local! {
    static SPIN_CREDIT: std::cell::Cell<i32> = const { std::cell::Cell::new(1) };
    static SPIN_PROBE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Iterations this thread may spin now. Threads in credit get the full
/// env-capped budget. Threads out of credit only probe every 8th blocked
/// wait with a small budget, so a workload that changed back to short
/// waits can re-earn credit without always-spin losers (e.g. a 1ms
/// vsync poll) paying a spin tax on every wait forever.
fn spin_budget() -> u32 {
    let max = spin_max_iters();
    if max == 0 {
        return 0;
    }
    SPIN_CREDIT.with(|c| {
        if c.get() > 0 {
            max
        } else {
            SPIN_PROBE.with(|p| {
                let n = p.get().wrapping_add(1);
                p.set(n);
                if n % 8 == 0 { (max / 8).clamp(8, 64) } else { 0 }
            })
        }
    })
}

fn spin_credit_reward() {
    SPIN_CREDIT.with(|c| c.set((c.get() + 1).min(16)));
}

fn spin_credit_penalize() {
    SPIN_CREDIT.with(|c| c.set((c.get() - 2).max(-8)));
}

/// Rate-limited, region-wide opportunistic sweep. Called only from wait
/// paths (before sleeping, and on timeout): threads with slack, so a
/// scan never lands on a latency-critical signaler. Costs ~1ns in the
/// common case: a thread-local counter gates the (vDSO, ~20ns) clock
/// read to every 256th call, and the region-wide interval check is then
/// one relaxed atomic load. The region lock serializes concurrent
/// sweepers; the interval is claimed under it before any scanning.
fn auto_sweep_dead(r: &Region) {
    // Thread-local counter: ~1ns per call, no cacheline sharing. Sweep
    // promptness is unaffected: under any real load 256 signal ops pass
    // in microseconds, and idle threads don't need to sweep.
    thread_local! {
        static CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }
    let n = CALLS.get();
    CALLS.set(n + 1);
    if n & 0xFF != 0 {
        return;
    }
    let interval = sweep_interval_ns();
    if interval == 0 {
        return;
    }
    let cell = unsafe { &*(&r.header().sweep_ns as *const u64 as *const AtomicU64) };
    let now = monotonic_ns();
    if now.saturating_sub(cell.load(Ordering::Relaxed)) < interval {
        return;
    }
    let Ok(g) = r.lock_guard() else { return };
    let now = monotonic_ns();
    if now.saturating_sub(cell.load(Ordering::Relaxed)) < interval {
        return; // another process just swept
    }
    // Claim the interval before doing any work. Stamping on completion
    // instead would leave a pass that finds nothing dead unrecorded, and
    // every thread's 256-call gate could then start a fresh region-wide
    // scan: the claim is what limits the sweep to one pass per interval.
    cell.store(now, Ordering::Release);
    let mut guard = Some(g);
    let mut scan = SweepScan::new(r);
    let mut cands = Vec::new();
    let mut verdicts: Vec<bool> = Vec::new();
    let mut cache = PidCache::new();
    let mut freed = 0u32;
    loop {
        // A chunk of candidates is collected under the lock, but liveness is
        // never probed here: `pid_alive` is a `kill(2)`, and holding the
        // shared region mutex across a whole table of them wedges every
        // other process's ntsync wait/signal for that entire span.
        scan.next_chunk(r, &mut cands);
        if cands.is_empty() {
            break;
        }
        drop(guard.take());
        verdicts.clear();
        verdicts.extend(cands.iter().map(|c| cand_alive(c, &mut cache)));
        guard = r.lock_guard().ok();
        if guard.is_none() {
            break;
        }
        // Re-validated in reap_dead_locked: a node or slot reused while the
        // lock was dropped is skipped instead of mutated.
        freed += reap_dead_locked(r, &cands, &verdicts);
    }
    drop(guard);
    if freed > 0 && debug_enabled() {
        debug_log(&format!("auto-sweep freed {freed} dead objects"));
    }
}

/// alloc_slot, but on a full table reap dead processes' objects first and
/// retry once. Caller holds the region lock.
fn alloc_slot_or_sweep(r: &Region) -> Option<u32> {
    alloc_slot(r).or_else(|| {
        if sweep_dead_locked(r) > 0 {
            alloc_slot(r)
        } else {
            None
        }
    })
}

/// Free all objects created by processes that no longer exist, and purge
/// waiter nodes left behind by dead waiters. Userspace has no fd-close-on-
/// death hook, so callers (e.g. a launcher or wineserver replacement)
/// should run this after a process exits.
pub fn sweep_dead() -> Result<u32, Error> {
    let r = region()?;
    let _g = r.lock_guard().map_err(Error::Init)?;
    Ok(sweep_dead_locked(r))
}

/// Tests: drive one `SweepScan` pass and report (chunks, candidates), so the
/// chunk iteration can be checked without forking dead owners.
#[cfg(test)]
pub(crate) fn test_scan_candidates() -> (usize, usize) {
    let Ok(r) = region() else { return (0, 0) };
    let Ok(_g) = r.lock_guard() else { return (0, 0) };
    let mut scan = SweepScan::new(r);
    let mut out = Vec::new();
    let (mut chunks, mut cands) = (0, 0);
    loop {
        scan.next_chunk(r, &mut out);
        if out.is_empty() {
            break;
        }
        chunks += 1;
        cands += out.len();
    }
    (chunks, cands)
}

/// Tests: slot index mask, so a test can tell whether two handles share a
/// slot.
#[cfg(test)]
pub(crate) const SLOT_MASK: u32 = INDEX_MASK;

/// Tests: build a dead-owner candidate for `handle`'s slot as (index, pid,
/// generation), so a caller can reap it later against a slot that may have
/// been reused. None if the handle is stale.
#[cfg(test)]
pub(crate) fn test_slot_candidate(handle: u32) -> Option<(u32, u32, u32)> {
    let r = region().ok()?;
    let _g = r.lock_guard().ok()?;
    let slot = r.slot(handle)?;
    unsafe { Some((handle & INDEX_MASK, (*slot).pid, handle >> INDEX_BITS)) }
}

/// Tests: (holders_len, holder pids) for a handle, for asserting retain and
/// release bookkeeping.
#[cfg(test)]
pub(crate) fn test_holders(handle: u32) -> Option<(u32, Vec<u32>)> {
    let r = region().ok()?;
    let _g = r.lock_guard().ok()?;
    let slot = r.slot(handle)?;
    unsafe {
        let len = (*slot).holders_len;
        let n = ((len & !HOLDER_OVERFLOW) as usize).min(MAX_HOLDERS);
        let all = (*slot).holders;
        Some((len, all[..n].to_vec()))
    }
}

/// Tests: reap a candidate from `test_slot_candidate` with the caller's
/// liveness verdict. Returns objects freed (0 when the slot was reused).
#[cfg(test)]
pub(crate) fn test_reap_slot_candidate(cand: (u32, u32, u32), alive: bool) -> u32 {
    let Ok(r) = region() else { return 0 };
    let Ok(_g) = r.lock_guard() else { return 0 };
    let c = DeadCand {
        node: false,
        index: cand.0,
        pid: cand.1,
        generation: cand.2,
        holders: [0; MAX_HOLDERS],
        holders_len: 0,
    };
    reap_dead_locked(r, &[c], &[alive])
}

/// Tests: (index, pid) of the first node registered on `handle`'s waiter
/// list, i.e. of the process currently blocked on it.
#[cfg(test)]
pub(crate) fn test_first_waiter(handle: u32) -> Option<(u32, u32)> {
    let r = region().ok()?;
    let _g = r.lock_guard().ok()?;
    let slot = r.slot(handle)?;
    let mut idx = unsafe { (*slot).waiter_head };
    let mut guard = 0;
    while idx != NIL && guard <= NODE_COUNT {
        guard += 1;
        let node = r.node_ptr(idx);
        unsafe {
            if (*node).registered != 0 {
                return Some((idx, (*node).pid));
            }
            idx = (*node).obj_next;
        }
    }
    None
}

/// Tests: reap a waiter-node candidate with the caller's verdict. Returns
/// true if the node was detached (and so returned to the node pool).
#[cfg(test)]
pub(crate) fn test_reap_node_candidate(cand: (u32, u32), alive: bool) -> bool {
    let Ok(r) = region() else { return false };
    let Ok(_g) = r.lock_guard() else { return false };
    let node = r.node_ptr(cand.0);
    let before = unsafe { (*node).registered != 0 };
    let c = DeadCand::node(cand.0, cand.1);
    reap_dead_locked(r, &[c], &[alive]);
    before && unsafe { (*node).registered == 0 }
}

/// Tests: sweep with the region lock held and report what happened to node
/// `idx` within that same locked section: (reaped, back in the pool, freed).
/// A separate sweep-then-inspect pair would race with the other tests, which
/// can allocate the freed node or reap the orphan themselves in between.
#[cfg(test)]
pub(crate) fn test_sweep_and_inspect_node(idx: u32) -> (bool, bool, u32) {
    let Ok(r) = region() else {
        return (false, false, 0);
    };
    let Ok(_g) = r.lock_guard() else {
        return (false, false, 0);
    };
    let registered_before = unsafe { (*r.node_ptr(idx)).registered != 0 };
    let freed = sweep_dead_locked(r);
    let registered_after = unsafe { (*r.node_ptr(idx)).registered != 0 };
    let mut pooled = false;
    let mut cur = r.header().node_free_head;
    let mut guard = 0;
    while cur != NIL && guard <= NODE_COUNT {
        guard += 1;
        if cur == idx {
            pooled = true;
            break;
        }
        cur = unsafe { (*r.node_ptr(cur)).waiter_next };
    }
    (registered_before && !registered_after, pooled, freed)
}

/// Tests: liveness verdicts a fresh `PidCache` returns for `pids`.
#[cfg(test)]
pub(crate) fn test_pid_cache(pids: &[u32]) -> Vec<bool> {
    let mut cache = PidCache::new();
    pids.iter().map(|&pid| cache.alive(pid)).collect()
}

/// Tests: current auto-sweep clock, i.e. when the last pass claimed its
/// interval (0 = never).
#[cfg(test)]
pub(crate) fn test_sweep_clock_ns() -> u64 {
    let Ok(r) = region() else { return 0 };
    let cell = unsafe { &*(&r.header().sweep_ns as *const u64 as *const AtomicU64) };
    cell.load(Ordering::Acquire)
}

/// Tests: force the auto-sweep rate limiter to consider the interval
/// elapsed, so the next signal op sweeps regardless of wall time.
#[cfg(test)]
pub(crate) fn test_reset_sweep_clock() {
    if let Ok(r) = region() {
        let cell = unsafe { &*(&r.header().sweep_ns as *const u64 as *const AtomicU64) };
        cell.store(0, Ordering::Release);
    }
}
