// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (C) 2026 Joshua Tam <297250+joshuatam@users.noreply.github.com>

//! C ABI mirroring the Linux /dev/ntsync ioctl interface, for use by
//! Proton/Wine's ntdll ntsync backend. Objects are identified by u32 handles
//! in place of kernel fds. All functions return 0 on success or a negative
//! errno, exactly like the kernel ioctls.
//!
//! Alertable waits: if `alert` is nonzero it names an event object that
//! aborts the wait; the wait then returns success with `index == count`,
//! exactly like the kernel ioctls.

use std::ffi::CStr;
use std::time::Duration;

use crate::core::{self, Error, WaitOutcome, MAX_WAIT_COUNT};

// errno values (Linux/Android)
pub const EPERM: i32 = 1;
pub const ENOMEM: i32 = 12;
pub const EFAULT: i32 = 14;
pub const EINVAL: i32 = 22;
pub const EOVERFLOW: i32 = 75;
pub const ETIMEDOUT: i32 = 110;
pub const EOWNERDEAD: i32 = 130;

pub const NTSYNC_WAIT_REALTIME: u32 = 0x1;
pub const NTSYNC_MAX_WAIT_COUNT: u32 = MAX_WAIT_COUNT as u32;

/// Same layout as <linux/ntsync.h>.
#[repr(C)]
pub struct ntsync_sem_args {
    pub count: u32,
    pub max: u32,
}

#[repr(C)]
pub struct ntsync_mutex_args {
    pub owner: u32,
    pub count: u32,
}

#[repr(C)]
pub struct ntsync_event_args {
    pub manual: u32,
    pub signaled: u32,
}

#[repr(C)]
pub struct ntsync_wait_args {
    /// Absolute timeout in ns (CLOCK_MONOTONIC, or CLOCK_REALTIME if
    /// NTSYNC_WAIT_REALTIME). U64_MAX = infinite.
    pub timeout: u64,
    /// Pointer to an array of `count` u32 handles.
    pub objs: u64,
    pub count: u32,
    /// Out: index of the object that satisfied the wait.
    pub index: u32,
    pub flags: u32,
    /// In: owner tid used to acquire mutexes.
    pub owner: u32,
    /// In: optional alert event handle (0 = none).
    pub alert: u32,
    pub pad: u32,
}

fn errno(e: Error) -> i32 {
    match e {
        Error::Invalid => -EINVAL,
        Error::Perm => -EPERM,
        Error::Overflow => -EOVERFLOW,
        Error::Init(e) => -e,
    }
}

fn timeout_to_duration(timeout: u64, flags: u32) -> Option<Duration> {
    if timeout == u64::MAX {
        return None;
    }
    let clock = if flags & NTSYNC_WAIT_REALTIME != 0 {
        libc::CLOCK_REALTIME
    } else {
        libc::CLOCK_MONOTONIC
    };
    let mut ts = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe { libc::clock_gettime(clock, &mut ts) };
    let now = ts.tv_sec as u64 * 1_000_000_000 + ts.tv_nsec as u64;
    Some(Duration::from_nanos(timeout.saturating_sub(now)))
}

/// Read the wait arguments into a stack buffer (no heap allocation on the
/// wait path; `count` is capped at NTSYNC_MAX_WAIT_COUNT == 64).
unsafe fn read_wait(
    args: *const ntsync_wait_args,
    handles: &mut [u32; NTSYNC_MAX_WAIT_COUNT as usize],
) -> Result<(usize, u32, u32, Option<Duration>), i32> {
    let args = unsafe { args.as_ref() }.ok_or(-EFAULT)?;
    if args.count == 0 || args.count > NTSYNC_MAX_WAIT_COUNT {
        return Err(-EINVAL);
    }
    let objs = args.objs as *const u32;
    if objs.is_null() {
        return Err(-EFAULT);
    }
    let count = args.count as usize;
    handles[..count].copy_from_slice(unsafe { std::slice::from_raw_parts(objs, count) });
    let timeout = timeout_to_duration(args.timeout, args.flags);
    Ok((count, args.owner, args.alert, timeout))
}

/// Initialize the shared region. `path` may be NULL to use
/// $NTSYNC_SHM_PATH or the built-in default. Idempotent. All other
/// functions auto-initialize on first use.
#[no_mangle]
pub unsafe extern "C" fn ntsync_init(path: *const std::ffi::c_char) -> i32 {
    let path = if path.is_null() {
        None
    } else {
        match unsafe { CStr::from_ptr(path) }.to_str() {
            Ok(p) => Some(p),
            Err(_) => return -EINVAL,
        }
    };
    match core::init(path) {
        Ok(()) => 0,
        Err(e) => errno(e),
    }
}

/// Free all objects whose creator process no longer exists. Returns the
/// number of freed objects, or a negative errno.
#[no_mangle]
pub extern "C" fn ntsync_sweep_dead() -> i32 {
    match core::sweep_dead() {
        Ok(n) => n as i32,
        Err(e) => errno(e),
    }
}

/// NTSYNC_IOC_CREATE_SEM
#[no_mangle]
pub unsafe extern "C" fn ntsync_create_sem(
    out_handle: *mut u32,
    args: *const ntsync_sem_args,
) -> i32 {
    let (out, args) = unsafe { (out_handle.as_mut(), args.as_ref()) };
    let (Some(out), Some(args)) = (out, args) else {
        return -EFAULT;
    };
    match core::create_semaphore(args.count, args.max) {
        Ok(h) => {
            *out = h;
            0
        }
        Err(e) => errno(e),
    }
}

/// NTSYNC_IOC_CREATE_MUTEX
#[no_mangle]
pub unsafe extern "C" fn ntsync_create_mutex(
    out_handle: *mut u32,
    args: *const ntsync_mutex_args,
) -> i32 {
    let (out, args) = unsafe { (out_handle.as_mut(), args.as_ref()) };
    let (Some(out), Some(args)) = (out, args) else {
        return -EFAULT;
    };
    match core::create_mutex(args.owner, args.count) {
        Ok(h) => {
            *out = h;
            0
        }
        Err(e) => errno(e),
    }
}

/// NTSYNC_IOC_CREATE_EVENT
#[no_mangle]
pub unsafe extern "C" fn ntsync_create_event(
    out_handle: *mut u32,
    args: *const ntsync_event_args,
) -> i32 {
    let (out, args) = unsafe { (out_handle.as_mut(), args.as_ref()) };
    let (Some(out), Some(args)) = (out, args) else {
        return -EFAULT;
    };
    match core::create_event(args.manual != 0, args.signaled != 0) {
        Ok(h) => {
            *out = h;
            0
        }
        Err(e) => errno(e),
    }
}

/// close() on an object fd.
#[no_mangle]
pub extern "C" fn ntsync_close(handle: u32) -> i32 {
    if core::close(handle) {
        0
    } else {
        -EINVAL
    }
}

/// Register the calling process as a holder of `handle`, so the sweep keeps
/// the object alive after its creator exits while this process still uses it.
/// Call it on a handle that arrived from another process (wineserver relays
/// objects); idempotent per process.
#[no_mangle]
pub extern "C" fn ntsync_retain(handle: u32) -> i32 {
    match core::retain(handle) {
        Ok(()) => 0,
        Err(core::Error::Invalid) => -EINVAL,
        Err(core::Error::Init(e)) => -e,
        Err(_) => -EINVAL,
    }
}

/// Drop this process's holder entry without closing the object.
#[no_mangle]
pub extern "C" fn ntsync_release(handle: u32) -> i32 {
    match core::release(handle) {
        Ok(()) => 0,
        Err(core::Error::Invalid) => -EINVAL,
        Err(core::Error::Init(e)) => -e,
        Err(_) => -EINVAL,
    }
}

/// NTSYNC_IOC_SEM_RELEASE. On success `count` is overwritten with the
/// previous count.
#[no_mangle]
pub unsafe extern "C" fn ntsync_sem_release(handle: u32, count: *mut u32) -> i32 {
    let count = unsafe { count.as_mut() };
    let Some(count) = count else {
        return -EFAULT;
    };
    match core::sem_release(handle, *count) {
        Ok(prev) => {
            *count = prev;
            0
        }
        Err(e) => errno(e),
    }
}

/// NTSYNC_IOC_SEM_READ
#[no_mangle]
pub unsafe extern "C" fn ntsync_sem_read(handle: u32, args: *mut ntsync_sem_args) -> i32 {
    let args = unsafe { args.as_mut() };
    let Some(args) = args else {
        return -EFAULT;
    };
    match core::read_sem(handle) {
        Ok((count, max)) => {
            *args = ntsync_sem_args { count, max };
            0
        }
        Err(e) => errno(e),
    }
}

/// NTSYNC_IOC_MUTEX_UNLOCK. `args->owner` is input; on success `args->count`
/// is overwritten with the previous recursion count.
#[no_mangle]
pub unsafe extern "C" fn ntsync_mutex_unlock(handle: u32, args: *mut ntsync_mutex_args) -> i32 {
    let args = unsafe { args.as_mut() };
    let Some(args) = args else {
        return -EFAULT;
    };
    match core::mutex_unlock(handle, args.owner) {
        Ok(prev) => {
            args.count = prev;
            0
        }
        Err(e) => errno(e),
    }
}

/// NTSYNC_IOC_MUTEX_KILL. `owner` must be the current owner tid.
#[no_mangle]
pub extern "C" fn ntsync_mutex_kill(handle: u32, owner: u32) -> i32 {
    match core::mutex_kill(handle, owner) {
        Ok(()) => 0,
        Err(e) => errno(e),
    }
}

/// NTSYNC_IOC_MUTEX_READ. Returns -EOWNERDEAD if the mutex is abandoned.
#[no_mangle]
pub unsafe extern "C" fn ntsync_mutex_read(handle: u32, args: *mut ntsync_mutex_args) -> i32 {
    let args = unsafe { args.as_mut() };
    let Some(args) = args else {
        return -EFAULT;
    };
    match core::read_mutex(handle) {
        Ok((count, owner, ownerdead)) => {
            *args = ntsync_mutex_args { owner, count };
            if ownerdead {
                -EOWNERDEAD
            } else {
                0
            }
        }
        Err(e) => errno(e),
    }
}

/// NTSYNC_IOC_EVENT_SET. On success stores the previous state in *prev (if
/// non-NULL), like the kernel ioctl.
#[no_mangle]
pub unsafe extern "C" fn ntsync_event_set(handle: u32, prev: *mut u32) -> i32 {
    match core::event_set(handle) {
        Ok(p) => {
            if let Some(prev) = unsafe { prev.as_mut() } {
                *prev = p;
            }
            0
        }
        Err(e) => errno(e),
    }
}

/// NTSYNC_IOC_EVENT_RESET. On success stores the previous state in *prev (if
/// non-NULL), like the kernel ioctl.
#[no_mangle]
pub unsafe extern "C" fn ntsync_event_reset(handle: u32, prev: *mut u32) -> i32 {
    match core::event_reset(handle) {
        Ok(p) => {
            if let Some(prev) = unsafe { prev.as_mut() } {
                *prev = p;
            }
            0
        }
        Err(e) => errno(e),
    }
}

/// NTSYNC_IOC_EVENT_PULSE. On success stores the previous state in *prev (if
/// non-NULL), like the kernel ioctl.
#[no_mangle]
pub unsafe extern "C" fn ntsync_event_pulse(handle: u32, prev: *mut u32) -> i32 {
    match core::event_pulse(handle) {
        Ok(p) => {
            if let Some(prev) = unsafe { prev.as_mut() } {
                *prev = p;
            }
            0
        }
        Err(e) => errno(e),
    }
}

/// NTSYNC_IOC_EVENT_READ
#[no_mangle]
pub unsafe extern "C" fn ntsync_event_read(handle: u32, args: *mut ntsync_event_args) -> i32 {
    let args = unsafe { args.as_mut() };
    let Some(args) = args else {
        return -EFAULT;
    };
    match core::read_event(handle) {
        Ok((manual, signaled)) => {
            *args = ntsync_event_args {
                manual: manual as u32,
                signaled: signaled as u32,
            };
            0
        }
        Err(e) => errno(e),
    }
}

unsafe fn do_wait(args: *mut ntsync_wait_args, all: bool) -> i32 {
    if args.is_null() {
        return -EFAULT;
    }
    let mut handles = [0u32; NTSYNC_MAX_WAIT_COUNT as usize];
    let (count, owner, alert, timeout) = match unsafe { read_wait(args, &mut handles) } {
        Ok(v) => v,
        Err(e) => return e,
    };
    let handles = &handles[..count];
    let outcome = if all {
        core::wait_all(handles, owner, timeout, alert)
    } else {
        core::wait_any(handles, owner, timeout, alert)
    };
    match outcome {
        WaitOutcome::Signaled { index, owner_dead } => {
            unsafe { (*args).index = index };
            if owner_dead {
                -EOWNERDEAD
            } else {
                0
            }
        }
        WaitOutcome::Timeout => -ETIMEDOUT,
        WaitOutcome::Invalid => -EINVAL,
    }
}

/// NTSYNC_IOC_WAIT_ANY
#[no_mangle]
pub unsafe extern "C" fn ntsync_wait_any(args: *mut ntsync_wait_args) -> i32 {
    unsafe { do_wait(args, false) }
}

/// NTSYNC_IOC_WAIT_ALL
#[no_mangle]
pub unsafe extern "C" fn ntsync_wait_all(args: *mut ntsync_wait_args) -> i32 {
    unsafe { do_wait(args, true) }
}
