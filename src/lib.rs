// MIT License
// 
// Copyright (c) 2026-present github/ErisianArchitect
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#[derive(Debug, thiserror::Error)]
#[error("Out of memory.")]
pub struct OutOfMemoryError;

use std::{
    alloc::{
        alloc, dealloc, Layout
    }, cell::UnsafeCell, mem::{
        MaybeUninit,
    }, ptr::NonNull, sync::atomic::{AtomicU8, Ordering}
};

const TAKEN: u8 = 0;
const WAITING: u8 = 1;
const ASSIGNING: u8 = 2;
const READY: u8 = 4;


pub mod strategy {
    pub trait SpawnStrategy {
        type Return<T>;
        fn spawn<F: FnOnce() + Send + 'static, T>(with: T, f: F) -> Self::Return<T>;
    }
    
    pub struct Std;
    
    impl SpawnStrategy for Std {
        type Return<T> = (T, ::std::thread::JoinHandle<()>);
        fn spawn<F: FnOnce() + Send + 'static, T>(with: T, f: F) -> Self::Return<T> {
            let handle = std::thread::spawn(f);
            (with, handle)
        }
    }
    
    #[cfg(feature = "rayon")]
    pub struct Rayon;
    
    #[cfg(feature = "rayon")]
    impl SpawnStrategy for Rayon {
        type Return<T> = T;
        fn spawn<F: FnOnce() + Send + 'static, T>(with: T, f: F) -> Self::Return<T> {
            rayon::spawn(f);
            with
        }
    }
    
    #[cfg(not(feature = "rayon"))]
    pub type DefaultStrategy = Std;
    
    #[cfg(feature = "rayon")]
    pub type DefaultStrategy = Rayon;
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, thiserror::Error)]
pub enum PendingError {
    #[error("Value already taken.")]
    Taken,
    #[error("Waiting for value.")]
    Waiting,
    #[error("Assigning value.")]
    Assigning,
}

#[repr(C)]
struct Inner<R> {
    // we only need AtomicU8 since there can only be one sender and one receiver.
    result: UnsafeCell<MaybeUninit<R>>,
    ref_count: AtomicU8,
    state: AtomicU8,
}

impl<R> Inner<R> {
    const LAYOUT: Layout = Layout::new::<Self>();

    fn alloc_new() -> NonNull<Inner<R>> {
        unsafe {
            let layout = Self::LAYOUT;
            let ptr = alloc(layout) as *mut Self;
            let Some(raw) = NonNull::new(ptr) else {
                ::std::alloc::handle_alloc_error(Self::LAYOUT);
            };
            raw.write(Self {
                result: UnsafeCell::new(MaybeUninit::uninit()),
                // initial reference count of 2 because there is one sender and one receiver.
                ref_count: AtomicU8::new(2),
                state: AtomicU8::new(WAITING),
            });
            raw
        }
    }

    /// Decrements the reference count and drops then deallocs if the reference count becomes 0.
    unsafe fn decrement_ref_count(raw: NonNull<Self>) -> bool {
        let inner_ref = unsafe { raw.as_ref() };
        if inner_ref.ref_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            // SAFETY: The ref count is zero, which means that this was the last instance.
            unsafe { Self::drop_and_dealloc(raw); }
            true
        } else {
            false
        }
    }

    /// This function should only ever be called on the last instance.
    unsafe fn drop_and_dealloc(mut raw: NonNull<Self>) {
        unsafe  {
            let inner_mut = raw.as_mut();
            let state = inner_mut.state.load(Ordering::Acquire);
            match state {
                TAKEN | WAITING => (/* Do nothing, there is no value. */),
                ASSIGNING => unreachable!("Invalid state on cleanup."),
                READY => inner_mut.result.get_mut().assume_init_drop(),
                unknown => unreachable!("Unknown state: {unknown}"),
            }
            dealloc(raw.as_ptr().cast(), Self::LAYOUT);
        }

    }
}

#[derive(Debug)]
pub struct Pending<R: Send + 'static> {
    raw: NonNull<Inner<R>>,
}

#[derive(Debug)]
pub struct Responder<R: Send + 'static> {
    raw: NonNull<Inner<R>>,
}

unsafe impl<R> Send for Pending<R>
where R: Send + 'static {}
unsafe impl<R> Sync for Pending<R>
where R: Send + Sync + 'static {}

unsafe impl<R> Send for Responder<R>
where R: Send + 'static {}
unsafe impl<R> Sync for Responder<R>
where R: Send + Sync + 'static {}

impl<R: Send + 'static> Responder<R> {
    #[must_use]
    #[inline]
    fn from_raw(raw: NonNull<Inner<R>>) -> Self {
        Self { raw }
    }

    #[inline(always)]
    pub fn respond(self, result: R) {
        // SAFETY: self.raw is guaranteed to be convertible to a reference.
        let inner_ref = unsafe {
            self.raw.as_ref()
        };
        // We use store because this is the only thing that can modify the value.
        // The responder is the writer, and the `Pending` is the reader.
        inner_ref.state.store(ASSIGNING, Ordering::Release);
        // SAFETY: dst is guaranteed to be valid for writes, and is properly aligned.
        unsafe {
            inner_ref.result.get().write(MaybeUninit::new(result));
        }
        inner_ref.state.store(READY, Ordering::Release);
    }
}

impl<R: Send + 'static> Pending<R> {

    #[must_use]
    #[inline]
    fn from_raw(raw: NonNull<Inner<R>>) -> Self {
        Self { raw }
    }

    #[must_use]
    #[inline]
    pub fn pair() -> (Self, Responder<R>) {
        let raw = Inner::<R>::alloc_new();
        (
            Self::from_raw(raw),
            Responder::from_raw(raw)
        )
    }
    
    pub fn spawn<S, F>(worker: F) -> S::Return<Self>
    where
        S: strategy::SpawnStrategy,
        F: FnOnce() -> R + Send + 'static,
    {
        let (pending, responder) = Self::pair();
        S::spawn(pending, #[inline(always)] move || {
            responder.respond(worker());
        })
    }

    #[inline]
    pub fn is_ready(&self) -> bool {
        let inner_ref = unsafe {
            self.raw.as_ref()
        };
        inner_ref.state.compare_exchange(READY, READY, Ordering::AcqRel, Ordering::Relaxed).is_ok()
    }

    #[must_use]
    #[inline]
    pub fn try_recv(&self) -> std::result::Result<R, PendingError> {
        unsafe {
            let inner_ref = self.raw.as_ref();
            match inner_ref.state.compare_exchange(READY, TAKEN, Ordering::AcqRel, Ordering::Relaxed) {
                Ok(_) => Ok(inner_ref.result.get().read().assume_init()),
                Err(TAKEN) => Err(PendingError::Taken),
                Err(WAITING) => Err(PendingError::Waiting),
                // The purpose of returning PendingError::Assigning is so that the caller can choose to call try_recv immediately after.
                // Pseudo-code:
                //  use ::std::thread::sleep;
                //  use ::std::time::Duration;
                //  let pending = Pending::spawn(|| {
                //      return some_expensive_function();
                //  });
                //  loop {
                //      // sleep in 50ms increments because the response is expected to take a few seconds, but we need a tight window for the response.
                //      sleep(Duration::from_millis(50));
                //      match pending.try_recv() {
                //          Ok(result) => return result,
                //          Err(PendingError::Assigning) => {
                //              // very brief sleep (OS may wake up later)
                //              sleep(Duration::from_nanos(10));
                //              let Ok(result) = pending.try_recv() else {
                //                  continue;
                //              };
                //              return result;
                //          }
                //          Err(PendingError::Taken) => unreachable!("This loop owns the pending instance. Nothing else can receive from it."),
                //          Err(_) => continue,
                //      }
                // }
                Err(ASSIGNING) => Err(PendingError::Assigning),
                Err(_) => unreachable!("Corrupted state; should not be possible."),
            }
        }
    }
}

impl<R: Send + 'static> Drop for Pending<R> {
    fn drop(&mut self) {
        // SAFETY: Pending<R> owns 1 reference count. It must be decremented on drop.
        unsafe {
            Inner::<R>::decrement_ref_count(self.raw);
        }
    }
}

impl<R: Send + 'static> Drop for Responder<R> {
    fn drop(&mut self) {
        // SAFETY: Responder<R> owns 1 reference count. It must be decremented on drop.
        unsafe {
            Inner::<R>::decrement_ref_count(self.raw);
        }
    }
}

#[must_use]
#[inline]
pub fn pair<R>() -> (Pending<R>, Responder<R>)
where R: Send + 'static {
    Pending::pair()
}

#[must_use]
#[inline]
pub fn spawn<S, R, F>(worker: F) -> S::Return<Pending<R>>
where
    S: strategy::SpawnStrategy,
    R: Send + 'static,
    F: FnOnce() -> R + Send + 'static, 
{
    Pending::spawn::<S, _>(worker)
}

#[must_use]
#[inline]
pub fn spawn_std<R, F>(worker: F) -> <strategy::Std as strategy::SpawnStrategy>::Return<Pending<R>>
where
    R: Send + 'static,
    F: FnOnce() -> R + Send + 'static,
{
    spawn::<strategy::Std, R, F>(worker)
}

#[cfg(feature = "rayon")]
#[must_use]
#[inline]
pub fn spawn_rayon<R, F>(worker: F) -> <strategy::Rayon as strategy::SpawnStrategy>::Return<Pending<R>> {
    spawn::<strategy::Rayon, R, F>(worker)
}

#[test]
fn main_test() {
    use std::thread::sleep;
    use std::time::Duration;
    let (pending, join_handle) = spawn::<strategy::Std, _, _>(|| {
        sleep(Duration::from_secs(3));
        0xDEADBEEFu32
    });

    sleep(Duration::from_millis(2750));
    if let Ok(result) = pending.try_recv() {
        println!("Result: 0x{result:0X}");
    } else {
        join_handle.join().expect("Failed to join.");
        let Ok(result) = pending.try_recv() else {
            panic!("Thread was joined with no response.");
        };
        println!("Result after join: 0x{result:0X}");
    }
}