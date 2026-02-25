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

use std::{
    alloc::{
        alloc, dealloc, Layout
    }, cell::UnsafeCell, mem::{
        MaybeUninit,
        transmute,
    }, ptr::NonNull, sync::atomic::{AtomicU8, Ordering}
};

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



#[repr(u8, align(1))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum State {
    Taken = 0,
    Waiting = 1,
    Assigning = 2,
    Ready = 3,
}

#[repr(u8, align(1))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reason {
    Taken = 0,
    Waiting = 1,
    /// The purpose of [Reason::Assigning] is so that the caller can choose to call [Pending::try_recv] immediately after.
    /// Pseudo-code:
    /// ```rust, ignore
    ///  use ::std::thread::sleep;
    ///  use ::std::time::Duration;
    ///  let pending = Pending::spawn(|| {
    ///      return some_expensive_function();
    ///  });
    ///  loop {
    ///      // sleep in 50ms increments because the response is expected to
    ///      // take a few seconds, but we need a tight window for the response.
    ///      sleep(Duration::from_millis(50));
    ///      match pending.try_recv() {
    ///          Ok(result) => return result,
    ///          Err(Reason::Assigning) => {
    ///              // very brief sleep (OS may wake up later)
    ///              sleep(Duration::from_nanos(10));
    ///              let Ok(result) = pending.try_recv() else {
    ///                  continue;
    ///              };
    ///              return result;
    ///          }
    ///          Err(Reason::Taken) => unreachable!(
    ///              "This loop owns the pending instance. Nothing else can receive from it."
    ///          ),
    ///          Err(_) => continue,
    ///      }
    /// }
    /// ```
    Assigning = 2,
}

#[repr(transparent)]
#[derive(Debug)]
struct AtomicState(AtomicU8);

impl AtomicState {
    #[inline(always)]
    const fn new(state: State) -> Self {
        Self(AtomicU8::new(unsafe { transmute(state) }))
    }
    
    #[inline(always)]
    fn store(&self, state: State) {
        self.0.store(unsafe { transmute(state) }, Ordering::Release);
    }
    
    #[must_use]
    #[inline(always)]
    fn load(&self) -> State {
        unsafe { transmute(self.0.load(Ordering::Acquire)) }
    }
    
    #[inline(always)]
    fn compare_exchange(&self, current: State, new: State) -> Result<State, State> {
        let current: u8 = unsafe { transmute(current) };
        let new: u8 = unsafe { transmute(new) };
        match self.0.compare_exchange(current, new, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(previous) => Ok(unsafe { transmute(previous) }),
            Err(previous) => Err(unsafe { transmute(previous) }),
        }
    }
    
    #[inline(always)]
    fn is_ready(&self) -> bool {
        self.compare_exchange(
            State::Ready,
            State::Ready,
        ).is_ok()
    }
    
    #[inline(always)]
    fn take_if_ready(&self) -> Result<(), Reason> {
        match self.compare_exchange(State::Ready, State::Taken) {
            Ok(_) => Ok(()),
            // SAFETY: Both State and Reason have the same layout. Reason is identical to State except that
            // it has no `Ready` variant. The `Ready` variant is the last discriminant, so this operation is perfectly
            // safe since the Err branch will never be `State::Ready`.
            Err(reason) => Err(unsafe { transmute(reason) })
        }
    }
}

#[repr(C)]
struct Inner<R> {
    result: UnsafeCell<MaybeUninit<R>>,
    // we only need AtomicU8 since there can only be one sender and one receiver.
    ref_count: AtomicU8,
    state: AtomicState,
}

impl<R> Inner<R> {
    const LAYOUT: Layout = Layout::new::<Self>();

    fn alloc_new() -> NonNull<Inner<R>> {
        unsafe {
            let layout = Self::LAYOUT;
            let ptr = alloc(layout).cast();
            let Some(raw) = NonNull::new(ptr) else {
                ::std::alloc::handle_alloc_error(Self::LAYOUT);
            };
            raw.write(Self {
                result: UnsafeCell::new(MaybeUninit::uninit()),
                // initial reference count of 2 because there is one sender and one receiver.
                ref_count: AtomicU8::new(2),
                state: AtomicState::new(State::Waiting),
            });
            raw
        }
    }

    /// Decrements the reference count and drops then deallocs if the reference count becomes 0.
    unsafe fn decrement_ref_count_and_maybe_free(raw: NonNull<Self>) {
        let inner_ref = unsafe { raw.as_ref() };
        if inner_ref.ref_count.fetch_sub(1, Ordering::AcqRel) == 1 {
            // SAFETY: The ref count is zero, which means that this was the last instance.
            unsafe { Self::drop_and_dealloc(raw); }
        }
    }

    /// This function should only ever be called when the ref_count reaches zero (the return value of `fetch_sub(1)` is `1`)
    unsafe fn drop_and_dealloc(mut raw: NonNull<Self>) {
        // temporary scope for the mutable reference to live in so it doesn't have any chance to cause conflicts.
        {
            // SAFETY: `drop_and_dealloc` is only called on the last handle during drop, so
            // we are certain that the pointer points to initialized and alive memory, and
            // we also know that we have exclusive access to it.
            let inner_mut = unsafe { raw.as_mut() };
            let state = inner_mut.state.load();
            use State::*;
            match state {
                Taken | Waiting => (/* Do nothing, there is no value. */),
                Assigning => unreachable!("Invalid state on cleanup."),
                // SAFETY: `drop_and_dealloc` is only called on the last instance during drop, and a `Ready` state indicates that
                // a value has been assigned but not taken.
                Ready => unsafe { inner_mut.result.get_mut().assume_init_drop() },
            }
        }
        // SAFETY: we have exclusive access to the pointer, the pointer is valid, and the memory is initialized.
        unsafe { raw.drop_in_place() };
        // SAFETY: pointer is exclusive, valid, and points to initialized memory.
        unsafe { dealloc(raw.as_ptr().cast(), Self::LAYOUT); }

    }
}

#[repr(transparent)]
#[derive(Debug)]
struct Handle<R> {
    raw: NonNull<Inner<R>>,
}

impl<R: Send + 'static> Handle<R> {
    #[must_use]
    #[inline(always)]
    fn pair() -> (Handle<R>, Handle<R>) {
        let raw = Inner::<R>::alloc_new();
        (
            Handle { raw },
            Handle { raw },
        )
    }
    
    #[inline(always)]
    unsafe fn as_ref<'a>(&'a self) -> &'a Inner<R> {
        unsafe { self.raw.as_ref() }
    }
}

impl<R> Drop for Handle<R> {
    fn drop(&mut self) {
        // SAFETY: Handle is only ever created in pairs, and the ref count always starts at 2, so
        // dropping each handle in the pair will bring the ref count to zero. The only way this
        // operation becomes unsafe is if an extra handle were created from the same raw pointer.
        unsafe {
            Inner::<R>::decrement_ref_count_and_maybe_free(self.raw);
        }
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Pending<R: Send + 'static> {
    handle: Handle<R>,
}

#[repr(transparent)]
#[derive(Debug)]
pub struct Responder<R: Send + 'static> {
    handle: Handle<R>,
}

impl<R: Send + 'static> Responder<R> {
    
    #[inline(always)]
    fn new(handle: Handle<R>) -> Self {
        Self { handle }
    }

    #[inline(always)]
    pub fn respond(self, result: R) {
        // SAFETY: self.handle is guaranteed to be convertible to a reference.
        let inner_ref = unsafe {
            self.handle.as_ref()
        };
        // We use store because this is the only thing that can modify the value.
        // The responder is the writer, and the `Pending` is the reader.
        inner_ref.state.store(State::Assigning);
        // SAFETY: dst is guaranteed to be valid for writes, and is properly aligned.
        unsafe {
            inner_ref.result.get().write(MaybeUninit::new(result));
        }
        inner_ref.state.store(State::Ready);
    }
}

impl<R: Send + 'static> Pending<R> {

    #[must_use]
    #[inline]
    fn new(handle: Handle<R>) -> Self {
        Self { handle }
    }

    #[must_use]
    #[inline]
    pub fn pair() -> (Pending<R>, Responder<R>) {
        let (recv, send) = Handle::<R>::pair();
        (
            Pending::new(recv),
            Responder::new(send),
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
            self.handle.as_ref()
        };
        inner_ref.state.is_ready()
    }

    #[must_use]
    #[inline]
    pub fn try_recv(&self) -> Result<R, Reason> {
        // SAFETY: handle is valid while self is alive.
        let inner_ref = unsafe { self.handle.as_ref() };
        inner_ref.state
            .take_if_ready()
            .map(move |_| {
                // SAFETY: Pending now has exclusive access to the result.
                unsafe { inner_ref.result.get().read().assume_init() }
            })
    }
}

unsafe impl<R> Send for Handle<R>
where R: Send + 'static {}
unsafe impl<R> Sync for Handle<R>
where R: Send + Sync + 'static {}

unsafe impl<R> Send for Pending<R>
where R: Send + 'static {}
unsafe impl<R> Sync for Pending<R>
where R: Send + Sync + 'static {}

unsafe impl<R> Send for Responder<R>
where R: Send + 'static {}
unsafe impl<R> Sync for Responder<R>
where R: Send + Sync + 'static {}

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