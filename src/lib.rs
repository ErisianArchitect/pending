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
        Layout, alloc, dealloc, handle_alloc_error
    }, cell::UnsafeCell, marker::PhantomData, mem::{
        MaybeUninit,
        transmute,
    }, ptr::NonNull, sync::atomic::{AtomicU8, Ordering}
};

/// The [strategy] module contains the [strategy::SpawnStrategy] trait, as well as
/// markers for different strategies. These strategies define how workers are spawned.
pub mod strategy {
    
    pub trait SpawnStrategy {
        type Return<T>;
        /// Spawn a worker, passing in part of the return value.
        /// 
        /// The `with` parameter is for the [`Pending<R>`] that is paired with the
        /// responder created for the worker. This allows strategies to be made that return
        /// different return types thats [`Pending<R>`].
        /// 
        /// [`Pending<R>`]: crate::Pending<R>
        fn spawn<F: FnOnce() + Send + 'static, T>(with: T, worker: F) -> Self::Return<T>;
    }
    
    /// Uses [std::thread::Thread] and returns `(`[`Pending<R>`]`,`[`Thread`]`)`.
    /// 
    /// [`Pending<R>`]: crate::Pending<R>
    /// [`Thread`]: std::thread::Thread
    pub struct Std;
    
    impl SpawnStrategy for Std {
        type Return<T> = (T, ::std::thread::JoinHandle<()>);
        #[inline]
        fn spawn<F: FnOnce() + Send + 'static, T>(with: T, f: F) -> Self::Return<T> {
            let handle = std::thread::spawn(f);
            (with, handle)
        }
    }
    
    /// Uses [rayon] to spawn a worker thread. Returns [`Pending<R>`].
    /// 
    /// [`Pending<R>`]: crate::Pending
    #[cfg(feature = "rayon")]
    pub struct Rayon;
    
    #[cfg(feature = "rayon")]
    impl SpawnStrategy for Rayon {
        type Return<T> = T;
        #[inline]
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



/// Represents the state that the response is in.
#[repr(u8, align(1))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum State {
    /// The response has been taken.
    Taken = 0,
    /// Waiting for a response.
    Waiting = 1,
    /// The response is being written.
    Assigning = 2,
    /// The response is ready.
    Ready = 3,
}

/// Represents the reason that [Pending::try_recv] was unable to receive a response.
#[repr(u8, align(1))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Reason {
    /// The response has already been taken.
    Taken = 0,
    /// Still waiting for a response.
    Waiting = 1,
    /// The response is being written. Used to determine when you should try to receive again immediately after.
    /// (You should time your delay according to how long you think it will take for the response to be written)
    /// 
    /// # Example:
    /// ```rust, ignore
    ///  use ::std::thread::sleep;
    ///  use ::std::time::Duration;
    ///  let pending = Pending::spawn(|| {
    ///      // Explicit return is, of course, not required. It's used to illustrate that a value is being returned, which
    ///      // isn't so obvious when using implicit return.
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

/// A wrapper around [AtomicU8] that stores [State] instead of [u8].
#[repr(transparent)]
#[derive(Debug)]
struct AtomicState(AtomicU8);

// AtomicState is a minimal implementation. It has only what it needs.
impl AtomicState {
    /// Create a new [AtomicState] initialized to [State::Waiting].
    #[inline(always)]
    const fn new() -> Self {
        Self(AtomicU8::new(unsafe { transmute(State::Waiting) }))
    }
    
    /// Stores the [State] with [Release] ordering.
    /// 
    /// [Release]: Ordering::Release
    #[inline(always)]
    fn store(&self, state: State) {
        self.0.store(unsafe { transmute(state) }, Ordering::Release);
    }
    
    /// Load the [State] with [Acquire] ordering.
    /// 
    /// [Acquire]: Ordering::Acquire
    #[must_use]
    #[inline(always)]
    fn load(&self) -> State {
        unsafe { transmute(self.0.load(Ordering::Acquire)) }
    }
    
    /// Perform a compare_exchange with [AcqRel] ordering for `success`, and
    /// [Relaxed] ordering for `failure`.
    /// 
    /// See [AtomicU8::compare_exchange].
    /// 
    /// [AcqRel]: Ordering::AcqRel
    /// [Relaxed]: Ordering::Relaxed
    #[inline(always)]
    fn compare_exchange(&self, current: State, new: State) -> Result<State, State> {
        let current: u8 = unsafe { transmute(current) };
        let new: u8 = unsafe { transmute(new) };
        match self.0.compare_exchange(current, new, Ordering::AcqRel, Ordering::Relaxed) {
            Ok(previous) => Ok(unsafe { transmute(previous) }),
            Err(previous) => Err(unsafe { transmute(previous) }),
        }
    }
    
    /// Checks if the [State] is [Ready] using a compare_exchange equality check.
    /// 
    /// [Ready]: State::Ready
    #[inline(always)]
    fn is_ready(&self) -> bool {
        self.compare_exchange(
            State::Ready,
            State::Ready,
        ).is_ok()
    }
    
    /// If the current state is [Ready], replace it with [Taken] and return Ok on success.
    /// 
    /// Returns the [Reason] on failure.
    /// 
    /// [Ready]: State::Ready
    /// [Taken]: State::Taken
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

/// The shared object between a [Pending] and [Responder] pair.
#[repr(C)]
struct Inner<R> {
    /// Stores the response given by [Responder::respond].
    response: UnsafeCell<MaybeUninit<R>>,
    // we only need AtomicU8 since there can only be one sender and one receiver.
    /// Stores the reference count. Since handles come in pairs, this will only have a maximum value of 2.
    ref_count: AtomicU8,
    /// The current state of the response.
    state: AtomicState,
}

impl<R> Inner<R> {
    const LAYOUT: Layout = Layout::new::<Self>();

    fn alloc_new() -> NonNull<Inner<R>> {
        unsafe {
            let layout = Self::LAYOUT;
            let ptr = alloc(layout).cast();
            let Some(raw) = NonNull::new(ptr) else {
                handle_alloc_error(Self::LAYOUT);
            };
            raw.write(Self {
                response: UnsafeCell::new(MaybeUninit::uninit()),
                // initial reference count of 2 because there is one sender and one receiver.
                ref_count: AtomicU8::new(2),
                state: AtomicState::new(),
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
        // temporary scope for the mutable reference (`inner_mut`) to live in so it doesn't have any chance to cause conflicts.
        {
            // SAFETY: `drop_and_dealloc` is only called on the last handle during drop, so
            // we are certain that the pointer points to initialized and alive memory, and
            // we also know that we have exclusive access to it.
            let inner_mut = unsafe { raw.as_mut() };
            let state = inner_mut.state.load();
            use State::*;
            match state {
                Taken | Waiting => (/* Do nothing, there is no value. */),
                // It shouldn't be possible for the responder to be dropped in the middle of assignment.
                // The only reason unreachable_unchecked is not used here is because it can't be guaranteed
                // that it won't happen.
                Assigning => unreachable!("Invalid state on cleanup."),
                // SAFETY: `drop_and_dealloc` is only called on the last instance during drop, and a `Ready` state indicates that
                // a value has been assigned but not taken.
                Ready => unsafe { inner_mut.response.get_mut().assume_init_drop() },
            }
        }
        // SAFETY: we have exclusive access to the pointer, the pointer is valid, and the memory is initialized.
        unsafe { raw.drop_in_place() };
        // SAFETY: pointer is exclusive, valid, and points to initialized memory.
        unsafe { dealloc(raw.as_ptr().cast(), Self::LAYOUT); }

    }
}

mod marker {
    pub trait HandleType: Send + Sized + 'static {}
    
    #[derive(Debug)]
    pub enum Sender {}
    #[derive(Debug)]
    pub enum Receiver {}
    
    impl HandleType for Sender {}
    impl HandleType for Receiver {}
    impl HandleType for () {}
}

#[repr(transparent)]
#[derive(Debug)]
struct Handle<R, Type: marker::HandleType = ()> {
    raw: NonNull<Inner<R>>,
    _phantom: PhantomData<*const Type>,
}

type SendHandle<R> = Handle<R, marker::Sender>;
type RecvHandle<R> = Handle<R, marker::Receiver>;
type SpawnOutput<R, S> = <S as strategy::SpawnStrategy>::Return<Pending<R>>;

impl<R: Send + 'static> Handle<R, ()> {
    #[must_use]
    #[inline(always)]
    fn pair() -> (Handle<R, marker::Sender>, Handle<R, marker::Receiver>) {
        let raw = Inner::<R>::alloc_new();
        (
            Handle::from_raw(raw),
            Handle::from_raw(raw),
        )
    }
}

impl<R: Send + 'static, Type: marker::HandleType> Handle<R, Type> {
    #[must_use]
    #[inline(always)]
    const fn from_raw(raw: NonNull<Inner<R>>) -> Self {
        Self {
            raw,
            _phantom: PhantomData,
        }
    }
}

impl<R> Handle<R, marker::Receiver>
where R: Send + 'static {
    #[inline(always)]
    fn is_ready(&self) -> bool {
        let inner_ref = unsafe {
            self.raw.as_ref()
        };
        inner_ref.state.is_ready()
    }
    
    #[inline(always)]
    fn try_recv(&self) -> Result<R, Reason> {
        // SAFETY: handle is valid while self is alive.
        let inner_ref = unsafe { self.raw.as_ref() };
        inner_ref.state
            .take_if_ready()
            .map(move |_| {
                // SAFETY: Handle now has exclusive access to the response.
                unsafe { inner_ref.response.get().read().assume_init() }
            })
    }
}

impl<R> Handle<R, marker::Sender>
where R: Send + 'static {
    #[inline(always)]
    fn respond(self, result: R) {
        // It might appear as if there is an aliasing bug in this code since `Inner` is
        // repr(C), and `response` is the first field, but `response` is UnsafeCell, which
        // prevents aliasing bugs.
        
        // SAFETY: self.handle is guaranteed to be convertible to a reference.
        let inner_ref = unsafe {
            self.raw.as_ref()
        };
        // We use store because this is the only thing that can modify the value
        // while the state is `Waiting`, and the state is always waiting until `respond`
        // modifies it right here, and `respond` consumes `Responder`, so it can't be
        // called again.
        // The responder is the writer, and the `Pending` is the reader.
        inner_ref.state.store(State::Assigning);
        // SAFETY: dst is guaranteed to be valid for writes, and is properly aligned.
        unsafe {
            inner_ref.response.get().write(MaybeUninit::new(result));
        }
        inner_ref.state.store(State::Ready);
    }
}

impl<R, Type: marker::HandleType> Drop for Handle<R, Type> {
    fn drop(&mut self) {
        // SAFETY: Handle is only ever created in pairs, and the ref count always starts at 2, so
        // dropping each handle in the pair will bring the ref count to zero. The only way this
        // operation becomes unsafe is if an extra handle were created from the same raw pointer.
        unsafe {
            Inner::<R>::decrement_ref_count_and_maybe_free(self.raw);
        }
    }
}

/// A oneshot single-producer/single-consumer receiver that works in tandem with [Responder<R>].
#[repr(transparent)]
#[derive(Debug)]
pub struct Pending<R: Send + 'static> {
    handle: RecvHandle<R>,
}

/// A oneshot single-producer/single-consumer sender that works in tandem with [Pending<R>].
#[repr(transparent)]
#[derive(Debug)]
pub struct Responder<R: Send + 'static> {
    handle: SendHandle<R>,
}

impl<R: Send + 'static> Responder<R> {
    #[inline(always)]
    pub fn respond(self, result: R) {
        self.handle.respond(result);
    }
}

impl<R: Send + 'static> Pending<R> {
    #[must_use]
    #[inline]
    pub fn pair() -> (Responder<R>, Pending<R>) {
        let (send_handle, receive_handle) = Handle::<R>::pair();
        (
            Responder { handle: send_handle },
            Pending { handle: receive_handle },
        )
    }
    
    #[must_use]
    pub fn spawn<S, F>(worker: F) -> S::Return<Self>
    where
        S: strategy::SpawnStrategy,
        F: FnOnce() -> R + Send + 'static,
    {
        let (responder, pending) = Self::pair();
        S::spawn(pending, #[inline(always)] move || {
            responder.respond(worker());
        })
    }

    #[must_use]
    #[inline]
    pub fn is_ready(&self) -> bool {
        self.handle.is_ready()
    }

    #[inline]
    pub fn try_recv(&self) -> Result<R, Reason> {
        self.handle.try_recv()
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
pub fn pair<R>() -> (Responder<R>, Pending<R>)
where R: Send + 'static {
    Pending::pair()
}

#[must_use]
#[inline]
pub fn spawn<S, R, F>(worker: F) -> SpawnOutput<R, S>
where
    S: strategy::SpawnStrategy,
    R: Send + 'static,
    F: FnOnce() -> R + Send + 'static, 
{
    Pending::spawn::<S, _>(worker)
}

#[must_use]
#[inline]
pub fn spawn_std<R, F>(worker: F) -> SpawnOutput<R, strategy::Std>
where
    R: Send + 'static,
    F: FnOnce() -> R + Send + 'static,
{
    spawn::<strategy::Std, R, F>(worker)
}

#[cfg(feature = "rayon")]
#[must_use]
#[inline]
pub fn spawn_rayon<R, F>(worker: F) -> SpawnOutput<R, strategy::Rayon> {
    spawn::<strategy::Rayon, R, F>(worker)
}