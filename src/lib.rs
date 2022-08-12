// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

//! This crate defines a `StackFuture` wrapper around futures that stores the wrapped
//! future in space provided by the caller. This can be used to emulate dyn async traits
//! without requiring heap allocation.
//!
//! For more details, see the documentation on the [`StackFuture`] struct.

// std is needed to run tests, but otherwise we don't need it.
#![cfg_attr(not(test), no_std)]
#![warn(missing_docs)]

use core::future::Future;
use core::marker::PhantomData;
use core::mem;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::ptr;
use core::task::Context;
use core::task::Poll;

/// A wrapper that stores a future in space allocated by the container
///
/// Often this space comes from the calling function's stack, but it could just
/// as well come from some other allocation.
///
/// A `StackFuture` can be used to emulate async functions in dyn Trait objects.
/// For example:
///
/// ```
/// # use stackfuture::*;
/// trait PseudoAsyncTrait {
///     fn do_something(&self) -> StackFuture<'_, (), { 512 }>;
/// }
///
/// impl PseudoAsyncTrait for i32 {
///     fn do_something(&self) -> StackFuture<'_, (), { 512 }> {
///         StackFuture::from(async {
///             // function body goes here
///         })
///     }
/// }
///
/// async fn use_dyn_async_trait(x: &dyn PseudoAsyncTrait) {
///     x.do_something().await;
/// }
///
/// async fn call_with_dyn_async_trait() {
///     use_dyn_async_trait(&42).await;
/// }
/// ```
///
/// This example defines `PseudoAsyncTrait` with a single method `do_something`.
/// The `do_something` method can be called as if it were declared as
/// `async fn do_something(&self)`. To implement `do_something`, the easiest thing
/// to do is to wrap the body of the function in `StackFuture::from(async { ... })`,
/// which creates an anonymous future for the body and stores it in a `StackFuture`.
///
/// Because `StackFuture` does not know the size of the future it wraps, the maximum
/// size of the future must be specified in the `STACK_SIZE` parameter. In the example
/// here, we've used a stack size of 512, which is probably much larger than necessary
/// but would accomodate many futures besides the simple one we've shown here.
///
/// `StackFuture` ensures when wrapping a future that enough space is available, and
/// it also respects any alignment requirements for the wrapped future. Note that the
/// wrapped future's alignment must be less than or equal to that of the overall
/// `StackFuture` struct.
///
/// The following example would panic because the future is too large:
/// ```should_panic
/// # use stackfuture::*;
/// fn large_stack_future() -> StackFuture<'static, u8, { 4 }> {
///     StackFuture::from(async {
///         let mut buf = [0u8; 256];
///         fill_buf(&mut buf).await;
///         buf[0]
///     })
/// }
///
/// async fn fill_buf(buf: &mut [u8]) {
///     buf[0] = 42;
/// }
///
/// let future = large_stack_future();
/// ```
///
/// The following example would panic because the alignment requirements can't be met:
/// ```should_panic
/// # use stackfuture::*;
/// #[repr(align(4096))]
/// struct LargeAlignment(i32);
///
/// fn large_alignment_future() -> StackFuture<'static, i32, { 8192 }> {
///     StackFuture::from(async {
///         let mut buf = LargeAlignment(0);
///         fill_buf(&mut buf).await;
///         buf.0
///     })
/// }
///
/// async fn fill_buf(buf: &mut LargeAlignment) {
///     buf.0 = 42;
/// }
///
/// let future = large_alignment_future();
/// ```
#[repr(C)] // Ensures the data first does not have any padding before it in the struct
pub struct StackFuture<'a, T, const STACK_SIZE: usize> {
    /// An array of bytes that is used to store the wrapped future.
    data: [MaybeUninit<u8>; STACK_SIZE],
    /// Since the type of `StackFuture` does not know the underlying future that it is wrapping,
    /// we keep a manual vtable that serves pointers to Poll::poll and Drop::drop. These are
    /// generated and filled in by `StackFuture::from`.
    ///
    /// This field stores a pointer to the poll function wrapper.
    poll_fn: fn(this: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T>,
    /// Stores a pointer to the drop function wrapper
    ///
    /// See the documentation on `poll_fn` for more details.
    drop_fn: fn(this: &mut Self),
    /// StackFuture can be used similarly to a `dyn Future`. We keep a PhantomData
    /// here so the type system knows this.
    _phantom: PhantomData<dyn Future<Output = T> + Send + 'a>,
}

impl<'a, T, const STACK_SIZE: usize> StackFuture<'a, T, { STACK_SIZE }> {
    /// Creates a StackFuture from an existing future
    ///
    /// See the documentation on `StackFuture` for examples of how to use this.
    pub fn from<F>(future: F) -> Self
    where
        F: Future<Output = T> + Send + 'a, // the bounds here should match those in the _phantom field
    {
        if mem::align_of::<F>() > mem::align_of::<Self>() {
            panic!(
                "cannot create StackFuture, required alignment is {} but maximum alignment is {}",
                mem::align_of::<F>(),
                mem::align_of::<Self>()
            )
        }

        if Self::has_space_for_val(&future) {
            let mut result = StackFuture {
                data: [MaybeUninit::uninit(); STACK_SIZE],
                poll_fn: Self::poll_inner::<F>,
                drop_fn: Self::drop_inner::<F>,
                _phantom: PhantomData,
            };

            // Ensure result.data is at the beginning of the struct so we don't need to do
            // alignment adjustments.
            assert_eq!(result.data.as_ptr() as usize, &result as *const _ as usize);

            // SAFETY: result.as_mut_ptr returns a pointer into result.data, which is an
            // uninitialized array of bytes. result.as_mut_ptr ensures the returned pointer
            // is correctly aligned, and the if expression we are in ensures the buffer is
            // large enough.
            //
            // Because `future` is bound by `'a` and `StackFuture` is also bound by `'a`,
            // we can be sure anything that `future` closes over will also outlive `result`.
            unsafe { result.as_mut_ptr::<F>().write(future) };

            result
        } else {
            panic!(
                "cannot create StackFuture, required size is {}, available space is {}",
                mem::size_of::<F>(),
                STACK_SIZE
            );
        }
    }

    /// A wrapper around the inner future's poll function, which we store in the poll_fn field
    /// of this struct.
    fn poll_inner<F: Future>(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<F::Output> {
        self.as_pin_mut_ref::<F>().poll(cx)
    }

    /// A wrapper around the inner future's drop function, which we store in the drop_fn field
    /// of this struct.
    fn drop_inner<F>(&mut self) {
        // SAFETY: *this.as_mut_ptr() was previously written as type F
        unsafe { ptr::drop_in_place(self.as_mut_ptr::<F>()) }
    }

    /// Returns a pointer into self.data that meets the alignment requirements for type `F`
    ///
    /// Before writing to the returned pointer, the caller must ensure that self.data is large
    /// enough to hold F and any required padding.
    fn as_mut_ptr<F>(&mut self) -> *mut F {
        assert!(Self::has_space_for::<F>());
        // SAFETY: Self is laid out so that the space for the future comes at offset 0.
        // This is checked by an assertion in Self::from. Thus it's safe to cast a pointer
        // to Self into a pointer to the wrapped future.
        unsafe { mem::transmute(self) }
    }

    /// Returns a pinned mutable reference to a type F stored in self.data
    fn as_pin_mut_ref<F>(self: Pin<&mut Self>) -> Pin<&mut F> {
        // SAFETY: `StackFuture` is only created by `StackFuture::from`, which
        // writes an `F` to `self.as_mut_ptr(), so it's okay to cast the `*mut F`
        // to an `&mut F` with the same lifetime as `self`.
        //
        // For pinning, since self is already pinned, we know the wrapped future
        // is also pinned.
        //
        // This function is only doing pointer arithmetic and casts, so we aren't moving
        // any pinned data.
        unsafe { self.map_unchecked_mut(|this| &mut *this.as_mut_ptr()) }
    }

    /// Computes how much space is required to store a value of type `F`
    fn required_space<F>() -> usize {
        mem::size_of::<F>()
    }

    /// Determines whether this `StackFuture` can hold a value of type `F`
    fn has_space_for<F>() -> bool {
        Self::required_space::<F>() <= STACK_SIZE
    }

    /// Determines whether this `StackFuture` can hold the referenced value
    fn has_space_for_val<F>(_: &F) -> bool {
        Self::has_space_for::<F>()
    }
}

impl<'a, T, const STACK_SIZE: usize> Future for StackFuture<'a, T, { STACK_SIZE }> {
    type Output = T;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        // SAFETY: This is doing pin projection. We unpin self so we can
        // access self.poll_fn, and then re-pin self to pass it into poll_in.
        // The part of the struct that needs to be pinned is data, since it
        // contains a potentially self-referential future object, but since we
        // do not touch that while self is unpinned and we do not move self
        // while unpinned we are okay.
        unsafe {
            let this = self.get_unchecked_mut();
            (this.poll_fn)(Pin::new_unchecked(this), cx)
        }
    }
}

impl<'a, T, const STACK_SIZE: usize> Drop for StackFuture<'a, T, { STACK_SIZE }> {
    fn drop(&mut self) {
        (self.drop_fn)(self);
    }
}

#[cfg(test)]
mod tests {
    use crate::StackFuture;
    use core::task::Poll;
    use futures::channel::mpsc;
    use futures::executor::block_on;
    use futures::pin_mut;
    use futures::Future;
    use futures::SinkExt;
    use futures::Stream;
    use futures::StreamExt;
    use std::sync::Arc;
    use std::task::Context;
    use std::task::Wake;
    use std::thread;

    #[test]
    fn create_and_run() {
        // A smoke test. Make sure we can create a future, run it, and get a value out.
        let f = StackFuture::<'_, _, 8>::from(async { 5 });
        assert_eq!(block_on(f), 5);
    }

    /// A type that is uninhabited and therefore can never be constructed.
    enum Never {}

    /// A future whose poll function always returns `Pending`
    ///
    /// Used to force a suspend point so we can test behaviors with suspended futures.
    struct SuspendPoint;
    impl Future for SuspendPoint {
        type Output = Never;

        fn poll(
            self: std::pin::Pin<&mut Self>,
            _cx: &mut std::task::Context<'_>,
        ) -> std::task::Poll<Self::Output> {
            Poll::Pending
        }
    }

    /// A waker that doesn't do anything.
    ///
    /// Needed so we can create a context and manually call poll.
    struct Waker;
    impl Wake for Waker {
        fn wake(self: std::sync::Arc<Self>) {
            unimplemented!()
        }
    }

    #[test]
    fn destructor_runs() {
        // A test to ensure `StackFuture` correctly calls the destructor of the underlying future.
        //
        // We do this by creating a manually implemented future whose destructor sets a boolean
        // indicating it ran. We create such a value (the `let _ = DropMe(&mut destructed))` line
        //  below), then use `SuspendPoint.await` to suspend the future.
        //
        // The driver code creates a context and then calls poll once on the future so that the
        // DropMe object will be created. We then let the future go out of scope so the destructor
        // will run.
        let mut destructed = false;
        let _poll_result = {
            let f = async {
                struct DropMe<'a>(&'a mut bool);
                impl Drop for DropMe<'_> {
                    fn drop(&mut self) {
                        *self.0 = true;
                    }
                }
                let _ = DropMe(&mut destructed);
                SuspendPoint.await
            };
            let f = StackFuture::<'_, _, 32>::from(f);

            let waker = Arc::new(Waker).into();
            let mut cx = Context::from_waker(&waker);
            pin_mut!(f);
            f.poll(&mut cx)
        };
        assert!(destructed);
    }

    #[test]
    fn test_alignment() {
        // A test to make sure we store the wrapped future with the correct alignment

        #[repr(align(8))]
        struct BigAlignment(u32);

        impl Future for BigAlignment {
            type Output = Never;

            fn poll(self: std::pin::Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
                Poll::Pending
            }
        }
        let mut f = StackFuture::<'_, _, 1016>::from(BigAlignment(42));
        assert!(is_aligned(f.as_mut_ptr::<BigAlignment>(), 8));
    }

    /// Returns whether `ptr` is aligned with the given alignment
    ///
    /// `alignment` must be a power of two.
    fn is_aligned<T>(ptr: *mut T, alignment: usize) -> bool {
        (ptr as usize) & (alignment - 1) == 0
    }

    #[test]
    fn stress_drop_sender() {
        // Regression test for #9

        const ITER: usize = if cfg!(miri) { 10 } else { 10000 };

        fn list() -> impl Stream<Item = i32> {
            let (tx, rx) = mpsc::channel(1);
            thread::spawn(move || {
                block_on(send_one_two_three(tx));
            });
            rx
        }

        for _ in 0..ITER {
            let v: Vec<_> = block_on(list().collect());
            assert_eq!(v, vec![1, 2, 3]);
        }
    }

    fn send_one_two_three(mut tx: mpsc::Sender<i32>) -> StackFuture<'static, (), 512> {
        StackFuture::from(async move {
            for i in 1..=3 {
                tx.send(i).await.unwrap();
            }
        })
    }
}
