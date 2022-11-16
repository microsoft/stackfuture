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

use core::fmt::Debug;
use core::fmt::Display;
use core::future::Future;
use core::marker::PhantomData;
use core::mem;
use core::mem::MaybeUninit;
use core::pin::Pin;
use core::ptr;
use core::task::Context;
use core::task::Poll;

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "alloc")]
use alloc::boxed::Box;

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
/// but would accommodate many futures besides the simple one we've shown here.
///
/// `StackFuture` ensures when wrapping a future that enough space is available, and
/// it also respects any alignment requirements for the wrapped future. Note that the
/// wrapped future's alignment must be less than or equal to that of the overall
/// `StackFuture` struct.
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
    /// Creates a `StackFuture` from an existing future
    ///
    /// See the documentation on [`StackFuture`] for examples of how to use this.
    ///
    /// The size and alignment requirements are statically checked, so it is a compiler error
    /// to use this with a future that does not fit within the [`StackFuture`]'s size and
    /// alignment requirements.
    ///
    /// The following example illustrates a compile error for a future that is too large.
    /// ```compile_fail
    /// # use stackfuture::StackFuture;
    /// // Fails because the future contains a large array and is therefore too big to fit in
    /// // a 16-byte `StackFuture`.
    /// let f = StackFuture::<_, { 16 }>::from(async {
    ///     let x = [0u8; 4096];
    ///     async {}.await;
    ///     println!("{}", x.len());
    /// });
    /// # #[cfg(miri)] break rust; // FIXME: miri doesn't detect this breakage for some reason...
    /// ```
    ///
    /// The example below illustrates a compiler error for a future whose alignment is too large.
    /// ```compile_fail
    /// # use stackfuture::StackFuture;
    ///
    /// #[derive(Debug)]
    /// #[repr(align(256))]
    /// struct BigAlignment(usize);
    ///
    /// // Fails because the future contains a large array and is therefore too big to fit in
    /// // a 16-byte `StackFuture`.
    /// let f = StackFuture::<_, { 16 }>::from(async {
    ///     let x = BigAlignment(42);
    ///     async {}.await;
    ///     println!("{x:?}");
    /// });
    /// # #[cfg(miri)] break rust; // FIXME: miri doesn't detect this breakage for some reason...
    /// ```
    pub fn from<F>(future: F) -> Self
    where
        F: Future<Output = T> + Send + 'a, // the bounds here should match those in the _phantom field
    {
        // Ideally we would provide this as:
        //
        //     impl<'a, F, const STACK_SIZE: usize> From<F> for  StackFuture<'a, F::Output, { STACK_SIZE }>
        //     where
        //         F: Future + Send + 'a
        //
        // However, libcore provides a blanket `impl<T> From<T> for T`, and since `StackFuture: Future`,
        // both impls end up being applicable to do `From<StackFuture> for StackFuture`.

        // Statically assert that `F` meets all the size and alignment requirements
        #[allow(clippy::let_unit_value)]
        let _ = AssertFits::<F, STACK_SIZE>::ASSERT;

        Self::try_from(future).unwrap()
    }

    /// Attempts to create a `StackFuture` from an existing future
    ///
    /// If the `StackFuture` is not large enough to hold `future`, this function returns an
    /// `Err` with the argument `future` returned to you.
    ///
    /// Panics
    ///
    /// If we cannot satisfy the alignment requirements for `F`, this function will panic.
    pub fn try_from<F>(future: F) -> Result<Self, IntoStackFutureError<F>>
    where
        F: Future<Output = T> + Send + 'a, // the bounds here should match those in the _phantom field
    {
        if Self::has_space_for_val(&future) && Self::has_alignment_for_val(&future) {
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

            Ok(result)
        } else {
            Err(IntoStackFutureError::new::<Self>(future))
        }
    }

    /// Creates a StackFuture from the given future, boxing if necessary
    ///
    /// This version will succeed even if the future is larger than `STACK_SIZE`. If the future
    /// is too large, `from_or_box` will allocate a `Box` on the heap and store the resulting
    /// boxed future in the `StackFuture`.
    ///
    /// The same thing also happens if the wrapped future's alignment is larger than StackFuture's
    /// alignment.
    ///
    /// This function requires the "alloc" crate feature.
    #[cfg(feature = "alloc")]
    pub fn from_or_box<F>(future: F) -> Self
    where
        F: Future<Output = T> + Send + 'a, // the bounds here should match those in the _phantom field
    {
        Self::try_from(future).unwrap_or_else(|err| Self::from(Box::pin(err.into_inner())))
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
    const fn required_space<F>() -> usize {
        mem::size_of::<F>()
    }

    /// Determines whether this `StackFuture` can hold a value of type `F`
    pub const fn has_space_for<F>() -> bool {
        Self::required_space::<F>() <= STACK_SIZE
    }

    /// Determines whether this `StackFuture` can hold the referenced value
    pub const fn has_space_for_val<F>(_: &F) -> bool {
        Self::has_space_for::<F>()
    }

    /// Determines whether this `StackFuture`'s alignment is compatible with the
    /// type `F`.
    pub const fn has_alignment_for<F>() -> bool {
        mem::align_of::<F>() <= mem::align_of::<Self>()
    }

    /// Determines whether this `StackFuture`'s alignment is compatible with the
    /// referenced value.
    pub const fn has_alignment_for_val<F>(_: &F) -> bool {
        Self::has_alignment_for::<F>()
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

struct AssertFits<F, const STACK_SIZE: usize>(PhantomData<F>);

impl<F, const STACK_SIZE: usize> AssertFits<F, STACK_SIZE> {
    const ASSERT: () = {
        if !StackFuture::<F, STACK_SIZE>::has_space_for::<F>() {
            panic!("F is too large");
        }

        if !StackFuture::<F, STACK_SIZE>::has_alignment_for::<F>() {
            panic!("F has incompatible alignment");
        }
    };
}

/// Captures information about why a future could not be converted into a [`StackFuture`]
///
/// It also contains the original future so that callers can still run the future in error
/// recovery paths, such as by boxing the future instead of wrapping it in [`StackFuture`].
pub struct IntoStackFutureError<F> {
    /// The size of the StackFuture we tried to convert the future into
    maximum_size: usize,
    /// The StackFuture's alignment
    maximum_alignment: usize,
    /// The future that was attempted to be wrapped
    future: F,
}

impl<F> IntoStackFutureError<F> {
    fn new<Target>(future: F) -> Self {
        Self {
            maximum_size: mem::size_of::<Target>(),
            maximum_alignment: mem::align_of::<Target>(),
            future,
        }
    }

    /// Returns true if the target [`StackFuture`] was too small to hold the given future.
    pub fn insufficient_space(&self) -> bool {
        self.maximum_size < mem::size_of_val(&self.future)
    }

    /// Returns true if the target [`StackFuture`]'s alignment was too small to accommodate the given future.
    pub fn alignment_too_small(&self) -> bool {
        self.maximum_alignment < mem::align_of_val(&self.future)
    }

    /// Returns the alignment of the wrapped future.
    pub fn required_alignment(&self) -> usize {
        mem::align_of_val(&self.future)
    }

    /// Returns the size of the wrapped future.
    pub fn required_space(&self) -> usize {
        mem::size_of_val(&self.future)
    }

    /// Returns the alignment of the target [`StackFuture`], which is also the maximum alignment
    /// that can be wrapped.
    pub const fn available_alignment(&self) -> usize {
        self.maximum_alignment
    }

    /// Returns the amount of space that was available in the target [`StackFuture`].
    pub const fn available_space(&self) -> usize {
        self.maximum_size
    }

    /// Returns the underlying future that caused this error
    ///
    /// Can be used to try again, either by directly awaiting the future, wrapping it in a `Box`,
    /// or some other method.
    fn into_inner(self) -> F {
        self.future
    }
}

impl<F> Display for IntoStackFutureError<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match (self.alignment_too_small(), self.insufficient_space()) {
            (true, true) => write!(f,
                "cannot create StackFuture, required size is {}, available space is {}; required alignment is {} but maximum alignment is {}",
                self.required_space(),
                self.available_space(),
                self.required_alignment(),
                self.available_alignment()
            ),
            (true, false) => write!(f,
                "cannot create StackFuture, required alignment is {} but maximum alignment is {}",
                self.required_alignment(),
                self.available_alignment()
            ),
            (false, true) => write!(f,
                "cannot create StackFuture, required size is {}, available space is {}",
                self.required_space(),
                self.available_space()
            ),
            // If we have space and alignment, then `try_from` would have succeeded
            (false, false) => unreachable!(),
        }
    }
}

impl<F> Debug for IntoStackFutureError<F> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("IntoStackFutureError")
            .field("maximum_size", &self.maximum_size)
            .field("maximum_alignment", &self.maximum_alignment)
            .field("future", &core::any::type_name::<F>())
            .finish()
    }
}

#[cfg(test)]
mod tests;
