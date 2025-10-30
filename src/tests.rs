// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

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
fn test_size_failure() {
    async fn fill_buf(buf: &mut [u8]) {
        buf[0] = 42;
    }

    let f = async {
        let mut buf = [0u8; 256];
        fill_buf(&mut buf).await;
        buf[0]
    };

    match StackFuture::<_, 4>::try_from(f) {
        Ok(_) => panic!("conversion to StackFuture should not have succeeded"),
        Err(e) => assert!(e.insufficient_space()),
    }
}

#[test]
fn test_alignment() {
    // A test to make sure we store the wrapped future with the correct alignment

    #[repr(align(8))]
    #[allow(dead_code)]
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

#[test]
fn test_alignment_failure() {
    // A test to make sure we store the wrapped future with the correct alignment

    #[repr(align(256))]
    #[allow(dead_code)]
    struct BigAlignment(u32);

    impl Future for BigAlignment {
        type Output = Never;

        fn poll(self: std::pin::Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }
    match StackFuture::<'_, _, 1016>::try_from(BigAlignment(42)) {
        Ok(_) => panic!("conversion to StackFuture should not have succeeded"),
        Err(e) => assert!(e.alignment_too_small()),
    }
}

#[cfg(feature = "alloc")]
#[test]
fn test_boxed_alignment() {
    // A test to make sure we store the wrapped future with the correct alignment

    #[repr(align(256))]
    struct BigAlignment(u32);

    impl Future for BigAlignment {
        type Output = Never;

        fn poll(self: std::pin::Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Pending
        }
    }
    StackFuture::<'_, _, 1016>::from_or_box(BigAlignment(42));
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

#[test]
fn try_from() {
    let big_future = StackFuture::<_, 1000>::from(async {});

    match StackFuture::<_, 10>::try_from(big_future) {
        Ok(_) => panic!("try_from should not have succeeded"),
        Err(big_future) => {
            assert!(StackFuture::<_, 1500>::try_from(big_future.into_inner()).is_ok())
        }
    };
}

#[cfg(feature = "alloc")]
#[test]
fn from_or_box() {
    let big_future = StackFuture::<_, 1000>::from(async {});
    StackFuture::<_, 32>::from_or_box(big_future);
}
