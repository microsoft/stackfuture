# StackFuture

This crate defines a `StackFuture` wrapper around futures that stores the wrapped future in space provided by the caller.
This can be used to emulate dynamic async traits without requiring heap allocation.
Below is an example of how use `StackFuture`:

```rust
use stackfuture::*;

trait PseudoAsyncTrait {
    fn do_something(&self) -> StackFuture<'static, (), { 512 }>;
}

impl PseudoAsyncTrait for i32 {
    fn do_something(&self) -> StackFuture<'static, (), { 512 }> {
        StackFuture::from(async {
            // function body goes here
        })
    }
}

async fn use_dyn_async_trait(x: &dyn PseudoAsyncTrait) {
    x.do_something().await;
}

async fn call_with_dyn_async_trait() {
    use_dyn_async_trait(&42).await;
}
```

This is most useful for cases where async functions in `dyn Trait` objects are needed but storing them in a `Box` is not feasible.
Such cases include embedded programming where allocation is not available, or in tight inner loops where the performance overhead for allocation is unacceptable.
Note that doing this involves tradeoffs.
In the case of `StackFuture`, you must set a compile-time limit on the maximum size of future that will be supported.
If you need to support async functions in `dyn Trait` objects but these constraints do not apply to you, you may be better served by the [`async-trait`](https://crates.io/crates/async-trait) crate.

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
