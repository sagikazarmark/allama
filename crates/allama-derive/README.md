# allama-derive

[![crates.io](https://img.shields.io/crates/v/allama-derive?style=flat-square)](https://crates.io/crates/allama-derive)
[![docs.rs](https://img.shields.io/docsrs/allama-derive?style=flat-square)](https://docs.rs/allama-derive)

Derive macros for `allama`'s typed decisions: `#[derive(Decision)]`,
`#[derive(ChoiceSet)]` and `#[derive(Scale)]`.

Use them through `allama` with the `derive` feature, which re-exports them
from `allama::decision` next to the traits they implement. The
`allama::decision::Decision` documentation describes the attributes.

Depend on `allama` with the `derive` feature rather than on this crate
directly: the generated code names `allama`'s items, so the two must be the
same release, which the feature guarantees.

## Tests

`cargo test -p allama-derive` checks the expansions and, in `tests/ui`, the
compile errors the macros emit themselves.

`tests/ui-rustc` holds compile errors that rustc reports about the generated
code (unsatisfied trait bounds, const evaluation). Their wording changes
between compiler releases, and they list `allama`'s `DecisionField`
implementations, so they are ignored by default. They were recorded with
rustc 1.98. Run them, and refresh them after a toolchain or impl change,
with:

```sh
cargo test -p allama-derive ui_rustc -- --ignored
TRYBUILD=overwrite cargo test -p allama-derive ui_rustc -- --ignored
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
