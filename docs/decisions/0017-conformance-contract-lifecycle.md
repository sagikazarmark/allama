# ADR 0017: Conformance contracts are added in minor releases, and required only when adapters pass them

- Status: accepted

## Context

`allama-conformance` gives provider authors aggregates
(`assert_chat_adapter`, `assert_embedding_adapter`,
`assert_decision_adapter`) that run every contract against an adapter from
one `*UnderTest` value. A contract added to an aggregate runs in every
adapter's suite, third-party suites included, without a change to them.
So when a contract is added, and when it can fail, is part of the crate's
compatibility promise.

A contract can also demand more than anyone has decided is worth it.
`decoding::assert_vendor_fields_kept` expects an unknown field at every
object level of a response to survive decoding; every bundled adapter and
the guide's `text_gateway` template fail it. Making it required would turn
every adapter written from the guide red, for a rule never announced.

The core and the conformance crate also move apart: `allama-conformance`
depends on `allama` with a caret requirement, so mismatched patch versions
are a legal pair.

## Decision

1. **Contracts are added only in minor releases** (`0.x` to `0.(x+1)`). A
   patch release never adds a contract to an aggregate, never makes a
   provisional contract required, and never adds an `ACCOUNTED_FIELDS`
   line or probe. So `cargo update` within a minor release never turns an
   adapter's suite red, and a new portable request field never reaches an
   adapter built against an older release without the probe that checks
   it. Such an adapter would ignore the field without a warning, breaking
   the rule that portable features are never dropped silently (ADR 0003).
   Before 1.0 this puts new contracts and new request fields in the same
   minor release as breaking changes; that is accepted.
2. **A contract in an aggregate is provisional until adapters pass it.**
   A contract may join an aggregate as provisional
   (`Contract::is_provisional`). It never fails a suite: without its input
   the aggregate prints "not run: needs ...", and when it fails the
   aggregate prints the failure and passes. A contract that needs an input
   older suites do not give always starts provisional. A later minor
   release may make it required, and its release notes say so; no release
   is promised in advance. A contract becomes required only when the
   bundled adapters and the `text_gateway` template pass it and the
   provider-authors guide says how to pass it.
3. **A contract outside the aggregates may be provisional with no
   promise.** A function no aggregate runs may be labelled provisional in
   its docs. It is opt-in: an author calls it to see what the adapter
   does, what it checks may change in any minor release, and it may never
   join an aggregate. `Contract` has a variant only for contracts an
   aggregate runs. `assert_vendor_fields_kept` is one: keeping unknown
   fields at the top level and in stream frames is settled, keeping them
   below the top level is not, and that needs its own decision (kept, or
   kept and warned about) before the contract can join an aggregate.
4. **A standalone part of an aggregated contract may be provisional.** The
   function that runs the part alone is provisional; the aggregate that
   runs the whole contract is not, and stays the check to call. The
   embedding and decision parts of the transparent wrapper contract
   (`assert_transparent_embedding_wrapper`,
   `assert_transparent_decision_wrapper`,
   `assert_transparent_embedding_data_path`,
   `assert_transparent_decision_data_path`) are provisional this way, so
   the transparency functions can be consolidated in a breaking release.
5. **A version skew fails the audit; it does not panic.** When
   `allama-conformance` meets an `ACCOUNTED_FIELDS` line it cannot read (a
   probe it does not know, an account other than `probe:` or `exempt:`),
   `accounted_fields()` leaves the line out, and
   `assert_request_fields_accounted` runs every probe it knows and then
   fails `RequestFieldsAccounted`, naming the lines and asking for an
   upgrade of `allama-conformance` to the version of `allama`. It never
   passes a field it did not check. Under point 1 this happens only when
   the two crates are at different minor versions, or a release breaks
   the rule.

## Consequences

- An adapter's suite fails only on contracts the bundled adapters already
  pass and the guide explains. The ecosystem is not committed to a cost,
  such as keeping nested vendor fields, before anyone has decided it is
  worth it.
- `Contract` is `#[non_exhaustive]`, so a contract joins an aggregate
  additively: first as provisional, then as required.
- A mismatched caret pair gives a test failure that says what to do,
  instead of a panic before the audit runs.
- New portable request fields wait for a minor release before 1.0.
  Allowing them in patch releases would need the core to report fields an
  adapter has not declared, which it does not.
