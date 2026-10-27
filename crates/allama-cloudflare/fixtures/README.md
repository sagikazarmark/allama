# Fixtures

Response bodies used by `tests/replay.rs`.

Transcribed from the official documentation (2026-10-27), `clairic/`:

- `invoice.json`: the example response of
  <https://developers.cloudflare.com/workers-ai/models/clairic/>, in
  Cloudflare's `{"result": ..}` envelope. Its spans are the page's; the
  replay test's input is written so that they place.

Synthetic (2026-10-27):

- `spans.json`: a response to a non-ASCII French input with a repeated
  value (`tests/replay.rs`'s `spans_place_on_non_ascii_input`), its
  offsets in characters as the blog describes them ("character offsets").
  The second `A-1` is the one `old_ref` names, so the nearest-occurrence
  rule is exercised; `refs` lists both. Written by hand since the model's
  offset unit is undocumented: the live `tests/e2e.rs` checks the same
  input against the endpoint.

Synthetic, from Workers AI's documented error codes
(<https://developers.cloudflare.com/workers-ai/platform/errors/>):

- `unknown_model.json` (HTTP 400, code 7000, a model that does not exist in
  the path), `bad_auth.json` (HTTP 401, code 10000), `bad_field.json`
  (HTTP 400, code 5006, an estimate without bounds; the wording follows
  LiteLLM's capture of the same error).

Re-record with `scripts/capture-fixture.sh` once the model leaves preview;
the live `tests/e2e.rs` checks the same shapes against the endpoint.
