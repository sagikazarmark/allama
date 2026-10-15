# Fixtures

Response bodies used by `tests/replay.rs`.

Copied verbatim from the official API documentation (2026-10-02):

- `ollama/label.json`: the example response of
  <https://docs.ollama.com/api/systemone>.
- `typesafe/department.json`, `frustration.json`, `is_urgent.json`: the
  per-answer-type example responses of <https://docs.typesafe.ai/api>.

Copied verbatim from the official API documentation (2026-10-15):

- `typesafe/box.json`: the example response of
  <https://docs.typesafe.ai/primitives/box>, a Box (locate) answer from
  `jev-1.14.0`.

Recorded from the live API (2026-10-02):

- `typesafe/recorded_triage.json`: `jev-latest` (served as `jev-1.13.0`)
  answering the request in `tests/replay.rs`'s `typesafe_recorded_response`.
  Probabilities come back rounded to two decimals, keys sorted.

Recorded from Cloudflare Workers AI (2026-10-13), `cloudflare/`:

- `triage.json`: `@cf/cloudflare/clef` answering the example request of
  <https://developers.cloudflare.com/workers-ai/models/clef/>
  (`tests/replay.rs`'s `clef_triage_request`), in Cloudflare's
  `{"result": ..}` envelope.
- `flash.json`: `@cf/cloudflare/clef-flash` answering one yes/no question.
- `image.json`: `@cf/cloudflare/clef` answering a choice about a 64×64 red
  PNG sent as `{"content_type", "base64"}`.
- Errors: `unknown_model.json` (HTTP 400, code 7000, a model that does not
  exist in the path), `bad_auth.json` (HTTP 401, code 10000),
  `bad_name.json` (HTTP 422, code 5012, a question name with a space) and
  `mismatch.json` (HTTP 422, code 5012, the selector `clef-flash` sent to
  `@cf/cloudflare/clef`).

Synthetic:

- `ollama/triage.json`: a response to the request of the multi-question
  example in <https://docs.ollama.com/capabilities/decision>, plus a `label`
  question. That guide only states two results: a refund probability of
  0.9989 and an urgency score of 0.8308. The other values are made up to be
  consistent with Ollama's definitions:
  - the urgency probabilities average to 0.8308;
  - confidences follow `1 − H(p)/ln N`;
  - the label answer reuses the distribution of the documented label
    example, for a billing ticket;
  - usage counts the shared input once per question (3 × 177 input
    tokens), and one output token per question plus the shared-prefix
    primer Ollama counts for three or more questions.

  Replace it with a recorded response once one is available.
