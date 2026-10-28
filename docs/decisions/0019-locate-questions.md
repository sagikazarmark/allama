# ADR 0019: Locate questions answer with one bounding box

- Status: accepted

## Context

Decision models have answered three kinds of question: yes/no, a choice
among options, and a score on a scale. Each answer is a distribution over
a fixed set of candidates. TypeSafe's Jev 1.14 adds a fourth, `box`: given
an image, return where a described target is, as the four edges of one box.
TypeSafe calls it a preview.

No other decision backend has it yet. Ollama's System One endpoint and
Cloudflare's Clef have no such question. Chat models locate objects too,
but each in its own convention: Gemini, for example, answers in text with
`[ymin, xmin, ymax, xmax]` scaled to 0–1000.

AGENTS.md asks for a neutral domain model before a provider's feature
reaches the core. So this ADR settles what a portable "where is it"
question is, and keeps TypeSafe's wire names out of it.

## Decision

1. **The question is *locate*: find one target and answer with its
   bounding box.** In computer vision, localization means one box for one
   object, and detection means any number of boxes. Locate is
   localization. `Question::locate(instructions)` builds a
   `LocateQuestion` whose instructions name the target and, when the
   request has several images, which image to search. Its kind is
   `QuestionKind::Locate` and its capability `DecisionCapability::Locate`.
   "Box" is TypeSafe's wire name and stays in the adapter.
2. **The answer is the box's edges as fractions of the image.**
   `LocateAnswer { left, top, right, bottom }` measures each edge from the
   image's top-left corner: `left` and `right` as fractions of the width,
   `top` and `bottom` as fractions of the height, with
   `0 ≤ left ≤ right ≤ 1` and `0 ≤ top ≤ bottom ≤ 1`. Fractions hold at
   any resolution, so the caller multiplies by the pixel size it holds.
   Edges, not corners or center and size, are the form the one backend
   uses, and they convert to any other form without loss. An adapter
   rescales a backend's own convention, for example 0–1000, into
   fractions.
3. **A locate answer carries no probability.** Unlike the other three
   kinds, it is not a distribution over candidates. A provider's
   confidence statistic stays provider data, as it does for the other
   kinds. There is no "not found" answer either: a model asked where
   something absent is still answers with a box. Callers who need to know
   whether the target is there ask a yes/no question in the same request.
4. **A locate question needs an image.** `DecisionRequest::validate`
   refuses a request with a locate question and no images, before
   anything is sent. Every backend would refuse it.
   `DecisionResponse::validate` refuses an answer whose edges are not
   finite, lie outside `[0, 1]`, or are out of order.
5. **Support is a capability answer (ADR 0015).** The System One adapter
   leaves `Locate` `Unknown`, since only some models serve it (Jev 1.14,
   not 1.13). A request is sent, and the provider decides. Another backend
   that can locate maps its own form onto the same answer.

## Consequences

- Typed decisions take a `LocateAnswer` field like any other answer, with
  instructions from the attribute or the doc comment.
- An adapter whose wire format cannot carry locate questions refuses
  them with `Error::Unsupported` naming the question, as it does for any
  other kind it lacks.
- Locating several things takes one locate question each. Detection, where
  the answer is any number of boxes, would be a different kind, with a
  different answer, if a backend ever offers it.
