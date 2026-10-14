#!/usr/bin/env bash
# Print a vendor's newest chat model id, from its models API.
#
# Usage:
#   scripts/newest-model.sh <anthropic|gemini|openai>
#
# The sampling drift check (crates/allama-examples/tests/newest_models.rs)
# runs against the model this prints:
#
#   ANTHROPIC_NEWEST_MODEL=$(scripts/newest-model.sh anthropic) \
#     cargo test -p allama-examples --test newest_models anthropic -- --ignored
#
# - anthropic: the first model listed, which is the most recently released.
# - gemini: the highest numbered `gemini-X.Y-flash` or `-pro` model.
# - openai: the most recently created `gpt-N[.M][-variant]` text model;
#   pro, chat, codex, audio, realtime, image, search, speech and live
#   variants are skipped.
#
# It reads ANTHROPIC_API_KEY, GEMINI_API_KEY or OPENAI_API_KEY, also from a
# .env file in the repository root (the parent of this script's directory),
# and needs curl and jq. The key reaches curl through a file descriptor, so
# it stays out of the process list.
set -euo pipefail

help() {
  sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p;}' "$0"
  exit 0
}

usage() {
  sed -n '4,5s/^# \{0,1\}//p' "$0" >&2
  echo "Run with --help for details." >&2
  exit 2
}

die() {
  echo "newest-model: $*" >&2
  exit 1
}

# GET $1 with the headers that follow, and print the body.
get() {
  local url=$1
  shift
  curl -fsS -H @<(printf '%s\n' "$@") "$url"
}

[[ ${1-} == -h || ${1-} == --help ]] && help
[[ $# -eq 1 ]] || usage

root=$(cd "$(dirname "$0")/.." && pwd)
if [[ -f "$root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$root/.env"
  set +a
fi

case "$1" in
  anthropic)
    [[ -n ${ANTHROPIC_API_KEY-} ]] || die "ANTHROPIC_API_KEY is not set"
    get "https://api.anthropic.com/v1/models?limit=1" \
      "x-api-key: $ANTHROPIC_API_KEY" "anthropic-version: 2023-06-01" |
      jq -er '.data[0].id'
    ;;
  gemini)
    [[ -n ${GEMINI_API_KEY-} ]] || die "GEMINI_API_KEY is not set"
    get "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000" \
      "x-goog-api-key: $GEMINI_API_KEY" |
      jq -er '[.models[].name | ltrimstr("models/") | select(test("^gemini-[0-9]+(\\.[0-9]+)?-(flash|pro)$"))]
        | sort_by(capture("^gemini-(?<v>[0-9.]+)").v | split(".") | map(tonumber)) | last'
    ;;
  openai)
    [[ -n ${OPENAI_API_KEY-} ]] || die "OPENAI_API_KEY is not set"
    get "https://api.openai.com/v1/models" "Authorization: Bearer $OPENAI_API_KEY" |
      jq -er '[.data[] | select(.id | test("^gpt-[0-9]+(\\.[0-9]+)*(-[a-z]+)?$"))
        | select(.id | test("-(pro|chat|codex|audio|realtime|image|search|transcribe|tts|live)$") | not)]
        | sort_by(.created) | last | .id'
    ;;
  *)
    echo "newest-model: unknown provider: $1" >&2
    usage
    ;;
esac
