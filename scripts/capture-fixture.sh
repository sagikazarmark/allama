#!/usr/bin/env bash
# Record a real provider exchange as a conformance fixture.
#
# Usage:
#   scripts/capture-fixture.sh raw <url> <request.json> <output-file> [--header 'Name: value' ...]
#   scripts/capture-fixture.sh <provider> <request.json> <output-file>
#
# raw POSTs <request.json> as-is to <url>, with `content-type:
# application/json` unless a --header sets another one. It works with any
# endpoint and needs only bash and curl, so you can copy this file into your
# own repository. In a header value, `$NAME` or `${NAME}` is replaced with
# environment variable NAME; single-quote the header so the shell leaves it
# alone, and the secret stays out of shell history:
#
#   scripts/capture-fixture.sh raw https://api.example.com/v1/chat \
#     request.json fixtures/stream.sse --header 'Authorization: Bearer $EXAMPLE_API_KEY'
#
# <provider> is a shortcut for a bundled provider: openai, openai-responses,
# anthropic, gemini, or gemini-stream. It reads OPENAI_API_KEY,
# ANTHROPIC_API_KEY, or GEMINI_API_KEY. Gemini takes the model from the
# body's "model" field (removed before sending, which needs jq).
#
# The body is sent as-is: set the streaming flag in it ("stream": true for
# OpenAI and Anthropic) for a streaming capture. Variables also come from a
# .env file in the repository root (the parent of this script's directory).
# Only the response body is written, whatever the HTTP status: request and
# response headers, which carry credentials and account identifiers, are
# never saved. Every value taken from the environment counts as a secret,
# and the script refuses to write a capture that contains one. Headers reach
# curl through a file descriptor, so secrets stay out of the process list.
set -euo pipefail

help() {
  sed -n '2,/^set -euo/{/^set -euo/d;s/^# \{0,1\}//;p;}' "$0"
  exit 0
}

usage() {
  sed -n '4,6s/^# \{0,1\}//p' "$0" >&2
  echo "Run with --help for details." >&2
  exit 2
}

die() {
  echo "capture-fixture: $*" >&2
  exit 1
}

# Expand `$NAME` and `${NAME}` in header $1 from the environment. Sets
# `expanded` and appends each value used to `secrets`. Messages name the
# header, never its value.
expand_header() {
  local rest=$1 out="" name value
  local header_name=${1%%:*}
  while [[ $rest == *'$'* ]]; do
    out+=${rest%%\$*}
    rest=${rest#*\$}
    if [[ $rest =~ ^\{([A-Za-z_][A-Za-z0-9_]*)\}(.*)$ ]] || [[ $rest =~ ^([A-Za-z_][A-Za-z0-9_]*)(.*)$ ]]; then
      name=${BASH_REMATCH[1]}
      rest=${BASH_REMATCH[2]}
    else
      die "header '$header_name' has a '\$' that does not name a variable"
    fi
    value=${!name-}
    [[ -n $value ]] || die "$name is not set (header '$header_name'); export it or add it to .env"
    secrets+=("$value")
    out+=$value
  done
  expanded=$out$rest
}

# Add header $1 (a `Name: value` template) to the request.
add_header() {
  local token='^[!#%&*+.^_|~0-9A-Za-z-]+:'
  [[ $1 =~ $token ]] || die "not a 'Name: value' header: '${1%%:*}'"
  expand_header "$1"
  [[ $expanded != *$'\n'* && $expanded != *$'\r'* ]] || die "header '${1%%:*}' contains a line break"
  headers+=("$expanded")
  shopt -s nocasematch
  [[ $1 == content-type:* ]] && has_content_type=1
  shopt -u nocasematch
  return 0
}

[[ ${1-} == -h || ${1-} == --help ]] && help
[[ $# -ge 3 ]] || usage
mode=$1
shift

root=$(cd "$(dirname "$0")/.." && pwd)
if [[ -f "$root/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  . "$root/.env"
  set +a
fi

headers=()
secrets=()
has_content_type=""
body=""

# Single quotes keep `$NAME` for expand_header.
# shellcheck disable=SC2016
case "$mode" in
  raw)
    [[ $# -ge 3 ]] || usage
    url=$1 request=$2 output=$3
    shift 3
    while [[ $# -gt 0 ]]; do
      case "$1" in
        -H | --header)
          [[ $# -ge 2 ]] || usage
          add_header "$2"
          shift 2
          ;;
        --header=*)
          add_header "${1#--header=}"
          shift
          ;;
        *) usage ;;
      esac
    done
    ;;
  openai | openai-responses | anthropic | gemini | gemini-stream)
    [[ $# -eq 2 ]] || usage
    request=$1 output=$2
    case "$mode" in
      openai)
        url=https://api.openai.com/v1/chat/completions
        add_header 'Authorization: Bearer $OPENAI_API_KEY'
        ;;
      openai-responses)
        url=https://api.openai.com/v1/responses
        add_header 'Authorization: Bearer $OPENAI_API_KEY'
        ;;
      anthropic)
        url=https://api.anthropic.com/v1/messages
        add_header 'x-api-key: $ANTHROPIC_API_KEY'
        add_header 'anthropic-version: 2023-06-01'
        ;;
      gemini | gemini-stream)
        model=$(jq -r '.model // empty' "$request")
        [[ -n $model ]] || die "the request body needs a \"model\" field for Gemini captures (it is removed before sending)"
        method=generateContent
        query=""
        if [[ "$mode" == gemini-stream ]]; then
          method=streamGenerateContent
          query="?alt=sse"
        fi
        url="https://generativelanguage.googleapis.com/v1beta/models/$model:$method$query"
        add_header 'x-goog-api-key: $GEMINI_API_KEY'
        body=$(jq -c 'del(.model)' "$request")
        ;;
    esac
    ;;
  *)
    echo "capture-fixture: unknown mode: $mode" >&2
    usage
    ;;
esac

[[ -f $request ]] || die "no such request body: $request"
[[ -n $has_content_type ]] || headers+=('content-type: application/json')

tmp="$output.tmp"
trap 'rm -f "$tmp"' EXIT
if [[ -n $body ]]; then
  status=$(printf '%s' "$body" | curl -sSN -o "$tmp" -w '%{http_code}' \
    -H @<(printf '%s\n' "${headers[@]}") --data-binary @- "$url")
else
  status=$(curl -sSN -o "$tmp" -w '%{http_code}' \
    -H @<(printf '%s\n' "${headers[@]}") --data-binary @"$request" "$url")
fi
touch "$tmp"

for secret in ${secrets[@]+"${secrets[@]}"}; do
  if grep -qF -- "$secret" "$tmp"; then
    die "refusing to write a capture that contains a secret from the environment"
  fi
done
mv "$tmp" "$output"
echo "wrote $output ($(wc -c <"$output" | tr -d ' ') bytes, HTTP $status); review it before committing"
