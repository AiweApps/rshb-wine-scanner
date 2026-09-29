#!/usr/bin/env bash

# Runs the unchanged participant_test.sh against the public HTTPS API with one
# guest Bearer token. The token lives only in a 0600 curl config inside a 0700
# temporary directory; a curl shim adds it solely to the exact predict URL.

set -uo pipefail

origin="https://wines.aiweapps.com"
images_dir=""
manifest=""
output="predictions.jsonl"

usage() {
  printf '%s\n' \
    "Usage: $0 --images-dir DIR --manifest FILE [--output FILE] [--origin https://HOST[:PORT]]" \
    "" \
    "Obtains one guest token via POST ORIGIN/auth/guest/token and runs" \
    "participant_test.sh with --endpoint ORIGIN/v1/eval/predict." \
    "Default origin: $origin"
}

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --images-dir)
      [ "$#" -ge 2 ] || die "--images-dir requires a value"
      images_dir="$2"
      shift 2
      ;;
    --manifest)
      [ "$#" -ge 2 ] || die "--manifest requires a value"
      manifest="$2"
      shift 2
      ;;
    --output)
      [ "$#" -ge 2 ] || die "--output requires a value"
      output="$2"
      shift 2
      ;;
    --origin)
      [ "$#" -ge 2 ] || die "--origin requires a value"
      origin="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1"
      ;;
  esac
done

[[ "$origin" =~ ^https://[A-Za-z0-9]([A-Za-z0-9.-]*[A-Za-z0-9])?(:[0-9]{1,5})?$ ]] || \
  die "origin must be https://HOST[:PORT] without credentials, path, query or fragment"

official_script="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/participant_test.sh"
[ -f "$official_script" ] || die "official script not found: $official_script"

for command_name in curl jq mktemp; do
  command -v "$command_name" >/dev/null 2>&1 || \
    die "required command not found: $command_name"
done
real_curl=$(command -v curl)
case "$real_curl" in
  /*) ;;
  *) die "curl must resolve to an executable file, got: $real_curl" ;;
esac

[ -n "$images_dir" ] || die "--images-dir is required"
[ -d "$images_dir" ] || die "images directory not found: $images_dir"
[ -n "$manifest" ] || die "--manifest is required"
[ -f "$manifest" ] || die "manifest not found: $manifest"
[ ! -e "$output" ] || die "output already exists: $output"

endpoint="$origin/v1/eval/predict"

umask 077
secret_dir=$(mktemp -d) || die "cannot create temporary directory"
chmod 700 "$secret_dir" || die "cannot restrict temporary directory"
child=""

cleanup() {
  rm -rf -- "$secret_dir"
}

on_signal() {
  trap - EXIT HUP INT TERM
  cleanup
  if [ -n "$child" ]; then
    command -v pkill >/dev/null 2>&1 && pkill -TERM -P "$child" 2>/dev/null
    kill -TERM "$child" 2>/dev/null
    wait "$child" 2>/dev/null
  fi
  exit "$1"
}

trap cleanup EXIT
trap 'on_signal 129' HUP
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

token_response="$secret_dir/token.json"
curl_config="$secret_dir/auth.curl"

token_http_code=$("$real_curl" -q \
  --silent \
  --show-error \
  --proto '=https' \
  --max-redirs 0 \
  --connect-timeout 5 \
  --max-time 15 \
  --request POST \
  --header 'Content-Type: application/json' \
  --data '{}' \
  --output "$token_response" \
  --write-out '%{http_code}' \
  "$origin/auth/guest/token") || die "guest token request failed"
[ "$token_http_code" = "200" ] || die "guest token request returned HTTP $token_http_code"

jq -er '
  select((.token_type | ascii_downcase) == "bearer")
  | .access_token
  | select(type == "string" and test("^[A-Za-z0-9._~+/=-]{16,4096}$"))
  | "header = \"Authorization: Bearer \(.)\""
' "$token_response" > "$curl_config" 2>/dev/null || die "guest token response is not a usable Bearer token"
rm -f -- "$token_response"

mkdir "$secret_dir/bin" || die "cannot create curl shim directory"
{
  printf '#!/usr/bin/env bash\n'
  printf 'real_curl=%q\n' "$real_curl"
  printf 'curl_config=%q\n' "$curl_config"
  printf 'endpoint=%q\n' "$endpoint"
  printf '%s\n' \
    'if [ "$#" -gt 0 ] && [ "${!#}" = "$endpoint" ]; then' \
    '  exec "$real_curl" -q --config "$curl_config" --proto "=https" "$@"' \
    'fi' \
    'exec "$real_curl" "$@"'
} > "$secret_dir/bin/curl" || die "cannot write curl shim"
chmod 700 "$secret_dir/bin/curl" || die "cannot make curl shim executable"

PATH="$secret_dir/bin:$PATH" bash "$official_script" \
  --images-dir "$images_dir" \
  --manifest "$manifest" \
  --endpoint "$endpoint" \
  --output "$output" &
child=$!
wait "$child"
status=$?
child=""
exit "$status"
