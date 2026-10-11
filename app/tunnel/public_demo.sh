#!/bin/bash
# CHAT_UI_PLAN.md P7-C — a public link to the chat server for a demo: cloudflared's quick tunnel (no account) to http://localhost:${LOCAL_PORT:-8000}.
# Runs on the laptop, after app/tunnel/tunnel.sh (or anything else) has put the server on that port. PUBLIC MODE ONLY (R6): it refuses
# unless that server's /healthz answers JSON whose "mode" is exactly "public", the mode that needs a token file, has the rate limit and
# shows nothing MIMIC-derived (R1). A private server never gets a public link from this script.
#
#   bash app/tunnel/public_demo.sh
#   LOCAL_PORT=8010 bash app/tunnel/public_demo.sh    # the port tunnel.sh was given
#
# The mode is checked again while the link is up, every DEMO_RECHECK_S seconds (default 10): a server that comes back in private mode (it was
# requeued, say, behind a link nobody closed) serves MIMIC-derived content (R6, the DUA) and must never sit behind a live public link. On
# anything but JSON whose mode is exactly "public" (private mode, a server that is down or does not answer within 5 s, an answer that is not
# JSON) the script stops cloudflared and exits 1 with one `refusing: ...` line that says the public link is closed. It fails closed, on one
# bad answer: a server too busy to answer /healthz within 5 s ends the demo, and the script is started again.
#
# Anyone with the link reaches the server until this script is stopped (Ctrl-C, or a kill: it stops cloudflared), so stop it as soon as
# the demo ends. Actually running it is P9-D, which waits for the user's word (U3). It needs cloudflared; when that is missing it says where
# the install instructions are and runs no installer.
#   LOCAL_PORT       the port the server is on (default 8000)
#   DEMO_RECHECK_S   seconds between two checks of the mode while the link is up (default 10; a whole number from 1 to 99999)
# Exit status: 1 for a refusal (also when the check closes the link), 2 for a setting that is not valid, 128 + n for signal n, and when
# cloudflared ends by itself, its own. A kill sent to the script alone is acted on when a check in progress ends (curl gives up after 5 s);
# Ctrl-C reaches the curl as well.
set -u

LOCAL_PORT="${LOCAL_PORT:-8000}"
DEMO_RECHECK_S="${DEMO_RECHECK_S:-10}"

# What the server says its mode is: public, private, or unknown (no answer, not JSON, no "mode", or a value that is neither of the two).
# The answer is parsed, never matched as text: a "public" inside another field is not the mode.
MODE_OF='import json, sys
try:
    mode = json.loads(sys.stdin.read(65537)).get("mode")
except Exception:
    mode = None
print(mode if mode in ("public", "private") else "unknown")'

die() { echo "$*"; exit 2; }
refuse() { echo "refusing: $*"; exit 1; }

case "${LOCAL_PORT}" in ''|0*|*[!0-9]*) die "LOCAL_PORT must be a number from 1 to 65535" ;; esac
[ "${#LOCAL_PORT}" -le 5 ] && [ "${LOCAL_PORT}" -le 65535 ] || die "LOCAL_PORT must be a number from 1 to 65535"
case "${DEMO_RECHECK_S}" in ''|0*|*[!0-9]*) die "DEMO_RECHECK_S must be a whole number of seconds from 1 to 99999" ;; esac
[ "${#DEMO_RECHECK_S}" -le 5 ] || die "DEMO_RECHECK_S must be a whole number of seconds from 1 to 99999"
command -v python3 >/dev/null 2>&1 || die "python3 is needed: it reads the server's /healthz answer"

# Ask the server for its mode: public, private or unknown.
public_mode() {
  local body
  body=$(curl -s --max-time 5 "http://localhost:${LOCAL_PORT}/healthz") || body=""
  python3 -c "${MODE_OF}" <<< "${body}"
}

# The mode first, then cloudflared: the refusal that matters most comes first.
SEEN=$(public_mode)
case "${SEEN}" in
  public) ;;
  private) refuse "the server on localhost:${LOCAL_PORT} is in private mode; a public link needs a server started with MODE=public (R6)" ;;
  *) refuse "no usable answer from http://localhost:${LOCAL_PORT}/healthz; it must be JSON with \"mode\": \"public\" (is the server up, and started with MODE=public?)" ;;
esac
command -v cloudflared >/dev/null 2>&1 || refuse "cloudflared is not installed; install it first, see https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"

# Stop what runs (cloudflared, or the pause between two checks) and wait for it. The traps are reset first, so that a second signal ends this
# script at once if cloudflared will not go.
end_jobs() {
  local pids
  trap - INT TERM HUP
  pids=$(jobs -p)
  [ -z "${pids}" ] || { kill ${pids}; wait; } 2>/dev/null
}
# A signal (Ctrl-C, kill, a closed terminal), and the normal end too (then no job is left and it only says so): stop cloudflared, end.
stop() {
  end_jobs
  echo "cloudflared stopped: the public link is closed."
  exit "$1"
}
# The check found the server is not public (any more): stop cloudflared, one line, status 1.
close_link() {
  end_jobs
  echo "refusing: $1; cloudflared is stopped and the public link is closed."
  exit 1
}
trap 'stop 130' INT
trap 'stop 143' TERM
trap 'stop 129' HUP

# Run a command as a background job and wait for it, so that a signal reaches stop() at once (bash holds a trap back until a foreground
# command has ended).
run_child() {
  "$@" &
  wait $!
}

echo "PUBLIC DEMO: anyone with the link can reach the server on localhost:${LOCAL_PORT} until this tunnel is stopped."
echo "Stop it (Ctrl-C) as soon as the demo ends. Every ${DEMO_RECHECK_S} s the server's mode is checked again; the link is closed as soon as it is not public."
cloudflared tunnel --url "http://localhost:${LOCAL_PORT}" &
LINK=$!
# Look at cloudflared once a second (so that it ending by itself is noticed within about a second) and, every DEMO_RECHECK_S seconds, at the mode.
waited=0
while kill -0 "${LINK}" 2>/dev/null; do
  run_child sleep 1
  waited=$((waited + 1))
  [ "${waited}" -ge "${DEMO_RECHECK_S}" ] || continue
  waited=0
  kill -0 "${LINK}" 2>/dev/null || break
  NOW=$(public_mode)
  case "${NOW}" in
    public) ;;
    private) close_link "the server on localhost:${LOCAL_PORT} is now in private mode (R6)" ;;
    *) close_link "no usable answer from http://localhost:${LOCAL_PORT}/healthz any more (the server is down, does not answer, or does not say public)" ;;
  esac
done
wait "${LINK}"
stop $?
