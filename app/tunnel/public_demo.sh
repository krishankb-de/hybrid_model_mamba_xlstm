#!/bin/bash
# CHAT_UI_PLAN.md P7-C — a public link to the chat server for a demo: cloudflared's quick tunnel (no account) to http://localhost:${LOCAL_PORT:-8000}.
# Runs on the laptop, after app/tunnel/tunnel.sh (or anything else) has put the server on that port. PUBLIC MODE ONLY (R6): it refuses
# unless that server's /healthz answers JSON whose "mode" is exactly "public", the mode that needs a token file, has the rate limit and
# shows nothing MIMIC-derived (R1). A private server never gets a public link from this script.
#
#   bash app/tunnel/public_demo.sh
#   LOCAL_PORT=8010 bash app/tunnel/public_demo.sh    # the port tunnel.sh was given
#
# Anyone with the link reaches the server until this script is stopped (Ctrl-C, or a kill: it stops cloudflared), so stop it as soon as
# the demo ends. Actually running it is P9-D, which waits for the user's word (U3). It needs cloudflared; when that is missing it says where
# the install instructions are and runs no installer. LOCAL_PORT (default 8000) is the port the server is on.
set -u

LOCAL_PORT="${LOCAL_PORT:-8000}"

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
command -v python3 >/dev/null 2>&1 || die "python3 is needed: it reads the server's /healthz answer"

# The mode first, then cloudflared: the refusal that matters most comes first.
HEALTH=$(curl -s --max-time 5 "http://localhost:${LOCAL_PORT}/healthz") || HEALTH=""
SEEN=$(python3 -c "${MODE_OF}" <<< "${HEALTH}")
case "${SEEN}" in
  public) ;;
  private) refuse "the server on localhost:${LOCAL_PORT} is in private mode; a public link needs a server started with MODE=public (R6)" ;;
  *) refuse "no usable answer from http://localhost:${LOCAL_PORT}/healthz; it must be JSON with \"mode\": \"public\" (is the server up, and started with MODE=public?)" ;;
esac
command -v cloudflared >/dev/null 2>&1 || refuse "cloudflared is not installed; install it first, see https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/"

# A signal (Ctrl-C, kill, a closed terminal), and the normal end too (then no job is left and it only says so): stop cloudflared, end. The
# traps are reset first, so that a second signal ends this script at once if cloudflared will not go.
stop() {
  trap - INT TERM HUP
  local pids
  pids=$(jobs -p)
  [ -z "${pids}" ] || { kill ${pids}; wait; } 2>/dev/null
  echo "cloudflared stopped: the public link is closed."
  exit "$1"
}
trap 'stop 130' INT
trap 'stop 143' TERM
trap 'stop 129' HUP

echo "PUBLIC DEMO: anyone with the link can reach the server on localhost:${LOCAL_PORT} until this tunnel is stopped."
echo "Stop it (Ctrl-C) as soon as the demo ends."
# A background job and a wait, so that a signal reaches stop() at once (bash holds a trap back until a foreground command has ended).
cloudflared tunnel --url "http://localhost:${LOCAL_PORT}" &
wait $!
stop $?
