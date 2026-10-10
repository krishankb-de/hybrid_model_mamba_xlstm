#!/bin/bash
# CHAT_UI_PLAN.md P7-C — keep http://localhost:${LOCAL_PORT:-8000} pointed at the chat server, wherever SLURM has (re)started it.
# Runs on the laptop. Ctrl-C to stop.
#
#   bash app/tunnel/tunnel.sh                    # then open http://localhost:8000
#   LOCAL_PORT=8010 bash app/tunnel/tunnel.sh    # when something else (a dev server) already listens on 8000
#
# The serve job (scripts/serve_chat_h100.sh, P7-B) writes where it is serving to chat_sessions/endpoint in your cluster home, once the API
# is up, as ONE JSON line {host, port, mode, pid, started_at}. A clean stop removes the file; a lost node can leave a stale one until the
# next job overwrites it. This script reads the file over ssh whenever it needs a connection, so a requeue onto another node costs a pause
# and no hand-work. The file lives on a shared cluster, so its content is data, not instructions: host must be a plain host name and port
# a number from 1 to 65535, or the content is ignored (and never printed) and the file is read again. Nothing from it reaches a shell.
#
# VIA=jump (default): the server listens on 127.0.0.1 of its node (the default bind, R6), so the forward goes to 127.0.0.1:PORT on the node
#   with the login node as the jump host. Compute-node host keys are not in ~/.ssh/known_hosts and an unattended forward cannot answer the
#   prompt for one (P1-B: "Host key verification failed"), so a node's key is accepted on first use into a file of its own,
#   known_hosts_hpi_nodes, and only through the login node. A key that later changes is still refused.
# VIA=login: the forward goes to NODE:PORT from the login node. That works only for a server started with a BIND other than 127.0.0.1,
#   which needs a token (R6).
#
# Settings, from the environment:
#   LOGIN          ssh host name or alias of the login node (default hpi-hpc)
#   LOCAL_PORT     the port on this machine (default 8000); the tunnel stops with a message when something already uses it
#   VIA            jump (default) | login
#   CLUSTER_USER   the user on the compute node, for VIA=jump (default krishankumar.bhushan)
#   RETRY_SLEEP    seconds between two attempts (default 15)
#   MAX_ATTEMPTS   stop after this many attempts and exit 0; 0, the default, never stops (for rehearsals and diagnosis)
# Exit status: 1 when the local port is already in use, 2 for a setting that is not valid, 128 + n for signal n, 0 after MAX_ATTEMPTS.
# One attempt is one read of the endpoint, then a forward for as long as its ssh lives; it prints one timestamped line. Ctrl-C (or a kill)
# stops the ssh together with the script. The forward is bound to loopback whatever ~/.ssh/config says (GatewayPorts=no, R6).
#
# On the cluster this runs one thing: `cat chat_sessions/endpoint`, under BatchMode, so it never waits for a prompt. The server's token file
# is never read, copied or printed. The local port is tested by binding it, never by connecting to it, so a dev server that listens there
# is not sent anything.
set -u

LOGIN="${LOGIN:-hpi-hpc}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
VIA="${VIA:-jump}"                                   # jump (loopback bind, P1-B) | login
CLUSTER_USER="${CLUSTER_USER:-krishankumar.bhushan}"
RETRY_SLEEP="${RETRY_SLEEP:-15}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-0}"                    # 0: never stop

# The endpoint file's content, parsed: "host port mode" on one line, made only of values that passed the checks, or no output and status 1.
# The host is a plain host name (the pattern anchored at both ends: a trailing newline does not pass), the port a whole number from 1 to
# 65535 (not a bool, a fraction or text), the mode shown only when it is one of the two the server writes. At most 4096 characters are
# read; the file is about 120.
PARSE_ENDPOINT='import json, re, sys
try:
    text = sys.stdin.read(4097)
    data = json.loads(text) if len(text) <= 4096 else None
    host, port, mode = data["host"], data["port"], data.get("mode")
    good = (isinstance(host, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]{0,252}", host) is not None
            and type(port) is int and 1 <= port <= 65535)
except Exception:
    good = False
if not good:
    sys.exit(1)
print(host, port, mode if mode in ("private", "public") else "unknown")'

# Is the local port free? A bind on 127.0.0.1, closed at once: nothing connects to the port and nothing is sent to it (a dev server may be
# listening there). SO_REUSEADDR is set as ssh sets it on its own listener: without it, the connections the last forward closed would read
# as "in use" for a minute after it ended, and a requeue would end the tunnel instead of reconnecting it.
PORT_FREE='import socket, sys
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
except OSError:
    sys.exit(1)
finally:
    s.close()'

die() { echo "$*"; exit 2; }
is_port() {   # a whole number from 1 to 65535: no sign, no leading zero, nothing else (the length is checked first, so nothing overflows)
  case "$1" in ''|0*|*[!0-9]*) return 1 ;; esac
  [ "${#1}" -le 5 ] && [ "$1" -le 65535 ]
}

is_port "${LOCAL_PORT}" || die "LOCAL_PORT must be a number from 1 to 65535"
case "${VIA}" in jump|login) ;; *) die "VIA must be jump or login" ;; esac
case "${LOGIN}" in ''|-*|*[!A-Za-z0-9._@-]*) die "LOGIN must be an ssh host name or alias" ;; esac
case "${CLUSTER_USER}" in ''|-*|*[!A-Za-z0-9._-]*) die "CLUSTER_USER must be a plain user name" ;; esac
case "${MAX_ATTEMPTS}" in ''|*[!0-9]*) die "MAX_ATTEMPTS must be a whole number (0 = never stop)" ;; esac
[ "${#MAX_ATTEMPTS}" -le 9 ] || die "MAX_ATTEMPTS must be a whole number (0 = never stop)"
case "${RETRY_SLEEP}" in ''|*[!0-9.]*|.*|*.|*.*.*) die "RETRY_SLEEP must be a number of seconds" ;; esac
command -v python3 >/dev/null 2>&1 || die "python3 is needed: it reads the endpoint file and tests the local port"

log() { echo "$(date +%T) $*"; }
port_free() { python3 -c "${PORT_FREE}" "${LOCAL_PORT}"; }
busy() { echo "localhost:${LOCAL_PORT} is already in use (a local dev server?); set LOCAL_PORT to a free port"; exit 1; }
parse_endpoint() { python3 -c "${PARSE_ENDPOINT}" 2>/dev/null; }

# A signal (Ctrl-C, kill, a closed terminal): stop the job that is running (the ssh forward or the pause, both started by run_child), then
# end. The traps are reset first, so that a second signal ends this script at once if a child will not go.
stop() {
  trap - INT TERM HUP
  local pids
  pids=$(jobs -p)
  [ -z "${pids}" ] || { kill ${pids}; wait; } 2>/dev/null
  echo "stopped"
  exit "$1"
}
trap 'stop 130' INT
trap 'stop 143' TERM
trap 'stop 129' HUP

# Run a command as a background job and wait for it. A signal then reaches stop() at once: bash holds a trap back until a foreground command
# has ended, and an ssh forward lasts for hours. Nothing here looks at the status it ends with: the next attempt reads the endpoint again.
run_child() {
  "$@" &
  wait $!
}

# One attempt. What it finds wrong is one line and a return: the loop pauses and goes round again.
attempt() {
  local rc
  EP=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${LOGIN}" cat chat_sessions/endpoint 2>/dev/null)
  rc=$?
  if [ "${rc}" -ne 0 ] || [ -z "${EP}" ]; then
    if [ "${rc}" -eq 255 ]; then
      log "no endpoint yet; is the job running? (ssh ${LOGIN} squeue --me); ssh itself failed (exit 255): check the connection and that your key is loaded"
    else
      log "no endpoint yet; is the job running? (ssh ${LOGIN} squeue --me)"
    fi
    return
  fi
  PARSED=$(parse_endpoint <<< "${EP}") || { log "endpoint unreadable; retrying"; return; }
  read -r NODE PORT MODE <<< "${PARSED}"
  port_free || busy
  log "forwarding localhost:${LOCAL_PORT} -> ${NODE}:${PORT} via ${VIA} (mode ${MODE})"
  if [ "${VIA}" = "jump" ]; then
    run_child ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=10 \
        -o GatewayPorts=no -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="${HOME}/.ssh/known_hosts_hpi_nodes" \
        -J "${LOGIN}" -L "${LOCAL_PORT}:127.0.0.1:${PORT}" "${CLUSTER_USER}@${NODE}"
  else
    run_child ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=10 \
        -o GatewayPorts=no -L "${LOCAL_PORT}:${NODE}:${PORT}" "${LOGIN}"
  fi
}

port_free || busy
[ "${VIA}" != "login" ] || echo "VIA=login needs a server started with a BIND other than 127.0.0.1, which needs a token (R6); the default VIA=jump needs neither"

ATTEMPTS=0
while true; do
  ATTEMPTS=$((ATTEMPTS + 1))
  attempt
  if [ "${MAX_ATTEMPTS}" -gt 0 ] && [ "${ATTEMPTS}" -ge "${MAX_ATTEMPTS}" ]; then
    log "stopping after ${ATTEMPTS} attempt(s) (MAX_ATTEMPTS)"
    exit 0
  fi
  run_child sleep "${RETRY_SLEEP}"   # the job moved or died, or is not up yet: read the endpoint again
done
