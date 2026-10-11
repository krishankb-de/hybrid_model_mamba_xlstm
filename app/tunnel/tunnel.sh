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
#   known_hosts_hpi_nodes, and only through the login node. A key that later changes is still refused. The file's path is handed to ssh in
#   double quotes, so a HOME with a space is fine; a double quote, a backslash or a % in HOME is refused (ssh could not read the path).
# VIA=login: the forward goes to NODE:PORT from the login node. That works only for a server started with a BIND other than 127.0.0.1,
#   which needs a token (R6).
#
# While a forward runs, the file is read again every RECHECK_S seconds (the same read, the same checks) and compared with the host, port and
# mode the forward was started for. A requeue puts the server on another node, or with --port 0 on another port, and an ssh forward to the
# old place stays up and points at nothing: ExitOnForwardFailure covers only the local bind, and ServerAlive only a dead connection.
#   changed     the forward is stopped and one line says so; the next attempt starts at once, with no RETRY_SLEEP pause
#   gone        the file is missing or empty (the server removes it first thing when it stops): the forward is stopped and one line says so,
#               then the usual hint and RETRY_SLEEP
#   unreadable  the content fails the checks above: the forward is stopped and one line says so, then "endpoint unreadable; retrying" and
#               RETRY_SLEEP
#   ssh failed  the read itself failed (exit 255, no connection): the forward is kept, ServerAlive ends it if its connection is dead; one line
#               is printed for each failed re-read, and the read is tried again at the next interval
#   the same    nothing is printed; a restarted server on the same node and port is the same place, and the forward is still right
# A forward that ends by itself is noticed within about a second (the script looks once a second) and the loop goes on as before: RETRY_SLEEP,
# then a new read.
#
# Settings, from the environment:
#   LOGIN          ssh host name or alias of the login node (default hpi-hpc)
#   LOCAL_PORT     the port on this machine (default 8000); the tunnel stops with a message when something already uses it
#   VIA            jump (default) | login
#   CLUSTER_USER   the user on the compute node, for VIA=jump (default krishankumar.bhushan)
#   RETRY_SLEEP    seconds between two attempts (default 15)
#   RECHECK_S      seconds between two re-reads of the endpoint while a forward runs (default 60; a whole number from 1 to 99999)
#   MAX_ATTEMPTS   stop after this many attempts and exit 0; 0, the default, never stops (for rehearsals and diagnosis). An attempt is one read
#                  of the endpoint and then its forward, for as long as that lives: the re-reads belong to the attempt and are not counted, so a
#                  live forward is never ended by MAX_ATTEMPTS, and the attempt that follows a change counts as a new one.
# Exit status: 1 when the local port is already in use, 2 for a setting that is not valid, 128 + n for signal n, 0 after MAX_ATTEMPTS.
# It prints one timestamped line per attempt. Ctrl-C (or a kill) stops the ssh together with the script, at once, also in the pauses; only
# while the endpoint is being read (ssh, which gives up connecting after 10 s) does a kill sent to the script alone wait for the read, and
# Ctrl-C reaches the read as well. The forward is bound to loopback whatever ~/.ssh/config says (GatewayPorts=no, R6).
#
# On the cluster this runs one thing: `cat chat_sessions/endpoint`, under BatchMode and with its stdin on /dev/null, so it never waits for a
# prompt and a tunnel started in the background of a terminal is not stopped by SIGTTIN. The server's token file is never read, copied or
# printed. The local port is tested by binding it on 127.0.0.1 and on ::1 (ssh -L listens on both for localhost, and a browser may take
# either), never by connecting to it, so a dev server that listens there is not sent anything. The limit of that test: on macOS a listener
# bound to a wildcard address (0.0.0.0 or ::) can read as free, because BSD lets a specific bind coexist with a wildcard one. A dev server
# that listens on 127.0.0.1 or ::1 is found.
set -u

LOGIN="${LOGIN:-hpi-hpc}"
LOCAL_PORT="${LOCAL_PORT:-8000}"
VIA="${VIA:-jump}"                                   # jump (loopback bind, P1-B) | login
CLUSTER_USER="${CLUSTER_USER:-krishankumar.bhushan}"
RETRY_SLEEP="${RETRY_SLEEP:-15}"
RECHECK_S="${RECHECK_S:-60}"
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

# Is the local port free? A bind on 127.0.0.1 and one on ::1, each closed at once: nothing connects to the port and nothing is sent to it (a
# dev server may be listening there). Both, because ssh -L binds localhost, which is both, and a browser may take either: a dev server that
# listens on ::1 only (node does) is in the tunnel's way as much as one on 127.0.0.1. A family the host does not have (EAFNOSUPPORT, or no
# ::1 address: EADDRNOTAVAIL) counts as free. SO_REUSEADDR is set as ssh sets it on its own listener: without it, the connections the last
# forward closed would read as "in use" for a minute after it ended, and a requeue would end the tunnel instead of reconnecting it.
PORT_FREE='import errno, socket, sys
port = int(sys.argv[1])
families = [(socket.AF_INET, "127.0.0.1")]
if socket.has_ipv6:
    families.append((socket.AF_INET6, "::1"))
for family, address in families:
    s = None
    try:
        s = socket.socket(family)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((address, port))
    except OSError as error:
        if error.errno not in (errno.EADDRNOTAVAIL, errno.EAFNOSUPPORT):
            sys.exit(1)
    finally:
        if s is not None:
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
# The nodes' known_hosts path goes to ssh in double quotes, so a HOME with a space is one file; a double quote, a backslash or a % in it cannot be quoted.
[ "${VIA}" != "jump" ] || case "${HOME}" in *[\"\\%]*) die "HOME must not contain a double quote, a backslash or a percent sign: ssh could not read the path of the nodes' known_hosts file" ;; esac
case "${MAX_ATTEMPTS}" in ''|*[!0-9]*) die "MAX_ATTEMPTS must be a whole number (0 = never stop)" ;; esac
[ "${#MAX_ATTEMPTS}" -le 9 ] || die "MAX_ATTEMPTS must be a whole number (0 = never stop)"
case "${RETRY_SLEEP}" in ''|*[!0-9.]*|.*|*.|*.*.*) die "RETRY_SLEEP must be a number of seconds" ;; esac
case "${RECHECK_S}" in ''|0*|*[!0-9]*) die "RECHECK_S must be a whole number of seconds from 1 to 99999" ;; esac
[ "${#RECHECK_S}" -le 5 ] || die "RECHECK_S must be a whole number of seconds from 1 to 99999"
command -v python3 >/dev/null 2>&1 || die "python3 is needed: it reads the endpoint file and tests the local port"

log() { echo "$(date +%T) $*"; }
port_free() { python3 -c "${PORT_FREE}" "${LOCAL_PORT}"; }
busy() { echo "localhost:${LOCAL_PORT} is already in use (a local dev server?); set LOCAL_PORT to a free port"; exit 1; }
parse_endpoint() { python3 -c "${PARSE_ENDPOINT}" 2>/dev/null; }

# A signal (Ctrl-C, kill, a closed terminal): stop the job that is running (the ssh forward or the pause, both started as background jobs
# below), then end. The traps are reset first, so that a second signal ends this script at once if a child will not go.
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

# Run a command (a pause) as a background job and wait for it. A signal then reaches stop() at once: bash holds a trap back until a
# foreground command has ended, and a pause lasts seconds, the ssh forward (started the same way in attempt(), watched by watch_forward())
# hours. Nothing here looks at the status it ends with: the next attempt reads the endpoint again.
run_child() {
  "$@" &
  wait $!
}

# One read of the endpoint file: the only place the cluster is asked anything, for the first read of an attempt and for every re-read. EP is
# what it printed, RC the status of the ssh. Its stdin is /dev/null: without that, a tunnel started in the background of a terminal is stopped
# by SIGTTIN, because ssh reads its stdin when it runs a command.
read_endpoint() {
  EP=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "${LOGIN}" cat chat_sessions/endpoint 2>/dev/null </dev/null)
  RC=$?
}

# Nothing to forward to: the hint, with a note when the read itself failed in ssh ($1 is the status of the read).
hint() {
  if [ "$1" -eq 255 ]; then
    log "no endpoint yet; is the job running? (ssh ${LOGIN} squeue --me); ssh itself failed (exit 255): check the connection and that your key is loaded"
  else
    log "no endpoint yet; is the job running? (ssh ${LOGIN} squeue --me)"
  fi
}

# Stop the forward that has the pid $1 and reap it, so that nothing is left behind.
end_forward() { { kill "$1"; wait "$1"; } 2>/dev/null; }

# While the forward $1 runs: look at it once a second, so that one that ends by itself is noticed within about a second, and every RECHECK_S
# seconds read the endpoint again and compare it with the one the forward was started for (PARSED, "host port mode"). The forward is stopped
# when the file has changed, is gone or is unreadable (a changed one sets NO_PAUSE: the next attempt starts at once). A re-read that fails in
# ssh itself says nothing about the file, so the forward stays.
watch_forward() {
  local fwd=$1 waited=0 latest n2 p2 m2
  while kill -0 "${fwd}" 2>/dev/null; do
    run_child sleep 1
    waited=$((waited + 1))
    [ "${waited}" -ge "${RECHECK_S}" ] || continue
    waited=0
    kill -0 "${fwd}" 2>/dev/null || break
    read_endpoint
    if [ "${RC}" -eq 255 ]; then
      log "endpoint re-check failed (ssh exit 255); keeping the forward to ${NODE}:${PORT}"
    elif [ "${RC}" -ne 0 ] || [ -z "${EP}" ]; then
      end_forward "${fwd}"
      log "endpoint file is gone; stopped the forward to ${NODE}:${PORT}"
      hint "${RC}"
      return
    elif ! latest=$(parse_endpoint <<< "${EP}"); then
      end_forward "${fwd}"
      log "endpoint file is unreadable; stopped the forward to ${NODE}:${PORT}"
      log "endpoint unreadable; retrying"
      return
    elif [ "${latest}" != "${PARSED}" ]; then
      read -r n2 p2 m2 <<< "${latest}"
      end_forward "${fwd}"
      log "endpoint changed: ${NODE}:${PORT} (mode ${MODE}) -> ${n2}:${p2} (mode ${m2}); restarting the forward"
      NO_PAUSE=1
      return
    fi
  done
  wait "${fwd}" 2>/dev/null
}

# One attempt. What it finds wrong is one line and a return: the loop pauses and goes round again.
attempt() {
  read_endpoint
  if [ "${RC}" -ne 0 ] || [ -z "${EP}" ]; then
    hint "${RC}"
    return
  fi
  PARSED=$(parse_endpoint <<< "${EP}") || { log "endpoint unreadable; retrying"; return; }
  read -r NODE PORT MODE <<< "${PARSED}"
  port_free || busy
  log "forwarding localhost:${LOCAL_PORT} -> ${NODE}:${PORT} via ${VIA} (mode ${MODE})"
  if [ "${VIA}" = "jump" ]; then
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=10 \
        -o GatewayPorts=no -o StrictHostKeyChecking=accept-new -o "UserKnownHostsFile=\"${HOME}/.ssh/known_hosts_hpi_nodes\"" \
        -J "${LOGIN}" -L "${LOCAL_PORT}:127.0.0.1:${PORT}" "${CLUSTER_USER}@${NODE}" &
  else
    ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ConnectTimeout=10 \
        -o GatewayPorts=no -L "${LOCAL_PORT}:${NODE}:${PORT}" "${LOGIN}" &
  fi
  watch_forward $!
}

port_free || busy
[ "${VIA}" != "login" ] || echo "VIA=login needs a server started with a BIND other than 127.0.0.1, which needs a token (R6); the default VIA=jump needs neither"

ATTEMPTS=0
while true; do
  ATTEMPTS=$((ATTEMPTS + 1))
  NO_PAUSE=0
  attempt
  if [ "${MAX_ATTEMPTS}" -gt 0 ] && [ "${ATTEMPTS}" -ge "${MAX_ATTEMPTS}" ]; then
    log "stopping after ${ATTEMPTS} attempt(s) (MAX_ATTEMPTS)"
    exit 0
  fi
  [ "${NO_PAUSE}" -eq 1 ] || run_child sleep "${RETRY_SLEEP}"   # the job moved or died, or is not up yet: read the endpoint again
done
