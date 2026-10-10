// app/static/api.js — the browser's data layer: the SSE parser, the streaming turn, polling, cancel and the image
// loader. Plain ES module over fetch streams: no dependencies and nothing Node-only, so the same file runs under
// `node --test` (tests/frontend/parsers.test.mjs).
//
// Events come out as {event, data} with data.seq, from the stream and from polling alike, and state.js folds both.
// A refused request throws an Error with `status` and `body`, the server's parsed error envelope
// {"type": "error", "error": {"type", "message"}} (null when the body is not JSON); a failure of the transport itself (fetch
// rejected, a body that could not be read) throws with `transport: true` (transportFailure). Every request carries authHeaders:
// an <img src> cannot send them, which is why images are fetched and shown through object URLs (D23), and why those
// take a path on this page's own origin only: the bearer token goes nowhere else.

export function parseSSE(buffer) {
  // Pure: feed text, get the complete frames and the unconsumed tail. ": ping" comments are skipped, and so is a frame
  // whose data is not JSON (a bare "data:" too) or is JSON that is no event: every event the server sends is an object
  // with an integer seq, and applyEvent reads it (null throws there, a missing seq would disable the dedupe). The
  // frames around a skipped one are still returned.
  buffer = buffer.replace(/\r\n/g, '\n');
  const events = [];
  let i;
  while ((i = buffer.indexOf('\n\n')) >= 0) {
    const frame = buffer.slice(0, i);
    buffer = buffer.slice(i + 2);
    let event = 'message';
    const data = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith(':')) continue;
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) data.push(line.slice(5).replace(/^ /, ''));
    }
    if (!data.length) continue;
    try {
      const d = JSON.parse(data.join('\n'));
      if (Number.isInteger(d?.seq)) events.push({ event, data: d });
    } catch { /* not JSON: drop this frame only */ }
  }
  return { events, rest: buffer };
}

// A failure of the transport itself, marked where it happens (P4-H): fetch rejected (the network, a server that is down, a tunnel that
// dropped, an abort) or reading a response's body failed. The page takes these, and only these, for the network's doing; a TypeError
// that is a bug (of this file's parsing, or of a callback it was given) carries no mark, and the page logs it. -> the error, marked.
export function transportFailure(err) {
  if (err !== null && typeof err === 'object' && !('transport' in err)) {
    try { err.transport = true; } catch { /* a frozen error stays as it is */ }
  }
  return err;
}

// fetch, with its rejection marked as the transport's.
async function fetched(url, init) {
  try {
    return await fetch(url, init);
  } catch (err) {
    throw transportFailure(err);
  }
}

export function authHeaders(token, clientId) {
  const h = {};
  if (token) h.Authorization = `Bearer ${token}`;
  if (clientId) h['X-Client-Id'] = clientId;
  return h;
}

// POSTs one turn and yields its events as they stream. onMessageId(id) gets the X-Message-Id header as soon as the
// server accepts the turn, before any of the body is read: a turn still queued has no event yet, and Stop needs the id.
// However it ends (the stream closes, the consumer leaves the loop, a read or onMessageId throws) the body is cancelled,
// so the connection is not left open. An aborted signal ends it with the AbortError fetch throws. A stream that ends
// without a message_stop just returns: the caller sees view.status still 'running' and falls back to pollMessage.
export async function* streamTurn({ base = '', sessionId, form, token, clientId, signal, onMessageId }) {
  const res = await fetched(`${base}/v1/sessions/${sessionId}/messages`,
                            { method: 'POST', body: form, signal, headers: authHeaders(token, clientId) });
  if (!res.ok) throw Object.assign(new Error('turn refused'), { status: res.status, body: await res.json().catch(() => null) });
  const reader = res.body.getReader();
  try {
    const id = res.headers.get('X-Message-Id');
    if (id && onMessageId) onMessageId(id);
    const dec = new TextDecoder();
    let buf = '';
    for (;;) {
      let chunk;
      try {
        chunk = await reader.read();
      } catch (err) {
        throw transportFailure(err);   // the connection broke mid-stream (or the read was aborted)
      }
      const { value, done } = chunk;
      if (done) return;
      const { events, rest } = parseSSE(buf + dec.decode(value, { stream: true }));
      buf = rest;
      yield* events;
    }
  } finally {
    await reader.cancel().catch(() => {});
  }
}

// What every request but the turn throws when the server refuses it.
async function refused(res, message) {
  return Object.assign(new Error(message), { status: res.status, body: await res.json().catch(() => null) });
}

// A pause the signal can cut short: it resolves on abort (it never rejects) and takes its timer with it.
function sleep(ms, signal) {
  return new Promise((resolve) => {
    if (signal?.aborted) return resolve();
    const done = () => { clearTimeout(timer); signal?.removeEventListener('abort', done); resolve(); };
    const timer = setTimeout(done, ms);
    signal?.addEventListener('abort', done, { once: true });
  });
}

const RETRY_STATUSES = new Set([429, 502, 503, 504]);   // busy, or a proxy or tunnel in between failed: worth asking again
const MAX_POLL_FAILURES = 20;                          // consecutive failed polls before pollMessage gives up

// Polls GET /v1/messages/{id}?after=N and yields each stored event once, in order, as the stream would have: the
// fallback when the stream drops (D7). Every intervalMs (500) it asks again for the events past the highest seq it has
// yielded; once the status leaves 'running' it yields what is left and returns.
// A poll that fails the way a flapping tunnel or a restarting server does (the network error fetch throws as a
// TypeError, or a 429, 502, 503 or 504) is asked again after a backoff that starts at intervalMs, doubles and stops
// growing at maxBackoffMs (5 s); a success starts it over, and the 20th failure in a row is thrown. Any other refusal
// (401, 403, 404, 422 ...) is final and is thrown at once with its status and body. An aborted signal ends it quietly,
// whether it is waiting, backing off or mid-request: that is the caller's own doing, not an error.
export async function* pollMessage({ base = '', messageId, after = 0, token, clientId, signal, intervalMs = 500,
                                     maxBackoffMs = 5000 }) {
  let seen = after;
  let failures = 0;
  while (!signal?.aborted) {
    let message;
    try {
      const url = `${base}/v1/messages/${messageId}?after=${seen}`;
      const res = await fetched(url, { signal, headers: authHeaders(token, clientId) });
      if (!res.ok) throw await refused(res, 'poll refused');
      message = await res.json().catch((err) => { throw err instanceof TypeError ? transportFailure(err) : err; });   // a body cut off
    } catch (err) {
      if (signal?.aborted) return;
      if (!(err instanceof TypeError || RETRY_STATUSES.has(err.status)) || ++failures >= MAX_POLL_FAILURES) throw err;
      await sleep(Math.min(maxBackoffMs, intervalMs * 2 ** (failures - 1)), signal);
      continue;
    }
    failures = 0;
    for (const row of message.events) {
      if (row.seq <= seen) continue;                    // a row an earlier poll delivered
      seen = row.seq;
      yield { event: row.event, data: row.data };       // data carries seq already
    }
    if (message.status !== 'running') return;
    await sleep(intervalMs, signal);
  }
}

// Asks the server to stop a queued or running turn; -> {id, status, cancel_requested}. Idempotent on the server.
export async function cancelMessage({ base = '', messageId, token, clientId }) {
  const res = await fetched(`${base}/v1/messages/${messageId}/cancel`, { method: 'POST', headers: authHeaders(token, clientId) });
  if (!res.ok) throw await refused(res, 'cancel refused');
  return res.json();
}

const images = new Map();       // url -> Promise<object URL>: concurrent calls share one fetch, later calls reuse its result
const objectUrls = new Set();   // every object URL made, so clearImageCache can revoke them all

// A path on this page's own origin: one "/" that is followed by neither a second "/" nor a backslash (a browser reads
// either as the start of a host), and no control character (a tab or newline is dropped before parsing, which can
// turn "/<tab>/host" into "//host").
const SAME_ORIGIN_PATH = /^\/(?![/\\])[^\x00-\x1f\x7f]*$/;

// The one filter for a URL that may be fetched with the token: loadImage applies it, and so does render.js before it asks
// for a thumbnail, so what one refuses the other never sees.
export const isSameOriginPath = (url) => typeof url === 'string' && SAME_ORIGIN_PATH.test(url);

// An image behind the token, as an object URL for <img src>: auth = {token, clientId}. A refusal (the server answered with an error: the
// image is gone, unreadable, not this client's) is kept too, until clearImageCache (P6 fix 1): a card is drawn again at every frame of a
// streamed turn, and a picture that cannot load must not be asked for at each of them. Another chat, or a new token, clears the cache and the
// picture is asked for again. A failure of the transport itself (fetch rejected, the body cut off) is not kept: it is no answer about the
// image, and the next call asks again, so a passing blip does not leave a picture broken for the rest of the chat.
// Only a same-origin path is fetched: anything else rejects before a request is made, so the token never leaves.
export function loadImage(url, auth = {}) {
  if (!isSameOriginPath(url)) {
    return Promise.reject(new Error('loadImage takes a path on this origin, starting with a single "/"'));
  }
  if (!images.has(url)) {
    const loading = (async () => {
      const res = await fetched(url, { headers: authHeaders(auth.token, auth.clientId) });
      if (!res.ok) throw await refused(res, 'image refused');
      const blob = await res.blob().catch((err) => { throw transportFailure(err); });
      const objectUrl = URL.createObjectURL(blob);
      objectUrls.add(objectUrl);
      return objectUrl;
    })();
    loading.catch((err) => {   // handled here as well, so a kept refusal is never an unhandled rejection
      if (err?.transport === true && images.get(url) === loading) images.delete(url);
    });
    images.set(url, loading);
  }
  return images.get(url);
}

// Forgets every image, and every failure, and revokes each object URL. A fetch still running when this is called finishes into
// the next cache's set, so the next clear revokes it.
export function clearImageCache() {
  for (const objectUrl of objectUrls) URL.revokeObjectURL(objectUrl);
  objectUrls.clear();
  images.clear();
}
