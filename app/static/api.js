// app/static/api.js — the browser's data layer: the SSE parser, the streaming turn, polling, cancel and the image
// loader. Plain ES module over fetch streams: no dependencies and nothing Node-only, so the same file runs under
// `node --test` (tests/frontend/parsers.test.mjs).
//
// Events come out as {event, data} with data.seq, from the stream and from polling alike, and state.js folds both.
// A refused request throws an Error with `status` and `body`, the server's parsed error envelope
// {"type": "error", "error": {"type", "message"}} (null when the body is not JSON). Every request carries authHeaders:
// an <img src> cannot send them, which is why images are fetched and shown through object URLs (D23).

export function parseSSE(buffer) {
  // Pure: feed text, get the complete frames and the unconsumed tail. ": ping" comments are skipped.
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
    if (data.length) events.push({ event, data: JSON.parse(data.join('\n')) });
  }
  return { events, rest: buffer };
}

export function authHeaders(token, clientId) {
  const h = {};
  if (token) h.Authorization = `Bearer ${token}`;
  if (clientId) h['X-Client-Id'] = clientId;
  return h;
}

// POSTs one turn and yields its events as they stream. onMessageId(id) gets the X-Message-Id header as soon as the
// server accepts the turn, before the first byte of the body: a turn still queued has no event yet, and Stop needs the id.
// An aborted signal ends it with the AbortError fetch throws. A stream that ends without a message_stop just returns:
// the caller sees view.status still 'running' and falls back to pollMessage from view.lastSeq.
export async function* streamTurn({ base = '', sessionId, form, token, clientId, signal, onMessageId }) {
  const res = await fetch(`${base}/v1/sessions/${sessionId}/messages`,
                          { method: 'POST', body: form, signal, headers: authHeaders(token, clientId) });
  if (!res.ok) throw Object.assign(new Error('turn refused'), { status: res.status, body: await res.json().catch(() => null) });
  const id = res.headers.get('X-Message-Id');
  if (id && onMessageId) onMessageId(id);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    const { events, rest } = parseSSE(buf + dec.decode(value, { stream: true }));
    buf = rest;
    yield* events;
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

// Polls GET /v1/messages/{id}?after=N and yields each stored event once, in order, as the stream would have: the
// fallback when the stream drops (D7). Every intervalMs (500) it asks again for the events past the highest seq it has
// yielded; once the status leaves 'running' it yields what is left and returns. An aborted signal ends it quietly,
// whether it is waiting or mid-request: that is the caller's own doing, not an error.
export async function* pollMessage({ base = '', messageId, after = 0, token, clientId, signal, intervalMs = 500 }) {
  let seen = after;
  while (!signal?.aborted) {
    let message;
    try {
      const url = `${base}/v1/messages/${messageId}?after=${seen}`;
      const res = await fetch(url, { signal, headers: authHeaders(token, clientId) });
      if (!res.ok) throw await refused(res, 'poll refused');
      message = await res.json();
    } catch (err) {
      if (signal?.aborted) return;
      throw err;
    }
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
  const res = await fetch(`${base}/v1/messages/${messageId}/cancel`, { method: 'POST', headers: authHeaders(token, clientId) });
  if (!res.ok) throw await refused(res, 'cancel refused');
  return res.json();
}

const images = new Map();       // url -> Promise<object URL>: concurrent calls share one fetch, later calls reuse its result
const objectUrls = new Set();   // every object URL made, so clearImageCache can revoke them all

// An image behind the token, as an object URL for <img src>: auth = {token, clientId}. A failed fetch is not cached.
export function loadImage(url, auth = {}) {
  if (!images.has(url)) {
    const loading = (async () => {
      const res = await fetch(url, { headers: authHeaders(auth.token, auth.clientId) });
      if (!res.ok) throw await refused(res, 'image refused');
      const objectUrl = URL.createObjectURL(await res.blob());
      objectUrls.add(objectUrl);
      return objectUrl;
    })().catch((err) => {
      if (images.get(url) === loading) images.delete(url);   // not the entry a clear and a newer call put in its place
      throw err;
    });
    images.set(url, loading);
  }
  return images.get(url);
}

// Forgets every image and revokes its object URL. A fetch still running when this is called finishes into the next
// cache's set, so the next clear revokes it.
export function clearImageCache() {
  for (const objectUrl of objectUrls) URL.revokeObjectURL(objectUrl);
  objectUrls.clear();
  images.clear();
}
