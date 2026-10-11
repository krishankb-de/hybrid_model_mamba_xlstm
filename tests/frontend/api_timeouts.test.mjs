// tests/frontend/api_timeouts.test.mjs — CHAT_UI_PLAN.md P8-A: the data layer's requests give up on a server that never answers.
//
// A half-open tunnel accepts a request and never replies, and fetch never fails by itself, so a Stop could hang at "Stopping…" for ever. Cancel, each
// poll and each image fetch run under a timeout (an AbortSignal that also carries the caller's own signal); the streaming read of a turn does not.
// An expired timeout is a failure of the transport: a TypeError marked `transport`, the shape fetch itself throws when the connection cannot be made,
// so the page's network-error path and pollMessage's backoff take it with no new code.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as api from '../../app/static/api.js';

const { cancelMessage, clearImageCache, loadImage, pollMessage, streamTurn } = api;

const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });

async function drain(events) {
  const got = [];
  for await (const e of events) got.push(e);
  return got;
}

// A promise's outcome within `ms`, so that a request that was never given a timeout fails a test cleanly instead of hanging the run.
async function outcome(promise, ms = 1500) {
  let timer;
  const hung = new Promise((resolve) => { timer = setTimeout(() => resolve({ hung: true }), ms); });
  try {
    return await Promise.race([promise.then((value) => ({ value }), (error) => ({ error })), hung]);
  } finally {
    clearTimeout(timer);
  }
}

// AbortSignal.timeout's timer does not keep node alive, so a test that waits on one holds the loop open itself.
function keepAlive(t) {
  const timer = setInterval(() => {}, 1000);
  t.after(() => clearInterval(timer));
}

// fetch that never answers and, like the real one, rejects with the signal's reason once the signal aborts.
function hangingFetch(t) {
  const calls = [];
  t.mock.method(globalThis, 'fetch', (url, init) => new Promise((resolve, reject) => {
    calls.push({ url: String(url), init });
    init.signal?.addEventListener('abort', () => reject(init.signal.reason), { once: true });
  }));
  return calls;
}

// Headers arrive at once and the body never ends: it errors with the signal's reason when the signal aborts, as a real body stream does.
function stalledBody(init) {
  return new Response(new ReadableStream({
    start(controller) { init.signal?.addEventListener('abort', () => controller.error(init.signal.reason), { once: true }); },
  }), { status: 200, headers: { 'Content-Type': 'application/json' } });
}

const timedOut = (err) => err instanceof TypeError && err.transport === true && /no answer in \d+ ms/.test(err.message);

test('the limits: 15 s for a cancel or a poll, a minute for a picture', () => {
  assert.deepEqual([api.REQUEST_TIMEOUT_MS, api.IMAGE_TIMEOUT_MS], [15000, 60000]);
});

test('a cancel that is never answered fails with the transport error instead of hanging Stop at "Stopping…"', async (t) => {
  keepAlive(t);
  const calls = hangingFetch(t);
  const got = await outcome(cancelMessage({ messageId: 'm1', token: 'tok', clientId: 'cid', timeoutMs: 30 }));
  assert.ok(!got.hung, 'cancelMessage never gave up on a server that does not answer');
  assert.ok(timedOut(got.error), String(got.error));
  assert.equal(calls.length, 1);
  assert.ok(calls[0].init.signal.aborted, 'the request itself was aborted, so the connection is not left open');
});

test('a cancel whose answer starts but whose body stalls is cut off too', async (t) => {
  keepAlive(t);
  t.mock.method(globalThis, 'fetch', async (url, init) => stalledBody(init));
  const got = await outcome(cancelMessage({ messageId: 'm1', timeoutMs: 30 }));
  assert.ok(!got.hung && timedOut(got.error), String(got.hung ? 'hung' : got.error));
});

test('the page\'s read of a chat after a cancel gives up on a server that never answers, as a cancel does (getSession, P6 fix 2)', async (t) => {
  keepAlive(t);
  const calls = hangingFetch(t);
  const got = await outcome(api.getSession({ sessionId: 's_1', token: 'tok', clientId: 'cid', timeoutMs: 30 }));
  assert.ok(!got.hung, 'getSession never gave up on a server that does not answer');
  assert.ok(timedOut(got.error), String(got.error));
  assert.deepEqual([calls.length, calls[0].url, calls[0].init.headers.Authorization, calls[0].init.headers['X-Client-Id']],
                   [1, '/v1/sessions/s_1', 'Bearer tok', 'cid']);
  assert.ok(calls[0].init.signal.aborted, 'the request itself was aborted, so the connection is not left open');
});

test('a chat read in time is the chat; a refusal is the server\'s answer, not the transport\'s; the limit is 15 s unless one is given (P6 fix 2)', async (t) => {
  const chat = { id: 's_1', title: 'A', messages: [] };
  const signals = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => { signals.push(init.signal); return json(chat); });
  assert.deepEqual(await api.getSession({ sessionId: 's_1' }), chat);
  assert.ok(signals[0] instanceof AbortSignal && !signals[0].aborted, 'the default limit runs under a signal of its own');
  t.mock.method(globalThis, 'fetch', async () => json({ type: 'error', error: { type: 'not_found_error', message: 'gone' } }, 404));
  await assert.rejects(api.getSession({ sessionId: 's_gone', timeoutMs: 5000 }), (err) => err.status === 404 && !('transport' in err));
});

test('a cancel that is answered in time is not touched by its timeout', async (t) => {
  const answer = { id: 'm1', status: 'aborted', cancel_requested: true };
  t.mock.method(globalThis, 'fetch', async () => json(answer));
  assert.deepEqual(await cancelMessage({ messageId: 'm1', timeoutMs: 5000 }), answer);
  t.mock.method(globalThis, 'fetch', async () => json({ type: 'error', error: { type: 'not_found_error', message: 'gone' } }, 404));
  await assert.rejects(cancelMessage({ messageId: 'nope', timeoutMs: 5000 }),
                       (err) => err.status === 404 && !('transport' in err));   // a refusal is still the server's answer, not the transport's
});

test('a poll that is never answered is a network error: pollMessage asks again after its backoff and goes on', async (t) => {
  keepAlive(t);
  const calls = [];
  t.mock.method(globalThis, 'fetch', (url, init) => new Promise((resolve, reject) => {
    calls.push({ url: String(url), init });
    if (calls.length === 1) {   // the first poll hangs; the second is answered
      init.signal?.addEventListener('abort', () => reject(init.signal.reason), { once: true });
    } else {
      resolve(json({ id: 'm1', status: 'done', events: [{ seq: 5, event: 'message_stop', data: { seq: 5 } }] }));
    }
  }));
  const got = await outcome(drain(pollMessage({ messageId: 'm1', after: 4, intervalMs: 1, maxBackoffMs: 2, timeoutMs: 30 })));
  assert.ok(!got.hung, 'a poll that is never answered stopped the polling for good');
  assert.deepEqual(got.value, [{ event: 'message_stop', data: { seq: 5 } }]);
  assert.equal(calls.length, 2);
  assert.ok(calls.every((c) => c.url === '/v1/messages/m1?after=4'), 'it asks again for the same events');
});

test('polls that are never answered end as the 20th failure in a row: the transport error, not a hang', async (t) => {
  keepAlive(t);
  const calls = hangingFetch(t);
  const got = await outcome(drain(pollMessage({ messageId: 'm1', intervalMs: 1, maxBackoffMs: 2, timeoutMs: 5 })), 5000);
  assert.ok(!got.hung && timedOut(got.error), String(got.hung ? 'hung' : got.error));
  assert.equal(calls.length, 20);
});

test('a poll whose body stalls is cut off like one that is never answered', async (t) => {
  keepAlive(t);
  let n = 0;
  t.mock.method(globalThis, 'fetch', async (url, init) => (++n === 1 ? stalledBody(init) : json({ id: 'm1', status: 'done', events: [] })));
  const got = await outcome(drain(pollMessage({ messageId: 'm1', intervalMs: 1, maxBackoffMs: 2, timeoutMs: 30 })));
  assert.deepEqual(got.value, []);
  assert.equal(n, 2);
});

test('the caller\'s own abort still ends a poll quietly, and is not mistaken for a timeout', async (t) => {
  const calls = hangingFetch(t);
  const ctl = new AbortController();
  const next = pollMessage({ messageId: 'm1', signal: ctl.signal }).next();   // the default 15 s timeout is far off
  ctl.abort();
  assert.deepEqual(await outcome(next), { value: { value: undefined, done: true } });
  assert.equal(calls.length, 1);
});

test('a picture that is never answered fails with the transport error, which is not remembered: the next call asks again', async (t) => {
  keepAlive(t);
  clearImageCache();
  t.after(clearImageCache);
  t.mock.method(URL, 'createObjectURL', () => 'blob:test/0');
  const calls = hangingFetch(t);
  const first = await outcome(loadImage('/v1/messages/u1/image?variant=thumb', { token: 'tok' }, { timeoutMs: 30 }));
  assert.ok(!first.hung && timedOut(first.error), String(first.hung ? 'hung' : first.error));
  t.mock.method(globalThis, 'fetch', async () => new Response(new Blob(['png'], { type: 'image/png' })));
  assert.equal(await loadImage('/v1/messages/u1/image?variant=thumb', { token: 'tok' }, { timeoutMs: 5000 }), 'blob:test/0');
  assert.equal(calls.length, 1);
});

test('asked is the one timeout helper: it takes the fetch it is given, gives up on it in time, and reads inside the limit what it returns (P8 follow-up)', async (t) => {
  keepAlive(t);
  const calls = [];
  const hang = (url, init) => new Promise((resolve, reject) => {
    calls.push({ url: String(url), init });
    init.signal?.addEventListener('abort', () => reject(init.signal.reason), { once: true });
  });
  const global = t.mock.method(globalThis, 'fetch', () => { throw new Error('the global fetch was used'); });
  assert.equal(typeof api.asked, 'function', 'api.js exports asked, for the page\'s own requests');
  const got = await outcome(api.asked('/v1/sessions', { headers: { 'X-Client-Id': 'cid' } }, 30, (res) => res.json(), hang));
  assert.ok(!got.hung && timedOut(got.error), String(got.hung ? 'hung' : got.error));
  assert.equal(calls.length, 1);
  assert.deepEqual([calls[0].url, calls[0].init.headers], ['/v1/sessions', { 'X-Client-Id': 'cid' }]);
  assert.ok(calls[0].init.signal.aborted, 'the request itself was aborted, so the connection is not left open');
  assert.equal(global.mock.callCount(), 0);   // the fetch it was given, not the global one
  assert.deepEqual(await api.asked('/x', {}, 5000, (res) => res.json(), async () => json({ a: 1 })), { a: 1 });   // answered in time
  const stalled = await outcome(api.asked('/x', {}, 30, (res) => res.json(), async (url, init) => stalledBody(init)));   // a body that never ends
  assert.ok(!stalled.hung && timedOut(stalled.error), String(stalled.hung ? 'hung' : stalled.error));
  const ctl = new AbortController();
  const aborted = outcome(api.asked('/x', { signal: ctl.signal }, 5000, (res) => res.json(), hang));
  ctl.abort();
  const cancelled = await aborted;
  assert.ok(!cancelled.hung && cancelled.error?.name !== undefined && !timedOut(cancelled.error), 'the caller\'s own abort is no timeout');
  assert.equal(cancelled.error.name, 'AbortError');
});

test('a refusal carries its Retry-After as whole seconds on the error every request of the data layer throws, and null for anything else (P8 follow-up)', async (t) => {
  const cases = [['45', 45], ['  7 ', 7], ['0', 0], ['soon', null], ['-3', null], ['1.5', null], ['', null], [undefined, null], ['Wed, 21 Oct 2026 07:28:00 GMT', null]];
  let n = 0;
  t.mock.method(globalThis, 'fetch', async () => {
    const [raw] = cases[Math.floor(n++ / 2)];   // each case answers a turn and a cancel
    return new Response(JSON.stringify({ type: 'error', error: { type: 'rate_limited', message: 'Too many turns.' } }),
                        { status: 429, headers: { 'Content-Type': 'application/json', ...(raw === undefined ? {} : { 'Retry-After': raw }) } });
  });
  for (const [raw, want] of cases) {
    await assert.rejects(drain(streamTurn({ sessionId: 's1', form: new FormData() })), (err) => err.status === 429 && err.retryAfter === want, `streamTurn, ${raw}`);
    await assert.rejects(cancelMessage({ messageId: 'm1' }), (err) => err.status === 429 && err.retryAfter === want, `cancelMessage, ${raw}`);
  }
  assert.equal(api.retryAfterSeconds(new Response('', { headers: { 'Retry-After': '12' } })), 12);
  assert.equal(api.retryAfterSeconds(null), null);
  assert.equal(api.retryAfterSeconds({}), null);
});

test('the streaming read of a turn has no timeout: its request carries the caller\'s signal as it is', async (t) => {
  const seen = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    seen.push(init.signal);
    return new Response('', { status: 200 });
  });
  const ctl = new AbortController();
  await drain(streamTurn({ sessionId: 's1', form: new FormData(), signal: ctl.signal }));
  await drain(streamTurn({ sessionId: 's1', form: new FormData() }));
  assert.equal(seen[0], ctl.signal);       // not wrapped, not combined with a timer
  assert.equal(seen[1], undefined);
});

test('where AbortSignal cannot time out (a browser before 2024) the requests run as they always did', async (t) => {
  const real = [AbortSignal.timeout, AbortSignal.any];
  t.after(() => { [AbortSignal.timeout, AbortSignal.any] = real; });
  AbortSignal.timeout = undefined;
  AbortSignal.any = undefined;
  const answer = { id: 'm1', status: 'running', cancel_requested: true };
  const signals = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    signals.push(init.signal);
    return url.includes('/cancel') ? json(answer) : json({ id: 'm1', status: 'done', events: [] });
  });
  assert.deepEqual(await cancelMessage({ messageId: 'm1' }), answer);
  const ctl = new AbortController();
  assert.deepEqual(await drain(pollMessage({ messageId: 'm1', signal: ctl.signal, intervalMs: 1 })), []);
  assert.deepEqual(signals, [undefined, ctl.signal]);   // no timer anywhere, and the caller's signal is the poll's own
});
