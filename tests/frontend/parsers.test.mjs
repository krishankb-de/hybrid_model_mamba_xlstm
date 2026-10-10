// tests/frontend/parsers.test.mjs
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import {
  authHeaders, cancelMessage, clearImageCache, loadImage, parseSSE, pollMessage, streamTurn,
} from '../../app/static/api.js';
import * as api from '../../app/static/api.js';
import { STAGES, applyEvent, initialView, labelsPending, replay, stageState } from '../../app/static/state.js';

test('a frame split across chunks reassembles', () => {
  const whole = 'event: stage_start\ndata: {"seq":1,"stage":"encode"}\n\n';
  let { events, rest } = parseSSE(whole.slice(0, 17));
  assert.equal(events.length, 0);
  ({ events, rest } = parseSSE(rest + whole.slice(17)));
  assert.deepEqual(events, [{ event: 'stage_start', data: { seq: 1, stage: 'encode' } }]);
  assert.equal(rest, '');
});

test('CRLF, a CR/LF split across chunks, multi-line data and ping comments', () => {
  let { events, rest } = parseSSE(': ping\r\n\r\nevent: x\r\ndata: {"a":\r');
  ({ events, rest } = parseSSE(rest + '\ndata: 1, "seq": 1}\r\n\r\n'));
  assert.deepEqual(events, [{ event: 'x', data: { a: 1, seq: 1 } }]);
});

test('replaying the stored log rebuilds the live view exactly', () => {
  const log = JSON.parse(readFileSync(new URL('./fixtures/turn_tiny.json', import.meta.url)));
  let live = initialView(log[0].data.message_id);
  for (const e of log) live = applyEvent(live, e);
  assert.deepEqual(replay(log), live);
  assert.equal(live.status, 'done');
  assert.deepEqual(Object.keys(live.stages), ['preprocess', 'encode', 'retrieve', 'generate', 'label', 'score']);
});

test('an event repeated by polling is ignored', () => {
  const log = JSON.parse(readFileSync(new URL('./fixtures/turn_tiny.json', import.meta.url)));
  const once = replay(log);
  assert.deepEqual(replay([...log, ...log.slice(3, 9)]), once);
});

// ---- beyond the brief: the rest of the data layer and the reducer edge cases (P4-B) ----------------------------------

const recorded = () => JSON.parse(readFileSync(new URL('./fixtures/turn_tiny.json', import.meta.url)));
const ev = (seq, event, data = {}) => ({ event, data: { ...data, seq } });
const wire = (event, data) => `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
const envelope = (type, message) => ({ type: 'error', error: { type, message } });   // the body of every refusal
const json = (body, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
const tick = () => new Promise((resolve) => setImmediate(resolve));

async function drain(events) {
  const got = [];
  for await (const e of events) got.push(e);
  return got;
}

async function until(condition) {
  for (let i = 0; i < 100 && !condition(); i++) await tick();
  assert.ok(condition(), 'the condition was never reached');
}

// What assert.rejects checks of a refused request: an Error with the status and the server's parsed error body.
const refusedWith = (status, body) => (err) => {
  assert.ok(err instanceof Error);
  assert.deepEqual([err.status, err.body], [status, body]);
  return true;
};

// Replaces fetch for one test (node:test puts the real one back at its end); every call is kept as { url, init }.
function stubFetch(t, respond) {
  const calls = [];
  t.mock.method(globalThis, 'fetch', async (url, init) => {
    calls.push({ url: String(url), init });
    return respond(calls.length, url, init);
  });
  return calls;
}

// Notes every delay a timer is asked for and skips it, so a backoff of seconds costs a test nothing.
function skipWaits(t) {
  const waits = [];
  const realSetTimeout = globalThis.setTimeout;
  globalThis.setTimeout = (fn, ms, ...args) => {
    waits.push(ms);
    return realSetTimeout(fn, 0, ...args);
  };
  t.after(() => { globalThis.setTimeout = realSetTimeout; });
  return waits;
}

// A response body the test feeds by hand: a turn can sit queued with its headers sent and no byte of body.
function openStream() {
  let controller;
  const body = new ReadableStream({ start(c) { controller = c; } });
  return { body, push: (bytes) => controller.enqueue(bytes), end: () => controller.close() };
}

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    Object.values(value).forEach(deepFreeze);
  }
  return value;
}

test('parseSSE: data without a space, other fields skipped, a bare frame is "message", an unfinished one is kept', () => {
  const { events, rest } = parseSSE('id: 7\nretry: 1000\nevent: warning\ndata:{"seq":2}\n\n: ping\n\nevent: stage_st');
  assert.deepEqual(events, [{ event: 'warning', data: { seq: 2 } }]);
  assert.equal(rest, 'event: stage_st');
  assert.deepEqual(parseSSE('data: {"seq":3}\n\n').events, [{ event: 'message', data: { seq: 3 } }]);
});

test('parseSSE skips a frame whose data is not JSON, a bare "data:" included, and keeps the frames around it', () => {
  const { events, rest } = parseSSE(
    'data: {"seq":1}\n\nevent: x\ndata: {oops\n\ndata:\n\nevent: y\ndata: not json at all\n\ndata: {"seq":5}\n\nevent: tail');
  assert.deepEqual(events, [{ event: 'message', data: { seq: 1 } }, { event: 'message', data: { seq: 5 } }]);
  assert.equal(rest, 'event: tail');   // the unfinished frame is still the rest
  assert.deepEqual(parseSSE('data:\n\n').events, []);   // a bare data: alone is not an event, and not an error
});

test('parseSSE yields a frame only when its data is an object with an integer seq, and keeps the frames around the rest', () => {
  // JSON that parses but is no event. applyEvent reads data.seq of every event it is given: null throws there, and an
  // object without a seq would set lastSeq to undefined and turn the dedupe off for the rest of the turn.
  const notEvents = ['null', '5', '"text"', 'true', '[1]', '[]', '{}', '{"stage":"encode"}', '{"seq":"2"}', '{"seq":2.5}', '{"seq":null}'];
  for (const json of notEvents) {
    assert.deepEqual(parseSSE(`event: x\ndata: ${json}\n\n`).events, [], `data: ${json} was yielded`);
  }
  const between = notEvents.map((json, i) => `event: junk${i}\ndata: ${json}\n\n`).join('');
  const { events, rest } = parseSSE(wire('message_start', { seq: 1, message_id: 'm1' }) + between
    + wire('message_stop', { seq: 2, message_id: 'm1', status: 'done' }) + 'event: tail');
  assert.deepEqual(events.map((e) => [e.event, e.data.seq]), [['message_start', 1], ['message_stop', 2]]);
  assert.equal(rest, 'event: tail');
  assert.equal(events.reduce(applyEvent, initialView('m1')).lastSeq, 2);   // what survives folds, dedupe intact
});

test('parseSSE reads event: and data: without the space after the colon', () => {
  const { events } = parseSSE('event:warning\ndata:{"seq":4}\n\nevent:  stage_end  \ndata:  {"seq":5}\n\n');
  assert.deepEqual(events, [{ event: 'warning', data: { seq: 4 } }, { event: 'stage_end', data: { seq: 5 } }]);
});

test('authHeaders sends the bearer token and the client id only when there are some', () => {
  assert.deepEqual(authHeaders(), {});
  assert.deepEqual(authHeaders(null, ''), {});
  assert.deepEqual(authHeaders('tok'), { Authorization: 'Bearer tok' });
  assert.deepEqual(authHeaders(undefined, 'cid'), { 'X-Client-Id': 'cid' });
  assert.deepEqual(authHeaders('tok', 'cid'), { Authorization: 'Bearer tok', 'X-Client-Id': 'cid' });
});

// ---- the recorded turn -------------------------------------------------------------------------------------------

test('the recorded turn is a gap-free log that folds into the view the card shows', () => {
  const log = recorded();
  assert.deepEqual(log.map((e) => e.data.seq), log.map((_, i) => i + 1));
  const [start, stop] = [log[0].data, log.at(-1).data];
  const view = replay(log);
  assert.equal(view.id, start.message_id);
  assert.equal(view.lastSeq, log.length);
  assert.deepEqual([view.mode, view.provenance, view.options, view.image],
                   [start.mode, start.model, start.options, start.image]);
  assert.deepEqual([view.report, view.displayReport, view.truncated, view.totalMs],
                   [stop.report, stop.display_report, stop.truncated_mid_sentence, stop.total_ms]);
  assert.deepEqual([view.provisional, view.error, view.notices], [false, null, []]);

  // A skipped stage contributes nothing to the fields derived from its detail. The recording has no gallery, labeler
  // or reference, so retrieve, label and score all end skipped, and none of them leaves a truthy empty value behind.
  assert.deepEqual([view.stages.retrieve.state, view.stages.label.state, view.stages.score.state],
                   ['skipped', 'skipped', 'skipped']);
  assert.deepEqual([view.neighbors, view.matches, view.trueRank], [[], [], null]);
  assert.deepEqual([view.labels, view.agreement, view.score], [null, null, null]);
  assert.deepEqual(STAGES, Object.keys(view.stages));   // the contract order the timeline draws

  const firstSnapshot = log.find((e) => e.event === 'content_block_delta');   // a reload in the middle of the turn
  const midway = replay(log.slice(0, log.indexOf(firstSnapshot) + 1));
  assert.deepEqual([midway.status, midway.provisional, midway.report],
                   ['running', true, firstSnapshot.data.delta.text]);
  assert.deepEqual(midway.stages.generate, { state: 'running' });
});

test('applyEvent never mutates the view or the event it is given', () => {
  const log = recorded();
  let view = initialView(log[0].data.message_id);
  for (const e of log) view = applyEvent(deepFreeze(view), deepFreeze(e));   // a write to a frozen object throws in a module
  assert.deepEqual(view, replay(recorded()));
});

test('replaying an empty log gives the initial running view', () => {
  assert.deepEqual(replay([]), initialView(null));
  assert.equal(initialView('m1').status, 'running');
});

// ---- reducer edge cases ------------------------------------------------------------------------------------------

const foldLive = (log) => log.reduce((view, e) => applyEvent(view, e), initialView(log[0].data.message_id));
// message_stop as Pipeline._stop sends it when no report was produced (a cancel, an error, a question).
const stopOf = (id, status, more = {}) => ({
  message_id: id, status, total_ms: 321.5, report: null, display_report: null, truncated_mid_sentence: false,
  disclaimer: 'Research prototype; not for clinical use.', ...more,
});
const cutAfterSnapshot = (log, n) => {   // the recorded turn up to its n-th report snapshot
  const snapshots = log.filter((e) => e.event === 'content_block_delta');
  return log.slice(0, log.indexOf(snapshots[n - 1]) + 1);
};

// What a finished turn's view satisfies however it ended: nothing provisional, text fields are strings, replay == live.
function assertSettled(log) {
  const view = replay(log);
  assert.notEqual(view.status, 'running');
  assert.equal(view.provisional, false);
  assert.equal(typeof view.report, 'string');
  assert.equal(typeof view.displayReport, 'string');
  assert.equal(typeof view.totalMs, 'number');
  assert.equal(view.lastSeq, log.at(-1).data.seq);
  assert.deepEqual(view, foldLive(log));
  return view;
}

test('a question turn (message_start with image null) folds into a settled view with no stages', () => {
  const start = recorded()[0].data;
  const view = assertSettled([
    ev(1, 'message_start', { ...start, image: null }),
    ev(2, 'warning', { code: 'not_a_command', message: 'This is a report generator, not a question answerer.' }),
    ev(3, 'message_stop', stopOf(start.message_id, 'done')),
  ]);
  assert.equal(view.image, null);
  assert.deepEqual(view.stages, {});
  assert.deepEqual(view.notices.map((n) => n.code), ['not_a_command']);
  assert.deepEqual([view.status, view.report, view.displayReport, view.truncated, view.error],
                   ['done', '', '', false, null]);
});

test('a cancelled turn with no content_block_stop ends aborted, keeps its partial report and settles', () => {
  const log = recorded();
  const id = log[0].data.message_id;
  const partial = cutAfterSnapshot(log, 3);
  const last = partial.at(-1);
  const midway = replay(partial);
  assert.deepEqual([midway.status, midway.provisional, midway.report], ['running', true, last.data.delta.text]);

  const view = assertSettled([...partial, ev(last.data.seq + 1, 'message_stop', stopOf(id, 'aborted'))]);
  assert.deepEqual([view.status, view.report, view.displayReport, view.truncated, view.error],
                   ['aborted', last.data.delta.text, last.data.delta.text, false, null]);
  // The log never ended generate, so what the timeline shows comes from stageState: the stages that finished stay as
  // they were, and the one the Stop caught and the ones it never reached are closed as stopped.
  const stopped = { state: 'skipped', skipped: 'stopped' };
  assert.deepEqual(STAGES.map((s) => stageState(view, s)),
                   [view.stages.preprocess, view.stages.encode, view.stages.retrieve, stopped, stopped, stopped]);
  assert.equal(stageState(midway, 'generate').state, 'running');   // the same log while the turn was still going

  const queued = assertSettled([log[0], ev(2, 'message_stop', stopOf(id, 'aborted'))]);   // stopped while still queued
  assert.deepEqual([queued.status, queued.report, queued.stages], ['aborted', '', {}]);
  assert.deepEqual(STAGES.map((s) => stageState(queued, s)), STAGES.map(() => stopped));
});

test('an error event then message_stop(error) keeps the error on a settled view', () => {
  const log = recorded();
  const id = log[0].data.message_id;
  const partial = cutAfterSnapshot(log, 2);
  const seq = partial.at(-1).data.seq;
  const failure = envelope('model_error', 'Internal error (RuntimeError)');
  const failing = [...partial, ev(seq + 1, 'error', failure)];
  const before = replay(failing);   // an error alone does not end the turn: message_stop does
  assert.deepEqual([before.status, before.error], ['running', failure.error]);

  const view = assertSettled([...failing, ev(seq + 2, 'message_stop', stopOf(id, 'error'))]);
  assert.deepEqual([view.status, view.error, view.report], ['error', failure.error, partial.at(-1).data.delta.text]);
  const notRun = { state: 'skipped', skipped: 'not_run' };   // generate was the stage that failed
  assert.deepEqual(STAGES.map((s) => stageState(view, s)),
                   [view.stages.preprocess, view.stages.encode, view.stages.retrieve, { state: 'error' }, notRun, notRun]);
});

// ---- stageState and labelsPending: what the timeline and the label chips show -----------------------------------------

test('stageState closes what a finished, stopped or failed turn left running or pending, and nothing else', () => {
  const records = {   // what view.stages holds for the stage under test; undefined: the log never mentioned it
    running: { state: 'running' },
    done: { state: 'done', ms: 12.5, detail: { tokens: 16 } },
    skipped: { state: 'skipped', skipped: 'gallery_unavailable' },
  };
  const stopped = { state: 'skipped', skipped: 'stopped' };
  const notRun = { state: 'skipped', skipped: 'not_run' };
  const pending = { state: 'pending' };
  // What the log holds, for the stage under test (generate) and for the turn: the stage's own record, and whether the
  // turn recorded some other stage (preprocess) or nothing at all.
  const logs = {
    'never mentioned, nothing else recorded': {},
    'never mentioned, another stage recorded': { preprocess: records.done },
    running: { generate: records.running },
    done: { generate: records.done },
    skipped: { generate: records.skipped },
  };
  const expected = {   // turn status -> what the log holds -> what stageState says of generate
    running: { 'never mentioned, nothing else recorded': pending, 'never mentioned, another stage recorded': pending,
               running: records.running, done: records.done, skipped: records.skipped },
    // Public mode never sends the score stage, so a finished turn can have stages its log never mentions: those are
    // settled as not run. A question turn ran no stage at all, records none, and stays as it is.
    done: { 'never mentioned, nothing else recorded': pending, 'never mentioned, another stage recorded': notRun,
            running: records.running, done: records.done, skipped: records.skipped },
    aborted: { 'never mentioned, nothing else recorded': stopped, 'never mentioned, another stage recorded': stopped,
               running: stopped, done: records.done, skipped: records.skipped },
    error: { 'never mentioned, nothing else recorded': notRun, 'never mentioned, another stage recorded': notRun,
             running: { state: 'error' }, done: records.done, skipped: records.skipped },
  };
  for (const [status, row] of Object.entries(expected)) {
    for (const [kind, want] of Object.entries(row)) {
      const view = deepFreeze({ ...initialView('m1'), status, stages: logs[kind] });   // frozen: the selector must not write to it
      assert.deepEqual(stageState(view, 'generate'), want, `a ${status} turn, ${kind}`);
    }
  }
});

test('a finished turn whose log lacks a stage settles it: the public log has no score stage, a question turn has none at all', () => {
  const without = (log, stage) => log.filter((e) => e.data.stage !== stage).map((e, i) => ({ ...e, data: { ...e.data, seq: i + 1 } }));
  const recordedLog = recorded();
  const publicLike = replay(without(recordedLog, 'score'));   // what app/redact.py sends: no stage_start or stage_end of score
  assert.equal('score' in publicLike.stages, false);
  assert.equal(publicLike.status, 'done');
  assert.deepEqual(stageState(publicLike, 'score'), { state: 'skipped', skipped: 'not_run' });
  assert.deepEqual(STAGES.map((s) => stageState(publicLike, s).state), ['done', 'done', 'skipped', 'done', 'skipped', 'skipped']);   // retrieve, label: skipped in the recording
  const question = replay([recordedLog[0], { event: 'message_stop', data: { ...recordedLog.at(-1).data, seq: 2 } }]);
  assert.deepEqual(question.stages, {});
  assert.deepEqual(STAGES.map((s) => stageState(question, s)), STAGES.map(() => ({ state: 'pending' })));   // nothing ran: nothing to settle
});

test('stageState and STAGES name the six stages in contract order', () => {
  assert.deepEqual(STAGES, ['preprocess', 'encode', 'retrieve', 'generate', 'label', 'score']);
  assert.deepEqual(STAGES.map((s) => stageState(initialView('m1'), s)), STAGES.map(() => ({ state: 'pending' })));
});

test('STAGES is frozen: a renderer that sorts or pushes to it throws instead of reordering every card', () => {
  assert.ok(Object.isFrozen(STAGES));
  assert.throws(() => STAGES.sort(), TypeError);   // the contract order is not alphabetical, so a sort would reorder it
  assert.throws(() => STAGES.push('extra'), TypeError);
  assert.deepEqual(STAGES, ['preprocess', 'encode', 'retrieve', 'generate', 'label', 'score']);   // and it is untouched
});

test('labelsPending is true only while the turn runs and the label stage has not ended', () => {
  const view = (status, label) => ({ ...initialView('m1'), status, stages: label ? { label } : {} });
  assert.equal(labelsPending(view('running')), true);                                          // not reached yet
  assert.equal(labelsPending(view('running', { state: 'running' })), true);                    // labelling now
  assert.equal(labelsPending(view('running', { state: 'done', ms: 5, detail: {} })), false);
  assert.equal(labelsPending(view('running', { state: 'skipped', skipped: 'label_off' })), false);
  for (const status of ['done', 'aborted', 'error']) {
    assert.equal(labelsPending(view(status)), false, `a ${status} turn`);
    assert.equal(labelsPending(view(status, { state: 'running' })), false, `a ${status} turn, label still running`);
  }
});

test('warnings accumulate in arrival order and do not end or stage anything', () => {
  let view = applyEvent(initialView('m1'), ev(1, 'warning', { code: 'reference_ignored_public', message: 'one' }));
  view = applyEvent(view, ev(2, 'warning', { code: 'not_a_command', message: 'two' }));
  assert.deepEqual(view.notices.map((n) => [n.seq, n.code]), [[1, 'reference_ignored_public'], [2, 'not_a_command']]);
  assert.deepEqual([view.status, view.stages], ['running', {}]);
});

test('message_stop keeps the raw report and the display copy apart and carries the truncation flag', () => {
  const stop = stopOf('m1', 'done', {
    report: 'Findings: a. Impression: b and', display_report: 'Findings: a. Impression: b.', truncated_mid_sentence: true,
  });
  const view = applyEvent(initialView('m1'), ev(1, 'message_stop', stop));
  assert.deepEqual([view.report, view.displayReport, view.truncated], [stop.report, stop.display_report, true]);
});

test('retrieve, label and score details land in their view fields; a skipped stage keeps its reason', () => {
  const end = (seq, stage, detail) => ev(seq, 'stage_end', { stage, ms: 3.2, detail });
  const retrieve = {
    image_neighbors: [{ rank: 1, similarity: 0.91 }], report_matches: [{ rank: 1, similarity: 0.62, report: 'synthetic' }],
    true_report_rank: { rank: 3, of: 2663 },
  };
  const label = {
    chexbert_14: { Cardiomegaly: 1 }, positives: ['Cardiomegaly'], neighbor_agreement: [{ rank: 1, agree: 13, of: 14 }],
  };
  const score = { rouge_l: 0.2, bleu_1: 0.3, bleu_4: 0.1, reference_source: 'user' };
  let view = applyEvent(initialView('m1'), end(1, 'retrieve', retrieve));
  assert.deepEqual([view.neighbors, view.matches, view.trueRank],
                   [retrieve.image_neighbors, retrieve.report_matches, retrieve.true_report_rank]);
  assert.deepEqual(view.stages.retrieve, { state: 'done', ms: 3.2, detail: retrieve });
  view = applyEvent(view, end(2, 'label', label));
  assert.deepEqual([view.labels, view.agreement], [label.chexbert_14, label.neighbor_agreement]);
  view = applyEvent(view, end(3, 'score', score));
  assert.deepEqual(view.score, score);
  view = applyEvent(view, ev(4, 'stage_end', { stage: 'generate', skipped: 'not_run' }));
  assert.deepEqual(view.stages.generate, { state: 'skipped', skipped: 'not_run' });

  // Public mode: the server keeps rank and similarity only, and the stage ends without the rest.
  const pub = applyEvent(initialView('m2'), end(1, 'retrieve', { image_neighbors: [{ rank: 1, similarity: 0.9 }] }));
  assert.deepEqual([pub.matches, pub.trueRank, pub.labels, pub.agreement, pub.score], [[], null, null, null, null]);
  assert.equal(applyEvent(initialView('m2'), end(1, 'label', {})).labels, null);
});

test('a skipped retrieve, label or score stage contributes nothing to the fields derived from its detail', () => {
  const skip = (seq, stage, reason) => ev(seq, 'stage_end', { stage, skipped: reason });
  let view = initialView('m1');
  view = applyEvent(view, skip(1, 'retrieve', 'gallery_unavailable'));
  view = applyEvent(view, skip(2, 'label', 'labeler_unavailable'));
  view = applyEvent(view, skip(3, 'score', 'no_reference'));
  assert.deepEqual([view.neighbors, view.matches, view.trueRank], [[], [], null]);
  assert.deepEqual([view.labels, view.agreement, view.score], [null, null, null]);   // null, not {}: {} is truthy
  assert.deepEqual([view.stages.retrieve, view.stages.label, view.stages.score], [
    { state: 'skipped', skipped: 'gallery_unavailable' }, { state: 'skipped', skipped: 'labeler_unavailable' },
    { state: 'skipped', skipped: 'no_reference' },
  ]);
  const done = applyEvent(initialView('m2'), ev(1, 'stage_end', { stage: 'score', ms: 2, detail: { rouge_l: 0.2 } }));
  assert.deepEqual(done.score, { rouge_l: 0.2 });   // a stage that ran still fills its field

  // Skipped means nothing is read from it, whatever else the event carries.
  const stray = (seq, stage, detail) => ev(seq, 'stage_end', { stage, skipped: 'not_run', detail });
  view = applyEvent(initialView('m3'), stray(1, 'retrieve', { image_neighbors: [{ rank: 1 }], report_matches: [{ rank: 1 }],
                                                                 true_report_rank: { rank: 1 } }));
  view = applyEvent(view, stray(2, 'label', { chexbert_14: { Cardiomegaly: 1 }, neighbor_agreement: [{ rank: 1 }] }));
  view = applyEvent(view, stray(3, 'score', { rouge_l: 0.9 }));
  assert.deepEqual([view.neighbors, view.matches, view.trueRank, view.labels, view.agreement, view.score],
                   [[], [], null, null, null, null]);
});

// ---- pollMessage -------------------------------------------------------------------------------------------------

test('pollMessage yields each stored event once, in order, and stops when the turn is no longer running', async (t) => {
  const row = (seq, event, data = {}) => ({ seq, event, data: { ...data, seq } });   // as GET /v1/messages/{id} returns them
  const stored = [
    row(1, 'message_start', { message_id: 'm1' }), row(2, 'stage_start', { stage: 'preprocess', index: 0 }),
    row(3, 'stage_end', { stage: 'preprocess', ms: 1.5, detail: {} }), row(4, 'message_stop', { message_id: 'm1', status: 'done' }),
  ];
  const pages = [
    { id: 'm1', status: 'running', events: [stored[0], stored[1]] },
    { id: 'm1', status: 'running', events: [stored[1], stored[2]] },   // the row polled before is sent again
    { id: 'm1', status: 'done', events: [stored[3]] },                 // the last page carries the end of the turn
  ];
  const calls = stubFetch(t, (n) => json(pages[n - 1]));
  const got = await drain(pollMessage({
    base: '/proxy', messageId: 'm1', after: 0, token: 'tok', clientId: 'cid', intervalMs: 1,
  }));

  assert.deepEqual(got, stored.map(({ event, data }) => ({ event, data })));
  assert.deepEqual(calls.map((c) => c.url), [
    '/proxy/v1/messages/m1?after=0', '/proxy/v1/messages/m1?after=2', '/proxy/v1/messages/m1?after=3',
  ]);
  assert.ok(calls.every((c) => c.init.headers.Authorization === 'Bearer tok' && c.init.headers['X-Client-Id'] === 'cid'));
  assert.deepEqual(replay(got), replay(stored));   // what polling yields folds as the stream would have
});

test('pollMessage resumes after the seq it is given', async (t) => {
  const row = (seq) => ({ seq, event: 'stage_start', data: { seq, stage: 'encode' } });
  const calls = stubFetch(t, () => json({ id: 'm1', status: 'error', events: [row(7), row(8)] }));
  const got = await drain(pollMessage({ messageId: 'm1', after: 7, intervalMs: 1 }));
  assert.deepEqual(got.map((e) => e.data.seq), [8]);
  assert.deepEqual(calls.map((c) => c.url), ['/v1/messages/m1?after=7']);   // a finished turn is read once
});

test('pollMessage waits 500 ms between polls unless told otherwise', async (t) => {
  const waits = skipWaits(t);
  stubFetch(t, (n) => json({ id: 'm1', status: n < 3 ? 'running' : 'done', events: [] }));
  assert.deepEqual(await drain(pollMessage({ messageId: 'm1' })), []);
  assert.deepEqual(waits, [500, 500]);
});

test('pollMessage ends quietly when its signal aborts during the wait, and leaves no timer behind', async (t) => {
  const unhandled = [];
  const onUnhandled = (reason) => unhandled.push(reason);
  process.on('unhandledRejection', onUnhandled);
  const [realSetTimeout, realClearTimeout] = [globalThis.setTimeout, globalThis.clearTimeout];
  const longTimers = new Set();   // the one-minute wait's timer, until it fires or is cleared
  globalThis.setTimeout = (fn, ms, ...args) => {
    const handle = realSetTimeout((...a) => { longTimers.delete(handle); fn(...a); }, ms, ...args);
    if (ms === 60_000) longTimers.add(handle);
    return handle;
  };
  globalThis.clearTimeout = (handle) => { longTimers.delete(handle); realClearTimeout(handle); };
  t.after(() => {
    process.off('unhandledRejection', onUnhandled);
    [globalThis.setTimeout, globalThis.clearTimeout] = [realSetTimeout, realClearTimeout];
  });
  stubFetch(t, () => json({ id: 'm1', status: 'running', events: [] }));

  const ctl = new AbortController();
  const next = pollMessage({ messageId: 'm1', signal: ctl.signal, intervalMs: 60_000 }).next();   // polls, then waits
  await until(() => longTimers.size === 1);
  ctl.abort();
  assert.deepEqual(await next, { value: undefined, done: true });
  assert.equal(longTimers.size, 0);
  await tick();
  await tick();
  assert.deepEqual(unhandled, []);
});

test('pollMessage ends quietly when its signal aborts mid-request, and makes no request once aborted', async (t) => {
  t.mock.method(globalThis, 'fetch', (url, init) => new Promise((resolve, reject) => {   // like fetch: rejects on abort
    init.signal.addEventListener('abort', () => reject(new DOMException('This operation was aborted', 'AbortError')));
  }));
  const ctl = new AbortController();
  const next = pollMessage({ messageId: 'm1', signal: ctl.signal }).next();
  ctl.abort();
  assert.deepEqual(await next, { value: undefined, done: true });

  const requests = globalThis.fetch.mock.callCount();
  assert.deepEqual(await pollMessage({ messageId: 'm1', signal: ctl.signal }).next(), { value: undefined, done: true });
  assert.equal(globalThis.fetch.mock.callCount(), requests);
});

test('pollMessage throws a refusal with its status and parsed body', async (t) => {
  const gone = envelope('not_found_error', 'Message not found.');
  stubFetch(t, () => json(gone, 404));
  await assert.rejects(drain(pollMessage({ messageId: 'nope', intervalMs: 1 })), refusedWith(404, gone));
});

// ---- pollMessage riding out failures (D7: it is the fallback when the stream drops, so it must outlast a flap) ------

const netDown = () => { throw new TypeError('fetch failed'); };   // what fetch throws when the connection cannot be made
const polled = (calls) => calls.map((c) => c.url.split('?')[1]);   // the query string of each request
const page = (status, events = []) => json({ id: 'm1', status, events });

test('pollMessage rides out two network errors and a 502: every event arrives once, asked for after the same seq', async (t) => {
  const row = (seq, event) => ({ seq, event, data: { seq } });
  const stored = [row(1, 'message_start'), row(2, 'stage_start'), row(3, 'stage_end'), row(4, 'message_stop')];
  const steps = [
    () => page('running', [stored[0], stored[1]]),
    netDown,
    netDown,
    () => new Response('<html>Bad Gateway</html>', { status: 502 }),   // a proxy's page, not the server's JSON
    () => page('done', [stored[2], stored[3]]),
  ];
  const calls = stubFetch(t, (n) => steps[n - 1]());
  const got = await drain(pollMessage({ messageId: 'm1', intervalMs: 1, maxBackoffMs: 4 }));
  assert.deepEqual(got, stored.map(({ event, data }) => ({ event, data })));   // nothing lost, nothing twice
  assert.deepEqual(polled(calls), ['after=0', 'after=2', 'after=2', 'after=2', 'after=2']);
});

test('pollMessage retries 429, 502, 503 and 504', async (t) => {
  let status;
  const calls = stubFetch(t, (n) => (n % 2 === 1 ? json(envelope('overloaded_error', 'busy'), status) : page('done')));
  for (status of [429, 502, 503, 504]) {
    calls.length = 0;
    assert.deepEqual(await drain(pollMessage({ messageId: 'm1', intervalMs: 1, maxBackoffMs: 2 })), []);
    assert.equal(calls.length, 2, `after a ${status}`);   // refused once, then asked again
  }
});

test('pollMessage throws at once on 400, 401, 403, 404, 422 and 500, with the status and the parsed body', async (t) => {
  const waits = skipWaits(t);
  let status;
  const calls = stubFetch(t, () => json(envelope('some_error', `refused with ${status}`), status));
  for (status of [400, 401, 403, 404, 422, 500]) {
    calls.length = 0;
    await assert.rejects(drain(pollMessage({ messageId: 'm1' })), refusedWith(status, envelope('some_error', `refused with ${status}`)));
    assert.equal(calls.length, 1, `a ${status} is final`);
  }
  assert.deepEqual(waits, []);   // and no backoff was waited
});

test('pollMessage gives up after 20 consecutive failures and throws the last one', async (t) => {
  let respond = netDown;
  const calls = stubFetch(t, () => respond());
  await assert.rejects(drain(pollMessage({ messageId: 'm1', intervalMs: 1, maxBackoffMs: 2 })), TypeError);
  assert.equal(calls.length, 20);

  const busy = envelope('overloaded_error', 'busy');   // an HTTP failure keeps its status and body when it is given up on
  respond = () => json(busy, 503);
  calls.length = 0;
  await assert.rejects(drain(pollMessage({ messageId: 'm1', intervalMs: 1, maxBackoffMs: 2 })), refusedWith(503, busy));
  assert.equal(calls.length, 20);
});

test('a successful poll resets the failure count', async (t) => {
  const plan = [...Array(19).fill('fail'), 'running', ...Array(19).fill('fail'), 'done'];   // never 20 in a row
  const calls = stubFetch(t, (n) => (plan[n - 1] === 'fail' ? netDown() : page(plan[n - 1])));
  assert.deepEqual(await drain(pollMessage({ messageId: 'm1', intervalMs: 1, maxBackoffMs: 2 })), []);
  assert.equal(calls.length, plan.length);
});

test('pollMessage backs off 500, 1000, 2000, 4000 and then 5000 ms, and starts over after a success', async (t) => {
  const waits = skipWaits(t);
  const plan = ['fail', 'fail', 'fail', 'fail', 'fail', 'fail', 'running', 'fail', 'done'];
  stubFetch(t, (n) => (plan[n - 1] === 'fail' ? netDown() : page(plan[n - 1])));
  await drain(pollMessage({ messageId: 'm1' }));
  // six failures, the normal wait after the success, then the first backoff again (5000 if the count had not reset)
  assert.deepEqual(waits, [500, 1000, 2000, 4000, 5000, 5000, 500, 500]);
});

test('pollMessage ends quietly when its signal aborts during a backoff wait, and leaves no timer behind', async (t) => {
  const unhandled = [];
  const onUnhandled = (reason) => unhandled.push(reason);
  process.on('unhandledRejection', onUnhandled);
  const [realSetTimeout, realClearTimeout] = [globalThis.setTimeout, globalThis.clearTimeout];
  const backoffTimers = new Set();   // the one-minute backoff's timer, until it fires or is cleared
  globalThis.setTimeout = (fn, ms, ...args) => {
    const handle = realSetTimeout((...a) => { backoffTimers.delete(handle); fn(...a); }, ms, ...args);
    if (ms === 60_000) backoffTimers.add(handle);
    return handle;
  };
  globalThis.clearTimeout = (handle) => { backoffTimers.delete(handle); realClearTimeout(handle); };
  t.after(() => {
    process.off('unhandledRejection', onUnhandled);
    [globalThis.setTimeout, globalThis.clearTimeout] = [realSetTimeout, realClearTimeout];
  });
  const calls = stubFetch(t, netDown);

  const ctl = new AbortController();
  const next = pollMessage({ messageId: 'm1', signal: ctl.signal, intervalMs: 60_000, maxBackoffMs: 60_000 }).next();   // fails, backs off
  await until(() => backoffTimers.size === 1);
  ctl.abort();
  assert.deepEqual(await next, { value: undefined, done: true });
  assert.equal(backoffTimers.size, 0);
  assert.equal(calls.length, 1);   // it did not try again
  await tick();
  await tick();
  assert.deepEqual(unhandled, []);
});

// ---- streamTurn and cancelMessage --------------------------------------------------------------------------------

test('streamTurn reports X-Message-Id before any body byte, then yields the parsed events', async (t) => {
  const stream = openStream();
  const calls = stubFetch(t, () => new Response(stream.body, {
    status: 200, headers: { 'Content-Type': 'text/event-stream', 'X-Message-Id': 'msg_1' },
  }));
  const form = new FormData();
  form.append('text', 'beam 5');
  const ctl = new AbortController();
  const ids = [];
  const turn = streamTurn({
    base: '/proxy', sessionId: 'ses_1', form, token: 'tok', clientId: 'cid', signal: ctl.signal,
    onMessageId: (id) => ids.push(id),
  });
  const events = [];
  const consumed = (async () => { for await (const e of turn) events.push(e); })();

  await until(() => ids.length > 0);
  assert.deepEqual(ids, ['msg_1']);
  assert.deepEqual(events, []);   // queued: the headers are in, the body has no byte yet, and Stop already has the id
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/proxy/v1/sessions/ses_1/messages');
  const { method, body, signal, headers } = calls[0].init;
  assert.deepEqual([method, body, signal], ['POST', form, ctl.signal]);
  assert.deepEqual(headers, { Authorization: 'Bearer tok', 'X-Client-Id': 'cid' });

  // A ping and three frames, cut in awkward places: inside the three bytes of the em dash, and just before the end.
  const snapshot = { seq: 2, index: 0, delta: { type: 'beam_snapshot', step: 1, text: 'No acute — change' } };
  const bytes = new TextEncoder().encode(': ping\n\n' + wire('message_start', { seq: 1, message_id: 'msg_1' })
    + wire('content_block_delta', snapshot) + wire('message_stop', { seq: 3, message_id: 'msg_1', status: 'done' }));
  const dash = bytes.indexOf(0xe2);
  const cuts = [0, 10, dash + 1, dash + 2, bytes.length - 3, bytes.length];
  for (let i = 1; i < cuts.length; i++) stream.push(bytes.slice(cuts[i - 1], cuts[i]));
  stream.end();
  await consumed;
  assert.deepEqual(events.map((e) => e.event), ['message_start', 'content_block_delta', 'message_stop']);
  assert.deepEqual(events.map((e) => e.data.seq), [1, 2, 3]);
  assert.equal(events[1].data.delta.text, 'No acute — change');
  assert.deepEqual(ids, ['msg_1']);   // once
});

test('streamTurn cancels the response body when the consumer stops early or onMessageId throws', async (t) => {
  const cancelled = [];
  const openBody = () => new ReadableStream({   // one frame, then it stays open like a turn still running
    start(c) { c.enqueue(new TextEncoder().encode(wire('message_start', { seq: 1, message_id: 'm1' }))); },
    cancel(reason) { cancelled.push(reason); },
  });
  stubFetch(t, () => new Response(openBody(), { status: 200, headers: { 'X-Message-Id': 'm1' } }));

  for await (const e of streamTurn({ sessionId: 's1', form: new FormData() })) {   // leaves without aborting
    assert.equal(e.event, 'message_start');
    break;
  }
  assert.equal(cancelled.length, 1);   // the connection is closed, not left open until the server ends the turn

  const boom = new Error('onMessageId failed');
  const turn = streamTurn({ sessionId: 's1', form: new FormData(), onMessageId: () => { throw boom; } });
  await assert.rejects(drain(turn), (err) => err === boom);   // the error is the consumer's own, and comes out
  assert.equal(cancelled.length, 2);   // the body is released all the same
});

test('streamTurn: a body whose cancel() rejects does not fail a consumer that stops early', async (t) => {
  // The cleanup in streamTurn's finally block is `await reader.cancel().catch(() => {})`. A plain `await reader.cancel()`
  // passes every other test here and turns a failing cleanup into an exception thrown at the consumer's `break`.
  const cancelAttempts = [];
  const body = new ReadableStream({   // one frame, then open; closing it fails
    start(c) { c.enqueue(new TextEncoder().encode(wire('message_start', { seq: 1, message_id: 'm1' }))); },
    cancel() { cancelAttempts.push('cancel'); throw new Error('the connection was already gone'); },
  });
  stubFetch(t, () => new Response(body, { status: 200, headers: { 'X-Message-Id': 'm1' } }));

  const seen = [];
  for await (const e of streamTurn({ sessionId: 's1', form: new FormData() })) {
    seen.push(e.event);
    break;   // must not throw from here
  }
  assert.deepEqual(seen, ['message_start']);
  assert.deepEqual(cancelAttempts, ['cancel']);   // it did try to close the body: the rejection was swallowed, not skipped
});

test('streamTurn marks a failure of the transport where it happens, and nothing else: the page swallows only those (P4-H)', async (t) => {
  const frame = new TextEncoder().encode(wire('message_start', { seq: 1, message_id: 'm1' }));
  const broken = () => new ReadableStream({   // one frame, then the connection dies
    start(c) { c.enqueue(frame); },
    pull(c) { c.error(new TypeError('network error')); },
  });
  let respond = () => { throw new TypeError('fetch failed'); };   // the connection cannot be made
  stubFetch(t, () => respond());
  const marked = (err) => err instanceof TypeError && err.transport === true;
  await assert.rejects(drain(streamTurn({ sessionId: 's1', form: new FormData() })), marked);
  respond = () => new Response(broken(), { status: 200, headers: { 'X-Message-Id': 'm1' } });
  await assert.rejects(drain(streamTurn({ sessionId: 's1', form: new FormData() })), marked);   // the body fails mid-stream
  const bug = new TypeError("Cannot read properties of undefined (reading 'id')");   // the caller's own code fails: no mark
  await assert.rejects(drain(streamTurn({ sessionId: 's1', form: new FormData(), onMessageId: () => { throw bug; } })),
                       (err) => err === bug && !('transport' in err));
  const invalid = envelope('validation_error', 'nope');   // a refusal is the server's answer, not the transport's
  respond = () => json(invalid, 422);
  await assert.rejects(drain(streamTurn({ sessionId: 's1', form: new FormData() })),
                       (err) => err.status === 422 && !('transport' in err));
});

test('pollMessage and cancelMessage mark a failure of the transport too', async (t) => {
  const calls = stubFetch(t, netDown);
  await assert.rejects(drain(pollMessage({ messageId: 'm1', intervalMs: 1, maxBackoffMs: 2 })), (err) => err.transport === true);
  assert.equal(calls.length, 20);
  await assert.rejects(cancelMessage({ messageId: 'm1' }), (err) => err instanceof TypeError && err.transport === true);
});

test('streamTurn goes on after a frame that is not JSON', async (t) => {
  const body = wire('message_start', { seq: 1, message_id: 'm1' }) + 'event: x\ndata: {oops\n\n' + wire('message_stop', { seq: 3 });
  stubFetch(t, () => new Response(body, { status: 200 }));
  const got = await drain(streamTurn({ sessionId: 's1', form: new FormData() }));
  assert.deepEqual(got.map((e) => e.event), ['message_start', 'message_stop']);
});

test('streamTurn throws a 422 with the parsed body, never reports an id, and tolerates a body that is not JSON', async (t) => {
  const invalid = envelope('validation_error', 'Invalid options: beam_size: Input should be less than or equal to 8');
  const calls = stubFetch(t, (n) => (n === 1 ? json(invalid, 422) : new Response('<html>Bad Gateway</html>', { status: 502 })));
  const ids = [];
  const turn = () => streamTurn({ sessionId: 's1', form: new FormData(), onMessageId: (id) => ids.push(id) });
  await assert.rejects(drain(turn()), refusedWith(422, invalid));
  await assert.rejects(drain(turn()), refusedWith(502, null));
  assert.deepEqual([ids, calls.length], [[], 2]);
});

test('cancelMessage POSTs the cancel route and returns the answer; a refusal throws its status and body', async (t) => {
  const answer = { id: 'm1', status: 'running', cancel_requested: true };
  const gone = envelope('not_found_error', 'Message not found.');
  const calls = stubFetch(t, (n) => (n === 1 ? json(answer) : json(gone, 404)));
  assert.deepEqual(await cancelMessage({ base: '/proxy', messageId: 'm1', token: 'tok', clientId: 'cid' }), answer);
  assert.equal(calls[0].url, '/proxy/v1/messages/m1/cancel');
  assert.equal(calls[0].init.method, 'POST');
  assert.deepEqual(calls[0].init.headers, { Authorization: 'Bearer tok', 'X-Client-Id': 'cid' });
  await assert.rejects(cancelMessage({ messageId: 'nope' }), refusedWith(404, gone));
});

// ---- loadImage ---------------------------------------------------------------------------------------------------

// The cache is module-wide: a test starts it empty and leaves it empty. Object URLs are stubbed to count and name them.
function stubObjectUrls(t) {
  const made = [];
  const revoked = [];
  t.mock.method(URL, 'createObjectURL', () => { made.push(`blob:test/${made.length}`); return made.at(-1); });
  t.mock.method(URL, 'revokeObjectURL', (objectUrl) => { revoked.push(objectUrl); });
  clearImageCache();
  t.after(clearImageCache);
  return { made, revoked };
}

test('loadImage fetches once with the auth headers for concurrent calls, and clearing revokes every object URL', async (t) => {
  const { revoked } = stubObjectUrls(t);
  const calls = stubFetch(t, () => new Response(new Blob(['png'], { type: 'image/png' })));
  const auth = { token: 'tok', clientId: 'cid' };
  const thumb = '/v1/messages/u1/image?variant=thumb';
  const [a, b] = await Promise.all([loadImage(thumb, auth), loadImage(thumb, auth)]);
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].init.headers, { Authorization: 'Bearer tok', 'X-Client-Id': 'cid' });
  assert.deepEqual([a, b], ['blob:test/0', 'blob:test/0']);

  assert.equal(await loadImage(thumb, auth), a);   // loaded already: no second fetch
  assert.equal(calls.length, 1);
  await loadImage('/v1/messages/u1/image?variant=original', auth);   // another URL is another fetch
  assert.equal(calls.length, 2);
  await loadImage('/v1/messages/u1/image?variant=model_input');   // no auth: no headers
  assert.deepEqual(calls[2].init.headers, {});

  clearImageCache();
  assert.deepEqual(revoked.sort(), ['blob:test/0', 'blob:test/1', 'blob:test/2']);   // every object URL, at once
  await loadImage(thumb, auth);
  assert.equal(calls.length, 4);   // the cache really was emptied
});

test('isSameOriginPath is the one filter for a path on this origin: a single slash and no control character, nothing else', () => {
  const { isSameOriginPath } = api;   // read off the namespace: a missing export fails this test only
  for (const ok of ['/', '/v1/x', '/v1/messages/u1/image?variant=a%2Fb&x=1#frag']) assert.equal(isSameOriginPath(ok), true, ok);
  const refused = ['//evil.example/x', '/\\evil.example/x', '\\\\evil.example/x', 'x.png', './x', '', 'https://e.example/x', 'blob:null/abc',
                   'data:image/png;base64,AAAA', 'javascript:alert(1)', '/\t/evil.example/x', '/\n/evil.example/x', '/\r/evil.example/x',
                   '/a\x00b', '/a\x1fb', '/a\x7fb', undefined, null, 42, {}, ['/x']];
  for (const bad of refused) assert.equal(isSameOriginPath(bad), false, JSON.stringify(bad));
});

test('loadImage takes a same-origin path only, so the bearer token never reaches another URL', async (t) => {
  stubObjectUrls(t);
  const calls = stubFetch(t, () => new Response(new Blob(['png'])));
  const auth = { token: 'tok', clientId: 'cid' };
  const refused = [
    'https://evil.example/x.png', 'http://evil.example/x.png', '//evil.example/x.png',   // another origin
    '/\\evil.example/x.png', '\\\\evil.example/x.png',                              // a backslash counts as a slash
    '/\t/evil.example/x.png', '/\n/evil.example/x.png', '/\r/evil.example/x.png',   // a tab, newline or CR is dropped, leaving //
    'x.png', './x.png', 'blob:null/abc', 'data:image/png;base64,AAAA', 'javascript:alert(1)', '', undefined, null, 42,
  ];
  for (const url of refused) {
    await assert.rejects(loadImage(url, auth), (err) => err instanceof Error, `${JSON.stringify(url)} was accepted`);
  }
  assert.equal(calls.length, 0);   // nothing was fetched, so no request carried the token anywhere

  assert.match(await loadImage('/v1/messages/u1/image?variant=thumb', auth), /^blob:/);   // a path on this origin is fine
  assert.match(await loadImage('/v1/messages/u1/image?variant=a%2Fb&x=1#frag'), /^blob:/);
  assert.equal(calls.length, 2);
  assert.equal(calls[0].url, '/v1/messages/u1/image?variant=thumb');
  assert.equal(calls[0].init.headers.Authorization, 'Bearer tok');
});

test('a refused image fetch rejects with its status, and is remembered until clearImageCache: one fetch per refused path (P6 fix 1)', async (t) => {
  stubObjectUrls(t);
  const gone = envelope('not_found_error', 'Message not found.');
  const calls = stubFetch(t, (n) => {
    if (n === 2) throw new TypeError('network down');
    return n === 1 ? json(gone, 404) : new Response(new Blob(['png']));
  });
  for (let i = 0; i < 10; i++) await assert.rejects(loadImage('/img/x', {}), refusedWith(404, gone));   // a card drawn ten times
  assert.equal(calls.length, 1);   // asked once: the server's answer is kept for this view, as a picture is
  clearImageCache();               // another chat, or a new token: asked again
  await assert.rejects(loadImage('/img/x', {}), (err) => err instanceof TypeError && err.transport === true);
  assert.equal(await loadImage('/img/x', {}), 'blob:test/0');   // the network failed, the server did not refuse: asked again at once, and it loads
  assert.equal(calls.length, 3);
});

test('an image whose body is cut off is a failure of the transport: not kept, and marked as the network\'s (P6 fix 1)', async (t) => {
  stubObjectUrls(t);
  let n = 0;
  const calls = stubFetch(t, () => {
    n += 1;
    return n === 1 ? { ok: true, status: 200, headers: new Headers(), blob: async () => { throw new TypeError('body stream lost'); } }
      : new Response(new Blob(['png']));
  });
  await assert.rejects(loadImage('/img/cut', {}), (err) => err instanceof TypeError && err.transport === true);
  assert.equal(await loadImage('/img/cut', {}), 'blob:test/0');
  assert.equal(calls.length, 2);
});
