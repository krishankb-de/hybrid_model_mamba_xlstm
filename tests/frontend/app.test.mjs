// tests/frontend/app.test.mjs — app/static/app.js (CHAT_UI_PLAN.md P4-D): the pure helpers, the watchdog state machine on
// an injected clock, and the page wiring on the DOM shim with a fake window, fake timers, a fake fetch and a fake api module.
//
// No MIMIC data and no server: events are synthetic (state.js folds them) or the recorded tiny turn (fixtures/turn_tiny.json).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { ShimEvent, installDom, serialize } from './dom_shim.mjs';
import {
  BOUNDS, CLIENT_KEY, COMMANDS, DEFAULT_SETTINGS, HEALTH_MS, HEALTH_SLOW_MS, SETTINGS_KEY, STALL_MS, browserStorage,
  checkImageFile, chosenCard, clientId, createApp, createWatchdog, errorMessage, exportFilename, healthText, isCommand, loadSettings,
  nearBottom, nextHealthDelay, optionsFromSettings, parseRoute, saveSettings, serverHas, sessionDate, sessionMeta, sessionTitle,
  userTurnMessage,
} from '../../app/static/app.js';
import { applyEvent, initialView } from '../../app/static/state.js';
import { el, renderAssistantCard } from '../../app/static/render.js';
import { transportFailure } from '../../app/static/api.js';

installDom();   // after the import above: with no #composer on the page, importing app.js started nothing

const SETTINGS_FILE = SETTINGS_KEY;
const memoryStorage = (initial = {}) => {
  const data = new Map(Object.entries(initial));
  return { data, getItem: (k) => (data.has(k) ? data.get(k) : null), setItem: (k, v) => { data.set(k, String(v)); }, removeItem: (k) => { data.delete(k); } };
};
const brokenStorage = () => ({ getItem() { throw new Error('blocked'); }, setItem() { throw new Error('blocked'); } });
const flush = async (n = 6) => { for (let i = 0; i < n; i++) await new Promise((resolve) => setImmediate(resolve)); };

// ---- routes ---------------------------------------------------------------------------------------------------------------

test('parseRoute reads #/s/<id> as a session, #/new as an empty chat, and anything else as home', () => {
  assert.deepEqual(parseRoute('#/s/s_01abc'), { kind: 'session', id: 's_01abc' });
  assert.deepEqual(parseRoute('#/s/s_01abc/'), { kind: 'session', id: 's_01abc' });   // a trailing slash is the same route
  assert.deepEqual(parseRoute('#/s/A-b_9'), { kind: 'session', id: 'A-b_9' });
  assert.deepEqual(parseRoute('#/s/%73_1'), { kind: 'session', id: 's_1' });          // decoded, then checked
  assert.deepEqual(parseRoute('#/new'), { kind: 'new' });
  for (const home of ['', '#', '#/', '#/s', '#/s/', '#/s//', '#/s/a/b', '#/s/a?x=1', '#/s/a#b', '#/s/..%2Fetc', '#/s/a b', '#/s/%E0%A4%A',
                      '#/s/' + 'a'.repeat(65), '#/other', '#new', 'x', '#/new/', null, undefined, 42, {}]) {
    assert.deepEqual(parseRoute(home), { kind: 'home' }, String(home));   // a malformed id never reaches a request
  }
  assert.deepEqual(parseRoute('#/s/' + 'a'.repeat(64)), { kind: 'session', id: 'a'.repeat(64) });
});

// ---- settings ---------------------------------------------------------------------------------------------------------------

test('loadSettings gives the defaults for no storage, an empty one, a broken one and a refusing one', () => {
  for (const storage of [undefined, null, memoryStorage(), brokenStorage(), {}, { getItem: () => undefined }, { getItem: () => 42 }]) {
    assert.deepEqual(loadSettings(storage), DEFAULT_SETTINGS);
  }
  assert.deepEqual(DEFAULT_SETTINGS, {
    model: '', decode: 'beam', beam_size: 3, max_new_tokens: 100, cached_decode: true, compile: false, k_images: 4, k_reports: 3,
    label: true, display_repair: true, stop_on_repeat: true, token: '',
  });   // the published protocol, but for two switches that the page has on and the server's Options has off: Display repair only changes what
        // is shown, and Stop when the report starts repeating ends a decoder that has no stop condition of its own (P4-G)
  assert.notEqual(loadSettings(undefined), DEFAULT_SETTINGS);   // a fresh object each time: the caller may change it
});

test('loadSettings reads what saveSettings wrote, and nothing it cannot trust', () => {
  const storage = memoryStorage();
  const mine = { ...DEFAULT_SETTINGS, model: 'm', decode: 'greedy', beam_size: 5, max_new_tokens: 150, cached_decode: false, compile: true,
                 k_images: 0, k_reports: 10, label: false, display_repair: true, token: 'secret' };
  assert.equal(saveSettings(storage, mine), true);
  assert.equal(typeof storage.data.get(SETTINGS_FILE), 'string');
  assert.deepEqual(loadSettings(storage), mine);

  const cases = [
    ['not JSON', '{nope'], ['an array', '[1,2]'], ['a number', '7'], ['null', 'null'], ['a string', '"x"'], ['empty', ''],
  ];
  for (const [name, raw] of cases) assert.deepEqual(loadSettings(memoryStorage({ [SETTINGS_FILE]: raw })), DEFAULT_SETTINGS, name);
  const junk = memoryStorage({ [SETTINGS_FILE]: JSON.stringify({
    model: 5, decode: 'sampling', beam_size: 'many', max_new_tokens: null, cached_decode: 'yes', compile: 1, k_images: {}, k_reports: [],
    label: 'no', display_repair: 0, stop_on_repeat: 'yes', token: 7, evil: '<img src=x onerror=alert(1)>', __proto__: { polluted: true },
  }) });
  assert.deepEqual(loadSettings(junk), DEFAULT_SETTINGS);   // each wrong type falls back to its own default
  assert.equal(Object.hasOwn(loadSettings(junk), 'evil'), false);   // and a key that is no setting is not carried
  const partial = loadSettings(memoryStorage({ [SETTINGS_FILE]: JSON.stringify({ beam_size: 6, token: 't' }) }));
  assert.deepEqual(partial, { ...DEFAULT_SETTINGS, beam_size: 6, token: 't' });   // the rest from the defaults
});

test('loadSettings clamps what it loads to the bounds', () => {
  const loaded = loadSettings(memoryStorage({ [SETTINGS_FILE]: JSON.stringify({ beam_size: 99, max_new_tokens: 1, k_images: -4, k_reports: 11.6 }) }));
  assert.deepEqual([loaded.beam_size, loaded.max_new_tokens, loaded.k_images, loaded.k_reports], [8, 16, 0, 10]);
  assert.deepEqual(Object.keys(BOUNDS).sort(), ['beam_size', 'k_images', 'k_reports', 'max_new_tokens']);
});

test('saveSettings says whether it stored, and never throws', () => {
  const storage = memoryStorage();
  assert.equal(saveSettings(storage, { ...DEFAULT_SETTINGS, beam_size: 4, evil: 'x' }), true);
  assert.deepEqual(JSON.parse(storage.data.get(SETTINGS_FILE)), { ...DEFAULT_SETTINGS, beam_size: 4 });   // sanitised on the way in too
  assert.equal(saveSettings(brokenStorage(), DEFAULT_SETTINGS), false);   // a full or blocked store
  assert.equal(saveSettings(undefined, DEFAULT_SETTINGS), false);
  assert.equal(saveSettings(null, DEFAULT_SETTINGS), false);
  assert.equal(saveSettings({}, DEFAULT_SETTINGS), false);
  assert.equal(saveSettings(storage, null), true);   // nothing to save is the defaults
});

test('browserStorage returns the page storage, or null where reading the property throws', () => {
  const store = memoryStorage();
  assert.equal(browserStorage({ localStorage: store }), store);
  assert.equal(browserStorage({}), null);
  assert.equal(browserStorage(undefined), null);
  assert.equal(browserStorage({ get localStorage() { throw new Error('SecurityError'); } }), null);   // a browser that blocks site data
});

// ---- options for a turn -----------------------------------------------------------------------------------------------------

const CARD_CACHED = { name: 'hybrid_150m_m3_rrg', cached_decode_available: true };
const CARD_PLAIN = { name: 'hybrid_150m_v2_rrg', cached_decode_available: false };
const MODELS = { default_model: 'hybrid_150m_m3_rrg', mode: 'private', allow_compile: false, models: [CARD_CACHED, CARD_PLAIN] };
const OPTION_KEYS = ['beam_size', 'cached_decode', 'compile', 'decode', 'display_repair', 'k_images', 'k_reports', 'label', 'max_new_tokens', 'stop_on_repeat'];

test('optionsFromSettings sends exactly the keys of the server Options that the drawer sets, at the published defaults (Display repair on)', () => {
  const options = optionsFromSettings(DEFAULT_SETTINGS, MODELS);
  assert.deepEqual(options, { decode: 'beam', beam_size: 3, max_new_tokens: 100, cached_decode: true, compile: false, k_images: 4, k_reports: 3,
                              label: true, display_repair: true, stop_on_repeat: true });   // no model: the server's default runs
  assert.deepEqual(Object.keys(optionsFromSettings({ ...DEFAULT_SETTINGS, model: CARD_PLAIN.name }, MODELS)).sort(), [...OPTION_KEYS, 'model'].sort());
  for (const key of ['reference', 'test_row', 'retrieval_k', 'token']) assert.equal(key in options, false, key);   // extra="forbid" would refuse them
  assert.doesNotThrow(() => JSON.stringify(options));
});

test('optionsFromSettings clamps every number to the server bounds, rounds, and replaces what is not a number', () => {
  const at = (patch) => optionsFromSettings({ ...DEFAULT_SETTINGS, ...patch }, MODELS);
  assert.deepEqual([at({ beam_size: 0 }).beam_size, at({ beam_size: 1 }).beam_size, at({ beam_size: 8 }).beam_size, at({ beam_size: 9 }).beam_size], [1, 1, 8, 8]);
  assert.deepEqual([at({ max_new_tokens: 15 }).max_new_tokens, at({ max_new_tokens: 16 }).max_new_tokens, at({ max_new_tokens: 200 }).max_new_tokens,
                    at({ max_new_tokens: 201 }).max_new_tokens], [16, 16, 200, 200]);
  assert.deepEqual([at({ k_images: -1 }).k_images, at({ k_images: 0 }).k_images, at({ k_images: 12 }).k_images, at({ k_images: 13 }).k_images], [0, 0, 12, 12]);
  assert.deepEqual([at({ k_reports: -1 }).k_reports, at({ k_reports: 0 }).k_reports, at({ k_reports: 10 }).k_reports, at({ k_reports: 11 }).k_reports], [0, 0, 10, 10]);
  assert.equal(at({ beam_size: 3.4 }).beam_size, 3);
  assert.equal(at({ beam_size: 3.6 }).beam_size, 4);
  assert.equal(at({ beam_size: '5' }).beam_size, 5);   // what a number input hands over
  for (const bad of [NaN, Infinity, -Infinity, '', ' ', 'x', null, undefined, {}, [], true]) {
    assert.equal(at({ beam_size: bad }).beam_size, 3, String(bad));   // the default, never NaN and never a throw
  }
  assert.equal(at({ decode: 'sampling' }).decode, 'beam');
  assert.equal(at({ decode: 'greedy' }).decode, 'greedy');
});

test('optionsFromSettings holds for any input: every key present, every number inside its bounds, no key of its own', () => {
  const pool = [NaN, Infinity, -1e9, 1e9, -1, 0, 1, 7.5, 8, 9, 15, 16, 200, 201, '3', 'x', '', null, undefined, true, false, {}, [], 'greedy', 'beam', 'sampling'];
  let seed = 7;
  const pick = () => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return pool[seed % pool.length]; };
  for (let i = 0; i < 2000; i++) {
    const messy = { beam_size: pick(), max_new_tokens: pick(), k_images: pick(), k_reports: pick(), decode: pick(), cached_decode: pick(), compile: pick(),
                    label: pick(), display_repair: pick(), stop_on_repeat: pick(), model: pick(), token: pick(), extra: pick() };
    const o = optionsFromSettings(messy, [MODELS, null, undefined, {}, [], [CARD_PLAIN], { models: 'no' }][i % 7]);
    assert.deepEqual(Object.keys(o).filter((k) => k !== 'model').sort(), OPTION_KEYS);
    assert.ok(Number.isInteger(o.beam_size) && o.beam_size >= 1 && o.beam_size <= 8);
    assert.ok(Number.isInteger(o.max_new_tokens) && o.max_new_tokens >= 16 && o.max_new_tokens <= 200);
    assert.ok(Number.isInteger(o.k_images) && o.k_images >= 0 && o.k_images <= 12);
    assert.ok(Number.isInteger(o.k_reports) && o.k_reports >= 0 && o.k_reports <= 10);
    assert.ok(['beam', 'greedy'].includes(o.decode));
    for (const flag of ['cached_decode', 'compile', 'label', 'display_repair', 'stop_on_repeat']) assert.equal(typeof o[flag], 'boolean', flag);
    assert.ok(!('model' in o) || typeof o.model === 'string');
  }
});

test('optionsFromSettings forces cached_decode off for a model whose card says it has no cache, and leaves it for one that has', () => {
  const on = { ...DEFAULT_SETTINGS, cached_decode: true };
  assert.equal(optionsFromSettings({ ...on, model: CARD_PLAIN.name }, MODELS).cached_decode, false);   // the 13D model: uncached only
  assert.equal(optionsFromSettings({ ...on, model: CARD_CACHED.name }, MODELS).cached_decode, true);
  assert.equal(optionsFromSettings({ ...on, model: '' }, MODELS).cached_decode, true);                  // the default model has the cache
  assert.equal(optionsFromSettings({ ...on, model: '' }, { ...MODELS, default_model: CARD_PLAIN.name }).cached_decode, false);   // and when the default has none
  assert.equal(optionsFromSettings({ ...on, cached_decode: false, model: CARD_CACHED.name }, MODELS).cached_decode, false);   // the user's own off stays off
  assert.equal(optionsFromSettings(on, null).cached_decode, true);   // models not loaded yet: nothing says otherwise
  assert.equal(optionsFromSettings(on, { models: [{ name: 'x' }], default_model: 'x' }).cached_decode, true);   // a card that does not say
});

test('optionsFromSettings forces compile off unless the server allows it', () => {
  const asked = { ...DEFAULT_SETTINGS, compile: true };
  assert.equal(optionsFromSettings(asked, MODELS).compile, false);
  assert.equal(optionsFromSettings(asked, { ...MODELS, allow_compile: true }).compile, true);
  assert.equal(optionsFromSettings(asked, { ...MODELS, allow_compile: 'yes' }).compile, false);   // only a real true allows it
  assert.equal(optionsFromSettings(asked, null).compile, false);
  assert.equal(optionsFromSettings(asked, [CARD_CACHED]).compile, false);
  assert.equal(optionsFromSettings({ ...asked, compile: false }, { ...MODELS, allow_compile: true }).compile, false);
});

test('optionsFromSettings names a model only when the server lists it', () => {
  assert.equal(optionsFromSettings({ ...DEFAULT_SETTINGS, model: CARD_PLAIN.name }, MODELS).model, CARD_PLAIN.name);
  assert.equal('model' in optionsFromSettings({ ...DEFAULT_SETTINGS, model: 'gone' }, MODELS), false);   // a stale choice: the default runs
  assert.equal('model' in optionsFromSettings({ ...DEFAULT_SETTINGS, model: CARD_PLAIN.name }, null), false);
  assert.equal(chosenCard({ model: CARD_PLAIN.name }, MODELS), CARD_PLAIN);
  assert.equal(chosenCard({ model: '' }, MODELS), CARD_CACHED);   // the default is the chosen one
  assert.equal(chosenCard({ model: 'gone' }, MODELS), CARD_CACHED);
  assert.equal(chosenCard({}, null), null);
  assert.equal(chosenCard({ model: 'a' }, [{ name: 'a' }]).name, 'a');
});

// ---- the client id ----------------------------------------------------------------------------------------------------------

test('clientId is random, kept under cxrchat.client, and fresh when storage fails', () => {
  const visible = /^[\x21-\x7e]{1,128}$/;   // the server accepts visible ASCII, 1 to 128 characters
  const storage = memoryStorage();
  const first = clientId(storage);
  assert.match(first, visible);
  assert.equal(CLIENT_KEY, 'cxrchat.client');
  assert.equal(storage.data.get(CLIENT_KEY), first);   // kept
  assert.equal(clientId(storage), first);              // and read back
  assert.notEqual(clientId(memoryStorage()), first);   // random

  const ids = new Set();
  for (let i = 0; i < 50; i++) ids.add(clientId(brokenStorage()));
  assert.equal(ids.size, 50);   // a storage that refuses: a fresh one every time
  for (const id of ids) assert.match(id, visible);
  assert.match(clientId(undefined), visible);
  assert.match(clientId(null), visible);
  assert.match(clientId({}), visible);
  const readOnly = { getItem: () => null, setItem() { throw new Error('quota'); } };
  assert.match(clientId(readOnly), visible);   // reads fine, cannot write: still an id

  for (const bad of ['', ' ', 'a b', 'x'.repeat(129), 'é', 'a\nb', 7]) {
    const held = memoryStorage({ [CLIENT_KEY]: bad });
    const got = clientId(held);
    assert.notEqual(got, bad);
    assert.match(got, visible);   // an id the server would refuse is replaced, and the replacement is kept
    assert.equal(held.data.get(CLIENT_KEY), got);
  }
});

// ---- error messages ---------------------------------------------------------------------------------------------------------

const refusal = (status, message, kind = 'validation_error') => Object.assign(new Error('refused'), {
  status, body: message === undefined ? null : { type: 'error', error: { type: kind, message } },
});

test('errorMessage maps the status and the server envelope to what the user reads', () => {
  assert.equal(errorMessage(refusal(401, 'Missing or wrong token: send Authorization: Bearer <token>.')), 'Enter the access token in Settings');
  assert.equal(errorMessage(refusal(429, 'The server is busy with 4 turns; try again shortly.')), 'The server is busy; try again shortly');
  assert.equal(errorMessage(refusal(422, 'Invalid options: beam_size: Input should be less than or equal to 8')), 'Invalid options: beam_size: Input should be less than or equal to 8');
  assert.equal(errorMessage(refusal(413, 'The image is over the 20 MB upload limit.')), 'The image is over the 20 MB upload limit.');
  assert.equal(errorMessage(refusal(422)), 'The server could not use that request.');   // no envelope: a sentence of our own
  assert.equal(errorMessage(refusal(413)), 'The image is too large.');
  assert.equal(errorMessage(refusal(404, 'Session not found.', 'not_found_error')), 'Session not found.');
  assert.equal(errorMessage(refusal(400, 'options must be a JSON object.')), 'options must be a JSON object.');
  assert.equal(errorMessage(refusal(403, 'Test-split studies are not available in public mode.')), 'Test-split studies are not available in public mode.');
  assert.equal(errorMessage(refusal(500, 'Could not store the image.')), 'Could not store the image.');
  assert.equal(errorMessage(refusal(404)), 'Not found. It may have been deleted.');
  assert.equal(errorMessage(refusal(500)), 'The server had a problem. Try again.');
  assert.equal(errorMessage(refusal(503)), 'The server had a problem. Try again.');
  assert.equal(errorMessage(refusal(400)), 'The server could not read the request.');
  assert.equal(errorMessage(refusal(403)), 'The server refused this request.');
  assert.equal(errorMessage(refusal(418)), 'The request failed.');
});

test('errorMessage copes with a network failure, an odd envelope and what is not an error at all', () => {
  assert.equal(errorMessage(new TypeError('Failed to fetch')), 'Cannot reach the server. Check the connection and try again.');
  assert.equal(errorMessage(new Error('boom')), 'boom');
  assert.equal(errorMessage(new Error('')), 'Something went wrong.');
  assert.equal(errorMessage('plain text'), 'plain text');
  assert.equal(errorMessage(''), 'Something went wrong.');
  for (const odd of [null, undefined, 42, {}, [], { status: 'x' }, { body: 7 }, { body: { error: 7 } }, { message: 7 }]) {
    assert.equal(typeof errorMessage(odd), 'string', String(odd));
    assert.ok(errorMessage(odd).length > 0);   // always something to show
  }
  assert.equal(errorMessage(refusal(422, { deep: 1 })), 'The server could not use that request.');   // a message that is no text is ignored
  assert.equal(errorMessage(refusal(422, '   ')), 'The server could not use that request.');
  assert.equal(errorMessage({ status: 422, body: { error: { message: 'x'.repeat(1000) } } }).length, 300);   // an unreasonable message is cut
  assert.equal(errorMessage({ status: 401, body: { error: { message: 'x' } } }), 'Enter the access token in Settings');
});

// ---- the watchdog -----------------------------------------------------------------------------------------------------------

test('the watchdog hands a running turn to polling after 3 s without bytes, once, and never touches a clock of its own', () => {
  assert.equal(STALL_MS, 3000);
  let t = 5000;
  const wd = createWatchdog({ now: () => t });
  assert.equal(wd.phase, 'idle');
  assert.equal(wd.tick('running'), 'none');   // nothing to watch before the turn is sent
  wd.arm();
  assert.equal(wd.phase, 'stream');
  t += 2999;
  assert.equal(wd.tick('running'), 'none');
  t += 1;   // exactly 3000 ms of silence
  assert.equal(wd.tick('running'), 'poll');
  assert.equal(wd.phase, 'poll');
  t += 10000;
  assert.equal(wd.tick('running'), 'none');   // asked once: polling is already the transport
});

test('bytes on the stream restart the silence, and a turn that is no longer running is never handed over', () => {
  let t = 0;
  const wd = createWatchdog({ now: () => t });
  wd.arm();
  t = 2900; wd.bytes();
  t = 5800;
  assert.equal(wd.tick('running'), 'none');   // 2900 ms since the last event
  t = 5900;
  assert.equal(wd.tick('running'), 'poll');

  const done = createWatchdog({ now: () => t });
  t = 0; done.arm();
  t = 60000;
  assert.equal(done.tick('done'), 'none');    // it ended: silence is what a finished stream looks like
  assert.equal(done.phase, 'idle');
  for (const status of ['aborted', 'error']) {
    const w = createWatchdog({ now: () => t });
    t = 0; w.arm(); t = 60000;
    assert.equal(w.tick(status), 'none', status);
  }
  const late = createWatchdog({ now: () => t });
  t = 0; late.arm(); late.settle(); late.bytes(); t = 60000;
  assert.equal(late.tick('running'), 'none');   // a settled turn is not revived by a stray byte or a tick
  assert.equal(createWatchdog({ now: () => t, stallMs: 100 }).tick('running'), 'none');
  const quick = createWatchdog({ now: () => t, stallMs: 100 });
  t = 0; quick.arm(); t = 100;
  assert.equal(quick.tick('running'), 'poll');   // the limit is a parameter
});

test('when the stream ends the watchdog says poll while the turn runs, and nothing once it has ended', () => {
  let t = 0;
  const wd = createWatchdog({ now: () => t });
  wd.arm();
  assert.equal(wd.ended('running'), 'poll');   // closed without a message_stop, or dropped by Stop or by the stall
  assert.equal(wd.phase, 'poll');
  wd.arm();
  assert.equal(wd.ended('done'), 'none');
  assert.equal(wd.phase, 'idle');
  assert.equal(wd.ended('running'), 'none');   // idle: nothing started this stream
  wd.arm(); t = 3000;
  assert.equal(wd.tick('running'), 'poll');
  assert.equal(wd.ended('running'), 'poll');   // the stall aborted the stream: its end asks for the same poll again, harmlessly
});

test('a poll that fails offers a retry that polls again, and a resumed turn starts in the poll phase', () => {
  const wd = createWatchdog({ now: () => 0 });
  assert.equal(wd.retry(), 'none');            // nothing failed
  wd.resume();
  assert.equal(wd.phase, 'poll');              // a turn found running on open has no stream
  wd.failed();
  assert.equal(wd.phase, 'failed');
  assert.equal(wd.ended('running'), 'none');   // a failed poll is not restarted by a stray stream end
  assert.equal(wd.tick('running'), 'none');
  assert.equal(wd.retry(), 'poll');
  assert.equal(wd.phase, 'poll');
  assert.equal(wd.retry(), 'none');            // one retry per failure
  wd.settle();
  assert.equal(wd.phase, 'idle');
  wd.failed();
  assert.equal(wd.phase, 'idle');              // failed() only follows a poll
});

// ---- small formatters ---------------------------------------------------------------------------------------------------------

test('session titles, dates and counts as the sidebar shows them', () => {
  assert.equal(sessionTitle({ title: 'chest.png' }), 'chest.png');
  assert.equal(sessionTitle({ title: '  ' }), 'New chat');
  assert.equal(sessionTitle({}), 'New chat');
  assert.equal(sessionTitle(null), 'New chat');
  const now = new Date('2026-10-03T12:00:00Z');
  assert.match(sessionDate('2026-10-01T09:00:00+00:00', now), /1/);
  assert.match(sessionDate('2025-10-01T09:00:00+00:00', now), /2025/);   // another year says so
  assert.doesNotMatch(sessionDate('2026-10-01T09:00:00+00:00', now), /2026/);
  for (const bad of ['', 'not a date', null, undefined, 5, {}]) assert.equal(sessionDate(bad, now), '', String(bad));
  assert.match(sessionMeta({ updated_at: '2026-10-01T09:00:00+00:00', turns: 2 }, now), /· 2 turns$/);
  assert.match(sessionMeta({ updated_at: '2026-10-01T09:00:00+00:00', turns: 1 }, now), /· 1 turn$/);
  assert.equal(sessionMeta({ updated_at: 'junk', turns: 0 }, now), '0 turns');
  assert.equal(sessionMeta({}, now), '0 turns');
  assert.match(sessionMeta({ created_at: '2026-10-01T09:00:00+00:00', turns: 3 }, now), /3 turns/);   // no updated_at: the creation date
});

test('the health strip text and its schedule: 10 s, 30 s after three failures, 10 s again at the first answer', () => {
  assert.equal(healthText({ status: 'ok', mode: 'private', turns_in_flight: 1, queue_cap: 4 }), 'private · 1 of 4 turns in flight');
  assert.equal(healthText({ mode: 'public', turns_in_flight: 0, queue_cap: 4 }), 'public · 0 of 4 turns in flight');
  assert.equal(healthText({ mode: 'private', turns_in_flight: 2 }), 'private · 2 turns in flight');
  assert.equal(healthText({ mode: 'private' }), 'private');
  assert.equal(healthText({}), 'server ok');
  assert.equal(healthText(null), 'server ok');
  assert.deepEqual([0, 1, 2, 3, 4, 50].map(nextHealthDelay), [10000, 10000, 10000, 30000, 30000, 30000]);
  assert.deepEqual([HEALTH_MS, HEALTH_SLOW_MS], [10000, 30000]);
});

test('an export is saved under the name the server gave it, or one made from the session id, never a path', () => {
  assert.equal(exportFilename('attachment; filename="session-s_1.json"', 's_1', 'json'), 'session-s_1.json');
  assert.equal(exportFilename('attachment; filename=session-s_1.md', 's_1', 'md'), 'session-s_1.md');
  assert.equal(exportFilename("attachment; filename*=UTF-8''session-s_1.md", 's_1', 'md'), 'session-s_1.md');
  for (const bad of [null, undefined, '', 'attachment', 'attachment; filename="../../etc/passwd"', 'attachment; filename="a/b.json"',
                     'attachment; filename="a\\b.json"', 'attachment; filename=""', 'attachment; filename=".hidden"', 42]) {
    assert.equal(exportFilename(bad, 's_9', 'json'), 'session-s_9.json', String(bad));
  }
});

test('a stored user message becomes the user turn: its text, its file name and the options the assistant row carries', () => {
  const options = { beam_size: 3 };
  assert.deepEqual(userTurnMessage({ text: 'beam 5', image_filename: 'chest.png' }, { options }),
                   { text: 'beam 5', image: { url: null, filename: 'chest.png' }, options });   // no image URL until P6-B
  assert.deepEqual(userTurnMessage({ text: '', image_filename: null }, { options: null }), { text: '', image: null, options: null });
  assert.deepEqual(userTurnMessage({ text: 'a' }, null), { text: 'a', image: null, options: null });
  assert.deepEqual(userTurnMessage(null, undefined), { text: '', image: null, options: null });
  assert.equal(userTurnMessage({ text: 5, image_filename: 7 }, { options: 'x' }).image, null);
});

test('nearBottom is true within 80 px of the end and when nothing has been measured', () => {
  assert.equal(nearBottom({ scrollTop: 920, clientHeight: 400, scrollHeight: 1400 }), true);    // 80 px left
  assert.equal(nearBottom({ scrollTop: 919, clientHeight: 400, scrollHeight: 1400 }), false);   // 81
  assert.equal(nearBottom({ scrollTop: 1000, clientHeight: 400, scrollHeight: 1400 }), true);   // at the end
  assert.equal(nearBottom({ scrollTop: 0, clientHeight: 400, scrollHeight: 400 }), true);       // nothing to scroll
  assert.equal(nearBottom({ scrollTop: 100, clientHeight: 400, scrollHeight: 1400 }, 2000), true);
  for (const unmeasured of [{}, null, undefined, { scrollTop: NaN, clientHeight: 1, scrollHeight: 9 }]) assert.equal(nearBottom(unmeasured), true);
});

test('checkImageFile lets through PNG, JPEG, WEBP and a file with no type, and says why it refuses the rest', () => {
  const file = (type, size = 1000) => ({ type, size, name: 'x' });
  for (const type of ['image/png', 'image/jpeg', 'image/webp', '']) assert.equal(checkImageFile(file(type)), null, type);
  for (const type of ['image/gif', 'image/svg+xml', 'application/pdf', 'text/html']) assert.equal(checkImageFile(file(type)), 'Choose a PNG, JPEG or WEBP image.', type);
  assert.equal(checkImageFile(file('image/png', 20 * 1024 * 1024)), null);                         // the limit itself is fine
  assert.equal(checkImageFile(file('image/png', 20 * 1024 * 1024 + 1)), 'The image is over the 20 MB limit.');
  assert.equal(checkImageFile(null), 'Choose an image.');
  assert.equal(checkImageFile({ type: 'image/png' }), null);   // no size known: the server decides
});

test('isCommand tells a command from a note as the server does: the notes of fixtures/commands.json, which the server is held to too (P4-H)', () => {
  const cases = JSON.parse(readFileSync(new URL('./fixtures/commands.json', import.meta.url)));
  assert.ok(cases.length >= 20 && cases.some((c) => c.command) && cases.some((c) => !c.command));
  for (const { note, command } of cases) assert.equal(isCommand(note), command, JSON.stringify(note));
  for (const odd of [null, undefined, 5, {}, ['beam 5']]) assert.equal(isCommand(odd), false);
});

test('every command rule of the page has an example in fixtures/commands.json, so a rule added without one fails (P4-H fix 1)', () => {
  const cases = JSON.parse(readFileSync(new URL('./fixtures/commands.json', import.meta.url)));
  const examples = cases.filter((c) => c.command).map((c) => c.note.split(/\s+/).filter(Boolean).join(' '));
  const uncovered = (rules) => rules.filter((rule) => !examples.some((note) => rule.test(note)));
  assert.ok(COMMANDS.length >= 6 && Object.isFrozen(COMMANDS));
  assert.deepEqual(uncovered(COMMANDS), []);
  assert.deepEqual(uncovered([...COMMANDS, /^stop (?:on|off)$/i]).map(String), ['/^stop (?:on|off)$/i']);   // the guard bites
});

test('a stored user message with its image URLs becomes a user turn that shows the server\'s thumbnail; a re-run keeps its file name (P6-B)', () => {
  const options = { beam_size: 3 };
  const urls = { original: '/v1/messages/u_1/image?variant=original', thumb: '/v1/messages/u_1/image?variant=thumb',
                 model_input: '/v1/messages/u_1/image?variant=model_input' };
  const user = { text: '', image_filename: 'chest.png', image_urls: urls };
  assert.deepEqual(userTurnMessage(user, { options }, { image: { source: 'upload' } }), { text: '', image: { url: urls.thumb, filename: 'chest.png' }, options });
  assert.deepEqual(userTurnMessage(user, { options }).image, { url: urls.thumb, filename: 'chest.png' });   // a log that could not be read: shown all the same
  assert.deepEqual(userTurnMessage({ ...user, text: 'greedy' }, { options }, { image: { source: 'previous' } }).image,
                   { url: null, filename: 'chest.png' });   // a re-run ran an earlier upload: its file name, as the live turn showed no picture
  assert.deepEqual(userTurnMessage({ text: '', image_filename: null, image_urls: urls, test_row: 3 }, { options }, { image: { source: 'test_split' } }),
                   { text: '', image: { url: urls.thumb, filename: null, source: 'test_split' }, options });   // a test study: its picture
  for (const odd of [null, 'x', { thumb: 7 }, { thumb: null }, []]) {
    assert.deepEqual(userTurnMessage({ ...user, image_urls: odd }, { options }).image, { url: null, filename: 'chest.png' }, JSON.stringify(odd));
  }
  assert.deepEqual(userTurnMessage({ text: 'a', image_filename: null, image_urls: null }, { options }).image, null);   // a question has none
});

test('a stored question (a user message with no image) becomes a user turn with no chips: it ran no model, so it used no settings (P4-H)', () => {
  const options = { beam_size: 3, max_new_tokens: 100 };
  assert.deepEqual(userTurnMessage({ text: 'is it pneumonia?', image_filename: null }, { options }), { text: 'is it pneumonia?', image: null, options: null });
  assert.deepEqual(userTurnMessage({ text: 'tokens 30', image_filename: 'chest.png' }, { options }).options, options);   // a re-run names its image
});

// ---- the page: a shim DOM shaped like index.html, a fake window, timers, fetch and api module ------------------------------

const $ = (id) => document.getElementById(id);
const q = (node, selector) => node.querySelector(selector);
const qa = (node, selector) => node.querySelectorAll(selector);
const texts = (nodes) => nodes.map((n) => n.textContent);

const frames = [];
globalThis.requestAnimationFrame = (callback) => { frames.push(callback); return frames.length; };   // a frame is run by hand
const nextFrame = () => { for (const run of frames.splice(0)) run(); };

// The shell of app/static/index.html: every id the script looks up (a test below checks the two agree).
function buildPage() {
  document.body.replaceChildren(
    el('header', { class: 'banner' },
      el('button', { id: 'sidebar-toggle', type: 'button', 'aria-controls': 'sidebar', 'aria-expanded': 'false', 'aria-label': 'Sessions' }, '☰'),
      el('p', { class: 'disclaimer' }, 'Research prototype — not for clinical use.'),
      el('span', { id: 'mode-badge' }), el('span', { id: 'health', 'aria-live': 'polite' })),
    el('div', { class: 'layout' },
      el('nav', { id: 'sidebar', 'aria-label': 'Sessions' }, el('button', { id: 'new-session', type: 'button' }, 'New chat'), el('ol', { id: 'session-list' })),
      el('main', { id: 'conversation', 'aria-live': 'polite' }),
      el('aside', { id: 'drawer', hidden: true, 'aria-label': 'Settings' })),
    el('form', { id: 'composer', 'aria-label': 'New turn' },
      el('div', { id: 'image-well', tabindex: 0, role: 'button' }, 'Drop, paste or click to attach an X-ray (PNG, JPEG, WEBP)'),
      el('div', { id: 'preview', hidden: true }),
      el('textarea', { id: 'prompt', rows: 2, 'aria-label': 'Note or command' }),
      el('div', { id: 'chips', role: 'group', 'aria-label': 'Current settings' }),
      el('button', { id: 'settings', type: 'button', 'aria-controls': 'drawer', 'aria-expanded': 'false' }, 'Settings'),
      el('button', { id: 'send', type: 'submit' }, 'Send'),
      el('button', { id: 'stop', type: 'button', hidden: true }, 'Stop'),
      el('input', { id: 'file', type: 'file', hidden: true })),
    el('div', { id: 'viewer', hidden: true }));
}

function fakeWindow(hash = '') {
  const listeners = new Map();
  const win = {
    short: false, replaced: [], copied: [], scrollY: 0, innerHeight: 800,
    location: {
      current: hash,
      get hash() { return this.current; },
      set hash(value) { if (String(value) === this.current) return; this.current = String(value); win.fire('hashchange'); },
    },
    history: { replaceState(_state, _title, url) { win.replaced.push(url); win.location.current = url; } },
    addEventListener(type, fn) { if (!listeners.has(type)) listeners.set(type, []); listeners.get(type).push(fn); },
    fire(type) { for (const fn of listeners.get(type) ?? []) fn({ type }); },
    matchMedia: (query) => ({ matches: win.short && /max-height/.test(query) }),
    navigator: { clipboard: { writeText: async (value) => { win.copied.push(value); } } },
  };
  return win;
}

function fakeTimers() {
  let now = 0;
  let next = 1;
  const timeouts = new Map();
  const intervals = new Map();
  return {
    now: () => now,
    setTimeout: (fn, ms) => { const id = next++; timeouts.set(id, { fn, at: now + ms, ms }); return id; },
    clearTimeout: (id) => { timeouts.delete(id); },
    setInterval: (fn, ms) => { const id = next++; intervals.set(id, { fn, ms, at: now + ms }); return id; },
    clearInterval: (id) => { intervals.delete(id); },
    get timeouts() { return [...timeouts.values()]; },
    get intervals() { return intervals.size; },
    advance(ms) {   // runs what is due in time order, the clock at each one's moment
      const end = now + ms;
      for (;;) {
        const due = [...timeouts.entries(), ...intervals.entries()].filter(([, v]) => v.at <= end).sort((a, b) => a[1].at - b[1].at)[0];
        if (!due) break;
        const [id, item] = due;
        now = Math.max(now, item.at);
        if (timeouts.has(id)) timeouts.delete(id); else item.at += item.ms;
        item.fn();
      }
      now = end;
    },
  };
}

// A route answers with a body (200), a refused() or a Response; a function gets (url, init).
const refused = (status, message, kind = 'validation_error') => ({ __status: status, body: { type: 'error', error: { type: kind, message } } });
function fakeFetch(routes) {
  const calls = [];
  const fn = async (url, init = {}) => {
    const method = init.method ?? 'GET';
    calls.push({ method, url: String(url), headers: init.headers ?? {}, body: init.body });
    const route = routes[`${method} ${String(url).split('?')[0]}`];
    if (route === undefined) return new Response(JSON.stringify({ type: 'error', error: { type: 'not_found_error', message: 'no route' } }), { status: 404 });
    const out = typeof route === 'function' ? await route(String(url), init) : route;
    if (out instanceof Response) return out;
    if (out && out.__status) {
      return new Response(out.body === undefined ? null : JSON.stringify(out.body), { status: out.__status, headers: { 'Content-Type': 'application/json' } });
    }
    return new Response(JSON.stringify(out), { status: 200, headers: { 'Content-Type': 'application/json' } });
  };
  fn.calls = calls;
  fn.to = (method, prefix) => calls.filter((c) => c.method === method && c.url.startsWith(prefix));
  return fn;
}

const abortError = () => Object.assign(new Error('The operation was aborted.'), { name: 'AbortError' });

// Items in, one at a time, and an end; iterate() throws AbortError as soon as its signal is aborted.
function channel() {
  const items = [];
  let wake = null;
  let ended = false;
  let failure = null;
  return {
    push(...more) { items.push(...more); wake?.(); },
    end() { ended = true; wake?.(); },
    fail(err) { failure = err; wake?.(); },
    async *iterate(signal, lenient = false) {   // lenient: events already queued still come out after an abort, as api.js yields a parsed batch
      for (;;) {
        if (signal?.aborted && !(lenient && items.length)) throw abortError();
        if (failure) throw failure;
        if (items.length) { yield items.shift(); continue; }
        if (ended) return;
        await new Promise((resolve) => { wake = resolve; signal?.addEventListener('abort', resolve, { once: true }); });
        wake = null;
      }
    },
  };
}

// api.js, faked: each streamTurn / pollMessage call is recorded with a channel the test pushes events into.
function fakeApi() {
  const api = {
    streams: [], polls: [], cancels: [], order: [], cleared: 0,
    refuse: null,        // an error streamTurn throws before it has an id
    silent: false,       // a server that sends no X-Message-Id: onMessageId is never called
    holdAbort: false,    // an abort does not end the stream at once: run.fail(err) does, whenever the test says
    lenient: false,      // events already queued still come out of a stream or a poll that was aborted
    cancelError: null,   // what cancelMessage rejects with
    cancelGate: null,    // a promise cancelMessage waits for before it answers
    streamTurn(opts) {
      const run = { opts, channel: channel(), id: null };
      run.accepted = new Promise((resolve) => { run.accept = (id) => { run.id = id; resolve(); }; });
      run.failed = new Promise((_, reject) => { run.fail = reject; });
      run.failed.catch(() => {});
      const held = api.holdAbort;
      const lenient = api.lenient;
      api.streams.push(run);
      opts.signal?.addEventListener('abort', () => api.order.push('abort'), { once: true });
      return (async function* stream() {
        if (api.refuse) throw api.refuse;
        if (!api.silent) {
          await Promise.race([run.accepted, run.failed, held ? new Promise(() => {})
            : new Promise((_, reject) => opts.signal?.addEventListener('abort', () => reject(abortError()), { once: true }))]);
          opts.onMessageId?.(run.id);
        }
        yield* run.channel.iterate(opts.signal, lenient);
      }());
    },
    pollMessage(opts) {
      const run = { opts, channel: channel() };
      api.polls.push(run);
      return run.channel.iterate(opts.signal, api.lenient);
    },
    async cancelMessage(opts) {
      api.order.push('cancel');
      api.cancels.push(opts);
      if (api.cancelGate) await api.cancelGate;
      if (api.cancelError) throw api.cancelError;
      return { id: opts.messageId, status: 'running', cancel_requested: true };
    },
    loadImage: async (url) => `blob:fake/${url}`,
    clearImageCache() { api.cleared += 1; },
  };
  return api;
}

const iso = (day) => `2026-10-${String(day).padStart(2, '0')}T10:00:00+00:00`;
const sess = (id, title, turns = 1, day = 3) => ({ id, title, turns, created_at: iso(day), updated_at: iso(day), mode: 'private' });
const userMsg = (id, text, filename = null) => ({ id, role: 'user', text, image_filename: filename, options: null, status: 'done' });
const botMsg = (id, status = 'done', options = null) => ({ id, role: 'assistant', text: '', options, status });
const TINY_CARD = { name: 'tiny', cached_decode_available: true, device: 'cpu', prefix_k: 4 };

let eventCount = 0;
const SHA = 'a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90';
const START_DATA = {
  message_id: 'm_a', user_message_id: 'u_a', session_id: 's_a', mode: 'private', model: TINY_CARD,
  options: { model: 'tiny', decode: 'beam', beam_size: 5, max_new_tokens: 100, cached_decode: true, compile: false, k_images: 4, k_reports: 3,
             label: true, reference: null, display_repair: true, stop_on_repeat: true, test_row: null },
  image: { sha256: SHA, filename: 'chest.png', source: 'upload', urls: {} },
};
const ev = (event, data = {}) => ({ event, data: { ...data, seq: ++eventCount } });
const startEv = (over = {}) => ev('message_start', { ...START_DATA, ...over });
const stageStartEv = (stage, index) => ev('stage_start', { stage, index });
const stageEndEv = (stage, ms = 5, detail = { x: 1 }) => ev('stage_end', { stage, ms, detail });
const skipEv = (stage, reason) => ev('stage_end', { stage, skipped: reason });
const snapEv = (text, step = 0) => ev('content_block_delta', { index: 0, delta: { type: 'beam_snapshot', step, text } });
const stopEv = (status = 'done', over = {}) => ev('message_stop', {
  message_id: 'm_a', status, total_ms: 12, report: null, display_report: null, truncated_mid_sentence: false, disclaimer: 'Research prototype; not for clinical use.', ...over,
});
const resetEvents = () => { eventCount = 0; };
// The events of a whole turn, 1..n, ending in message_stop done.
const fullTurn = (id = 'm_a') => {
  resetEvents();
  return [startEv({ message_id: id }), stageStartEv('preprocess', 0), stageEndEv('preprocess'), stageStartEv('encode', 1), stageEndEv('encode'),
          skipEv('retrieve', 'gallery_unavailable'), stageStartEv('generate', 3), ev('content_block_start', { index: 0, content_block: { type: 'report', text: '' } }),
          snapEv('Findings: clear.'), ev('content_block_stop', { index: 0 }), stageEndEv('generate', 9), skipEv('label', 'labeler_unavailable'),
          skipEv('score', 'no_reference'), stopEv('done', { message_id: id, report: 'Findings: clear.', display_report: 'Findings: clear.' })];
};
const rows = (events) => events.map((e) => ({ seq: e.data.seq, event: e.event, data: e.data }));

function harness({ hash = '', sessions = [], models = MODELS, storage = memoryStorage(), routes = {}, confirmAnswer = true, urls, noConfirm = false } = {}) {
  installDom();   // a new document each time: the body of the last test still holds that app's listeners
  buildPage();
  frames.length = 0;
  const timers = fakeTimers();
  const win = fakeWindow(hash);
  const api = fakeApi();
  const confirms = [];
  const h = { timers, win, api, storage, confirms, answer: confirmAnswer, sessions };
  h.fetch = fakeFetch({
    'GET /healthz': { status: 'ok', mode: 'private', default_model: 'tiny', turns_in_flight: 0, queue_cap: 4 },
    'GET /v1/models': models,
    'GET /v1/sessions': () => ({ sessions: h.sessions, next_cursor: null }),
    ...routes,
  });
  h.app = createApp({
    document, window: win, storage, fetch: h.fetch, api, now: timers.now, setTimeout: timers.setTimeout, clearTimeout: timers.clearTimeout,
    setInterval: timers.setInterval, clearInterval: timers.clearInterval,
    ...(noConfirm ? {} : { confirm: (message) => { confirms.push(message); return h.answer; } }),
    ...(urls ? { URL: urls } : {}),
  });
  return h;
}

const imageFile = (name = 'chest.png', type = 'image/png', size = 2048) => new File([new Uint8Array(size)], name, { type });
const press = (target, key, init = {}) => {
  const event = new ShimEvent('keydown', { key, bubbles: true, cancelable: true, ...init });
  target.dispatchEvent(event);
  return event;
};
const attach = (file) => { $('file').files = [file]; $('file').dispatchEvent(new ShimEvent('change', { bubbles: true })); };

// ---- start and routes --------------------------------------------------------------------------------------------------------

test('every id the script looks up is in the page, and the page names none the shell test does not', () => {
  const html = readFileSync(new URL('../../app/static/index.html', import.meta.url), 'utf8');
  const source = readFileSync(new URL('../../app/static/app.js', import.meta.url), 'utf8');
  const asked = new Set([...source.matchAll(/\$\('([\w-]+)'\)/g)].map((m) => m[1]));
  assert.ok(asked.size >= 17, `the script looks up ${asked.size} ids`);
  for (const id of asked) assert.match(html, new RegExp(`id="${id}"`), `index.html has #${id}`);
  buildPage();
  for (const id of asked) assert.ok(document.getElementById(id), `the test page has #${id}`);
});

test('the ids the script adds to the page are unique, and an existing status region is used instead of a second one', async () => {
  const h = harness({ sessions: [sess('s_a', 'A', 0)], routes: EXISTING });
  document.body.append(el('div', { id: 'status', class: 'existing' }));
  await h.app.start();
  await flush();
  const ids = qa(document.body, '[id]').map((n) => n.getAttribute('id'));
  assert.equal(new Set(ids).size, ids.length, 'no id twice');
  for (const added of ['drawer-close', 'models-section', 'exports', 'session-more', 'notice', 'cached-note', 'status', 'apply-note', 'running-note',
                       'retrieval-note', 'labels-note', 'rerun-hint', 'settings-form', 'drawer-save', 'stop-hint', 'saved',
                       'beam_size-error', 'max_new_tokens-error', 'k_images-error', 'k_reports-error']) assert.ok(ids.includes(added), added);
  assert.equal($('status').getAttribute('class'), 'existing');   // the one that was there
  assert.equal($('status').parentNode, document.body);
  const fresh = harness();   // none there: the page makes a visually hidden, polite one
  await fresh.app.start();
  assert.deepEqual(['class', 'role', 'aria-live'].map((a) => $('status').getAttribute(a)), ['visually-hidden', 'status', 'polite']);
  assert.equal($('status').parentNode, document.body);
});

test('start with no hash opens the newest session and replays it: its user turn, and a card identical to the live one', async () => {
  const fixture = JSON.parse(readFileSync(new URL('./fixtures/turn_tiny.json', import.meta.url)));
  const botId = fixture[0].data.message_id;
  const options = fixture[0].data.options;
  const h = harness({
    sessions: [sess('s_b', 'chest.png', 1, 3), sess('s_a', 'an older chat', 2, 2)],
    routes: {
      'GET /v1/sessions/s_b': { ...sess('s_b', 'chest.png', 1), messages: [userMsg('u_b', 'beam 5\nplease', 'chest.png'), botMsg(botId, 'done', options)] },
      [`GET /v1/messages/${botId}`]: { ...botMsg(botId, 'done', options), events: rows(fixture) },
    },
  });
  await h.app.start();
  await flush();
  assert.deepEqual(h.win.replaced, ['#/s/s_b']);   // home became the newest session, without a history entry
  assert.equal($('mode-badge').textContent, 'private');
  assert.equal($('health').textContent, 'private · 0 of 4 turns in flight');

  const items = qa($('session-list'), 'li');
  assert.deepEqual(items.map((li) => li.getAttribute('data-session')), ['s_b', 's_a']);   // newest first, as the server sent them
  assert.deepEqual(items.map((li) => q(li, '.session-title').textContent), ['chest.png', 'an older chat']);
  assert.match(q(items[0], '.session-meta').textContent, /· 1 turn$/);
  assert.match(q(items[1], '.session-meta').textContent, /· 2 turns$/);
  assert.equal(q(items[0], 'a').getAttribute('href'), '#/s/s_b');
  assert.equal(q(items[0], 'a').getAttribute('aria-current'), 'page');
  assert.equal(q(items[1], 'a').hasAttribute('aria-current'), false);
  assert.equal($('session-list').getAttribute('role'), 'list');
  assert.equal($('session-more').hidden, true);

  const [user, card] = $('conversation').children;
  assert.deepEqual([user.localName, user.getAttribute('class'), card.localName, card.getAttribute('data-message-id')], ['article', 'turn user', 'article', botId]);
  assert.equal(q(user, '.user-text').textContent, 'beam 5\nplease');
  assert.equal(q(user, '.chip').textContent, 'chest.png');                                // a replayed upload: its file name, until P6-B
  assert.equal(qa(user, 'img').length, 0);
  assert.deepEqual(texts(qa(user, '.options .chip')), ['beam 3', '16 tok', 'cached', 'k 4/3', 'raw text', 'full budget']);   // the options the assistant row carries (this turn ran with Display repair and the stop switch off)
  assert.equal(user.getAttribute('aria-label'), 'Your message, turn 1');
  const replayed = fixture.reduce(applyEvent, initialView(botId));
  const fresh = renderAssistantCard(replayed, { copy() {}, showModels() {}, loadImage: h.api.loadImage, turn: 1, ui: new Map() });   // P6-B: its images too
  await flush();
  assert.equal(serialize(card), serialize(fresh));   // the same builders over the same log: the same card
  assert.equal(h.fetch.calls.every((c) => c.headers['X-Client-Id'] && !c.headers.Authorization || c.url === '/healthz'), true);
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
});

test('a reloaded chat shows each upload\'s thumbnail from the server, and its card the images row (P6-B)', async () => {
  const fixture = JSON.parse(readFileSync(new URL('./fixtures/turn_tiny.json', import.meta.url)));
  const { message_id: botId, user_message_id: userId, options, image } = fixture[0].data;
  const h = harness({
    sessions: [sess('s_b', 'chest.png', 1, 3)],
    routes: {
      'GET /v1/sessions/s_b': { ...sess('s_b', 'chest.png', 1),
                                messages: [{ ...userMsg(userId, '', 'chest.png'), image_urls: image.urls }, botMsg(botId, 'done', options)] },
      [`GET /v1/messages/${botId}`]: { ...botMsg(botId, 'done', options), events: rows(fixture) },
    },
  });
  await h.app.start();
  await flush();
  const [user, card] = $('conversation').children;
  const thumb = q(user, 'img');
  assert.equal(thumb.getAttribute('data-src'), image.urls.thumb);
  assert.equal(thumb.getAttribute('src'), `blob:fake/${image.urls.thumb}`);   // fetched with the page's auth (D23): never an <img src> to the server
  assert.equal(thumb.getAttribute('alt'), 'Uploaded X-ray: chest.png');
  assert.equal(qa(user, '.chip').some((c) => c.textContent === 'chest.png'), false);   // the picture now, not its name
  assert.deepEqual(qa(card, 'section.images img').map((i) => i.getAttribute('data-src')), [image.urls.thumb, image.urls.model_input]);
  assert.deepEqual(texts(qa(card, 'section.images figcaption')), ['Your X-ray · 320×320 px', 'What the model saw (224×224)']);
});

test('a deep link opens that session and not the newest, and a malformed one is home', async () => {
  const routes = { 'GET /v1/sessions/s_a': { ...sess('s_a', 'older', 1), messages: [] } };
  const deep = harness({ hash: '#/s/s_a', sessions: [sess('s_b', 'newer'), sess('s_a', 'older')], routes });
  await deep.app.start();
  await flush();
  assert.deepEqual(deep.win.replaced, []);   // the address already named it
  assert.deepEqual(deep.fetch.to('GET', '/v1/sessions/').map((c) => c.url), ['/v1/sessions/s_a']);
  assert.equal(q($('session-list'), '[aria-current]').parentNode.getAttribute('data-session'), 's_a');

  const bad = harness({ hash: '#/s/../etc', sessions: [sess('s_b', 'newer')], routes: { 'GET /v1/sessions/s_b': { ...sess('s_b', 'newer', 0), messages: [] } } });
  await bad.app.start();
  await flush();
  assert.deepEqual(bad.win.replaced, ['#/s/s_b']);
});

test('with no session at all the page shows an empty New chat, and Send without an image says so and asks the server nothing', async () => {
  const h = harness({ sessions: [] });
  await h.app.start();
  await flush();
  assert.deepEqual(h.win.replaced, ['#/new']);
  assert.equal($('conversation').children.length, 0);
  assert.equal(h.fetch.to('POST', '/v1/sessions').length, 0);
  await h.app.send();
  assert.equal($('notice').hidden, false);
  assert.equal(q($('notice'), 'p').textContent, 'Attach an X-ray first.');
  assert.equal($('notice').getAttribute('role'), 'alert');
  assert.equal(h.fetch.to('POST', '/v1/sessions').length, 0);   // no empty session is made for nothing
  assert.equal(h.api.streams.length, 0);
  q($('notice'), 'button[aria-label="Dismiss"]').click();
  assert.equal($('notice').hidden, true);
});

test('a session the server does not have falls back to the newest one that it does, and never loops on the missing one', async () => {
  const h = harness({
    hash: '#/s/s_gone', sessions: [sess('s_gone', 'stale entry'), sess('s_ok', 'fine')],
    routes: { 'GET /v1/sessions/s_gone': refused(404, 'Session not found.'), 'GET /v1/sessions/s_ok': { ...sess('s_ok', 'fine', 0), messages: [] } },
  });
  await h.app.start();
  await flush(12);
  assert.equal(h.fetch.to('GET', '/v1/sessions/s_gone').length, 1);   // asked once
  assert.equal(h.fetch.to('GET', '/v1/sessions/s_ok').length, 1);
  assert.deepEqual(h.win.replaced.at(-1), '#/s/s_ok');
  assert.equal(q($('notice'), 'p').textContent, 'Session not found.');
});

test('a server that needs a token: the first 401 says to enter it in Settings, and the page stays usable', async () => {
  const h = harness({ routes: { 'GET /v1/models': refused(401, 'Missing or wrong token'), 'GET /v1/sessions': refused(401, 'Missing or wrong token') } });
  await h.app.start();
  await flush();
  assert.equal(q($('notice'), 'p').textContent, 'Enter the access token in Settings');
  assert.deepEqual(h.win.replaced, ['#/new']);   // an empty chat: nothing to list
  assert.equal($('send').disabled, false);
});

// ---- a turn: send, refusal, keys, Stop, the watchdog, resume ------------------------------------------------------------------

const EXISTING = { 'GET /v1/sessions/s_a': { ...sess('s_a', 'earlier', 0), messages: [] } };
// An open, empty session s_a with an image attached: what a turn needs.
async function ready(extra = {}) {
  const h = harness({ sessions: [sess('s_a', 'earlier', 0)], routes: { ...EXISTING, ...extra.routes }, ...extra.options });
  await h.app.start();
  await flush();
  attach(imageFile('chest.png'));
  return h;
}
const cardOf = () => q($('conversation'), 'article.card');
const buttonOf = (node, label) => qa(node, 'button').find((b) => b.textContent === label);

test('Send streams a turn: Send is off and Stop is on while it runs, the card fills in, and the turn settles', async () => {
  const h = harness({ routes: { 'POST /v1/sessions': { id: 's_new', title: '', mode: 'private', turns: 0, created_at: iso(3), updated_at: iso(3) } } });
  await h.app.start();
  await flush();
  assert.equal($('exports').hidden, true);   // nothing to export before the chat has a session
  attach(imageFile('chest.png', 'image/png', 2048));
  assert.equal($('preview').hidden, false);
  assert.match(q($('preview'), 'span').textContent, /^chest\.png · 2 KB$/);
  assert.equal(q($('preview'), 'img').getAttribute('alt'), '');
  $('prompt').value = 'beam 5';
  const events = fullTurn();
  const turn = h.app.send();
  await flush();

  const created = h.fetch.to('POST', '/v1/sessions')[0];   // the empty chat became a session, with a JSON body
  assert.equal(created.url, '/v1/sessions');
  assert.equal(created.headers['Content-Type'], 'application/json');
  assert.deepEqual(h.win.replaced, ['#/new', '#/s/s_new']);   // the address follows without a navigation: a hashchange would tear the turn down
  assert.equal($('exports').hidden, false);

  const run = h.api.streams[0];
  assert.equal(run.opts.sessionId, 's_new');
  assert.equal(run.opts.form.get('text'), 'beam 5');
  assert.equal(run.opts.form.get('image').name, 'chest.png');
  assert.deepEqual(JSON.parse(run.opts.form.get('options')), { decode: 'beam', beam_size: 3, max_new_tokens: 100, cached_decode: true, compile: false,
                                                                 k_images: 4, k_reports: 3, label: true, display_repair: true, stop_on_repeat: true });
  assert.equal(typeof run.opts.clientId, 'string');
  assert.equal(run.opts.token, '');
  assert.equal($('send').disabled, true);
  assert.equal($('stop').hidden, false);
  assert.equal($('stop').disabled, true);   // the server has not said the turn's id yet: nothing to cancel
  assert.equal($('conversation').getAttribute('aria-busy'), 'true');
  const [user] = $('conversation').children;
  assert.equal(user.getAttribute('class'), 'turn user');   // drawn at once, from the local file
  assert.equal(q(user, '.user-text').textContent, 'beam 5');
  assert.match(q(user, 'img').getAttribute('src'), /^blob:/);
  assert.equal(cardOf(), null);   // and no card until the turn is accepted
  assert.equal($('prompt').value, 'beam 5');   // the composer keeps its content until then

  run.accept('m_a');
  await flush();
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_a');
  assert.equal($('stop').disabled, false);
  assert.equal($('prompt').value, '');   // accepted: the text and the file are spent
  assert.equal($('preview').hidden, true);
  for (const e of events.slice(0, 4)) run.channel.push(e);   // message_start .. encode running
  await flush();
  nextFrame();
  assert.equal(cardOf().getAttribute('data-status'), 'running');
  assert.equal($('status').textContent, 'encode running');
  assert.deepEqual(texts(qa(q($('conversation'), '.turn.user'), '.options .chip')), ['beam 5', '100 tok', 'cached', 'k 4/3']);   // the options the server resolved

  const listed = h.fetch.to('GET', '/v1/sessions?').length;
  for (const e of events.slice(4)) run.channel.push(e);
  run.channel.end();
  await turn;
  assert.equal(h.fetch.to('GET', '/v1/sessions?').length, listed + 1);   // its title and count changed: the sidebar asks again, once
  assert.equal(cardOf().getAttribute('data-status'), 'done');   // drawn at once when the turn ends, not on the next frame
  assert.deepEqual([q(cardOf(), '.report-section h3').textContent, q(cardOf(), '.report-section p').textContent], ['Findings', 'clear.']);
  assert.equal($('status').textContent, 'Report ready');
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
  assert.equal($('conversation').getAttribute('aria-busy'), 'true');   // taken off a frame later, not in the task that ends the turn
  assert.ok(h.fetch.to('GET', '/v1/sessions?').length >= 2, 'the sidebar was refreshed');
  nextFrame();
  assert.equal(cardOf().getAttribute('data-status'), 'done');   // a frame that was still queued changes nothing
  assert.equal($('conversation').hasAttribute('aria-busy'), false);   // and that frame is the one that takes aria-busy off
});

test('the status region says a thing only when it changes', async () => {
  const h = await ready();
  const spoken = [];
  const status = $('status');
  let current = '';
  Object.defineProperty(status, 'textContent', { get: () => current, set: (v) => { current = v; spoken.push(v); } });   // every write is counted
  const run = (events) => events.forEach((e) => h.api.streams[0].channel.push(e));
  const turn = h.app.send();
  await flush();
  h.api.streams[0].accept('m_a');
  await flush();
  resetEvents();
  run([startEv(), stageStartEv('encode', 1)]);
  await flush();
  nextFrame();
  run([snapEv('Findings: a', 0)]);   // still "encode running": the stage has not ended
  await flush();
  nextFrame();
  run([snapEv('Findings: a b', 1)]);
  await flush();
  nextFrame();
  assert.deepEqual(spoken, ['Queued', 'encode running']);   // accepted, then the stage; three more frames said nothing new
  run([stageEndEv('encode'), stopEv('done', { report: 'Findings: a b' })]);
  h.api.streams[0].channel.end();
  await turn;
  assert.deepEqual(spoken, ['Queued', 'encode running', 'Report ready']);
});

const NEW_SESSION = { id: 's_new', title: '', mode: 'private', turns: 0, created_at: iso(3), updated_at: iso(3) };

test('a new chat is listed under its title as soon as the server has its first turn, not as an empty "New chat" until the turn ends (P4-H)', async () => {
  const h = harness({ routes: { 'POST /v1/sessions': NEW_SESSION } });
  await h.app.start();
  await flush();
  const listed = h.fetch.to('GET', '/v1/sessions?').length;
  attach(imageFile('chest.png'));
  const turn = h.app.send();
  await flush();
  assert.equal(h.fetch.to('GET', '/v1/sessions?').length, listed);   // made, but with no turn yet: nothing to list that the turn will not change
  h.sessions = [sess('s_new', 'chest.png', 1)];                      // what the server says once it has the turn: the turn titled the chat
  h.api.streams[0].accept('m_a');
  await flush();
  assert.equal(h.fetch.to('GET', '/v1/sessions?').length, listed + 1);
  const [row] = qa($('session-list'), 'li');
  assert.deepEqual([row.getAttribute('data-session'), q(row, '.session-title').textContent], ['s_new', 'chest.png']);
  assert.match(q(row, '.session-meta').textContent, /· 1 turn$/);
  assert.equal(q(row, 'a').getAttribute('aria-current'), 'page');
  assert.equal(cardOf().getAttribute('data-status'), 'running');   // while it runs
  h.api.streams[0].channel.push(...fullTurn());
  h.api.streams[0].channel.end();
  await turn;
});

test('a first turn the server refuses leaves no empty chat behind: the chat made for it is deleted and the page is back on an empty one (P4-H)', async (t) => {
  const logged = captureErrors(t);
  const h = harness({ routes: { 'POST /v1/sessions': NEW_SESSION, 'DELETE /v1/sessions/s_new': () => new Response(null, { status: 204 }) } });
  await h.app.start();
  await flush();
  attach(imageFile('notes.png'));
  $('prompt').value = 'a note';
  h.api.refuse = refusal(422, 'Use a PNG, JPEG or WEBP image.');
  await h.app.send();
  await flush();
  assert.deepEqual(h.fetch.to('DELETE', '/v1/sessions/').map((c) => c.url), ['/v1/sessions/s_new']);
  assert.equal(h.win.location.hash, '#/new');
  assert.equal(q($('notice'), 'p').textContent, 'Use a PNG, JPEG or WEBP image.');   // the reason stays
  assert.equal($('prompt').value, 'a note');                                         // and so does the composer
  assert.equal($('preview').hidden, false);
  assert.equal($('exports').hidden, true);                                           // no chat to export
  assert.deepEqual(qa($('session-list'), 'li'), []);
  assert.equal($('send').disabled, false);
  assert.deepEqual(logged, []);
  h.api.refuse = null;                                                               // Send again makes a chat again
  const turn = h.app.send();
  await flush();
  assert.equal(h.fetch.to('POST', '/v1/sessions').length, 2);
  h.api.streams.at(-1).accept('m_a');
  await flush();
  h.api.streams.at(-1).channel.push(...fullTurn());
  h.api.streams.at(-1).channel.end();
  await turn;
  assert.equal(h.win.location.hash, '#/s/s_new');
  assert.equal(h.fetch.to('DELETE', '/v1/sessions/').length, 1);                    // a chat with a turn is never deleted
});

test('a stream the network drops mid-turn is followed by the poll and logs nothing: a network failure is not a bug of the page (P4-H)', async (t) => {
  const logged = captureErrors(t);
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 9));   // up to the first snapshot of the report
  await flush();
  run.channel.fail(transportFailure(new TypeError('network error')));   // what api.js throws when the server dies mid-stream
  await flush();
  assert.equal(h.api.polls.length, 1);   // the poll carries the turn on from its last seq
  assert.equal(h.api.polls[0].opts.after, 9);
  assert.deepEqual(logged, []);           // and nothing is logged as a bug
  assert.equal($('notice').hidden, true);
  h.api.polls[0].channel.push(...events.slice(9));
  h.api.polls[0].channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.deepEqual(logged, []);
});

test('a TypeError out of the stream or the poll that api.js did not mark as the transport\'s is a bug, and is logged (P4-H)', async (t) => {
  const logged = captureErrors(t);
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 9));
  await flush();
  const bug = new TypeError("Cannot read properties of undefined (reading 'delta')");   // the stream handling's own failure
  run.channel.fail(bug);
  await flush();
  assert.deepEqual(logged.map((args) => args[0]), [bug]);   // logged, though it is a TypeError
  assert.equal(h.api.polls.length, 1);                      // and the poll still carries the turn on
  const broken = new TypeError('x is not iterable');        // the poll gives up on a failure that is no network's
  h.api.polls[0].channel.fail(broken);
  await flush();
  assert.deepEqual(logged.map((args) => args[0]), [bug, broken]);
  assert.equal(q($('notice'), 'p').textContent, 'Something went wrong — see the console');   // and the console has it
  await turn;                                               // the send itself is over: the poll it ended on failed
  buttonOf($('notice'), 'Retry').click();                   // Retry follows the turn again
  await flush();
  h.api.polls.at(-1).channel.push(...events.slice(9));
  h.api.polls.at(-1).channel.end();
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.deepEqual(logged.map((args) => args[0]), [bug, broken]);   // nothing more
});

test('a Send right after a refused first turn makes a new chat, and does not send its turn into the one being deleted (P4-H)', async () => {
  let release;
  const held = new Promise((resolve) => { release = resolve; });
  let made = 0;
  const h = harness({ routes: {
    'POST /v1/sessions': () => ({ ...NEW_SESSION, id: made++ ? 's_two' : 's_new' }),
    'DELETE /v1/sessions/s_new': async () => { await held; return new Response(null, { status: 204 }); },   // a slow delete
  } });
  await h.app.start();
  await flush();
  attach(imageFile('notes.png'));
  h.api.refuse = refusal(422, 'Use a PNG, JPEG or WEBP image.');
  await h.app.send();
  await flush();
  assert.equal(h.win.location.hash, '#/new');   // back on the empty chat at once, while the delete is still on its way
  h.api.refuse = null;
  const turn = h.app.send();   // the composer is free again: the user sends straight away
  await flush();
  assert.equal(h.fetch.to('POST', '/v1/sessions').length, 2);
  assert.equal(h.api.streams.at(-1).opts.sessionId, 's_two');   // not the chat being deleted
  release();
  h.api.streams.at(-1).accept('m_a');
  await flush();
  h.api.streams.at(-1).channel.push(...fullTurn());
  h.api.streams.at(-1).channel.end();
  await turn;
  assert.equal(h.win.location.hash, '#/s/s_two');
});

test('a refused turn in a chat that was there before deletes nothing, even when the chat has no turn yet', async () => {
  const h = await ready();   // s_a: an existing chat with no turns
  h.api.refuse = refusal(422, 'Use a PNG, JPEG or WEBP image.');
  await h.app.send();
  await flush();
  assert.deepEqual(h.fetch.to('DELETE', '/v1/sessions/'), []);
  assert.equal(h.win.location.hash, '#/s/s_a');
});

test('Send pressed while it has the focus hands the focus to the note field before it is switched off, so the keyboard keeps its place (P4-H)', async () => {
  const h = await ready();
  press($('prompt'), 'Tab');   // the keyboard moves the focus: it shows its ring (:focus-visible)
  $('send').focus();
  assert.equal(document.activeElement, $('send'));
  assert.equal($('send').matches(':focus-visible'), true);
  const turn = h.app.send();   // what the composer's submit runs when Send is pressed with Enter, Space or a click
  assert.equal($('send').disabled, true);
  assert.equal(document.activeElement, $('prompt'));   // not the page: a disabled button loses the focus, and Chrome gives it to the body
  h.api.streams[0].accept('m_a');
  await flush();
  assert.equal(document.activeElement, $('prompt'));
  h.api.streams[0].channel.push(...fullTurn().slice(1));
  h.api.streams[0].channel.end();
  await turn;
  assert.equal($('send').disabled, false);
  assert.equal(document.activeElement, $('prompt'));   // where the next note is typed
  $('prompt').focus();
  const second = h.app.send();   // sent from the note field: the focus stays there and is not moved
  assert.equal(document.activeElement, $('prompt'));
  h.api.streams[1].accept('m_b');
  await flush();
  h.api.streams[1].channel.push(...fullTurn('m_b').slice(1));
  h.api.streams[1].channel.end();
  await second;
});

test('Send and Stop pressed with a pointer (a click, a tap) leave the focus alone: on a phone the note field would open the keyboard (P4-H)', async () => {
  const h = await ready();
  $('send').dispatchEvent(new ShimEvent('pointerdown', { bubbles: true }));
  $('send').focus();   // the focus a click or a tap gives a button: no ring
  assert.equal($('send').matches(':focus-visible'), false);
  const turn = h.app.send();
  assert.notEqual(document.activeElement, $('prompt'));   // no soft keyboard over the card that is about to stream
  assert.equal(document.activeElement, document.body);    // the disabled Send gave it up to the page, as Chrome does
  h.api.streams[0].accept('m_a');
  await flush();
  $('stop').dispatchEvent(new ShimEvent('pointerdown', { bubbles: true }));
  $('stop').focus();
  const stopping = h.app.stopTurn();
  assert.notEqual(document.activeElement, $('prompt'));
  await stopping;
  await flush();   // the dropped stream hands the turn to a poll
  h.api.polls[0].channel.push(stopEv('aborted'));
  h.api.polls[0].channel.end();
  await turn;
  assert.notEqual(document.activeElement, $('prompt'));
});

test('a turn that the server refuses before its stream opens: the notice says why, its user turn is taken back, and the composer is as it was', async (t) => {
  const logged = captureErrors(t);
  const h = await ready();
  $('prompt').value = 'beam 99';
  const cases = [
    [refusal(422, 'Invalid options: beam_size: Input should be less than or equal to 8'), 'Invalid options: beam_size: Input should be less than or equal to 8'],
    [refusal(413, 'The image is over the 20 MB upload limit.'), 'The image is over the 20 MB upload limit.'],
    [refusal(401, 'Missing or wrong token'), 'Enter the access token in Settings'],
    [refusal(429, 'The server is busy with 4 turns; try again shortly.'), 'The server is busy; try again shortly'],
    [transportFailure(new TypeError('Failed to fetch')), 'Cannot reach the server. Check the connection and try again.'],   // as api.js throws it
  ];
  for (const [err, shown] of cases) {
    h.api.refuse = err;
    await h.app.send();
    assert.equal(q($('notice'), 'p').textContent, shown);
    assert.equal($('notice').hidden, false);
    assert.equal($('conversation').children.length, 0, 'the optimistic user turn is gone');
    assert.equal($('prompt').value, 'beam 99');   // nothing to type again
    assert.equal($('preview').hidden, false);
    assert.equal(q($('preview'), 'span').textContent.startsWith('chest.png'), true);
    assert.equal($('send').disabled, false);
    assert.equal($('stop').hidden, true);
    nextFrame();   // aria-busy comes off a frame after the page stops being busy
    assert.equal($('conversation').hasAttribute('aria-busy'), false);
  }
  assert.deepEqual(logged, []);   // a refusal, the network being down, and a cancel are what they say, not bugs to be logged
  h.api.refuse = null;
  const turn = h.app.send();   // the retry needs no re-entry, and clears the notice
  await flush();
  assert.equal($('notice').hidden, true);
  h.api.streams.at(-1).accept('m_a');
  await flush();
  h.api.streams.at(-1).channel.push(...fullTurn().slice(-1));
  h.api.streams.at(-1).channel.end();
  await turn;
  assert.equal($('conversation').children.length, 2);   // the user turn and the card, once each
  assert.equal($('prompt').value, '');
});

test('Enter sends and Shift+Enter is a newline; an input method\'s Enter is not a send; Enter during a turn does nothing', async () => {
  const h = await ready();
  const shift = press($('prompt'), 'Enter', { shiftKey: true });
  assert.equal(shift.defaultPrevented, false);   // the browser inserts the newline
  const composing = press($('prompt'), 'Enter', { isComposing: true });
  const legacy = press($('prompt'), 'Enter', { keyCode: 229 });
  assert.deepEqual([composing.defaultPrevented, legacy.defaultPrevented], [false, false]);
  const letter = press($('prompt'), 'a');
  assert.equal(letter.defaultPrevented, false);
  await flush();
  assert.equal(h.api.streams.length, 0);

  const enter = press($('prompt'), 'Enter');
  assert.equal(enter.defaultPrevented, true);
  await flush();
  assert.equal(h.api.streams.length, 1);   // sent
  const again = press($('prompt'), 'Enter');
  assert.equal(again.defaultPrevented, true);   // no newline either
  await flush();
  assert.equal(h.api.streams.length, 1);   // and not sent twice
  assert.equal($('send').disabled, true);

  const submit = new ShimEvent('submit', { bubbles: true, cancelable: true });   // the Send button
  $('composer').dispatchEvent(submit);
  assert.equal(submit.defaultPrevented, true);   // the form never reloads the page
  await flush();
  assert.equal(h.api.streams.length, 1);
});

test('Stop sends the cancel first and drops the stream second; the AbortError is quiet and the poll brings the turn\'s own message_stop', async (t) => {
  const errors = [];
  const original = console.error;
  console.error = (...args) => errors.push(args);
  t.after(() => { console.error = original; });
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  assert.equal($('stop').disabled, true);
  $('stop').click();   // before the id is known: nothing happens
  await flush();
  assert.deepEqual(h.api.cancels, []);
  run.accept('m_a');
  await flush();
  for (const e of events.slice(0, 9)) run.channel.push(e);   // up to the first snapshot of the report
  await flush();
  nextFrame();
  assert.equal($('stop').disabled, false);

  $('stop').focus();
  assert.equal(document.activeElement, $('stop'));
  $('stop').click();
  await flush();
  assert.deepEqual(h.api.order, ['cancel', 'abort']);   // the server is asked first, then the stream is dropped
  assert.deepEqual(h.api.cancels.map((c) => c.messageId), ['m_a']);
  assert.equal($('stop').textContent, 'Stopping…');
  assert.equal($('stop').disabled, true);
  $('stop').click();
  await flush();
  assert.equal(h.api.cancels.length, 1);   // asked once
  assert.equal(h.api.polls.length, 1);     // the dropped stream is followed by a poll, from the last seq the view has
  assert.equal(h.api.polls[0].opts.messageId, 'm_a');
  assert.equal(h.api.polls[0].opts.after, 9);
  assert.equal($('send').disabled, true);   // not over until the turn says so
  assert.equal(cardOf().getAttribute('data-status'), 'running');

  const aborted = stopEv('aborted');
  h.api.polls[0].channel.push(aborted);
  h.api.polls[0].channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'aborted');
  assert.equal(q(cardOf(), '.note.stopped').textContent, 'Turn stopped');
  assert.equal($('status').textContent, 'Turn stopped');
  assert.equal($('stop').hidden, true);
  assert.equal($('stop').textContent, 'Stop');   // ready for the next turn
  assert.equal($('stop').disabled, false);
  assert.equal($('send').disabled, false);
  assert.equal(document.activeElement, $('prompt'));   // Stop is gone: focus moved to where the next turn is typed, not to the page's top
  assert.deepEqual(errors, []);   // an AbortError is the user's doing: nothing is reported
});

test('a cancel the server refuses leaves the stream running and Stop usable again, with the reason in the notice', async () => {
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(events[0]);
  await flush();
  h.api.cancelError = refusal(404, 'Message not found.', 'not_found_error');
  $('stop').click();
  await flush();
  assert.deepEqual(h.api.order, ['cancel']);   // no abort: the turn is still running on the server
  assert.equal(q($('notice'), 'p').textContent, 'Message not found.');
  assert.equal($('stop').textContent, 'Stop');
  assert.equal($('stop').disabled, false);
  run.channel.push(...events.slice(1, 4));   // and the stream still brings events
  await flush();
  nextFrame();
  assert.equal($('status').textContent, 'encode running');
  run.channel.push(...events.slice(4));
  run.channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');
});

test('3 s without an event drops the stream and polls from the last seq until the turn ends; an event restarts the clock', async () => {
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  assert.equal(h.timers.intervals, 1);   // the watchdog's ticker
  run.channel.push(...events.slice(0, 3));
  await flush();
  h.timers.advance(2500);
  run.channel.push(events[3]);   // an event at 2.5 s
  await flush();
  h.timers.advance(2900);        // 5.4 s: 2.9 s of silence
  await flush();
  assert.deepEqual(h.api.order, []);
  assert.equal(h.api.polls.length, 0);
  h.timers.advance(100);         // 5.5 s: 3 s of silence
  await flush();
  assert.deepEqual(h.api.order, ['abort']);   // the stream is dropped, and nothing is cancelled
  assert.deepEqual(h.api.cancels, []);
  assert.equal(h.api.polls.length, 1);
  assert.equal(h.api.polls[0].opts.after, 4);   // from the last seq the view has
  assert.equal(h.timers.intervals, 0);          // no stall check while polling
  assert.equal($('send').disabled, true);
  h.api.polls[0].channel.push(...events.slice(4));
  h.api.polls[0].channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal(q(cardOf(), '.report-section p').textContent, 'clear.');
  assert.equal($('send').disabled, false);
  assert.equal(h.api.polls.length, 1);
});

test('a turn still queued sends no bytes: after 3 s the page polls it from the start, and the queue\'s events arrive by polling', async () => {
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  h.api.streams[0].accept('m_a');
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'running');
  assert.equal(statusOf(), 'Queued');
  h.timers.advance(3000);
  await flush();
  assert.equal(h.api.polls.length, 1);
  assert.equal(h.api.polls[0].opts.after, 0);
  h.api.polls[0].channel.push(...events);
  h.api.polls[0].channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  function statusOf() { return $('status').textContent; }
});

test('a stream that closes without a message_stop is followed by a poll too', async () => {
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 5));
  run.channel.end();   // the connection closed, no message_stop
  await flush();
  assert.equal(h.api.polls.length, 1);
  assert.equal(h.api.polls[0].opts.after, 5);
  h.api.polls[0].channel.push(...events.slice(5));
  h.api.polls[0].channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');
});

test('a server that sends no X-Message-Id still gets its card, from message_start, and Stop has the id it needs', async () => {
  const h = await ready();
  h.api.silent = true;
  const events = fullTurn('m_s');
  const turn = h.app.send();
  await flush();
  assert.equal(cardOf(), null);   // nothing to draw yet
  assert.equal($('stop').disabled, true);
  h.api.streams[0].channel.push(events[0], events[1]);
  await flush();
  nextFrame();
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_s');
  assert.equal($('stop').disabled, false);
  assert.equal($('prompt').value, '');
  $('stop').click();
  await flush();
  assert.deepEqual(h.api.cancels.map((c) => c.messageId), ['m_s']);
  h.api.polls[0].channel.push(stopEv('aborted', { message_id: 'm_s' }));
  h.api.polls[0].channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'aborted');
});

test('a refusal that reaches a chat the user already left does not stop the watchdog of the turn running in the chat they are in now', async () => {
  const h = harness({ sessions: [sess('s_a', 'A', 0), sess('s_b', 'B', 0)], routes: { ...EXISTING, 'GET /v1/sessions/s_b': { ...sess('s_b', 'B', 0), messages: [] } } });
  h.api.holdAbort = true;   // the first stream does not notice its abort for a while
  await h.app.start();
  await flush();
  attach(imageFile());
  const first = h.app.send();
  await flush();
  h.win.location.hash = '#/s/s_b';   // leaves the chat with that upload in flight
  await flush();
  h.api.holdAbort = false;
  attach(imageFile('second.png'));
  const second = h.app.send();
  await flush();
  h.api.streams[1].accept('m_b');
  await flush();
  assert.equal(h.timers.intervals, 1);   // the second turn's stall check
  h.api.streams[0].fail(abortError());   // now the first one ends, as a refusal of a turn that nobody is waiting for any more
  await first;
  assert.equal(h.timers.intervals, 1);   // and does not take that check away
  assert.equal($('notice').hidden, true);
  assert.equal($('send').disabled, true);
  h.timers.advance(3000);
  await flush();
  assert.equal(h.api.polls.length, 1);   // so the second turn is still handed to polling after 3 s
  assert.equal(h.api.polls[0].opts.messageId, 'm_b');
  h.api.polls[0].channel.push(stopEv('done', { message_id: 'm_b' }));
  h.api.polls[0].channel.end();
  await second;
  assert.equal($('send').disabled, false);
});

test('a chat made for a first turn does not move the address of a chat the user has gone to meanwhile', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const h = harness({
    hash: '#/new', sessions: [sess('s_b', 'B', 1)],
    routes: {
      ...doneSession('s_b', 'B', 'm_b'),
      'POST /v1/sessions': async () => { await gate; return { id: 's_new', title: '', mode: 'private', turns: 0, created_at: iso(3), updated_at: iso(3) }; },
    },
  });
  await h.app.start();
  await flush();
  assert.equal($('conversation').children.length, 0);   // an empty chat
  attach(imageFile());
  const turn = h.app.send();   // the chat is being made
  await flush();
  assert.equal(h.api.streams.length, 0);
  assert.equal($('send').disabled, true);
  h.win.location.hash = '#/s/s_b';   // the user opens another chat before it is there
  await flush();
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');
  assert.equal($('send').disabled, false);
  release();
  await turn;
  await flush();
  assert.equal(h.win.location.hash, '#/s/s_b');   // still where the user went
  assert.equal(h.win.replaced.includes('#/s/s_new'), false);
  assert.equal(h.api.streams.length, 0);   // and no turn was started in a chat that is not on screen
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');
  assert.equal($('send').disabled, false);
  assert.equal($('exports').hidden, false);   // the open chat is s_b, which can be exported
});

test('a poll that fails for good shows its error with a Retry that polls again from the last seq', async () => {
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 6));
  await flush();
  h.timers.advance(3000);
  await flush();
  const first = h.api.polls[0];
  first.channel.push(events[6]);
  await flush();
  first.channel.fail(refusal(404, 'Message not found.', 'not_found_error'));   // not one of the failures api.js retries
  await turn;
  await flush();
  assert.equal(q($('notice'), 'p').textContent, 'Message not found.');
  const retry = buttonOf($('notice'), 'Retry');
  assert.equal(retry.hidden, false);
  assert.equal($('send').disabled, true);    // the turn is still running as far as the page knows
  assert.equal(cardOf().getAttribute('data-status'), 'running');
  retry.click();
  await flush();
  assert.equal($('notice').hidden, true);
  assert.equal(h.api.polls.length, 2);
  assert.equal(h.api.polls[1].opts.after, 7);   // re-created from the view's last seq
  h.api.polls[1].channel.push(...events.slice(7));
  h.api.polls[1].channel.end();
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal($('send').disabled, false);
  retry.click();   // a second click on the same button does nothing
  await flush();
  assert.equal(h.api.polls.length, 2);
});

test('a poll that ends with the turn still running leaves a notice and does not leave the page waiting', async () => {
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  h.api.streams[0].accept('m_a');
  await flush();
  h.api.streams[0].channel.push(...events.slice(0, 4));
  h.api.streams[0].channel.end();
  await flush();
  h.api.polls[0].channel.end();   // the server says the turn left running, and the log has no end
  await turn;
  await flush();
  assert.match(q($('notice'), 'p').textContent, /ended without a result/);
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
});

test('a session opened with its last turn still running polls it from its last seq, and settles when it ends', async () => {
  resetEvents();
  const partial = [startEv(), stageStartEv('preprocess', 0), stageEndEv('preprocess'), stageStartEv('encode', 1)];
  const rest = [stageEndEv('encode'), skipEv('retrieve', 'gallery_unavailable'), stopEv('done', { report: 'Findings: ok.', display_report: 'Findings: ok.' })];
  const h = harness({
    sessions: [sess('s_a', 'running one', 1)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'running one', 1), messages: [userMsg('u_a', '', 'chest.png'), botMsg('m_a', 'running', START_DATA.options)] },
      'GET /v1/messages/m_a': { ...botMsg('m_a', 'running'), events: rows(partial) },
    },
  });
  await h.app.start();
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'running');
  assert.equal($('send').disabled, true);
  assert.equal($('stop').hidden, false);
  assert.equal($('stop').disabled, false);   // the id is known: Stop can cancel it
  assert.equal($('conversation').getAttribute('aria-busy'), 'true');
  assert.equal(h.api.streams.length, 0);     // there is no stream to rejoin
  assert.equal(h.api.polls.length, 1);
  assert.deepEqual([h.api.polls[0].opts.messageId, h.api.polls[0].opts.after], ['m_a', 4]);
  h.api.polls[0].channel.push(...rest);
  h.api.polls[0].channel.end();
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
  nextFrame();   // aria-busy comes off a frame after the page stops being busy
  assert.equal($('conversation').hasAttribute('aria-busy'), false);
});

test('a question turn and a turn stopped before its content block closed both replay', async () => {
  resetEvents();
  const question = [startEv({ message_id: 'm_q', image: null }), ev('warning', { code: 'not_a_command', message: 'This is a report generator, not a question answerer.' }),
                    stopEv('done', { message_id: 'm_q' })];
  const cancelled = [startEv({ message_id: 'm_c' }), stageStartEv('preprocess', 0), stageEndEv('preprocess'), stageStartEv('generate', 3),
                     ev('content_block_start', { index: 0, content_block: { type: 'report', text: '' } }), snapEv('Findings: part'),
                     stopEv('aborted', { message_id: 'm_c' })];   // no content_block_stop, no stage_end of generate
  const h = harness({
    sessions: [sess('s_a', 'two turns', 2)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'two turns', 2), messages: [userMsg('u_q', 'what is this?'), botMsg('m_q', 'done', START_DATA.options),
                                                                           userMsg('u_c', '', 'chest.png'), botMsg('m_c', 'aborted', START_DATA.options)] },
      'GET /v1/messages/m_q': { ...botMsg('m_q', 'done'), events: rows(question) },
      'GET /v1/messages/m_c': { ...botMsg('m_c', 'aborted'), events: rows(cancelled) },
    },
  });
  await h.app.start();
  await flush();
  const [u1, c1, u2, c2] = $('conversation').children;
  assert.deepEqual([u1, c1, u2, c2].map((n) => n.getAttribute('aria-label')), ['Your message, turn 1', 'Assistant report, turn 1', 'Your message, turn 2', 'Assistant report, turn 2']);
  assert.equal(q(c1, '.note.notice').textContent, 'This is a report generator, not a question answerer.');
  assert.equal(q(u1, '.user-text').textContent, 'what is this?');
  assert.equal(c1.getAttribute('data-status'), 'done');
  assert.equal(q(c2, '.note.stopped').textContent, 'Turn stopped');
  assert.equal(q(c2, '.report-section p').textContent, 'part');
  assert.equal(qa(c2, '[data-state="running"]').length, 0);   // nothing is left spinning
  assert.equal(h.api.polls.length, 0);   // both are over: nothing to follow
  assert.equal($('send').disabled, false);
});

// ---- leaving a session, a new token ---------------------------------------------------------------------------------------------

const stageToggle = (card, stage) => q(card, `li[data-stage="${stage}"] > button`);
const doneSession = (id, title, botId, text = 'from ' + id) => ({
  [`GET /v1/sessions/${id}`]: { ...sess(id, title, 1), messages: [userMsg(`u_${id}`, text, `${id}.png`), botMsg(botId, 'done', START_DATA.options)] },
  [`GET /v1/messages/${botId}`]: () => ({ ...botMsg(botId, 'done'), events: rows(fullTurn(botId)) }),
});

test('switching session replaces the conversation, clears the image cache and the open-stage memory, and moves the highlight', async () => {
  const h = harness({ sessions: [sess('s_b', 'B'), sess('s_a', 'A')], routes: { ...doneSession('s_b', 'B', 'm_b'), ...doneSession('s_a', 'A', 'm_a2') } });
  await h.app.start();
  await flush();
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');
  const cleared = h.api.cleared;
  stageToggle(cardOf(), 'encode').click();   // a stage opened here must not stay open in a card the next view builds
  assert.equal(stageToggle(cardOf(), 'encode').getAttribute('aria-expanded'), 'true');

  h.win.location.hash = '#/s/s_a';   // the user follows a link in the sidebar: a hashchange
  await flush();
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_a');
  assert.equal($('conversation').children.length, 2);
  assert.ok(h.api.cleared > cleared, 'the image cache was cleared');
  assert.equal(q(q($('session-list'), '[aria-current]').parentNode, '.session-title').textContent, 'A');
  h.win.location.hash = '#/s/s_b';
  await flush();
  assert.equal(stageToggle(cardOf(), 'encode').getAttribute('aria-expanded'), 'false');   // the memory belonged to the view that was left
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_b');
});

test('leaving a chat with a turn running drops its stream and cancels nothing; coming back finds it running and polls it', async () => {
  let aRunning = false;
  const events = fullTurn();
  const h = harness({
    sessions: [sess('s_a', 'A', 0), sess('s_b', 'B', 1)],
    routes: {
      ...doneSession('s_b', 'B', 'm_b'),
      'GET /v1/sessions/s_a': () => ({ ...sess('s_a', 'A', aRunning ? 1 : 0), messages: aRunning ? [userMsg('u_a', '', 'a.png'), botMsg('m_a', 'running', START_DATA.options)] : [] }),
      'GET /v1/messages/m_a': () => ({ ...botMsg('m_a', 'running'), events: rows(events.slice(0, 4)) }),
    },
  });
  await h.app.start();
  await flush();
  attach(imageFile('a.png'));
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 4));
  await flush();
  assert.equal($('send').disabled, true);

  h.win.location.hash = '#/s/s_b';
  await flush();
  await turn;
  assert.deepEqual(h.api.order, ['abort']);   // dropped, not cancelled
  assert.equal(run.opts.signal.aborted, true);
  assert.deepEqual(h.api.cancels, []);
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');
  assert.equal($('send').disabled, false);   // this view has no turn running
  assert.equal($('stop').hidden, true);
  nextFrame();   // aria-busy comes off a frame after the page stops being busy
  assert.equal($('conversation').hasAttribute('aria-busy'), false);
  assert.equal(h.timers.intervals, 0);   // and no ticker left over
  run.channel.push(...events.slice(4, 9));   // what was already on the wire when the stream was dropped
  await flush();
  nextFrame();
  assert.equal($('conversation').children.length, 2);   // is not drawn into this view
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');

  aRunning = true;
  h.win.location.hash = '#/s/s_a';
  await flush();
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_a');
  assert.equal($('send').disabled, true);   // found running: polled from the last seq it has
  assert.deepEqual([h.api.polls[0].opts.messageId, h.api.polls[0].opts.after], ['m_a', 4]);
});

test('leaving a chat whose turn is being polled drops the poll, and a message_stop that was already read changes nothing', async () => {
  resetEvents();
  const partial = [startEv(), stageStartEv('preprocess', 0), stageEndEv('preprocess'), stageStartEv('encode', 1)];
  const h = harness({
    sessions: [sess('s_a', 'running', 1), sess('s_b', 'B', 1)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'running', 1), messages: [userMsg('u_a', '', 'chest.png'), botMsg('m_a', 'running', START_DATA.options)] },
      'GET /v1/messages/m_a': { ...botMsg('m_a', 'running'), events: rows(partial) },
      ...doneSession('s_b', 'B', 'm_b'),
    },
  });
  h.api.lenient = true;   // what a poll had already fetched still comes out after the abort
  await h.app.start();
  await flush();
  assert.equal(h.api.polls.length, 1);
  const poll = h.api.polls[0];
  assert.equal($('send').disabled, true);
  const listed = h.fetch.to('GET', '/v1/sessions?').length;
  poll.channel.push(stopEv('done', { report: 'Findings: late.', display_report: 'Findings: late.' }));   // fetched just as the user leaves
  h.win.location.hash = '#/s/s_b';
  await flush();
  assert.equal(poll.opts.signal.aborted, true);   // the poll was dropped
  assert.deepEqual(h.api.cancels, []);            // and the turn was not cancelled
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');
  assert.equal(h.fetch.to('GET', '/v1/sessions?').length, listed);   // that turn did not "end" in the chat now on screen: no refresh, no settle
  assert.equal($('send').disabled, false);
  assert.equal($('conversation').children.length, 2);
});

test('a chat that loads after the user has moved on is not drawn', async () => {
  let release;
  const gate = new Promise((resolve) => { release = resolve; });
  const h = harness({
    hash: '#/s/s_b', sessions: [sess('s_b', 'B'), sess('s_a', 'A')],
    routes: {
      ...doneSession('s_b', 'B', 'm_b'),
      'GET /v1/sessions/s_a': async () => { await gate; return { ...sess('s_a', 'A', 1), messages: [userMsg('u_a', 'from s_a', 'a.png'), botMsg('m_a2', 'done', START_DATA.options)] }; },
      'GET /v1/messages/m_a2': () => ({ ...botMsg('m_a2', 'done'), events: rows(fullTurn('m_a2')) }),
    },
  });
  await h.app.start();
  await flush();
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');
  h.win.location.hash = '#/s/s_a';   // a slow one
  await flush();
  assert.equal($('send').disabled, true);   // loading
  assert.equal($('conversation').children.length, 0);
  h.win.location.hash = '#/s/s_b';   // the user changes their mind
  await flush();
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');
  assert.equal($('send').disabled, false);
  release();
  await flush(12);
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');   // s_a arrived late and was not drawn over it
  assert.equal($('conversation').children.length, 2);
  assert.equal($('send').disabled, false);
  assert.equal(q($('session-list'), '[aria-current]').parentNode.getAttribute('data-session'), 's_b');
});

test('a turn whose id arrives after the user left its chat draws nothing into the chat they are in now', async () => {
  const h = harness({ sessions: [sess('s_a', 'A', 0), sess('s_b', 'B', 1)], routes: { ...doneSession('s_b', 'B', 'm_b'), ...EXISTING } });
  await h.app.start();
  await flush();
  attach(imageFile());
  const turn = h.app.send();
  await flush();
  const { opts } = h.api.streams[0];
  h.win.location.hash = '#/s/s_b';
  await flush();
  await turn;
  opts.onMessageId('m_late');   // the server's answer to an upload that was in flight
  nextFrame();
  assert.equal($('conversation').children.length, 2);
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');
  assert.equal($('notice').hidden, true);   // leaving is not a failure
  assert.equal($('send').disabled, false);
});

test('a new access token is stored, clears the image cache, drops a running stream without cancelling it, and loads everything again with it', async () => {
  const h = await ready();
  const turn = h.app.send();
  await flush();
  h.api.streams[0].accept('m_a');
  await flush();
  const cleared = h.api.cleared;
  const modelCalls = h.fetch.to('GET', '/v1/models').length;
  $('settings').click();
  const token = q($('drawer'), 'input[type="password"]');
  assert.equal(token.getAttribute('autocomplete'), 'off');
  assert.match($('drawer').textContent, /Saved in this browser on this device, because you typed it here\./);
  token.value = '  s3cret  ';
  token.dispatchEvent(new ShimEvent('change', { bubbles: true }));
  assert.deepEqual(h.api.order, ['abort']);   // before anything is fetched again: the stream of the old identity goes first
  assert.ok(h.api.cleared > cleared, 'and so do its images');
  await flush(12);
  await turn;
  assert.equal(JSON.parse(h.storage.data.get(SETTINGS_KEY)).token, 's3cret');   // trimmed, and kept where the user typed it
  assert.deepEqual(h.api.order, ['abort']);
  assert.deepEqual(h.api.cancels, []);
  assert.ok(h.api.cleared > cleared);
  assert.equal(h.fetch.to('GET', '/v1/models').length, modelCalls + 1);
  assert.equal(h.fetch.to('GET', '/v1/models').at(-1).headers.Authorization, 'Bearer s3cret');
  assert.equal(h.fetch.to('GET', '/v1/sessions/s_a').at(-1).headers.Authorization, 'Bearer s3cret');   // the open chat was fetched again
  assert.equal($('send').disabled, false);

  const calls = h.fetch.calls.length;
  token.dispatchEvent(new ShimEvent('change', { bubbles: true }));   // the same token again: nothing to reload
  await flush();
  assert.equal(h.fetch.calls.length, calls);
});

// ---- the drawer ----------------------------------------------------------------------------------------------------------------------

const saved = (h) => JSON.parse(h.storage.data.get(SETTINGS_KEY));
const chipsText = () => texts(qa($('chips'), 'span'));
const change = (node) => node.dispatchEvent(new ShimEvent('change', { bubbles: true }));
const ownText = (node) => node.childNodes.filter((n) => n.nodeType === 3).map((n) => n.data).join('');   // a label's words, not its select's options
const labelled = (text) => qa($('drawer'), 'label').find((l) => (text instanceof RegExp ? text.test(ownText(l)) : ownText(l) === text));
const control = (text) => q(labelled(text), 'input, select');
// The number fields as the drawer labels them (P4-G): the range is in the label, built from BOUNDS.
const BEAM = 'Beam size (1–8)';
const BUDGET = 'Token budget (16–200)';
const SIMILAR = 'Similar images (0–12)';
const MATCHING = 'Matching reports (0–10)';

test('Settings opens and closes the drawer with the hidden attribute and aria-expanded; the close button and Esc close it and give focus back', async () => {
  const h = harness();
  await h.app.start();
  assert.equal($('drawer').hidden, true);
  assert.equal($('settings').getAttribute('aria-expanded'), 'false');
  assert.equal($('drawer').hasAttribute('style'), false);   // never style.display

  $('settings').focus();
  $('settings').click();
  assert.equal($('drawer').hidden, false);
  assert.equal($('settings').getAttribute('aria-expanded'), 'true');
  for (const other of ['a', 'Enter', 'Tab', ' ', 'Backspace', 'Esc']) {   // only "Escape" closes it
    assert.equal(press(document.activeElement, other).defaultPrevented, false, other);
    assert.equal($('drawer').hidden, false, other);
  }
  const close = $('drawer-close');
  assert.equal(close.getAttribute('aria-label'), 'Close settings');
  assert.equal(close.textContent, '✕');
  assert.equal(document.activeElement, close);   // focus moves into the panel it opened
  assert.equal(q($('drawer'), 'h2').textContent, 'Settings');

  $('settings').click();   // the same button closes it
  assert.deepEqual([$('drawer').hidden, $('settings').getAttribute('aria-expanded')], [true, 'false']);
  $('settings').click();
  close.click();
  assert.deepEqual([$('drawer').hidden, $('settings').getAttribute('aria-expanded')], [true, 'false']);
  assert.equal(document.activeElement, $('settings'));   // back where it was opened from

  $('settings').click();
  const escape = press(document.activeElement, 'Escape');
  assert.equal($('drawer').hidden, true);
  assert.equal(escape.defaultPrevented, true);
  assert.equal(document.activeElement, $('settings'));
  assert.equal(press(document.body, 'Escape').defaultPrevented, false);   // nothing open: Esc is left alone
  assert.equal(press(document.body, 'a').defaultPrevented, false);

  $('prompt').focus();   // opened while focus is elsewhere (a link in a card, say): Esc returns focus there
  $('settings').click();
  press(document.body, 'Escape');
  assert.equal(document.activeElement, $('prompt'));
  $('prompt').focus();
  $('settings').click();
  $('prompt').remove();   // and if that control is gone by then, Settings is where it lands
  press(document.body, 'Escape');
  assert.equal(document.activeElement, $('settings'));
});

test('the drawer lists the models with the default chosen; a model with no cache turns cached decode off, disabled and explained', async () => {
  const h = harness({ models: MODELS });
  await h.app.start();
  await flush();
  $('settings').click();
  const model = control('Model');
  assert.deepEqual(qa(model, 'option').map((o) => o.getAttribute('value')), [CARD_CACHED.name, CARD_PLAIN.name]);
  assert.equal(model.value, CARD_CACHED.name);   // the default is preselected
  const cached = control('Cached decode');
  assert.equal(cached.disabled, false);
  assert.equal(cached.checked, true);
  assert.equal($('cached-note').hidden, true);
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3']);

  model.value = CARD_PLAIN.name;
  change(model);
  assert.equal(saved(h).model, CARD_PLAIN.name);
  assert.equal(cached.disabled, true);
  assert.equal(cached.hasAttribute('disabled'), true);
  assert.equal(cached.checked, false);
  assert.equal($('cached-note').hidden, false);
  assert.equal(cached.getAttribute('aria-describedby'), 'cached-note');
  assert.match($('cached-note').textContent, /no decode cache/);
  assert.equal(chipsText().includes('uncached'), true);
  assert.equal(saved(h).cached_decode, true);   // the user's own choice is kept for the model that has the cache

  model.value = CARD_CACHED.name;   // the default again
  change(model);
  assert.equal(saved(h).model, '');   // stored as "the default", so a new server default is followed
  assert.deepEqual([cached.disabled, cached.checked, $('cached-note').hidden], [false, true, true]);

  const names = qa($('models-section'), 'h4').map((n) => n.textContent);
  assert.deepEqual(names, [CARD_CACHED.name, CARD_PLAIN.name]);   // the provenance link opens these
  assert.equal(qa($('models-section'), 'table').length, 2);
});

test('compile is offered only when the server allows it', async () => {
  const closed = harness({ models: MODELS });
  await closed.app.start();
  await flush();
  assert.equal(labelled(/^Compile/).hidden, true);
  const open = harness({ models: { ...MODELS, allow_compile: true } });
  await open.app.start();
  await flush();
  assert.equal(labelled(/^Compile/).hidden, false);
  const compile = control(/^Compile/);
  compile.checked = true;
  change(compile);
  assert.equal(saved(open).compile, true);
  assert.equal(chipsText().includes('compiled'), true);
});

test('each number field shows its range in its label and as min and max, and nothing typed out of range is changed into something else', async () => {
  const h = harness({ models: MODELS });
  await h.app.start();
  await flush();
  $('settings').click();
  for (const [label, key] of [[BEAM, 'beam_size'], [BUDGET, 'max_new_tokens'], [SIMILAR, 'k_images'], [MATCHING, 'k_reports']]) {
    const field = control(label);
    const [low, high] = BOUNDS[key];
    assert.deepEqual([field.getAttribute('data-setting'), field.getAttribute('min'), field.getAttribute('max')], [key, String(low), String(high)], label);
    assert.equal(ownText(labelled(label)), `${label.split(' (')[0]} (${low}–${high})`, label);   // the label says what the field takes (an en dash)
    typeInto(field, String(high + 1));                   // out of range: the field keeps what was typed, it is not snapped to the bound
    assert.equal(field.value, String(high + 1), label);
  }
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3']);   // the chips are what will be sent, and that has not changed
  assert.equal(h.storage.data.has(SETTINGS_KEY), false);                     // and nothing was stored
  for (const field of [control(BEAM), control(BUDGET), control(SIMILAR), control(MATCHING)]) change(field);   // leaving the field does not clamp it either
  assert.deepEqual([control(BEAM).value, control(BUDGET).value, control(SIMILAR).value, control(MATCHING).value], ['9', '201', '13', '11']);
  assert.equal(h.storage.data.has(SETTINGS_KEY), false);
});

test('decode, labels and display repair are settings too; greedy has no beam to size; the mode is shown and cannot be edited', async () => {
  const h = harness({ models: MODELS });
  await h.app.start();
  await flush();
  const decode = control('Decode');
  decode.value = 'greedy';
  change(decode);
  assert.equal(saved(h).decode, 'greedy');
  assert.equal(control(BEAM).disabled, true);
  assert.equal(chipsText()[0], 'greedy');
  decode.value = 'beam';
  change(decode);
  assert.equal(control(BEAM).disabled, false);
  const labels = control('CheXbert labels');
  labels.checked = false;
  change(labels);
  const repair = control('Display repair');
  assert.equal(repair.checked, true);   // on by default
  repair.checked = false;
  change(repair);
  assert.deepEqual([saved(h).label, saved(h).display_repair], [false, false]);
  assert.deepEqual(chipsText().slice(-2), ['labels off', 'raw text']);
  const mode = control('Mode');
  assert.equal(mode.value, 'private');
  assert.equal(mode.hasAttribute('readonly'), true);
  assert.equal(q(labelled('Access token'), 'input').getAttribute('type'), 'password');
  assert.equal(qa($('drawer'), 'input[type="url"]').length, 0);   // no API base URL: the page is served by its server
});

test('settings are kept in storage, read at the next load, and applied when storage is unavailable', async () => {
  const first = harness({ models: MODELS });
  await first.app.start();
  await flush();
  const beam = control(BEAM);
  beam.value = '6';
  change(beam);
  assert.equal(saved(first).beam_size, 6);
  assert.equal($('drawer').textContent.includes('Browser storage is unavailable'), true);
  assert.equal(labelled('Browser storage is unavailable'), undefined);   // it is a note, not a field
  assert.equal(qa($('drawer'), '.hint').find((n) => n.textContent.startsWith('Browser storage')).hidden, true);   // hidden while storage works

  const second = harness({ models: MODELS, storage: first.storage });   // the next page load
  await second.app.start();
  await flush();
  assert.equal(control(BEAM).value, '6');
  assert.equal(chipsText()[0], 'beam 6');

  const blocked = harness({ models: MODELS, storage: brokenStorage() });
  await blocked.app.start();
  await flush();
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3']);   // the defaults
  const note = qa($('drawer'), '.hint').find((n) => n.textContent.startsWith('Browser storage'));
  assert.equal(note.hidden, true);
  const input = control(BEAM);
  input.value = '5';
  change(input);
  assert.equal(chipsText()[0], 'beam 5');   // still applied for this page
  assert.equal(note.hidden, false);          // and the user is told it will not last
});

// ---- the sidebar ----------------------------------------------------------------------------------------------------------------------

test('the sidebar toggle flips body.sidebar-open and its aria-expanded; the scrim (a click on the body itself) and Esc close it', async () => {
  const h = harness();
  await h.app.start();
  const toggle = $('sidebar-toggle');
  const open = () => document.body.classList.contains('sidebar-open');
  assert.equal(open(), false);
  toggle.focus();
  toggle.click();
  assert.deepEqual([open(), toggle.getAttribute('aria-expanded')], [true, 'true']);
  $('sidebar').click();   // a click inside the panel is not the scrim
  assert.equal(open(), true);
  document.body.dispatchEvent(new ShimEvent('click', { bubbles: true }));   // target === the body: the scrim is its ::after
  assert.deepEqual([open(), toggle.getAttribute('aria-expanded')], [false, 'false']);
  toggle.click();
  toggle.click();
  assert.equal(open(), false);

  toggle.click();
  $('prompt').focus();
  press($('prompt'), 'Enter');   // any other key leaves it open
  press($('prompt'), 'a');
  assert.equal(open(), true);
  press($('prompt'), 'Escape');
  assert.deepEqual([open(), toggle.getAttribute('aria-expanded')], [false, 'false']);
  assert.equal(document.activeElement, toggle);   // back to the control that opened it
  toggle.click();
  $('settings').focus();
  $('settings').click();   // both open: one Esc closes both, and focus returns to the drawer's opener
  press(document.body, 'Escape');
  assert.deepEqual([open(), $('drawer').hidden], [false, true]);
  assert.equal(document.activeElement, $('settings'));
});

test('a session link closes the sidebar; New chat opens an empty chat, clears the composer and closes the sidebar', async () => {
  const h = harness({ sessions: [sess('s_b', 'B')], routes: { ...doneSession('s_b', 'B', 'm_b') } });
  await h.app.start();
  await flush();
  attach(imageFile());
  $('prompt').value = 'draft';
  $('sidebar-toggle').click();
  q($('session-list'), 'a').focus();
  q($('session-list'), 'a').click();   // following the link: the sidebar gets out of the way of the conversation
  assert.equal(document.body.classList.contains('sidebar-open'), false);
  assert.equal(document.activeElement, $('sidebar-toggle'));   // and focus goes back to what opened it, not into a panel that is now hidden

  $('sidebar-toggle').click();
  $('new-session').click();
  await flush();
  assert.equal(document.body.classList.contains('sidebar-open'), false);
  assert.equal(h.win.location.hash, '#/new');
  assert.equal($('conversation').children.length, 0);
  assert.equal($('prompt').value, '');
  assert.equal($('preview').hidden, true);
  assert.equal(qa($('session-list'), '[aria-current]').length, 0);
  assert.equal($('exports').hidden, true);
  $('new-session').click();   // already there: nothing to do, and nothing breaks
  await flush();
  assert.equal(h.win.location.hash, '#/new');
});

test('Delete asks once; a no changes nothing, a yes deletes, and deleting the open chat goes to the newest one left', async () => {
  const h = harness({
    sessions: [sess('s_c', 'C'), sess('s_b', 'B'), sess('s_a', 'A')],
    routes: {
      ...doneSession('s_c', 'C', 'm_c'), ...doneSession('s_b', 'B', 'm_b'), ...doneSession('s_a', 'A', 'm_a2'),
      'DELETE /v1/sessions/s_b': () => { h.sessions = h.sessions.filter((s) => s.id !== 's_b'); return new Response(null, { status: 204 }); },
      'DELETE /v1/sessions/s_c': () => { h.sessions = h.sessions.filter((s) => s.id !== 's_c'); return new Response(null, { status: 204 }); },
    },
  });
  await h.app.start();
  await flush();
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_c');
  const deleteOf = (id) => q(qa($('session-list'), 'li').find((li) => li.getAttribute('data-session') === id), 'button.session-delete');
  assert.equal(deleteOf('s_b').getAttribute('aria-label'), 'Delete chat: B');

  h.answer = false;
  deleteOf('s_b').click();
  await flush();
  assert.equal(h.confirms.length, 1);   // asked once
  assert.match(h.confirms[0], /Delete "B"/);
  assert.equal(h.fetch.to('DELETE', '/v1/sessions').length, 0);   // a no changes nothing

  h.answer = true;
  deleteOf('s_b').click();   // not the open chat: it just goes
  await flush();
  assert.equal(h.confirms.length, 2);
  assert.equal(h.fetch.to('DELETE', '/v1/sessions/s_b').length, 1);
  assert.deepEqual(qa($('session-list'), 'li').map((li) => li.getAttribute('data-session')), ['s_c', 's_a']);
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_c');   // still on C

  const cleared = h.api.cleared;
  deleteOf('s_c').click();   // the open chat: the view is reset and the newest one left is opened
  await flush(12);
  assert.equal(h.fetch.to('DELETE', '/v1/sessions/s_c').length, 1);
  assert.ok(h.api.cleared > cleared);
  assert.deepEqual(qa($('session-list'), 'li').map((li) => li.getAttribute('data-session')), ['s_a']);
  assert.equal(h.win.replaced.at(-1), '#/s/s_a');
});

test('deleting the last chat leaves an empty New chat, and a delete the server refuses says why', async () => {
  const h = harness({
    sessions: [sess('s_b', 'B')],
    routes: { ...doneSession('s_b', 'B', 'm_b'), 'DELETE /v1/sessions/s_b': () => refused(500, 'Could not delete.') },
  });
  await h.app.start();
  await flush();
  q($('session-list'), 'button.session-delete').click();
  await flush();
  assert.equal(q($('notice'), 'p').textContent, 'Could not delete.');
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_b');   // nothing was reset
  h.fetch.calls.length = 0;
  const handlers = { 'DELETE /v1/sessions/s_b': () => { h.sessions = []; return new Response(null, { status: 204 }); } };
  Object.assign(h, { routes: handlers });
  // a fresh harness that deletes: the session list is empty afterwards, so the page goes to an empty chat
  const g = harness({ sessions: [sess('s_b', 'B')], routes: { ...doneSession('s_b', 'B', 'm_b'), 'DELETE /v1/sessions/s_b': () => { g.sessions = []; return new Response(null, { status: 204 }); } } });
  await g.app.start();
  await flush();
  q($('session-list'), 'button.session-delete').click();
  await flush(12);
  assert.equal(g.win.replaced.at(-1), '#/new');
  assert.equal($('conversation').children.length, 0);
  assert.equal(qa($('session-list'), 'li').length, 0);
  assert.equal($('exports').hidden, true);
});

test('deleting the open chat clears it from the screen at once, before the list is asked for again', async () => {
  let release;
  let hold = false;
  const gate = new Promise((resolve) => { release = resolve; });
  const h = harness({
    sessions: [sess('s_b', 'B')],
    routes: {
      ...doneSession('s_b', 'B', 'm_b'),
      'GET /v1/sessions': async () => { if (hold) await gate; return { sessions: h.sessions, next_cursor: null }; },
      'DELETE /v1/sessions/s_b': () => { h.sessions = []; return new Response(null, { status: 204 }); },
    },
  });
  await h.app.start();
  await flush();
  assert.equal($('conversation').children.length, 2);
  assert.equal($('exports').hidden, false);
  hold = true;
  q($('session-list'), 'button.session-delete').click();
  await flush();
  assert.equal($('conversation').children.length, 0);   // a deleted chat is not left on screen while the list loads
  assert.equal($('exports').hidden, true);               // and cannot be exported
  release();
  await flush(12);
  assert.equal(h.win.replaced.at(-1), '#/new');
});

test('a delete that finds the chat already gone is not an error: the list is simply asked for again', async () => {
  const h = harness({
    sessions: [sess('s_b', 'B'), sess('s_a', 'A')],
    routes: { ...doneSession('s_b', 'B', 'm_b'), 'DELETE /v1/sessions/s_a': () => refused(404, 'Session not found.', 'not_found_error') },
  });
  await h.app.start();
  await flush();
  const listed = h.fetch.to('GET', '/v1/sessions?').length;
  h.sessions = [sess('s_b', 'B')];   // what the server says now: the other tab deleted it
  const row = qa($('session-list'), 'li').find((li) => li.getAttribute('data-session') === 's_a');
  q(row, 'button.session-delete').click();
  await flush();
  assert.equal($('notice').hidden, true);
  assert.equal(h.fetch.to('GET', '/v1/sessions?').length, listed + 1);
  assert.deepEqual(qa($('session-list'), 'li').map((li) => li.getAttribute('data-session')), ['s_b']);
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_b');
});

test('More pages the sidebar with the cursor, and the list that a refresh gives keeps the pages already loaded', async () => {
  const h = harness({
    sessions: [], routes: {
      'GET /v1/sessions': (url) => (url.includes('cursor=c1') ? { sessions: [sess('s_old', 'old', 1, 1)], next_cursor: null }
        : { sessions: [sess('s_b', 'B', 1, 3), sess('s_a', 'A', 1, 2)], next_cursor: 'c1' }),
      ...doneSession('s_b', 'B', 'm_b'),
    },
  });
  await h.app.start();
  await flush();
  assert.equal($('session-more').hidden, false);
  assert.equal($('session-more').textContent, 'More');
  $('session-more').click();
  await flush();
  assert.deepEqual(qa($('session-list'), 'li').map((li) => li.getAttribute('data-session')), ['s_b', 's_a', 's_old']);
  assert.equal(h.fetch.to('GET', '/v1/sessions?').at(-1).url, '/v1/sessions?limit=50&cursor=c1');
  assert.equal($('session-more').hidden, true);
  await h.app.refreshSessions();   // after a turn, say: three loaded, so ask for at least that many
  assert.match(h.fetch.to('GET', '/v1/sessions?').at(-1).url, /^\/v1\/sessions\?limit=50$/);
});

test('a refresh asks for as many sessions as are loaded, so a long list is not cut back to one page', async () => {
  const many = (top, n) => Array.from({ length: n }, (_, i) => sess(`s_${String(top - i).padStart(3, '0')}`, `chat ${top - i}`, 1, 3));
  const h = harness({
    hash: '#/new',
    routes: { 'GET /v1/sessions': (url) => (url.includes('cursor=') ? { sessions: many(10, 10), next_cursor: null } : { sessions: many(60, 50), next_cursor: 'c1' }) },
  });
  await h.app.start();
  await flush();
  assert.equal(qa($('session-list'), 'li').length, 50);
  $('session-more').click();
  await flush();
  assert.equal(qa($('session-list'), 'li').length, 60);
  await h.app.refreshSessions();
  assert.equal(h.fetch.to('GET', '/v1/sessions?').at(-1).url, '/v1/sessions?limit=60');   // sixty are on screen: ask for sixty
});

// ---- exports ---------------------------------------------------------------------------------------------------------------------------

test('Export fetches the export with the auth headers and saves it through a Blob link, then revokes the object URL', async () => {
  const made = [];
  const revoked = [];
  const urls = { createObjectURL: (blob) => { made.push(blob); return 'blob:fake/export'; }, revokeObjectURL: (u) => { revoked.push(u); } };
  const storage = memoryStorage({ [SETTINGS_KEY]: JSON.stringify({ token: 't0k' }) });
  const h = harness({
    storage, urls, sessions: [sess('s_a', 'A', 0)],
    routes: {
      ...EXISTING,
      'GET /v1/sessions/s_a/export': (url) => new Response(url.endsWith('md') ? '# Session' : '{"session":{}}', {
        status: 200, headers: { 'Content-Type': 'text/plain', 'Content-Disposition': `attachment; filename="session-s_a.${url.endsWith('md') ? 'md' : 'json'}"` } }),
    },
  });
  await h.app.start();
  await flush();
  assert.equal($('exports').hidden, false);
  assert.deepEqual(texts(qa($('exports'), 'button')), ['Export JSON', 'Export Markdown']);
  const clicked = [];
  document.body.addEventListener('click', (e) => { if (e.target.localName === 'a') clicked.push(e.target); });
  qa($('exports'), 'button')[0].click();
  await flush();
  const call = h.fetch.to('GET', '/v1/sessions/s_a/export')[0];
  assert.equal(call.url, '/v1/sessions/s_a/export?format=json');
  assert.equal(call.headers.Authorization, 'Bearer t0k');
  assert.ok(call.headers['X-Client-Id']);
  assert.equal(clicked.length, 1);
  assert.equal(clicked[0].getAttribute('download'), 'session-s_a.json');
  assert.equal(clicked[0].getAttribute('href'), 'blob:fake/export');
  assert.equal(await made[0].text(), '{"session":{}}');
  assert.equal(clicked[0].parentNode, null);   // the link was only there to click
  assert.deepEqual(revoked, []);
  h.timers.advance(1000);
  assert.deepEqual(revoked, ['blob:fake/export']);   // revoked afterwards

  qa($('exports'), 'button')[1].click();
  await flush();
  assert.equal(h.fetch.to('GET', '/v1/sessions/s_a/export').at(-1).url, '/v1/sessions/s_a/export?format=md');
  assert.equal(clicked[1].getAttribute('download'), 'session-s_a.md');
  assert.equal(await made[1].text(), '# Session');
  h.timers.advance(1000);
  assert.equal(revoked.length, 2);
});

test('an export the server refuses shows why and saves nothing', async () => {
  const made = [];
  const h = harness({
    urls: { createObjectURL: (b) => { made.push(b); return 'blob:x'; }, revokeObjectURL() {} }, sessions: [sess('s_a', 'A', 0)],
    routes: { ...EXISTING, 'GET /v1/sessions/s_a/export': () => refused(404, 'Session not found.', 'not_found_error') },
  });
  await h.app.start();
  await flush();
  qa($('exports'), 'button')[0].click();
  await flush();
  assert.equal(q($('notice'), 'p').textContent, 'Session not found.');
  assert.deepEqual(made, []);
});

// ---- the health strip ------------------------------------------------------------------------------------------------------------------

test('the strip says "server restarting…" while /healthz fails, backs off to 30 s after three failures, and is back at 10 s on the first answer', async () => {
  let failing = false;
  let down = 'status';
  const h = harness({
    routes: {
      'GET /healthz': () => {
        if (!failing) return { status: 'ok', mode: 'private', turns_in_flight: 1, queue_cap: 4 };
        if (down === 'network') throw new TypeError('Failed to fetch');
        return new Response('bad gateway', { status: 502 });
      },
    },
  });
  await h.app.start();
  await flush();
  const delays = () => h.timers.timeouts.map((t) => t.ms);
  assert.equal($('health').textContent, 'private · 1 of 4 turns in flight');
  assert.equal($('health').hasAttribute('data-state'), false);
  assert.deepEqual(delays(), [10000]);

  failing = true;
  const check = async (ms) => { h.timers.advance(ms); await flush(); };
  await check(10000);   // failure 1: a proxy answering 502
  assert.equal($('health').textContent, 'server restarting…');
  assert.equal($('health').getAttribute('data-state'), 'down');
  assert.deepEqual(delays(), [10000]);
  down = 'network';
  await check(10000);   // failure 2: the connection itself
  assert.deepEqual(delays(), [10000]);
  await check(10000);   // failure 3
  assert.equal($('health').textContent, 'server restarting…');
  assert.deepEqual(delays(), [30000]);   // slower from here on
  await check(29999);
  assert.deepEqual(delays(), [30000]);   // not before
  failing = false;
  await check(1);
  assert.equal($('health').textContent, 'private · 1 of 4 turns in flight');   // the first answer
  assert.equal($('health').hasAttribute('data-state'), false);
  assert.deepEqual(delays(), [10000]);   // and back to the quick pace
  assert.equal(h.fetch.to('GET', '/healthz').length, 5);
});

test('the health text is written only when it changes, and a mode in it fills the badge', async () => {
  let load = 0;
  const h = harness({ routes: { 'GET /healthz': () => ({ status: 'ok', mode: 'public', turns_in_flight: load, queue_cap: 4 }) } });
  await h.app.start();
  await flush();
  assert.equal($('mode-badge').textContent, 'private');   // /v1/models said private first; healthz carries the mode too
  const strip = $('health');
  let writes = 0;
  let value = strip.textContent;
  Object.defineProperty(strip, 'textContent', { get: () => value, set: (v) => { value = v; writes += 1; } });
  h.timers.advance(10000);
  await flush();
  h.timers.advance(10000);
  await flush();
  assert.equal(writes, 0);   // the same answer twice: the live region is not poked
  load = 2;
  h.timers.advance(10000);
  await flush();
  assert.equal(writes, 1);
  assert.equal(strip.textContent, 'public · 2 of 4 turns in flight');
  assert.equal($('mode-badge').textContent, 'public');
});

// ---- scroll, focus, copy ---------------------------------------------------------------------------------------------------------------

test('the page follows a streaming turn only while the reader is near the bottom of #conversation', async () => {
  const h = await ready();
  const conversation = $('conversation');
  Object.assign(conversation, { clientHeight: 400, scrollHeight: 2000, scrollTop: 1600 });
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  assert.equal(conversation.scrollTop, 2000);   // a turn the user just sent is shown
  conversation.scrollTop = 1550;   // 50 px from the end: still at the bottom
  run.channel.push(events[0]);
  await flush();
  conversation.scrollTop = 1550;
  nextFrame();
  assert.equal(conversation.scrollTop, 2000);   // followed
  conversation.scrollTop = 500;   // scrolled up to read something
  run.channel.push(events[1]);
  await flush();
  nextFrame();
  assert.equal(conversation.scrollTop, 500);   // not yanked back
  run.channel.push(...events.slice(2));
  run.channel.end();
  await turn;
  assert.equal(conversation.scrollTop, 500);   // not even by the last card
});

test('where the document scrolls (a short viewport) the newest card is scrolled into view, and only for a reader near the end', async () => {
  const proto = Object.getPrototypeOf(document.createElement('div'));
  const scrolled = [];
  proto.scrollIntoView = function scrollIntoView(options) { scrolled.push([this.getAttribute('data-message-id') ?? this.getAttribute('class'), options]); };
  try {
    const h = await ready();
    h.win.short = true;
    document.documentElement = { scrollHeight: 3000, scrollTop: 0, clientHeight: 800 };
    $('conversation').scrollTop = 0;
    const events = fullTurn();
    const turn = h.app.send();
    await flush();
    const run = h.api.streams[0];
    run.accept('m_a');
    await flush();
    assert.deepEqual(scrolled.at(-1), ['m_a', { block: 'end' }]);   // the card the user just asked for
    assert.equal($('conversation').scrollTop, 0);                    // #conversation is not the scroller here
    const before = scrolled.length;
    h.win.scrollY = 100;   // 3000 - 100 - 800 is far from the end: the reader is up the page
    run.channel.push(events[0]);
    await flush();
    nextFrame();
    assert.equal(scrolled.length, before);
    h.win.scrollY = 2150;  // 50 px from the end
    run.channel.push(events[1]);
    await flush();
    nextFrame();
    assert.equal(scrolled.length, before + 1);
    assert.deepEqual(scrolled.at(-1), ['m_a', { block: 'end' }]);
    run.channel.push(...events.slice(2));
    run.channel.end();
    await turn;
  } finally {
    delete proto.scrollIntoView;
    delete document.documentElement;
  }
});

test('a card rebuilt mid-stream keeps focus on the same control; focus in another card or outside the cards is left alone', async () => {
  const h = harness({ sessions: [sess('s_b', 'B', 1)], routes: { ...doneSession('s_b', 'B', 'm_b') } });
  await h.app.start();
  await flush();
  attach(imageFile());
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_n');
  await flush();
  const events = fullTurn('m_n');
  run.channel.push(...events.slice(0, 5));   // up to the end of encode: its stage has a detail, so a button
  await flush();
  nextFrame();
  const cards = () => qa($('conversation'), 'article.card');
  const [settled, live] = cards();
  assert.equal(live.getAttribute('data-message-id'), 'm_n');
  const encode = stageToggle(live, 'encode');
  encode.focus();
  assert.equal(document.activeElement, encode);
  run.channel.push(events[5]);
  await flush();
  nextFrame();
  const rebuilt = cards()[1];
  assert.notEqual(rebuilt, live);   // the whole card was replaced
  assert.equal(document.activeElement, stageToggle(rebuilt, 'encode'));   // and focus is on the same control of the new one

  run.channel.push(...events.slice(6, 9));   // generate runs and the first snapshot arrives: the live card has a Show raw of its own now
  await flush();
  nextFrame();
  assert.ok(buttonOf(cards()[1], 'Show raw'));
  const raw = buttonOf(settled, 'Show raw');
  raw.focus();
  run.channel.push(events[9]);   // the report closes: the live card is rebuilt again, and gains its Copy
  await flush();
  nextFrame();
  assert.ok(buttonOf(cards()[1], 'Copy'));
  assert.equal(document.activeElement, raw);   // a control of another card keeps its focus, and is not taken by the live card's
  $('prompt').focus();
  run.channel.push(events[10]);
  await flush();
  nextFrame();
  assert.equal(document.activeElement, $('prompt'));
  run.channel.push(...events.slice(11));
  run.channel.end();
  await turn;
});

test('Copy in a card writes the report to the clipboard; with no clipboard it says "Copy failed"', async (t) => {
  const timers = [];
  const realSetTimeout = globalThis.setTimeout;
  globalThis.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return timers.length; };
  t.after(() => { globalThis.setTimeout = realSetTimeout; });
  const h = harness({ sessions: [sess('s_b', 'B', 1)], routes: { ...doneSession('s_b', 'B', 'm_b') } });
  await h.app.start();
  await flush();
  buttonOf(cardOf(), 'Copy').click();
  await flush();
  assert.deepEqual(h.win.copied, ['Findings: clear.']);
  assert.equal(buttonOf(cardOf(), 'Copy').getAttribute('aria-label'), 'Copy report, turn 1');
  h.win.navigator = {};
  buttonOf(cardOf(), 'Copy').click();
  await flush();
  assert.equal(buttonOf(cardOf(), 'Copy failed').textContent, 'Copy failed');
  const link = buttonOf(cardOf(), 'model details');
  link.click();   // the provenance link opens the drawer at the models
  assert.equal($('drawer').hidden, false);
  assert.equal(document.activeElement, $('drawer-close'));
});

// ---- the composer (minimal) -----------------------------------------------------------------------------------------------------------------

test('the image well opens the picker from a click, Enter and Space; a chosen file is previewed, replaceable and removable', async () => {
  const made = [];
  const revoked = [];
  const urls = { createObjectURL: (blob) => { made.push(blob); return `blob:fake/${made.length}`; }, revokeObjectURL: (u) => { revoked.push(u); } };
  const h = harness({ urls });
  await h.app.start();
  await flush();
  let picks = 0;
  $('file').addEventListener('click', () => { picks += 1; });
  $('image-well').click();
  assert.equal(picks, 1);
  const enter = press($('image-well'), 'Enter');
  const space = press($('image-well'), ' ');
  assert.deepEqual([picks, enter.defaultPrevented, space.defaultPrevented], [3, true, true]);   // Space must not scroll the page
  press($('image-well'), 'a');
  press($('image-well'), 'Tab');
  assert.equal(picks, 3);

  attach(imageFile('x-ray.png', 'image/png', 51200));
  assert.equal($('preview').hidden, false);
  assert.equal(q($('preview'), 'img').getAttribute('src'), 'blob:fake/1');
  assert.equal(q($('preview'), 'span').textContent, 'x-ray.png · 50 KB');
  assert.equal(buttonOf($('preview'), 'Remove').getAttribute('aria-label'), 'Remove attached image');
  assert.equal($('file').value, '');   // so choosing the same file again fires change again
  attach(imageFile('second.jpg', 'image/jpeg', 100));
  assert.equal(qa($('preview'), 'img').length, 1);   // replaced, not added
  assert.equal(q($('preview'), 'span').textContent, 'second.jpg · 1 KB');
  assert.deepEqual(revoked, ['blob:fake/1']);   // the first preview's object URL went with it
  buttonOf($('preview'), 'Remove').click();
  assert.equal($('preview').hidden, true);
  assert.equal($('preview').children.length, 0);
  assert.deepEqual(revoked, ['blob:fake/1', 'blob:fake/2']);
  $('file').files = [];
  change($('file'));   // a cancelled picker: nothing
  assert.equal($('preview').hidden, true);
});

test('a file that cannot work is refused with a notice and not attached', async () => {
  const h = harness();
  await h.app.start();
  await flush();
  for (const [file, why] of [[imageFile('doc.pdf', 'application/pdf'), 'Choose a PNG, JPEG or WEBP image.'],
                             [{ name: 'big.png', type: 'image/png', size: 21 * 1024 * 1024 }, 'The image is over the 20 MB limit.']]) {
    attach(file);
    assert.equal(q($('notice'), 'p').textContent, why);
    assert.equal($('preview').hidden, true);
  }
  attach(imageFile('ok.png'));   // a good one clears the notice
  assert.equal($('notice').hidden, true);
  assert.equal($('preview').hidden, false);
});

test('a file dropped on the well or pasted into the note is attached; a file dropped elsewhere does not navigate the page; text is left alone', async () => {
  const h = harness();
  await h.app.start();
  await flush();
  const fire = (target, type, init) => { const e = new ShimEvent(type, { bubbles: true, cancelable: true, ...init }); target.dispatchEvent(e); return e; };
  const files = { types: ['Files'], files: [imageFile('dropped.png')] };
  const over = fire($('image-well'), 'dragover', { dataTransfer: files });
  assert.equal(over.defaultPrevented, true);   // without it the browser would not let the drop happen here
  assert.equal($('image-well').classList.contains('dragging'), true);
  fire($('image-well'), 'dragleave', {});
  assert.equal($('image-well').classList.contains('dragging'), false);
  fire($('image-well'), 'dragover', { dataTransfer: files });
  const dropped = fire($('image-well'), 'drop', { dataTransfer: files });
  assert.equal(dropped.defaultPrevented, true);
  assert.equal($('image-well').classList.contains('dragging'), false);
  assert.match(q($('preview'), 'span').textContent, /^dropped\.png/);

  const webImage = { types: ['text/uri-list', 'text/html'], files: [] };   // an image dragged from another web page: no file, but a link the browser would follow
  assert.equal(fire($('image-well'), 'dragover', { dataTransfer: webImage }).defaultPrevented, true);
  assert.equal(fire($('image-well'), 'drop', { dataTransfer: webImage }).defaultPrevented, true);   // dropped on the well it opens nothing
  assert.match(q($('preview'), 'span').textContent, /^dropped\.png/);   // and attaches nothing
  const elsewhere = fire($('conversation'), 'drop', { dataTransfer: { types: ['Files'], files: [imageFile('stray.png')] } });
  assert.equal(elsewhere.defaultPrevented, true);   // the page stays on the chat instead of opening the file
  assert.match(q($('preview'), 'span').textContent, /^dropped\.png/);   // and only the well attaches
  assert.equal(fire($('prompt'), 'drop', { dataTransfer: { types: ['text/plain'], files: [] } }).defaultPrevented, false);   // text may be dropped into the note
  assert.equal(fire($('prompt'), 'dragover', { dataTransfer: { types: ['text/plain'], files: [] } }).defaultPrevented, false);

  buttonOf($('preview'), 'Remove').click();
  const pasted = fire($('prompt'), 'paste', { clipboardData: { files: [imageFile('shot.png')] } });
  assert.equal(pasted.defaultPrevented, true);
  assert.match(q($('preview'), 'span').textContent, /^shot\.png/);
  buttonOf($('preview'), 'Remove').click();
  assert.equal(fire($('prompt'), 'paste', { clipboardData: { files: [] } }).defaultPrevented, false);   // text pastes as text
  assert.equal(fire($('prompt'), 'paste', { clipboardData: { files: [imageFile('x.gif', 'image/gif')] } }).defaultPrevented, false);
  assert.equal(fire($('prompt'), 'paste', {}).defaultPrevented, false);
  assert.equal($('preview').hidden, true);
});

test('the page makes no request to any other origin and sets no style: every URL is a path of this server', async () => {
  const h = harness({ sessions: [sess('s_b', 'B', 1)], routes: { ...doneSession('s_b', 'B', 'm_b'), 'GET /v1/sessions/s_b/export': new Response('{}', { status: 200 }) } });
  await h.app.start();
  await flush();
  attach(imageFile());
  const turn = h.app.send();
  await flush();
  h.api.streams[0].accept('m_a');
  await flush();
  h.api.streams[0].channel.push(...fullTurn());
  h.api.streams[0].channel.end();
  await turn;
  for (const call of h.fetch.calls) assert.match(call.url, /^\/(?:v1|healthz)/, call.url);   // same origin, always a path
  assert.equal(qa(document.body, '[style]').length, 0);
});

// ---- the module's own start-up ------------------------------------------------------------------------------------------------------------

test('importing app.js starts nothing without a #composer on the page, and starts the app on a page that has one', async (t) => {
  const asked = [];
  const realFetch = globalThis.fetch;
  const realSetTimeout = globalThis.setTimeout;
  const realSetInterval = globalThis.setInterval;
  globalThis.fetch = async (url) => { asked.push(String(url)); return new Response(JSON.stringify({ sessions: [], next_cursor: null, models: [] }), { status: 200 }); };
  globalThis.setTimeout = () => 0;   // the health loop would keep this process alive for ever
  globalThis.setInterval = () => 0;
  const storage = memoryStorage();   // the page's own storage (and not node's, which warns when it is touched without a file)
  const realStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', { value: storage, configurable: true, writable: true });
  t.after(() => {
    Object.assign(globalThis, { fetch: realFetch, setTimeout: realSetTimeout, setInterval: realSetInterval });
    if (realStorage) Object.defineProperty(globalThis, 'localStorage', realStorage); else delete globalThis.localStorage;
    installDom();
  });

  const logged = [];
  const realError = console.error;
  console.error = (...args) => logged.push(args);
  t.after(() => { console.error = realError; });
  installDom();   // a page with no composer: a test importing the helpers is this
  await import('../../app/static/app.js?bare');
  await flush();
  assert.deepEqual(asked, []);
  assert.deepEqual(logged, []);   // it did not start and fail either: starting on a page without the shell would throw
  console.error = realError;

  installDom();
  buildPage();
  await import('../../app/static/app.js?started');
  await flush(12);
  assert.ok(asked.includes('/v1/models'), `the app started and asked for the models: ${asked}`);
  assert.ok(asked.some((u) => u.startsWith('/v1/sessions?')));
  assert.ok(asked.includes('/healthz'));
  assert.ok($('status'), 'it built its status region');
  assert.equal($('drawer').children.length > 0, true);   // and the drawer
  assert.equal($('image-well').getAttribute('role'), 'button');
  assert.match(storage.data.get(CLIENT_KEY), /^[\x21-\x7e]{1,128}$/);   // it read the page's storage, and kept its client id there
});

// ---- keeping the keyboard's place when the page rebuilds what it was on -----------------------------------------------------------------------

test('a rebuilt session list keeps focus on the same link or delete button, and the last More hands focus to its first new row', async () => {
  const h = harness({
    routes: {
      'GET /v1/sessions': (url) => (url.includes('cursor=c1') ? { sessions: [sess('s_z', 'z', 1, 1), sess('s_y', 'y', 1, 1)], next_cursor: null }
        : { sessions: [sess('s_b', 'B', 1, 3), sess('s_a', 'A', 1, 2)], next_cursor: 'c1' }),
      ...doneSession('s_b', 'B', 'm_b'), ...doneSession('s_a', 'A', 'm_a2'),
    },
  });
  await h.app.start();
  await flush();
  const rowOf = (id) => qa($('session-list'), 'li').find((li) => li.getAttribute('data-session') === id);
  const link = rowOf('s_a').querySelector('a');
  link.focus();
  await h.app.refreshSessions();
  const again = rowOf('s_a').querySelector('a');
  assert.notEqual(again, link);   // the list was rebuilt
  assert.equal(document.activeElement, again);
  const del = rowOf('s_b').querySelector('button');
  del.focus();
  await h.app.refreshSessions();
  assert.equal(document.activeElement, rowOf('s_b').querySelector('button'));
  link.remove();
  $('prompt').focus();
  await h.app.refreshSessions();
  assert.equal(document.activeElement, $('prompt'));   // focus elsewhere is left alone

  $('session-more').focus();
  assert.equal(document.activeElement, $('session-more'));
  $('session-more').click();
  await flush();
  assert.equal($('session-more').hidden, true);   // the last page: the button is gone
  assert.equal(document.activeElement, qa($('session-list'), 'li a')[2]);   // so focus goes to the first row it added
  assert.equal(qa($('session-list'), 'li').length, 4);
});

test('dismissing a notice or removing the attached image does not leave focus on a control that has gone', async () => {
  const h = harness();
  await h.app.start();
  await flush();
  await h.app.send();   // no image: the notice
  const dismiss = buttonOf($('notice'), '✕');
  dismiss.focus();
  dismiss.click();
  assert.equal($('notice').hidden, true);
  assert.equal(document.activeElement, $('prompt'));

  attach(imageFile());
  const remove = buttonOf($('preview'), 'Remove');
  remove.focus();
  remove.click();
  assert.equal($('preview').hidden, true);
  assert.equal(document.activeElement, $('image-well'));   // the control that brought the image in
  $('prompt').focus();
  attach(imageFile());
  buttonOf($('preview'), 'Remove').click();
  assert.equal(document.activeElement, $('prompt'));   // focus that was elsewhere stays there
});

// ---- fix round 1/5 ---------------------------------------------------------------------------------------------------------------------
// Written before the changes they cover, and run red first (the two Important defects are missing branches, which a mutation of the
// existing code cannot find).

const deferred = () => {
  let resolve;
  const promise = new Promise((done) => { resolve = done; });
  return { promise, resolve };
};
const captureErrors = (t) => {
  const logged = [];
  const real = console.error;
  console.error = (...args) => logged.push(args);
  t.after(() => { console.error = real; });
  return logged;
};
const GENERIC = 'Something went wrong — see the console';

// I1: the watchdog starts when the server has the turn, not when Send is pressed ------------------------------------------------------------

test('a slow upload is not silence: nothing is dropped while the server has not accepted the turn, and the 3 s clock starts at the accept', async () => {
  const h = await ready();
  const events = fullTurn();
  $('prompt').value = 'beam 5';
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  h.timers.advance(10000);   // the POST has been going for 10 s: a big PNG over a tunnel, and the server decoding it
  await flush();
  assert.deepEqual(h.api.order, []);                   // not aborted
  assert.equal($('notice').hidden, true);              // and nothing said
  assert.equal($('send').disabled, true);              // the turn is simply pending
  assert.equal($('stop').disabled, true);
  assert.equal(h.timers.intervals, 0);                 // no stall check: there is nothing to be silent about yet
  assert.equal(qa($('conversation'), '.turn.user').length, 1);   // and the user turn stays
  assert.equal($('prompt').value, 'beam 5');           // the composer keeps its content while the upload is on its way

  run.accept('m_a');
  await flush();
  assert.equal($('prompt').value, '');                 // and spends it when the server has the turn
  assert.equal(h.timers.intervals, 1);                 // armed now
  run.channel.push(events[0]);                         // message_start, 10 s after Send
  await flush();
  h.timers.advance(2900);
  await flush();
  assert.deepEqual(h.api.order, []);                   // 2.9 s after the accept: still the stream
  h.timers.advance(100);
  await flush();
  assert.deepEqual(h.api.order, ['abort']);            // 3 s after it: handed to polling, as before
  assert.equal(h.api.polls.length, 1);
  assert.equal(h.api.polls[0].opts.after, 1);
  h.api.polls[0].channel.push(...events.slice(1));
  h.api.polls[0].channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal($('send').disabled, false);
});

test('an upload that is slow and then refused still says why, and a refusal after 10 s is a refusal like any other', async () => {
  const h = await ready();
  $('prompt').value = 'beam 99';
  const turn = h.app.send();
  await flush();
  h.timers.advance(10000);
  await flush();
  assert.equal($('notice').hidden, true);
  h.api.streams[0].fail(refusal(422, 'Invalid options: beam_size: Input should be less than or equal to 8'));
  await turn;
  assert.equal(q($('notice'), 'p').textContent, 'Invalid options: beam_size: Input should be less than or equal to 8');
  assert.equal($('conversation').children.length, 0);
  assert.equal($('prompt').value, 'beam 99');
  assert.equal($('send').disabled, false);
});

test('a request that was aborted before the server gave an id says it was cancelled, never the browser\'s own words', async () => {
  assert.equal(errorMessage(abortError()), 'The request was cancelled.');
  assert.equal(errorMessage(new DOMException('signal is aborted without reason', 'AbortError')), 'The request was cancelled.');
  const h = await ready();
  h.api.refuse = new DOMException('signal is aborted without reason', 'AbortError');
  await h.app.send();
  assert.equal(q($('notice'), 'p').textContent, 'The request was cancelled.');
  assert.doesNotMatch($('notice').textContent, /aborted/);
  assert.equal($('send').disabled, false);
  assert.equal($('conversation').children.length, 0);
});

test('a stream that closes before the server ever accepted the turn does not leave the page waiting for it', async () => {
  const h = await ready();
  h.api.silent = true;
  $('prompt').value = 'beam 4';
  const turn = h.app.send();
  await flush();
  h.api.streams[0].channel.end();   // a clean end: no header, no event
  await turn;
  assert.match(q($('notice'), 'p').textContent, /closed the connection before it accepted the turn/);
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
  assert.equal($('conversation').children.length, 0);   // the user turn is taken back
  assert.equal($('prompt').value, 'beam 4');            // and the composer is as it was
  nextFrame();   // aria-busy comes off a frame after the page stops being busy
  assert.equal($('conversation').hasAttribute('aria-busy'), false);
});

// I2: a resumed queued turn keeps what the user sent ---------------------------------------------------------------------------------------

test('a chat opened with a queued turn (no events yet) keeps the user\'s text and file name when its message_start arrives by polling', async () => {
  const h = harness({
    sessions: [sess('s_a', 'queued', 1)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'queued', 1), messages: [userMsg('u_a', 'beam 5 please', 'chest.png'), botMsg('m_a', 'running', START_DATA.options)] },
      'GET /v1/messages/m_a': { ...botMsg('m_a', 'running'), events: [] },
    },
  });
  await h.app.start();
  await flush();
  const shown = () => q($('conversation'), '.turn.user');
  assert.equal(q(shown(), '.user-text').textContent, 'beam 5 please');
  assert.equal(q(shown(), '.chip').textContent, 'chest.png');
  assert.equal(h.api.polls.length, 1);
  assert.equal(h.api.polls[0].opts.after, 0);
  resetEvents();
  h.api.polls[0].channel.push(startEv());   // the worker has started the turn
  await flush();
  nextFrame();
  assert.equal(q(shown(), '.user-text')?.textContent, 'beam 5 please');   // the bubble was rebuilt with the options the server resolved ...
  assert.equal(q(shown(), '.chip')?.textContent, 'chest.png');             // ... and kept what the user sent
  assert.deepEqual(texts(qa(shown(), '.options .chip')), ['beam 5', '100 tok', 'cached', 'k 4/3']);
  assert.equal(shown().getAttribute('aria-label'), 'Your message, turn 1');
});

// M1: Stop always settles the card, and a failed poll does not strand the page ------------------------------------------------------------

async function strandedTurn() {   // a turn whose poll failed for good: the Retry notice is up and nothing follows the turn
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 7));
  await flush();
  h.timers.advance(3000);   // 3 s of silence: polling
  await flush();
  h.api.polls[0].channel.fail(refusal(404, 'Message not found.', 'not_found_error'));
  await turn;
  await flush();
  return { h, events };
}

test('Stop after a poll that failed for good follows the turn again from its last seq, so the card settles', async () => {
  const { h } = await strandedTurn();
  assert.equal(q($('notice'), 'p').textContent, 'Message not found.');
  assert.equal(buttonOf($('notice'), 'Retry').hidden, false);
  assert.equal($('send').disabled, true);
  $('stop').click();
  await flush();
  assert.deepEqual(h.api.cancels.map((c) => c.messageId), ['m_a']);   // the server is asked to stop it
  assert.equal(h.api.polls.length, 2);                                  // and, with nothing following the turn, the page follows it again
  assert.equal(h.api.polls[1].opts.after, 7);
  assert.equal($('notice').hidden, true);                               // the Retry notice has done its job
  h.api.polls[1].channel.push(stopEv('aborted'));
  h.api.polls[1].channel.end();
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'aborted');
  assert.equal(q(cardOf(), '.note.stopped').textContent, 'Turn stopped');
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
  assert.equal($('stop').textContent, 'Stop');
  assert.equal($('stop').disabled, false);
});

test('dismissing the Retry notice does not leave Send disabled for ever: Stop still follows the turn to its end', async () => {
  const { h } = await strandedTurn();
  buttonOf($('notice'), '✕').click();   // the notice is dismissed, and with it the Retry
  assert.equal($('notice').hidden, true);
  assert.equal($('send').disabled, true);
  assert.equal($('stop').hidden, false);
  assert.equal($('stop').disabled, false);
  $('stop').click();
  await flush();
  assert.equal(h.api.polls.length, 2);
  h.api.polls[1].channel.push(stopEv('done', { report: 'Findings: finished meanwhile.', display_report: 'Findings: finished meanwhile.' }));   // it had ended on its own
  h.api.polls[1].channel.end();
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal($('send').disabled, false);
});

test('after a cancel and then a poll that fails, Stop is usable again, and so is the Retry', async () => {
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 4));
  await flush();
  $('stop').click();   // the cancel goes through and the stream is dropped ...
  await flush();
  assert.equal($('stop').textContent, 'Stopping…');
  h.api.polls[0].channel.fail(refusal(404, 'Message not found.', 'not_found_error'));   // ... and then the poll that should have brought the stop fails
  await turn;
  await flush();
  assert.equal($('stop').textContent, 'Stop');   // not "Stopping…" for ever
  assert.equal($('stop').disabled, false);
  assert.equal(buttonOf($('notice'), 'Retry').hidden, false);
  $('stop').click();
  await flush();
  assert.equal(h.api.cancels.length, 2);   // asked again: a cancel is idempotent on the server
  assert.equal(h.api.polls.length, 2);
  assert.equal(h.api.polls[1].opts.after, 4);
  h.api.polls[1].channel.push(stopEv('aborted'));
  h.api.polls[1].channel.end();
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'aborted');
  assert.equal($('send').disabled, false);
});

test('a cancel that fails after the turn has already ended shows nothing', async () => {
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 5));
  await flush();
  const gate = deferred();
  h.api.cancelGate = gate.promise;
  h.api.cancelError = refusal(404, 'Message not found.', 'not_found_error');
  $('stop').click();   // the cancel is on its way ...
  await flush();
  run.channel.push(...events.slice(5));   // ... and the turn ends on its own
  run.channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal($('send').disabled, false);
  gate.resolve();   // now the cancel fails: nothing it could say is true any more
  await flush();
  assert.equal($('notice').hidden, true);
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
  assert.equal($('stop').textContent, 'Stop');
});

// M2: a chat that cannot be opened can be opened again, and one bad turn does not blank it ----------------------------------------------------

test('a chat that failed to open can be opened again: by its link, and by the Retry in the notice', async () => {
  let fail = true;
  const ok = doneSession('s_a', 'A', 'm_a');
  const build = () => harness({
    hash: '#/s/s_a', sessions: [sess('s_a', 'A', 1)],
    routes: { ...ok, 'GET /v1/sessions/s_a': () => (fail ? refused(500, 'Could not read the session.') : ok['GET /v1/sessions/s_a']) },
  });
  let h = build();
  await h.app.start();
  await flush();
  assert.equal(q($('notice'), 'p').textContent, 'Could not read the session.');
  assert.equal(buttonOf($('notice'), 'Retry').hidden, false);
  assert.equal($('conversation').children.length, 0);
  assert.equal($('exports').hidden, true);   // no chat is open
  const other = q($('session-list'), 'a');
  assert.equal(h.fetch.to('GET', '/v1/sessions/s_a').length, 1);
  fail = false;
  other.click();   // the address is the same, so the browser fires no hashchange: the link itself must try again
  await flush();
  assert.equal(h.fetch.to('GET', '/v1/sessions/s_a').length, 2);
  assert.equal($('conversation').children.length, 2);
  assert.equal($('notice').hidden, true);
  assert.equal($('exports').hidden, false);

  fail = true;
  h = build();
  await h.app.start();
  await flush();
  fail = false;
  buttonOf($('notice'), 'Retry').click();
  await flush();
  assert.equal(h.fetch.to('GET', '/v1/sessions/s_a').length, 2);
  assert.equal($('conversation').children.length, 2);
  assert.equal($('notice').hidden, true);
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_a');
});

test('one turn that cannot be loaded does not blank the chat: its card says so, with a Retry that loads it', async () => {
  let failSecond = true;
  const first = fullTurn('m_1');
  const second = fullTurn('m_2');
  const h = harness({
    sessions: [sess('s_a', 'two turns', 2)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'two turns', 2), messages: [userMsg('u_1', 'first', 'a.png'), botMsg('m_1', 'done', START_DATA.options),
                                                                           userMsg('u_2', 'second', 'b.png'), botMsg('m_2', 'done', START_DATA.options)] },
      'GET /v1/messages/m_1': () => ({ ...botMsg('m_1', 'done'), events: rows(first) }),
      'GET /v1/messages/m_2': () => (failSecond ? refused(500, 'Could not read the turn.') : { ...botMsg('m_2', 'done'), events: rows(second) }),
    },
  });
  await h.app.start();
  await flush();
  const cards = () => qa($('conversation'), 'article.card');
  assert.equal($('conversation').children.length, 4);
  assert.equal(cards()[0].getAttribute('data-message-id'), 'm_1');
  assert.equal(q($('conversation').children[2], '.user-text').textContent, 'second');
  assert.match(cards()[1].textContent, /Couldn.t load this turn/);
  assert.equal(cards()[1].getAttribute('aria-label'), 'Assistant report, turn 2');
  assert.equal($('notice').hidden, true);   // the failure is on its card
  assert.equal($('send').disabled, false);
  buttonOf(cards()[1], 'Retry').click();   // still failing: it says so again and stays
  await flush();
  assert.equal(h.fetch.to('GET', '/v1/messages/m_2').length, 2);
  assert.match(cards()[1].textContent, /Couldn.t load this turn/);
  failSecond = false;
  buttonOf(cards()[1], 'Retry').focus();
  buttonOf(cards()[1], 'Retry').click();
  await flush();
  assert.equal(cards().length, 2);
  assert.ok(cards()[1].contains(document.activeElement) && document.activeElement !== cards()[1], 'focus went into the card that replaced the Retry');
  assert.equal(cards()[1].getAttribute('data-message-id'), 'm_2');
  assert.equal(cards()[1].getAttribute('data-status'), 'done');
  assert.equal(buttonOf($('conversation'), 'Retry'), undefined);
  assert.equal($('send').disabled, false);
});

test('a last turn that could not be loaded and is still running is followed once it loads', async () => {
  resetEvents();
  const partial = [startEv({ message_id: 'm_2' }), stageStartEv('preprocess', 0)];
  let failing = true;
  const h = harness({
    sessions: [sess('s_a', 'running', 1)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'running', 1), messages: [userMsg('u_2', 'second', 'b.png'), botMsg('m_2', 'running', START_DATA.options)] },
      'GET /v1/messages/m_2': () => (failing ? refused(500, 'Could not read the turn.') : { ...botMsg('m_2', 'running'), events: rows(partial) }),
    },
  });
  await h.app.start();
  await flush();
  assert.equal(h.api.polls.length, 0);   // nothing is known of it yet
  assert.equal($('send').disabled, false);
  failing = false;
  buttonOf(cardOf(), 'Retry').click();
  await flush();
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_2');
  assert.equal(h.api.polls.length, 1);
  assert.equal(h.api.polls[0].opts.after, 2);
  assert.equal($('send').disabled, true);
  h.api.polls[0].channel.push(stopEv('done', { message_id: 'm_2' }));
  h.api.polls[0].channel.end();
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal($('send').disabled, false);
});

// M3: More cannot append a page twice -------------------------------------------------------------------------------------------------------

test('More is off while its page loads, and a second click asks for nothing and appends nothing twice', async () => {
  const gate = deferred();
  const h = harness({
    hash: '#/new',
    routes: {
      'GET /v1/sessions': async (url) => {
        if (!url.includes('cursor=c1')) return { sessions: [sess('s_b', 'B', 1, 3), sess('s_a', 'A', 1, 2)], next_cursor: 'c1' };
        await gate.promise;
        return { sessions: [sess('s_z', 'z', 1, 1)], next_cursor: null };
      },
    },
  });
  await h.app.start();
  await flush();
  $('session-more').click();
  $('session-more').click();   // a double click
  assert.equal($('session-more').disabled, true);
  await flush();
  assert.equal(h.fetch.to('GET', '/v1/sessions?').filter((c) => c.url.includes('cursor=c1')).length, 1);
  gate.resolve();
  await flush();
  assert.deepEqual(qa($('session-list'), 'li').map((li) => li.getAttribute('data-session')), ['s_b', 's_a', 's_z']);
  assert.equal($('session-more').hidden, true);
  assert.equal($('session-more').disabled, false);   // and it works again for a later page
});

// M4: /healthz has a timeout ------------------------------------------------------------------------------------------------------------------

test('a /healthz that never answers is given up after 5 s: the strip says the server is restarting', async () => {
  const h = harness({
    routes: { 'GET /healthz': (url, init) => new Promise((_, reject) => init.signal?.addEventListener('abort', () => reject(abortError()), { once: true })) },
  });
  await h.app.start();
  await flush();
  assert.equal($('health').textContent, '');   // still waiting
  h.timers.advance(4999);
  await flush();
  assert.equal($('health').textContent, '');
  h.timers.advance(1);
  await flush();
  assert.equal($('health').textContent, 'server restarting…');
  assert.equal($('health').getAttribute('data-state'), 'down');
  assert.deepEqual(h.timers.timeouts.map((t) => t.ms), [10000]);   // and the next check is scheduled as for any failure
});

// M5: unexpected exceptions ---------------------------------------------------------------------------------------------------------------------

test('errorMessage says the server cannot be reached only for a network failure, and something else went wrong otherwise', () => {
  const network = Object.assign(new TypeError('whatever the browser says'), { network: true });
  assert.equal(errorMessage(network), 'Cannot reach the server. Check the connection and try again.');
  assert.equal(errorMessage(new TypeError('Failed to fetch')), 'Cannot reach the server. Check the connection and try again.');   // what the browsers say
  assert.equal(errorMessage(new TypeError('NetworkError when attempting to fetch resource.')), 'Cannot reach the server. Check the connection and try again.');
  assert.equal(errorMessage(new TypeError('Load failed')), 'Cannot reach the server. Check the connection and try again.');
  for (const words of ['fetch failed', 'network error', 'The network connection was lost.', 'The Internet connection appears to be offline.', 'Network request failed']) {
    assert.equal(errorMessage(new TypeError(words)), 'Cannot reach the server. Check the connection and try again.', words);   // node, Chrome's body, Safari twice, React Native
  }
  assert.equal(errorMessage(new TypeError("Cannot read properties of undefined (reading 'delta')")), GENERIC);   // a bug of ours, not the network
  assert.equal(errorMessage(new TypeError('x is not iterable')), GENERIC);
  assert.equal(errorMessage(new Error('boom')), 'boom');   // other errors still say what they say
});

test('an event that breaks the reducer or the render says so, is logged, and neither stops the turn nor sends it to polling', async (t) => {
  const logged = captureErrors(t);
  const h = await ready();
  const events = fullTurn();
  const bad = ev('content_block_delta', { index: 0 });   // no delta: the reducer throws a TypeError
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 4), bad, ...events.slice(4));
  run.channel.end();
  await turn;
  assert.equal(q($('notice'), 'p').textContent, GENERIC);
  assert.ok(logged.some(([e]) => e instanceof TypeError), 'it was logged');
  assert.equal(h.api.polls.length, 0);                                  // the stream carried on, and was not abandoned
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal($('send').disabled, false);
});

test('a render failure while the turn ends still frees the composer', async (t) => {
  const logged = captureErrors(t);
  const h = await ready();
  const proto = Object.getPrototypeOf(document.createElement('div'));
  const original = proto.replaceWith;
  let boom = false;
  proto.replaceWith = function replaceWith(...nodes) {
    if (boom && this.localName === 'article' && this.getAttribute('class') === 'card') throw new TypeError('render failed');
    return original.apply(this, nodes);
  };
  t.after(() => { delete proto.replaceWith; });
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 13));
  await flush();
  nextFrame();
  boom = true;
  run.channel.push(events[13]);   // message_stop: the final paint throws
  run.channel.end();
  await turn;
  assert.equal($('send').disabled, false);   // restored although the paint threw
  assert.equal($('stop').hidden, true);
  nextFrame();   // aria-busy comes off a frame after the page stops being busy
  assert.equal($('conversation').hasAttribute('aria-busy'), false);
  assert.equal(q($('notice'), 'p').textContent, GENERIC);
  assert.ok(logged.some(([e]) => e instanceof TypeError && e.message === 'render failed'));
});

test('something unexpected before the request leaves the composer as it was, and says so', async (t) => {
  const logged = captureErrors(t);
  let calls = 0;
  const urls = { createObjectURL: () => { calls += 1; if (calls > 1) throw new TypeError('no object URLs'); return 'blob:fake/1'; }, revokeObjectURL() {} };
  const h = harness({ urls, sessions: [sess('s_a', 'A', 0)], routes: EXISTING });
  await h.app.start();
  await flush();
  attach(imageFile());   // the first object URL: the preview
  $('prompt').value = 'beam 4';
  await h.app.send();    // the second one, for the user turn, throws
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
  assert.equal(q($('notice'), 'p').textContent, GENERIC);
  assert.equal($('conversation').children.length, 0);
  assert.equal($('prompt').value, 'beam 4');
  assert.equal($('preview').hidden, false);
  assert.equal(h.api.streams.length, 0);
  assert.ok(logged.some(([e]) => e instanceof TypeError && e.message === 'no object URLs'));
});

test('a chat the server answers with something the page cannot read is not reported as a network failure', async () => {
  const h = harness({ hash: '#/s/s_a', sessions: [sess('s_a', 'A', 1)], routes: { 'GET /v1/sessions/s_a': { id: 's_a' } } });   // no "messages"
  await h.app.start();
  await flush();
  assert.equal(q($('notice'), 'p').textContent, GENERIC);
  assert.doesNotMatch($('notice').textContent, /reach the server/);
  const down = harness({ hash: '#/new', routes: { 'GET /v1/models': () => { throw new TypeError('a message no browser uses'); } } });
  await down.app.start();
  await flush();
  assert.equal(q($('notice'), 'p').textContent, 'Cannot reach the server. Check the connection and try again.');   // a fetch that failed is the network, marked as such
});

// M6: accessibility -------------------------------------------------------------------------------------------------------------------------------

test('a chat is drawn into #conversation while it is aria-busy, so a screen reader does not read the whole history as it arrives', async () => {
  const h = harness({ sessions: [sess('s_b', 'B', 1)], routes: { ...doneSession('s_b', 'B', 'm_b') } });
  const conversation = $('conversation');
  const seen = [];
  const replaceChildren = conversation.replaceChildren.bind(conversation);
  conversation.replaceChildren = (...nodes) => { seen.push([nodes.length, conversation.getAttribute('aria-busy')]); return replaceChildren(...nodes); };
  await h.app.start();
  await flush();
  assert.deepEqual(seen.filter(([n]) => n > 0), [[2, 'true']]);   // the user turn and the card went in while it was busy
  nextFrame();   // aria-busy comes off a frame after the page stops being busy
  assert.equal(conversation.hasAttribute('aria-busy'), false);    // and it is not left busy
  assert.equal(conversation.children.length, 2);
});

test('after a Delete, focus goes to the chat that is now open, or to New chat when none is left', async () => {
  const h = harness({
    sessions: [sess('s_c', 'C'), sess('s_b', 'B'), sess('s_a', 'A')],
    routes: {
      ...doneSession('s_c', 'C', 'm_c'), ...doneSession('s_b', 'B', 'm_b'), ...doneSession('s_a', 'A', 'm_a2'),
      'DELETE /v1/sessions/s_a': () => { h.sessions = h.sessions.filter((s) => s.id !== 's_a'); return new Response(null, { status: 204 }); },
      'DELETE /v1/sessions/s_c': () => { h.sessions = h.sessions.filter((s) => s.id !== 's_c'); return new Response(null, { status: 204 }); },
      'DELETE /v1/sessions/s_b': () => { h.sessions = h.sessions.filter((s) => s.id !== 's_b'); return new Response(null, { status: 204 }); },
    },
  });
  await h.app.start();
  await flush();
  const row = (id) => qa($('session-list'), 'li').find((li) => li.getAttribute('data-session') === id);
  const del = (id) => { const b = q(row(id), 'button.session-delete'); b.focus(); b.click(); };
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_c');
  del('s_a');   // not the open chat: focus goes to the one that is open
  await flush(12);
  assert.equal(document.activeElement, q(row('s_c'), 'a'));
  del('s_c');   // the open chat: B opens and its sidebar item gets focus
  await flush(12);
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_b');
  assert.equal(document.activeElement, q(row('s_b'), 'a'));
  del('s_b');   // the last one
  await flush(12);
  assert.equal($('conversation').children.length, 0);
  assert.equal(document.activeElement, $('new-session'));
});

// M7: polish -------------------------------------------------------------------------------------------------------------------------------------

test('the accept clears what was sent from the note and the file, and keeps what was typed or attached since', async () => {
  const h = await ready();
  const events = fullTurn();
  $('prompt').value = 'beam 5';
  const turn = h.app.send();
  await flush();   // the upload is on its way
  $('prompt').value = 'beam 5 and a second thought';   // typed meanwhile
  attach(imageFile('second.png'));                      // and another image chosen
  h.api.streams[0].accept('m_a');
  await flush();
  assert.equal($('prompt').value, ' and a second thought');   // only what was sent is gone
  assert.equal($('preview').hidden, false);
  assert.match(q($('preview'), 'span').textContent, /^second\.png/);   // and the new attachment is still there
  h.api.streams[0].channel.push(...events);
  h.api.streams[0].channel.end();
  await turn;

  attach(imageFile('third.png'));   // the plain case: nothing changed since Send
  $('prompt').value = 'beam 3';
  const again = h.app.send();
  await flush();
  h.api.streams[1].accept('m_b');
  await flush();
  assert.equal($('prompt').value, '');
  assert.equal($('preview').hidden, true);
  h.api.streams[1].channel.push(stopEv('done', { message_id: 'm_b' }));
  h.api.streams[1].channel.end();
  await again;

  attach(imageFile('fourth.png'));   // a note that was replaced, not added to, is the next turn's whole
  $('prompt').value = 'greedy';
  const third = h.app.send();
  await flush();
  $('prompt').value = 'tokens 30';
  h.api.streams[2].accept('m_c');
  await flush();
  assert.equal($('prompt').value, 'tokens 30');
  h.api.streams[2].channel.push(stopEv('done', { message_id: 'm_c' }));
  h.api.streams[2].channel.end();
  await third;
});

test('with no confirm() to ask, Delete does nothing rather than deleting unasked', async () => {
  const h = harness({
    noConfirm: true, sessions: [sess('s_b', 'B')],
    routes: { ...doneSession('s_b', 'B', 'm_b'), 'DELETE /v1/sessions/s_b': () => new Response(null, { status: 204 }) },
  });
  await h.app.start();
  await flush();
  q($('session-list'), 'button.session-delete').click();
  await flush();
  assert.equal(h.fetch.to('DELETE', '/v1/sessions').length, 0);
  assert.equal(qa($('session-list'), 'li').length, 1);
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_b');
});

test('a session id is never put into a selector: an odd one keeps focus through a rebuild and breaks nothing', async () => {
  const odd = 's"] , [x';
  const h = harness({ hash: '#/new', sessions: [sess(odd, 'odd one', 1), sess('s_ok', 'fine', 1)] });
  await h.app.start();
  await flush();
  const row = (id) => qa($('session-list'), 'li').find((li) => li.getAttribute('data-session') === id);
  q(row(odd), 'a').focus();
  await h.app.refreshSessions();
  assert.equal($('notice').hidden, true);
  assert.equal(document.activeElement, q(row(odd), 'a'));
  q(row(odd), 'button').focus();
  await h.app.refreshSessions();
  assert.equal(document.activeElement, q(row(odd), 'button'));
});

// hardening: the same changes, from the sides the first tests do not reach ---------------------------------------------------------------------

test('a chat that cannot be made says why and frees the composer, whether the server refuses or the network is down', async () => {
  let answer = refused(500, 'Could not create the session.');
  const h = harness({ hash: '#/new', routes: { 'POST /v1/sessions': () => { if (answer instanceof Error) throw answer; return answer; } } });
  await h.app.start();
  await flush();
  attach(imageFile());
  $('prompt').value = 'beam 4';
  await h.app.send();
  assert.equal(q($('notice'), 'p').textContent, 'Could not create the session.');
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
  assert.equal($('conversation').children.length, 0);
  assert.equal($('prompt').value, 'beam 4');
  assert.equal($('preview').hidden, false);
  assert.equal(h.api.streams.length, 0);
  answer = new TypeError('Failed to fetch');
  await h.app.send();
  assert.equal(q($('notice'), 'p').textContent, 'Cannot reach the server. Check the connection and try again.');
  assert.equal($('send').disabled, false);
  assert.equal(h.fetch.to('POST', '/v1/sessions').length, 2);   // nothing was made, so the next Send asks again
});

test('a cancel that fails because the network does says so, whatever the browser calls its TypeError', async () => {
  const h = await ready();
  const turn = h.app.send();
  await flush();
  h.api.streams[0].accept('m_a');
  await flush();
  h.api.cancelError = new TypeError('x is not a function (the browser\'s own words are not the test)');
  $('stop').click();
  await flush();
  assert.equal(q($('notice'), 'p').textContent, 'Cannot reach the server. Check the connection and try again.');
  assert.equal($('stop').textContent, 'Stop');
  assert.equal($('stop').disabled, false);
  h.api.streams[0].channel.push(stopEv('done'));
  h.api.streams[0].channel.end();
  await turn;
});

test('an export whose body fails to arrive is a network failure, and saves nothing', async () => {
  const made = [];
  const h = harness({
    urls: { createObjectURL: (b) => { made.push(b); return 'blob:x'; }, revokeObjectURL() {} }, sessions: [sess('s_a', 'A', 0)],
    routes: { ...EXISTING, 'GET /v1/sessions/s_a/export': () => new Response(new ReadableStream({ start(controller) { controller.error(new TypeError('terminated')); } })) },
  });
  await h.app.start();
  await flush();
  qa($('exports'), 'button')[0].click();
  await flush();
  assert.equal(q($('notice'), 'p').textContent, 'Cannot reach the server. Check the connection and try again.');
  assert.deepEqual(made, []);
});

test('a TypeError out of the stream or the poll is the network, and one out of the page\'s own handling of an event is not', async (t) => {
  const logged = captureErrors(t);
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 4));
  run.channel.end();
  await flush();                                    // the stream closed without an end: polling
  const poll = h.api.polls[0];
  poll.channel.push(ev('content_block_delta', { index: 0 }));   // an event that breaks the reducer, in the poll this time
  await flush();
  assert.equal(q($('notice'), 'p').textContent, GENERIC);       // a bug of the page's, said as one ...
  assert.equal(buttonOf($('notice'), 'Retry').hidden, true);    // ... with no Retry: the poll did not fail
  poll.channel.push(...events.slice(4));
  poll.channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');   // and the poll carried on to the end
  assert.ok(logged.some(([e]) => e instanceof TypeError));

  const second = harness({ sessions: [sess('s_a', 'A', 0)], routes: EXISTING });
  await second.app.start();
  await flush();
  attach(imageFile());
  const again = second.app.send();
  await flush();
  second.api.streams[0].accept('m_a');
  await flush();
  second.api.streams[0].channel.push(...fullTurn().slice(0, 4));
  second.api.streams[0].channel.end();
  await flush();
  second.api.polls[0].channel.fail(transportFailure(new TypeError('whatever the browser says')));   // the poll itself failed: the network
  await again;
  await flush();
  assert.equal(q($('notice'), 'p').textContent, 'Cannot reach the server. Check the connection and try again.');
  assert.equal(buttonOf($('notice'), 'Retry').hidden, false);
});

test('Retry on a card that could not be loaded asks once, however often it is clicked while the answer is awaited', async () => {
  const gate = deferred();
  let calls = 0;
  const h = harness({
    sessions: [sess('s_a', 'A', 1)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'A', 1), messages: [userMsg('u_1', 'first', 'a.png'), botMsg('m_1', 'done', START_DATA.options)] },
      'GET /v1/messages/m_1': async () => {
        calls += 1;
        if (calls === 1) return refused(500, 'Could not read the turn.');
        await gate.promise;
        return { ...botMsg('m_1', 'done'), events: rows(fullTurn('m_1')) };
      },
    },
  });
  await h.app.start();
  await flush();
  const retry = buttonOf(cardOf(), 'Retry');
  retry.click();
  retry.click();
  retry.click();
  await flush();
  assert.equal(calls, 2);   // the first load, and one retry
  gate.resolve();
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal(calls, 2);
});

test('a stale Retry, from a card of a chat the user has left, resumes nothing in the chat now on screen', async () => {
  resetEvents();
  const partial = [startEv({ message_id: 'm_1' }), stageStartEv('preprocess', 0)];
  let calls = 0;
  const h = harness({
    hash: '#/s/s_a', sessions: [sess('s_a', 'A', 1), sess('s_b', 'B', 1)],
    routes: {
      ...doneSession('s_b', 'B', 'm_b'),
      'GET /v1/sessions/s_a': { ...sess('s_a', 'A', 1), messages: [userMsg('u_1', 'first', 'a.png'), botMsg('m_1', 'running', START_DATA.options)] },
      'GET /v1/messages/m_1': () => { calls += 1; return calls === 1 ? refused(500, 'Could not read the turn.') : { ...botMsg('m_1', 'running'), events: rows(partial) }; },
    },
  });
  await h.app.start();
  await flush();
  const stale = buttonOf(cardOf(), 'Retry');
  h.win.location.hash = '#/s/s_b';   // the user leaves for another chat; this card goes with the old one
  await flush();
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');
  stale.click();   // a click on the old card's button, which would now find the turn running
  await flush();
  assert.equal(calls, 2);                                   // it did ask ...
  assert.equal(h.api.polls.length, 0);                      // ... and followed nothing
  assert.equal($('send').disabled, false);                  // so this chat's composer is its own
  assert.equal($('stop').hidden, true);
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');
  assert.equal($('conversation').children.length, 2);
});

test('Stop while a poll is already following the turn asks the server to cancel and starts no second poll', async () => {
  resetEvents();
  const partial = [startEv(), stageStartEv('preprocess', 0)];
  const h = harness({
    sessions: [sess('s_a', 'running', 1)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'running', 1), messages: [userMsg('u_a', '', 'chest.png'), botMsg('m_a', 'running', START_DATA.options)] },
      'GET /v1/messages/m_a': { ...botMsg('m_a', 'running'), events: rows(partial) },
    },
  });
  await h.app.start();
  await flush();
  assert.equal(h.api.polls.length, 1);
  $('stop').click();
  await flush();
  assert.deepEqual(h.api.cancels.map((c) => c.messageId), ['m_a']);
  assert.equal(h.api.polls.length, 1);   // the poll that is there brings the stop
  assert.equal($('stop').textContent, 'Stopping…');
  h.api.polls[0].channel.push(stopEv('aborted'));
  h.api.polls[0].channel.end();
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'aborted');
  assert.equal($('send').disabled, false);
});

test('something unexpected while the chat is made is not hidden: it is logged, said as ours, and the composer is freed', async (t) => {
  const logged = captureErrors(t);
  const h = harness({ hash: '#/new', routes: { 'POST /v1/sessions': { id: 's_new', title: '', mode: 'private', turns: 0, created_at: iso(3), updated_at: iso(3) } } });
  await h.app.start();
  await flush();
  h.win.history.replaceState = () => { throw new TypeError('the address cannot be set'); };
  attach(imageFile());
  await h.app.send();
  assert.equal(q($('notice'), 'p').textContent, GENERIC);
  assert.ok(logged.some(([e]) => e instanceof TypeError && e.message === 'the address cannot be set'));
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
  assert.equal(h.api.streams.length, 0);
  assert.equal($('prompt').value ?? '', '');
  assert.equal($('preview').hidden, false);   // the image is still attached, for the next try
});

test('a turn in the middle of a chat that loads late and is found running is shown as it is, and only the last turn is ever followed', async () => {
  resetEvents();
  const partial = [startEv({ message_id: 'm_1' }), stageStartEv('preprocess', 0)];
  const second = fullTurn('m_2');
  let calls = 0;
  const h = harness({
    sessions: [sess('s_a', 'two', 2)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'two', 2), messages: [userMsg('u_1', 'first', 'a.png'), botMsg('m_1', 'running', START_DATA.options),
                                                                     userMsg('u_2', 'second', 'b.png'), botMsg('m_2', 'done', START_DATA.options)] },
      'GET /v1/messages/m_1': () => { calls += 1; return calls === 1 ? refused(500, 'Could not read the turn.') : { ...botMsg('m_1', 'running'), events: rows(partial) }; },
      'GET /v1/messages/m_2': () => ({ ...botMsg('m_2', 'done'), events: rows(second) }),
    },
  });
  await h.app.start();
  await flush();
  buttonOf(qa($('conversation'), 'article.card')[0], 'Retry').click();
  await flush();
  assert.equal(qa($('conversation'), 'article.card')[0].getAttribute('data-status'), 'running');   // shown as it was last heard of
  assert.equal(h.api.polls.length, 0);                                                              // but it is not the last turn: not followed
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
});

test('a link to another chat, while one failed to open, opens that one and does not try the failed one again', async () => {
  const h = harness({
    hash: '#/s/s_a', sessions: [sess('s_b', 'B', 1), sess('s_a', 'A', 1)],
    routes: { ...doneSession('s_b', 'B', 'm_b'), 'GET /v1/sessions/s_a': () => refused(500, 'Could not read the session.') },
  });
  await h.app.start();
  await flush();
  assert.equal(h.fetch.to('GET', '/v1/sessions/s_a').length, 1);
  const link = qa($('session-list'), 'li').find((li) => li.getAttribute('data-session') === 's_b').querySelector('a');
  link.click();                      // what the browser does first: the link's own handler, at the old address ...
  h.win.location.hash = '#/s/s_b';   // ... and then the address changes
  await flush();
  assert.equal(q($('conversation'), '.user-text').textContent, 'from s_b');
  assert.equal(h.fetch.to('GET', '/v1/sessions/s_a').length, 1);   // the one that failed was not asked for again on the way
  assert.equal($('notice').hidden, true);
});

test('More can be used again for the next page, keeps focus while there are more pages, and hands it on at the last one', async () => {
  const h = harness({
    hash: '#/new',
    routes: {
      'GET /v1/sessions': (url) => {
        if (url.includes('cursor=c2')) return { sessions: [sess('s_e', 'e', 1, 1)], next_cursor: null };
        if (url.includes('cursor=c1')) return { sessions: [sess('s_c', 'c', 1, 2), sess('s_d', 'd', 1, 2)], next_cursor: 'c2' };
        return { sessions: [sess('s_a', 'a', 1, 3), sess('s_b', 'b', 1, 3)], next_cursor: 'c1' };
      },
    },
  });
  await h.app.start();
  await flush();
  const rows = () => qa($('session-list'), 'li').map((li) => li.getAttribute('data-session'));
  $('session-more').focus();
  $('session-more').click();
  await flush();
  assert.deepEqual(rows(), ['s_a', 's_b', 's_c', 's_d']);
  assert.equal($('session-more').hidden, false);
  assert.equal($('session-more').disabled, false);
  assert.equal(document.activeElement, $('session-more'));   // more pages to come: focus stays where it was
  $('session-more').click();                                  // and the next page can be asked for
  await flush();
  assert.deepEqual(rows(), ['s_a', 's_b', 's_c', 's_d', 's_e']);
  assert.equal($('session-more').hidden, true);
  assert.equal(document.activeElement, qa($('session-list'), 'li a')[4]);   // the last page: the first row it added
});

// ---- P4-D follow-up ------------------------------------------------------------------------------------------------------------------------
// Seven residuals of the P4-D re-review (task-P4-D-followup.md). Each test was written before its change and run red on the old code.

// 1: an exception while the accepted turn is drawn must not lock the page ---------------------------------------------------------------------

test('an exception while the accepted turn is drawn does not lock the page: the stall clock runs, Stop works, and the turn settles', async (t) => {
  const logged = captureErrors(t);
  const h = await ready();
  const conversation = $('conversation');
  const append = conversation.append.bind(conversation);
  let boom = true;
  conversation.append = (...nodes) => {   // the accepted turn's card is the first article.card to go in
    if (boom && nodes.some((n) => n.getAttribute?.('class') === 'card')) { boom = false; throw new TypeError('append failed'); }
    return append(...nodes);
  };
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  h.api.streams[0].accept('m_a');   // the drawing in accept() throws
  await flush();
  assert.equal(h.timers.intervals, 1);          // the stall clock was armed before the drawing
  assert.equal($('stop').disabled, false);      // and so was Stop: the server has the turn
  assert.equal(q($('notice'), 'p').textContent, GENERIC);
  assert.ok(logged.some(([e]) => e instanceof TypeError && e.message === 'append failed'), 'it was logged');
  h.api.streams[0].channel.push(...events);
  h.api.streams[0].channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');   // the paint put the card in
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
});

test('when the accept fails and the stream then goes silent, the stall clock hands the turn to the poll, which settles it', async (t) => {
  captureErrors(t);
  const h = await ready();
  const conversation = $('conversation');
  const append = conversation.append.bind(conversation);
  let boom = true;
  conversation.append = (...nodes) => {
    if (boom && nodes.some((n) => n.getAttribute?.('class') === 'card')) { boom = false; throw new TypeError('append failed'); }
    return append(...nodes);
  };
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  h.api.streams[0].accept('m_a');
  await flush();
  h.timers.advance(3000);   // 3 s without an event
  await flush();
  assert.equal(h.api.polls.length, 1);
  assert.equal(h.api.polls[0].opts.after, 0);
  h.api.polls[0].channel.push(...events);
  h.api.polls[0].channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal($('send').disabled, false);
});

// 2: a notice belongs to the route it was raised on -----------------------------------------------------------------------------------------

test('New chat after a chat failed to open clears that chat\'s notice and its Retry, and does not ask for the failed chat again', async () => {
  const h = harness({ hash: '#/s/s_a', sessions: [sess('s_a', 'A', 1)], routes: { 'GET /v1/sessions/s_a': () => refused(500, 'Could not read the session.') } });
  await h.app.start();
  await flush();
  assert.equal(q($('notice'), 'p').textContent, 'Could not read the session.');
  assert.equal(buttonOf($('notice'), 'Retry').hidden, false);
  $('new-session').click();   // #/new: another route
  await flush();
  assert.equal($('notice').hidden, true);
  assert.equal(buttonOf($('notice'), 'Retry').hidden, true);
  assert.equal(q($('notice'), 'p').textContent, '');
  assert.equal(h.win.location.hash, '#/new');
  assert.equal(h.fetch.to('GET', '/v1/sessions/s_a').length, 1);
  assert.equal($('send').disabled, false);
});

test('a notice raised while the page starts on an empty chat is not cleared by the first route', async () => {
  const h = harness({ routes: { 'GET /v1/models': refused(401, 'Missing or wrong token') } });   // the 401 notice comes before the route is handled
  await h.app.start();
  await flush();
  assert.equal(q($('notice'), 'p').textContent, 'Enter the access token in Settings');
  assert.equal($('notice').hidden, false);
});

// 3: the Retry of a card that could not be loaded says which turn ----------------------------------------------------------------------------

test('the Retry of a failed card is named by its turn and starts with its visible text', async () => {
  const h = harness({
    sessions: [sess('s_a', 'two turns', 2)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'two turns', 2), messages: [userMsg('u_1', 'first', 'a.png'), botMsg('m_1', 'done', START_DATA.options),
                                                                           userMsg('u_2', 'second', 'b.png'), botMsg('m_2', 'done', START_DATA.options)] },
      'GET /v1/messages/m_1': () => refused(500, 'Could not read the turn.'),
      'GET /v1/messages/m_2': () => refused(500, 'Could not read the turn.'),
    },
  });
  await h.app.start();
  await flush();
  const retries = qa($('conversation'), 'article.card').map((card) => buttonOf(card, 'Retry'));
  assert.deepEqual(retries.map((b) => b.getAttribute('aria-label')), ['Retry loading turn 1', 'Retry loading turn 2']);
  for (const b of retries) assert.ok(b.getAttribute('aria-label').startsWith(b.textContent), 'label in name');
  retries[1].click();   // still failing: the card is rebuilt, and is named again
  await flush();
  assert.equal(buttonOf(qa($('conversation'), 'article.card')[1], 'Retry').getAttribute('aria-label'), 'Retry loading turn 2');
});

// 4: a Retry that comes late follows nothing ---------------------------------------------------------------------------------------------------

test('a Retry clicked after its turn was left starts no poller', async () => {
  const gate = deferred();
  let hold = false;
  const h = await ready({ routes: { 'GET /v1/models': async () => { if (hold) await gate.promise; return MODELS; } } });
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  h.api.streams[0].accept('m_a');
  await flush();
  h.api.streams[0].channel.push(...events.slice(0, 7));
  await flush();
  h.timers.advance(3000);
  await flush();
  h.api.polls[0].channel.fail(refusal(404, 'Message not found.', 'not_found_error'));
  await turn;
  await flush();
  const retry = buttonOf($('notice'), 'Retry');
  assert.equal(retry.hidden, false);
  hold = true;
  const reload = h.app.reloadAll();   // a new token: the turn is left at once, and its chat is opened again when /v1/models answers
  await flush();
  assert.equal($('notice').hidden, false);   // so the Retry is still on screen for a moment
  retry.click();
  await flush();
  assert.equal(h.api.polls.length, 1);   // and follows a turn nobody is looking at no more
  gate.resolve();
  await reload;
  await flush();
  assert.equal(h.api.polls.length, 1);
});

// 5: a failed Stop that did not matter is not left on screen -------------------------------------------------------------------------------

test('a "couldn\'t stop" notice goes away when the turn ends by itself, whether the stream or the poll brings its end', async () => {
  for (const via of ['stream', 'poll']) {
    const h = await ready();
    const events = fullTurn();
    const turn = h.app.send();
    await flush();
    const run = h.api.streams[0];
    run.accept('m_a');
    await flush();
    run.channel.push(...events.slice(0, 5));
    await flush();
    h.api.cancelError = refusal(404, 'Message not found.', 'not_found_error');
    $('stop').click();
    await flush();
    assert.equal(q($('notice'), 'p').textContent, 'Message not found.', via);
    assert.equal($('notice').hidden, false, via);
    if (via === 'stream') {
      run.channel.push(...events.slice(5));
      run.channel.end();
    } else {
      h.timers.advance(3000);   // the stream goes quiet: the poll brings the end
      await flush();
      h.api.polls[0].channel.push(...events.slice(5));
      h.api.polls[0].channel.end();
    }
    await turn;
    assert.equal(cardOf().getAttribute('data-status'), 'done', via);
    assert.equal($('notice').hidden, true, via);
    assert.equal(q($('notice'), 'p').textContent, '', via);
  }
});

test('a notice that replaced the "couldn\'t stop" one is still there when the turn ends', async (t) => {
  captureErrors(t);
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 4));
  await flush();
  h.api.cancelError = refusal(404, 'Message not found.', 'not_found_error');
  $('stop').click();
  await flush();
  run.channel.push(ev('content_block_delta', { index: 0 }));   // an event that breaks the reducer: its own notice
  await flush();
  assert.equal(q($('notice'), 'p').textContent, GENERIC);
  run.channel.push(...events.slice(4));
  run.channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal(q($('notice'), 'p').textContent, GENERIC);   // not the failed Stop's, and not for the turn's end to clear
  assert.equal($('notice').hidden, false);
});

test('a Stop that works takes down the notice of the Stop that failed before it', async () => {
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 4));
  await flush();
  h.api.cancelError = refusal(404, 'Message not found.', 'not_found_error');
  $('stop').click();
  await flush();
  assert.equal($('notice').hidden, false);
  h.api.cancelError = null;   // the second try goes through
  $('stop').click();
  await flush();
  assert.equal($('notice').hidden, true);   // "Stopping…" is true now, and the old complaint is not
  assert.equal($('stop').textContent, 'Stopping…');
  h.api.polls[0].channel.push(stopEv('aborted'));
  h.api.polls[0].channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'aborted');
  assert.equal($('notice').hidden, true);
});

// 6: the sidebar does not mark a chat that did not open ---------------------------------------------------------------------------------------

test('a chat that failed to open is not marked as the current page in the sidebar, and is again when it opens', async () => {
  let fail = true;
  const ok = doneSession('s_a', 'A', 'm_a');
  const h = harness({
    hash: '#/s/s_a', sessions: [sess('s_b', 'B', 1), sess('s_a', 'A', 1)],
    routes: { ...ok, 'GET /v1/sessions/s_a': () => (fail ? refused(500, 'Could not read the session.') : ok['GET /v1/sessions/s_a']) },
  });
  await h.app.start();
  await flush();
  assert.equal(buttonOf($('notice'), 'Retry').hidden, false);
  assert.equal(q($('session-list'), '[aria-current]'), null);   // nothing is open
  assert.equal(qa($('session-list'), 'li').length, 2);          // and the list is still all there
  fail = false;
  buttonOf($('notice'), 'Retry').click();
  await flush();
  assert.equal(q($('session-list'), '[aria-current]').parentNode.getAttribute('data-session'), 's_a');
});

test('a failed open keeps the keyboard\'s place in the sidebar', async () => {
  const h = harness({ hash: '#/s/s_a', sessions: [sess('s_a', 'A', 1)], routes: { 'GET /v1/sessions/s_a': () => refused(500, 'Could not read the session.') } });
  await h.app.start();
  await flush();
  q(q($('session-list'), 'li'), 'a').focus();
  q(q($('session-list'), 'li'), 'a').click();   // the same address: the link tries again, and fails again
  await flush();
  assert.equal(q($('session-list'), '[aria-current]'), null);
  assert.equal(document.activeElement, q(q($('session-list'), 'li'), 'a'));
});

// 7: aria-busy comes off a frame after it went on ---------------------------------------------------------------------------------------------

test('a chat that is drawn stays aria-busy until the next frame, so a screen reader sees the attribute come off', async () => {
  const h = harness({ sessions: [sess('s_b', 'B', 1)], routes: { ...doneSession('s_b', 'B', 'm_b') } });
  await h.app.start();
  await flush();
  assert.equal($('conversation').children.length, 2);
  assert.equal($('conversation').getAttribute('aria-busy'), 'true');   // set, the chat drawn, and not yet taken off
  nextFrame();
  assert.equal($('conversation').hasAttribute('aria-busy'), false);
});

test('a turn that ends takes aria-busy off one frame later', async () => {
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  h.api.streams[0].accept('m_a');
  await flush();
  assert.equal($('conversation').getAttribute('aria-busy'), 'true');
  h.api.streams[0].channel.push(...events);
  h.api.streams[0].channel.end();
  await turn;
  assert.equal($('send').disabled, false);                              // the composer is free at once ...
  assert.equal($('conversation').getAttribute('aria-busy'), 'true');   // ... and the attribute is seen to go a frame later
  nextFrame();
  assert.equal($('conversation').hasAttribute('aria-busy'), false);
});

test('a chat opened with its last turn running stays aria-busy after the frame that ends the drawing', async () => {
  resetEvents();
  const partial = [startEv(), stageStartEv('preprocess', 0)];
  const h = harness({
    sessions: [sess('s_a', 'running one', 1)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'running one', 1), messages: [userMsg('u_a', '', 'chest.png'), botMsg('m_a', 'running', START_DATA.options)] },
      'GET /v1/messages/m_a': { ...botMsg('m_a', 'running'), events: rows(partial) },
    },
  });
  await h.app.start();
  await flush();
  nextFrame();
  assert.equal($('conversation').getAttribute('aria-busy'), 'true');   // the turn runs: the drawing's frame must not take it off
  h.api.polls[0].channel.push(...fullTurn().slice(2, 3), stopEv('done'));
  h.api.polls[0].channel.end();
  await flush();
  nextFrame();
  assert.equal($('conversation').hasAttribute('aria-busy'), false);
});

test('aria-busy that was put on again before the frame is not taken off by the frame of the first', async () => {
  const h = await ready();
  const turn = h.app.send();   // busy on
  await flush();
  h.api.streams[0].accept('m_a');
  await flush();
  h.api.streams[0].channel.push(stopEv('done'));
  h.api.streams[0].channel.end();
  await turn;                  // busy off: the removal waits for a frame
  attach(imageFile('second.png'));
  const again = h.app.send();  // busy on again before that frame
  await flush();
  nextFrame();                 // the first turn's removal runs now
  assert.equal($('conversation').getAttribute('aria-busy'), 'true');
  h.api.streams[1].accept('m_b');
  await flush();
  h.api.streams[1].channel.push(stopEv('done', { message_id: 'm_b' }));
  h.api.streams[1].channel.end();
  await again;
  nextFrame();
  assert.equal($('conversation').hasAttribute('aria-busy'), false);
});

// ---- P4-E fix round 1 ----------------------------------------------------------------------------------------------------------------------
// Written before the changes they cover, and run red first (task-P4-E-fix1.md, M1 and M2).

// M1: a poll that fails after the turn's own end was read ----------------------------------------------------------------------------------

test('a poll that fails after the turn\'s message_stop was read says nothing: a finished turn is offered no Retry', async () => {
  const h = await ready();
  const events = fullTurn();
  const turn = h.app.send();
  await flush();
  const run = h.api.streams[0];
  run.accept('m_a');
  await flush();
  run.channel.push(...events.slice(0, 5));
  await flush();
  h.timers.advance(3000);   // silence: the poll takes over
  await flush();
  const poll = h.api.polls[0];
  poll.channel.push(...events.slice(5));   // up to and including message_stop
  await flush();
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal($('send').disabled, false);
  // the server stores message_stop before it marks the message finished: the poll that read it asks once more, and that request fails
  poll.channel.fail(refusal(404, 'Message not found.', 'not_found_error'));
  await turn;
  await flush();
  assert.equal($('notice').hidden, true);                       // nothing to say about a turn that is over
  assert.equal(buttonOf($('notice'), 'Retry').hidden, true);
  assert.equal(h.api.polls.length, 1);
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal($('send').disabled, false);
  assert.equal($('stop').hidden, true);
});

test('a poll that fails while the turn is still running still offers its Retry', async () => {
  const { h } = await strandedTurn();   // the poll failed for good, and the turn has not ended
  assert.equal(q($('notice'), 'p').textContent, 'Message not found.');
  assert.equal(buttonOf($('notice'), 'Retry').hidden, false);
});

// M2: what was sent is spent before the card is drawn -----------------------------------------------------------------------------------------

test('an exception while the accepted turn is drawn still spends what was sent, so the next Send does not send it again', async (t) => {
  captureErrors(t);
  const h = await ready();
  const conversation = $('conversation');
  const append = conversation.append.bind(conversation);
  let boom = true;
  conversation.append = (...nodes) => {   // the accepted turn's card is the first article.card to go in
    if (boom && nodes.some((n) => n.getAttribute?.('class') === 'card')) { boom = false; throw new TypeError('append failed'); }
    return append(...nodes);
  };
  $('prompt').value = 'beam 5';
  const turn = h.app.send();
  await flush();
  assert.equal($('prompt').value, 'beam 5');   // until the server has the turn
  h.api.streams[0].accept('m_a');   // the drawing throws
  await flush();
  assert.equal($('prompt').value, '');          // the note that was sent is spent
  assert.equal($('preview').hidden, true);      // and so is the image
  h.api.streams[0].channel.push(...fullTurn());
  h.api.streams[0].channel.end();
  await turn;
  assert.equal($('send').disabled, false);
  assert.equal($('prompt').value, '');
  assert.equal($('preview').hidden, true);
  const again = h.app.send();   // nothing is left to send twice
  await flush();
  const second = h.api.streams[1];
  assert.equal(second.opts.form.get('image'), null);
  assert.equal(second.opts.form.get('text'), '');
  second.accept('m_b');
  await flush();
  second.channel.push(stopEv('done', { message_id: 'm_b' }));
  second.channel.end();
  await again;
});

test('a failure while what was sent is spent does not stop the card from being drawn, nor the turn from being followed', async (t) => {
  const logged = captureErrors(t);
  const h = await ready();
  const events = fullTurn();
  const prompt = $('prompt');
  prompt.value = 'beam 5';
  Object.defineProperty(prompt, 'value', { get: () => 'beam 5', set: () => { throw new TypeError('the note refuses to change'); } });
  const turn = h.app.send();
  await flush();
  h.api.streams[0].accept('m_a');   // spending the note throws
  await flush();
  assert.equal(cardOf().getAttribute('data-message-id'), 'm_a');   // the card is drawn all the same
  assert.equal(q($('notice'), 'p').textContent, GENERIC);
  assert.ok(logged.some(([e]) => e instanceof TypeError && e.message === 'the note refuses to change'), 'it was logged');
  assert.equal(h.timers.intervals, 1);
  h.api.streams[0].channel.push(...events);
  h.api.streams[0].channel.end();
  await turn;
  assert.equal(cardOf().getAttribute('data-status'), 'done');
  assert.equal($('send').disabled, false);
});

test('what was typed or attached since Send stays when the drawing of the accepted turn fails', async (t) => {
  captureErrors(t);
  const h = await ready();
  const conversation = $('conversation');
  const append = conversation.append.bind(conversation);
  let boom = true;
  conversation.append = (...nodes) => {
    if (boom && nodes.some((n) => n.getAttribute?.('class') === 'card')) { boom = false; throw new TypeError('append failed'); }
    return append(...nodes);
  };
  $('prompt').value = 'beam 5';
  const turn = h.app.send();
  await flush();
  $('prompt').value = 'beam 5 and a second thought';   // typed while the upload was on its way
  attach(imageFile('second.png'));
  h.api.streams[0].accept('m_a');
  await flush();
  assert.equal($('prompt').value, ' and a second thought');   // only what was sent is gone
  assert.match(q($('preview'), 'span').textContent, /^second\.png/);
  h.api.streams[0].channel.push(...fullTurn());
  h.api.streams[0].channel.end();
  await turn;
});


// ---- P4-F: settings that visibly apply --------------------------------------------------------------------------------------------

const typeInto = (input, value) => { input.value = value; input.dispatchEvent(new ShimEvent('input', { bubbles: true })); };   // what a key does to a number field
const NO_STAGES = { ...MODELS, features: { retrieval: false, labels: false } };   // what /v1/models says of a server with no gallery and no labeller

test('Display repair is on by default in the page (the server keeps it off), and a stored choice wins', () => {
  assert.equal(DEFAULT_SETTINGS.display_repair, true);
  assert.equal(optionsFromSettings(DEFAULT_SETTINGS, MODELS).display_repair, true);
  const off = loadSettings(memoryStorage({ [SETTINGS_FILE]: JSON.stringify({ display_repair: false }) }));
  assert.equal(off.display_repair, false);
  assert.equal(optionsFromSettings(off, MODELS).display_repair, false);
});

test('the composer chips say nothing of Display repair while it is on, and "raw text" when it is off', async () => {
  const h = harness({ models: MODELS });
  await h.app.start();
  await flush();
  const repair = control('Display repair');
  assert.equal(repair.checked, true);
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3']);
  repair.checked = false;
  change(repair);
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3', 'raw text']);
  assert.equal(saved(h).display_repair, false);
  repair.checked = true;
  change(repair);
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3']);
});

test('a number field applies a whole number inside its bounds as it is typed, and says so when what is typed is not one', async () => {
  const h = harness({ models: MODELS });
  await h.app.start();
  await flush();
  $('settings').click();   // the drawer is open, as it is for whoever types in it
  const tokens = control(BUDGET);
  assert.equal(tokens.getAttribute('data-setting'), 'max_new_tokens');   // a stable hook for the browser check
  tokens.focus();
  typeInto(tokens, '1');    // on the way to 150, and below the bound of 16
  typeInto(tokens, '15');
  assert.equal(tokens.value, '15');   // never changed under the hand that is typing
  assert.equal(chipsText()[1], '100 tok');
  assert.equal(h.storage.data.has(SETTINGS_KEY), false);   // nothing was applied, so nothing was stored
  typeInto(tokens, '150');
  assert.equal(chipsText()[1], '150 tok');   // at once: no blur and no Enter
  assert.equal(saved(h).max_new_tokens, 150);
  assert.deepEqual([tokens.value, document.activeElement === tokens], ['150', true]);
  for (const bad of ['', '1e', '16.5', '-20', 'x', '250', '15']) {
    typeInto(tokens, bad);
    assert.deepEqual([chipsText()[1], saved(h).max_new_tokens, tokens.value], ['100 tok', 100, bad], bad);   // not a whole number inside the bounds: the setting is what it was before this edit
    assert.equal($('max_new_tokens-error').hidden, false, bad);                                               // and the line says so
  }
  change(tokens);   // leaving the field does not clamp it: 15 stays 15, and the setting stays the 100 it was before the edit
  assert.deepEqual([tokens.value, chipsText()[1], saved(h).max_new_tokens], ['15', '100 tok', 100]);
  typeInto(tokens, '0120');   // applied, and the field is not rewritten while it is being typed in ...
  assert.deepEqual([tokens.value, chipsText()[1]], ['0120', '120 tok']);
  assert.equal($('max_new_tokens-error').hidden, true);
  change(tokens);
  assert.equal(tokens.value, '120');   // ... until change writes the setting back
  typeInto(control(BEAM), '5');
  typeInto(control(SIMILAR), '0');
  typeInto(control(MATCHING), '10');
  assert.deepEqual(chipsText(), ['beam 5', '120 tok', 'cached', 'k 0/10']);   // every number field, not only the budget
  typeInto(control(MATCHING), '11');   // above the bound of 10: the setting goes back to the 3 it was before this edit, not the 10 typed on the way
  assert.equal(chipsText().at(-1), 'k 0/3');
});

test('the drawer says that changes apply from the next Send, and while a turn runs that the running turn keeps the settings it started with', async () => {
  const h = await ready({ options: { models: MODELS } });
  const [head, applies, running] = $('drawer').children;
  assert.equal(head.getAttribute('class'), 'drawer-head');
  assert.deepEqual([applies.getAttribute('id'), applies.textContent, applies.hidden], ['apply-note', 'Changes apply from your next Send.', false]);   // first, always
  assert.deepEqual([running.getAttribute('id'), running.textContent, running.hidden],
                   ['running-note', 'The running turn keeps the settings it started with.', true]);
  const turn = h.app.send();
  await flush();
  assert.equal(running.hidden, false);   // from Send, before the server has answered
  h.api.streams[0].accept('m_a');
  await flush();
  typeInto(control(BUDGET), '64');   // the settings keep changing while it runs
  assert.equal(chipsText()[1], '64 tok');   // for the next Send
  assert.deepEqual(texts(qa(q($('conversation'), '.turn.user'), '.options .chip')).slice(0, 2), ['beam 3', '100 tok']);   // and the running turn is not touched
  h.api.streams[0].channel.push(...fullTurn());
  h.api.streams[0].channel.end();
  await turn;
  assert.deepEqual([running.hidden, applies.hidden], [true, false]);
});

test('with no new image the hint follows the note as it is typed: a command or no note re-runs the last X-ray, any other note gets no report (P4-H)', async () => {
  const h = harness({ sessions: [sess('s_a', 'A')], routes: doneSession('s_a', 'A', 'm_a') });
  await h.app.start();
  await flush();
  const hint = $('rerun-hint');
  const RERUN = 'No new image: Send re-runs s_a.png with these settings.';
  const type = (note) => { $('prompt').value = note; $('prompt').dispatchEvent(new ShimEvent('input', { bubbles: true })); return hint.textContent; };
  assert.equal(hint.textContent, RERUN);
  assert.equal(type('Is there pneumonia?'), 'This note is not a command: with no new image, Send gets no report.');   // the server's rule
  assert.equal(hint.hidden, false);
  assert.equal($('send').getAttribute('aria-describedby'), 'rerun-hint');   // Send still says what it will do
  assert.equal(type('tokens 30'), RERUN);
  assert.equal(type('Reference: Heart size is normal.'), RERUN);
  assert.equal(type('   '), RERUN);
  type('what is this?');
  attach(imageFile('new.png'));
  assert.equal(hint.hidden, true);   // a note with a new image goes with it
  buttonOf($('preview'), 'Remove').click();
  assert.equal(hint.textContent, 'This note is not a command: with no new image, Send gets no report.');
});

test('with no new image and an earlier one in the chat, the composer says that Send re-runs it; not in an empty chat, with a new image, or while a turn runs', async () => {
  const empty = harness({ sessions: [] });
  await empty.app.start();
  await flush();
  assert.equal($('rerun-hint').hidden, true);   // nothing to re-run in an empty chat
  const h = harness({ sessions: [sess('s_a', 'A')], routes: doneSession('s_a', 'A', 'm_a') });
  await h.app.start();
  await flush();
  const hint = $('rerun-hint');
  assert.equal(hint.hidden, false);
  assert.equal(hint.textContent, 'No new image: Send re-runs s_a.png with these settings.');
  assert.deepEqual(['role', 'aria-live', 'aria-atomic'].filter((a) => hint.hasAttribute(a)), []);   // information, not an alert
  assert.equal($('send').getAttribute('aria-describedby'), 'rerun-hint');   // Send says what it will do, to whoever reaches it by tab
  const rowsOfComposer = $('composer').children;
  assert.equal(rowsOfComposer[rowsOfComposer.indexOf($('image-well')) + 1], hint);   // next to the image well
  attach(imageFile('new.png'));
  assert.equal(hint.hidden, true);   // Send sends this one
  assert.equal($('send').hasAttribute('aria-describedby'), false);   // a hidden note is still read out if it is named
  buttonOf($('preview'), 'Remove').click();
  assert.equal(hint.hidden, false);   // removed: Send re-runs again
  assert.equal($('send').getAttribute('aria-describedby'), 'rerun-hint');
  attach(imageFile('second.png'));
  const turn = h.app.send();
  await flush();
  assert.equal(hint.hidden, true);   // a turn runs
  h.api.streams[0].accept('m_b');
  await flush();
  assert.equal(hint.hidden, true);
  h.api.streams[0].channel.push(...fullTurn('m_b'));
  h.api.streams[0].channel.end();
  await turn;
  assert.equal(hint.hidden, false);
  assert.equal(hint.textContent, 'No new image: Send re-runs second.png with these settings.');   // the newest image of the chat
  const rerun = h.app.send();   // and Send with no new image does what the hint says
  await flush();
  assert.equal(h.api.streams[1].opts.form.get('image'), null);
  assert.deepEqual(h.api.streams[1].opts.sessionId, 's_a');
  h.api.streams[1].accept('m_c');
  await flush();
  h.api.streams[1].channel.push(...fullTurn('m_c'));
  h.api.streams[1].channel.end();
  await rerun;
  assert.equal(hint.textContent, 'No new image: Send re-runs second.png with these settings.');   // a re-run does not change which image is the newest
});

test('the re-run hint names the newest image of the chat, not the newest turn', async () => {
  resetEvents();
  const question = [startEv({ message_id: 'm_3', image: null }), ev('warning', { code: 'not_a_command', message: 'This is a report generator.' }),
                    stopEv('done', { message_id: 'm_3' })];
  const h = harness({
    sessions: [sess('s_a', 'three turns', 3)],
    routes: {
      'GET /v1/sessions/s_a': { ...sess('s_a', 'three turns', 3), messages: [
        userMsg('u_1', '', 'first.png'), botMsg('m_1'), userMsg('u_2', '', 'second.png'), botMsg('m_2'), userMsg('u_3', 'is it pneumonia?'), botMsg('m_3')] },
      'GET /v1/messages/m_1': () => ({ ...botMsg('m_1'), events: rows(fullTurn('m_1')) }),
      'GET /v1/messages/m_2': () => ({ ...botMsg('m_2'), events: rows(fullTurn('m_2')) }),
      'GET /v1/messages/m_3': () => ({ ...botMsg('m_3'), events: rows(question) }),
    },
  });
  await h.app.start();
  await flush();
  assert.equal($('rerun-hint').hidden, false);
  assert.equal($('rerun-hint').textContent, 'No new image: Send re-runs second.png with these settings.');   // the question after it had no image
});

test('a user turn shows the chips of what ran: no k where the server runs no retrieval, and none for a question, which runs no model (P4-H)', async () => {
  const h = harness({ models: NO_STAGES, sessions: [sess('s_a', 'earlier', 0)], routes: EXISTING });
  await h.app.start();
  await flush();
  const lastChips = () => texts(qa(qa($('conversation'), '.turn.user').at(-1), '.options .chip'));
  attach(imageFile('chest.png'));
  const turn = h.app.send();
  await flush();
  assert.deepEqual(lastChips(), ['beam 3', '100 tok', 'cached']);   // drawn from what is sent, at once: no k
  h.api.streams[0].accept('m_a');
  await flush();
  h.api.streams[0].channel.push(...fullTurn());
  h.api.streams[0].channel.end();
  await turn;
  assert.deepEqual(lastChips(), ['beam 5', '100 tok', 'cached']);   // what the server resolved: still no k, the stage was skipped
  $('prompt').value = 'Is there pneumonia?';
  const question = h.app.send();
  await flush();
  assert.deepEqual(lastChips(), []);   // a note that is no command, with no image: no model will run
  h.api.streams[1].accept('m_q');
  await flush();
  resetEvents();
  h.api.streams[1].channel.push(startEv({ message_id: 'm_q', image: null }), ev('warning', { code: 'not_a_command', message: 'Not a command.' }),
                                stopEv('done', { message_id: 'm_q' }));
  h.api.streams[1].channel.end();
  await question;
  assert.deepEqual(lastChips(), []);   // and message_start, with no image, says none ran
  assert.equal(q(qa($('conversation'), '.turn.user').at(-1), '.user-text').textContent, 'Is there pneumonia?');
});

test('replayed, a question turn has no chips, and a turn shows no k where the server runs no retrieval (an older server is taken to)', async () => {
  resetEvents();
  const question = [startEv({ message_id: 'm_q', image: null }), ev('warning', { code: 'not_a_command', message: 'Not a command.' }),
                    stopEv('done', { message_id: 'm_q' })];
  const routes = {
    'GET /v1/sessions/s_a': { ...sess('s_a', 'two', 2), messages: [userMsg('u_1', '', 'chest.png'), botMsg('m_1', 'done', START_DATA.options),
                                                                  userMsg('u_q', 'what is this?'), botMsg('m_q', 'done', START_DATA.options)] },
    'GET /v1/messages/m_1': () => ({ ...botMsg('m_1'), events: rows(fullTurn('m_1')) }),
    'GET /v1/messages/m_q': { ...botMsg('m_q'), events: rows(question) },
  };
  for (const [models, chips] of [[NO_STAGES, ['beam 5', '100 tok', 'cached']], [MODELS, ['beam 5', '100 tok', 'cached', 'k 4/3']]]) {
    const h = harness({ models, sessions: [sess('s_a', 'two', 2)], routes });
    await h.app.start();
    await flush();
    const [first, , second] = $('conversation').children;
    assert.deepEqual(texts(qa(first, '.options .chip')), chips);
    assert.deepEqual(texts(qa(second, '.options .chip')), []);
    assert.equal(q(second, '.user-text').textContent, 'what is this?');
  }
});

test('a server with no retrieval gallery and no labeller: those controls are disabled and say why, and the chips leave out k', async () => {
  const h = harness({ models: NO_STAGES });
  await h.app.start();
  await flush();
  const retrieval = $('retrieval-note');
  const labels = $('labels-note');
  assert.equal(retrieval.textContent, 'This server has no retrieval gallery, so similar X-rays and matching reports are skipped.');
  assert.equal(labels.textContent, 'This server has no CheXbert labeller, so labels are skipped.');
  assert.deepEqual([retrieval.hidden, labels.hidden], [false, false]);
  for (const [field, note] of [[SIMILAR, 'retrieval-note'], [MATCHING, 'retrieval-note'], ['CheXbert labels', 'labels-note']]) {
    const input = control(field);
    assert.deepEqual([input.disabled, input.getAttribute('aria-describedby')], [true, note], String(field));
  }
  assert.equal(control('CheXbert labels').checked, false);   // nothing to switch on
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached']);   // no "k 4/3" for a stage that is skipped
  // What is sent is what the user chose: the server skips the stages, and says so in the turn.
  const sent = optionsFromSettings(loadSettings(h.storage), NO_STAGES);
  assert.deepEqual([sent.k_images, sent.k_reports, sent.label], [4, 3, true]);
  // The notes' places: after the field they explain (and the field's own, hidden, error line).
  const form = $('settings-form').children;
  assert.deepEqual(form.slice(form.indexOf(labelled(MATCHING)), form.indexOf(labelled(MATCHING)) + 3), [labelled(MATCHING), $('k_reports-error'), retrieval]);
  assert.equal(form[form.indexOf(labelled('CheXbert labels')) + 1], labels);
});

test('a server that runs the stages, and an older one that does not say, leave those controls available', async () => {
  const cases = [['it runs both', { ...MODELS, features: { retrieval: true, labels: true } }], ['an older server', MODELS],
                 ['empty features', { ...MODELS, features: {} }], ['features of the wrong type', { ...MODELS, features: 'none' }],
                 ['null features', { ...MODELS, features: null }]];
  for (const [name, models] of cases) {
    const h = harness({ models });
    await h.app.start();
    await flush();
    for (const field of [SIMILAR, MATCHING, 'CheXbert labels']) {
      assert.deepEqual([control(field).disabled, control(field).hasAttribute('aria-describedby')], [false, false], `${name}: ${field}`);
    }
    assert.deepEqual([$('retrieval-note').hidden, $('labels-note').hidden], [true, true], name);
    assert.equal(control('CheXbert labels').checked, true, name);
    assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3'], name);
  }
  const half = harness({ models: { ...MODELS, features: { retrieval: true, labels: false } } });   // each stage on its own
  await half.app.start();
  await flush();
  assert.deepEqual([control(SIMILAR).disabled, control('CheXbert labels').disabled], [false, true]);
  assert.deepEqual([$('retrieval-note').hidden, $('labels-note').hidden], [true, false]);
  const other = harness({ models: { ...MODELS, features: { retrieval: false, labels: true } } });
  await other.app.start();
  await flush();
  assert.deepEqual([control(MATCHING).disabled, control('CheXbert labels').disabled], [true, false]);
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached']);
});

// P5-E fix 1: GET /v1/models also says per model whether its retrieval and labels run (a model whose tower is not the gallery's has no
// retrieval); the top-level `features` stays the default model's.
const PER_MODEL = { ...MODELS, features: { retrieval: true, labels: true },
                    models: [{ ...CARD_CACHED, features: { retrieval: true, labels: true } }, { ...CARD_PLAIN, features: { retrieval: false, labels: true } }] };

test('serverHas reads the named model\'s features, else the default model\'s, else the server-wide ones', () => {
  assert.equal(serverHas(PER_MODEL, 'retrieval', CARD_PLAIN.name), false);
  assert.equal(serverHas(PER_MODEL, 'labels', CARD_PLAIN.name), true);
  assert.equal(serverHas(PER_MODEL, 'retrieval', CARD_CACHED.name), true);
  const defaultSays = { ...PER_MODEL, models: [{ ...CARD_CACHED, features: { retrieval: false, labels: true } }, CARD_PLAIN] };
  assert.equal(serverHas(defaultSays, 'retrieval'), false);                   // no model named: the default's card
  assert.equal(serverHas(defaultSays, 'retrieval', ''), false);
  assert.equal(serverHas(defaultSays, 'retrieval', 'not listed'), false);     // a model the server does not list runs as the default
  assert.equal(serverHas(defaultSays, 'retrieval', CARD_PLAIN.name), true);   // a card that does not say: the server-wide value
  assert.equal(serverHas(NO_STAGES, 'retrieval', CARD_PLAIN.name), false);    // cards without features: the server-wide value
  assert.equal(serverHas({ ...MODELS, models: [{ ...CARD_CACHED, features: 'none' }] }, 'retrieval'), true);   // not told: available
});

test('the drawer and the chips follow the chosen model\'s features: k is off, and explained, for a model with no retrieval', async () => {
  const h = harness({ models: PER_MODEL });
  await h.app.start();
  await flush();
  $('settings').click();
  assert.deepEqual([control(SIMILAR).disabled, control(MATCHING).disabled, $('retrieval-note').hidden], [false, false, true]);
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3']);
  const model = control('Model');
  model.value = CARD_PLAIN.name;
  change(model);
  assert.equal(saved(h).model, CARD_PLAIN.name);
  assert.deepEqual([control(SIMILAR).disabled, control(MATCHING).disabled, $('retrieval-note').hidden], [true, true, false]);
  assert.equal(control(SIMILAR).getAttribute('aria-describedby'), 'retrieval-note');
  assert.equal(control('CheXbert labels').disabled, false);                  // its labels still run
  assert.equal(chipsText().some((c) => c.startsWith('k ')), false);
  model.value = CARD_CACHED.name;
  change(model);
  assert.deepEqual([control(SIMILAR).disabled, $('retrieval-note').hidden], [false, true]);
  assert.equal(chipsText().includes('k 4/3'), true);
});

test('a replayed user turn shows k only if the model it ran on runs retrieval', async () => {
  resetEvents();
  const routes = {
    'GET /v1/sessions/s_a': { ...sess('s_a', 'two', 2), messages: [
      userMsg('u_1', '', 'chest.png'), botMsg('m_1', 'done', { ...START_DATA.options, model: CARD_PLAIN.name }),
      userMsg('u_2', '', 'chest.png'), botMsg('m_2', 'done', { ...START_DATA.options, model: CARD_CACHED.name })] },
    'GET /v1/messages/m_1': () => ({ ...botMsg('m_1'), events: rows(fullTurn('m_1')) }),
    'GET /v1/messages/m_2': () => ({ ...botMsg('m_2'), events: rows(fullTurn('m_2')) }),
  };
  const h = harness({ models: PER_MODEL, sessions: [sess('s_a', 'two', 2)], routes });
  await h.app.start();
  await flush();
  const [first, , second] = $('conversation').children;
  assert.equal(texts(qa(first, '.options .chip')).some((c) => c.startsWith('k ')), false);   // ran on the model with no retrieval
  assert.equal(texts(qa(second, '.options .chip')).includes('k 4/3'), true);
});

test('serverHas is false only where the server says that a stage is not there', () => {
  assert.equal(serverHas(NO_STAGES, 'retrieval'), false);
  assert.equal(serverHas(NO_STAGES, 'labels'), false);
  assert.equal(serverHas({ features: { retrieval: false, labels: true } }, 'labels'), true);
  for (const models of [null, undefined, MODELS, [], 'x', 3, { features: null }, { features: {} }, { features: [] }, { features: 'no' },
                        { features: { retrieval: true } }, { features: { retrieval: 0 } }, { features: { retrieval: 'false' } }]) {
    assert.equal(serverHas(models, 'retrieval'), true, JSON.stringify(models));   // not told: available, as an older server is
  }
});

// ---- P4-G: a drawer that saves visibly -------------------------------------------------------------------------------------------

const FIELD_ERROR = (low, high) => `Enter a whole number from ${low} to ${high}.`;
const SAVED = 'Settings saved. They apply from your next Send.';
const STORAGE_NOTE = 'Browser storage is unavailable, so these settings last until the page closes.';
const save = () => buttonOf($('drawer'), 'Save');
const mouse = (target, type) => {   // the events of a click on a field: mousedown, then focus, then mouseup
  const event = new ShimEvent(type, { bubbles: true, cancelable: true });
  target.dispatchEvent(event);
  return event;
};
const withDrawerOpen = async (options = {}) => {
  const h = harness({ models: MODELS, ...options });
  await h.app.start();
  await flush();
  $('settings').click();
  return h;
};

test('Stop when the report starts repeating is on by default in the page (the server keeps it off), and a stored setting that lacks it gets the default', () => {
  assert.equal(DEFAULT_SETTINGS.stop_on_repeat, true);
  assert.equal(optionsFromSettings(DEFAULT_SETTINGS, MODELS).stop_on_repeat, true);
  const older = loadSettings(memoryStorage({ [SETTINGS_FILE]: JSON.stringify({ beam_size: 6, display_repair: false }) }));   // saved before the switch existed
  assert.deepEqual([older.stop_on_repeat, older.beam_size, older.display_repair], [true, 6, false]);
  const off = loadSettings(memoryStorage({ [SETTINGS_FILE]: JSON.stringify({ stop_on_repeat: false }) }));
  assert.equal(off.stop_on_repeat, false);
  assert.equal(optionsFromSettings(off, MODELS).stop_on_repeat, false);
  for (const wrong of ['no', 0, 1, null, [], {}]) {
    assert.equal(loadSettings(memoryStorage({ [SETTINGS_FILE]: JSON.stringify({ stop_on_repeat: wrong }) })).stop_on_repeat, true, JSON.stringify(wrong));
  }
  const storage = memoryStorage();
  assert.equal(saveSettings(storage, { ...DEFAULT_SETTINGS, stop_on_repeat: false }), true);
  assert.equal(JSON.parse(storage.data.get(SETTINGS_KEY)).stop_on_repeat, false);
});

test('the drawer has the stop switch, on, with its hint; its chip reads "full budget" when it is off and nothing when it is on', async () => {
  const h = await withDrawerOpen();
  const stop = control('Stop when the report starts repeating');
  assert.deepEqual([stop.getAttribute('type'), stop.getAttribute('data-setting'), stop.checked], ['checkbox', 'stop_on_repeat', true]);
  assert.equal($('stop-hint').textContent, 'Off: the published protocol, which always decodes the whole token budget.');
  assert.equal(stop.getAttribute('aria-describedby'), 'stop-hint');   // the words that say what off means are read with the switch
  const form = $('settings-form').children;
  assert.equal(form[form.indexOf(labelled('Stop when the report starts repeating')) + 1], $('stop-hint'));   // the hint is right under the switch
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3']);   // on is what the page does: nothing to say
  stop.checked = false;
  change(stop);
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3', 'full budget']);
  assert.equal(saved(h).stop_on_repeat, false);
  const repair = control('Display repair');
  repair.checked = false;
  change(repair);
  assert.deepEqual(chipsText().slice(-2), ['raw text', 'full budget']);
  stop.checked = true;
  change(stop);
  assert.equal(chipsText().includes('full budget'), false);
  assert.equal(saved(h).stop_on_repeat, true);
});

test('a Send carries the stop switch as the drawer has it', async () => {
  const h = await ready({ options: { models: MODELS } });
  $('settings').click();
  const stop = control('Stop when the report starts repeating');
  stop.checked = false;
  change(stop);
  const turn = h.app.send();
  await flush();
  assert.equal(JSON.parse(h.api.streams[0].opts.form.get('options')).stop_on_repeat, false);
  h.api.streams[0].accept('m_a');
  await flush();
  h.api.streams[0].channel.push(...fullTurn());
  h.api.streams[0].channel.end();
  await turn;
});

test('a number field selects its whole value when it gets focus; the mouseup of the click that gave it focus does not undo that, and a Tab leaves the first click alone', async () => {
  await withDrawerOpen();
  const tokens = control(BUDGET);
  assert.equal(tokens.value, '100');
  assert.equal(mouse(tokens, 'mouseup').defaultPrevented, false);   // nothing pressed it and it has no focus: nothing to guard
  tokens.focus();   // a Tab: the whole value is selected
  assert.deepEqual([tokens.selectionStart, tokens.selectionEnd], [0, 3]);
  tokens.selectionStart = tokens.selectionEnd = 3;   // the caret at the end of "100", where a click leaves it
  mouse(tokens, 'pointerdown');   // the first click after the Tab, in a field that has focus: it places the caret, as any click does there
  mouse(tokens, 'mousedown');
  assert.equal(mouse(tokens, 'mouseup').defaultPrevented, false);
  tokens.blur();
  mouse(tokens, 'pointerdown');   // a click on a field that has no focus: pointerdown, mousedown, then focus, then mouseup
  mouse(tokens, 'mousedown');
  tokens.focus();
  assert.deepEqual([tokens.selectionStart, tokens.selectionEnd], [0, 3]);
  assert.equal(mouse(tokens, 'mouseup').defaultPrevented, true);    // else the browser puts the caret back, and "150" is typed after "100"
  assert.equal(mouse(tokens, 'mouseup').defaultPrevented, false);   // once: the next click in the field places the caret as it always does
  tokens.blur();
  mouse(tokens, 'mousedown');   // a browser with no pointer events: mousedown alone is the click
  tokens.focus();
  assert.equal(mouse(tokens, 'mouseup').defaultPrevented, true);
  tokens.blur();
  mouse(tokens, 'pointerdown');   // a press that never gave the field focus (dragged away) is not left armed for the focus of a later Tab
  mouse(tokens, 'mouseup');
  tokens.focus();
  assert.equal(mouse(tokens, 'mouseup').defaultPrevented, false);
  tokens.blur();
  tokens.focus();   // and a Tab that is followed by a mouseup of its own (a click begun elsewhere) is not a click on the field
  tokens.blur();
  assert.equal(mouse(tokens, 'mouseup').defaultPrevented, false);
  for (const [label, value] of [[BEAM, '3'], [SIMILAR, '4'], [MATCHING, '3']]) {   // every number field, not only the budget
    const field = control(label);
    mouse(field, 'pointerdown');
    field.focus();
    assert.deepEqual([field.selectionStart, field.selectionEnd], [0, value.length], label);
    assert.equal(mouse(field, 'mouseup').defaultPrevented, true, label);
  }
  control('Decode').focus();   // the other controls are left alone
  assert.equal(control('Decode').selectionStart, undefined);
  assert.equal(mouse(control('Decode'), 'mouseup').defaultPrevented, false);
});

test('a value that is not a whole number inside the range is not applied: the field says why, the chips and the stored setting stay, and a valid value clears it', async () => {
  const h = await withDrawerOpen();
  const tokens = control(BUDGET);
  tokens.focus();
  for (const bad of ['200150', '300', '8', '', '16.5', '-1', 'x']) {   // "200150" is the caret after the 200 with "150" typed; "300" is a user's guess
    typeInto(tokens, bad);
    const error = $('max_new_tokens-error');
    assert.deepEqual([error.hidden, error.textContent], [false, 'Enter a whole number from 16 to 200.'], bad);
    assert.deepEqual([tokens.getAttribute('aria-invalid'), tokens.getAttribute('aria-describedby')], ['true', 'max_new_tokens-error'], bad);
    assert.equal(tokens.value, bad, bad);   // not rewritten
    assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3'], bad);
    assert.equal(h.storage.data.has(SETTINGS_KEY), false, bad);   // and not stored
  }
  typeInto(tokens, '150');
  assert.deepEqual([$('max_new_tokens-error').hidden, tokens.hasAttribute('aria-invalid'), tokens.hasAttribute('aria-describedby')], [true, false, false]);
  assert.deepEqual([chipsText()[1], saved(h).max_new_tokens], ['150 tok', 150]);
  for (const [label, key, text, bounds] of [[BEAM, 'beam_size', '9', [1, 8]], [SIMILAR, 'k_images', '13', [0, 12]], [MATCHING, 'k_reports', '11', [0, 10]]]) {
    typeInto(control(label), text);
    assert.equal($(`${key}-error`).textContent, FIELD_ERROR(...bounds), label);   // each field names its own bounds
    assert.equal($(`${key}-error`).hidden, false, label);
  }
  assert.deepEqual([saved(h).beam_size, saved(h).k_images, saved(h).k_reports], [3, 4, 3]);
  assert.equal($('max_new_tokens-error').parentNode, $('settings-form'));   // under its own field, in the form
  const form = $('settings-form').children;
  assert.equal(form[form.indexOf(labelled(BUDGET)) + 1], $('max_new_tokens-error'));
});

test('Enter in a field submits the form, which is Save: it keeps the value, closes the drawer to Settings and says "Settings saved." for about 4 seconds', async () => {
  const h = await withDrawerOpen();
  const tokens = control(BUDGET);
  tokens.focus();
  typeInto(tokens, '150');
  assert.equal($('saved').textContent, '');
  const enter = press(tokens, 'Enter');   // the form submits as a browser's does
  assert.equal($('drawer').hidden, true);
  assert.equal(document.activeElement, $('settings'));
  assert.equal($('settings').getAttribute('aria-expanded'), 'false');
  assert.equal(saved(h).max_new_tokens, 150);
  assert.equal(enter.defaultPrevented, false);   // the page leaves the key to the browser
  const message = $('saved');
  assert.deepEqual([message.textContent, message.hasAttribute('hidden')], [SAVED, false]);
  assert.deepEqual(['role', 'aria-live'].map((a) => message.getAttribute(a)), ['status', 'polite']);
  h.timers.advance(3900);
  assert.equal(message.textContent, SAVED);   // still there
  h.timers.advance(200);
  assert.equal(message.textContent, '');      // gone after about 4 s
});

test('Enter in a field that holds a value outside the range does not close the drawer: the error stays, the focus stays, and nothing is saved or confirmed', async () => {
  const h = await withDrawerOpen();
  const tokens = control(BUDGET);
  tokens.focus();
  typeInto(tokens, '300');
  press(tokens, 'Enter');
  assert.equal($('drawer').hidden, false);
  assert.equal(document.activeElement, tokens);
  assert.deepEqual([$('max_new_tokens-error').hidden, $('max_new_tokens-error').textContent], [false, 'Enter a whole number from 16 to 200.']);
  assert.equal(tokens.value, '300');   // not changed to 200
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3']);
  assert.equal(h.storage.data.has(SETTINGS_KEY), false);
  assert.equal($('saved').textContent, '');
  typeInto(tokens, '150');   // fixed: the same key now saves
  press(tokens, 'Enter');
  assert.equal($('drawer').hidden, true);
  assert.deepEqual([saved(h).max_new_tokens, $('saved').textContent], [150, SAVED]);
});

test('the drawer ends in a primary Save button that persists, closes to Settings and confirms; the close button stays at the top', async () => {
  const h = await withDrawerOpen();
  const form = $('settings-form');
  assert.equal(form.localName, 'form');
  assert.equal(form.hasAttribute('novalidate'), true);   // our messages, not the browser's bubbles, which would also stop the submit
  const button = save();
  assert.deepEqual([button.localName, button.getAttribute('type'), button.id, button.disabled], ['button', 'submit', 'drawer-save', false]);
  assert.equal(form.children.at(-1).contains(button), true);   // it ends the form
  assert.equal(qa($('drawer'), 'button')[0], $('drawer-close'));   // ✕ stays at the top
  assert.equal(qa($('drawer'), 'h2')[0].textContent, 'Settings');
  assert.equal(form.contains($('drawer-close')), false);
  typeInto(control(BEAM), '5');   // applies as it is typed ...
  assert.equal(saved(h).beam_size, 5);
  h.storage.data.delete(SETTINGS_KEY);   // ... and Save writes the settings again, whole, whether or not anything changed since the last write
  button.click();
  assert.deepEqual([saved(h).beam_size, saved(h).decode, saved(h).max_new_tokens, saved(h).stop_on_repeat], [5, 'beam', 100, true]);
  assert.equal($('drawer').hidden, true);
  assert.equal(document.activeElement, $('settings'));
  assert.equal($('saved').textContent, SAVED);
});

test('Save with an invalid field stays open: it focuses the first invalid field, shows every error, and saves and confirms nothing', async () => {
  const h = await withDrawerOpen();
  typeInto(control(MATCHING), '11');
  typeInto(control(BEAM), '9');
  typeInto(control(BUDGET), '15');   // in the form's order: beam size, the budget, then the retrieval counts
  control(MATCHING).focus();   // focus is elsewhere when Save is pressed
  save().click();
  assert.equal($('drawer').hidden, false);
  assert.equal(document.activeElement, control(BEAM));
  assert.deepEqual(['beam_size', 'max_new_tokens', 'k_images', 'k_reports'].map((key) => $(`${key}-error`).hidden), [false, false, true, false]);
  assert.equal(h.storage.data.has(SETTINGS_KEY), false);
  assert.equal($('saved').textContent, '');
  assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3']);
  assert.equal($('status').textContent, FIELD_ERROR(1, 8));   // a screen reader hears the first
  typeInto(control(BEAM), '4');   // fix it: the next Save goes to the next
  save().click();
  assert.equal(document.activeElement, control(BUDGET));
  assert.equal($('drawer').hidden, false);
  assert.equal($('saved').textContent, '');
});

test('closing the drawer with an invalid field (the close button, Esc or Settings) puts the saved value back and clears the error; a valid value typed before stays', async () => {
  const closers = [['the close button', () => $('drawer-close').click()], ['Esc', () => press(document.activeElement, 'Escape')],
                   ['Settings', () => $('settings').click()]];
  for (const [name, close] of closers) {
    const h = await withDrawerOpen();
    const tokens = control(BUDGET);
    tokens.focus();
    typeInto(tokens, '150');
    change(tokens);   // Enter, or leaving the field: 150 is the setting the next edit starts from
    typeInto(control(BEAM), '5');
    typeInto(tokens, '300');
    assert.equal($('max_new_tokens-error').hidden, false, name);
    close();
    assert.equal($('drawer').hidden, true, name);
    assert.equal(tokens.value, '150', name);   // the saved value: not the 100 it started with, not the 300, not a 30 on the way to it
    assert.deepEqual([$('max_new_tokens-error').hidden, tokens.hasAttribute('aria-invalid'), tokens.hasAttribute('aria-describedby')], [true, false, false], name);
    assert.deepEqual([saved(h).max_new_tokens, saved(h).beam_size, chipsText().slice(0, 2)], [150, 5, ['beam 5', '150 tok']], name);
    assert.equal($('saved').textContent, '', name);   // closed, not saved
    $('settings').click();   // reopened: the saved value, and no error
    assert.deepEqual([control(BUDGET).value, $('max_new_tokens-error').hidden], ['150', true], name);
  }
});

test('Save when the browser will not store the settings closes like any Save and says so beside the chips, never "saved"; the settings still apply in this tab', async () => {
  const h = await withDrawerOpen({ storage: brokenStorage() });
  const note = qa($('drawer'), '.hint').find((n) => n.textContent.startsWith('Browser storage'));
  assert.deepEqual([note.textContent, note.hidden], [STORAGE_NOTE, true]);   // nothing has been tried yet
  save().click();
  assert.equal($('drawer').hidden, true);   // closed, as on success: a Save that leaves the drawer open looks dead
  assert.equal(document.activeElement, $('settings'));
  assert.deepEqual([$('saved').textContent, $('saved').getAttribute('data-state')], [STORAGE_NOTE, 'warn']);   // the storage note in the visible region, in the warning's colour
  assert.notEqual($('saved').textContent, SAVED);
  assert.equal(note.hidden, false);   // and the existing note in the drawer, for when it is opened again
  h.timers.advance(4100);
  assert.deepEqual([$('saved').textContent, $('saved').hasAttribute('data-state')], ['', false]);   // for as long as a confirmation stays, and the colour goes with it
  $('settings').click();
  typeInto(control(BUDGET), '150');
  assert.equal(chipsText()[1], '150 tok');   // the settings still apply for this page
  save().click();
  assert.deepEqual([$('drawer').hidden, $('saved').textContent], [true, STORAGE_NOTE]);
  assert.equal(chipsText()[1], '150 tok');
});

test('a second Save restarts the 4 seconds, and an earlier timer does not take the new message down', async () => {
  const h = await withDrawerOpen();
  save().click();
  assert.equal($('saved').textContent, SAVED);
  h.timers.advance(3000);
  $('settings').click();
  save().click();   // saved again, 3 s into the first
  h.timers.advance(2000);   // 5 s in all: the first timer would have gone
  assert.equal($('saved').textContent, SAVED);
  h.timers.advance(2100);
  assert.equal($('saved').textContent, '');
});

test('a field that holds an invalid draft keeps it, and its error, while another control changes; a field that becomes disabled loses its error', async () => {
  const h = await withDrawerOpen();
  typeInto(control(BUDGET), '300');
  const decode = control('Decode');
  decode.value = 'greedy';
  change(decode);   // syncs the drawer from the settings
  assert.deepEqual([control(BUDGET).value, $('max_new_tokens-error').hidden], ['300', false]);   // not rewritten under the user
  decode.value = 'beam';
  change(decode);
  typeInto(control(BEAM), '9');
  assert.equal($('beam_size-error').hidden, false);
  decode.value = 'greedy';
  change(decode);   // an invalid draft in a field that becomes disabled is dropped: it has nothing to say, and Save does not look at it
  assert.deepEqual([control(BEAM).disabled, control(BEAM).value, $('beam_size-error').hidden, control(BEAM).hasAttribute('aria-invalid')], [true, '3', true, false]);
  save().click();
  assert.equal($('drawer').hidden, false);   // the budget is still invalid
  assert.equal(document.activeElement, control(BUDGET));
  assert.equal(saved(h).decode, 'greedy');
});

test('the confirmation is a polite status region right after the composer chips, empty until a Save and not the page\'s hidden status line', async () => {
  const h = harness();
  await h.app.start();
  const region = $('saved');
  const rowsOfComposer = $('composer').children;
  assert.equal(rowsOfComposer[rowsOfComposer.indexOf($('chips')) + 1], region);
  assert.deepEqual(['role', 'aria-live'].map((a) => region.getAttribute(a)), ['status', 'polite']);
  assert.equal(region.textContent, '');
  assert.equal(region.hasAttribute('hidden'), false);   // present from the start, so that a screen reader has it before its first message
  assert.equal(region.classList.contains('visually-hidden'), false);   // it is for the eye too
  assert.equal($('status').getAttribute('class'), 'visually-hidden');   // the page's own status line stays what it was
  assert.notEqual($('status'), region);
});

// ---- P4-G fix round 1 -------------------------------------------------------------------------------------------------------------

// The setting of one number field as the page holds it: what the chips say and what is stored (null while nothing has been stored).
const chipOf = (key) => ({ beam_size: () => chipsText()[0], max_new_tokens: () => chipsText().find((c) => c.endsWith(' tok')),
                           k_images: () => chipsText().at(-1), k_reports: () => chipsText().at(-1) })[key]();
const storedOf = (h, key) => (h.storage.data.has(SETTINGS_KEY) ? saved(h)[key] : null);
const typedKeys = (input, text) => { for (let n = 1; n <= text.length; n++) typeInto(input, text.slice(0, n)); };

test('a number typed key by key goes back to what the setting was before the edit when it stops being a number the setting can take: 300 over 100 is 100, 30, 100', async () => {
  // [label, key, the keys, chips and stored after each key]: every prefix of a too-big number that is itself a number inside the bounds
  // ("30" of 300, "1" of 12, "25" of 250) used to stay in force, and be stored, under the line that says the whole is wrong
  const cases = [
    [BUDGET, 'max_new_tokens', '300', ['100 tok', '30 tok', '100 tok'], [null, 30, 100]],
    [BUDGET, 'max_new_tokens', '250', ['100 tok', '25 tok', '100 tok'], [null, 25, 100]],
    [BEAM, 'beam_size', '12', ['beam 1', 'beam 3'], [1, 3]],
    [SIMILAR, 'k_images', '13', ['k 1/3', 'k 4/3'], [1, 4]],
    [MATCHING, 'k_reports', '20', ['k 4/2', 'k 4/3'], [2, 3]],
  ];
  for (const focused of [true, false]) {   // a field that had focus when the first key came, and an input that no focus came before
    for (const [label, key, text, chips, stored] of cases) {
      const h = await withDrawerOpen();
      const field = control(label);
      if (focused) field.focus();
      const seen = [];
      for (let n = 1; n <= text.length; n++) {
        typeInto(field, text.slice(0, n));
        seen.push([chipOf(key), storedOf(h, key)]);
      }
      const name = `${label} ${text}${focused ? '' : ' (no focus)'}`;
      assert.deepEqual(seen.slice(-chips.length).map((x) => x[0]), chips, name);   // the chips follow the keys, and end where they began
      assert.deepEqual(seen.slice(-stored.length).map((x) => x[1]), stored, name);   // so does the stored setting
      assert.equal(field.value, text, name);               // the field is left as typed
      assert.equal($(`${key}-error`).hidden, false, name);   // under its line
      assert.deepEqual(chipsText(), ['beam 3', '100 tok', 'cached', 'k 4/3'], name);   // every setting is what it was before the edit
    }
  }
});

test('Enter or Save while a number is wrong keeps the setting it was before the edit, with the line showing', async () => {
  const h = await withDrawerOpen();
  const tokens = control(BUDGET);
  tokens.focus();
  typedKeys(tokens, '300');
  assert.deepEqual([chipOf('max_new_tokens'), storedOf(h, 'max_new_tokens')], ['100 tok', 100]);
  press(tokens, 'Enter');
  assert.deepEqual([$('drawer').hidden, $('max_new_tokens-error').hidden, tokens.value, chipOf('max_new_tokens'), storedOf(h, 'max_new_tokens')],
                   [false, false, '300', '100 tok', 100]);   // the value stays 100: Enter did not apply a 30, nor a 200
  save().click();
  assert.deepEqual([$('drawer').hidden, $('max_new_tokens-error').hidden, tokens.value, chipOf('max_new_tokens'), storedOf(h, 'max_new_tokens'), $('saved').textContent],
                   [false, false, '300', '100 tok', 100, '']);
  typedKeys(tokens, '30');   // Backspace: the 30 on the way is applied again, and is a number the setting can take
  assert.deepEqual([chipOf('max_new_tokens'), $('max_new_tokens-error').hidden], ['30 tok', true]);
});

test('a stray digit after a number that was fine takes the setting back to what it was before the edit, Backspace applies it again, and a committed edit is the next one\'s start', async () => {
  const h = await withDrawerOpen();
  const tokens = control(BUDGET);
  tokens.focus();
  typeInto(tokens, '150');
  assert.deepEqual([chipOf('max_new_tokens'), storedOf(h, 'max_new_tokens')], ['150 tok', 150]);
  typeInto(tokens, '1500');   // a stray 0
  assert.deepEqual([chipOf('max_new_tokens'), storedOf(h, 'max_new_tokens'), tokens.value, $('max_new_tokens-error').hidden], ['100 tok', 100, '1500', false]);
  typeInto(tokens, '150');    // Backspace
  assert.deepEqual([chipOf('max_new_tokens'), storedOf(h, 'max_new_tokens'), $('max_new_tokens-error').hidden], ['150 tok', 150, true]);
  change(tokens);             // Enter, or leaving the field: committed, and the next edit starts from 150
  typeInto(tokens, '1500');
  assert.deepEqual([chipOf('max_new_tokens'), storedOf(h, 'max_new_tokens')], ['150 tok', 150]);
  tokens.blur();              // leaving the field with the wrong text in it ends that edit too
  typeInto(tokens, '20');     // a new edit: it applies ...
  assert.equal(chipOf('max_new_tokens'), '20 tok');
  typeInto(tokens, '2000');   // ... and goes back to the 150 this edit started from
  assert.deepEqual([chipOf('max_new_tokens'), storedOf(h, 'max_new_tokens')], ['150 tok', 150]);
  tokens.focus();             // focus begins an edit as well: it starts from what the setting is now
  typeInto(tokens, '30');
  typeInto(tokens, '3000');
  assert.equal(chipOf('max_new_tokens'), '150 tok');
  typeInto(tokens, '40');     // applies, in the edit that began at the focus
  tokens.blur();              // which ends with the blur alone, with no change before it
  typeInto(tokens, '4000');   // the next edit starts from the 40 the setting is now, not from the 150 before the one that ended
  assert.equal(chipOf('max_new_tokens'), '40 tok');
});

test('a refused Enter or Save in a field that already has focus selects the bad text again, so that the next key replaces it, and says the same message again', async () => {
  await withDrawerOpen();
  const tokens = control(BUDGET);
  tokens.focus();
  typeInto(tokens, '200150');
  tokens.selectionStart = tokens.selectionEnd = 6;   // the caret at the end, where the typing left it
  const status = $('status');
  const written = [];
  const original = status.replaceChildren.bind(status);
  status.replaceChildren = (...nodes) => { written.push(nodes.map(String).join('')); return original(...nodes); };
  press(tokens, 'Enter');
  assert.deepEqual([tokens.selectionStart, tokens.selectionEnd], [0, 6]);   // no focus event comes to a field that has focus, so the page selects it itself
  tokens.selectionStart = tokens.selectionEnd = 6;
  press(tokens, 'Enter');   // refused again, with the very same words
  save().click();           // and once more, from Save
  assert.deepEqual([tokens.selectionStart, tokens.selectionEnd], [0, 6]);
  assert.deepEqual(written.filter(Boolean), [FIELD_ERROR(16, 200), FIELD_ERROR(16, 200), FIELD_ERROR(16, 200)]);   // every refusal is said, the identical ones too
  assert.equal(status.textContent, FIELD_ERROR(16, 200));
});

test('opening the drawer again takes a confirmation that is still showing down, and its timer with it', async () => {
  const h = await withDrawerOpen();
  save().click();
  assert.equal($('saved').textContent, SAVED);
  assert.equal(h.timers.timeouts.filter((t) => t.ms === 4000).length, 1);   // the 4 s that will take it down
  $('settings').click();   // reopened within the 4 s: the message is about a Save that is behind the user now
  assert.deepEqual([$('saved').textContent, $('saved').hasAttribute('data-state')], ['', false]);
  assert.equal(h.timers.timeouts.filter((t) => t.ms === 4000).length, 0);
  $('drawer-close').click();
  assert.equal($('saved').textContent, '');   // closing says nothing: it is not a Save
});
