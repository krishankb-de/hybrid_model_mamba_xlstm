// app/static/app.js — wires the page: routing, the sessions sidebar, the composer, one streaming turn with Stop and the
// polling watchdog, the settings drawer, exports and the health strip (CHAT_UI_PLAN.md P4-D). api.js, state.js and render.js
// do the data and the drawing; this file owns the page and what happens when.
//
// Two halves. The first is pure and exported, so node tests it without a page: parseRoute, loadSettings / saveSettings,
// optionsFromSettings, clientId, errorMessage, createWatchdog (a state machine whose only input from outside is the clock it
// is given) and the small formatters. The second, createApp(env), is everything that touches the page, and takes what it
// touches (document, window, fetch, storage, timers, the api module) as env, so a test hands it a shim page and fakes. The
// last lines start it, and only on a page that has a #composer: importing this file in node does nothing.
//
// A turn is one POST that streams. When no event arrives for 3 s while the turn runs, the page drops that stream (which does
// not stop the turn on the server: D7) and polls GET /v1/messages/{id}?after=<lastSeq> until the turn ends; Stop sends the
// cancel first and then drops the stream, and the poll brings the turn's own message_stop. Leaving a session drops the
// stream and does not poll: coming back finds the turn running and polls it from its last seq.
//
// The settings drawer is a form that ends in Save (P4-G). A number applies as it is typed while it is a whole number inside its
// bounds; anything else is left as typed, with a line under the field that says what it takes, and is never clamped, applied or stored.
// Save (or Enter in a field) checks every field, then stores, closes the drawer and says so; closing it any other way puts the saved
// value back in a field that holds a draft.
//
// Every control is shown or hidden with the hidden attribute and every string reaches the page as a text node (el() and
// textContent), never as markup. Nothing is stored but the settings and the client id, both inside try/catch.
import * as realApi from './api.js';
import { applyEvent, initialView } from './state.js';
import {
  detailTable, el, focusKey, optionChips, renderAssistantCard, renderUserTurn, restoreFocus, scheduleRender, statusText,
} from './render.js';

const { authHeaders } = realApi;

export const SETTINGS_KEY = 'cxrchat.settings';
export const CLIENT_KEY = 'cxrchat.client';
export const STALL_MS = 3000;          // silence on the stream that hands a running turn to polling
export const TICK_MS = 500;            // how often a running turn asks the watchdog
export const HEALTH_MS = 10000;        // /healthz, while it answers
export const HEALTH_SLOW_MS = 30000;   // and after HEALTH_SLOW_AFTER failures in a row, until it answers again
export const HEALTH_SLOW_AFTER = 3;
export const HEALTH_TIMEOUT_MS = 5000;  // a /healthz that has not answered by then has failed: a half-open tunnel never says so itself
const PAGE = 50;                       // sessions per request of the sidebar
const NEAR_PX = 80;                    // "at the bottom" for auto-scroll: within this many pixels of it
const REVOKE_MS = 1000;                // an export's object URL is revoked this long after its link was clicked
const MAX_IMAGE_BYTES = 20 * 1024 * 1024;   // the server's upload limit
const IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/webp'];
const SESSION_ID = /^[A-Za-z0-9_-]{1,64}$/;   // what the server's new_id produces
const CLIENT_ID = /^[\x21-\x7e]{1,128}$/;     // what the server accepts as X-Client-Id: visible ASCII
const SHORT_VIEWPORT = '(max-height: 480px)'; // the page scrolls instead of #conversation (styles.css)

// ---- routes ---------------------------------------------------------------------------------------------------------------

// #/s/<id> is a session, #/new an empty chat that has no session until its first turn is sent, and anything else (no hash,
// "#/", a malformed id) is home: the page resolves it to the newest session, or to #/new when there is none.
export function parseRoute(hash) {
  const h = typeof hash === 'string' ? hash : '';
  const m = /^#\/s\/([^/?#]+)\/?$/.exec(h);
  if (m) {
    let id = '';
    try { id = decodeURIComponent(m[1]); } catch { id = ''; }
    if (SESSION_ID.test(id)) return { kind: 'session', id };
  }
  return h === '#/new' ? { kind: 'new' } : { kind: 'home' };
}

// ---- settings and options --------------------------------------------------------------------------------------------------

export const BOUNDS = Object.freeze({   // the server's bounds (app/schemas.py Options), in the order the drawer lists the fields
  beam_size: [1, 8], max_new_tokens: [16, 200], k_images: [0, 12], k_reports: [0, 10],
});
const NUMBER_KEYS = Object.keys(BOUNDS);
const rangeLabel = (title, key) => `${title} (${BOUNDS[key][0]}–${BOUNDS[key][1]})`;           // "Token budget (16–200)": the range is in the label
const rangeMessage = (key) => `Enter a whole number from ${BOUNDS[key][0]} to ${BOUNDS[key][1]}.`;   // what a field says when its text is not one
const SAVED_TEXT = 'Settings saved. They apply from your next Send.';
const SAVED_MS = 4000;   // how long the page says so
const STORAGE_TEXT = 'Browser storage is unavailable, so these settings last until the page closes.';
// The published protocol, as every Options default has it, but for two switches that are on here and off in the server's Options (its
// API contract). display_repair changes only what is shown: the report is the decoder's own text either way, and Show raw shows that
// text. stop_on_repeat ends a decoder that has no stop condition of its own once the report begins to repeat itself, so that a turn
// does not go on writing the same sentence to its token budget; off, the whole budget is decoded, which is the published protocol.
// model '' is the server's default; token is the access token.
export const DEFAULT_SETTINGS = Object.freeze({
  model: '', decode: 'beam', beam_size: 3, max_new_tokens: 100, cached_decode: true, compile: false,
  k_images: 4, k_reports: 3, label: true, display_repair: true, stop_on_repeat: true, token: '',
});

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

function clampInt(value, [low, high], fallback) {
  const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN;
  return Number.isFinite(n) ? Math.min(high, Math.max(low, Math.round(n))) : fallback;
}

// A whole number inside the bounds, as that number; null for anything else (half typed, out of range, not a number).
function wholeNumber(value, [low, high]) {
  const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN;
  return Number.isInteger(n) && n >= low && n <= high ? n : null;
}

// Anything in, a complete settings object out: a key that is missing or of the wrong type takes its default, a number is
// clamped to its bounds, and a key that is not a setting is dropped.
function sanitize(raw) {
  const o = isObject(raw) ? raw : {};
  const d = DEFAULT_SETTINGS;
  const flag = (key) => (typeof o[key] === 'boolean' ? o[key] : d[key]);
  return {
    model: typeof o.model === 'string' ? o.model : d.model,
    decode: o.decode === 'greedy' || o.decode === 'beam' ? o.decode : d.decode,
    beam_size: clampInt(o.beam_size, BOUNDS.beam_size, d.beam_size),
    max_new_tokens: clampInt(o.max_new_tokens, BOUNDS.max_new_tokens, d.max_new_tokens),
    cached_decode: flag('cached_decode'),
    compile: flag('compile'),
    k_images: clampInt(o.k_images, BOUNDS.k_images, d.k_images),
    k_reports: clampInt(o.k_reports, BOUNDS.k_reports, d.k_reports),
    label: flag('label'),
    display_repair: flag('display_repair'),
    stop_on_repeat: flag('stop_on_repeat'),
    token: typeof o.token === 'string' ? o.token : d.token,
  };
}

// The stored settings, or the defaults. storage may be absent, may throw on any access (a browser that blocks site data),
// and may hold anything: all of it ends in the defaults, never in an exception.
export function loadSettings(storage) {
  let parsed = null;
  try {
    const raw = storage?.getItem(SETTINGS_KEY);
    parsed = typeof raw === 'string' ? JSON.parse(raw) : null;
  } catch { parsed = null; }
  return sanitize(parsed);
}

// true when the settings were stored; false when there is no storage, or it refused (full, blocked, private window).
export function saveSettings(storage, settings) {
  try {
    storage.setItem(SETTINGS_KEY, JSON.stringify(sanitize(settings)));
    return true;
  } catch { return false; }
}

// Does the server run this stage ('retrieval' or 'labels')? GET /v1/models says in `features`; false means its pipeline skips the stage.
// A server that does not say (an older one, or before /v1/models has answered) is taken to run it: a control is not disabled on a guess.
export function serverHas(models, feature) {
  const features = isObject(models) ? models.features : null;
  return !(isObject(features) && features[feature] === false);
}

const cardsOf = (models) => (Array.isArray(models) ? models : Array.isArray(models?.models) ? models.models : []).filter(isObject);

// The card of the model the settings choose: the named one, else the server's default. null while /v1/models is unknown.
export function chosenCard(settings, models) {
  const wanted = typeof settings?.model === 'string' ? settings.model : '';
  const cards = cardsOf(models);
  return cards.find((c) => c.name === wanted) ?? cards.find((c) => c.name === models?.default_model) ?? null;
}

// The options field of a turn: exactly the keys of the server's Options that the drawer sets (it refuses any other), every
// one inside its bounds. models is the /v1/models payload ({default_model, allow_compile, models: [card, ...]}), or null
// before it has loaded. A model the server does not list is left out, so the server's default runs; cached decoding is
// off when the chosen card says there is no cache, and compile unless the server allows it.
export function optionsFromSettings(settings, models) {
  const s = sanitize(settings);
  const card = chosenCard(s, models);
  const listed = cardsOf(models).some((c) => c.name === s.model);
  const options = {};
  if (s.model && listed) options.model = s.model;
  return Object.assign(options, {
    decode: s.decode,
    beam_size: s.beam_size,
    max_new_tokens: s.max_new_tokens,
    cached_decode: s.cached_decode && !(card && card.cached_decode_available === false),
    compile: s.compile && !Array.isArray(models) && models?.allow_compile === true,
    k_images: s.k_images,
    k_reports: s.k_reports,
    label: s.label,
    display_repair: s.display_repair,
    stop_on_repeat: s.stop_on_repeat,
  });
}

// ---- identity -------------------------------------------------------------------------------------------------------------

function randomId() {
  const c = globalThis.crypto;
  if (typeof c?.randomUUID === 'function') return c.randomUUID();
  if (typeof c?.getRandomValues === 'function') {
    return Array.from(c.getRandomValues(new Uint8Array(16)), (b) => b.toString(16).padStart(2, '0')).join('');
  }
  return Array.from({ length: 4 }, () => Math.random().toString(16).slice(2, 10).padEnd(8, '0')).join('');
}

// The X-Client-Id: random, kept under cxrchat.client. A storage that refuses (or an id it holds that the server would
// refuse) gives a fresh one, so the page still has an identity, for as long as the caller keeps this one.
export function clientId(storage) {
  try {
    const kept = storage?.getItem(CLIENT_KEY);
    if (typeof kept === 'string' && CLIENT_ID.test(kept)) return kept;
  } catch { /* read refused: make one */ }
  const fresh = randomId();
  try { storage?.setItem(CLIENT_KEY, fresh); } catch { /* write refused: it lives as long as the page */ }
  return fresh;
}

// window.localStorage, or null: the property itself throws in a browser that blocks site data.
export function browserStorage(win) {
  try { return win?.localStorage ?? null; } catch { return null; }
}

// ---- messages for the user ---------------------------------------------------------------------------------------------------

const text = (v) => (typeof v === 'string' && v.trim() ? v.trim() : '');
const clip = (s, n = 300) => (s.length > n ? `${s.slice(0, n - 1)}…` : s);
const isAbort = (err) => err?.name === 'AbortError';   // what fetch throws for a request whose signal was aborted
const GENERIC_ERROR = 'Something went wrong — see the console';   // for what is a bug of the page's, not the server's or the network's
// What the browsers' fetch says when the network fails, each in its own words (Chrome, Firefox, Safari, node).
const FETCH_FAILURE = /failed to fetch|fetch failed|load failed|networkerror|network error|network connection|internet connection|network request failed/i;

// A TypeError is how fetch reports a network failure, and also how a bug of ours reports itself. Where only fetch can have
// thrown (a request, the stream or the poll of api.js, a cancel) the error is marked, and errorMessage trusts the mark.
function markNetwork(err) {
  if (err instanceof TypeError && !('network' in err)) err.network = true;
  return err;
}

// A failure that is the server's (it refused), the network's or the user's own cancel is shown; anything else is ours and is logged.
const expected = (err) => typeof err?.status === 'number' || err?.network === true || isAbort(err);

// What a failed request says to the user. err is what api.js throws: an Error with `status` and `body` (the server's
// envelope {type: "error", error: {type, message}}), or what fetch throws when the network fails (a TypeError).
export function errorMessage(err) {
  if (typeof err === 'string') return text(err) || 'Something went wrong.';
  if (isAbort(err)) return 'The request was cancelled.';   // never the browser's own "signal is aborted without reason"
  const status = err?.status;
  const server = text(err?.body?.error?.message);
  if (status === 401) return 'Enter the access token in Settings';
  if (status === 429) return 'The server is busy; try again shortly';
  if (status === 413) return clip(server || 'The image is too large.');
  if (status === 422) return clip(server || 'The server could not use that request.');
  if (typeof status === 'number') {
    if (server) return clip(server);
    if (status === 400) return 'The server could not read the request.';
    if (status === 403) return 'The server refused this request.';
    if (status === 404) return 'Not found. It may have been deleted.';
    return status >= 500 ? 'The server had a problem. Try again.' : 'The request failed.';
  }
  if (err?.network === true || (err instanceof TypeError && FETCH_FAILURE.test(err.message))) {
    return 'Cannot reach the server. Check the connection and try again.';
  }
  if (err instanceof TypeError) return GENERIC_ERROR;
  return clip(text(err?.message) || 'Something went wrong.');
}

// What comes out of an api.js generator's next() is the transport's: a stream that failed, a poll that was refused. (What the page
// does with each event is its own, and is caught where it is done.) A TypeError out of it is a network failure, and is marked.
async function* fromTransport(source) {
  try {
    yield* source;
  } catch (err) {
    throw markNetwork(err);
  }
}

// ---- the watchdog ----------------------------------------------------------------------------------------------------------

// A state machine for one turn's transport, and nothing else: no timers, no page, no network. The clock is the only input
// from outside, and every method returns what the page should do. Phases: idle, stream (the POST is open), poll (the stream
// is given up and GET /v1/messages/{id} is followed), failed (the poll threw a terminal error: the page offers a retry).
export function createWatchdog({ now = () => Date.now(), stallMs = STALL_MS } = {}) {
  let phase = 'idle';
  let heard = 0;
  return {
    get phase() { return phase; },
    arm() { phase = 'stream'; heard = now(); },                       // the turn was sent
    bytes() { if (phase === 'stream') heard = now(); },               // an event came in on the stream
    // Asked every TICK_MS with the view's status: 'poll' once, when the turn is running and the stream has been silent for
    // stallMs (the page drops the stream); otherwise 'none'. A turn that has ended is idle.
    tick(status) {
      if (phase !== 'stream') return 'none';
      if (status !== 'running') { phase = 'idle'; return 'none'; }
      if (now() - heard < stallMs) return 'none';
      phase = 'poll';
      return 'poll';
    },
    // The stream is over (it closed, was dropped, or Stop dropped it): 'poll' while the turn is still running.
    ended(status) {
      if (phase === 'idle' || phase === 'failed') return 'none';
      phase = status === 'running' ? 'poll' : 'idle';
      return phase === 'poll' ? 'poll' : 'none';
    },
    resume() { phase = 'poll'; },                                     // a turn found running on open: no stream, poll it
    failed() { if (phase === 'poll') phase = 'failed'; },             // the poller threw a terminal error
    retry() { if (phase !== 'failed') return 'none'; phase = 'poll'; return 'poll'; },
    settle() { phase = 'idle'; },                                     // the turn ended, or the page left it
  };
}

// ---- small formatters --------------------------------------------------------------------------------------------------------

export const sessionTitle = (session) => text(session?.title) || 'New chat';

// "Oct 3", or "Oct 3, 2025" for another year; '' for what is not a date.
export function sessionDate(iso, now = new Date()) {
  const d = new Date(iso);
  if (typeof iso !== 'string' || Number.isNaN(d.getTime())) return '';
  const options = d.getFullYear() === now.getFullYear() ? { month: 'short', day: 'numeric' } : { year: 'numeric', month: 'short', day: 'numeric' };
  return d.toLocaleDateString(undefined, options);
}

export function sessionMeta(session, now = new Date()) {
  const n = Number.isInteger(session?.turns) ? session.turns : 0;
  return [sessionDate(session?.updated_at ?? session?.created_at, now), `${n} ${n === 1 ? 'turn' : 'turns'}`].filter(Boolean).join(' · ');
}

// What the strip says: the mode and the load. A payload that lacks a number says what it has.
export function healthText(h) {
  const mode = text(h?.mode);
  const n = Number.isInteger(h?.turns_in_flight) ? h.turns_in_flight : null;
  const cap = Number.isInteger(h?.queue_cap) ? h.queue_cap : null;
  const load = n === null ? '' : `${n}${cap === null ? '' : ` of ${cap}`} ${cap === null && n === 1 ? 'turn' : 'turns'} in flight`;   // "1 of 4 turns", "1 turn"
  return [mode, load].filter(Boolean).join(' · ') || 'server ok';
}

// 10 s while the server answers, 30 s after three failed checks in a row, and 10 s again at the first answer.
export const nextHealthDelay = (failures) => (failures >= HEALTH_SLOW_AFTER ? HEALTH_SLOW_MS : HEALTH_MS);

// The file name an export is saved under: the server's Content-Disposition when it is a plain file name, else one made
// from the session id. Never anything with a path in it.
export function exportFilename(contentDisposition, sessionId, format) {
  const m = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(typeof contentDisposition === 'string' ? contentDisposition : '');
  return m && /^\w[\w.-]{0,100}$/.test(m[1]) ? m[1] : `session-${sessionId}.${format}`;
}

// A user message and the assistant message that answers it, as the user turn draws them. The assistant row carries the
// turn's resolved options. A replayed upload has no image URL until the image endpoint exists (P6-B): the file name only.
export function userTurnMessage(user, assistant) {
  const filename = text(user?.image_filename);
  return {
    text: typeof user?.text === 'string' ? user.text : '',
    image: filename ? { url: null, filename } : null,
    options: isObject(assistant?.options) ? assistant.options : null,
  };
}

// Is the scroller within NEAR_PX of its end? m is an element, or {scrollTop, clientHeight, scrollHeight}. Without those
// numbers (a page that has not laid out) it says yes: the page follows the turn.
export function nearBottom(m, slack = NEAR_PX) {
  const { scrollTop, clientHeight, scrollHeight } = m ?? {};
  if (![scrollTop, clientHeight, scrollHeight].every(Number.isFinite)) return true;
  return scrollHeight - scrollTop - clientHeight <= slack;
}

// null when the file may be attached; else why not. The server decides on the content: this only saves an upload that
// cannot work (a type the picker would not offer, a file over the limit). A file with no type is let through.
export function checkImageFile(file) {
  if (!file) return 'Choose an image.';
  if (file.type && !IMAGE_TYPES.includes(file.type)) return 'Choose a PNG, JPEG or WEBP image.';
  if (Number.isFinite(file.size) && file.size > MAX_IMAGE_BYTES) return 'The image is over the 20 MB limit.';
  return null;
}

// ---- the page -----------------------------------------------------------------------------------------------------------------

// env: document, window (location, history, addEventListener, matchMedia, navigator, confirm), storage, fetch, api (the
// api.js functions), now, setTimeout / setInterval / clearInterval, URL, confirm. All optional but document.
export function createApp(env) {
  const doc = env.document;
  const win = env.window ?? {};
  const api = env.api ?? realApi;
  const storage = env.storage ?? null;
  const urls = env.URL ?? globalThis.URL;
  const doFetch = (url, init) => (env.fetch ?? globalThis.fetch)(url, init);
  const later = (fn, ms) => (env.setTimeout ? env.setTimeout(fn, ms) : globalThis.setTimeout(fn, ms));
  const cancelLater = (id) => (env.clearTimeout ? env.clearTimeout(id) : globalThis.clearTimeout(id));
  const every = (fn, ms) => (env.setInterval ? env.setInterval(fn, ms) : globalThis.setInterval(fn, ms));
  const cancelEvery = (id) => (env.clearInterval ? env.clearInterval(id) : globalThis.clearInterval(id));
  // The next animation frame: env's, else the page's, else a timer (a page with none). A test hands one it runs by hand.
  const frame = (fn) => {
    const raf = env.requestAnimationFrame ?? win.requestAnimationFrame ?? globalThis.requestAnimationFrame;
    return typeof raf === 'function' ? raf.call(globalThis, fn) : later(fn, 16);
  };
  // Delete is the one thing this asks about, and it fails closed: with no way to ask, nothing is deleted.
  const confirmed = (message) => (env.confirm ? env.confirm(message) : typeof win.confirm === 'function' ? win.confirm(message) : false);
  const watchdog = createWatchdog({ now: env.now ?? (() => Date.now()) });

  const $ = (id) => doc.getElementById(id);
  const qa = (node, selector) => Array.from(node.querySelectorAll(selector));
  const ui = {
    toggle: $('sidebar-toggle'), sidebar: $('sidebar'), newChat: $('new-session'), list: $('session-list'),
    conversation: $('conversation'), drawer: $('drawer'), composer: $('composer'), well: $('image-well'),
    preview: $('preview'), prompt: $('prompt'), chips: $('chips'), settings: $('settings'), send: $('send'),
    stop: $('stop'), file: $('file'), badge: $('mode-badge'), health: $('health'),
  };

  const state = {
    settings: loadSettings(storage),
    clientId: clientId(storage),
    models: null,            // the /v1/models payload
    session: { id: null, turns: [], ui: new Map(), blobUrls: new Set() },   // what the conversation shows
    sessions: [],            // the sidebar
    nextCursor: null,
    sessionsLoaded: false,
    turn: null,              // the turn this view is following, if one is running
    busy: false,             // a turn runs here: Send is off and Stop is on
    loading: false,          // a session is being fetched
    file: null,              // the attached image, not yet sent
    previewUrl: null,
    said: '',                // the last text written to the status region
    gen: 0,                  // bumped by every route change, so a late answer can tell it is stale
    ticker: null,
    drawerOpener: null,
    healthFailures: 0,
    loadingMore: false,      // a page of the sidebar is being fetched: More is off, so it cannot be asked twice
    noticeRetry: null,
    noticeOwner: null,       // the turn whose failed Stop the notice says, if it is that: the end of the turn takes it down
    savedTimer: null,        // the timer that takes "Settings saved." down again
    route: null,             // the route handleRoute showed last ('new' or 's/<id>'): a notice belongs to the route it was raised on
    missing: null,           // a session the server said it does not have: home never picks it
  };

  const auth = () => ({ token: state.settings.token, clientId: state.clientId });
  const report = (err) => { try { console.error(err); } catch { /* no console */ } };
  const detach = (promise) => { Promise.resolve(promise).catch(report); };   // a handler's async work: failures are shown, never unhandled

  // ---- requests --------------------------------------------------------------------------------------------------------------

  // A JSON request with the page's auth headers. A refusal throws what api.js throws: an Error with status and body.
  async function request(path, { method = 'GET', body, headers, raw = false, signal } = {}) {
    let res;
    try {
      res = await doFetch(path, { method, body, signal, headers: { ...authHeaders(state.settings.token, state.clientId), ...headers } });
    } catch (err) {
      throw markNetwork(err);   // only fetch itself can have thrown here
    }
    if (!res.ok) throw Object.assign(new Error('request refused'), { status: res.status, body: await res.json().catch(() => null) });
    if (raw) return res;
    return res.status === 204 ? null : res.json();
  }

  // ---- the notice above the composer and the live status ----------------------------------------------------------------------

  function showNotice(message, retry = null, owner = null) {
    ui.noticeText.textContent = message;
    ui.noticeRetry.hidden = !retry;
    state.noticeRetry = retry;
    state.noticeOwner = owner;
    ui.notice.hidden = false;
  }

  function clearNotice() {
    if (doc.activeElement && ui.notice.contains(doc.activeElement)) ui.prompt.focus();   // its Retry or Dismiss is about to go
    ui.notice.hidden = true;
    ui.noticeText.textContent = '';
    ui.noticeRetry.hidden = true;
    state.noticeRetry = null;
    state.noticeOwner = null;
  }

  // The status region says a thing only when it changes, so the same word is not read twice. again: a message that is news every time
  // (a refusal) is said again, the same words too.
  function announce(message, again = false) {
    if (message === state.said && !again) return;
    if (again) ui.status.textContent = '';   // a live region reads what is put into it: empty, then the words, so that the same words are new
    state.said = message;
    ui.status.textContent = message;
  }

  // ---- the conversation --------------------------------------------------------------------------------------------------------

  const cardCtx = (n) => ({
    loadImage: (url) => api.loadImage(url, auth()),
    copy,
    showModels: () => openDrawer({ models: true }),
    labelNames: Array.isArray(state.models?.label_names) ? state.models.label_names : undefined,
    ui: state.session.ui,
    turn: n,
  });

  function copy(value) {
    const clipboard = win.navigator?.clipboard;
    return typeof clipboard?.writeText === 'function' ? clipboard.writeText(value) : Promise.reject(new Error('no clipboard'));
  }

  // Where the page scrolls: #conversation, or the document where the layout is short (styles.css, max-height 480px).
  const scroller = {
    short: () => !!win.matchMedia?.(SHORT_VIEWPORT)?.matches,
    near() {
      if (this.short()) {
        const root = doc.documentElement ?? {};
        return nearBottom({ scrollTop: win.scrollY ?? root.scrollTop, clientHeight: win.innerHeight ?? root.clientHeight, scrollHeight: root.scrollHeight });
      }
      return nearBottom(ui.conversation);
    },
    toEnd(node) {
      if (this.short()) node?.scrollIntoView?.({ block: 'end' });
      else ui.conversation.scrollTop = ui.conversation.scrollHeight;
    },
  };

  function makeTurn(session, n) {
    const turn = { n, id: null, session, view: initialView(null), userEl: null, card: null, text: '', image: null, file: null,
                   controller: null, poller: null, streaming: false, polling: false, stopping: false, left: false, settled: false,
                   dirty: false, reloading: false };
    turn.draw = () => paint(turn);   // one function per card, so scheduleRender draws it at most once a frame
    return turn;
  }

  // Replaces the card with one built from the latest view, keeps focus on the same control and the scroll where the
  // reader had it, and says what changed. Called at most once per animation frame per card (scheduleRender), and straight
  // away when the turn ends.
  function paint(turn) {
    if (turn.left || !turn.dirty || !turn.card) return;
    turn.dirty = false;
    const old = turn.card;
    const near = scroller.near();
    const active = doc.activeElement;
    const key = active && old.contains(active) ? focusKey(active) : null;   // a control of this card only
    const next = renderAssistantCard(turn.view, cardCtx(turn.n));
    if (old.parentNode) old.replaceWith(next); else ui.conversation.append(next);
    turn.card = next;
    if (key) restoreFocus(next, key);
    announce(statusText(turn.view));
    if (near) scroller.toEnd(next);
  }

  function refreshUserBubble(turn, options) {
    if (!turn.userEl) return;
    const next = renderUserTurn({ text: turn.text, image: turn.image, options }, cardCtx(turn.n));
    turn.userEl.replaceWith(next);
    turn.userEl = next;
  }

  // The state of the controls that depend on whether a turn runs.
  function syncControls() {
    ui.send.disabled = state.busy || state.loading;
    ui.stop.hidden = !state.busy;
    ui.exports.hidden = !state.session.id;
    fields.runningNote.hidden = !state.busy;
    syncRerun();
  }

  // The file name of the newest user turn that had an image: the one a text-only turn runs again (the server's rule, post_message).
  function newestImage() {
    const turns = state.session.turns;
    for (let i = turns.length - 1; i >= 0; i--) if (turns[i].image?.filename) return turns[i].image.filename;
    return '';
  }

  // Under the image well, when Send with no new image would run the chat's last X-ray again: with a file attached, in an empty chat
  // and while a turn runs (or a chat loads) it says nothing. Information, not an alert.
  function syncRerun() {
    const name = state.file || state.busy || state.loading ? '' : newestImage();
    ui.rerun.hidden = !name;
    if (name) {
      ui.rerun.textContent = `No new image: Send re-runs ${name} with these settings.`;
      ui.send.setAttribute('aria-describedby', 'rerun-hint');   // Send says what it will do, to whoever reaches it by tab
    } else {
      ui.send.removeAttribute('aria-describedby');   // a hidden note that is still named would still be read out
    }
  }

  // aria-busy on #conversation goes on at once and comes off a frame after the page stops being busy: put on and taken off inside
  // one task, the attribute would never reach assistive technology. A frame that finds a turn running again leaves it on.
  function markBusy(on) {
    if (on) ui.conversation.setAttribute('aria-busy', 'true');
    else frame(() => { if (!state.busy) ui.conversation.removeAttribute('aria-busy'); });
  }

  function setBusy(on) {
    const wasBusy = state.busy;
    state.busy = on;
    if (on) {
      markBusy(true);
    } else {
      markBusy(false);
      ui.stop.disabled = false;
      ui.stop.textContent = 'Stop';
      if (wasBusy && (doc.activeElement === ui.stop || doc.activeElement === ui.send)) ui.prompt.focus();   // the control that had focus is going away
    }
    syncControls();
  }

  // ---- one turn: the stream, the watchdog, the poll, Stop ----------------------------------------------------------------

  function stopTicker() {
    if (state.ticker !== null) cancelEvery(state.ticker);
    state.ticker = null;
  }

  function startTicker(turn) {
    stopTicker();
    state.ticker = every(() => {
      if (watchdog.tick(turn.view.status) === 'poll') turn.controller?.abort();   // the stream goes; the poll starts as it unwinds
    }, TICK_MS);
  }

  // The turn is over: draw its final card, free the composer, refresh the sidebar (its title and count changed).
  function settle(turn) {
    if (turn.settled) return;
    turn.settled = true;
    stopTicker();
    watchdog.settle();
    try {
      paint(turn);
    } finally {   // a card that cannot be drawn must not leave the composer locked
      if (state.turn === turn) {
        state.turn = null;
        setBusy(false);
      }
      if (state.noticeOwner === turn) clearNotice();   // "couldn't stop": the turn ended by itself
      detach(refreshSessions());
    }
  }

  function feed(turn, event) {
    if (turn.left) return;   // events already read when the view was left still come out of the stream: they are not drawn
    try {
      if (!turn.card && typeof event.data?.message_id === 'string') accept(turn, event.data.message_id);   // a server that sent no X-Message-Id
      turn.view = applyEvent(turn.view, event);
      turn.dirty = true;
      if (event.event === 'message_start') refreshUserBubble(turn, event.data.options);   // the options the server resolved, commands included
      if (turn.view.status !== 'running') settle(turn);
      else scheduleRender(turn.draw);
    } catch (err) {   // a reducer or a builder that threw on this event: say so, log it, and carry on with the next one
      report(err);
      showNotice(GENERIC_ERROR);
    }
  }

  // The poll that takes over from the stream: from the last seq the view has, until the turn leaves running. A terminal
  // error (not the transient ones api.js retries) is shown with a Retry that starts the poll again from that seq.
  async function pollTurn(turn) {
    stopTicker();
    const poller = new AbortController();
    turn.poller = poller;
    turn.polling = true;
    try {
      for await (const event of fromTransport(api.pollMessage({ messageId: turn.id, after: turn.view.lastSeq, ...auth(), signal: poller.signal }))) {
        feed(turn, event);
      }
    } catch (err) {
      // A poll that fails after the turn's own message_stop was read (the server stores that before it marks the message finished, so
      // the poll that read it asks once more) has nothing left to ask: a Retry for a finished turn would be a button that does nothing.
      if (turn.left || turn.settled || isAbort(err)) return;
      watchdog.failed();
      if (state.turn === turn) {   // Stop works again, whatever it said: pressing it follows the turn from here
        turn.stopping = false;
        ui.stop.disabled = false;
        ui.stop.textContent = 'Stop';
      }
      showNotice(errorMessage(err), () => follow(turn));
      return;
    } finally {
      turn.polling = false;
    }
    if (turn.left || turn.settled) return;
    if (turn.view.status === 'running') {   // the poll ended and the log has no end: do not leave the page waiting for it
      showNotice('The turn ended without a result. Reload the chat to see it.');
      settle(turn);
    }
  }

  // Polls a turn that nothing is following (its poll failed for good), from the last seq the view has. Retry in the notice and
  // Stop both come here. A turn that was left is not followed, so a Retry that comes late starts nothing; one that is already
  // being polled needs nothing. (A turn that has ended never gets here: pollTurn offers no Retry for a failure after the end, and
  // Stop looks at `settled` before it asks.)
  function follow(turn) {
    if (turn.left || turn.polling) return;
    if (watchdog.retry() !== 'poll') watchdog.resume();
    if (state.noticeRetry) clearNotice();
    detach(pollTurn(turn));
  }

  async function runTurn(session, { text: note, file, options }) {
    const turn = makeTurn(session, session.turns.length + 1);
    turn.text = note;
    turn.file = file;
    try {
      if (file) {
        const url = urls.createObjectURL(file);
        session.blobUrls.add(url);
        turn.image = { url, filename: file.name };
      }
      turn.userEl = renderUserTurn({ text: note, image: turn.image, options }, cardCtx(turn.n));
      session.turns.push(turn);
      ui.conversation.append(turn.userEl);
      scroller.toEnd(turn.userEl);
      state.turn = turn;
      syncControls();

      const form = new FormData();
      if (file) form.append('image', file, file.name);
      form.append('text', note);
      form.append('options', JSON.stringify(options));

      turn.controller = new AbortController();
      ui.stop.disabled = true;   // until the server has the turn and has said its id: that is what Stop cancels
      turn.streaming = true;
      // No stall clock yet: accept() starts it when the server has the turn. Until then the request is an upload, and the server
      // decoding and saving it, which can take longer than 3 s on a slow tunnel, and is not silence.
      for await (const event of fromTransport(api.streamTurn({
        sessionId: session.id, form, ...auth(), signal: turn.controller.signal, onMessageId: (id) => accept(turn, id),
      }))) {
        watchdog.bytes();
        feed(turn, event);
      }
    } catch (err) {
      if (!turn.id) { refuse(turn, err); return; }   // it never started: no id was ever given
      if (!isAbort(err) && !turn.left) report(err);   // a network error mid-stream: the poll below carries on
    } finally {
      turn.streaming = false;
    }
    if (turn.left || turn.settled) return;
    if (!turn.id) { refuse(turn, new Error('The server closed the connection before it accepted the turn.')); return; }
    if (watchdog.ended(turn.view.status) === 'poll') await pollTurn(turn);
  }

  // The page's own work around a turn must not stop the turn, nor the work after it: a failure is logged and said, and the caller goes on.
  function attempt(work) {
    try {
      work();
    } catch (err) {
      report(err);
      showNotice(GENERIC_ERROR);
    }
  }

  // What was sent is spent: the text of the note that was sent, and the file that was attached. What was typed or attached since
  // Send is the next turn's, and stays.
  function spendComposer(turn) {
    const typed = ui.prompt.value ?? '';
    ui.prompt.value = typed.startsWith(turn.text) ? typed.slice(turn.text.length) : typed;
    if (state.file === turn.file) clearFile();
  }

  // The server has the turn: Stop works, and the 3 s of silence that mean "poll instead" are counted from here. Stop, the clock, the
  // ticker and the spending of the composer come before any drawing, and each is guarded: a page that cannot draw its card still
  // follows the turn to its end, and does not send the same note and image again at the next Send.
  function accept(turn, id) {
    if (turn.left) return;   // the user moved to another chat while the upload was in flight: its turn is theirs to find later
    turn.id = id;
    turn.view = initialView(id);
    ui.stop.disabled = false;
    watchdog.arm();
    startTicker(turn);
    attempt(() => spendComposer(turn));
    attempt(() => {   // a card that is made but not yet in the page is put there by its first paint
      turn.card = renderAssistantCard(turn.view, cardCtx(turn.n));
      ui.conversation.append(turn.card);
      scroller.toEnd(turn.card);
      announce(statusText(turn.view));
    });
  }

  // The turn never started (the request was refused, or the network was down): take its user turn back out, show why,
  // and leave the composer as the user had it, so Send tries again with nothing to re-enter.
  function refuse(turn, err) {
    if (turn.left) return;   // the user left this chat: nothing here to undo, and the ticker is now another turn's
    stopTicker();
    watchdog.settle();
    turn.userEl?.remove();
    const at = turn.session.turns.indexOf(turn);
    if (at >= 0) turn.session.turns.splice(at, 1);
    if (turn.image) { urls.revokeObjectURL(turn.image.url); turn.session.blobUrls.delete(turn.image.url); }
    if (state.turn === turn) state.turn = null;
    setBusy(false);
    showNotice(errorMessage(err));
    if (!expected(err)) report(err);
  }

  async function send() {
    if (state.busy || state.loading) return;
    const note = ui.prompt.value ?? '';
    const file = state.file;
    const session = state.session;
    if (!file && !session.turns.length) { showNotice('Attach an X-ray first.'); return; }
    clearNotice();
    const options = optionsFromSettings(state.settings, state.models);
    setBusy(true);
    ui.stop.disabled = true;
    try {
      if (!session.id) await createSession(session);
      if (state.session !== session) return;   // the user moved to another chat while this one was being made
      await runTurn(session, { text: note, file, options });
    } catch (err) {   // the chat could not be made, or something broke that runTurn does not deal with itself
      if (state.session === session) {
        showNotice(errorMessage(err));
        if (!expected(err)) report(err);
      }
    } finally {
      if (state.busy && !state.turn && state.session === session) setBusy(false);   // nothing is following a turn: the composer is not locked
    }
  }

  // An empty chat becomes a session at its first turn. The address follows without a navigation (replaceState fires no
  // hashchange), so the turn that is about to stream is not torn down by the route handler.
  async function createSession(session) {
    const made = await request('/v1/sessions', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
    session.id = made.id;
    if (state.session === session) {   // not if the user has gone to another chat meanwhile: the address is theirs
      setAddress(`#/s/${made.id}`);
      syncControls();
    }
    detach(refreshSessions());
  }

  // Stop: ask the server to cancel, then drop the stream. The turn's own message_stop (status aborted) settles the card;
  // the poll that follows the dropped stream is what brings it.
  async function stopTurn() {
    const turn = state.turn;
    if (!turn || !turn.id || turn.stopping || turn.settled) return;
    turn.stopping = true;
    ui.stop.disabled = true;
    ui.stop.textContent = 'Stopping…';
    try {
      await api.cancelMessage({ messageId: turn.id, ...auth() });
    } catch (err) {
      turn.stopping = false;
      if (turn.settled || turn.left || state.turn !== turn) return;   // it ended while the cancel was on its way: nothing to report
      ui.stop.disabled = false;
      ui.stop.textContent = 'Stop';
      showNotice(errorMessage(markNetwork(err)), null, turn);   // the turn's end takes it down again
      return;
    }
    if (turn.settled || turn.left) return;
    if (state.noticeOwner === turn) clearNotice();   // an earlier Stop failed and this one did not
    if (turn.streaming) turn.controller?.abort();   // the poll that brings the turn's own message_stop starts as the stream unwinds
    else follow(turn);                              // nothing is following it (its poll failed for good): follow it again
  }

  // Leaving a session (or a token change): the stream and the poll are dropped, the turn is not cancelled and is not
  // followed. Opening that session later finds it running and polls it.
  function leaveTurn() {
    stopTicker();
    watchdog.settle();
    const turn = state.turn;
    if (!turn) return;
    turn.left = true;
    turn.controller?.abort();
    turn.poller?.abort();
    state.turn = null;
  }

  // A turn found running when its session was opened: no stream to rejoin, so poll it from the last seq it has.
  function resume(turn) {
    state.turn = turn;
    setBusy(true);
    ui.stop.disabled = false;
    watchdog.resume();
    detach(pollTurn(turn));
  }

  // ---- sessions: the view, the sidebar, the routes ------------------------------------------------------------------------

  function resetView(id) {
    leaveTurn();
    api.clearImageCache();
    for (const url of state.session.blobUrls) urls.revokeObjectURL(url);
    state.session = { id, turns: [], ui: new Map(), blobUrls: new Set() };
    ui.conversation.replaceChildren();
    announce('');
    clearNotice();
    setBusy(false);
  }

  // One message with its events, as {log} or {error}: a message that cannot be read costs its own card and not the whole chat.
  const loadLog = (messageId) => request(`/v1/messages/${encodeURIComponent(messageId)}?after=0`).then((log) => ({ log }), (error) => ({ error }));

  async function loadSession(id) {
    const session = await request(`/v1/sessions/${encodeURIComponent(id)}`);
    const assistants = session.messages.filter((m) => m.role === 'assistant');
    const loaded = await Promise.all(assistants.map((m) => loadLog(m.id)));
    return { session, logs: new Map(assistants.map((m, i) => [m.id, loaded[i]])) };
  }

  // The card of a turn whose log could not be read: what is wrong, and a Retry that reads it again.
  function failedCard(turn, error) {
    return el('article', { class: 'card', 'aria-label': `Assistant report, turn ${turn.n}`, 'data-message-id': turn.id },
      el('p', { class: 'note error' }, `Couldn't load this turn. ${errorMessage(error)}`),
      el('button', { type: 'button', 'aria-label': `Retry loading turn ${turn.n}`, onclick: () => detach(reloadTurn(turn)) }, 'Retry'));
  }

  async function reloadTurn(turn) {
    if (turn.reloading) return;
    turn.reloading = true;
    const view = turn.session;
    let loaded;
    try {
      loaded = await loadLog(turn.id);
    } finally {
      turn.reloading = false;
    }
    if (state.session !== view) return;   // the user has gone to another chat meanwhile
    let next;
    if (loaded.error) {
      next = failedCard(turn, loaded.error);
    } else {
      turn.view = loaded.log.events.reduce(applyEvent, initialView(turn.id));
      next = renderAssistantCard(turn.view, cardCtx(turn.n));
    }
    const hadFocus = doc.activeElement && turn.card.contains(doc.activeElement);
    turn.card.replaceWith(next);
    turn.card = next;
    if (hadFocus) qa(next, 'button')[0]?.focus();   // the Retry that had focus is gone: the first control of what replaced it
    if (!loaded.error && view.turns[view.turns.length - 1] === turn && turn.view.status === 'running' && loaded.log.status === 'running') resume(turn);
  }

  // Draws every turn from the stored log: the same reducers and builders the live stream goes through.
  function showSession({ session, logs }) {
    const view = state.session;
    const pairs = [];
    let user = null;
    for (const m of session.messages) {
      if (m.role === 'user') {
        if (user) pairs.push({ user, assistant: null });   // a user message with no answer: drawn alone
        user = m;
      } else {
        pairs.push({ user, assistant: m });
        user = null;
      }
    }
    if (user) pairs.push({ user, assistant: null });
    const nodes = [];
    for (const { user: u, assistant: a } of pairs) {
      const turn = makeTurn(view, view.turns.length + 1);
      if (u) {
        const shown = userTurnMessage(u, a);
        turn.text = shown.text;      // a turn that was still queued has no events: its message_start rebuilds this bubble,
        turn.image = shown.image;    // and rebuilds it from these
        turn.userEl = renderUserTurn(shown, cardCtx(turn.n));
        nodes.push(turn.userEl);
      }
      if (a) {
        turn.id = a.id;
        const loaded = logs.get(a.id);
        if (loaded?.log) {
          turn.view = loaded.log.events.reduce(applyEvent, initialView(a.id));
          turn.card = renderAssistantCard(turn.view, cardCtx(turn.n));
        } else {
          turn.card = failedCard(turn, loaded?.error);
        }
        nodes.push(turn.card);
      }
      view.turns.push(turn);
    }
    markBusy(true);   // a screen reader does not read the whole history as it goes in
    try {
      ui.conversation.replaceChildren(...nodes);
    } finally {
      markBusy(false);
    }
    scroller.toEnd(nodes[nodes.length - 1]);
    const last = view.turns[view.turns.length - 1];
    const lastLog = last?.id ? logs.get(last.id)?.log : null;
    if (lastLog && last.view.status === 'running' && lastLog.status === 'running') resume(last);
  }

  async function openSession(id) {
    const gen = ++state.gen;
    resetView(id);
    state.loading = true;
    syncControls();
    renderSessions();
    try {
      const data = await loadSession(id);
      if (gen !== state.gen) return;
      showSession(data);
    } catch (err) {
      if (gen !== state.gen) return;
      if (err?.status === 404) {   // not there (any more): the newest chat that is, from a fresh list, or an empty one
        state.missing = id;
        state.sessionsLoaded = false;
        state.loading = false;
        setAddress('#/');
        await handleRoute();
        showNotice(errorMessage(err));   // after the fallback has opened its chat, which clears the notices of the one before
      } else {
        state.session.id = null;   // nothing is open: its link (or the route) opens it again, and so does the Retry
        renderSessions();          // and the sidebar does not mark it as the page
        showNotice(errorMessage(err), () => detach(openSession(id)));
      }
    } finally {
      if (gen === state.gen) {
        state.loading = false;
        syncControls();
      }
    }
  }

  function openNew() {
    ++state.gen;
    resetView(null);
    ui.prompt.value = '';
    clearFile();
    state.loading = false;
    syncControls();
    renderSessions();
  }

  // Shows the route the address names. Home has no view of its own: it becomes the newest session, or #/new when there is
  // none (and never the session that was just found missing).
  async function handleRoute() {
    let route = parseRoute(win.location?.hash);
    if (route.kind === 'home') {
      const gen = ++state.gen;
      if (!state.sessionsLoaded) await refreshSessions();
      if (gen !== state.gen) return;
      const newest = state.sessions.find((s) => s.id !== state.missing);
      const hash = newest ? `#/s/${newest.id}` : '#/new';
      setAddress(hash);
      route = parseRoute(hash);
    }
    const key = route.kind === 'session' ? `s/${route.id}` : 'new';
    if (state.route !== null && state.route !== key) clearNotice();   // a notice belongs to the route it was raised on
    state.route = key;
    if (route.kind === 'session') {
      if (state.session.id !== route.id) await openSession(route.id);
    } else if (state.session.id !== null || state.session.turns.length) {
      openNew();
    }
  }

  // The address changes without a history entry and without a hashchange: the caller shows the route itself.
  function setAddress(hash) {
    if (typeof win.history?.replaceState === 'function') win.history.replaceState(null, '', hash);
    else if (win.location) win.location.hash = hash;
  }

  // A link the user follows: a new history entry. The same address fires no hashchange, so it is handled here.
  function navigate(hash) {
    if (!win.location || win.location.hash === hash) return handleRoute();
    win.location.hash = hash;
    return undefined;
  }

  async function refreshSessions({ more = false } = {}) {
    const shown = state.sessions.length;
    const onMore = more && doc.activeElement === ui.more;
    if (more) {
      if (state.loadingMore) return;   // a double click asks once
      state.loadingMore = true;
      ui.more.disabled = true;
    }
    try {
      const limit = more ? PAGE : Math.min(200, Math.max(PAGE, state.sessions.length));
      const query = `limit=${limit}${more && state.nextCursor ? `&cursor=${encodeURIComponent(state.nextCursor)}` : ''}`;
      const page = await request(`/v1/sessions?${query}`);
      state.sessions = more ? [...state.sessions, ...page.sessions] : page.sessions;
      state.nextCursor = page.next_cursor ?? null;
      state.sessionsLoaded = true;
      renderSessions();
    } catch (err) {
      state.sessionsLoaded = true;
      showNotice(errorMessage(err));
    } finally {
      if (more) {
        state.loadingMore = false;
        ui.more.disabled = false;
        if (onMore) (ui.more.hidden ? qa(ui.list, 'li a')[shown] : ui.more)?.focus();   // More keeps focus; the last page hands it to its first new row
      }
    }
  }

  const rowOf = (id) => qa(ui.list, 'li').find((li) => li.getAttribute('data-session') === id);   // an id is compared, never put into a selector

  function renderSessions() {
    const active = doc.activeElement;
    const held = active && ui.list.contains(active) ? { id: active.closest('li')?.getAttribute('data-session'), tag: active.localName } : null;
    ui.list.replaceChildren(...state.sessions.map((s) => {
      const title = sessionTitle(s);
      const current = s.id === state.session.id;
      return el('li', { 'data-session': s.id },
        el('a', {
          href: `#/s/${s.id}`, 'aria-current': current ? 'page' : null,
          onclick: () => {
            closeSidebar();
            if (win.location?.hash === `#/s/${s.id}`) detach(handleRoute());   // the same address fires no hashchange: a chat that failed to open tries again here
          },
        },
          el('span', { class: 'session-title' }, title), el('span', { class: 'session-meta' }, sessionMeta(s))),
        el('button', { type: 'button', class: 'session-delete', 'aria-label': `Delete chat: ${title}`, onclick: () => detach(deleteSession(s)) }, '✕'));
    }));
    ui.more.hidden = !state.nextCursor;
    syncControls();
    if (held?.id) rowOf(held.id)?.querySelector(held.tag)?.focus();   // a rebuilt list must not drop the keyboard's place
  }

  async function deleteSession(session) {
    if (!confirmed(`Delete "${sessionTitle(session)}"? This cannot be undone.`)) return;
    try {
      await request(`/v1/sessions/${encodeURIComponent(session.id)}`, { method: 'DELETE' });
    } catch (err) {
      if (err?.status !== 404) { showNotice(errorMessage(err)); return; }   // already gone: the list below says so
    }
    const here = state.session.id === session.id;
    if (here) resetView(null);   // its image URLs and its stream go with it
    await refreshSessions();
    if (here) {
      setAddress('#/');
      await handleRoute();   // home: the newest chat that is left, or an empty one
    }
    const open = state.session.id ? rowOf(state.session.id) : null;
    (open?.querySelector('a') ?? ui.newChat).focus();   // the Delete that had focus is gone: the chat that is open, or New chat
  }

  // ---- export --------------------------------------------------------------------------------------------------------------------

  async function exportSession(format) {
    const id = state.session.id;
    if (!id) return;
    try {
      const res = await request(`/v1/sessions/${encodeURIComponent(id)}/export?format=${format}`, { raw: true });
      const blob = await res.blob().catch((err) => { throw markNetwork(err); });
      const url = urls.createObjectURL(blob);
      const link = doc.createElement('a');
      link.setAttribute('href', url);
      link.setAttribute('download', exportFilename(res.headers?.get?.('Content-Disposition'), id, format));
      doc.body.append(link);
      link.click();
      link.remove();
      later(() => urls.revokeObjectURL(url), REVOKE_MS);
    } catch (err) {
      showNotice(errorMessage(err));
    }
  }

  // ---- the health strip ----------------------------------------------------------------------------------------------------------

  function setHealth(message, down) {
    if (ui.health.textContent !== message) ui.health.textContent = message;
    if (down) ui.health.setAttribute('data-state', 'down'); else ui.health.removeAttribute('data-state');
  }

  async function checkHealth() {
    const controller = new AbortController();
    const timer = later(() => controller.abort(), HEALTH_TIMEOUT_MS);   // a half-open tunnel never answers and never fails by itself
    try {
      const res = await doFetch('/healthz', { cache: 'no-store', signal: controller.signal });
      if (!res.ok) throw new Error('unhealthy');
      const h = await res.json();
      state.healthFailures = 0;
      if (text(h?.mode)) ui.badge.textContent = h.mode;
      setHealth(healthText(h), false);
    } catch {
      state.healthFailures += 1;
      setHealth('server restarting…', true);
    } finally {
      cancelLater(timer);
    }
    later(() => detach(checkHealth()), nextHealthDelay(state.healthFailures));
  }

  // ---- the shell: sidebar and drawer ---------------------------------------------------------------------------------------------

  function openSidebar() {
    doc.body.classList.add('sidebar-open');
    ui.toggle.setAttribute('aria-expanded', 'true');
  }

  function closeSidebar({ restore = true } = {}) {
    if (!doc.body.classList.contains('sidebar-open')) return false;
    doc.body.classList.remove('sidebar-open');
    ui.toggle.setAttribute('aria-expanded', 'false');
    if (restore) ui.toggle.focus();
    return true;
  }

  function openDrawer({ models = false } = {}) {
    if (ui.drawer.hidden) {
      const from = doc.activeElement;
      state.drawerOpener = from && from !== doc.body ? from : ui.settings;
      clearSaved();   // "Settings saved." that is still on the page is about a Save that is behind the user now
      ui.drawer.hidden = false;
      ui.settings.setAttribute('aria-expanded', 'true');
      ui.drawerClose.focus();
    }
    if (models) ui.modelsSection.scrollIntoView?.({ block: 'start' });
  }

  function closeDrawer({ restore = true } = {}) {
    if (ui.drawer.hidden) return false;
    discardDrafts();
    ui.drawer.hidden = true;
    ui.settings.setAttribute('aria-expanded', 'false');
    if (restore) {
      const to = state.drawerOpener && doc.body.contains(state.drawerOpener) ? state.drawerOpener : ui.settings;
      to.focus();
    }
    state.drawerOpener = null;
    return true;
  }

  function onKeydown(event) {
    if (event.key !== 'Escape') return;
    const drawer = closeDrawer();
    const sidebar = closeSidebar({ restore: !drawer });
    if (drawer || sidebar) event.preventDefault();
  }

  // ---- the settings drawer ----------------------------------------------------------------------------------------------------------

  const fields = { inputs: {}, errors: {} };
  const drafts = new Set();   // the number fields whose text the settings did not take: not a whole number inside the bounds

  // true when the settings were stored; false when the browser would not (the note under the fields says so).
  function persist() {
    const ok = saveSettings(storage, state.settings);
    fields.storageNote.hidden = ok;
    return ok;
  }

  // sync false leaves the drawer's controls as they are: the field being typed in is not rewritten under the hand that types.
  function update(patch, { sync = true } = {}) {
    state.settings = sanitize({ ...state.settings, ...patch });
    const ok = persist();
    if (sync) syncDrawer();
    renderChips();
    return ok;
  }

  // A number field's text is a draft until it is a whole number inside the bounds: the field then has the line that says what it takes
  // under it (aria-invalid, and aria-describedby naming the line), and the settings, the chips and the stored copy stay as they were.
  function showDraft(key, on) {
    const input = fields.inputs[key];
    if (on) drafts.add(key); else drafts.delete(key);
    fields.errors[key].hidden = !on;
    if (on) input.setAttribute('aria-invalid', 'true'); else input.removeAttribute('aria-invalid');
    describe(input, fields.errors[key].id, on);
  }

  // Closing the drawer puts the saved value back in every field that holds a draft, and takes its line down.
  function discardDrafts() {
    if (!drafts.size) return;
    for (const key of [...drafts]) showDraft(key, false);
    syncDrawer();
  }

  // The visible live region beside the chips, emptied (and its timer stopped).
  function clearSaved() {
    if (state.savedTimer !== null) cancelLater(state.savedTimer);
    state.savedTimer = null;
    ui.saved.textContent = '';
    ui.saved.removeAttribute('data-state');
  }

  // "Settings saved." in that region for SAVED_MS; a second Save starts the time over. warn: where the browser would not store the settings
  // the same region says that instead (the storage note), in the warning's colour, and never "saved".
  function confirmSaved(message, warn = false) {
    clearSaved();
    ui.saved.textContent = message;
    if (warn) ui.saved.setAttribute('data-state', 'warn');
    state.savedTimer = later(clearSaved, SAVED_MS);
  }

  // Save, which is the form's submit: Enter in a text field submits it too. Every number field the page lets the user edit must hold a
  // whole number inside its bounds, else nothing is stored and the drawer stays open on the first that does not (with every wrong
  // field's line showing). Otherwise the settings are stored, the drawer closes to where it was opened from, and the page says so beside
  // the chips. Where the browser will not store them, the drawer closes all the same (one that stays open looks dead), the settings still
  // apply in this tab, and what the page says beside the chips is the storage note, in the warning's colour, and never "saved".
  function saveDrawer() {
    const wrong = NUMBER_KEYS.filter((key) => !fields.inputs[key].disabled && wholeNumber(fields.inputs[key].value, BOUNDS[key]) === null);
    if (wrong.length) {
      for (const key of wrong) showDraft(key, true);
      const first = fields.inputs[wrong[0]];
      first.focus();
      first.select();   // a field that already has focus gets no focus event, so its bad text is selected here: the next key replaces it
      announce(rangeMessage(wrong[0]), true);   // a refusal is news every time, the same words too
      return false;
    }
    const patch = {};
    for (const key of NUMBER_KEYS) {
      if (!fields.inputs[key].disabled) patch[key] = wholeNumber(fields.inputs[key].value, BOUNDS[key]);
      showDraft(key, false);
    }
    const stored = update(patch);
    closeDrawer();
    confirmSaved(stored ? SAVED_TEXT : STORAGE_TEXT, !stored);
    return true;
  }

  function buildDrawer() {
    // A number field: a whole number inside the bounds applies at once, so that the chips follow the keys; anything else stays as typed,
    // with its line under it, until it is right or the drawer closes. Nothing is clamped (the "1" on the way to "150" is not 16), and
    // nothing wrong is left in force: the "30" on the way to "300" does apply, and when the text stops being a number the setting can take,
    // the setting goes back to what it was before this edit (an edit runs from the focus, or the first key, to the change or the blur).
    const number = (key) => {
      const input = el('input', { type: 'number', min: BOUNDS[key][0], max: BOUNDS[key][1], step: 1, inputmode: 'numeric', 'data-setting': key });
      fields.inputs[key] = input;
      fields.errors[key] = el('p', { class: 'field-error', id: `${key}-error`, hidden: true }, rangeMessage(key));
      const read = () => {
        const n = wholeNumber(input.value, BOUNDS[key]);
        showDraft(key, n === null);
        return n;
      };
      let before;   // the setting when this edit began; undefined between edits
      const begin = () => { if (before === undefined) before = state.settings[key]; };
      input.addEventListener('input', () => {
        begin();   // here as well as at the focus: a key can come with no focus event before it
        const n = read();
        if (n !== null) {
          if (n !== state.settings[key]) update({ [key]: n }, { sync: false });
        } else if (state.settings[key] !== before) {
          update({ [key]: before }, { sync: false });
        }
      });
      input.addEventListener('change', () => {
        const n = read();
        if (n !== null) update({ [key]: n });   // writes the number back as the setting has it: "0120" is 120
        before = undefined;                      // committed: the next edit starts from what the setting is now
      });
      // Typing replaces what the field holds: it is selected whole when it gets focus, by Tab or by a click. A click ends in a mouseup,
      // which would put the caret back and undo that, so the mouseup of the click that gave the field its focus is cancelled; without
      // this, "150" is typed after the "100" that was there, and 100150 is out of range. Only that click: the first click after a Tab, and
      // every later click in a field that has focus, place the caret as they always do.
      let pressed = false;   // a pointer went down on the field while it had no focus: the focus that follows is that click's
      let guard = false;     // and the mouseup after it must not put the caret back over the selection
      const down = () => { pressed = doc.activeElement !== input; };
      input.addEventListener('pointerdown', down);
      input.addEventListener('mousedown', down);   // a browser with no pointer events
      input.addEventListener('focus', () => {
        input.select();
        guard = pressed;
        pressed = false;
        begin();
      });
      input.addEventListener('mouseup', (event) => {
        if (guard) event.preventDefault();
        guard = pressed = false;
      });
      input.addEventListener('blur', () => {
        guard = pressed = false;
        before = undefined;
      });
      return input;
    };
    const choice = (key, ...options) => {
      const select = el('select', { 'data-setting': key }, ...options.map(([value, label]) => el('option', { value }, label)));
      select.addEventListener('change', () => update({ [key]: select.value }));
      return select;
    };
    const toggle = (key) => {
      const input = el('input', { type: 'checkbox', 'data-setting': key });
      input.addEventListener('change', () => update({ [key]: !!input.checked }));
      return input;
    };
    fields.model = el('select', {});
    fields.model.addEventListener('change', () => update({ model: fields.model.value === state.models?.default_model ? '' : fields.model.value }));
    fields.decode = choice('decode', ['beam', 'Beam search'], ['greedy', 'Greedy']);
    fields.beam = number('beam_size');
    fields.tokens = number('max_new_tokens');
    fields.cached = toggle('cached_decode');
    fields.cachedNote = el('p', { class: 'note', id: 'cached-note' }, 'This model has no decode cache, so decoding runs uncached.');
    fields.cached.setAttribute('aria-describedby', 'cached-note');
    fields.compile = toggle('compile');
    fields.compileRow = el('label', { class: 'check' }, fields.compile, 'Compile the model (torch.compile)');
    fields.kImages = number('k_images');
    fields.kReports = number('k_reports');
    fields.retrievalNote = el('p', { class: 'note', id: 'retrieval-note', hidden: true },
      'This server has no retrieval gallery, so similar X-rays and matching reports are skipped.');
    fields.label = toggle('label');
    fields.labelsNote = el('p', { class: 'note', id: 'labels-note', hidden: true }, 'This server has no CheXbert labeller, so labels are skipped.');
    fields.repair = toggle('display_repair');
    fields.stop = toggle('stop_on_repeat');
    fields.stopHint = el('p', { class: 'hint', id: 'stop-hint' }, 'Off: the published protocol, which always decodes the whole token budget.');
    fields.stop.setAttribute('aria-describedby', 'stop-hint');
    fields.applyNote = el('p', { class: 'hint', id: 'apply-note' }, 'Changes apply from your next Send.');
    fields.runningNote = el('p', { class: 'hint', id: 'running-note', hidden: true }, 'The running turn keeps the settings it started with.');
    fields.mode = el('input', { type: 'text', readonly: true });
    fields.token = el('input', { type: 'password', autocomplete: 'off', spellcheck: 'false' });
    fields.token.addEventListener('change', () => {
      const token = fields.token.value.trim();
      if (token === state.settings.token) return;
      update({ token });
      detach(reloadAll());
    });
    fields.storageNote = el('p', { class: 'hint', hidden: true }, STORAGE_TEXT);
    fields.save = el('button', { type: 'submit', id: 'drawer-save' }, 'Save');
    // novalidate: the browser's own bubbles for min and max would stop the submit before the page could say what its fields take.
    fields.form = el('form', { id: 'settings-form', novalidate: true },
      el('label', {}, 'Model', fields.model),
      el('label', {}, 'Decode', fields.decode),
      el('label', {}, rangeLabel('Beam size', 'beam_size'), fields.beam), fields.errors.beam_size,
      el('label', {}, rangeLabel('Token budget', 'max_new_tokens'), fields.tokens), fields.errors.max_new_tokens,
      el('label', { class: 'check' }, fields.cached, 'Cached decode'), fields.cachedNote,
      fields.compileRow,
      el('label', {}, rangeLabel('Similar images', 'k_images'), fields.kImages), fields.errors.k_images,
      el('label', {}, rangeLabel('Matching reports', 'k_reports'), fields.kReports), fields.errors.k_reports, fields.retrievalNote,
      el('label', { class: 'check' }, fields.label, 'CheXbert labels'), fields.labelsNote,
      el('label', { class: 'check' }, fields.repair, 'Display repair'),
      el('label', { class: 'check' }, fields.stop, 'Stop when the report starts repeating'), fields.stopHint,
      el('label', {}, 'Mode', fields.mode),
      el('label', {}, 'Access token', fields.token),
      el('p', { class: 'hint' }, 'Saved in this browser on this device, because you typed it here.'),
      fields.storageNote,
      el('div', { class: 'drawer-actions' }, fields.save));
    fields.form.addEventListener('submit', (event) => {
      event.preventDefault();   // the page never reloads
      saveDrawer();
    });
    ui.drawerClose = el('button', { type: 'button', id: 'drawer-close', 'aria-label': 'Close settings', onclick: () => closeDrawer() }, '✕');
    ui.modelsSection = el('section', { id: 'models-section', 'aria-label': 'Models' });
    ui.drawer.replaceChildren(
      el('div', { class: 'drawer-head' }, el('h2', {}, 'Settings'), ui.drawerClose),
      fields.applyNote, fields.runningNote, fields.form, ui.modelsSection);
    fields.token.value = state.settings.token;
    syncDrawer();
  }

  // The drawer's controls from the settings and the models: the model list, what the chosen card allows, the numbers (a draft is kept).
  function fillModels() {
    const cards = cardsOf(state.models);
    fields.model.replaceChildren(...(cards.length ? cards.map((c) => el('option', { value: c.name }, c.name)) : [el('option', { value: '' }, 'Server default')]));
    ui.modelsSection.replaceChildren(el('h3', {}, 'Models'),
      ...cards.flatMap((c) => [el('h4', {}, c.name), detailTable(c, `${c.name} details`)]));
    syncDrawer();
  }

  // aria-describedby is a list of ids: this adds or takes out one of them, and names a note only while the note is shown (a hidden
  // element that is named is still read out).
  function describe(field, id, on) {
    const ids = (field.getAttribute('aria-describedby') ?? '').split(/\s+/).filter((x) => x && x !== id);
    if (on) ids.push(id);
    if (ids.length) field.setAttribute('aria-describedby', ids.join(' ')); else field.removeAttribute('aria-describedby');
  }

  function syncDrawer() {
    const s = state.settings;
    const card = chosenCard(s, state.models);
    const cachedOk = !(card && card.cached_decode_available === false);
    const retrieval = serverHas(state.models, 'retrieval');
    const labelling = serverHas(state.models, 'labels');
    fields.model.value = card?.name ?? '';
    fields.decode.value = s.decode;
    fields.beam.disabled = s.decode === 'greedy';
    for (const field of [fields.kImages, fields.kReports]) {   // a stage the server skips has nothing to set; the note says so
      field.disabled = !retrieval;
      describe(field, 'retrieval-note', !retrieval);
    }
    for (const key of NUMBER_KEYS) {
      const field = fields.inputs[key];
      if (field.disabled) showDraft(key, false);              // nothing can be typed in it: there is no draft to keep
      if (!drafts.has(key)) field.value = String(s[key]);    // a draft is not rewritten under the hand that is typing it
    }
    fields.cached.disabled = !cachedOk;
    fields.cached.checked = cachedOk && s.cached_decode;
    fields.cachedNote.hidden = cachedOk;
    fields.compileRow.hidden = state.models?.allow_compile !== true;
    fields.compile.checked = s.compile && state.models?.allow_compile === true;
    fields.retrievalNote.hidden = retrieval;
    fields.label.disabled = !labelling;
    fields.label.checked = labelling && s.label;
    describe(fields.label, 'labels-note', !labelling);
    fields.labelsNote.hidden = labelling;
    fields.repair.checked = s.display_repair;
    fields.stop.checked = s.stop_on_repeat;
    fields.mode.value = text(state.models?.mode) || '—';
  }

  function renderChips() {
    const options = optionsFromSettings(state.settings, state.models);
    if (!serverHas(state.models, 'retrieval')) { delete options.k_images; delete options.k_reports; }   // that stage is skipped: no "k 4/3" to show
    ui.chips.replaceChildren(...optionChips(options).map((c) => el('span', {}, c)));
  }

  async function refreshModels() {
    try {
      state.models = await request('/v1/models');
      if (text(state.models.mode)) ui.badge.textContent = state.models.mode;
      fillModels();
      renderChips();
    } catch (err) {
      showNotice(errorMessage(err));
    }
  }

  // A new token, or a new client id, can change what the server shows: everything is fetched again, and what the old
  // identity had open (images, a running stream) is dropped first.
  async function reloadAll() {
    leaveTurn();
    api.clearImageCache();
    const id = state.session.id;
    await Promise.all([refreshModels(), refreshSessions()]);
    if (id) await openSession(id); else await handleRoute();
  }

  // ---- the composer (minimal: P6-A replaces it with composer.js) ----------------------------------------------------------------------

  function clearFile() {
    if (doc.activeElement && ui.preview.contains(doc.activeElement)) ui.well.focus();   // its Remove is about to go
    if (state.previewUrl) urls.revokeObjectURL(state.previewUrl);
    state.previewUrl = null;
    state.file = null;
    ui.preview.replaceChildren();
    ui.preview.hidden = true;
    syncControls();   // with no file attached, Send may be a re-run
  }

  function attach(file) {
    const problem = checkImageFile(file);
    if (problem) { showNotice(problem); return; }
    clearNotice();
    clearFile();
    state.file = file;
    state.previewUrl = urls.createObjectURL(file);
    const kb = Number.isFinite(file.size) ? ` · ${Math.max(1, Math.round(file.size / 1024))} KB` : '';
    ui.preview.replaceChildren(
      el('img', { src: state.previewUrl, alt: '' }),
      el('span', {}, `${file.name || 'image'}${kb}`),
      el('button', { type: 'button', 'aria-label': 'Remove attached image', onclick: () => clearFile() }, 'Remove'));
    ui.preview.hidden = false;
    syncControls();   // a new image: Send sends it
  }

  function wireComposer() {
    ui.composer.addEventListener('submit', (event) => { event.preventDefault(); detach(send()); });
    ui.prompt.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' || event.shiftKey || event.isComposing || event.keyCode === 229) return;   // Shift+Enter is a newline; an IME's Enter is not a send
      event.preventDefault();
      detach(send());
    });
    ui.well.addEventListener('click', () => ui.file.click());
    ui.well.addEventListener('keydown', (event) => {
      if (event.key !== 'Enter' && event.key !== ' ') return;
      event.preventDefault();
      ui.file.click();
    });
    ui.file.addEventListener('change', () => {
      const file = ui.file.files?.[0];
      if (file) attach(file);
      ui.file.value = '';   // choosing the same file again must fire change again
    });
    ui.well.addEventListener('dragover', (event) => { event.preventDefault(); ui.well.classList.add('dragging'); });
    ui.well.addEventListener('dragleave', () => ui.well.classList.remove('dragging'));
    ui.well.addEventListener('drop', (event) => {
      event.preventDefault();
      ui.well.classList.remove('dragging');
      const file = event.dataTransfer?.files?.[0];
      if (file) attach(file);
    });
    for (const type of ['dragover', 'drop']) {   // a file dropped anywhere else must not navigate the page away from the chat
      doc.body.addEventListener(type, (event) => { if (Array.from(event.dataTransfer?.types ?? []).includes('Files')) event.preventDefault(); });
    }
    ui.prompt.addEventListener('paste', (event) => {
      const file = Array.from(event.clipboardData?.files ?? []).find((f) => IMAGE_TYPES.includes(f.type));
      if (!file) return;
      event.preventDefault();
      attach(file);
    });
    ui.stop.addEventListener('click', () => detach(stopTurn()));
  }

  // ---- building and starting ----------------------------------------------------------------------------------------------------------

  function build() {
    ui.list.setAttribute('role', 'list');   // list-style: none drops the semantics in Safari
    ui.status = doc.getElementById('status') ?? el('div', { id: 'status', class: 'visually-hidden', role: 'status', 'aria-live': 'polite' });
    if (!ui.status.parentNode) doc.body.append(ui.status);

    ui.noticeText = el('p', {});
    ui.noticeRetry = el('button', { type: 'button', hidden: true, onclick: () => state.noticeRetry?.() }, 'Retry');
    ui.notice = el('div', { id: 'notice', role: 'alert', hidden: true }, ui.noticeText, ui.noticeRetry,
      el('button', { type: 'button', 'aria-label': 'Dismiss', onclick: () => clearNotice() }, '✕'));
    ui.rerun = el('p', { id: 'rerun-hint', hidden: true });
    // The page's own status line is visually hidden; this one is for the eye as well: "Settings saved." beside the chips, for a few seconds.
    // It is there from the start (empty), so that a screen reader has the region before its first message.
    ui.saved = el('p', { id: 'saved', role: 'status', 'aria-live': 'polite' });
    const rows = Array.from(ui.composer.children);
    rows.splice(rows.indexOf(ui.well) + 1, 0, ui.rerun);   // next to the image well
    rows.splice(rows.indexOf(ui.chips) + 1, 0, ui.saved);   // and under the chips
    ui.composer.replaceChildren(ui.notice, ...rows);

    ui.exports = el('div', { id: 'exports', role: 'group', 'aria-label': 'Export this chat', hidden: true },
      el('button', { type: 'button', 'data-format': 'json', onclick: () => detach(exportSession('json')) }, 'Export JSON'),
      el('button', { type: 'button', 'data-format': 'md', onclick: () => detach(exportSession('md')) }, 'Export Markdown'));
    ui.more = el('button', { type: 'button', id: 'session-more', hidden: true, onclick: () => detach(refreshSessions({ more: true })) }, 'More');
    ui.sidebar.replaceChildren(ui.newChat, ui.exports, ui.list, ui.more);

    buildDrawer();
    renderChips();
  }

  function wire() {
    ui.toggle.addEventListener('click', () => { if (doc.body.classList.contains('sidebar-open')) closeSidebar(); else openSidebar(); });
    doc.body.addEventListener('click', (event) => { if (event.target === doc.body) closeSidebar(); });   // the scrim is the body's ::after
    ui.settings.addEventListener('click', () => { if (ui.drawer.hidden) openDrawer(); else closeDrawer(); });
    doc.body.addEventListener('keydown', onKeydown);
    ui.newChat.addEventListener('click', () => { closeSidebar(); detach(navigate('#/new')); });
    win.addEventListener?.('hashchange', () => detach(handleRoute()));
    wireComposer();
  }

  async function start() {
    build();
    wire();
    detach(checkHealth());
    await Promise.all([refreshModels(), refreshSessions()]);
    await handleRoute();
  }

  return { start, send, stopTurn, handleRoute, navigate, refreshSessions, exportSession, openDrawer, closeDrawer, reloadAll, state, ui };
}

// ---- start ----------------------------------------------------------------------------------------------------------------------------

export function browserEnv() {
  return { document, window: globalThis, storage: browserStorage(globalThis), api: realApi };
}

// Only a page that has the composer starts the app: a test importing this file gets the helpers and nothing else.
if (typeof document !== 'undefined' && document.getElementById?.('composer')) {
  createApp(browserEnv()).start().catch((err) => { try { console.error(err); } catch { /* no console */ } });
}
