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

export const BOUNDS = Object.freeze({   // the server's bounds (app/schemas.py Options)
  beam_size: [1, 8], max_new_tokens: [16, 200], k_images: [0, 12], k_reports: [0, 10],
});
// The published protocol, as every Options default has it. model '' is the server's default; token is the access token.
export const DEFAULT_SETTINGS = Object.freeze({
  model: '', decode: 'beam', beam_size: 3, max_new_tokens: 100, cached_decode: true, compile: false,
  k_images: 4, k_reports: 3, label: true, display_repair: false, token: '',
});

const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);

function clampInt(value, [low, high], fallback) {
  const n = typeof value === 'number' ? value : typeof value === 'string' && value.trim() !== '' ? Number(value) : NaN;
  return Number.isFinite(n) ? Math.min(high, Math.max(low, Math.round(n))) : fallback;
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

// What a failed request says to the user. err is what api.js throws: an Error with `status` and `body` (the server's
// envelope {type: "error", error: {type, message}}), or what fetch throws when the network fails (a TypeError).
export function errorMessage(err) {
  if (typeof err === 'string') return text(err) || 'Something went wrong.';
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
  if (err instanceof TypeError) return 'Cannot reach the server. Check the connection and try again.';
  return clip(text(err?.message) || 'Something went wrong.');
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

const isAbort = (err) => err?.name === 'AbortError';

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
  const every = (fn, ms) => (env.setInterval ? env.setInterval(fn, ms) : globalThis.setInterval(fn, ms));
  const cancelEvery = (id) => (env.clearInterval ? env.clearInterval(id) : globalThis.clearInterval(id));
  const confirmed = (message) => (env.confirm ? env.confirm(message) : typeof win.confirm === 'function' ? win.confirm(message) : true);
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
    noticeRetry: null,
    missing: null,           // a session the server said it does not have: home never picks it
  };

  const auth = () => ({ token: state.settings.token, clientId: state.clientId });
  const report = (err) => { try { console.error(err); } catch { /* no console */ } };
  const detach = (promise) => { Promise.resolve(promise).catch(report); };   // a handler's async work: failures are shown, never unhandled

  // ---- requests --------------------------------------------------------------------------------------------------------------

  // A JSON request with the page's auth headers. A refusal throws what api.js throws: an Error with status and body.
  async function request(path, { method = 'GET', body, headers, raw = false, signal } = {}) {
    const res = await doFetch(path, { method, body, signal, headers: { ...authHeaders(state.settings.token, state.clientId), ...headers } });
    if (!res.ok) throw Object.assign(new Error('request refused'), { status: res.status, body: await res.json().catch(() => null) });
    if (raw) return res;
    return res.status === 204 ? null : res.json();
  }

  // ---- the notice above the composer and the live status ----------------------------------------------------------------------

  function showNotice(message, retry = null) {
    ui.noticeText.textContent = message;
    ui.noticeRetry.hidden = !retry;
    state.noticeRetry = retry;
    ui.notice.hidden = false;
  }

  function clearNotice() {
    if (doc.activeElement && ui.notice.contains(doc.activeElement)) ui.prompt.focus();   // its Retry or Dismiss is about to go
    ui.notice.hidden = true;
    ui.noticeText.textContent = '';
    ui.noticeRetry.hidden = true;
    state.noticeRetry = null;
  }

  // The status region says a thing only when it changes, so the same word is not read twice.
  function announce(message) {
    if (message === state.said) return;
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
    const turn = { n, id: null, session, view: initialView(null), userEl: null, card: null, text: '', image: null,
                   controller: null, poller: null, stopping: false, left: false, settled: false, dirty: false };
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
  }

  function setBusy(on) {
    const wasBusy = state.busy;
    state.busy = on;
    if (on) {
      ui.conversation.setAttribute('aria-busy', 'true');
    } else {
      ui.conversation.removeAttribute('aria-busy');
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
    paint(turn);
    if (state.turn === turn) {
      state.turn = null;
      setBusy(false);
    }
    detach(refreshSessions());
  }

  function feed(turn, event) {
    if (turn.left) return;   // events already read when the view was left still come out of the stream: they are not drawn
    if (!turn.card && typeof event.data?.message_id === 'string') accept(turn, event.data.message_id);   // a server that sent no X-Message-Id
    turn.view = applyEvent(turn.view, event);
    turn.dirty = true;
    if (event.event === 'message_start') refreshUserBubble(turn, event.data.options);   // the options the server resolved, commands included
    if (turn.view.status !== 'running') settle(turn);
    else scheduleRender(turn.draw);
  }

  // The poll that takes over from the stream: from the last seq the view has, until the turn leaves running. A terminal
  // error (not the transient ones api.js retries) is shown with a Retry that starts the poll again from that seq.
  async function pollTurn(turn) {
    stopTicker();
    const poller = new AbortController();
    turn.poller = poller;
    try {
      for await (const event of api.pollMessage({ messageId: turn.id, after: turn.view.lastSeq, ...auth(), signal: poller.signal })) {
        feed(turn, event);
      }
    } catch (err) {
      if (turn.left || isAbort(err)) return;
      watchdog.failed();
      showNotice(errorMessage(err), () => {
        if (watchdog.retry() !== 'poll') return;
        clearNotice();
        detach(pollTurn(turn));
      });
      return;
    }
    if (turn.left || turn.settled) return;
    if (turn.view.status === 'running') {   // the poll ended and the log has no end: do not leave the page waiting for it
      showNotice('The turn ended without a result. Reload the chat to see it.');
      settle(turn);
    }
  }

  async function runTurn(session, { text: note, file, options }) {
    const turn = makeTurn(session, session.turns.length + 1);
    turn.text = note;
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

    const controller = new AbortController();
    turn.controller = controller;
    watchdog.arm();
    startTicker(turn);
    ui.stop.disabled = true;   // until the server has the turn and has said its id: that is what Stop cancels
    try {
      for await (const event of api.streamTurn({
        sessionId: session.id, form, ...auth(), signal: controller.signal, onMessageId: (id) => accept(turn, id),
      })) {
        watchdog.bytes();
        feed(turn, event);
      }
    } catch (err) {
      if (!turn.id) { refuse(turn, err); return; }   // it never started: no id was ever given
      if (!isAbort(err) && !turn.left) report(err);   // a network error mid-stream: the poll below carries on
    }
    if (turn.left || turn.settled) return;
    if (watchdog.ended(turn.view.status) === 'poll') await pollTurn(turn);
  }

  // The server has the turn: from here the composer's text and file are spent.
  function accept(turn, id) {
    if (turn.left) return;   // the user moved to another chat while the upload was in flight: its turn is theirs to find later
    turn.id = id;
    turn.view = initialView(id);
    turn.card = renderAssistantCard(turn.view, cardCtx(turn.n));
    ui.conversation.append(turn.card);
    ui.prompt.value = '';
    clearFile();
    ui.stop.disabled = false;
    scroller.toEnd(turn.card);
    announce(statusText(turn.view));
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
    } catch (err) {
      setBusy(false);
      showNotice(errorMessage(err));
      return;
    }
    if (state.session !== session) return;   // the user moved to another chat while this one was being made
    await runTurn(session, { text: note, file, options });
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
      ui.stop.disabled = false;
      ui.stop.textContent = 'Stop';
      showNotice(errorMessage(err));
      return;
    }
    turn.controller?.abort();
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

  async function loadSession(id) {
    const session = await request(`/v1/sessions/${encodeURIComponent(id)}`);
    const assistants = session.messages.filter((m) => m.role === 'assistant');
    const logs = await Promise.all(assistants.map((m) => request(`/v1/messages/${encodeURIComponent(m.id)}?after=0`)));
    return { session, logs: new Map(logs.map((log) => [log.id, log])) };
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
        turn.userEl = renderUserTurn(userTurnMessage(u, a), cardCtx(turn.n));
        nodes.push(turn.userEl);
      }
      if (a) {
        turn.id = a.id;
        turn.view = (logs.get(a.id)?.events ?? []).reduce(applyEvent, initialView(a.id));
        turn.card = renderAssistantCard(turn.view, cardCtx(turn.n));
        nodes.push(turn.card);
      }
      view.turns.push(turn);
    }
    ui.conversation.replaceChildren(...nodes);
    scroller.toEnd(nodes[nodes.length - 1]);
    const last = view.turns[view.turns.length - 1];
    if (last?.id && last.view.status === 'running' && logs.get(last.id)?.status === 'running') resume(last);
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
      }
      showNotice(errorMessage(err));   // after the fallback has opened its chat, which clears the notices of the one before
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
    try {
      const limit = more ? PAGE : Math.min(200, Math.max(PAGE, state.sessions.length));
      const query = `limit=${limit}${more && state.nextCursor ? `&cursor=${encodeURIComponent(state.nextCursor)}` : ''}`;
      const page = await request(`/v1/sessions?${query}`);
      const shown = state.sessions.length;
      const onMore = doc.activeElement === ui.more;
      state.sessions = more ? [...state.sessions, ...page.sessions] : page.sessions;
      state.nextCursor = page.next_cursor ?? null;
      state.sessionsLoaded = true;
      renderSessions();
      if (more && onMore && ui.more.hidden) qa(ui.list, 'li a')[shown]?.focus();   // that was the last page: More is gone, so focus moves on
    } catch (err) {
      state.sessionsLoaded = true;
      showNotice(errorMessage(err));
    }
  }

  function renderSessions() {
    const active = doc.activeElement;
    const held = active && ui.list.contains(active) ? { id: active.closest('li')?.getAttribute('data-session'), tag: active.localName } : null;
    ui.list.replaceChildren(...state.sessions.map((s) => {
      const title = sessionTitle(s);
      const current = s.id === state.session.id;
      return el('li', { 'data-session': s.id },
        el('a', { href: `#/s/${s.id}`, 'aria-current': current ? 'page' : null, onclick: () => closeSidebar() },
          el('span', { class: 'session-title' }, title), el('span', { class: 'session-meta' }, sessionMeta(s))),
        el('button', { type: 'button', class: 'session-delete', 'aria-label': `Delete chat: ${title}`, onclick: () => detach(deleteSession(s)) }, '✕'));
    }));
    ui.more.hidden = !state.nextCursor;
    syncControls();
    if (held?.id) qa(ui.list, `li[data-session="${held.id}"] ${held.tag}`)[0]?.focus();   // a rebuilt list must not drop the keyboard's place
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
  }

  // ---- export --------------------------------------------------------------------------------------------------------------------

  async function exportSession(format) {
    const id = state.session.id;
    if (!id) return;
    try {
      const res = await request(`/v1/sessions/${encodeURIComponent(id)}/export?format=${format}`, { raw: true });
      const blob = await res.blob();
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
    try {
      const res = await doFetch('/healthz', { cache: 'no-store' });
      if (!res.ok) throw new Error('unhealthy');
      const h = await res.json();
      state.healthFailures = 0;
      if (text(h?.mode)) ui.badge.textContent = h.mode;
      setHealth(healthText(h), false);
    } catch {
      state.healthFailures += 1;
      setHealth('server restarting…', true);
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
      ui.drawer.hidden = false;
      ui.settings.setAttribute('aria-expanded', 'true');
      ui.drawerClose.focus();
    }
    if (models) ui.modelsSection.scrollIntoView?.({ block: 'start' });
  }

  function closeDrawer({ restore = true } = {}) {
    if (ui.drawer.hidden) return false;
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

  const fields = {};

  function persist() {
    const ok = saveSettings(storage, state.settings);
    fields.storageNote.hidden = ok;
  }

  function update(patch) {
    state.settings = sanitize({ ...state.settings, ...patch });
    persist();
    syncDrawer();
    renderChips();
  }

  function buildDrawer() {
    const number = (key) => {
      const input = el('input', { type: 'number', min: BOUNDS[key][0], max: BOUNDS[key][1], step: 1, inputmode: 'numeric' });
      input.addEventListener('change', () => update({ [key]: clampInt(input.value, BOUNDS[key], state.settings[key]) }));
      return input;
    };
    const choice = (key, ...options) => {
      const select = el('select', {}, ...options.map(([value, label]) => el('option', { value }, label)));
      select.addEventListener('change', () => update({ [key]: select.value }));
      return select;
    };
    const toggle = (key) => {
      const input = el('input', { type: 'checkbox' });
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
    fields.label = toggle('label');
    fields.repair = toggle('display_repair');
    fields.mode = el('input', { type: 'text', readonly: true });
    fields.token = el('input', { type: 'password', autocomplete: 'off', spellcheck: 'false' });
    fields.token.addEventListener('change', () => {
      const token = fields.token.value.trim();
      if (token === state.settings.token) return;
      update({ token });
      detach(reloadAll());
    });
    fields.storageNote = el('p', { class: 'hint', hidden: true }, 'Browser storage is unavailable, so these settings last until the page closes.');
    ui.drawerClose = el('button', { type: 'button', id: 'drawer-close', 'aria-label': 'Close settings', onclick: () => closeDrawer() }, '✕');
    ui.modelsSection = el('section', { id: 'models-section', 'aria-label': 'Models' });
    ui.drawer.replaceChildren(
      el('div', { class: 'drawer-head' }, el('h2', {}, 'Settings'), ui.drawerClose),
      el('label', {}, 'Model', fields.model),
      el('label', {}, 'Decode', fields.decode),
      el('label', {}, 'Beam size', fields.beam),
      el('label', {}, 'Token budget', fields.tokens),
      el('label', { class: 'check' }, fields.cached, 'Cached decode'), fields.cachedNote,
      fields.compileRow,
      el('label', {}, 'Similar images (k_images)', fields.kImages),
      el('label', {}, 'Matching reports (k_reports)', fields.kReports),
      el('label', { class: 'check' }, fields.label, 'CheXbert labels'),
      el('label', { class: 'check' }, fields.repair, 'Display repair'),
      el('label', {}, 'Mode', fields.mode),
      el('label', {}, 'Access token', fields.token),
      el('p', { class: 'hint' }, 'Saved in this browser on this device, because you typed it here.'),
      fields.storageNote,
      ui.modelsSection);
    fields.token.value = state.settings.token;
    syncDrawer();
  }

  // The drawer's controls from the settings and the models: the model list, what the chosen card allows, the clamped numbers.
  function fillModels() {
    const cards = cardsOf(state.models);
    fields.model.replaceChildren(...(cards.length ? cards.map((c) => el('option', { value: c.name }, c.name)) : [el('option', { value: '' }, 'Server default')]));
    ui.modelsSection.replaceChildren(el('h3', {}, 'Models'),
      ...cards.flatMap((c) => [el('h4', {}, c.name), detailTable(c, `${c.name} details`)]));
    syncDrawer();
  }

  function syncDrawer() {
    const s = state.settings;
    const card = chosenCard(s, state.models);
    const cachedOk = !(card && card.cached_decode_available === false);
    fields.model.value = card?.name ?? '';
    fields.decode.value = s.decode;
    fields.beam.value = String(s.beam_size);
    fields.beam.disabled = s.decode === 'greedy';
    fields.tokens.value = String(s.max_new_tokens);
    fields.cached.disabled = !cachedOk;
    fields.cached.checked = cachedOk && s.cached_decode;
    fields.cachedNote.hidden = cachedOk;
    fields.compileRow.hidden = state.models?.allow_compile !== true;
    fields.compile.checked = s.compile && state.models?.allow_compile === true;
    fields.kImages.value = String(s.k_images);
    fields.kReports.value = String(s.k_reports);
    fields.label.checked = s.label;
    fields.repair.checked = s.display_repair;
    fields.mode.value = text(state.models?.mode) || '—';
  }

  function renderChips() {
    ui.chips.replaceChildren(...optionChips(optionsFromSettings(state.settings, state.models)).map((c) => el('span', {}, c)));
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
    ui.composer.replaceChildren(ui.notice, ...Array.from(ui.composer.children));

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
