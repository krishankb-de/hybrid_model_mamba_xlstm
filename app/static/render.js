// app/static/render.js — the card builders: one turn of the chat, drawn from the view that state.js folds from the log.
//
// Every builder is a pure function of a view (and a ctx) and returns a fresh element, so an update is a rebuild: the
// caller replaces the old card with renderAssistantCard(view, ctx), at most once per animation frame (scheduleRender).
// Nothing here touches the page (no #conversation, no fetch, no storage) and nothing writes to the view, whose arrays
// and objects alias the stored log. Every field of a view may be missing: a public turn lacks what the server redacts,
// a turn that has not started has no card and no stage, a skipped stage leaves its fields null or empty. A builder
// draws what it has and nothing for the rest, never the word "undefined".
//
// Text is always a text node: el() appends a string as text and never parses it, and no script in app/static has an
// HTML-string API (tests/frontend/render.test.mjs greps for them), so report text (model output) and a note (user
// input) cannot become markup. The only URLs set on an element are blob: object URLs and data: images; an image comes
// through ctx.loadImage because an <img src> cannot carry the bearer token.
//
// ctx = { loadImage, openViewer, copy, showModels, labelNames, ui }. Every member is optional and a missing callback
// hides its control.
//   loadImage(path) -> Promise<object URL>   api.js loadImage with the page's auth: a user turn's thumbnail
//   openViewer(image)                        a click on the thumbnail; image is what renderUserTurn was given
//   copy(text)                               the report's Copy button
//   showModels()                             the provenance link, which opens /v1/models in the drawer
//   labelNames: [14 names]                   CHEXBERT_14 order, from /v1/models; else the order of view.labels
//   ui: Map                                  keeps the open stage details and Show raw, per message id, across the
//                                            whole-card replace; without it a re-rendered card starts closed
import { STAGES, initialView, labelsPending, stageState } from './state.js';

export { STAGES };

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
    else if (v !== false && v != null) node.setAttribute(k, v === true ? '' : v);
  }
  node.append(...children.filter((c) => c != null));
  return node;
}

// ---- small helpers ---------------------------------------------------------------------------------------------------

const NONE = '—';          // what a null or an empty value reads as
const CLIP = 200;          // a string longer than this is clipped in a detail table until "show more"
const MAX_DEPTH = 4;       // how deep a detail table nests before it shows the rest as JSON text

const isNum = (v) => typeof v === 'number' && Number.isFinite(v);
const str = (v) => (typeof v === 'string' ? v : isNum(v) ? String(v) : '');   // text comes from a string or a number, nothing else
const said = (prefix, value, suffix = '') => (str(value) ? `${prefix}${str(value)}${suffix}` : null);   // null when there is none
const isObject = (v) => v !== null && typeof v === 'object' && !Array.isArray(v);
const isPrimitive = (v) => v === null || (typeof v !== 'object' && typeof v !== 'function');

// The view with every field present: what a builder reads, so a field that is missing is the empty value.
function whole(view) {
  const base = initialView(null);
  const given = isObject(view) ? view : {};
  return Object.fromEntries(Object.keys(base).map((k) => [k, given[k] ?? base[k]]));
}

const noticesOf = (v) => (Array.isArray(v.notices) ? v.notices.filter((n) => isObject(n) && (str(n.message) || str(n.code))) : []);

// What outlives a re-render: the stage details that are open and Show raw. ctx.ui keeps it per message id; without
// a store the record lives and dies with the card.
function uiOf(view, ctx) {
  const store = ctx?.ui;
  const fresh = () => ({ open: new Set(), raw: false });
  if (!store || typeof store.get !== 'function' || typeof store.set !== 'function') return fresh();
  let record = store.get(view.id);
  if (!record) {
    record = fresh();
    store.set(view.id, record);
  }
  return record;
}

// ---- the timeline -----------------------------------------------------------------------------------------------------

export function renderTimeline(view, ctx) {
  const v = whole(view);
  const ui = uiOf(v, ctx);
  return el('ol', { class: 'timeline', 'aria-label': 'Pipeline stages' }, ...STAGES.map((s) => {
    const st = stageState(v, s);
    const state = typeof st.state === 'string' ? st.state : 'pending';
    const ms = isNum(st.ms) ? Math.round(st.ms) : null;
    const why = str(st.skipped);
    // A stop is the user's own doing, so it is said on screen; every other skip reason is in the spoken label.
    const label = state === 'done' ? (ms == null ? `${s} · done` : `${s} · ${ms} ms`)
                : state === 'skipped' ? `${s} · skipped${why === 'stopped' ? ' (stopped)' : ''}` : s;
    const spoken = state === 'done' ? (ms == null ? `${s}, done` : `${s}, done, ${ms} milliseconds`)
                 : state === 'skipped' ? `${s}, skipped${why ? `: ${why}` : ''}` : `${s}, ${state}`;
    const li = el('li', { 'data-stage': s, 'data-state': state, 'aria-label': spoken, tabindex: 0 }, label);
    if (isObject(st.detail) && Object.keys(st.detail).length) {
      const table = detailTable(st.detail, `${s} details`);
      const toggle = () => { if (li.classList.toggle('open')) ui.open.add(s); else ui.open.delete(s); };
      if (ui.open.has(s)) li.classList.add('open');
      li.addEventListener('click', (e) => { if (!table.contains(e.target)) toggle(); });   // not a click inside the table
      li.addEventListener('keydown', (e) => {   // Enter or Space on the item itself, not on a button inside its table
        if (e.target === li && (e.key === 'Enter' || e.key === ' ')) {
          e.preventDefault();
          toggle();
        }
      });
      li.append(table);
    }
    return li;
  }));
}

// A stage's detail as a table: a row per key; an array of objects as a nested table with a row per object and a column
// per key; an object as a table of its own; a long string clipped with "show more". label names the table.
export function detailTable(detail, label) {
  const entries = isObject(detail) ? Object.entries(detail) : [];
  return el('table', { 'aria-label': label }, el('tbody', {}, ...entries.map(([key, value]) => row(key, value, 1))));
}

const row = (key, value, depth) => el('tr', {}, el('th', { scope: 'row' }, key), el('td', {}, valueNode(value, depth)));

const numberText = (n) => (Number.isInteger(n) ? String(n) : Number.isFinite(n) ? String(Number(n.toPrecision(6))) : NONE);

function valueNode(v, depth) {
  if (v == null) return NONE;
  if (typeof v === 'string') return v.length > CLIP ? clipped(v) : v;
  if (typeof v === 'number') return numberText(v);
  if (typeof v !== 'object') return String(v);
  if (depth >= MAX_DEPTH) return valueNode(jsonText(v), depth);
  if (Array.isArray(v)) return arrayNode(v, depth);
  const entries = Object.entries(v);
  if (!entries.length) return NONE;
  return el('table', { class: 'nested' }, el('tbody', {}, ...entries.map(([key, value]) => row(key, value, depth + 1))));
}

function jsonText(v) {
  try { return JSON.stringify(v); } catch { return NONE; }
}

function arrayNode(items, depth) {
  if (!items.length) return NONE;
  const inline = (x) => isPrimitive(x) && !(typeof x === 'string' && x.length > CLIP);
  if (items.every(inline)) return items.map((x) => (x == null ? NONE : typeof x === 'number' ? numberText(x) : String(x))).join(', ');
  if (items.every(isObject)) return recordsTable(items, depth);
  return el('ol', { class: 'values' }, ...items.map((x) => el('li', {}, valueNode(x, depth + 1))));
}

function recordsTable(items, depth) {
  const columns = [...new Set(items.flatMap((item) => Object.keys(item)))];
  if (!columns.length) return NONE;
  return el('table', { class: 'nested' },
    el('thead', {}, el('tr', {}, ...columns.map((c) => el('th', { scope: 'col' }, c)))),
    el('tbody', {}, ...items.map((item) => el('tr', {}, ...columns.map(
      (c) => el('td', {}, valueNode(Object.hasOwn(item, c) ? item[c] : undefined, depth + 1)))))));
}

// The first CLIP characters and a button that shows the rest, and hides it again.
function clipped(text) {
  let end = CLIP;
  const last = text.charCodeAt(end - 1);
  if (last >= 0xd800 && last <= 0xdbff) end -= 1;   // not between the halves of a surrogate pair
  const head = `${text.slice(0, end)}…`;
  const shown = el('span', { class: 'clip-text' }, head);
  const more = el('button', { type: 'button', class: 'more', 'aria-expanded': 'false' }, 'show more');
  more.addEventListener('click', () => {
    const open = more.getAttribute('aria-expanded') !== 'true';
    shown.textContent = open ? text : head;
    more.textContent = open ? 'show less' : 'show more';
    more.setAttribute('aria-expanded', String(open));
  });
  return el('span', { class: 'clip' }, shown, ' ', more);
}

// ---- notes: warnings, a stop, an error ---------------------------------------------------------------------------------

const errorText = (v) => `Error: ${str(v.error?.message) || 'the turn failed'}`;

export function renderNotes(view) {
  const v = whole(view);
  const notes = noticesOf(v).map((n) => el('p', { class: 'note notice', 'data-code': str(n.code) || null }, str(n.message) || str(n.code)));
  if (v.error || v.status === 'error') notes.push(el('p', { class: 'note error' }, errorText(v)));
  else if (v.status === 'aborted') notes.push(el('p', { class: 'note stopped' }, 'Turn stopped'));
  return el('div', { class: 'notes', hidden: !notes.length }, ...notes);
}

// What the live region says: the stage that runs, then how the turn ended. Not set on the card itself: the page owns
// the announcement (a polite status node that changes only when this text does).
export function statusText(view) {
  const v = whole(view);
  if (v.error || v.status === 'error') return errorText(v);
  if (v.status === 'aborted') return 'Turn stopped';
  if (v.status === 'done') return v.report ? 'Report ready' : str(noticesOf(v).at(-1)?.message) || 'Turn finished';
  const running = STAGES.find((s) => stageState(v, s).state === 'running');
  if (running) return `${running} running`;
  return v.lastSeq ? 'Working' : 'Queued';
}

// ---- the report --------------------------------------------------------------------------------------------------------

// The text as sections: before the first header, then one per literal "Findings:" or "Impression:" in text order.
export function splitReport(text) {
  const parts = str(text).split(/(Findings:|Impression:)/);
  const sections = [];
  if (parts[0].trim()) sections.push({ title: null, body: parts[0].trim() });
  for (let i = 1; i < parts.length; i += 2) sections.push({ title: parts[i].slice(0, -1), body: parts[i + 1].trim() });
  return sections;
}

export function renderReport(view, ctx) {
  const v = whole(view);
  const cx = ctx ?? {};
  const ui = uiOf(v, cx);
  const raw = str(v.report) || str(v.displayReport);
  const shown = str(v.displayReport) || raw;   // the stream has only the raw snapshot until message_stop brings the copy
  if (!raw) return el('div', { class: 'report', hidden: true });
  const bodyOf = () => (ui.raw
    ? el('pre', { class: 'report-raw' }, raw)
    : el('div', { class: 'report-body' }, ...splitReport(shown).map(({ title, body }) => el('section', { class: 'report-section' },
      title ? el('h3', {}, title) : null, body ? el('p', {}, body) : null))));
  let body = bodyOf();
  const toggle = el('button', { type: 'button', 'aria-pressed': String(ui.raw) }, 'Show raw');
  toggle.addEventListener('click', () => {
    ui.raw = !ui.raw;
    toggle.setAttribute('aria-pressed', String(ui.raw));
    const next = bodyOf();
    body.replaceWith(next);
    body = next;
  });
  const copy = typeof cx.copy === 'function'
    ? el('button', { type: 'button', onclick: () => cx.copy(ui.raw ? raw : shown) }, 'Copy') : null;
  return el('div', { class: v.provisional ? 'report provisional' : 'report' },
    body,
    v.truncated ? el('p', { class: 'note truncated' }, 'Report stopped at the token budget mid-sentence') : null,
    el('div', { class: 'report-actions' }, copy, toggle));
}

// ---- labels and scores -------------------------------------------------------------------------------------------------

const isOn = (x) => x === 1 || x === true;   // a label is 1 or 0

const CHIP_CLASS = { positive: 'chip label positive', negative: 'chip label negative', unknown: 'chip label unknown' };
const CHIP_SPOKEN = { positive: 'positive', negative: 'negative', unknown: 'not reported' };
const SCORES = [['ROUGE-L', 'rouge_l'], ['BLEU-1', 'bleu_1'], ['BLEU-4', 'bleu_4'], ['CheXbert-14 F1', 'chexbert_14_micro_f1']];
const REFERENCES = { user: 'your reference', test_split: 'test-split reference' };

export function renderLabels(view, ctx) {
  const v = whole(view);
  const cx = ctx ?? {};
  const note = (text) => el('p', { class: 'note' }, text);
  let body = null;
  let marked = false;
  if (isObject(v.labels)) {
    ({ node: body, marked } = chipList(v, cx));
  } else if (labelsPending(v)) {
    body = note('labelling…');
  } else {
    const st = stageState(v, 'label');   // a stop or an error settles it; a settled turn that never got here shows nothing
    const why = str(st.skipped);
    if (st.state === 'skipped') body = note(`labels unavailable${why ? ` (${why})` : ''}`);
    else if (st.state === 'error') body = note('labels unavailable (error)');
    else if (st.state === 'done') body = note('labels unavailable');
  }
  const score = isObject(v.score) ? scoreBlock(v.score, marked) : null;
  return el('div', { class: 'labels', hidden: !body && !score }, body, score);
}

// A chip per name, in the order of ctx.labelNames (a label the list lacks follows, so none is dropped). A chip is
// positive, negative, or unknown when the labels do not carry its name, and shows whether it agrees with the reference
// once the score carries the reference's labels (score.reference_chexbert_14, {name: 0|1}).
function chipList(v, cx) {
  const labels = v.labels;
  const given = Array.isArray(cx.labelNames) ? cx.labelNames.filter((n) => typeof n === 'string') : [];
  const names = [...given, ...Object.keys(labels).filter((n) => !given.includes(n))];
  const reference = isObject(v.score) && isObject(v.score.reference_chexbert_14) ? v.score.reference_chexbert_14 : null;
  let marked = false;
  const chips = names.map((name) => {
    const value = Object.hasOwn(labels, name) ? labels[name] : null;
    const kind = value == null ? 'unknown' : isOn(value) ? 'positive' : 'negative';
    const known = reference && kind !== 'unknown' && Object.hasOwn(reference, name) && reference[name] != null;
    const agrees = known ? isOn(reference[name]) === (kind === 'positive') : null;
    if (agrees !== null) marked = true;
    const spoken = `${name}: ${CHIP_SPOKEN[kind]}${agrees === null ? '' : agrees ? ', matches reference' : ', differs from reference'}`;
    return el('li', {
      class: CHIP_CLASS[kind], 'data-label': name, 'data-value': kind === 'unknown' ? null : kind === 'positive' ? '1' : '0',
      'data-agree': agrees === null ? null : String(agrees), 'aria-label': spoken,
    }, name, agrees === null ? null : el('span', { class: 'mark', 'aria-hidden': 'true' }, agrees ? '✓' : '✗'));
  });
  return { node: el('ul', { class: 'label-chips', 'aria-label': 'CheXbert-14 labels of the generated report' }, ...chips), marked };
}

function scoreBlock(score, marked) {
  const entries = SCORES.filter(([, key]) => isNum(score[key])).map(([name, key]) => [name, score[key].toFixed(3)]);
  if (typeof score.exact_match_14 === 'boolean') entries.push(['Exact match', score.exact_match_14 ? 'yes' : 'no']);
  if (!entries.length) return null;
  const source = typeof score.reference_source === 'string'
    ? (Object.hasOwn(REFERENCES, score.reference_source) ? REFERENCES[score.reference_source] : score.reference_source) : null;
  const caption = [source ? `vs ${source}` : null, marked ? '✓ same as the reference, ✗ different' : null].filter(Boolean).join(' · ');
  return el('div', { class: 'score-block' },
    el('dl', { class: 'score' }, ...entries.map(([name, value]) => el('div', {}, el('dt', {}, name), el('dd', {}, value)))),
    caption ? el('p', { class: 'score-source' }, caption) : null);
}

// ---- provenance ----------------------------------------------------------------------------------------------------------

// model · ckpt <first 8 of sha256> · scan <impl>/tfla <impl> · prefix_k <k> · beam <n> · <tokens> tok · <total> ms · <device>
// A part the card does not have is left out. What ran (the generate detail) wins over what was asked (the options).
export function provenanceText(view) {
  const v = whole(view);
  const card = isObject(v.provenance) ? v.provenance : {};
  const generate = isObject(v.stages.generate?.detail) ? v.stages.generate.detail : {};
  const options = isObject(v.options) ? v.options : {};
  const decode = [generate.decode, options.decode].find((d) => typeof d === 'string');   // what ran, else what was asked
  return [
    said('', card.name),
    said('ckpt ', str(card.checkpoint_sha256).slice(0, 8)),
    [said('scan ', card.scan_impl), said('tfla ', card.tfla_impl)].filter(Boolean).join('/') || null,
    said('prefix_k ', card.prefix_k),
    decode === 'greedy' ? 'greedy' : said('beam ', generate.beam_size) ?? said('beam ', options.beam_size),
    said('', generate.tokens, ' tok'),
    isNum(v.totalMs) ? `${Math.round(v.totalMs)} ms` : null,
    said('', card.device) ?? said('', generate.device),
  ].filter(Boolean).join(' · ');
}

export function renderProvenance(view, ctx) {
  const v = whole(view);
  const cx = ctx ?? {};
  const card = isObject(v.provenance) ? v.provenance : {};
  const generate = isObject(v.stages.generate?.detail) ? v.stages.generate.detail : {};
  const text = provenanceText(v);
  const drift = str(card.drift_note) || str(generate.drift_note);
  // A button, not an anchor: a link to a hash would change the page's route.
  const link = (text || drift) && typeof cx.showModels === 'function'
    ? el('button', { type: 'button', onclick: () => cx.showModels() }, 'model details') : null;
  return el('footer', { class: 'provenance', hidden: !text && !drift },
    text || link ? el('p', {}, text, text && link ? ' · ' : null, link) : null,
    drift ? el('p', { class: 'drift' }, drift) : null);
}

// ---- the user turn -----------------------------------------------------------------------------------------------------

const IMAGE_URL = /^(?:blob:|data:image\/(?:png|jpeg|webp|gif);base64,)/;
const usable = (url) => typeof url === 'string' && IMAGE_URL.test(url);
const SERVER_PATH = /^\/(?![/\\])/;   // one slash, then neither a slash nor a backslash: a path on this origin

// The resolved options as small chips: beam 3 · 100 tok · cached · k 4/3, and what else deviates from the defaults.
export function optionChips(options) {
  if (!isObject(options)) return [];
  const o = options;
  return [
    o.decode === 'greedy' ? 'greedy' : said('beam ', o.beam_size),
    said('', o.max_new_tokens, ' tok'),
    o.cached_decode === true ? 'cached' : o.cached_decode === false ? 'uncached' : null,
    str(o.k_images) || str(o.k_reports) ? `k ${str(o.k_images) || NONE}/${str(o.k_reports) || NONE}` : null,
    o.label === false ? 'labels off' : null,
    o.display_repair === true ? 'repair on' : null,
    o.compile === true ? 'compiled' : null,
    typeof o.reference === 'string' && o.reference ? 'reference' : null,
    Number.isInteger(o.test_row) ? `test row ${o.test_row}` : null,
  ].filter(Boolean);
}

// msg = { text, image: { url, filename }, options }. image.url is either a URL the page can show as it is (the local
// preview: a blob: or data: image) or a path on the server ("/v1/..."), which ctx.loadImage fetches with the token.
export function renderUserTurn(msg, ctx) {
  const m = isObject(msg) ? msg : {};
  const cx = ctx ?? {};
  const text = str(m.text).trim();
  const chips = optionChips(m.options);
  const picture = isObject(m.image) ? thumbnail(m.image, cx) : null;
  return el('article', { class: 'turn user', hidden: !picture && !text && !chips.length },
    el('div', { class: 'bubble' }, picture, text ? el('p', { class: 'user-text' }, text) : null,
      chips.length ? el('ul', { class: 'options', 'aria-label': 'Settings used' }, ...chips.map((c) => el('li', { class: 'chip' }, c))) : null));
}

function thumbnail(image, cx) {
  const name = str(image.filename);
  const shown = usable(image.url);
  const fetched = !shown && typeof image.url === 'string' && SERVER_PATH.test(image.url) && typeof cx.loadImage === 'function';
  if (!shown && !fetched) return name ? el('span', { class: 'chip' }, name) : null;
  const img = el('img', { class: 'thumb', alt: name ? `Uploaded X-ray: ${name}` : 'Uploaded X-ray' });
  if (shown) {
    img.setAttribute('src', image.url);
  } else {
    Promise.resolve().then(() => cx.loadImage(image.url)).then(
      (url) => { if (usable(url)) img.setAttribute('src', url); },
      () => img.setAttribute('data-failed', ''));
  }
  if (typeof cx.openViewer !== 'function') return img;
  return el('button', { type: 'button', class: 'thumb-button', 'aria-label': 'Open X-ray in the viewer', onclick: () => cx.openViewer(image) }, img);
}

// ---- the assistant card and the throttle ---------------------------------------------------------------------------------

export function renderAssistantCard(view, ctx) {
  const v = whole(view);
  const timeline = renderTimeline(v, ctx);
  const provenance = renderProvenance(v, ctx);
  // A turn that finished having run no stage (a question the server answered with a warning) has no pipeline to show,
  // and no model to attribute: nothing ran. One that was stopped or failed before its first stage still shows its stages.
  if (v.status === 'done' && !Object.keys(v.stages).length) {
    timeline.setAttribute('hidden', '');
    provenance.setAttribute('hidden', '');
  }
  return el('article', { class: 'card', 'data-message-id': str(v.id) || null, 'data-status': str(v.status) || null },
    timeline, renderNotes(v), renderReport(v, ctx), renderLabels(v, ctx), provenance);
}

const pending = new WeakSet();

// Runs fn once on the next animation frame however often it is asked before then. Pass the same function each time and
// let it read the latest view itself: fifty snapshots in one frame draw one card. A turn in a background tab draws when
// the tab is shown again. Without requestAnimationFrame (a test) a timer stands in.
export function scheduleRender(fn) {
  if (pending.has(fn)) return;
  pending.add(fn);
  const run = () => {
    pending.delete(fn);
    fn();
  };
  if (typeof globalThis.requestAnimationFrame === 'function') globalThis.requestAnimationFrame(run);
  else setTimeout(run, 16);
}
