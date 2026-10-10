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
// ctx = { loadImage, openViewer, copy, showModels, labelNames, retrieval, ui, turn }. Every member is optional and a missing
// callback hides its control.
//   loadImage(path) -> Promise<object URL>   api.js loadImage with the page's auth: a user turn's thumbnail
//   openViewer(image)                        a click on the thumbnail; image is what renderUserTurn was given
//   copy(text)                               the report's Copy button; a throw or a rejected promise shows "Copy failed"
//   showModels()                             the provenance link, which opens /v1/models in the drawer
//   labelNames: [14 names]                   CHEXBERT_14 order, from /v1/models; else the order of view.labels
//   retrieval: false                         the server runs no retrieval stage (/v1/models features): a user turn shows no k chip,
//                                            since k was never used; left out, it is taken to run one
//   ui: Map                                  keeps the open stage details and Show raw, per message id, across the
//                                            whole-card replace; without it a re-rendered card starts closed
//   turn: 3                                  the turn's number in the session: it names the card and tells the same
//                                            control of two cards apart ("Copy report, turn 3")
// Every control a card rebuilds each frame has a data-action (copy, raw, models, more, stage), so the page can give
// focus back to the same control after the replace: focusKey before it, restoreFocus after.
import { isSameOriginPath } from './api.js';
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

// The turn number a ctx (or the options of detailTable) carries, if it is one; and a spoken name with it appended.
const turnOf = (cx) => (Number.isInteger(cx?.turn) && cx.turn > 0 ? cx.turn : null);
const named = (text, cx) => (turnOf(cx) ? `${text}, turn ${turnOf(cx)}` : text);
const path = (...parts) => parts.map(str).filter(Boolean).join(' ');   // "retrieve report_matches 2 report"
const squeezed = (t) => str(t).split(/\s+/).filter(Boolean).join(' ');   // a text as the dumps have it: its whitespace in single spaces

// The view with every field present: what a builder reads, so a field that is missing is the empty value.
function whole(view) {
  const base = initialView(null);
  const given = isObject(view) ? view : {};
  return Object.fromEntries(Object.keys(base).map((k) => [k, given[k] ?? base[k]]));
}

const noticesOf = (v) => (Array.isArray(v.notices) ? v.notices.filter((n) => isObject(n) && (str(n.message) || str(n.code))) : []);

// What outlives a re-render: the stage details that are open, the show-more buttons that are expanded ("<stage>:<n>")
// and Show raw. ctx.ui keeps it per message id; without a store the record lives and dies with the card.
function uiOf(view, ctx) {
  const store = ctx?.ui;
  const fresh = () => ({ open: new Set(), more: new Set(), raw: false });
  if (!store || typeof store.get !== 'function' || typeof store.set !== 'function') return fresh();
  let record = store.get(view.id);
  if (!record) {
    record = fresh();
    store.set(view.id, record);
  }
  return record;
}

// ---- the timeline -----------------------------------------------------------------------------------------------------

// A stage with a detail is a disclosure: a <button aria-expanded aria-controls> inside its item carries the label and
// opens the table, and Enter and Space come with the button. A stage without a detail is plain text: nothing to open,
// so nothing to focus. The item keeps data-stage, data-state and .open for the stylesheet.
// What a screen reader gets beyond the visible label is its `said` part, and the name is the label and then that
// ("encode · 612 ms, done, turn 2"), so the visible text is the start of it (WCAG 2.5.3). A button carries it in its
// aria-label; a plain item as text the eye does not see, because browse mode reads a list item's text, not its label.
export function renderTimeline(view, ctx) {
  const v = whole(view);
  const cx = ctx ?? {};
  const ui = uiOf(v, cx);
  return el('ol', { class: 'timeline', role: 'list', 'aria-label': 'Pipeline stages' }, ...STAGES.map((s) => {
    const st = stageState(v, s);
    const state = typeof st.state === 'string' ? st.state : 'pending';
    const ms = isNum(st.ms) ? Math.round(st.ms) : null;
    const why = str(st.skipped);
    // A stop is the user's own doing, so it is said on screen; every other skip reason is said to a screen reader only.
    const label = state === 'done' ? (ms == null ? `${s} · done` : `${s} · ${ms} ms`)
                : state === 'skipped' ? `${s} · skipped${why === 'stopped' ? ' (stopped)' : ''}` : s;
    const said = state === 'done' ? (ms == null ? '' : ', done')   // the glyph is a shape, so the state is also said in words
               : state === 'skipped' ? (why && why !== 'stopped' ? `: ${why}` : '') : `, ${state}`;
    if (!(isObject(st.detail) && Object.keys(st.detail).length)) {
      return el('li', { 'data-stage': s, 'data-state': state }, label, said ? el('span', { class: 'visually-hidden' }, said) : null);
    }
    const id = `detail-${str(v.id).replace(/[^\w-]/g, '_') || 'turn'}-${s}`;   // unique per message, and a usable id reference
    const table = detailTable(st.detail, `${s} details`, { id, name: s, turn: turnOf(cx), more: ui.more });
    const open = ui.open.has(s);
    const toggle = el('button', {
      type: 'button', 'data-action': 'stage', 'data-stage': s, 'aria-expanded': String(open), 'aria-controls': id,
      'aria-label': named(`${label}${said}`, cx),
    }, label);
    const li = el('li', { 'data-stage': s, 'data-state': state, class: open ? 'open' : null }, toggle, table);
    toggle.addEventListener('click', () => {
      const now = li.classList.toggle('open');
      toggle.setAttribute('aria-expanded', String(now));
      if (now) ui.open.add(s); else ui.open.delete(s);
    });
    return li;
  }));
}

// A stage's detail as a table: a row per key; an array of objects as a nested table with a row per object and a column
// per key; an object as a table of its own; a long string clipped with "show more". label names the table. options:
// { id } for the table (what a button's aria-controls names), { name } the stage, and { turn }, which together name each
// show-more button so no two sound alike ("show more of retrieve report_matches 2 report, turn 3"). { more } is a Set the
// table reads and writes the expanded strings to, as "<stage>:<n>" with n the string's number in this table in document
// order (the n of focusKey's more:<stage>:<n>): the page keeps it across rebuilds. Without it a table starts clipped.
export function detailTable(detail, label, options) {
  const o = isObject(options) ? options : {};
  const entries = isObject(detail) ? Object.entries(detail) : [];
  const env = { turn: turnOf(o), stage: str(o.name), more: o.more instanceof Set ? o.more : null, count: 0 };
  return el('table', { id: str(o.id) || null, 'aria-label': label },
    el('tbody', {}, ...entries.map(([key, value]) => row(key, value, 1, env, str(o.name)))));
}

const row = (key, value, depth, env, parent) => el('tr', {},
  el('th', { scope: 'row' }, key), el('td', {}, valueNode(value, depth, path(parent, key), env)));

const numberText = (n) => (Number.isInteger(n) ? String(n) : Number.isFinite(n) ? String(Number(n.toPrecision(6))) : NONE);

function valueNode(v, depth, name, env) {
  if (v == null) return NONE;
  if (typeof v === 'string') return v.length > CLIP ? clipped(v, name, env) : v;
  if (typeof v === 'number') return numberText(v);
  if (typeof v !== 'object') return String(v);
  if (depth >= MAX_DEPTH) return valueNode(jsonText(v), depth, name, env);
  if (Array.isArray(v)) return arrayNode(v, depth, name, env);
  const entries = Object.entries(v);
  if (!entries.length) return NONE;
  return el('table', { class: 'nested' }, el('tbody', {}, ...entries.map(([key, value]) => row(key, value, depth + 1, env, name))));
}

function jsonText(v) {
  try { return JSON.stringify(v); } catch { return NONE; }
}

function arrayNode(items, depth, name, env) {
  if (!items.length) return NONE;
  const inline = (x) => isPrimitive(x) && !(typeof x === 'string' && x.length > CLIP);
  if (items.every(inline)) return items.map((x) => (x == null ? NONE : typeof x === 'number' ? numberText(x) : String(x))).join(', ');
  if (items.every(isObject)) return recordsTable(items, depth, name, env);
  return el('ol', { class: 'values' }, ...items.map((x, i) => el('li', {}, valueNode(x, depth + 1, path(name, i + 1), env))));
}

function recordsTable(items, depth, name, env) {
  const columns = [...new Set(items.flatMap((item) => Object.keys(item)))];
  if (!columns.length) return NONE;
  return el('table', { class: 'nested' },
    el('thead', {}, el('tr', {}, ...columns.map((c) => el('th', { scope: 'col' }, c)))),
    el('tbody', {}, ...items.map((item, i) => el('tr', {}, ...columns.map(
      (c) => el('td', {}, valueNode(Object.hasOwn(item, c) ? item[c] : undefined, depth + 1, path(name, i + 1, c), env)))))));
}

// The first CLIP characters and a button that shows the rest, and hides it again. name says which string it is. env.more,
// when there is one, remembers that it is expanded: the next frame builds it expanded again.
function clipped(text, name, env) {
  let end = CLIP;
  const last = text.charCodeAt(end - 1);
  if (last >= 0xd800 && last <= 0xdbff) end -= 1;   // not between the halves of a surrogate pair
  const head = `${text.slice(0, end)}…`;
  const key = `${env.stage}:${env.count++}`;   // built in document order, so the nth call of a table is its nth show-more button
  let open = !!env.more?.has(key);
  const shown = el('span', { class: 'clip-text' }, open ? text : head);
  const of = name ? ` of ${name}` : '';
  const verb = () => (open ? 'show less' : 'show more');
  const more = el('button', {
    type: 'button', class: 'more', 'data-action': 'more', 'aria-expanded': String(open), 'aria-label': named(`${verb()}${of}`, env),
  }, verb());
  more.addEventListener('click', () => {
    open = !open;
    shown.textContent = open ? text : head;
    more.textContent = verb();
    more.setAttribute('aria-expanded', String(open));
    more.setAttribute('aria-label', named(`${verb()}${of}`, env));
    if (open) env.more?.add(key); else env.more?.delete(key);
  });
  return el('span', { class: 'clip' }, shown, ' ', more);
}

// ---- notes: warnings, a stop, an error ---------------------------------------------------------------------------------

const errorText = (v) => `Error: ${str(v.error?.message) || 'the turn failed'}`;
// The server says "Internal error (<class>)" for a failure it did not foresee, and only the class: never the exception's text, which can
// hold report text and paths (a public server says a fixed sentence instead). A line under it says what to try (P4-H A3): the one such
// error users have met is a server that ran on while its code changed on disk, which a restart cures.
const INTERNAL_ERROR = /^Internal error \(/;
const INTERNAL_ERROR_HINT = 'The server hit an internal error. If you just updated the code, restart the server.';

export function renderNotes(view) {
  const v = whole(view);
  const notes = noticesOf(v).map((n) => el('p', { class: 'note notice', 'data-code': str(n.code) || null }, str(n.message) || str(n.code)));
  if (v.error || v.status === 'error') {
    notes.push(el('p', { class: 'note error' }, errorText(v)));
    if (INTERNAL_ERROR.test(str(v.error?.message))) notes.push(el('p', { class: 'note error-hint' }, INTERNAL_ERROR_HINT));
  } else if (v.status === 'aborted') {
    notes.push(el('p', { class: 'note stopped' }, 'Turn stopped'));
  }
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

const COPY_FAILED_MS = 2000;   // how long the Copy button says "Copy failed"

// The text as sections: before the first header, then one per literal "Findings:" or "Impression:" in text order.
export function splitReport(text) {
  const parts = str(text).split(/(Findings:|Impression:)/);
  const sections = [];
  if (parts[0].trim()) sections.push({ title: null, body: parts[0].trim() });
  for (let i = 1; i < parts.length; i += 2) sections.push({ title: parts[i].slice(0, -1), body: parts[i + 1].trim() });
  return sections;
}

// Why the report ends where it does (P4-G, P9-G2), from the generate stage's detail ({stopped, tokens}) and the options the turn ran with:
//   stopped on a repeat   a quiet note, and never the budget's: the turn chose to stop
//   stopped on the model's own end of report (eos)   no note at all: the report ends where the model meant it to, so nothing was cut off
//   the budget, with the display repair on and a sentence cut off   what the card hides, and where Show raw has it
//   the budget, otherwise   the plain note (a repaired card hides the unfinished sentence, an unrepaired one shows it)
// A log from before the stop reason was recorded has no `stopped`: it ran to the budget, as every report then did.
function reportNote(v) {
  const generate = isObject(v.stages.generate?.detail) ? v.stages.generate.detail : {};
  const stopped = str(generate.stopped);
  if (stopped === 'repeat') return el('p', { class: 'note', 'data-stopped': 'repeat' }, 'Stopped when the model began repeating itself.');
  if (stopped === 'eos') return null;   // the engine never flags an EOS stop as cut off; whatever a server sent, no budget is to blame
  if (!v.truncated) return null;
  // The card hides something only if the repair left something out of it: with no complete sentence to cut back to, the repair keeps the
  // text as it is, and the card shows all of it.
  const hides = squeezed(v.displayReport) !== squeezed(v.report);
  if (!(isObject(v.options) && v.options.display_repair === true && (stopped === 'budget' || stopped === '') && hides)) {
    return el('p', { class: 'note truncated' }, 'Report stopped at the token budget mid-sentence');
  }
  const budget = isNum(generate.tokens) ? `${generate.tokens}-token budget` : 'token budget';
  return el('p', { class: 'note truncated' }, `Reached the ${budget}; the unfinished last sentence is hidden (Show raw shows it).`);
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
  const toggle = el('button', { type: 'button', 'data-action': 'raw', 'aria-pressed': String(ui.raw), 'aria-label': named('Show raw report', cx) }, 'Show raw');
  toggle.addEventListener('click', () => {
    ui.raw = !ui.raw;
    toggle.setAttribute('aria-pressed', String(ui.raw));
    const next = bodyOf();
    body.replaceWith(next);
    body = next;
  });
  return el('div', { class: v.provisional ? 'report provisional' : 'report' },
    body,
    reportNote(v),
    el('div', { class: 'report-actions' }, typeof cx.copy === 'function' && !v.provisional ? copyButton(cx, () => (ui.raw ? raw : shown)) : null, toggle));
}

// Copy hands ctx.copy the text. A copy that throws or whose promise is rejected (a refused clipboard) must not pass
// unnoticed: the button says "Copy failed" for a moment, and the failure is not left as an unhandled rejection. That state
// lives on the button, and a streaming turn rebuilds the card every frame, so the button is not offered while the report is
// provisional (the best beam so far, not the report); it appears when the block closes.
function copyButton(cx, textNow) {
  const idle = () => { copy.textContent = 'Copy'; copy.setAttribute('aria-label', named('Copy report', cx)); };
  const failed = () => {
    copy.textContent = 'Copy failed';
    copy.setAttribute('aria-label', named('Copy failed', cx));
    setTimeout(idle, COPY_FAILED_MS);
  };
  const copy = el('button', { type: 'button', 'data-action': 'copy', 'aria-label': named('Copy report', cx) }, 'Copy');
  copy.addEventListener('click', () => {
    try {
      const done = cx.copy(textNow());
      if (done && typeof done.then === 'function') done.then(undefined, failed);
    } catch {
      failed();
    }
  });
  return copy;
}

// ---- labels and scores -------------------------------------------------------------------------------------------------

const isOn = (x) => x === 1 || x === true;   // a label is 1 or 0

const CHIP_CLASS = { positive: 'chip label positive', negative: 'chip label negative', unknown: 'chip label unknown' };
const CHIP_SPOKEN = { positive: 'positive', negative: 'negative', unknown: 'not reported' };
// What each number is, as the thesis scores it for one report: ROUGE-L is the sentence-level F-measure (beta 1.2) of
// scripts/evaluate_report_generation.py, BLEU-n is its corpus formula applied to this one pair, and the CheXbert F1 is
// micro-averaged over the 14 labels of this report (the corpus-level tables average over many).
const SCORES = [['ROUGE-L', 'rouge_l'], ['BLEU-1', 'bleu_1'], ['BLEU-4', 'bleu_4'], ['CheXbert-14 micro F1', 'chexbert_14_micro_f1']];
const REFERENCES = { user: 'your reference', test_split: 'test-split reference' };

// What the chips say once the label stage has ended skipped (P5-E), so the placeholder never outlives the stage: the user's own
// setting is "labels off" whatever else is true, and any other reason is named.
function skippedLabels(why, off) {
  if (off || why === 'label_off') return 'labels off';
  return `labels unavailable${why ? ` (${why})` : ''}`;
}

// Said under the chips while the gallery's own labels are still being built (the label detail's neighbor_agreement_pending): the
// report is labelled, but no similar X-ray has labels to agree with yet. Server state, so public mode says it too.
const AGREEMENT_PENDING = 'agreement pending: the gallery is still being labelled';

export function renderLabels(view, ctx) {
  const v = whole(view);
  const cx = ctx ?? {};
  const note = (text) => el('p', { class: 'note' }, text);
  const off = isObject(v.options) && v.options.label === false;   // the user's own setting: said as that, never as "unavailable"
  let body = null;
  let pending = null;
  let marked = false;
  if (isObject(v.labels)) {
    ({ node: body, marked } = chipList(v, cx));
    if (isObject(v.stages.label?.detail) && v.stages.label.detail.neighbor_agreement_pending === true) pending = note(AGREEMENT_PENDING);
  } else if (labelsPending(v)) {
    body = note(off ? 'labels off' : 'labelling…');   // off: say so now, not at the end
  } else {
    const st = stageState(v, 'label');   // a stop or an error settles it; a settled turn that never got here shows nothing
    const why = str(st.skipped);
    if (st.state === 'skipped') body = note(skippedLabels(why, off));
    else if (st.state === 'error') body = note('labels unavailable (error)');
    else if (st.state === 'done') body = note('labels unavailable');
  }
  const score = isObject(v.score) ? scoreBlock(v.score, marked) : null;
  return el('div', { class: 'labels', hidden: !body && !score }, body, pending, score);
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
    // The state is text inside the chip, hidden from the eye: a screen reader in browse mode reads the text of a list
    // item and not its aria-label, so a label there would never be heard.
    const spoken = `: ${CHIP_SPOKEN[kind]}${agrees === null ? '' : agrees ? ', matches reference' : ', differs from reference'}`;
    return el('li', {
      class: CHIP_CLASS[kind], 'data-label': name, 'data-value': kind === 'unknown' ? null : kind === 'positive' ? '1' : '0',
      'data-agree': agrees === null ? null : String(agrees),
    }, name, el('span', { class: 'visually-hidden' }, spoken),
    agrees === null ? null : el('span', { class: 'mark', 'aria-hidden': 'true' }, agrees ? '✓' : '✗'));
  });
  return { node: el('ul', { class: 'label-chips', role: 'list', 'aria-label': 'CheXbert-14 labels of the generated report' }, ...chips), marked };
}

function scoreBlock(score, marked) {
  const entries = SCORES.filter(([, key]) => isNum(score[key])).map(([name, key]) => [name, score[key].toFixed(3)]);
  if (typeof score.exact_match_14 === 'boolean') entries.push(['CheXbert-14 exact match', score.exact_match_14 ? 'yes' : 'no']);
  if (!entries.length) return null;
  const source = typeof score.reference_source === 'string'
    ? (Object.hasOwn(REFERENCES, score.reference_source) ? REFERENCES[score.reference_source] : score.reference_source) : null;
  const caption = [source ? `vs ${source}` : null, 'this report only', marked ? '✓ same as the reference, ✗ different' : null]
    .filter(Boolean).join(' · ');
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
    ? el('button', { type: 'button', 'data-action': 'models', 'aria-label': named('model details', cx), onclick: () => cx.showModels() }, 'model details') : null;
  return el('footer', { class: 'provenance', hidden: !text && !drift },
    text || link ? el('p', {}, text, text && link ? ' · ' : null, link) : null,
    drift ? el('p', { class: 'drift' }, drift) : null);
}

// ---- the user turn -----------------------------------------------------------------------------------------------------

const IMAGE_URL = /^(?:blob:|data:image\/(?:png|jpeg|webp|gif);base64,)/;
const usable = (url) => typeof url === 'string' && IMAGE_URL.test(url);

// The resolved options as small chips: beam 3 · 100 tok · cached · k 4/3, and what else deviates from the defaults. Display repair is
// on by default in the page, so only its being off is said: "raw text", the report as the decoder wrote it. So is stopping when the
// report starts repeating: off is "full budget", the published protocol, which always decodes every token it is given. A turn whose
// options do not say (a log from before the switch) makes no claim.
export function optionChips(options) {
  if (!isObject(options)) return [];
  const o = options;
  return [
    o.decode === 'greedy' ? 'greedy' : said('beam ', o.beam_size),
    said('', o.max_new_tokens, ' tok'),
    o.cached_decode === true ? 'cached' : o.cached_decode === false ? 'uncached' : null,
    str(o.k_images) || str(o.k_reports) ? `k ${str(o.k_images) || NONE}/${str(o.k_reports) || NONE}` : null,
    o.label === false ? 'labels off' : null,
    o.display_repair === false ? 'raw text' : null,
    o.stop_on_repeat === false ? 'full budget' : null,
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
  const chips = optionChips(cx.retrieval === false && isObject(m.options) ? { ...m.options, k_images: null, k_reports: null } : m.options);
  const picture = isObject(m.image) ? thumbnail(m.image, cx) : null;
  return el('article', { class: 'turn user', 'aria-label': named('Your message', cx), hidden: !picture && !text && !chips.length },
    el('div', { class: 'bubble' }, picture, text ? el('p', { class: 'user-text' }, text) : null,
      chips.length ? el('ul', { class: 'options', role: 'list', 'aria-label': 'Settings used' }, ...chips.map((c) => el('li', { class: 'chip' }, c))) : null));
}

function thumbnail(image, cx) {
  const name = str(image.filename);
  const shown = usable(image.url);
  const fetched = !shown && isSameOriginPath(image.url) && typeof cx.loadImage === 'function';   // api.js's filter: the token goes nowhere else
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
  return el('button', { type: 'button', class: 'thumb-button', 'aria-label': named('Open X-ray in the viewer', cx), onclick: () => cx.openViewer(image) }, img);
}

// ---- the assistant card and the throttle ---------------------------------------------------------------------------------

export function renderAssistantCard(view, ctx) {
  const v = whole(view);
  const cx = ctx ?? {};
  const timeline = renderTimeline(v, cx);
  const provenance = renderProvenance(v, cx);
  // A turn that finished having run no stage (a question the server answered with a warning) has no pipeline to show,
  // and no model to attribute: nothing ran. One that was stopped or failed before its first stage still shows its stages.
  if (v.status === 'done' && !Object.keys(v.stages).length) {
    timeline.setAttribute('hidden', '');
    provenance.setAttribute('hidden', '');
  }
  return el('article', { class: 'card', 'aria-label': named('Assistant report', cx), 'data-message-id': str(v.id) || null, 'data-status': str(v.status) || null },
    timeline, renderNotes(v), renderReport(v, cx), renderLabels(v, cx), provenance);
}

// ---- keeping focus through the whole-card replace -------------------------------------------------------------------------

// The control a node is in (or is), as a key that survives the rebuild: copy, raw, models, stage:<stage> or
// more:<stage>:<n>, the nth show-more button of that stage's table. null when the node is not in one of those.
export function focusKey(node) {
  const control = node?.closest?.('[data-action]');
  if (!control) return null;
  const action = control.getAttribute('data-action');
  if (action === 'copy' || action === 'raw' || action === 'models') return action;
  const item = control.closest('li[data-stage]');
  const stage = item?.getAttribute('data-stage');
  if (!stage) return null;
  if (action === 'stage') return `stage:${stage}`;
  if (action !== 'more') return null;
  const index = Array.from(item.querySelectorAll('[data-action="more"]')).indexOf(control);
  return index < 0 ? null : `more:${stage}:${index}`;
}

const FOCUS_KEY = /^(?:(copy|raw|models)|stage:([a-z]+)|more:([a-z]+):(\d+))$/;

// Puts focus on the control of the rebuilt card that key names: after card.replaceWith(next), restoreFocus(next, key),
// with key taken by focusKey before it. true when the control took focus; false when there is no such control (a Copy
// the page no longer offers, a stage that has no detail) or it cannot be focused (a stage closed, so its buttons are not
// shown: keep ctx.ui so it is open again). Never scrolls the pane.
export function restoreFocus(card, key) {
  const m = typeof key === 'string' ? FOCUS_KEY.exec(key) : null;
  if (!m || typeof card?.querySelector !== 'function') return false;
  let target = null;
  if (m[1]) target = card.querySelector(`[data-action="${m[1]}"]`);
  else if (m[2]) target = STAGES.includes(m[2]) ? card.querySelector(`li[data-stage="${m[2]}"] > [data-action="stage"]`) : null;
  else if (STAGES.includes(m[3])) target = Array.from(card.querySelectorAll(`li[data-stage="${m[3]}"] [data-action="more"]`))[Number(m[4])];
  if (!target) return false;
  target.focus({ preventScroll: true });
  return globalThis.document?.activeElement === target;
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
