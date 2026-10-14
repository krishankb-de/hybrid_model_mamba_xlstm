// tests/frontend/render.test.mjs — the card builders of app/static/render.js (CHAT_UI_PLAN.md P4-C) on the DOM shim.
//
// Views come from state.js: the recorded tiny turn (fixtures/turn_tiny.json) through replay(), and turns that the tiny
// engine cannot produce yet (labels, scores, errors, a public mode) folded from synthetic events: no MIMIC data here.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { ShimEvent, installDom, serialize } from './dom_shim.mjs';
import { initialView, labelsPending, replay } from '../../app/static/state.js';
import { clearImageCache, loadImage as apiLoadImage } from '../../app/static/api.js';
import * as render from '../../app/static/render.js';
import {
  STAGES, detailTable, el, optionChips, provenanceText, renderAssistantCard, renderLabels, renderNotes, renderProvenance,
  renderReport, renderTimeline, renderUserTurn, scheduleRender, splitReport, statusText,
} from '../../app/static/render.js';

const { focusKey, restoreFocus } = render;   // fix round 1: read off the namespace, so a missing export fails its own tests only
// P6-B..D: read off the namespace too, and called through a guard, so that a builder that does not exist yet fails its own tests and no other
const builder = (name) => (...args) => {
  assert.equal(typeof render[name], 'function', `render.js exports ${name}`);
  return render[name](...args);
};
const renderImages = builder('renderImages');
const renderNeighbors = builder('renderNeighbors');
const renderMatches = builder('renderMatches');
const renderPublished = builder('renderPublished');
const settleClamps = builder('settleClamps');   // P6 fix 1: the page measures, once a card is in it, which clamped report overflows
const ownXraySide = builder('ownXraySide');     // P6-G: the viewer's side for the turn's own X-ray, from the card's tile and from the user turn's thumbnail
const splitSentences = builder('splitSentences');       // P6-H: the report card a person can read: a report's sentences, one to a line
const refersToEarlier = builder('refersToEarlier');     // P6-H: whether a sentence writes about an earlier X-ray
const summaryText = builder('summaryText');             // P6-H: the plain-language summary, from the labels and the glossary only

installDom();   // this file's process only: the other test files never see a document

const LABEL_NAMES = [   // CHEXBERT_14 of CHAT_UI_PLAN.md P5-A: the 14 names in the order the labeller reports them
  'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia',
  'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding',
];
const SHA = 'a1b2c3d4e5f60718293a4b5c6d7e8f90a1b2c3d4e5f60718293a4b5c6d7e8f90';

const recorded = () => JSON.parse(readFileSync(new URL('./fixtures/turn_tiny.json', import.meta.url)));
const tinyView = () => replay(recorded());

const q = (node, selector) => node.querySelector(selector);
const qa = (node, selector) => node.querySelectorAll(selector);
const texts = (nodes) => nodes.map((n) => n.textContent);
const tick = () => new Promise((resolve) => setImmediate(resolve));
const stageItem = (card, stage) => q(card, `li[data-stage="${stage}"]`);
const stageButton = (card, stage) => q(card, `li[data-stage="${stage}"] > button`);   // the disclosure control of a stage with a detail
const button = (card, label) => qa(card, 'button').find((b) => b.textContent === label) ?? null;
// What the page shows of a node: its text without what is visually hidden (that is for a screen reader).
const visibleText = (node) => (node.nodeType === 3 ? node.data
  : node.classList.contains('visually-hidden') ? '' : node.childNodes.map(visibleText).join(''));
// What Tab can reach under root: a control that takes focus and is not taken out of the tab order.
const tabStops = (root) => qa(root, 'button, input, select, textarea, a[href], [tabindex]')
  .filter((n) => n.getAttribute('tabindex') !== '-1' && !n.hasAttribute('disabled'));
const part = (table, name) => table.children.find((c) => c.localName === name);   // a table's own thead or tbody
const bodyRows = (table) => part(table, 'tbody').children;                        // its own rows, not those of nested tables
const pair = (row) => [row.children[0].textContent, row.children[1].textContent];

function deepFreeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.freeze(value);
    Object.values(value).forEach(deepFreeze);
  }
  return value;
}

// ---- turns folded from synthetic events ----------------------------------------------------------------------------

const step = (event, data = {}) => ({ event, data });
const stageStart = (stage, index) => step('stage_start', { stage, index });
const stageEnd = (stage, ms, detail) => step('stage_end', { stage, ms, detail });
const skipped = (stage, reason) => step('stage_end', { stage, skipped: reason });
const snapshot = (text, n = 0) => step('content_block_delta', { index: 0, delta: { type: 'beam_snapshot', step: n, text } });
const stopOf = (status, more = {}) => step('message_stop', {
  message_id: 'm_test', status, total_ms: 612.4, report: null, display_report: null, truncated_mid_sentence: false,
  disclaimer: 'Research prototype; not for clinical use.', ...more,
});
const START = step('message_start', {
  message_id: 'm_test', user_message_id: 'u_test', session_id: 's_test', mode: 'private',
  model: {
    name: 'hybrid_150m_m3_rrg', checkpoint_sha256: SHA, prefix_k: 32, scan_impl: 'exact', tfla_impl: 'exact', device: 'cpu',
    drift_note: 'CPU decode can drift from the published GPU run',
  },
  options: {
    model: 'hybrid_150m_m3_rrg', decode: 'beam', beam_size: 3, max_new_tokens: 100, cached_decode: true, compile: false,
    k_images: 4, k_reports: 3, label: true, reference: null, display_repair: false, test_row: null,
  },
  image: { sha256: SHA, filename: 'chest.png', source: 'upload', urls: { thumb: '/v1/messages/u_test/image?variant=thumb' } },
});
const GENERATE = { decode: 'beam', beam_size: 3, tokens: 87, stopped: 'budget', cached_decode: true, device: 'cpu',
                   drift_note: 'CPU decode can drift from the published GPU run' };

const viewOf = (steps) => replay(steps.map((s, i) => ({ event: s.event, data: { ...s.data, seq: i + 1 } })));

// A finished turn. labels and score are what P5 will send; with neither the label and score stages end skipped.
function finished({ report = 'Findings: The lungs are clear. Impression: No acute disease.', labels = null, score = null,
                    truncated = false, display = report, options = {}, generate = {} } = {}) {
  return viewOf([
    step('message_start', { ...START.data, options: { ...START.data.options, ...options } }),
    stageStart('preprocess', 0), stageEnd('preprocess', 1.6, { format: 'PNG', input_px: [320, 320] }),
    stageStart('encode', 1), stageEnd('encode', 611.6, { patch_grid: [197, 768], pooled_dim: 512, device: 'cpu' }),
    skipped('retrieve', 'gallery_unavailable'),
    stageStart('generate', 3), step('content_block_start', { index: 0, content_block: { type: 'report', text: '' } }),
    snapshot(report), step('content_block_stop', { index: 0 }), stageEnd('generate', 5234.2, { ...GENERATE, ...generate }),
    labels ? stageEnd('label', 40, { chexbert_14: labels }) : skipped('label', 'labeler_unavailable'),
    score ? stageEnd('score', 12, score) : skipped('score', 'no_reference'),
    stopOf('done', { report, display_report: display, truncated_mid_sentence: truncated }),
  ]);
}

const labelled = (positives) => Object.fromEntries(LABEL_NAMES.map((n) => [n, positives.includes(n) ? 1 : 0]));
const SCORE = { rouge_l: 0.19049, bleu_1: 0.3, bleu_4: 0.1, chexbert_14_micro_f1: 0.4736, reference_source: 'user' };

// The turn while the label stage runs, a report so far and the stages before it done.
const labelling = () => viewOf([
  START, stageStart('preprocess', 0), stageEnd('preprocess', 1, {}), stageStart('generate', 3), snapshot('Findings: so far'),
  stageEnd('generate', 9, GENERATE), stageStart('label', 4),
]);

// A finished turn with strings long enough to be clipped: three in the retrieve stage (a private turn's matched reports and
// a note) and one in generate (its drift note). Each stage's first clipped string is its number 0, so a show-more key that
// forgot the stage would take the two for one.
const withClips = () => viewOf([
  START, stageStart('retrieve', 2),
  stageEnd('retrieve', 8, { report_matches: [{ report: 'x'.repeat(300) }, { report: 'y'.repeat(300) }], note: 'z'.repeat(300) }),
  stageStart('generate', 3), snapshot('Findings: ok'), stageEnd('generate', 9, { ...GENERATE, drift_note: 'd'.repeat(300) }),
  stopOf('done', { report: 'Findings: ok', display_report: 'Findings: ok' }),
]);

// ---- el ------------------------------------------------------------------------------------------------------------

test('el sets attributes, wires on* handlers, drops null children and appends strings as text, never as markup', () => {
  const calls = [];
  const node = el('p', { class: 'a b', hidden: true, title: false, lang: null, tabindex: 0, onclick: () => calls.push('click') },
                  'plain ', null, '<b onclick=alert(1)>x</b>', undefined);
  assert.equal(node.getAttribute('class'), 'a b');
  assert.equal(node.getAttribute('hidden'), '');      // true is a bare attribute
  assert.equal(node.hasAttribute('title') || node.hasAttribute('lang'), false);   // false and null set nothing
  assert.equal(node.getAttribute('tabindex'), '0');   // 0 is a value
  node.click();
  assert.deepEqual(calls, ['click']);
  assert.equal(node.textContent, 'plain <b onclick=alert(1)>x</b>');
  assert.equal(node.children.length, 0);               // no element was parsed out of the string
});

test('the shim follows the DOM where these tests lean on it: selectors, events, strings, and the HTML-string tripwires', () => {
  const root = el('div', { id: 'r' },
    el('ol', { class: 'timeline' },
      el('li', { 'data-stage': 'a', class: 'open x' }, 'A'),
      el('li', { 'data-stage': 'b' }, el('table', {}, el('tbody', {}, el('tr', {}, el('td', {}, 'cell')))))),
    el('p', { class: 'x', title: 'a b' }, 'p'));
  assert.deepEqual(texts(qa(root, 'li')), ['A', 'cell']);
  assert.deepEqual(texts(qa(root, 'ol > li')), ['A', 'cell']);
  assert.deepEqual(texts(qa(root, 'ol td')), ['cell']);
  assert.deepEqual(texts(qa(root, 'li > td')), []);   // a td is a descendant of the li, not a child
  assert.deepEqual(texts(qa(root, 'li[data-stage="b"]')), ['cell']);
  assert.deepEqual(texts(qa(root, '[data-stage]')), ['A', 'cell']);
  assert.deepEqual(texts(qa(root, '.x')), ['A', 'p']);
  assert.deepEqual(texts(qa(root, 'li.open.x')), ['A']);
  assert.deepEqual(texts(qa(root, 'p[title="a b"], td')), ['cell', 'p']);   // a list, in document order
  assert.equal(q(root, '#r'), null);   // descendants only
  const td = q(root, 'td');
  assert.equal(td.closest('li').getAttribute('data-stage'), 'b');
  assert.ok(root.contains(td) && !td.contains(root) && td.contains(td));
  const seen = [];
  root.addEventListener('click', (e) => seen.push(`${e.target.localName} reached ${e.currentTarget.localName}`));
  td.click();   // the event bubbles from the cell to the root, and keeps its target
  assert.deepEqual(seen, ['td reached div']);

  const p = document.createElement('p');
  p.append(undefined, false, 0, 'x');   // what is not a node is made a string, as the DOM does
  assert.equal(p.textContent, 'undefinedfalse0x');
  p.textContent = '<b>x</b>';
  assert.deepEqual([p.children.length, p.textContent], [0, '<b>x</b>']);

  document.body.replaceChildren(root);   // focus follows a browser's rules: controls take it, plain elements do not, and only in the page
  const box = document.createElement('li');
  const push = el('button', { type: 'button' }, 'b');
  root.append(box, push, el('li', { tabindex: -1 }, 'x'));
  box.focus();
  assert.equal(document.activeElement, document.body);
  push.focus();
  assert.equal(document.activeElement, push);
  push.remove();
  assert.equal(document.activeElement, document.body);   // a removed element has lost focus
  const loose = el('button', { type: 'button' }, 'loose');
  loose.focus();
  assert.equal(document.activeElement, document.body);   // not in the page: no focus
  root.children.at(-1).focus();
  assert.equal(document.activeElement, root.children.at(-1));   // tabindex -1: focusable by script
  assert.deepEqual(tabStops(root).map((n) => n.localName), []);   // ... but not a tab stop
  document.body.replaceChildren();

  assert.throws(() => { p.innerHTML = '<b>x</b>'; }, /no innerHTML/);
  assert.throws(() => p.outerHTML, /no outerHTML/);
  assert.throws(() => p.insertAdjacentHTML('beforeend', 'x'), /no insertAdjacentHTML/);
  assert.throws(() => document.write('x'), /no document\.write/);
});

// ---- the timeline --------------------------------------------------------------------------------------------------

test('the timeline is an ordered list of the six stages in contract order', () => {
  const ol = renderTimeline(initialView('m1'));
  assert.equal(ol.localName, 'ol');
  assert.ok(ol.classList.contains('timeline'));
  assert.equal(ol.getAttribute('aria-label'), 'Pipeline stages');
  assert.deepEqual(qa(ol, 'li').map((li) => li.getAttribute('data-stage')), ['preprocess', 'encode', 'retrieve', 'generate', 'label', 'score']);
  assert.deepEqual(STAGES, ['preprocess', 'encode', 'retrieve', 'generate', 'label', 'score']);
  assert.equal(qa(ol, '[tabindex]').length, 0);   // no item is made a tab stop by hand
});

// A plain item (a stage with no detail) says its state in text the eye does not see, after the visible label: browse mode
// reads the text of a list item and not its aria-label. visibleText is what is shown, textContent what is read.
const hiddenText = (li) => q(li, '.visually-hidden')?.textContent ?? null;

test('a pending stage shows its bare name, a running one its name, a done one its time; hidden text after it says the state', () => {
  const pending = renderTimeline(initialView('m1'));
  assert.deepEqual(qa(pending, 'li').map((li) => [li.getAttribute('data-state'), visibleText(li), li.textContent, li.hasAttribute('aria-label')]),
                   STAGES.map((s) => ['pending', s, `${s}, pending`, false]));
  assert.deepEqual(qa(pending, 'li').map(hiddenText), STAGES.map(() => ', pending'));

  const mid = renderTimeline(viewOf([START, stageStart('preprocess', 0), stageEnd('preprocess', 1.6, {}), stageStart('encode', 1),
                                     stageEnd('encode', 611.6, { device: 'cpu' }), skipped('retrieve', 'gallery_unavailable'),
                                     stageStart('generate', 3)]));
  const encode = stageButton(mid, 'encode');   // a stage with a detail: its label is on the button that opens it
  assert.deepEqual([stageItem(mid, 'encode').getAttribute('data-state'), encode.textContent, encode.getAttribute('aria-label')],
                   ['done', 'encode · 612 ms', 'encode · 612 ms, done']);   // the name starts with the visible text (WCAG 2.5.3)
  const preprocess = stageItem(mid, 'preprocess');   // one without: plain text on the item
  assert.deepEqual([visibleText(preprocess), preprocess.textContent, hiddenText(preprocess), preprocess.hasAttribute('aria-label')],
                   ['preprocess · 2 ms', 'preprocess · 2 ms, done', ', done', false]);   // rounded
  const generate = stageItem(mid, 'generate');
  assert.deepEqual([generate.getAttribute('data-state'), visibleText(generate), generate.textContent, hiddenText(generate)],
                   ['running', 'generate', 'generate, running', ', running']);
});

test('a skipped stage says so, with its reason in the hidden text', () => {
  const view = finished();
  const retrieve = stageItem(renderTimeline(view), 'retrieve');
  assert.deepEqual([retrieve.getAttribute('data-state'), visibleText(retrieve), retrieve.textContent, hiddenText(retrieve), retrieve.hasAttribute('aria-label')],
                   ['skipped', 'retrieve · skipped', 'retrieve · skipped: gallery_unavailable', ': gallery_unavailable', false]);
});

test('after a stop no stage stays running: the stage and the ones not reached read "skipped (stopped)"', () => {
  const view = viewOf([START, stageStart('preprocess', 0), stageEnd('preprocess', 1, {}), stageStart('generate', 3),
                       snapshot('Findings: part'), stopOf('aborted')]);
  const items = Object.fromEntries(qa(renderTimeline(view), 'li').map((li) => [li.getAttribute('data-stage'), li]));
  assert.equal(items.preprocess.getAttribute('data-state'), 'done');
  for (const stage of ['encode', 'retrieve', 'generate', 'label', 'score']) {   // encode never started; generate was running
    assert.equal(items[stage].getAttribute('data-state'), 'skipped', stage);
    assert.equal(items[stage].textContent, `${stage} · skipped (stopped)`, stage);   // the label says why: nothing is added to it
    assert.equal(hiddenText(items[stage]), null, stage);
    assert.equal(items[stage].hasAttribute('aria-label'), false, stage);
  }
  assert.equal(qa(renderTimeline(view), 'li[data-state="running"]').length, 0);
});

test('after an error the failed stage reads error and the ones not reached read skipped', () => {
  const view = viewOf([START, stageStart('preprocess', 0), stageEnd('preprocess', 1, {}), stageStart('generate', 3),
                       step('error', { type: 'error', error: { type: 'model_error', message: 'Internal error (RuntimeError)' } }),
                       stopOf('error')]);
  const ol = renderTimeline(view);
  assert.deepEqual([stageItem(ol, 'generate').getAttribute('data-state'), visibleText(stageItem(ol, 'generate')),
                    stageItem(ol, 'generate').textContent], ['error', 'generate', 'generate, error']);
  assert.deepEqual([stageItem(ol, 'label').getAttribute('data-state'), stageItem(ol, 'label').textContent],
                   ['skipped', 'label · skipped: not_run']);
  assert.equal(qa(ol, 'li[data-state="running"]').length, 0);
});

test('a time that is missing or not a number never prints NaN', () => {
  const view = viewOf([START, step('stage_end', { stage: 'encode', detail: {} }), step('stage_end', { stage: 'preprocess', ms: 'soon', detail: {} })]);
  const ol = renderTimeline(view);
  assert.equal(stageItem(ol, 'encode').textContent, 'encode · done');   // the label already says it: nothing is added
  assert.equal(stageItem(ol, 'preprocess').textContent, 'preprocess · done');
  assert.doesNotMatch(serialize(ol), /NaN|undefined/);
});

// ---- the detail table ----------------------------------------------------------------------------------------------

test('a stage with a detail is a disclosure button: aria-expanded and aria-controls follow its table, and a click toggles both', () => {
  const card = renderAssistantCard(finished());
  const li = stageItem(card, 'encode');
  const toggle = stageButton(card, 'encode');
  const table = q(li, 'table');
  assert.ok(toggle && table, 'the done stage carries its detail and the button that opens it');
  assert.equal(toggle.localName, 'button');
  assert.equal(toggle.getAttribute('type'), 'button');
  assert.match(table.getAttribute('id'), /^[\w-]+$/);   // an id that aria-controls can name
  assert.equal(toggle.getAttribute('aria-controls'), table.getAttribute('id'));
  assert.equal(toggle.getAttribute('aria-expanded'), 'false');
  assert.equal(toggle.textContent, 'encode · 612 ms');   // the visible label
  assert.equal(toggle.getAttribute('aria-label'), 'encode · 612 ms, done');   // the spoken one starts with the visible one
  assert.deepEqual([li.getAttribute('data-stage'), li.getAttribute('data-state')], ['encode', 'done']);   // the hooks stay on the item
  assert.equal(li.hasAttribute('aria-label') || li.hasAttribute('tabindex'), false);   // the name and the focus are the button's
  assert.equal(li.classList.contains('open'), false);   // styles.css shows the table only while li.open
  toggle.click();
  assert.deepEqual([toggle.getAttribute('aria-expanded'), li.classList.contains('open')], ['true', true]);
  toggle.click();
  assert.deepEqual([toggle.getAttribute('aria-expanded'), li.classList.contains('open')], ['false', false]);

  for (const stage of ['retrieve', 'label', 'score']) {   // skipped: nothing to open, and no control
    const skippedItem = stageItem(card, stage);
    assert.equal(q(skippedItem, 'table'), null, stage);
    assert.equal(q(skippedItem, 'button'), null, stage);
    skippedItem.click();
    assert.equal(skippedItem.classList.contains('open'), false, stage);
  }
});

test('the ids a card hands out are unique, and differ between two messages', () => {
  const idsOf = (card) => qa(card, '[id]').map((n) => n.getAttribute('id'));
  const one = idsOf(renderAssistantCard(finished()));
  const two = idsOf(renderAssistantCard({ ...finished(), id: 'm_other' }));
  assert.ok(one.length >= 3);
  assert.equal(new Set(one).size, one.length);
  assert.deepEqual(one.filter((id) => two.includes(id)), []);
});

test('a stage whose detail is empty has nothing to open and no button', () => {
  const ol = renderTimeline(viewOf([START, stageStart('encode', 1), stageEnd('encode', 3, {})]));
  assert.equal(q(stageItem(ol, 'encode'), 'table'), null);
  assert.equal(q(stageItem(ol, 'encode'), 'button'), null);
});

test('only a stage with a detail can be focused: the rest of the timeline is plain text and not a tab stop', () => {
  const card = renderAssistantCard(finished(), {});
  document.body.replaceChildren(card);
  const ol = q(card, 'ol.timeline');
  assert.deepEqual(tabStops(ol).map((n) => [n.localName, n.getAttribute('data-stage')]),
                   [['button', 'preprocess'], ['button', 'encode'], ['button', 'generate']]);   // retrieve, label and score were skipped
  assert.equal(qa(ol, '[tabindex]').length, 0);
  for (const li of qa(ol, 'li')) {   // the control is the button: Enter and Space come with it, so no item listens for keys or clicks
    assert.deepEqual([(li.listeners.get('keydown') ?? []).length, (li.listeners.get('click') ?? []).length], [0, 0], li.getAttribute('data-stage'));
    li.focus();
    assert.equal(document.activeElement, document.body, `${li.getAttribute('data-stage')}: an item cannot take focus`);
  }
  stageButton(card, 'encode').focus();
  assert.equal(document.activeElement, stageButton(card, 'encode'));

  const running = renderTimeline(viewOf([START, stageStart('preprocess', 0), stageEnd('preprocess', 1, { format: 'PNG' }), stageStart('encode', 1)]));
  assert.deepEqual(tabStops(running).map((n) => n.getAttribute('data-stage')), ['preprocess']);   // encode runs, the rest are pending
  assert.equal(tabStops(renderTimeline(initialView('m1'))).length, 0);
  document.body.replaceChildren();
});

test('a click inside the open table does not close it', () => {
  const card = renderAssistantCard(finished());
  stageButton(card, 'encode').click();
  const li = stageItem(card, 'encode');
  q(li, 'td').click();   // text selected in the table
  q(li, 'th').click();
  li.click();            // or the pill around the button
  assert.equal(li.classList.contains('open'), true);
  assert.equal(stageButton(card, 'encode').getAttribute('aria-expanded'), 'true');
});

test('the table lists keys and values; arrays of numbers read inline, booleans and nothing read as words', () => {
  const table = detailTable({ patch_grid: [197, 768], pooled_dim: 512, device: 'cpu', one_pass: true, skipped_by: null, ratio: 0.123456789, none: [] }, 'encode details');
  assert.equal(table.getAttribute('aria-label'), 'encode details');
  assert.deepEqual(bodyRows(table).map(pair), [
    ['patch_grid', '197, 768'], ['pooled_dim', '512'], ['device', 'cpu'], ['one_pass', 'true'], ['skipped_by', '—'],
    ['ratio', '0.123457'], ['none', '—'],
  ]);
  assert.ok(qa(table, 'th').every((th) => th.getAttribute('scope') === 'row'));
});

test('an array of objects reads as nested rows, one per object, a column per key; nested objects nest again', () => {
  const detail = {
    image_neighbors: [
      { rank: 1, similarity: 0.91, labels: { Edema: 0, Cardiomegaly: 1 } },
      { rank: 2, similarity: 0.88, labels: { Edema: 1 } },
    ],
    gallery: { build_id: 'b1', images: 20 },
    report_matches: [],
  };
  const outer = bodyRows(detailTable(detail));
  const records = q(outer[0].children[1], 'table');
  assert.ok(records, 'the array is a table inside the cell');
  assert.deepEqual(texts(part(records, 'thead').children[0].children), ['rank', 'similarity', 'labels']);
  const rows = bodyRows(records);
  assert.equal(rows.length, 2);
  assert.deepEqual(texts(rows[1].children).slice(0, 2), ['2', '0.88']);
  const labelsOfFirst = q(rows[0], 'table');   // an object in a cell is a key/value table of its own
  assert.deepEqual(bodyRows(labelsOfFirst).map(pair), [['Edema', '0'], ['Cardiomegaly', '1']]);
  const gallery = outer.find((r) => r.children[0].textContent === 'gallery');
  assert.deepEqual(bodyRows(q(gallery, 'table')).map(pair), [['build_id', 'b1'], ['images', '20']]);
  assert.equal(outer.find((r) => r.children[0].textContent === 'report_matches').children[1].textContent, '—');
});

test('a key missing from some objects of an array leaves its cell empty, and a mixed array lists its items', () => {
  const outer = bodyRows(detailTable({ rows: [{ a: 1 }, { a: 2, b: 3 }], mixed: [1, { x: 'y' }, 'z'] }));
  assert.deepEqual(bodyRows(q(outer[0], 'table')).map((r) => texts(r.children)), [['1', '—'], ['2', '3']]);
  assert.equal(qa(outer[1], 'li').length, 3);
  assert.deepEqual(pair(bodyRows(q(outer[1], 'table'))[0]), ['x', 'y']);
  assert.doesNotMatch(serialize(outer[0]) + serialize(outer[1]), /undefined|NaN/);
});

test('a string over 200 characters is clipped with "show more", which shows it all and back; 200 is not clipped', () => {
  const long = '0123456789'.repeat(30);   // 300 characters
  const table = detailTable({ drift_note: long, short: 'x'.repeat(200) });
  const [longRow, shortRow] = bodyRows(table);
  const more = q(longRow, 'button');
  assert.equal(more.textContent, 'show more');
  assert.equal(more.getAttribute('type'), 'button');
  assert.equal(more.getAttribute('aria-expanded'), 'false');
  assert.equal(q(longRow, '.clip-text').textContent, `${long.slice(0, 200)}…`);
  more.click();
  assert.equal(q(longRow, '.clip-text').textContent, long);
  assert.equal(more.textContent, 'show less');
  assert.equal(more.getAttribute('aria-expanded'), 'true');
  more.click();
  assert.equal(q(longRow, '.clip-text').textContent, `${long.slice(0, 200)}…`);
  assert.equal(more.textContent, 'show more');
  assert.equal(q(shortRow, 'button'), null);
  assert.equal(q(shortRow, 'td').textContent, 'x'.repeat(200));
});

test('a clip never cuts a character in two', () => {
  const text = `${'a'.repeat(199)}😀${'b'.repeat(50)}`;   // the emoji is a surrogate pair across the 200th code unit
  const clipText = q(detailTable({ t: text }), '.clip-text').textContent;
  assert.equal(clipText, `${'a'.repeat(199)}…`);
  assert.doesNotMatch(clipText, /[\ud800-\udfff]/);
});

test('a very deep detail stops nesting and shows the rest as JSON text', () => {
  let deep = { leaf: 'end' };
  for (let i = 0; i < 12; i++) deep = { inner: deep };
  const table = detailTable(deep);
  assert.ok(qa(table, 'table').length <= 6);
  assert.match(table.textContent, /leaf/);
});

// ---- the report ----------------------------------------------------------------------------------------------------

test('a report splits on the literal Findings: and Impression: headers into two sections', () => {
  const report = renderReport(finished({ report: 'Findings: The lungs are clear. Impression: No acute disease.' }));
  const sections = qa(report, '.report-section');
  assert.deepEqual(sections.map((s) => [q(s, 'h3').textContent, bare(q(s, 'li'))]),   // P6-H: a list of sentences, no longer a paragraph
                   [['Findings', 'The lungs are clear.'], ['Impression', 'No acute disease.']]);
});

test('splitReport keeps text order, text before a header, a repeated header; headers are literal and case-sensitive', () => {
  assert.deepEqual(splitReport('Findings: a. Impression: b.'), [{ title: 'Findings', body: 'a.' }, { title: 'Impression', body: 'b.' }]);
  assert.deepEqual(splitReport('Impression: b. Findings: a.'), [{ title: 'Impression', body: 'b.' }, { title: 'Findings', body: 'a.' }]);
  assert.deepEqual(splitReport('No headers here.'), [{ title: null, body: 'No headers here.' }]);
  assert.deepEqual(splitReport('Lead in. Findings: a.'), [{ title: null, body: 'Lead in.' }, { title: 'Findings', body: 'a.' }]);
  assert.deepEqual(splitReport('Findings: a. Findings: b.'), [{ title: 'Findings', body: 'a.' }, { title: 'Findings', body: 'b.' }]);
  assert.deepEqual(splitReport('findings: a. IMPRESSION: b.'), [{ title: null, body: 'findings: a. IMPRESSION: b.' }]);
  assert.deepEqual(splitReport('Findings:'), [{ title: 'Findings', body: '' }]);   // a header whose text has not come yet
  assert.deepEqual(splitReport(''), []);
  assert.deepEqual(splitReport(null), []);
});

test('a report with no headers is one section without a heading', () => {
  const report = renderReport(finished({ report: 'The lungs are clear.' }));
  assert.equal(qa(report, '.report-section').length, 1);
  assert.equal(q(report, 'h3'), null);
  assert.deepEqual(lines(report), ['The lungs are clear.']);
});

test('the report is provisional exactly while view.provisional', () => {
  const live = viewOf([START, stageStart('generate', 3), snapshot('Findings: the best beam so far')]);
  assert.equal(live.provisional, true);
  const report = renderReport(live);
  assert.ok(report.classList.contains('provisional'));
  assert.ok(report.classList.contains('report'));
  assert.equal(q(report, 'p').textContent, 'the best beam so far');
  assert.equal(renderReport(finished()).classList.contains('provisional'), false);   // message_stop ends it
  assert.equal(renderReport(viewOf([START, stageStart('generate', 3), snapshot('x'), step('content_block_stop', { index: 0 })])).classList.contains('provisional'), false);
});

test('the truncated note appears when the report stopped mid-sentence, and only then', () => {
  const note = (view) => q(renderReport(view), '.note');
  assert.equal(note(finished({ truncated: true })).textContent, 'Report stopped at the token budget mid-sentence');
  assert.equal(note(finished({ truncated: false })), null);
  assert.equal(note(tinyView()).textContent, 'Report stopped at the token budget mid-sentence');   // the recorded turn hit its budget
});

test('the card says why the report ended: a repeat stop gets a quiet note and no budget note; the budget with repair on says what is hidden; repair off keeps its note (P4-G)', () => {
  const notes = (view) => texts(qa(renderReport(view), '.note'));
  const REPEAT = 'Stopped when the model began repeating itself.';
  const repeat = finished({ options: { display_repair: true, stop_on_repeat: true }, generate: { stopped: 'repeat', tokens: 21 } });
  assert.deepEqual(notes(repeat), [REPEAT]);
  assert.equal(q(renderReport(repeat), '.note.truncated'), null);   // it is not the budget's note
  const note = q(renderReport(repeat), '.note');
  assert.equal(note.getAttribute('data-stopped'), 'repeat');
  assert.equal(note.classList.contains('error') || note.classList.contains('truncated'), false);   // quiet: the plain muted note
  // a server that still flags the cut-off ending of a repeat stop: it was not the budget that ended it
  assert.deepEqual(notes(finished({ truncated: true, options: { display_repair: true }, generate: { stopped: 'repeat', tokens: 21 } })), [REPEAT]);
  assert.deepEqual(notes(finished({ truncated: true, options: { display_repair: false }, generate: { stopped: 'repeat', tokens: 21 } })), [REPEAT]);

  // a report cut off by the budget, and what the repair shows of it: the unfinished last sentence is left out of the card
  const cutOff = { report: 'Findings: The lungs are clear. Impression: No acute', display: 'Findings: The lungs are clear.' };
  const budget = finished({ ...cutOff, truncated: true, options: { display_repair: true }, generate: { stopped: 'budget', tokens: 200 } });
  assert.deepEqual(notes(budget), ['Reached the 200-token budget; the unfinished last sentence is hidden (Show raw shows it).']);
  assert.ok(q(renderReport(budget), '.note.truncated'));
  assert.deepEqual(notes(finished({ ...cutOff, truncated: true, options: { display_repair: true }, generate: { stopped: 'budget', tokens: 100 } })),
                   ['Reached the 100-token budget; the unfinished last sentence is hidden (Show raw shows it).']);   // the number is the tokens decoded
  assert.deepEqual(notes(finished({ ...cutOff, truncated: true, options: { display_repair: false }, generate: { stopped: 'budget', tokens: 100 } })),
                   ['Report stopped at the token budget mid-sentence']);   // repair off: the unfinished sentence is on the card, as the existing note says
  assert.deepEqual(notes(finished({ ...cutOff, truncated: true, generate: { stopped: 'budget', tokens: 100 } })), ['Report stopped at the token budget mid-sentence']);   // a turn that does not say
  assert.deepEqual(notes(finished({ ...cutOff, truncated: false, options: { display_repair: true }, generate: { stopped: 'budget', tokens: 100 } })), []);   // nothing was hidden: nothing to say
  assert.deepEqual(notes(finished({ ...cutOff, truncated: true, options: { display_repair: true }, generate: { stopped: undefined, tokens: 64 } })),
                   ['Reached the 64-token budget; the unfinished last sentence is hidden (Show raw shows it).']);   // a log from before the stop reason: it was the budget
  assert.deepEqual(notes(finished({ ...cutOff, truncated: true, options: { display_repair: true }, generate: { stopped: 'budget', tokens: undefined } })),
                   ['Reached the token budget; the unfinished last sentence is hidden (Show raw shows it).']);   // no number to give
  assert.deepEqual(notes(finished({ ...cutOff, truncated: true, options: { display_repair: true }, generate: { stopped: 'timeout', tokens: 87 } })),
                   ['Report stopped at the token budget mid-sentence']);   // any other reason keeps the plain note
});

test('an EOS stop has no note: the model ended the report itself, so nothing was cut off and no budget is named (P9-G2)', () => {
  const notes = (view) => texts(qa(renderReport(view), '.note'));
  const cutOff = { report: 'Findings: The lungs are clear. Impression: No acute', display: 'Findings: The lungs are clear.' };
  const eos = { stopped: 'eos', tokens: 87 };
  assert.deepEqual(notes(finished({ generate: eos })), []);
  assert.equal(q(renderReport(finished({ generate: eos })), '.note'), null);
  // what the engine sends is truncated false whatever the text ends in; a server that flagged it anyway still gets no budget note
  for (const display_repair of [true, false, undefined]) {
    for (const truncated of [false, true]) {
      assert.deepEqual(notes(finished({ ...cutOff, truncated, options: { display_repair }, generate: eos })), [], `repair ${display_repair}, truncated ${truncated}`);
    }
  }
  assert.deepEqual(notes(finished({ options: { display_repair: true, stop_on_repeat: true }, generate: eos })), []);   // the switch being on changes nothing
  // the neighbours keep their notes: the repeat's, and the budget's
  assert.deepEqual(notes(finished({ generate: { stopped: 'repeat', tokens: 21 } })), ['Stopped when the model began repeating itself.']);
  assert.deepEqual(notes(finished({ ...cutOff, truncated: true, generate: { stopped: 'budget', tokens: 87 } })), ['Report stopped at the token budget mid-sentence']);
  // the turn's report is still the card's body: only the note is absent
  assert.equal(bare(q(renderReport(finished({ generate: eos })), '.report-body li')), 'The lungs are clear.');
});

test('with the repair on, the budget note says what is hidden only when the card does hide something: no complete sentence is kept as it is (P4-G fix 1)', () => {
  const notes = (view) => texts(qa(renderReport(view), '.note'));
  const PLAIN = ['Report stopped at the token budget mid-sentence'];
  const repair = { options: { display_repair: true }, generate: { stopped: 'budget', tokens: 16 }, truncated: true };
  // the repair has no complete sentence to cut back to, keeps the text as it is and calls that a fallback: the card shows all of it
  assert.deepEqual(notes(finished({ ...repair, report: 'The lungs are clear and the heart', display: 'The lungs are clear and the heart' })), PLAIN);
  assert.deepEqual(notes(finished({ ...repair, report: 'The lungs   are\nclear', display: 'The lungs are clear' })), PLAIN);   // spacing alone hides nothing
  assert.deepEqual(notes(finished({ ...repair, report: 'The lungs are clear. The heart', display: 'The lungs are clear.' })),
                   ['Reached the 16-token budget; the unfinished last sentence is hidden (Show raw shows it).']);   // a sentence left out: say so
  assert.deepEqual(notes(finished({ ...repair, report: 'The lungs are clear. The lungs are clear. The heart', display: 'The lungs are clear.' })),
                   ['Reached the 16-token budget; the unfinished last sentence is hidden (Show raw shows it).']);   // repeats and a fragment left out
});

test('the recorded turn keeps its budget note, and a card that has not reached generate has no reason to give yet', () => {
  const notesOf = (view) => texts(qa(renderReport(view), '.note'));
  assert.deepEqual(notesOf(tinyView()), ['Report stopped at the token budget mid-sentence']);   // the recorded turn stopped at its budget, repair off
  const early = viewOf([START, stageStart('generate', 3), snapshot('Findings: so far.')]);
  assert.deepEqual(notesOf(early), []);   // the best beam so far
});

test('Show raw swaps the display copy for the raw report and back; the button says which it is on', () => {
  const view = finished({ report: 'Findings: a lung  clear Impression: none and', display: 'Findings: A lung is clear. Impression: None.' });
  const card = renderAssistantCard(view, { copy() {} });
  const report = q(card, '.report');
  assert.deepEqual(lines(report), ['A lung is clear.', 'None.']);   // the display copy, in sections, a sentence to a line
  assert.equal(q(report, 'pre'), null);
  const raw = button(report, 'Show raw');
  assert.equal(raw.getAttribute('aria-pressed'), 'false');
  raw.click();
  assert.equal(q(report, 'pre').textContent, 'Findings: a lung  clear Impression: none and');   // verbatim, unsplit
  assert.equal(q(report, '.report-section'), null);
  assert.equal(raw.getAttribute('aria-pressed'), 'true');
  assert.equal(raw.textContent, 'Show raw');   // the label stays; aria-pressed carries the state
  raw.click();
  assert.deepEqual(lines(report), ['A lung is clear.', 'None.']);
  assert.equal(q(report, 'pre'), null);
  assert.equal(raw.getAttribute('aria-pressed'), 'false');
});

test('Copy hands ctx.copy the text on show, the display copy or the raw one; without ctx.copy there is no button', () => {
  const view = finished({ report: 'Findings: raw text.', display: 'Findings: display text.' });
  const copied = [];
  const report = renderReport(view, { copy: (t) => copied.push(t) });
  button(report, 'Copy').click();
  button(report, 'Show raw').click();
  button(report, 'Copy').click();
  assert.deepEqual(copied, ['Findings: display text.', 'Findings: raw text.']);
  assert.equal(button(renderReport(view, {}), 'Copy'), null);
  assert.equal(button(renderReport(view), 'Copy'), null);
  assert.ok(button(renderReport(view, {}), 'Show raw'), 'Show raw needs nothing from the page');
});

test('there is no report block, and no button, before the first snapshot', () => {
  const report = renderReport(viewOf([START]));
  assert.ok(report.hasAttribute('hidden'));
  assert.equal(qa(report, 'button').length, 0);
});

// ---- the report card a person can read (P6-H) ----------------------------------------------------------------------
//
// A finished report reads as a plain summary of its labels, then its sentences one to a line, then what explains them: a note for a [removed], a
// note for a ⚠ sentence, and the Terms. All of it is drawn from the display copy as it is: the text of a sentence is the clean display's own,
// apart from the [removed] that stands for a run of underscores, the ⚠ in front of a sentence about an earlier X-ray, and the number after a term.
// Every sentence here is neutral and made up for the test (R7): no line of a real report.

const REMOVED_NOTE = '[removed] marks text the training reports hid for privacy, usually a date or a name. The model copies the pattern, so nothing is missing from your image.';
const EARLIER_NOTE = '⚠ marks a sentence that refers to an earlier X-ray. The model never saw one; it writes the way reports are written.';
const SUMMARY_TITLE = 'What this means (plain language)';
const SUMMARY_NOTE = "A plain-language summary of the findings detected in the model's report. It is not a diagnosis.";

// A sentence as the clean display has it: without the ⚠ and the number of a term, which are drawn on it (a [removed] is text, and stays).
const bare = (node) => (node.nodeType === 3 ? node.data
  : node.localName === 'sup' || node.classList.contains('flag') ? '' : node.childNodes.map(bare).join(''));
const lines = (report) => qa(report, '.report-section li').map(bare);                       // the sentences of the card, in order
const notesOf = (report) => texts(qa(report, '.report-key'));                               // the notes under the sentences
const termsOf = (report) => texts(qa(report, '.report-terms ol > li'));                     // the Terms list
const reportOf = (text, extra = {}, ctx = {}) => renderReport(finished({ report: text, ...extra }), ctx);
const classes = (nodes) => nodes.map((n) => n.getAttribute('class'));

const GLOSSARY_TABLE = [   // the plain explanations as the user was given them: each row is its terms, then what it says
  ['no finding', 'no abnormality is reported'],
  ['enlarged cardiomediastinum', 'the heart and the central chest look wider than normal'],
  ['cardiomegaly', 'an enlarged heart'],
  ['lung opacity / opacity', 'an area of the lung that looks whiter, or denser, than normal'],
  ['lung lesion / lesion', 'a spot or mass in the lung'],
  ['edema', 'fluid in the lung tissue'],
  ['consolidation', 'part of the lung filled with fluid or inflammation instead of air'],
  ['pneumonia', 'a lung infection'],
  ['atelectasis', 'part of a lung is partly collapsed or not fully inflated'],
  ['pneumothorax', 'air between the lung and the chest wall, which can make the lung collapse'],
  ['pleural effusion / effusion', 'fluid between the lung and the chest wall'],
  ['pleural thickening', 'a thickened lining around the lung'],
  ['fracture', 'a broken bone, for example a rib'],
  ['support devices / device', 'tubes, lines, wires or devices placed in the body'],
  ['vascular congestion', 'the blood vessels in the lungs look fuller than normal'],
  ['hilar / hila', 'around the roots of the lungs, where vessels and airways enter'],
  ['basilar / base / bases', 'the lower parts of the lungs'],
  ['costophrenic angle', 'the lower outer corner of the lung on the image'],
  ['mediastinum / mediastinal', 'the central chest between the lungs'],
  ['interstitial', 'in the tissue between the air sacs'],
  ['nodule', 'a small round spot'],
  ['granuloma', 'a small healed spot, often from an old infection'],
  ['hemidiaphragm', 'one half of the breathing muscle under the lungs'],
  ['emphysema', 'damaged air sacs that trap air'],
  ['sternotomy wires', 'wires that close the breastbone after chest surgery'],
  ['endotracheal tube', 'a breathing tube in the windpipe'],
  ['nasogastric tube', 'a tube through the nose into the stomach'],
  ['picc / picc line', 'a thin tube in an arm vein that reaches near the heart'],
  ['pacemaker', 'a device that helps the heart keep a regular beat'],
  ['scoliosis', 'a sideways curve of the spine'],
  ['osseous', 'about the bones'],
  ['focal', 'in one small area'],
  ['bilateral', 'on both sides'],
  ['acute', 'new or sudden'],
  ['chronic', 'long-standing'],
];

test('the glossary is the table of plain explanations as given, word for word, the 14 labels first (P6-H)', async () => {
  const { GLOSSARY } = await import('../../app/static/glossary.js');
  assert.deepEqual(GLOSSARY.map((row) => [row.terms.join(' / '), row.explanation]), GLOSSARY_TABLE);
  assert.equal(GLOSSARY.length, 35);
  assert.ok(Object.isFrozen(GLOSSARY) && GLOSSARY.every((row) => Object.isFrozen(row) && Object.isFrozen(row.terms)), 'static data nobody can edit');
});

test('splitSentences cuts after . ? ! and keeps every sentence as it was written, its spacing included (P6-H)', () => {
  assert.deepEqual(splitSentences('Moderate left pleural effusion. Heart size is normal. No pneumothorax.'),
                   ['Moderate left pleural effusion.', 'Heart size is normal.', 'No pneumothorax.']);
  assert.deepEqual(splitSentences('Is it clear? Yes! It is.'), ['Is it clear?', 'Yes!', 'It is.']);
  assert.deepEqual(splitSentences('Heart  size is   normal.\n  No pneumothorax.'), ['Heart  size is   normal.', 'No pneumothorax.']);
  assert.deepEqual(splitSentences('  One.   Two.  '), ['One.', 'Two.']);
  assert.deepEqual(splitSentences('The heart size is normal'), ['The heart size is normal']);   // no full stop: still one line
  assert.deepEqual(splitSentences('A whole sentence. and a fragment'), ['A whole sentence.', 'and a fragment']);
  for (const nothing of ['', '   ', '\n', null, undefined, 7, {}]) assert.deepEqual(splitSentences(nothing), [], String(nothing));
  const text = 'Heart size is normal. No pneumothorax.  Left vs. right. Seen at 3 p.m. today.';
  assert.equal(splitSentences(text).join(' '), text.replace(/\s+/g, ' '));   // nothing is lost, nothing is added
});

test('splitSentences does not split inside a decimal or after an abbreviation: Dr. e.g. i.e. vs. approx. No. before a digit, St. a.m. p.m. (P6-H)', () => {
  const whole = (text) => assert.deepEqual(splitSentences(text), [text], text);
  whole('The nodule measures 1.5 cm in the left upper lobe.');
  whole('It grew from 0.8 to 1.25 cm.');
  whole('Reviewed by Dr. Lee today.');
  whole('A small finding, e.g. a rib fracture.');
  whole('The left lower lobe, i.e. the base.');
  whole('Left vs. right is not told apart.');
  whole('The lesion is approx. 3 cm wide.');
  whole('The tube is No. 3 in the series.');
  whole('Seen at St. Mary clinic.');
  whole('Taken at 3 a.m. today and again at 4 p.m. today.');
  whole('Seen by Mr. Lee and Mrs. Day and Ms. Fox and Prof. Ray.');   // the other titles, which no sentence ends on
  whole('Tubes, lines, etc. are in place.');
  whole('See Fig. 2 and cf. the film.');
  assert.deepEqual(splitSentences('The nodule measures 1.5 cm. It is small. Dr. Lee agrees.'), ['The nodule measures 1.5 cm.', 'It is small.', 'Dr. Lee agrees.']);
  assert.deepEqual(splitSentences('Pneumothorax: No. Effusion: small.'), ['Pneumothorax: No.', 'Effusion: small.']);   // "No." is guarded before a digit, and only there
  assert.deepEqual(splitSentences('Tube No. 3 is in place. No. 4 is not.'), ['Tube No. 3 is in place.', 'No. 4 is not.']);   // before a digit it is a number, wherever it stands
  assert.deepEqual(splitSentences('1. Small effusion. 2. Normal heart size.'), ['1. Small effusion.', '2. Normal heart size.']);   // a list number is no stop
  assert.deepEqual(splitSentences('Compared with the image from ___. The lungs are clear.'), ['Compared with the image from ___.', 'The lungs are clear.']);
  assert.deepEqual(splitSentences('Wait ... then more.'), ['Wait ... then more.']);   // dots with nothing before them end nothing
});

test('a sentence that compares with an earlier study is told by a short list of words, each pinned (P6-H)', () => {
  const flagged = [
    'Compared with the image from ___, the lungs are clear.', 'Comparison: none.',
    'There is a prior sternotomy.', 'Previous studies are not available.', 'The tube was previously removed.',
    'Since the last film a tube was placed.', 'Since prior, the lungs have cleared.',
    'Interval development of a tube.', 'Interval change is noted.',
    'Unchanged heart size.', 'No change in the heart size.', 'No significant change in the lungs.', 'Little change in the lungs.', 'Little overall change in the lungs.',
    'Stable position of the tube.',
    'Again, a left tube is seen.', 'Again seen is a left tube.',
    'Persistent left tube.', 'The tube persists.',
    'A rib is redemonstrated.', 'A rib is re-demonstrated.',
    'The tube is in the same place as before.',
    'The lungs have improved.', 'The lungs are improving.', 'The lungs have worsened.', 'Worsening of the lungs.',
    'The heart has increased in size.', 'The heart has decreased in size.',
  ];
  for (const sentence of flagged) assert.equal(refersToEarlier(sentence), true, sentence);
  assert.equal(refersToEarlier('COMPARED WITH THE STUDY OF ___.'), true);   // case does not matter
  const plain = [
    'No pneumothorax.', 'No acute abnormality in the upper zones.', 'There is a new left tube.', 'Increased opacity at the left base.',
    'Heart size is normal.', 'The lungs are clear.', 'The two lungs are comparable in size.',   // comparable is not compared
    'The patient is unstable.', 'Priority is given to the left side.', 'The lungs are persistently clear.',   // a word that only holds another
    'There is again a small left tube.',   // "again" counts only where a sentence begins with it
    'Nothing has changed color.', 'A change in the chart.', 'The heart size is increased.',   // no temporal word: not a comparison
    '', 'Findings:',
  ];
  for (const sentence of plain) assert.equal(refersToEarlier(sentence), false, sentence);
  assert.equal(refersToEarlier(null), false);
  assert.equal(refersToEarlier(undefined), false);
});

test('the summary is made of the labels and the glossary only: positives in a fixed order, each as its plain phrase and its term (P6-H)', () => {
  const said = (...positives) => summaryText(labelled(positives));
  const PHRASES = [
    ['Enlarged Cardiomediastinum', 'the heart and the central chest look wider than normal (enlarged cardiomediastinum)'],
    ['Cardiomegaly', 'an enlarged heart (cardiomegaly)'],
    ['Lung Opacity', 'an area of the lung that looks whiter, or denser, than normal (lung opacity)'],
    ['Lung Lesion', 'a spot or mass in the lung (lung lesion)'],
    ['Edema', 'fluid in the lung tissue (edema)'],
    ['Consolidation', 'part of the lung filled with fluid or inflammation instead of air (consolidation)'],
    ['Pneumonia', 'a lung infection (pneumonia)'],
    ['Atelectasis', 'part of a lung is partly collapsed or not fully inflated (atelectasis)'],
    ['Pneumothorax', 'air between the lung and the chest wall, which can make the lung collapse (pneumothorax)'],
    ['Pleural Effusion', 'fluid between the lung and the chest wall (pleural effusion)'],
    ['Pleural Other', 'a thickened lining around the lung (pleural thickening)'],
    ['Fracture', 'a broken bone, for example a rib (fracture)'],
    ['Support Devices', 'tubes, lines, wires or devices placed in the body (support devices)'],
  ];
  for (const [name, phrase] of PHRASES) assert.equal(said(name), `The report mentions: ${phrase}.`, name);
  assert.equal(said('Pleural Effusion', 'Atelectasis'),   // CHEXBERT_14 order, whatever order the labels came in
                   'The report mentions: part of a lung is partly collapsed or not fully inflated (atelectasis); fluid between the lung and the chest wall (pleural effusion).');
  const every = PHRASES.map(([name]) => name);
  const all = `The report mentions: ${PHRASES.map(([, phrase]) => phrase).join('; ')}.`;
  assert.equal(summaryText(Object.fromEntries([...every].reverse().map((n) => [n, 1]).concat([['No Finding', 0]]))), all);   // the last in, the last said
  assert.ok(all.endsWith('(support devices).'), 'Support Devices comes last');
  assert.equal(summaryText(Object.fromEntries([['Support Devices', true], ['Edema', true], ['Cardiomegaly', 0]])),   // true counts as 1
               'The report mentions: fluid in the lung tissue (edema); tubes, lines, wires or devices placed in the body (support devices).');
  const NONE = 'The report mentions no abnormal finding.';
  assert.equal(said(), NONE);                                       // no positive label
  assert.equal(said('No Finding'), NONE);                           // No Finding positive
  assert.equal(said('No Finding', 'Edema', 'Cardiomegaly'), NONE);  // it is what the labeller says of the whole report
  for (const unknown of [null, undefined, {}, [], 'Edema', 7, { Edema: null }, { Unheard: 1 }, { Unheard: 0, Other: null }]) {
    assert.equal(summaryText(unknown), null, JSON.stringify(unknown));   // nothing to say: it says nothing
  }
  assert.equal(summaryText({ Edema: 1, Unheard: 1 }), 'The report mentions: fluid in the lung tissue (edema).');   // a name it has no phrase for is left out
});

test('a finished report is a list under each heading, one sentence to a line, in the text as the clean display has it (P6-H)', () => {
  const report = reportOf('Findings: The lungs are clear.  The heart size is normal. Impression: Nothing else to report.');
  const sections = qa(report, '.report-section');
  assert.deepEqual(sections.map((s) => q(s, 'h3').textContent), ['Findings', 'Impression']);
  assert.deepEqual(sections.map((s) => qa(s, 'li').map((li) => li.textContent)), [['The lungs are clear.', 'The heart size is normal.'], ['Nothing else to report.']]);
  assert.ok(sections.every((s) => q(s, 'ul') !== null && q(s, 'p') === null), 'a list, and no paragraph');
  assert.equal(q(report, '.report-body').children.length, 2);
  assert.deepEqual(notesOf(report), []);                         // nothing to explain: no note,
  assert.equal(q(report, '.report-terms'), null);                // no Terms,
  assert.equal(q(report, '.report-summary'), null);              // and, with no labels, no summary: the card is what it was but for the list
  assert.deepEqual(classes([...report.children]), ['report-body', 'report-actions']);
  const untitled = reportOf('The lungs are clear. Nothing else to report.');   // untitled text shows without a heading
  assert.equal(q(untitled, 'h3'), null);
  assert.deepEqual(lines(untitled), ['The lungs are clear.', 'Nothing else to report.']);
  assert.deepEqual(lines(reportOf('Findings:')), []);   // a header whose text has not come yet: its heading, and no list
  assert.equal(q(reportOf('Findings:'), 'ul'), null);
});

test('while the report is provisional the plain display stays: a paragraph a section, no list, no summary, no notes, no terms (P6-H)', () => {
  const live = viewOf([START, stageStart('generate', 3), snapshot('Findings: Compared with the image from ___, a pleural effusion. Impression: Stable.')]);
  const report = renderReport(live, { copy() {} });
  assert.ok(report.classList.contains('provisional'));
  assert.deepEqual(texts(qa(report, '.report-section p')), ['Compared with the image from ___, a pleural effusion.', 'Stable.']);   // the text as it is, underscores and all
  for (const selector of ['li', 'ul', '.report-summary', '.report-extras', '.report-key', '.report-terms', '.removed', '.term', '.flag']) {
    assert.equal(q(report, selector), null, selector);
  }
  const odd = renderReport({ ...finished({ labels: labelled(['Edema']), report: 'Findings: A pleural effusion.' }), provisional: true });   // labels that came before the block closed
  assert.equal(q(odd, '.report-summary'), null);
  assert.equal(qa(odd, 'li').length, 0);
});

test('a run of three or more underscores is shown as a small [removed], and a note says why, once and only when there is one (P6-H)', () => {
  const report = reportOf('Findings: The date was ___. The name is _____ and the code ________. Two underscores __ stay. Impression: Nothing else.');
  assert.deepEqual(texts(qa(report, '.removed')), ['[removed]', '[removed]', '[removed]']);
  assert.ok(qa(report, '.removed').every((n) => n.localName === 'span' && n.parentNode.localName === 'li'), 'inline, inside its sentence');
  assert.deepEqual(lines(report), ['The date was [removed].', 'The name is [removed] and the code [removed].', 'Two underscores __ stay.', 'Nothing else.']);
  assert.ok(!qa(report, '.report-section').map((s) => s.textContent).join('').includes('___'), 'no run of underscores is left on the card');
  assert.deepEqual(notesOf(report), [REMOVED_NOTE]);   // once, however many there are
  assert.equal(q(report, '.report-key').getAttribute('data-note'), 'removed');
  assert.deepEqual(notesOf(reportOf('Findings: Two underscores __ stay. A single _ too.')), []);   // none to hide: none to explain
  assert.deepEqual(notesOf(reportOf('Findings: The lungs are clear.')), []);
  assert.deepEqual(notesOf(reportOf('Findings: The date was ___. Impression: Nothing else.', {}, { copy() {} })), [REMOVED_NOTE]);
});

test('a sentence about an earlier study gets a quiet ⚠ and a muted look, and a note says what ⚠ means, once and only then (P6-H)', () => {
  const report = reportOf('Findings: Compared with the image from ___, the lungs are clear. The heart size is normal. Impression: Stable. Nothing else to report.');
  const flagged = qa(report, '.report-section li.earlier');
  assert.deepEqual(flagged.map(bare), ['Compared with the image from [removed], the lungs are clear.', 'Stable.']);
  assert.deepEqual(flagged.map((li) => texts(qa(li, '.flag'))), [['⚠'], ['⚠']]);
  assert.equal(qa(report, '.flag').length, 2);   // not on the others
  assert.equal(qa(report, '.report-section li').length, 4);
  const flag = q(flagged[0], '.flag');
  assert.equal(flag.getAttribute('role'), 'img');                                    // a shape that is also said in words
  assert.equal(flag.getAttribute('aria-label'), 'refers to an earlier X-ray');
  assert.equal(flag.getAttribute('title'), 'This sentence refers to an earlier X-ray. The model never saw one.');
  assert.equal(flagged[0].childNodes[0], flag, 'in front of its sentence');
  assert.deepEqual(notesOf(report), [REMOVED_NOTE, EARLIER_NOTE]);                   // the ⚠ is told once, after the [removed]
  assert.deepEqual(classes(qa(report, '.report-key')), ['report-key', 'report-key']);
  assert.deepEqual(qa(report, '.report-key').map((n) => n.getAttribute('data-note')), ['removed', 'earlier']);
  const alone = reportOf('Findings: The lungs are clear. Stable position of the tube. No change in the heart size.');
  assert.deepEqual(notesOf(alone), [EARLIER_NOTE]);                                  // told once for two sentences
  assert.equal(qa(alone, '.flag').length, 2);
  assert.deepEqual(notesOf(reportOf('Findings: No pneumothorax. There is a new left tube. Increased opacity at the base.')), []);   // none flagged: no note
});

test('terms get a dotted-underline hook, a number and a title; each is numbered once, in the order it is first used, and explained once in the Terms list (P6-H)', () => {
  const report = reportOf('Findings: Moderate left pleural effusion with basilar atelectasis. The heart size is normal. No pneumothorax. Impression: Left pleural effusion.');
  assert.deepEqual(lines(report), ['Moderate left pleural effusion with basilar atelectasis.', 'The heart size is normal.', 'No pneumothorax.', 'Left pleural effusion.']);
  const marked = qa(report, '.term');
  assert.deepEqual(marked.map(bare), ['pleural effusion', 'basilar', 'atelectasis', 'pneumothorax']);   // the second "pleural effusion" has its number already
  assert.deepEqual(marked.map((t) => q(t, 'sup').textContent), ['1', '2', '3', '4']);
  assert.deepEqual(marked.map((t) => t.getAttribute('title')), [
    '1: fluid between the lung and the chest wall', '2: the lower parts of the lungs', '3: part of a lung is partly collapsed or not fully inflated',
    '4: air between the lung and the chest wall, which can make the lung collapse']);
  assert.ok(marked.every((t) => t.localName === 'span' && t.parentNode.localName === 'li'), 'inline, in its sentence');
  const first = qa(report, '.report-section li')[0];
  assert.deepEqual(first.childNodes.map((n) => (n.nodeType === 3 ? n.data : n.getAttribute('class'))), ['Moderate left ', 'term', ' with ', 'term', ' ', 'term', '.']);   // the text round a term is untouched
  const list = q(report, '.report-terms');
  assert.equal(q(list, 'h4').textContent, 'Terms');
  assert.deepEqual(termsOf(report), [
    'pleural effusion: fluid between the lung and the chest wall', 'basilar: the lower parts of the lungs',
    'atelectasis: part of a lung is partly collapsed or not fully inflated',
    'pneumothorax: air between the lung and the chest wall, which can make the lung collapse']);
  assert.equal(q(list, 'ol') !== null && q(list, 'ol').parentNode === list, true);   // an ordered list: its own numbers are the numbers in the text
  // nothing in it can be tabbed to: spans with a title, and a list
  assert.deepEqual(tabStops(reportOf('Findings: A pleural effusion. Stable.', {}, { copy() {} })).map((n) => n.getAttribute('data-action')), ['copy', 'raw']);
});

test('a term is found whatever its case, on word boundaries only, and a longer term wins over a shorter one inside it (P6-H)', () => {
  const upper = reportOf('Findings: PLEURAL EFFUSION and Pneumonia.');
  assert.deepEqual(qa(upper, '.term').map(bare), ['PLEURAL EFFUSION', 'Pneumonia']);   // the text as written
  assert.deepEqual(termsOf(upper), ['pleural effusion: fluid between the lung and the chest wall', 'pneumonia: a lung infection']);   // the table's own lower case
  const inside = reportOf('Findings: Two nodules, a subacute change, a unifocal area, a baseline film and lung basal fullness. The lungs are clear.');
  assert.equal(qa(inside, '.term').length, 0);          // nodules, subacute, unifocal, baseline: each holds a term and is not one
  assert.equal(q(inside, '.report-terms'), null);
  assert.deepEqual(qa(reportOf('Findings: The lung Bases are clear. Wide base. Chronic and focal and bilateral.'), '.term').map(bare),
                   ['Bases', 'Chronic', 'focal', 'bilateral']);   // base is the same term as bases: the second one has its number
  const longest = reportOf('Findings: A left lung opacity and a small opacity.');
  assert.deepEqual(qa(longest, '.term').map(bare), ['lung opacity']);   // not "lung" and "opacity" apart; the same row once
  assert.deepEqual(termsOf(longest), ['lung opacity: an area of the lung that looks whiter, or denser, than normal']);
  assert.deepEqual(termsOf(reportOf('Findings: A PICC line is in place.')), ['picc line: a thin tube in an arm vein that reaches near the heart']);   // not "picc"
  assert.deepEqual(termsOf(reportOf('Findings: An endotracheal tube, a nasogastric tube and sternotomy wires are in place.')), [
    'endotracheal tube: a breathing tube in the windpipe', 'nasogastric tube: a tube through the nose into the stomach',
    'sternotomy wires: wires that close the breastbone after chest surgery']);
  assert.deepEqual(termsOf(reportOf('Findings: A small effusion. Another pleural effusion.')), ['effusion: fluid between the lung and the chest wall']);   // one row: its first words
  assert.deepEqual(qa(reportOf('Findings: A small effusion. Another pleural effusion.'), '.term').map(bare), ['effusion']);
  assert.deepEqual(qa(reportOf('Findings: Costophrenic angle and vascular congestion at the hila.'), '.term').map((t) => q(t, 'sup').textContent), ['1', '2', '3']);
  assert.deepEqual(qa(reportOf('Findings: Pleural  effusion\nat the base.'), '.term').map(bare), ['Pleural  effusion', 'base']);   // a term may span a space or a line break
});

test('a [removed] and a ⚠ and a term can share a sentence: each is drawn, and the text round it is the display copy (P6-H)', () => {
  const report = reportOf('Findings: Compared with ___, the pleural effusion is stable at the base. Impression: Nothing else.');
  const [first] = qa(report, '.report-section li');
  assert.equal(bare(first), 'Compared with [removed], the pleural effusion is stable at the base.');
  assert.ok(first.classList.contains('earlier') && q(first, '.flag') && q(first, '.removed') && qa(first, '.term').length === 2);
  assert.equal(visibleText(first), '⚠Compared with [removed], the pleural effusion1 is stable at the base2.');
  assert.deepEqual(termsOf(report), ['pleural effusion: fluid between the lung and the chest wall', 'base: the lower parts of the lungs']);
});

test('the summary comes above the report once the labels are known: a box, its plain sentence, and under it what it is and is not (P6-H)', () => {
  const report = reportOf('Findings: A pleural effusion. Impression: Nothing else.', { labels: labelled(['Pleural Effusion', 'Atelectasis']) });
  const summary = q(report, '.report-summary');
  assert.deepEqual(classes([...summary.children]), ['summary-box', 'summary-note']);   // the box, and under it the note
  assert.equal(q(summary, '.summary-box h4').textContent, SUMMARY_TITLE);
  assert.equal(q(summary, '.summary-box p').textContent,
               'The report mentions: part of a lung is partly collapsed or not fully inflated (atelectasis); fluid between the lung and the chest wall (pleural effusion).');
  assert.equal(q(summary, '.summary-note').textContent, SUMMARY_NOTE);
  assert.deepEqual(classes([...report.children]), ['report-summary', 'report-body', 'report-extras', 'report-actions']);   // above the sentences
  assert.equal(q(reportOf('Findings: Nothing else.', { labels: labelled([]) }), '.summary-box p').textContent, 'The report mentions no abnormal finding.');
  assert.equal(q(reportOf('Findings: Nothing else.', { labels: labelled(['No Finding']) }), '.summary-box p').textContent, 'The report mentions no abnormal finding.');
  assert.equal(q(reportOf('Findings: Nothing else.', { labels: labelled(['No Finding', 'Edema']) }), '.summary-box p').textContent, 'The report mentions no abnormal finding.');
  const box = q(renderReport(finished({ labels: labelled(['Edema']) }), { copy() {} }), '.report-summary');
  assert.deepEqual(tabStops(box), []);   // a box to read, not a control
});

test('there is no summary while the labels are pending, when there are none, while the report is provisional; public mode has it as private does (P6-H)', () => {
  const closed = [START, stageStart('generate', 3), snapshot('Findings: The lungs are clear.'), step('content_block_stop', { index: 0 }), stageEnd('generate', 9, GENERATE)];
  const pending = renderReport(viewOf([...closed, stageStart('label', 4)]));   // the report is final, its labels are on their way
  assert.equal(q(pending, '.report-summary'), null);
  assert.deepEqual(lines(pending), ['The lungs are clear.']);                   // the sentences come as soon as the block closes
  assert.equal(q(renderReport(viewOf([...closed, skipped('label', 'labeler_unavailable')])), '.report-summary'), null);   // no labeller: the card is as it was
  assert.equal(q(reportOf('Findings: The lungs are clear.'), '.report-summary'), null);   // the label stage skipped
  assert.equal(q(reportOf('Findings: The lungs are clear.', { options: { label: false } }), '.report-summary'), null);   // the user's own setting
  const labelsOnly = viewOf([...closed, stageEnd('label', 4, { chexbert_14: labelled(['Edema']) })]);   // labelled, the turn still scoring
  assert.equal(q(renderReport(labelsOnly), '.report-summary .summary-box p').textContent, 'The report mentions: fluid in the lung tissue (edema).');
  assert.equal(q(renderReport({ ...labelsOnly, provisional: true }), '.report-summary'), null);
  const publicTurn = { ...finished({ labels: labelled(['Edema']), report: 'Findings: The lungs are clear.' }), mode: 'public' };
  assert.equal(q(renderReport(publicTurn), '.report-summary .summary-box p').textContent, 'The report mentions: fluid in the lung tissue (edema).');
  assert.equal(q(renderReport({ ...publicTurn, labels: {} }), '.report-summary'), null);   // labels with nothing in them say nothing
});

test('a finished report reads in this order: the summary, the sentences, the stop note, the notes, the terms, then its buttons (P6-H)', () => {
  const text = 'Findings: Compared with the image from ___, a small pleural effusion is seen. The heart size is normal. Impression: Nothing else.';
  const report = reportOf(text, { labels: labelled(['Pleural Effusion']), truncated: true }, { copy() {} });
  assert.deepEqual(classes([...report.children]), ['report-summary', 'report-body', 'note truncated', 'report-extras', 'report-actions']);
  assert.deepEqual(classes([...q(report, '.report-extras').children]), ['report-key', 'report-key', 'report-terms']);
  assert.deepEqual(notesOf(report), [REMOVED_NOTE, EARLIER_NOTE]);
  assert.deepEqual(texts(qa(report, '.report .note')), ['Report stopped at the token budget mid-sentence']);   // the stop note is still the report's one `.note`
});

test('Show raw swaps the whole card text for the original, notes and terms with it, and Copy never carries a mark (P6-H)', () => {
  const text = 'Findings: Compared with the image from ___, a pleural effusion is stable.  Impression: Nothing else.';
  const copied = [];
  const report = reportOf(text, { labels: labelled(['Pleural Effusion']) }, { copy: (t) => copied.push(t) });
  assert.deepEqual(notesOf(report), [REMOVED_NOTE, EARLIER_NOTE]);
  assert.equal(q(report, '.report-terms') !== null, true);
  button(report, 'Copy').click();
  button(report, 'Show raw').click();
  assert.equal(q(report, 'pre').textContent, text);                               // the original, byte for byte: underscores, spacing, all
  assert.equal(q(report, '.report-body'), null);
  assert.deepEqual(notesOf(report), []);                                          // nothing in view for them to explain
  assert.equal(q(report, '.report-terms'), null);
  assert.equal(q(report, '.term') === null && q(report, '.removed') === null && q(report, '.flag') === null, true);
  assert.equal(q(report, '.report-summary .summary-box p').textContent, 'The report mentions: fluid between the lung and the chest wall (pleural effusion).');   // it is about the report, whichever way it is read
  assert.equal(button(report, 'Show raw').getAttribute('aria-pressed'), 'true');
  button(report, 'Copy').click();
  button(report, 'Show raw').click();                                             // and back
  assert.equal(q(report, 'pre'), null);
  assert.deepEqual(lines(report), ['Compared with the image from [removed], a pleural effusion is stable.', 'Nothing else.']);
  assert.deepEqual(notesOf(report), [REMOVED_NOTE, EARLIER_NOTE]);
  assert.equal(termsOf(report).length, 1);
  assert.deepEqual(copied, [text, text]);                                         // the clean text (this turn's display copy is its report), with no marker, number or note
  assert.ok(copied.every((t) => !/\[removed\]|⚠|Terms/.test(t)));
});

test('a card drawn anew while Show raw is on keeps the original in place of the notes and terms, and the next click brings them back (P6-H)', () => {
  const view = finished({ report: 'Findings: A pleural effusion. Stable position of the tube.' });
  const ui = new Map();
  const ctx = { ui, copy() {} };
  button(renderReport(view, ctx), 'Show raw').click();
  const again = renderReport(view, ctx);                                          // the next throttled frame rebuilds the whole card
  assert.equal(q(again, 'pre').textContent, 'Findings: A pleural effusion. Stable position of the tube.');
  assert.equal(button(again, 'Show raw').getAttribute('aria-pressed'), 'true');
  assert.equal(q(again, '.report-body'), null);
  assert.equal(q(again, '.report-extras').hasAttribute('hidden'), true);          // nothing in view for the notes and terms to explain
  assert.deepEqual([notesOf(again), termsOf(again), qa(again, '.term, .flag').length], [[], [], 0]);
  button(again, 'Show raw').click();
  assert.deepEqual(lines(again), ['A pleural effusion.', 'Stable position of the tube.']);
  assert.equal(q(again, '.report-extras').hasAttribute('hidden'), false);
  assert.deepEqual([notesOf(again), termsOf(again)], [[EARLIER_NOTE], ['pleural effusion: fluid between the lung and the chest wall']]);
});

// ---- labels --------------------------------------------------------------------------------------------------------

test('while the label stage has not ended the chips are a "labelling…" placeholder', () => {
  const labels = renderLabels(labelling(), { labelNames: LABEL_NAMES });
  assert.equal(labels.textContent, 'labelling…');
  assert.equal(qa(labels, '.chip').length, 0);
  assert.equal(renderLabels(viewOf([START, stageStart('generate', 3)]), { labelNames: LABEL_NAMES }).textContent, 'labelling…');   // not reached yet
});

test('a skipped label stage says "labels unavailable" with its reason', () => {
  for (const reason of ['labeler_unavailable', 'some_other_reason']) {
    const view = viewOf([START, skipped('label', reason), stopOf('done')]);
    const labels = renderLabels(view, { labelNames: LABEL_NAMES });
    assert.equal(labels.textContent, `labels unavailable (${reason})`, reason);
    assert.equal(qa(labels, '.chip').length, 0);
  }
  assert.equal(renderLabels(tinyView()).textContent, 'labels unavailable (labeler_unavailable)');   // the recorded turn
  const stopped = viewOf([START, stageStart('generate', 3), stopOf('aborted')]);
  assert.equal(renderLabels(stopped).textContent, 'labels unavailable (stopped)');   // never "labelling…" on a finished turn
});

test('the label states follow the stage (P5-E): labelling while it runs, then labels or why there are none, never stuck', () => {
  const live = [START, stageStart('preprocess', 0), stageEnd('preprocess', 1, {}), stageStart('generate', 3), snapshot('Findings: so far'),
                stageEnd('generate', 9, GENERATE)];
  const text = (...steps) => renderLabels(viewOf([...live, ...steps]), { labelNames: LABEL_NAMES }).textContent;
  assert.equal(text(), 'labelling…');                                  // the stage is next
  assert.equal(text(stageStart('label', 4)), 'labelling…');            // the stage runs
  const done = renderLabels(viewOf([...live, stageStart('label', 4), stageEnd('label', 3, { chexbert_14: labelled(['Edema']) })]),
                            { labelNames: LABEL_NAMES });
  assert.equal(qa(done, '.chip').length, 14);                          // then the labels, and no placeholder beside them
  assert.equal(q(done, '.note'), null);
  const reasons = { label_off: 'labels off', labeler_unavailable: 'labels unavailable (labeler_unavailable)' };
  for (const [reason, said] of Object.entries(reasons)) {
    assert.equal(text(skipped('label', reason)), said, reason);                          // the turn still runs: the stage has ended
    assert.equal(text(skipped('label', reason), skipped('score', 'no_reference'), stopOf('done')), said, reason);   // and settled
    assert.equal(text(stageStart('label', 4), skipped('label', reason), stopOf('done')), said, reason);           // started, then skipped
  }
});

test('while the gallery is still being labelled the chips come with an "agreement pending" note (P5-E fix 1), in private mode only (P6-C ruling)', () => {
  const live = [START, stageStart('generate', 3), snapshot('Findings: so far'), stageEnd('generate', 9, GENERATE), stageStart('label', 4)];
  const detail = { chexbert_14: labelled(['Edema']), positives: ['Edema'], neighbor_agreement: [], neighbor_agreement_pending: true };
  const labels = renderLabels(viewOf([...live, stageEnd('label', 3, detail), stopOf('done')]), { labelNames: LABEL_NAMES });
  assert.equal(qa(labels, '.chip').length, 14);                         // the report is labelled all the same
  assert.deepEqual(texts(qa(labels, '.note')), ['agreement pending: the gallery is still being labelled']);
  assert.equal(labels.children.at(-1).textContent, 'agreement pending: the gallery is still being labelled');   // after the chips
  // The agreement with the similar X-rays is private (R1): public mode never shows it, so it does not say it is pending either.
  const pub = renderLabels(viewOf([step('message_start', { ...START.data, mode: 'public' }), ...live.slice(1), stageEnd('label', 3, detail),
                                   stopOf('done')]), { labelNames: LABEL_NAMES });
  assert.equal(qa(pub, '.chip').length, 14);
  assert.equal(q(pub, '.note'), null);
  const plain = renderLabels(viewOf([...live, stageEnd('label', 3, { ...detail, neighbor_agreement_pending: undefined }), stopOf('done')]),
                             { labelNames: LABEL_NAMES });
  assert.equal(q(plain, '.note'), null);                                   // no marker, no note
  const odd = renderLabels(viewOf([...live, stageEnd('label', 3, { ...detail, neighbor_agreement_pending: 'yes' }), stopOf('done')]),
                           { labelNames: LABEL_NAMES });
  assert.equal(q(odd, '.note'), null);                                     // only a true marker says so
  const scored = renderLabels(viewOf([...live, stageEnd('label', 3, detail), stageStart('score', 5), stageEnd('score', 1, SCORE), stopOf('done')]),
                              { labelNames: LABEL_NAMES });
  assert.deepEqual([...scored.children].map((c) => c.getAttribute('class')), ['label-chips', 'note', 'score-block']);   // chips, note, scores
});

test('the chips are the 14 names in ctx.labelNames order: positives filled, negatives outlined', () => {
  const view = finished({ labels: labelled(['Cardiomegaly', 'Support Devices']) });
  const labels = renderLabels(view, { labelNames: LABEL_NAMES });
  const chips = qa(labels, '.chip');
  assert.deepEqual(chips.map(visibleText), LABEL_NAMES);
  assert.deepEqual(chips.map((c) => c.getAttribute('data-label')), LABEL_NAMES);
  const positive = chips.filter((c) => c.classList.contains('positive'));
  assert.deepEqual(positive.map(visibleText), ['Cardiomegaly', 'Support Devices']);
  assert.ok(chips.filter((c) => !positive.includes(c)).every((c) => c.classList.contains('negative')));
  assert.deepEqual(chips.filter((c) => c.getAttribute('data-value') === '1').length, 2);
  assert.equal(chips[1].textContent, 'Cardiomegaly: positive');   // the state is text in the chip, which a screen reader reads
  assert.equal(chips[0].textContent, 'Enlarged Cardiomediastinum: negative');
  assert.equal(qa(labels, '.chip').every((c) => c.classList.contains('label')), true);
  assert.equal(q(labels, '.note'), null);
  assert.equal(q(labels, '.score'), null);   // no score without a reference
});

test('without ctx.labelNames the chips follow the order of the labels in the view', () => {
  const view = finished({ labels: { Edema: 1, 'No Finding': 0, Cardiomegaly: 0 } });
  assert.deepEqual(qa(renderLabels(view), '.chip').map(visibleText), ['Edema', 'No Finding', 'Cardiomegaly']);
  assert.deepEqual(qa(renderLabels(view, { labelNames: [] }), '.chip').map(visibleText), ['Edema', 'No Finding', 'Cardiomegaly']);
});

test('a name in the list that the labels lack reads unknown, and a label outside the list is not dropped', () => {
  const view = finished({ labels: { Edema: 1, Stranger: 1 } });
  const chips = qa(renderLabels(view, { labelNames: ['Edema', 'Cardiomegaly'] }), '.chip');
  assert.deepEqual(chips.map((c) => [visibleText(c), c.classList.contains('unknown'), c.classList.contains('positive')]),
                   [['Edema', false, true], ['Cardiomegaly', true, false], ['Stranger', false, true]]);
  assert.equal(chips[1].textContent, 'Cardiomegaly: not reported');
});

test('with a score each chip also shows whether it agrees with the reference, and the score numbers sit beside the chips', () => {
  const reference = labelled(['Cardiomegaly', 'Edema']);
  const view = finished({
    labels: labelled(['Cardiomegaly', 'Support Devices']),
    score: { ...SCORE, exact_match_14: false, reference_chexbert_14: reference },
  });
  const labels = renderLabels(view, { labelNames: LABEL_NAMES });
  const byName = Object.fromEntries(qa(labels, '.chip').map((c) => [c.getAttribute('data-label'), c]));
  assert.equal(byName.Cardiomegaly.getAttribute('data-agree'), 'true');           // both positive
  assert.equal(byName.Edema.getAttribute('data-agree'), 'false');                 // the reference has it, the report not
  assert.equal(byName['Support Devices'].getAttribute('data-agree'), 'false');    // the report has it, the reference not
  assert.equal(byName.Fracture.getAttribute('data-agree'), 'true');               // both negative
  assert.match(byName.Edema.textContent, /Edema: negative, differs from reference/);
  assert.match(byName.Cardiomegaly.textContent, /Cardiomegaly: positive, matches reference/);
  assert.equal(q(byName.Edema, '.mark').textContent, '✗');
  assert.equal(q(byName.Fracture, '.mark').textContent, '✓');
  assert.equal(q(byName.Edema, '.mark').getAttribute('aria-hidden'), 'true');   // the glyph is not read out twice
  assert.equal(qa(labels, '.chip[data-agree]').length, 14);

  const score = q(labels, '.score');
  assert.deepEqual(qa(score, 'div').map((d) => [q(d, 'dt').textContent, q(d, 'dd').textContent]), [
    ['ROUGE-L', '0.190'], ['BLEU-1', '0.300'], ['BLEU-4', '0.100'], ['CheXbert-14 micro F1', '0.474'], ['CheXbert-14 exact match', 'no'],
  ]);   // the F1 says which average it is: micro, over the 14 labels of this one report
  assert.match(labels.textContent, /vs your reference · this report only · ✓ same as the reference, ✗ different/);   // and the marks are explained
  assert.equal(labels.querySelector('.label-chips').parentNode, score.parentNode.parentNode);   // chips and numbers share the section
});

test('a chip says its state in text inside the chip, not in an aria-label on the list item', () => {
  const view = finished({
    labels: labelled(['Cardiomegaly', 'Support Devices']),
    score: { ...SCORE, reference_chexbert_14: labelled(['Cardiomegaly', 'Edema']) },
  });
  const chips = qa(renderLabels(view, { labelNames: [...LABEL_NAMES, 'Stranger'] }), '.chip');
  const byName = Object.fromEntries(chips.map((c) => [c.getAttribute('data-label'), c]));
  assert.equal(chips.some((c) => c.hasAttribute('aria-label')), false);   // browse mode reads the text of a list item, not its label
  const said = (name) => q(byName[name], '.visually-hidden').textContent;
  assert.equal(said('Cardiomegaly'), ': positive, matches reference');
  assert.equal(said('Edema'), ': negative, differs from reference');
  assert.equal(said('Support Devices'), ': positive, differs from reference');
  assert.equal(said('Fracture'), ': negative, matches reference');
  assert.equal(said('Stranger'), ': not reported');
  assert.equal(byName.Cardiomegaly.textContent, 'Cardiomegaly: positive, matches reference✓');   // read out, then the glyph
  assert.equal(visibleText(byName.Cardiomegaly), 'Cardiomegaly✓');                                  // shown: the name and the glyph
  assert.equal(q(byName.Cardiomegaly, '.mark').getAttribute('aria-hidden'), 'true');
  const plain = Object.fromEntries(qa(renderLabels(finished({ labels: labelled(['Edema']) }), { labelNames: LABEL_NAMES }), '.chip')
    .map((c) => [c.getAttribute('data-label'), c]));
  assert.equal(q(plain.Edema, '.visually-hidden').textContent, ': positive');   // with no reference there is no agreement to say
  assert.equal(q(plain.Fracture, '.visually-hidden').textContent, ': negative');
});

test('with labels off the card says "labels off" from the start instead of "labelling…"', () => {
  const off = (...steps) => viewOf([step('message_start', { ...START.data, options: { ...START.data.options, label: false } }), ...steps]);
  assert.equal(renderLabels(off(stageStart('preprocess', 0))).textContent, 'labels off');
  assert.equal(renderLabels(off(stageStart('generate', 3), snapshot('Findings: x'), stageEnd('generate', 9, GENERATE), stageStart('label', 4))).textContent,
               'labels off');
  assert.equal(renderLabels(off(skipped('label', 'label_off'), stopOf('done'))).textContent, 'labels off');   // settled: still "off", never "unavailable (label_off)"
  assert.equal(renderLabels(labelling()).textContent, 'labelling…');   // labels on: as before
  assert.equal(renderLabels({ ...labelling(), options: null }).textContent, 'labelling…');
  assert.equal(renderLabels({ ...labelling(), options: { label: true } }).textContent, 'labelling…');
});

test('a label stage that ends skipped as label_off keeps saying "labels off", with or without the options in the view', () => {
  const off = (...steps) => viewOf([step('message_start', { ...START.data, options: { ...START.data.options, label: false } }), ...steps]);
  const settled = [stageStart('preprocess', 0), stageEnd('preprocess', 1, {}), stageStart('generate', 3), snapshot('Findings: x'),
                   stageEnd('generate', 9, GENERATE), skipped('label', 'label_off'), skipped('score', 'no_reference')];
  const text = (view) => renderLabels(view, { labelNames: LABEL_NAMES }).textContent;
  assert.equal(text(off(...settled)), 'labels off');   // still running: the label stage has ended (skipped), the turn has not
  assert.equal(text(off(...settled, stopOf('done'))), 'labels off');
  assert.equal(text(viewOf([START, skipped('label', 'label_off'), stopOf('done')])), 'labels off');   // options say label on, the stage says off: the stage wins
  assert.equal(text({ ...viewOf([START, skipped('label', 'label_off'), stopOf('done')]), options: null }), 'labels off');   // a view with no options
  assert.equal(text(off(stageStart('generate', 3), stopOf('aborted'))), 'labels off');   // stopped before the stage: it was off all along
  assert.equal(text(off(stageStart('generate', 3), step('error', { type: 'error', error: { type: 'model_error', message: 'x' } }), stopOf('error'))), 'labels off');
  assert.equal(text(viewOf([START, stageStart('generate', 3), stopOf('aborted')])), 'labels unavailable (stopped)');   // labels on: a stop is still a stop
  const card = renderAssistantCard(off(...settled, stopOf('done')), { labelNames: LABEL_NAMES });
  assert.equal(q(card, '.labels').textContent, 'labels off');
  assert.doesNotMatch(q(card, '.labels').textContent, /label_off|unavailable/);
  assert.equal(q(stageItem(card, 'label'), '.visually-hidden').textContent, ': label_off');   // the timeline item still says why it was skipped
});

test('the timeline, the label chips and the option chips are lists to a screen reader even where the style drops the bullets', () => {
  assert.equal(renderTimeline(initialView('m1')).getAttribute('role'), 'list');   // Safari removes list semantics from list-style: none
  const chips = q(renderLabels(finished({ labels: labelled(['Edema']) }), { labelNames: LABEL_NAMES }), 'ul.label-chips');
  assert.equal(chips.getAttribute('role'), 'list');
  const user = renderUserTurn({ text: 'a', options: tinyView().options }, {});
  assert.equal(q(user, 'ul.options').getAttribute('role'), 'list');
  assert.ok(qa(user, 'ul.options > li').length >= 1);
  assert.equal(qa(renderAssistantCard(finished({ labels: labelled([]) }), { labelNames: LABEL_NAMES }), '[role="list"]').length, 2);   // the timeline and the chips
});

test('a score without per-label reference labels shows its numbers and no agree marks; no score, no numbers', () => {
  const labels = renderLabels(finished({ labels: labelled(['Edema']), score: SCORE }), { labelNames: LABEL_NAMES });
  assert.equal(qa(labels, '.chip[data-agree]').length, 0);
  assert.equal(qa(labels, '.mark').length, 0);
  assert.equal(qa(q(labels, '.score'), 'dd').length, 4);
  assert.doesNotMatch(labels.textContent, /✓|✗/);   // no marks, so no legend for them
  assert.equal(q(renderLabels(finished({ labels: labelled([]) }), { labelNames: LABEL_NAMES }), '.score'), null);
  const noLabels = renderLabels(finished({ score: SCORE }));   // labels skipped, scores there: the numbers still show
  assert.match(noLabels.textContent, /labels unavailable \(labeler_unavailable\)/);
  assert.equal(qa(q(noLabels, '.score'), 'dd').length, 4);
  const partial = renderLabels(finished({ labels: labelled([]), score: { rouge_l: 0.5, bleu_1: NaN, bleu_4: null, reference_source: 'test_split' } }));
  assert.deepEqual(qa(partial, '.score dt').map((n) => n.textContent), ['ROUGE-L']);   // only the numbers that are numbers
  assert.match(partial.textContent, /vs test-split reference · this report only/);
  assert.doesNotMatch(serialize(partial), /NaN|undefined|null/);
});

test('renderLabels renders nothing for a turn that never reached the label stage and is settled', () => {
  const question = viewOf([START, step('warning', { code: 'not_a_command', message: 'Not a question answerer.' }), stopOf('done')]);
  const labels = renderLabels(question);
  assert.ok(labels.hasAttribute('hidden'));
  assert.equal(labels.textContent, '');
});

// ---- provenance ----------------------------------------------------------------------------------------------------

test('the provenance footer reads model, checkpoint, operators, prefix, beam, tokens, time and device', () => {
  const view = finished();
  assert.equal(provenanceText(view), `hybrid_150m_m3_rrg · ckpt a1b2c3d4 · scan exact/tfla exact · prefix_k 32 · beam 3 · 87 tok · 612 ms · cpu`);
  const footer = renderProvenance(view, {});
  assert.equal(footer.localName, 'footer');
  assert.equal(q(footer, 'p').textContent, provenanceText(view));
  assert.equal(q(footer, '.drift').textContent, 'CPU decode can drift from the published GPU run');
});

test('the footer for the recorded tiny turn drops what the card does not have: no checkpoint', () => {
  const view = tinyView();
  assert.equal(provenanceText(view), 'tiny · scan legacy/tfla exact · prefix_k 4 · beam 3 · 16 tok · 19 ms · cpu');
  assert.equal(q(renderProvenance(view), '.drift').textContent, 'tiny random-init model');
});

test('a greedy run says greedy, not a beam size', () => {
  const view = finished();
  const greedy = { ...view, options: { ...view.options, decode: 'greedy' }, stages: { ...view.stages, generate: { ...view.stages.generate, detail: { ...GENERATE, decode: 'greedy' } } } };
  assert.match(provenanceText(greedy), /· greedy ·/);
  assert.doesNotMatch(provenanceText(greedy), /beam/);
  const asked = { ...view, options: { ...view.options, decode: 'greedy' },   // what ran is not readable: what was asked
                  stages: { ...view.stages, generate: { ...view.stages.generate, detail: { decode: {}, beam_size: 3 } } } };
  assert.match(provenanceText(asked), /· greedy ·/);
});

test('the provenance link calls ctx.showModels; without it there is no link', () => {
  const view = finished();
  let opened = 0;
  const footer = renderProvenance(view, { showModels: () => { opened += 1; } });
  const link = button(footer, 'model details');
  assert.ok(link);
  assert.equal(link.getAttribute('type'), 'button');   // a button, not an anchor: a hash link would change the route
  link.click();
  assert.equal(opened, 1);
  assert.equal(qa(renderProvenance(view, {}), 'button, a').length, 0);
  assert.equal(qa(renderProvenance(view), 'button, a').length, 0);
});

test('the footer is hidden until the turn has started, and shows what a partial card has', () => {
  assert.ok(renderProvenance(initialView('m1'), { showModels() {} }).hasAttribute('hidden'));
  const partial = { ...initialView('m1'), provenance: { name: 'tiny' } };
  assert.equal(provenanceText(partial), 'tiny');
  assert.equal(renderProvenance(partial).hasAttribute('hidden'), false);
  const asked = { ...partial, options: { decode: 'beam', beam_size: 5 } };
  assert.equal(provenanceText(asked), 'tiny · beam 5');   // before generate has run: what was asked
});

// ---- markup in the data stays text (ruling 3) ----------------------------------------------------------------------

const PAYLOAD = '<img src=x onerror=alert(1)>';
// The attributes that would run script: any on* attribute anywhere under the node. (A listener added with
// addEventListener is not an attribute, and the page's own handlers are all of that kind.)
function handlerAttributes(node) {
  const names = [];
  const walk = (n) => {
    names.push(...[...n.attrs.keys()].filter((name) => name.startsWith('on')));
    n.children.forEach(walk);
  };
  walk(node);
  return names;
}

test('a report containing markup renders as literal text: no element, no handler', () => {
  const view = finished({ report: `Findings: ${PAYLOAD} Impression: <script>alert(2)</script>` });
  const card = renderAssistantCard(view, { copy() {} });
  assert.equal(qa(card, 'img').length, 0);
  assert.equal(qa(card, 'script').length, 0);
  assert.deepEqual(handlerAttributes(card), []);
  assert.ok(card.textContent.includes(PAYLOAD));
  assert.ok(card.textContent.includes('<script>alert(2)</script>'));
  const html = serialize(card);
  assert.ok(html.includes('&lt;img src=x onerror=alert(1)&gt;'), 'escaped text, as a browser would write it');
  assert.equal(html.includes('<img'), false);
  button(card, 'Show raw').click();   // and the raw view
  assert.equal(q(card, 'pre').textContent.includes(PAYLOAD), true);
  assert.equal(qa(card, 'img').length, 0);
});

test('markup in a user note, a filename, a label name, a notice, an error and a stage detail stays text too', () => {
  const user = renderUserTurn({ text: PAYLOAD, image: { filename: `${PAYLOAD}.png` }, options: null }, {});
  assert.equal(qa(user, 'img').length, 0);
  assert.deepEqual(handlerAttributes(user), []);
  assert.ok(user.textContent.includes(PAYLOAD));

  const base = finished({ labels: { [PAYLOAD]: 1 } });
  const view = { ...base, notices: [{ code: 'x', message: PAYLOAD }], error: { type: 'model_error', message: PAYLOAD },
                 stages: { ...base.stages, encode: { state: 'done', ms: 1, detail: { [PAYLOAD]: PAYLOAD, nested: [{ [PAYLOAD]: PAYLOAD }] } } } };
  const card = renderAssistantCard(view, { labelNames: [PAYLOAD] });
  assert.equal(qa(card, 'img').length, 0);   // an <img inside an attribute value (the chip's aria-label) is only a string
  assert.deepEqual(handlerAttributes(card), []);
  // The chip and the label stage's own detail, the notice, the error, and the encode detail's key, value, column heading
  // and cell: eight places, each showing it as text.
  assert.equal(card.textContent.split(PAYLOAD).length - 1, 8);
});

// ---- a public-mode or partial view: missing fields are not errors (ruling 5) ----------------------------------------

function everyBuilder(view, ctx) {
  const built = {
    card: renderAssistantCard(view, ctx), timeline: renderTimeline(view, ctx), report: renderReport(view, ctx),
    labels: renderLabels(view, ctx), provenance: renderProvenance(view, ctx), notes: renderNotes(view, ctx),
  };
  for (const name of SECTION_BUILDERS) if (typeof render[name] === 'function') built[name] = render[name](view, ctx);   // P6-B..D
  return built;
}
const SECTION_BUILDERS = ['renderImages', 'renderNeighbors', 'renderMatches', 'renderPublished'];
// What a missing field must never print, in the text of a card or in any attribute of it.
function assertClean(node, name) {
  assert.doesNotMatch(node.textContent, /undefined|NaN|\bnull\b|\[object|\bfalse\b/, `${name}: text`);
  const walk = (n) => {
    for (const [attr, value] of n.attrs) assert.doesNotMatch(value, /undefined|NaN|\[object|^null$/, `${name}: ${attr}`);
    n.children.forEach(walk);
  };
  walk(node);
}

test('a public-mode turn, as app/redact.py sends it, renders without errors or stray words and ends with every stage settled', () => {
  const pub = viewOf([
    step('message_start', { message_id: 'm_pub', user_message_id: 'u', session_id: 's', mode: 'public',
                            model: { name: 'hybrid_150m_m3_rrg', checkpoint: 'last.ckpt' }, options: { decode: 'beam', beam_size: 3 },
                            image: { sha256: SHA, filename: 'x.png', source: 'upload', urls: {} } }),
    stageStart('preprocess', 0), stageEnd('preprocess', 1, { format: 'PNG', input_px: [320, 320] }),
    stageStart('encode', 1), stageEnd('encode', 6, { device: 'cpu' }),
    stageStart('retrieve', 2),
    stageEnd('retrieve', 8, { image_neighbors: [{ rank: 1, similarity: 0.9 }], gallery: { images: 20 } }),   // rank and similarity only
    stageStart('generate', 3), snapshot('Findings: ok'), stageEnd('generate', 9, { decode: 'beam', beam_size: 3 }),
    stageStart('label', 4), stageEnd('label', 4, { chexbert_14: { Cardiomegaly: 1 } }),   // no neighbor_agreement
    // No score event of any kind: public mode has no reference, and redact.py drops stage_start and stage_end of score.
    stopOf('done', { report: 'Findings: ok', display_report: 'Findings: ok' }),
  ]);
  assert.equal('score' in pub.stages, false);   // the public log never mentions it
  assert.deepEqual([pub.score, pub.agreement, pub.trueRank, pub.matches], [null, null, null, []]);
  const built = everyBuilder(pub, { labelNames: LABEL_NAMES });
  for (const [name, node] of Object.entries(built)) assertClean(node, name);
  assert.deepEqual(qa(built.labels, '.chip').map(visibleText), LABEL_NAMES);   // 14 chips, the 13 the labels lack read unknown
  assert.equal(qa(built.labels, '.chip.positive').length, 1);
  assert.equal(qa(built.labels, '.chip.unknown').length, 13);

  // The finished turn is settled all the way down: the stage the public log never sent is skipped, not pending forever.
  assert.deepEqual(qa(built.timeline, 'li').map((li) => li.getAttribute('data-state')), ['done', 'done', 'done', 'done', 'done', 'skipped']);
  const score = stageItem(built.timeline, 'score');
  assert.deepEqual([visibleText(score), score.textContent], ['score · skipped', 'score · skipped: not_run']);
  assert.equal(qa(built.card, '[data-state="pending"]').length, 0);
  assert.equal(statusText(pub), 'Report ready');
});

test('a view that lacks fields outright still renders: every builder takes {} and a null-filled view', () => {
  const bare = { id: null, status: 'done', stages: { generate: { state: 'done' }, encode: null }, provenance: { name: 'x' },
                 report: undefined, displayReport: undefined, labels: undefined, score: undefined, agreement: undefined,
                 neighbors: undefined, matches: undefined, trueRank: undefined, notices: undefined, options: undefined,
                 image: undefined, totalMs: undefined };
  for (const view of [{}, bare, initialView(null), { ...initialView('m'), provenance: { name: null, device: null, prefix_k: null } }]) {
    for (const ctx of [undefined, null, {}, { labelNames: null, copy: null, ui: null }, { turn: NaN }, { turn: -1 }, { turn: '3' }, { turn: 2.5 }]) {
      const built = everyBuilder(view, ctx);
      for (const [name, node] of Object.entries(built)) assertClean(node, name);
    }
  }
  assert.equal(statusText({}), 'Queued');
  assert.equal(renderUserTurn(undefined).localName, 'article');
  assert.equal(renderUserTurn(null, null).localName, 'article');
  assert.equal(renderUserTurn({}).hasAttribute('hidden'), true);
});

test('a value that is neither text nor a number prints nothing: an object, an array, NaN, an object with no prototype', () => {
  const base = finished({ labels: labelled(['Edema']) });
  const bare = Object.create(null);   // String() and Number() throw on it
  const confused = {
    ...base, report: { a: 1 }, displayReport: ['x'], error: { message: { deep: 1 } }, totalMs: NaN,
    notices: [{ code: {}, message: [] }, { message: NaN }, null, 5, { message: bare }],
    options: { beam_size: {}, max_new_tokens: [], k_images: {}, k_reports: NaN, test_row: 1.5 },
    provenance: { name: {}, checkpoint_sha256: {}, scan_impl: [], prefix_k: NaN, device: bare, drift_note: {} },
    labels: { Edema: bare, Cardiomegaly: {}, 'No Finding': [1] },
    score: { rouge_l: bare, reference_source: {}, exact_match_14: 'yes', reference_chexbert_14: { Edema: bare } },
  };
  const built = everyBuilder(confused, { labelNames: [...LABEL_NAMES, 7, null, {}] });
  for (const [name, node] of Object.entries(built)) assertClean(node, name);
  assert.equal(provenanceText(confused), 'beam 3 · 87 tok · cpu');   // what ran still speaks; the rest is left out
  assert.equal(statusText(confused), 'Error: the turn failed');
  assert.deepEqual(optionChips(confused.options), []);
  assert.equal(qa(built.labels, '.chip.positive').length, 0);   // only a 1 or true is a positive label
  assert.equal(qa(built.notes, '.notice').length, 0);
  assert.ok(built.report.hasAttribute('hidden'));
});

test('a detail table copes with any value: null, empty, numbers, a bare array of nothing', () => {
  const table = detailTable({ a: null, b: undefined, c: [null, undefined], d: {}, e: [[1, 2], [3]], f: -0, g: 1e-7, h: 12345.678 });
  assert.doesNotMatch(serialize(table), /undefined|NaN|\[object/);
  assert.equal(detailTable(null).textContent, '');
  assert.equal(detailTable({}).textContent, '');
});

// ---- statusText for the live region (ruling 7) -----------------------------------------------------------------------

test('statusText names the running stage, then Report ready, Turn stopped or the error', () => {
  assert.equal(statusText(initialView('m1')), 'Queued');
  const midGenerate = viewOf([START, stageStart('preprocess', 0), stageEnd('preprocess', 1, {}), stageStart('encode', 1)]);
  assert.equal(statusText(midGenerate), 'encode running');
  assert.equal(statusText(viewOf([START, stageStart('generate', 3), snapshot('x')])), 'generate running');
  assert.equal(statusText(labelling()), 'label running');
  assert.equal(statusText(viewOf([START, stageStart('preprocess', 0), stageEnd('preprocess', 1, {})])), 'Working');   // between two stages
  assert.equal(statusText(finished()), 'Report ready');
  assert.equal(statusText(tinyView()), 'Report ready');
  const aborted = viewOf([START, stageStart('generate', 3), snapshot('x'), stopOf('aborted')]);
  assert.equal(statusText(aborted), 'Turn stopped');
  const failed = viewOf([START, stageStart('generate', 3), step('error', { type: 'error', error: { type: 'model_error', message: 'Internal error (RuntimeError)' } }), stopOf('error')]);
  assert.equal(statusText(failed), 'Error: Internal error (RuntimeError)');
  const failing = viewOf([START, stageStart('generate', 3), step('error', { type: 'error', error: { type: 'model_error', message: 'Out of memory' } })]);
  assert.equal(failing.status, 'running');
  assert.equal(statusText(failing), 'Error: Out of memory');   // announced as soon as it is known
  assert.equal(statusText({ ...failed, error: null }), 'Error: the turn failed');
  const question = viewOf([START, step('warning', { code: 'not_a_command', message: 'This is a report generator, not a question answerer.' }), stopOf('done')]);
  assert.equal(statusText(question), 'This is a report generator, not a question answerer.');
  assert.equal(statusText(viewOf([START, stopOf('done')])), 'Turn finished');
});

// ---- the whole card -------------------------------------------------------------------------------------------------

test('the card of the recorded turn: timeline, report with its note, labels unavailable, provenance', () => {
  const view = tinyView();
  const card = renderAssistantCard(view, { labelNames: LABEL_NAMES, copy() {}, showModels() {} });
  assert.equal(card.localName, 'article');
  assert.ok(card.classList.contains('card'));
  assert.equal(card.getAttribute('data-message-id'), view.id);
  assert.equal(card.getAttribute('data-status'), 'done');
  assert.equal(card.hasAttribute('aria-live'), false);   // P4-D announces through its own status node
  assert.deepEqual(qa(card, 'ol.timeline > li').map((li) => [li.getAttribute('data-stage'), li.getAttribute('data-state'), li.childNodes[0].textContent]), [
    ['preprocess', 'done', 'preprocess · 2 ms'], ['encode', 'done', 'encode · 0 ms'], ['retrieve', 'skipped', 'retrieve · skipped'],
    ['generate', 'done', 'generate · 14 ms'], ['label', 'skipped', 'label · skipped'], ['score', 'skipped', 'score · skipped'],
  ]);
  assert.equal(lines(card).join(' '), view.report);   // its sentences, one to a line, are the report
  assert.equal(q(card, '.report .note').textContent, 'Report stopped at the token budget mid-sentence');
  assert.equal(q(card, '.labels').textContent, 'labels unavailable (labeler_unavailable)');
  assert.equal(q(card, '.provenance p').textContent.startsWith('tiny · scan legacy/tfla exact · prefix_k 4'), true);
  assert.equal(button(card, 'Copy') !== null && button(card, 'Show raw') !== null && button(card, 'model details') !== null, true);
  assert.deepEqual(['ol.timeline', '.notes', '.test-notice', '.report', '.labels', '.provenance'].map((s) => card.children.indexOf(q(card, s))),
                   [0, 1, 2, 3, 4, 5]);   // the recorded turn is the tiny engine's: its card says so, between the notes and the report (P9-G5b)
});

test('a card is a pure function of its view: the same view gives the same markup, and a re-render replaces the card whole', () => {
  const view = tinyView();
  const ctx = { labelNames: LABEL_NAMES, copy() {}, showModels() {} };
  const first = renderAssistantCard(view, ctx);
  assert.equal(serialize(renderAssistantCard(view, ctx)), serialize(first));
  document.body.replaceChildren(first);
  const second = renderAssistantCard(viewOf([START, stageStart('generate', 3), snapshot('Findings: partly')]), ctx);
  first.replaceWith(second);
  assert.equal(document.body.children.length, 1);
  assert.equal(document.body.children[0], second);
  assert.equal(first.parentNode, null);
});

test('a question turn shows its notice and no timeline of stages that never ran', () => {
  const view = viewOf([START, step('warning', { code: 'not_a_command', message: 'This is a report generator, not a question answerer.' }), stopOf('done')]);
  const card = renderAssistantCard(view, {});
  assert.ok(q(card, 'ol.timeline').hasAttribute('hidden'));
  assert.ok(q(card, '.provenance').hasAttribute('hidden'), 'no model ran, so none is named');
  assert.equal(q(card, '.note.notice').textContent, 'This is a report generator, not a question answerer.');
  assert.equal(q(card, '.note.notice').getAttribute('data-code'), 'not_a_command');
  assert.ok(q(card, '.report').hasAttribute('hidden'));
  const queued = renderAssistantCard(initialView('m1'), {});   // a turn still waiting: all six pending, and shown
  assert.equal(q(queued, 'ol.timeline').hasAttribute('hidden'), false);
  assert.equal(qa(queued, 'li[data-state="pending"]').length, 6);
  const stoppedQueued = renderAssistantCard(viewOf([START, stopOf('aborted')]), {});   // stopped before its first stage
  assert.equal(q(stoppedQueued, 'ol.timeline').hasAttribute('hidden'), false);
  assert.equal(qa(stoppedQueued, 'li[data-state="skipped"]').length, 6);
  assert.equal(q(stoppedQueued, '.note.stopped').textContent, 'Turn stopped');
});

test('a failed turn shows its error message; a stopped one says so', () => {
  const failed = viewOf([START, stageStart('generate', 3), snapshot('Findings: part'),
                         step('error', { type: 'error', error: { type: 'model_error', message: 'Internal error (RuntimeError)' } }), stopOf('error')]);
  const card = renderAssistantCard(failed, {});
  assert.equal(q(card, '.note.error').textContent, 'Error: Internal error (RuntimeError)');
  assert.deepEqual(lines(card), ['part']);   // what had been generated stays
  const failing = viewOf([START, stageStart('generate', 3),
                          step('error', { type: 'error', error: { type: 'model_error', message: 'Out of memory' } })]);
  assert.equal(failing.status, 'running');   // the error is known before message_stop ends the turn
  assert.equal(q(renderAssistantCard(failing, {}), '.note.error').textContent, 'Error: Out of memory');
  const stopped = renderAssistantCard(viewOf([START, stageStart('generate', 3), stopOf('aborted')]), {});
  assert.equal(q(stopped, '.note.stopped').textContent, 'Turn stopped');
  assert.equal(q(stopped, '.note.error'), null);
  assert.equal(q(renderAssistantCard(finished(), {}), '.note.error, .note.stopped'), null);
});

test('an internal error also says what to try, on a line of its own; its error text and the status are unchanged (P4-H A3)', () => {
  const HINT = 'The server hit an internal error. If you just updated the code, restart the server.';
  const failed = viewOf([START, stageStart('generate', 3),
                         step('error', { type: 'error', error: { type: 'model_error', message: 'Internal error (ImportError)' } }), stopOf('error')]);
  const card = renderAssistantCard(failed, {});
  assert.equal(q(card, '.note.error').textContent, 'Error: Internal error (ImportError)');   // the class name only, never exception text
  assert.equal(q(card, '.note.error-hint').textContent, HINT);
  const notes = q(card, '.notes').children;                                                   // right under the error it explains
  assert.equal(notes.indexOf(q(card, '.note.error-hint')), notes.indexOf(q(card, '.note.error')) + 1);
  assert.equal(statusText(failed), 'Error: Internal error (ImportError)');
  assert.equal(q(renderNotes(failed), '.note.error-hint').textContent, HINT);
  // Only the server's own "Internal error (<class>)", which private mode shows. A public server's fixed message, an upload it refused, a
  // restart: none is about code on disk, so none gets the line.
  const others = [{ type: 'model_error', message: 'The model could not finish this turn.' },
                  { type: 'validation_error', message: 'Each side must be at least 64 pixels.' },
                  { type: 'server_restart', message: 'The server restarted while this turn was running.' },
                  { type: 'model_error', message: 'Out of memory' }];
  for (const error of others) {
    const view = viewOf([START, stageStart('generate', 3), step('error', { type: 'error', error }), stopOf('error')]);
    assert.equal(q(renderAssistantCard(view, {}), '.note.error-hint'), null, error.message);
  }
  assert.equal(q(renderAssistantCard(viewOf([START, stopOf('error')]), {}), '.note.error-hint'), null);   // an error with no message
});

test('warnings are shown in arrival order', () => {
  const view = viewOf([START, step('warning', { code: 'reference_ignored_public', message: 'First.' }),
                       step('warning', { code: 'other', message: 'Second.' }), stopOf('done')]);
  assert.deepEqual(texts(qa(renderNotes(view), '.note')), ['First.', 'Second.']);
  assert.equal(renderNotes(finished()).hasAttribute('hidden'), true);
});

test('no builder writes to the view, the labels list or the ctx object it is given; only the ui store it is handed changes', () => {
  const view = deepFreeze(finished({ labels: labelled(['Edema']), score: { ...SCORE, reference_chexbert_14: labelled([]) }, truncated: true }));
  const names = deepFreeze([...LABEL_NAMES]);
  const ui = new Map();   // the one thing a builder is meant to write to
  const ctx = Object.freeze({ labelNames: names, copy() {}, showModels() {}, openViewer() {}, turn: 4, ui });
  const card = renderAssistantCard(view, ctx);   // a write to a frozen object throws in a module
  assert.ok(qa(card, 'li[data-stage] > button').length >= 3);   // there are controls to work
  for (const toggle of qa(card, 'li[data-stage] > button')) { toggle.click(); toggle.click(); }
  button(card, 'Show raw').click();
  const tiny = deepFreeze(tinyView());
  everyBuilder(tiny, ctx);
  everyBuilder(deepFreeze(viewOf([START, stageStart('generate', 3), stopOf('aborted')])), ctx);
  renderUserTurn(deepFreeze({ text: 'beam 5', image: { url: 'blob:x/1', filename: 'a.png' }, options: tinyView().options }), ctx);
  assert.deepEqual(names, LABEL_NAMES);
  assert.deepEqual([...ui.keys()].sort(), ['m_test', tiny.id].sort());   // a record per message, and nothing else was written
});

// ---- the user turn ---------------------------------------------------------------------------------------------------

test('the options read as small chips: beam, tokens, cached, k images/reports', () => {
  assert.deepEqual(optionChips({ decode: 'beam', beam_size: 3, max_new_tokens: 100, cached_decode: true, k_images: 4, k_reports: 3 }),
                   ['beam 3', '100 tok', 'cached', 'k 4/3']);   // the brief's example
  assert.deepEqual(optionChips(tinyView().options), ['beam 3', '16 tok', 'cached', 'k 4/3', 'raw text', 'full budget']);   // this recorded turn ran with Display repair and the stop switch off
  assert.deepEqual(optionChips({ decode: 'greedy', beam_size: 3, max_new_tokens: 150, cached_decode: false, k_images: 0, k_reports: 10,
                                 label: false, display_repair: true, compile: true, reference: 'a note', test_row: 17 }),
                   ['greedy', '150 tok', 'uncached', 'k 0/10', 'labels off', 'compiled', 'reference', 'test row 17']);   // repair on is the default: no chip
  assert.deepEqual(optionChips({ decode: 'beam', beam_size: 5, reference: '', test_row: null, label: true, display_repair: false, compile: false }),
                   ['beam 5', 'raw text']);
  assert.deepEqual(optionChips(null), []);
  assert.deepEqual(optionChips({}), []);
});

test('Display repair has a chip only when it is off, and only a real false is a choice', () => {
  const base = { decode: 'beam', beam_size: 3 };
  assert.deepEqual(optionChips({ ...base, display_repair: true }), ['beam 3']);                 // the default says nothing
  assert.deepEqual(optionChips({ ...base, display_repair: false }), ['beam 3', 'raw text']);   // the report as the decoder wrote it
  for (const unsaid of [undefined, null, 0, '', 'no', 'false', [], {}]) {
    assert.deepEqual(optionChips({ ...base, display_repair: unsaid }), ['beam 3'], String(unsaid));   // a turn that does not say: no claim
  }
  assert.equal(optionChips({ ...base, display_repair: false }).includes('repair on'), false);   // the old chip is gone
});

test('Stop when the report starts repeating has a chip only when it is off, and only a real false is a choice: "full budget"', () => {
  const base = { decode: 'beam', beam_size: 3 };
  assert.deepEqual(optionChips({ ...base, stop_on_repeat: true }), ['beam 3']);   // the page's default says nothing
  assert.deepEqual(optionChips({ ...base, stop_on_repeat: false }), ['beam 3', 'full budget']);   // the published protocol: the whole budget is decoded
  for (const unsaid of [undefined, null, 0, '', 'false']) {
    assert.deepEqual(optionChips({ ...base, stop_on_repeat: unsaid }), ['beam 3'], String(unsaid));   // an older turn does not say: no claim
  }
  assert.deepEqual(optionChips({ ...base, display_repair: false, stop_on_repeat: false }), ['beam 3', 'raw text', 'full budget']);
  assert.deepEqual(optionChips({ ...base, stop_on_repeat: false, compile: true, reference: 'a note', test_row: 17 }),
                   ['beam 3', 'full budget', 'compiled', 'reference', 'test row 17']);
});

// ---- P9-G5b: a model trained to end its reports (eos_trained) ---------------------------------------------------------------------

test('a report that ended at its own EOS says so where the chip would say "full budget": the stop reason, "ended the report"', () => {
  const off = { decode: 'beam', beam_size: 3, stop_on_repeat: false };
  assert.deepEqual(optionChips(off, { stopped: 'eos' }), ['beam 3', 'ended the report']);   // not "full budget": the report ended where the model meant it to
  assert.deepEqual(optionChips(off, { stopped: 'eos', eosTrained: true }), ['beam 3', 'ended the report']);
  assert.deepEqual(optionChips(off, { stopped: 'budget', eosTrained: true }), ['beam 3', 'full budget']);   // an EOS model that did reach the budget: it did decode it all
  assert.deepEqual(optionChips(off, { stopped: 'budget' }), ['beam 3', 'full budget']);
  assert.deepEqual(optionChips(off, { eosTrained: true }), ['beam 3']);   // an EOS model whose turn has not ended (or the next turn's settings): no claim
  assert.deepEqual(optionChips(off, { stopped: '', eosTrained: true }), ['beam 3']);
  for (const published of [undefined, null, {}, { eosTrained: false }, { stopped: '', eosTrained: false }, 'eos', 7]) {
    assert.deepEqual(optionChips(off, published), ['beam 3', 'full budget'], JSON.stringify(published));   // a published model, as ever
  }
  assert.deepEqual(optionChips(off), ['beam 3', 'full budget']);   // and a call with no second argument is the call it was
  const on = { decode: 'beam', beam_size: 3, stop_on_repeat: true };
  for (const outcome of [{ stopped: 'eos' }, { stopped: 'budget' }, { stopped: 'repeat' }, { eosTrained: true }, { stopped: 'eos', eosTrained: true }]) {
    assert.deepEqual(optionChips(on, outcome), ['beam 3'], JSON.stringify(outcome));   // the switch on says nothing, whatever ended the report
  }
  assert.deepEqual(optionChips({ ...off, stop_on_repeat: undefined }, { stopped: 'eos' }), ['beam 3']);   // a turn that does not say: no claim
  assert.deepEqual(optionChips({ ...off, display_repair: false, compile: true, test_row: 4 }, { stopped: 'eos' }),
                   ['beam 3', 'raw text', 'ended the report', 'compiled', 'test row 4']);   // in the place of the chip it replaces
  assert.deepEqual(optionChips({ ...off, stop_on_repeat: false }, { stopped: 'repeat' }), ['beam 3']);   // a repeat stop is not "full budget" either
});

test('the user turn of a report that ended at EOS reads "ended the report" and never "full budget"; the card of that turn adds no budget note', () => {
  const options = { ...tinyView().options, stop_on_repeat: false };
  const chips = (outcome) => texts(qa(renderUserTurn({ text: 'x', options, outcome }, {}), '.options .chip'));
  assert.deepEqual(chips({ stopped: 'eos', eosTrained: true }), ['beam 3', '16 tok', 'cached', 'k 4/3', 'raw text', 'ended the report']);
  assert.deepEqual(chips({ stopped: 'budget', eosTrained: true }), ['beam 3', '16 tok', 'cached', 'k 4/3', 'raw text', 'full budget']);
  assert.deepEqual(chips({ eosTrained: true }), ['beam 3', '16 tok', 'cached', 'k 4/3', 'raw text']);   // still running: nothing claimed
  assert.deepEqual(chips(undefined), ['beam 3', '16 tok', 'cached', 'k 4/3', 'raw text', 'full budget']);   // a message with no outcome: as before
  const card = renderAssistantCard(finished({ options: { stop_on_repeat: false }, generate: { stopped: 'eos', tokens: 31 } }), {});
  assert.deepEqual(texts(qa(card, '.report .note')), []);   // no budget note: the report ends where the model meant it to
});

test('the user turn replaces nothing it was given: its message, options and outcome stay as they were', () => {
  const message = deepFreeze({ text: 'x', options: { ...tinyView().options, stop_on_repeat: false }, outcome: { stopped: 'eos', eosTrained: true } });
  assert.doesNotThrow(() => renderUserTurn(message, {}));   // a write to a frozen object throws in a module
});

// ---- P9-G5b: the tiny engine's random weights are announced on every card -----------------------------------------------------------

const TEST_NOTICE = 'Test model with random weights: this text is not a report. Real reports come from the cluster server.';

test('every card of the tiny model has the notice, above its report: "Test model with random weights: this text is not a report..."', () => {
  assert.equal(tinyView().provenance.name, 'tiny');   // the recorded turn is the tiny engine's own
  const card = renderAssistantCard(tinyView(), { copy() {}, showModels() {} });
  const notice = q(card, '.test-notice');
  assert.ok(notice, 'the card has the notice');
  assert.equal(notice.textContent, TEST_NOTICE);
  assert.equal(notice.hidden, false);
  assert.equal(notice.localName, 'p');
  assert.equal(notice.getAttribute('role'), 'note');   // a statement about the card: not an alert, which would be read out at every frame
  assert.equal(render.TEST_NOTICE, TEST_NOTICE);
  const kids = card.children;
  assert.ok(kids.indexOf(notice) < kids.indexOf(q(card, '.report')), 'above the report');
  assert.equal(kids.filter((n) => n.classList.contains('test-notice')).length, 1);   // once, not once per section
  assert.equal(texts(qa(card, '.notes .note')).includes(TEST_NOTICE), false);   // it is not one of the notes a test of the card counts
  assert.equal(texts(qa(card, '.report .note')).includes(TEST_NOTICE), false);
});

test('the notice is on the tiny card in every state of the turn, and only a turn that ran the model has one', () => {
  const tiny = { ...START.data.model, name: 'tiny' };
  const start = step('message_start', { ...START.data, model: tiny });
  const running = viewOf([start, stageStart('preprocess', 0)]);
  const done = viewOf([start, stageStart('generate', 3), snapshot('Findings: x.'), stopOf('done', { report: 'Findings: x.', display_report: 'Findings: x.' })]);
  const stopped = viewOf([start, stageStart('generate', 3), stopOf('aborted')]);
  const failed = viewOf([start, stageStart('generate', 3), step('error', { type: 'error', error: { type: 'model_error', message: 'Out of memory' } }), stopOf('error')]);
  for (const [why, view] of [['running', running], ['done', done], ['stopped', stopped], ['failed', failed]]) {
    assert.equal(q(renderAssistantCard(view, {}), '.test-notice')?.hidden, false, why);
  }
  const queued = renderAssistantCard(initialView('m1'), {});   // not yet started: no model is known
  assert.equal(q(queued, '.test-notice'), null);
  const accepted = viewOf([start]);   // the card of the model is known, and no stage has begun: nothing has been written that could be mistaken
  assert.equal(q(renderAssistantCard(accepted, {}), '.test-notice'), null);
  const question = viewOf([start, step('warning', { code: 'not_a_command', message: 'Send an image.' }), stopOf('done')]);   // no stage ran: no model ran
  assert.equal(q(renderAssistantCard(question, {}), '.test-notice'), null);
  const asking = viewOf([start, step('warning', { code: 'not_a_command', message: 'Send an image.' })]);   // and not for the moment before it is done
  assert.equal(q(renderAssistantCard(asking, {}), '.test-notice'), null);
  const begun = viewOf([start, stageStart('preprocess', 0)]);   // the first stage: from here every frame of the card has it
  assert.equal(q(renderAssistantCard(begun, {}), '.test-notice')?.textContent, TEST_NOTICE);
});

test('a real engine never gets the notice: not the published models, not the EOS model, not a card that names nothing', () => {
  for (const name of ['hybrid_150m_m3_rrg', 'hybrid_150m_v2_rrg', 'hybrid_150m_m3_rrg_eos', 'tiny2', 'Tiny', ' tiny', '', undefined]) {
    const model = { ...START.data.model, name };
    const view = viewOf([step('message_start', { ...START.data, model }), stageStart('generate', 3), snapshot('Findings: x.'),
                         stopOf('done', { report: 'Findings: x.', display_report: 'Findings: x.' })]);
    const card = renderAssistantCard(view, {});
    assert.equal(q(card, '.test-notice')?.hidden ?? true, true, String(name));
    assert.equal(card.textContent.includes('random weights'), false, String(name));
  }
  const bare = viewOf([step('message_start', { ...START.data, model: null }), stopOf('aborted')]);   // a log with no card at all
  assert.equal(q(renderAssistantCard(bare, {}), '.test-notice')?.hidden ?? true, true);
  assert.equal(q(renderAssistantCard(finished(), {}), '.test-notice')?.hidden ?? true, true);   // the card most of this file draws
});

test('isTestModel is true for the card named tiny and for nothing else', () => {
  assert.equal(typeof render.isTestModel, 'function');
  assert.equal(render.TEST_MODEL, 'tiny');
  assert.equal(render.isTestModel({ name: 'tiny' }), true);
  for (const other of [{ name: 'hybrid_150m_m3_rrg' }, { name: 'Tiny' }, { name: 'tiny ' }, { name: null }, {}, null, undefined, 'tiny', ['tiny'], 7]) {
    assert.equal(render.isTestModel(other), false, JSON.stringify(other));
  }
});

test('the user turn shows the note, the thumbnail and the option chips', async () => {
  const loaded = [];
  const ctx = { loadImage: async (path) => { loaded.push(path); return 'blob:test/1'; } };
  const turn = renderUserTurn({ text: 'beam 5\nplease', image: { url: '/v1/messages/u_test/image?variant=thumb', filename: 'chest.png' },
                                options: tinyView().options }, ctx);
  assert.equal(turn.localName, 'article');
  assert.ok(turn.classList.contains('turn') && turn.classList.contains('user'));
  assert.equal(q(turn, '.user-text').textContent, 'beam 5\nplease');
  assert.deepEqual(texts(qa(turn, '.options .chip')), ['beam 3', '16 tok', 'cached', 'k 4/3', 'raw text', 'full budget']);
  const img = q(turn, 'img');
  assert.equal(img.getAttribute('alt'), 'Uploaded X-ray: chest.png');
  await tick();
  assert.deepEqual(loaded, ['/v1/messages/u_test/image?variant=thumb']);
  assert.equal(img.getAttribute('src'), 'blob:test/1');   // through ctx.loadImage: the token never rides an <img src>
});

test('with ctx.retrieval false the user turn shows no k chip: the server ran no retrieval, so k was not used (P4-H)', () => {
  const options = tinyView().options;
  const chips = (ctx) => texts(qa(renderUserTurn({ text: 'x', options }, ctx), '.options .chip'));
  assert.deepEqual(chips({ retrieval: false }), ['beam 3', '16 tok', 'cached', 'raw text', 'full budget']);
  assert.deepEqual(chips({ retrieval: true }), ['beam 3', '16 tok', 'cached', 'k 4/3', 'raw text', 'full budget']);
  assert.deepEqual(chips({}), chips({ retrieval: true }));   // not said: taken to run, as serverHas takes an older server
  assert.equal(options.k_images, 4);   // the view is not written to
});

test('a preview URL is used as it is and a server path goes through ctx.loadImage; nothing else ever reaches src', async () => {
  const loaded = [];
  const ctx = { loadImage: async (p) => { loaded.push(p); return 'blob:test/1'; } };
  const srcOf = async (image, c = ctx) => { const t = renderUserTurn({ image }, c); await tick(); return q(t, 'img')?.getAttribute('src') ?? null; };
  assert.equal(await srcOf({ url: 'blob:http://localhost/abc-123' }), 'blob:http://localhost/abc-123');
  assert.equal(await srcOf({ url: 'data:image/png;base64,AAAA' }), 'data:image/png;base64,AAAA');
  assert.deepEqual(loaded, []);   // nothing fetched for a preview
  assert.equal(await srcOf({ url: '/v1/messages/u/image?variant=thumb' }), 'blob:test/1');
  assert.deepEqual(loaded, ['/v1/messages/u/image?variant=thumb']);

  loaded.length = 0;
  const refused = ['javascript:alert(1)', 'data:text/html,<b>x</b>', 'data:image/svg+xml;base64,AAAA', 'https://evil.example/x.png',
                   '//evil.example/x.png', '/\\evil.example/x.png', '\\\\evil.example/x.png', 'x.png', '', 42, {}, null,
                   '/\t/evil.example/x.png', '/\n/evil.example/x.png', '/\r/evil.example/x.png', '/ok\x00', '/ok\x7f'];   // a control character is dropped by the browser's URL parser, and "//host" is what is left
  for (const bad of refused) assert.equal(await srcOf({ url: bad }), null, `url ${JSON.stringify(bad)} was used`);
  assert.deepEqual(loaded, []);   // not even asked: neither the loader nor the token saw them, as api.js's own filter
  const hostile = { loadImage: async () => 'javascript:alert(1)' };   // an answer from the loader is checked too
  assert.equal(await srcOf({ url: '/v1/x' }, hostile), null);
  assert.equal(await srcOf({ url: '/v1/x' }, {}), null);   // no loader and no preview: nothing to show, nothing thrown
});

test('a thumbnail that fails to load leaves the turn intact and raises nothing', async () => {
  const unhandled = [];
  const onUnhandled = (reason) => unhandled.push(reason);
  process.on('unhandledRejection', onUnhandled);
  try {
    const turn = renderUserTurn({ text: 'hello', image: { url: '/v1/x', filename: 'a.png' } }, { loadImage: async () => { throw new Error('refused'); } });
    const thrower = renderUserTurn({ image: { url: '/v1/x' } }, { loadImage: () => { throw new Error('sync'); } });
    await tick();
    await tick();
    assert.equal(q(turn, 'img').hasAttribute('src'), false);
    assert.equal(q(turn, 'img').getAttribute('data-failed'), '');
    assert.ok(q(thrower, 'img'));
    assert.equal(q(turn, '.user-text').textContent, 'hello');
    assert.deepEqual(unhandled, []);
  } finally {
    process.off('unhandledRejection', onUnhandled);
  }
});

test('the thumbnail opens the viewer when the page can, handing over the image and the button it came from; without ctx.openViewer it is a picture only', () => {
  const image = { url: 'blob:x/1', filename: 'a.png' };
  const opened = [];
  const withViewer = renderUserTurn({ image }, { openViewer: (i, opener) => opened.push([i, opener]) });
  const open = q(withViewer, 'button.thumb-button');
  assert.ok(open);
  assert.equal(open.getAttribute('aria-label'), 'Open X-ray in the viewer');
  assert.ok(q(open, 'img'));
  open.click();
  assert.equal(opened.length, 1);
  assert.equal(opened[0][0], image);   // what renderUserTurn was given: the page says which side that is (ownXraySide)
  assert.ok(opened[0][1] === open, 'the second argument is the thumbnail\'s button, not ' + typeof opened[0][1]);   // the control it was opened from, for the focus to go back to, as for the card's pictures (P6-G)
  const plain = renderUserTurn({ image }, {});
  assert.equal(q(plain, 'button'), null);
  assert.ok(q(plain, 'img'));
});

test('a turn without an image has no picture; an image with no usable source shows its file name', () => {
  assert.equal(qa(renderUserTurn({ text: 'a note' }), 'img').length, 0);
  const named = renderUserTurn({ image: { filename: 'chest.png' } }, {});
  assert.equal(qa(named, 'img').length, 0);
  assert.equal(q(named, '.chip').textContent, 'chest.png');
  assert.equal(renderUserTurn({ text: 'a note' }).hasAttribute('hidden'), false);
});

// ---- the images row: your X-ray and what the model saw (P6-B) ----------------------------------------------------------

const URLS = { original: '/v1/messages/u_test/image?variant=original', thumb: '/v1/messages/u_test/image?variant=thumb',
               model_input: '/v1/messages/u_test/image?variant=model_input' };
const imageStart = (image = {}, more = {}) => step('message_start', { ...START.data, ...more, image: { ...START.data.image, urls: URLS, ...image } });
const PREPROCESSED = stageEnd('preprocess', 1.6, { format: 'PNG', input_px: [2544, 3056], source: 'upload' });
// A loader that answers every path with an object URL named after its variant, and keeps what it was asked.
function loader() {
  const asked = [];
  return { asked, loadImage: async (path) => { asked.push(path); return `blob:test/${String(path).split('variant=')[1] ?? path}`; } };
}

test('the images row shows your X-ray, labelled with its pixel size, and what the model saw, both through ctx.loadImage (P6-B)', async () => {
  const { asked, loadImage } = loader();
  const section = renderImages(viewOf([imageStart(), stageStart('preprocess', 0), PREPROCESSED]), { loadImage, turn: 2 });
  assert.equal(section.localName, 'section');
  assert.ok(section.classList.contains('images'));
  assert.equal(section.hasAttribute('hidden'), false);
  assert.equal(section.getAttribute('aria-label'), 'Images, turn 2');
  assert.equal(q(section, 'h3').textContent, 'Images');
  assert.deepEqual(texts(qa(section, 'figure figcaption')), ['Your X-ray · 2544×3056 px', 'What the model saw (224×224)']);
  const imgs = qa(section, 'img');
  assert.deepEqual(imgs.map((i) => i.getAttribute('data-src')), [URLS.thumb, URLS.model_input]);   // the 512 px thumbnail, not the original
  assert.deepEqual(imgs.map((i) => i.getAttribute('alt')), ['Your X-ray', 'What the model saw']);
  assert.ok(imgs.every((i) => !i.hasAttribute('src')));   // no source until a load resolves: an <img src> cannot carry the token
  await tick();
  assert.deepEqual(asked, [URLS.thumb, URLS.model_input]);
  assert.deepEqual(imgs.map((i) => i.getAttribute('src')), ['blob:test/thumb', 'blob:test/model_input']);
  assert.equal(qa(section, 'button').length, 0);   // no viewer to open: pictures only
});

test('your X-ray has no size until preprocess says it, a test study is named as one, and a public turn shows both pictures too', async () => {
  const { loadImage } = loader();
  const captions = (view) => texts(qa(renderImages(view, { loadImage }), 'figcaption'));
  assert.deepEqual(captions(viewOf([imageStart()])), ['Your X-ray', 'What the model saw (224×224)']);   // message_start only
  const study = viewOf([imageStart({ source: 'test_split', filename: null }), stageEnd('preprocess', 1, { input_px: [320, 320], source: 'test_split' })]);
  assert.deepEqual(captions(study), ['Test-split X-ray · 320×320 px', 'What the model saw (224×224)']);
  assert.equal(q(renderImages(study, { loadImage }), 'img').getAttribute('alt'), 'Test-split X-ray');
  const pub = viewOf([imageStart({ urls: { thumb: URLS.thumb, model_input: URLS.model_input } }, { mode: 'public' }), PREPROCESSED]);
  assert.deepEqual(captions(pub), ['Your X-ray · 2544×3056 px', 'What the model saw (224×224)']);   // a user's own upload, in both modes
  for (const odd of [[320], ['a', 'b'], [0, 5], [1.5, 2], 'x', null]) {
    const view = viewOf([imageStart(), stageEnd('preprocess', 1, { input_px: odd })]);
    assert.equal(captions(view)[0], 'Your X-ray', JSON.stringify(odd));   // a size it cannot read is no size, never "undefined"
  }
});

test('with nothing to show the images row is hidden: no loader, no image, no usable path; a failed load marks its own picture', async () => {
  const { asked, loadImage } = loader();
  const view = viewOf([imageStart(), PREPROCESSED]);
  assert.equal(renderImages(view, {}).hasAttribute('hidden'), true);   // no loader: nothing can be fetched with the token
  assert.equal(renderImages(viewOf([step('message_start', { ...START.data, image: null })]), { loadImage }).hasAttribute('hidden'), true);   // a question
  assert.equal(renderImages(initialView('m'), { loadImage }).hasAttribute('hidden'), true);
  const hostile = viewOf([imageStart({ urls: { thumb: 'https://evil.example/x.png', model_input: '//evil.example/y.png', original: 'blob:x' } })]);
  assert.equal(renderImages(hostile, { loadImage }).hasAttribute('hidden'), true);
  await tick();
  assert.deepEqual(asked, []);   // not even asked: only a path on this origin is fetched with the token
  const half = renderImages(viewOf([imageStart({ urls: { thumb: URLS.thumb } })]), { loadImage });
  assert.deepEqual(texts(qa(half, 'figcaption')), ['Your X-ray']);   // the tile that has a path is shown, alone
  const failing = renderImages(view, { loadImage: async (path) => { if (path === URLS.thumb) throw new Error('refused'); return 'blob:ok'; } });
  await tick();
  await tick();
  assert.deepEqual(qa(failing, 'img').map((i) => [i.hasAttribute('data-failed'), i.getAttribute('src')]), [[true, null], [false, 'blob:ok']]);
});

test('with ctx.openViewer each picture is a button that opens it in viewer.js: your X-ray at full size, what the model saw (P6-E)', () => {
  const { loadImage } = loader();
  const opened = [];
  const ctx = { loadImage, openViewer: (spec, opener) => opened.push([spec, opener]), turn: 2 };
  const section = renderImages(viewOf([imageStart(), PREPROCESSED]), ctx);
  const buttons = qa(section, 'button[data-action="view"]');
  assert.deepEqual(buttons.map((b) => b.getAttribute('aria-label')), ['Open your X-ray in the viewer, turn 2', 'Open what the model saw in the viewer, turn 2']);
  assert.ok(buttons.every((b) => q(b, 'img') && b.getAttribute('type') === 'button'));   // a button: Enter and Space open it too
  buttons.forEach((b) => b.click());
  // A side for viewer.js: the size is the loaded picture's own (ruling 10), so a label is a name; the original, which no card shows, is
  // transient: the viewer lets go of it when it closes.
  assert.deepEqual(opened.map(([spec]) => spec), [{ left: { url: URLS.original, label: 'Your X-ray', note: '', transient: true } },
                                                  { left: { url: URLS.model_input, label: 'What the model saw', note: '', transient: false } }]);
  assert.deepEqual(opened.map(([, opener]) => opener), buttons);   // and the control it was opened from, for the focus to go back to
  opened.length = 0;
  const pub = renderImages(viewOf([imageStart({ urls: { thumb: URLS.thumb, model_input: URLS.model_input } })]), ctx);
  q(pub, 'button').click();
  assert.deepEqual(opened[0][0], { left: { url: URLS.thumb, label: 'Your X-ray', note: '(thumbnail)', transient: false } });   // no original in public mode: what the card shows
  opened.length = 0;
  const study = renderImages(viewOf([imageStart({ source: 'test_split', filename: null }), stageEnd('preprocess', 1, { input_px: [320, 320] })]), ctx);
  q(study, 'button').click();
  assert.deepEqual(opened[0][0], { left: { url: URLS.original, label: 'Test-split X-ray', note: '(stored gallery size)', transient: true } });   // a 320 px JPEG
});

// ---- the viewer from the user's own message: the same side as the card's "Your X-ray" tile (P6-G) --------------------------

test('the thumbnail of the user turn and the "Your X-ray" tile of the card are given one side: the original where the mode sends one, else the thumbnail, a test study at its stored size (P6-G)', () => {
  const { loadImage } = loader();
  const opened = [];
  const ctx = { loadImage, openViewer: (spec) => opened.push(spec) };
  const tileSide = (view) => {   // what a click on the card's tile hands the viewer
    opened.length = 0;
    q(renderImages(view, ctx), 'button[data-action="view"]').click();   // the first picture of the row: your X-ray
    return opened[0].left;
  };
  const thumb = { url: URLS.thumb, filename: 'chest.png' };
  const studyThumb = { url: URLS.thumb, filename: null, source: 'test_split' };
  const upload = viewOf([imageStart(), PREPROCESSED]);
  const pub = viewOf([imageStart({ urls: { thumb: URLS.thumb, model_input: URLS.model_input } }, { mode: 'public' }), PREPROCESSED]);
  const study = viewOf([imageStart({ source: 'test_split', filename: null }), stageEnd('preprocess', 1, { input_px: [320, 320] })]);
  for (const [view, image] of [[upload, thumb], [pub, thumb], [study, studyThumb]]) {
    assert.deepEqual(ownXraySide(view, image, ctx), tileSide(view));   // not a copy of the tile's choice: its own answer, held to equal
  }
  assert.deepEqual(ownXraySide(upload, thumb, ctx), { url: URLS.original, label: 'Your X-ray', note: '', transient: true });
  assert.deepEqual(ownXraySide(pub, thumb, ctx), { url: URLS.thumb, label: 'Your X-ray', note: '(thumbnail)', transient: false });   // public mode sends no original
  assert.deepEqual(ownXraySide(study, studyThumb, ctx), { url: URLS.original, label: 'Test-split X-ray', note: '(stored gallery size)', transient: true });
  const frozen = deepFreeze(viewOf([imageStart(), PREPROCESSED]));
  assert.deepEqual(ownXraySide(frozen, deepFreeze({ ...thumb }), ctx), ownXraySide(upload, thumb, ctx));   // it writes to neither the view nor the image
});

test('with no card to say, the side is what the thumbnail itself shows: the page\'s own preview of the file, else the server\'s thumbnail marked as such (P6-G)', () => {
  const { loadImage } = loader();
  const ctx = { loadImage };
  const preview = { url: 'blob:http://localhost/abc-123', filename: 'chest.png' };
  const thumb = { url: URLS.thumb, filename: 'chest.png' };
  const studyThumb = { url: URLS.thumb, filename: null, source: 'test_split' };
  const noCard = [null, undefined, {}, initialView('m_x'),                                     // a live turn before message_start
                  viewOf([imageStart({ urls: {} })]),                                          // a log whose message_start names no picture
                  viewOf([step('message_start', { ...START.data, image: null })]),             // one with no image at all
                  viewOf([imageStart({ urls: { thumb: 'https://evil.example/x.png', original: '//evil.example/y.png' } })])];   // and none this page may fetch
  for (const view of noCard) {
    const named = JSON.stringify(view);
    assert.deepEqual(ownXraySide(view, preview, ctx), { url: preview.url, label: 'Your X-ray', note: '', transient: false }, named);   // the file as it was attached: no note, nothing for the viewer to let go of
    assert.deepEqual(ownXraySide(view, thumb, ctx), { url: URLS.thumb, label: 'Your X-ray', note: '(thumbnail)', transient: false }, named);
    assert.deepEqual(ownXraySide(view, studyThumb, ctx), { url: URLS.thumb, label: 'Test-split X-ray', note: '(stored gallery size)', transient: false }, named);
  }
  assert.deepEqual(ownXraySide(null, { url: 'data:image/png;base64,AAAA' }, ctx).url, 'data:image/png;base64,AAAA');   // a data: picture is the page's own too
  for (const bad of ['https://evil.example/x.png', '//evil.example/x.png', 'javascript:alert(1)', 'data:image/svg+xml;base64,AAAA', 'x.png', '', 42, null, undefined]) {
    assert.equal(ownXraySide(null, { url: bad }, ctx), null, `url ${JSON.stringify(bad)} was given to the viewer`);   // nothing this page may show or fetch with the token
  }
  assert.equal(ownXraySide(null, null, ctx), null);                                          // no image
  assert.equal(ownXraySide(null, 'x', ctx), null);
  assert.equal(ownXraySide(null, thumb, {}), null);                                          // a server path needs the page's loader
  assert.equal(ownXraySide(null, thumb, undefined), null);
  assert.equal(ownXraySide(null, preview, undefined).url, preview.url);                      // the page's own picture needs none
});

test('the card shows the images row after the labels when the page can load images, and is the card it always was without a loader', () => {
  const { loadImage } = loader();
  const view = viewOf([imageStart(), stageStart('preprocess', 0), PREPROCESSED, stageStart('generate', 3), snapshot('Findings: ok'),
                       stageEnd('generate', 9, GENERATE), skipped('label', 'labeler_unavailable'), skipped('score', 'no_reference'),
                       stopOf('done', { report: 'Findings: ok', display_report: 'Findings: ok' })]);
  const card = renderAssistantCard(view, { loadImage, copy() {}, showModels() {} });
  assert.deepEqual(card.children.map((c) => c.getAttribute('class')), ['timeline', 'notes', 'report', 'labels', 'images', 'provenance']);
  const plain = renderAssistantCard(view, { copy() {}, showModels() {} });
  assert.deepEqual(plain.children.map((c) => c.getAttribute('class')), ['timeline', 'notes', 'report', 'labels', 'provenance']);
});

test('a user turn\'s server thumbnail carries its path in data-src, and a test study\'s thumbnail says what it is', async () => {
  const { loadImage } = loader();
  const upload = renderUserTurn({ image: { url: URLS.thumb, filename: 'chest.png' } }, { loadImage });
  assert.equal(q(upload, 'img').getAttribute('data-src'), URLS.thumb);
  assert.equal(q(upload, 'img').getAttribute('alt'), 'Uploaded X-ray: chest.png');
  const study = renderUserTurn({ image: { url: URLS.thumb, filename: null, source: 'test_split' } }, { loadImage });
  assert.equal(q(study, 'img').getAttribute('alt'), 'Test-split X-ray');
  await tick();
  assert.equal(q(study, 'img').getAttribute('src'), 'blob:test/thumb');
  const preview = renderUserTurn({ image: { url: 'blob:http://localhost/abc', filename: 'chest.png' } }, { loadImage });
  assert.equal(q(preview, 'img').hasAttribute('data-src'), false);   // a local preview is the page's own: no server path
});

test('focusKey and restoreFocus find a viewer button again in the rebuilt card', () => {
  const { loadImage } = loader();
  const view = viewOf([imageStart(), PREPROCESSED, stageStart('generate', 3), snapshot('Findings: so far')]);
  const ctx = { loadImage, openViewer() {}, ui: new Map() };
  let card = renderAssistantCard(view, ctx);
  document.body.replaceChildren(card);
  const second = qa(card, 'button[data-action="view"]')[1];
  second.focus();
  assert.equal(focusKey(document.activeElement), 'view:image:model_input');   // by what it shows, not its place in the card (P6 fix 1)
  const next = renderAssistantCard(view, ctx);
  card.replaceWith(next);
  card = next;
  assert.equal(restoreFocus(card, 'view:image:model_input'), true);
  assert.equal(document.activeElement, qa(card, 'button[data-action="view"]')[1]);
  assert.equal(restoreFocus(card, 'view:image:nope'), false);   // no such button: false, quietly
  document.body.replaceChildren();
});

// ---- similar X-rays: the neighbour grid (P6-C) ----------------------------------------------------------------------------

const neighbour = (rank, more = {}) => ({ rank, similarity: Number((0.95 - 0.05 * rank).toFixed(3)), gallery_row: 100 + rank,
                                          study_id: 50000000 + rank, txt_row: 200 + rank, image_url: `/v1/gallery/images/${100 + rank}`,
                                          labels: labelled(rank === 1 ? ['Edema'] : []), ...more });
const agreementOf = (rank, more = {}) => ({ rank, agree: 13, of: 14, both_positive: [], neighbor_only: [], generated_only: [], ...more });
const retrieved = (neighbours, more = {}) => stageEnd('retrieve', 3, { image_neighbors: neighbours, report_matches: [], gallery: { images: 200 }, ...more });
const gridSteps = (neighbours = [neighbour(1), neighbour(2)]) => [imageStart(), PREPROCESSED, stageStart('retrieve', 2), retrieved(neighbours),
                                                                  stageStart('generate', 3), snapshot('Findings: so far'), stageEnd('generate', 9, GENERATE)];
const gridView = (...more) => viewOf([...gridSteps(), ...more]);
const items = (section) => qa(section, 'li.neighbor');
const agreements = (section) => items(section).map((li) => q(li, '.agreement')?.textContent ?? null);

test('the similar X-rays are a grid: each neighbour\'s picture, "#rank · similarity" and a bar as long as the similarity (P6-C)', async () => {
  const { asked, loadImage } = loader();
  const section = renderNeighbors(gridView(), { loadImage, turn: 2 });
  assert.equal(section.localName, 'section');
  assert.ok(section.classList.contains('neighbors'));
  assert.equal(section.hasAttribute('hidden'), false);
  assert.equal(section.getAttribute('aria-label'), 'Similar X-rays, turn 2');
  assert.equal(q(section, 'h3').textContent, 'Similar X-rays (13D tower)');
  assert.equal(q(section, 'ol').getAttribute('role'), 'list');   // a list to a screen reader where the style drops the numbers
  assert.deepEqual(items(section).map((li) => li.getAttribute('data-rank')), ['1', '2']);
  assert.deepEqual(items(section).map((li) => q(li, '.neighbor-score').textContent), ['#1 · 0.900', '#2 · 0.850']);
  assert.deepEqual(items(section).map((li) => q(li, '.sim-fill').getAttribute('style')), ['width: 90%', 'width: 85%']);
  assert.ok(items(section).every((li) => q(li, '.sim-bar').getAttribute('aria-hidden') === 'true'));   // the number is said; the bar draws it
  const imgs = qa(section, 'img');
  assert.deepEqual(imgs.map((i) => [i.getAttribute('data-src'), i.getAttribute('alt')]),
                   [['/v1/gallery/images/101', 'Similar X-ray #1'], ['/v1/gallery/images/102', 'Similar X-ray #2']]);
  await tick();
  assert.deepEqual(asked, ['/v1/gallery/images/101', '/v1/gallery/images/102']);   // through the loader, with the page's auth
  assert.equal(qa(section, '.neighbor-placeholder').length, 0);
  assert.equal(q(section, '.section-note'), null);   // nothing to explain in private mode
});

test('a similarity outside 0..1 keeps its bar inside the box, and a neighbour that lacks fields still renders cleanly', () => {
  const { loadImage } = loader();
  const odd = renderNeighbors(viewOf([...gridSteps([neighbour(1, { similarity: 1.4 }), neighbour(2, { similarity: -0.2 }),
                                                    { similarity: 'x', rank: 'y' }, null, 7])]), { loadImage });
  assert.deepEqual(items(odd).map((li) => q(li, '.sim-fill')?.getAttribute('style') ?? null), ['width: 100%', 'width: 0%', null]);
  assert.equal(items(odd)[2].hasAttribute('data-rank'), false);
  assertClean(odd, 'odd neighbours');
});

test('n/14 labels agree once the label stage has ended, with chips for the positives that differ; "labels…" before it', () => {
  const { loadImage } = loader();
  assert.deepEqual(agreements(renderNeighbors(gridView(), { loadImage })), ['labels…', 'labels…']);   // generate has ended, label is next
  assert.deepEqual(agreements(renderNeighbors(gridView(stageStart('label', 4)), { loadImage })), ['labels…', 'labels…']);
  const detail = { chexbert_14: labelled(['Cardiomegaly']), positives: ['Cardiomegaly'],
                   neighbor_agreement: [agreementOf(1, { agree: 12, neighbor_only: ['Edema'], generated_only: ['Cardiomegaly'] }), agreementOf(2)] };
  const section = renderNeighbors(gridView(stageStart('label', 4), stageEnd('label', 3, detail)), { loadImage });
  assert.deepEqual(agreements(section), ['12/14 labels agree', '13/14 labels agree']);
  assert.deepEqual(texts(qa(items(section)[0], 'li.chip.diff')), ['neighbour: Edema', 'report: Cardiomegaly']);
  assert.equal(q(items(section)[0], 'ul').getAttribute('aria-label'), 'Labels that differ, similar X-ray #1');
  assert.equal(qa(items(section)[1], '.chip').length, 0);   // every label agrees: nothing to name
  assert.equal(qa(section, 'li.chip.label').length, 0);     // not the report's own label chips (the card has exactly 14 of those)
});

test('the agreement line says why there is none: labels off, the labeller unavailable or stopped, the gallery still being labelled', () => {
  const { loadImage } = loader();
  const line = (...more) => agreements(renderNeighbors(gridView(...more), { loadImage }));
  assert.deepEqual(line(skipped('label', 'labeler_unavailable')), ['labels unavailable', 'labels unavailable']);
  assert.deepEqual(line(skipped('label', 'label_off')), ['labels off', 'labels off']);   // the user's own setting, said as that
  assert.deepEqual(line(stopOf('aborted')), ['labels unavailable', 'labels unavailable']);   // stopped before the label stage
  assert.deepEqual(line(stageStart('label', 4), stopOf('error')), ['labels unavailable', 'labels unavailable']);
  const pending = viewOf([...gridSteps([neighbour(1, { labels: null }), neighbour(2, { labels: null })])]);   // the gallery's labels are not built
  assert.deepEqual(agreements(renderNeighbors(pending, { loadImage })), ['labels pending', 'labels pending']);
  const off = viewOf([step('message_start', { ...imageStart().data, options: { ...START.data.options, label: false } }), ...gridSteps().slice(1)]);
  assert.deepEqual(agreements(renderNeighbors(off, { loadImage })), ['labels off', 'labels off']);   // said from the start, as the chips say it
});

test('a server with no labeller (features.labels false) never says "labelling…" or "labels…": no labels are coming (P6-C ruling)', () => {
  const { loadImage } = loader();
  const running = labelling();   // the label stage runs
  assert.equal(renderLabels(running, { labelNames: LABEL_NAMES }).textContent, 'labelling…');
  const none = renderLabels(running, { labelNames: LABEL_NAMES, labels: false });
  assert.equal(none.textContent.includes('labelling'), false);
  assert.deepEqual(agreements(renderNeighbors(gridView(), { loadImage, labels: false })), ['labels unavailable', 'labels unavailable']);
  const ended = viewOf([START, stageStart('generate', 3), stageEnd('generate', 9, GENERATE), skipped('label', 'labeler_unavailable'), stopOf('done')]);
  assert.equal(renderLabels(ended, { labelNames: LABEL_NAMES, labels: false }).textContent, 'labels unavailable (labeler_unavailable)');   // its end is said as before
  const off = viewOf([step('message_start', { ...START.data, options: { ...START.data.options, label: false } }), stageStart('generate', 3)]);
  assert.equal(renderLabels(off, { labelNames: LABEL_NAMES, labels: false }).textContent, 'labels off');   // the user's own setting still shows
  assert.equal(labelsPending(running), true);
  assert.equal(labelsPending(running, false), false);   // state.js: nothing pending from a server that runs no labeller
  assert.equal(labelsPending(running, true), true);
});

test('public mode: rank and similarity only, a neutral placeholder for each picture, and not one request for a gallery image (R1)', async () => {
  const { asked, loadImage } = loader();
  const pubStart = imageStart({ urls: { thumb: URLS.thumb, model_input: URLS.model_input } }, { mode: 'public' });
  const pub = viewOf([pubStart, PREPROCESSED, stageStart('retrieve', 2),
                      retrieved([{ rank: 1, similarity: 0.9 }, { rank: 2, similarity: 0.85 }, { rank: 3, similarity: 0.8 }]),
                      stageStart('generate', 3), snapshot('Findings: ok'), stageEnd('generate', 9, GENERATE),
                      stageStart('label', 4), stageEnd('label', 3, { chexbert_14: labelled(['Edema']), positives: ['Edema'] }),
                      stopOf('done', { report: 'Findings: ok', display_report: 'Findings: ok' })]);
  const section = renderNeighbors(pub, { loadImage, openViewer() {} });
  assert.equal(section.hasAttribute('hidden'), false);
  assert.equal(q(section, '.section-note').textContent, 'Public mode shows rank and similarity only.');
  assert.deepEqual(items(section).map((li) => q(li, '.neighbor-score').textContent), ['#1 · 0.900', '#2 · 0.850', '#3 · 0.800']);
  assert.equal(qa(section, '.neighbor-placeholder').length, 3);
  assert.ok(qa(section, '.neighbor-placeholder').every((p) => p.getAttribute('aria-hidden') === 'true'));
  assert.deepEqual([qa(section, 'img').length, qa(section, 'button').length, qa(section, '.agreement').length, qa(section, '.chip').length], [0, 0, 0, 0]);
  const card = renderAssistantCard(pub, { loadImage, labelNames: LABEL_NAMES });   // the whole card: its images row asks for the user's own upload
  await tick();
  assert.deepEqual(asked.filter((path) => path.startsWith('/v1/gallery/')), []);
  assert.ok(asked.every((path) => path.startsWith('/v1/messages/u_test/image?')), asked.join(', '));
  assertClean(card, 'public card');
  // Defence in depth: a public turn whose neighbour somehow kept an image path still shows a placeholder and asks for nothing.
  const leaked = viewOf([pubStart, stageStart('retrieve', 2), retrieved([{ rank: 1, similarity: 0.9, image_url: '/v1/gallery/images/5' }])]);
  asked.length = 0;
  const kept = renderNeighbors(leaked, { loadImage });
  await tick();
  assert.deepEqual([asked, qa(kept, '.neighbor-placeholder').length, qa(kept, 'img').length], [[], 1, 0]);
});

test('a neighbour that is the upload itself, byte for byte, says so', () => {
  const { loadImage } = loader();
  const identical = (to) => viewOf([imageStart(), stageEnd('preprocess', 1, { input_px: [320, 320], identical_to: to }), stageStart('retrieve', 2),
                                    retrieved([neighbour(1), neighbour(2)])]);
  const badges = (view) => items(renderNeighbors(view, { loadImage })).map((li) => q(li, '.badge')?.textContent ?? null);
  assert.deepEqual(badges(identical({ split: 'train', row: 102 })), [null, 'identical to your upload']);
  assert.deepEqual(badges(identical({ split: 'test', row: 102 })), [null, null]);   // a test image is in no neighbour list
  assert.deepEqual(badges(identical(null)), [null, null]);
  assert.deepEqual(badges(gridView()), [null, null]);
});

test('with ctx.openViewer a similar X-ray opens beside your X-ray, its note the stored gallery size (P6-E)', () => {
  const { loadImage } = loader();
  const opened = [];
  const section = renderNeighbors(gridView(), { loadImage, openViewer: (spec, opener) => opened.push([spec, opener]), turn: 3 });
  const buttons = qa(section, 'button[data-action="view"]');
  assert.deepEqual(buttons.map((b) => b.getAttribute('aria-label')),
                   ['Open similar X-ray #1 beside your X-ray in the viewer, turn 3', 'Open similar X-ray #2 beside your X-ray in the viewer, turn 3']);
  buttons[0].click();
  assert.deepEqual(opened, [[{ left: { url: URLS.original, label: 'Your X-ray', note: '', transient: true },
                               right: { url: '/v1/gallery/images/101', label: 'Similar X-ray #1', note: '(stored gallery size)', transient: false } },
                             buttons[0]]]);
  assert.equal(qa(section, 'li > button').length, 0);   // the button is inside the figure, not a child of the list item
  opened.length = 0;
  const alone = renderNeighbors(viewOf([imageStart({ urls: {} }), stageStart('retrieve', 2), retrieved([neighbour(1)])]), { loadImage, openViewer: (spec) => opened.push(spec) });
  q(alone, 'button[data-action="view"]').click();   // nothing of the turn's own to set beside it: the similar X-ray alone
  assert.deepEqual(opened, [{ left: { url: '/v1/gallery/images/101', label: 'Similar X-ray #1', note: '(stored gallery size)', transient: false } }]);
  assert.equal(q(alone, 'button').getAttribute('aria-label'), 'Open similar X-ray #1 in the viewer');
});

test('the card puts the similar X-rays after the images row; a skipped retrieve or k 0 leaves the section out', () => {
  const { loadImage } = loader();
  const card = renderAssistantCard(gridView(stopOf('done', { report: 'Findings: so far', display_report: 'Findings: so far' })), { loadImage });
  assert.deepEqual(card.children.map((c) => c.getAttribute('class')), ['timeline', 'notes', 'report', 'labels', 'images', 'neighbors', 'provenance']);
  const skip = viewOf([imageStart(), PREPROCESSED, skipped('retrieve', 'gallery_unavailable')]);
  assert.equal(renderNeighbors(skip, { loadImage }).hasAttribute('hidden'), true);
  assert.equal(q(renderAssistantCard(skip, { loadImage }), 'section.neighbors'), null);
  assert.equal(renderNeighbors(viewOf([imageStart(), stageStart('retrieve', 2), retrieved([])]), { loadImage }).hasAttribute('hidden'), true);
  assert.equal(renderNeighbors(viewOf([imageStart(), stageStart('retrieve', 2)]), { loadImage }).hasAttribute('hidden'), true);   // still running
});

// ---- matching reports, the own-report badge and the published lines (P6-D) ---------------------------------------------------

const LONG_REPORT = `Findings: ${'The lungs are clear without focal consolidation. '.repeat(6)}Impression: No acute cardiopulmonary process.`;   // > 180
const match = (rank, more = {}) => ({ rank, similarity: Number((0.6 - 0.02 * rank).toFixed(3)), group: 300 + rank, group_size: rank === 1 ? 37 : 1,
                                      txt_row: 400 + rank, report: rank === 1 ? LONG_REPORT : `Findings: short report ${rank}.`,
                                      labels: labelled(rank === 1 ? ['Edema', 'Cardiomegaly'] : []), ...more });
const OWN_RANK = { rank: 3, of: 2663, rank_dedup: 2, n_tied: 1, hit_at_10: true, protocol: 'i2t, official test split, strict pairing' };
const matchesView = (detail = {}, ...after) => viewOf([imageStart(), PREPROCESSED, stageStart('retrieve', 2),
  stageEnd('retrieve', 3, { image_neighbors: [], report_matches: [match(1), match(2)], gallery: { images: 200 }, ...detail }), ...after]);
const matchItems = (section) => qa(section, 'li.match');

test('the matching reports are a ranked list: "#rank · similarity", how many identical reports, the report and its positive labels (P6-D)', () => {
  const section = renderMatches(matchesView(), { labelNames: LABEL_NAMES, turn: 2 });
  assert.equal(section.localName, 'section');
  assert.ok(section.classList.contains('matches'));
  assert.equal(section.hasAttribute('hidden'), false);
  assert.equal(section.getAttribute('aria-label'), 'Matching reports, turn 2');
  assert.equal(q(section, 'h3').textContent, 'Matching reports (13D image and text encoders)');
  assert.equal(q(section, 'ol').getAttribute('role'), 'list');
  assert.deepEqual(matchItems(section).map((li) => li.getAttribute('data-rank')), ['1', '2']);
  assert.deepEqual(matchItems(section).map((li) => q(li, '.match-score').textContent), ['#1 · 0.580', '#2 · 0.560']);
  assert.deepEqual(matchItems(section).map((li) => q(li, '.group-size')?.textContent ?? null), ['×37 identical reports', null]);   // a group of one says nothing
  assert.deepEqual(matchItems(section).map((li) => q(li, '.clamp-text').textContent), [LONG_REPORT, 'Findings: short report 2.']);
  const [long, short] = matchItems(section).map((li) => q(li, '.clamp-text'));
  assert.ok([long, short].every((t) => t.classList.contains('clamped') && !t.classList.contains('open')));   // every report: three lines (P6 fix 1)
  const toggle = q(matchItems(section)[0], 'button[data-action="show"]');
  assert.deepEqual([toggle.textContent, toggle.getAttribute('aria-expanded'), toggle.getAttribute('aria-label'), toggle.getAttribute('aria-controls'),
                    toggle.getAttribute('data-key')],
                   ['Show all', 'false', 'Show all of matching report 1, turn 2', long.getAttribute('id'), 'match:1']);
  assert.equal(q(matchItems(section)[1], 'button[data-action="show"]').getAttribute('data-key'), 'match:2');
  assert.deepEqual(matchItems(section).map((li) => q(li, '.clamp-actions').hidden), [true, true]);   // until the page has measured them
  assert.deepEqual(texts(qa(matchItems(section)[0], 'li.chip.finding')), ['Cardiomegaly', 'Edema']);   // the positives, in the labeller's order
  assert.equal(q(matchItems(section)[0], 'ul').getAttribute('aria-label'), 'Labels of matching report 1');
  assert.equal(qa(matchItems(section)[1], '.chip').length, 0);
  assert.equal(qa(section, 'li.chip.label').length, 0);   // not the report's own 14 chips
  assert.equal(qa(section, 'li > button').length, 0);     // a control sits in its own row, not straight in the list item
  assert.equal(q(section, '.own-rank'), null);           // an upload: no own report to rank
});

test('Show all opens a clamped report and Show less closes it; ctx.ui keeps it open through the whole-card replace, and focus comes back', () => {
  const view = matchesView({}, stageStart('generate', 3), snapshot('Findings: so far'));
  const ui = new Map();
  const ctx = { labelNames: LABEL_NAMES, ui, turn: 1 };
  let card = renderAssistantCard(view, ctx);
  document.body.replaceChildren(card);
  settleClamps(card, () => true);   // as the page does once the card is in it: here every report overflows its three lines
  const toggle = q(card, 'section.matches button[data-action="show"]');
  toggle.click();
  const text = () => q(card, 'section.matches li.match .clamp-text');
  assert.ok(text().classList.contains('open'));
  assert.deepEqual([toggle.textContent, toggle.getAttribute('aria-expanded'), toggle.getAttribute('aria-label')],
                   ['Show less', 'true', 'Show less of matching report 1, turn 1']);
  toggle.focus();
  assert.equal(focusKey(document.activeElement), 'show:match:1');
  const next = renderAssistantCard(view, ctx);   // the next frame
  card.replaceWith(next);
  card = next;
  assert.ok(text().classList.contains('open'));   // still open, and its Show less with it: an open report is not measured away
  assert.equal(q(card, 'section.matches .clamp-actions').hidden, false);
  assert.equal(restoreFocus(card, 'show:match:1'), true);
  assert.equal(document.activeElement, q(card, 'section.matches button[data-action="show"]'));
  document.activeElement.click();                 // closed again, and the next frame keeps it closed
  card.replaceWith(renderAssistantCard(view, ctx));
  assert.ok(!q(document.body, 'section.matches li.match .clamp-text').classList.contains('open'));
  const fresh = renderAssistantCard(view, { labelNames: LABEL_NAMES });   // without a store a card starts closed
  assert.ok(!q(fresh, 'section.matches li.match .clamp-text').classList.contains('open'));
  document.body.replaceChildren();
});

test('Show less on a report that fits its three lines by now (widened while open) takes its row away: no Show all is left that reveals nothing (P6-A g)', () => {
  const layout = (text, lines) => {   // what the page would measure: three lines of 21 px shown, `lines` needed
    Object.defineProperty(text, 'scrollHeight', { configurable: true, value: 21 * lines });
    Object.defineProperty(text, 'clientHeight', { configurable: true, value: 21 * Math.min(3, lines) });
  };
  const open = (lines) => {
    const card = renderAssistantCard(matchesView(), { labelNames: LABEL_NAMES, ui: new Map(), turn: 1 });
    document.body.replaceChildren(card);
    settleClamps(card, () => true);   // narrow when it was drawn: the report overflowed its three lines
    const parts = { card, text: q(card, 'section.matches li.match .clamp-text'), toggle: q(card, 'section.matches button[data-action="show"]') };
    parts.row = parts.toggle.parentNode;
    parts.toggle.click();   // Show all
    layout(parts.text, lines);   // the window changes while it is open
    parts.toggle.focus();
    parts.toggle.click();   // Show less
    return parts;
  };
  const fits = open(2);   // wider now: two lines hold it
  assert.equal(fits.text.classList.contains('open'), false);
  assert.equal(fits.row.hidden, true);   // measured as it closed, not at the next paint or resize
  assert.equal(document.activeElement, fits.text);   // the focus is not lost with its button: it is on the report the button was for
  assert.equal(fits.text.getAttribute('tabindex'), '-1');
  const still = open(5);   // still too long for three lines: its Show all stays
  assert.deepEqual([still.text.classList.contains('open'), still.row.hidden, still.toggle.textContent], [false, false, 'Show all']);
  assert.equal(document.activeElement, still.toggle);
  assert.equal(still.text.getAttribute('tabindex'), '-1');   // the page can focus a report at any time (P6 fix 1); Tab never stops at it
  document.body.replaceChildren();
});

test('the report a Show less left the focus on is found again by its key when a running card is painted again: never the page body (P6 fix 1)', () => {
  const view = matchesView({}, stageStart('generate', 3), snapshot('Findings: so far'));   // the turn runs on: each frame paints a new card
  const ctx = { labelNames: LABEL_NAMES, ui: new Map(), turn: 1 };
  let card = renderAssistantCard(view, ctx);
  document.body.replaceChildren(card);
  settleClamps(card, () => true);
  const toggle = q(card, 'section.matches button[data-action="show"]');
  toggle.click();   // Show all
  const text = q(card, 'section.matches li.match .clamp-text');
  Object.defineProperty(text, 'scrollHeight', { configurable: true, value: 42 });   // widened meanwhile: two lines hold it
  Object.defineProperty(text, 'clientHeight', { configurable: true, value: 42 });
  toggle.focus();
  toggle.click();   // Show less: the row goes, and its focus to the report
  assert.equal(document.activeElement, text);
  const key = focusKey(document.activeElement);   // what app.js paint() reads before it replaces the card ...
  assert.equal(key, 'clamp:match:1');
  const next = renderAssistantCard(view, ctx);
  card.replaceWith(next);
  card = next;
  assert.equal(restoreFocus(card, key), true);   // ... and gives back after
  assert.equal(document.activeElement, q(card, 'section.matches li.match .clamp-text'));
  document.body.replaceChildren();
});

test('every clamped report shows Show all only when it really overflows its three lines: measured in the page, never counted (P6 fix 1)', () => {
  const section = renderMatches(matchesView({ report_matches: [match(1), match(2), match(3, { report: 'x'.repeat(170) })] }), { turn: 2 });
  const rows = () => matchItems(section).map((li) => q(li, '.clamp-actions').hidden);
  const measured = [];
  settleClamps(section, (text) => { measured.push(text.getAttribute('id')); return text.textContent.length > 100; });
  assert.deepEqual(rows(), [false, true, false]);   // the long one and the 170-character one overflow here; the short one fits
  assert.equal(measured.length, 3);
  settleClamps(section, () => false);               // wider now: nothing overflows, and no button reveals nothing
  assert.deepEqual(rows(), [true, true, true]);
  settleClamps(section);                            // the page's own measure; with no layout to read (here) it shows no button
  assert.deepEqual(rows(), [true, true, true]);
  q(matchItems(section)[0], 'button[data-action="show"]').click();   // an open report keeps its Show less, whatever is measured
  settleClamps(section, () => false);
  assert.deepEqual(rows(), [false, true, true]);
  for (const odd of [null, undefined, {}, 'x']) settleClamps(odd);   // nothing to settle: no throw
});

test('Show all keeps the focus on the same report when the score stage puts the published lines above it (P6 fix 1)', () => {
  const steps = [imageStart(), PREPROCESSED, stageStart('retrieve', 2),
                 stageEnd('retrieve', 3, { image_neighbors: [], report_matches: [match(1), match(2, { report: LONG_REPORT })], gallery: {} }),
                 stageStart('generate', 3), snapshot('Findings: live.'), stageEnd('generate', 9, GENERATE), stageStart('score', 5)];
  const ctx = { ui: new Map() };
  let card = renderAssistantCard(viewOf(steps), ctx);
  document.body.replaceChildren(card);
  settleClamps(card, () => true);
  q(card, 'section.matches button[data-key="match:2"]').focus();
  const key = focusKey(document.activeElement);
  assert.equal(key, 'show:match:2');
  const scored = [...steps, stageEnd('score', 1, { ...SCORE, published: { model_report: LONG_REPORT, floor_report: LONG_REPORT, live_equals_published: false } })];
  const next = renderAssistantCard(viewOf(scored), ctx);
  card.replaceWith(next);
  card = next;
  settleClamps(card, () => true);
  assert.equal(qa(card, 'section.published button[data-action="show"]').length, 2);   // two more Show all buttons, before the reports' own
  assert.equal(restoreFocus(card, key), true);
  assert.equal(document.activeElement.getAttribute('data-key'), 'match:2');            // the same report's, not the second of the card
  assert.equal(focusKey(document.activeElement), 'show:match:2');
  for (const bad of ['show:', 'show:match:2"]', 'show:../x', 'view:' + 'a'.repeat(300)]) assert.equal(restoreFocus(card, bad), false, bad);
  document.body.replaceChildren();
});

test('a picture that failed to load is fetched once however often its card is drawn again (P6 fix 1)', async (t) => {
  clearImageCache();
  t.after(clearImageCache);
  const fetched = [];
  t.mock.method(globalThis, 'fetch', async (url) => {
    fetched.push(String(url));
    return new Response(JSON.stringify({ type: 'error', error: { type: 'not_found_error', message: 'gone' } }), { status: 404 });
  });
  const view = gridView();   // the turn's two pictures and two similar X-rays
  const ctx = { loadImage: (path) => apiLoadImage(path, {}), ui: new Map() };
  let card = null;
  for (let paint = 0; paint < 10; paint++) {   // a streamed turn draws its card again at every frame
    card = renderAssistantCard(view, ctx);
    await tick();
    await tick();
  }
  assert.deepEqual([...fetched].sort(), [URLS.model_input, URLS.thumb, '/v1/gallery/images/101', '/v1/gallery/images/102'].sort());
  assert.ok(qa(card, 'img').every((img) => img.hasAttribute('data-failed') && !img.hasAttribute('src')));   // each shows it failed, at once
  clearImageCache();                           // another chat, or a new token: asked again, once
  renderAssistantCard(view, ctx);
  await tick();
  await tick();
  assert.equal(fetched.length, 8);
});

test('a test study\'s card ranks its own report: "Own report: rank 3 of 2,663 test reports · R@10 hit", the dedup rank and protocol in the tooltip', () => {
  const badge = (rank, detail = {}) => q(renderMatches(matchesView({ true_report_rank: rank, ...detail }), {}), '.own-rank');
  const hit = badge(OWN_RANK);
  assert.equal(visibleText(hit), 'Own report: rank 3 of 2,663 test reports · R@10 hit');
  const tip = 'Dedup-aware rank 2: every identical report counted once. 1 other report scores exactly the same. '
            + 'Protocol: i2t, official test split, strict pairing.';
  assert.equal(hit.getAttribute('title'), tip);
  assert.equal(q(hit, '.visually-hidden').textContent, ` ${tip}`);   // what a pointer shows, a screen reader hears
  assert.equal(visibleText(badge({ ...OWN_RANK, rank: 14, hit_at_10: false, n_tied: 0 })), 'Own report: rank 14 of 2,663 test reports · R@10 miss');
  assert.equal(badge({ ...OWN_RANK, n_tied: 3 }).getAttribute('title').includes('3 other reports score exactly the same.'), true);
  const alone = renderMatches(matchesView({ true_report_rank: OWN_RANK, report_matches: [] }), {});   // k_reports 0: the badge alone
  assert.equal(alone.hasAttribute('hidden'), false);
  assert.deepEqual([q(alone, '.own-rank') !== null, q(alone, 'ol')], [true, null]);
  for (const odd of [{ rank: '3', of: 2663 }, { rank: 3 }, { of: 40 }, null, 'x']) assert.equal(badge(odd), null, JSON.stringify(odd));
});

test('public mode: the matching reports keep rank and similarity only, with a note; no text, no labels, no group, no badge (U2)', () => {
  const pubStart = imageStart({ urls: { thumb: URLS.thumb, model_input: URLS.model_input } }, { mode: 'public' });
  const pub = viewOf([pubStart, stageStart('retrieve', 2), stageEnd('retrieve', 3, {
    image_neighbors: [], report_matches: [{ rank: 1, similarity: 0.41 }, { rank: 2, similarity: 0.4 }], gallery: { images: 200 } })]);
  const section = renderMatches(pub, { labelNames: LABEL_NAMES });
  assert.deepEqual(matchItems(section).map((li) => q(li, '.match-score').textContent), ['#1 · 0.410', '#2 · 0.400']);
  assert.equal(q(section, '.section-note').textContent, 'Public mode shows rank and similarity only.');
  assert.deepEqual([qa(section, '.clamp-text').length, qa(section, '.chip').length, qa(section, 'button').length, q(section, '.own-rank')],
                   [0, 0, 0, null]);
  const leaked = viewOf([pubStart, stageStart('retrieve', 2), stageEnd('retrieve', 3, { report_matches: [match(1)] })]);   // defence in depth
  assert.deepEqual([qa(renderMatches(leaked, {}), '.clamp-text').length, qa(renderMatches(leaked, {}), '.chip').length], [0, 0]);
  assertClean(section, 'public matches');
});

test('a test study\'s published lines: the GPU report with identical or differs against the live one, and the retrieval floor (P6-D)', () => {
  const scored = (published) => viewOf([imageStart({ source: 'test_split', filename: null }), stageStart('generate', 3), snapshot('Findings: live.'),
                                        stageEnd('generate', 9, GENERATE), stageStart('score', 5),
                                        stageEnd('score', 1, { ...SCORE, reference_source: 'test_split', published }),
                                        stopOf('done', { report: 'Findings: live.', display_report: 'Findings: live.' })]);
  const lines = (section) => qa(section, 'li.published-line');
  const same = renderPublished(scored({ model_report: LONG_REPORT, floor_report: 'Findings: floor.', live_equals_published: true }), { turn: 4 });
  assert.equal(same.hasAttribute('hidden'), false);
  assert.equal(same.getAttribute('aria-label'), 'Published reports, turn 4');
  assert.equal(q(same, 'h3').textContent, 'Published reports (test split)');
  assert.deepEqual(lines(same).map((li) => li.getAttribute('data-kind')), ['model', 'floor']);
  assert.deepEqual(lines(same).map((li) => q(li, '.published-head').textContent), ['Published (GPU) report · identical', 'Retrieval-floor report']);
  assert.ok(q(same, '.verdict').classList.contains('same'));
  assert.deepEqual(lines(same).map((li) => q(li, '.clamp-text').textContent), [LONG_REPORT, 'Findings: floor.']);
  assert.equal(q(lines(same)[0], 'button[data-action="show"]').getAttribute('aria-label'), 'Show all of the published report, turn 4');
  assert.equal(q(same, 'p.note'), null);   // compared: nothing to explain
  const differs = renderPublished(scored({ model_report: 'Findings: other.', floor_report: 'Findings: floor.', live_equals_published: false }), {});
  assert.equal(q(differs, '.published-head').textContent, 'Published (GPU) report · differs');
  assert.ok(q(differs, '.verdict').classList.contains('differs'));
  const unsaid = renderPublished(scored({ model_report: 'Findings: other.', floor_report: 'Findings: floor.', live_equals_published: null }), {});
  assert.equal(q(unsaid, '.published-head').textContent, 'Published (GPU) report');   // not compared: no verdict, and the note says when it is
  assert.equal(q(unsaid, 'p.note').textContent,
               'Compared with the live report only when the turn is decoded as the published dump was: beam 3, 100 tokens, the whole budget, not compiled.');
  const floorOnly = renderPublished(scored({ model_report: null, floor_report: 'Findings: floor.', live_equals_published: null }), {});
  assert.deepEqual([lines(floorOnly).map((li) => li.getAttribute('data-kind')), q(floorOnly, 'p.note')], [['floor'], null]);   // the dump is another model's
  assert.equal(renderPublished(scored(undefined), {}).hasAttribute('hidden'), true);   // an upload's score has no published lines
  assert.equal(renderPublished(finished(), {}).hasAttribute('hidden'), true);
  assertClean(renderPublished(scored({ model_report: 7, floor_report: null, live_equals_published: 'yes' }), {}), 'odd published');
});

test('the card: report, labels, the published lines, then the images, the similar X-rays and the matching reports, then provenance', () => {
  const { loadImage } = loader();
  const view = viewOf([imageStart({ source: 'test_split', filename: null }), PREPROCESSED, stageStart('retrieve', 2),
                       stageEnd('retrieve', 3, { image_neighbors: [neighbour(1)], report_matches: [match(1)], true_report_rank: OWN_RANK, gallery: {} }),
                       stageStart('generate', 3), snapshot('Findings: live.'), stageEnd('generate', 9, GENERATE),
                       stageStart('label', 4), stageEnd('label', 3, { chexbert_14: labelled([]), positives: [], neighbor_agreement: [agreementOf(1)] }),
                       stageStart('score', 5), stageEnd('score', 1, { ...SCORE, published: { model_report: 'a', floor_report: 'b', live_equals_published: true } }),
                       stopOf('done', { report: 'Findings: live.', display_report: 'Findings: live.' })]);
  const card = renderAssistantCard(view, { loadImage, labelNames: LABEL_NAMES });
  assert.deepEqual(card.children.map((c) => c.getAttribute('class')),
                   ['timeline', 'notes', 'report', 'labels', 'published', 'images', 'neighbors', 'matches', 'provenance']);
  assert.equal(qa(card, 'li.chip.label').length, 14);   // the report's own chips, and only those, carry .label
});

// ---- UI state that outlives a re-render (ctx.ui) ----------------------------------------------------------------------

test('with ctx.ui the open stages and the raw toggle survive the whole-card replace; without it they start over', () => {
  const view = finished({ report: 'Findings: a. Impression: b.', display: 'Findings: A. Impression: B.' });
  const ui = new Map();
  const ctx = { ui, copy() {} };
  let card = renderAssistantCard(view, ctx);
  stageButton(card, 'encode').click();
  stageButton(card, 'generate').click();
  stageButton(card, 'generate').click();   // opened and closed again
  button(card, 'Show raw').click();

  card = renderAssistantCard(view, ctx);   // the next throttled frame
  const open = (stage) => [stageItem(card, stage).classList.contains('open'), stageButton(card, stage).getAttribute('aria-expanded')];
  assert.deepEqual(open('encode'), [true, 'true']);   // the class and the state the button announces agree
  assert.deepEqual(open('generate'), [false, 'false']);
  assert.deepEqual(open('preprocess'), [false, 'false']);
  assert.equal(q(card, 'pre').textContent, 'Findings: a. Impression: b.');
  assert.equal(button(card, 'Show raw').getAttribute('aria-pressed'), 'true');

  button(card, 'Show raw').click();
  stageButton(card, 'encode').click();
  card = renderAssistantCard(view, ctx);
  assert.deepEqual(open('encode'), [false, 'false']);
  assert.equal(q(card, 'pre'), null);

  stageButton(card, 'encode').click();
  const other = renderAssistantCard({ ...view, id: 'm_other' }, ctx);   // another message has its own state
  assert.equal(stageItem(other, 'encode').classList.contains('open'), false);
  const bare = renderAssistantCard(view, { copy() {} });   // no store: nothing remembered
  assert.equal(stageItem(bare, 'encode').classList.contains('open'), false);
});

test('with ctx.ui an expanded "show more" survives the whole-card replace, and belongs to its stage', () => {
  const view = withClips();   // retrieve has three clipped strings (0, 1, 2), generate one (its own 0)
  const ui = new Map();
  const ctx = { ui, copy() {} };
  const moreOf = (card) => qa(card, 'button.more');
  const state = (card) => moreOf(card).map((b) => [b.textContent, b.getAttribute('aria-expanded')].join('/'));
  const shown = (card) => qa(card, '.clip-text').map((n) => n.textContent.length);
  let card = renderAssistantCard(view, ctx);
  assert.deepEqual(state(card), ['show more/false', 'show more/false', 'show more/false', 'show more/false']);
  assert.deepEqual(shown(card), [201, 201, 201, 201]);   // 200 characters and the ellipsis

  moreOf(card)[1].click();   // retrieve's second string
  moreOf(card)[3].click();   // and generate's only one
  card = renderAssistantCard(view, ctx);   // the next frame of a streaming turn rebuilds the whole card
  assert.deepEqual(state(card), ['show more/false', 'show less/true', 'show more/false', 'show less/true']);
  assert.deepEqual(shown(card), [201, 300, 201, 300]);   // the strings are shown in full again
  assert.deepEqual(moreOf(card).map((b) => b.getAttribute('aria-label')), [
    'show more of retrieve report_matches 1 report', 'show less of retrieve report_matches 2 report',
    'show more of retrieve note', 'show less of generate drift_note']);   // and the names say what the buttons now do

  moreOf(card)[1].click();   // collapse one of them: the next frame keeps that too
  card = renderAssistantCard(view, ctx);
  assert.deepEqual(state(card), ['show more/false', 'show more/false', 'show more/false', 'show less/true']);
  moreOf(card)[3].click();
  card = renderAssistantCard(view, ctx);
  assert.deepEqual(state(card), ['show more/false', 'show more/false', 'show more/false', 'show more/false']);

  // The stage is part of the key: generate's number 0 must not open retrieve's number 0, nor the reverse.
  const generateOnly = new Map();
  let g = renderAssistantCard(view, { ui: generateOnly });
  moreOf(g)[3].click();
  g = renderAssistantCard(view, { ui: generateOnly });
  assert.deepEqual(state(g), ['show more/false', 'show more/false', 'show more/false', 'show less/true']);
  const retrieveOnly = new Map();
  let r = renderAssistantCard(view, { ui: retrieveOnly });
  moreOf(r)[0].click();
  r = renderAssistantCard(view, { ui: retrieveOnly });
  assert.deepEqual(state(r), ['show less/true', 'show more/false', 'show more/false', 'show more/false']);

  const other = renderAssistantCard({ ...view, id: 'm_other' }, { ui: generateOnly });   // another message has its own state
  assert.deepEqual(state(other), ['show more/false', 'show more/false', 'show more/false', 'show more/false']);
  moreOf(g)[3].click();   // and a click after the rebuild still works, in both directions
  assert.deepEqual(state(g), ['show more/false', 'show more/false', 'show more/false', 'show more/false']);
  const bare = renderAssistantCard(view, { copy() {} });   // no store: nothing remembered
  moreOf(bare)[0].click();
  assert.deepEqual(state(renderAssistantCard(view, { copy() {} })), ['show more/false', 'show more/false', 'show more/false', 'show more/false']);
});

test('detailTable keeps no show-more state of its own: without a store it starts clipped, and a store it is handed is the one it writes to', () => {
  const long = 'q'.repeat(300);
  const table = detailTable({ a: long, b: long }, 'x details', { name: 'encode' });
  const [first] = qa(table, 'button.more');
  first.click();
  assert.deepEqual(qa(detailTable({ a: long, b: long }, 'x details', { name: 'encode' }), 'button.more').map((b) => b.textContent), ['show more', 'show more']);
  const store = new Set();
  const withStore = detailTable({ a: long, b: long }, 'x details', { name: 'encode', more: store });
  qa(withStore, 'button.more')[1].click();
  assert.deepEqual([...store], ['encode:1']);   // the stage, then the string's number in the table
  assert.deepEqual(qa(detailTable({ a: long, b: long }, 'x details', { name: 'encode', more: store }), 'button.more').map((b) => b.textContent),
                   ['show more', 'show less']);
  qa(withStore, 'button.more')[1].click();
  assert.deepEqual([...store], []);
});

test('Copy is not offered while the report is provisional, so a "Copy failed" on its button cannot be lost to a mid-stream rebuild', () => {
  const streaming = viewOf([START, stageStart('generate', 3), snapshot('Findings: so far')]);   // a snapshot: the best beam so far
  assert.equal(streaming.provisional, true);
  const card = renderAssistantCard(streaming, { copy() {}, showModels() {} });
  assert.equal(button(card, 'Copy'), null);
  assert.equal(qa(card, '[data-action="copy"]').length, 0);
  assert.ok(button(card, 'Show raw'), 'the raw view is still offered');
  assert.equal(q(card, '.report').getAttribute('class'), 'report provisional');
  assert.equal(button(renderReport(streaming, { copy() {} }), 'Copy'), null);   // the same for the report on its own
  const closed = viewOf([START, stageStart('generate', 3), snapshot('Findings: so far'), step('content_block_stop', { index: 0 })]);
  assert.equal(closed.provisional, false);   // the report is final once its block closes, before the labels and scores are
  assert.ok(button(renderAssistantCard(closed, { copy() {} }), 'Copy'));
  assert.ok(button(renderAssistantCard(finished(), { copy() {} }), 'Copy'));
  assert.equal(button(renderAssistantCard(finished(), {}), 'Copy'), null);   // and still none without a ctx.copy
});

// ---- focus hooks for the whole-card replace (M2) ---------------------------------------------------------------------

test('Copy, Show raw, model details and show more carry a stable data-action, and a stage button its stage', () => {
  const card = renderAssistantCard(withClips(), { copy() {}, showModels() {} });
  assert.equal(button(card, 'Copy').getAttribute('data-action'), 'copy');
  assert.equal(button(card, 'Show raw').getAttribute('data-action'), 'raw');
  assert.equal(button(card, 'model details').getAttribute('data-action'), 'models');
  const more = qa(card, 'button.more');
  assert.equal(more.length, 4);   // three in retrieve, one in generate
  assert.ok(more.every((b) => b.getAttribute('data-action') === 'more'));
  assert.deepEqual(qa(card, 'li > button').map((b) => [b.getAttribute('data-action'), b.getAttribute('data-stage')]),
                   [['stage', 'retrieve'], ['stage', 'generate']]);   // only the stages that have a detail
});

test('focusKey names the focused control and restoreFocus puts focus on its twin in the rebuilt card', () => {
  const view = withClips();
  const ctx = { copy() {}, showModels() {}, ui: new Map() };
  let card = renderAssistantCard(view, ctx);
  document.body.replaceChildren(card);
  stageButton(card, 'retrieve').click();   // open, so that their show-more buttons are on screen and can take focus
  stageButton(card, 'generate').click();
  const controls = (c) => ({
    'stage:retrieve': stageButton(c, 'retrieve'), 'stage:generate': stageButton(c, 'generate'),
    copy: button(c, 'Copy'), raw: button(c, 'Show raw'), models: button(c, 'model details'),
    'more:retrieve:0': qa(c, 'button.more')[0], 'more:retrieve:2': qa(c, 'button.more')[2],
    'more:generate:0': qa(c, 'button.more')[3],   // the second stage's first string is its number 0, not 3
  });
  for (const key of Object.keys(controls(card))) {
    const control = controls(card)[key];
    control.focus();
    assert.equal(document.activeElement, control, key);
    assert.equal(focusKey(document.activeElement), key);
    const next = renderAssistantCard(view, ctx);   // the next frame: the card is rebuilt and replaces the old one
    card.replaceWith(next);
    card = next;
    assert.equal(document.activeElement, document.body, `${key}: the replace takes focus away`);
    assert.equal(restoreFocus(card, key), true, key);
    assert.equal(document.activeElement, controls(card)[key], `${key}: focus is on the same control of the new card`);
    assert.equal(focusKey(document.activeElement), key);
  }
  document.body.replaceChildren();
});

test('focusKey reads the control from anything inside it; restoreFocus gives up quietly when it cannot restore', () => {
  const card = renderAssistantCard(finished(), {});   // no copy callback and no models link
  document.body.replaceChildren(card);
  assert.equal(focusKey(null), null);
  assert.equal(focusKey(document.body), null);
  assert.equal(focusKey(q(card, 'p, .note') ?? q(card, 'ol')), null);   // not inside a control
  const raw = button(card, 'Show raw');
  assert.equal(focusKey(raw), 'raw');
  raw.append(el('span', {}, 'inner'));
  assert.equal(focusKey(q(raw, 'span')), 'raw');   // a click or focus can land on a child of the button
  for (const key of [null, undefined, '', 'copy', 'models', 'stage:retrieve', 'stage:nope', 'more:encode:0', 'more:retrieve:0',
                     'stage:"] x', 'x:y:z', 'copy:extra', 'raw:1', 'more:encode:-1', 'more:encode:x']) {
    assert.equal(restoreFocus(card, key), false, String(key));   // absent, unknown or malformed: false, and no exception
  }
  assert.equal(document.activeElement, document.body);
  assert.equal(restoreFocus(card, 'raw'), true);
  assert.equal(document.activeElement, raw);
  document.body.replaceChildren();
});

test('restoreFocus says whether focus really moved, and never scrolls the pane to do it', () => {
  const card = renderAssistantCard(finished(), {});
  const raw = button(card, 'Show raw');
  const asked = [];
  const focus = raw.focus.bind(raw);
  raw.focus = (options) => { asked.push(options); focus(options); };   // what the page would be told to do
  assert.equal(restoreFocus(card, 'raw'), false);   // the card is not in the page: the control exists but cannot take focus
  assert.equal(document.activeElement, document.body);
  document.body.replaceChildren(card);
  assert.equal(restoreFocus(card, 'raw'), true);
  assert.equal(document.activeElement, raw);
  assert.deepEqual(asked, [{ preventScroll: true }, { preventScroll: true }]);   // a frame that rebuilds the card must not jump the pane
  document.body.replaceChildren();
});

test('the shim gives focus to nothing inside a hidden subtree, and takes it away from what becomes hidden', () => {
  const inner = el('button', { type: 'button' }, 'inner');
  const section = el('section', {}, inner);
  const wrap = el('div', {}, section);
  document.body.replaceChildren(wrap);
  inner.focus();
  assert.equal(document.activeElement, inner);   // shown: it takes focus
  section.setAttribute('hidden', '');   // a browser blurs what it hides
  assert.equal(document.activeElement, document.body);
  section.removeAttribute('hidden');
  assert.equal(document.activeElement, document.body);   // and does not give it back when the subtree is shown again

  for (const hide of [wrap, section]) {   // hidden on the element's own parent or further up: refused either way
    hide.setAttribute('hidden', '');
    inner.focus();
    assert.equal(document.activeElement, document.body, `${hide.localName} is hidden`);
    hide.removeAttribute('hidden');
  }
  inner.setAttribute('hidden', '');   // the control itself
  inner.focus();
  assert.equal(document.activeElement, document.body);
  inner.removeAttribute('hidden');
  inner.focus();
  assert.equal(document.activeElement, inner);   // and takes it once nothing above it is hidden
  document.body.replaceChildren();
});

test('the shim fires focus and blur as focus moves, selects a field\'s whole value, and submits a form as a browser does (P4-G)', () => {
  const form = el('form', {}, el('input', { type: 'number', id: 'n' }), el('input', { type: 'text', id: 't' }), el('textarea', { id: 'a' }),
                  el('button', { type: 'button', id: 'plain' }, 'plain'), el('button', { type: 'submit', id: 'go' }, 'Go'),
                  el('button', { id: 'bare' }, 'no type'));
  document.body.replaceChildren(form);
  const [n, t, area, plain, go, bare] = ['n', 't', 'a', 'plain', 'go', 'bare'].map((id) => document.getElementById(id));
  const seen = [];
  for (const node of [n, t]) {
    for (const type of ['focus', 'blur']) node.addEventListener(type, (event) => seen.push(`${type} ${node.id} bubbles=${event.bubbles}`));
  }
  n.focus();
  t.focus();   // the field that had focus is blurred first
  t.blur();
  n.focus();
  n.focus();   // already there: nothing fires again
  assert.deepEqual(seen, ['focus n bubbles=false', 'blur n bubbles=false', 'focus t bubbles=false', 'blur t bubbles=false', 'focus n bubbles=false']);

  n.value = '150';
  n.select();   // the whole value, as a click or a Tab into the field leaves it
  assert.deepEqual([n.selectionStart, n.selectionEnd], [0, 3]);
  n.value = '';
  n.select();
  assert.deepEqual([n.selectionStart, n.selectionEnd], [0, 0]);   // nothing typed yet is nothing selected

  const submits = [];
  form.addEventListener('submit', (event) => { submits.push(event.target.localName); event.preventDefault(); });
  plain.click();
  assert.deepEqual(submits, []);               // a button of type button submits nothing
  go.click();
  bare.click();                                // a button with no type is a submit button
  assert.deepEqual(submits, ['form', 'form']);
  go.setAttribute('disabled', '');
  go.click();
  assert.equal(submits.length, 2);             // a disabled one does not
  go.removeAttribute('disabled');

  const enter = (target, init = {}) => { const e = new ShimEvent('keydown', { key: 'Enter', bubbles: true, cancelable: true, ...init }); target.dispatchEvent(e); return e; };
  enter(n);
  enter(t);
  assert.equal(submits.length, 4);             // Enter in a text-like field presses the form's default button, which submits
  enter(area);
  enter(plain);
  assert.equal(submits.length, 4);             // in a textarea it is a newline, and on a button it is that button's own press
  const seenKey = [];
  form.addEventListener('keydown', (event) => { seenKey.push(event.key); event.preventDefault(); });
  enter(n);
  assert.deepEqual([submits.length, seenKey], [4, ['Enter']]);   // a handler that took the key leaves the form alone
  document.body.replaceChildren();
});

test('a hidden footer keeps its "model details" button out of reach: it cannot take focus, and restoreFocus says so', () => {
  const view = viewOf([START, stopOf('done')]);   // finished and no stage run: the card hides its timeline and its footer
  const card = renderAssistantCard(view, { showModels() {} });
  document.body.replaceChildren(card);
  const footer = q(card, 'footer.provenance');
  assert.ok(footer.hasAttribute('hidden'));
  const models = button(card, 'model details');
  assert.ok(models && footer.contains(models), 'the button is built, inside the hidden footer');
  models.focus();
  assert.equal(document.activeElement, document.body);
  assert.equal(restoreFocus(card, 'models'), false);   // the control exists but cannot take focus, so the page is told it failed
  assert.equal(document.activeElement, document.body);
  footer.removeAttribute('hidden');
  assert.equal(restoreFocus(card, 'models'), true);   // the same call works once the footer is shown
  assert.equal(document.activeElement, models);
  document.body.replaceChildren();
});

// ---- names of repeated controls (M3) ----------------------------------------------------------------------------------

test('repeated controls have distinct names that contain their visible text, and the turn number tells the cards apart', () => {
  const view = withClips();
  const ctxOf = (turn) => ({ copy() {}, showModels() {}, openViewer() {}, ...(turn ? { turn } : {}) });
  const three = renderAssistantCard(view, ctxOf(3));
  const nameOf = (card, label) => button(card, label).getAttribute('aria-label');
  assert.equal(three.getAttribute('aria-label'), 'Assistant report, turn 3');
  assert.equal(nameOf(three, 'Copy'), 'Copy report, turn 3');
  assert.equal(nameOf(three, 'Show raw'), 'Show raw report, turn 3');
  assert.equal(nameOf(three, 'model details'), 'model details, turn 3');
  assert.deepEqual(qa(three, 'button.more').map((b) => b.getAttribute('aria-label')), [
    'show more of retrieve report_matches 1 report, turn 3', 'show more of retrieve report_matches 2 report, turn 3',
    'show more of retrieve note, turn 3', 'show more of generate drift_note, turn 3',
  ]);   // the stage and the path to the string, so no two of them sound alike
  assert.equal(stageButton(three, 'generate').getAttribute('aria-label'), 'generate · 9 ms, done, turn 3');
  qa(three, 'button.more')[2].click();
  assert.equal(qa(three, 'button.more')[2].getAttribute('aria-label'), 'show less of retrieve note, turn 3');

  const unnumbered = renderAssistantCard(view, ctxOf());
  assert.equal(unnumbered.getAttribute('aria-label'), 'Assistant report');
  assert.deepEqual([nameOf(unnumbered, 'Copy'), nameOf(unnumbered, 'Show raw'), nameOf(unnumbered, 'model details')],
                   ['Copy report', 'Show raw report', 'model details']);
  assert.equal(stageButton(unnumbered, 'generate').getAttribute('aria-label'), 'generate · 9 ms, done');

  const names = (card) => qa(card, 'button').map((b) => b.getAttribute('aria-label'));
  const seven = renderAssistantCard(view, ctxOf(7));
  const [a, b] = [names(three), names(seven)];
  assert.equal(new Set(a).size, a.length);   // distinct within a card
  assert.deepEqual(a.filter((n) => b.includes(n)), []);   // and between two cards

  for (const control of qa(three, 'button[data-action]')) {   // the stage buttons too: their text is "encode · 612 ms", and so is the start of the name
    const [name, text] = [control.getAttribute('aria-label').toLowerCase(), control.textContent.toLowerCase()];
    assert.ok(name.includes(text), `"${control.getAttribute('aria-label')}" should contain "${control.textContent}"`);   // WCAG 2.5.3, label in name
    if (control.getAttribute('data-action') === 'stage') assert.ok(name.startsWith(text), `"${name}" should start with "${text}"`);
  }
  assert.deepEqual(qa(three, 'button[data-action="stage"]').map((b) => b.getAttribute('aria-label')),
                   ['retrieve · 8 ms, done, turn 3', 'generate · 9 ms, done, turn 3']);

  const user = renderUserTurn({ text: 'hi', image: { url: 'blob:x/1', filename: 'a.png' } }, ctxOf(3));
  assert.equal(user.getAttribute('aria-label'), 'Your message, turn 3');
  assert.equal(q(user, 'button.thumb-button').getAttribute('aria-label'), 'Open X-ray in the viewer, turn 3');
  const tile = q(renderImages(viewOf([imageStart(), PREPROCESSED]), { ...ctxOf(3), loadImage: loader().loadImage }), 'button[data-action="view"]');
  assert.notEqual(tile.getAttribute('aria-label'), q(user, 'button.thumb-button').getAttribute('aria-label'));   // P6-G: two openers of one X-ray, told apart by name
  const plainUser = renderUserTurn({ text: 'hi', image: { url: 'blob:x/1' } }, ctxOf());
  assert.equal(plainUser.getAttribute('aria-label'), 'Your message');
  assert.equal(q(plainUser, 'button.thumb-button').getAttribute('aria-label'), 'Open X-ray in the viewer');
});

test('a rejected ctx.copy shows "Copy failed" on the button for a moment, then "Copy" again; success changes nothing', async (t) => {
  const timers = [];
  const realSetTimeout = globalThis.setTimeout;
  globalThis.setTimeout = (fn, ms) => { timers.push({ fn, ms }); return timers.length; };   // a timer is run by hand
  const unhandled = [];
  const onUnhandled = (reason) => unhandled.push(reason);
  process.on('unhandledRejection', onUnhandled);
  t.after(() => { globalThis.setTimeout = realSetTimeout; process.off('unhandledRejection', onUnhandled); });

  const failures = [async () => { throw new Error('denied'); }, () => Promise.reject(new Error('denied')), () => { throw new Error('sync'); }];
  for (const copy of failures) {
    timers.length = 0;
    const copyButton = button(renderReport(finished(), { copy, turn: 2 }), 'Copy');
    copyButton.click();   // a throw from copy itself must not escape the click either
    await tick();
    assert.equal(copyButton.textContent, 'Copy failed');
    assert.equal(copyButton.getAttribute('aria-label'), 'Copy failed, turn 2');
    assert.deepEqual(timers.map((x) => x.ms), [2000]);
    timers[0].fn();   // the moment is over
    assert.equal(copyButton.textContent, 'Copy');
    assert.equal(copyButton.getAttribute('aria-label'), 'Copy report, turn 2');
  }
  for (const copy of [async () => {}, () => undefined, () => true, () => ({ then: undefined })]) {
    timers.length = 0;
    const copyButton = button(renderReport(finished(), { copy }), 'Copy');
    copyButton.click();
    await tick();
    assert.equal(copyButton.textContent, 'Copy');
    assert.equal(copyButton.getAttribute('aria-label'), 'Copy report');
    assert.deepEqual(timers, []);
  }
  await tick();
  assert.deepEqual(unhandled, []);
});

// ---- the throttle (ruling 9) -------------------------------------------------------------------------------------------

test('scheduleRender runs a function once on the next animation frame however often it is asked', () => {
  const frames = [];
  const real = globalThis.requestAnimationFrame;
  globalThis.requestAnimationFrame = (cb) => { frames.push(cb); return frames.length; };
  try {
    const seen = [];
    let latest = 0;
    const draw = () => seen.push(latest);
    for (let i = 1; i <= 50; i++) { latest = i; scheduleRender(draw); }   // fifty events in one frame
    assert.equal(frames.length, 1);
    assert.deepEqual(seen, []);   // not synchronously
    frames.shift()();
    assert.deepEqual(seen, [50]);   // once, with the latest state
    scheduleRender(draw);   // the next burst gets the next frame
    assert.equal(frames.length, 1);
    frames.shift()();
    assert.equal(seen.length, 2);
    const other = () => seen.push('other');   // another function has its own pending frame
    scheduleRender(draw);
    scheduleRender(other);
    assert.equal(frames.length, 2);
    const failing = () => { throw new Error('draw failed'); };
    scheduleRender(failing);
    assert.throws(() => frames.at(-1)(), /draw failed/);
    scheduleRender(failing);   // a throwing frame does not block the next
    assert.equal(frames.length, 4);
  } finally {
    globalThis.requestAnimationFrame = real;
  }
});

test('scheduleRender falls back to a timer where there is no requestAnimationFrame', async () => {
  const real = globalThis.requestAnimationFrame;
  delete globalThis.requestAnimationFrame;
  try {
    let runs = 0;
    const draw = () => { runs += 1; };
    scheduleRender(draw);
    scheduleRender(draw);
    await new Promise((resolve) => setTimeout(resolve, 60));
    assert.equal(runs, 1);
  } finally {
    if (real) globalThis.requestAnimationFrame = real;
  }
});

// ---- what app/static may contain (rulings 3 and 10, and M6) ---------------------------------------------------------------

const STATIC_PATH = fileURLToPath(new URL('../../app/static/', import.meta.url));
// Every way to turn a string into markup, or into a document of its own: none may appear in a script or page of the app.
const HTML_SINKS = /innerHTML|outerHTML|insertAdjacentHTML|document\s*\.\s*write|DOMParser|createContextualFragment|setHTMLUnsafe|parseHTMLUnsafe|srcdoc/;

// The .js, .mjs and .html files under dir, however deep: [path relative to dir, text].
const staticSources = (dir) => readdirSync(dir, { recursive: true })
  .filter((name) => /\.(?:m?js|html)$/.test(name))
  .map((name) => [name, readFileSync(join(dir, name), 'utf8')]);

test('no script or page in app/static uses an HTML-string API: report and user text can only ever be text', () => {
  const found = staticSources(STATIC_PATH);
  assert.ok(found.some(([name]) => name === 'render.js') && found.some(([name]) => name === 'index.html'), 'the scan reads render.js and the page');
  for (const [name, source] of found) assert.doesNotMatch(source, HTML_SINKS, name);
});

test('the HTML-string scan reads nested directories and knows every sink', () => {
  const dir = mkdtempSync(join(tmpdir(), 'static-scan-'));
  try {
    const sinks = ['x.innerHTML = a', 'x.outerHTML = a', 'x.insertAdjacentHTML("beforeend", a)', 'document.write(a)', 'document . write(a)',
                   'new DOMParser()', 'range.createContextualFragment(a)', 'el.setHTMLUnsafe(a)', 'Document.parseHTMLUnsafe(a)', 'frame.srcdoc = a'];
    mkdirSync(join(dir, 'deep', 'er'), { recursive: true });
    sinks.forEach((code, i) => writeFileSync(join(dir, i % 2 ? join('deep', 'er') : 'deep', `s${i}.js`), `${code};\n`));
    writeFileSync(join(dir, 'deep', 'page.html'), '<iframe srcdoc="x"></iframe>');
    writeFileSync(join(dir, 'clean.js'), 'export const a = 1;\n');
    writeFileSync(join(dir, 'deep', 'notes.txt'), 'innerHTML in a text file is not a script');
    const flagged = staticSources(dir).filter(([, source]) => HTML_SINKS.test(source)).map(([name]) => name);
    assert.equal(flagged.length, sinks.length + 1);   // every sink file, at two depths, and the page; not clean.js, not the .txt
    assert.ok(!flagged.includes('clean.js') && !flagged.some((n) => n.endsWith('.txt')));
    for (const [i, code] of sinks.entries()) assert.ok(HTML_SINKS.test(code), code);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('app/static has no EventSource and no absolute URL, and imports only its own files', () => {
  const scripts = staticSources(STATIC_PATH).filter(([name]) => name.endsWith('.js'));
  assert.ok(scripts.length >= 3);   // api, state, render, and whatever the page adds
  for (const [name, source] of scripts) {
    assert.doesNotMatch(source, /EventSource/, name);
    assert.doesNotMatch(source, /https?:\/\//, name);
    const specifiers = [
      ...source.matchAll(/\b(?:import|export)\b[^;'"]*?\bfrom\s*['"]([^'"]+)['"]/g),   // import { a, b } from '...', across lines too
      ...source.matchAll(/\bimport\s*\(\s*['"]([^'"]+)['"]/g),                          // import('...')
      ...source.matchAll(/(?:^|\n)\s*import\s*['"]([^'"]+)['"]/g),                       // import '...'
    ].map((m) => m[1]);
    for (const specifier of specifiers) {
      assert.match(specifier, /^\.\/[\w-]+\.js$/, `${name} imports ${specifier}`);   // browser-native modules, no packages
    }
  }
});
