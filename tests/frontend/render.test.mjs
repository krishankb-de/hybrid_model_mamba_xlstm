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
import { initialView, replay } from '../../app/static/state.js';
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
  assert.deepEqual(sections.map((s) => [q(s, 'h3').textContent, q(s, 'p').textContent]),
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
  assert.equal(q(report, '.report-section p').textContent, 'The lungs are clear.');
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
  assert.equal(q(renderReport(finished({ generate: eos })), '.report-body p').textContent, 'The lungs are clear.');
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
  assert.deepEqual(texts(qa(report, '.report-section p')), ['A lung is clear.', 'None.']);   // the display copy, in sections
  assert.equal(q(report, 'pre'), null);
  const raw = button(report, 'Show raw');
  assert.equal(raw.getAttribute('aria-pressed'), 'false');
  raw.click();
  assert.equal(q(report, 'pre').textContent, 'Findings: a lung  clear Impression: none and');   // verbatim, unsplit
  assert.equal(q(report, '.report-section'), null);
  assert.equal(raw.getAttribute('aria-pressed'), 'true');
  assert.equal(raw.textContent, 'Show raw');   // the label stays; aria-pressed carries the state
  raw.click();
  assert.deepEqual(texts(qa(report, '.report-section p')), ['A lung is clear.', 'None.']);
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

test('while the gallery is still being labelled the chips come with an "agreement pending" note (P5-E fix 1), in either mode', () => {
  const live = [START, stageStart('generate', 3), snapshot('Findings: so far'), stageEnd('generate', 9, GENERATE), stageStart('label', 4)];
  const detail = { chexbert_14: labelled(['Edema']), positives: ['Edema'], neighbor_agreement: [], neighbor_agreement_pending: true };
  for (const start of [START, step('message_start', { ...START.data, mode: 'public' })]) {
    const labels = renderLabels(viewOf([start, ...live.slice(1), stageEnd('label', 3, detail), stopOf('done')]), { labelNames: LABEL_NAMES });
    assert.equal(qa(labels, '.chip').length, 14);                         // the report is labelled all the same
    assert.deepEqual(texts(qa(labels, '.note')), ['agreement pending: the gallery is still being labelled']);
    assert.equal(labels.children.at(-1).textContent, 'agreement pending: the gallery is still being labelled');   // after the chips
  }
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
const SECTION_BUILDERS = ['renderImages'];
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
  assert.equal(q(card, '.report-section p').textContent, view.report);
  assert.equal(q(card, '.report .note').textContent, 'Report stopped at the token budget mid-sentence');
  assert.equal(q(card, '.labels').textContent, 'labels unavailable (labeler_unavailable)');
  assert.equal(q(card, '.provenance p').textContent.startsWith('tiny · scan legacy/tfla exact · prefix_k 4'), true);
  assert.equal(button(card, 'Copy') !== null && button(card, 'Show raw') !== null && button(card, 'model details') !== null, true);
  assert.deepEqual(['ol.timeline', '.notes', '.report', '.labels', '.provenance'].map((s) => card.children.indexOf(q(card, s))), [0, 1, 2, 3, 4]);
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
  assert.equal(q(card, '.report-section p').textContent, 'part');   // what had been generated stays
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

test('the thumbnail opens the viewer when the page can; without ctx.openViewer it is a picture only', () => {
  const image = { url: 'blob:x/1', filename: 'a.png' };
  const opened = [];
  const withViewer = renderUserTurn({ image }, { openViewer: (i) => opened.push(i) });
  const open = q(withViewer, 'button.thumb-button');
  assert.ok(open);
  assert.equal(open.getAttribute('aria-label'), 'Open X-ray in the viewer');
  assert.ok(q(open, 'img'));
  open.click();
  assert.deepEqual(opened, [image]);
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

test('with ctx.openViewer each picture is a button that opens it: your X-ray at full size, what the model saw at 224 (the hook P6-E fills)', () => {
  const { loadImage } = loader();
  const opened = [];
  const ctx = { loadImage, openViewer: (spec) => opened.push(spec), turn: 2 };
  const section = renderImages(viewOf([imageStart(), PREPROCESSED]), ctx);
  const buttons = qa(section, 'button[data-action="view"]');
  assert.deepEqual(buttons.map((b) => b.getAttribute('aria-label')), ['Open your X-ray in the viewer, turn 2', 'Open what the model saw in the viewer, turn 2']);
  assert.ok(buttons.every((b) => q(b, 'img') && b.getAttribute('type') === 'button'));
  buttons.forEach((b) => b.click());
  assert.deepEqual(opened, [{ images: [{ url: URLS.original, label: 'Your X-ray · 2544×3056 px' }] },
                            { images: [{ url: URLS.model_input, label: 'What the model saw (224×224)' }] }]);
  opened.length = 0;
  const pub = renderImages(viewOf([imageStart({ urls: { thumb: URLS.thumb, model_input: URLS.model_input } })]), ctx);
  q(pub, 'button').click();
  assert.deepEqual(opened, [{ images: [{ url: URLS.thumb, label: 'Your X-ray' }] }]);   // no original in public mode: the thumbnail
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
  assert.equal(focusKey(document.activeElement), 'view:1');
  const next = renderAssistantCard(view, ctx);
  card.replaceWith(next);
  card = next;
  assert.equal(restoreFocus(card, 'view:1'), true);
  assert.equal(document.activeElement, qa(card, 'button[data-action="view"]')[1]);
  assert.equal(restoreFocus(card, 'view:7'), false);   // no such button: false, quietly
  document.body.replaceChildren();
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
