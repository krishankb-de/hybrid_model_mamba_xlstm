// tests/frontend/viewer.test.mjs — app/static/viewer.js (CHAT_UI_PLAN.md P6-E): the view's arithmetic, the footer and the keys, and the viewer
// on the DOM shim: what it draws, how a picture is loaded (never a server path as an <img src>), zoom and pan by button, key, wheel, drag, pinch
// and double-click, the panes linked and not, brightness and contrast, the focus trap, the page behind made inert, and the close.
//
// No server and no MIMIC data: pictures are paths answered by a fake loader with made-up object URLs.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ShimEvent, installDom } from './dom_shim.mjs';
import {
  MAX_SCALE, MIN_SCALE, PAN_STEP, ZOOM_STEP, centred, clamp, clampPan, filterCss, fitScale, footerText, keyAction, openViewer, recentred,
  wheelFactor, zoomAt,
} from '../../app/static/viewer.js';
import { el } from '../../app/static/render.js';

installDom();

const near = (a, b, eps = 1e-9) => Math.abs(a - b) <= eps;

// ---- the pure part ------------------------------------------------------------------------------------------------------------------------

test('zoomAt keeps the image point under the cursor where it is, and holds the scale to [0.25, 16]', () => {
  const view = { scale: 1, x: 10, y: 20 };
  const under = (v, cx, cy) => [(cx - v.x) / v.scale, (cy - v.y) / v.scale];   // the image point at (cx, cy)
  for (const [factor, cx, cy] of [[2, 100, 50], [0.5, 30, 300], [1.25, 0, 0], [3, 512.5, 99]]) {
    const next = zoomAt(view, factor, cx, cy);
    assert.equal(next.scale, factor);
    const [a, b] = under(view, cx, cy);
    const [c, d] = under(next, cx, cy);
    assert.ok(near(a, c) && near(b, d), `${factor} at ${cx},${cy}`);
  }
  assert.deepEqual([MIN_SCALE, MAX_SCALE], [0.25, 16]);
  assert.equal(zoomAt(view, 100, 0, 0).scale, MAX_SCALE);
  assert.equal(zoomAt(view, 0.001, 0, 0).scale, MIN_SCALE);
  const clampedIn = zoomAt(view, 100, 50, 60);   // held to the bound, the point under the cursor still stays
  const [a, b] = under(view, 50, 60);
  const [c, d] = under(clampedIn, 50, 60);
  assert.ok(near(a, c) && near(b, d));
  assert.deepEqual(zoomAt({ scale: 16, x: 5, y: 5 }, 2, 40, 40), { scale: 16, x: 5, y: 5 });   // at the bound already: nothing moves
  assert.equal(zoomAt({ ...view, extra: 'kept' }, 2, 0, 0).extra, 'kept');
});

test('clampPan keeps at least a tenth of the image inside the pane, on every side', () => {
  const view = { scale: 2, x: 0, y: 0 };   // a 100×50 picture at scale 2 is 200×100 in a 300×200 pane
  const far = clampPan({ ...view, x: -1000, y: -1000 }, 100, 50, 300, 200);
  assert.deepEqual([far.x, far.y], [-180, -90]);   // 20 px and 10 px of it are still in the pane: a tenth
  const out = clampPan({ ...view, x: 1000, y: 1000 }, 100, 50, 300, 200);
  assert.deepEqual([out.x, out.y], [280, 190]);
  assert.deepEqual(clampPan({ ...view, x: 30, y: 40 }, 100, 50, 300, 200), { scale: 2, x: 30, y: 40 });   // in bounds: as it was
  const shown = (v, size, box, axis) => Math.min(v[axis] + size, box) - Math.max(v[axis], 0);
  for (const v of [far, out]) assert.ok(shown(v, 200, 300, 'x') >= 20 - 1e-9 && shown(v, 100, 200, 'y') >= 10 - 1e-9);
});

test('filterCss is the CSS filter of the brightness and the contrast, each 100% unless set', () => {
  assert.equal(filterCss({}), 'brightness(100%) contrast(100%)');
  assert.equal(filterCss({ brightness: 150, contrast: 40 }), 'brightness(150%) contrast(40%)');
  assert.equal(filterCss({ brightness: 0, contrast: 300 }), 'brightness(0%) contrast(300%)');
});

test('fitScale, centred and recentred: a picture whole in its pane, centred, and kept at its centre when the pane changes size', () => {
  assert.equal(fitScale(2544, 3056, 400, 300), 300 / 3056);   // a tall X-ray: the height decides
  assert.equal(fitScale(224, 224, 600, 400), 400 / 224);      // a small picture is drawn larger
  for (const none of [[0, 10, 10, 10], [10, 10, 0, 10], [NaN, 1, 1, 1]]) assert.equal(fitScale(...none), 1);
  assert.deepEqual(centred(1, 200, 100, 300, 200), { scale: 1, x: 50, y: 50 });
  assert.deepEqual(centred(99, 10, 10, 100, 100), { scale: 16, x: -30, y: -30 });   // held to the bounds
  assert.equal(clamp(5, 0, 3), 3);
  const view = { scale: 2, x: -50, y: -20 };   // the pane's centre (150, 100) shows the picture point (0.5, 0.6) of a 200×100 base
  const moved = recentred(view, [200, 100], [300, 200], [400, 200], [600, 400]);   // twice the pane, twice the base
  assert.ok(near((300 - moved.x) / (400 * 2), (150 - view.x) / (200 * 2)) && near((200 - moved.y) / (200 * 2), (100 - view.y) / (100 * 2)));
  assert.equal(moved.scale, 2);
});

test('the footer says the label and the pixel size of the loaded picture, with its note', () => {
  assert.equal(footerText('Your X-ray', 2544, 3056), 'Your X-ray · 2544×3056 px');
  assert.equal(footerText('Similar X-ray #2', 320, 320, '(stored gallery size)'), 'Similar X-ray #2 · 320×320 px (stored gallery size)');
  assert.equal(footerText('Your X-ray'), 'Your X-ray');   // not loaded yet: the label alone
  assert.equal(footerText('Your X-ray', 0, 0, '(stored gallery size)'), 'Your X-ray');
});

test('the keys: Esc closes, + and − zoom (= and _ too), 0 resets, an arrow pans so that more of that side shows; others do nothing', () => {
  assert.deepEqual(keyAction('Escape'), { type: 'close' });
  for (const key of ['+', '=']) assert.deepEqual(keyAction(key), { type: 'zoom', factor: ZOOM_STEP });
  for (const key of ['-', '_', '−']) assert.deepEqual(keyAction(key), { type: 'zoom', factor: 1 / ZOOM_STEP });
  assert.deepEqual(keyAction('0'), { type: 'reset' });
  assert.deepEqual(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].map(keyAction).map((a) => [a.dx, a.dy]),
                   [[PAN_STEP, 0], [-PAN_STEP, 0], [0, PAN_STEP], [0, -PAN_STEP]]);
  for (const other of ['Enter', ' ', 'a', 'Tab', 'Esc', '1', undefined]) assert.equal(keyAction(other), null, String(other));
});

test('a turn of the wheel zooms smoothly, whatever unit it counts in, and never more than double or half', () => {
  assert.equal(wheelFactor(0), 1);
  assert.ok(wheelFactor(-100) > 1 && wheelFactor(100) < 1);   // up zooms in
  assert.ok(near(wheelFactor(-100) * wheelFactor(100), 1));    // and back out by as much
  assert.ok(near(wheelFactor(3, 1), wheelFactor(48)));          // three lines are 48 px
  assert.deepEqual([wheelFactor(-1e6), wheelFactor(1e6), wheelFactor(1, 2) >= 0.5], [2, 0.5, true]);
  assert.equal(wheelFactor(NaN), 1);
});

// ---- the viewer on a page ----------------------------------------------------------------------------------------------------------------

// The page around #viewer as index.html has it: the banner, the layout (with a control that opens the viewer), the composer, and a node the
// page made inert itself.
function page() {
  installDom();
  const root = el('div', { id: 'viewer', hidden: true, role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Image viewer' });
  const banner = el('header', { class: 'banner' }, el('button', { id: 'sidebar-toggle', type: 'button' }, '☰'));
  const layout = el('div', { class: 'layout' }, el('main', { id: 'conversation' }, el('button', { id: 'opener', type: 'button' }, 'open')));
  const composer = el('form', { id: 'composer' }, el('textarea', { id: 'prompt' }));
  const already = el('div', { id: 'elsewhere', inert: true });
  document.body.replaceChildren(banner, layout, composer, root, already);
  return { root, banner, layout, composer, already, opener: document.getElementById('opener') };
}
const fakeLoader = () => {
  const asked = [];
  const forgotten = [];
  return { asked, forgotten, loadImage: async (path) => { asked.push(path); return `blob:fake/${asked.length}`; }, forgetImage: (path) => forgotten.push(path) };
};
const tick = async () => { for (let i = 0; i < 4; i += 1) await new Promise((resolve) => setImmediate(resolve)); };
const panesOf = (root) => root.querySelectorAll('.viewer-pane');
const stageOf = (pane) => pane.querySelector('.viewer-stage');
const scaleOf = (pane) => Number(stageOf(pane).getAttribute('data-scale'));
const control = (root, name) => root.querySelector(`[data-viewer="${name}"]`);
// The pane's picture has decoded with this natural size, in a stage of this size.
function decoded(pane, natW, natH, boxW = 400, boxH = 300) {
  const stage = stageOf(pane);
  Object.defineProperty(stage, 'clientWidth', { configurable: true, value: boxW });
  Object.defineProperty(stage, 'clientHeight', { configurable: true, value: boxH });
  stage.getBoundingClientRect = () => ({ left: 0, top: 0, right: boxW, bottom: boxH, width: boxW, height: boxH });
  const img = pane.querySelector('img');
  Object.assign(img, { naturalWidth: natW, naturalHeight: natH });
  img.dispatchEvent(new ShimEvent('load'));
}
const fire = (target, type, init = {}) => { const e = new ShimEvent(type, { bubbles: true, cancelable: true, ...init }); target.dispatchEvent(e); return e; };
const press = (target, key, init = {}) => fire(target, 'keydown', { key, ...init });
const ORIGINAL = { url: '/v1/messages/u_1/image?variant=original', label: 'Your X-ray', note: '', transient: true };
const NEIGHBOUR = { url: '/v1/gallery/images/101', label: 'Similar X-ray #1', note: '(stored gallery size)' };

test('one picture: a dialog over the page with its controls and one pane; the picture comes through loadImage and its footer says its size', async () => {
  const p = page();
  const fake = fakeLoader();
  const handle = openViewer({ left: ORIGINAL }, { root: p.root, loadImage: fake.loadImage, forgetImage: fake.forgetImage });
  assert.ok(handle && typeof handle.close === 'function');
  assert.equal(p.root.hidden, false);
  assert.deepEqual(['role', 'aria-modal', 'aria-label'].map((a) => p.root.getAttribute(a)), ['dialog', 'true', 'Image viewer: Your X-ray']);
  assert.deepEqual(['out', 'in', 'fit', 'actual', 'reset', 'close'].map((n) => [control(p.root, n).textContent, control(p.root, n).getAttribute('aria-label')]),
                   [['−', 'Zoom out'], ['+', 'Zoom in'], ['Fit', null], ['1:1', '1:1, actual size'], ['Reset', null], ['✕', 'Close viewer']]);
  assert.equal(control(p.root, 'link'), null);   // one picture: nothing to link
  for (const name of ['brightness', 'contrast']) {
    const range = control(p.root, name);
    assert.deepEqual(['type', 'min', 'max', 'value'].map((a) => range.getAttribute(a)), ['range', '0', '300', '100']);
    assert.equal(range.closest('label').textContent.startsWith(name === 'brightness' ? 'Brightness' : 'Contrast'), true);
  }
  const [pane] = panesOf(p.root);
  assert.equal(panesOf(p.root).length, 1);
  const img = pane.querySelector('img');
  assert.equal(img.hasAttribute('src'), false);   // nothing until the loader answers: a server path is never a src
  assert.equal(img.getAttribute('data-src'), ORIGINAL.url);
  assert.equal(pane.querySelector('figcaption').textContent, 'Your X-ray');
  assert.equal(document.activeElement, stageOf(pane));   // the keys act on the picture at once
  await tick();
  assert.deepEqual(fake.asked, [ORIGINAL.url]);
  assert.equal(img.getAttribute('src'), 'blob:fake/1');
  decoded(pane, 2544, 3056);
  assert.equal(pane.querySelector('figcaption').textContent, 'Your X-ray · 2544×3056 px');
  assert.equal(pane.querySelector('.viewer-state').hidden, true);
  assert.equal(scaleOf(pane), 1);
  assert.match(img.style.transform, /^translate\(\d+\.\d+px, 0\.00px\) scale\(0\.09817\)$/);   // whole, centred: 300 / 3056
  assert.match(stageOf(pane).getAttribute('aria-label'), /^Your X-ray, 2544×3056 px\. /);
  handle.close();
});

test('the page behind is inert and does not scroll while the viewer is open, and the close puts it all back and gives the focus back', async () => {
  const p = page();
  const closes = [];
  const handle = openViewer({ left: ORIGINAL }, { root: p.root, loadImage: fakeLoader().loadImage, onClose: (o) => closes.push(o) });
  assert.deepEqual([p.banner, p.layout, p.composer].map((n) => n.hasAttribute('inert')), [true, true, true]);
  assert.equal(p.root.hasAttribute('inert'), false);
  assert.equal(document.body.classList.contains('viewer-open'), true);   // styles.css: overflow hidden
  handle.close();
  assert.deepEqual([p.banner, p.layout, p.composer].map((n) => n.hasAttribute('inert')), [false, false, false]);
  assert.equal(p.already.hasAttribute('inert'), true);   // what the page made inert itself stays so
  assert.deepEqual([document.body.classList.contains('viewer-open'), p.root.hidden, p.root.children.length], [false, true, 0]);
  assert.equal(p.root.getAttribute('aria-label'), 'Image viewer');
  assert.deepEqual(closes, [{ restore: true }]);
  handle.close();   // twice: nothing more
  assert.equal(closes.length, 1);
});

test('the focus is trapped in the viewer: Tab from its last control goes to its first, Shift+Tab from its first to its last', () => {
  const p = page();
  const handle = openViewer({ left: ORIGINAL, right: NEIGHBOUR }, { root: p.root, loadImage: fakeLoader().loadImage });
  const all = p.root.querySelectorAll('button, input, [tabindex]');
  const first = all[0];
  const last = all.at(-1);
  last.focus();
  assert.equal(press(last, 'Tab').defaultPrevented, true);
  assert.equal(document.activeElement, first);
  assert.equal(press(first, 'Tab', { shiftKey: true }).defaultPrevented, true);
  assert.equal(document.activeElement, last);
  all[1].focus();
  assert.equal(press(all[1], 'Tab').defaultPrevented, false);   // inside: the browser moves the focus on as usual
  handle.close();
});

test('two pictures side by side, linked by default: a zoom or a pan applies to both; unlinked, to the pane it was made in', async () => {
  const p = page();
  const handle = openViewer({ left: ORIGINAL, right: NEIGHBOUR }, { root: p.root, loadImage: fakeLoader().loadImage });
  assert.equal(p.root.getAttribute('aria-label'), 'Image viewer: Your X-ray beside Similar X-ray #1');
  const [left, right] = panesOf(p.root);
  assert.deepEqual([left.getAttribute('data-side'), right.getAttribute('data-side')], ['left', 'right']);
  await tick();
  decoded(left, 2544, 3056);
  decoded(right, 320, 320);
  assert.equal(right.querySelector('figcaption').textContent, 'Similar X-ray #1 · 320×320 px (stored gallery size)');
  const link = control(p.root, 'link');
  assert.equal(link.checked, true);
  assert.equal(link.closest('label').textContent, 'Link panes');
  control(p.root, 'in').click();
  assert.deepEqual([scaleOf(left), scaleOf(right)], [ZOOM_STEP, ZOOM_STEP]);
  const before = [left.querySelector('img').style.transform, right.querySelector('img').style.transform];
  fire(stageOf(right), 'pointerdown', { pointerId: 1, button: 0, clientX: 200, clientY: 150 });
  fire(stageOf(right), 'pointermove', { pointerId: 1, clientX: 180, clientY: 140 });   // a drag in the right pane
  fire(stageOf(right), 'pointerup', { pointerId: 1 });
  assert.notEqual(left.querySelector('img').style.transform, before[0]);   // both moved
  assert.notEqual(right.querySelector('img').style.transform, before[1]);
  link.checked = false;
  link.dispatchEvent(new ShimEvent('change', { bubbles: true }));
  fire(stageOf(right), 'pointerdown', { pointerId: 2, button: 0, clientX: 10, clientY: 10 });   // the right pane is the one in use now
  fire(stageOf(right), 'pointerup', { pointerId: 2 });
  control(p.root, 'in').click();
  assert.deepEqual([scaleOf(left), scaleOf(right)], [ZOOM_STEP, ZOOM_STEP * ZOOM_STEP]);
  assert.equal(right.hasAttribute('data-active'), true);
  stageOf(left).focus();
  press(stageOf(left), '-');
  assert.deepEqual([scaleOf(left), scaleOf(right)], [1, ZOOM_STEP * ZOOM_STEP]);
  handle.close();
});

test('the buttons: + and − zoom by a step, 1:1 shows one picture pixel to a screen pixel, Fit shows it whole, Reset the fit and 100% filters', async () => {
  const p = page();
  const handle = openViewer({ left: ORIGINAL }, { root: p.root, loadImage: fakeLoader().loadImage });
  await tick();
  const [pane] = panesOf(p.root);
  decoded(pane, 800, 600, 400, 300);   // fit 0.5
  control(p.root, 'in').click();
  assert.equal(scaleOf(pane), ZOOM_STEP);
  control(p.root, 'out').click();
  control(p.root, 'out').click();
  assert.equal(scaleOf(pane), Number((1 / ZOOM_STEP).toFixed(3)));
  control(p.root, 'actual').click();
  assert.equal(scaleOf(pane), 2);   // 1:1 is twice the fit of 0.5
  assert.match(pane.querySelector('img').style.transform, / scale\(1\.00000\)$/);
  control(p.root, 'fit').click();
  assert.equal(scaleOf(pane), 1);
  const brightness = control(p.root, 'brightness');
  brightness.value = '150';
  brightness.dispatchEvent(new ShimEvent('input', { bubbles: true }));
  control(p.root, 'in').click();
  control(p.root, 'reset').click();
  assert.equal(scaleOf(pane), 1);
  assert.equal(brightness.value, '100');
  assert.equal(pane.querySelector('img').style.filter, 'brightness(100%) contrast(100%)');
  for (let i = 0; i < 30; i += 1) control(p.root, 'in').click();
  assert.equal(scaleOf(pane), MAX_SCALE);   // held to 16
  handle.close();
});

test('the keys: + − and 0 zoom and reset, the arrows pan, Esc closes; an arrow on a slider moves the slider, and Ctrl+− stays the browser\'s', async () => {
  const p = page();
  const closes = [];
  const handle = openViewer({ left: ORIGINAL }, { root: p.root, loadImage: fakeLoader().loadImage, onClose: (o) => closes.push(o) });
  await tick();
  const [pane] = panesOf(p.root);
  decoded(pane, 800, 600, 400, 300);
  const stage = stageOf(pane);
  assert.equal(press(stage, '+').defaultPrevented, true);
  assert.equal(scaleOf(pane), ZOOM_STEP);
  const at = () => pane.querySelector('img').style.transform;
  const before = at();
  press(stage, 'ArrowRight');   // more of the right side: the picture moves left
  const x = (t) => Number(/translate\((-?[\d.]+)px/.exec(t)[1]);
  assert.ok(near(x(at()), x(before) - PAN_STEP, 0.01), `${before} -> ${at()}`);
  press(stage, '0');
  assert.equal(scaleOf(pane), 1);
  const range = control(p.root, 'contrast');
  range.focus();
  assert.equal(press(range, 'ArrowRight').defaultPrevented, false);   // the slider's own key
  assert.equal(press(stage, '-', { ctrlKey: true }).defaultPrevented, false);   // the browser's zoom
  assert.equal(press(stage, 'q').defaultPrevented, false);
  let heard = 0;
  document.body.addEventListener('keydown', () => { heard += 1; });
  assert.equal(press(stage, 'Escape').defaultPrevented, true);
  assert.equal(heard, 0);   // the page under the viewer never heard its Escape: the drawer and the study list stay as they are
  assert.deepEqual([p.root.hidden, closes], [true, [{ restore: true }]]);
  handle.close();
});

test('the wheel zooms at the cursor, a drag pans, two fingers pinch, a double-click goes to 1:1 at the point and back to the fit', async () => {
  const p = page();
  const handle = openViewer({ left: ORIGINAL }, { root: p.root, loadImage: fakeLoader().loadImage });
  await tick();
  const [pane] = panesOf(p.root);
  decoded(pane, 800, 600, 400, 300);   // fit 0.5: at the fit the picture is 400×300, at (0, 0)
  const stage = stageOf(pane);
  const wheel = fire(stage, 'wheel', { deltaY: -100, deltaMode: 0, clientX: 100, clientY: 75 });
  assert.equal(wheel.defaultPrevented, true);   // the page behind does not scroll
  const zoomed = scaleOf(pane);
  assert.ok(zoomed > 1);
  const t = () => /translate\((-?[\d.]+)px, (-?[\d.]+)px\) scale\(([\d.]+)\)/.exec(pane.querySelector('img').style.transform).slice(1).map(Number);
  const [x, y, s] = t();
  assert.ok(near((100 - x) / s, 100 / 0.5, 0.05) && near((75 - y) / s, 75 / 0.5, 0.05));   // the picture point under the cursor stayed under it
  fire(stage, 'pointerdown', { pointerId: 7, button: 0, clientX: 50, clientY: 50 });
  assert.equal(stage.classList.contains('dragging'), true);
  fire(stage, 'pointermove', { pointerId: 7, clientX: 60, clientY: 45 });
  fire(stage, 'pointerup', { pointerId: 7 });
  const [x2, y2] = t();
  assert.ok(near(x2, x + 10, 0.01) && near(y2, y - 5, 0.01));
  assert.equal(stage.classList.contains('dragging'), false);
  fire(stage, 'pointerdown', { pointerId: 1, button: 0, clientX: 100, clientY: 100 });
  fire(stage, 'pointerdown', { pointerId: 2, button: 0, clientX: 200, clientY: 100 });
  fire(stage, 'pointermove', { pointerId: 2, clientX: 300, clientY: 100 });   // the fingers twice as far apart
  fire(stage, 'pointerup', { pointerId: 1 });
  fire(stage, 'pointerup', { pointerId: 2 });
  assert.ok(near(scaleOf(pane), Math.min(16, zoomed * 2), 0.002));
  control(p.root, 'fit').click();
  fire(stage, 'dblclick', { clientX: 120, clientY: 90 });
  assert.equal(scaleOf(pane), 2);   // 1:1, about the point
  fire(stage, 'dblclick', { clientX: 120, clientY: 90 });
  assert.equal(scaleOf(pane), 1);   // and back to the fit
  fire(stage, 'pointerdown', { pointerId: 9, button: 2, clientX: 5, clientY: 5 });   // a right button does not drag
  assert.equal(stage.classList.contains('dragging'), false);
  handle.close();
});

test('brightness and contrast run 0 to 300% and are a CSS filter on the viewer\'s own pictures, nothing else: display only', async () => {
  const p = page();
  const handle = openViewer({ left: ORIGINAL, right: NEIGHBOUR }, { root: p.root, loadImage: fakeLoader().loadImage });
  await tick();
  const imgs = p.root.querySelectorAll('img');
  const srcs = imgs.map((i) => i.getAttribute('src'));
  const set = (name, value) => { const r = control(p.root, name); r.value = value; r.dispatchEvent(new ShimEvent('input', { bubbles: true })); return r; };
  const brightness = set('brightness', '175');
  const contrast = set('contrast', '40');
  assert.deepEqual(imgs.map((i) => i.style.filter), ['brightness(175%) contrast(40%)', 'brightness(175%) contrast(40%)']);
  assert.deepEqual([brightness.getAttribute('aria-valuetext'), contrast.closest('label').querySelector('.viewer-value').textContent], ['175%', '40%']);
  assert.deepEqual(imgs.map((i) => i.getAttribute('src')), srcs);   // the pictures themselves are untouched
  set('brightness', '999');
  assert.equal(brightness.value, '300');   // held to 300%
  set('contrast', 'x');
  assert.equal(contrast.value, '40');      // not a number: as it was
  assert.equal(document.body.querySelectorAll('[style]').every((n) => p.root.contains(n)), true);   // nothing outside the viewer is styled
  handle.close();
  assert.equal(p.root.querySelectorAll('[style]').length, 0);
});

test('a picture no card shows is let go of when the viewer closes; one a card shows is left in the page\'s cache', async () => {
  const p = page();
  const fake = fakeLoader();
  const handle = openViewer({ left: ORIGINAL, right: NEIGHBOUR }, { root: p.root, loadImage: fake.loadImage, forgetImage: fake.forgetImage });
  await tick();
  assert.deepEqual(fake.asked, [ORIGINAL.url, NEIGHBOUR.url]);   // both through the page's loader, with its auth
  handle.close();
  assert.deepEqual(fake.forgotten, [ORIGINAL.url]);   // the full-size original: its object URL is revoked (api.js forgetImage)
});

test('after a click on the bar or a footer the viewer itself has the focus, and +, −, 0 and the arrows still work (P6 fix 1)', async () => {
  const p = page();
  const handle = openViewer({ left: ORIGINAL }, { root: p.root, loadImage: fakeLoader().loadImage });
  await tick();
  const [pane] = panesOf(p.root);
  decoded(pane, 800, 600, 400, 300);
  assert.equal(p.root.getAttribute('tabindex'), '-1');   // a click can focus it, Tab cannot
  p.root.focus();   // a click on a part that is no control: the browser gives the focus to the nearest focusable ancestor
  assert.equal(document.activeElement, p.root);
  assert.equal(press(p.root, '+').defaultPrevented, true);
  assert.equal(scaleOf(pane), ZOOM_STEP);
  const x = () => Number(/translate\((-?[\d.]+)px/.exec(pane.querySelector('img').style.transform)[1]);
  const before = x();
  press(p.root, 'ArrowRight');
  assert.ok(near(x(), before - PAN_STEP, 0.01));
  press(p.root, '-');
  assert.equal(scaleOf(pane), 1);
  press(p.root, '+');
  press(p.root, '0');
  assert.equal(scaleOf(pane), 1);
  const controls = p.root.querySelectorAll('button, input, [tabindex]');
  assert.equal(press(p.root, 'Tab', { shiftKey: true }).defaultPrevented, true);   // Shift+Tab from the viewer itself stays inside it
  assert.equal(document.activeElement, controls.at(-1));
  handle.close();
  assert.equal(p.root.hasAttribute('tabindex'), false);   // closed, it is nothing to focus
});

test('a viewer closed before its pictures were asked for asks for none (P6 fix 1)', async () => {
  const p = page();
  const fake = fakeLoader();
  const handle = openViewer({ left: ORIGINAL, right: NEIGHBOUR }, { root: p.root, loadImage: fake.loadImage, forgetImage: fake.forgetImage });
  handle.close();   // in the same task: before the microtask that would ask
  await tick();
  assert.deepEqual(fake.asked, []);
  assert.equal(p.root.querySelectorAll('img').length, 0);
});

test('a picture that cannot load says so; a path on another origin is not even asked for; a blob: or data: picture is shown as it is', async () => {
  const p = page();
  const fake = fakeLoader();
  const handle = openViewer({ left: { url: 'https://evil.example/x.png', label: 'Elsewhere' }, right: { url: 'blob:http://x/1', label: 'Local' } },
                            { root: p.root, loadImage: fake.loadImage });
  await tick();
  const [left, right] = panesOf(p.root);
  assert.deepEqual(fake.asked, []);   // the token never goes to another origin, and a picture the page made needs no fetch
  assert.equal(left.querySelector('img').hasAttribute('src'), false);
  assert.equal(left.querySelector('.viewer-state').textContent, 'This picture could not be loaded.');
  assert.equal(right.querySelector('img').getAttribute('src'), 'blob:http://x/1');
  handle.close();
  const refusing = openViewer({ left: ORIGINAL }, { root: p.root, loadImage: async () => { throw new Error('refused'); } });
  await tick();
  assert.equal(panesOf(p.root)[0].querySelector('.viewer-state').hidden, false);
  refusing.close();
  const odd = openViewer({ left: ORIGINAL }, { root: p.root, loadImage: async () => '/v1/not-an-object-url' });   // never a server path as a src
  await tick();
  assert.equal(panesOf(p.root)[0].querySelector('img').hasAttribute('src'), false);
  odd.close();
  assert.equal(openViewer({}, { root: p.root }), null);   // nothing to show: nothing opens
  assert.equal(p.root.hidden, true);
});

test('a second viewer closes the first, which gives no focus back; a resize keeps each pane\'s scale and the point at its centre', async () => {
  const p = page();
  const observers = [];
  const ResizeObserver = class { constructor(cb) { this.cb = cb; observers.push(this); } observe(node) { this.node = node; } disconnect() { this.gone = true; } };
  const closes = [];
  const first = openViewer({ left: ORIGINAL }, { root: p.root, loadImage: fakeLoader().loadImage, onClose: (o) => closes.push(['first', o]), ResizeObserver });
  const second = openViewer({ left: NEIGHBOUR }, { root: p.root, loadImage: fakeLoader().loadImage, onClose: (o) => closes.push(['second', o]), ResizeObserver });
  assert.deepEqual(closes, [['first', { restore: false }]]);
  assert.equal(observers[0].gone, true);
  assert.equal(panesOf(p.root).length, 1);
  await tick();
  const [pane] = panesOf(p.root);
  decoded(pane, 320, 320, 400, 300);
  press(stageOf(pane), '+');
  decoded(pane, 320, 320, 400, 300);   // a second load event changes nothing that matters: the view starts over at the fit
  press(stageOf(pane), '+');
  const stage = stageOf(pane);
  Object.defineProperty(stage, 'clientWidth', { configurable: true, value: 800 });
  Object.defineProperty(stage, 'clientHeight', { configurable: true, value: 600 });
  observers[1].cb([]);
  assert.equal(scaleOf(pane), ZOOM_STEP);   // the same zoom, in the bigger pane
  assert.equal(pane.querySelector('img').style.transform, 'translate(25.00px, -75.00px) scale(2.34375)');   // fit 600 / 320 × 1.25, its centre kept
  second.close();
  first.close();
  assert.deepEqual(closes, [['first', { restore: false }], ['second', { restore: true }]]);
});
