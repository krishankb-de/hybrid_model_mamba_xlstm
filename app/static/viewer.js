// app/static/viewer.js — the full-screen image viewer (CHAT_UI_PLAN.md P6-E): one picture, or two side by side (the turn's X-ray and the
// similar X-ray it was opened from; stacked below 700 px), with zoom, pan, brightness and contrast by pointer and by keyboard, and a focus trap.
//
// The first half is pure (the view's arithmetic, the footer, the keys), so node tests it as it is. openViewer({left, right}, env) draws the
// viewer into the page's #viewer and returns its handle, { close, root }. env: document, root (the #viewer element), window (its resize, where
// there is no ResizeObserver), loadImage(path) -> Promise<object URL> (api.js's loadImage with the page's auth), forgetImage(path) (api.js's),
// onClose({ restore }) (the page gives the focus back to the control that opened it).
//
// A side is { url, label, note, transient }. url is a path on this page's own origin, fetched through env.loadImage with its auth headers and
// its cache of refusals (an <img src> cannot carry the bearer token, D23), or a blob: or data: picture the page made itself, which is shown as
// it is: a server path is never an <img src>. label names the picture ("Similar X-ray #2"), note follows its pixel size in the footer
// ("(stored gallery size)"), and transient says that no card shows it (the full-size original): the viewer lets go of it when it closes, so
// that its object URL is revoked and its bytes are not held for the rest of the chat. The pixel size is the loaded picture's own (ruling 10).
// Brightness and contrast are a CSS filter on the viewer's own pictures: display only, nothing sent or shown elsewhere changes (ruling 11).
//
// A pane's view is { scale, x, y }: scale is relative to the fit (1 shows the picture whole), and x, y place the picture's top-left corner in
// the pane, in CSS px. The picture is laid out at its natural size and drawn with transform: translate(x, y) scale(fit × scale). "Link
// panes" (on by default) makes every zoom and pan apply to both panes alike; off, it applies to the pane it was made in, and a button or a
// key acts on the pane last used.
import { isSameOriginPath } from './api.js';
import { el } from './render.js';

export const MIN_SCALE = 0.25, MAX_SCALE = 16;
export const ZOOM_STEP = 1.25;   // one press of + or −, by key or button
export const PAN_STEP = 48;      // one arrow key, in CSS px

export function clamp(v, lo, hi) { return Math.min(hi, Math.max(lo, v)); }

export function zoomAt(view, factor, cx, cy) {
  // Zoom about the cursor: the image point under (cx, cy) stays under it.
  const scale = clamp(view.scale * factor, MIN_SCALE, MAX_SCALE);
  const k = scale / view.scale;
  return { ...view, scale, x: cx - (cx - view.x) * k, y: cy - (cy - view.y) * k };
}

export function clampPan(view, imgW, imgH, boxW, boxH) {
  // Keep at least 10% of the image inside the pane.
  const w = imgW * view.scale, h = imgH * view.scale;
  return { ...view, x: clamp(view.x, -w + 0.1 * w, boxW - 0.1 * w), y: clamp(view.y, -h + 0.1 * h, boxH - 0.1 * h) };
}

export function filterCss({ brightness = 100, contrast = 100 }) {
  return `brightness(${brightness}%) contrast(${contrast}%)`;
}

// The scale at which a picture w×h fits a box boxW×boxH whole, in CSS px per picture px (above 1 for a small picture in a large pane); 1 where
// either has no size yet.
export function fitScale(w, h, boxW, boxH) {
  return w > 0 && h > 0 && boxW > 0 && boxH > 0 ? Math.min(boxW / w, boxH / h) : 1;
}

// A view at this scale (relative to the fit, held to the bounds) with the picture centred in the box; baseW×baseH is the picture at the fit.
export function centred(scale, baseW, baseH, boxW, boxH) {
  const s = clamp(scale, MIN_SCALE, MAX_SCALE);
  return { scale: s, x: (boxW - baseW * s) / 2, y: (boxH - baseH * s) / 2 };
}

// The view once the pane has changed size (a resize, the panes stacking): the same scale, with the picture point that was at the centre of
// the old pane at the centre of the new one. base and box are [w, h], before and after.
export function recentred(view, [oldBaseW, oldBaseH], [oldBoxW, oldBoxH], [baseW, baseH], [boxW, boxH]) {
  const u = oldBaseW > 0 ? (oldBoxW / 2 - view.x) / (oldBaseW * view.scale) : 0.5;
  const v = oldBaseH > 0 ? (oldBoxH / 2 - view.y) / (oldBaseH * view.scale) : 0.5;
  return { ...view, x: boxW / 2 - u * baseW * view.scale, y: boxH / 2 - v * baseH * view.scale };
}

// A pane's footer: "Your X-ray · 2544×3056 px", "Similar X-ray #2 · 320×320 px (stored gallery size)"; the label alone until the picture has
// a size.
export function footerText(label, w = 0, h = 0, note = '') {
  const size = w > 0 && h > 0 ? `${w}×${h} px${note ? ` ${note}` : ''}` : '';
  return [label, size].filter(Boolean).join(' · ');
}

// The zoom of one turn of the wheel: smooth, whatever unit the wheel counts in (pixels, lines or pages), and never more than double or half.
export function wheelFactor(deltaY, deltaMode = 0) {
  const px = (Number.isFinite(deltaY) ? deltaY : 0) * (deltaMode === 1 ? 16 : deltaMode === 2 ? 400 : 1);
  return clamp(Math.exp(-px * 0.002), 0.5, 2);
}

// What a key does in the viewer, or null. Esc closes; + and − zoom (and = and _, the same keys without and with Shift); 0 resets; an arrow
// pans, so that more of that side of the picture shows.
export function keyAction(key) {
  switch (key) {
    case 'Escape': return { type: 'close' };
    case '+': case '=': return { type: 'zoom', factor: ZOOM_STEP };
    case '-': case '_': case '−': return { type: 'zoom', factor: 1 / ZOOM_STEP };
    case '0': return { type: 'reset' };
    case 'ArrowLeft': return { type: 'pan', dx: PAN_STEP, dy: 0 };
    case 'ArrowRight': return { type: 'pan', dx: -PAN_STEP, dy: 0 };
    case 'ArrowUp': return { type: 'pan', dx: 0, dy: PAN_STEP };
    case 'ArrowDown': return { type: 'pan', dx: 0, dy: -PAN_STEP };
    default: return null;
  }
}

// ---- the viewer on the page --------------------------------------------------------------------------------------------------------------

const PICTURE = /^(?:blob:|data:image\/(?:png|jpeg|webp|gif);base64,)/;   // a picture the page made itself: shown as it is
const isSide = (side) => side !== null && typeof side === 'object' && (typeof side.url === 'string' || typeof side.label === 'string');
const text = (v) => (typeof v === 'string' ? v : '');
const FILTERS = Object.freeze({ brightness: 100, contrast: 100 });
let current = null;   // the viewer that is open: one at a time

export function openViewer(spec, env = {}) {
  const doc = env.document ?? globalThis.document;
  const root = env.root ?? doc?.getElementById?.('viewer');
  const sides = [spec?.left, spec?.right].filter(isSide);
  if (!root || !sides.length) return null;
  current?.close({ restore: false });
  const win = env.window ?? null;
  let closed = false;
  let linked = true;
  const filters = { ...FILTERS };

  // -- the panes
  const panes = sides.map((side, index) => {
    const label = text(side.label) || 'X-ray';
    const img = el('img', { class: 'viewer-image', alt: label, draggable: 'false' });
    const note = el('p', { class: 'viewer-state' }, 'Loading…');
    const stage = el('div', { class: 'viewer-stage', tabindex: '0', role: 'img', 'aria-label': stageName(label) }, img, note);
    const footer = el('figcaption', { class: 'viewer-footer' }, footerText(label));
    const node = el('figure', { class: 'viewer-pane', 'data-side': index ? 'right' : 'left' }, stage, footer);
    return { side, label, img, note, stage, footer, node, natW: 0, natH: 0, fit: 1, box: [0, 0], view: { scale: 1, x: 0, y: 0 },
             loaded: false, pointers: new Map() };
  });
  let active = panes[0];

  function stageName(label, size = '') {
    return `${label}${size ? `, ${size}` : ''}. Drag or use the arrow keys to pan; plus and minus zoom.`;
  }
  const baseOf = (pane) => [pane.natW * pane.fit, pane.natH * pane.fit];

  function draw(pane) {
    pane.img.style.filter = filterCss(filters);
    pane.stage.setAttribute('data-scale', String(Number(pane.view.scale.toFixed(4))));   // the zoom, for the page's checks
    if (!pane.loaded) return;
    const { x, y, scale } = pane.view;
    pane.img.style.transform = `translate(${x.toFixed(2)}px, ${y.toFixed(2)}px) scale(${(pane.fit * scale).toFixed(5)})`;
  }

  // The pane's box and the fit it gives; a pane that has changed size keeps its scale and the picture point at its centre (recentred).
  function measure(pane) {
    const box = [pane.stage.clientWidth || 0, pane.stage.clientHeight || 0];
    if (!pane.loaded) { pane.box = box; return; }
    const oldBase = baseOf(pane);
    const oldBox = pane.box;
    pane.fit = fitScale(pane.natW, pane.natH, box[0], box[1]);
    pane.box = box;
    const base = baseOf(pane);
    const atFit = Math.abs(pane.view.scale - 1) < 1e-6;
    pane.view = atFit || !(oldBox[0] > 0) ? centred(pane.view.scale, base[0], base[1], box[0], box[1])
      : clampPan(recentred(pane.view, oldBase, oldBox, base, box), base[0], base[1], box[0], box[1]);
    draw(pane);
  }

  function loaded(pane) {
    if (closed) return;
    pane.natW = pane.img.naturalWidth || 0;
    pane.natH = pane.img.naturalHeight || 0;
    pane.loaded = pane.natW > 0 && pane.natH > 0;
    if (!pane.loaded) { failed(pane); return; }
    pane.note.hidden = true;
    pane.footer.textContent = footerText(pane.label, pane.natW, pane.natH, text(pane.side.note));
    pane.stage.setAttribute('aria-label', stageName(pane.label, footerText('', pane.natW, pane.natH, text(pane.side.note))));
    pane.box = [0, 0];
    pane.view = { scale: 1, x: 0, y: 0 };
    measure(pane);
  }

  function failed(pane) {
    if (closed) return;
    pane.loaded = false;
    pane.note.textContent = 'This picture could not be loaded.';
    pane.note.hidden = false;
    pane.img.removeAttribute('src');
  }

  // A picture comes through env.loadImage, with the page's auth, unless the page made it itself (blob:, data:). Anything else is not loaded.
  function load(pane) {
    const url = pane.side.url;
    pane.img.addEventListener('load', () => loaded(pane));
    pane.img.addEventListener('error', () => failed(pane));
    if (typeof url === 'string' && PICTURE.test(url)) { pane.img.setAttribute('src', url); return; }
    if (!isSameOriginPath(url) || typeof env.loadImage !== 'function') { failed(pane); return; }
    pane.img.setAttribute('data-src', url);   // whose picture it is, for the page and its checks; never its src
    Promise.resolve().then(() => (closed ? null : env.loadImage(url))).then(   // a viewer closed meanwhile asks for nothing (P6 fix 1)
      (objectUrl) => {
        if (closed) return;
        if (typeof objectUrl === 'string' && PICTURE.test(objectUrl)) pane.img.setAttribute('src', objectUrl); else failed(pane);
      },
      () => failed(pane));
  }

  // -- zoom and pan
  const targets = (pane) => (linked ? panes : [pane ?? active]);

  function zoomPane(pane, factor, cx = pane.box[0] / 2, cy = pane.box[1] / 2) {
    if (!pane.loaded) return;
    const [w, h] = baseOf(pane);
    pane.view = clampPan(zoomAt(pane.view, factor, cx, cy), w, h, pane.box[0], pane.box[1]);
    draw(pane);
  }

  function panPane(pane, dx, dy) {
    if (!pane.loaded) return;
    const [w, h] = baseOf(pane);
    pane.view = clampPan({ ...pane.view, x: pane.view.x + dx, y: pane.view.y + dy }, w, h, pane.box[0], pane.box[1]);
    draw(pane);
  }

  function showAt(pane, scale) {   // Fit (1) or 1:1 (one picture px to one CSS px), centred
    if (!pane.loaded) return;
    const [w, h] = baseOf(pane);
    pane.view = centred(scale, w, h, pane.box[0], pane.box[1]);
    draw(pane);
  }
  const actualScale = (pane) => 1 / pane.fit;

  // A point of one pane at the same place in another: the panes can differ in size where the page is uneven.
  const at = (from, to, cx, cy) => [from.box[0] > 0 ? (cx / from.box[0]) * to.box[0] : cx, from.box[1] > 0 ? (cy / from.box[1]) * to.box[1] : cy];

  function zoomFrom(pane, factor, cx, cy) {
    for (const p of targets(pane)) zoomPane(p, factor, ...at(pane, p, cx, cy));
  }

  // Double-click: from the fit (or nearer the fit than 1:1) to 1:1 about the point, else back to the fit.
  function toggleFrom(pane, cx, cy) {
    for (const p of targets(pane)) {
      if (!p.loaded) continue;
      const actual = actualScale(p);
      if (Math.abs(p.view.scale - 1) <= Math.abs(p.view.scale - actual)) zoomPane(p, actual / p.view.scale, ...at(pane, p, cx, cy));
      else showAt(p, 1);
    }
  }

  function setFilter(name, value) {
    const control = ranges[name];
    const v = clamp(Math.round(Number(value)), 0, 300);
    if (!Number.isFinite(v)) { control.input.value = String(filters[name]); return; }   // not a number: the slider keeps what it had
    filters[name] = v;
    control.input.value = String(v);
    control.input.setAttribute('aria-valuetext', `${v}%`);
    control.shown.textContent = `${v}%`;
    for (const p of panes) draw(p);
  }

  function reset() {
    for (const p of panes) showAt(p, 1);
    setFilter('brightness', 100);
    setFilter('contrast', 100);
  }

  // -- the toolbar
  const button = (name, label, aria = null) => el('button', { type: 'button', 'data-viewer': name, 'aria-label': aria,
                                                             onclick: () => act(name) }, label);
  const range = (name, title) => {
    const input = el('input', { type: 'range', min: '0', max: '300', step: '5', value: '100', 'data-viewer': name, 'aria-valuetext': '100%' });
    input.value = '100';
    const shown = el('span', { class: 'viewer-value', 'aria-hidden': 'true' }, '100%');
    input.addEventListener('input', () => setFilter(name, input.value));
    return { input, shown, label: el('label', { class: 'viewer-range' }, title, input, shown) };
  };
  const ranges = { brightness: range('brightness', 'Brightness'), contrast: range('contrast', 'Contrast') };
  const link = el('input', { type: 'checkbox', 'data-viewer': 'link', checked: true });
  link.checked = true;
  link.addEventListener('change', () => {
    linked = !!link.checked;
    if (linked) panesNode.setAttribute('data-linked', ''); else panesNode.removeAttribute('data-linked');
  });
  const tools = el('div', { class: 'viewer-tools', role: 'group', 'aria-label': 'Zoom, pan and display' },
    button('out', '−', 'Zoom out'), button('in', '+', 'Zoom in'), button('fit', 'Fit'), button('actual', '1:1', '1:1, actual size'),
    button('reset', 'Reset'),
    panes.length > 1 ? el('label', { class: 'viewer-check' }, link, 'Link panes') : null,
    ranges.brightness.label, ranges.contrast.label);
  const closer = button('close', '✕', 'Close viewer');
  const bar = el('div', { class: 'viewer-bar' }, tools, closer);
  const panesNode = el('div', { class: 'viewer-panes', 'data-count': String(panes.length), 'data-linked': '' }, ...panes.map((p) => p.node));

  function act(name) {
    if (name === 'close') { close(); return; }
    if (name === 'reset') { reset(); return; }
    for (const p of targets(null)) {
      if (name === 'in') zoomPane(p, ZOOM_STEP);
      else if (name === 'out') zoomPane(p, 1 / ZOOM_STEP);
      else if (name === 'fit') showAt(p, 1);
      else if (name === 'actual') showAt(p, actualScale(p));
    }
  }

  function markActive(pane) {
    active = pane;
    for (const p of panes) {
      if (p === pane) p.node.setAttribute('data-active', ''); else p.node.removeAttribute('data-active');
    }
  }

  // -- pointers: drag pans, two fingers pinch, the wheel zooms at the cursor, a double-click is the fit or 1:1
  for (const pane of panes) {
    const s = pane.stage;
    const local = (e) => {
      const r = s.getBoundingClientRect?.() ?? { left: 0, top: 0 };
      return [e.clientX - r.left, e.clientY - r.top];
    };
    s.addEventListener('focus', () => markActive(pane));
    s.addEventListener('pointerdown', (e) => {
      if (e.button !== undefined && e.button !== 0) return;
      markActive(pane);
      try { s.setPointerCapture?.(e.pointerId); } catch { /* a pointer the browser has already let go of */ }
      pane.pointers.set(e.pointerId, [e.clientX, e.clientY]);
      s.classList.add('dragging');
    });
    s.addEventListener('pointermove', (e) => {
      const before = pane.pointers.get(e.pointerId);
      if (!before) return;
      const now = [e.clientX, e.clientY];
      if (pane.pointers.size === 1) {
        for (const p of targets(pane)) panPane(p, now[0] - before[0], now[1] - before[1]);
      } else {
        const other = [...pane.pointers.entries()].find(([id]) => id !== e.pointerId)?.[1];
        const apart = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
        if (other && apart(before, other) > 0) {
          const [mx, my] = local({ clientX: (now[0] + other[0]) / 2, clientY: (now[1] + other[1]) / 2 });
          zoomFrom(pane, apart(now, other) / apart(before, other), mx, my);
        }
      }
      pane.pointers.set(e.pointerId, now);
    });
    const lift = (e) => {
      pane.pointers.delete(e.pointerId);
      if (!pane.pointers.size) s.classList.remove('dragging');
    };
    s.addEventListener('pointerup', lift);
    s.addEventListener('pointercancel', lift);
    s.addEventListener('wheel', (e) => {
      e.preventDefault();   // the page behind does not scroll
      const [cx, cy] = local(e);
      zoomFrom(pane, wheelFactor(e.deltaY, e.deltaMode), cx, cy);
    }, { passive: false });
    s.addEventListener('dblclick', (e) => {
      e.preventDefault();
      const [cx, cy] = local(e);
      toggleFrom(pane, cx, cy);
    });
  }

  // -- keys, and the focus trap
  const focusables = () => Array.from(root.querySelectorAll('button, input, [tabindex]'))
    .filter((n) => !n.hasAttribute('disabled') && n.getAttribute('tabindex') !== '-1' && !n.closest('[hidden]'));

  function onKey(e) {
    if (e.key === 'Tab') {
      const list = focusables();
      const first = list[0];
      const last = list[list.length - 1];
      const now = doc.activeElement;
      if (!first) return;
      if (e.shiftKey ? (now === first || now === root || !root.contains(now)) : (now === last || !root.contains(now))) {
        e.preventDefault();
        (e.shiftKey ? last : first).focus();
      }
      return;
    }
    const action = keyAction(e.key);
    if (!action || e.ctrlKey || e.metaKey || e.altKey) return;   // Ctrl+− and Cmd+0 stay the browser's own zoom
    if (action.type === 'pan' && e.target?.getAttribute?.('type') === 'range') return;   // the arrows move a slider
    e.preventDefault();
    e.stopPropagation();   // the viewer is the topmost layer: the drawer and the study list under it never hear its keys
    if (action.type === 'close') { close(); return; }
    if (action.type === 'reset') { reset(); return; }
    const pane = panes.find((p) => p.stage === e.target) ?? null;
    for (const p of targets(pane)) {
      if (action.type === 'zoom') zoomPane(p, action.factor);
      else panPane(p, action.dx, action.dy);
    }
  }

  // -- open: the page behind is inert and does not scroll; the panes follow the window's size
  const inerted = Array.from(doc.body.children).filter((node) => node !== root && !node.hasAttribute('inert'));
  const place = () => {   // where the page scrolls (a short viewport) the viewer is fixed under the banner: it starts where the banner ends
    const banner = doc.querySelector?.('.banner');
    const bottom = banner?.getBoundingClientRect?.().bottom;
    if (Number.isFinite(bottom)) root.style.setProperty('--viewer-top', `${Math.max(0, Math.round(bottom))}px`);
  };
  const onResize = () => { place(); panes.forEach(measure); };
  const Observer = env.ResizeObserver ?? globalThis.ResizeObserver;
  const observer = typeof Observer === 'function' ? new Observer(onResize) : null;

  function close({ restore = true } = {}) {
    if (closed) return;
    closed = true;
    if (current === handle) current = null;
    observer?.disconnect();
    win?.removeEventListener?.('resize', onResize);
    root.removeEventListener('keydown', onKey);
    root.removeAttribute('tabindex');
    for (const node of inerted) node.removeAttribute('inert');
    doc.body.classList.remove('viewer-open');
    root.hidden = true;
    root.replaceChildren();   // its pictures go with it
    root.setAttribute('aria-label', 'Image viewer');
    root.style.removeProperty('--viewer-top');
    for (const pane of panes) {   // a picture no card shows is let go of: its object URL is revoked (api.js forgetImage)
      if (pane.side.transient === true && isSameOriginPath(pane.side.url)) env.forgetImage?.(pane.side.url);
    }
    env.onClose?.({ restore });
  }

  const handle = { close, root };
  current = handle;
  root.replaceChildren(bar, panesNode);
  root.setAttribute('aria-label', panes.length > 1 ? `Image viewer: ${panes[0].label} beside ${panes[1].label}` : `Image viewer: ${panes[0].label}`);
  for (const node of inerted) node.setAttribute('inert', '');
  doc.body.classList.add('viewer-open');
  // A click on a part that is no control (the bar, a footer, the dark ground) gives the focus to the viewer itself, which a Tab never stops at:
  // its keys then still reach it (P6 fix 1). Without this the focus would go to the page behind, and + and the arrows would do nothing.
  root.setAttribute('tabindex', '-1');
  root.addEventListener('keydown', onKey);
  root.hidden = false;
  place();
  if (observer) observer.observe(panesNode); else win?.addEventListener?.('resize', onResize);
  markActive(panes[0]);
  panes.forEach(load);
  panes.forEach(draw);
  panes[0].stage.focus({ preventScroll: true });   // the keys act on the picture at once; a viewer that scrolls (a short page) shows its controls first
  return handle;
}
