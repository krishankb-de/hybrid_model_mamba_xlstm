// tests/frontend/composer.test.mjs — app/static/composer.js (CHAT_UI_PLAN.md P6-A): the client-side checks, the clipboard and the size
// formatter, and the intake itself on the DOM shim: the well's click and keys, drag-drop, the document-level paste, the preview and Remove.
//
// No server and no MIMIC data: every file is synthetic bytes, and object URLs come from a fake that records them.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { ShimEvent, installDom } from './dom_shim.mjs';
import { ACCEPTED, MAX_BYTES, attachComposer, checkFile, formatBytes, pickImageFromClipboard } from '../../app/static/composer.js';
import { el } from '../../app/static/render.js';

installDom();

const MB = 1024 * 1024;
const fileOf = (name, type = 'image/png', size = 2048) => new File([new Uint8Array(size)], name, { type });
const described = (name, type, size) => ({ name, type, size });   // what checkFile reads of a File: its name, type and size
const DICOM = 'DICOM is not supported. Export the image as PNG or JPEG first.';

// ---- the checks, the clipboard, the sizes ----------------------------------------------------------------------------------------------

test('checkFile accepts PNG, JPEG and WEBP up to 20 MB, and refuses DICOM, any other type and a larger file in the brief\'s words', () => {
  assert.deepEqual(ACCEPTED, ['image/png', 'image/jpeg', 'image/webp']);
  assert.equal(MAX_BYTES, 20 * MB);
  for (const type of ACCEPTED) assert.equal(checkFile(described('x', type, 1000)), null, type);
  assert.equal(checkFile(described('x.png', 'image/png', MAX_BYTES)), null);   // the limit itself is fine
  assert.equal(checkFile(described('x.png', 'image/png', MAX_BYTES + 1)), 'Image is larger than 20 MB.');
  assert.equal(checkFile(described('chest.dcm', '', 1000)), DICOM);
  assert.equal(checkFile(described('CHEST.DCM', 'application/octet-stream', 1000)), DICOM);   // by its name, whatever the browser calls its type
  assert.equal(checkFile(described('scan', 'application/dicom', 1000)), DICOM);
  for (const type of ['image/gif', 'image/svg+xml', 'application/pdf', 'text/html', 'image/bmp']) {
    assert.equal(checkFile(described('x', type, 1000)), 'Use a PNG, JPEG or WEBP image.', type);
  }
  assert.equal(checkFile(described('big.gif', 'image/gif', MAX_BYTES + 1)), 'Use a PNG, JPEG or WEBP image.');   // the type is said first
  assert.equal(checkFile(null), 'No file.');
  assert.equal(checkFile(undefined), 'No file.');
});

test('checkFile keeps P4\'s two allowances: a file with no type, or with no size known, goes to the server, which decides', () => {
  assert.equal(checkFile(described('xray', '', 1000)), null);   // a browser that cannot name the type: the server reads the bytes
  assert.equal(checkFile({ name: 'x.png', type: 'image/png' }), null);
});

test('pickImageFromClipboard takes the first file item of an accepted type and ignores text and other files', () => {
  const item = (kind, type, file = null) => ({ kind, type, getAsFile: () => file });
  const png = fileOf('shot.png');
  const jpeg = fileOf('second.jpg', 'image/jpeg');
  assert.equal(pickImageFromClipboard([item('string', 'text/plain'), item('file', 'image/png', png), item('file', 'image/jpeg', jpeg)]), png);
  assert.equal(pickImageFromClipboard([item('string', 'text/html'), item('string', 'text/plain')]), null);   // a text paste
  assert.equal(pickImageFromClipboard([item('file', 'image/gif', fileOf('x.gif', 'image/gif'))]), null);
  assert.equal(pickImageFromClipboard([item('string', 'image/png')]), null);   // a string item is no file, whatever its type says
  assert.equal(pickImageFromClipboard([item('file', 'image/png', null), item('file', 'image/webp', jpeg)]), jpeg);   // one that yields no file is passed over
  for (const none of [null, undefined, []]) assert.equal(pickImageFromClipboard(none), null);
  assert.equal(pickImageFromClipboard({ length: 1, 0: item('file', 'image/webp', png) }), png);   // array-like, as a DataTransferItemList is
});

test('formatBytes says KB below a megabyte, at least 1, and MB with one decimal from there', () => {
  assert.deepEqual([0, 1, 1023, 1024, 2048, 51200, MB - 1].map(formatBytes), ['1 KB', '1 KB', '1 KB', '1 KB', '2 KB', '50 KB', '1024 KB']);
  assert.deepEqual([MB, 1.5 * MB, 2.4 * MB, 20 * MB].map(formatBytes), ['1.0 MB', '1.5 MB', '2.4 MB', '20.0 MB']);
});

// ---- the intake on a page -------------------------------------------------------------------------------------------------------------------

// The composer's parts as app/static/index.html has them, a conversation beside them, and a composer attached with fakes that record.
function page() {
  installDom();
  const parts = {
    well: el('div', { id: 'image-well', tabindex: 0, role: 'button' }, 'Drop, paste or click to attach an X-ray (PNG, JPEG, WEBP)'),
    preview: el('div', { id: 'preview', hidden: true }),
    prompt: el('textarea', { id: 'prompt' }),
    input: el('input', { id: 'file', type: 'file', hidden: true }),
    elsewhere: el('main', { id: 'conversation' }),
  };
  parts.form = el('form', { id: 'composer' }, parts.well, parts.preview, parts.prompt, parts.input);
  document.body.replaceChildren(parts.elsewhere, parts.form);
  const made = [];
  const revoked = [];
  const urls = { createObjectURL: (blob) => { made.push(blob); return `blob:fake/${made.length}`; }, revokeObjectURL: (u) => { revoked.push(u); } };
  const changes = [];
  const errors = [];
  const composer = attachComposer({ well: parts.well, input: parts.input, preview: parts.preview, doc: document, urls,
                                    onChange: (file) => changes.push(file), onError: (message) => errors.push(message) });
  return { ...parts, made, revoked, changes, errors, composer };
}

const choose = (p, file) => { p.input.files = file ? [file] : []; p.input.dispatchEvent(new ShimEvent('change', { bubbles: true })); };
const fire = (target, type, init = {}) => { const e = new ShimEvent(type, { bubbles: true, cancelable: true, ...init }); target.dispatchEvent(e); return e; };
const press = (target, key) => fire(target, 'keydown', { key });
const line = (p) => p.preview.querySelector('span')?.textContent ?? null;
const removeOf = (p) => p.preview.querySelector('button');
const loaded = (p, width, height) => {   // the preview's picture has decoded: a browser now knows its natural size
  const img = p.preview.querySelector('img');
  Object.assign(img, { naturalWidth: width, naturalHeight: height });
  img.dispatchEvent(new ShimEvent('load'));
};
const clipboard = (...files) => ({ items: files.map((f) => (typeof f === 'string' ? { kind: 'string', type: f, getAsFile: () => null }
                                                                               : { kind: 'file', type: f.type, getAsFile: () => f })) });

test('the well opens the file chooser from a click, Enter and Space, and from no other key', () => {
  const p = page();
  let picks = 0;
  p.input.addEventListener('click', () => { picks += 1; });
  p.well.click();
  assert.equal(picks, 1);
  const enter = press(p.well, 'Enter');
  const space = press(p.well, ' ');
  assert.deepEqual([picks, enter.defaultPrevented, space.defaultPrevented], [3, true, true]);   // Space must not scroll the page
  for (const other of ['a', 'Tab', 'Escape']) assert.equal(press(p.well, other).defaultPrevented, false, other);
  assert.equal(picks, 3);
});

test('a chosen file is previewed: its thumbnail, its name and size at once, its pixel size once the picture has decoded, and Remove', () => {
  const p = page();
  const file = fileOf('x-ray.png', 'image/png', 51200);
  choose(p, file);
  assert.equal(p.preview.hidden, false);
  const img = p.preview.querySelector('img');
  assert.deepEqual([img.getAttribute('src'), img.getAttribute('alt')], ['blob:fake/1', 'Preview of x-ray.png']);
  assert.equal(line(p), 'x-ray.png · 50 KB');   // before the picture has decoded: what the file says
  loaded(p, 2544, 3056);
  assert.equal(line(p), 'x-ray.png · 2544×3056 px · 50 KB');
  const meta = p.preview.querySelector('span');   // the name, and what the file is, on lines of their own (a narrow composer cuts the name short)
  assert.deepEqual(meta.children.map((c) => [c.getAttribute('class'), c.textContent]),
                   [['preview-name', 'x-ray.png'], ['preview-sep', ' · '], ['preview-size', '2544×3056 px · 50 KB']]);
  assert.equal(meta.getAttribute('title'), 'x-ray.png · 2544×3056 px · 50 KB');   // whole, where the line is cut short
  const remove = removeOf(p);
  assert.deepEqual([remove.textContent, remove.getAttribute('aria-label'), remove.getAttribute('type')], ['×', 'Remove image', 'button']);
  assert.deepEqual(p.preview.children.map((c) => c.localName), ['img', 'span', 'button']);
  assert.deepEqual(p.changes, [file]);
  assert.equal(p.composer.get(), file);
  assert.equal(p.input.value, '');   // so choosing the same file again fires change again
  assert.deepEqual(p.made, [file]);
  choose(p, null);                   // a cancelled chooser: nothing changes
  assert.deepEqual([p.changes.length, p.composer.get()], [1, file]);
});

test('another file replaces the attached one and its object URL; Remove takes it away and hands the focus on to the well', () => {
  const p = page();
  choose(p, fileOf('first.png'));
  choose(p, fileOf('second.jpg', 'image/jpeg', 100));
  assert.equal(p.preview.querySelectorAll('img').length, 1);   // replaced, not added
  assert.equal(line(p), 'second.jpg · 1 KB');
  assert.deepEqual(p.revoked, ['blob:fake/1']);   // the first preview's object URL went with it
  removeOf(p).focus();
  removeOf(p).click();
  assert.deepEqual([p.preview.hidden, p.preview.children.length, p.composer.get()], [true, 0, null]);
  assert.deepEqual(p.revoked, ['blob:fake/1', 'blob:fake/2']);
  assert.equal(p.changes.at(-1), null);
  assert.equal(document.activeElement, p.well);   // the control that brought the image in, not the body
  p.prompt.focus();
  choose(p, fileOf('third.png'));
  removeOf(p).click();
  assert.equal(document.activeElement, p.prompt);   // focus that was elsewhere stays there
});

test('a picture that decodes after it was replaced does not write over the line of the one that replaced it', () => {
  const p = page();
  choose(p, fileOf('old.png'));
  const stale = p.preview.querySelector('img');
  choose(p, fileOf('new.png', 'image/png', 4096));
  Object.assign(stale, { naturalWidth: 64, naturalHeight: 64 });
  stale.dispatchEvent(new ShimEvent('load'));
  assert.equal(line(p), 'new.png · 4 KB');
  loaded(p, 320, 288);
  assert.equal(line(p), 'new.png · 320×288 px · 4 KB');
});

test('a file the page refuses is said through onError and leaves what is attached as it was', () => {
  const p = page();
  const ok = fileOf('ok.png');
  choose(p, ok);
  for (const [file, why] of [[fileOf('scan.gif', 'image/gif'), 'Use a PNG, JPEG or WEBP image.'], [fileOf('chest.dcm', ''), DICOM],
                             [described('huge.png', 'image/png', 21 * MB), 'Image is larger than 20 MB.']]) {
    choose(p, file);
    assert.equal(p.errors.at(-1), why);
    assert.equal(p.composer.get(), ok);
    assert.equal(line(p), 'ok.png · 2 KB');
  }
  assert.deepEqual([p.changes, p.made.length], [[ok], 1]);   // nothing was attached, and no object URL made, for a refused file
});

test('a file dropped on the well is attached; dragging over it marks the drop target; a drop that holds no file attaches nothing', () => {
  const p = page();
  const files = { types: ['Files'], files: [fileOf('dropped.png')] };
  for (const type of ['dragenter', 'dragover']) {
    assert.equal(fire(p.well, type, { dataTransfer: files }).defaultPrevented, true, type);   // without it the browser would not drop here
    assert.equal(p.well.classList.contains('dragging'), true, type);
  }
  fire(p.well, 'dragleave', {});
  assert.equal(p.well.classList.contains('dragging'), false);
  fire(p.well, 'dragover', { dataTransfer: files });
  assert.equal(fire(p.well, 'drop', { dataTransfer: files }).defaultPrevented, true);
  assert.equal(p.well.classList.contains('dragging'), false);
  assert.match(line(p), /^dropped\.png/);
  const webImage = { types: ['text/uri-list', 'text/html'], files: [] };   // an image dragged from another page: a link, no file
  assert.equal(fire(p.well, 'dragover', { dataTransfer: webImage }).defaultPrevented, true);
  assert.equal(fire(p.well, 'drop', { dataTransfer: webImage }).defaultPrevented, true);   // dropped on the well it opens nothing
  assert.match(line(p), /^dropped\.png/);   // and attaches nothing
  const refusedDrop = fire(p.well, 'drop', { dataTransfer: { types: ['Files'], files: [fileOf('x.gif', 'image/gif')] } });
  assert.deepEqual([refusedDrop.defaultPrevented, p.errors.at(-1)], [true, 'Use a PNG, JPEG or WEBP image.']);
});

test('a file dropped anywhere else on the page never navigates away from the chat, and attaches nothing; text may still be dragged into the note', () => {
  const p = page();
  const stray = { types: ['Files'], files: [fileOf('stray.png')] };
  for (const target of [p.elsewhere, p.prompt, document.body]) {
    assert.equal(fire(target, 'dragover', { dataTransfer: stray }).defaultPrevented, true);
    assert.equal(fire(target, 'drop', { dataTransfer: stray }).defaultPrevented, true);   // the page stays on the chat instead of opening the file
  }
  assert.equal(p.composer.get(), null);   // only the well attaches
  const text = { types: ['text/plain'], files: [] };
  assert.equal(fire(p.prompt, 'dragover', { dataTransfer: text }).defaultPrevented, false);
  assert.equal(fire(p.prompt, 'drop', { dataTransfer: text }).defaultPrevented, false);
});

test('a paste anywhere on the page attaches the image on the clipboard; a text paste is left to the browser, unchanged', () => {
  const p = page();
  const pasted = fire(p.prompt, 'paste', { clipboardData: clipboard('text/plain', fileOf('shot.png')) });
  assert.equal(pasted.defaultPrevented, true);
  assert.match(line(p), /^shot\.png/);
  const onPage = fire(p.elsewhere, 'paste', { clipboardData: clipboard(fileOf('from-the-page.webp', 'image/webp')) });   // the focus is not in the composer
  assert.deepEqual([onPage.defaultPrevented, line(p).startsWith('from-the-page.webp')], [true, true]);
  removeOf(p).click();
  assert.equal(fire(p.prompt, 'paste', { clipboardData: clipboard('text/plain', 'text/html') }).defaultPrevented, false);   // text pastes as text
  assert.equal(fire(p.prompt, 'paste', { clipboardData: clipboard(fileOf('x.gif', 'image/gif')) }).defaultPrevented, false);
  assert.equal(fire(p.prompt, 'paste', {}).defaultPrevented, false);
  assert.equal(fire(p.prompt, 'paste', { clipboardData: { files: [] } }).defaultPrevented, false);
  assert.deepEqual([p.preview.hidden, p.composer.get()], [true, null]);
});

test('clear() empties the preview, resets the file input and says so; a size that is not known is left out of the line', () => {
  const p = page();
  choose(p, fileOf('chest.png'));
  p.input.value = 'C:\\fakepath\\chest.png';
  p.composer.clear();
  assert.deepEqual([p.preview.hidden, p.preview.children.length, p.input.value, p.composer.get(), p.changes.at(-1)], [true, 0, '', null, null]);
  assert.deepEqual(p.revoked, ['blob:fake/1']);
  p.composer.clear();   // nothing attached: nothing to revoke twice
  assert.deepEqual(p.revoked, ['blob:fake/1']);
  choose(p, { name: 'sizeless.png', type: 'image/png' });
  assert.equal(line(p), 'sizeless.png');
  loaded(p, 640, 480);
  assert.equal(line(p), 'sizeless.png · 640×480 px');
  choose(p, new File([new Uint8Array(10)], '', { type: 'image/png' }));   // a pasted screenshot can have no name
  assert.equal(line(p), 'pasted image · 1 KB');
  assert.equal(p.preview.querySelector('img').getAttribute('alt'), 'Preview of pasted image');
});

test('a composer made inert takes no paste: the viewer over the page has made the page behind it inert (P6-E)', () => {
  const p = page();
  p.form.setAttribute('inert', '');
  const pasted = fire(document.body, 'paste', { clipboardData: clipboard(fileOf('behind.png')) });
  assert.deepEqual([pasted.defaultPrevented, p.composer.get(), p.preview.hidden], [false, null, true]);
  p.form.removeAttribute('inert');   // the viewer closed: a paste attaches again
  assert.equal(fire(document.body, 'paste', { clipboardData: clipboard(fileOf('back.png')) }).defaultPrevented, true);
  assert.match(line(p), /^back\.png/);
});
