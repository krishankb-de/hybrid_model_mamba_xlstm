// app/static/composer.js — image intake for the composer: click, drag-drop, paste; a preview before sending (CHAT_UI_PLAN.md P6-A).
// The server re-validates everything; these checks only save a round trip.
//
// The first half is pure (checkFile, pickImageFromClipboard, formatBytes), so node tests it as it is. attachComposer wires the well, the
// hidden file input and the preview, and takes the page's document and URL (object URLs) so that a test hands it a shim page and a fake.
// It is the page's one intake path (P6-A ruling 1: app.js keeps no copy of its own). Every string reaches the page as a text node.
export const ACCEPTED = ['image/png', 'image/jpeg', 'image/webp'];
export const MAX_BYTES = 20 * 1024 * 1024;   // the server's upload limit

// -> null, or the message to show. A file whose type the browser could not name (no extension it knows) and one whose size is not known are
// let through, as before P6-A: the server reads the bytes and decides.
export function checkFile(file) {
  if (!file) return 'No file.';
  if (/\.dcm$/i.test(file.name || '') || file.type === 'application/dicom') return 'DICOM is not supported. Export the image as PNG or JPEG first.';
  if (file.type && !ACCEPTED.includes(file.type)) return 'Use a PNG, JPEG or WEBP image.';
  if (file.size > MAX_BYTES) return 'Image is larger than 20 MB.';
  return null;
}

// DataTransferItemList-like (an array, or anything with a length and indexes) -> the first image File of an accepted type, or null. Text
// items are left alone, so a text paste stays the browser's.
export function pickImageFromClipboard(items) {
  for (const it of Array.from(items ?? [])) {
    if (it?.kind !== 'file' || !ACCEPTED.includes(it.type)) continue;
    const file = it.getAsFile();
    if (file) return file;
  }
  return null;
}

export function formatBytes(n) {
  return n < 1024 * 1024 ? `${Math.max(1, Math.round(n / 1024))} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;
}

// What the file is, after its name: "2544×3056 px · 2.4 MB". The pixel size is there once the picture has decoded; a size that is not known
// is left out.
const sizesOf = (file, img) => [img?.naturalWidth > 0 && img.naturalHeight > 0 ? `${img.naturalWidth}×${img.naturalHeight} px` : null,
                                Number.isFinite(file.size) ? formatBytes(file.size) : null].filter(Boolean).join(' · ');

// well, input (type=file), preview: the composer's parts. onChange(file | null) hears each attach and each clear; onError(message) hears
// a file the checks refuse, which leaves what is attached as it was. doc and urls stand in for document and URL in a test.
// -> { get, clear }: the attached file (or null), and a clear that the page calls once the server has accepted the turn it went with.
export function attachComposer({ well, input, preview, onChange, onError, doc = globalThis.document, urls = globalThis.URL }) {
  let current = null;
  let url = null;
  // The preview's line, "chest.png · 2544×3056 px · 2.4 MB": the name, the separator and the sizes are spans of their own, so that styles.css
  // puts a long name on a line of its own, cut short, with the sizes still in sight under it (P6-A: a narrow composer). The text, and the
  // title that shows it whole, are the one line.
  const describe = (meta, name, file, img) => {
    const span = (cls, text) => { const s = doc.createElement('span'); s.setAttribute('class', cls); s.textContent = text; return s; };
    const sizes = sizesOf(file, img);
    meta.replaceChildren(span('preview-name', name), ...(sizes ? [span('preview-sep', ' · '), span('preview-size', sizes)] : []));
    meta.setAttribute('title', meta.textContent);
  };
  const set = (file) => {
    const err = checkFile(file);
    if (err) { onError(err); return; }
    if (url) urls.revokeObjectURL(url);
    current = file;
    url = urls.createObjectURL(file);
    const name = file.name || 'pasted image';
    const img = doc.createElement('img');
    const meta = doc.createElement('span');
    const remove = doc.createElement('button');
    img.setAttribute('alt', `Preview of ${name}`);
    img.addEventListener('load', () => describe(meta, name, file, img));   // a picture replaced since writes to a line no longer shown
    img.setAttribute('src', url);
    describe(meta, name, file, null);
    remove.setAttribute('type', 'button');
    remove.setAttribute('aria-label', 'Remove image');
    remove.textContent = '×';
    remove.addEventListener('click', () => clear());
    preview.replaceChildren(img, meta, remove);
    preview.hidden = false;
    onChange(current);
  };
  const clear = () => {
    if (doc.activeElement && preview.contains(doc.activeElement)) well.focus();   // its Remove is about to go: the well brought the image in
    if (url) urls.revokeObjectURL(url);
    current = url = null;
    preview.replaceChildren();
    preview.hidden = true;
    input.value = '';
    onChange(null);
  };
  well.addEventListener('click', () => input.click());
  well.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); } });
  input.addEventListener('change', () => {
    const file = input.files?.[0];
    if (file) set(file);
    input.value = '';   // choosing the same file again must fire change again
  });
  for (const t of ['dragenter', 'dragover']) well.addEventListener(t, (e) => { e.preventDefault(); well.classList.add('dragging'); });
  for (const t of ['dragleave', 'drop']) well.addEventListener(t, () => well.classList.remove('dragging'));
  well.addEventListener('drop', (e) => { e.preventDefault(); const f = e.dataTransfer?.files?.[0]; if (f) set(f); });
  // A file dropped anywhere else must not navigate the page away from the chat; text may still be dropped into the note.
  for (const t of ['dragover', 'drop']) {
    doc.addEventListener(t, (e) => { if (Array.from(e.dataTransfer?.types ?? []).includes('Files')) e.preventDefault(); });
  }
  // A paste is the document's, wherever the focus is. Only an image is taken, and only then is the paste the page's: text pastes as text.
  // A composer made inert (the viewer is open over the page, P6-E) takes none.
  doc.addEventListener('paste', (e) => {
    if (well.closest?.('[inert]')) return;
    const f = pickImageFromClipboard(e.clipboardData && e.clipboardData.items);
    if (f) { e.preventDefault(); set(f); }
  });
  return { get: () => current, clear };
}
