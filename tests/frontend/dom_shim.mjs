// tests/frontend/dom_shim.mjs — the little DOM the render tests run on (CHAT_UI_PLAN.md P4-C). No dependencies.
//
// It covers what app/static/render.js builds a card with: createElement and createTextNode; setAttribute,
// getAttribute, hasAttribute, removeAttribute; append, replaceChildren, replaceWith, remove, contains, closest;
// textContent (read, and write); classList; children; hidden and disabled as properties that follow their attributes;
// addEventListener with dispatchEvent and click, events bubbling to the parents; document.getElementById;
// querySelector and querySelectorAll over tag, #id, .class, [attr], [attr=value] and the descendant
// and child combinators; focus, blur and document.activeElement, with a browser's rules for what can be focused: a
// button, an input, an enabled select or textarea, a link with an href, or anything with a tabindex, and only while
// it is in the document (the body of the installed document) and neither it nor anything above it has the hidden
// attribute: a card that was replaced has lost its focus, and so has a control that is hidden under a focus holder.
// It follows the DOM where a test could tell: append turns a string into a text node and never parses it (so a report
// that contains markup stays text), and a value that is not a node is made a string, so an `undefined` or `false`
// that slips into a card shows up as that word.
//
// The HTML-string APIs are tripwires: reading or writing them throws, so a renderer that reached for one fails its
// test instead of passing on a shim that quietly accepts it. serialize(node) is the test's own way to look at markup:
// text is escaped as a browser does, so an `<img` in the output is an element and `&lt;img` is text.
//
// installDom() puts a fresh document on globalThis for one test file and returns restore().

const VOID_TAGS = new Set(['area', 'br', 'col', 'hr', 'img', 'input', 'link', 'meta', 'wbr']);
const FOCUSABLE_TAGS = new Set(['button', 'input', 'select', 'textarea']);
let documentBody = null;   // the body of the document installed last: focus works inside it only
let focused = null;        // the element that has focus, if it is still in that body
const tripwire = (name) => { throw new Error(`the DOM shim has no ${name}: a renderer must build nodes, not parse strings`); };
// The element or an ancestor has the hidden attribute: nothing in it is rendered, so nothing in it can have focus.
const inHiddenSubtree = (node) => { for (let n = node; n && n.nodeType === 1; n = n.parentNode) if (n.hasAttribute('hidden')) return true; return false; };

export class ShimEvent {
  constructor(type, init = {}) {
    Object.assign(this, init);   // key, and whatever else a test wants the handler to read
    this.type = type;
    this.bubbles = !!init.bubbles;
    this.cancelable = !!init.cancelable;
    this.defaultPrevented = false;
    this.cancelBubble = false;
    this.target = null;
    this.currentTarget = null;
  }
  preventDefault() { if (this.cancelable) this.defaultPrevented = true; }
  stopPropagation() { this.cancelBubble = true; }
}

const toNode = (value) => (value instanceof ShimNode ? value : new ShimText(value));   // String(value), as the DOM does

function detach(node) {
  const parent = node.parentNode;
  if (parent) {
    parent.childNodes.splice(parent.childNodes.indexOf(node), 1);
    node.parentNode = null;
  }
}

class ShimNode {
  constructor() {
    this.parentNode = null;
    this.childNodes = [];
  }
  get textContent() { return this.childNodes.map((n) => n.textContent).join(''); }
  contains(other) {
    for (let n = other; n; n = n.parentNode) if (n === this) return true;
    return false;
  }
  remove() { detach(this); }
  replaceWith(...nodes) {
    const parent = this.parentNode;
    if (!parent) return;
    const fresh = nodes.map(toNode);
    fresh.forEach(detach);   // before the index is read: a node that was a sibling of this one moves it
    parent.childNodes.splice(parent.childNodes.indexOf(this), 1, ...fresh);
    fresh.forEach((n) => { n.parentNode = parent; });
    this.parentNode = null;
  }
}

class ShimText extends ShimNode {
  constructor(data) {
    super();
    this.nodeType = 3;
    this.data = String(data);
  }
  get textContent() { return this.data; }
}

function classListOf(element) {
  const read = () => (element.getAttribute('class') ?? '').split(/\s+/).filter(Boolean);
  const write = (names) => element.setAttribute('class', names.join(' '));
  const list = {
    add: (...names) => { const now = read(); for (const n of names) if (!now.includes(n)) now.push(n); write(now); },
    remove: (...names) => write(read().filter((n) => !names.includes(n))),
    contains: (name) => read().includes(name),
    toggle: (name, force) => {
      const want = force ?? !read().includes(name);
      if (want) list.add(name); else list.remove(name);
      return want;
    },
    [Symbol.iterator]: () => read()[Symbol.iterator](),
  };
  return list;
}

// ---- selectors -----------------------------------------------------------------------------------------------------

function splitTop(source, separators) {   // split outside [...] and quotes; -> [{ text, sep }] with the separator that followed
  const out = [];
  let depth = 0;
  let quote = '';
  let start = 0;
  for (let i = 0; i < source.length; i++) {
    const c = source[i];
    if (quote) { if (c === quote) quote = ''; continue; }
    if (c === '"' || c === "'") quote = c;
    else if (c === '[') depth++;
    else if (c === ']') depth--;
    else if (depth === 0 && separators.test(c)) {
      out.push({ text: source.slice(start, i), sep: c });
      start = i + 1;
    }
  }
  out.push({ text: source.slice(start), sep: '' });
  return out;
}

function parseCompound(source) {
  const head = /^(\*|[A-Za-z][\w-]*)?/.exec(source);
  const compound = { tag: head[1] && head[1] !== '*' ? head[1].toLowerCase() : null, id: null, classes: [], attrs: [] };
  const rest = source.slice(head[0].length);
  const piece = /#([\w-]+)|\.([\w-]+)|\[\s*([\w-]+)\s*(?:=\s*(?:"([^"]*)"|'([^']*)'|([^\]\s]*)))?\s*\]/y;
  let at = 0;
  while (at < rest.length) {
    piece.lastIndex = at;
    const m = piece.exec(rest);
    if (!m) throw new Error(`the DOM shim cannot read the selector part "${rest.slice(at)}" of "${source}"`);
    if (m[1]) compound.id = m[1];
    else if (m[2]) compound.classes.push(m[2]);
    else compound.attrs.push({ name: m[3].toLowerCase(), value: m[4] ?? m[5] ?? m[6] ?? null });
    at = piece.lastIndex;
  }
  return compound;
}

function parseComplex(source) {   // [{ combinator, compound }]: how each compound relates to the one on its left
  const parts = [];
  let combinator = null;
  let pendingChild = false;
  for (const { text, sep } of splitTop(source.trim(), /[\s>]/)) {
    if (text !== '') {
      parts.push({ combinator: parts.length ? (pendingChild ? '>' : ' ') : null, compound: parseCompound(text) });
      pendingChild = false;
    }
    if (sep === '>') pendingChild = true;
  }
  return parts;
}

function matchesCompound(node, c) {
  if (node.nodeType !== 1) return false;
  if (c.tag && node.localName !== c.tag) return false;
  if (c.id && node.getAttribute('id') !== c.id) return false;
  if (!c.classes.every((name) => node.classList.contains(name))) return false;
  return c.attrs.every((a) => (a.value === null ? node.hasAttribute(a.name) : node.getAttribute(a.name) === a.value));
}

function matchesFrom(node, parts, i) {   // node matches parts[i]: does the rest of the selector match above it?
  if (i === 0) return true;
  const left = parts[i - 1].compound;
  if (parts[i].combinator === '>') {
    const parent = node.parentNode;
    return !!parent && matchesCompound(parent, left) && matchesFrom(parent, parts, i - 1);
  }
  for (let p = node.parentNode; p && p.nodeType === 1; p = p.parentNode) {
    if (matchesCompound(p, left) && matchesFrom(p, parts, i - 1)) return true;
  }
  return false;
}

const compile = (selector) => splitTop(selector, /,/).map(({ text }) => parseComplex(text)).filter((parts) => parts.length);

function matchesAny(node, complexes) {
  return complexes.some((parts) => matchesCompound(node, parts.at(-1).compound) && matchesFrom(node, parts, parts.length - 1));
}

// ---- elements --------------------------------------------------------------------------------------------------------

class ShimElement extends ShimNode {
  constructor(tag) {
    super();
    this.nodeType = 1;
    this.localName = String(tag).toLowerCase();
    this.tagName = this.localName.toUpperCase();
    this.attrs = new Map();
    this.listeners = new Map();
    this.classList = classListOf(this);
  }
  get children() { return this.childNodes.filter((n) => n.nodeType === 1); }
  set textContent(value) {
    const text = String(value);
    this.replaceChildren(...(text === '' ? [] : [text]));
  }
  get textContent() { return super.textContent; }
  focus() {
    const focusable = this.hasAttribute('tabindex') || (FOCUSABLE_TAGS.has(this.localName) && !this.hasAttribute('disabled'))
      || (this.localName === 'a' && this.hasAttribute('href'));
    if (focusable && documentBody && documentBody.contains(this) && !inHiddenSubtree(this)) focused = this;
  }
  blur() { if (focused === this) focused = null; }
  get id() { return this.getAttribute('id') ?? ''; }
  get className() { return this.getAttribute('class') ?? ''; }
  get hidden() { return this.hasAttribute('hidden'); }
  set hidden(on) { if (on) this.setAttribute('hidden', ''); else this.removeAttribute('hidden'); }
  get disabled() { return this.hasAttribute('disabled'); }
  set disabled(on) { if (on) this.setAttribute('disabled', ''); else this.removeAttribute('disabled'); }

  setAttribute(name, value) { this.attrs.set(String(name).toLowerCase(), String(value)); }
  getAttribute(name) { const k = String(name).toLowerCase(); return this.attrs.has(k) ? this.attrs.get(k) : null; }
  hasAttribute(name) { return this.attrs.has(String(name).toLowerCase()); }
  removeAttribute(name) { this.attrs.delete(String(name).toLowerCase()); }

  append(...nodes) {
    for (const node of nodes.map(toNode)) {
      detach(node);
      node.parentNode = this;
      this.childNodes.push(node);
    }
  }
  replaceChildren(...nodes) {
    for (const child of [...this.childNodes]) detach(child);
    this.append(...nodes);
  }

  addEventListener(type, listener) {
    if (typeof listener !== 'function') throw new TypeError(`the listener for "${type}" is not a function`);
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(listener);
  }
  removeEventListener(type, listener) {
    this.listeners.set(type, (this.listeners.get(type) ?? []).filter((l) => l !== listener));
  }
  dispatchEvent(event) {
    if (!event.target) event.target = this;
    for (let node = this; node && node.nodeType === 1; node = node.parentNode) {
      event.currentTarget = node;
      for (const listener of [...(node.listeners.get(event.type) ?? [])]) listener.call(node, event);
      if (!event.bubbles || event.cancelBubble) break;
    }
    return !event.defaultPrevented;
  }
  click() { return this.dispatchEvent(new ShimEvent('click', { bubbles: true, cancelable: true })); }

  matches(selector) { return matchesAny(this, compile(selector)); }
  closest(selector) {
    const complexes = compile(selector);
    for (let n = this; n && n.nodeType === 1; n = n.parentNode) if (matchesAny(n, complexes)) return n;
    return null;
  }
  querySelectorAll(selector) {
    const complexes = compile(selector);
    const found = [];
    const walk = (node) => {
      for (const child of node.children) {
        if (matchesAny(child, complexes)) found.push(child);
        walk(child);
      }
    };
    walk(this);
    return found;
  }
  querySelector(selector) { return this.querySelectorAll(selector)[0] ?? null; }

  get innerHTML() { return tripwire('innerHTML'); }
  set innerHTML(_) { tripwire('innerHTML'); }
  get outerHTML() { return tripwire('outerHTML'); }
  set outerHTML(_) { tripwire('outerHTML'); }
  insertAdjacentHTML() { return tripwire('insertAdjacentHTML'); }
}

export function createDocument() {
  const body = new ShimElement('body');
  documentBody = body;
  focused = null;
  return {
    get activeElement() {
      if (focused && (!body.contains(focused) || inHiddenSubtree(focused))) focused = null;   // a browser blurs what leaves the page or is hidden, for good
      return focused ?? body;
    },
    createElement: (tag) => new ShimElement(tag),
    createTextNode: (data) => new ShimText(data),
    body,
    getElementById: (id) => body.querySelector(`#${id}`),
    querySelector: (selector) => body.querySelector(selector),
    querySelectorAll: (selector) => body.querySelectorAll(selector),
    write: () => tripwire('document.write'),
  };
}

export function installDom() {
  const had = Object.hasOwn(globalThis, 'document');
  const previous = globalThis.document;
  globalThis.document = createDocument();
  return () => { if (had) globalThis.document = previous; else delete globalThis.document; };
}

const escapeText = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
const escapeAttr = (s) => s.replace(/&/g, '&amp;').replace(/"/g, '&quot;');

export function serialize(node) {
  if (node.nodeType === 3) return escapeText(node.data);
  const attrs = [...node.attrs].map(([k, v]) => ` ${k}="${escapeAttr(v)}"`).join('');
  if (VOID_TAGS.has(node.localName)) return `<${node.localName}${attrs}>`;
  return `<${node.localName}${attrs}>${node.childNodes.map(serialize).join('')}</${node.localName}>`;
}
