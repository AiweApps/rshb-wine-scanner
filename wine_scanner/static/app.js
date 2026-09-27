'use strict';

const MAX_BYTES = 20 * 1024 * 1024;
const TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const SITE_PREFIX = 'https://vino-svoe.ru/wines/';
const AUTO_RETRIES = 3;
const REQUEST_TIMEOUT_MS = 200000;
const SVG = 'http://www.w3.org/2000/svg';

const $ = (id) => document.getElementById(id);
const el = {
  status: $('status'), statusText: $('status-text'), drop: $('drop'), file: $('file'), photo: $('photo'),
  frame: $('frame'), preview: $('preview'), overlay: $('overlay'), caption: $('photo-caption'),
  tools: $('stage-tools'), drawBtn: $('draw-btn'), newBtn: $('new-btn'), drawHint: $('draw-hint'),
  idle: $('idle'), loading: $('loading'), loadingTitle: $('loading-title'), loadingNote: $('loading-note'),
  error: $('error'), errorTitle: $('error-title'), errorText: $('error-text'), retryBtn: $('retry-btn'),
  resetBtn: $('reset-btn'), outcome: $('outcome'), notice: $('notice'), tabs: $('tabs'), bottle: $('bottle'),
  tech: $('tech-list'), download: $('download-btn'), cardTpl: $('card-tpl'),
};

const state = {
  file: null, url: null, size: null, pending: null, response: null, overview: null,
  selected: null, drawing: false, draft: null, autoRetries: 0, timer: null, controller: null, status: null,
};

// ---------- service status ----------
async function refreshStatus() {
  try {
    const r = await fetch('/api/status', { cache: 'no-store' });
    const s = await r.json();
    state.status = s;
    const q = s.queue || {};
    el.status.className = 'status';
    if (!s.backend_reachable) {
      el.status.classList.add('down'); el.statusText.textContent = 'сервис распознавания недоступен';
    } else if (!s.profile_match) {
      el.status.classList.add('down'); el.statusText.textContent = 'сервис временно недоступен';
    } else if (q.in_flight) {
      el.status.classList.add('busy');
      el.statusText.textContent = 'занят' + (q.waiting ? ` · в очереди ${q.waiting}` : '');
    } else {
      el.status.classList.add('ready'); el.statusText.textContent = 'готов';
    }
  } catch (e) {
    el.status.className = 'status down'; el.statusText.textContent = 'нет связи с интерфейсом';
  }
}

// ---------- photo selection ----------
function pick(file) {
  if (!file) return;
  reset(true);
  const name = (file.name || '').toLowerCase();
  if (/\.(heic|heif)$/.test(name) || /heic|heif/.test(file.type)) {
    return fail('Формат HEIC не поддерживается',
      'Сохраните фото как JPEG. На iPhone: Настройки → Камера → Форматы → «Наиболее совместимый», или отправьте фото через «Экспорт» в JPEG.', false);
  }
  if (file.type && !TYPES.includes(file.type)) {
    return fail('Неподдерживаемый формат', 'Нужен JPEG, PNG или WebP.', false);
  }
  if (file.size > MAX_BYTES) {
    return fail('Файл слишком большой', `Размер ${(file.size / 1048576).toFixed(1)} МиБ, допустимо до 20 МиБ.`, false);
  }
  state.file = file;
  state.url = URL.createObjectURL(file);
  el.preview.src = state.url;
  el.drop.hidden = true; el.photo.hidden = false; el.tools.hidden = false;
  el.caption.textContent = `${file.name || 'фото'} · ${(file.size / 1048576).toFixed(1)} МиБ`;
  el.preview.onload = () => { drawOverlay(); };
  recognize(null);
}

el.file.addEventListener('change', () => pick(el.file.files[0]));
['dragenter', 'dragover'].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.add('over'); }));
['dragleave', 'drop'].forEach((t) => el.drop.addEventListener(t, (e) => { e.preventDefault(); el.drop.classList.remove('over'); }));
el.drop.addEventListener('drop', (e) => pick(e.dataTransfer.files[0]));
el.newBtn.addEventListener('click', () => { reset(true); el.file.click(); });
el.resetBtn.addEventListener('click', () => { reset(true); el.file.click(); });
el.retryBtn.addEventListener('click', () => { state.autoRetries = 0; state.file ? recognize(state.pending) : el.file.click(); });

function reset(clearPhoto) {
  clearTimeout(state.timer);
  if (state.controller) state.controller.abort();
  Object.assign(state, { controller: null, pending: null, response: null, overview: null, selected: null, draft: null, autoRetries: 0 });
  setDrawing(false);
  show('idle');
  if (clearPhoto) {
    if (state.url) URL.revokeObjectURL(state.url);
    Object.assign(state, { file: null, url: null, size: null });
    el.preview.removeAttribute('src'); el.overlay.replaceChildren();
    el.photo.hidden = true; el.tools.hidden = true; el.drop.hidden = false; el.file.value = '';
  }
}

function show(which) {
  for (const k of ['idle', 'loading', 'error', 'outcome']) el[k].hidden = k !== which;
}

// ---------- recognition ----------
async function recognize(roi) {
  clearTimeout(state.timer);
  if (state.controller) state.controller.abort();
  state.pending = roi;
  show('loading');
  el.loadingTitle.textContent = roi ? 'Распознаём выбранную бутылку…' : 'Читаем этикетку…';
  el.loadingNote.textContent = 'Обычно несколько секунд; фото никуда не сохраняется.';
  el.retryBtn.hidden = false;
  const form = new FormData();
  form.append('image', state.file, state.file.name || 'photo');
  if (roi) form.append('target_roi', JSON.stringify(roi));
  const controller = new AbortController();
  state.controller = controller;
  const timeout = setTimeout(() => controller.abort('timeout'), REQUEST_TIMEOUT_MS);
  let r, body;
  try {
    r = await fetch('/api/recognize', { method: 'POST', body: form, signal: controller.signal });
    body = await r.json().catch(() => ({}));
  } catch (e) {
    if (state.controller !== controller) return;
    return fail('Нет ответа', controller.signal.aborted ? 'Время ожидания истекло. Повторите запрос.' : 'Интерфейс недоступен. Проверьте, что он запущен, и повторите.');
  } finally {
    clearTimeout(timeout);
  }
  if (state.controller !== controller) return;
  state.controller = null;
  refreshStatus();
  if (r.ok) {
    state.autoRetries = 0;
    state.response = body;
    if (!roi) state.overview = body;
    state.selected = initialSelection(body.view);
    return render();
  }
  if (r.status === 503 && body.retry_after && state.autoRetries < AUTO_RETRIES) {
    state.autoRetries += 1;
    return countdown(body.retry_after, body.message);
  }
  const titles = { 413: 'Слишком большое изображение', 415: 'Формат не поддерживается', 422: 'Изображение не обработано',
    502: 'Ошибка сервиса', 503: 'Сервис занят или недоступен', 504: 'Время ожидания истекло' };
  const retryable = ![413, 415].includes(r.status) && body.error !== 'bad_roi';
  fail(titles[r.status] || `Ошибка ${r.status}`, body.message || 'Повторите попытку.', retryable);
}

function countdown(seconds, message) {
  let left = Math.max(1, Math.round(seconds));
  show('loading');
  el.loadingTitle.textContent = message || 'Сервис занят';
  const tick = () => {
    el.loadingNote.textContent = `Повторим автоматически через ${left} с (попытка ${state.autoRetries} из ${AUTO_RETRIES}).`;
    if (left-- <= 0) return recognize(state.pending);
    state.timer = setTimeout(tick, 1000);
  };
  tick();
}

function fail(title, text, retryable = true) {
  show('error');
  el.errorTitle.textContent = title;
  el.errorText.textContent = text;
  el.retryBtn.hidden = !retryable || !state.file;
}

function initialSelection(view) {
  if (view.bottles.length === 1) return view.bottles[0].instance_id;
  if (view.mode === 'explicit_roi' && view.primary_instance_id) return view.primary_instance_id;
  return null;
}

// ---------- rendering ----------
function render() {
  const view = state.response.view;
  state.size = view.frame_size || view.image.size;
  show('outcome');
  renderNotice(view);
  renderTabs(view);
  renderBottle(view);
  renderTech(view);
  drawOverlay();
}

function renderNotice(view) {
  const n = view.bottles.length;
  el.notice.replaceChildren();
  const add = (strong, text) => {
    const s = document.createElement('strong'); s.textContent = strong;
    el.notice.append(s);
    if (text) el.notice.append(document.createTextNode(' ' + text));
  };
  if (view.mode === 'explicit_roi') {
    add('Распознано по выбранной рамке.', n ? 'Ниже — ответ для этой области.' : 'В рамке не найдена бутылка с этикеткой — попробуйте обвести шире.');
    const back = button('ghost small', 'Все бутылки на фото', () => { state.response = state.overview; state.selected = null; render(); });
    back.hidden = !state.overview;
    el.notice.append(document.createElement('br'), back);
  } else if (n === 0) {
    add('Бутылка с этикеткой не найдена.', 'Снимите этикетку крупнее и ровнее или обведите бутылку вручную кнопкой «Выделить рамку».');
  } else if (n === 1) {
    add('Найдена одна бутылка.', '');
  } else {
    add(`На фото ${n} бутыл${plural(n, 'ка', 'ки', 'ок')}.`,
      'У каждой свой ответ. Выберите нужную бутылку — рамкой на фото или вкладкой ниже.');
  }
  el.notice.hidden = false;
}

function renderTabs(view) {
  el.tabs.replaceChildren();
  el.tabs.hidden = view.bottles.length < 2;
  if (el.tabs.hidden) return;
  el.tabs.append(tab('Все', null, state.selected === null));
  for (const b of view.bottles) {
    const hint = b.primary && view.primary_basis === 'central_suggestion' ? ' · у центра' : '';
    el.tabs.append(tab(`Бутылка ${b.number}`, b.instance_id, state.selected === b.instance_id, hint));
  }
}

function tab(label, id, selected, hint = '') {
  const t = button('tab', label, () => select(id));
  t.setAttribute('role', 'tab'); t.setAttribute('aria-selected', String(selected));
  if (hint) { const h = document.createElement('span'); h.className = 'hint'; h.textContent = hint; t.append(h); }
  return t;
}

function select(id) {
  state.selected = id;
  const view = state.response.view;
  renderTabs(view); renderBottle(view); renderTech(view); drawOverlay();
}

function renderBottle(view) {
  el.bottle.replaceChildren();
  const bottle = view.bottles.find((b) => b.instance_id === state.selected);
  if (!bottle) {
    if (!view.bottles.length) {
      el.bottle.append(emptyCard(view.decision_text || 'Совпадение не найдено'));
      return;
    }
    const list = document.createElement('div'); list.className = 'alts overview';
    for (const b of view.bottles) {
      const tag = `Бутылка ${b.number}` + (b.primary && view.primary_basis === 'central_suggestion' ? ' · у центра' : '');
      const c = b.best ? card(b.best, tag) : emptyCard(`Бутылка ${b.number}: совпадение не найдено`);
      c.classList.add('pickable'); c.tabIndex = 0; c.setAttribute('role', 'button');
      c.addEventListener('click', (e) => { if (!e.target.closest('a')) select(b.instance_id); });
      c.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(b.instance_id); } });
      list.append(c);
    }
    el.bottle.append(list);
    return;
  }
  const head = document.createElement('div'); head.className = 'bottle-head';
  const h = document.createElement('h3');
  h.textContent = view.bottles.length > 1 || view.mode === 'explicit_roi' ? `Бутылка ${bottle.number}` : 'Ответ сканера';
  head.append(h);
  if (bottle.decision_text) {
    const verdict = document.createElement('span'); verdict.className = 'verdict'; verdict.textContent = bottle.decision_text;
    head.append(verdict);
  }
  if (view.mode === 'automatic' && view.bottles.length > 1 && bottle.select_roi) {
    const acts = document.createElement('div'); acts.className = 'actions';
    acts.append(button('ghost small', 'Распознать только её', () => recognize(bottle.select_roi)));
    head.append(acts);
  }
  el.bottle.append(head);
  el.bottle.append(bottle.best ? card(bottle.best, 'лучшее совпадение', true) : emptyCard('Для этой бутылки совпадение не найдено.'));
  if (bottle.alternatives.length) {
    const t = document.createElement('h4'); t.className = 'alts-title';
    t.textContent = 'Похожие варианты';
    const list = document.createElement('div'); list.className = 'alts';
    bottle.alternatives.forEach((a) => list.append(card(a)));
    el.bottle.append(t, list);
  }
}

function card(info, tag = '', best = false) {
  const node = el.cardTpl.content.firstElementChild.cloneNode(true);
  if (best) node.classList.add('best');
  const ref = node.querySelector('.card-ref'), img = ref.querySelector('img');
  if (info.reference) {
    img.src = info.reference; img.alt = `Эталон: ${info.title}`;
    img.addEventListener('error', () => ref.classList.add('empty'));
  } else ref.classList.add('empty');
  node.querySelector('.card-tag').textContent = tag;
  node.querySelector('.card-title').textContent = info.title;
  node.querySelector('.card-producer').textContent = info.producer || '';
  node.querySelector('.card-meta').textContent = [info.category, info.region, info.grapes].filter(Boolean).join(' · ');
  const link = node.querySelector('.card-link');
  if (info.page_url && info.page_url.startsWith(SITE_PREFIX)) {
    const a = document.createElement('a');
    a.href = info.page_url; a.target = '_blank'; a.rel = 'noopener noreferrer';
    a.textContent = 'Открыть на «Своём вине» ↗';
    link.append(a);
  } else {
    const na = document.createElement('span'); na.className = 'na';
    na.textContent = 'Адрес страницы на «Своём вине» в каталоге не указан';
    link.append(na);
  }
  return node;
}

function emptyCard(text) {
  const d = document.createElement('div'); d.className = 'empty-card'; d.textContent = text; return d;
}

function renderTech(view) {
  const bottle = view.bottles.find((b) => b.instance_id === state.selected);
  const best = bottle ? bottle.best : view.published_card;
  const status = state.status || {};
  const snapshot = { true: 'есть в снимках сайта 17–21.09', false: 'не было в снимках сайта 17–21.09' };
  const rows = [
    ['решение', view.decision], ['основание цели', view.primary_basis], ['режим', view.mode],
    ['raw.slug', view.raw_slug === null ? 'null' : view.raw_slug], ['best_candidate', view.best_slug],
    ['slug карточки', best && best.slug], ['адрес страницы', best && best.page_url],
    ['источник адреса', best && best.page_source], ['снимок сайта', best && snapshot[best.page_in_site_snapshot]],
    ['рамка запроса', view.roi ? view.roi.map((v) => Math.round(v)).join(', ') : '—'], ['бутылок', view.bottle_count],
    ['кадр (EXIF)', (view.frame_size || view.image.size || []).join(' × ')],
    ['файл', view.image ? `${view.image.format} · ${view.image.bytes} байт` : '—'],
    ['вероятность', 'не рассчитывается (без калибровки)'],
    ['время сервиса', view.backend_ms ? `${(view.backend_ms / 1000).toFixed(1)} с` : '—'],
    ['профиль ответа', view.profile_checksum], ['профиль выпуска', status.expected_profile],
    ['профиль backend', status.runtime_descriptor], ['причины', (view.reasons || []).join(', ') || '—'],
  ];
  el.tech.replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement('dt'), dd = document.createElement('dd');
    dt.textContent = k; dd.textContent = String(v ?? '—'); return [dt, dd];
  }));
}

el.download.addEventListener('click', () => {
  if (!state.response) return;
  const blob = new Blob([JSON.stringify(state.response.view, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `scan-${new Date().toISOString().replace(/[:.]/g, '-')}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
});

// ---------- overlay & manual ROI (EXIF-oriented original pixels) ----------
function frameSize() {
  return state.size || (el.preview.naturalWidth ? [el.preview.naturalWidth, el.preview.naturalHeight] : null);
}

function drawOverlay() {
  const size = frameSize();
  el.overlay.replaceChildren();
  if (!size) return;
  const [w, h] = size;
  el.overlay.setAttribute('viewBox', `0 0 ${w} ${h}`);
  el.overlay.setAttribute('preserveAspectRatio', 'xMidYMid meet');
  const nw = el.preview.naturalWidth, nh = el.preview.naturalHeight;
  const mismatch = nw && Math.abs(nw / nh - w / h) > 0.02;
  if (state.file) {
    el.caption.textContent = `${state.file.name || 'фото'} · ${(state.file.size / 1048576).toFixed(1)} МиБ` +
      (mismatch ? ' · браузер показал фото без EXIF-поворота, рамки могут не совпасть' : '');
  }
  const view = state.response && state.response.view;
  const unit = Math.max(w, h) / 42;
  if (view && view.roi) {
    const [x1, y1, x2, y2] = view.roi;
    el.overlay.append(svg('rect', { class: 'roi', x: x1, y: y1, width: x2 - x1, height: y2 - y1 }));
  }
  if (view) {
    for (const b of view.bottles) {
      if (!b.geometry) continue;
      const [x1, y1, x2, y2] = b.geometry;
      const active = state.selected === b.instance_id;
      el.overlay.append(svg('rect', { class: 'box-shadow', x: x1, y: y1, width: x2 - x1, height: y2 - y1, rx: unit / 3 }));
      const r = svg('rect', { class: 'box' + (active ? ' active' : ''), x: x1, y: y1, width: x2 - x1, height: y2 - y1,
        rx: unit / 3, tabindex: 0, role: 'button', 'aria-label': `Бутылка ${b.number}` });
      r.addEventListener('click', () => { if (!state.drawing) select(b.instance_id); });
      r.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); select(b.instance_id); } });
      el.overlay.append(r);
      if (view.bottles.length > 1 || view.mode === 'explicit_roi') {
        const cx = x1 + unit * 1.1, cy = y1 + unit * 1.1;
        el.overlay.append(svg('circle', { class: 'badge-bg' + (active ? ' active' : ''), cx, cy, r: unit }));
        const t = svg('text', { class: 'badge', x: cx, y: cy, 'text-anchor': 'middle', 'dominant-baseline': 'central', 'font-size': unit * 1.1 });
        t.textContent = b.number; el.overlay.append(t);
      }
    }
  }
  if (state.draft) {
    const [x1, y1, x2, y2] = state.draft;
    el.overlay.append(svg('rect', { class: 'roi', x: x1, y: y1, width: x2 - x1, height: y2 - y1 }));
  }
}

function svg(name, attrs) {
  const n = document.createElementNS(SVG, name);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  return n;
}

function toImage(event) {
  const size = frameSize();
  const rect = el.preview.getBoundingClientRect();
  const scale = Math.min(rect.width / size[0], rect.height / size[1]);
  const ox = rect.left + (rect.width - size[0] * scale) / 2, oy = rect.top + (rect.height - size[1] * scale) / 2;
  const clamp = (v, m) => Math.max(0, Math.min(m, v));
  return [clamp((event.clientX - ox) / scale, size[0]), clamp((event.clientY - oy) / scale, size[1]), scale];
}

function setDrawing(on) {
  state.drawing = on;
  el.drawBtn.setAttribute('aria-pressed', String(on));
  el.drawBtn.textContent = on ? 'Отменить выделение' : 'Выделить рамку';
  el.frame.classList.toggle('drawing', on);
  el.drawHint.hidden = !on;
  if (!on) { state.draft = null; drawOverlay(); }
}

el.drawBtn.addEventListener('click', () => setDrawing(!state.drawing));
let start = null;
el.frame.addEventListener('pointerdown', (e) => {
  if (!state.drawing || !frameSize()) return;
  e.preventDefault();
  el.frame.setPointerCapture(e.pointerId);
  start = toImage(e);
  state.draft = null;
});
el.frame.addEventListener('pointermove', (e) => {
  if (!start) return;
  const [x, y] = toImage(e);
  state.draft = [Math.min(start[0], x), Math.min(start[1], y), Math.max(start[0], x), Math.max(start[1], y)];
  drawOverlay();
});
const finish = () => {
  if (!start) return;
  const scale = start[2];
  start = null;
  const d = state.draft;
  if (!d || (d[2] - d[0]) * scale < 16 || (d[3] - d[1]) * scale < 16) { state.draft = null; drawOverlay(); return; }
  setDrawing(false);
  recognize(d);
};
el.frame.addEventListener('pointerup', finish);
el.frame.addEventListener('pointercancel', () => { start = null; state.draft = null; drawOverlay(); });
window.addEventListener('keydown', (e) => { if (e.key === 'Escape' && state.drawing) setDrawing(false); });

// ---------- helpers ----------
function button(cls, label, onClick) {
  const b = document.createElement('button'); b.type = 'button'; b.className = cls; b.textContent = label;
  b.addEventListener('click', onClick); return b;
}
function plural(n, one, few, many) {
  const m10 = n % 10, m100 = n % 100;
  if (m10 === 1 && m100 !== 11) return one;
  if (m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14)) return few;
  return many;
}

el.status.addEventListener('click', refreshStatus);
refreshStatus();
setInterval(() => { if (!document.hidden) refreshStatus(); }, 15000);
