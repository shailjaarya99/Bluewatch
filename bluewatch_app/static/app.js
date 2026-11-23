/* BlueWatch page logic. Every control on the page is wired to something real. */
(() => {
  'use strict';
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const nf = new Intl.NumberFormat('en-US');
  const pct = (x, d = 1) => (x * 100).toFixed(d) + '%';
  const J = { busy: b => window.Jelly && window.Jelly.busy(b), bloom: () => window.Jelly && window.Jelly.bloom() };

  // class id -> [name, colour]. Names and ids follow the MADOS dataset.
  const CLASS = [null,
    ['Marine Debris', '#ff4fd8'], ['Dense Sargassum', '#0b8f4d'], ['Sparse Floating Algae', '#8fd14f'],
    ['Natural Organic Material', '#7a4a24'], ['Ship', '#ffe14d'], ['Oil Spill', '#ff4b1f'], ['Marine Water', '#1b3a6b'],
    ['Sediment-Laden Water', '#c8a165'], ['Foam', '#ffffff'], ['Turbid Water', '#4aa3d6'], ['Shallow Water', '#2ec4b6'],
    ['Waves & Wakes', '#9fa8da'], ['Oil Platform', '#ff9f1c'], ['Jellyfish', '#d9a7ff'], ['Sea snot', '#6b705c']];
  const OIL = 6, WATER = 7;
  const PAL = new Uint8Array(16 * 3);
  CLASS.forEach((c, i) => { if (c) for (let k = 0; k < 3; k++) PAL[i * 3 + k] = parseInt(c[1].slice(1 + 2 * k, 3 + 2 * k), 16); });

  const dec = s => { const b = atob(s), u = new Uint8Array(b.length); for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i); return u; };
  const setMsg = (el, t, err) => { el.textContent = t || ''; el.classList.toggle('err', !!err); };
  const offline = "Can't reach the BlueWatch server. Start it again with: py app.py";
  const save = (blob, name) => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 2000);
  };

  /* ------------------------------------------------------------ legend rows */
  function legendHTML(counts, total, interactive, focusId) {
    const rows = counts.map((n, c) => ({ n, c })).filter(r => r.c > 0 && r.n > 0).sort((a, b) => b.n - a.n);
    if (!rows.length) return '<p class="note">Nothing to show.</p>';
    const tag = interactive ? 'button' : 'div';
    return rows.map(r => {
      const share = r.n / total, hex = CLASS[r.c][1];
      const on = interactive && focusId === r.c;
      const attrs = interactive ? ` type="button" aria-pressed="${on}"` : '';
      return `<${tag} class="lg${on ? ' on' : ''}" data-c="${r.c}"${attrs}>` +
        `<i class="sw" style="--c:${hex}"></i><span>${CLASS[r.c][0]}</span>` +
        `<span class="pc">${(share * 100).toFixed(share < 0.001 ? 2 : 1)}%</span>` +
        `<span class="bar"><u style="width:${Math.max(share * 100, 1.2)}%;--c:${hex}"></u></span></${tag}>`;
    }).join('');
  }
  const tile = (v, l, wide) => `<div class="score${wide ? ' wide' : ''}"><b>${v}</b><span>${l}</span></div>`;

  /* ------------------------------------------------------------ scene viewer */
  const mapCv = $('#map'), mctx = mapCv.getContext('2d');
  const MODELS = { unet: 'U-Net', rf: 'Random Forest' };
  function repFor(m) { return R && (m === 'unet' && R.ensemble ? R.ensemble : R[m]); }
  let S = null, view = 'overlay', focus = 0, currentId = null;
  let R = null, model = 'rf', lastReq = null;   // R = numbers from report.json

  function classify() {
    const n = S.w * S.h, cut = Math.round($('#thr').value / 100 * 255);
    const counts = new Array(16).fill(0), pred = S.pred;
    for (let i = 0; i < n; i++) { const c = S.oilp[i] >= cut ? OIL : S.nonoil[i]; pred[i] = c; counts[c]++; }
    S.counts = counts;
  }

  function draw() {
    const { w, h, rgb, pred, truth } = S, n = w * h;
    if (mapCv.width !== w || mapCv.height !== h) { mapCv.width = w; mapCv.height = h; }
    const img = mctx.createImageData(w, h), d = img.data;
    for (let i = 0; i < n; i++) {
      const c = view === 'truth' ? truth[i] : pred[i], o = c * 3;
      let r, g, b;
      if (view === 'sat') { r = rgb[3 * i]; g = rgb[3 * i + 1]; b = rgb[3 * i + 2]; }
      else if (view === 'pred') { r = PAL[o]; g = PAL[o + 1]; b = PAL[o + 2]; }
      else if (view === 'truth') { if (c) { r = PAL[o]; g = PAL[o + 1]; b = PAL[o + 2]; } else { r = 18; g = 4; b = 22; } }
      else { // overlay: clear water stays as the satellite sees it, everything else is tinted
        const sr = rgb[3 * i], sg = rgb[3 * i + 1], sb = rgb[3 * i + 2];
        if (c === WATER) { r = sr; g = sg; b = sb; }
        else { const k = 0.66; r = sr * (1 - k) + PAL[o] * k; g = sg * (1 - k) + PAL[o + 1] * k; b = sb * (1 - k) + PAL[o + 2] * k; }
      }
      if (focus && view !== 'sat' && c !== focus) { r = r * 0.18 + 8; g = g * 0.18; b = b * 0.18 + 12; }
      const j = 4 * i; d[j] = r; d[j + 1] = g; d[j + 2] = b; d[j + 3] = 255;
    }
    mctx.putImageData(img, 0, 0);
  }

  function renderLegend() {
    $('#legend').innerHTML = legendHTML(S.counts, S.w * S.h, true, focus);
    $('#legendHint').hidden = view === 'sat';
  }

  function renderAlert() {
    const p = S.counts[OIL] / (S.w * S.h);
    const rep = repFor(S.model);
    const trust = rep
      ? `Only about ${Math.round(rep.oil_precision * 10)} in 10 oil flags from the ${MODELS[S.model]} on unseen scenes were real oil, so confirm with a second source.`
      : 'Confirm with a second source.';
    let level, title, text;
    if (p < 0.005) { level = 'calm'; title = 'No significant oil flagged'; text = `Only ${pct(p, 2)} of this scene is called oil at this setting.`; }
    else if (p < 0.05) { level = 'watch'; title = 'A small oil-like area'; text = `${pct(p)} of the scene is flagged. Check the Satellite view before trusting it.`; }
    else if (p < 0.2) { level = 'likely'; title = 'Oil-like area found'; text = `${pct(p)} of the scene is flagged. ${trust}`; }
    else { level = 'large'; title = 'Large oil-like area'; text = `${pct(p)} of the scene is flagged. ${trust}`; }
    $('#alert').dataset.level = level;
    $('#alertTitle').textContent = title;
    $('#alertText').textContent = text + ' (Alert level is a simple rule on how much of the scene is flagged: 0.5%, 5%, 20%.)';
  }

  function renderScores() {
    const el = $('#scores');
    if (!S.truth) { el.hidden = true; $('#wrongBox').hidden = true; return; }
    let tp = 0, fp = 0, fn = 0, tn = 0, lab = 0, real = 0;
    const missedTo = new Array(16).fill(0), falseFrom = new Array(16).fill(0);
    const n = S.w * S.h;
    for (let i = 0; i < n; i++) {
      const t = S.truth[i]; if (!t) continue;
      lab++;
      const flagged = S.pred[i] === OIL;
      if (t === OIL) { real++; if (flagged) tp++; else { fn++; missedTo[S.pred[i]]++; } } else { if (flagged) { fp++; falseFrom[t]++; } else tn++; }
    }
    el.hidden = false;
    const top = a => a.map((n, c) => ({ n, c })).filter(r => r.c > 0 && r.n > 0).sort((x, y) => y.n - x.n)[0];
    const parts = [];
    if (real > 0) { const m = top(missedTo); parts.push(`${nf.format(fn)} of ${nf.format(real)} real oil pixels were missed (${pct(fn / real, 0)})` + (m ? `, mostly called ${CLASS[m.c][0]}.` : '.')); }
    if (fp > 0) { const f = top(falseFrom); parts.push(`${nf.format(fp)} pixels were flagged as oil by mistake` + (f ? `, mostly really ${CLASS[f.c][0]}.` : '.')); }
    if (!parts.length) parts.push('Nothing was missed and nothing was flagged by mistake among the labelled pixels of this scene.');
    $('#wrongText').textContent = parts.join(' ');
    $('#wrongBox').hidden = false;
    const div = (a, b) => b ? a / b : null;
    const show = (x, d = 0) => x === null ? 'n/a' : pct(x, d);
    const foot = `Scored on the ${nf.format(lab)} pixels experts labelled (${pct(lab / n, 1)} of this scene).`;
    if (real === 0) {
      el.innerHTML = tile(nf.format(fp), 'pixels wrongly flagged as oil') +
        tile(show(div(fp, fp + tn), 1), 'of the labelled pixels flagged by mistake') +
        tile('No real oil here', 'Every oil flag in this scene is a false alarm. This is the false alarm check.', true) +
        `<p class="note score wide" style="margin:0">${foot}</p>`;
    } else {
      el.innerHTML = tile(show(div(tp, tp + fn)), 'of the real oil caught') +
        tile(show(div(tp, tp + fp)), 'of the oil flags were right') +
        tile(show(div(fp, fp + tn), 1), 'of clean pixels wrongly flagged') +
        tile(div(tp, tp + fp + fn) === null ? 'n/a' : div(tp, tp + fp + fn).toFixed(2), 'overlap score for oil (IoU)') +
        `<p class="note score wide" style="margin:0">${foot}</p>`;
    }
  }

  let queued = false;
  function update() {
    classify(); draw(); renderLegend(); renderAlert(); renderScores();
  }
  function updateSoon() { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; if (S) update(); }); }

  function setView(v) {
    view = v;
    $$('#viewBtns button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === v)));
    if (S) { draw(); renderLegend(); }
  }

  function showScene(j) {
    const n = j.w * j.h;
    S = { w: j.w, h: j.h, oilp: dec(j.oilp), nonoil: dec(j.nonoil), rgb: dec(j.rgb), truth: j.truth ? dec(j.truth) : null, pred: new Uint8Array(n), name: j.name, model: j.model };
    setModel(j.model);
    focus = 0;
    $('#sceneName').textContent = j.name;
    $('#sceneSub').textContent = `${j.w} x ${j.h} pixels, read by the ${MODELS[j.model]} in ${j.seconds} s.` + (j.note ? ' ' + j.note : '');
    const tb = $('#viewBtns [data-view="truth"]');
    tb.hidden = !S.truth;
    if (!S.truth && view === 'truth') view = 'overlay';
    $$('#viewBtns button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.view === view)));
    $('#empty').hidden = true; $('#result').hidden = false;
    update();
    J.bloom();
    if (innerWidth < 981) $('#viewer').scrollIntoView({ behavior: 'smooth', block: 'start' });
  }

  function setBusy(on, text) {
    $('#busy').hidden = !on;
    if (text) $('#busyText').textContent = text;
    J.busy(on);
  }

  async function analyse(fd) {
    const msg = $('#scanMsg');
    setMsg(msg, '');
    setBusy(true, 'Reading 11 bands and classifying pixels...');
    try {
      const res = await fetch('/api/predict', { method: 'POST', body: fd });
      let j; try { j = await res.json(); } catch (e) { throw new Error('The server sent back something unexpected.'); }
      if (!res.ok) throw new Error(j.error || 'Something went wrong.');
      showScene(j);
    } catch (e) {
      setMsg(msg, e instanceof TypeError ? offline : e.message, true);
    } finally { setBusy(false); }
  }

  function scan(req) {
    lastReq = req;
    const fd = new FormData();
    if (req.sample) fd.append('sample', req.sample); else req.files.forEach(f => fd.append('files', f));
    fd.append('model', model);
    analyse(fd);
  }

  function setModel(m) {
    model = m;
    $$('#modelBtns button').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.model === m)));
    $('#modelNote').textContent = m === 'unet'
      ? 'Averages several U-Nets. Each also looks at the pixels around a pixel. Takes scenes up to 512 x 512.'
      : 'Looks at one pixel at a time, with five extra formulas. Takes bigger scenes.';
  }
  $('#modelBtns').addEventListener('click', e => {
    const b = e.target.closest('button');
    if (!b || b.dataset.model === model) return;
    setModel(b.dataset.model);
    if (lastReq) scan(lastReq);          // run the same scene again with the other model
  });

  /* events: map */
  $('#viewBtns').addEventListener('click', e => { const b = e.target.closest('button'); if (b) setView(b.dataset.view); });
  $('#thr').addEventListener('input', e => { $('#thrVal').textContent = e.target.value + '%'; updateSoon(); });
  $('#legend').addEventListener('click', e => {
    const b = e.target.closest('.lg'); if (!b || !S) return;
    const c = +b.dataset.c; focus = focus === c ? 0 : c;
    draw(); renderLegend();
  });
  $('#dlPng').addEventListener('click', () => {
    if (!S) return;
    const sc = Math.max(1, Math.round(900 / S.w)), c = document.createElement('canvas');
    c.width = S.w * sc; c.height = S.h * sc;
    const x = c.getContext('2d'); x.imageSmoothingEnabled = false; x.drawImage(mapCv, 0, 0, c.width, c.height);
    c.toBlob(b => save(b, 'bluewatch_' + S.name.replace(/[^\w]+/g, '_') + '_' + view + '.png'));
  });
  $('#dlCsv').addEventListener('click', () => {
    if (!S) return;
    const n = S.w * S.h;
    let t = 'class_id,class,pixels,percent_of_scene\n';
    for (let c = 1; c <= 15; c++) t += `${c},"${CLASS[c][0]}",${S.counts[c]},${(S.counts[c] / n * 100).toFixed(3)}\n`;
    save(new Blob([t], { type: 'text/csv' }), 'bluewatch_' + S.name.replace(/[^\w]+/g, '_') + '_counts.csv');
  });

  /* ------------------------------------------------------------ choosing a scene */
  function showTab(which) {
    const sample = which === 'sample';
    $('#tabSample').setAttribute('aria-selected', String(sample));
    $('#tabUpload').setAttribute('aria-selected', String(!sample));
    $('#paneSample').hidden = !sample; $('#paneUpload').hidden = sample;
  }
  $('#tabSample').addEventListener('click', () => showTab('sample'));
  $('#tabUpload').addEventListener('click', () => showTab('upload'));

  let SAMPLES = null, kind = 'oil';
  function renderSamples() {
    const box = $('#sampleList'), list = (SAMPLES && SAMPLES[kind]) || [];
    box.innerHTML = '';
    if (!list.length) { box.innerHTML = '<p class="note">No scenes in this group.</p>'; return; }
    for (const r of list) {
      const b = document.createElement('button');
      b.type = 'button'; b.className = 'sample' + (r.id === currentId ? ' on' : ''); b.dataset.id = r.id;
      const mix = r.oil > 0 && r.water > 0;
      const sub = r.oil > 0
        ? `${nf.format(r.oil)} real oil pixels` + (r.water > 0 ? `, ${nf.format(r.water)} clean water pixels` : '')
        : `${nf.format(r.labelled)} labelled pixels, no oil`;
      b.innerHTML = `<strong>Scene ${r.scene.replace('Scene_', '')}, crop ${r.crop}${mix ? '<i class="badge">oil + water</i>' : ''}</strong><span>${sub}</span>`;
      b.addEventListener('click', () => { currentId = r.id; renderSamples(); scan({ sample: r.id }); });
      box.appendChild(b);
    }
  }
  $$('.chip').forEach(ch => ch.addEventListener('click', () => {
    kind = ch.dataset.kind;
    $$('.chip').forEach(c => c.setAttribute('aria-pressed', String(c === ch)));
    renderSamples();
  }));

  async function loadSamples() {
    const box = $('#sampleList');
    box.innerHTML = '<p class="note">Looking through your test scenes. The first time takes a few seconds...</p>';
    try { SAMPLES = await (await fetch('/api/samples')).json(); renderSamples(); }
    catch (e) { box.innerHTML = `<p class="note">${offline}</p>`; }
  }

  function dropzone(label, input, onFiles) {
    ['dragenter', 'dragover'].forEach(ev => label.addEventListener(ev, e => { e.preventDefault(); label.classList.add('over'); }));
    ['dragleave', 'drop'].forEach(ev => label.addEventListener(ev, e => { e.preventDefault(); label.classList.remove('over'); }));
    label.addEventListener('drop', e => { if (e.dataTransfer && e.dataTransfer.files.length) onFiles([...e.dataTransfer.files]); });
    input.addEventListener('change', () => { if (input.files.length) onFiles([...input.files]); input.value = ''; });
  }
  dropzone($('#drop'), $('#fileScene'), files => {
    const tifs = files.filter(f => /\.tiff?$/i.test(f.name));
    if (!tifs.length) { setMsg($('#scanMsg'), 'Those are not .tif files. Use the 11-band file from MADOS_nearest.', true); return; }
    currentId = null; if (SAMPLES) renderSamples();
    scan({ files: tifs });
  });

  /* ------------------------------------------------------------ csv */
  async function analyseCsv(fd) {
    const msg = $('#csvMsg');
    setMsg(msg, ''); $('#csvBusy').hidden = false; J.busy(true);
    try {
      const res = await fetch('/api/predict_csv', { method: 'POST', body: fd });
      let j; try { j = await res.json(); } catch (e) { throw new Error('The server sent back something unexpected.'); }
      if (!res.ok) throw new Error(j.error || 'Something went wrong.');
      showCsv(j);
    } catch (e) { setMsg(msg, e instanceof TypeError ? offline : e.message, true); }
    finally { $('#csvBusy').hidden = true; J.busy(false); }
  }
  function showCsv(j) {
    $('#csvEmpty').hidden = true; $('#csvResult').hidden = false;
    $('#csvName').textContent = j.name;
    $('#csvSub').textContent = `${nf.format(j.rows)} pixels read by the Random Forest in ${j.seconds} s` + (j.sampled ? ` (random sample of ${nf.format(j.total_rows)} rows).` : '.');
    const counts = [0, ...j.counts], total = counts.reduce((a, b) => a + b, 0) || 1;
    $('#csvLegend').innerHTML = legendHTML(counts, total, false);
    const sc = $('#csvScores'), s = j.scores;
    if (!s) { sc.innerHTML = '<p class="note" style="grid-column:1/-1">This file has no label column, so there is no score. The breakdown below is what the model found.</p>'; }
    else {
      const f = (x, d = 0) => x === null || x === undefined ? 'n/a' : pct(x, d);
      sc.innerHTML = tile(f(s.accuracy), 'of pixels classified correctly') +
        tile(s.miou === null ? 'n/a' : s.miou.toFixed(2), 'average overlap score over classes (mIoU)') +
        tile(f(s.oil_recall), 'of the real oil caught') + tile(f(s.oil_precision), 'of the oil flags were right') +
        tile(f(s.false_alarm_rate, 1), 'of clean pixels wrongly flagged as oil', true);
    }
    J.bloom();
  }
  dropzone($('#dropCsv'), $('#fileCsv'), files => {
    const f = files[0];
    if (!/\.csv$/i.test(f.name)) { setMsg($('#csvMsg'), 'That is not a .csv file.', true); return; }
    const fd = new FormData(); fd.append('file', f); analyseCsv(fd);
  });
  $('#getSample').addEventListener('click', () => { location.href = '/api/sample_csv'; });

  /* ------------------------------------------------------------ evidence: numbers from report.json and analysis/ */
  const esc = t => String(t).replace(/[&<>]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
  const li = t => `<li>${t}</li>`;
  let A = null;

  async function loadReport() {
    try { R = await (await fetch('/static/report.json')).json(); } catch (e) { return; }
    try { const a = await (await fetch('/api/analysis')).json(); A = a.available ? a : null; } catch (e) { A = null; }
    const rf = R.rf, un = R.unet, en = R.ensemble;
    $('#tPixels').textContent = nf.format(rf.pixels);

    // table: one column per model
    const cols = [];
    if (en) cols.push(['U-Net ensemble', en]);
    if (un) cols.push([en ? 'U-Net (one run)' : 'U-Net', un]);
    cols.push(['Random Forest', rf]);
    const rows = [
      ['Real oil it finds', 'oil_recall', v => pct(v, 0)],
      ['Oil flags that are right', 'oil_precision', v => pct(v, 0)],
      ['Overlap score for oil (IoU)', 'oil_iou', v => v.toFixed(2)],
      ['Average over all 15 classes (mIoU)', 'miou', v => v.toFixed(2)],
      ['Pixels given the right class (all 15)', 'accuracy', v => pct(v, 1)],
      ['Oil or not oil, right per pixel', 'oil_vs_rest_accuracy', v => pct(v, 1)],
    ];
    $('#oilTable').innerHTML = '<tr><th></th>' + cols.map(c => `<th>${c[0]}</th>`).join('') + '</tr>' +
      rows.map(([label, key, f]) => {
        const have = cols.filter(c => typeof c[1][key] === 'number');
        const best = Math.max(...have.map(c => c[1][key]));
        return `<tr><td>${label}</td>` + cols.map(c => typeof c[1][key] !== 'number' ? '<td>not measured</td>' : `<td${have.length > 1 && c[1][key] === best ? ' class="up"' : ''}>${f(c[1][key])}</td>`).join('') + '</tr>';
      }).join('');
    const a = R.bands_only_validation, b = R.validation;
    $('#cmp').innerHTML =
      '<tr><th></th><th>11 bands only</th><th>Bands + 5 formulas</th></tr>' +
      `<tr><td>Oil overlap score (IoU)</td><td>${a.oil_iou.toFixed(2)}</td><td class="up">${b.oil_iou.toFixed(2)}</td></tr>` +
      `<tr><td>Average over classes (mIoU)</td><td>${a.miou.toFixed(2)}</td><td class="up">${b.miou.toFixed(2)}</td></tr>`;

    // score per class
    const main = un || rf;
    const list = Object.keys(main.per_class_iou).map(Number)
      .map(c => ({ c, v: main.per_class_iou[c], r: rf.per_class_iou[c] })).sort((x, y) => y.v - x.v);
    $('#perClass').innerHTML = list.map(({ c, v, r }) => {
      const hex = CLASS[c][1];
      return `<div class="lg"><i class="sw" style="--c:${hex}"></i><span>${CLASS[c][0]}</span>` +
        `<span class="pc">${v.toFixed(2)}${un ? `<small>RF ${r.toFixed(2)}</small>` : ''}</span>` +
        `<span class="bar"><u style="width:${Math.max(v * 100, 1.2)}%;--c:${hex}"></u></span>` +
        (un ? `<span class="bar thin"><u style="width:${Math.max(r * 100, 1.2)}%"></u></span>` : '') + '</div>';
    }).join('');

    // works well / struggles, every line comes from a measured number
    const best = en || un || rf;
    const names3 = arr => arr.map(x => `${CLASS[x.c][0]} ${x.v.toFixed(2)}`).join(', ');
    const pros = [], cons = [];
    if (typeof best.accuracy === 'number') pros.push(`${pct(best.accuracy, 1)} of labelled pixels get the right class` + (R.baseline_all_water ? `, against ${pct(R.baseline_all_water, 1)} for always guessing "water".` : '.'));
    pros.push(`Average overlap over classes (mIoU) is ${best.miou.toFixed(2)}` + (best !== rf ? `, against ${rf.miou.toFixed(2)} for the Random Forest.` : '.'));
    pros.push(`Strongest classes (U-Net, IoU): ${names3(list.slice(0, 3))}.`);
    cons.push(`Oil: ${pct(best.oil_recall, 0)} of real oil is found, and only ${pct(best.oil_precision, 0)} of the oil flags are right.`);
    if (A && A.called_oil_really && A.called_oil_really.length) {
      const w = A.called_oil_really.find(x => x.class !== 'Oil Spill' && x.pct >= 1);
      if (w) cons.push(`${w.pct}% of the pixels it calls oil are really ${esc(w.class)}.`);
    }
    if (A && A.oil_called && A.oil_called.length) {
      const m = A.oil_called.find(x => x.class !== 'Oil Spill' && x.pct >= 1);
      if (m) cons.push(`${m.pct}% of real oil pixels are called ${esc(m.class)}.`);
    }
    cons.push(`Weakest classes (U-Net, IoU): ${names3(list.slice(-3).reverse())}.`);
    cons.push('Scores on new scenes are lower than on the scenes used to pick the model, so results vary from scene to scene.');
    $('#prosList').innerHTML = pros.map(li).join('');
    $('#consList').innerHTML = cons.map(li).join('');

    renderOilTab(); renderFailTab(); sliderNote();
  }

  function sliderNote() {
    const el = $('#thrNote');
    if (!A || !A.slider_points) { el.hidden = true; return; }
    const p = A.slider_points, lo = p['0.3'], mid = p['0.5'], hi = p['0.8'];
    if (!lo || !mid || !hi) { el.hidden = true; return; }
    el.hidden = false;
    el.textContent = `On the test set: at 30% the model finds ${pct(lo[0], 0)} of the oil and ${pct(lo[1], 0)} of its flags are right; at 50%, ${pct(mid[0], 0)} and ${pct(mid[1], 0)}; at 80%, ${pct(hi[0], 0)} and ${pct(hi[1], 0)}. Raising the slider mostly loses oil without making the flags more reliable.`;
  }

  function renderOilTab() {
    const el = $('#oilMistakes');
    if (!A) { el.innerHTML = '<p class="note">Not measured yet. Run <code>py -3.12 analysis_report.py</code> in your Bluewatch folder, then refresh this page.</p>'; return; }
    const list = arr => arr.slice(0, 4).map(x => `<li><b>${esc(x.class)}</b> ${x.pct}%</li>`).join('');
    const pts = A.slider_points || {};
    const prow = Object.keys(pts).map(k => `<tr><td>${Math.round(k * 100)}%</td><td>${pct(pts[k][0], 0)}</td><td>${pct(pts[k][1], 0)}</td></tr>`).join('');
    el.innerHTML = `<div class="two">
      <figure><img src="/analysis/oil_confusion.png" alt="What oil is confused with"><figcaption>Left: what the model calls real oil. Right: what the pixels it calls oil really are.</figcaption></figure>
      <figure><img src="/analysis/pr_curve.png" alt="Precision recall curve for oil"><figcaption>Every slider setting: oil caught against flags that are right.</figcaption></figure></div>
      <div class="two">
        <div><h3>Real oil was called</h3><ul class="plain">${list(A.oil_called || [])}</ul>
             <h3 class="mt">Pixels called oil were really</h3><ul class="plain">${list(A.called_oil_really || [])}</ul></div>
        <div><h3>Slider setting</h3><table class="cmp"><tr><th>Sure at least</th><th>Oil caught</th><th>Flags right</th></tr>${prow}</table></div>
      </div>`;
  }

  function renderFailTab() {
    const el = $('#failList');
    if (!A || !A.failures || !A.failures.length) { el.innerHTML = '<p class="note">Not measured yet. Run <code>py -3.12 analysis_report.py</code> in your Bluewatch folder, then refresh this page.</p>'; return; }
    el.innerHTML = '<p class="note">The four test crops with real oil where the model did worst. Left to right: satellite, model, real labels.</p>' +
      A.failures.map(f => `<figure><img src="/${esc(f.file)}" alt="Failure case ${esc(f.scene)} crop ${esc(f.crop)}"><figcaption>${esc(f.scene.replace('_', ' '))}, crop ${esc(f.crop)}: ${nf.format(f.real_oil)} real oil pixels, ${nf.format(f.found)} found, ${nf.format(f.false_flags)} false flags.</figcaption></figure>`).join('');
  }

  $('#qTabs').addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    $$('#qTabs button').forEach(x => x.setAttribute('aria-selected', String(x === b)));
    ['scores', 'oil', 'fails', 'how'].forEach(k => { $('#q-' + k).hidden = k !== b.dataset.q; });
  });

  /* ------------------------------------------------------------ start */
  (async function init() {
    loadReport();
    let st = {};
    try { st = await (await fetch('/api/status')).json(); }
    catch (e) { setMsg($('#scanMsg'), offline, true); }
    $('#unetHint').hidden = !st.unet_hint;
    if (st.unet && st.n_unets > 1) MODELS.unet = `U-Net ensemble (${st.n_unets} models)`;
    $('#status').textContent = [st.unet ? (st.n_unets > 1 ? `U-Net ensemble of ${st.n_unets}` : 'U-Net') : '', st.rf ? 'Random Forest' : '', st.device ? `running on ${st.device}` : ''].filter(Boolean).join(' + ').replace(' + running', ', running');
    $('#modelPick').hidden = !(st.unet && st.rf);
    setModel(st.unet ? 'unet' : 'rf');
    if (st.samples) { loadSamples(); showTab('sample'); }
    else {
      $('#tabSample').hidden = true; showTab('upload');
      const n = document.createElement('p');
      n.className = 'note';
      n.textContent = 'Sample scenes appear here when the MADOS/MADOS_nearest folder sits next to the app.';
      $('#paneUpload').appendChild(n);
    }
    if (st.test_csv && st.rf !== false) $('#getSample').hidden = false;
  })();
})();
