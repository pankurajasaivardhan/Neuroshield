/* NeuroShield demo frontend logic */
const state = {
  t1: { sample: null, file: null, pair: null, polling: null },
  t2: { sample: null, files: {}, polling: null },
};

function $(id) { return document.getElementById(id); }

function switchTrack(which) {
  const t1 = which === 't1';
  $('tab-t1').classList.toggle('active', t1);
  $('tab-t2').classList.toggle('active', !t1);
  $('tab-t1').setAttribute('aria-selected', t1);
  $('tab-t2').setAttribute('aria-selected', !t1);
  $('panel-t1').classList.toggle('hidden', !t1);
  $('panel-t2').classList.toggle('hidden', t1);
}

/* ---------- samples ---------- */
async function loadSamples() {
  try {
    const res = await fetch('/api/samples');
    const data = await res.json();
    renderSamples('t1', data.track1 || []);
    renderSamples('t2', data.track2 || []);
  } catch (e) {
    $('t1-samples').innerHTML = '<div class="loading">Could not load samples. You can still upload a scan.</div>';
    $('t2-samples').innerHTML = '<div class="loading">Could not load samples. You can still upload a case.</div>';
  }
}

function renderSamples(track, samples) {
  const box = $(track + '-samples');
  if (!samples.length) {
    box.innerHTML = '<div class="loading">No bundled samples on the server yet - please upload a scan below.</div>';
    return;
  }
  box.innerHTML = '';
  samples.forEach((s) => {
    const b = document.createElement('button');
    b.className = 'sample-card';
    b.innerHTML = `<div class="sample-name">${esc(s.name)}</div><div class="sample-meta">${esc(s.description || '')}</div>`;
    b.onclick = () => {
      state[track].sample = s.id;
      box.querySelectorAll('.sample-card').forEach((c) => c.classList.remove('selected'));
      b.classList.add('selected');
      if (track === 't1') { state.t1.file = null; $('t1-file').value = ''; $('t1-file-pair').value = ''; }
      else { state.t2.files = {}; ['flair','t1','t1ce','t2'].forEach((m) => { $('t2-' + m).value = ''; }); }
      updateRunButtons();
    };
    box.appendChild(b);
  });
}

function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }

/* ---------- uploads ---------- */
$('t1-file').addEventListener('change', (e) => {
  const f = e.target.files[0];
  state.t1.file = f || null;
  if (f && /\.(img|hdr)$/i.test(f.name)) {
    $('t1-file-pair').classList.remove('hidden');
    $('t1-pair-hint').classList.remove('hidden');
  } else {
    $('t1-file-pair').classList.add('hidden');
    $('t1-pair-hint').classList.add('hidden');
    state.t1.pair = null;
  }
  if (f) { state.t1.sample = null; document.querySelectorAll('#t1-samples .sample-card').forEach((c) => c.classList.remove('selected')); }
  updateRunButtons();
});
$('t1-file-pair').addEventListener('change', (e) => {
  state.t1.pair = e.target.files[0] || null;
  updateRunButtons();
});
['flair', 't1', 't1ce', 't2'].forEach((m) => {
  $('t2-' + m).addEventListener('change', (e) => {
    state.t2.files[m] = e.target.files[0] || null;
    if (state.t2.files[m]) {
      state.t2.sample = null;
      document.querySelectorAll('#t2-samples .sample-card').forEach((c) => c.classList.remove('selected'));
    }
    updateRunButtons();
  });
});

function updateRunButtons() {
  const t1ok = state.t1.sample || (state.t1.file && (!$('t1-file-pair') || $('t1-file-pair').classList.contains('hidden') || state.t1.pair));
  $('t1-run').disabled = !t1ok;
  const uploadedAll = ['flair', 't1', 't1ce', 't2'].every((m) => state.t2.files[m]);
  $('t2-run').disabled = !(state.t2.sample || uploadedAll);
}

/* ---------- run + poll ---------- */
function setStatus(track, html, isError) {
  const el = $(track + '-status');
  el.classList.remove('hidden');
  el.classList.toggle('error', !!isError);
  el.innerHTML = html;
}
function clearResult(track) {
  $(track + '-empty').classList.add('hidden');
  $(track + '-out').classList.remove('hidden');
  $(track + '-out').innerHTML = '';
}

async function runTrack1() {
  const fd = new FormData();
  if (state.t1.sample) fd.append('sample_id', state.t1.sample);
  else {
    fd.append('scan', state.t1.file);
    if (state.t1.pair) fd.append('scan_pair', state.t1.pair);
  }
  $('t1-run').disabled = true;
  setStatus('t1', '<span class="spinner"></span>Starting the models…');
  try {
    const res = await fetch('/api/track1/run', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not start the run.');
    poll('t1', data.job_id);
  } catch (e) {
    setStatus('t1', '⚠ ' + esc(e.message), true);
    $('t1-run').disabled = false;
  }
}

async function runTrack2() {
  const fd = new FormData();
  fd.append('remove_flair', $('t2-flair-toggle').checked);
  if (state.t2.sample) fd.append('sample_id', state.t2.sample);
  else ['flair', 't1', 't1ce', 't2'].forEach((m) => fd.append(m, state.t2.files[m]));
  $('t2-run').disabled = true;
  setStatus('t2', '<span class="spinner"></span>Starting the segmentation…');
  try {
    const res = await fetch('/api/track2/run', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Could not start the run.');
    poll('t2', data.job_id);
  } catch (e) {
    setStatus('t2', '⚠ ' + esc(e.message), true);
    $('t2-run').disabled = false;
  }
}

function poll(track, jobId) {
  const started = Date.now();
  if (state[track].polling) clearInterval(state[track].polling);
  state[track].polling = setInterval(async () => {
    try {
      const res = await fetch('/api/jobs/' + jobId);
      const job = await res.json();
      const secs = Math.round((Date.now() - started) / 1000);
      if (job.status === 'done') {
        clearInterval(state[track].polling);
        setStatus(track, '✅ Finished in ' + secs + 's');
        const rres = await fetch('/api/jobs/' + jobId + '/result');
        const result = await rres.json();
        if (track === 't1') renderTrack1(result, jobId); else renderTrack2(result, jobId);
        $(track + '-run').disabled = false;
      } else if (job.status === 'error') {
        clearInterval(state[track].polling);
        setStatus(track, '⚠ ' + esc(job.error || 'The run failed.'), true);
        $(track + '-run').disabled = false;
      } else {
        setStatus(track, '<span class="spinner"></span>' + esc(job.stage || 'Working…') + ' <span class="muted">(' + secs + 's)</span>');
      }
    } catch (e) { /* transient poll error: keep waiting */ }
  }, 2000);
}

/* ---------- render track 1 ---------- */
function renderTrack1(r, jobId) {
  clearResult('t1');
  const demented = r.prediction === 1;
  const confPct = Math.round((demented ? r.probability_dementia : r.probability_no_dementia) * 100);
  const chips = Object.entries(r.per_seed).map(([seed, s]) => {
    const agree = s.prediction === r.prediction;
    return `<span class="chip ${agree ? 'agree' : 'disagree'}">Model trained run ${seed}: ${s.prediction === 1 ? 'Dementia pattern' : 'No dementia pattern'}</span>`;
  }).join('');
  let html = `
    <div class="verdict ${demented ? 'warn' : 'good'}">
      <div>
        <div class="verdict-label">${demented ? 'Model prediction: dementia-pattern class' : 'Model prediction: no-dementia class'}</div>
        <div class="verdict-sub">Model score: ${confPct}% (a model output score, not clinical certainty) · ${r.slices_used} slices read</div>
      </div>
    </div>
    <div class="conf-row">
      <div class="conf-head"><span>Probability of dementia pattern</span><strong>${Math.round(r.probability_dementia * 100)}%</strong></div>
      <div class="conf-bar"><div class="conf-fill" style="width:${Math.round(r.probability_dementia * 100)}%"></div></div>
    </div>
    <h3>Do the 3 models agree?</h3>
    <div class="chips">${chips}</div>
    <p class="muted small">${r.seed_agreement
      ? 'All three independently trained models voted the same way. Agreement does not make the result correct - the project audit found cases where all three agreed and were still wrong.'
      : `Only ${r.seeds_agreeing} of ${r.seeds_total} models voted for this result - treat it with extra caution. Agreement alone never proves correctness.`}</p>
    <div class="note">${esc(r.class_note)}</div>`;
  $('t1-out').innerHTML = html;
  if (r.sample_true_label_text) {
    const wrap = document.createElement('div');
    wrap.className = 'reveal';
    const btn = document.createElement('button');
    btn.className = 'btn btn-secondary';
    btn.textContent = "Reveal this sample's true dataset label";
    btn.onclick = () => {
      const note = document.createElement('div');
      note.className = 'note';
      const matched = r.sample_true_label === r.prediction;
      note.innerHTML = '<strong>This sample scan\'s dataset label:</strong> ' + esc(r.sample_true_label_text)
        + (matched ? ' - the model matched it ✅' : ' - the model did NOT match it ❌ (a real error, shown honestly)');
      wrap.replaceWith(note);
    };
    wrap.appendChild(btn);
    $('t1-out').appendChild(wrap);
  }
}

/* ---------- render track 2 ---------- */
function renderTrack2(r, jobId) {
  clearResult('t2');
  const base = '/api/jobs/' + jobId + '/files/';
  const imgs = (r.overlay_slices || []).map((u) => `<img src="${base}${u}" alt="MRI slice with tumor overlay" loading="lazy">`).join('');
  const volRows = Object.entries(r.volumes_cm3).map(([k, v]) =>
    `<tr><td>${k === 'Whole tumor' ? '<strong>' + esc(k) + '</strong>' : esc(k)}</td><td>${v.toFixed(2)} cm³</td></tr>`).join('');
  let html = `
    <div class="verdict ${r.tumor_found ? 'warn' : 'good'}">
      <div>
        <div class="verdict-label">${r.tumor_found ? 'Tumor regions outlined' : 'No tumor region found'}</div>
        <div class="verdict-sub">Case ${esc(r.case_id)} · colours show the three tumor parts</div>
      </div>
    </div>
    <div class="legend">
      <span><span class="swatch" style="background:rgb(59,130,246)"></span>Necrotic / non-enhancing core</span>
      <span><span class="swatch" style="background:rgb(34,197,94)"></span>Swelling (edema)</span>
      <span><span class="swatch" style="background:rgb(239,68,68)"></span>Enhancing tumor</span>
    </div>
    <div class="slices">${imgs}</div>
    <h3 style="margin-top:16px">Estimated region sizes</h3>
    <table class="vol"><tr><th>Region</th><th>Volume</th></tr>${volRows}</table>`;
  if (r.vs_expert_dice) {
    const rows = Object.entries(r.vs_expert_dice).map(([k, v]) =>
      `<tr><td>${esc(k)}</td><td>${Math.round(v * 100)}%</td></tr>`).join('');
    html += `
    <h3 style="margin-top:16px">How close is this to the expert's outline? <span class="tag">Sample cases only</span></h3>
    <table class="vol"><tr><th>Region</th><th>Match (Dice)</th></tr>${rows}</table>
    <p class="muted small">This sample case has a human expert's outline, so we can score the model directly. 100% = perfect match.</p>`;
  }
  if (r.flair_removal) {
    const fr = r.flair_removal;
    const pct = Math.round(fr.whole_tumor_stability_dice * 100);
    const color = fr.stable ? 'var(--good)' : 'var(--bad)';
    const cfImgs = (fr.overlay_slices || []).map((u) => `<img src="${base}${u}" alt="Overlay without FLAIR" loading="lazy">`).join('');
    html += `
    <div class="stability">
      <h3>Reliability stress test: FLAIR sequence removed</h3>
      <div class="conf-head"><span>Whole-tumor stability</span><strong style="color:${color}">${pct}%</strong></div>
      <div class="stability-bar"><div class="stability-fill" style="width:${pct}%;background:${color}"></div></div>
      <p class="muted small">${fr.stable ? 'The outline stayed mostly the same without FLAIR.' : '⚠ The outline changed a lot without FLAIR - this prediction is fragile.'} ${esc(fr.explanation)}</p>
      <div class="slices">${cfImgs}</div>
    </div>`;
  }
  html += `<div class="note">Settings: the model's accuracy-boosting test-time augmentation is switched off so results return in minutes on CPU. Its held-out test whole-tumor Dice is 0.90, but the model is not reliable on scans unlike its training data.</div>`;
  $('t2-out').innerHTML = html;
}

/* overlay lightbox: click a slice to inspect full size, click anywhere to close */
document.addEventListener('click', (e) => {
  const lb = $('lightbox');
  if (!lb) return;
  if (e.target.matches('.slices img')) {
    lb.querySelector('img').src = e.target.src;
    lb.querySelector('img').alt = e.target.alt || 'Enlarged overlay';
    lb.classList.add('open');
  } else if (e.target.closest('#lightbox')) {
    lb.classList.remove('open');
  }
});

loadSamples();
