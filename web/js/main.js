// Wiring: fetch schemas, build controls, solve, animate, report.

import { fetchProblems, solve } from './api.js';
import { TrajectoryScene } from './scene.js';
import { ParamPanel } from './ui.js';
import { TelemetryCharts } from './charts.js';

const $ = (id) => document.getElementById(id);

const app = {
  problems: [],
  problem: null,
  traj: null,
  time: 0,
  playing: false,
  speed: 1,
  lastFrame: performance.now(),
  solveToken: 0,
};

const scene = new TrajectoryScene($('viewport'));
const charts = new TelemetryCharts($('charts'));
const panel = new ParamPanel($('param-groups'), () => {
  if ($('auto-solve').checked) run();
});

// ── status + toast ────────────────────────────────────────
function setStatus(text, kind = '') {
  $('status-text').textContent = text;
  $('status-chip').className = `status-chip ${kind}`;
}

let toastTimer = null;
function toast(msg) {
  const t = $('toast');
  t.textContent = msg;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 5200);
}

// ── solve ─────────────────────────────────────────────────
async function run() {
  if (!app.problem) return;
  const token = ++app.solveToken;
  setStatus('SOLVING', 'busy');

  let traj;
  try {
    traj = await solve(app.problem.slug, panel.values());
  } catch (err) {
    if (token !== app.solveToken) return;
    setStatus('ERROR', 'bad');
    toast(String(err.message ?? err));
    return;
  }
  if (token !== app.solveToken) return;   // a newer solve already landed

  app.traj = traj;
  scene.setTrajectory(traj, app.problem);
  charts.build(traj);
  renderResults(traj);

  if (!traj.feasible) {
    setStatus(String(traj.status).toUpperCase(), 'bad');
    toast(traj.notes?.[0] ?? 'No feasible trajectory for these parameters.');
    app.playing = false;
    $('btn-play').textContent = '▶';
    setTime(0);
    return;
  }

  // Optimisers report "optimal"; the simulation reports what actually happened,
  // and "crash" is a successful solve with a bad outcome.
  const label = String(traj.status).toUpperCase();
  const severity = /CRASH|DIVERG|ERROR/.test(label) ? 'bad'
    : /HARD|AIRBORNE|INACCURATE/.test(label) ? 'warn'
      : 'ok';
  setStatus(label, severity);
  const T = traj.t_state.at(-1) ?? 1;
  $('scrub').max = T;
  $('scrub').step = T / 600;
  setTime(0);
  app.playing = true;
  $('btn-play').textContent = '❚❚';
}

// ── results panel ─────────────────────────────────────────
function renderResults(traj) {
  const g = $('kv-grid');
  g.innerHTML = '';

  const rows = [
    ['Status', traj.status, traj.feasible ? 'good' : 'bad'],
    ['Solver', traj.solver, ''],
    ['Objective', traj.cost != null ? traj.cost.toFixed(4) : '—', ''],
    ['Solve time', traj.solve_time_ms != null ? `${traj.solve_time_ms.toFixed(2)} ms` : '—', ''],
  ];

  for (const [k, v] of Object.entries(traj.diagnostics ?? {})) {
    if (v == null) continue;
    rows.push([
      k.replace(/_/g, ' ').replace(/\b\w/g, (m) => m.toUpperCase()),
      typeof v === 'number' ? formatNum(v) : String(v),
      '',
    ]);
  }

  for (const [k, v, cls] of rows) {
    const kd = document.createElement('div');
    kd.className = 'kv-key'; kd.textContent = k;
    const vd = document.createElement('div');
    vd.className = `kv-val ${cls}`; vd.textContent = v;
    g.append(kd, vd);
  }

  const n = $('notes');
  n.innerHTML = '';
  for (const note of traj.notes ?? []) {
    const d = document.createElement('div');
    d.className = 'note';
    d.textContent = note;
    n.appendChild(d);
  }
}

// Thrust is m/s^2 for the optimiser problems and newtons for the simulation,
// so the top-bar readout has to cope with both 12.4 and 6,900,000.
function compact(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return `${(v / 1e9).toFixed(2)}G`;
  if (a >= 1e6) return `${(v / 1e6).toFixed(2)}M`;
  if (a >= 1e4) return `${(v / 1e3).toFixed(1)}k`;
  return v.toFixed(1);
}

function formatNum(v) {
  if (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0)) return v.toExponential(3);
  return v.toFixed(3);
}

// ── playback ──────────────────────────────────────────────
function setTime(t) {
  app.time = t;
  $('scrub').value = t;
  $('time-readout').textContent = `${t.toFixed(2)} s`;
}

function tick(now) {
  const dt = Math.min((now - app.lastFrame) / 1000, 0.1);
  app.lastFrame = now;

  if (app.playing && app.traj?.feasible) {
    const T = app.traj.t_state.at(-1) ?? 1;
    let t = app.time + dt * app.speed;
    if (t >= T) { t = T; app.playing = false; $('btn-play').textContent = '▶'; }
    setTime(t);
  }

  const s = scene.update(app.time);
  if (s) {
    $('stat-alt').textContent = `${s.altitude.toFixed(1)} m`;
    $('stat-speed').textContent = `${s.speed.toFixed(1)} m/s`;
    $('stat-thrust').textContent = compact(s.thrust);
    $('stat-tilt').textContent = `${s.tilt.toFixed(1)}°`;
    $('stat-time').textContent = `${s.time.toFixed(2)} s`;
    charts.update(s.time);
  }
  scene.render();
  requestAnimationFrame(tick);
}

// ── events ────────────────────────────────────────────────
$('btn-solve').addEventListener('click', run);

$('btn-reset').addEventListener('click', () => { panel.reset(); run(); });

$('btn-export').addEventListener('click', () => {
  if (!app.traj) { toast('Nothing to export yet — solve first.'); return; }
  const payload = {
    problem: app.problem.slug,
    generated: new Date().toISOString(),
    values: panel.values(),
    trajectory: app.traj,
  };
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = `${app.problem.slug}-run.json`;
  a.click();
  URL.revokeObjectURL(url);
});

$('btn-play').addEventListener('click', () => {
  if (!app.traj?.feasible) return;
  const T = app.traj.t_state.at(-1) ?? 1;
  if (!app.playing && app.time >= T - 1e-6) setTime(0);
  app.playing = !app.playing;
  $('btn-play').textContent = app.playing ? '❚❚' : '▶';
});

$('scrub').addEventListener('input', (e) => {
  app.playing = false;
  $('btn-play').textContent = '▶';
  setTime(parseFloat(e.target.value));
});

function segment(id, attr, fn, exclusive = true) {
  $(id).addEventListener('click', (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;
    if (exclusive) {
      [...$(id).children].forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
    } else {
      btn.classList.toggle('active');
    }
    fn(btn.dataset[attr], btn.classList.contains('active'));
  });
}

segment('speed-seg', 'speed', (v) => { app.speed = parseFloat(v); });
segment('cam-seg', 'cam', (v) => scene.setCamera(v));
segment('toggle-seg', 'toggle', (k, on) => scene.setToggle(k, on), false);

$('collapse-left').addEventListener('click', () => {
  $('panel-left').classList.toggle('collapsed');
});
$('collapse-right').addEventListener('click', () => {
  $('panel-right').classList.toggle('collapsed');
});

$('problem-select').addEventListener('change', (e) => {
  selectProblem(e.target.value);
});

addEventListener('keydown', (e) => {
  if (e.target.matches('input, select, textarea')) return;
  if (e.code === 'Space') { e.preventDefault(); $('btn-play').click(); }
  if (e.key === 'r' || e.key === 'R') run();
});

// ── boot ──────────────────────────────────────────────────
function selectProblem(slug) {
  app.problem = app.problems.find((p) => p.slug === slug) ?? app.problems[0];
  $('problem-phase').textContent = app.problem.phase.toUpperCase();
  $('problem-summary').textContent = app.problem.summary;
  panel.build(app.problem);
  run();
}

async function boot() {
  try {
    app.problems = await fetchProblems();
  } catch (err) {
    setStatus('OFFLINE', 'bad');
    toast(`Could not reach the solver backend: ${err.message}`);
    return;
  }

  const sel = $('problem-select');
  sel.innerHTML = '';
  for (const p of app.problems) {
    const o = document.createElement('option');
    o.value = p.slug;
    o.textContent = `${p.phase} — ${p.title}`;
    sel.appendChild(o);
  }

  // Open on the 3-D problem: it is the one worth looking at in a 3-D viewer.
  const start = app.problems.find((p) => p.slug === 'landing-3dof') ?? app.problems[0];
  sel.value = start.slug;
  selectProblem(start.slug);

  requestAnimationFrame(tick);
}

// Debug handle: lets you poke the scene from the browser console, e.g.
//   __gnc.app.traj.diagnostics
//   __gnc.scene.setCamera('top')
window.__gnc = { app, scene, panel, charts, run, setTime };

boot();
