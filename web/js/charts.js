// Compact SVG telemetry plots. No chart library — these are simple polylines
// with a playhead, and staying dependency-free keeps the repo easy to clone.

const NS = 'http://www.w3.org/2000/svg';
const W = 280;
const H = 62;

function el(name, attrs = {}) {
  const node = document.createElementNS(NS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  return node;
}

export class TelemetryCharts {
  constructor(container) {
    this.container = container;
    this.charts = [];
  }

  build(traj) {
    this.container.innerHTML = '';
    this.charts = [];
    if (!traj?.feasible) return;

    for (const s of traj.series) {
      const t = s.on === 'control' ? traj.t_control : traj.t_state;
      if (!t.length || t.length !== s.values.length) continue;

      const wrap = document.createElement('div');
      wrap.className = 'chart';

      const title = document.createElement('div');
      title.className = 'chart-title';
      const name = document.createElement('span');
      name.textContent = s.label.toUpperCase();
      const val = document.createElement('span');
      title.append(name, val);
      wrap.appendChild(title);

      const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'none' });

      const lo = Math.min(...s.values);
      const hi = Math.max(...s.values);
      const span = hi - lo || 1;
      const pad = span * 0.12;
      const yMin = lo - pad;
      const yMax = hi + pad;

      const tMin = t[0];
      const tMax = t[t.length - 1] || 1;

      const X = (tt) => ((tt - tMin) / (tMax - tMin || 1)) * W;
      const Y = (vv) => H - ((vv - yMin) / (yMax - yMin)) * H;

      // zero line, when zero is inside the visible range
      if (yMin < 0 && yMax > 0) {
        svg.appendChild(el('line', {
          x1: 0, x2: W, y1: Y(0), y2: Y(0),
          stroke: 'rgba(125,141,163,0.35)', 'stroke-width': 1,
          'stroke-dasharray': '3 3', 'vector-effect': 'non-scaling-stroke',
        }));
      }

      const pts = s.values.map((v, i) => `${X(t[i]).toFixed(2)},${Y(v).toFixed(2)}`).join(' ');
      const shape = s.on === 'control' ? 'polyline' : 'polyline';
      svg.appendChild(el(shape, {
        points: pts, fill: 'none', stroke: '#4dd2ff', 'stroke-width': 1.5,
        'vector-effect': 'non-scaling-stroke',
        'stroke-linejoin': 'round',
      }));

      const head = el('line', {
        x1: 0, x2: 0, y1: 0, y2: H,
        stroke: '#ff9d4d', 'stroke-width': 1,
        'vector-effect': 'non-scaling-stroke',
      });
      svg.appendChild(head);

      wrap.appendChild(svg);
      this.container.appendChild(wrap);

      this.charts.push({ series: s, t, X, head, val, tMin, tMax });
    }
  }

  // Move each playhead and print the value under the cursor.
  update(time) {
    for (const c of this.charts) {
      const x = c.X(Math.min(Math.max(time, c.tMin), c.tMax));
      c.head.setAttribute('x1', x);
      c.head.setAttribute('x2', x);

      const { t, series } = c;
      let i = 0;
      while (i < t.length - 1 && t[i + 1] < time) i += 1;
      const v = series.values[i];
      c.val.textContent = `${v >= 0 ? '' : ''}${v.toFixed(2)} ${series.unit}`;
    }
  }
}
