// Builds the control panel from a problem's declared parameter schema.
// Adding a Param in Python is all it takes to get a working control here.

export class ParamPanel {
  constructor(container, onChange) {
    this.container = container;
    this.onChange = onChange;
    this.inputs = new Map();
    this.problem = null;
  }

  build(problem) {
    this.problem = problem;
    this.container.innerHTML = '';
    this.inputs.clear();

    // Preserve declaration order of groups.
    const groups = [];
    const byGroup = new Map();
    for (const p of problem.params) {
      if (!byGroup.has(p.group)) { byGroup.set(p.group, []); groups.push(p.group); }
      byGroup.get(p.group).push(p);
    }

    for (const name of groups) {
      const sec = document.createElement('div');
      sec.className = 'group';

      const title = document.createElement('div');
      title.className = 'group-title';
      title.textContent = name.toUpperCase();
      sec.appendChild(title);

      for (const p of byGroup.get(name)) sec.appendChild(this._control(p));
      this.container.appendChild(sec);
    }
  }

  _control(p) {
    const wrap = document.createElement('div');
    wrap.className = 'ctrl';

    if (p.kind === 'bool') {
      const label = document.createElement('label');
      label.className = 'switch';
      const box = document.createElement('input');
      box.type = 'checkbox';
      box.checked = Boolean(p.default);
      const text = document.createElement('span');
      text.className = 'ctrl-name';
      text.textContent = p.label;
      label.append(box, text);
      wrap.appendChild(label);

      box.addEventListener('change', () => this.onChange());
      this.inputs.set(p.key, { spec: p, read: () => box.checked, set: (v) => { box.checked = Boolean(v); } });

    } else if (p.kind === 'choice') {
      const head = document.createElement('div');
      head.className = 'ctrl-head';
      const nm = document.createElement('span');
      nm.className = 'ctrl-name';
      nm.textContent = p.label;
      head.appendChild(nm);
      wrap.appendChild(head);

      const sel = document.createElement('select');
      for (const c of p.choices ?? []) {
        const opt = document.createElement('option');
        opt.value = c; opt.textContent = c;
        if (c === p.default) opt.selected = true;
        sel.appendChild(opt);
      }
      wrap.appendChild(sel);

      sel.addEventListener('change', () => this.onChange());
      this.inputs.set(p.key, { spec: p, read: () => sel.value, set: (v) => { sel.value = v; } });

    } else {
      const head = document.createElement('div');
      head.className = 'ctrl-head';
      const nm = document.createElement('span');
      nm.className = 'ctrl-name';
      nm.textContent = p.label;
      const val = document.createElement('span');
      val.className = 'ctrl-val';
      head.append(nm, val);
      wrap.appendChild(head);

      const range = document.createElement('input');
      range.type = 'range';
      range.min = p.min ?? 0;
      range.max = p.max ?? 100;
      range.step = p.step ?? (p.kind === 'int' ? 1 : 0.1);
      range.value = p.default;
      wrap.appendChild(range);

      const show = () => {
        const n = Number(range.value);
        val.innerHTML = `${p.kind === 'int' ? n : n.toFixed(this._dp(range.step))}` +
          (p.unit ? `<span class="unit">${p.unit}</span>` : '');
      };
      show();

      // `input` updates the label live; `change` fires the solve, so dragging a
      // slider does not queue dozens of solver runs.
      range.addEventListener('input', show);
      range.addEventListener('change', () => this.onChange());

      this.inputs.set(p.key, {
        spec: p,
        read: () => (p.kind === 'int' ? parseInt(range.value, 10) : parseFloat(range.value)),
        set: (v) => { range.value = v; show(); },
      });
    }

    if (p.help) {
      const help = document.createElement('div');
      help.className = 'ctrl-help';
      help.textContent = p.help;
      wrap.appendChild(help);
    }
    return wrap;
  }

  _dp(step) {
    const s = String(step);
    return s.includes('.') ? s.split('.')[1].length : 0;
  }

  values() {
    const out = {};
    for (const [k, c] of this.inputs) out[k] = c.read();
    return out;
  }

  reset() {
    for (const [, c] of this.inputs) c.set(c.spec.default);
  }
}
