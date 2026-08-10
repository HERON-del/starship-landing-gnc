// 3-D trajectory scene.
//
// Frame convention matches the solver exactly: right-handed, +Y up, pad at the
// origin. Nothing is transformed on the way in.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const COLD = new THREE.Color(0x2f6bff);   // low thrust
const HOT  = new THREE.Color(0xff7a1a);   // at the limit

export class TrajectoryScene {
  constructor(container) {
    this.container = container;
    this.traj = null;
    this.scale = 100;
    this.camMode = 'orbit';
    this.show = { grid: true, cone: true, trail: true, vectors: false };

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setSize(container.clientWidth, container.clientHeight);
    this.renderer.shadowMap.enabled = false;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x05070c);
    this.scene.fog = new THREE.FogExp2(0x05070c, 0.0009);

    this.camera = new THREE.PerspectiveCamera(
      50, container.clientWidth / container.clientHeight, 0.5, 60000);
    this.camera.position.set(180, 130, 220);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.07;
    this.controls.maxPolarAngle = Math.PI * 0.495;   // never go under the ground

    this._lights();
    this._stars();
    this._ground();
    this._rocket();

    this.pathGroup = new THREE.Group();
    this.scene.add(this.pathGroup);

    this.coneMesh = null;

    this._arrows();

    addEventListener('resize', () => this.resize());
  }

  // ── static furniture ──────────────────────────────────────
  _lights() {
    this.scene.add(new THREE.HemisphereLight(0x9fc4ff, 0x101820, 1.15));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(1, 1.4, 0.7);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0x4dd2ff, 0.7);
    rim.position.set(-1, 0.35, -0.9);
    this.scene.add(rim);
  }

  _stars() {
    const n = 1800;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      // uniform on a sphere shell, far away
      const u = Math.random() * 2 - 1;
      const th = Math.random() * Math.PI * 2;
      const r = 26000 * (0.75 + Math.random() * 0.25);
      const s = Math.sqrt(1 - u * u);
      pos[i * 3]     = r * s * Math.cos(th);
      pos[i * 3 + 1] = r * Math.abs(u);      // keep them above the horizon
      pos[i * 3 + 2] = r * s * Math.sin(th);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.scene.add(new THREE.Points(g, new THREE.PointsMaterial({
      color: 0xbfd4ee, size: 34, sizeAttenuation: true, transparent: true, opacity: 0.75,
    })));
  }

  _ground() {
    this.ground = new THREE.Group();
    this.scene.add(this.ground);

    this.grid = new THREE.GridHelper(1, 40, 0x2a4468, 0x16283f);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.55;
    this.ground.add(this.grid);

    this.pad = new THREE.Group();
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.8, 1, 64),
      new THREE.MeshBasicMaterial({ color: 0x4dd2ff, side: THREE.DoubleSide,
        transparent: true, opacity: 0.95 }));
    ring.rotation.x = -Math.PI / 2;
    this.pad.add(ring);

    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(0.8, 64),
      new THREE.MeshBasicMaterial({ color: 0x0d2233, side: THREE.DoubleSide,
        transparent: true, opacity: 0.85 }));
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = 0.01;
    this.pad.add(disc);

    const cross = new THREE.Group();
    for (const rot of [0, Math.PI / 2]) {
      const bar = new THREE.Mesh(
        new THREE.PlaneGeometry(1.25, 0.09),
        new THREE.MeshBasicMaterial({ color: 0x4dd2ff, transparent: true, opacity: 0.6 }));
      bar.rotation.x = -Math.PI / 2;
      bar.rotation.z = rot;
      cross.add(bar);
    }
    cross.position.y = 0.02;
    this.pad.add(cross);
    this.ground.add(this.pad);
  }

  _rocket() {
    this.rocket = new THREE.Group();

    const hull = new THREE.MeshStandardMaterial({
      color: 0xc9d4e2, metalness: 0.85, roughness: 0.34 });
    const dark = new THREE.MeshStandardMaterial({
      color: 0x2b3442, metalness: 0.6, roughness: 0.5 });

    // Body is built along +Y so the solver's attitude quaternion applies directly.
    const body = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 5.2, 28), hull);
    body.position.y = 2.6;
    this.rocket.add(body);

    const nose = new THREE.Mesh(new THREE.ConeGeometry(1, 2.1, 28), hull);
    nose.position.y = 6.25;
    this.rocket.add(nose);

    const skirt = new THREE.Mesh(new THREE.CylinderGeometry(1.06, 1.14, 0.55, 28), dark);
    skirt.position.y = 0.28;
    this.rocket.add(skirt);

    for (let i = 0; i < 4; i++) {
      const fin = new THREE.Mesh(new THREE.BoxGeometry(0.13, 1.5, 0.85), dark);
      const a = (i / 4) * Math.PI * 2;
      fin.position.set(Math.cos(a) * 1.06, 0.95, Math.sin(a) * 1.06);
      fin.rotation.y = -a;
      this.rocket.add(fin);
    }

    // Exhaust plume, drawn downward from the skirt.
    this.plume = new THREE.Mesh(
      new THREE.ConeGeometry(0.85, 4.2, 22, 1, true),
      new THREE.MeshBasicMaterial({
        color: 0xffa233, transparent: true, opacity: 0.55,
        blending: THREE.AdditiveBlending, side: THREE.DoubleSide, depthWrite: false }));
    this.plume.rotation.x = Math.PI;      // point it down
    this.plume.position.y = -2.1;
    this.rocket.add(this.plume);

    this.plumeCore = new THREE.Mesh(
      new THREE.ConeGeometry(0.36, 2.6, 18, 1, true),
      new THREE.MeshBasicMaterial({
        color: 0xfff0c0, transparent: true, opacity: 0.8,
        blending: THREE.AdditiveBlending, side: THREE.DoubleSide, depthWrite: false }));
    this.plumeCore.rotation.x = Math.PI;
    this.plumeCore.position.y = -1.3;
    this.rocket.add(this.plumeCore);

    this.scene.add(this.rocket);
  }

  _arrows() {
    const mk = (color) => {
      const a = new THREE.ArrowHelper(
        new THREE.Vector3(0, 1, 0), new THREE.Vector3(), 1, color, 0.28, 0.16);
      a.visible = false;
      this.scene.add(a);
      return a;
    };
    this.velArrow = mk(0x4ade80);
    this.thrArrow = mk(0xff9d4d);
  }

  // ── trajectory ────────────────────────────────────────────
  setTrajectory(traj, problem) {
    this.traj = traj?.feasible ? traj : null;
    this.scale = problem?.scene_scale ?? 100;

    this._layoutWorld();
    this._buildPath();
    this._buildCorridor(traj, problem);

    if (!this.traj) {
      this.rocket.visible = false;
      this.velArrow.visible = false;
      this.thrArrow.visible = false;
      return;
    }
    this.rocket.visible = true;
    this.frameCamera();
    this.update(0);
  }

  _layoutWorld() {
    const s = this.scale;
    this.grid.scale.setScalar(s * 3.2);
    this.pad.scale.setScalar(s * 0.055);

    const rs = s * 0.014;                   // rocket stays legible at any scale
    this.rocket.scale.setScalar(rs);
    this.scene.fog.density = 0.28 / (s * 10);
    this.controls.maxDistance = s * 14;
    this.controls.minDistance = s * 0.05;
  }

  _buildPath() {
    while (this.pathGroup.children.length) {
      const c = this.pathGroup.children.pop();
      c.geometry?.dispose();
      c.material?.dispose();
    }
    if (!this.traj) return;

    const pts = this.traj.position.map((p) => new THREE.Vector3(p[0], p[1], p[2]));
    if (pts.length < 2) return;

    const tmax = this.traj.thrust_max || 1;
    const mag = this.traj.thrust.map((a) => Math.hypot(a[0], a[1], a[2]));
    const colAt = (i) => {
      const m = mag[Math.min(i, mag.length - 1)] ?? 0;
      return COLD.clone().lerp(HOT, Math.min(m / tmax, 1));
    };

    // Fat, colour-coded tube: reads far better in 3-D than a hairline.
    const curve = new THREE.CatmullRomCurve3(pts);
    const seg = Math.min(pts.length * 3, 600);
    const tube = new THREE.TubeGeometry(curve, seg, this.scale * 0.0035, 8, false);

    const count = tube.attributes.position.count;
    const radial = 9;                                   // radialSegments + 1
    const colors = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const along = Math.floor(i / radial) / seg;       // 0..1 down the tube
      const c = colAt(Math.round(along * (pts.length - 1)));
      colors[i * 3] = c.r; colors[i * 3 + 1] = c.g; colors[i * 3 + 2] = c.b;
    }
    tube.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    this.pathGroup.add(new THREE.Mesh(tube, new THREE.MeshBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.92 })));

    // Ground shadow of the path, so downrange distance is readable.
    const flat = pts.map((p) => new THREE.Vector3(p.x, 0.001 * this.scale, p.z));
    const fg = new THREE.BufferGeometry().setFromPoints(flat);
    this.pathGroup.add(new THREE.Line(fg, new THREE.LineBasicMaterial({
      color: 0x4dd2ff, transparent: true, opacity: 0.22 })));

    // Vertical drop line at the start, an altitude cue.
    const s0 = pts[0];
    const dg = new THREE.BufferGeometry().setFromPoints([
      s0, new THREE.Vector3(s0.x, 0, s0.z)]);
    this.pathGroup.add(new THREE.Line(dg, new THREE.LineDashedMaterial({
      color: 0x4dd2ff, transparent: true, opacity: 0.3,
      dashSize: this.scale * 0.02, gapSize: this.scale * 0.02 })));
    this.pathGroup.children.at(-1).computeLineDistances();

    this.pathGroup.visible = this.show.trail;
  }

  _buildCorridor(traj, problem) {
    if (this.coneMesh) {
      this.scene.remove(this.coneMesh);
      this.coneMesh.geometry.dispose();
      this.coneMesh.material.dispose();
      this.coneMesh = null;
    }
    const gamma = traj?.diagnostics?.glideslope_deg;
    if (!traj?.feasible || gamma == null || gamma <= 0) return;

    const maxAlt = Math.max(...traj.position.map((p) => p[1]));
    const h = maxAlt * 1.12;
    const radius = h / Math.tan(THREE.MathUtils.degToRad(gamma));

    const geo = new THREE.ConeGeometry(radius, h, 72, 1, true);
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: 0x4dd2ff, transparent: true, opacity: 0.07,
      side: THREE.DoubleSide, depthWrite: false }));
    mesh.rotation.x = Math.PI;      // apex down
    mesh.position.y = h / 2;        // apex lands on the pad

    const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: 0x4dd2ff, wireframe: true, transparent: true, opacity: 0.09,
      depthWrite: false }));
    mesh.add(wire);

    mesh.visible = this.show.cone;
    this.coneMesh = mesh;
    this.scene.add(mesh);
  }

  // ── per-frame ─────────────────────────────────────────────
  /** Sample the trajectory at time `t` (seconds) and pose everything. */
  update(t) {
    if (!this.traj) { this.controls.update(); return null; }

    const ts = this.traj.t_state;
    const T = ts[ts.length - 1];
    const time = Math.min(Math.max(t, 0), T);

    let i = 0;
    while (i < ts.length - 2 && ts[i + 1] < time) i += 1;
    const span = ts[i + 1] - ts[i] || 1;
    const f = Math.min(Math.max((time - ts[i]) / span, 0), 1);

    const lerp3 = (arr) => {
      const a = arr[i], b = arr[i + 1] ?? arr[i];
      return new THREE.Vector3(
        a[0] + (b[0] - a[0]) * f,
        a[1] + (b[1] - a[1]) * f,
        a[2] + (b[2] - a[2]) * f);
    };

    const pos = lerp3(this.traj.position);
    const vel = lerp3(this.traj.velocity);

    const qa = this.traj.attitude[i];
    const qb = this.traj.attitude[i + 1] ?? qa;
    const q = new THREE.Quaternion(qa[0], qa[1], qa[2], qa[3])
      .slerp(new THREE.Quaternion(qb[0], qb[1], qb[2], qb[3]), f);

    this.rocket.position.copy(pos);
    this.rocket.quaternion.copy(q);

    // Control is zero-order hold: pick the interval, do not interpolate.
    const ci = Math.min(
      Math.max(Math.floor(time / (T / this.traj.thrust.length)), 0),
      this.traj.thrust.length - 1);
    const a = this.traj.thrust[ci];
    const aMag = Math.hypot(a[0], a[1], a[2]);
    const frac = Math.min(aMag / (this.traj.thrust_max || 1), 1);

    const lit = frac > 0.015;
    this.plume.visible = lit;
    this.plumeCore.visible = lit;
    if (lit) {
      const flick = 0.92 + Math.random() * 0.16;
      this.plume.scale.set(0.6 + frac * 0.7, frac * 2.4 * flick, 0.6 + frac * 0.7);
      this.plume.position.y = -2.1 * (frac * 2.4 * flick) / 2.4 - 0.2;
      this.plumeCore.scale.set(0.7 + frac * 0.5, frac * 2.2 * flick, 0.7 + frac * 0.5);
      this.plumeCore.position.y = this.plume.position.y * 0.62;
      this.plume.material.opacity = 0.32 + frac * 0.4;
    }

    this._updateArrows(pos, vel, a, aMag);
    this._updateCamera(pos);
    this.controls.update();

    return {
      time,
      altitude: pos.y,
      speed: vel.length(),
      thrust: aMag,
      tilt: aMag > 1e-6
        ? THREE.MathUtils.radToDeg(Math.acos(Math.min(Math.max(a[1] / aMag, -1), 1)))
        : 0,
      position: pos,
    };
  }

  _updateArrows(pos, vel, a, aMag) {
    const on = this.show.vectors;
    this.velArrow.visible = on;
    this.thrArrow.visible = on;
    if (!on) return;

    const L = this.scale * 0.22;
    const vLen = vel.length();
    if (vLen > 1e-6) {
      this.velArrow.position.copy(pos);
      this.velArrow.setDirection(vel.clone().normalize());
      this.velArrow.setLength(L * Math.min(vLen / 80, 1.4), L * 0.16, L * 0.09);
    }
    if (aMag > 1e-6) {
      this.thrArrow.position.copy(pos);
      this.thrArrow.setDirection(new THREE.Vector3(a[0], a[1], a[2]).normalize());
      this.thrArrow.setLength(
        L * Math.min(aMag / (this.traj.thrust_max || 1), 1.4), L * 0.16, L * 0.09);
    }
  }

  _updateCamera(pos) {
    const s = this.scale;
    if (this.camMode === 'orbit') return;          // user drives it

    if (this.camMode === 'chase') {
      const back = new THREE.Vector3(0.55, 0.34, 0.55).multiplyScalar(s * 0.42);
      this.camera.position.lerp(pos.clone().add(back), 0.09);
      this.controls.target.lerp(pos, 0.14);
    } else if (this.camMode === 'side') {
      this.camera.position.lerp(new THREE.Vector3(0, s * 0.45, s * 1.5), 0.09);
      this.controls.target.lerp(new THREE.Vector3(0, s * 0.35, 0), 0.09);
    } else if (this.camMode === 'top') {
      this.camera.position.lerp(new THREE.Vector3(0.01, s * 1.9, 0), 0.09);
      this.controls.target.lerp(new THREE.Vector3(0, 0, 0), 0.09);
    }
  }

  frameCamera() {
    if (!this.traj) return;
    const box = new THREE.Box3();
    for (const p of this.traj.position) box.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
    box.expandByPoint(new THREE.Vector3(0, 0, 0));

    const c = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3()).length() || this.scale;

    this.controls.target.copy(c);
    this.camera.position.copy(c).add(
      new THREE.Vector3(0.75, 0.5, 0.95).normalize().multiplyScalar(size * 1.15));
    this.camera.near = size / 900;
    this.camera.far = Math.max(size * 400, 40000);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  setCamera(mode) {
    this.camMode = mode;
    this.controls.enabled = mode === 'orbit';
    if (mode === 'orbit') this.frameCamera();
  }

  setToggle(key, on) {
    this.show[key] = on;
    if (key === 'grid') this.ground.visible = on;
    if (key === 'trail') this.pathGroup.visible = on;
    if (key === 'cone' && this.coneMesh) this.coneMesh.visible = on;
    if (key === 'vectors') { this.velArrow.visible = on; this.thrArrow.visible = on; }
  }

  resize() {
    const w = this.container.clientWidth;
    const h = this.container.clientHeight;
    if (!w || !h) return;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  render() { this.renderer.render(this.scene, this.camera); }
}
