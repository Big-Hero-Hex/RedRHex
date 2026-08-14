/**
 * Physics-page robot viewer.
 *
 * Renders the RedRHex layout served by /api/physics/robot-geometry and mirrors the
 * values being edited in the Physics preset editor. It shows *parameters*, never live
 * robot state -- the panel has no joint telemetry, and pretending otherwise would make
 * the picture a lie.
 *
 * Frames: the URDF/CAD frame is X = fore-aft, Y = up, Z = lateral, which is already
 * three.js's Y-up convention, so the scene is built in CAD coordinates unchanged. The
 * simulator's +90-deg-about-X spawn rotation is *not* applied here; it exists to map the
 * asset into Isaac's Z-up world, and re-applying it would tip this scene on its side.
 * Physics fields authored in the simulator frame are converted per-value by simToCad().
 *
 * Leg identity is always the canonical index. Leg *labels* are shown as reported by the
 * server, which also reports where the URDF actually puts each leg; the two disagree on
 * the right side today, so the viewer surfaces the mismatch instead of hiding it.
 *
 * This module is the panel's only ES module. It exposes a small imperative API on
 * window.RedRHexRobotView so that the classic-script app.js can drive it.
 */

import * as THREE from "three";

const TAU = Math.PI * 2;
const GROUP_KEYS = ["abad", "main", "damper"];
const MAX_PIXEL_RATIO = 2;
const DRAG_SELECT_SLOP_PX = 4;

/** Coil wound around +Z, used to draw the passive torsion springs. */
class HelixCurve extends THREE.Curve {
  constructor(radius, length, turns) {
    super();
    this.radius = radius;
    this.length = length;
    this.turns = turns;
  }

  getPoint(t, target = new THREE.Vector3()) {
    const angle = t * TAU * this.turns;
    return target.set(
      Math.cos(angle) * this.radius,
      Math.sin(angle) * this.radius,
      (t - 0.5) * this.length,
    );
  }
}

/** Physics fields are authored in the simulator frame (Z up). The scene is CAD (Y up). */
function simToCad(x, y, z) {
  return new THREE.Vector3(x, z, -y);
}

function clamp(value, low, high) {
  return Math.min(high, Math.max(low, value));
}

function lerp(a, b, t) {
  return a + (b - a) * t;
}

function finite(value, fallback = 0) {
  const numeric = typeof value === "number" ? value : Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

/** Read a themed color from the stylesheet so the scene tracks light/dark mode. */
function cssColor(name, fallback) {
  const raw = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  if (!raw) return new THREE.Color(fallback);
  try {
    return new THREE.Color(raw);
  } catch (error) {
    return new THREE.Color(fallback);
  }
}

function palette() {
  return {
    body: cssColor("--surface-2", "#e8eef1"),
    bodyEdge: cssColor("--line-strong", "#aebcc1"),
    leg: cssColor("--red", "#b7362d"),
    legAccent: cssColor("--red-dark", "#8f241e"),
    joints: {
      // Actively driven joints are coloured; the passive spring stays bare metal.
      abad: cssColor("--green", "#16735f"),
      main: cssColor("--blue", "#315e9f"),
      damper: cssColor("--line-strong", "#aebcc1"),
    },
    com: cssColor("--amber", "#8a5a00"),
    ground: cssColor("--surface", "#ffffff"),
    grid: cssColor("--line", "#d4dde1"),
    highlight: cssColor("--green", "#16735f"),
    text: cssColor("--text", "#1f2523"),
  };
}

/* ------------------------------------------------------------------ field parsing */

/**
 * Map a physics field key onto the parts of the scene it affects.
 * Returns { legs: number[] | "all", parts: string[] } or null when the field has no
 * spatial meaning (those live in the readout strip instead).
 */
function fieldTargets(key) {
  if (!key) return null;
  let match = key.match(/passive_spring\.damper_(\d)\./);
  if (match) return { legs: [Number(match[1])], parts: ["damper"] };

  match = key.match(/joint_(?:friction|dynamic_friction|viscous_friction)\.(main|abad|damper)_(\d)$/);
  if (match) return { legs: [Number(match[2])], parts: [match[1]] };

  match = key.match(/hardware_mapping\.abad_target_(?:scale|offset_rad)\.abad_(\d)$/);
  if (match) return { legs: [Number(match[1])], parts: ["abad"] };

  match = key.match(/simulation_physics\.(main_drive|abad|damper)\./);
  if (match) {
    const part = match[1] === "main_drive" ? "main" : match[1];
    return { legs: "all", parts: [part] };
  }
  if (key.startsWith("simulation_physics.mass.")) return { legs: [], parts: ["body", "com"] };
  if (key.startsWith("simulation_physics.rigid_body.")) return { legs: [], parts: ["body"] };
  if (key.startsWith("simulation_physics.ground.")) return { legs: [], parts: ["ground"] };
  return null;
}

/* ------------------------------------------------------------------ orbit control */

/**
 * Minimal orbit control: drag to rotate, wheel to dolly, shift/middle-drag to pan.
 * Hand-written rather than vendoring three's OrbitControls, which would pull in the
 * addons import path for ~60 lines of behaviour.
 */
class Orbit {
  constructor(camera, element, target, radius = 1.15) {
    this.camera = camera;
    this.element = element;
    this.target = target.clone();
    this.home = { theta: -0.85, phi: 1.24, radius, target: target.clone() };
    this.theta = this.home.theta;
    this.phi = this.home.phi;
    this.radius = this.home.radius;
    this.goal = { theta: this.theta, phi: this.phi, radius: this.radius, target: this.target.clone() };
    this.dragging = null;
    this.moved = 0;
    this.onPointerDown = this.handleDown.bind(this);
    this.onPointerMove = this.handleMove.bind(this);
    this.onPointerUp = this.handleUp.bind(this);
    this.onWheel = this.handleWheel.bind(this);
    element.addEventListener("pointerdown", this.onPointerDown);
    element.addEventListener("wheel", this.onWheel, { passive: false });
    window.addEventListener("pointermove", this.onPointerMove);
    window.addEventListener("pointerup", this.onPointerUp);
  }

  handleDown(event) {
    if (event.button !== 0 && event.button !== 1) return;
    this.dragging = {
      x: event.clientX,
      y: event.clientY,
      pan: event.button === 1 || event.shiftKey,
    };
    this.moved = 0;
    this.element.setPointerCapture?.(event.pointerId);
  }

  handleMove(event) {
    if (!this.dragging) return;
    const dx = event.clientX - this.dragging.x;
    const dy = event.clientY - this.dragging.y;
    this.dragging.x = event.clientX;
    this.dragging.y = event.clientY;
    this.moved += Math.abs(dx) + Math.abs(dy);
    if (this.dragging.pan) {
      const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
      const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
      const scale = this.goal.radius * 0.0016;
      this.goal.target.addScaledVector(right, -dx * scale).addScaledVector(up, dy * scale);
    } else {
      this.goal.theta -= dx * 0.007;
      this.goal.phi = clamp(this.goal.phi - dy * 0.007, 0.08, Math.PI - 0.08);
    }
  }

  handleUp(event) {
    if (!this.dragging) return;
    this.element.releasePointerCapture?.(event.pointerId);
    this.dragging = null;
  }

  handleWheel(event) {
    event.preventDefault();
    this.goal.radius = clamp(
      this.goal.radius * Math.exp(event.deltaY * 0.0012),
      this.home.radius * 0.3,
      this.home.radius * 3.5,
    );
  }

  /** True when the last pointer gesture was a drag, so clicks do not select mid-orbit. */
  get wasDrag() {
    return this.moved > DRAG_SELECT_SLOP_PX;
  }

  reset() {
    this.goal.theta = this.home.theta;
    this.goal.phi = this.home.phi;
    this.goal.radius = this.home.radius;
    this.goal.target.copy(this.home.target);
  }

  update() {
    this.theta = lerp(this.theta, this.goal.theta, 0.18);
    this.phi = lerp(this.phi, this.goal.phi, 0.18);
    this.radius = lerp(this.radius, this.goal.radius, 0.18);
    this.target.lerp(this.goal.target, 0.18);
    const sinPhi = Math.sin(this.phi);
    this.camera.position.set(
      this.target.x + this.radius * sinPhi * Math.sin(this.theta),
      this.target.y + this.radius * Math.cos(this.phi),
      this.target.z + this.radius * sinPhi * Math.cos(this.theta),
    );
    this.camera.lookAt(this.target);
  }

  dispose() {
    this.element.removeEventListener("pointerdown", this.onPointerDown);
    this.element.removeEventListener("wheel", this.onWheel);
    window.removeEventListener("pointermove", this.onPointerMove);
    window.removeEventListener("pointerup", this.onPointerUp);
  }
}

/* ------------------------------------------------------------------ scene builder */

class RobotScene {
  constructor(host, layout) {
    this.host = host;
    this.layout = layout;
    this.legs = [];
    this.disposables = [];
    this.frames = 0;
    this.running = false;
    this.hoverLeg = null;
    this.highlight = { legs: [], parts: [] };
    this.colors = palette();

    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, MAX_PIXEL_RATIO));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    host.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(38, 1, 0.05, 40);

    this.root = new THREE.Group();
    this.scene.add(this.root);

    this.buildLights();
    this.buildBody();
    this.buildLegs();
    this.buildGround();
    this.buildCom();
    // Pose first: where the feet land depends on the joint angles, so settling the model
    // onto the ground plane is only meaningful once the default pose is applied.
    this.apply({});
    this.recentre();
    this.apply({});

    this.orbit = new Orbit(this.camera, this.renderer.domElement, this.focus, this.framingRadius);
    this.raycaster = new THREE.Raycaster();
    this.pointer = new THREE.Vector2();
    this.bindPicking();

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(host);
    this.resize();
  }

  track(...objects) {
    this.disposables.push(...objects);
    return objects[0];
  }

  buildLights() {
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x404448, 1.35));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(0.6, 1.4, 0.8);
    key.castShadow = true;
    key.shadow.mapSize.set(1024, 1024);
    key.shadow.camera.near = 0.1;
    key.shadow.camera.far = 5;
    const extent = 0.7;
    key.shadow.camera.left = -extent;
    key.shadow.camera.right = extent;
    key.shadow.camera.top = extent;
    key.shadow.camera.bottom = -extent;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0xffffff, 0.4);
    fill.position.set(-0.8, 0.5, -0.6);
    this.scene.add(fill);
  }

  buildBody() {
    const body = this.layout.body;
    const [xMin, xMax] = body.fore_aft_span_m;
    const centre = body.lateral_center_m;
    // Front and rear legs bolt straight to the shell; the middle pair is splayed
    // outboard so it clears them. The narrower pair therefore sets the shell width,
    // and the splayed pair reaches it on visible outriggers.
    // Pull the flank inboard of the nearest mount so the hip hardware stands clear of
    // the shell instead of being half-swallowed by it.
    const halfWidth = Math.min(
      ...this.layout.legs.map((leg) => Math.abs(leg.mount[2] - centre)),
    ) - 0.016;
    this.deck = { halfWidth, centre, y: body.vertical_m };

    // Hips sit above the mount plane, on the standoffs the URDF calls top/bottom
    // connectors. Size the shell to reach them so the legs bolt to its flank instead of
    // perching on the lid.
    const hipY = this.layout.legs.reduce(
      (total, leg) => total + leg.mount[1] + leg.joints.abad.origin_m[1],
      0,
    ) / this.layout.legs.length;
    const top = hipY + 0.003;
    const bottom = hipY - 0.058;
    const height = top - bottom;
    const length = xMax - xMin + 0.075;
    const geometry = this.track(new THREE.BoxGeometry(length, height, halfWidth * 2));
    const material = this.track(
      new THREE.MeshStandardMaterial({ color: this.colors.body, roughness: 0.55, metalness: 0.2 }),
    );
    this.body = new THREE.Mesh(geometry, material);
    this.body.position.set((xMin + xMax) / 2, (top + bottom) / 2, centre);
    this.body.castShadow = true;
    this.body.receiveShadow = true;
    this.body.userData.part = "body";
    this.root.add(this.body);
    this.deck.top = top;
    this.deck.hipY = hipY;

    const edges = this.track(new THREE.EdgesGeometry(geometry));
    const edgeMaterial = this.track(new THREE.LineBasicMaterial({ color: this.colors.bodyEdge }));
    this.bodyEdges = new THREE.LineSegments(edges, edgeMaterial);
    this.body.add(this.bodyEdges);
  }

  /** Orient a mesh whose geometry is built along +Y so that it lies along `axis`. */
  static alignTo(mesh, axis) {
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), axis.clone().normalize());
  }

  /** Structural bar spanning `from` -> `to`, both in the parent's frame. */
  makeStrut(from, to, thickness, material) {
    const delta = new THREE.Vector3().subVectors(to, from);
    const length = delta.length();
    if (length < 1e-5) return null;
    const geometry = this.track(new THREE.BoxGeometry(thickness, length, thickness));
    const bar = new THREE.Mesh(geometry, material);
    bar.position.copy(from).addScaledVector(delta, 0.5);
    RobotScene.alignTo(bar, delta);
    bar.castShadow = true;
    return bar;
  }

  buildLegs() {
    const arc = Math.PI * 1.45;
    // A torus arc starts at +X and sweeps counter-clockwise, so its midpoint sits at
    // arc/2. Rotating by this puts the opening of the C facing straight down.
    const arcDown = Math.PI * 1.5 - arc / 2;
    const legGeometry = this.track(new THREE.TorusGeometry(0.0845, 0.0075, 10, 40, arc));

    // Each joint family gets its own shape, because they do not work alike: the ABAD is
    // a servoed hinge, the main drive is a continuously rotating motor, and the torsion
    // spring is passive. Shape carries the mechanism; colour only reinforces it.
    const hingeGeometry = this.track(new THREE.CylinderGeometry(0.0085, 0.0085, 0.032, 16));
    const hingeCapGeometry = this.track(new THREE.CylinderGeometry(0.0115, 0.0115, 0.005, 16));
    const swingGeometry = this.track(new THREE.TorusGeometry(0.019, 0.0016, 6, 20, Math.PI * 0.5));
    const motorGeometry = this.track(new THREE.CylinderGeometry(0.0145, 0.0145, 0.026, 20));
    const rotorGeometry = this.track(new THREE.BoxGeometry(0.0045, 0.027, 0.019));
    const driveArcGeometry = this.track(new THREE.TorusGeometry(0.0215, 0.0016, 6, 24, Math.PI * 1.4));
    const springGeometry = this.track(new THREE.TubeGeometry(new HelixCurve(0.0105, 0.019, 3.5), 96, 0.0018, 7, false));
    const ringGeometry = this.track(new THREE.TorusGeometry(0.021, 0.0022, 8, 22));

    const structureMaterial = this.track(
      new THREE.MeshStandardMaterial({ color: this.colors.bodyEdge, roughness: 0.6, metalness: 0.3 }),
    );

    for (const spec of this.layout.legs) {
      const group = new THREE.Group();
      group.position.fromArray(spec.mount);
      this.root.add(group);

      const abadOrigin = new THREE.Vector3().fromArray(spec.joints.abad.origin_m);
      // The splayed middle legs stand outboard of the shell and reach it on a bracket;
      // the front and rear pairs bolt straight to the flank, so their strut is empty.
      const shellEdge = this.deck.centre + Math.sign(spec.mount[2] - this.deck.centre) * this.deck.halfWidth;
      const outrigger = this.makeStrut(
        new THREE.Vector3(abadOrigin.x, abadOrigin.y, shellEdge - spec.mount[2]),
        abadOrigin,
        0.014,
        structureMaterial,
      );
      if (outrigger) group.add(outrigger);

      const pivots = {};
      let parent = group;
      for (const key of GROUP_KEYS) {
        const joint = spec.joints[key];
        const pivot = new THREE.Group();
        pivot.position.fromArray(joint.origin_m);
        const axis = new THREE.Vector3().fromArray(joint.axis);
        if (axis.lengthSq() === 0) axis.set(0, 0, 1);
        axis.normalize();
        pivot.userData.axis = axis;
        pivot.userData.initRad = finite(joint.init_rad, 0);
        parent.add(pivot);

        const material = this.track(
          new THREE.MeshStandardMaterial({
            color: this.colors.joints[key],
            roughness: key === "damper" ? 0.35 : 0.45,
            metalness: key === "damper" ? 0.65 : 0.25,
          }),
        );
        const marker = new THREE.Mesh(
          key === "abad" ? hingeGeometry : key === "main" ? motorGeometry : springGeometry,
          material,
        );
        marker.castShadow = true;
        marker.userData.part = key;
        marker.userData.legIndex = spec.index;

        if (key === "damper") {
          // The coil is modelled around +Z, so it only needs turning onto the joint axis.
          marker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
        } else {
          RobotScene.alignTo(marker, axis);
        }
        pivot.add(marker);

        if (key === "abad") {
          // Hinge pin with visible end caps, plus the arc it abducts through.
          for (const side of [-1, 1]) {
            const cap = new THREE.Mesh(hingeCapGeometry, material);
            RobotScene.alignTo(cap, axis);
            cap.position.copy(axis).multiplyScalar(side * 0.016);
            pivot.add(cap);
          }
          const swing = new THREE.Mesh(swingGeometry, material);
          swing.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
          swing.rotation.z += Math.PI * 0.75;
          pivot.add(swing);
        } else if (key === "main") {
          // Rotor key and a sweep arc: this one turns continuously under velocity control.
          const rotor = new THREE.Mesh(rotorGeometry, material);
          RobotScene.alignTo(rotor, axis);
          pivot.add(rotor);
          const sweep = new THREE.Mesh(driveArcGeometry, material);
          sweep.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
          pivot.add(sweep);
        }

        const ring = new THREE.Mesh(
          ringGeometry,
          this.track(new THREE.MeshBasicMaterial({ color: this.colors.joints[key], transparent: true, opacity: 0.4 })),
        );
        ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), axis);
        ring.visible = false;
        pivot.add(ring);

        pivots[key] = { pivot, marker, ring, material };
        parent = pivot;
      }

      // The torsion spring hangs off the end of a real link, not off the motor itself.
      const arm = this.makeStrut(
        new THREE.Vector3(),
        new THREE.Vector3().fromArray(spec.joints.damper.origin_m),
        0.0105,
        structureMaterial,
      );
      if (arm) pivots.main.pivot.add(arm);

      const spawnHeading = GROUP_KEYS.reduce((total, key) => {
        const joint = spec.joints[key];
        return total + finite(joint.init_rad, 0) * finite(joint.axis?.[2], 0);
      }, 0);
      const legMaterial = this.track(
        new THREE.MeshStandardMaterial({ color: this.colors.leg, roughness: 0.45, metalness: 0.1 }),
      );
      const leg = new THREE.Mesh(legGeometry, legMaterial);
      leg.castShadow = true;
      leg.rotation.z = arcDown - spawnHeading;
      leg.userData.part = "damper";
      leg.userData.legIndex = spec.index;
      pivots.damper.pivot.add(leg);

      this.legs.push({ spec, group, pivots, leg, legMaterial });
    }
  }

  buildGround() {
    const geometry = this.track(new THREE.PlaneGeometry(4, 4));
    const material = this.track(
      new THREE.MeshStandardMaterial({ color: this.colors.ground, roughness: 0.95, metalness: 0 }),
    );
    this.ground = new THREE.Mesh(geometry, material);
    this.ground.rotation.x = -Math.PI / 2;
    this.ground.receiveShadow = true;
    this.ground.userData.part = "ground";
    this.scene.add(this.ground);

    this.grid = new THREE.GridHelper(2, 24, this.colors.grid, this.colors.grid);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.5;
    this.scene.add(this.grid);
    this.disposables.push(this.grid.geometry, this.grid.material);
  }

  buildCom() {
    const geometry = this.track(new THREE.SphereGeometry(0.019, 20, 14));
    const material = this.track(
      new THREE.MeshStandardMaterial({ color: this.colors.com, roughness: 0.3, emissive: this.colors.com, emissiveIntensity: 0.25 }),
    );
    this.com = new THREE.Mesh(geometry, material);
    this.com.userData.part = "com";
    this.root.add(this.com);

    const lineGeometry = this.track(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]));
    const lineMaterial = this.track(new THREE.LineDashedMaterial({ color: this.colors.com, dashSize: 0.012, gapSize: 0.01 }));
    this.comDrop = new THREE.Line(lineGeometry, lineMaterial);
    this.root.add(this.comDrop);

    this.nominalCom = new THREE.Vector3().fromArray(this.layout.body.nominal_com_m);
    this.setComOffset(new THREE.Vector3());
  }

  /** Drop the model so its lowest point rests on the ground plane, then centre it. */
  recentre() {
    this.root.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(this.root);
    const centre = box.getCenter(new THREE.Vector3());
    this.root.position.x -= centre.x;
    this.root.position.z -= centre.z;
    this.root.position.y -= box.min.y;
    this.root.updateMatrixWorld(true);
    const settled = new THREE.Box3().setFromObject(this.root);
    this.focus = settled.getCenter(new THREE.Vector3());
    this.modelBox = settled;
    this.framingRadius = this.computeFraming();
  }

  /**
   * Smallest camera distance that keeps the whole model on screen at the home angle.
   *
   * Margin-tuned bounding-sphere fits either waste the frame or clip the feet, because
   * the stage is far wider than it is tall. This projects the eight bounding-box corners
   * onto the camera basis and solves each for the distance at which it reaches the edge
   * of the frustum, then takes the binding one -- exact for any aspect ratio.
   */
  computeFraming() {
    const box = this.modelBox;
    if (!box) return 1.15;
    const aspect = this.camera.aspect || 1;
    const tanV = Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2);
    const tanH = tanV * aspect;

    const { theta, phi } = this.orbit ? this.orbit.home : { theta: -0.85, phi: 1.24 };
    const forward = new THREE.Vector3(
      Math.sin(phi) * Math.sin(theta),
      Math.cos(phi),
      Math.sin(phi) * Math.cos(theta),
    ).normalize();
    const right = new THREE.Vector3().crossVectors(new THREE.Vector3(0, 1, 0), forward).normalize();
    const up = new THREE.Vector3().crossVectors(forward, right).normalize();

    const centre = box.getCenter(new THREE.Vector3());
    let distance = 0;
    for (let corner = 0; corner < 8; corner += 1) {
      const point = new THREE.Vector3(
        corner & 1 ? box.max.x : box.min.x,
        corner & 2 ? box.max.y : box.min.y,
        corner & 4 ? box.max.z : box.min.z,
      ).sub(centre);
      const depth = point.dot(forward);
      distance = Math.max(
        distance,
        depth + Math.abs(point.dot(right)) / tanH,
        depth + Math.abs(point.dot(up)) / tanV,
      );
    }
    // The collapsed sticky strip is short and very wide, so spend less of it on margin.
    const compact = this.host.clientHeight > 0 && this.host.clientHeight < 260;
    return Math.max(distance * (compact ? 1.0 : 1.12), 0.25);
  }

  bindPicking() {
    const canvas = this.renderer.domElement;
    this.onCanvasMove = (event) => {
      const rect = canvas.getBoundingClientRect();
      this.pointer.set(
        ((event.clientX - rect.left) / rect.width) * 2 - 1,
        -((event.clientY - rect.top) / rect.height) * 2 + 1,
      );
      this.pointerFresh = true;
    };
    this.onCanvasLeave = () => {
      this.pointerFresh = false;
      this.setHover(null);
    };
    this.onCanvasUp = (event) => {
      if (event.button !== 0 || this.orbit.wasDrag) return;
      const hit = this.pickLeg();
      if (hit === null) return;
      const entry = this.legs.find((item) => item.spec.index === hit);
      if (!entry) return;
      const spec = entry.spec;
      canvas.dispatchEvent(
        new CustomEvent("redrhex:leg-selected", {
          bubbles: true,
          detail: {
            index: spec.index,
            contractLabel: spec.contract_label,
            geometricLabel: spec.geometric_label,
          },
        }),
      );
    };
    canvas.addEventListener("pointermove", this.onCanvasMove);
    canvas.addEventListener("pointerleave", this.onCanvasLeave);
    canvas.addEventListener("pointerup", this.onCanvasUp);
  }

  pickLeg() {
    this.raycaster.setFromCamera(this.pointer, this.camera);
    const hits = this.raycaster.intersectObject(this.root, true);
    for (const hit of hits) {
      const index = hit.object.userData?.legIndex;
      if (typeof index === "number") return index;
    }
    return null;
  }

  setHover(index) {
    if (this.hoverLeg === index) return;
    this.hoverLeg = index;
    this.renderer.domElement.style.cursor = index === null ? "grab" : "pointer";
    this.host.dispatchEvent(
      new CustomEvent("redrhex:leg-hover", {
        bubbles: true,
        detail: index === null ? null : { index, ...(this.legs.find((item) => item.spec.index === index)?.spec || {}) },
      }),
    );
    this.applyMaterials();
  }

  /* ---------------------------------------------------------------- value binding */

  setComOffset(offset) {
    const position = this.nominalCom.clone().add(offset);
    this.com.position.copy(position);
    const points = this.comDrop.geometry.attributes.position;
    points.setXYZ(0, position.x, position.y, position.z);
    points.setXYZ(1, position.x, -this.root.position.y, position.z);
    points.needsUpdate = true;
    this.comDrop.computeLineDistances();
  }

  apply(values) {
    const get = (key, fallback) => finite(values?.[key], fallback);

    for (const entry of this.legs) {
      const index = entry.spec.index;
      const rest = get(
        `simulation_physics.passive_spring.damper_${index}.rest_position_rad`,
        entry.pivots.damper.pivot.userData.initRad,
      );
      const abadOffset = get(`hardware_mapping.abad_target_offset_rad.abad_${index}`, 0);

      for (const key of GROUP_KEYS) {
        const { pivot } = entry.pivots[key];
        let angle = pivot.userData.initRad;
        if (key === "damper") angle = rest;
        if (key === "abad") angle += abadOffset;
        pivot.quaternion.setFromAxisAngle(pivot.userData.axis, angle);
      }

      // Stiffness reads as colour saturation: a stiffer spring looks harder.
      const stiffness = get(`simulation_physics.passive_spring.damper_${index}.stiffness`, 200);
      const t = clamp(stiffness / 400, 0, 1);
      entry.legMaterial.color.copy(this.colors.leg).lerp(this.colors.legAccent, t);

      for (const key of GROUP_KEYS) {
        const friction = get(`simulation_physics.joint_friction.${key}_${index}`, 0);
        const scale = 1 + clamp(friction * 4, 0, 1.6);
        entry.pivots[key].ring.scale.setScalar(scale);
        entry.pivots[key].ring.visible = friction > 0;
      }
    }

    const offset = simToCad(
      get("simulation_physics.mass.com_offset_m.0", 0),
      get("simulation_physics.mass.com_offset_m.1", 0),
      get("simulation_physics.mass.com_offset_m.2", 0),
    );
    this.setComOffset(offset);

    const massScale = get("simulation_physics.mass.scale", 1);
    this.body.material.color.copy(this.colors.body).lerp(this.colors.text, clamp((massScale - 1) * 0.5, 0, 0.45));

    const staticFriction = get("simulation_physics.ground.static_friction", 1.2);
    const slick = new THREE.Color(0.62, 0.76, 0.9);
    const grippy = new THREE.Color(0.82, 0.62, 0.42);
    this.ground.material.color.copy(this.colors.ground).lerp(
      staticFriction < 1.2 ? slick : grippy,
      clamp(Math.abs(staticFriction - 1.2) / 1.8, 0, 0.55),
    );
    this.applyMaterials();
  }

  setHighlight(targets) {
    this.highlight = targets || { legs: [], parts: [] };
    this.applyMaterials();
  }

  applyMaterials() {
    const { legs, parts } = this.highlight;
    const legMatches = (index) => legs === "all" || (Array.isArray(legs) && legs.includes(index));
    for (const entry of this.legs) {
      const hovered = this.hoverLeg === entry.spec.index;
      const active = hovered || (legMatches(entry.spec.index) && parts.includes("damper"));
      entry.legMaterial.emissive = active ? this.colors.highlight : new THREE.Color(0x000000);
      entry.legMaterial.emissiveIntensity = active ? 0.45 : 0;
      for (const key of GROUP_KEYS) {
        const marker = entry.pivots[key].marker;
        const on = hovered || (legMatches(entry.spec.index) && parts.includes(key));
        marker.material.emissive = on ? this.colors.highlight : new THREE.Color(0x000000);
        marker.material.emissiveIntensity = on ? 0.6 : 0;
        marker.scale.setScalar(on ? 1.5 : 1);
      }
    }
    const bodyOn = parts.includes("body");
    this.body.material.emissive = bodyOn ? this.colors.highlight : new THREE.Color(0x000000);
    this.body.material.emissiveIntensity = bodyOn ? 0.25 : 0;
    const comOn = parts.includes("com");
    this.com.material.emissiveIntensity = comOn ? 0.9 : 0.25;
    this.com.scale.setScalar(comOn ? 1.35 : 1);
    const groundOn = parts.includes("ground");
    this.grid.material.opacity = groundOn ? 0.95 : 0.5;
  }

  refreshTheme() {
    this.colors = palette();
    this.body.material.color.copy(this.colors.body);
    this.bodyEdges.material.color.copy(this.colors.bodyEdge);
    this.ground.material.color.copy(this.colors.ground);
    this.grid.material.color.copy(this.colors.grid);
    this.com.material.color.copy(this.colors.com);
    this.com.material.emissive.copy(this.colors.com);
    this.comDrop.material.color.copy(this.colors.com);
    for (const entry of this.legs) {
      entry.legMaterial.color.copy(this.colors.leg);
      for (const key of GROUP_KEYS) {
        entry.pivots[key].material.color.copy(this.colors.joints[key]);
        entry.pivots[key].ring.material.color.copy(this.colors.joints[key]);
      }
    }
    this.applyMaterials();
  }

  resize() {
    const width = this.host.clientWidth;
    const height = this.host.clientHeight;
    if (!width || !height) return;
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    if (!this.orbit) return;
    const radius = this.computeFraming();
    const untouched = Math.abs(this.orbit.goal.radius - this.orbit.home.radius) < 1e-6;
    this.orbit.home.radius = radius;
    if (untouched) this.orbit.goal.radius = radius;
  }

  start() {
    if (this.running) return;
    this.running = true;
    const tick = () => {
      if (!this.running) return;
      this.raf = requestAnimationFrame(tick);
      this.orbit.update();
      if (this.pointerFresh) this.setHover(this.pickLeg());
      this.renderer.render(this.scene, this.camera);
      this.frames += 1;
    };
    this.raf = requestAnimationFrame(tick);
  }

  stop() {
    this.running = false;
    if (this.raf) cancelAnimationFrame(this.raf);
    this.raf = null;
  }

  dispose() {
    this.stop();
    this.resizeObserver.disconnect();
    this.orbit.dispose();
    const canvas = this.renderer.domElement;
    canvas.removeEventListener("pointermove", this.onCanvasMove);
    canvas.removeEventListener("pointerleave", this.onCanvasLeave);
    canvas.removeEventListener("pointerup", this.onCanvasUp);
    for (const item of this.disposables) item.dispose?.();
    this.renderer.dispose();
    canvas.remove();
  }
}

/* ------------------------------------------------------------------ SVG fallback */

/**
 * Top-down schematic used when WebGL is unavailable or the scene fails to build.
 * It keeps the two things that carry the most meaning -- true leg positions and
 * click-to-filter -- so the feature degrades rather than disappearing.
 */
function buildFallback(host, layout) {
  const NS = "http://www.w3.org/2000/svg";
  const [xMin, xMax] = layout.body.fore_aft_span_m;
  const [zMin, zMax] = layout.body.lateral_span_m;
  const pad = 0.09;
  const width = xMax - xMin + pad * 2;
  const height = zMax - zMin + pad * 2;

  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `${xMin - pad} ${zMin - pad} ${width} ${height}`);
  svg.setAttribute("class", "robot-fallback-svg");
  svg.setAttribute("role", "group");
  svg.setAttribute("aria-label", "Top-down robot layout");

  const shell = document.createElementNS(NS, "rect");
  shell.setAttribute("x", xMin);
  shell.setAttribute("y", zMin + (zMax - zMin) * 0.22);
  shell.setAttribute("width", xMax - xMin);
  shell.setAttribute("height", (zMax - zMin) * 0.56);
  shell.setAttribute("rx", 0.03);
  shell.setAttribute("class", "robot-fallback-body");
  svg.appendChild(shell);

  const com = document.createElementNS(NS, "circle");
  com.setAttribute("r", 0.016);
  com.setAttribute("class", "robot-fallback-com");
  svg.appendChild(com);

  for (const spec of layout.legs) {
    const node = document.createElementNS(NS, "g");
    node.setAttribute("class", "robot-fallback-leg");
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", "button");
    node.dataset.legIndex = String(spec.index);
    node.setAttribute("aria-label", `Leg ${spec.index}: ${spec.geometric_label}`);

    const dot = document.createElementNS(NS, "circle");
    dot.setAttribute("cx", spec.mount[0]);
    dot.setAttribute("cy", spec.mount[2]);
    dot.setAttribute("r", 0.026);
    node.appendChild(dot);

    const text = document.createElementNS(NS, "text");
    text.setAttribute("x", spec.mount[0]);
    text.setAttribute("y", spec.mount[2] + 0.011);
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("font-size", "0.03");
    text.textContent = String(spec.index);
    node.appendChild(text);

    const select = () =>
      svg.dispatchEvent(
        new CustomEvent("redrhex:leg-selected", {
          bubbles: true,
          detail: { index: spec.index, contractLabel: spec.contract_label, geometricLabel: spec.geometric_label },
        }),
      );
    node.addEventListener("click", select);
    node.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        select();
      }
    });
    svg.appendChild(node);
  }

  host.appendChild(svg);
  const nominal = layout.body.nominal_com_m;
  return {
    apply(values) {
      const get = (key) => finite(values?.[key], 0);
      // Fore-aft and lateral only; the top-down view has no vertical axis to show.
      com.setAttribute("cx", nominal[0] + get("simulation_physics.mass.com_offset_m.0"));
      com.setAttribute("cy", nominal[2] - get("simulation_physics.mass.com_offset_m.1"));
    },
    dispose() {
      svg.remove();
    },
  };
}

function webglAvailable() {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch (error) {
    return false;
  }
}

/* ------------------------------------------------------------------ public facade */

const view = {
  scene: null,
  fallback: null,
  layout: null,
  host: null,
  values: {},
  mode: "idle",

  /**
   * Build the viewer into `host`. Never throws: any failure downgrades to the SVG
   * schematic so an exception here can never take the Physics editor down with it.
   */
  mount(host, layout, options = {}) {
    this.dispose();
    this.host = host;
    this.layout = layout;
    const forceFallback = options.forceFallback || !webglAvailable();
    if (!forceFallback) {
      try {
        this.scene = new RobotScene(host, layout);
        this.scene.apply(this.values);
        this.mode = "webgl";
        this.scene.renderer.domElement.addEventListener("webglcontextlost", (event) => {
          event.preventDefault();
          this.downgrade("WebGL context lost");
        });
        this.bindTheme();
        return this.mode;
      } catch (error) {
        console.warn("RedRHex robot viewer: falling back to schematic", error);
        this.scene?.dispose?.();
        this.scene = null;
      }
    }
    this.fallback = buildFallback(host, layout);
    this.fallback.apply(this.values);
    this.mode = "fallback";
    return this.mode;
  },

  downgrade(reason) {
    if (this.mode !== "webgl" || !this.host || !this.layout) return;
    console.warn(`RedRHex robot viewer: ${reason}`);
    this.scene?.dispose?.();
    this.scene = null;
    this.fallback = buildFallback(this.host, this.layout);
    this.fallback.apply(this.values);
    this.mode = "fallback";
    this.host.dispatchEvent(new CustomEvent("redrhex:robot-view-downgraded", { bubbles: true, detail: { reason } }));
  },

  bindTheme() {
    if (this.themeObserver) this.themeObserver.disconnect();
    this.themeObserver = new MutationObserver(() => this.scene?.refreshTheme());
    this.themeObserver.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
  },

  update(values) {
    this.values = values || {};
    try {
      this.scene?.apply(this.values);
      this.fallback?.apply(this.values);
    } catch (error) {
      console.warn("RedRHex robot viewer: update failed", error);
    }
  },

  highlightField(key) {
    try {
      this.scene?.setHighlight(fieldTargets(key) || { legs: [], parts: [] });
    } catch (error) {
      /* highlighting is decorative; never surface a failure here */
    }
  },

  setRunning(running) {
    if (!this.scene) return;
    if (running) this.scene.start();
    else this.scene.stop();
  },

  resetView() {
    this.scene?.orbit.reset();
  },

  /** Frame counter, used by the UI tests to prove the loop pauses when hidden. */
  frameCount() {
    return this.scene ? this.scene.frames : 0;
  },

  dispose() {
    this.themeObserver?.disconnect();
    this.themeObserver = null;
    this.scene?.dispose();
    this.fallback?.dispose();
    this.scene = null;
    this.fallback = null;
    this.mode = "idle";
  },
};

window.RedRHexRobotView = view;
document.dispatchEvent(new CustomEvent("redrhex:robot-view-ready"));

export default view;
