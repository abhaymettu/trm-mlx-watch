// The same TRM forward pass as trm/model.py, in WebGPU compute shaders, from the exported float32 weights.
// net(h) = 2 x MixerBlock; MixerBlock: h = rms(h + cellMLP(h^T)^T); h = rms(h + chanMLP(h)); MLP = SwiGLU.
const CELLS = 81, CLASSES = 9;

const WGSL = /* wgsl */ `
struct MM { M: u32, N: u32, K: u32, sa0: u32, sa1: u32, sb0: u32, sb1: u32, so0: u32, so1: u32, bOff: u32 }
@group(0) @binding(0) var<uniform> mm: MM;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read> W: array<f32>;
@group(0) @binding(3) var<storage, read_write> O: array<f32>;
var<workgroup> tA: array<f32, 256>;
var<workgroup> tB: array<f32, 256>;
@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>,
          @builtin(workgroup_id) wg: vec3<u32>) {
  let i = g.x; let j = g.y;
  let bi = wg.x * 16u; let bj = wg.y * 16u;
  var acc = 0.0;
  let tiles = (mm.K + 15u) / 16u;
  for (var t = 0u; t < tiles; t++) {
    let k = t * 16u;
    let ai = bi + l.y; let ak = k + l.x;
    tA[l.y * 16u + l.x] = select(0.0, A[ai * mm.sa0 + ak * mm.sa1], ai < mm.M && ak < mm.K);
    let bjj = bj + l.y;
    tB[l.y * 16u + l.x] = select(0.0, W[mm.bOff + bjj * mm.sb0 + ak * mm.sb1], bjj < mm.N && ak < mm.K);
    workgroupBarrier();
    for (var q = 0u; q < 16u; q++) { acc += tA[l.x * 16u + q] * tB[l.y * 16u + q]; }
    workgroupBarrier();
  }
  if (i < mm.M && j < mm.N) { O[i * mm.so0 + j * mm.so1] = acc; }
}`;
const WGSL_SWIGLU = /* wgsl */ `
struct SG { rows: u32, hidden: u32 }
@group(0) @binding(0) var<uniform> sg: SG;
@group(0) @binding(1) var<storage, read> G: array<f32>;
@group(0) @binding(2) var<storage, read_write> U: array<f32>;
@compute @workgroup_size(64)
fn swiglu(@builtin(global_invocation_id) g: vec3<u32>) {
  let idx = g.x;
  if (idx >= sg.rows * sg.hidden) { return; }
  let r = idx / sg.hidden; let j = idx % sg.hidden;
  let a = G[r * 2u * sg.hidden + j]; let b = G[r * 2u * sg.hidden + sg.hidden + j];
  U[idx] = a / (1.0 + exp(-a)) * b;
}`;
const WGSL_NORM = /* wgsl */ `
struct NM { rows: u32, dim: u32, wOff: u32 }
@group(0) @binding(0) var<uniform> nm: NM;
@group(0) @binding(1) var<storage, read> X: array<f32>;
@group(0) @binding(2) var<storage, read> T: array<f32>;
@group(0) @binding(3) var<storage, read> W: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(64)
fn addnorm(@builtin(global_invocation_id) g: vec3<u32>) {
  let r = g.x;
  if (r >= nm.rows) { return; }
  var ss = 0.0;
  for (var d = 0u; d < nm.dim; d++) { let v = X[r * nm.dim + d] + T[r * nm.dim + d]; ss += v * v; }
  let s = inverseSqrt(ss / f32(nm.dim) + 1e-5);
  for (var d = 0u; d < nm.dim; d++) { O[r * nm.dim + d] = (X[r * nm.dim + d] + T[r * nm.dim + d]) * s * W[nm.wOff + d]; }
}`;
const WGSL_ADD = /* wgsl */ `
@group(0) @binding(0) var<uniform> n: u32;
@group(0) @binding(1) var<storage, read> P: array<f32>;
@group(0) @binding(2) var<storage, read> Q: array<f32>;
@group(0) @binding(3) var<storage, read> R: array<f32>;
@group(0) @binding(4) var<storage, read_write> O: array<f32>;
@compute @workgroup_size(64)
fn add3(@builtin(global_invocation_id) g: vec3<u32>) { if (g.x < n) { O[g.x] = P[g.x] + Q[g.x] + R[g.x]; } }`;

export class LiveTRM {
  static async create(assets) {
    if (!navigator.gpu) throw new Error("this browser has no WebGPU");
    const adapter = await navigator.gpu.requestAdapter();
    if (!adapter) throw new Error("no WebGPU adapter");
    const device = await adapter.requestDevice();
    const [manifest, bin] = await Promise.all([
      fetch(`${assets}/weights.json`).then((r) => r.json()),
      fetch(`${assets}/weights.bin`).then((r) => r.arrayBuffer()),
    ]);
    const live = new LiveTRM(device, manifest, new Float32Array(bin));
    const i = adapter.info || {};
    live.adapter = [i.vendor, i.architecture, i.description].filter(Boolean).join(" ") || "unknown adapter";
    return live;
  }

  constructor(device, manifest, w) {
    this.dev = device; this.cfg = manifest.config; this.w = w;
    this.t = Object.fromEntries(manifest.tensors.map((t) => [t.name, t]));
    const d = this.cfg.d, n = CELLS * d;
    this.hidC = this.t["blocks.0.cell_mlp.down.weight"].shape[1];
    this.hidD = this.t["blocks.0.chan_mlp.down.weight"].shape[1];
    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const buf = (len) => device.createBuffer({ size: len * 4, usage: S });
    this.W = device.createBuffer({ size: w.byteLength, usage: S });
    device.queue.writeBuffer(this.W, 0, w);
    Object.assign(this, { x: buf(n), y: buf(n), z: buf(n), zero: buf(n), s: buf(n), o: buf(n), h1: buf(n), h2: buf(n),
      tt: buf(n), g: buf(Math.max(d * 2 * this.hidC, CELLS * 2 * this.hidD)), u: buf(Math.max(d * this.hidC, CELLS * this.hidD)) });
    this.stage = device.createBuffer({ size: n * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const pipe = (code, entry) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: entry } });
    this.p = { mm: pipe(WGSL, "matmul"), sg: pipe(WGSL_SWIGLU, "swiglu"), nm: pipe(WGSL_NORM, "addnorm"), add: pipe(WGSL_ADD, "add3") };
    this.netOps = this.buildNet();
    this.addZ = this.op(this.p.add, [n], [this.x, this.y, this.z, this.s], [Math.ceil(n / 64)]);
    this.addY = this.op(this.p.add, [n], [this.y, this.z, this.zero, this.s], [Math.ceil(n / 64)]);
  }

  op(pipeline, uni, bufs, groups) {
    const ub = this.dev.createBuffer({ size: Math.max(16, Math.ceil(uni.length * 4 / 16) * 16), usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.dev.queue.writeBuffer(ub, 0, new Uint32Array(uni));
    const entries = [{ binding: 0, resource: { buffer: ub } }, ...bufs.map((b, i) => ({ binding: i + 1, resource: { buffer: b } }))];
    return { pipeline, bind: this.dev.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries }), groups };
  }

  off(name) { return this.t[name].offset; }

  buildNet() {
    const d = this.cfg.d, hc = this.hidC, hd = this.hidD, ops = [];
    const mm = (M, N, K, sa0, sa1, sb0, sb1, so0, so1, name, A, O) =>
      ops.push(this.op(this.p.mm, [M, N, K, sa0, sa1, sb0, sb1, so0, so1, this.off(name)], [A, this.W, O], [Math.ceil(M / 16), Math.ceil(N / 16)]));
    const sg = (rows, hid) => ops.push(this.op(this.p.sg, [rows, hid], [this.g, this.u], [Math.ceil(rows * hid / 64)]));
    const nm = (name, X, O) => ops.push(this.op(this.p.nm, [CELLS, d, this.off(name)], [X, this.tt, this.W, O], [2]));
    for (let b = 0; b < this.cfg.layers; b++) {
      const X = b === 0 ? this.s : this.h2, out = b === this.cfg.layers - 1 ? this.o : this.h2;
      const P = `blocks.${b}.`;
      // cell mixing: rows are channels (d), contraction over the 81 cells, written back as (81, d)
      mm(d, 2 * hc, CELLS, 1, d, CELLS, 1, 2 * hc, 1, P + "cell_mlp.gate_up.weight", X, this.g);
      sg(d, hc);
      mm(d, CELLS, hc, hc, 1, hc, 1, 1, d, P + "cell_mlp.down.weight", this.u, this.tt);
      nm(P + "norm1.weight", X, this.h1);
      mm(CELLS, 2 * hd, d, d, 1, d, 1, 2 * hd, 1, P + "chan_mlp.gate_up.weight", this.h1, this.g);
      sg(CELLS, hd);
      mm(CELLS, d, hd, hd, 1, hd, 1, d, 1, P + "chan_mlp.down.weight", this.u, this.tt);
      nm(P + "norm2.weight", this.h1, out);
    }
    return ops;
  }

  encode(enc, ops) {
    for (const o of ops) {
      const pass = enc.beginComputePass();
      pass.setPipeline(o.pipeline); pass.setBindGroup(0, o.bind); pass.dispatchWorkgroups(...o.groups); pass.end();
    }
  }

  vec(name) { const t = this.t[name]; return this.w.subarray(t.offset, t.offset + t.shape.reduce((a, b) => a * b, 1)); }

  async solve(puzzle, steps) {
    const d = this.cfg.d, dev = this.dev, n = CELLS * d, bytes = n * 4;
    const emb = this.vec("embed.weight"), x = new Float32Array(n), y0 = new Float32Array(n), z0 = new Float32Array(n);
    const yi = this.vec("y_init"), zi = this.vec("z_init"), sc = Math.sqrt(d);
    for (let c = 0; c < CELLS; c++) for (let k = 0; k < d; k++) {
      x[c * d + k] = emb[puzzle[c] * d + k] * sc; y0[c * d + k] = yi[k]; z0[c * d + k] = zi[k];
    }
    dev.queue.writeBuffer(this.x, 0, x); dev.queue.writeBuffer(this.y, 0, y0); dev.queue.writeBuffer(this.z, 0, z0);
    const head = this.vec("head.weight"), qw = this.vec("q_head.weight"), qb = this.vec("q_head.bias")[0];
    const out = [];
    for (let k = 0; k < steps; k++) {
      const passes = [];
      for (let t = 0; t < this.cfg.T; t++) {
        const enc = dev.createCommandEncoder();
        for (let i = 0; i < this.cfg.n; i++) {
          this.encode(enc, [this.addZ, ...this.netOps]);
          enc.copyBufferToBuffer(this.o, 0, this.z, 0, bytes);
        }
        this.encode(enc, [this.addY, ...this.netOps]);
        enc.copyBufferToBuffer(this.o, 0, this.y, 0, bytes);
        enc.copyBufferToBuffer(this.y, 0, this.stage, 0, bytes);
        dev.queue.submit([enc.finish()]);
        await this.stage.mapAsync(GPUMapMode.READ);
        const y = new Float32Array(this.stage.getMappedRange().slice(0));
        this.stage.unmap();
        const grid = new Array(CELLS), mean = new Float32Array(d);
        for (let c = 0; c < CELLS; c++) {
          let best = 0, bv = -Infinity;
          for (let j = 0; j < CLASSES; j++) {
            let s = 0; for (let e = 0; e < d; e++) s += y[c * d + e] * head[j * d + e];
            if (s > bv) { bv = s; best = j; }
          }
          grid[c] = puzzle[c] > 0 ? puzzle[c] : best + 1;
          for (let e = 0; e < d; e++) mean[e] += y[c * d + e] / CELLS;
        }
        let q = qb; for (let e = 0; e < d; e++) q += mean[e] * qw[e];
        passes.push({ grid, q });
      }
      out.push(passes);
    }
    return out;
  }
}
