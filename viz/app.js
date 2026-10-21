// Replays a real solve trace exported from the MLX checkpoint (scripts/export.py), and can recompute the same
// solve live in the browser with WebGPU (webgpu.js) to show the weights really do this.
const params = new URLSearchParams(location.search);
const ASSETS = (params.get("assets") || "assets").replace(/\/$/, "");
const $ = (id) => document.getElementById(id);
const state = { traces: [], info: null, frames: [], i: 0, playing: !params.has("paused"), ti: 0, timer: 0, acc: 0, last: 0 };
window.__viz = { ready: false, state };

const cells = [];
function buildBoard() {
  const b = $("board");
  for (let i = 0; i < 81; i++) {
    const c = document.createElement("div");
    const r = Math.floor(i / 9), col = i % 9;
    c.className = "cell" + (col % 3 === 0 && col ? " c" + col : "") + (r % 3 === 0 && r ? " r" + r : "");
    const s = document.createElement("span");
    c.appendChild(s);
    b.appendChild(c);
    cells.push({ el: c, span: s, base: c.className });
  }
}

function framesFor(tr, cfg) {
  const f = [{ kind: "start", step: -1, t: -1, z: -1, grid: tr.puzzle, conf: null, think: null, q: null }];
  let grid = tr.puzzle, conf = null, q = null;
  const halt = tr.halted_at ?? tr.steps.length - 1;
  for (const st of tr.steps.slice(0, halt + 1)) {
    st.passes.forEach((p, t) => {
      p.think.forEach((th, z) => f.push({ kind: "z", step: st.step, t, z, grid, conf, think: th, q }));
      grid = p.grid; conf = p.conf; q = p.q;
      f.push({ kind: "y", step: st.step, t, z: cfg.n, grid, conf, think: null, q, right: p.correct_cells });
    });
  }
  f.push({ ...f[f.length - 1], kind: "end" });
  return f;
}

function render() {
  const tr = state.traces[state.ti], fr = state.frames[state.i], cfg = state.info.config;
  let mx = 0;
  if (fr.think) for (const v of fr.think) mx = Math.max(mx, v);
  for (let i = 0; i < 81; i++) {
    const c = cells[i], given = tr.puzzle[i] > 0, v = fr.grid[i];
    let cls = c.base;
    if (given) cls += " given";
    else if (fr.conf) cls += v === tr.solution[i] ? " ok" : " bad";
    c.el.className = cls;
    c.span.textContent = given || fr.conf ? v : "";
    c.span.style.opacity = given || !fr.conf ? 1 : (0.35 + 0.65 * fr.conf[i]).toFixed(2);
    const t = fr.think && !given ? Math.sqrt(fr.think[i] / (mx || 1)) * 0.5 : 0;
    c.el.style.setProperty("--t", t.toFixed(3));
  }
  const steps = (tr.halted_at ?? tr.steps.length - 1) + 1;
  $("r-step").textContent = fr.step < 0 ? "-" : `${fr.step + 1} / ${cfg.n_sup}`;
  $("r-rec").textContent = fr.t < 0 ? "-" : `${fr.t + 1} / ${cfg.T}`;
  $("r-z").textContent = fr.kind === "z" ? `${fr.z + 1} / ${cfg.n}` : fr.kind === "y" ? "answer update" : "-";
  const blanks = tr.puzzle.filter((v) => v === 0).length;
  const right = fr.conf ? fr.grid.filter((v, i) => tr.puzzle[i] === 0 && v === tr.solution[i]).length : 0;
  $("r-right").textContent = fr.conf ? `${right} / ${blanks}` : `0 / ${blanks}`;
  $("phase").textContent = fr.kind === "start" ? "Puzzle loaded" : fr.kind === "z" ? "Thinking: updating latent z"
    : fr.kind === "y" ? "Updating the answer y" : tr.solved_at_halt ? "Halted: solved" : "Halted: not solved";
  const q = fr.q;
  $("r-q").textContent = q == null ? "-" : q.toFixed(2);
  const fill = $("q-fill"), w = q == null ? 0 : Math.min(50, Math.abs(q) * 5);
  fill.style.width = w + "%";
  fill.style.left = q != null && q < 0 ? 50 - w + "%" : "50%";
  fill.classList.toggle("pos", q != null && q > 0);
  const per = cfg.T * (cfg.n + 1);
  [...$("steps").children].forEach((b, k) => {
    const done = fr.step > k || fr.kind === "end" && k <= fr.step ? 1 : fr.step === k ? ((fr.t * (cfg.n + 1) + (fr.kind === "y" ? cfg.n + 1 : fr.z + 1)) / per) : 0;
    b.querySelector(".fill").style.transform = `scaleX(${Math.min(1, done)})`;
    b.classList.toggle("halt", k === tr.halted_at);
    b.style.opacity = k < steps ? 1 : 0.35;
  });
}

function load(ti) {
  state.ti = ti;
  state.frames = framesFor(state.traces[ti], state.info.config);
  state.i = 0;
  $("live-status").textContent = "";
  render();
}

function tick(now) {
  if (state.playing) {
    const fps = +$("speed").value;
    state.acc += Math.min(200, now - (state.last || now));
    while (state.acc > 1000 / fps) {
      state.acc -= 1000 / fps;
      if (state.i < state.frames.length - 1) state.i++;
      else { state.playing = params.has("loop"); if (state.playing) state.i = 0; $("play").textContent = state.playing ? "Pause" : "Play"; }
    }
  }
  if (state.i !== state.drawn) { state.drawn = state.i; render(); }
  state.last = now;
  requestAnimationFrame(tick);
}

async function main() {
  buildBoard();
  const data = await (await fetch(`${ASSETS}/traces.json`)).json();
  state.traces = data.traces; state.info = data.info;
  const inf = data.info, ev = inf.last_eval;
  $("params").textContent = (inf.params / 1e6).toFixed(1) + "M-parameter";
  $("badge").textContent = `Partially trained: ${inf.trained_iters.toLocaleString()} steps, ${Math.round(inf.trained_minutes)} min on an M4 Pro. ` +
    (inf.eval_full ? `Solves ${(inf.eval_full.test_solved_valid_sudoku * 100).toFixed(1)}% of ${inf.eval_full.test_puzzles.toLocaleString()} held-out puzzles.`
      : `Held-out exact solves: ${(ev.exact_at_halt * 100).toFixed(1)}% of ${ev.n}.`);
  $("foot").innerHTML = `Every frame is a real forward pass of the checkpoint on a test puzzle it never trained on. ` +
    `Per supervision step the model runs ${inf.config.T} recursions; each does <b>${inf.config.n} latent passes</b> (z, the glow shows how much each cell's latent changed) ` +
    `then <b>one answer update</b> (y, the digits). Digit brightness is the model's confidence. ` +
    `Blank-cell accuracy on ${ev.n} held-out puzzles: <b>${(ev.blank_cell_acc_at_halt * 100).toFixed(1)}%</b>. ` +
    `Puzzles are from my own hard-puzzle generator, not the paper's Sudoku-Extreme set.`;
  const steps = $("steps");
  for (let k = 0; k < inf.config.n_sup; k++) {
    const b = document.createElement("button");
    b.innerHTML = `<div class="fill"></div><div class="n">${k + 1}</div>`;
    b.onclick = () => { const j = state.frames.findIndex((f) => f.step === k); if (j >= 0) { state.i = j; render(); } };
    steps.appendChild(b);
  }
  state.traces.forEach((t, k) => {
    const o = document.createElement("option");
    o.value = k; o.textContent = `#${t.test_index} (${t.solved_at_halt ? "solved" : "not solved"})`;
    $("pick").appendChild(o);
  });
  const start = Math.min(+(params.get("trace") || 0), state.traces.length - 1);
  $("pick").value = start;
  $("pick").onchange = (e) => load(+e.target.value);
  $("play").onclick = () => { state.playing = !state.playing; if (state.playing && state.i >= state.frames.length - 1) state.i = 0; $("play").textContent = state.playing ? "Pause" : "Play"; };
  $("restart").onclick = () => { state.i = 0; render(); };
  $("play").textContent = state.playing ? "Pause" : "Play";
  $("live").onclick = runLive;
  load(start);
  window.__viz.ready = true;
  requestAnimationFrame(tick);
}

async function runLive() {
  const status = $("live-status");
  try {
    status.textContent = "loading weights...";
    const { LiveTRM } = await import("./webgpu.js");
    const live = await LiveTRM.create(ASSETS);
    const tr = state.traces[state.ti];
    status.textContent = "running on your GPU...";
    const t0 = performance.now();
    const res = await live.solve(tr.puzzle, (tr.halted_at ?? tr.steps.length - 1) + 1);
    const ms = performance.now() - t0;
    let gridMatch = 0, total = 0, qDiff = 0;
    res.forEach((st, k) => st.forEach((p, t) => {
      const ref = tr.steps[k].passes[t];
      total++; if (p.grid.every((v, i) => v === ref.grid[i])) gridMatch++;
      qDiff = Math.max(qDiff, Math.abs(p.q - ref.q));
    }));
    window.__viz.live = { gridMatch, total, qDiff, ms, adapter: live.adapter };
    status.textContent = `WebGPU recomputed ${total} answer updates in ${(ms / 1000).toFixed(1)} s: ${gridMatch}/${total} grids identical to the MLX trace, max |q diff| ${qDiff.toFixed(4)} (${live.adapter})`;
  } catch (e) {
    window.__viz.live = { error: String(e) };
    status.textContent = "Live mode unavailable here: " + e.message + " (the replay above is still the real trace)";
  }
}

main().catch((e) => { $("badge").textContent = "Could not load trace: " + e.message; window.__viz.error = String(e); });
