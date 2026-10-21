// Visualizer smoke test: serve viz/, load a trace, check the board renders and replays, then run the
// live WebGPU solve and check it agrees with the MLX trace.
//   node tests/viz_smoke.mjs [assets-dir-relative-to-viz]   (default: assets)
import { chromium } from "playwright-core";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), "../viz");
const assets = process.argv[2] || "assets";
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".bin": "application/octet-stream" };
const server = http.createServer((req, res) => {
  const p = path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (!p.startsWith(root) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) {
    const idx = path.join(p, "index.html");
    if (fs.existsSync(idx)) { res.writeHead(200, { "content-type": "text/html" }); return res.end(fs.readFileSync(idx)); }
    res.writeHead(404); return res.end();
  }
  res.writeHead(200, { "content-type": types[path.extname(p)] || "application/octet-stream" });
  res.end(fs.readFileSync(p));
}).listen(0);
const port = server.address().port;
const fail = (m) => { console.error("FAIL", m); process.exitCode = 1; };

const browser = await chromium.launch({ headless: true, args: ["--enable-unsafe-webgpu", "--enable-features=Vulkan,WebGPU"] });
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 900 } });
  const errors = [];
  page.on("pageerror", (e) => errors.push(String(e)));
  await page.goto(`http://127.0.0.1:${port}/?assets=${assets}&paused`);
  await page.waitForFunction(() => window.__viz && (window.__viz.ready || window.__viz.error), null, { timeout: 20000 });
  const info = await page.evaluate(() => ({ err: window.__viz.error, frames: window.__viz.state.frames.length,
    cells: document.querySelectorAll(".cell").length, given: document.querySelectorAll(".cell.given").length,
    badge: document.getElementById("badge").textContent }));
  if (info.err) fail("load: " + info.err);
  if (info.cells !== 81) fail("expected 81 cells, got " + info.cells);
  if (info.given < 17) fail("expected the puzzle's givens on the board, got " + info.given);
  if (!/Partially trained/.test(info.badge)) fail("badge missing partial-training label: " + info.badge);
  await page.click("#play");
  await page.waitForTimeout(1500);
  const moved = await page.evaluate(() => window.__viz.state.i);
  if (!(moved > 3)) fail("replay did not advance, frame " + moved);
  const t0 = Date.now();
  await page.click("#live");
  await page.waitForFunction(() => window.__viz.live, null, { timeout: 240000 });
  const live = await page.evaluate(() => window.__viz.live);
  if (live.error) console.log("live WebGPU unavailable in this browser:", live.error);
  else {
    console.log(`live WebGPU: ${live.gridMatch}/${live.total} grids identical to the MLX trace, max |q diff| ${live.qDiff.toFixed(5)}, ${(live.ms / 1000).toFixed(1)}s`);
    if (live.gridMatch < live.total * 0.9) fail("WebGPU disagrees with the MLX trace");
  }
  if (errors.length) fail("page errors: " + errors.join("; "));
  console.log(`frames ${info.frames}, givens ${info.given}, replay advanced to ${moved}, live took ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  console.log(process.exitCode ? "viz smoke: FAIL" : "viz smoke: PASS");
} finally {
  await browser.close();
  server.close();
}
