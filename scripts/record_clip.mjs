// Record the visualizer replaying a real trace to an mp4 (headless Chromium video, then ffmpeg).
//   node scripts/record_clip.mjs [--assets assets] [--trace 0] [--seconds 20] [--out clip/trm-solve.mp4] [--shot file.png]
import { chromium } from "playwright-core";
import { execFileSync } from "node:child_process";
import http from "node:http";
import fs from "node:fs";
import path from "node:path";

const arg = (k, d) => { const i = process.argv.indexOf("--" + k); return i > 0 ? process.argv[i + 1] : d; };
const repo = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const root = path.join(repo, "viz");
const assets = arg("assets", "assets"), trace = arg("trace", "0"), seconds = +arg("seconds", "20");
const out = path.resolve(repo, arg("out", "clip/trm-solve.mp4")), shot = arg("shot", null), speed = arg("speed", null);
const types = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json" };
const server = http.createServer((req, res) => {
  let p = path.join(root, decodeURIComponent(new URL(req.url, "http://x").pathname));
  if (p.endsWith("/")) p += "index.html";
  if (!p.startsWith(root) || !fs.existsSync(p)) { res.writeHead(404); return res.end(); }
  res.writeHead(200, { "content-type": types[path.extname(p)] || "application/octet-stream" });
  res.end(fs.readFileSync(p));
}).listen(0);
const raw = path.join(repo, "clip/raw");
fs.mkdirSync(raw, { recursive: true });
const size = { width: 1280, height: 800 };
const browser = await chromium.launch({ headless: true });
const ctx = await browser.newContext({ viewport: size, deviceScaleFactor: 1, recordVideo: shot ? undefined : { dir: raw, size } });
const page = await ctx.newPage();
await page.goto(`http://127.0.0.1:${server.address().port}/?assets=${assets}&trace=${trace}&paused`);
await page.waitForFunction(() => window.__viz?.ready);
if (speed) await page.evaluate((s) => { document.getElementById("speed").value = s; }, speed);
const t0 = Date.now();
await page.waitForTimeout(700);
if (shot) {
  await page.evaluate((f) => { const st = window.__viz.state; st.i = Math.floor(st.frames.length * f); }, +arg("at", "0.6"));
  await page.waitForTimeout(400);
  await page.screenshot({ path: shot });
} else {
  await page.click("#play");
  await page.waitForTimeout(seconds * 1000);
}
const video = page.video();
await ctx.close();
await browser.close();
server.close();
if (video) {
  const webm = await video.path();
  const skip = ((Date.now() - t0) / 1000 - seconds - 0.7 > 0 ? 0 : 0) + 0.6;
  fs.mkdirSync(path.dirname(out), { recursive: true });
  execFileSync("ffmpeg", ["-y", "-loglevel", "error", "-ss", String(skip), "-i", webm, "-t", String(seconds),
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart", out]);
  console.log("wrote", out);
}
