// PRAESENS HUD -- browser side.
// 1. Emitter: paints the challenge on the four edge strips every animation frame and reports
//    exactly what it painted, stamped on the engine's clock (min-RTT offset), so the optical lane
//    scores against the real display -- the same contract as the OpenCV emitter's redraw log.
// 2. Telemetry: draws the live lanes (LED-style canvases) and the face hologram (Three.js).
import * as THREE from "/static/vendor/three.module.min.js";

const $ = (id) => document.getElementById(id);
const C = { holo: "#7ec4d2", plasma: "#c489ae", phos: "#83c79d", amber: "#d8a96e", alarm: "#cf7680",
            dim: "#767ca0", ghost: "#282d55", ink: "#d9dfee" };
const reduceMotion = matchMedia("(prefers-reduced-motion: reduce)").matches;
// one font for canvas text, matching the stylesheet: a plain humanist sans, no LED/techno face
const FONT = '"Segoe UI", Inter, system-ui, sans-serif';

// ------------------------------------------------------------------ websocket + clock
let ws = null, hello = null, T = null, lastResult = null;
const sync = { samples: [], offset: null, rtt: null };   // engine_perf_s = performance.now()/1000 + offset

function send(o) { if (ws && ws.readyState === 1) ws.send(JSON.stringify(o)); }
function engineNow() { return sync.offset === null ? null : performance.now() / 1000 + sync.offset; }

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.binaryType = "blob";
  ws.onopen = () => { send({ type: "role", role: "emitter" }); $("liveChip").textContent = "linked"; burstPings(); };
  ws.onclose = () => { $("liveChip").textContent = "offline"; $("scanBtn").disabled = true; setTimeout(connect, 1000); };
  ws.onmessage = (ev) => {
    if (typeof ev.data !== "string") { onFrame(ev.data); return; }
    const m = JSON.parse(ev.data);
    if (m.type === "pong") onPong(m);
    else if (m.type === "hello") onHello(m);
    else if (m.type === "challenge") onChallenge(m);
    else if (m.type === "telemetry") onTelemetry(m);
    else if (m.type === "result") onResult(m.result);
    else if (m.type === "scan") onScan(m);
    else if (m.type === "toast") toast(m.text);
  };
}
function ping() { send({ type: "ping", c: performance.now() }); }
function burstPings() { for (let i = 0; i < 20; i++) setTimeout(ping, i * 60); }
setInterval(ping, 1500);
function onPong(m) {
  const c1 = performance.now(), rtt = c1 - m.c;
  sync.samples.push({ rtt, off: m.s - (m.c + c1) / 2000 });
  if (sync.samples.length > 40) sync.samples.shift();
  const best = sync.samples.reduce((a, b) => (b.rtt < a.rtt ? b : a));
  sync.offset = best.off; sync.rtt = best.rtt;
  $("syncChip").textContent = `clock Â±${(best.rtt / 2).toFixed(1)} ms`;
}

function onHello(m) {
  hello = m;
  const bf = m.border_fraction ?? 0.13;
  document.documentElement.style.setProperty("--bf-x", `${bf * 100}vw`);
  document.documentElement.style.setProperty("--bf-y", `${bf * 100}vh`);
  $("synthChip").hidden = !m.synthetic;
  $("sim").classList.toggle("on", !!m.synthetic);
  $("camSel").style.display = m.synthetic ? "none" : "flex";
  $("lanePressure").style.display = m.lanes.includes("pressure") ? "" : "none";
}

// ------------------------------------------------------------------ emitter
const strips = [...document.querySelectorAll(".emit")];
let current = null, pending = null, emitBuf = [];

function onChallenge(m) {
  const ch = { ...m, n: m.chips.length };
  if (current && current.id === m.id) { current.depth = m.depth; return; }   // depth boost
  if (pending && pending.id === m.id) { pending.depth = m.depth; return; }
  if (!current) current = ch; else pending = ch;
}
function chipAt(ch, s) {
  const el = s - ch.start_perf;
  let i = Math.floor(el * ch.chip_rate_hz);
  if (ch.loop) i = ((i % ch.n) + ch.n) % ch.n;
  else i = Math.min(Math.max(i, 0), ch.n - 1);
  return ch.chips[i];
}
function emitFrame() {
  const s = engineNow();
  if (s === null || !current) return;
  if (pending && s >= pending.start_perf) { flushEmit(); current = pending; pending = null; }
  const chip = chipAt(current, s);
  const lum = Math.round(Math.min(255, Math.max(0, current.base + current.depth * chip)));
  const col = `rgb(${lum},${lum},${lum})`;
  for (const el of strips) el.style.backgroundColor = col;
  // stamp: this rAF callback's time, mapped to the engine clock (display latency lands in the lag search)
  emitBuf.push([+s.toFixed(5), chip, lum]);
}
function flushEmit() {
  if (emitBuf.length && current) send({ type: "emitlog", id: current.id, entries: emitBuf });
  emitBuf = [];
}
setInterval(flushEmit, 200);

// ------------------------------------------------------------------ camera preview
let camURL = null, camAspect = 4 / 3;
function onFrame(blob) {
  const url = URL.createObjectURL(blob);
  const img = $("camImg");
  img.onload = () => { if (img.naturalWidth) camAspect = img.naturalWidth / img.naturalHeight; };
  img.src = url;
  if (camURL) URL.revokeObjectURL(camURL);
  camURL = url;
}

// ------------------------------------------------------------------ LED canvas kit
function fit(cv) {
  const r = cv.getBoundingClientRect(), d = devicePixelRatio || 1;
  const w = Math.max(1, Math.round(r.width * d)), h = Math.max(1, Math.round(r.height * d));
  if (cv.width !== w || cv.height !== h) { cv.width = w; cv.height = h; }
  const ctx = cv.getContext("2d");
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return { ctx, w, h, d };
}
function dotGrid(ctx, w, h, step, color = "rgba(92,242,255,.07)") {
  ctx.fillStyle = color;
  for (let y = step / 2; y < h; y += step) for (let x = step / 2; x < w; x += step) ctx.fillRect(x, y, 1.2, 1.2);
}
function led(ctx, x, y, r, color, a = 1) {
  ctx.globalAlpha = a * 0.25; ctx.fillStyle = color;
  ctx.beginPath(); ctx.arc(x, y, r * 2.4, 0, 7); ctx.fill();
  ctx.globalAlpha = a; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill();
  ctx.globalAlpha = 1;
}
function ledTrace(ctx, data, w, h, color, { pad = 6, range = 3, r = 1.6, x0 = 0, x1 = 1 } = {}) {
  const n = data.length; if (n < 2) return;
  for (let i = 0; i < n; i++) {
    const v = data[i]; if (v === null || v === undefined) continue;
    const x = (x0 + (x1 - x0) * i / (n - 1)) * (w - 2 * pad) + pad;
    const y = h / 2 - Math.max(-1, Math.min(1, v / range)) * (h / 2 - pad);
    led(ctx, x, y, r, color, 0.9);
  }
}
function ledMeter(cv, value, threshold, colorFor) {
  const { ctx, w, h } = fit(cv);
  const n = 24, gap = 2 * (devicePixelRatio || 1), sw = (w - gap * (n - 1)) / n;
  for (let i = 0; i < n; i++) {
    const lit = value !== null && value !== undefined && (i + 0.5) / n <= value;
    const col = colorFor((i + 0.5) / n);
    ctx.fillStyle = lit ? col : "rgba(124,131,184,.14)";
    if (lit) { ctx.shadowColor = col; ctx.shadowBlur = 8; } else ctx.shadowBlur = 0;
    ctx.fillRect(i * (sw + gap), 0, sw, h);
  }
  ctx.shadowBlur = 0;
  if (threshold !== null && threshold !== undefined) {
    const x = threshold * w; ctx.fillStyle = C.ink; ctx.fillRect(x - 1, 0, 2, h);
  }
}
const zone = (thr) => (v) => (v >= thr ? C.phos : v >= thr * 0.6 ? C.amber : C.alarm);
const fmt = (v, d = 2, blank = "-.--") => (v === null || v === undefined ? blank : v.toFixed(d));
function setState(id, text, cls) { const el = $(id); el.textContent = text; el.className = `state ${cls}`; }

// ------------------------------------------------------------------ lanes
function drawOptical(o) {
  const { ctx, w, h, d } = fit($("optScope"));
  if (o.paused) {
    // live correlation is paused while the acoustic probe records (it made the audio drop
    // buffers); the scored optical result is computed when the scan ends
    setState("optState", "measuring", "warn");
    ctx.fillStyle = C.dim; ctx.font = `${11 * d}px ${FONT}`;
    ctx.fillText("light pattern being recorded", 8 * d, h / 2 - 6 * d);
    ctx.fillText("scored when the scan ends", 8 * d, h / 2 + 10 * d);
    $("optScore").textContent = "···";
    return;
  }
  dotGrid(ctx, w, h, 9 * d);
  ctx.strokeStyle = "rgba(92,242,255,.18)"; ctx.beginPath(); ctx.moveTo(0, h / 2); ctx.lineTo(w, h / 2); ctx.stroke();
  ledTrace(ctx, o.emitted || [], w, h, C.plasma, { r: 1.3 * d });
  ledTrace(ctx, o.measured || [], w, h, C.holo, { r: 1.7 * d });
  ctx.font = `${11 * d}px ${FONT}`; ctx.fillStyle = C.plasma; ctx.fillText("screen pattern", 6 * d, 13 * d);
  ctx.fillStyle = C.holo; ctx.fillText("light off your face", 6 * d, 26 * d);
  $("optScore").textContent = fmt(o.score);
  $("optLag").textContent = o.lag_ms === null ? "---" : String(Math.round(o.lag_ms));
  $("optSnr").textContent = fmt(o.snr_db, 1, "--.-");
  ledMeter($("optMeter"), o.score === null ? null : Math.max(0, o.score), o.pass, zone(o.pass ?? 0.3));
  if (o.score === null) setState("optState", o.face_frac < 0.3 ? "no face" : "acquiring", "idle");
  else if (o.score >= o.pass && (o.snr_db ?? 0) >= 3) setState("optState", "coherent", "ok");
  else setState("optState", "incoherent", "bad");
}

function drawTyping(t, p) {
  const { ctx, w, h, d } = fit($("typRhythm"));
  dotGrid(ctx, w, h, 9 * d);
  const span = 6;
  const keys = (p && p.recent_keys) || [];
  // each key-down is a lit LED column; older keys fade out as they scroll left
  for (const dt of keys) {
    const x = (1 + dt / span) * (w - 8 * d) + 4 * d, age = Math.min(1, -dt / span);
    const rows = 7;
    for (let k = 0; k < rows; k++) led(ctx, x, h - 10 * d - k * ((h - 34 * d) / rows), 1.5 * d, C.phos, (1 - age) * (1 - k * 0.1));
  }
  ctx.font = `${11 * d}px ${FONT}`; ctx.fillStyle = C.dim;
  ctx.fillText("key-down rhythm, last 6 s", 6 * d, 13 * d);
  const scan = t.scan;
  if (!(scan && t.mode === "scan") && !keys.length) {
    ctx.font = `${13 * d}px ${FONT}`; ctx.fillStyle = C.dim; ctx.textAlign = "center";
    ctx.fillText("A scan shows a fresh phrase to type.", w / 2, h / 2);
    ctx.fillText("Only key timing is kept, never the characters.", w / 2, h / 2 + 18 * d);
    ctx.textAlign = "left";
  }
  if (scan && t.mode === "scan") {
    $("typCount").textContent = String(scan.n_keys).padStart(2, "0");
    const frac = scan.matched / Math.max(1, scan.phrase.length);
    setState("typState", `${Math.round(frac * 100)}% typed`, frac >= 0.7 ? "ok" : "warn");
  } else {
    $("typCount").textContent = "--";
    setState("typState", keys.length ? "listening" : "idle", "idle");
  }
}

// ------------------------------------------------------------------ acoustic lane
// The probe is a 20 s duplex tone, so this lane has nothing to show between scans --
// it reports the state honestly instead of animating something that isn't measured.
function drawAcoustic(a, mode) {
  const { ctx, w, h, d } = fit($("acoScope"));
  if (!a || !a.in_lanes) {
    setState("acoState", "not in this run", "idle");
    ctx.fillStyle = C.dim; ctx.font = `${11 * d}px ${FONT}`;
    ctx.fillText("add +acoustic to --lanes", 8 * d, h / 2);
    $("acoScore").textContent = "-.--"; $("acoLag").textContent = "---";
    return;
  }
  if (a.state === "probing") {
    setState("acoState", "probe playing", "warn");
    $("acoWhy").textContent = `${Math.round(a.carrier_hz)} Hz through the speaker`;
    const t = performance.now() / 1000;
    ctx.strokeStyle = C.plasma; ctx.lineWidth = 1.4 * d; ctx.beginPath();
    for (let x = 0; x < w; x++) {                 // a tone, drawn as a tone
      const y = h / 2 + Math.sin(x * 0.22 + t * 6) * h * 0.3 * Math.exp(-Math.abs(x - w / 2) / (w * 0.9));
      x ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    $("acoScore").textContent = "···"; $("acoLag").textContent = "---";
    return;
  }
  const last = a.last;
  if (!last) {
    setState("acoState", "waiting", "idle");
    ctx.fillStyle = C.dim; ctx.font = `${11 * d}px ${FONT}`;
    ctx.fillText("runs during a scan", 8 * d, h / 2);
    $("acoScore").textContent = "-.--"; $("acoLag").textContent = "---";
    return;
  }
  const ok = last.status === "ok";
  setState("acoState", ok ? (last.passes ? "coherent" : "weak") : last.status.replace("_", " "),
           ok ? (last.passes ? "ok" : "bad") : "warn");
  $("acoScore").textContent = ok && last.score !== null ? last.score.toFixed(2) : "-.--";
  $("acoLag").textContent = last.lag_ms === null || last.lag_ms === undefined ? "---" : String(Math.round(last.lag_ms));
  $("acoWhy").textContent = (last.diagnostics || "").split(",")[0] || "last scan";
  // one bar: how far the measured score sits above the pass bar
  const frac = Math.max(0, Math.min(1, (last.score ?? 0)));
  ctx.fillStyle = "rgba(126,196,210,.10)"; ctx.fillRect(8 * d, h / 2 - 7 * d, w - 16 * d, 14 * d);
  ctx.fillStyle = last.passes ? C.phos : C.alarm;
  ctx.fillRect(8 * d, h / 2 - 7 * d, (w - 16 * d) * frac, 14 * d);
}

function drawPressure(p, now) {
  const { ctx, w, h, d } = fit($("preEnv"));
  dotGrid(ctx, w, h, 9 * d);
  const env = p.env || [];
  const n = env.length, cw = w / Math.max(1, n);
  const top = h * 0.74;
  for (let i = 0; i < n; i++) {
    const v = Math.max(0, Math.min(1, (env[i] || 0) / 40));
    const segs = Math.round(v * 12);
    for (let k = 0; k < segs; k++) {
      ctx.fillStyle = k > 8 ? C.plasma : C.holo;
      ctx.globalAlpha = 0.45 + 0.55 * (k / 12);
      ctx.fillRect(i * cw, top - (k + 1) * (top / 13), Math.max(1, cw - 1), top / 13 - 1.5 * d);
    }
  }
  ctx.globalAlpha = 1;
  // key-downs reported by the OS (plasma ticks) -- their impacts should follow at a fixed lag
  for (const dt of p.recent_keys || []) {
    if (dt < -3) continue;
    const x = (1 + dt / 3) * w;
    ctx.fillStyle = C.plasma; ctx.beginPath(); ctx.moveTo(x - 4 * d, 0); ctx.lineTo(x + 4 * d, 0); ctx.lineTo(x, 7 * d); ctx.fill();
  }
  // per-key verdict row: did this keystroke have a physical strike behind it?
  const keys = p.keys || [];
  // one pip per recent keystroke: green = a strike was heard for it, red = none (height = strike force)
  const base = h - 2 * d, pw = Math.min(9 * d, (w - 4 * d) / 26);
  keys.forEach((k, i) => {
    const col = k.matched ? C.phos : C.alarm;
    const f = k.force_db === null ? 0.2 : Math.max(0.2, Math.min(1, k.force_db / 40));
    const ph = f * (h - top - 6 * d);
    ctx.fillStyle = col; ctx.shadowColor = col; ctx.shadowBlur = 6;
    ctx.fillRect(2 * d + i * (pw + 2 * d), base - ph, pw, ph);
  });
  ctx.shadowBlur = 0;
  ctx.font = `${11 * d}px ${FONT}`; ctx.fillStyle = C.dim;
  ctx.fillText(p.status === "scan" ? "this scan: one pip per key" : "mic, last 3 s  â–¼ key-down", 6 * d, 20 * d);
  $("preScore").textContent = fmt(p.score);
  $("preLag").textContent = p.lag_ms === null || p.lag_ms === undefined ? "---" : String(Math.round(p.lag_ms));
  $("preJit").textContent = fmt(p.jitter_ms, 1, "--.-");
  if (p.status === "ok") {
    const ok = p.score >= 0.5;
    setState("preState", ok ? "strikes match" : "no strikes", ok ? "ok" : "bad");
    $("preWhy").textContent = `${Math.round((p.hit || 0) * 100)}% of keys heard Â· chance ${Math.round((p.chance || 0) * 100)}%`;
  } else if (p.status === "shared") {
    setState("preState", "recording", "warn");
    $("preWhy").textContent = "keys shown live · strikes scored when the scan ends";
  } else if (p.status === "scan") {
    const ok = p.score !== null && p.score >= 0.5;
    setState("preState", p.score === null ? "unreliable" : ok ? "strikes match" : "no strikes",
             p.score === null ? "warn" : ok ? "ok" : "bad");
    $("preWhy").textContent = p.score === null ? (p.reason || "").split(";")[0]
      : `this scan: ${Math.round((p.hit || 0) * 100)}% of ${p.n_keys} keys heard · chance ${Math.round((p.chance || 0) * 100)}%`;
  } else if (p.status === "stale") {
    setState("preState", "last burst", "idle");
    $("preWhy").textContent = `${Math.round((p.hit || 0) * 100)}% of keys heard in the last burst`;
  } else {
    setState("preState", "waiting", "idle");
    $("preWhy").textContent = p.reason || "type anything to see strikes";
  }
}

function drawFuse(t) {
  const items = [];
  const o = t.optical;
  items.push(["Optical", o.score === null ? "" : o.score >= o.pass ? "ok" : "bad"]);
  if (t.lanes.includes("typing")) {
    const s = t.scan; let st = "";
    if (s && t.mode === "scan") st = s.matched / Math.max(1, s.phrase.length) >= 0.7 ? "ok" : "warn";
    items.push(["Typing", st]);
  }
  if (t.lanes.includes("acoustic")) {
    const a = t.acoustic || {};
    items.push(["Acoustic", a.state === "probing" ? "warn" : a.last ? (a.last.passes ? "ok" : "bad") : ""]);
  }
  if (t.lanes.includes("pressure")) {
    const p = t.pressure;
    items.push(["Key impact", p.status === "ok" ? (p.score >= 0.5 ? "ok" : "bad") : p.status === "shared" ? "warn" : ""]);
  }
  $("fuse").innerHTML = items.map(([n, s]) => `<span class="led ${s}"><i></i>${n}</span>`).join("");
}

// ------------------------------------------------------------------ scan ring + prompt
let scanInfo = null;
function onScan(m) { scanInfo = m; $("verdict").classList.remove("on"); }
function drawRing(t) {
  const { ctx, w, h, d } = fit($("ring"));
  const cx = w / 2, cy = h * 0.55, r = Math.min(w, h) * 0.42;
  const s = t.scan, inScan = s && t.mode === "scan";
  ctx.lineWidth = 2 * d;
  // reticle corners
  ctx.strokeStyle = "rgba(92,242,255,.45)";
  const q = r * 0.95, k = r * 0.16;
  for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
    ctx.beginPath(); ctx.moveTo(cx + sx * q, cy + sy * (q - k)); ctx.lineTo(cx + sx * q, cy + sy * q); ctx.lineTo(cx + sx * (q - k), cy + sy * q); ctx.stroke();
  }
  if (inScan) {
    const frac = Math.max(0, Math.min(1, s.elapsed / s.duration));
    const ticks = 90;
    for (let i = 0; i < ticks; i++) {
      const a = -Math.PI / 2 + (i / ticks) * Math.PI * 2;
      const on = i / ticks <= frac;
      ctx.strokeStyle = on ? C.plasma : "rgba(124,131,184,.25)";
      ctx.lineWidth = (on ? 3 : 1.5) * d;
      ctx.beginPath(); ctx.moveTo(cx + Math.cos(a) * r * 0.99, cy + Math.sin(a) * r * 0.99);
      ctx.lineTo(cx + Math.cos(a) * r * 1.05, cy + Math.sin(a) * r * 1.05); ctx.stroke();
    }
    ctx.font = `600 ${13 * d}px ${FONT}`; ctx.fillStyle = C.plasma; ctx.textAlign = "center";
    const left = Math.max(0, s.duration - s.elapsed);
    ctx.fillText(s.elapsed < 0 ? "Get ready" : `${left.toFixed(0)} s left`, cx, cy - r * 1.05 - 10 * d);
    ctx.textAlign = "left";
  }
}
function updatePrompt(t) {
  const s = t.scan, on = !!(s && t.mode === "scan");
  $("prompt").classList.toggle("on", on);
  if (!on) return;
  const p = s.phrase, m = Math.min(s.matched, p.length);
  const esc = (x) => x.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/ /g, "&nbsp;");
  $("phrase").innerHTML = `<span class="done">${esc(p.slice(0, m))}</span><span class="next">${esc(p.slice(m, m + 1))}</span>${esc(p.slice(m + 1))}`;
}

// ------------------------------------------------------------------ verdict
function onResult(r) {
  setMood(r.verdict);
  lastResult = r;
  const v = $("verdict");
  v.className = `verdict on ${r.verdict}`;
  $("vWord").textContent = r.verdict === "RE-CHALLENGE" ? "RE-CHALLENGE" : r.verdict;
  $("vReason").textContent = r.reason || "";
  const names = { optical: "Optical", typing: "Typing", acoustic: "Acoustic", pressure: "Key impact" };
  $("vLanes").innerHTML = (r.lanes || []).filter((l) => l.name !== "pulse").map((l) => {
    const verdict = l.status !== "ok" ? ["na", "No evidence"] : !l.gated ? ["na", "Advisory"] : l.passes ? ["ok", "Pass"] : ["bad", "Fail"];
    const lag = l.lag_ms === null || !l.passes ? "" : `${Math.round(l.lag_ms)}<small> ms</small>`;
    return `<tr><td>${names[l.name] || l.name}</td><td class="${verdict[0]}">${verdict[1]}</td>
      <td class="n">${l.subscore === null ? "-.--" : l.subscore.toFixed(2)}</td><td class="n">${lag}</td>
      <td class="diag">${(l.diagnostics || "").replace(/</g, "&lt;")}</td></tr>`;
  }).join("");
  $("vLog").textContent = `${r.synthetic ? "SYNTHETIC â€” " : ""}saved ${r.log || ""}`;
  const tr = r.optical_trace || {};
  const { ctx, w, h, d } = fit($("vTrace"));
  dotGrid(ctx, w, h, 9 * d);
  ledTrace(ctx, tr.emitted || [], w, h, C.plasma, { r: 1.1 * d });
  ledTrace(ctx, tr.measured || [], w, h, C.holo, { r: 1.4 * d });
  $("vClose").focus();
}
$("vClose").onclick = () => { $("verdict").classList.remove("on"); setMood("idle"); send({ type: "dismiss" }); };

// ------------------------------------------------------------------ telemetry
function onTelemetry(t) {
  try {
    onTelemetryInner(t);
  } catch (e) {
    console.error("telemetry render failed", e);
    toast(`UI error: ${e.message}`);
  }
}

function onTelemetryInner(t) {
  T = t;
  $("status").innerHTML = `<i>${t.mode}</i> â€” ${String(t.status).replace(/</g, "&lt;")}`;
  $("fpsChip").textContent = `${t.fps === null ? "â€”" : t.fps.toFixed(1)} fps`;
  $("scanBtn").disabled = t.mode !== "idle";
  $("scanBtn").textContent = t.mode === "idle" ? "Start scan" : t.mode === "scan" ? "Scanning" : t.mode === "result" ? "Done" : "Working";
  $("camLabel").textContent = t.synthetic ? "synthetic source" : `camera ${t.camera}`;
  $("faceTag").textContent = t.face ? "LOCKED" : "SEARCH";
  $("faceSub").textContent = t.face ? "478 landmarks tracked" : "no face in view";
  $("stageNote").textContent = t.mode === "scan" ? "Keep your face in the frame and type the phrase."
    : t.face ? "Live. The edge light is today's challenge; your skin answers it." : "Face the screen to begin.";
  drawOptical(t.optical);
  drawTyping(t, t.pressure);
  drawAcoustic(t.acoustic, t.mode);
  if (t.lanes.includes("pressure")) drawPressure(t.pressure, t.now);
  drawFuse(t);
  updatePrompt(t);
  $("sim").hidden = !t.synthetic;
  $("attacks").hidden = !!t.synthetic;
  if (t.sim) for (const b of document.querySelectorAll("[data-sim]")) b.setAttribute("aria-pressed", String(!!t.sim[b.dataset.sim]));
  if (t.cameras && t.cameras.length) for (const b of document.querySelectorAll("[data-role]")) {
    const c = t.cameras.find((x) => x.role === b.dataset.role);
    b.disabled = !c;
    b.classList.toggle("on", !!c && c.index === t.camera);
    if (c) b.title = `${c.name} (index ${c.index})`;
  }
  if (t.landmarks) setFaceTarget(t.landmarks);
  if (t.mode === "scan") setMood("scan");
  else if (t.mode !== "result") setMood("idle");   // "result" keeps the verdict colour on the face
  // the engine went back to live: close the verdict (unless it was replayed by hand via __hud)
  if (t.mode === "idle" && $("verdict").classList.contains("on") && !window.__hud.hold) $("verdict").classList.remove("on");
}

$("scanBtn").onclick = () => send({ type: "scan" });

// The edge strips are the optical emitter. Windowed, they are a thin frame around a dark page and
// light almost nothing (the 2026-10-24 night scans failed at 20-33% face coverage, all windowed).
// On ASUS laptops plain F11 is a media key, so give the page its own way in, and say when it's out.
function isFullscreen() {
  return !!document.fullscreenElement || (Math.abs(window.innerHeight - screen.height) < 4 &&
                                          Math.abs(window.innerWidth - screen.width) < 4);
}
function syncFullscreen() {
  const fs = isFullscreen();
  $("fsChip").hidden = fs;
  $("fsBtn").textContent = fs ? "Exit fullscreen" : "Fullscreen";
}
$("fsBtn").onclick = () => {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen().catch((e) => toast(`fullscreen refused: ${e.message}`));
};
document.addEventListener("fullscreenchange", syncFullscreen);
window.addEventListener("resize", syncFullscreen);
syncFullscreen();
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && $("verdict").classList.contains("on")) $("vClose").click();
});
for (const b of document.querySelectorAll("[data-sim]"))
  b.onclick = () => send({ type: "sim", key: b.dataset.sim, value: b.getAttribute("aria-pressed") !== "true" });
for (const b of document.querySelectorAll("[data-role]"))
  b.onclick = () => send({ type: "camera", role: b.dataset.role });
$("injectBtn").onclick = () => send({ type: "inject_typing" });

let toastTimer = null;
function toast(text) {
  const el = $("toast"); el.textContent = text; el.classList.add("on");
  clearTimeout(toastTimer); toastTimer = setTimeout(() => el.classList.remove("on"), 2200);
}

// ------------------------------------------------------------------ hologram
const N = 478;
const canvas = $("holo");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(2, devicePixelRatio || 1));
const scene = new THREE.Scene();
// Orthographic in canvas NDC (-1..1 both axes): landmarks are image coordinates, so the mesh
// lands exactly on the face in the preview underneath. Perspective would re-project them.
const camera3 = new THREE.OrthographicCamera(-1, 1, 1, -1, -10, 10);
camera3.position.set(0, 0, 5);
const group = new THREE.Group();
scene.add(group);

const pos = new Float32Array(N * 3);
const target = new Float32Array(N * 3);
const posAttr = new THREE.BufferAttribute(pos, 3);
posAttr.setUsage(THREE.DynamicDrawUsage);

const uniforms = { uTime: { value: 0 }, uScan: { value: 0 }, uSize: { value: 9 }, uPR: { value: renderer.getPixelRatio() },
  uPxScale: { value: 0.5 },
  uBeat: { value: 0 }, uAlpha: { value: 1 }, uColA: { value: new THREE.Color(C.holo) }, uColB: { value: new THREE.Color(C.plasma) } };
const pointVS = `
  uniform float uSize, uPR, uScan, uBeat, uTime, uPxScale;
  attribute float aHl;
  varying float vDepth, vScan, vHl;
  void main() {
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    vDepth = clamp(position.z * 2.2 + 0.5, 0.0, 1.0);
    vScan = exp(-pow((position.y - uScan) * 16.0, 2.0));
    vHl = aHl;
    float s = uSize * (1.0 + 0.7 * vScan + aHl * (0.45 + 0.7 * uBeat));
    gl_PointSize = s * uPR * uPxScale;
    gl_Position = projectionMatrix * mv;
  }`;
const pointFS = `
  uniform vec3 uColA, uColB; uniform float uAlpha, uBeat;
  varying float vDepth, vScan, vHl;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.0, d);
    float core = smoothstep(0.18, 0.0, d);
    vec3 col = mix(uColB, uColA, vDepth);
    col = mix(col, vec3(0.80, 0.52, 0.70), vHl * (0.30 + 0.25 * uBeat));
    col += vScan * vec3(0.22);
    gl_FragColor = vec4(col * (0.42 + 0.5 * core), a * uAlpha * (0.38 + 0.3 * vDepth + 0.35 * vScan));
  }`;
const hl = new Float32Array(N);
// skin patches the optical + pulse lanes read: forehead and both cheeks (MediaPipe indices)
const ROI = [10, 338, 297, 332, 284, 251, 21, 54, 103, 67, 109, 151, 108, 69, 104, 68, 337, 299, 333, 298, 9, 107, 336,
  50, 101, 36, 205, 187, 123, 116, 117, 118, 119, 100, 142, 203, 206, 207, 147,
  280, 330, 266, 425, 411, 352, 345, 346, 347, 348, 329, 371, 423, 426, 427, 376];
for (const i of ROI) hl[i] = 1;
const pGeo = new THREE.BufferGeometry();
pGeo.setAttribute("position", posAttr);
pGeo.setAttribute("aHl", new THREE.BufferAttribute(hl, 1));
const pMat = new THREE.ShaderMaterial({ uniforms, vertexShader: pointVS, fragmentShader: pointFS,
  transparent: true, depthWrite: false, blending: THREE.AdditiveBlending });
group.add(new THREE.Points(pGeo, pMat));

const lGeo = new THREE.BufferGeometry();
lGeo.setAttribute("position", posAttr);
const lMat = new THREE.LineBasicMaterial({ color: C.holo, transparent: true, opacity: 0.13,
  blending: THREE.AdditiveBlending, depthWrite: false });
const wire = new THREE.LineSegments(lGeo, lMat);
group.add(wire);

const floor = new THREE.PolarGridHelper(1.25, 24, 7, 96, 0x5cf2ff, 0x5cf2ff);
floor.material.transparent = true; floor.material.opacity = 0.16; floor.material.blending = THREE.AdditiveBlending;
floor.position.y = -0.95;
scene.add(floor);
const scanRing = new THREE.Mesh(new THREE.TorusGeometry(0.78, 0.004, 6, 160),
  new THREE.MeshBasicMaterial({ color: C.plasma, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending }));
scanRing.rotation.x = Math.PI / 2;
scene.add(scanRing);

let canon = null, faceTarget = null, haveFace = false;

// The mesh is tinted by what the rest of the HUD is saying, so the face itself reads as the
// verdict: cyan idle, magenta while a scan runs, then green / amber / red on the result.
const MOOD = {
  idle: [C.holo, C.plasma, C.holo],
  scan: [C.plasma, C.holo, C.plasma],
  ACCEPT: [C.phos, C.holo, C.phos],
  "RE-CHALLENGE": [C.amber, C.plasma, C.amber],
  REJECT: [C.alarm, C.plasma, C.alarm],
};
let mood = "idle";
const moodA = new THREE.Color(C.holo), moodB = new THREE.Color(C.plasma), moodW = new THREE.Color(C.holo);
function setMood(name) { if (MOOD[name]) mood = name; }
function mixMood(dt) {
  const [a, b, w] = MOOD[mood] || MOOD.idle, k = Math.min(1, dt * 3.5);
  moodA.lerp(new THREE.Color(a), k); moodB.lerp(new THREE.Color(b), k); moodW.lerp(new THREE.Color(w), k);
  uniforms.uColA.value.copy(moodA); uniforms.uColB.value.copy(moodB); lMat.color.copy(moodW);
}
fetch("/static/facemesh.json").then((r) => r.json()).then((m) => {
  const edges = new Set(), idx = [];
  for (const [a, b, c] of m.triangles) for (const [u, v] of [[a, b], [b, c], [c, a]]) {
    const k = u < v ? `${u}-${v}` : `${v}-${u}`;
    if (!edges.has(k)) { edges.add(k); idx.push(u, v); }
  }
  lGeo.setIndex(idx);
  // canonical face: cm, y up, z toward viewer -> same normalisation as live landmarks
  const v = m.vertices, flat = new Float32Array(N * 3);
  for (let i = 0; i < 468; i++) { flat[i * 3] = v[i][0]; flat[i * 3 + 1] = v[i][1]; flat[i * 3 + 2] = v[i][2]; }
  const eye = (a, b) => [0, 1, 2].map((k) => (v[a][k] + v[b][k]) / 2);
  for (let i = 468; i < N; i++) { const e = i < 473 ? eye(33, 133) : eye(362, 263); flat[i * 3] = e[0]; flat[i * 3 + 1] = e[1]; flat[i * 3 + 2] = e[2]; }
  canon = normalise(flat);
  if (!faceTarget) { target.set(canon); pos.set(canon); }
});

function normalise(flat) {
  let cx = 0, cy = 0, cz = 0, ymin = 1e9, ymax = -1e9;
  for (let i = 0; i < 468; i++) { cx += flat[i * 3]; cy += flat[i * 3 + 1]; cz += flat[i * 3 + 2]; ymin = Math.min(ymin, flat[i * 3 + 1]); ymax = Math.max(ymax, flat[i * 3 + 1]); }
  cx /= 468; cy /= 468; cz /= 468;
  const s = 1.45 / Math.max(1e-6, ymax - ymin), out = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    out[i * 3] = (flat[i * 3] - cx) * s; out[i * 3 + 1] = (flat[i * 3 + 1] - cy) * s; out[i * 3 + 2] = (flat[i * 3 + 2] - cz) * s;
  }
  return out;
}
function imageRect() {
  // where the <img> actually draws inside the stage: object-fit: contain letterboxes it
  const r = canvas.getBoundingClientRect();
  const W = Math.max(1, r.width), H = Math.max(1, r.height);
  const h = Math.min(H, W / camAspect), w = h * camAspect;
  return { W, H, x: (W - w) / 2, y: (H - h) / 2, w, h };
}

function toNdc(px, py, out, i) {
  const { W, H } = rect;
  out[i] = (px / W) * 2 - 1;
  out[i + 1] = 1 - (py / H) * 2;
}

let rect = { W: 1, H: 1, x: 0, y: 0, w: 1, h: 1 };

let lastFaceT = 0;
// Detection drops out constantly in dim light (33% face coverage measured on the demo laptop).
// Snapping between "locked on the face" and "idle mesh centre-stage" made that look like a
// tracking failure, so the mesh HOLDS its last fit, fades as confidence ages, and only returns
// to the idle mesh after FACE_DROP_S of nothing at all.
const FACE_GRACE_S = 0.45;  // full opacity: the fit is still trusted (~3-4 missed detections at 16 fps)
const FACE_DROP_S = 1.4;    // after this, give up and idle. Held longer, a stale fit trails the
                            // moving head and reads as lag (seen at 33% face coverage, 2026-10-24)

function setFaceTarget(lm) {
  lastFaceT = performance.now() / 1000;
  // MediaPipe gives x,y normalised to the IMAGE (y down) and z as depth in x-scale units.
  // The preview is mirrored (scaleX(-1)), so x is flipped to match what the operator sees;
  // the result lands on the face pixel-for-pixel, which is the point of the overlay.
  rect = imageRect();
  const f = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    const px = rect.x + (1 - lm[i * 3]) * rect.w;
    const py = rect.y + lm[i * 3 + 1] * rect.h;
    toNdc(px, py, f, i * 3);
    f[i * 3 + 2] = -lm[i * 3 + 2] * 1.8;      // depth: shading + scan only, ortho ignores it
  }
  faceTarget = f;
}

function canonTarget() {
  // no face in view: the canonical mesh idles in the middle of the stage, isotropically scaled
  if (!canon) return null;
  rect = imageRect();
  const k = 0.34 * Math.min(rect.W, rect.H);
  const f = new Float32Array(N * 3);
  for (let i = 0; i < N; i++) {
    toNdc(rect.W / 2 + canon[i * 3] * k, rect.H / 2 - canon[i * 3 + 1] * k, f, i * 3);
    f[i * 3 + 2] = canon[i * 3 + 2] * 0.5;
  }
  return f;
}

function resize() {
  const r = canvas.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  // ortho stays -1..1 in both axes (canvas NDC); only the point size tracks the stage size
  uniforms.uPxScale.value = Math.max(0.28, Math.min(1.0, Math.min(r.width, r.height) / 900));
  rect = imageRect();
}
new ResizeObserver(resize).observe(canvas);

const clock = new THREE.Clock();
function frame() {
  emitFrame();
  const dt = Math.min(0.05, clock.getDelta()), t = clock.elapsedTime;
  // A dropped frame or a quick head turn makes MediaPipe miss a detection or two.
  // Falling straight back to the idle mesh makes the overlay flicker off the face,
  // so the last landmarks hold for FACE_GRACE_S before the mesh gives up.
  const ageS = performance.now() / 1000 - lastFaceT;
  haveFace = !!faceTarget && ageS < FACE_DROP_S;
  if (!haveFace) faceTarget = null;
  // 1 while the fit is fresh, easing to 0.25 as it ages out -- it dims rather than jumping away
  const faceConf = !haveFace ? 0 : ageS <= FACE_GRACE_S ? 1
    : 1 - 0.75 * (ageS - FACE_GRACE_S) / (FACE_DROP_S - FACE_GRACE_S);
  const tgt = faceTarget || canonTarget();
  if (tgt) {
    // snappier when locked to a face: the mesh has to sit ON the head, not trail it
    const k = 1 - Math.pow(haveFace ? 0.00000012 : 0.0015, dt);
    for (let i = 0; i < N * 3; i++) pos[i] += (tgt[i] - pos[i]) * k;
    posAttr.needsUpdate = true;
  }
  // locked to a face: no spin, or the overlay would slide off the head
  group.rotation.y = haveFace ? 0 : (reduceMotion ? 0 : t * 0.35);
  floor.visible = scanRing.visible = !haveFace;
  uniforms.uAlpha.value += ((haveFace ? 0.62 * faceConf : 0.3) - uniforms.uAlpha.value) * 0.08;
  lMat.opacity += ((haveFace ? 0.17 * faceConf : 0.05) - lMat.opacity) * 0.08;
  mixMood(dt);
  const scanY = reduceMotion ? 0 : Math.sin(t * 0.9) * 0.78;
  uniforms.uScan.value = scanY; scanRing.position.y = scanY;
  scanRing.scale.setScalar(Math.sqrt(Math.max(0.05, 1 - (scanY / 0.95) ** 2)));
  floor.rotation.y = -t * 0.1;
  uniforms.uBeat.value = 0;   // the pulse lane is not shown: nothing on screen may imply it
  uniforms.uTime.value = t;
  renderer.render(scene, camera3);
  if (T) drawRing(T);
  requestAnimationFrame(frame);
}
resize();
requestAnimationFrame(frame);
connect();
window.__hud = { onResult, hold: false };  // debug hook: __hud.hold = true; __hud.onResult(result)
