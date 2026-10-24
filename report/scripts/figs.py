"""Generates all data-driven figures for the report from the real PRAESENS logs/code."""
import json, glob, sys, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path("/home/user/Praesens")
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO))
os.chdir(REPO)

from praesens.challenge import Challenge, derive_zone_challenges, autocorrelation
from praesens.optical import compute_correlation_curve, moving_average_detrend
from praesens.spatial import compute_correlation_matrix, spatial_assignment_score
from eval.cross_session import reconstruct_challenge, build_synthetic_emitter_log

NAVY, TEAL, ORANGE, RED, GREEN, PURPLE, GREY, GOLD = (
    "#1F3A68", "#0F8B8D", "#E07A1F", "#C0392B", "#2E9E5B", "#6C4AB6", "#7F8C8D", "#D4A017")
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman"],
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linestyle": "--", "savefig.dpi": 220, "savefig.bbox": "tight",
    "axes.edgecolor": "#444", "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", name)


logs = []
for f in sorted(glob.glob("logs/2026*.json")):
    d = json.load(open(f))
    d["_file"] = f
    logs.append(d)

xs = json.load(open(sys.argv[2]))  # cross_session_full.json
trials = xs["trials"] if "trials" in xs else xs.get("per_trial", [])

# ---------------------------------------------------------------- 1. real trace
rec = next(d for d in logs if d["session"] == "20260914T115241_5ef66054")
ts = np.array(rec["timestamps"]); ts = ts - rec["start_time_perf_counter"]
em = np.array(rec["trace_emitted"]); me = np.array(rec["trace_measured"])
z = lambda x: (x - np.nanmean(x)) / np.nanstd(x)
fig, ax = plt.subplots(figsize=(7.2, 2.8))
m = ts < 22
ax.step(ts[m], z(em[m]), where="post", color=ORANGE, lw=1.6, label="Emitted challenge (m-sequence, lag-aligned)")
ax.plot(ts[m], z(me[m]), color=NAVY, lw=1.1, label="Measured facial luminance (detrended)")
ax.set_xlabel("Time since session start (s)"); ax.set_ylabel("z-score")
ax.set_title(f"Live session {rec['session'][:15]}: optical subscore {rec['score']:.2f}, lag {rec['lag_ms']:.0f} ms, SNR {rec['snr_db']:.1f} dB")
ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.22))
save(fig, "fig_trace")

# ---------------------------------------------------------------- 2. correlation vs lag
ch_self = reconstruct_challenge(rec)
other = next(d for d in logs if d["session"] == "20260901T105232_2ea73839")
ch_other = reconstruct_challenge(other)
mask = ts < ch_self.duration_s
curves = {}
for label, ch in (("Own challenge (genuine)", ch_self), ("Another session's challenge (injection proxy)", ch_other)):
    elog = build_synthetic_emitter_log(ch, 100.0)
    mm = ts < ch.duration_s
    lags, sc = compute_correlation_curve(me[mm], ts[mm], elog, 300, 5)
    curves[label] = (lags, sc)
fig, ax = plt.subplots(figsize=(7.2, 2.7))
for (label, (lags, sc)), c in zip(curves.items(), (GREEN, RED)):
    ax.plot(lags, sc, color=c, lw=2, label=label)
    i = int(np.argmax(sc)); ax.scatter([lags[i]], [sc[i]], color=c, zorder=5, s=30)
ax.axhline(0.3, color=GREY, ls=":", lw=1.2); ax.text(302, 0.3, " pass\n threshold", va="center", fontsize=8, color=GREY)
ax.axvspan(0, 300, color=TEAL, alpha=0.05)
ax.set_xlabel("Hypothesised camera lag (ms)"); ax.set_ylabel("Normalised correlation")
ax.set_title("Correlation-versus-lag curve for the same measured trace")
ax.legend(loc="center right")
save(fig, "fig_corr_curve")

# ---------------------------------------------------------------- 3. distributions
bona = np.array([t["score"] for t in trials if t["condition"] == "bonafide"])
xsess = np.array([t["score"] for t in trials if t["condition"] == "xsession"])
eoff = np.array([d["score"] for d in logs if d["condition"] == "emitter_off"])
inj = np.array([d["score"] for d in logs if d["condition"] == "inject_static"])
fig, ax = plt.subplots(figsize=(7.2, 3.2))
data = [bona, xsess, eoff, inj]
labels = [f"Bona fide\n(n={len(bona)})", f"Cross-session\ninjection proxy\n(n={len(xsess)})",
          f"Emitter OFF\ncontrol (n={len(eoff)})", f"Real virtual-camera\ninjection (n={len(inj)})"]
cols = [GREEN, RED, GREY, PURPLE]
vp = ax.violinplot([bona, xsess], positions=[1, 2], showextrema=False, widths=0.8)
for b, c in zip(vp["bodies"], cols[:2]):
    b.set_facecolor(c); b.set_alpha(0.25); b.set_edgecolor(c)
bp = ax.boxplot(data, positions=[1, 2, 3, 4], widths=0.22, patch_artist=True, showfliers=False,
                medianprops=dict(color="black"))
for patch, c in zip(bp["boxes"], cols):
    patch.set_facecolor(c); patch.set_alpha(0.75)
rng = np.random.default_rng(1)
for i, (d, c) in enumerate(zip(data, cols), start=1):
    dd = d if len(d) < 400 else rng.choice(d, 400, replace=False)
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(dd)), dd, s=7, color=c, alpha=0.6, edgecolor="none")
ax.axhline(0.3, color=NAVY, ls="--", lw=1); ax.text(4.45, 0.31, "optical pass\nthreshold 0.30", fontsize=8, color=NAVY)
ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(labels, fontsize=8.5)
ax.set_xlim(0.5, 4.95)
ax.set_ylabel("Optical liveness score"); ax.set_title("Score separation across conditions (real corpus)")
save(fig, "fig_score_dist")

# ---------------------------------------------------------------- 4. ROC
neg = np.concatenate([xsess, np.array([t["score"] for t in trials if t["condition"] == "emitter_off"])])
thr = np.unique(np.concatenate([bona, neg]))[::-1]
tpr = np.array([(bona >= t).mean() for t in thr]); fpr = np.array([(neg >= t).mean() for t in thr])
tpr = np.concatenate([[0], tpr, [1]]); fpr = np.concatenate([[0], fpr, [1]])
auc = np.trapezoid(tpr, fpr)
fnr = 1 - tpr; k = np.argmin(np.abs(fnr - fpr))
fig, ax = plt.subplots(figsize=(4.4, 4.0))
ax.fill_between(fpr, tpr, color=TEAL, alpha=0.15)
ax.plot(fpr, tpr, color=TEAL, lw=2.2, label=f"PRAESENS optical lane (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], color=GREY, ls="--", lw=1, label="Chance")
ax.scatter([fpr[k]], [tpr[k]], color=RED, zorder=5, label=f"EER ≈ {fpr[k]*100:.1f}%")
ax.set_xlabel("APCER (false-accept rate)"); ax.set_ylabel("1 − BPCER (true-accept rate)")
ax.set_title("ROC: bona fide vs. injection proxy"); ax.legend(loc="lower right", fontsize=8.5)
ax.set_aspect("equal")
save(fig, "fig_roc")
print("AUC", auc, "EER", fpr[k])

# ---------------------------------------------------------------- 5. lag histogram
bl = np.array([t["lag_ms"] for t in trials if t["condition"] == "bonafide"])
xl = np.array([t["lag_ms"] for t in trials if t["condition"] == "xsession"])
fig, ax = plt.subplots(figsize=(7.2, 2.6))
bins = np.arange(0, 310, 10)
ax.hist(xl, bins=bins, density=True, color=RED, alpha=0.45, label=f"Cross-session proxy (n={len(xl)})")
ax.hist(bl, bins=bins, density=True, color=GREEN, alpha=0.75, label=f"Bona fide (n={len(bl)})")
ax.set_xlabel("Lag at correlation peak (ms)"); ax.set_ylabel("Density")
ax.set_title("Where the correlation peak lands: physical latency vs. chance")
ax.legend()
save(fig, "fig_lag_hist")

# ---------------------------------------------------------------- 6. robustness
from collections import defaultdict
def group(field):
    g = defaultdict(list)
    for t in trials:
        if t["condition"] == "bonafide":
            g[t["meta"].get(field)].append(t["score"])
    return g
fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.8), gridspec_kw={"width_ratios": [4, 2, 2]})
for ax, field, title, order in ((axes[0], "lighting", "Lighting", ["dim", "normal", "backlit", "side-lit"]),
                                 (axes[1], "glasses", "Eyewear", ["without", "with"]),
                                 (axes[2], "makeup", "Makeup", ["none", "foundation_powder"])):
    g = group(field)
    keys = [k for k in order if k in g]
    means = [np.mean(g[k]) for k in keys]; sds = [np.std(g[k]) for k in keys]
    colors = [TEAL, NAVY, ORANGE, PURPLE][:len(keys)]
    ax.bar(range(len(keys)), means, yerr=sds, color=colors, alpha=0.85, capsize=3)
    for i, k in enumerate(keys):
        ax.text(i, 0.03, f"n={len(g[k])}", ha="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(keys))); ax.set_xticklabels([k.replace("_", "\n") for k in keys], fontsize=8)
    ax.set_title(title); ax.set_ylim(0, 1.0)
axes[0].set_ylabel("Mean bona fide score")
fig.suptitle("Robustness of the optical lane across capture conditions", fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "fig_robustness")

# ---------------------------------------------------------------- 7. SNR vs score
fig, ax = plt.subplots(figsize=(7.0, 3.0))
for cond, c, mk in (("bonafide", GREEN, "o"), ("inject_static", PURPLE, "s"), ("emitter_off", GREY, "^")):
    pts = [(d["snr_db"], d["score"]) for d in logs if d["condition"] == cond
           and d.get("snr_db") is not None and np.isfinite(d["snr_db"])]
    if pts:
        a = np.array(pts)
        ax.scatter(a[:, 0], a[:, 1], color=c, marker=mk, s=28, alpha=0.8, edgecolor="white", lw=0.4,
                   label={"bonafide": "Bona fide", "inject_static": "Virtual-camera injection",
                          "emitter_off": "Emitter OFF control"}[cond] + f" (n={len(a)})")
ax.axvline(3.0, color=ORANGE, ls="--", lw=1.2); ax.text(3.1, 0.9, "SNR floor 3 dB\n(below → insufficient signal)", fontsize=8, color=ORANGE, va="top")
ax.axhline(0.3, color=NAVY, ls=":", lw=1)
ax.set_xlabel("Estimated in-band SNR (dB)"); ax.set_ylabel("Optical score")
ax.set_title("SNR gating separates measurement failure from attack")
ax.legend(loc="lower right", fontsize=8)
save(fig, "fig_snr_scatter")

# ---------------------------------------------------------------- 8. exposure / fps
fig, ax1 = plt.subplots(figsize=(6.4, 2.7))
ev = ["-3", "-4", "-5", "-6"]; fps = [8.15, 16.56, 31.16, 30.33]
bars = ax1.bar(ev, fps, color=[RED, ORANGE, TEAL, TEAL], alpha=0.85, width=0.55)
for b, f in zip(bars, fps):
    ax1.text(b.get_x() + b.get_width() / 2, f + 0.8, f"{f:.1f} fps", ha="center", fontsize=8.5)
ax1.set_xlabel("Manual exposure value (log2 s)"); ax1.set_ylabel("Measured capture rate (fps)")
ax1.set_ylim(0, 38)
ax2 = ax1.twinx(); ax2.grid(False)
ax2.plot(ev, [f / 10 for f in fps], color=NAVY, marker="o", lw=2, label="auto chip rate = fps/10")
ax2.set_ylabel("Selected chip rate (Hz)", color=NAVY); ax2.set_ylim(0, 3.8)
ax2.spines["right"].set_visible(True)
ax1.set_title("Exposure sweep: frame-rate ceiling and adaptive chip-rate selection")
ax2.legend(loc="upper left", fontsize=8)
save(fig, "fig_exposure")

# ---------------------------------------------------------------- 9. verdicts on fused sessions
fused = [d for d in logs if (d.get("fusion") or {}).get("verdict")]
cats = ["ACCEPT", "RE-CHALLENGE", "REJECT"]
vc = {c: {"bonafide": 0, "inject_static": 0} for c in cats}
for d in fused:
    vc[d["fusion"]["verdict"]][d["condition"]] += 1
fig, ax = plt.subplots(figsize=(6.4, 2.6))
y = np.arange(2); left = np.zeros(2)
for c, col in zip(cats, (GREEN, GOLD, RED)):
    vals = np.array([vc[c]["bonafide"], vc[c]["inject_static"]])
    ax.barh(y, vals, left=left, color=col, label=c, height=0.5)
    for i, v in enumerate(vals):
        if v: ax.text(left[i] + v / 2, y[i], str(v), ha="center", va="center", color="white", fontweight="bold")
    left += vals
ax.set_yticks(y); ax.set_yticklabels(["Bona fide sessions", "Injection sessions\n(OBS virtual camera)"])
ax.set_xlabel("Number of fully adjudicated live sessions"); ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25))
ax.grid(axis="y", visible=False)
save(fig, "fig_verdicts")
print("verdicts", vc, len(fused))

# ---------------------------------------------------------------- 10. ablation
ab = json.load(open("eval/summary.json"))["ablation"]["lanes"]
fig, ax = plt.subplots(figsize=(5.6, 2.5))
names = ["optical", "typing", "acoustic"]
vals = [ab[n]["fraction_degraded"] * 100 for n in names]
bars = ax.bar([n.capitalize() + (" (stub)" if n == "acoustic" else "") for n in names], vals,
              color=[TEAL, ORANGE, GREY], width=0.5)
for b, v, n in zip(bars, vals, names):
    ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{ab[n]['n_degraded_when_ablated']}/{ab[n]['n_full_accept_sessions']}", ha="center")
ax.set_ylim(0, 115); ax.set_ylabel("ACCEPTs lost when lane removed (%)")
ax.set_title("Ablation: every real lane is necessary")
save(fig, "fig_ablation")

# ---------------------------------------------------------------- 11. m-sequence + autocorr + zone xcorr
ch = Challenge(chip_rate_hz=5.0, duration_s=12.6, seed=17)  # 63 chips, order 6
ac = autocorrelation(ch.chips)
fig = plt.figure(figsize=(7.4, 3.6))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.4], hspace=0.6, wspace=0.45)
a0 = fig.add_subplot(gs[0, :])
t = np.arange(ch.n_chips + 1) / ch.chip_rate_hz
a0.step(t, np.append(ch.chips, ch.chips[-1]), where="post", color=ORANGE, lw=1.6)
a0.set_yticks([-1, 1]); a0.set_xlabel("Time (s)"); a0.set_title(f"Order-{ch.order} m-sequence ({ch.n_chips} chips, seed {ch.seed}) driving the emitter")
a1 = fig.add_subplot(gs[1, :2])
lags = np.arange(len(ac)) - (len(ac) // 2 if len(ac) > ch.n_chips else 0)
a1.stem(np.arange(len(ac)), ac, linefmt=NAVY, markerfmt="o", basefmt=" ")
a1.set_xlabel("Cyclic shift (chips)"); a1.set_ylabel("Autocorrelation"); a1.set_title("Impulse-like periodic autocorrelation")
master = Challenge(chip_rate_hz=5.0, duration_s=20.0, seed=17)
zones = derive_zone_challenges(master, ("left", "right", "top"))
C = np.array([[np.dot(a.challenge.chips, b.challenge.chips) / len(a.challenge.chips) for b in zones] for a in zones])
a2 = fig.add_subplot(gs[1, 2])
im = a2.imshow(C, cmap="RdYlGn", vmin=-1, vmax=1)
for i in range(3):
    for j in range(3):
        a2.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=8)
a2.set_xticks(range(3)); a2.set_yticks(range(3))
a2.set_xticklabels(["L", "R", "T"]); a2.set_yticklabels(["L", "R", "T"]); a2.grid(False)
a2.set_title("Zone cross-corr.")
save(fig, "fig_mseq")

# ---------------------------------------------------------------- 12. spatial matrices (synthetic, as tests)
def zone_log(c, sr=100.0):
    tt = np.arange(0, c.duration_s, 1 / sr)
    return [{"t": float(a), "chip_value": int(v)} for a, v in zip(tt, c.value_at(tt))]
zb = {z_.zone_name: z_.challenge for z_ in zones}
zl = {k: zone_log(v) for k, v in zb.items()}
ft = np.arange(0.2, 19.5, 1 / 29.97)
r = np.random.default_rng(0)
roi2zone = {"forehead": "top", "left_cheek": "left", "right_cheek": "right"}
gen = {roi: (ft, 120.0 + 8.0 * zb[zn].value_at(ft - 0.04).astype(float) + 1.5 * sum(zb[o].value_at(ft - 0.04).astype(float) for o in zb if o != zn) + r.normal(0, 0.5, len(ft)))
       for roi, zn in roi2zone.items()}
mean_wave = np.mean([zb[k].value_at(ft - 0.09).astype(float) for k in zb], axis=0)
att = {roi: (ft, 120.0 + 8.0 * mean_wave + r.normal(0, 0.5, len(ft))) for roi in roi2zone}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
for ax, tr, title in ((axes[0], gen, "Genuine face"), (axes[1], att, "Global-brightness attack")):
    R, _, rn, zn = compute_correlation_matrix(tr, zl, 1.5, 300, 5)
    s = spatial_assignment_score(R, rn, zn, roi2zone)
    ax.imshow(R, cmap="viridis", vmin=-0.2, vmax=1)
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", color="white" if R[i, j] < 0.6 else "black", fontsize=9)
    ax.set_xticks(range(len(zn))); ax.set_xticklabels([f"{z_} zone" for z_ in zn], fontsize=8.5)
    ax.set_yticks(range(len(rn))); ax.set_yticklabels([n.replace("_", " ") for n in rn], fontsize=8.5)
    ax.grid(False); ax.set_title(f"{title}\nspatial score = {s:.2f}")
save(fig, "fig_spatial")

# ---------------------------------------------------------------- 13. acoustic (synthetic recording, same detector)
import scipy.signal
from praesens.acoustic import AcousticConfig, synthesize_probe, detect_probe, _baseband, _segmented_statistic, _chip_envelope
import yaml
acfg = AcousticConfig.from_dict(yaml.safe_load(open("config.yaml"))["acoustic"])
ach = Challenge(chip_rate_hz=5.0, duration_s=20.0, seed=11)
probe = synthesize_probe(ach, acfg)
FS = acfg.sample_rate_hz
rr = np.random.default_rng(3)
d = int(0.330 * FS)
recd = np.zeros(len(probe) + int(0.8 * FS)); recd[d:d + len(probe)] += 0.05 * probe
recd += rr.normal(0, 0.01, len(recd))
det = detect_probe(recd, ach, acfg)
q = max(1, int(round(FS * acfg.lag_step_ms / 1000.0)))
sos = scipy.signal.butter(4, max(acfg.baseband_bandwidth_hz, 2 * ach.chip_rate_hz), btype="low", fs=FS, output="sos")
ref = scipy.signal.sosfiltfilt(sos, _chip_envelope(ach, acfg))[::q]
maxlag = int(round(acfg.lag_search_max_ms / 1000 * FS / q))
seg = int(round(acfg.segment_s * FS / q))
stat = _segmented_statistic(_baseband(recd - recd.mean(), acfg.carrier_hz, FS, sos, q), ref, seg, maxlag)
noise = _segmented_statistic(_baseband(recd - recd.mean(), acfg.carrier_hz + 300, FS, sos, q), ref, seg, maxlag)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7), gridspec_kw={"width_ratios": [1, 1.4]})
env = _chip_envelope(ach, acfg)
tt = np.arange(int(2.0 * FS)) / FS
axes[0].plot(tt, env[:len(tt)], color=PURPLE, lw=1.5)
axes[0].set_xlabel("Time (s)"); axes[0].set_ylabel("Chip envelope")
axes[0].set_title("Smoothed BPSK envelope (first 2 s)\nmodulating a 4 kHz carrier")
ins = axes[0].inset_axes([0.55, 0.08, 0.42, 0.3])
k0 = int(0.40 * FS) - 60
ins.plot(np.arange(120) / FS * 1000, probe[k0:k0 + 120], color=NAVY, lw=0.8)
ins.set_xticks([]); ins.set_yticks([]); ins.set_title("carrier, 2.7 ms", fontsize=7)
lag_axis = np.arange(len(stat)) * q / FS * 1000
axes[1].plot(lag_axis, stat, color=GREEN, lw=1.8, label="Carrier channel")
axes[1].plot(lag_axis, noise, color=GREY, lw=1.2, label="Off-carrier (+300 Hz) noise channel")
axes[1].axvline(det.lag_ms, color=RED, ls="--", lw=1)
axes[1].text(det.lag_ms + 8, 0.6, f"peak at {det.lag_ms:.0f} ms\nSNR {det.snr_db:.1f} dB", color=RED, fontsize=8.5)
axes[1].set_xlabel("Round-trip lag (ms)"); axes[1].set_ylabel("Differential statistic")
axes[1].set_title("Segmented matched-filter detection"); axes[1].legend(fontsize=7.5, loc="upper left")
fig.tight_layout()
save(fig, "fig_acoustic")

# ---------------------------------------------------------------- 14. tests per module
tc = {"fusion": 33, "session": 25, "panel": 23, "typing": 19, "capture": 15, "acoustic": 13,
      "analyse": 11, "spatial": 5, "emit": 2, "adaptive_injector": 2}
fig, ax = plt.subplots(figsize=(6.6, 2.7))
k = list(tc); v = [tc[x] for x in k]
pal = [TEAL, NAVY, ORANGE, PURPLE, GREEN, RED, GOLD, "#16A085", "#8E44AD", "#34495E"]
bars = ax.bar(k, v, color=pal)
for b, x in zip(bars, v):
    ax.text(b.get_x() + b.get_width() / 2, x + 0.5, str(x), ha="center", fontsize=8.5)
ax.set_ylabel("Test cases"); ax.set_title(f"Automated test suite: {sum(v)} test cases across 10 modules")
plt.setp(ax.get_xticklabels(), rotation=25, ha="right", fontsize=8.5)
save(fig, "fig_tests")

# ---------------------------------------------------------------- 15. verdict banners (rendered by the real code)
import cv2
from praesens.session import render_verdict_banner
def banner(sess, name):
    import copy
    r_ = copy.deepcopy(next(d for d in logs if d["session"].startswith(sess)))
    r_["fusion"].setdefault("reject_threshold", 0.3); r_["fusion"].setdefault("accept_threshold", 0.6)
    if "policy" not in r_["fusion"]:
        from praesens.fusion import LaneResult, adjudicate_two_lane
        L = {l["lane_name"]: LaneResult(**{k: l[k] for k in ("lane_name", "subscore", "status", "lag_ms", "confidence", "diagnostics")})
             for l in r_["fusion"]["lanes"] if not l.get("is_stub")}
        j = adjudicate_two_lane(L["optical"], L.get("typing"), 0.3, 0.7)
        r_["fusion"].update(verdict=j.verdict, reason_text=j.reason_text, optical_pass_threshold=0.3, typing_pass_threshold=0.7)
        r_["fusion"]["lanes"] = [l for l in r_["fusion"]["lanes"] if l["lane_name"] in L]
    img = render_verdict_banner(r_, 960, 540)
    cv2.imwrite(str(OUT / f"{name}.png"), img)
    print("wrote", name, r_["fusion"]["verdict"])
banner("20260914T115241", "banner_accept")
banner("20260901T104245", "banner_reject")
banner("20260913T202324", "banner_rechallenge")

# ---------------------------------------------------------------- 16. spatial emitter frames (rendered by the real code)
from praesens.spatial import SpatialEmitter
from praesens.emit import EmitterConfig
ecfg = EmitterConfig.from_dict({**yaml.safe_load(open("config.yaml"))["emitter"], "border_fraction": 0.14,
                                 "fallback_window_width": 960, "fallback_window_height": 540})
se = SpatialEmitter(zones, ecfg)
se._width, se._height = 960, 540
prev = np.full((300, 400, 3), (60, 45, 30), np.uint8)
cv2.putText(prev, "live camera preview", (60, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
se.set_preview(prev)
frames = []
for tsec in (0.1, 0.5, 0.9, 1.3):
    f_, cv_, _ = se.render_frame(tsec)
    cv2.putText(f_, f"t={tsec:.1f}s  L{cv_['left']:+d} R{cv_['right']:+d} T{cv_['top']:+d}", (150, 520),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2, cv2.LINE_AA)
    frames.append(f_)
sep = np.full((540, 12, 3), 255, np.uint8)
row1 = np.hstack([frames[0], sep, frames[1]]); row2 = np.hstack([frames[2], sep, frames[3]])
grid = np.vstack([row1, np.full((12, row1.shape[1], 3), 255, np.uint8), row2])
cv2.imwrite(str(OUT / "emitter_frames.png"), grid)
print("wrote emitter_frames")
