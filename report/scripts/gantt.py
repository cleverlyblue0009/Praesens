import sys, datetime as dt
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as md
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif"], "font.size": 9.5})
D = lambda s: dt.date.fromisoformat(s)
tasks = [
 ("Research", "Literature survey and threat model", "2026-07-06", 21),
 ("Research", "Problem formulation and system design", "2026-07-20", 14),
 ("Core pipeline", "M0 hardware probe, M1 m-sequence, M2 emitter", "2026-08-01", 4),
 ("Core pipeline", "M3 optical lane, M4 session runner, M5 analysis", "2026-08-03", 3),
 ("Core pipeline", "M6–M8 live dashboard, attack harness, panel", "2026-08-04", 3),
 ("Core pipeline", "First real corpus, exposure fixes", "2026-08-05", 4),
 ("Extended defences", "M9 spatially-differential challenge", "2026-08-09", 8),
 ("Extended defences", "M10 typing lane, M11 fusion", "2026-08-16", 5),
 ("Extended defences", "M12 attack tooling, M13 evaluation, M14 panel", "2026-08-18", 4),
 ("Hardening", "Cross-platform, capture and fps fixes", "2026-08-24", 9),
 ("Hardening", "Concurrent lanes, verdict banner", "2026-09-01", 3),
 ("Hardening", "Acoustic lane and three-lane gate", "2026-09-08", 7),
 ("Evaluation", "Corpus evaluation and ablation", "2026-09-15", 14),
 ("Evaluation", "Report writing and review", "2026-10-01", 25),
]
col = {"Research": "#1F3A68", "Core pipeline": "#0F8B8D", "Extended defences": "#6C4AB6", "Hardening": "#E07A1F", "Evaluation": "#2E9E5B"}
fig, ax = plt.subplots(figsize=(7.4, 4.2))
for i, (ph, name, s, d) in enumerate(tasks):
    ax.barh(i, d, left=md.date2num(D(s)), color=col[ph], height=0.62, alpha=0.9)
ax.set_yticks(range(len(tasks))); ax.set_yticklabels([t[1] for t in tasks], fontsize=8.5)
ax.invert_yaxis(); ax.xaxis_date(); ax.xaxis.set_major_formatter(md.DateFormatter("%d %b"))
ax.xaxis.set_major_locator(md.WeekdayLocator(byweekday=0, interval=2))
ax.grid(axis="x", ls="--", alpha=0.3); ax.spines[["top", "right"]].set_visible(False)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=c, label=k) for k, c in col.items()], ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.4, 1.1), frameon=False)
plt.setp(ax.get_xticklabels(), fontsize=8.5)
fig.savefig(sys.argv[1], dpi=220, bbox_inches="tight")
