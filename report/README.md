# PRAESENS — B.Tech. BCSE497J Project-I report

`PRAESENS_Project_I_Report.pdf` is the final report (VIT template structure).

Rebuild:
- `scripts/figs.py <fig_dir> <cross_session_full.json>` regenerates the data figures from `logs/` (run under `xvfb-run`; the JSON comes from `python -m eval.cross_session --output-json ...`).
- `scripts/gantt.py fig/fig_gantt.png` draws the project plan.
- `node diagrams/render.js fig` renders the Mermaid design diagrams (needs `mermaid` + `playwright`).
- `python src/build.py` assembles `src/report.html`; render with WeasyPrint: `weasyprint src/report.html report.pdf`.
