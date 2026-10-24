import re, sys, ast, html
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from refs import REFS
D = Path(__file__).parent
REPO = Path("/home/user/Praesens")
src = "".join((D / f).read_text() for f in ["p0_front.html", "p1_ch1_4.html", "p2_ch5_6.html", "p3_ch7_9.html"])

# ---- citations: number in order of first appearance
order = []
def cite(m):
    keys = [k.strip() for k in m.group(1).split(",")]
    nums = []
    for k in keys:
        if k not in REFS: raise SystemExit(f"unknown ref {k}")
        if k not in order: order.append(k)
        nums.append(order.index(k) + 1)
    return "[" + ", ".join(str(n) for n in nums) + "]"
src = re.sub(r"\{\{c:([^}]+)\}\}", cite, src)
unused = [k for k in REFS if k not in order]
if unused: print("UNUSED refs:", unused)
refs_html = '<ol class="refs">' + "".join(f"<li>{REFS[k]}</li>" for k in order) + "</ol>"
src = src.replace("{{REFS}}", refs_html)

# ---- figure / table numbering per chapter
labels = {}; lof = []; lot = []
def number_section(sec):
    ch = re.search(r'data-ch="([^"]+)"', sec).group(1)
    fi = [0]; ti = [0]
    def fig(m):
        fi[0] += 1; fid = m.group(1); n = f"{ch}.{fi[0]}"; labels[fid] = f"Fig. {n}"
        return m.group(0)
    for m in re.finditer(r'<figure id="(f-[^"]+)"', sec): fig(m)
    for m in re.finditer(r'<div class="tblcap" id="(t-[^"]+)">', sec):
        ti[0] += 1; labels[m.group(1)] = f"Table {ch}.{ti[0]}"
    def figcap(m):
        fid = m.group(1); body = m.group(2)
        cap = re.search(r"<figcaption>(.*?)</figcaption>", body, re.S).group(1)
        lof.append((labels[fid].replace("Fig. ", ""), cap, fid))
        body = body.replace(f"<figcaption>{cap}</figcaption>", f"<figcaption><b>{labels[fid]}</b> &nbsp;{cap}</figcaption>")
        return f'<figure id="{fid}"{body}</figure>'
    sec = re.sub(r'<figure id="(f-[^"]+)"(.*?)</figure>', figcap, sec, flags=re.S)
    def tblcap(m):
        tid = m.group(1); cap = m.group(2)
        lot.append((labels[tid].replace("Table ", ""), cap, tid))
        return f'<div class="tblcap" id="{tid}"><b>{labels[tid]}</b> &nbsp;{cap}</div>'
    sec = re.sub(r'<div class="tblcap" id="(t-[^"]+)">(.*?)</div>', tblcap, sec)
    return sec
parts = re.split(r'(?=<section class="main)', src)
src = parts[0] + "".join(number_section(p) for p in parts[1:])
src = re.sub(r"\{\{([ft]-[a-z0-9-]+)\}\}", lambda m: labels[m.group(1)], src)

strip_tags = lambda s: re.sub(r"<[^>]+>", "", s)
src = src.replace("{{LOF}}", '<table class="lst"><tr><td class="no">Figure No.</td><td>Title</td><td class="pg">Page No.</td></tr>' +
    "".join(f'<tr><td class="no">{n}</td><td>{c}</td><td class="pg"><a href="#{i}"></a></td></tr>' for n, c, i in lof) + "</table>")
src = src.replace("{{LOT}}", '<table class="lst"><tr><td class="no">Table No.</td><td>Title</td><td class="pg">Page No.</td></tr>' +
    "".join(f'<tr><td class="no">{n}</td><td>{c}</td><td class="pg"><a href="#{i}"></a></td></tr>' for n, c, i in lot) + "</table>")

# ---- TOC from headings
rows = ['<tr class="head"><td class="no">Sl. No</td><td>Contents</td><td class="pg">Page No.</td></tr>',
        '<tr class="ch"><td class="no"></td><td class="ttl"><a href="#abstract">Abstract</a></td><td class="pg"><a class="roman" href="#abstract"></a></td></tr>']
for m in re.finditer(r'<h([123]) id="([^"]+)">(.*?)</h\1>', src):
    lvl, hid, txt = m.group(1), m.group(2), strip_tags(m.group(3))
    if lvl == "1":
        num = hid[2:] + "." if hid.startswith("ch") else ""
        title = txt.upper() if hid.startswith("ch") else "APPENDIX A – " + txt.upper()
        rows.append(f'<tr class="ch"><td class="no">{num}</td><td class="ttl"><a href="#{hid}">{title}</a></td><td class="pg"><a class="arabic" href="#{hid}"></a></td></tr>')
    else:
        cls = "sub" if lvl == "2" else "sub2"
        rows.append(f'<tr class="{cls}"><td class="no"></td><td class="ttl"><a href="#{hid}">{txt}</a></td><td class="pg"><a class="arabic" href="#{hid}"></a></td></tr>')
src = src.replace("{{TOC}}", '<table class="toc">' + "".join(rows) + "</table>")

# ---- terminal block and code appendix
src = src.replace("{{CLEANBLOCK}}", html.escape((D.parent / "cleanblock.txt").read_text().strip("\n")))
def func_src(path, names):
    text = (REPO / path).read_text(); tree = ast.parse(text); lines = text.splitlines()
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            seg = lines[node.lineno - 1 - len(node.decorator_list): node.end_lineno]
            body0 = node.body[0]
            if isinstance(body0, ast.Expr) and isinstance(getattr(body0, "value", None), ast.Constant) and isinstance(body0.value.value, str):
                a, b = body0.lineno - node.lineno, body0.end_lineno - node.lineno
                seg = seg[:a] + seg[b + 1:]
            out.append((node.lineno, "\n".join(seg)))
    return "\n\n".join(s for _, s in sorted(out))
code = [
 ("A.1 LFSR m-sequence generation", "praesens/challenge.py", ["_step", "_lfsr_bits"]),
 ("A.2 Lag-searched normalised cross-correlation (optical lane)", "praesens/optical.py", ["_lag_sweep", "cross_correlate_lag_search"]),
 ("A.3 Spatial assignment score", "praesens/spatial.py", ["spatial_assignment_score"]),
 ("A.4 Privacy-preserving keystroke scoring", "praesens/typing.py", ["keystroke_normality_score"]),
 ("A.5 Segmented differential matched filter (acoustic lane)", "praesens/acoustic.py", ["_segmented_statistic"]),
 ("A.6 Binary-gate adjudication", "praesens/fusion.py", ["lane_passes", "adjudicate_two_lane"]),
]
ch = ""
for title, path, names in code:
    ch += f'<h3>{title}</h3><p class="small"><code>{path}</code></p><pre>{html.escape(func_src(path, names))}</pre>'
src = src.replace("{{CODE}}", ch)

i = src.index('<section class="front'); src = src[:i] + '<div class="front">' + src[i:]
j = src.index('<section class="main'); src = src[:j] + '</div><div class="main">' + src[j:] + '</div>'
src = src.replace('<section class="main main-first', '<section class="mainfirst')
src = re.sub(r'<section class="(front|main)( front-first)?(?=[ "])', '<section class="', src)
assert "{{" not in src, re.findall(r"\{\{[^}]*\}\}", src)[:5]
doc = f'<!doctype html><html><head><meta charset="utf-8"><title>PRAESENS Project-I Report</title><link rel="stylesheet" href="style.css"></head><body>{src}</body></html>'
(D / "report.html").write_text(doc)
print("figures", len(lof), "tables", len(lot), "refs", len(order))
