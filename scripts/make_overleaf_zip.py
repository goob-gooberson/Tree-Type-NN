"""Bundle the report for Overleaf: report/*.tex, report/tables/ and the figures it uses -> overleaf.zip

    python scripts/make_overleaf_zip.py

On overleaf.com: New Project -> Upload Project -> choose overleaf.zip -> Recompile.
"""
import glob
import os
import re
import zipfile

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

if __name__ == "__main__":
    rep = os.path.join(ROOT, "report")
    tex = sorted(glob.glob(os.path.join(rep, "*.tex")))
    used = set()
    for f in tex:
        used |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", open(f, encoding="utf-8").read()))
    out = os.path.join(ROOT, "overleaf.zip")
    missing = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in tex:
            z.write(f, os.path.basename(f))
        for f in sorted(glob.glob(os.path.join(rep, "tables", "*.tex"))):
            z.write(f, "tables/" + os.path.basename(f))
        for name in sorted(used):
            src = os.path.join(ROOT, "figures", name)
            if os.path.exists(src):
                z.write(src, "figures/" + name)
            else:
                missing.append(name)
    print(f"wrote {out}: {len(tex)} .tex files, {len(used) - len(missing)} figures")
    if missing:
        print("MISSING figures (run scripts/make_figures.py / receptive_fields.py first):", missing)
