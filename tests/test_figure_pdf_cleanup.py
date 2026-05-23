import unittest
from pathlib import Path
import re


class TestFigurePdfCleanup(unittest.TestCase):
    def test_no_local_pdf_usage(self):
        repo_root = Path(__file__).resolve().parents[1]

        # Collect markdown files: README.md, SUMMARY.md, and chapters/**/*.md
        md_paths = []
        for p in (repo_root / "README.md", repo_root / "SUMMARY.md"):
            if p.exists():
                md_paths.append(p)
        md_paths.extend(sorted(repo_root.glob("chapters/**/*.md")))

        md_hits = []
        for p in md_paths:
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                text = p.read_text(encoding="latin-1")
            # Scan line-by-line and token-by-token to find local assets/figures/*.pdf references
            for lineno, line in enumerate(text.splitlines(), start=1):
                for raw_tok in re.split(r"\s+", line):
                    tok = raw_tok.strip("()[],:;\'\"")
                    ltok = tok.lower()
                    if "assets/figures/" in ltok and ".pdf" in ltok:
                        # ignore remote URLs containing a scheme like http:// or https://
                        if "://" in tok:
                            continue
                        md_hits.append(f"{p}:{lineno}:{tok}")

        # Collect figure scripts under assets/figures/*.py and detect local PDF export logic
        figures_dir = repo_root / "assets" / "figures"
        script_hits = []
        if figures_dir.exists():
            for s in sorted(figures_dir.glob("*.py")):
                try:
                    text = s.read_text(encoding="utf-8")
                except Exception:
                    text = s.read_text(encoding="latin-1")
                # Heuristics for local PDF export (avoid broad literal ".pdf" matches):
                # - savefig(..."some.pdf"...)
                # - savefig(..., format='pdf')
                # - use of PdfPages / backend_pdf
                if (
                    re.search(r"savefig\([^\)]*\.pdf", text, re.IGNORECASE)
                    or re.search(r"savefig\([^\)]*format\s*=\s*['\"]pdf['\"]", text, re.IGNORECASE)
                    or re.search(r"\bPdfPages\b", text)
                    or re.search(r"matplotlib\.backends?\.backend_pdf", text)
                ):
                    script_hits.append(str(s))

        # Detect any leftover PDF artifacts in assets/figures/*.pdf
        leftover_pdfs = []
        if figures_dir.exists():
            for p in sorted(figures_dir.glob("*.pdf")):
                leftover_pdfs.append(str(p))

        # Assertions: all three lists must be empty
        self.assertListEqual([], md_hits, f"Markdown files reference local PDFs: {md_hits}")
        self.assertListEqual([], script_hits, f"Figure scripts export PDFs locally: {script_hits}")
        self.assertListEqual([], leftover_pdfs, f"Leftover PDF artifacts found: {leftover_pdfs}")


if __name__ == "__main__":
    unittest.main()
