# Figure PDF Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove unused local figure PDFs from the repository and stop figure scripts from generating PDF outputs while preserving existing PNG-based tutorial references.

**Architecture:** Add one regression test that treats local PDF usage as forbidden state: no Markdown file should reference `assets/figures/*.pdf`, no figure script should emit `.pdf`, and no local PDF artifact should remain under `assets/figures/`. Then make the smallest possible cleanup to the affected figure scripts and delete the orphaned PDF artifacts so the test turns green.

**Tech Stack:** Python 3, `unittest`, repository Markdown files, `assets/figures/*.py`

---

## File map

- Create: `tests/test_figure_pdf_cleanup.py` — regression test for local PDF references, script exports, and leftover PDF artifacts
- Modify:
  - `assets/figures/3-1-xor-scatter.py`
  - `assets/figures/fig1_loss_surface.py`
  - `assets/figures/fig2_mse_2d.py`
  - `assets/figures/fig3_bce_cross_entropy.py`
  - `assets/figures/fig4_triplet_loss.py`
  - `assets/figures/fig5_loss_decision_tree.py`
  - `assets/figures/fig_convolution_kernel.py`
  - `assets/figures/fig_stride_comparison.py`
  - `assets/figures/fig_inception_module.py`
  - `assets/figures/fig_googlenet_architecture.py`
  - `assets/figures/resnet_block.py`
  - `assets/figures/gradient_flow_comparison.py`
  - `assets/figures/fig11-2-transposed-conv.py`
  - `assets/figures/fig11-2-unet.py`
- Delete:
  - `assets/figures/3-1-xor-scatter.pdf`
  - `assets/figures/fig1_loss_surface.pdf`
  - `assets/figures/fig2_mse_2d.pdf`
  - `assets/figures/fig3_bce_cross_entropy.pdf`
  - `assets/figures/fig4_triplet_loss.pdf`
  - `assets/figures/fig5_loss_decision_tree.pdf`
  - `assets/figures/fig_convolution_kernel.pdf`
  - `assets/figures/fig_stride_comparison.pdf`
  - `assets/figures/fig_inception_module.pdf`
  - `assets/figures/fig_googlenet_architecture.pdf`
  - `assets/figures/resnet_block.pdf`
  - `assets/figures/gradient_flow_comparison.pdf`
  - `assets/figures/fig11-2-transposed-conv.pdf`
  - `assets/figures/fig11-2-unet.pdf`

### Task 1: Add a regression test for forbidden local PDF usage

**Files:**
- Create: `tests/test_figure_pdf_cleanup.py`

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path
import re
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
MARKDOWN_FILES = [
    REPO_ROOT / "README.md",
    REPO_ROOT / "SUMMARY.md",
    *sorted((REPO_ROOT / "chapters").rglob("*.md")),
]
FIGURE_SCRIPTS = sorted((REPO_ROOT / "assets" / "figures").glob("*.py"))
LOCAL_PDFS = sorted((REPO_ROOT / "assets" / "figures").glob("*.pdf"))
PDF_EXPORT_PATTERNS = [
    ".pdf",
    "with_suffix('.pdf')",
    'with_suffix(".pdf")',
]


class FigurePdfCleanupTest(unittest.TestCase):
    def test_markdown_and_figure_scripts_do_not_use_local_pdf_outputs(self) -> None:
        markdown_hits = []
        for path in MARKDOWN_FILES:
            text = path.read_text(encoding="utf-8")
            for line_no, line in enumerate(text.splitlines(), start=1):
                if "assets/figures/" in line and ".pdf" in line:
                    markdown_hits.append(f"{path.relative_to(REPO_ROOT)}:{line_no}:{line.strip()}")

        script_hits = []
        for path in FIGURE_SCRIPTS:
            text = path.read_text(encoding="utf-8")
            if any(pattern in text for pattern in PDF_EXPORT_PATTERNS):
                script_hits.append(str(path.relative_to(REPO_ROOT)))

        leftover_pdfs = [str(path.relative_to(REPO_ROOT)) for path in LOCAL_PDFS]

        self.assertEqual([], markdown_hits, f"Unexpected local PDF Markdown references: {markdown_hits}")
        self.assertEqual([], script_hits, f"Unexpected PDF exports in figure scripts: {script_hits}")
        self.assertEqual([], leftover_pdfs, f"Unexpected local PDF figure artifacts: {leftover_pdfs}")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_figure_pdf_cleanup -v`
Expected: `FAIL` because the current repository still contains `assets/figures/*.pdf` files and several figure scripts still include `.pdf` export code.

### Task 2: Remove PDF exports and orphaned PDF artifacts

**Files:**
- Modify:
  - `assets/figures/3-1-xor-scatter.py`
  - `assets/figures/fig1_loss_surface.py`
  - `assets/figures/fig2_mse_2d.py`
  - `assets/figures/fig3_bce_cross_entropy.py`
  - `assets/figures/fig4_triplet_loss.py`
  - `assets/figures/fig5_loss_decision_tree.py`
  - `assets/figures/fig_convolution_kernel.py`
  - `assets/figures/fig_stride_comparison.py`
  - `assets/figures/fig_inception_module.py`
  - `assets/figures/fig_googlenet_architecture.py`
  - `assets/figures/resnet_block.py`
  - `assets/figures/gradient_flow_comparison.py`
  - `assets/figures/fig11-2-transposed-conv.py`
  - `assets/figures/fig11-2-unet.py`
- Delete:
  - `assets/figures/3-1-xor-scatter.pdf`
  - `assets/figures/fig1_loss_surface.pdf`
  - `assets/figures/fig2_mse_2d.pdf`
  - `assets/figures/fig3_bce_cross_entropy.pdf`
  - `assets/figures/fig4_triplet_loss.pdf`
  - `assets/figures/fig5_loss_decision_tree.pdf`
  - `assets/figures/fig_convolution_kernel.pdf`
  - `assets/figures/fig_stride_comparison.pdf`
  - `assets/figures/fig_inception_module.pdf`
  - `assets/figures/fig_googlenet_architecture.pdf`
  - `assets/figures/resnet_block.pdf`
  - `assets/figures/gradient_flow_comparison.pdf`
  - `assets/figures/fig11-2-transposed-conv.pdf`
  - `assets/figures/fig11-2-unet.pdf`

- [ ] **Step 1: Write minimal implementation**

For every listed script, delete the `.pdf` save branch and keep the existing `.png` output unchanged. The resulting shape should look like this:

```python
output_path = output_dir / "example.png"
plt.savefig(output_path, bbox_inches="tight", facecolor="white")
print(f"Saved figure to {output_path}")
```

For scripts that currently create a `pdf_path`, remove that variable entirely:

```python
output_path = output_dir / "example.png"
plt.savefig(output_path, bbox_inches="tight", facecolor="white")
plt.close(fig)
```

Then delete the matching `.pdf` files from `assets/figures/`.

- [ ] **Step 2: Run test to verify it passes**

Run: `python -m unittest tests.test_figure_pdf_cleanup -v`
Expected: `OK`

- [ ] **Step 3: Re-run the repository checks used for this change**

Run: `rg "\.pdf" README.md SUMMARY.md chapters -g '*.md'`
Expected: only external paper links or code snippets such as `norm.pdf(...)`; no hits under `assets/figures/*.pdf`

Run: `rg "with_suffix\('\.pdf'\)|with_suffix\\(\"\\.pdf\"\\)|savefig\(.*pdf" assets/figures -g '*.py'`
Expected: no matches

## Self-review

- Spec coverage: the plan covers all spec requirements — reference scan, script cleanup, and artifact deletion
- Placeholder scan: no `TODO`/`TBD` items remain
- Type consistency: one test file checks the exact cleanup contract the implementation is meant to satisfy
