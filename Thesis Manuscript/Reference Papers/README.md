# SP Manuscript — VSCode LaTeX Setup

## Prerequisites

MiKTeX is already installed. The only remaining prerequisite before the workflow is usable is **Strawberry Perl**, which `latexmk` requires to run.

**Strawberry Perl** — download the full installer from https://strawberryperl.com and run it. After install, open a new terminal and verify with:

```
perl -v
```

It should print a version string. If the command is not recognized, restart VSCode and the terminal — Perl was not added to PATH yet.

**latexmk** — after Perl is confirmed, open MiKTeX Console → Packages, search for "latexmk", and install it if not already present. MiKTeX may auto-install it on the first build attempt if "Install missing packages on-the-fly" is enabled.

## One-time VSCode setup

LaTeX Workshop by James Yu must be installed from the Extensions panel. Once installed, no manual configuration is needed — `.vscode/settings.json` in this repo already pre-configures LaTeX Workshop for this project.

## Daily workflow

1. Open the **project folder** in VSCode (not a single file)
2. Open `rodriguez_sp-manuscript.tex` — LaTeX Workshop detects the root file automatically via the magic comment at the top of the file
3. Open the PDF preview: **Ctrl+Alt+V**, or click the PDF icon in the top-right toolbar, or run "LaTeX Workshop: View LaTeX PDF" from the command palette (**Ctrl+Shift+P**)
4. Tile the editor and preview side-by-side using VSCode's split editor (drag the tab to the right half)
5. Edit any chapter file, save with **Ctrl+S** — latexmk rebuilds automatically and the preview refreshes

**SyncTeX (bidirectional sync between editor and PDF):**

- **Ctrl+Alt+J** from the editor jumps to the corresponding location in the PDF preview
- **Ctrl+click** in the PDF panel jumps back to the source line in the editor

## Understanding the build sequence

A complete LaTeX build with bibliography requires four passes: `pdflatex` → `bibtex` → `pdflatex` → `pdflatex`. The first pass produces auxiliary files that BibTeX reads; the second and third passes resolve cross-references and the table of contents. `latexmk` detects which passes are needed and runs them automatically — you never need to run the passes manually.

## Troubleshooting

**1. "perl is not recognized" or recipe fails immediately**
Strawberry Perl is not on PATH. Reinstall Strawberry Perl and fully restart VSCode after installation.

**2. "Recipe failed" on first build**
Usually a missing MiKTeX package. Open the LaTeX Workshop Output panel (click "LaTeX" in the bottom status bar) to read the error and package name. Open MiKTeX Console → Packages and install it manually, or enable auto-install under MiKTeX Console → Settings → "Install missing packages on-the-fly".

**3. PDF preview is blank**
The build may not have completed. Watch the status bar for the spinning LaTeX icon and wait for it to stop before concluding there is an error.

**4. References show as [?]**
BibTeX did not run or `biblio.bib` has an error. Trigger a full manual build once with **Ctrl+Alt+B**.

**5. Auxiliary files cluttering the folder**
Run "LaTeX Workshop: Clean up auxiliary files" from the command palette, or run `latexmk -c` in the terminal.

## File map

| File | Content |
|------|---------|
| `chapter1.tex` | Introduction (Background, Problem Statement, Objectives, Significance, Scope, Assumptions) |
| `chapter2.tex` | Review of Related Literature |
| `chapter3.tex` | (Methodology) |
| `chapter4.tex` – `chapter7.tex` | (Results, Discussion, etc.) |
| `chapter8.tex` | (Conclusion) |
| Appendix & Acknowledgment | Defined inline at end of `rodriguez_sp-manuscript.tex` |
