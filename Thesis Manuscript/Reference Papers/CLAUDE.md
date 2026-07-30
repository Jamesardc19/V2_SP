# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a LaTeX manuscript template for a **Special Problem (SP)** submitted to the University of the Philippines Manila, College of Arts and Sciences, Department of Physical Sciences and Mathematics, for the Bachelor of Science in Computer Science degree.

## Build Commands

```bash
# Compile the document (run twice for TOC/references to resolve)
pdflatex "rodriguez_sp-manuscript.tex"
bibtex "SP Docs Template"
pdflatex "rodriguez_sp-manuscript.tex"
pdflatex "rodriguez_sp-manuscript.tex"

# Or use latexmk for automatic dependency handling
latexmk -pdf "rodriguez_sp-manuscript.tex"

# Clean auxiliary files
latexmk -c
```

## Document Structure

The main entry point is `rodriguez_sp-manuscript.tex`, which inputs all other files:

- `titlepage.tex` — Title page using `\MyTitle` and `\MyAuthor` from `\title{}`/`\author{}` in the main file
- `acceptancesheet.tex` — Approval sheet template; parameters set in the preamble of the main file
- `chapter1.tex` – `chapter8.tex` — One file per chapter (Introduction through conclusion/appendices)
- `biblio.bib` — BibTeX bibliography in IEEE citation style (`\bibliographystyle{ieeetr}`)
- `source-code/` — Source code files included via `\lstinputlisting{}`
- `images/` — Figures included via `\includegraphics{}`

## Key Configuration in Main File

Update these fields in `rodriguez_sp-manuscript.tex` before use:

```latex
\title{SP Title}
\author{Juan Dela Cruz}
\date{June 2026}

\defadviser{Adviser Name, M.Sc.}
\defchair{Chair Name, Ph.D.}
\defdean{Dean Name, Ph.D.}
\defadviserblank{6.5cm}   % width of signature blank
\defdeanblank{6.5cm}
```

## Formatting Conventions

- **Page margins**: top/bottom 1in, left 1.5in (binding), right 1in — set via `geometry` package
- **Spacing**: double-spaced body text (`\doublespacing`); single-spaced in acceptance sheet and code listings
- **Section numbering**: Roman numerals for sections (`\Roman{section}`), uppercase letters for subsections (`\Alph{subsection}`)
- **Each section starts on a new page** — `\section` is redefined to call `\newpage` automatically
- **Citations**: IEEE style via BibTeX; use `\cite{key}` and add entries to `biblio.bib`

## Chapter Outline

| File | Content |
|------|---------|
| `chapter1.tex` | Introduction (Background, Problem Statement, Objectives, Significance, Scope, Assumptions) |
| `chapter2.tex` | Review of Related Literature |
| `chapter3.tex` | (Methodology) |
| `chapter4.tex` – `chapter7.tex` | (Results, Discussion, etc.) |
| `chapter8.tex` | (Conclusion) |
| Appendix & Acknowledgment | Defined inline at end of `rodriguez_sp-manuscript.tex` |

## Including Source Code

Two methods are available (see end of `rodriguez_sp-manuscript.tex`):

```latex
% Inline listing
\begin{lstlisting}
// code here
\end{lstlisting}

% From file
\lstinputlisting{source-code/Node.java}
```

Code listings render in `\tiny` font with `\singlespacing` inside the Appendix section.
