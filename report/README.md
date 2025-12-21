# Research Report

## Final Report: Robust Fingerprinting for Language Model Lineage Detection

This directory contains the LaTeX source for the semester research report.

### Files

- `final_report.tex`: Main report document
- `references.bib`: Bibliography with all cited papers
- `neurips_2019.sty`: NeurIPS 2019 style file (formatting)
- `template.tex`: Original template (for reference)

### Compiling the Report

#### Option 1: Using pdflatex (recommended)

```bash
cd report
pdflatex final_report.tex
bibtex final_report
pdflatex final_report.tex
pdflatex final_report.tex
```

This will generate `final_report.pdf`.

#### Option 2: Using Overleaf

1. Create a new project on [Overleaf](https://www.overleaf.com)
2. Upload all files: `final_report.tex`, `references.bib`, `neurips_2019.sty`
3. Set the main document to `final_report.tex`
4. Click "Recompile"

#### Option 3: Using latexmk (automatic)

```bash
latexmk -pdf final_report.tex
```

### Report Structure

1. **Abstract**: Overview of the research and key findings
2. **Introduction**: Motivation and research questions
3. **Background**: Related work (RoFL, watermarking)
4. **Methodology**: 
   - Approach 1: RoFL-style fingerprinting
   - Approach 2: Bottom-k vocabulary subspace
5. **Experimental Setup**: Models, datasets, timeline
6. **Results**: 
   - RoFL-style results
   - Bottom-k vocabulary results
   - Comparison of approaches
7. **Discussion**: Why it works, limitations, practical implications
8. **Future Work**: Short-term and long-term directions
9. **Conclusion**: Summary of contributions

### Key Findings Summary

- **RoFL-style fingerprinting** achieves 0.74 score for same-lineage vs 0.15 for different-lineage (gap: 0.59)
- **Bottom-k vocabulary overlap** shows 68% overlap for same-lineage vs 22% for different-lineage
- **Constrained generation** achieves 0.78 score for same-lineage vs 0.20 for different-lineage (gap: 0.58)
- Optimal parameters: 20 fingerprints, k=1000-2000

### Figures and Tables

The report includes 7 tables summarizing experimental results:
- Table 1: Overall RoFL-style results
- Table 2: Effect of fingerprint count
- Table 3: Effect of bottom-k size
- Table 4: Vocabulary overlap results
- Table 5: Constrained generation results
- Table 6: Comparison of approaches
- Additional tables for hyperparameter analysis

### Notes

- The report is ~15 pages (excluding references)
- All experiments are documented with parameters and results
- Timeline shows the evolution of the research from September to December
- Includes discussion of limitations and future work

### Contact

For questions about the report, contact: jl6647@columbia.edu


