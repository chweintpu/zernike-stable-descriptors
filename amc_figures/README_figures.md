# Figures: data and code

`make_figures.py` regenerates every figure of the manuscript and the Supplementary Material from the saved
results, with the labels used in the text, and writes the plotted numbers to `figure_data/`.
Place the folder `amc_figures` in the `Zernike` folder and press F5 in Spyder. Output: `Zernike/figures_final/`
(PNG at 300 dpi and PDF) and `Zernike/figures_final/figure_data/` (CSV).

| Figure | Source data (in the Zernike folder) | Produced by | Plotted data |
|---|---|---|---|
| Figure S8 | `results_amc_phase6/fourier_bessel_check.csv` | phase 6a | `FigureS8_fourier_bessel_check.csv` |
| Figure 3 | `results_amc_phase6/kN_surface.csv` | phase 6b | `Figure3_kN_surface_retrieval.csv` |
| Figure 4 | `results_amc_phase8/surface_val/surface_val_mean.csv` | phase 8b | `Figure4_classification_surface_val.csv` |
| Figure 5 | `results_amc_phase6/phase6_records.csv`, `results_amc_phase8/full/classification_comparisons.csv` | phases 6b, 8 | `Figure5_cross_task_gain.csv` |
| Figures 1, 2 and S7 | no data needed (schematic) | `amc_schematics/make_schematics.py` | output/FigA, FigB, FigC |
| Figures S1, S2 | `results_amc_phase9/stability_per_shape_MPEG-7.csv` | phase 9 | `FigureS1_S2_perturbations_summary.csv` |
| Figures S3, S4 | `results_amc_phase1/moments_all.npz`, `results_amc_phase1b/moments_extra.npz`, `results_amc_phase4/moments_phase4.npz` | phases 1, 1b, 4a | `FigureS3_error_by_order.csv`, `FigureS4_convergence_order.csv` |
| Figure S5 | `results_amc_phase4/moments_phase4.npz` | phase 4a | `FigureS5_rotation_by_order.csv` |
| Figure S6 | `results_amc_phase6/k_curve.csv` | phase 6b | `FigureS6_k_extended.csv` |

Figures 3-5, S1, S2, S6 and S8 in `figures_final/` were generated from the uploaded result files (seeds 3001-3010).
Figures S3-S5 need the moment files, which exist only on the local machine; run the script locally to produce them.

Changes with respect to the figures produced by the phase scripts: all labels use the names of the manuscript
(BOX-INS, BOX-CIR, CEN-MAX, CEN-AREA, CEN-RG, "baseline");
Figure 1 omits the order n = 1, whose moments vanish because of centroid centering; Figure S3 adds the
exact-coverage implementation (BOX-CIR), which appears in Table 2; plot titles were removed in favor of the captions.
