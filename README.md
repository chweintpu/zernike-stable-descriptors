# Code for "Stable and Invariant Zernike Descriptors for Shape Analysis"

This repository reproduces all numerical results of the manuscript. The scripts are organized by
experimental phase. Each phase reads the outputs of earlier phases and writes to `results_amc_phase*/`.

## Requirements

Python 3.11, NumPy, SciPy, scikit-learn, Matplotlib, mpmath, Pillow. Kimia-216, Kimia-99 and ETHZ
are read through the preprocessing functions of the original experiment scripts, which additionally
need PyTorch, OpenCV and pandas. Optional: `psutil`, `threadpoolctl` (hardware information).
All scripts restrict BLAS to one thread per process and parallelize over processes.

## Data

Place the datasets under `data/` (MPEG-7 CE-Shape-1 Part B in `data/mpeg7`, Kimia-216, Kimia-99 and
the ETHZ Shape Classes as in the original scripts) and the cached frozen deep features in the
`feature_cache/` folders referenced in `phase5_data.py`.

## Run order and mapping to the manuscript

| Step | Script(s) | Produces | Manuscript |
|---|---|---|---|
| 0 | `amc_phase1/phase0_validate.py` | engine and cache checks | Section 4.1 |
| 1 | `amc_phase1/phase1_run.py`, `phase1_analyze.py` | native references, discretization errors | Table 1, Supplementary Figure S3 |
| 1b | `amc_phase1b3/phase1b.py` | convergence orders, ranking certificate | Table 1, Supplementary Table S7, Figure S4 |
| 3 | `amc_phase1b3/phase3_spectrum.py` | order spectrum of fused retrieval | Supplementary Section S2 |
| 4 | `amc_phase4/phase4a_moments.py`, `phase4b_select.py` | exact coverage, normalizations, rotation, weighting, angular analysis | Table 2, Supplementary Figure S5, Section S2 |
| 5 | `amc_phase5/phase5a_moments.py`, `phase5b_eval.py` | cross-dataset retrieval | Section 4.2 |
| 6 | `amc_phase6/phase6a_moments.py`, `phase6b_eval.py` | small-support and Fourier–Bessel asymptotics, extended scales, (k, N) surfaces | Table 5, Figure 3, Supplementary Tables S8, S9, Figures S6, S8 |
| 7 | `amc_phase7/phase7a_moments.py`, `phase7b_eval.py` | CEN-AREA, rotation | Tables 2, 4 |
| 7c | `amc_phase7/phase7c_numerics.py` | evaluation methods, run times, Richardson surrogate | Supplementary Tables S5, S7, S14 |
| 7d | `amc_phase7/phase7d_manufactured.py` | manufactured shapes | Supplementary Table S6 |
| 7e | `amc_phase7/phase7e_run.py` | CEN-AREA at large scales, clipping, batched timings | Section 4.2, Supplementary Table S5 |
| 7f | `amc_phase7/phase7f_fair.py` | symmetric CEN-RG / CEN-AREA comparison, hardware information | Sections 4.1, 4.2 |
| 8 | `amc_phase8/phase8_classification.py` (`PILOT = False`) | classification | Table 6, Supplementary Tables S1, S2, S10–S13, S15 |
| 9 | `amc_phase9/phase9_stability.py` | controlled perturbations | Section 4.1, Supplementary Table S3, Figures S1, S2 |
| 8b | `amc_phase8/phase8b_surface_val.py` | validation-only classification surface | Figure 4, Section 4.3 |

The amplification constants of Supplementary Table S4 are computed exactly by `cheb_coeffs` in
`amc_phase7/zernike_methods.py`. Figure 5 is drawn from the paired differences of Table 5 and Table 6.

Schematic Figures 1, 2 and Supplementary Figure S7 are drawn by `amc_schematics/make_schematics.py`, which needs no data.

## Core modules

- `zernike_amc.py`: moment engine (Chebyshev form), native-resolution reference quadrature,
  normalizations (BOX-INS, BOX-CIR, CEN-MAX, CEN-AREA, CEN-RG), exact area coverage (10).
- `zernike_methods.py`: factorial form, Kintner and q-recursive recurrences, batched Kintner (5)–(6)
  and batched Chebyshev (8) evaluation.
- `phase5_data.py`: dataset loading with the file order, labels and splits of the original protocol.

## Notes

- All split-dependent results use the seeds 3001–3010 defined in `phase5_data.py`, `phase3_worker.py` and `phase4b_worker.py`.

- File order is case-insensitive (Windows convention) so that moments align with cached features.
- All model selection uses validation data only. Test data are evaluated once per final model.

## Figures

`amc_figures/make_figures.py` redraws the data figures from the result files in `results_amc_phase*/`, and
`amc_schematics/make_schematics.py` draws the schematic figures without data. Output files map to the
manuscript as follows: Figure 1 = `FigA_normalization`, Figure 2 = `FigB_bandwidth`,
Figures 3–5 = `Figure3_*`–`Figure5_*`, Supplementary Figure S7 = `FigC_discretization`,
Supplementary Figures S1–S6 and S8 = `FigureS*_*`.

## Dataset modules and deep features

`dataset_scripts/` contains the dataset utilities used by `phase5_data.py`: file scanning, mask loading
and splitting for Kimia-216 (`data_kimia216.py`), Kimia-99 (`data_kimia99.py`) and the ETHZ Shape Classes
(`data_ethz.py`), together with the functions that extract the frozen deep features. For MPEG-7,
`data_mpeg7.py` provides the preprocessing, `features_mpeg7_backbones.py` and `features_mpeg7_dinov2.py`
extract the features of the six backbones, and `extract_mpeg7_features.py` recomputes all MPEG-7 caches. Paths are resolved
relative to the repository root, or to the folder given by the environment variable `ZERNIKE_BASE`.
The frozen deep features are read from the `feature_cache/` folders listed in `phase5_data.py`.
The datasets themselves are not redistributed. MPEG-7 CE-Shape-1 Part B, Kimia-216, Kimia-99 and the
ETHZ Shape Classes are available from their original providers.

## Result files

The `results_amc_phase*/` folders contain the summary result files (seeds 3001–3010) from which
`amc_figures/make_figures.py` redraws Figures 3–5 and Supplementary Figures S1, S2, S6 and S8 without
rerunning the experiments. Supplementary Figures S3–S5 additionally need the moment files of phases 1,
1b and 4, which are produced by the corresponding scripts and are not stored in the repository.

## License

MIT License, see `LICENSE`.

## Citation

C.-H. Wei, Stable and invariant Zernike descriptors for shape analysis: discretization error and
Fourier–Bessel spectral limits, submitted to Applied Mathematics and Computation.

