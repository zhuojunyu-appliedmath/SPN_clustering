This repository contains the analysis code and derived data used to generate figures for "Striatal state context multiplexes the role of basal ganglia pathways
during decision-making" [link]

## Quick start

Run Jupyter from the repository root.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter lab
```

Run the notebooks in numerical order. 
Notebooks 01-07 create standardized derived data tables; notebooks 04, 05, 06, and 08 generate the manuscript figures.

## Repository structure
```text
notebooks/              ordered analysis notebooks
spn_figures/            shared analysis and plotting functions
data/source/            source data and manifests
data/derived/           compact outputs created by notebooks
assets/static/          original task and circuit artwork
figures/main/           main-text figure outputs
figures/supporting/     Supporting Information figure outputs
README.md               repository documentation

## Figures in the Manuscript
Fig. 1A: schematic of mouse two-choice task

Fig. 1B: schematic of reduced CBGT circuit

Fig. 1C(i): simulated 12D SPN profiles | `01_cbgt_reference_clustering.ipynb` | `figures/main/Fig1C_i_cbgt_12D_profiles`

Fig. 1C(ii): fast/slow SPN correlations | `01_cbgt_reference_clustering.ipynb` | `figures/main/Fig1C_ii_cbgt_fast_slow_correlations`

Fig. 1D: two-stage clustering schematic and validation | `01_cbgt_reference_clustering.ipynb` | `figures/main/Fig1D_two_stage_clustering`

Fig. 2: example empirical SPN profiles | `04_spn_clustering_figures.ipynb` | `figures/main/Fig2_example_empirical_spn_profiles`

Fig. 3A: CBGT CLAW | `05_claw_and_control_ensemble_figures.ipynb` | `figures/main/Fig3A_cbgt_claw`

Fig. 3B: IBL CLAW | `05_claw_and_control_ensemble_figures.ipynb` | `figures/main/Fig3B_ibl_claw`

Fig. 4A: schematic of control ensembles 

Fig. 4B: changes in control ensemble engagement along the CBGT CLAW | `05_claw_and_control_ensemble_figures.ipynb` | `figures/main/Fig4B_control_ensemble_transitions`

Fig. 5: model and empirical prediction tests | `08_prediction_boxplots.ipynb` | `figures/main/Fig5_multiplexed_iSPN_roles`

## Figures in the Supporting Information 
Fig. S1A: schematic of full CBGT circuit 

Fig. S1B: firing rates of CBGT populations 

Fig. S1C: median decision time distributions of 300 CBGT networks

Fig. S1D: SPN thresholds (Fig. S1A-D reproduced from https://doi.org/10.1371/journal.pcbi.1012966)

Fig. S2: Steinmetz ISI-shuffle control | `06_steinmetz_isi_shuffle.ipynb` | `figures/supporting/FigS2_isi_shuffle_control`

Fig. S3: all Steinmetz clustering results | `04_spn_clustering_figures.ipynb` | `figures/supporting/FigS3_steinmetz_profiles_and_correlations`

Fig. S4: all IBL clustering results | `04_spn_clustering_figures.ipynb` | `figures/supporting/FigS4_ibl_profiles_and_correlations`

Fig. S5: schematic of control ensemble in the full CBGT circuit, reproduced from https://doi.org/10.64898/2026.02.17.706272

Fig. S6: Steinmetz CLAW | `05_claw_and_control_ensemble_figures.ipynb` | `figures/supporting/FigS6_steinmetz_claw`

Fig. S7: changes in drift rate and boundary height along the CBGT CLAW | `05_claw_and_control_ensemble_figures.ipynb` | `figures/supporting/FigS7_ddm_parameter_transitions`


## Notebook workflow

### 01. CBGT reference patterns and validation

`01_cbgt_reference_clustering.ipynb` reads the 300 simulated networks, constructs the 12-dimensional pre-decision features and full pre-decision firing rate table, validates the two-stage clustering against known pathway/channel labels, prepares the simulated CLAW state table, and generates Fig. 1C-D.

### 02-03. Empirical SPN inference

`02_steinmetz_spn_clustering.ipynb` and `03_ibl_spn_clustering.ipynb` apply the same inference pipeline to the Steinmetz and IBL recordings. 

Both notebooks save four standardized outputs:

- `unit_labels.csv`
- `unit_profiles.csv.gz`
- `correlations.csv`
- `population_activity_bins.csv.gz`

### 04. Empirical clustering figures

`04_spn_clustering_figures.ipynb` uses only the derived files from notebooks 02-03. 
It generates Fig. 2 and SI Figs. S3-S4.

### 05. CLAWs and control ensembles

`05_claw_and_control_ensemble_figures.ipynb` binarizes the four inferred SPN populations, compresses consecutive repetitions of the same state, estimates transition and terminal probabilities, and generates Fig. 3 and SI Fig. S6. 
It then projects CBGT state transitions onto the choice, responsiveness, and pliancy ensembles to generate Fig. 4B and SI Fig. S7.

### 06. ISI-shuffle control

`06_steinmetz_isi_shuffle.ipynb` permutes each unit’s interspike intervals, reconstructs surrogate spike trains, reruns the full clustering procedure 50 times per recording, and reports the fraction of shuffles satisfying the prespecified SPN temporal-pattern criteria. It generates SI Fig. S2.

### 07-08. Prediction analyses and boxplots

`07_prediction_statistics.ipynb` performs statistical analyses for three predictions:

1. Left-choice probability in iSPN-only versus dSPN-containing states.
2. Terminal probability before and after same-channel dSPN+iSPN coactivation.
3. Decision time with and without later opponent-channel iSPN recruitment.

`08_prediction_boxplots.ipynb` generates Fig. 5 from the standardized bootstrap, raw decision time, and significance test tables.

## Data layout

See `data/README.md` for the exact source and derived file names, required columns, and upload instructions.
