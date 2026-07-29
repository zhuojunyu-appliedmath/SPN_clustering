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
Notebooks 01-07 create standardized derived data tables; notebooks 04, 05, 06, 08, and 09 generate the manuscript figures.

## Figures in the Manuscript
Fig. 1A: schematic of mouse two-choice task

Fig. 1B: schematic of reduced CBGT circuit

Fig. 1C(i): simulated 12D SPN profiles | 01_cbgt_reference_clustering.ipynb | figures/main/Fig1C_i_cbgt_12D_profiles

Fig. 1C(ii): fast/slow SPN correlations | 01_cbgt_reference_clustering.ipynb | figures/main/Fig1C_ii_cbgt_fast_slow_correlations

Fig. 1D: two-stage clustering schematic and validation | 01_cbgt_reference_clustering.ipynb | figures/main/Fig1D_two_stage_clustering

Fig. 2: example empirical SPN profiles | 04_spn_clustering_figures.ipynb | figures/main/Fig2_example_empirical_spn_profiles

Fig. 3A: CBGT CLAW | 05_claw_and_control_ensemble_figures.ipynb | figures/main/Fig3A_cbgt_claw

Fig. 3B: IBL CLAW | 05_claw_and_control_ensemble_figures.ipynb | figures/main/Fig3B_ibl_claw

Fig. 4A: schematic of control ensembles 

Fig. 4B: changes in control ensemble engagement along the CBGT CLAW | 05_claw_and_control_ensemble_figures.ipynb | figures/main/Fig4B_control_ensemble_transitions

Fig. 5: model and empirical prediction tests | 08_prediction_boxplots.ipynb | figures/main/Fig5_multiplexed_iSPN_roles

## Figures in the Supporting Information 
Fig. S1A: schematic of full CBGT circuit 

Fig. S1B: firing rates of CBGT populations 

Fig. S1C: median decision time distributions of 300 CBGT networks

Fig. S1D: SPN thresholds (Fig. S1A-D reproduced from https://doi.org/10.1371/journal.pcbi.1012966)

Fig. S2: Steinmetz ISI-shuffle control | 06_steinmetz_isi_shuffle.ipynb | figures/supporting/FigS2_isi_shuffle_control

Fig. S3: all Steinmetz clustering results | 04_spn_clustering_figures.ipynb | figures/supporting/FigS3_steinmetz_profiles_and_correlations

Fig. S4: all IBL clustering results | 04_spn_clustering_figures.ipynb | figures/supporting/FigS4_ibl_profiles_and_correlations

Fig. S5: schematic of control ensemble in the full CBGT circuit, reproduced from https://doi.org/10.64898/2026.02.17.706272

Fig. S6: Steinmetz CLAW | 05_claw_and_control_ensemble_figures.ipynb | figures/supporting/FigS6_steinmetz_claw

Fig. S7: changes in drift rate and boundary height along the CBGT CLAW | 05_claw_and_control_ensemble_figures.ipynb | figures/supporting/FigS7_ddm_parameter_transitions
