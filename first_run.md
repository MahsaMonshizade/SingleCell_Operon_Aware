/home/mmonshiz/micromamba/envs/scvi_env/lib/python3.8/site-packages/setuptools_scm/_integration/setuptools.py:31: RuntimeWarning: 
ERROR: setuptools==59.5.0 is used in combination with setuptools_scm>=8.x

Your build configuration is incomplete and previously worked by accident!
setuptools_scm requires setuptools>=61

Suggested workaround if applicable:
 - migrating from the deprecated setup_requires mechanism to pep517/518
   and using a pyproject.toml to declare build dependencies
   which are reliably pre-installed before running the build tools

  warnings.warn(
==================================================
Training OperonAware SCVI
==================================================
INFO     No batch_key inputted, assuming all cells are same batch               
INFO     No label_key inputted, assuming all cells have same label              
INFO     Using data from adata.X                                                
INFO     Computing library size prior per batch                                 
INFO     Successfully registered anndata object containing 57627 cells, 3070    
         vars, 1 batches, 1 labels, and 0 proteins. Also registered 0 extra     
         categorical covariates and 0 extra continuous covariates.              
INFO     Please do not further modify adata until model is trained.             
  Cells: 57,627  |  Genes: 3,070
[load_neighbor_indices] Warning: dropping 808 pairs with genes not in adata.var_names
[load_neighbor_indices] Loaded 1323 operon neighbor pairs
  Operon pairs: 1,323
  Lambda:       1.0
SCVI Model with the following params: 
n_hidden: 64, n_latent: 5, n_layers: 2, dropout_rate: 0.1, dispersion: gene, 
gene_likelihood: zinb, latent_distribution: normal
Training status: Not Trained


To print summary of associated AnnData, use: 
scvi.data.view_anndata_setup(model.adata)



Training...
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1,2,3]
Epoch 1/139:   0%|                                      | 0/139 [00:00<?, ?it/s]/home/mmonshiz/micromamba/envs/scvi_env/lib/python3.8/site-packages/pytorch_lightning/trainer/callback_hook.py:100: LightningDeprecationWarning: The signature of `Callback.on_train_epoch_end` has changed in v1.3. `outputs` parameter has been removed. Support for the old signature will be removed in v1.5
  warning_cache.deprecation(
Epoch 139/139: 100%|███████| 139/139 [07:25<00:00,  3.21s/it, loss=375, v_num=1]

Saved → pountain_data/outputs/lb_scVI_model_operon_aware



/home/mmonshiz/micromamba/envs/scvi_env/lib/python3.8/site-packages/setuptools_scm/_integration/setuptools.py:31: RuntimeWarning: 
ERROR: setuptools==59.5.0 is used in combination with setuptools_scm>=8.x

Your build configuration is incomplete and previously worked by accident!
setuptools_scm requires setuptools>=61

Suggested workaround if applicable:
 - migrating from the deprecated setup_requires mechanism to pep517/518
   and using a pyproject.toml to declare build dependencies
   which are reliably pre-installed before running the build tools

  warnings.warn(
==================================================
Loading data, models, and RegulonDB...
INFO     No batch_key inputted, assuming all cells are same batch               
INFO     No label_key inputted, assuming all cells have same label              
INFO     Using data from adata.X                                                
INFO     Computing library size prior per batch                                 
INFO     Successfully registered anndata object containing 57627 cells, 3070    
         vars, 1 batches, 1 labels, and 0 proteins. Also registered 0 extra     
         categorical covariates and 0 extra continuous covariates.              
INFO     Please do not further modify adata until model is trained.             
INFO     Using data from adata.X                                                
INFO     Computing library size prior per batch                                 
INFO     Registered keys:['X', 'batch_indices', 'local_l_mean', 'local_l_var',  
         'labels']                                                              
INFO     Successfully registered anndata object containing 57627 cells, 3070    
         vars, 1 batches, 1 labels, and 0 proteins. Also registered 0 extra     
         categorical covariates and 0 extra continuous covariates.              
INFO     Using data from adata.X                                                
INFO     Computing library size prior per batch                                 
INFO     Registered keys:['X', 'batch_indices', 'local_l_mean', 'local_l_var',  
         'labels']                                                              
INFO     Successfully registered anndata object containing 57627 cells, 3070    
         vars, 1 batches, 1 labels, and 0 proteins. Also registered 0 extra     
         categorical covariates and 0 extra continuous covariates.              
  Cells: 57,627  |  Genes: 3,070
[gene_mapping] Parsed 4464 symbol→blattner pairs from GFF
[gene_mapping] 3070/4464 symbols mapped to adata (1394 not found — likely filtered in preprocessing)
[operon_utils] Parsed 848 multi-gene operons from RegulonDB
Weak      787
Strong     61

[operon_utils] 560 operons with ≥2 genes in adata (2368 total gene pairs)
[operon_utils] Coverage by confidence level:
  Confirmed: 0/0 operons have ≥2 genes in adata
  Strong   : 46/61 operons have ≥2 genes in adata
  Weak     : 514/787 operons have ≥2 genes in adata

Computing denoised expression...
INFO     No batch_key inputted, assuming all cells are same batch               
INFO     No label_key inputted, assuming all cells have same label              
INFO     Using data from adata.X                                                
INFO     Computing library size prior per batch                                 
INFO     Successfully registered anndata object containing 3000 cells, 3070     
         vars, 1 batches, 1 labels, and 0 proteins. Also registered 0 extra     
         categorical covariates and 0 extra continuous covariates.              
INFO     Please do not further modify adata until model is trained.             
Computing per-operon intra-operon correlations...
  Raw: 560 operons | NaN std: 0 | NaN op: 0

  Overall (560 operons):
    Standard : 0.3380 ± 0.2280
    Operon   : 0.3431 ± 0.2273
    Δ        : +0.0051  (Mann-Whitney p=3.616e-01)
    % improved: 50.7%

  By confidence level:
    [Strong   ] n=  46 | Std=0.3367  Op=0.3622  Δ=+0.0255  (50% improved)
    [Weak     ] n= 514 | Std=0.3381  Op=0.3414  Δ=+0.0033  (51% improved)

  ── Strong operons only (n=46) ──
    Δ = +0.0255  |  50.0% improved

Computing intra- vs inter-operon contrast...
  Intra | Std=0.3622  Op=0.3642
  Inter | Std=0.0461  Op=0.0543
  Δr    | Std=0.3162  Op=0.3099  (✗ standard)

Computing within-operon gene similarity...
  Std=0.3435  Op=0.3461  (✓ operon)

Top 10 most improved operons:
   operon_name  n_genes confidence  r_standard  r_operon    delta
     hydN-hypF        2       Weak   -0.191758  0.177218 0.368976
paaABCDEFGHIJK        2       Weak   -0.124656  0.239580 0.364236
      rph-pyrE        2       Weak   -0.063952  0.285048 0.349000
xanQ-guaD-ghxQ        2       Weak   -0.308237  0.028037 0.336274
         tehAB        2       Weak   -0.261288  0.058150 0.319438
         fadBA        2       Weak    0.077166  0.391798 0.314632
     clsA-yciU        2       Weak    0.179367  0.464819 0.285452
    relBE-hokD        2       Weak    0.068338  0.334996 0.266658
         kdpDE        2     Strong    0.252137  0.502938 0.250801
     pldB-yigL        2       Weak    0.178674  0.421389 0.242715

Top 10 where standard was better:
 operon_name  n_genes confidence  r_standard  r_operon     delta
   acs-yjcHG        2       Weak    0.256247 -0.040954 -0.297201
       gntXY        2       Weak    0.353475  0.116027 -0.237448
   serC-aroA        2       Weak    0.759524  0.524877 -0.234647
   mhpR-lacI        2       Weak    0.302871  0.085066 -0.217805
   ratA-yfjF        2       Weak    0.325124  0.107908 -0.217215
       dsdXA        2       Weak    0.094406 -0.117793 -0.212199
garPLRK-rnpB        2       Weak   -0.113602 -0.320225 -0.206624
   talA-tktB        2       Weak    0.846002  0.640925 -0.205078
   nadA-pnuC        2       Weak    0.697684  0.495997 -0.201686
       yccFS        2       Weak    0.192828 -0.008775 -0.201602

Saved → pountain_data/outputs/regulondb_operon_evaluation.png
Saved → pountain_data/outputs/per_operon_results.csv

==================================================
  ✓ Intra-operon correlation: Operon wins
  ✗ Intra-inter contrast: Standard wins
  ✓ % operons improved (>50%): Operon wins
  ✓ Within-operon gene similarity: Operon wins

  Operon SCVI wins 3/4 RegulonDB metrics
==================================================