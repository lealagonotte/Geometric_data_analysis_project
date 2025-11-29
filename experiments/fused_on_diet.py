import sys
import os
import numpy as np
from sklearn.decomposition import PCA
import scanpy as sc
from scanpy.preprocessing import subsample
import anndata
import matplotlib.pyplot as plt
import ot

# Import data
rna_diet = sc.read_h5ad("RNA_scaled_diet_conditions_20240411.h5ad")
print(rna_diet)
prot_diet = sc.read_h5ad("protein_intensities_diet_conditions_20240411.h5ad")
print(prot_diet)

# code

GENES = list(rna_diet.var.index)
PROTEINS = list(prot_diet.var.index)
CONDITIONS = list(np.unique(rna_diet.obs['cond']))

print(f"Number of Genes: {len(GENES)}")
print(f"Number of Proteins: {len(PROTEINS)}")
print(f"Number of conditions: {len(CONDITIONS)} ({CONDITIONS})")

print("Maximum expression", np.max(rna_diet.X))
print("Minimum expression", np.min(rna_diet.X))

print("Maximum intensity", np.max(prot_diet.X))
print("Minimum intensity", np.min(prot_diet.X))

print("There are nans:", np.isnan(prot_diet.X).sum())

nan_mask_rowise = np.where(np.isnan(prot_diet.X).sum(axis=1) > 0, True, False) 
print(f"These nans come from {nan_mask_rowise.sum()} rows")
prot_diet.X[nan_mask_rowise,:]

prot_diet_noNA = prot_diet[~nan_mask_rowise,:]
print("Maximum intensity", np.max(prot_diet_noNA.X))
print("Minimum intensity", np.min(prot_diet_noNA.X))

prot_diet.obs["cond"].value_counts() / prot_diet.shape[0]

rna_diet.obs["cond"].value_counts() / rna_diet.shape[0]

def condition_subsample(adata, frac, cond_col):
    all_conditions = np.unique(adata.obs[cond_col])
    adata_list = []
    for cond in all_conditions:
        cond_adata = adata[adata.obs[cond_col] == cond]
        adata_list.append(subsample(cond_adata,frac,copy=True))
    return anndata.concat(adata_list, axis=0)

FRAC = 0.001

sub_rna_diet = condition_subsample(rna_diet, FRAC, 'cond')
sub_rna_diet

sub_prot_diet = condition_subsample(prot_diet_noNA, FRAC, 'cond') #sample just on non NA
sub_prot_diet

def pca_data(adata):
    # Computing reduced dimension
    print("Reducing dimension..")
    sc.tl.pca(adata)
    print("Plotting PCA Variance ratios..")
    sc.pl.pca_variance_ratio(adata, n_pcs=15, log=False)

pca_data(sub_rna_diet)
pca_data(sub_prot_diet)

RNA = {i:sub_rna_diet[sub_rna_diet.obs['cond'] == cond].X for i,cond in enumerate(CONDITIONS)}
RNA_reduced = {i:sub_rna_diet[sub_rna_diet.obs['cond'] == cond].obsm['X_pca'] for i,cond in enumerate(CONDITIONS)}
PROT = {i:sub_prot_diet[sub_prot_diet.obs['cond'] == cond].X for i,cond in enumerate(CONDITIONS)} # no pca used since only 18 features
PROT_reduced = {i:sub_prot_diet[sub_prot_diet.obs['cond'] == cond].obsm['X_pca'] for i,cond in enumerate(CONDITIONS)}


X_list, Y_list, X_labels_list, Y_labels_list = [], [], [], []
Blocs = [] #register each category for plotting
for i, cond in enumerate(CONDITIONS):
    X_cond = RNA_reduced[i]            # complete RNA for cond i
    Y_cond = PROT_reduced[i]           # complete PROT for cond i
    X_n_cells = X_cond.shape[0]
    Y_n_cells = Y_cond.shape[0]

    X_list.append(X_cond)
    Y_list.append(Y_cond)
    X_labels_list.extend([i] * X_n_cells)  # label = index of condition
    Y_labels_list.extend([i] * Y_n_cells)
    Blocs.append((X_n_cells, Y_n_cells))

X_cum = np.cumsum([0] + [b[0] for b in Blocs])
Y_cum = np.cumsum([0] + [b[1] for b in Blocs])

X_full = np.vstack(X_list)
Y_full = np.vstack(Y_list)

C1 = ot.dist(X_full, X_full, metric='euclidean')**2
C2 = ot.dist(Y_full, Y_full, metric='euclidean')**2

# Normalisation pour plus de stabilité
C1 = C1 / C1.max()
C2 = C2 / C2.max()

M = (np.array(X_labels_list)[:, None] != np.array(Y_labels_list)[None, :]).astype(int)

print("C1 :", C1.shape, C1)
print("C2 :", C2.shape, C2)
print("M :", M.shape, M)

epsilon = 0.01
alpha = 0.07
max_iter = 500

print("Computing transport plan for Fused Gromov-Wasserstein")
T = ot.gromov.entropic_fused_gromov_wasserstein(
    M, C1, C2, alpha=alpha, epsilon=epsilon, max_iter=max_iter, verbose=True
    )

print(f"Transport plan T (shape): {T.shape}")
print(T)
import matplotlib.pyplot as plt
plt.imshow(T, cmap='viridis')
plt.colorbar()
plt.title(f" Total Coupling, alpha= {alpha}, epsilon = {epsilon}")
plt.show()
plt.imshow(T[X_cum[0]:X_cum[1], Y_cum[0]:Y_cum[1]], cmap='viridis')
plt.colorbar()
plt.title(f" Coupling, first label, alpha= {alpha}, epsilon = {epsilon}")
plt.show()
plt.imshow(T[X_cum[1]:X_cum[2], Y_cum[1]:Y_cum[2]], cmap='viridis')
plt.colorbar()
plt.title(f" Total Coupling, second label, alpha= {alpha}, epsilon = {epsilon}")
plt.show()
plt.imshow(T[X_cum[2]:, Y_cum[2]:], cmap='viridis')
plt.colorbar()
plt.title(f" Total Coupling, third label, alpha= {alpha}, epsilon = {epsilon}")
plt.show()

