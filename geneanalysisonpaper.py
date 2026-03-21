import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import anndata
from sklearn.decomposition import PCA
import os

# Step 0: Instructions for downloading data
# Breast cancer data: GSE176078 from GEO[](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE176078)
# Download the supplementary files, which include count matrices in .tar format. Extract to a folder, e.g., 'breast_data/'
# Often, it's in 10X format: barcodes.tsv.gz, features.tsv.gz, matrix.mtx.gz per sample.

# Lung cancer data: GSE131907 from GEO[](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE131907)
# Similarly, download and extract count matrices to 'lung_data/'

# For multi-sample datasets, you may need to load each sample and concatenate.
# Here, assume single combined h5ad files for simplicity. In practice, use sc.read_10x_mtx for each and merge.

# Placeholder: Assume data is loaded as AnnData
# In real use, replace with:
# breast_adata = sc.read_10x_mtx('breast_data/sample1/')  # Repeat for all samples and concatenate
# Or if processed h5ad available from portals.

# For demonstration, use placeholder data (remove in real analysis)
breast_adata = sc.datasets.pbmc3k_processed()  # Placeholder for breast
lung_adata = sc.datasets.pbmc3k_processed()    # Placeholder for lung
breast_adata.obs['cancer_type'] = 'breast'
lung_adata.obs['cancer_type'] = 'lung'

# Actual loading example (commented):
# import tarfile
# with tarfile.open('GSE176078_RAW.tar', 'r') as tar:
#     tar.extractall('breast_data/')
# Then load each sample.

# Step 1: Explore and clean the data
def explore_and_clean(adata, cancer_type):
    print(f"Exploring {cancer_type} data:")
    print(adata)  # Overview
    sc.pl.highest_expr_genes(adata, n_top=20, save=f'_{cancer_type}_top_genes.png')  # Top expressed genes
    
    # Quality control
    sc.pp.calculate_qc_metrics(adata, percent_top=None, log1p=False, inplace=True)
    sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True, save=f'_{cancer_type}_qc.png')
    
    # Filtering
    adata = adata[adata.obs.n_genes_by_counts < 2500, :]  # Example thresholds; adjust based on data
    adata = adata[adata.obs.n_genes_by_counts > 200, :]
    adata = adata[adata.obs.pct_counts_mt < 5, :]
    
    # Normalize and log-transform
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Highly variable genes
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
    adata = adata[:, adata.var.highly_variable]
    
    # Scale
    sc.pp.scale(adata, max_value=10)
    
    return adata

breast_adata = explore_and_clean(breast_adata, 'breast')
lung_adata = explore_and_clean(lung_adata, 'lung')

# Step 2: Visualize gene-expression patterns
def visualize_patterns(adata, cancer_type):
    # PCA
    sc.tl.pca(adata, svd_solver='arpack')
    sc.pl.pca(adata, color='cancer_type', save=f'_{cancer_type}.png')
    
    # UMAP
    sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    sc.tl.umap(adata)
    sc.pl.umap(adata, color=['leiden', 'cancer_type'], save=f'_{cancer_type}.png')  # Assuming clustering done
    
    # Cluster
    sc.tl.leiden(adata)
    
    # Heatmap of top genes per cluster
    sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
    sc.pl.rank_genes_groups_heatmap(adata, n_genes=5, save=f'_{cancer_type}.png')

visualize_patterns(breast_adata, 'breast')
visualize_patterns(lung_adata, 'lung')

# Step 3: Compare differences between the two cancer types
# Integrate for comparison
adata_combined = anndata.concat([breast_adata, lung_adata], label='cancer_type', keys=['breast', 'lung'])

# Batch correction (e.g., Harmony)
import harmonypy as hm
sc.tl.pca(adata_combined)
ho = hm.run_harmony(adata_combined.obsm['X_pca'], adata_combined.obs, ['cancer_type'])
adata_combined.obsm['X_pca_harmony'] = ho.Z_corr.T
sc.pp.neighbors(adata_combined, use_rep='X_pca_harmony')
sc.tl.umap(adata_combined)

sc.pl.umap(adata_combined, color='cancer_type', save='_combined.png')

# Differential expression
sc.tl.rank_genes_groups(adata_combined, 'cancer_type', method='wilcoxon')
sc.pl.rank_genes_groups(adata_combined, n_genes=25, sharey=False, save='_de.png')

# Cell type proportions
proportions = pd.crosstab(adata_combined.obs['cancer_type'], adata_combined.obs['leiden'], normalize='index')
proportions.plot(kind='bar', stacked=True)
plt.title('Cell Type Proportions: Breast vs Lung')
plt.savefig('proportions.png')

# Pathway enrichment (example with top DE genes)
top_genes = sc.get.rank_genes_groups_df(adata_combined, group='breast')['names'].head(100).tolist()
import gseapy as gp
enr = gp.enrichr(gene_list=top_genes, gene_sets='KEGG_2021_Human', outdir='pathways')
enr.results.head(10).to_csv('pathways_breast_vs_lung.csv')

# Step 4: Draw conclusions (interpret results manually or print insights)
# Example: Based on DE genes and pathways, differences in immune response genes might indicate varying immunotherapy responses.
# Breast cancer often shows higher estrogen signaling, while lung may have more smoking-related mutations affecting pathways.
# Print top DE genes and enriched pathways for interpretation.
print("Top DE genes (breast higher):")
print(sc.get.rank_genes_groups_df(adata_combined, group='breast')['names'].head(10))
print("\nEnriched pathways:")
print(enr.results[['Term', 'Adjusted P-value']].head(10))

# Conclusions:
# - If immune cells differ, it might mean different tumor microenvironments, affecting treatment.
# - Gene expression differences could highlight unique biomarkers for diagnosis.
# - For medicine: Targeted therapies based on specific pathways (e.g., EGFR in lung vs HER2 in breast).

print("Analysis complete. Check saved figures and CSV files for details.")