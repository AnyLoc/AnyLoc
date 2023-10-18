# Jointly project global descriptors from multiple datasets
"""
    Get global database and query descriptors from multiple datasets.
    Say there are 'n' datasets.
    The database tensors are of shape:
    - db-ds1: N1_db, D_in
    - db-ds2: N2_db, D_in
    ...
    - db-dsn: Nn_db, D_in
    
    We have similarly, the queries
    - qu-ds1: N1_qu, D_in
    - qu-ds2: N2_qu, D_in
    ...
    ...
    - qu-dsn: Nn_qu, D_in
    
    All these datasets belong to the same "domain" (general setting).
    This program stacks the database tensors along the first dim,
    then projects it to lower dim, and then applies the same 
    projection to all query descriptors (PCA is fit only on database)
    accordingly.
    Basically, the process is
    - db-all_stacked: (N1_db + N2_db + ... + Nn_db), D_in
    - db-all_pca: (N1_db + N2_db + ... + Nn_db), D_out <- fit and tf
    - qu-all_stacked: (N1_qu + N2_qu + ... + Nn_qu), D_in
    - qu-all_pca: (N1_qu + N2_qu + ... + Nn_qu), D_out <- only tf
    
    Then unstack and save in corresponding file names
"""

if __name__ != "__main__":
    print("Please run this script directly. It cannot be imported.")
    exit(1)

# %%
import os
import numpy as np
import torch
from sklearn.decomposition import PCA


# %%
# Cache file directory (where global descriptors are stored)
exp_dir = "/scratch/avneesh.mishra/vl-vpr/cache/experiments/pca_downsample"
gd_stored_dir = f"{exp_dir}/vlad_descs"
# Dataset names (files '[db,qu]-ds_name.pt' in stored directory)
# ds_names = ["17places", "baidu_datasets", "gardens"]    # Indoor
ds_names = ["Oxford_25m", "pitts30k", "st_lucia"]   # Urban
# PCA settings
pca_lower_dim = 512
pca_whiten = True
# Output directory (same files, lower dimensionality)
pca_out_dir = f"{exp_dir}/pca_{pca_lower_dim}"

# Input validation
assert os.path.isdir(gd_stored_dir), f"NotFound: {gd_stored_dir = }"
if os.path.isdir(pca_out_dir):
    print(f"Directory already exists: {pca_out_dir}")
else:
    os.makedirs(pca_out_dir)
    print(f"Created directory: {pca_out_dir}")


# %%
# Read all tensors into numpy array
all_db_gds = []
all_qu_gds = []
for ds_name in ds_names:
    db_gd_file = f"{gd_stored_dir}/db-{ds_name}.pt"
    qu_gd_file = f"{gd_stored_dir}/qu-{ds_name}.pt"
    assert os.path.isfile(db_gd_file), f"NotFound: {db_gd_file = }"
    assert os.path.isfile(qu_gd_file), f"NotFound: {qu_gd_file = }"
    db_gd = torch.load(db_gd_file)
    qu_gd = torch.load(qu_gd_file)
    all_db_gds.append(db_gd.numpy())
    all_qu_gds.append(qu_gd.numpy())
# All descriptors
db_descs = np.concatenate(all_db_gds, axis=0)
qu_descs = np.concatenate(all_qu_gds, axis=0)

# %%
# PCA projection
pca = PCA(n_components=pca_lower_dim, whiten=pca_whiten)
down_db_descs = pca.fit_transform(db_descs)
down_qu_descs = pca.transform(qu_descs)

# %%
# Split into dataset-wise
db_shapes = [k.shape[0] for k in all_db_gds]
qu_shapes = [k.shape[0] for k in all_qu_gds]
saved_db = 0
saved_qu = 0
for ds_idx, ds_name in enumerate(ds_names):
    db_gd = down_db_descs[saved_db:saved_db+db_shapes[ds_idx]]
    saved_db += db_shapes[ds_idx]
    qu_gd = down_qu_descs[saved_qu:saved_qu+qu_shapes[ds_idx]]
    saved_qu += qu_shapes[ds_idx]
    print(f"{ds_name = }, {db_gd.shape = }, {qu_gd.shape = }")
    db_gd = torch.from_numpy(db_gd)
    qu_gd = torch.from_numpy(qu_gd)
    torch.save(db_gd, f"{pca_out_dir}/db-{ds_name}.pt")
    torch.save(qu_gd, f"{pca_out_dir}/qu-{ds_name}.pt")
print("....Program completed....")

# %%
