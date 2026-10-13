# data/ (contents gitignored)

```
data/
  raw/<source>/<class folder>/...   # you place datasets here (jmuben, uganda, bracol, rocole)
  metadata.csv                      # 02_build_metadata.py, updated by 03_cluster_split.py
  processed/<split>/<label>/*.jpg   # 04_make_processed.py (+ processed/metadata.csv)
  coffee_leaf_kaggle.zip            # 04_make_processed.py, upload this to Kaggle/Colab
```

Class folder names are mapped to the 5 classes through `label_aliases` in `configs/config.yaml`.
Datasets that ship as one image folder plus a label CSV/XLSX (e.g. RoCoLe) must first be
sorted into class folders. See the README.
