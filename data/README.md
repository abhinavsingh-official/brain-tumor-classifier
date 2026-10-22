# Dataset

This project trains on the public **Brain Tumor MRI Dataset** by Masoud Nickparvar on Kaggle:

https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

7,023 human brain MRI images across 4 classes (glioma, meningioma, no tumor, pituitary),
split into `Training/` (1,400 images/class) and `Testing/` (400 images/class — notumor has
slightly more, matching the published dataset).

The raw images are not committed to this repository (it's ~175MB of data and is freely
available on Kaggle). To retrain or run the demo's training script yourself:

1. Download the dataset from the Kaggle link above.
2. Extract it so you have:
   ```
   data/Training/{glioma,meningioma,notumor,pituitary}/*.jpg
   data/Testing/{glioma,meningioma,notumor,pituitary}/*.jpg
   ```
3. Run `python src/train.py --data-dir data --epochs 15`.
