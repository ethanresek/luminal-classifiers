# How Many Genes Does It Take? Minimal gene expression classifiers for distinguishing Luminal A and Luminal B breast cancer

## Project Abstract
Breast cancer is a leading cause of death among women, and finding accurate diagnoses quickly is important for creating treatment plans. This is particularly challenging for Luminal A and Luminal B subtypes, which are both ER-positive and HER2-negative by standard clinical testing and therefore cannot be reliably distinguished without molecular profiling. The PAM50 assay addresses this by measuring the activity of 50 genes, but the question of how many genes are actually necessary remains open. Our project asks: can we select a smaller set of genes while still reliably classifying the cancer subtype? Our project uses the METABRIC dataset, which contains 331 gene expression features from 1,140 patients that will be used as feature dimensions and samples, respectively. We will train three different classifier models: a support vector machine with RBF kernel, a Random Forest model, and a deep neural network across five gene panel sizes (5, 10, 20, 50, 331 genes) from the dataset. We will use these models to classify a set of training samples into the two subtypes and compare the three models’ effectiveness in their classification.

## Dataset
We use the METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset, publicly available on [Kaggle](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric/) and [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric). METABRIC is a large breast cancer study from Canada and the UK that followed 1,904 patients for an average of about 10 years. After keeping only Luminal A and Luminal B patients, the working dataset is:
- 1,140 patients: 679 Luminal A (59.6%) and 461 Luminal B (40.4%)
- 331 gene expression features, each measuring how active a cancer-relevant gene is in the tumor relative to normal tissue. These include cell division genes (CDK1, AURKA, CCNB1), DNA repair genes (RAD51, BRCA1, CHEK1), known cancer driver genes (TP53, PIK3CA, MYC), and cell identity genes (CDH1, RUNX1, TGFB3). 
All gene expression values are already normalized, but we needed to filter out any non-Luminal cancer, apply binary encoding to the two types, and split the dataset into input and ground truth sets.

The dataset CSV is available in our repo in the data folder (METABRIC_RNA_Mutation.csv) or from the links above.

## Methods
We used a support vector machine (SVM) with Radial Basis Function (RBF) kernel, Random Forest (RF), and Multi-Layer Perceptron (MLP) models for our project. An SVM with RBF kernel is effective for high-feature problems in reducing overfitting, and the RBF kernel allows for fitting non-linear boundaries around the data. Random Forest splits features and data into random subsets, which can better protect against overfitting on specific features. Due to our small dataset, we kept our MLP model simple, with a maximum of two hidden layers. An overly large model without the dataset to back it up could lead to overfitting. We have also used Dropout, L2 decay, batch normalization, early stopping, and a sparsity penalty to help regularize our MLP results.

## How to run

The code has been written to run smoothly on local machines and on Google Colab. File paths resolve differently, but this is handled in a try-except block.

Google Colab: The code clones the git repo and imports any required files. If any updates are made to the repo while you are working in Colab, you must rerun the cloning code block to get the updated repo code.

Local: Please run the following in the terminal to install the required dependencies:

```bash
pip install -r requirements.txt
```
If running on Colab, necessary pip installs are handled by the code.

### Models
We have saved all our trained models in the "models" directory as joblib files. To access any of the joblib files, please run:

```python
import joblib
file_name = "<your_file_name_here>"  # replace with actual filename
data = joblib.load(file_name)
```

The specific shape of each joblib is specified below. -- TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1

#### SVM
-- TODO!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

#### Random Forest
The Random Forest Jupyter notebook saves two models on joblib, creates ENTERPLOTNUMBERHERE plots, and reports various statistics regarding the models. The first joblib file (ENTERMODELNAMEHERE) contains the model that uses all 489 genes. 

- `model`: the best estimator from hyperparameter search
- `X_train`, `X_test`, `y_train`, `y_test`: the train/test split used for that model

Then import that model into the CS6140_RandomForest_Features.ipynb by changing the file path where marked in the code. The features journal uses the base model to select the top k features, and retrains new models with each of the top k features. This also creates a joblib file with the following fields:

- `searches`: The results of the searches for every top k feature set
- `feature_indices`: 

#### Multi-Layer Perceptron