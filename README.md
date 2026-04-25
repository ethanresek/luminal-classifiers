# How Many Genes Does It Take? Minimal gene expression classifiers for distinguishing Luminal A and Luminal B breast cancer

## File Structure

```
├── data/
│   └── METABRIC_RNA_Mutation.csv
├── models/
│   ├── final_MLP_base_search_20260424_191855.joblib
│   ├── final_MLP_grid_searches_20260424_200512.joblib
│   ├── final_rf_base_random_search_20260424_092839.joblib
│   ├── final_rf_grid_searches_20260424_124604.joblib
│   └── final_svm_results_20260424_204351.joblib
├── plots/
│   ├── MLP/
│   │   ├── mlp_confusion_matrices.png
│   │   ├── mlp_feature_importances.png
│   │   ├── mlp_gene_scatter_plot.png
│   │   └── mlp_performance_across_panels.png
│   ├── RF/
│   │   ├── rf_confusion_matrices.png
│   │   ├── rf_feature_importances.png
│   │   ├── rf_gene_scatter_plot.png
│   │   └── rf_performance_across_panels.png
│   ├── SVM/
│       ├── svm_confusion_matrices.png
│       ├── svm_feature_importances.png
│       ├── svm_gene_scatter_plot.png
│       └── svm_performance_across_panels.png
├── CS6140_MLP.ipynb
├── CS6140_RandomForest.ipynb
├── CS6140_SVM.ipynb
├── pre_process.py
└── requirements.txt
```

## Project Abstract
Breast cancer is a leading cause of death among women, and obtaining accurate diagnoses quickly is important for creating treatment plans. This is particularly challenging for Luminal A and Luminal B subtypes, which are both ER-positive and HER2-negative by standard clinical testing and therefore cannot be reliably distinguished without molecular profiling. The PAM50 assay addresses this by measuring the activity of 50 genes, but the question of how many genes are actually necessary remains open. Our project asks: can we select a smaller set of genes while still reliably classifying the cancer subtype? Our project uses the METABRIC dataset, which contains 489 gene expression features from 1,140 patients that will be used as feature dimensions and samples, respectively. We will train three different classifier models: a support vector machine with RBF kernel, a Random Forest model, and a deep neural network across five gene panel sizes (5, 10, 20, 30, 40, 50, 489 genes) from the dataset. We will use these models to classify a set of training samples into the two subtypes and compare the three models’ effectiveness in their classification.

## Dataset
We use the METABRIC (Molecular Taxonomy of Breast Cancer International Consortium) dataset, publicly available on [Kaggle](https://www.kaggle.com/datasets/raghadalharbi/breast-cancer-gene-expression-profiles-metabric/) and [cBioPortal](https://www.cbioportal.org/study/summary?id=brca_metabric). METABRIC is a large breast cancer study from Canada and the UK that followed 1,904 patients for an average of about 10 years. After keeping only Luminal A and Luminal B patients, the working dataset is:
- 1,140 patients: 679 Luminal A (59.6%) and 461 Luminal B (40.4%)
- 489 gene expression features, each measuring how active a cancer-relevant gene is in the tumor relative to normal tissue. These include cell division genes (CDK1, AURKA, CCNB1), DNA repair genes (RAD51, BRCA1, CHEK1), known cancer driver genes (TP53, PIK3CA, MYC), and cell identity genes (CDH1, RUNX1, TGFB3). 
All gene expression values are already normalized, but we needed to filter out any non-Luminal cancer, apply binary encoding to the two types (LumA=1, LumB=0), and split the dataset into input and ground truth sets.

The dataset CSV is available in our repo in the data folder (METABRIC_RNA_Mutation.csv) or from the links above.

## Methods
We used a support vector machine (SVM) with Radial Basis Function (RBF) kernel, Random Forest (RF), and Multi-Layer Perceptron (MLP) models for our project. An SVM with RBF kernel is effective for high-feature problems in reducing overfitting, and the RBF kernel allows for fitting non-linear boundaries around the data. Random Forest splits features and data into random subsets, which can better protect against overfitting on specific features. Due to our small dataset, we kept our MLP model simple, with one to two hidden layers. An overly large model without the dataset to back it up could lead to overfitting. We have also used Dropout, L2 decay, batch normalization, early stopping, and a sparsity penalty to help regularize our MLP results.

F1 was used as the scoring metric for all hyperparameter searches because it would handle the unbalanced dataset well. RF and MLP use an 85/15 train/test split while SVM uses a nested CV on the full dataset.

## How to run

The code has been written to run smoothly on local machines and on Google Colab. File paths resolve differently, but this is handled in a try-except block.

Google Colab: The code clones the git repo and imports any required files. If any updates are made to the repo while you are working in Colab, you must rerun the cloning code block to get the updated repo code. Also, if running on Colab, necessary pip installs are handled by the code. 

Local: Please run the following in the terminal to install the required dependencies:

```bash
pip install -r requirements.txt
```

### Models
The models we created can be reproduced by running the corresponding notebook. If the repo has not been edited, they can run as is. Where to edit the CSV file path has been noted in each of the notebooks in case it changes. The random state seed is consistent throughout (42).

We have saved all our trained models in the "models" directory as joblib files. To access any of the joblib files, please run:

```
file_name = "<your_file_name_here>"  # replace with actual filename
data = joblib.load(file_name)
```

The specific shape of each joblib is specified below.

#### SVM
The Support Vector Machine Jupyter notebook saves one joblib file, creates and saves 4 plots as PNG files, and reports various statistics regarding the models. The joblib file (models/final_svm_results_20260424_204351.joblib) contains all the searches from each panel size.

- `results`: A dictionary where each key is a panel size and each value is a list of five dictionaries (one per fold). For example, results\[20]\[2] gives the metrics and predictions of the third fold of the 20-gene panel. Each inner dictionary contains:
  - `f1`
  - `balanced_accuracy`
  - `auc`
  - `C`
  - `gamma`
  - `y_true`
  - `y_pred`
  - `y_prob`
- `best_params_per_panel`: An array of objects containing the best hyperparameters selected by the CV
- `panel_sizes`: The number of features included in each top k model. Stored in an array.
- `t_stats`: The importance vector for every gene.

The joblib does not store the models themselves. It is our fastest to train by far (only takes a few minutes), so the models can easily be replicated by running the code.

#### Random Forest
The Random Forest Jupyter notebook saves two joblib files, creates and saves 4 plots as PNG files, and reports various statistics regarding the models. The first joblib file (models/final_rf_base_random_search_20260424_092839.joblib) contains the model that uses all 489 genes. The joblib file contains:

- `model`: the result of the RandomizedSearchCV
- `X_train`, `X_test`, `y_train`, `y_test`: the train/test split used for that model

To extract the model itself, load the joblib file and run:

```
model = data['model'].best_estimator_
```

The second joblib file (models/final_rf_grid_searches_20260424_124604.joblib) contains the rest of the models. This contains:

- `searches`: The results of the searches for every top k feature set. Stored in an array.
- `feature_indices`: The indices of X that contain the top k features. 2d array where each inner array contains k indices
- `feature_counts`: The number of features included in each top k model. Stored in an array.
- `rf_results`: An array of objects that contain accuracy scoring for each feature size
- `best_params_per_panel`: An array of objects containing the best hyperparameters selected by the CV

The indices of each outermost array map to each other. So, for example, the search result, indices, count, results, and params for the top-5 model are stored in index 0 of each array.

#### Multi-Layer Perceptron
The MLP Jupyter notebook saves two joblib files, creates and saves 4 plots as PNG files, and reports various statistics regarding the models. The first joblib file (models/final_MLP_base_search_20260424_191855.joblib) contains the model that uses all 489 genes. The joblib file contains:

- `model`: the result of the RandomizedSearchCV
- `X_train`, `X_test`, `y_train`, `y_test`: the train/test split used for that model

To extract the model itself, load the joblib file and run:

```
model = data['model'].best_estimator_
```
The second joblib file (models/final_MLP_grid_searches_20260424_200512.joblib) contains the rest of the models. This contains:

- `searches`: The results of the searches for every top k feature set. Stored in an array.
- `feature_indices`: The indices of X that contain the top k features. 2d array where each inner array contains k indices
- `feature_counts`: The number of features included in each top k model. Stored in an array.
- `fold_evals`: A dictionary where each key is a panel size and each value is a list of five dictionaries (one per fold). For example, fold_evals\[20]\[2] gives the metrics and predictions of the third fold of the 20-gene panel. Each inner dictionary contains:
  -  `f1`
  - `balanced_accuracy`
  - `roc_auc`
  - `y_true`
  - `y_pred`
  - `y_pred_prob`

The indices of each outermost array map to each other. So, for example, the search result, indices, count, results, and params for the top-5 model are stored in index 0 of each array.

## Evaluation Metrics
All models are compared using F1-Score, ROC-AUC, and balanced accuracy.

- F1-Score: Combination of precision and recall. Precision measures how many positive predictions were correct, and recall measures how many actual positives you caught. This measures how well the model performs with a threshold of 0.5.
- ROC-AUC: Measures how well model distinguishes between classes across all decision thresholds from 0 to 1, not just the 0.5 used by F1. The ROC curve measures the ability to distinguish, and AUC quantifies the overall performance, with 0.5 being equivalent to random guessing and 1.0 being perfect. Evaluates predicted probabilities rather than just final binary predictions.
- Balanced Accuracy: Averages recall of each class separately, so each class is treated equally regardless of any class imbalance. This is used rather than standard average because of our imbalanced dataset.
