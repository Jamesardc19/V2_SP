# Early Detection of the Risk of Diabetes Using Calibrations on Explainable Artificial Intelligence (XAI) with Philippine Data

This repository contains the implementation for the thesis project on early diabetes risk detection using machine learning models with Philippine data.

## Project Overview

This project aims to develop and evaluate machine learning models for early detection of diabetes risk using data from the National Nutrition Survey Dataset (2018-2021). The implementation includes:

1. Comprehensive data preprocessing
2. Training and evaluation of multiple machine learning models
3. Implementation of calibration techniques
4. Application of explainable AI methods

## Directory Structure

```
├── DATASETS/
│   └── MAIN DATASET/
│       ├── data-set_anthrop.csv       # Anthropometric data
│       ├── data-set_biochemical.csv   # Biochemical data
│       ├── data-set_clinical.csv      # Clinical health data
│       └── data-set_dietary_indiv.csv # Dietary data
├── EDA_Results/                       # Exploratory Data Analysis results
├── Model_Results/                     # Model evaluation results
├── LIME_Explanations/                 # LIME model interpretability visualizations
├── Trained_Models/                    # Saved trained models
├── data_preprocessing.py              # Data preprocessing script
├── model_training.py                  # Model training and evaluation script
└── processed_dataset.csv              # Preprocessed dataset ready for modeling
```

## Requirements

The following Python packages are required to run the scripts:

```
pandas
numpy
scikit-learn
matplotlib
seaborn
imblearn
xgboost
pytorch-tabnet
torch
bayesian-optimization
lime
```

You can install all required packages using:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn imblearn xgboost pytorch-tabnet bayesian-optimization lime
```

## Data Preprocessing

The data preprocessing script (`data_preprocessing.py`) performs the following operations:

1. Loads the four datasets (anthropometric, biochemical, clinical, and dietary)
2. Merges the datasets based on common columns (`regcode`, `provhuc`, `enns_year`, `hhnum`, `member_code`)
3. Drops rows with missing target values (`fbs`)
4. Identifies columns with missing values and segregates them between numerical and categorical
5. Imputes missing numerical values with mean and categorical values with '99'
6. Transforms the target attribute (`fbs`) into a binary categorical variable:
   - Values < 100 are categorized as non-diabetic (0)
   - Values >= 100 are categorized as diabetic (1)
7. Scales numerical features using standardization
8. Performs exploratory data analysis and saves results
9. Saves the preprocessed dataset for further analysis

### Running the Data Preprocessing Script

```bash
python data_preprocessing.py
```

This will generate:
- `processed_dataset.csv`: The preprocessed dataset
- Various EDA results in the `EDA_Results/` directory

## Model Training and Evaluation

The model training script (`model_training.py`) implements and evaluates the following machine learning models:

1. Naive Bayes
2. Logistic Regression
3. Support Vector Machines (SVM)
4. Random Forest
5. k-Nearest Neighbors (k-NN)
6. XGBoost
7. BO-TabNet (TabNet with Bayesian Optimization)

The script also:
- Applies SMOTE to handle class imbalance
- Implements calibration techniques (Platt Scaling and Isotonic Regression)
- Uses LIME for model interpretability
- Evaluates models using Precision, Recall, F2-score, and AUPRC

### Running the Model Training Script

```bash
python model_training.py
```

This will:
1. Load the preprocessed dataset
2. Split the data into training and testing sets
3. Apply SMOTE to handle class imbalance
4. Train and evaluate all models
5. Apply calibration techniques
6. Generate model interpretability visualizations using LIME
7. Save trained models and evaluation results

## Results

After running the scripts, you can find:

1. **EDA Results** in the `EDA_Results/` directory:
   - Basic statistics
   - Missing values analysis
   - Correlation matrix
   - Target distribution
   - FBS distribution

2. **Model Results** in the `Model_Results/` directory:
   - Model performance metrics
   - Confusion matrices
   - ROC curves
   - Precision-Recall curves
   - Calibration curves

3. **LIME Explanations** in the `LIME_Explanations/` directory:
   - Feature importance visualizations for each model

4. **Trained Models** in the `Trained_Models/` directory:
   - Saved model files for future use or deployment

## Performance Metrics

The models are evaluated using the following metrics:

1. **Precision**: The ability of the model to avoid false positives
2. **Recall**: The ability of the model to find all positive samples
3. **F2-score**: A weighted harmonic mean of precision and recall that gives more weight to recall
4. **AUPRC (Area Under the Precision-Recall Curve)**: Summarizes the precision-recall trade-off

## Notes on Model Training

- **SVM Training**: The SVM model has been optimized with parameters to improve training speed while maintaining performance.
- **TabNet**: The TabNet model uses Bayesian Optimization to find optimal hyperparameters, which may take longer to train but can provide better performance. Note that TabNet requires NumPy arrays as input (not Pandas DataFrames), which is handled automatically in the code.
- **SMOTE**: Class imbalance is addressed using SMOTE, which creates synthetic samples of the minority class.
- **Calibration**: Both Platt Scaling and Isotonic Regression are applied to improve the reliability of predicted probabilities.

## Troubleshooting

If you encounter any issues:

1. **Memory Errors**: Reduce batch sizes in the TabNet model or use a subset of the data for initial testing.
2. **Long Training Times**: SVM and TabNet may take longer to train. Consider reducing the dataset size or adjusting parameters for faster training.
3. **Missing Dependencies**: Ensure all required packages are installed using the pip command provided above.

## Citation

If you use this code or the findings in your research, please cite:

```
Early Detection of the Risk of Diabetes Using Calibrations on Explainable Artificial Intelligence (XAI) with Philippine Data
```
