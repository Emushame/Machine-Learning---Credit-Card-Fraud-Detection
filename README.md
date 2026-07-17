# Machine Learning - Credit Card Fraud Detection

## Overview
This project trains a machine learning model to detect fraudulent credit card transactions using the popular **Credit Card Fraud Detection** dataset (Kaggle-style).

The main workflow is implemented in the notebook:
- `1. Credit Card Fraud Detection - ML.ipynb`

## Dataset
The notebook loads the dataset from:
- `data/creditcard.csv`

### Target
- `Class`:
  - `0` = Normal transaction
  - `1` = Fraud transaction

### Features
All columns except `Class` are used as input features (`V1`..`V28`, plus `Amount`).

## What the notebook does
1. **Import libraries**
   - `pandas`, `numpy`, `matplotlib`, `seaborn`
   - `scikit-learn` utilities for model training and evaluation
2. **Load the data** using `pd.read_csv('data/creditcard.csv')`.
3. **Analyze class distribution** to measure fraud vs. normal imbalance.
4. **Explore `Amount` statistics** for fraud and normal subsets.
5. **Plot correlation matrix** of numeric features.
6. **Prepare training data**
   - `X = df.drop(['Class'], axis=1)`
   - `Y = df['Class']`
7. **Train/test split**
   - `train_test_split(..., test_size=0.2, random_state=42)`
8. **Model training**
   - Uses `RandomForestClassifier` with default hyperparameters.
9. **Model evaluation** on the test set using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Matthews Correlation Coefficient (MCC)
   - Confusion matrix (plotted as a heatmap)

## Requirements
The notebook uses Python and the following libraries:
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## How to run
### Option A: Jupyter Notebook
1. Open `1. Credit Card Fraud Detection - ML.ipynb` in Jupyter / VS Code.
2. Ensure the dataset file exists at `data/creditcard.csv`.
3. Run the notebook cells from top to bottom.

### Option B: Run as a notebook with Jupyter
From the project root:
```bash
jupyter notebook "1. Credit Card Fraud Detection - ML.ipynb"
```
Then execute cells.

## Project structure
```text
.
├── Credit Card Fraud Detection - ML.ipynb
├── data/
│   └── creditcard.csv
└── .gitignore
```

## Notes / Considerations
- The dataset is **heavily imbalanced** (fraud is much rarer than normal). Metrics like **Recall**, **F1**, and **MCC** are therefore more informative than accuracy alone.
- The current implementation uses a **baseline** model (`RandomForestClassifier` with default settings). For improved performance, consider:
  - hyperparameter tuning (e.g., `n_estimators`, `max_depth`)
  - class imbalance handling (e.g., `class_weight`, resampling)
  - threshold tuning based on precision/recall trade-offs

## License
Add a license file if you want to specify usage terms.

