# mlops-2025
Repo for MLOps Course of class 2025-2026

## Pipeline Execution

### run_pipeline_with_classes.sh

Runs the complete MLOps pipeline using classes from `mlops_2025` module. Executes preprocessing, feature extraction, training (3 models), evaluation, and prediction.

**Pipeline Steps:**

1. **Preprocessing** (`mlops_2025.preprocessing.titanic_preprocessor`)
   - **Input:** `data/train.csv`, `data/test.csv`
   - **Process:** 
     - Combines train and test data for consistent feature engineering
     - Handles missing values, creates features (Title, Family_size, etc.)
     - **Note:** Test data (rows without `Survived`) is preprocessed but stored in the combined file
   - **Output:** `output/processed_data.csv` (combined and cleaned data with both train and test rows)

2. **Feature Extraction** (`mlops_2025.feature_extraction.titanic_feature_extractor`)
   - **Input:** `output/processed_data.csv`
   - **Process:** 
     - Filters training data (rows with `Survived` labels)
     - **Excludes preprocessed test data** (rows where `Survived` is NaN)
     - Splits training data into train/test (80/20) for validation
     - **Fits transformers ONLY on training data** (`num_cat_transformer`, `bins_transformer`)
     - Transforms training features only
   - **Output:**
     - `output/train_features.csv` (transformed)
     - `output/train_labels.csv`
     - `output/test_features.csv` (raw, from 80/20 split - used for validation)
     - `output/test_labels.csv`
     - `transformers/num_cat_transformer.pkl`
     - `transformers/bins_transformer.pkl`

3. **Training** (3 models)
   - **Input:** `output/train_features.csv`, `output/train_labels.csv`
   - **Output:**
     - `models/logistic_regression.pkl`
     - `models/xgboost.pkl`
     - `models/random_forest.pkl`

4. **Evaluation** (`scripts/evaluate.py`)
   - **Input:** Model files, `output/test_features.csv`, `output/test_labels.csv`, `transformers/`
   - **Process:** Applies transformers to validation test features, evaluates model performance
   - **Uses:** `DataLoader`, `ModelLoader`, `TransformersLoader` for consistent I/O
   - **Output:**
     - `metrics/metrics_logistic_regression.json`
     - `metrics/metrics_xgboost.json`
     - `metrics/metrics_random_forest.json`

5. **Prediction** (`scripts/predict.py`)
   - **Input:** Model files, `output/processed_data.csv`, `transformers/`
   - **Process:** 
     - Extracts preprocessed test data from `processed_data.csv` (rows where `Survived` is NaN)
     - Applies transformers to preprocessed test data
     - Makes predictions on original test.csv data (that wasn't used in training)
   - **Uses:** `DataLoader`, `ModelLoader`, `TransformersLoader`, `DataSaver` for consistent I/O
   - **Output:**
     - `predictions/predictions_logistic_regression.csv`
     - `predictions/predictions_xgboost.csv`
     - `predictions/predictions_random_forest.csv`

**Usage:**
```bash
./runners/run_pipeline_with_classes.sh
```

## Architecture

### Loaders (`mlops_2025.loaders`)

Abstract base class `BaseLoader` with `load()` method. Implementations:

- **DataLoader**: Loads CSV files into pandas DataFrames
- **TransformersLoader**: Loads pickle files containing transformer objects (can load single or all from directory)
- **ModelLoader**: Loads pickle files containing trained models

### Savers (`mlops_2025.savers`)

Abstract base class `BaseSaver` with `save()` method. Implementations:

- **DataSaver**: Saves pandas DataFrames/Series to CSV files
- **TransformersSaver**: Saves transformer objects to pickle files (can save single or multiple to directory)
- **ModelSaver**: Saves trained model objects to pickle files

All pipeline components use these loaders and savers for consistent data I/O operations.

## Key Design Decisions

### Test Data Usage

- **`test.csv` is used in preprocessing** to ensure consistent feature engineering (e.g., Age imputation using group medians from combined data)
- **Preprocessed test data is stored** in `processed_data.csv` (rows where `Survived` is NaN)
- **Test data is excluded from training** - feature extraction filters out rows without `Survived` labels
- **Preprocessed test data is reused for predictions** - `predict.py` extracts test rows from `processed_data.csv` instead of preprocessing again
- This ensures:
  - No data leakage (test data never used in training)
  - Consistent preprocessing (same transformations applied to train and test)
  - No duplicate preprocessing (test data preprocessed once, reused for predictions)

