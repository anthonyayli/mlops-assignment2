# mlops-2025
Repo for MLOps Course of class 2025-2026

## Pipeline Execution

### run_pipeline_with_classes.sh

Runs the complete MLOps pipeline using classes from `mlops_2025` module. Executes preprocessing, feature extraction, training (3 models), and evaluation.

**Pipeline Steps:**

1. **Preprocessing** (`mlops_2025.preprocessing.titanic_preprocessor`)
   - **Input:** `data/train.csv`, `data/test.csv`
   - **Output:** `output/processed_data.csv` (combined and cleaned data)

2. **Feature Extraction** (`mlops_2025.feature_extraction.titanic_feature_extractor`)
   - **Input:** `output/processed_data.csv`
   - **Process:** 
     - Filters training data (rows with `Survived`)
     - Splits into train/test (80/20)
     - **Fits transformers ONLY on training data** (`num_cat_transformer`, `bins_transformer`)
     - !!!!Transforms training features only
   - **Output:**
     - `output/train_features.csv` (transformed)
     - `output/train_labels.csv`
     - `output/test_features.csv` (raw, from split)
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
   - **Process:** Applies transformers to test features, evaluates model
   - **Output:**
     - `metrics/metrics_logistic_regression.json`
     - `metrics/metrics_xgboost.json`
     - `metrics/metrics_random_forest.json`

**Usage:**
```bash
./runners/run_pipeline_with_classes.sh
```
