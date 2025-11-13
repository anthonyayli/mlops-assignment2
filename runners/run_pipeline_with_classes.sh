#!/bin/bash

# MLOps Pipeline Script using mlops_2025 classes
# Runs the complete pipeline: preprocessing -> feature extraction -> training (3 models) -> evaluation -> prediction
# Uses classes from mlops_2025 module
# Uses predict.py on the preprocessed test data from processed_data.csv for all 3 models

set -e  # Exit on any error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Change to project root (one level up from runners/)
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${PROJECT_ROOT}"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths (can be overridden with environment variables)
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
MODELS_DIR="${MODELS_DIR:-models}"
TRANSFORMERS_DIR="${TRANSFORMERS_DIR:-transformers}"
METRICS_DIR="${METRICS_DIR:-metrics}"
PREDICTIONS_DIR="${PREDICTIONS_DIR:-predictions}"

# Input data files
TRAIN_CSV="${TRAIN_CSV:-${DATA_DIR}/train.csv}"
TEST_CSV="${TEST_CSV:-${DATA_DIR}/test.csv}"

# Intermediate outputs
PROCESSED_DATA="${OUTPUT_DIR}/processed_data.csv"
TRAIN_FEATURES="${OUTPUT_DIR}/train_features.csv"
TRAIN_LABELS="${OUTPUT_DIR}/train_labels.csv"
TEST_FEATURES="${OUTPUT_DIR}/test_features.csv"
TEST_LABELS="${OUTPUT_DIR}/test_labels.csv"

# Model paths
LOGISTIC_REGRESSION_MODEL="${MODELS_DIR}/logistic_regression.pkl"
XGBOOST_MODEL="${MODELS_DIR}/xgboost.pkl"
RANDOM_FOREST_MODEL="${MODELS_DIR}/random_forest.pkl"

# Transformers output
TRANSFORMERS_OUTPUT="${TRANSFORMERS_DIR}"

# Metrics outputs
METRICS_LR="${METRICS_DIR}/metrics_logistic_regression.json"
METRICS_XGB="${METRICS_DIR}/metrics_xgboost.json"
METRICS_RF="${METRICS_DIR}/metrics_random_forest.json"

# Predictions outputs
PREDICTIONS_LR="${PREDICTIONS_DIR}/predictions_logistic_regression.csv"
PREDICTIONS_XGB="${PREDICTIONS_DIR}/predictions_xgboost.csv"
PREDICTIONS_RF="${PREDICTIONS_DIR}/predictions_random_forest.csv"

# Create necessary directories
echo -e "${GREEN}Creating output directories...${NC}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${TRANSFORMERS_DIR}"
mkdir -p "${METRICS_DIR}"
mkdir -p "${PREDICTIONS_DIR}"

# Check if input files exist
if [ ! -f "${TRAIN_CSV}" ]; then
    echo -e "${RED}Error: Training data not found at ${TRAIN_CSV}${NC}"
    echo "Please set TRAIN_CSV environment variable or place train.csv in ${DATA_DIR}/"
    exit 1
fi

if [ ! -f "${TEST_CSV}" ]; then
    echo -e "${RED}Error: Test data not found at ${TEST_CSV}${NC}"
    echo "Please set TEST_CSV environment variable or place test.csv in ${DATA_DIR}/"
    exit 1
fi

echo -e "${GREEN}Input files found:${NC}"
echo "  Train: ${TRAIN_CSV}"
echo "  Test: ${TEST_CSV}"
echo ""

# Step 1: Preprocessing using mlops_2025.preprocessing.TitanicPreprocessor
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 1: Preprocessing${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python -m mlops_2025.preprocessing.titanic_preprocessor \
    --input-train "${TRAIN_CSV}" \
    --input-test "${TEST_CSV}" \
    --output "${PROCESSED_DATA}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Preprocessing failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Preprocessing completed successfully!${NC}"
echo ""

# Step 2: Feature Extraction using mlops_2025.feature_extraction.TitanicFeatureExtractor
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 2: Feature Extraction${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python -m mlops_2025.feature_extraction.titanic_feature_extractor \
    --input-frame "${PROCESSED_DATA}" \
    --output-train-features "${TRAIN_FEATURES}" \
    --output-train-labels "${TRAIN_LABELS}" \
    --output-test-features "${TEST_FEATURES}" \
    --output-test-labels "${TEST_LABELS}" \
    --transformers-output "${TRANSFORMERS_OUTPUT}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Feature extraction failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Feature extraction completed successfully!${NC}"
echo ""

# Step 3: Training - Logistic Regression
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 3a: Training Logistic Regression${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python -m mlops_2025.models.logistic_regression \
    --train-features "${TRAIN_FEATURES}" \
    --train-labels "${TRAIN_LABELS}" \
    --model-output "${LOGISTIC_REGRESSION_MODEL}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Logistic Regression training failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Logistic Regression training completed successfully!${NC}"
echo ""

# Step 3: Training - XGBoost
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 3b: Training XGBoost${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python -m mlops_2025.models.xgboost_model \
    --train-features "${TRAIN_FEATURES}" \
    --train-labels "${TRAIN_LABELS}" \
    --model-output "${XGBOOST_MODEL}"

if [ $? -ne 0 ]; then
    echo -e "${RED}XGBoost training failed!${NC}"
    exit 1
fi

echo -e "${GREEN}XGBoost training completed successfully!${NC}"
echo ""

# Step 3: Training - Random Forest
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 3c: Training Random Forest${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python -m mlops_2025.models.random_forest \
    --train-features "${TRAIN_FEATURES}" \
    --train-labels "${TRAIN_LABELS}" \
    --model-output "${RANDOM_FOREST_MODEL}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Random Forest training failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Random Forest training completed successfully!${NC}"
echo ""

# Step 4: Evaluation - Logistic Regression
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 4a: Evaluating Logistic Regression${NC}"
echo -e "${BLUE}========================================${NC}"
uv run python scripts/evaluate.py \
    --model "${LOGISTIC_REGRESSION_MODEL}" \
    --test-features "${TEST_FEATURES}" \
    --test-labels "${TEST_LABELS}" \
    --transformers-dir "${TRANSFORMERS_OUTPUT}" \
    --metrics-output "${METRICS_LR}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Logistic Regression evaluation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Logistic Regression evaluation completed successfully!${NC}"
echo ""

# Step 4: Evaluation - XGBoost
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 4b: Evaluating XGBoost${NC}"
echo -e "${BLUE}========================================${NC}"
uv run python scripts/evaluate.py \
    --model "${XGBOOST_MODEL}" \
    --test-features "${TEST_FEATURES}" \
    --test-labels "${TEST_LABELS}" \
    --transformers-dir "${TRANSFORMERS_OUTPUT}" \
    --metrics-output "${METRICS_XGB}"

if [ $? -ne 0 ]; then
    echo -e "${RED}XGBoost evaluation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}XGBoost evaluation completed successfully!${NC}"
echo ""

# Step 4: Evaluation - Random Forest
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Step 4c: Evaluating Random Forest${NC}"
echo -e "${BLUE}========================================${NC}"
uv run python scripts/evaluate.py \
    --model "${RANDOM_FOREST_MODEL}" \
    --test-features "${TEST_FEATURES}" \
    --test-labels "${TEST_LABELS}" \
    --transformers-dir "${TRANSFORMERS_OUTPUT}" \
    --metrics-output "${METRICS_RF}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Random Forest evaluation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Random Forest evaluation completed successfully!${NC}"
echo ""

# Step 5: Prediction on Preprocessed Test Data - Logistic Regression
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 5a: Prediction on Preprocessed Test Data (Logistic Regression)${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python scripts/predict.py \
    --model "${LOGISTIC_REGRESSION_MODEL}" \
    --processed-data "${PROCESSED_DATA}" \
    --transformers-dir "${TRANSFORMERS_OUTPUT}" \
    --output "${PREDICTIONS_LR}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Logistic Regression prediction failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Logistic Regression prediction completed successfully!${NC}"
echo ""

# Step 5: Prediction on Preprocessed Test Data - XGBoost
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 5b: Prediction on Preprocessed Test Data (XGBoost)${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python scripts/predict.py \
    --model "${XGBOOST_MODEL}" \
    --processed-data "${PROCESSED_DATA}" \
    --transformers-dir "${TRANSFORMERS_OUTPUT}" \
    --output "${PREDICTIONS_XGB}"

if [ $? -ne 0 ]; then
    echo -e "${RED}XGBoost prediction failed!${NC}"
    exit 1
fi

echo -e "${GREEN}XGBoost prediction completed successfully!${NC}"
echo ""

# Step 5: Prediction on Preprocessed Test Data - Random Forest
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 5c: Prediction on Preprocessed Test Data (Random Forest)${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python scripts/predict.py \
    --model "${RANDOM_FOREST_MODEL}" \
    --processed-data "${PROCESSED_DATA}" \
    --transformers-dir "${TRANSFORMERS_OUTPUT}" \
    --output "${PREDICTIONS_RF}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Random Forest prediction failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Random Forest prediction completed successfully!${NC}"
echo ""

# Summary
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Output files:"
echo "  - Processed data: ${PROCESSED_DATA}"
echo "  - Train features: ${TRAIN_FEATURES}"
echo "  - Train labels: ${TRAIN_LABELS}"
echo "  - Test features: ${TEST_FEATURES}"
echo "  - Test labels: ${TEST_LABELS}"
echo ""
echo "Models saved:"
echo "  - Logistic Regression: ${LOGISTIC_REGRESSION_MODEL}"
echo "  - XGBoost: ${XGBOOST_MODEL}"
echo "  - Random Forest: ${RANDOM_FOREST_MODEL}"
echo ""
echo "Transformers saved:"
echo "  - ${TRANSFORMERS_OUTPUT}/num_cat_transformer.pkl"
echo "  - ${TRANSFORMERS_OUTPUT}/bins_transformer.pkl"
echo ""
echo "Metrics saved:"
echo "  - Logistic Regression: ${METRICS_LR}"
echo "  - XGBoost: ${METRICS_XGB}"
echo "  - Random Forest: ${METRICS_RF}"
echo ""
echo "Predictions on preprocessed test data:"
echo "  - Logistic Regression: ${PREDICTIONS_LR}"
echo "  - XGBoost: ${PREDICTIONS_XGB}"
echo "  - Random Forest: ${PREDICTIONS_RF}"
echo ""

