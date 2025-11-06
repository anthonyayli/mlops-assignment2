#!/bin/bash

# MLOps Pipeline Script
# Runs the complete pipeline: preprocessing -> feature extraction -> training -> evaluation
# Excludes predict.py as requested

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
NC='\033[0m' # No Color

# Default paths (can be overridden with environment variables)
DATA_DIR="${DATA_DIR:-data}"
OUTPUT_DIR="${OUTPUT_DIR:-output}"
MODELS_DIR="${MODELS_DIR:-models}"
TRANSFORMERS_DIR="${TRANSFORMERS_DIR:-transformers}"
METRICS_DIR="${METRICS_DIR:-metrics}"

# Input data files
TRAIN_CSV="${TRAIN_CSV:-${DATA_DIR}/train.csv}"
TEST_CSV="${TEST_CSV:-${DATA_DIR}/test.csv}"

# Intermediate outputs
PROCESSED_DATA="${OUTPUT_DIR}/processed_data.csv"
TRAIN_FEATURES="${OUTPUT_DIR}/train_features.csv"
TRAIN_LABELS="${OUTPUT_DIR}/train_labels.csv"
TEST_FEATURES="${OUTPUT_DIR}/test_features.csv"
TEST_LABELS="${OUTPUT_DIR}/test_labels.csv"

# Model and transformers
MODEL_PATH="${MODELS_DIR}/logistic_regression.pkl"
TRANSFORMERS_OUTPUT="${TRANSFORMERS_DIR}"

# Metrics output
METRICS_OUTPUT="${METRICS_DIR}/metrics.json"

# Create necessary directories
echo -e "${GREEN}Creating output directories...${NC}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${MODELS_DIR}"
mkdir -p "${TRANSFORMERS_DIR}"
mkdir -p "${METRICS_DIR}"

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

# Step 1: Preprocessing
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 1: Preprocessing${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python scripts/preprocessing.py \
    --input-train "${TRAIN_CSV}" \
    --input-test "${TEST_CSV}" \
    --output "${PROCESSED_DATA}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Preprocessing failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Preprocessing completed successfully!${NC}"
echo ""

# Step 2: Feature Extraction
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 2: Feature Extraction${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python scripts/feature_extraction.py \
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

# Step 3: Training
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 3: Model Training${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python scripts/train.py \
    --train-features "${TRAIN_FEATURES}" \
    --train-labels "${TRAIN_LABELS}" \
    --model-output "${MODEL_PATH}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Training completed successfully!${NC}"
echo ""

# Step 4: Evaluation
echo -e "${YELLOW}========================================${NC}"
echo -e "${YELLOW}Step 4: Model Evaluation${NC}"
echo -e "${YELLOW}========================================${NC}"
uv run python scripts/evaluate.py \
    --model "${MODEL_PATH}" \
    --test-features "${TEST_FEATURES}" \
    --test-labels "${TEST_LABELS}" \
    --transformers-dir "${TRANSFORMERS_OUTPUT}" \
    --metrics-output "${METRICS_OUTPUT}"

if [ $? -ne 0 ]; then
    echo -e "${RED}Evaluation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}Evaluation completed successfully!${NC}"
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
echo "  - Model: ${MODEL_PATH}"
echo "  - Transformers: ${TRANSFORMERS_OUTPUT}/"
echo "  - Metrics: ${METRICS_OUTPUT}"
echo ""

