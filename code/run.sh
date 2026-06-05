#!/usr/bin/env bash
# =============================================================================
# COMP5434 Review Rating Prediction — Main Entry Point
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Colors ----
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

usage() {
    cat <<EOF
╔══════════════════════════════════════════════════════════════╗
║       COMP5434 Review Rating Prediction Pipeline            ║
╚══════════════════════════════════════════════════════════════╝

Usage: bash run.sh <stage> [options]

Stages:
  etl         Data extraction, transformation, loading
  features    Feature engineering (text, user, product, time)
  train       Model training with hyperparameter tuning
  predict     Generate predictions on test set
  submit      Create Kaggle submission CSV
  ablation    Run ablation experiments
  all         Run full pipeline (etl → features → train → predict → submit)

Options:
  --help, -h  Show this help message
  --verbose   Enable verbose logging
  --dry-run   Print commands without executing

Examples:
  bash run.sh all              # Run full pipeline
  bash run.sh etl              # Run only ETL stage
  bash run.sh features         # Run only feature engineering
  bash run.sh train --verbose  # Train with verbose output

Environment Variables:
  SPARK_MASTER     Spark master URL (default: local[*])
  DATA_DIR         Path to data directory (default: ../data)
  ARTIFACT_DIR     Path to artifacts (default: ../artifacts)
  N_PARTITIONS     Number of Spark partitions (default: 8)
EOF
}

# ---- Parse arguments ----
STAGE=""
VERBOSE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        *)
            if [[ -z "$STAGE" ]]; then
                STAGE="$1"
            else
                echo -e "${RED}Error: Unknown argument '$1'${NC}" >&2
                usage >&2
                exit 1
            fi
            shift
            ;;
    esac
done

if [[ -z "$STAGE" ]]; then
    echo -e "${RED}Error: No stage specified.${NC}" >&2
    usage >&2
    exit 1
fi

# ---- Environment ----
export SPARK_MASTER="${SPARK_MASTER:-local[*]}"
export DATA_DIR="${DATA_DIR:-${SCRIPT_DIR}/../data}"
export ARTIFACT_DIR="${ARTIFACT_DIR:-${SCRIPT_DIR}/../artifacts}"
export N_PARTITIONS="${N_PARTITIONS:-8}"

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║       COMP5434 Review Rating Prediction Pipeline            ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Stage:${NC}      $STAGE"
echo -e "${YELLOW}Spark Master:${NC} $SPARK_MASTER"
echo -e "${YELLOW}Data Dir:${NC}    $DATA_DIR"
echo -e "${YELLOW}Artifacts:${NC}   $ARTIFACT_DIR"
echo ""

# ---- Stage runners (stubs) ----
run_etl() {
    echo -e "${GREEN}[ETL]${NC} Data extraction, transformation, loading..."
    echo "  → Reading train.csv, test.csv, prodInfo.csv"
    echo "  → Handling missing values, type conversions"
    echo "  → Output: artifacts/etl/"
    # TODO: Implement ETL pipeline
    # python -m code.etl.main
}

run_features() {
    echo -e "${GREEN}[FEATURES]${NC} Feature engineering..."
    echo "  → Text features: TF-IDF, sentence embeddings"
    echo "  → User/product aggregation features"
    echo "  → Time-based features"
    echo "  → Output: artifacts/features/"
    # TODO: Implement feature engineering
    # python -m code.features.main
}

run_train() {
    echo -e "${GREEN}[TRAIN]${NC} Model training..."
    echo "  → Training LightGBM, XGBoost, CatBoost"
    echo "  → Hyperparameter tuning with Optuna"
    echo "  → Cross-validation and model selection"
    echo "  → Output: artifacts/models/"
    # TODO: Implement model training
    # python -m code.models.train
}

run_predict() {
    echo -e "${GREEN}[PREDICT]${NC} Generating predictions..."
    echo "  → Loading trained models"
    echo "  → Predicting on test set"
    echo "  → Output: artifacts/predictions/"
    # TODO: Implement prediction
    # python -m code.models.predict
}

run_submit() {
    echo -e "${GREEN}[SUBMIT]${NC} Creating Kaggle submission..."
    echo "  → Formatting predictions as submission.csv"
    echo "  → Validating submission format"
    echo "  → Output: kaggle/submission.csv"
    # TODO: Implement submission generation
    # python -m code.kaggle.generate
}

run_ablation() {
    echo -e "${GREEN}[ABLATION]${NC} Running ablation experiments..."
    echo "  → Testing feature group contributions"
    echo "  → Comparing individual model performance"
    echo "  → Output: artifacts/ablation/"
    # TODO: Implement ablation experiments
    # python -m code.ablation.main
}

# ---- Execute stage ----
case "$STAGE" in
    etl)
        run_etl
        ;;
    features)
        run_features
        ;;
    train)
        run_train
        ;;
    predict)
        run_predict
        ;;
    submit)
        run_submit
        ;;
    ablation)
        run_ablation
        ;;
    all)
        run_etl
        run_features
        run_train
        run_predict
        run_submit
        ;;
    *)
        echo -e "${RED}Error: Unknown stage '$STAGE'${NC}" >&2
        echo "" >&2
        echo "Valid stages: etl, features, train, predict, submit, ablation, all" >&2
        echo "Run 'bash run.sh --help' for usage." >&2
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}✓ Stage '$STAGE' completed.${NC}"
