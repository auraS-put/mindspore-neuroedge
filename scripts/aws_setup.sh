#!/bin/bash
# AWS infrastructure setup for AURAS training
# Run this AFTER configuring AWS CLI credentials (aws configure)
#
# Creates: S3 bucket, ECR repo, IAM role, builds & pushes Docker image
# Region: eu-west-1 (Ireland) — cheapest EU region for GPU instances
#
# Usage: bash scripts/aws_setup.sh

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────
REGION="${AWS_REGION:-eu-west-1}"
BUCKET_NAME="auras-experiments-$(aws sts get-caller-identity --query Account --output text)"
ECR_REPO_NAME="auras-training"
ROLE_NAME="AurasSageMakerRole"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== AURAS AWS Setup ==="
echo "Region:  $REGION"
echo "Bucket:  $BUCKET_NAME"
echo "ECR:     $ECR_REPO_NAME"
echo "Role:    $ROLE_NAME"
echo ""

# ── Step 1: Create S3 Bucket ────────────────────────────────────────────
echo "[1/5] Creating S3 bucket..."
if aws s3api head-bucket --bucket "$BUCKET_NAME" 2>/dev/null; then
    echo "  Bucket already exists: $BUCKET_NAME"
else
    aws s3api create-bucket \
        --bucket "$BUCKET_NAME" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION"
    echo "  Created: s3://$BUCKET_NAME"
fi

# ── Step 2: Create IAM Role for SageMaker ───────────────────────────────
echo "[2/5] Creating SageMaker IAM role..."
TRUST_POLICY='{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {"Service": "sagemaker.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }
  ]
}'

ROLE_ARN=$(aws iam get-role --role-name "$ROLE_NAME" --query 'Role.Arn' --output text 2>/dev/null || true)
if [ -n "$ROLE_ARN" ] && [ "$ROLE_ARN" != "None" ]; then
    echo "  Role already exists: $ROLE_ARN"
else
    ROLE_ARN=$(aws iam create-role \
        --role-name "$ROLE_NAME" \
        --assume-role-policy-document "$TRUST_POLICY" \
        --query 'Role.Arn' --output text)
    echo "  Created role: $ROLE_ARN"
    
    # Attach SageMaker full access + S3 access
    aws iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
    aws iam attach-role-policy --role-name "$ROLE_NAME" \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
    echo "  Attached policies: SageMakerFullAccess, S3FullAccess"
    
    # Wait for role propagation
    echo "  Waiting 10s for IAM propagation..."
    sleep 10
fi

# ── Step 3: Create ECR Repository ───────────────────────────────────────
echo "[3/5] Creating ECR repository..."
ECR_URI=$(aws ecr describe-repositories --repository-names "$ECR_REPO_NAME" \
    --region "$REGION" --query 'repositories[0].repositoryUri' --output text 2>/dev/null || true)
if [ -n "$ECR_URI" ] && [ "$ECR_URI" != "None" ]; then
    echo "  ECR repo already exists: $ECR_URI"
else
    ECR_URI=$(aws ecr create-repository \
        --repository-name "$ECR_REPO_NAME" \
        --region "$REGION" \
        --query 'repository.repositoryUri' --output text)
    echo "  Created: $ECR_URI"
fi
IMAGE_URI="$ECR_URI:latest"

# ── Step 4: Build & Push Docker Image ───────────────────────────────────
echo "[4/5] Building & pushing Docker image..."
echo "  Logging into ECR..."
aws ecr get-login-password --region "$REGION" | \
    docker login --username AWS --password-stdin "$ECR_URI"

echo "  Building image (this may take a few minutes)..."
docker build -t "$ECR_REPO_NAME:latest" "$PROJECT_ROOT/docker/"

echo "  Tagging & pushing..."
docker tag "$ECR_REPO_NAME:latest" "$IMAGE_URI"
docker push "$IMAGE_URI"
echo "  Pushed: $IMAGE_URI"

# ── Step 5: Print env vars to add to .env ───────────────────────────────
echo ""
echo "[5/5] Done! Add these to your .env file:"
echo ""
echo "# === AWS Configuration ==="
echo "AWS_REGION=$REGION"
echo "AWS_S3_BUCKET=$BUCKET_NAME"
echo "SAGEMAKER_ROLE_ARN=$ROLE_ARN"
echo "SAGEMAKER_IMAGE_URI=$IMAGE_URI"
echo "CLOUD_PROVIDER=aws"
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Upload data:    aws s3 cp data/processed/siena_sop_merged.npz s3://$BUCKET_NAME/data/"
echo "  2. Upload code:    tar -czf /tmp/auras_code.tar.gz -C $PROJECT_ROOT/src . -C $PROJECT_ROOT/configs . -C $PROJECT_ROOT/scripts cloud_boot_benchmark.py"
echo "     Then:           aws s3 cp /tmp/auras_code.tar.gz s3://$BUCKET_NAME/data/"
echo "  3. Test storage:   python scripts/storage.py -p aws ls"
echo "  4. Submit job:     python scripts/submit_job.py -p aws --benchmark --env EPOCHS=1 MODELS=pyramidal_cnn_bilstm --flavor ml.g4dn.xlarge"
