"""
Submit training job to Vertex AI.

Usage:
    python scripts/submit_vertex_training.py \
        --project-id ocr-training-12345 \
        --region us-central1 \
        --bucket-name ocr-data-12345 \
        --image-uri us-central1-docker.pkg.dev/ocr-training-12345/vertex-ai/ocr-classifier-training:latest
"""

import argparse
from google.cloud import aiplatform


def submit_training_job(
    project_id: str,
    region: str,
    bucket_name: str,
    image_uri: str,
    job_name: str = "classifier-training",
    epochs: int = 30,
    batch_size: int = 64,
    machine_type: str = "n1-standard-4",
    gpu_type: str = None,
    gpu_count: int = 0,
):
    """
    Submit a training job to Vertex AI.
    
    Args:
        project_id: GCP project ID
        region: GCP region (e.g., us-central1)
        bucket_name: GCS bucket name
        image_uri: Docker image URI
        job_name: Display name for the job
        epochs: Number of training epochs
        batch_size: Batch size
        machine_type: Machine type (n1-standard-4, n1-standard-8, etc.)
        gpu_type: GPU type (NVIDIA_TESLA_K80, NVIDIA_TESLA_T4, NVIDIA_TESLA_V100)
        gpu_count: Number of GPUs
    """
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=region)
    
    # Create container job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=image_uri,
    )
    
    # Prepare command-line arguments
    args = [
        "--gcs-bucket", bucket_name,
        "--gcs-data-prefix", "data/difficulty_labels",
        "--gcs-model-prefix", "models/classifier",
        "--epochs", str(epochs),
        "--batch-size", str(batch_size),
    ]
    
    print("=" * 80)
    print("SUBMITTING VERTEX AI TRAINING JOB")
    print("=" * 80)
    print(f"Project ID:       {project_id}")
    print(f"Region:           {region}")
    print(f"Job Name:         {job_name}")
    print(f"Image:            {image_uri}")
    print(f"Machine Type:     {machine_type}")
    if gpu_type:
        print(f"GPU:              {gpu_count}x {gpu_type}")
    else:
        print(f"GPU:              None (CPU only)")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size}")
    print(f"Data:             gs://{bucket_name}/data/difficulty_labels")
    print("=" * 80)
    print()
    
    try:
        # Submit job
        model = job.run(
            args=args,
            replica_count=1,
            machine_type=machine_type,
            accelerator_type=gpu_type if gpu_type else None,
            accelerator_count=gpu_count if gpu_type else 0,
            sync=False,  # Don't wait for job to complete
        )
        
        print("✓ Job submitted successfully!")
        print()
        print(f"Job Resource Name:")
        print(f"  {model.resource_name}")
        print()
        print("Monitor your training job:")
        print(f"  Web UI: https://console.cloud.google.com/vertex-ai/training/custom-jobs?")
        print(f"          project={project_id}")
        print()
        print("Stream logs:")
        print(f"  gcloud ai custom-jobs stream-logs {model.resource_name.split('/')[-1]} \\")
        print(f"    --region={region} --project={project_id}")
        print()
        print("Download results when complete:")
        print(f"  gsutil -m cp -r gs://{bucket_name}/models/classifier ./")
        print()
        print("=" * 80)
        
        return model
    
    except Exception as e:
        print(f"✗ Error submitting job: {e}")
        print()
        print("Troubleshooting tips:")
        print("  1. Check that you're authenticated: gcloud auth login")
        print("  2. Verify project ID: gcloud config list")
        print("  3. Check services enabled: gcloud services list --enabled")
        print("  4. Verify Docker image exists: gcloud artifacts docker images list")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Submit Vertex AI training job",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

  # Basic CPU training
  python scripts/submit_vertex_training.py \\
      --project-id ocr-training-12345 \\
      --region us-central1 \\
      --bucket-name ocr-data-12345 \\
      --image-uri us-central1-docker.pkg.dev/ocr-training-12345/vertex-ai/ocr-classifier-training:latest

  # With GPU
  python scripts/submit_vertex_training.py \\
      --project-id ocr-training-12345 \\
      --region us-central1 \\
      --bucket-name ocr-data-12345 \\
      --image-uri us-central1-docker.pkg.dev/ocr-training-12345/vertex-ai/ocr-classifier-training:latest \\
      --gpu-type NVIDIA_TESLA_T4 \\
      --gpu-count 1

  # Larger machine
  python scripts/submit_vertex_training.py \\
      --project-id ocr-training-12345 \\
      --region us-central1 \\
      --bucket-name ocr-data-12345 \\
      --image-uri us-central1-docker.pkg.dev/ocr-training-12345/vertex-ai/ocr-classifier-training:latest \\
      --machine-type n1-standard-8
        """
    )
    
    parser.add_argument("--project-id", required=True, help="GCP project ID")
    parser.add_argument("--region", default="us-central1", help="GCP region (default: us-central1)")
    parser.add_argument("--bucket-name", required=True, help="GCS bucket name")
    parser.add_argument("--image-uri", required=True, help="Docker image URI")
    parser.add_argument("--job-name", default="classifier-training", help="Display name for training job")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--machine-type", default="n1-standard-4", 
                       help="Machine type: n1-standard-4, n1-standard-8, etc. (default: n1-standard-4)")
    parser.add_argument("--gpu-type", default=None, 
                       help="GPU type: NVIDIA_TESLA_K80, NVIDIA_TESLA_T4, NVIDIA_TESLA_V100, etc.")
    parser.add_argument("--gpu-count", type=int, default=0, help="Number of GPUs (default: 0)")
    
    args = parser.parse_args()
    
    submit_training_job(
        project_id=args.project_id,
        region=args.region,
        bucket_name=args.bucket_name,
        image_uri=args.image_uri,
        job_name=args.job_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        machine_type=args.machine_type,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
    )
