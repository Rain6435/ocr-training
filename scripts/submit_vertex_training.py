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
from typing import Optional
from google.cloud import aiplatform


VALID_GPU_TYPES = {
    "NVIDIA_TESLA_K80",
    "NVIDIA_TESLA_P100",
    "NVIDIA_TESLA_T4",
    "NVIDIA_TESLA_V100",
    "NVIDIA_TESLA_P4",
    "NVIDIA_TESLA_A100",
    "NVIDIA_A100_80GB",
    "NVIDIA_L4",
}

DEFAULT_BUCKET_NAME = "ocr-data-70106"


def _normalize_bucket_name(bucket_name: str) -> str:
    """Normalize bucket input to a bare bucket name."""
    return bucket_name.replace("gs://", "").strip("/")


def _normalize_gpu_type(gpu_type: str | None) -> str | None:
    """Normalize GPU aliases to Vertex accelerator enum names."""
    if not gpu_type:
        return None

    normalized = gpu_type.strip().upper().replace("-", "_")

    aliases = {
        "L4": "NVIDIA_L4",
        "TESLA_T4": "NVIDIA_TESLA_T4",
        "T4": "NVIDIA_TESLA_T4",
        "TESLA_V100": "NVIDIA_TESLA_V100",
        "V100": "NVIDIA_TESLA_V100",
        "A100": "NVIDIA_TESLA_A100",
    }
    normalized = aliases.get(normalized, normalized)

    if normalized not in VALID_GPU_TYPES:
        raise ValueError(
            f"Unsupported GPU type '{gpu_type}'. Supported values include: {sorted(VALID_GPU_TYPES)}"
        )

    return normalized


def submit_training_job(
    project_id: str,
    region: str,
    bucket_name: str,
    image_uri: str,
    task: str = "classifier",
    job_name: str = "classifier-training",
    epochs: int = 30,
    batch_size: int = 64,
    machine_type: str = "n1-standard-4",
    gpu_type: Optional[str] = None,
    gpu_count: int = 0,
    grad_clip_norm: float = 1.0,
    metric_decode_strategy: str = "greedy",
    metric_beam_width: int = 10,
    val_metric_batches: int = 10,
    enable_lm_post_correction: bool = False,
    lm_post_correction_mode: str = "compound",
    lm_dictionary_path: str = "data/dictionaries/en_dict.txt",
    lm_historical_dict_path: str = "data/dictionaries/historical_en.txt",
    lm_max_edit_distance: int = 2,
) -> Optional[object]:
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
    
    bucket_name = _normalize_bucket_name(bucket_name)
    gpu_type = _normalize_gpu_type(gpu_type)

    if gpu_type and gpu_count < 1:
        raise ValueError("gpu_count must be >= 1 when gpu_type is provided")

    if not gpu_type and gpu_count != 0:
        raise ValueError("gpu_count must be 0 when gpu_type is not provided")

    # L4 is supported on g2 machine families, not n1.
    if gpu_type == "NVIDIA_L4" and machine_type.startswith("n1-"):
        print("[INFO] NVIDIA_L4 requested with n1 machine type; switching to g2-standard-8.")
        machine_type = "g2-standard-8"

    # Initialize Vertex AI with staging bucket
    aiplatform.init(
        project=project_id, 
        location=region,
        staging_bucket=f"gs://{bucket_name}"
    )
    
    # Create container job
    job = aiplatform.CustomContainerTrainingJob(
        display_name=job_name,
        container_uri=image_uri,
    )
    
    # Dockerfile.training uses ENTRYPOINT ["python", "-m"].
    # Vertex args override CMD, so the module path must be the first arg.
    if task == "ocr":
        args = [
            "src.ocr.custom_model.train",
            "--gcs-bucket", bucket_name,
            "--gcs-processed-prefix", "data/processed",
            "--gcs-raw-prefix", "data/raw",
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
            "--grad-clip-norm", str(grad_clip_norm),
            "--val-metric-batches", str(val_metric_batches),
            "--metric-decode-strategy", metric_decode_strategy,
            "--metric-beam-width", str(metric_beam_width),
        ]
        if enable_lm_post_correction:
            args.extend([
                "--enable-lm-post-correction",
                "--lm-post-correction-mode", lm_post_correction_mode,
                "--lm-dictionary-path", lm_dictionary_path,
                "--lm-historical-dict-path", lm_historical_dict_path,
                "--lm-max-edit-distance", str(lm_max_edit_distance),
            ])
        data_location = f"gs://{bucket_name}/data/processed + gs://{bucket_name}/data/raw"
        result_location = f"gs://{bucket_name}/aiplatform-custom-training-*"
    else:
        args = [
            "src.classifier.train_vertex",
            "--gcs-bucket", bucket_name,
            "--gcs-data-prefix", "data/difficulty_labels",
            "--gcs-model-prefix", "models/classifier",
            "--epochs", str(epochs),
            "--batch-size", str(batch_size),
        ]
        data_location = f"gs://{bucket_name}/data/difficulty_labels"
        result_location = f"gs://{bucket_name}/models/classifier"
    
    print("=" * 80)
    print("SUBMITTING VERTEX AI TRAINING JOB")
    print("=" * 80)
    print(f"Project ID:       {project_id}")
    print(f"Region:           {region}")
    print(f"Task:             {task}")
    print(f"Job Name:         {job_name}")
    print(f"Image:            {image_uri}")
    print(f"Machine Type:     {machine_type}")
    if gpu_type:
        print(f"GPU:              {gpu_count}x {gpu_type}")
    else:
        print(f"GPU:              None (CPU only)")
    print(f"Epochs:           {epochs}")
    print(f"Batch Size:       {batch_size}")
    if task == "ocr":
        print(f"Grad Clip Norm:   {grad_clip_norm}")
        print(f"Metric Decode:    {metric_decode_strategy} (beam_width={metric_beam_width})")
        print(f"Val Metric Batches: {val_metric_batches}")
        print(f"LM Post-Correction: {'enabled' if enable_lm_post_correction else 'disabled'}")
    print(f"Data:             {data_location}")
    print("=" * 80)
    print()
    
    try:
        # Submit job asynchronously
        print()
        print("Starting training job... (this may take 30+ minutes)")
        print()
        model: Optional[object] = None
        
        try:
            model = job.run(
                args=args,
                replica_count=1,
                machine_type=machine_type,
                accelerator_type=gpu_type if gpu_type else "ACCELERATOR_TYPE_UNSPECIFIED",
                accelerator_count=gpu_count if gpu_type else 0,
                sync=False,
                create_request_timeout=300.0,
            )
        except Exception as submit_error:
            print(f"[ERROR] Exception in job.run(): {submit_error}")
            print(f"[ERROR] Exception type: {type(submit_error)}")
            raise
        
        # CustomContainerTrainingJob.run(sync=False) may return None when no model is uploaded.
        # The created training pipeline resource is available on the job object.
        try:
            resource_name = job.resource_name
        except RuntimeError:
            resource_name = None

        print(f"\n[DEBUG] job.run() returned: {type(model)}")
        print(f"[DEBUG] training pipeline resource: {resource_name}")
        
        print()
        print("=" * 80)
        print("TRAINING JOB SUBMITTED SUCCESSFULLY!")
        print("=" * 80)
        print()
        if resource_name:
            job_id = resource_name.split('/')[-1]
            print(f"Job Resource Name:")
            print(f"  {resource_name}")
            print()
            print("Tip: This is a TrainingPipeline resource. A CustomJob appears shortly after pipeline start.")
            print()
            print("Monitor your training pipeline/job:")
            print(f"  Web UI: https://console.cloud.google.com/vertex-ai/training/custom-jobs?project={project_id}")
            print()
            print("Stream logs in real-time:")
            print(f"  gcloud ai custom-jobs list --region={region} --project={project_id}")
            print(f"  gcloud ai training-pipelines list --region={region} --project={project_id}")
            print(f"  gcloud ai training-pipelines describe {job_id} --region={region} --project={project_id}")
        else:
            print("Note: Job submitted in background. Check the output directory:")
            print(f"     gs://{bucket_name}/aiplatform-custom-training-*")
        print()
        print("Download results when complete:")
        print(f"  gsutil -m cp -r {result_location} ./trained_models/")
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
    parser.add_argument("--task", choices=["classifier", "ocr"], default="classifier", help="Training task to run")
    parser.add_argument(
        "--bucket-name",
        default=DEFAULT_BUCKET_NAME,
        help=f"GCS bucket name (default: {DEFAULT_BUCKET_NAME})",
    )
    parser.add_argument("--image-uri", required=True, help="Docker image URI")
    parser.add_argument("--job-name", default="classifier-training", help="Display name for training job")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs (default: 30)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--machine-type", default="n1-standard-4", 
                       help="Machine type: n1-standard-4, n1-standard-8, etc. (default: n1-standard-4)")
    parser.add_argument("--gpu-type", default=None, 
                       help="GPU type: NVIDIA_TESLA_K80, NVIDIA_TESLA_T4, NVIDIA_TESLA_V100, etc.")
    parser.add_argument("--gpu-count", type=int, default=0, help="Number of GPUs (default: 0)")
    parser.add_argument("--grad-clip-norm", type=float, default=1.0, help="Gradient clipping norm for OCR training")
    parser.add_argument(
        "--metric-decode-strategy",
        choices=["greedy", "beam"],
        default="greedy",
        help="Decoding strategy for OCR validation/test metrics",
    )
    parser.add_argument("--metric-beam-width", type=int, default=10, help="Beam width for beam decoding metrics")
    parser.add_argument("--val-metric-batches", type=int, default=10, help="Validation batches for CER/WER")
    parser.add_argument(
        "--enable-lm-post-correction",
        action="store_true",
        help="Enable LM-style post correction for OCR metric computation",
    )
    parser.add_argument(
        "--lm-post-correction-mode",
        choices=["compound", "word"],
        default="compound",
        help="LM post correction mode",
    )
    parser.add_argument(
        "--lm-dictionary-path",
        default="data/dictionaries/en_dict.txt",
        help="Main dictionary path for LM post-correction",
    )
    parser.add_argument(
        "--lm-historical-dict-path",
        default="data/dictionaries/historical_en.txt",
        help="Historical dictionary path for LM post-correction",
    )
    parser.add_argument(
        "--lm-max-edit-distance",
        type=int,
        default=2,
        help="Max edit distance for LM post-correction",
    )
    
    args = parser.parse_args()
    
    submit_training_job(
        project_id=args.project_id,
        region=args.region,
        bucket_name=args.bucket_name,
        image_uri=args.image_uri,
        task=args.task,
        job_name=args.job_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        machine_type=args.machine_type,
        gpu_type=args.gpu_type,
        gpu_count=args.gpu_count,
        grad_clip_norm=args.grad_clip_norm,
        metric_decode_strategy=args.metric_decode_strategy,
        metric_beam_width=args.metric_beam_width,
        val_metric_batches=args.val_metric_batches,
        enable_lm_post_correction=args.enable_lm_post_correction,
        lm_post_correction_mode=args.lm_post_correction_mode,
        lm_dictionary_path=args.lm_dictionary_path,
        lm_historical_dict_path=args.lm_historical_dict_path,
        lm_max_edit_distance=args.lm_max_edit_distance,
    )