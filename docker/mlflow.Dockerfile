# Base image with MLflow
FROM ghcr.io/mlflow/mlflow:latest-full

# Install pure-python drivers for GCP environment
RUN pip install --no-cache-dir \
    google-cloud-storage \
    pg8000 \
    sqlalchemy-collectd

# Cloud Run listens on port 8080 by default
ENV PORT 8080
EXPOSE 8080

# Disable Nginx for Cloud Run compatibility
ENV DISABLE_NGINX true

# Command to run the tracking server
# We use --serve-artifacts to proxy GCS requests through the server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "8080", "--serve-artifacts"]
