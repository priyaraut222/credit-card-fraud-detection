# Credit Card Fraud Detection -- Streamlit Dashboard
#
# This image serves the DASHBOARD ONLY (app.py). It expects models/ to
# already contain trained artifacts -- run train_models.py locally first
# and make sure models/ is present in the build context, since training
# (especially the hyperparameter search) is too slow to redo on every
# container build.
#
# Build:
#   docker build -t fraud-dashboard .
# Run:
#   docker run -p 8501:8501 fraud-dashboard

FROM python:3.12-slim

WORKDIR /app

# System deps needed by some scientific-python wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv pip install --system -e .

COPY src/ src/
COPY app.py ./
COPY models/ models/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
