FROM python:3.13-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt
COPY src/ src/
COPY models/ models/
COPY data/ data/
ENV PYTHONPATH=/app/src ETA_MODEL=lgb ETA_REPLAY_UNTIL=2025-03-27T18:00:00
EXPOSE 8000
CMD ["uvicorn", "eta.serve:app", "--host", "0.0.0.0", "--port", "8000"]
