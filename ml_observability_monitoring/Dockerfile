FROM python:3.11-slim

WORKDIR /app

COPY ml_observability_monitoring/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml_evaluation_pipeline/ ./ml_evaluation_pipeline/
COPY ml_observability_monitoring/ ./ml_observability_monitoring/

EXPOSE 50052
CMD ["python", "ml_observability_monitoring/grpc_server.py"]
