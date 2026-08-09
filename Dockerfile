FROM python:3.11-slim

WORKDIR /app

# System deps for transformers/onnx
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/ ./gateway/
COPY router_model/ ./router_model/
COPY gateway-config.json gateway-policy.json ./
COPY tests/ ./tests/

RUN useradd --create-home --uid 10001 glint && chown -R glint:glint /app
USER glint

EXPOSE 8076

CMD ["python", "-m", "gateway.app", "--host", "0.0.0.0", "--port", "8076"]
