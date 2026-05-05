FROM python:3.11-slim AS builder

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system dependencies
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libgomp1 && \
#     apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Copy installed Python packages from builder
COPY --from=builder /usr/local /usr/local

# Application code
COPY . .

# Create data directories
RUN mkdir -p data/models

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
