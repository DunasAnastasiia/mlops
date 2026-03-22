FROM python:3.10-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim AS final

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY src/ ./src/
COPY conf/ ./conf/
COPY .dvc/ ./.dvc/
COPY params.yaml .
COPY dvc.yaml .
COPY dvc.lock .
COPY requirements.txt .

RUN mkdir -p models data

CMD ["python", "src/train.py"]
