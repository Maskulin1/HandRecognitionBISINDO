# temp stage
FROM python:3.11-slim as builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install dependencies from packages.txt
RUN apt-get update && \
    xargs apt-get install -y --no-install-recommends < /app/packages.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install -r requirements.txt

# final stage
FROM python:3.11-slim

# Copy the installed packages from builder stage
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

ENV PATH="/opt/venv/bin:$PATH"

# Copy the application code to the container
COPY . .

# Ensure packages.txt is used in the final image
RUN apt-get update && \
    xargs apt-get install -y --no-install-recommends < /app/packages.txt

CMD ["python", "inference_classifier.py"]
