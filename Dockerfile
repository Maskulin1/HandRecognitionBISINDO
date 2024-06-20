# Use the official lightweight Python image.
# https://hub.docker.com/_/python
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install dependencies
COPY requirements.txt .
COPY packages.txt .
RUN apt-get update && \
    xargs -a packages.txt apt-get install -y && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . /app
WORKDIR /app

# Expose the port Streamlit is running on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "infrence_classifier.py"]
