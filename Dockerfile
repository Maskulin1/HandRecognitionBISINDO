# Start with the official Python image.
FROM python:3.11.5

# Set environment variables to ensure Python runs in a consistent environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies from packages.txt
COPY packages.txt .
RUN apt-get update && \
    xargs -a packages.txt apt-get install -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application directory into the Docker image
COPY . .

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run Streamlit app
ENTRYPOINT ["streamlit", "run", "infrence_classifier.py", "--server.port=8501", "--server.address=0.0.0.0"]
