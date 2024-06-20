# Use the official lightweight Python image.
FROM python:3.11.5

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
COPY packages.txt .
RUN apt-get update && \
    xargs -a packages.txt apt-get install -y && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the port Streamlit is running on
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "infrence_classifier.py"]
