# Use an official Python runtime as the base image
FROM python:3.11-slim

# Set the working directory
RUN mkdir /app
WORKDIR /app

# Copy the Python script and requirements file into the container
COPY . /app
# Copy necessary files (adjust based on your setup)
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set the entry point for the container
ENTRYPOINT ["python3", "clustering_ctrp_blosum.py"]