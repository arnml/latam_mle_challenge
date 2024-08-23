# syntax=docker/dockerfile:1.2
# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . /usr/src/app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Define environment variable
ENV PORT 8080

# Run the application
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8080"]
