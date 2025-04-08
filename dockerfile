# Use the official Python 3.11.4 slim image as the base for the build stage
FROM python:3.11.4-slim AS builder

# Create a directory to hold installed Python packages
RUN mkdir -p /install

# Copy the requirements file into the image
COPY requirements.txt /tmp/requirements.txt

# Install the Python dependencies from requirements.txt into /install
# --no-cache-dir avoids caching to keep the image small
# --target specifies a custom install location
RUN pip install --no-cache-dir --target=/install -r /tmp/requirements.txt


# Start a new, clean image for the final stage to keep it lightweight
FROM python:3.11.4-slim

# Copy the previously installed dependencies from the builder stage
COPY --from=builder /install /app/libs

# Copy the application code into the image
COPY . /app

# Set the environment variable so Python knows where to find the installed libraries
ENV PYTHONPATH=/app/libs

# Set the working directory for any subsequent commands
WORKDIR /app
