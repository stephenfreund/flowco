# Use the official lightweight Python image.
FROM python:3.11-slim

# Install graphviz, build tools, Node.js, and npm
RUN apt-get update && \
    apt-get install -y --no-install-recommends graphviz build-essential make curl && \
    # Install Node.js (version 16.x)
    curl -fsSL https://deb.nodesource.com/setup_16.x | bash - && \
    apt-get install -y nodejs && \
    # Clean up to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Add arguments for commit SHA and build date
ARG COMMIT_SHA
ARG BUILD_DATE
ARG RELEASE_VERSION

# Set environment variables for the app
ENV COMMIT_SHA=$COMMIT_SHA
ENV BUILD_DATE=$BUILD_DATE
ENV RELEASE_VERSION=$RELEASE_VERSION

# Set work directory
WORKDIR /app

# Update pip and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Navigate to mxgraph_component and run `make build`
RUN make -C src/flowco/mxgraph_component build

# Install the mxgraph_component package
RUN pip3 install src/mxgraph_component/dist/mxgraph_component-0.0.1-py3-none-any.whl -e .

# Ensure the .streamlit directory exists
RUN mkdir -p /app/.streamlit

# Copy the secrets.toml file into the .streamlit directory
COPY .streamlit/secrets.toml /app/.streamlit/secrets.toml
COPY .streamlit/config.toml /app/.streamlit/config.toml

# Modify the environment setting in secrets.toml
RUN sed -i 's/FLOWCO_ENVIRONMENT = "local"/FLOWCO_ENVIRONMENT = "production"/' /app/.streamlit/secrets.toml

# Expose the port Streamlit runs on
EXPOSE 80

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/flowco/flowco.py", "--server.port=80", "--server.address=0.0.0.0"]
