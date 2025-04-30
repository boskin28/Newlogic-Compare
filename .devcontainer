// .devcontainer/devcontainer.json
{
  "name": "Python Streamlit DevContainer",
  "build": {
    "dockerfile": "Dockerfile",
    "context": "..",
    "args": {
      "VARIANT": "3.11"
    }
  },
  "workspaceFolder": "/workspace",
  "extensions": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-azuretools.vscode-docker"
  ],
  "settings": {
    "python.pythonPath": "/usr/local/bin/python",
    "terminal.integrated.shell.linux": "/bin/bash"
  },
  "forwardPorts": [8501],
  "postStartCommand": "pip install -r requirements.txt",
  "remoteUser": "vscode"
}

// .devcontainer/Dockerfile
FROM mcr.microsoft.com/vscode/devcontainers/python:${VARIANT}

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create app directory
WORKDIR /workspace

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Default command: launch Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
