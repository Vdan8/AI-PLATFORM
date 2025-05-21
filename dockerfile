# Dockerfile for your AI Tool Sandbox

# Use the official Python 3.11 slim-bookworm image as the base.
# This is the image specified in your .env: MCP_SANDBOX_IMAGE=python:3.11-slim-bookworm
FROM python:3.11-slim-bookworm

# Set the working directory inside the container.
# Your `sandbox_service.py` mounts the tool script into `/sandbox`.
WORKDIR /sandbox

# Install the necessary Python packages.
# `aiohttp` is for making HTTP requests (like in your weather tool).
# `nltk` is for natural language processing (like in your summarization tool).
# `--no-cache-dir` is used to prevent pip from storing downloaded packages,
# which helps keep the image size smaller.
RUN pip install --no-cache-dir aiohttp nltk

# (Optional but Recommended for NLTK) Download NLTK data.
# Many NLTK functions (especially for summarization) require specific data files
# (e.g., 'punkt' for tokenization, 'stopwords' for text cleaning).
# Downloading them during the image build process ensures they are always available
# inside the sandbox container without needing to be downloaded repeatedly.
# `quiet=True` prevents verbose output during download.
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('stopwords', quiet=True)"

# You typically don't need a CMD or ENTRYPOINT here because your `sandbox_service.py`
# explicitly tells the container what command to run (`python /sandbox/tool_script.py`).