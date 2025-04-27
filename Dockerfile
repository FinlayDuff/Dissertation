FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy poetry files
COPY pyproject.toml poetry.lock ./

# Configure poetry to not create virtual environment in container
RUN poetry config virtualenvs.create false

# Install dependencies
RUN poetry install --without dev --no-root

# Copy application code
COPY . .

# Create volume mount points
VOLUME ["/app/results"]

# Set environment variables
ENV PYTHONPATH=/app

# Default command
CMD ["poetry", "run", "python", "main.py"]