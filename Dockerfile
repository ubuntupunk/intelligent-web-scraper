# Multi-stage Dockerfile for Intelligent Web Scraper
# Optimized for production deployment with security best practices

# Build stage
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=0.1.0
ARG VCS_REF

# Add metadata
LABEL org.opencontainers.image.title="Intelligent Web Scraper" \
      org.opencontainers.image.description="An advanced example application for the Atomic Agents framework" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Atomic Agents Team" \
      org.opencontainers.image.source="https://github.com/atomic-agents/intelligent-web-scraper" \
      org.opencontainers.image.documentation="https://github.com/atomic-agents/intelligent-web-scraper/docs" \
      org.opencontainers.image.licenses="MIT"

# Install system dependencies for building
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==1.7.1

# Set Poetry configuration
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --only=main --no-root && \
    rm -rf $POETRY_CACHE_DIR

# Runtime stage
FROM python:3.11-slim as runtime

# Set runtime arguments
ARG APP_USER=scraper
ARG APP_UID=1000
ARG APP_GID=1000

# Install runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user
RUN groupadd -g ${APP_GID} ${APP_USER} && \
    useradd -u ${APP_UID} -g ${APP_GID} -m -s /bin/bash ${APP_USER}

# Set working directory
WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder --chown=${APP_USER}:${APP_USER} /app/.venv /app/.venv

# Copy application code
COPY --chown=${APP_USER}:${APP_USER} . .

# Create necessary directories
RUN mkdir -p /app/results /app/logs /app/config && \
    chown -R ${APP_USER}:${APP_USER} /app

# Set environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RESULTS_DIRECTORY="/app/results" \
    LOG_DIRECTORY="/app/logs"

# Switch to non-root user
USER ${APP_USER}

# Install the application
RUN pip install --no-deps -e .

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import intelligent_web_scraper; print('OK')" || exit 1

# Expose port (if running as web service)
EXPOSE 8000

# Create entrypoint script
COPY --chown=${APP_USER}:${APP_USER} docker-entrypoint.sh /app/
RUN chmod +x /app/docker-entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command
CMD ["intelligent-web-scraper"]