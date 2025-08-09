# Deployment Guide

This guide provides comprehensive instructions for deploying the Intelligent Web Scraper in various environments, from development to production.

## Table of Contents

- [Quick Start](#quick-start)
- [Environment Setup](#environment-setup)
- [Configuration Management](#configuration-management)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Health Checks](#health-checks)
- [Monitoring and Logging](#monitoring-and-logging)
- [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Python 3.11 or higher
- Poetry (recommended) or pip
- OpenAI API key or compatible LLM service

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/atomic-agents/intelligent-web-scraper.git
   cd intelligent-web-scraper
   ```

2. **Install dependencies:**
   ```bash
   # Using Poetry (recommended)
   poetry install
   
   # Or using pip
   pip install -e .
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Run the application:**
   ```bash
   # Using Poetry
   poetry run intelligent-web-scraper
   
   # Or directly
   intelligent-web-scraper
   ```

## Environment Setup

### Development Environment

1. **Install development dependencies:**
   ```bash
   poetry install --with dev
   ```

2. **Set up pre-commit hooks:**
   ```bash
   poetry run pre-commit install
   ```

3. **Run tests:**
   ```bash
   poetry run pytest
   ```

### Staging Environment

1. **Use production-like configuration:**
   ```bash
   export ENVIRONMENT=staging
   export ORCHESTRATOR_MODEL=gpt-4o-mini
   export ENABLE_MONITORING=true
   export MAX_CONCURRENT_REQUESTS=3
   ```

2. **Enable detailed logging:**
   ```bash
   export LOG_LEVEL=INFO
   export LOG_FORMAT=json
   ```

### Production Environment

1. **Use optimized configuration:**
   ```bash
   export ENVIRONMENT=production
   export ORCHESTRATOR_MODEL=gpt-4
   export PLANNING_AGENT_MODEL=gpt-4o-mini
   export MAX_CONCURRENT_REQUESTS=10
   export ENABLE_MONITORING=true
   export MONITORING_INTERVAL=5.0
   ```

2. **Security considerations:**
   ```bash
   export RESPECT_ROBOTS_TXT=true
   export ENABLE_RATE_LIMITING=true
   export REQUEST_DELAY=2.0
   ```

## Configuration Management

### Environment Variables

The application supports comprehensive configuration through environment variables:

#### Core Configuration
```bash
# LLM Models
ORCHESTRATOR_MODEL=gpt-4o-mini          # Main orchestrator model
PLANNING_AGENT_MODEL=gpt-4o-mini        # Planning agent model

# OpenAI Configuration
OPENAI_API_KEY=your_api_key_here        # Required for OpenAI models
OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: custom endpoint
```

#### Scraping Configuration
```bash
# Quality and Performance
DEFAULT_QUALITY_THRESHOLD=50.0          # Quality threshold (0-100)
MAX_CONCURRENT_REQUESTS=5               # Concurrent HTTP requests
REQUEST_DELAY=1.0                       # Delay between requests (seconds)

# Output Configuration
EXPORT_FORMAT=json                      # Default export format
RESULTS_DIRECTORY=./results             # Results storage directory
```

#### Compliance and Ethics
```bash
# Ethical Scraping
RESPECT_ROBOTS_TXT=true                 # Respect robots.txt files
ENABLE_RATE_LIMITING=true               # Enable automatic rate limiting

# User Agent
USER_AGENT="Intelligent-Web-Scraper/0.1.0 (+https://github.com/atomic-agents/intelligent-web-scraper)"
```

#### Monitoring and Performance
```bash
# Monitoring
ENABLE_MONITORING=true                  # Enable real-time monitoring
MONITORING_INTERVAL=1.0                 # Update interval (seconds)

# Concurrency
MAX_INSTANCES=5                         # Maximum scraper instances
MAX_WORKERS=10                          # Maximum worker threads
MAX_ASYNC_TASKS=50                      # Maximum async tasks
```

### Configuration Files

#### JSON Configuration
Create a `config.json` file for complex configurations:

```json
{
  "orchestrator_model": "gpt-4",
  "planning_agent_model": "gpt-4o-mini",
  "default_quality_threshold": 75.0,
  "max_concurrent_requests": 8,
  "request_delay": 1.5,
  "default_export_format": "json",
  "results_directory": "/app/results",
  "respect_robots_txt": true,
  "enable_rate_limiting": true,
  "enable_monitoring": true,
  "monitoring_interval": 2.0,
  "max_instances": 3,
  "max_workers": 8,
  "max_async_tasks": 30
}
```

Use with:
```bash
intelligent-web-scraper --config config.json
```

#### Environment File (.env)
```bash
# .env file
ORCHESTRATOR_MODEL=gpt-4
PLANNING_AGENT_MODEL=gpt-4o-mini
OPENAI_API_KEY=your_api_key_here
DEFAULT_QUALITY_THRESHOLD=75.0
MAX_CONCURRENT_REQUESTS=8
ENABLE_MONITORING=true
RESULTS_DIRECTORY=/app/results
```

## Docker Deployment

### Basic Docker Setup

1. **Build the Docker image:**
   ```bash
   docker build -t intelligent-web-scraper:latest .
   ```

2. **Run the container:**
   ```bash
   docker run -it \
     -e OPENAI_API_KEY=your_api_key \
     -e ORCHESTRATOR_MODEL=gpt-4o-mini \
     -v $(pwd)/results:/app/results \
     intelligent-web-scraper:latest
   ```

### Docker Compose

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  intelligent-web-scraper:
    build: .
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - ORCHESTRATOR_MODEL=gpt-4o-mini
      - PLANNING_AGENT_MODEL=gpt-4o-mini
      - DEFAULT_QUALITY_THRESHOLD=75.0
      - MAX_CONCURRENT_REQUESTS=5
      - ENABLE_MONITORING=true
      - RESULTS_DIRECTORY=/app/results
    volumes:
      - ./results:/app/results
      - ./config:/app/config
    stdin_open: true
    tty: true
    restart: unless-stopped

  # Optional: Add monitoring services
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    
  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-storage:/var/lib/grafana

volumes:
  grafana-storage:
```

Run with:
```bash
docker-compose up -d
```

### Production Docker Configuration

For production deployments, use multi-stage builds and optimized configurations:

```dockerfile
# Production Dockerfile
FROM python:3.11-slim as builder

WORKDIR /app
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --only=main --no-dev

FROM python:3.11-slim as runtime

# Create non-root user
RUN useradd --create-home --shell /bin/bash scraper

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy application
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY . .

# Set ownership and permissions
RUN chown -R scraper:scraper /app
USER scraper

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import intelligent_web_scraper; print('OK')" || exit 1

# Default command
CMD ["intelligent-web-scraper"]
```

## Production Deployment

### System Requirements

#### Minimum Requirements
- **CPU:** 2 cores
- **RAM:** 4 GB
- **Storage:** 10 GB available space
- **Network:** Stable internet connection

#### Recommended Requirements
- **CPU:** 4+ cores
- **RAM:** 8+ GB
- **Storage:** 50+ GB SSD
- **Network:** High-bandwidth connection for concurrent scraping

### Deployment Strategies

#### 1. Systemd Service (Linux)

Create `/etc/systemd/system/intelligent-web-scraper.service`:

```ini
[Unit]
Description=Intelligent Web Scraper
After=network.target

[Service]
Type=simple
User=scraper
Group=scraper
WorkingDirectory=/opt/intelligent-web-scraper
Environment=PATH=/opt/intelligent-web-scraper/.venv/bin
EnvironmentFile=/opt/intelligent-web-scraper/.env
ExecStart=/opt/intelligent-web-scraper/.venv/bin/intelligent-web-scraper
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=/opt/intelligent-web-scraper/results

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable intelligent-web-scraper
sudo systemctl start intelligent-web-scraper
sudo systemctl status intelligent-web-scraper
```

#### 2. Kubernetes Deployment

Create `k8s-deployment.yaml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: intelligent-web-scraper
  labels:
    app: intelligent-web-scraper
spec:
  replicas: 3
  selector:
    matchLabels:
      app: intelligent-web-scraper
  template:
    metadata:
      labels:
        app: intelligent-web-scraper
    spec:
      containers:
      - name: intelligent-web-scraper
        image: intelligent-web-scraper:latest
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        - name: ORCHESTRATOR_MODEL
          value: "gpt-4o-mini"
        - name: MAX_CONCURRENT_REQUESTS
          value: "5"
        - name: ENABLE_MONITORING
          value: "true"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: results-storage
          mountPath: /app/results
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import intelligent_web_scraper; print('OK')"
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import intelligent_web_scraper; print('OK')"
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: results-storage
        persistentVolumeClaim:
          claimName: results-pvc
---
apiVersion: v1
kind: Secret
metadata:
  name: openai-secret
type: Opaque
data:
  api-key: <base64-encoded-api-key>
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: results-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
```

Deploy:
```bash
kubectl apply -f k8s-deployment.yaml
```

### Load Balancing and Scaling

#### Horizontal Scaling
```bash
# Scale deployment
kubectl scale deployment intelligent-web-scraper --replicas=5

# Auto-scaling
kubectl autoscale deployment intelligent-web-scraper --cpu-percent=70 --min=2 --max=10
```

#### Load Balancer Configuration (nginx)
```nginx
upstream intelligent_web_scraper {
    least_conn;
    server 10.0.1.10:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.11:8000 weight=1 max_fails=3 fail_timeout=30s;
    server 10.0.1.12:8000 weight=1 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name scraper.example.com;
    
    location / {
        proxy_pass http://intelligent_web_scraper;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_connect_timeout 30s;
        proxy_send_timeout 30s;
        proxy_read_timeout 30s;
    }
}
```

## Health Checks

### Application Health Check

The application includes built-in health check capabilities:

```python
# health_check.py
import sys
import asyncio
from intelligent_web_scraper.config import IntelligentScrapingConfig
from intelligent_web_scraper.agents.orchestrator import IntelligentScrapingOrchestrator

async def health_check():
    """Perform application health check."""
    try:
        # Test configuration loading
        config = IntelligentScrapingConfig.from_env()
        
        # Test orchestrator initialization
        orchestrator = IntelligentScrapingOrchestrator(config=config)
        
        # Test basic functionality
        test_input = {
            "scraping_request": "Health check test",
            "target_url": "https://httpbin.org/json"
        }
        
        # This would be a minimal test - in production you might want
        # a dedicated health check endpoint
        print("Health check passed")
        return True
        
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    result = asyncio.run(health_check())
    sys.exit(0 if result else 1)
```

### External Health Checks

#### HTTP Health Check Endpoint
If running as a web service, implement a health endpoint:

```python
from fastapi import FastAPI, HTTPException
from intelligent_web_scraper import validate_ecosystem_compatibility

app = FastAPI()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check ecosystem compatibility
        compatibility = validate_ecosystem_compatibility()
        
        if not all(compatibility.values()):
            raise HTTPException(status_code=503, detail="Service unhealthy")
        
        return {
            "status": "healthy",
            "compatibility": compatibility,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Health check failed: {e}")

@app.get("/ready")
async def readiness_check():
    """Readiness check endpoint."""
    # Check if service is ready to accept requests
    return {"status": "ready"}
```

#### Monitoring Integration
```bash
# Prometheus health check
curl -f http://localhost:8000/health || exit 1

# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 30

# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```

## Monitoring and Logging

### Logging Configuration

#### Structured Logging
```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/intelligent-web-scraper.log')
    ]
)

# Use JSON formatter for production
if os.getenv('ENVIRONMENT') == 'production':
    for handler in logging.root.handlers:
        handler.setFormatter(JSONFormatter())
```

#### Log Rotation
```bash
# logrotate configuration: /etc/logrotate.d/intelligent-web-scraper
/var/log/intelligent-web-scraper.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 644 scraper scraper
    postrotate
        systemctl reload intelligent-web-scraper
    endscript
}
```

### Metrics Collection

#### Prometheus Metrics
```python
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define metrics
scraping_requests_total = Counter('scraping_requests_total', 'Total scraping requests', ['status'])
scraping_duration_seconds = Histogram('scraping_duration_seconds', 'Scraping duration')
active_scrapers = Gauge('active_scrapers', 'Number of active scrapers')
quality_score = Histogram('quality_score', 'Quality scores of scraped data')

# Use in application
@scraping_duration_seconds.time()
async def scrape_with_metrics(request):
    active_scrapers.inc()
    try:
        result = await scrape(request)
        scraping_requests_total.labels(status='success').inc()
        quality_score.observe(result.quality_score)
        return result
    except Exception as e:
        scraping_requests_total.labels(status='error').inc()
        raise
    finally:
        active_scrapers.dec()

# Start metrics server
start_http_server(8001)
```

#### Grafana Dashboard
```json
{
  "dashboard": {
    "title": "Intelligent Web Scraper",
    "panels": [
      {
        "title": "Scraping Requests Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(scraping_requests_total[5m])",
            "legendFormat": "{{status}}"
          }
        ]
      },
      {
        "title": "Average Quality Score",
        "type": "stat",
        "targets": [
          {
            "expr": "avg(quality_score)",
            "legendFormat": "Quality Score"
          }
        ]
      },
      {
        "title": "Active Scrapers",
        "type": "stat",
        "targets": [
          {
            "expr": "active_scrapers",
            "legendFormat": "Active"
          }
        ]
      }
    ]
  }
}
```

## Troubleshooting

### Common Issues

#### 1. API Key Issues
```bash
# Check API key configuration
echo $OPENAI_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

#### 2. Memory Issues
```bash
# Monitor memory usage
docker stats intelligent-web-scraper

# Adjust memory limits
docker run --memory=4g intelligent-web-scraper
```

#### 3. Rate Limiting
```bash
# Check rate limiting configuration
intelligent-web-scraper --config - <<EOF
{
  "enable_rate_limiting": true,
  "request_delay": 2.0,
  "max_concurrent_requests": 3
}
EOF
```

#### 4. Network Connectivity
```bash
# Test network connectivity
curl -I https://example.com

# Check DNS resolution
nslookup example.com

# Test with proxy
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080
```

### Debug Mode

Enable debug mode for detailed troubleshooting:

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with debug flag
intelligent-web-scraper --debug --verbose

# Check system compatibility
python -c "
from intelligent_web_scraper import validate_ecosystem_compatibility
import json
print(json.dumps(validate_ecosystem_compatibility(), indent=2))
"
```

### Performance Tuning

#### 1. Optimize Concurrency
```bash
# Tune for high-throughput scenarios
export MAX_CONCURRENT_REQUESTS=20
export MAX_WORKERS=16
export MAX_ASYNC_TASKS=100

# Tune for memory-constrained environments
export MAX_CONCURRENT_REQUESTS=2
export MAX_WORKERS=4
export MAX_ASYNC_TASKS=10
```

#### 2. Model Selection
```bash
# Fast but less accurate
export ORCHESTRATOR_MODEL=gpt-4o-mini
export PLANNING_AGENT_MODEL=gpt-4o-mini

# Slower but more accurate
export ORCHESTRATOR_MODEL=gpt-4
export PLANNING_AGENT_MODEL=gpt-4
```

#### 3. Quality vs Speed Trade-offs
```bash
# Prioritize speed
export DEFAULT_QUALITY_THRESHOLD=30.0
export REQUEST_DELAY=0.5

# Prioritize quality
export DEFAULT_QUALITY_THRESHOLD=80.0
export REQUEST_DELAY=2.0
```

### Support and Maintenance

#### Log Analysis
```bash
# Analyze error patterns
grep "ERROR" /var/log/intelligent-web-scraper.log | \
  awk '{print $4}' | sort | uniq -c | sort -nr

# Monitor performance
grep "processing_time" /var/log/intelligent-web-scraper.log | \
  awk '{print $NF}' | sort -n | tail -10
```

#### Backup and Recovery
```bash
# Backup configuration
tar -czf config-backup-$(date +%Y%m%d).tar.gz \
  .env config/ results/

# Backup results
rsync -av results/ backup/results/

# Database backup (if using database)
pg_dump scraper_db > backup/scraper_db_$(date +%Y%m%d).sql
```

For additional support, please refer to:
- [GitHub Issues](https://github.com/atomic-agents/intelligent-web-scraper/issues)
- [Documentation](https://github.com/atomic-agents/intelligent-web-scraper/docs)
- [Community Forum](https://github.com/atomic-agents/intelligent-web-scraper/discussions)