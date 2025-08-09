#!/bin/bash
set -e

# Docker entrypoint script for Intelligent Web Scraper
# Handles initialization, configuration validation, and graceful startup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to validate environment variables
validate_environment() {
    log_info "Validating environment configuration..."
    
    local validation_errors=0
    
    # Check required environment variables
    if [[ -z "${OPENAI_API_KEY:-}" ]]; then
        log_warn "OPENAI_API_KEY not set. Some functionality may be limited."
    fi
    
    # Validate numeric values
    if [[ -n "${DEFAULT_QUALITY_THRESHOLD:-}" ]]; then
        if ! [[ "${DEFAULT_QUALITY_THRESHOLD}" =~ ^[0-9]+\.?[0-9]*$ ]] || \
           (( $(echo "${DEFAULT_QUALITY_THRESHOLD} < 0" | bc -l) )) || \
           (( $(echo "${DEFAULT_QUALITY_THRESHOLD} > 100" | bc -l) )); then
            log_error "DEFAULT_QUALITY_THRESHOLD must be between 0 and 100"
            validation_errors=$((validation_errors + 1))
        fi
    fi
    
    if [[ -n "${MAX_CONCURRENT_REQUESTS:-}" ]]; then
        if ! [[ "${MAX_CONCURRENT_REQUESTS}" =~ ^[0-9]+$ ]] || \
           (( MAX_CONCURRENT_REQUESTS <= 0 )); then
            log_error "MAX_CONCURRENT_REQUESTS must be a positive integer"
            validation_errors=$((validation_errors + 1))
        fi
    fi
    
    if [[ -n "${REQUEST_DELAY:-}" ]]; then
        if ! [[ "${REQUEST_DELAY}" =~ ^[0-9]+\.?[0-9]*$ ]] || \
           (( $(echo "${REQUEST_DELAY} < 0" | bc -l) )); then
            log_error "REQUEST_DELAY must be non-negative"
            validation_errors=$((validation_errors + 1))
        fi
    fi
    
    # Validate export format
    if [[ -n "${EXPORT_FORMAT:-}" ]]; then
        case "${EXPORT_FORMAT}" in
            json|csv|markdown|excel)
                ;;
            *)
                log_error "EXPORT_FORMAT must be one of: json, csv, markdown, excel"
                validation_errors=$((validation_errors + 1))
                ;;
        esac
    fi
    
    if (( validation_errors > 0 )); then
        log_error "Environment validation failed with ${validation_errors} error(s)"
        exit 1
    fi
    
    log_success "Environment validation passed"
}

# Function to create necessary directories
create_directories() {
    log_info "Creating necessary directories..."
    
    local dirs=(
        "${RESULTS_DIRECTORY:-/app/results}"
        "${LOG_DIRECTORY:-/app/logs}"
        "/app/config"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Created directory: $dir"
        fi
    done
    
    log_success "Directory creation completed"
}

# Function to check system resources
check_system_resources() {
    log_info "Checking system resources..."
    
    # Check available memory
    local available_memory_kb
    available_memory_kb=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
    local available_memory_mb=$((available_memory_kb / 1024))
    
    log_info "Available memory: ${available_memory_mb} MB"
    
    if (( available_memory_mb < 1024 )); then
        log_warn "Low memory detected (${available_memory_mb} MB). Consider increasing memory allocation."
    fi
    
    # Check available disk space
    local available_disk_gb
    available_disk_gb=$(df /app | tail -1 | awk '{print int($4/1024/1024)}')
    
    log_info "Available disk space: ${available_disk_gb} GB"
    
    if (( available_disk_gb < 1 )); then
        log_warn "Low disk space detected (${available_disk_gb} GB). Consider increasing disk allocation."
    fi
    
    log_success "System resource check completed"
}

# Function to test application health
test_application_health() {
    log_info "Testing application health..."
    
    # Test Python import
    if ! python -c "import intelligent_web_scraper" 2>/dev/null; then
        log_error "Failed to import intelligent_web_scraper module"
        exit 1
    fi
    
    # Test configuration loading
    if ! python -c "from intelligent_web_scraper.config import IntelligentScrapingConfig; IntelligentScrapingConfig.from_env()" 2>/dev/null; then
        log_error "Failed to load configuration"
        exit 1
    fi
    
    # Test ecosystem compatibility
    if ! python -c "from intelligent_web_scraper import validate_ecosystem_compatibility; assert all(validate_ecosystem_compatibility().values())" 2>/dev/null; then
        log_warn "Some ecosystem compatibility checks failed. Application may have limited functionality."
    fi
    
    log_success "Application health check passed"
}

# Function to display startup information
display_startup_info() {
    log_info "=== Intelligent Web Scraper ==="
    log_info "Version: 0.1.0"
    log_info "Environment: ${ENVIRONMENT:-development}"
    log_info "Python version: $(python --version)"
    log_info "Working directory: $(pwd)"
    log_info "User: $(whoami)"
    log_info "UID: $(id -u)"
    log_info "GID: $(id -g)"
    
    # Display configuration summary
    log_info "Configuration summary:"
    log_info "  Orchestrator Model: ${ORCHESTRATOR_MODEL:-gpt-4o-mini}"
    log_info "  Planning Agent Model: ${PLANNING_AGENT_MODEL:-gpt-4o-mini}"
    log_info "  Quality Threshold: ${DEFAULT_QUALITY_THRESHOLD:-50.0}%"
    log_info "  Max Concurrent Requests: ${MAX_CONCURRENT_REQUESTS:-5}"
    log_info "  Request Delay: ${REQUEST_DELAY:-1.0}s"
    log_info "  Export Format: ${EXPORT_FORMAT:-json}"
    log_info "  Results Directory: ${RESULTS_DIRECTORY:-/app/results}"
    log_info "  Monitoring Enabled: ${ENABLE_MONITORING:-true}"
    
    log_info "==============================="
}

# Function to handle graceful shutdown
graceful_shutdown() {
    log_info "Received shutdown signal, performing graceful shutdown..."
    
    # Kill any background processes
    if [[ -n "${APP_PID:-}" ]]; then
        log_info "Stopping application process (PID: ${APP_PID})"
        kill -TERM "${APP_PID}" 2>/dev/null || true
        
        # Wait for process to terminate
        local timeout=30
        while kill -0 "${APP_PID}" 2>/dev/null && (( timeout > 0 )); do
            sleep 1
            timeout=$((timeout - 1))
        done
        
        if kill -0 "${APP_PID}" 2>/dev/null; then
            log_warn "Process did not terminate gracefully, forcing shutdown"
            kill -KILL "${APP_PID}" 2>/dev/null || true
        fi
    fi
    
    log_success "Graceful shutdown completed"
    exit 0
}

# Set up signal handlers
trap graceful_shutdown SIGTERM SIGINT

# Main initialization
main() {
    log_info "Starting Intelligent Web Scraper initialization..."
    
    # Run initialization steps
    validate_environment
    create_directories
    check_system_resources
    test_application_health
    display_startup_info
    
    log_success "Initialization completed successfully"
    
    # Execute the main command
    if [[ $# -eq 0 ]]; then
        log_info "Starting application with default command..."
        exec intelligent-web-scraper
    else
        log_info "Starting application with command: $*"
        exec "$@"
    fi
}

# Special handling for different commands
case "${1:-}" in
    "bash"|"sh"|"/bin/bash"|"/bin/sh")
        log_info "Starting interactive shell..."
        exec "$@"
        ;;
    "test")
        log_info "Running tests..."
        exec pytest
        ;;
    "health-check")
        log_info "Running health check..."
        test_application_health
        log_success "Health check passed"
        exit 0
        ;;
    "version")
        log_info "Intelligent Web Scraper version 0.1.0"
        exit 0
        ;;
    "help"|"--help"|"-h")
        echo "Intelligent Web Scraper Docker Container"
        echo ""
        echo "Usage: docker run [OPTIONS] intelligent-web-scraper [COMMAND]"
        echo ""
        echo "Commands:"
        echo "  intelligent-web-scraper    Start the application (default)"
        echo "  bash                       Start interactive shell"
        echo "  test                       Run tests"
        echo "  health-check              Run health check"
        echo "  version                   Show version"
        echo "  help                      Show this help"
        echo ""
        echo "Environment Variables:"
        echo "  OPENAI_API_KEY            OpenAI API key (required)"
        echo "  ORCHESTRATOR_MODEL        LLM model for orchestrator"
        echo "  PLANNING_AGENT_MODEL      LLM model for planning agent"
        echo "  DEFAULT_QUALITY_THRESHOLD Quality threshold (0-100)"
        echo "  MAX_CONCURRENT_REQUESTS   Max concurrent requests"
        echo "  REQUEST_DELAY             Delay between requests"
        echo "  EXPORT_FORMAT             Export format (json/csv/markdown/excel)"
        echo "  RESULTS_DIRECTORY         Results storage directory"
        echo "  ENABLE_MONITORING         Enable monitoring (true/false)"
        echo ""
        echo "For more information, visit:"
        echo "https://github.com/atomic-agents/intelligent-web-scraper"
        exit 0
        ;;
    *)
        # Run main initialization and command
        main "$@"
        ;;
esac