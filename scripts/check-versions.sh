#!/bin/bash
# Check version consistency across repositories
# This script helps maintain version synchronization in our atomic ecosystem

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}📊 Atomic Ecosystem Version Status Report${NC}"
echo -e "${BLUE}=========================================${NC}"

# Function to get version from pyproject.toml
get_version() {
    local file="$1"
    if [ -f "$file" ]; then
        grep 'version = ' "$file" | cut -d'"' -f2
    else
        echo "NOT_FOUND"
    fi
}

# Function to get git commit hash
get_git_hash() {
    local dir="$1"
    if [ -d "$dir/.git" ]; then
        cd "$dir" && git rev-parse --short HEAD 2>/dev/null || echo "NO_GIT"
    else
        echo "NO_GIT"
    fi
}

# Check atomic_scraper_tool (source of truth)
if [ -d "atomic_scraper_tool" ]; then
    SCRAPER_VERSION=$(get_version "atomic_scraper_tool/pyproject.toml")
    SCRAPER_HASH=$(get_git_hash "atomic_scraper_tool")
    echo -e "${GREEN}🔧 atomic_scraper_tool (SOURCE):${NC} v$SCRAPER_VERSION ($SCRAPER_HASH)"
else
    echo -e "${RED}❌ atomic_scraper_tool directory not found${NC}"
    SCRAPER_VERSION="NOT_FOUND"
fi

# Check atomic-forge version
if [ -f "atomic-agents/atomic-forge/tools/atomic_scraper_tool/pyproject.toml" ]; then
    FORGE_VERSION=$(get_version "atomic-agents/atomic-forge/tools/atomic_scraper_tool/pyproject.toml")
    FORGE_HASH=$(get_git_hash "atomic-agents")
    echo -e "${CYAN}⚛️  atomic-forge/atomic_scraper_tool:${NC} v$FORGE_VERSION ($FORGE_HASH)"
else
    echo -e "${RED}❌ atomic-forge/atomic_scraper_tool not found${NC}"
    FORGE_VERSION="NOT_FOUND"
fi

# Check intelligent-web-scraper
if [ -f "intelligent-web-scraper/pyproject.toml" ]; then
    IWS_VERSION=$(get_version "intelligent-web-scraper/pyproject.toml")
    IWS_HASH=$(get_git_hash "intelligent-web-scraper")
    
    # Check dependency specification
    if grep -q "atomic-scraper-tool.*path.*atomic_scraper_tool" "intelligent-web-scraper/pyproject.toml"; then
        IWS_DEP="path dependency (local)"
    elif grep -q "atomic-scraper-tool" "intelligent-web-scraper/pyproject.toml"; then
        IWS_DEP=$(grep "atomic-scraper-tool" "intelligent-web-scraper/pyproject.toml" | cut -d'=' -f2 | tr -d ' "')
    else
        IWS_DEP="not specified"
    fi
    
    echo -e "${YELLOW}🚀 intelligent-web-scraper:${NC} v$IWS_VERSION ($IWS_HASH)"
    echo -e "   ${CYAN}Dependency:${NC} $IWS_DEP"
else
    echo -e "${RED}❌ intelligent-web-scraper not found${NC}"
fi

echo ""
echo -e "${BLUE}🔍 Analysis:${NC}"

# Version consistency check
if [ "$SCRAPER_VERSION" != "NOT_FOUND" ] && [ "$FORGE_VERSION" != "NOT_FOUND" ]; then
    if [ "$SCRAPER_VERSION" = "$FORGE_VERSION" ]; then
        echo -e "${GREEN}✅ Versions are synchronized${NC}"
    else
        echo -e "${RED}⚠️  VERSION MISMATCH DETECTED!${NC}"
        echo -e "   Source: v$SCRAPER_VERSION"
        echo -e "   Forge:  v$FORGE_VERSION"
        echo -e "${YELLOW}💡 Run: ./scripts/sync-to-forge.sh${NC}"
    fi
else
    echo -e "${RED}❌ Cannot compare versions - missing repositories${NC}"
fi

# Git status check
echo ""
echo -e "${BLUE}📝 Git Status:${NC}"

# Check atomic_scraper_tool status
if [ -d "atomic_scraper_tool/.git" ]; then
    cd atomic_scraper_tool
    if [ -n "$(git status --porcelain)" ]; then
        echo -e "${YELLOW}🔧 atomic_scraper_tool: Has uncommitted changes${NC}"
    else
        echo -e "${GREEN}🔧 atomic_scraper_tool: Clean${NC}"
    fi
    cd ..
fi

# Check atomic-agents status
if [ -d "atomic-agents/.git" ]; then
    cd atomic-agents
    if git status --porcelain | grep -q "atomic-forge/tools/atomic_scraper_tool"; then
        echo -e "${YELLOW}⚛️  atomic-agents: atomic_scraper_tool changes pending${NC}"
    else
        echo -e "${GREEN}⚛️  atomic-agents: atomic_scraper_tool up to date${NC}"
    fi
    cd ..
fi

# Check intelligent-web-scraper status
if [ -d "intelligent-web-scraper/.git" ]; then
    cd intelligent-web-scraper
    if [ -n "$(git status --porcelain)" ]; then
        echo -e "${YELLOW}🚀 intelligent-web-scraper: Has uncommitted changes${NC}"
    else
        echo -e "${GREEN}🚀 intelligent-web-scraper: Clean${NC}"
    fi
    cd ..
fi

echo ""
echo -e "${BLUE}🛠️  Recommended Actions:${NC}"

if [ "$SCRAPER_VERSION" != "$FORGE_VERSION" ]; then
    echo -e "${YELLOW}1. Sync versions: ./scripts/sync-to-forge.sh${NC}"
fi

echo -e "${CYAN}2. Test pipeline: ./scripts/test-pipeline.sh${NC}"
echo -e "${CYAN}3. Update dependencies: poetry update (in each project)${NC}"

echo ""
echo -e "${GREEN}📋 Report complete!${NC}"