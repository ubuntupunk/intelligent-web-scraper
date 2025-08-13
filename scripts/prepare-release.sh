#!/bin/bash
# Prepare release for atomic ecosystem
# This script coordinates releases across multiple repositories

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

echo -e "${MAGENTA}🚀 Atomic Ecosystem Release Preparation${NC}"
echo -e "${MAGENTA}=======================================${NC}"

# Function to get current version
get_version() {
    local file="$1"
    if [ -f "$file" ]; then
        grep 'version = ' "$file" | cut -d'"' -f2
    else
        echo "NOT_FOUND"
    fi
}

# Function to update version in pyproject.toml
update_version() {
    local file="$1"
    local new_version="$2"
    
    if [ -f "$file" ]; then
        sed -i "s/version = \".*\"/version = \"$new_version\"/" "$file"
        echo -e "${GREEN}✅ Updated $file to v$new_version${NC}"
    else
        echo -e "${RED}❌ File $file not found${NC}"
    fi
}

# Get current versions
echo -e "${BLUE}📊 Current Version Status:${NC}"
SCRAPER_VERSION=$(get_version "atomic_scraper_tool/pyproject.toml")
IWS_VERSION=$(get_version "intelligent-web-scraper/pyproject.toml")

echo -e "${CYAN}🔧 atomic_scraper_tool: v$SCRAPER_VERSION${NC}"
echo -e "${CYAN}🚀 intelligent-web-scraper: v$IWS_VERSION${NC}"

# Ask for new version
echo ""
echo -e "${YELLOW}📝 Release Version Management:${NC}"
echo -e "${CYAN}Current versions are synchronized at v$SCRAPER_VERSION${NC}"

read -p "Enter new version (e.g., 0.2.0, 1.0.0): " NEW_VERSION

if [ -z "$NEW_VERSION" ]; then
    echo -e "${RED}❌ No version provided, exiting${NC}"
    exit 1
fi

# Validate version format (basic check)
if ! [[ "$NEW_VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}❌ Invalid version format. Use semantic versioning (e.g., 1.0.0)${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}🔄 Preparing release v$NEW_VERSION...${NC}"

# Step 1: Update versions
echo -e "${BLUE}1. Updating versions...${NC}"
update_version "atomic_scraper_tool/pyproject.toml" "$NEW_VERSION"
update_version "intelligent-web-scraper/pyproject.toml" "$NEW_VERSION"

# Step 2: Sync to atomic-forge
echo -e "${BLUE}2. Syncing to atomic-forge...${NC}"
./scripts/sync-to-forge.sh

# Step 3: Run tests
echo -e "${BLUE}3. Running test pipeline...${NC}"
./scripts/test-pipeline.sh | grep -E "(✅|❌|⚠️)" || true

# Step 4: Generate changelog entries
echo -e "${BLUE}4. Generating changelog information...${NC}"

CHANGELOG_ENTRY="
## [v$NEW_VERSION] - $(date +%Y-%m-%d)

### 🚀 Release Highlights
- Nuclear-powered web scraping platform
- Reactor-grade monitoring and performance tracking
- Batch processing with multi-URL orchestration
- AI-powered strategy planning and optimization

### ⚛️ Atomic Ecosystem Updates
- Synchronized atomic_scraper_tool and intelligent-web-scraper
- Enhanced atomic-forge integration
- Improved development pipeline and automation

### 🔧 Technical Improvements
- Enhanced error handling and recovery mechanisms
- Improved CLI functionality and user experience
- Better dependency management and version synchronization
- Comprehensive test coverage and validation

### 📚 Documentation
- Updated README with nuclear-themed terminology
- Enhanced development pipeline documentation
- Improved installation and usage instructions
"

echo -e "${CYAN}📝 Suggested changelog entry:${NC}"
echo "$CHANGELOG_ENTRY"

# Step 5: Git operations
echo -e "${BLUE}5. Git operations...${NC}"

# Check git status for each repository
repos=("atomic_scraper_tool" "intelligent-web-scraper" "atomic-agents")

for repo in "${repos[@]}"; do
    if [ -d "$repo/.git" ]; then
        echo -e "${CYAN}📋 Git status for $repo:${NC}"
        cd "$repo"
        
        if [ -n "$(git status --porcelain)" ]; then
            echo -e "${YELLOW}⚠️  $repo has uncommitted changes${NC}"
            git status --short
        else
            echo -e "${GREEN}✅ $repo is clean${NC}"
        fi
        
        cd ..
    fi
done

# Step 6: Release checklist
echo ""
echo -e "${MAGENTA}📋 Release Checklist:${NC}"
echo -e "${CYAN}□ Version numbers updated to v$NEW_VERSION${NC}"
echo -e "${CYAN}□ atomic-forge synchronized${NC}"
echo -e "${CYAN}□ Tests passing${NC}"
echo -e "${CYAN}□ Documentation updated${NC}"
echo -e "${CYAN}□ Changelog entries prepared${NC}"

echo ""
echo -e "${YELLOW}🎯 Next Manual Steps:${NC}"
echo -e "${CYAN}1. Review and commit changes in each repository${NC}"
echo -e "${CYAN}2. Create git tags: git tag v$NEW_VERSION${NC}"
echo -e "${CYAN}3. Push tags: git push origin v$NEW_VERSION${NC}"
echo -e "${CYAN}4. Update CHANGELOG.md files${NC}"
echo -e "${CYAN}5. Create GitHub releases${NC}"

echo ""
echo -e "${GREEN}🎉 Release preparation completed for v$NEW_VERSION!${NC}"
echo -e "${MAGENTA}Ready for nuclear-powered deployment! ⚛️${NC}"