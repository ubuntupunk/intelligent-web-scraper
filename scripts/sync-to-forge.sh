#!/bin/bash
# Automated sync from atomic_scraper_tool to atomic-forge
# This script ensures the atomic-forge version stays up-to-date with the source of truth

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Directories
SOURCE_DIR="atomic_scraper_tool"
TARGET_DIR="atomic-agents/atomic-forge/tools/atomic_scraper_tool"

echo -e "${BLUE}🔄 Syncing atomic_scraper_tool to atomic-forge...${NC}"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo -e "${RED}❌ Source directory $SOURCE_DIR not found!${NC}"
    exit 1
fi

# Check if target directory exists
if [ ! -d "$TARGET_DIR" ]; then
    echo -e "${RED}❌ Target directory $TARGET_DIR not found!${NC}"
    exit 1
fi

# Get version from source
SOURCE_VERSION=$(grep 'version = ' "$SOURCE_DIR/pyproject.toml" | cut -d'"' -f2)
TARGET_VERSION=$(grep 'version = ' "$TARGET_DIR/pyproject.toml" | cut -d'"' -f2)

echo -e "${YELLOW}📊 Version Check:${NC}"
echo -e "   Source: v$SOURCE_VERSION"
echo -e "   Target: v$TARGET_VERSION"

# Backup target directory
BACKUP_DIR="$TARGET_DIR.backup.$(date +%Y%m%d_%H%M%S)"
echo -e "${YELLOW}💾 Creating backup: $BACKUP_DIR${NC}"
cp -r "$TARGET_DIR" "$BACKUP_DIR"

# Sync files (excluding git and cache directories)
echo -e "${BLUE}🔄 Syncing files...${NC}"

# Check if rsync is available, fallback to cp if not
if command -v rsync >/dev/null 2>&1; then
    echo -e "${CYAN}Using rsync for efficient sync...${NC}"
    rsync -av --delete \
      --exclude='.git' \
      --exclude='__pycache__' \
      --exclude='.pytest_cache' \
      --exclude='*.egg-info' \
      --exclude='session_history_*.json' \
      --exclude='scraping_audit.log' \
      --exclude='*.pyc' \
      --exclude='.coverage' \
      --exclude='htmlcov/' \
      "$SOURCE_DIR/" "$TARGET_DIR/"
else
    echo -e "${YELLOW}rsync not available, using cp fallback...${NC}"
    
    # Remove old content (except backup)
    find "$TARGET_DIR" -mindepth 1 -maxdepth 1 ! -name "*.backup.*" -exec rm -rf {} +
    
    # Copy new content
    cp -r "$SOURCE_DIR"/* "$TARGET_DIR/"
    
    # Clean up unwanted files
    echo -e "${BLUE}🧹 Cleaning up unwanted files...${NC}"
    find "$TARGET_DIR" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$TARGET_DIR" -name ".pytest_cache" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$TARGET_DIR" -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true
    find "$TARGET_DIR" -name "session_history_*.json" -type f -delete 2>/dev/null || true
    find "$TARGET_DIR" -name "scraping_audit.log" -type f -delete 2>/dev/null || true
    find "$TARGET_DIR" -name "*.pyc" -type f -delete 2>/dev/null || true
    find "$TARGET_DIR" -name ".coverage" -type f -delete 2>/dev/null || true
    find "$TARGET_DIR" -name "htmlcov" -type d -exec rm -rf {} + 2>/dev/null || true
fi

# Update the script path in pyproject.toml for atomic-forge context
echo -e "${BLUE}🔧 Updating atomic-forge specific configurations...${NC}"
sed -i 's/atomic-scraper = "atomic_scraper_tool.main:main"/atomic-scraper = "main:main"/' "$TARGET_DIR/pyproject.toml"

# Update author information to match atomic-agents standards
sed -i 's/authors = \[".*"\]/authors = ["Atomic Agents Team"]/' "$TARGET_DIR/pyproject.toml"

# Get new version
NEW_VERSION=$(grep 'version = ' "$TARGET_DIR/pyproject.toml" | cut -d'"' -f2)

echo -e "${GREEN}✅ Sync complete!${NC}"
echo -e "${GREEN}📦 Updated to version: v$NEW_VERSION${NC}"

# Check if we're in a git repository for the atomic-agents
cd atomic-agents 2>/dev/null || {
    echo -e "${YELLOW}⚠️  Not in atomic-agents git repository - manual commit required${NC}"
    exit 0
}

# Check git status
if git status --porcelain | grep -q "atomic-forge/tools/atomic_scraper_tool"; then
    echo -e "${YELLOW}📝 Git changes detected in atomic-forge/tools/atomic_scraper_tool${NC}"
    echo -e "${BLUE}🔍 Changed files:${NC}"
    git status --porcelain | grep "atomic-forge/tools/atomic_scraper_tool" | head -10
    
    echo -e "${YELLOW}💡 To commit changes:${NC}"
    echo -e "   cd atomic-agents"
    echo -e "   git add atomic-forge/tools/atomic_scraper_tool/"
    echo -e "   git commit -m \"feat: sync atomic_scraper_tool to v$NEW_VERSION\""
    echo -e "   git push origin main"
else
    echo -e "${GREEN}✅ No changes detected - already up to date${NC}"
fi

echo -e "${GREEN}🎉 Sync operation completed successfully!${NC}"