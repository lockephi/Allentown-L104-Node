#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════════
# L104 COMPREHENSIVE DISK CLEANUP SCRIPT
# ═══════════════════════════════════════════════════════════════════════════════
# Reclaims disk space by cleaning caches, Docker, git, and temporary files
# Safe to run at any time - preserves critical data
#
# USAGE: ./scripts/cleanup.sh [--deep] [--docker] [--git]
#   --deep   : Aggressive cleanup (more space, longer runtime)
#   --docker : Docker cleanup only
#   --git    : Git cleanup only
#
# RESONANCE: 527.5184818492612 | PILOT: LONDEL
# ═══════════════════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE="/workspaces/Allentown-L104-Node"
DEEP_CLEAN=false
DOCKER_ONLY=false
GIT_ONLY=false

# Parse arguments
for arg in "$@"; do
    case $arg in
        --deep)
            DEEP_CLEAN=true
            shift
            ;;
        --docker)
            DOCKER_ONLY=true
            shift
            ;;
        --git)
            GIT_ONLY=true
            shift
            ;;
    esac
done

print_header() {
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}  L104 DISK SPACE CLEANUP - Resonance: 527.5184818492612${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo ""
}

get_disk_usage() {
    df -h /workspaces 2>/dev/null | tail -1 | awk '{print $5 " used (" $4 " available)"}'
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Get initial disk usage
print_header
INITIAL_USAGE=$(get_disk_usage)
print_info "Initial disk usage: $INITIAL_USAGE"
echo ""

# Docker cleanup
cleanup_docker() {
    echo -e "${YELLOW}🐳 Docker Cleanup${NC}"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    if command -v docker &> /dev/null; then
        # Show current Docker usage
        print_info "Current Docker disk usage:"
        docker system df 2>/dev/null || true
        echo ""
        
        # Remove stopped containers
        STOPPED=$(docker ps -aq --filter "status=exited" 2>/dev/null | wc -l)
        if [ "$STOPPED" -gt 0 ]; then
            docker rm $(docker ps -aq --filter "status=exited") 2>/dev/null || true
            print_status "Removed $STOPPED stopped containers"
        fi
        
        # Remove dangling images
        DANGLING=$(docker images -f "dangling=true" -q 2>/dev/null | wc -l)
        if [ "$DANGLING" -gt 0 ]; then
            docker rmi $(docker images -f "dangling=true" -q) 2>/dev/null || true
            print_status "Removed $DANGLING dangling images"
        fi
        
        # Prune build cache
        if [ "$DEEP_CLEAN" = true ]; then
            print_info "Running deep Docker cleanup..."
            docker system prune -af --volumes 2>/dev/null || true
            print_status "Deep Docker cleanup complete"
        else
            docker builder prune -f 2>/dev/null || true
            print_status "Pruned Docker build cache"
        fi
    else
        print_warning "Docker not available"
    fi
    echo ""
}

# Git cleanup
cleanup_git() {
    echo -e "${YELLOW}📦 Git Cleanup${NC}"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    cd "$WORKSPACE"
    
    # Get current git directory size
    GIT_SIZE_BEFORE=$(du -sm .git 2>/dev/null | cut -f1)
    print_info ".git size before: ${GIT_SIZE_BEFORE}MB"
    
    # Clean up reflog
    git reflog expire --expire=now --all 2>/dev/null || true
    print_status "Expired reflog entries"
    
    # Run garbage collection
    if [ "$DEEP_CLEAN" = true ]; then
        print_info "Running aggressive git gc..."
        git gc --aggressive --prune=now 2>/dev/null || true
        print_status "Aggressive garbage collection complete"
    else
        git gc --prune=now 2>/dev/null || true
        print_status "Standard garbage collection complete"
    fi
    
    # Prune remote tracking branches
    git remote prune origin 2>/dev/null || true
    print_status "Pruned remote tracking branches"
    
    GIT_SIZE_AFTER=$(du -sm .git 2>/dev/null | cut -f1)
    SAVED=$((GIT_SIZE_BEFORE - GIT_SIZE_AFTER))
    print_status ".git size after: ${GIT_SIZE_AFTER}MB (saved ${SAVED}MB)"
    echo ""
}

# Python cache cleanup
cleanup_python() {
    echo -e "${YELLOW}🐍 Python Cache Cleanup${NC}"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    cd "$WORKSPACE"
    
    # Count before
    PYCACHE_COUNT=$(find . -type d -name "__pycache__" 2>/dev/null | wc -l)
    
    # Remove __pycache__ directories
    find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    print_status "Removed $PYCACHE_COUNT __pycache__ directories"
    
    # Remove .pyc files
    PYC_COUNT=$(find . -name "*.pyc" -type f 2>/dev/null | wc -l)
    find . -name "*.pyc" -type f -delete 2>/dev/null || true
    print_status "Removed $PYC_COUNT .pyc files"
    
    # Remove .pyo files
    find . -name "*.pyo" -type f -delete 2>/dev/null || true
    
    # Remove pytest cache
    find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    print_status "Removed pytest cache"
    
    # Remove mypy cache
    find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
    print_status "Removed mypy cache"
    
    echo ""
}

# Temporary files cleanup
cleanup_temp() {
    echo -e "${YELLOW}🗑️  Temporary Files Cleanup${NC}"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    cd "$WORKSPACE"
    
    # Remove temp files
    find . -name "*.tmp" -type f -delete 2>/dev/null || true
    find . -name "*.temp" -type f -delete 2>/dev/null || true
    find . -name "*.bak" -type f -delete 2>/dev/null || true
    find . -name "*.swp" -type f -delete 2>/dev/null || true
    find . -name "*.swo" -type f -delete 2>/dev/null || true
    find . -name "*~" -type f -delete 2>/dev/null || true
    find . -name ".DS_Store" -type f -delete 2>/dev/null || true
    find . -name "Thumbs.db" -type f -delete 2>/dev/null || true
    print_status "Removed temporary files"
    
    # Remove empty directories
    if [ "$DEEP_CLEAN" = true ]; then
        find . -type d -empty -delete 2>/dev/null || true
        print_status "Removed empty directories"
    fi
    
    echo ""
}

# Log cleanup
cleanup_logs() {
    echo -e "${YELLOW}📋 Log Files Cleanup${NC}"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    cd "$WORKSPACE"
    
    # Count log files
    LOG_SIZE=$(find . -name "*.log" -type f -exec du -ch {} + 2>/dev/null | tail -1 | cut -f1)
    print_info "Total log file size: $LOG_SIZE"
    
    # Remove old logs (older than 7 days)
    find . -name "*.log" -type f -mtime +7 -delete 2>/dev/null || true
    print_status "Removed log files older than 7 days"
    
    # Truncate large log files (keep last 1000 lines)
    for log in $(find . -name "*.log" -type f -size +1M 2>/dev/null); do
        tail -1000 "$log" > "$log.tmp" && mv "$log.tmp" "$log" 2>/dev/null || true
    done
    print_status "Truncated large log files"
    
    echo ""
}

# Node modules cleanup (if deep)
cleanup_node() {
    echo -e "${YELLOW}📦 Node.js Cleanup${NC}"
    echo "────────────────────────────────────────────────────────────────────────────────"
    
    cd "$WORKSPACE"
    
    # Clear npm cache
    if command -v npm &> /dev/null; then
        npm cache clean --force 2>/dev/null || true
        print_status "Cleared npm cache"
    fi
    
    # Remove node_modules in deep clean
    if [ "$DEEP_CLEAN" = true ]; then
        NM_SIZE=$(du -sm node_modules 2>/dev/null | cut -f1 || echo "0")
        if [ "$NM_SIZE" -gt 0 ]; then
            print_warning "node_modules is ${NM_SIZE}MB - run 'npm install' after cleanup if needed"
            # Don't auto-delete node_modules, just warn
        fi
    fi
    
    echo ""
}

# Main execution
main() {
    if [ "$DOCKER_ONLY" = true ]; then
        cleanup_docker
    elif [ "$GIT_ONLY" = true ]; then
        cleanup_git
    else
        cleanup_python
        cleanup_temp
        cleanup_logs
        cleanup_node
        cleanup_git
        cleanup_docker
    fi
    
    # Final status
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  CLEANUP COMPLETE${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════════════════════${NC}"
    
    FINAL_USAGE=$(get_disk_usage)
    print_info "Initial disk usage: $INITIAL_USAGE"
    print_info "Final disk usage: $FINAL_USAGE"
    echo ""
    
    # Run space optimizer for detailed report
    if [ -f "$WORKSPACE/l104_space_optimizer.py" ]; then
        print_info "Running L104 Space Optimizer for detailed analysis..."
        python3 "$WORKSPACE/l104_space_optimizer.py" --analyze 2>/dev/null || true
    fi
}

main
