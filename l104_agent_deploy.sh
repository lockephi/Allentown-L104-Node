#!/bin/bash
# ═══════════════════════════════════════════════════════════════════
# L104 Agent Deployment Script
# Deploys, manages, and monitors agents in the L104 Sovereign Node
# ═══════════════════════════════════════════════════════════════════

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# L104 Paths
L104_ROOT="/Users/carolalvarez/Applications/Allentown-L104-Node"
VENV="$L104_ROOT/.venv/bin"
PYTHON="$VENV/python"

# Agent types
AGENT_TYPES=("general" "researcher" "coder" "analyst" "planner" "critic" "creative" "executor" "learner" "oracle" "synthesizer")

# Priority levels
PRIORITIES=("critical" "high" "normal" "low" "idle")

# ═══════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════

log_info() { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_venv() {
    if [ ! -d "$L104_ROOT/.venv" ]; then
        log_error "Virtual environment not found at $L104_ROOT/.venv"
        log_info "Run: cd $L104_ROOT && python -m venv .venv && .venv/bin/pip install -r requirements.txt"
        exit 1
    fi
}

# ═══════════════════════════════════════════════════════════════════
# Deployment Commands
# ═══════════════════════════════════════════════════════════════════

deploy_agent() {
    local GOAL="$1"
    local TYPE="${2:-general}"
    local PRIO="${3:-normal}"

    check_venv

    log_info "Deploying agent: Type=$TYPE, Priority=$PRIO"
    log_info "Goal: $GOAL"

    $PYTHON -c "
import sys
sys.path.insert(0, '$L104_ROOT')
from l104_agent_system import AgentOrchestrator, AgentType, AgentPriority, AgentTask

# Convert strings to enums
agent_type = AgentType.$TYPE.upper() if hasattr(AgentType, '$TYPE'.upper()) else AgentType.GENERAL
priority = AgentPriority.$PRIO.upper() if hasattr(AgentPriority, '$PRIO'.upper()) else AgentPriority.NORMAL

task = AgentTask(
    prompt='$GOAL',
    agentType=agent_type,
    priority=priority,
    source='cli_deploy'
)

orchestrator = AgentOrchestrator.shared
task_id = orchestrator.submit(task)
print(f'Task ID: {task_id}')
print(f'Status: {task.status}')
" 2>&1

    log_success "Agent deployed successfully!"
}

list_agents() {
    check_venv

    log_info "Active Agents:"
    $PYTHON -c "
import sys
sys.path.insert(0, '$L104_ROOT')
from l104_agent_system import AgentOrchestrator

orchestrator = AgentOrchestrator.shared
active = orchestrator.list_active()
completed = orchestrator.list_completed()

print(f'\nRunning/Queued: {len([t for t in active if t.status.value == \"running\"])}')
print(f'Queued: {len([t for t in active if t.status.value == \"queued\"])}')
print(f'Completed: {len(completed)}')
print('\n--- Active Tasks ---')
for task in active[:20]:
    print(f'  [{task.status.value}] {task.agentType.value}: {task.prompt[:50]}...')
" 2>&1
}

agent_status() {
    check_venv

    $PYTHON -c "
import sys
sys.path.insert(0, '$L104_ROOT')
from l104_agent_system import AgentOrchestrator

orchestrator = AgentOrchestrator.shared
active = orchestrator.list_active()
print(f'Total Active: {len(active)}')
for task in active:
    print(f'  {task.task_id}: {task.status.value} - {task.agentType.value}')
" 2>&1
}

cancel_agent() {
    local TASK_ID="$1"

    check_venv

    $PYTHON -c "
import sys
sys.path.insert(0, '$L104_ROOT')
from l104_agent_system import AgentOrchestrator

orchestrator = AgentOrchestrator.shared
result = orchestrator.cancel('$TASK_ID')
print(f'Cancelled: {result}')
" 2>&1

    log_success "Agent cancelled!"
}

clear_completed() {
    check_venv

    $PYTHON -c "
import sys
sys.path.insert(0, '$L104_ROOT')
from l104_agent_system import AgentOrchestrator

orchestrator = AgentOrchestrator.shared
orchestrator.clear_completed()
print('Cleared completed tasks')
" 2>&1

    log_success "Completed tasks cleared!"
}

# ═══════════════════════════════════════════════════════════════════
# Server Management
# ═══════════════════════════════════════════════════════════════════

start_server() {
    check_venv
    log_info "Starting L104 Server..."
    cd "$L104_ROOT"
    $VENV/uvicorn l104_server.app:app --host 0.0.0.0 --port 8080 --reload &
    log_success "Server started on http://localhost:8080"
}

stop_server() {
    log_info "Stopping L104 Server..."
    pkill -f "uvicorn l104_server.app:app" || log_warn "No server process found"
    log_success "Server stopped"
}

restart_server() {
    stop_server
    sleep 2
    start_server
}

# ═══════════════════════════════════════════════════════════════════
# Build & Deploy
# ═══════════════════════════════════════════════════════════════════

build_app() {
    log_info "Building L104 Swift App..."
    cd "$L104_ROOT/L104SwiftApp"
    ./quick_build.sh
    log_success "App built!"
}

deploy_all() {
    log_info "Deploying full L104 stack..."

    # Start server
    start_server

    # Wait for server
    sleep 3

    # Build app
    build_app

    log_success "Full stack deployed!"
}

# ═══════════════════════════════════════════════════════════════════
# Status
# ═══════════════════════════════════════════════════════════════════

status() {
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  L104 SOVEREIGN NODE - STATUS${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════${NC}"

    # Check server
    if curl -s http://localhost:8080/api/v1/health > /dev/null 2>&1; then
        log_success "Server: Running"
    else
        log_warn "Server: Not running"
    fi

    # Check agents
    list_agents

    # Check memory
    echo ""
    log_info "Memory Usage:"
    free -h 2>/dev/null || vm_stat | head -5
}

# ═══════════════════════════════════════════════════════════════════
# Help
# ═══════════════════════════════════════════════════════════════════

help() {
    echo -e "${CYAN}L104 Agent Deployment Script${NC}"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  deploy <goal> [type] [priority]  Deploy a new agent"
    echo "  list                               List active agents"
    echo "  status                             Show system status"
    echo "  cancel <task_id>                   Cancel an agent"
    echo "  clear                              Clear completed tasks"
    echo "  start-server                       Start L104 server"
    echo "  stop-server                        Stop L104 server"
    echo "  restart-server                     Restart L104 server"
    echo "  build                              Build Swift app"
    echo "  deploy-all                         Deploy full stack"
    echo ""
    echo "Agent Types:"
    for t in "${AGENT_TYPES[@]}"; do echo "    $t"; done
    echo ""
    echo "Priorities:"
    for p in "${PRIORITIES[@]}"; do echo "    $p"; done
    echo ""
    echo "Examples:"
    echo "  $0 deploy \"Analyze the codebase\" researcher high"
    echo "  $0 deploy \"Fix memory leak\" coder critical"
    echo "  $0 list"
}

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

case "${1:-help}" in
    deploy)
        deploy_agent "${2:-}" "${3:-general}" "${4:-normal}"
        ;;
    list)
        list_agents
        ;;
    status)
        status
        ;;
    cancel)
        cancel_agent "${2:-}"
        ;;
    clear)
        clear_completed
        ;;
    start-server)
        start_server
        ;;
    stop-server)
        stop_server
        ;;
    restart-server)
        restart_server
        ;;
    build)
        build_app
        ;;
    deploy-all)
        deploy_all
        ;;
    help|--help|-h)
        help
        ;;
    *)
        log_error "Unknown command: $1"
        help
        exit 1
        ;;
esac