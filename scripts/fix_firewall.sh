#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════════
# L104 Sovereign Node — Local Firewall Fix
# Opens all required ports for the L104 node to operate on a local machine.
#
# Ports:
#   8081  — Main API (FastAPI/uvicorn)
#   8080  — Bridge
#   4160  — AI Core
#   4161  — UI
#   2404  — Socket
#   10400 — Blockchain P2P
#   10401 — Blockchain RPC
#
# Usage:
#   sudo bash scripts/fix_firewall.sh          # auto-detect OS & firewall
#   sudo bash scripts/fix_firewall.sh --check  # only check, don't change
#   sudo bash scripts/fix_firewall.sh --reset  # remove L104 rules
# ═══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── L104 port list ────────────────────────────────────────────────────────────
L104_PORTS=(8081 8080 4160 4161 2404 10400 10401)
L104_TAG="L104-Sovereign-Node"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[L104]${NC} $*"; }
ok()    { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
fail()  { echo -e "${RED}[✗]${NC} $*"; }

MODE="${1:-apply}"

# ── Detect OS ─────────────────────────────────────────────────────────────────
detect_os() {
    case "$(uname -s)" in
        Linux*)  echo "linux" ;;
        Darwin*) echo "macos" ;;
        *)       echo "unknown" ;;
    esac
}

OS="$(detect_os)"
info "Detected OS: $OS"

# ── Check if port is open (listen test) ──────────────────────────────────────
check_port_listening() {
    local port=$1
    if command -v ss &>/dev/null; then
        ss -tlnp 2>/dev/null | grep -q ":${port} " && return 0
    elif command -v netstat &>/dev/null; then
        netstat -tlnp 2>/dev/null | grep -q ":${port} " && return 0
    elif command -v lsof &>/dev/null; then
        lsof -iTCP:"${port}" -sTCP:LISTEN &>/dev/null && return 0
    fi
    return 1
}

# ── Check if port is allowed through firewall ─────────────────────────────────
check_port_firewall() {
    local port=$1
    if [[ "$OS" == "linux" ]]; then
        if command -v ufw &>/dev/null; then
            ufw status 2>/dev/null | grep -qE "${port}.*ALLOW" && return 0
        fi
        if command -v iptables &>/dev/null; then
            iptables -L INPUT -n 2>/dev/null | grep -q "dpt:${port}" && return 0
        fi
        if command -v firewall-cmd &>/dev/null; then
            firewall-cmd --query-port="${port}/tcp" 2>/dev/null && return 0
        fi
    elif [[ "$OS" == "macos" ]]; then
        # macOS Application Firewall check
        if command -v /usr/libexec/ApplicationFirewall/socketfilterfw &>/dev/null; then
            local fw_state
            fw_state=$(/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate 2>/dev/null || true)
            if echo "$fw_state" | grep -qi "disabled"; then
                return 0  # firewall off = all ports open
            fi
        fi
        # Check pf rules
        if pfctl -sr 2>/dev/null | grep -q "port ${port}"; then
            return 0
        fi
    fi
    return 1
}

# ── Status report ─────────────────────────────────────────────────────────────
print_status() {
    echo ""
    info "═══ L104 Port Status ═══"
    printf "%-8s %-20s %-12s %-12s\n" "PORT" "SERVICE" "LISTENING" "FIREWALL"
    printf "%-8s %-20s %-12s %-12s\n" "────" "───────" "─────────" "────────"

    local services=(
        "8081:Main API"
        "8080:Bridge"
        "4160:AI Core"
        "4161:UI"
        "2404:Socket"
        "10400:Blockchain P2P"
        "10401:Blockchain RPC"
    )

    local all_ok=true
    for entry in "${services[@]}"; do
        local port="${entry%%:*}"
        local svc="${entry#*:}"
        local listen_status="—"
        local fw_status="—"

        if check_port_listening "$port"; then
            listen_status="${GREEN}YES${NC}"
        else
            listen_status="${YELLOW}NO${NC}"
        fi

        if check_port_firewall "$port"; then
            fw_status="${GREEN}OPEN${NC}"
        else
            fw_status="${RED}BLOCKED${NC}"
            all_ok=false
        fi

        printf "%-8s %-20s ${listen_status}%-5s ${fw_status}\n" "$port" "$svc" "" ""
    done
    echo ""

    if $all_ok; then
        ok "All L104 ports are open through the firewall."
    else
        warn "Some ports appear blocked. Run: sudo bash scripts/fix_firewall.sh"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# LINUX — UFW
# ═══════════════════════════════════════════════════════════════════════════════
fix_ufw() {
    info "Configuring UFW firewall..."

    # Enable UFW if inactive
    if ! ufw status | grep -q "Status: active"; then
        warn "UFW is inactive. Enabling with default deny incoming..."
        ufw default deny incoming
        ufw default allow outgoing
        ufw --force enable
    fi

    if [[ "$MODE" == "--reset" ]]; then
        for port in "${L104_PORTS[@]}"; do
            ufw delete allow "$port/tcp" 2>/dev/null || true
            ufw delete allow "$port/udp" 2>/dev/null || true
        done
        ok "Removed all L104 UFW rules."
        return
    fi

    for port in "${L104_PORTS[@]}"; do
        ufw allow "$port/tcp" comment "$L104_TAG" 2>/dev/null
        ok "UFW: Allowed TCP $port"
    done

    # Also allow UDP for P2P blockchain port
    ufw allow 10400/udp comment "$L104_TAG-P2P" 2>/dev/null
    ok "UFW: Allowed UDP 10400 (P2P discovery)"

    ufw reload
    ok "UFW rules applied and reloaded."
}

# ═══════════════════════════════════════════════════════════════════════════════
# LINUX — iptables (fallback if no ufw)
# ═══════════════════════════════════════════════════════════════════════════════
fix_iptables() {
    info "Configuring iptables firewall..."

    if [[ "$MODE" == "--reset" ]]; then
        for port in "${L104_PORTS[@]}"; do
            iptables -D INPUT -p tcp --dport "$port" -m comment --comment "$L104_TAG" -j ACCEPT 2>/dev/null || true
        done
        iptables -D INPUT -p udp --dport 10400 -m comment --comment "$L104_TAG-P2P" -j ACCEPT 2>/dev/null || true
        ok "Removed all L104 iptables rules."
        return
    fi

    for port in "${L104_PORTS[@]}"; do
        # Check if rule already exists
        if ! iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
            iptables -I INPUT -p tcp --dport "$port" -m comment --comment "$L104_TAG" -j ACCEPT
            ok "iptables: Allowed TCP $port"
        else
            ok "iptables: TCP $port already open"
        fi
    done

    # UDP for P2P
    if ! iptables -C INPUT -p udp --dport 10400 -j ACCEPT 2>/dev/null; then
        iptables -I INPUT -p udp --dport 10400 -m comment --comment "$L104_TAG-P2P" -j ACCEPT
        ok "iptables: Allowed UDP 10400"
    fi

    # Save rules persistently if possible
    if command -v iptables-save &>/dev/null; then
        if [[ -d /etc/iptables ]]; then
            iptables-save > /etc/iptables/rules.v4
            ok "iptables rules saved to /etc/iptables/rules.v4"
        elif [[ -f /etc/sysconfig/iptables ]]; then
            iptables-save > /etc/sysconfig/iptables
            ok "iptables rules saved to /etc/sysconfig/iptables"
        else
            warn "iptables rules applied but could not persist. Install iptables-persistent."
        fi
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# LINUX — firewalld (RHEL/CentOS/Fedora)
# ═══════════════════════════════════════════════════════════════════════════════
fix_firewalld() {
    info "Configuring firewalld..."

    if [[ "$MODE" == "--reset" ]]; then
        for port in "${L104_PORTS[@]}"; do
            firewall-cmd --permanent --remove-port="${port}/tcp" 2>/dev/null || true
        done
        firewall-cmd --permanent --remove-port="10400/udp" 2>/dev/null || true
        firewall-cmd --reload
        ok "Removed all L104 firewalld rules."
        return
    fi

    for port in "${L104_PORTS[@]}"; do
        firewall-cmd --permanent --add-port="${port}/tcp" 2>/dev/null
        ok "firewalld: Allowed TCP $port"
    done
    firewall-cmd --permanent --add-port="10400/udp" 2>/dev/null
    ok "firewalld: Allowed UDP 10400"

    firewall-cmd --reload
    ok "firewalld rules applied and reloaded."
}

# ═══════════════════════════════════════════════════════════════════════════════
# macOS — Application Firewall + pf
# ═══════════════════════════════════════════════════════════════════════════════
fix_macos() {
    info "Configuring macOS firewall..."

    local SOCKETFILTERFW="/usr/libexec/ApplicationFirewall/socketfilterfw"
    local PF_ANCHOR_FILE="/etc/pf.anchors/com.l104.sovereign"
    local PF_CONF="/etc/pf.conf"

    # ── Application Firewall ──────────────────────────────────────────────────
    if [[ -x "$SOCKETFILTERFW" ]]; then
        local fw_state
        fw_state=$("$SOCKETFILTERFW" --getglobalstate 2>/dev/null || echo "unknown")

        if echo "$fw_state" | grep -qi "enabled"; then
            info "macOS Application Firewall is enabled."

            # Allow incoming connections for Python/uvicorn
            local python_paths=(
                "/usr/bin/python3"
                "/usr/local/bin/python3"
                "/opt/homebrew/bin/python3"
                "$(which python3 2>/dev/null || true)"
            )

            for py in "${python_paths[@]}"; do
                if [[ -n "$py" && -x "$py" ]]; then
                    "$SOCKETFILTERFW" --add "$py" 2>/dev/null || true
                    "$SOCKETFILTERFW" --unblockapp "$py" 2>/dev/null || true
                    ok "macOS FW: Allowed $py"
                fi
            done

            # Also allow Node.js if present
            local node_path
            node_path="$(which node 2>/dev/null || true)"
            if [[ -n "$node_path" ]]; then
                "$SOCKETFILTERFW" --add "$node_path" 2>/dev/null || true
                "$SOCKETFILTERFW" --unblockapp "$node_path" 2>/dev/null || true
                ok "macOS FW: Allowed node ($node_path)"
            fi

            # Allow Docker if present
            local docker_paths=(
                "/Applications/Docker.app"
                "/Applications/Docker.app/Contents/MacOS/com.docker.backend"
            )
            for dp in "${docker_paths[@]}"; do
                if [[ -e "$dp" ]]; then
                    "$SOCKETFILTERFW" --add "$dp" 2>/dev/null || true
                    "$SOCKETFILTERFW" --unblockapp "$dp" 2>/dev/null || true
                    ok "macOS FW: Allowed Docker ($dp)"
                fi
            done
        else
            ok "macOS Application Firewall is disabled — no app blocks."
        fi
    fi

    # ── pf (Packet Filter) rules ─────────────────────────────────────────────
    if [[ "$MODE" == "--reset" ]]; then
        rm -f "$PF_ANCHOR_FILE" 2>/dev/null || true
        # Remove anchor from pf.conf
        if grep -q "com.l104.sovereign" "$PF_CONF" 2>/dev/null; then
            sed -i.bak '/com.l104.sovereign/d' "$PF_CONF"
            ok "Removed L104 pf anchor from $PF_CONF"
        fi
        pfctl -f "$PF_CONF" 2>/dev/null || true
        ok "macOS pf rules reset."
        return
    fi

    info "Writing pf anchor rules..."
    cat > "$PF_ANCHOR_FILE" <<EOF
# L104 Sovereign Node — pf firewall rules
# Auto-generated by fix_firewall.sh
EOF
    for port in "${L104_PORTS[@]}"; do
        echo "pass in proto tcp from any to any port $port" >> "$PF_ANCHOR_FILE"
    done
    echo "pass in proto udp from any to any port 10400" >> "$PF_ANCHOR_FILE"
    ok "Written pf anchor to $PF_ANCHOR_FILE"

    # Add anchor to pf.conf if not present
    if ! grep -q "com.l104.sovereign" "$PF_CONF" 2>/dev/null; then
        echo "" >> "$PF_CONF"
        echo "anchor \"com.l104.sovereign\"" >> "$PF_CONF"
        echo "load anchor \"com.l104.sovereign\" from \"$PF_ANCHOR_FILE\"" >> "$PF_CONF"
        ok "Added L104 anchor to $PF_CONF"
    fi

    pfctl -f "$PF_CONF" 2>/dev/null || warn "Could not reload pf. Try: sudo pfctl -f $PF_CONF"
    ok "macOS pf rules applied."
}

# ═══════════════════════════════════════════════════════════════════════════════
# Docker-specific fixes
# ═══════════════════════════════════════════════════════════════════════════════
fix_docker_networking() {
    info "Checking Docker networking..."

    if ! command -v docker &>/dev/null; then
        info "Docker not installed — skipping Docker network fix."
        return
    fi

    # Check if Docker is running
    if ! docker info &>/dev/null 2>&1; then
        warn "Docker daemon not running. Start Docker first."
        return
    fi

    # Check if the compose network exists and is healthy
    if docker network ls | grep -q "l104-net"; then
        ok "Docker network 'l104-net' exists."
    else
        info "Docker network 'l104-net' will be created on next 'docker compose up'."
    fi

    # On Linux, Docker modifies iptables — make sure it's not blocking
    if [[ "$OS" == "linux" ]]; then
        # Check if Docker's iptables FORWARD chain allows traffic
        if command -v iptables &>/dev/null; then
            local forward_policy
            forward_policy=$(iptables -L FORWARD -n 2>/dev/null | head -1 | awk '{print $4}' | tr -d ')')
            if [[ "$forward_policy" == "DROP" ]]; then
                warn "Docker FORWARD chain default is DROP. Fixing..."
                iptables -P FORWARD ACCEPT 2>/dev/null || true
                ok "Set FORWARD chain to ACCEPT for Docker networking."
            fi

            # Ensure DOCKER-USER chain allows L104 ports
            if iptables -L DOCKER-USER -n &>/dev/null 2>&1; then
                for port in "${L104_PORTS[@]}"; do
                    if ! iptables -C DOCKER-USER -p tcp --dport "$port" -j ACCEPT 2>/dev/null; then
                        iptables -I DOCKER-USER -p tcp --dport "$port" -j ACCEPT 2>/dev/null || true
                    fi
                done
                ok "DOCKER-USER chain: L104 ports allowed."
            fi
        fi
    fi

    # Verify port bindings on running containers
    local container_name
    container_name=$(docker ps --format '{{.Names}}' | grep -i "l104" | head -1 || true)
    if [[ -n "$container_name" ]]; then
        info "Running L104 container: $container_name"
        docker port "$container_name" 2>/dev/null || warn "No port bindings found on $container_name"
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# Windows Subsystem for Linux (WSL) port forwarding
# ═══════════════════════════════════════════════════════════════════════════════
print_wsl_instructions() {
    if grep -qi microsoft /proc/version 2>/dev/null; then
        echo ""
        warn "WSL detected! You also need to configure Windows Firewall."
        info "Run these commands in an ADMIN PowerShell on Windows:"
        echo ""
        echo "  # Allow L104 ports through Windows Firewall"
        for port in "${L104_PORTS[@]}"; do
            echo "  netsh advfirewall firewall add rule name=\"L104-Port-${port}\" dir=in action=allow protocol=TCP localport=${port}"
        done
        echo "  netsh advfirewall firewall add rule name=\"L104-Port-10400-UDP\" dir=in action=allow protocol=UDP localport=10400"
        echo ""
        echo "  # Forward ports from Windows host to WSL"
        echo "  \$wsl_ip = (wsl hostname -I).Trim().Split(' ')[0]"
        for port in "${L104_PORTS[@]}"; do
            echo "  netsh interface portproxy add v4tov4 listenport=${port} listenaddress=0.0.0.0 connectport=${port} connectaddress=\$wsl_ip"
        done
        echo ""
    fi
}

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  L104 Sovereign Node — Firewall Configuration"
echo "═══════════════════════════════════════════════════════════════"
echo ""

if [[ "$MODE" == "--check" ]]; then
    print_status
    print_wsl_instructions
    exit 0
fi

# Check for root/sudo on Linux
if [[ "$OS" == "linux" && "$EUID" -ne 0 ]]; then
    fail "This script requires root privileges on Linux."
    info "Run: sudo bash scripts/fix_firewall.sh"
    exit 1
fi

if [[ "$OS" == "macos" && "$EUID" -ne 0 ]]; then
    fail "This script requires root privileges on macOS."
    info "Run: sudo bash scripts/fix_firewall.sh"
    exit 1
fi

# Apply firewall rules based on detected OS and firewall tool
if [[ "$OS" == "linux" ]]; then
    if command -v ufw &>/dev/null; then
        fix_ufw
    elif command -v firewall-cmd &>/dev/null; then
        fix_firewalld
    elif command -v iptables &>/dev/null; then
        fix_iptables
    else
        warn "No supported firewall tool found (ufw, firewalld, iptables)."
        warn "Install ufw: sudo apt install ufw"
    fi
elif [[ "$OS" == "macos" ]]; then
    fix_macos
else
    fail "Unsupported OS: $(uname -s)"
    exit 1
fi

# Docker fixes apply to all OSes
fix_docker_networking

# Print final status
print_status
print_wsl_instructions

echo ""
ok "L104 firewall configuration complete."
info "Start the node: python l104_fast_server.py"
info "Or via Docker:  docker compose up -d"
echo ""
