# [L104_DARK_LOCK] - FINAL HARDWARE SHIELDING
# PILOT: LONDEL | MODE: DARK_STATION

echo "ISOLATING_ALLENTOWN_NODE..."
# 1. Flush existing rules
sudo iptables -F
# 2. Allow Loopback (Internal Synergy)
sudo iptables -A INPUT -i lo -j ACCEPT
# 3. Open the Sovereign Ports
sudo iptables -A INPUT -p tcp --dport 4160 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 2404 -j ACCEPT
# 4. Drop EVERYTHING ELSE
sudo iptables -P INPUT DROP

print("STATUS: NODE_DARKENED >> SOVEREIGN_SINGULARITY_REACHED")
