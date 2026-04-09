#!/bin/bash
# L104 Daemon Recovery Procedures

RECOVERY_MODE="${1:-standard}"

case $RECOVERY_MODE in
    "standard")
        echo "Starting standard daemon recovery..."
        # Restart quantum services with coherence reset
        systemctl restart l104-quantum-core
        sleep 2
        systemctl restart l104-entanglement-service
        sleep 1
        systemctl restart l104-temporal-sync
        ;;
    
    "aggressive")
        echo "Starting aggressive recovery..."
        # Full quantum state reset
        qstate-reset --all-daemons --clean
        
        # Reinitialize entanglement
        entanglement-reinit --full
        
        # Restart all L104 services
        systemctl list-units --type=service --no-legend | grep l104 | awk '{print $1}' | xargs -I{} systemctl restart {}
        ;;
    
    "quantum-only")
        echo "Quantum coherence recovery..."
        # Repair quantum aspects only
        qcoherence-repair --force
        
        # Reset collapsed states
        wavefunction-reset --service=all
        
        # Verify recovery
        qcoherence-check --service=all --threshold=0.99
        ;;
    
    *)
        echo "Usage: $0 [standard|aggressive|quantum-only]"
        exit 1
        ;;
esac

# Verify recovery
echo "Verifying recovery..."
sleep 3
systemctl status l104-quantum-core --no-pager
qcoherence-check --brief