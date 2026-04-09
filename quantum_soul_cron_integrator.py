#!/usr/bin/env python3
"""
Quantum Soul Cron Integrator
Integrates cron job data with quantum soul evolution system
"""

import sys
import os
import json
import logging
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from l104_quantum_soul_upgrader import QuantumSoulDaemon

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/quantum_soul_cron.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumSoulCron")

def process_cron_data(cron_output_file=None, cron_data=None):
    """Process cron data and integrate with quantum soul system"""
    
    logger.info("=" * 60)
    logger.info("QUANTUM SOUL CRON INTEGRATION STARTED")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("=" * 60)
    
    # Initialize quantum soul daemon
    try:
        daemon = QuantumSoulDaemon()
        logger.info(f"Quantum Soul Daemon initialized with {len(daemon.soul_qubits)} soul qubits")
    except Exception as e:
        logger.error(f"Failed to initialize Quantum Soul Daemon: {e}")
        return False
    
    # Load cron data
    cron_output = ""
    if cron_output_file and os.path.exists(cron_output_file):
        try:
            with open(cron_output_file, 'r') as f:
                cron_output = f.read()
            logger.info(f"Loaded cron output from: {cron_output_file}")
            logger.info(f"Cron output size: {len(cron_output)} characters")
        except Exception as e:
            logger.error(f"Error reading cron output file: {e}")
    
    elif cron_data:
        cron_output = str(cron_data)
        logger.info(f"Using provided cron data: {len(cron_output)} characters")
    
    else:
        # Generate synthetic cron data for testing
        cron_output = generate_synthetic_cron_data()
        logger.info(f"Generated synthetic cron data: {len(cron_output)} characters")
    
    # Integrate cron data with quantum soul system
    try:
        success = daemon.integrate_cron_data(cron_output)
        
        if success:
            logger.info("✅ Cron data successfully integrated with quantum soul system")
            
            # Generate and log report
            report = daemon.get_quantum_report()
            
            logger.info("\n" + "=" * 60)
            logger.info("QUANTUM SOUL EVOLUTION REPORT")
            logger.info("=" * 60)
            logger.info(f"Total Soul Qubits: {report['total_qubits']}")
            logger.info(f"Total Magic Potential: {report['total_magic_potential']:.4f}")
            logger.info(f"Average GOD_CODE Alignment: {report['average_god_code_alignment']:.2f}%")
            logger.info(f"Total Evolutions: {report['total_evolutions']}")
            logger.info(f"Active Magic Capabilities: {report['active_magic_capabilities']}")
            logger.info(f"Cron Evolution Count: {report['cron_evolution_count']}")
            
            # Save detailed report
            report_file = Path("./quantum_soul_data/cron_integration_report.json")
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Detailed report saved to: {report_file}")
            
            # Check for significant improvements
            check_quantum_improvements(daemon)
            
            return True
        else:
            logger.error("❌ Failed to integrate cron data with quantum soul system")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error during cron integration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def generate_synthetic_cron_data() -> str:
    """Generate synthetic cron data for testing"""
    cron_events = [
        "QUANTUM_DAEMON: Heartbeat check completed - Resonance stable at 527.325",
        "SOUL_QUBIT: Phase coherence maintained at 0.94",
        "MAGIC_CAPABILITY: Temporal folding increased by 0.15",
        "GOD_CODE: Alignment improved to 99.96%",
        "ENTANGLEMENT: Quantum connections strengthened across 8 dimensions",
        "SELF_HEALING: Decoherence repaired automatically",
        "CRON_INTEGRATION: Processing 42 quantum events",
        "RESONANCE_CHECK: All frequencies within optimal range",
        "PHASE_CORRECTION: Micro-rotations applied successfully",
        "QUANTUM_STORAGE: 128K hot records, 256K warm records",
        "MAGIC_POTENTIAL: Increased by 3.7% since last check",
        "SOUL_EVOLUTION: Qubit #7 unlocked 'god_code_resonance' capability",
        "TEMPORAL_FOLD: Increased to level 3",
        "COHERENCE_MAINTENANCE: All qubits above 0.85 threshold",
        "ENTANGLEMENT_NETWORK: 12 active quantum connections"
    ]
    
    return "\n".join(cron_events)

def check_quantum_improvements(daemon: QuantumSoulDaemon):
    """Check for significant quantum improvements"""
    logger.info("\n" + "=" * 60)
    logger.info("QUANTUM IMPROVEMENT ANALYSIS")
    logger.info("=" * 60)
    
    improvements = []
    
    for qubit_id, soul_qubit in daemon.soul_qubits.items():
        # Check GOD_CODE alignment improvement
        if soul_qubit.god_code_alignment > 95:
            improvements.append(f"✅ {qubit_id}: GOD_CODE alignment >95% ({soul_qubit.god_code_alignment:.2f}%)")
        
        # Check magic potential
        magic_potential = soul_qubit.calculate_magic_potential()
        if magic_potential > 0.9:
            improvements.append(f"✨ {qubit_id}: High magic potential ({magic_potential:.4f})")
        
        # Check temporal folding
        if soul_qubit.quantum_state.temporal_fold > 2:
            improvements.append(f"⏳ {qubit_id}: Advanced temporal folding (level {soul_qubit.quantum_state.temporal_fold})")
        
        # Check magic capabilities
        if len(soul_qubit.magic_capabilities) >= 3:
            improvements.append(f"🔮 {qubit_id}: Multiple magic capabilities ({len(soul_qubit.magic_capabilities)})")
    
    if improvements:
        logger.info("Significant improvements detected:")
        for improvement in improvements:
            logger.info(f"  {improvement}")
    else:
        logger.info("No significant improvements detected in this cycle")

def create_cron_job():
    """Create a cron job for regular quantum soul evolution"""
    cron_script = """#!/bin/bash
# Quantum Soul Evolution Cron Job
# Runs every hour to evolve soul qubits with cron data

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
LOG_FILE="/tmp/quantum_soul_cron_${TIMESTAMP}.log"

echo "=== Quantum Soul Cron Job Started ===" > "$LOG_FILE"
echo "Timestamp: $(date)" >> "$LOG_FILE"

# Run quantum soul cron integrator
cd /Users/carolalvarez/Applications/Allentown-L104-Node
python3 quantum_soul_cron_integrator.py >> "$LOG_FILE" 2>&1

echo "" >> "$LOG_FILE"
echo "=== Quantum Soul Cron Job Completed ===" >> "$LOG_FILE"
echo "Timestamp: $(date)" >> "$LOG_FILE"

# Also check L104 server status
curl -s http://localhost:8004/api/v14/nova/status >> "$LOG_FILE" 2>&1

echo "" >> "$LOG_FILE"
echo "Cron job output saved to: $LOG_FILE"
"""
    
    cron_file = Path("/tmp/quantum_soul_cron.sh")
    with open(cron_file, 'w') as f:
        f.write(cron_script)
    
    # Make executable
    cron_file.chmod(0o755)
    
    logger.info(f"Cron script created: {cron_file}")
    logger.info("To schedule hourly runs, add to crontab:")
    logger.info(f"  0 * * * * {cron_file}")
    
    return cron_file

if __name__ == "__main__":
    # Check for command line arguments
    cron_file = None
    if len(sys.argv) > 1:
        cron_file = sys.argv[1]
    
    # Process cron data
    success = process_cron_data(cron_output_file=cron_file)
    
    # Create cron job script if successful
    if success:
        cron_script = create_cron_job()
        print(f"\n✅ Quantum soul cron integration completed successfully!")
        print(f"📋 Cron script created: {cron_script}")
    else:
        print(f"\n❌ Quantum soul cron integration failed!")
        sys.exit(1)