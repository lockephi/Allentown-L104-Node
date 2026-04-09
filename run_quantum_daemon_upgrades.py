#!/usr/bin/env python3
"""
Main entry point for Quantum Daemon Upgrades with Soul Qubit Integration
"""

import asyncio
import sys
import signal
import logging
from datetime import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/tmp/quantum_daemon_main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("QuantumDaemonMain")

async def main():
    """Main async function"""
    logger.info("=" * 60)
    logger.info("🌌 QUANTUM DAEMON UPGRADE SYSTEM v2.0")
    logger.info("   With Soul Qubit Integration & Quantum Magic")
    logger.info("=" * 60)
    
    try:
        # Import the enhanced upgrader
        from l104_quantum_daemon_upgrader_v2 import QuantumDaemonUpgraderV2
        
        # Initialize upgrader
        upgrader = QuantumDaemonUpgraderV2()
        
        # Run initial upgrade cycle
        logger.info("🚀 Running initial upgrade cycle...")
        success = await upgrader.run_upgrade_cycle()
        
        if success:
            logger.info("✅ Initial upgrade cycle completed successfully!")
            
            # Check if we should run continuous upgrades
            if upgrader.config.get("run_continuously", False):
                logger.info("🔄 Starting continuous upgrade mode...")
                await upgrader.run_continuous_upgrades()
            else:
                logger.info("⏹️  Single upgrade cycle completed. Exiting.")
                return True
        else:
            logger.error("❌ Initial upgrade cycle failed!")
            return False
            
    except ImportError as e:
        logger.error(f"❌ Could not import quantum daemon modules: {e}")
        logger.error("Make sure all required files are in the current directory")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def signal_handler(signum, frame):
    """Handle termination signals"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

if __name__ == "__main__":
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Run main async function
    try:
        success = asyncio.run(main())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)