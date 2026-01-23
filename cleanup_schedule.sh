#!/bin/bash
# L104 Automated Space Cleanup
# Runs daily cleanup to prevent space issues

cd /workspaces/Allentown-L104-Node
python3 l104_space_optimizer.py --auto-cleanup

# Log cleanup results
echo "$(date): Space cleanup completed" >> .space_cleanup.log
