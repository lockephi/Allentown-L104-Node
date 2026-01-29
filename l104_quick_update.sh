#!/bin/bash
# [L104_QUICK_UPDATE] - FAST REBUILD OF ALL MODALITIES
# INVARIANT: 527.5184818492611 | PILOT: LONDEL

echo "==================================================="
echo "   L104 SOVEREIGN NODE :: QUICK UPDATE PROCESS"
echo "==================================================="

# 1. Run Modality Sync
echo ">>> [STEP 1]: SYNCING LOGIC ACROSS MODALITIES..."
python3 l104_modality_sync.py

# 2. Recompile C++ Core
echo ">>> [STEP 2]: RECOMPILING C++ CORE..."
if command -v g++ &> /dev/null; then
    g++ l104_core.cpp -o l104_core_cpp
    echo "--- [CPP]: REBUILD SUCCESSFUL ---"
else
    echo "--- [CPP]: WARNING -> g++ NOT FOUND, SKIPPING ---"
fi

# 3. Recompile Java Core
echo ">>> [STEP 3]: RECOMPILING JAVA CORE..."
if command -v javac &> /dev/null; then
    mkdir -p com/l104/sovereign
    cp L104Core.java com/l104/sovereign/
    javac com/l104/sovereign/L104Core.java
    echo "--- [JAVA]: REBUILD SUCCESSFUL ---"
else
    echo "--- [JAVA]: WARNING -> javac NOT FOUND, SKIPPING ---"
fi

# 4. Verify Python Singularity
echo ">>> [STEP 4]: VERIFYING PYTHON SINGULARITY..."
python3 -c "import l104_asi_core; print('--- [PYTHON]: ASI CORE LOADED ---')"

echo "==================================================="
echo "   UPDATE COMPLETE :: ALL MODALITIES SYNCED"
echo "==================================================="
