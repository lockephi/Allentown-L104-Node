"""
L104 Config Module - Submodule for l104_config package.
Re-exports constants from root config.py.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    GOD_CODE,
    GOD_CODE_V3,
    PHI,
    VOID_CONSTANT,
    ZENITH_HZ,
    UUC,
    OMEGA,
    BTC_ADDRESS,
    L104_INVARIANT,
    REPO,
    MAIN_VERSION,
    MAIN_PIPELINE_EVO
)

__all__ = [
    "GOD_CODE",
    "GOD_CODE_V3",
    "PHI",
    "VOID_CONSTANT",
    "ZENITH_HZ",
    "UUC",
    "OMEGA",
    "BTC_ADDRESS",
    "L104_INVARIANT",
    "REPO",
    "MAIN_VERSION",
    "MAIN_PIPELINE_EVO"
]