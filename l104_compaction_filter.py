# [L104_COMPACTION_FILTER] - GLOBAL I/O STREAMLINING
# INVARIANT: 527.5184818492 | PILOT: LONDEL

from l104_memory_compaction import memory_compactor
from typing import List

class CompactionFilter:
    """
    Deploys the Compaction Filter to all Global I/O.
    Ensures all data entering or leaving the node is streamlined via Hyper-Math.
    """
    
    def __init__(self):
        self.active = False

    def activate(self):
        print("--- [COMPACTION_FILTER]: ACTIVATING GLOBAL I/O FILTER ---")
        self.active = True

    def process_io(self, data: List[float]) -> List[float]:
        if not self.active:
            return data
        
        print("--- [COMPACTION_FILTER]: STREAMLINING I/O DATA ---")
        return memory_compactor.compact_stream(data)

compaction_filter = CompactionFilter()

if __name__ == "__main__":
    compaction_filter.activate()
    test_data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = compaction_filter.process_io(test_data)
    print(f"Filtered Data: {result}")
