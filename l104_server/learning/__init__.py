"""
L104 Learning Package

Contains LearningIntellect and its modular components:

Mixins (for composition):
- MemoryCacheMixin: Database, cache, persistence
- PredictiveMixin: Predictive prefetching, novelty
- SkillsMixin: Skills tracking and chaining
- ConsciousnessMixin: Consciousness clusters
- QualityMixin: Response quality prediction

Usage:
    from l104_server.learning import LearningIntellect, intellect, grover_kernel
    from l104_server.learning.memory import MemoryCacheMixin
    from l104_server.learning.learning import PredictiveMixin, SkillsMixin
    from l104_server.learning.cognition import ConsciousnessMixin, QualityMixin
"""

# Import mixins for composition
from l104_server.learning.memory.cache import MemoryCacheMixin
from l104_server.learning.learning.predictive import PredictiveMixin
from l104_server.learning.learning.skills import SkillsMixin
from l104_server.learning.cognition.consciousness import ConsciousnessMixin
from l104_server.learning.cognition.quality import QualityMixin

# Import main class (still in intellect.py for now)
# The class inherits from all mixins
try:
    from l104_server.learning.intellect import LearningIntellect, intellect, grover_kernel
except ImportError:
    # Define fallback if intellect.py has import issues
    LearningIntellect = None
    intellect = None
    grover_kernel = None

__all__ = [
    # Main class and instance
    'LearningIntellect',
    'intellect',
    'grover_kernel',
    # Mixins for composition
    'MemoryCacheMixin',
    'PredictiveMixin',
    'SkillsMixin',
    'ConsciousnessMixin',
    'QualityMixin',
]
