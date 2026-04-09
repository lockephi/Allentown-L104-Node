"""
Learning Submodule for LearningIntellect

Contains learning-related functionality:
- Predictive prefetching
- Skills tracking
- Meta-evolution
"""

from l104_server.learning.learning.predictive import PredictiveMixin
from l104_server.learning.learning.skills import SkillsMixin

__all__ = ['PredictiveMixin', 'SkillsMixin']
