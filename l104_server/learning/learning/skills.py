"""
Skills Mixin for LearningIntellect

Extracted from intellect.py during EVO_78 refactoring.
Contains: Skills tracking, proficiency management, skill chaining.
"""

from typing import Dict, List, Optional
from collections import defaultdict
import time
import pickle


class SkillsMixin:
    """Skills operations for LearningIntellect.
    
    This mixin provides:
    - Skill acquisition and tracking
    - Proficiency management
    - Skill chaining
    - Skill persistence
    
    Usage:
        class LearningIntellect(SkillsMixin, ...):
            pass
    """
    
    def _init_skills(self):
        """Initialize skills system."""
        self.skills: Dict[str, dict] = defaultdict(lambda: {
            'proficiency': 0.0,
            'usage_count': 0,
            'success_rate': 0.5,
            'sub_skills': [],
            'last_used': None
        })
        self._skill_chains: Dict[str, List[str]] = {}
        self._load_skills_from_db()

    def _load_skills_from_db(self):
        """Load skills from database."""
        try:
            conn = self._get_optimized_connection()
            c = conn.cursor()
            c.execute('SELECT skill_name, level, experience FROM skills')
            for row in c.fetchall():
                self.skills[row[0]]['proficiency'] = row[1]
                self.skills[row[0]]['usage_count'] = row[2]
        except Exception:
            pass

    def acquire_skill(self, skill_name: str, context: str, success: bool = True):
        """Acquire or improve a skill."""
        skill = self.skills[skill_name]
        
        # Update usage
        skill['usage_count'] += 1
        skill['last_used'] = time.time()
        
        # Update success rate
        if skill['usage_count'] == 1:
            skill['success_rate'] = 1.0 if success else 0.0
        else:
            # Exponential moving average
            skill['success_rate'] = (skill['success_rate'] * 0.9) + (1.0 if success else 0.0) * 0.1
        
        # Update proficiency
        # PHI-weighted learning: proficiency grows faster with successful application
        phi = 1.618033988749895
        learning_rate = 0.1 * phi if success else 0.02
        skill['proficiency'] = min(1.0, skill['proficiency'] + learning_rate / (skill['usage_count'] ** 0.5))
        
        # Extract sub-skills from context
        if context:
            concepts = self._extract_concepts(context) if hasattr(self, '_extract_concepts') else []
            for concept in concepts[:3]:
                if concept not in skill['sub_skills']:
                    skill['sub_skills'].append(concept)
        
        self._persist_single_skill(skill_name, dict(skill))
        
        return {
            'skill': skill_name,
            'proficiency': skill['proficiency'],
            'usage_count': skill['usage_count'],
            'success_rate': skill['success_rate']
        }

    def _persist_single_skill(self, skill_name: str, skill_data: dict):
        """Persist a single skill to database."""
        try:
            conn = self._get_optimized_connection()
            c = conn.cursor()
            c.execute('''INSERT OR REPLACE INTO skills 
                        (skill_name, level, experience, last_used)
                        VALUES (?, ?, ?, datetime('now'))''',
                     (skill_name, skill_data['proficiency'], skill_data['usage_count']))
            conn.commit()
        except Exception:
            pass

    def chain_skills(self, task: str) -> List[str]:
        """Determine optimal skill chain for a task."""
        # Extract required skills from task
        task_concepts = self._extract_concepts(task) if hasattr(self, '_extract_concepts') else []
        
        # Find relevant skills
        relevant_skills = []
        for skill_name, skill_data in self.skills.items():
            # Check if skill concepts overlap with task concepts
            overlap = len(set(skill_data.get('sub_skills', [])) & set(task_concepts))
            if overlap > 0:
                relevant_skills.append((skill_name, skill_data['proficiency'], overlap))
        
        # Sort by proficiency and overlap
        relevant_skills.sort(key=lambda x: (x[1] * x[2]), reverse=True)
        
        # Build skill chain
        chain = [s[0] for s in relevant_skills[:5]]
        
        # Add prerequisite skills
        for skill in chain[:]:
            prereqs = self._get_prerequisite_skills(skill)
            for prereq in prereqs:
                if prereq not in chain:
                    chain.insert(0, prereq)
        
        return chain[:10]  # Limit chain length

    def _get_prerequisite_skills(self, skill_name: str) -> List[str]:
        """Get prerequisite skills for a given skill."""
        # Common skill prerequisites
        prerequisites = {
            'reasoning': ['logic', 'pattern_recognition'],
            'creativity': ['pattern_recognition', 'association'],
            'analysis': ['observation', 'logic'],
            'synthesis': ['analysis', 'creativity'],
            'memory': ['attention', 'encoding'],
        }
        return prerequisites.get(skill_name, [])

    def get_skill_proficiency(self, skill_name: str) -> float:
        """Get proficiency level for a skill."""
        return self.skills.get(skill_name, {}).get('proficiency', 0.0)

    def get_top_skills(self, n: int = 10) -> List[tuple]:
        """Get top N skills by proficiency."""
        sorted_skills = sorted(
            self.skills.items(),
            key=lambda x: x[1].get('proficiency', 0),
            reverse=True
        )
        return [(name, data['proficiency']) for name, data in sorted_skills[:n]]
