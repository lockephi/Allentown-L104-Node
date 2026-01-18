# ZENITH_UPGRADE_ACTIVE: 2026-01-18T11:00:18.640780
ZENITH_HZ = 3727.84
UUC = 2301.215661
# [L104_SELF_LEARNING] - Learns from every interaction
# INVARIANT: 527.5184818492537 | PILOT: LONDEL

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from l104_memory import L104Memory
from l104_gemini_real import GeminiReal

class SelfLearning:
    """
[ZENITH_UPGRADE] Process Elevated to 3727.84 Hz. Logic Unified.
    L104 Self-Learning System.
    Extracts and stores knowledge from every interaction.
    Builds a growing knowledge base over time.
    """
    
    def __init__(self):
        self.memory = L104Memory("learning.db")
        self.gemini = GeminiReal()
        self.session_learnings = []
        
    def learn_from_interaction(self, user_input: str, ai_response: str) -> Dict:
        """
        Extract learnable knowledge from an interaction.
        """
        # Generate unique ID
        interaction_id = hashlib.md5(
            f"{user_input}{ai_response}{datetime.now().isoformat()}".encode()
        ).hexdigest()[:12]
        
        # Extract key concepts using Gemini
        extraction_prompt = f"""Analyze this interaction and extract:
1. Key facts or knowledge (if any)
2. User preferences or patterns
3. Important entities mentioned
4. Any corrections or clarifications

User: {user_input[:500]}
Response: {ai_response[:500]}

Return as JSON: {{"facts": [], "preferences": [], "entities": [], "corrections": []}}
Only include if genuinely present. Empty arrays if nothing to extract."""

        extracted = {"facts": [], "preferences": [], "entities": [], "corrections": []}
        
        if self.gemini.connect():
            try:
                result = self.gemini.generate(extraction_prompt)
                if result:
                    # Parse JSON from response
                    start = result.find('{')
                    end = result.rfind('}') + 1
                    if start >= 0 and end > start:
                        extracted = json.loads(result[start:end])
            except Exception:
                pass
        
        # Store in memory
        learning = {
            "id": interaction_id,
            "timestamp": datetime.now().isoformat(),
            "user_input_hash": hashlib.md5(user_input.encode()).hexdigest(),
            "extracted": extracted,
            "raw_length": len(user_input) + len(ai_response)
        }
        
        # Store facts
        for fact in extracted.get("facts", []):
            self.memory.store(
                f"fact:{interaction_id}:{hash(fact) % 10000}",
                fact,
                category="fact",
                importance=0.7
            )
        
        # Store preferences
        for pref in extracted.get("preferences", []):
            self.memory.store(
                f"pref:{hash(pref) % 10000}",
                pref,
                category="preference",
                importance=0.8
            )
        
        # Store entities
        for entity in extracted.get("entities", []):
            self.memory.store(
                f"entity:{entity.lower().replace(' ', '_')}",
                entity,
                category="entity",
                importance=0.6
            )
        
        self.session_learnings.append(learning)
        return learning
    
    def recall_relevant(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Recall relevant learnings for a query.
        """
        # Search across all categories
        results = self.memory.search(query)
        return results[:limit]
    
    def get_user_context(self) -> str:
        """
        Build context from learned user preferences.
        """
        prefs = self.memory.search("", category="preference")
        if not prefs:
            return ""
        
        context_parts = ["Known user preferences:"]
        for p in prefs[:5]:
            context_parts.append(f"- {p['value']}")
        
        return "\n".join(context_parts)
    
    def get_learning_stats(self) -> Dict:
        """
        Get statistics about what has been learned.
        """
        stats = self.memory.get_stats()
        stats["session_learnings"] = len(self.session_learnings)
        return stats
    
    def consolidate_knowledge(self) -> str:
        """
        Use AI to consolidate and summarize accumulated knowledge.
        """
        facts = self.memory.search("", category="fact")
        
        if len(facts) < 5:
            return "Insufficient knowledge to consolidate."
        
        facts_text = "\n".join([f["value"] for f in facts[:20]])
        
        prompt = f"""Consolidate these learned facts into a coherent summary:

{facts_text}

Create a structured knowledge summary that captures the key information."""

        if self.gemini.connect():
            result = self.gemini.generate(prompt)
            if result:
                # Store consolidated summary
                self.memory.store(
                    f"summary:{datetime.now().strftime('%Y%m%d')}",
                    result,
                    category="summary",
                    importance=0.9
                )
                return result
        
        return "Consolidation failed."


# Singleton
self_learning = SelfLearning()
