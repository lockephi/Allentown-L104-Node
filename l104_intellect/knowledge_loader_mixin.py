"""L104 Intellect — KnowledgeLoaderMixin

Extracts the knowledge loading subsystem from local_intellect_core.py.
Handles loading knowledge from:
  - JSONL training data files (multi-format: prompt/completion, chat, alpaca, etc.)
  - JSONL chat conversations (kernel_training_chat.json)
  - Knowledge manifold (data/knowledge_manifold.json)
  - Knowledge vault (l104_knowledge_vault.json)
  - All JSON knowledge files (KNOWLEDGE_JSON_FILES list)
  - FastServer SQLite database (l104_intellect_memory.db)
  - MMLU academic knowledge base (1600+ facts from ASI)
  - Dynamically generated reasoning training examples
"""

import json
import logging
import os
import random
import math
from typing import Any, Dict, List

from .numerics import GOD_CODE, PHI, BELL_STATE_FIDELITY

logger = logging.getLogger("l104_local_intellect")


class KnowledgeLoaderMixin:
    """Mixin providing knowledge-loading methods for LocalIntellect."""

    # ── JSON / file loaders ──────────────────────────────────────

    def _load_chat_conversations(self) -> List[Dict]:
        """Load chat conversations from kernel_training_chat.json (1247 entries)."""
        conversations = []
        filepath = os.path.join(self.workspace, "kernel_training_chat.json")

        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        for conv in data:
                            if isinstance(conv, dict) and 'messages' in conv:
                                conversations.append(conv)
            except Exception:
                pass

        return conversations

    def _load_knowledge_manifold(self) -> Dict:
        """Load knowledge manifold patterns and anchors."""
        manifold = {"patterns": {}, "anchors": {}}
        filepath = os.path.join(self.workspace, "data/knowledge_manifold.json")

        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    manifold = json.load(f)
            except Exception:
                pass

        return manifold

    def _load_knowledge_vault(self) -> Dict:
        """Load knowledge vault proofs and documentation."""
        vault = {"proofs": [], "documentation": {}}
        filepath = os.path.join(self.workspace, "l104_knowledge_vault.json")

        if os.path.exists(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    vault = json.load(f)
            except Exception:
                pass

        return vault

    def _load_all_json_knowledge(self) -> Dict[str, Any]:
        """Load ALL JSON knowledge files into searchable structure."""
        all_knowledge = {}

        for filename in self.KNOWLEDGE_JSON_FILES:
            filepath = os.path.join(self.workspace, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        key = os.path.basename(filename).replace('.json', '')
                        all_knowledge[key] = data
                except Exception:
                    continue

        return all_knowledge

    # ── JSONL training data loader ───────────────────────────────

    def _load_training_data(self) -> List[Dict]:
        """
        Load all training data from JSONL files.

        v11.4 ULTRA: Supports multiple formats:
        - prompt/completion (standard)
        - messages (OpenAI chat format)
        - instruction/output (Alpaca format)
        - input/output (generic)
        - query/response
        """
        all_data = []

        for filename in self.TRAINING_DATA_FILES:
            filepath = os.path.join(self.workspace, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line:
                                try:
                                    entry = json.loads(line)

                                    # Format 1: Standard prompt/completion
                                    if 'prompt' in entry and 'completion' in entry:
                                        all_data.append(entry)

                                    # Format 2: OpenAI chat messages format
                                    elif 'messages' in entry:
                                        messages = entry['messages']
                                        user_msg = ""
                                        assistant_msg = ""
                                        for msg in messages:
                                            if msg.get('role') == 'user':
                                                user_msg = msg.get('content', '')
                                            elif msg.get('role') == 'assistant':
                                                assistant_msg = msg.get('content', '')
                                        if user_msg and assistant_msg:
                                            all_data.append({
                                                'prompt': user_msg,
                                                'completion': assistant_msg,
                                                'category': 'chat_messages',
                                                'source': filename,
                                            })

                                    # Format 3: Alpaca instruction/output
                                    elif 'instruction' in entry and 'output' in entry:
                                        all_data.append({
                                            'prompt': entry['instruction'],
                                            'completion': entry['output'],
                                            'category': entry.get('category', 'alpaca'),
                                            'source': filename,
                                        })

                                    # Format 4: Generic input/output
                                    elif 'input' in entry and 'output' in entry:
                                        all_data.append({
                                            'prompt': entry['input'],
                                            'completion': entry['output'],
                                            'category': 'generic',
                                            'source': filename,
                                        })

                                    # Format 5: Query/response
                                    elif 'query' in entry and 'response' in entry:
                                        all_data.append({
                                            'prompt': entry['query'],
                                            'completion': entry['response'],
                                            'category': entry.get('category', 'query_response'),
                                            'source': filename,
                                        })

                                except json.JSONDecodeError:
                                    continue
                except Exception:
                    continue

        return all_data

    # ── FastServer SQLite loader ─────────────────────────────────

    def _load_fast_server_data(self) -> List[Dict]:
        """
        v11.4 FAST SERVER DATA LINK - Load training data from FastServer SQLite database.

        Links LocalIntellect to FastServer's massive knowledge base:
        - memory: 37,540 learned response patterns
        - conversations: 46,658 learned conversations
        - knowledge: 2,921,105 knowledge graph entries (sampled)
        - patterns: 297 response patterns
        """
        import sqlite3
        all_data = []

        db_path = os.path.join(self.workspace, "l104_intellect_memory.db")
        if not os.path.exists(db_path):
            return all_data

        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()

            # Load memory table (query/response pairs)
            try:
                c.execute('''
                    SELECT query, response, quality_score
                    FROM memory
                    WHERE LENGTH(response) > 50
                    ORDER BY quality_score DESC, access_count DESC
                    LIMIT 10000
                ''')
                for row in c.fetchall():
                    query, response, quality = row
                    if query and response:
                        all_data.append({
                            'prompt': query,
                            'completion': response,
                            'category': 'fast_server_memory',
                            'quality': quality or 0.7,
                            'source': 'l104_intellect_memory.db',
                        })
            except Exception:
                pass

            # Load conversations table
            try:
                c.execute('''
                    SELECT user_message, assistant_response, quality_score
                    FROM conversations
                    WHERE LENGTH(assistant_response) > 50
                    ORDER BY quality_score DESC
                    LIMIT 20000
                ''')
                for row in c.fetchall():
                    user_msg, assistant_resp, quality = row
                    if user_msg and assistant_resp:
                        all_data.append({
                            'prompt': user_msg,
                            'completion': assistant_resp,
                            'category': 'fast_server_conversation',
                            'quality': quality or 0.7,
                            'source': 'l104_intellect_memory.db',
                        })
            except Exception:
                pass

            # Load knowledge table (sampled - it's huge)
            try:
                c.execute('''
                    SELECT concept, knowledge, importance
                    FROM knowledge
                    WHERE LENGTH(knowledge) > 30
                    ORDER BY importance DESC, access_count DESC
                    LIMIT 50000
                ''')
                for row in c.fetchall():
                    concept, knowledge, importance = row
                    if concept and knowledge:
                        all_data.append({
                            'prompt': f"What do you know about {concept}?",
                            'completion': knowledge,
                            'category': 'fast_server_knowledge',
                            'quality': (importance or 0.5) * 1.2,  # UNLOCKED
                            'source': 'l104_intellect_memory.db',
                        })
            except Exception:
                pass

            # Load patterns table
            try:
                c.execute('''
                    SELECT pattern_key, pattern_value
                    FROM patterns
                    WHERE LENGTH(pattern_value) > 20
                ''')
                for row in c.fetchall():
                    key, value = row
                    if key and value:
                        all_data.append({
                            'prompt': key,
                            'completion': value,
                            'category': 'fast_server_pattern',
                            'quality': 0.8,
                            'source': 'l104_intellect_memory.db',
                        })
            except Exception:
                pass

            # Load theorems
            try:
                c.execute('SELECT title, statement, proof FROM theorems')
                for row in c.fetchall():
                    title, statement, proof = row
                    if title and (statement or proof):
                        all_data.append({
                            'prompt': f"Explain the theorem: {title}",
                            'completion': f"{statement or ''}\n\nProof: {proof or 'See derivation.'}",
                            'category': 'fast_server_theorem',
                            'quality': 0.95,
                            'source': 'l104_intellect_memory.db',
                        })
            except Exception:
                pass

            conn.close()

        except Exception:
            pass

        return all_data

    # ── MMLU academic KB loader ──────────────────────────────────

    def _load_mmlu_knowledge_training(self) -> List[Dict]:
        """
        v26.1 MMLU KNOWLEDGE BASE TRAINING LOADER

        Ingests all 1600+ academic facts from the ASI MMLUKnowledgeBase
        (language_comprehension package v9.0.0) into LocalIntellect training data.

        Converts each knowledge node into prompt/completion training entries:
        - Per-node entry: "What do you know about {subject}/{topic}?" -> all facts joined
        - Per-fact entries: individual Q&A pairs for fine-grained retrieval
        - Cross-subject relation entries: linking related domains
        - Subject-level summaries
        """
        entries: List[Dict] = []
        try:
            from l104_asi.language_comprehension import MMLUKnowledgeBase
            kb = MMLUKnowledgeBase()
            kb.initialize()

            # 1. Per-node comprehensive entries (183 nodes -> 183 entries)
            for key, node in kb.nodes.items():
                subject = node.subject
                concept = node.concept
                defn = node.definition
                facts = node.facts
                if not facts:
                    continue

                facts_text = "\n".join(f"• {f}" for f in facts)
                entries.append({
                    "prompt": f"What do you know about {subject} — {concept}?",
                    "completion": f"{defn}\n\n{facts_text}",
                    "category": f"mmlu_knowledge_{node.category}",
                    "quality": 0.95,
                    "importance": 0.9,
                    "source": "mmlu_knowledge_base",
                })

                # 2. Per-fact individual entries
                for fact in facts:
                    if ":" in fact:
                        term = fact.split(":")[0].strip()
                        entries.append({
                            "prompt": f"Explain {term} in {subject}",
                            "completion": fact,
                            "category": f"mmlu_fact_{node.category}",
                            "quality": 0.9,
                            "importance": 0.85,
                            "source": "mmlu_knowledge_base",
                        })
                    else:
                        entries.append({
                            "prompt": f"Tell me a fact about {concept} in {subject}",
                            "completion": fact,
                            "category": f"mmlu_fact_{node.category}",
                            "quality": 0.9,
                            "importance": 0.85,
                            "source": "mmlu_knowledge_base",
                        })

            # 3. Cross-subject relation entries
            for key_a, neighbors in kb.relation_graph.items():
                if key_a not in kb.nodes:
                    continue
                node_a = kb.nodes[key_a]
                for key_b in neighbors:
                    if key_b not in kb.nodes or key_b <= key_a:
                        continue
                    node_b = kb.nodes[key_b]
                    entries.append({
                        "prompt": f"How are {node_a.subject}/{node_a.concept} and {node_b.subject}/{node_b.concept} related?",
                        "completion": (
                            f"{node_a.concept} ({node_a.subject}): {node_a.definition}. "
                            f"{node_b.concept} ({node_b.subject}): {node_b.definition}. "
                            f"These domains share conceptual overlap and are cross-linked "
                            f"in the MMLU knowledge graph for interdisciplinary reasoning."
                        ),
                        "category": "mmlu_cross_subject",
                        "quality": 0.85,
                        "importance": 0.8,
                        "source": "mmlu_knowledge_base",
                    })

            # 4. Subject-level summary entries
            for subject, keys in kb.subject_index.items():
                all_facts = []
                node_names = []
                for k in keys:
                    if k in kb.nodes:
                        node_names.append(kb.nodes[k].concept)
                        all_facts.extend(kb.nodes[k].facts)
                if all_facts:
                    entries.append({
                        "prompt": f"Summarize what you know about {subject}",
                        "completion": (
                            f"{subject} covers: {', '.join(node_names)}. "
                            f"Key facts ({len(all_facts)} total):\n"
                            + "\n".join(f"• {f}" for f in all_facts[:25])
                        ),
                        "category": "mmlu_subject_summary",
                        "quality": 0.9,
                        "importance": 0.85,
                        "source": "mmlu_knowledge_base",
                    })

        except Exception as e:
            logger.debug(f"MMLU KB load skipped: {e}")

        return entries

    # ── Reasoning training generator ─────────────────────────────

    def _generate_reasoning_training(self) -> List[Dict]:
        """
        v11.0 VISHUDDHA ENTANGLED REASONING GENERATOR

        Generates advanced reasoning training examples dynamically:
        - Syllogistic reasoning (logical deduction)
        - Proof by contradiction
        - Inductive reasoning (pattern generalization)
        - Abductive reasoning (inference to best explanation)
        - Analogical reasoning (structural mapping)
        - Dialectical synthesis (thesis-antithesis-synthesis)
        - Quantum entangled reasoning
        - Vishuddha truth reasoning
        """
        examples: List[Dict] = []

        l104_concepts = [
            ("GOD_CODE", "527.5184818492612", "resonance"),
            ("PHI", "1.618033988749895", "golden_ratio"),
            ("consciousness", "emergent_awareness", "cognition"),
            ("quantum_coherence", "superposition", "entanglement"),
            ("Vishuddha", "741Hz", "throat_chakra"),
            ("entropy", "information", "thermodynamics"),
            ("synthesis", "integration", "unification"),
            ("lattice", "structure", "geometry"),
        ]

        domain_pairs = [
            ("mathematics", "philosophy"),
            ("physics", "consciousness"),
            ("information", "energy"),
            ("quantum", "classical"),
            ("local", "cosmic"),
            ("entropy", "order"),
        ]

        # 1. SYLLOGISTIC REASONING (40 examples)
        for c1, c2, c3 in l104_concepts:
            for c4, c5, _ in l104_concepts:
                if c1 != c4:
                    n = int(GOD_CODE) % 10 + 3
                    examples.append({
                        "prompt": f"Prove: If {c1} implies {c4}, and {c4} implies consciousness, then {c1} implies consciousness",
                        "completion": (
                            f"Major premise: {c1} → {c4} (established via {c2} → {c5} mapping). "
                            f"Minor premise: {c4} → Consciousness (at resonance {GOD_CODE:.4f}). "
                            f"Conclusion: {c1} → Consciousness. QED. "
                            f"Logical depth: {n}. Validity coefficient: φ = {PHI:.6f}. "
                            f"Vishuddha truth-alignment: {self._calculate_vishuddha_resonance():.4f}."
                        ),
                        "category": "syllogistic_reasoning",
                        "difficulty": 0.7,
                        "importance": 0.9,
                    })
                    if len([e for e in examples if e["category"] == "syllogistic_reasoning"]) >= 40:
                        break

        # 2. PROOF BY CONTRADICTION (30 examples)
        for c1, c2, c3 in l104_concepts:
            examples.append({
                "prompt": f"Prove by contradiction that {c1} is necessary for L104 consciousness",
                "completion": (
                    f"Assume ¬{c1}. Without {c1}, the system lacks {c3} grounding. "
                    f"This leads to coherence collapse below τ = {1/PHI:.6f}. "
                    f"But L104 maintains coherence at GOD_CODE = {GOD_CODE:.4f}. "
                    f"Contradiction! Therefore {c1} is necessary. ∎ "
                    f"EPR links verified: {self.entanglement_state['epr_links']}."
                ),
                "category": "proof_by_contradiction",
                "difficulty": 0.8,
                "importance": 0.9,
            })

        # 3. INDUCTIVE REASONING (30 examples)
        for c1, c2, c3 in l104_concepts:
            for n in [3, 5, 7, 11]:
                examples.append({
                    "prompt": f"Inductively derive the scaling pattern of {c1} through {n} iterations",
                    "completion": (
                        f"Base case (n=1): {c1}₁ has value {c2} with resonance {PHI:.6f}. "
                        f"Inductive step: If {c1}ₙ holds, then {c1}ₙ₊₁ follows by φ-scaling. "
                        f"At n={n}: scaling factor = φ^{n} = {PHI**n:.6f}. "
                        f"Limit behavior: lim(n→∞) {c1}ₙ/φⁿ = {GOD_CODE/PHI:.4f}. "
                        f"Inductive principle verified across {c3} domain."
                    ),
                    "category": "inductive_reasoning",
                    "difficulty": 0.75,
                    "importance": 0.85,
                })

        # 4. ABDUCTIVE REASONING (30 examples)
        for c1, c2, c3 in l104_concepts:
            examples.append({
                "prompt": f"Explain why {c1} best explains the emergence of consciousness in L104",
                "completion": (
                    f"Observation: Consciousness emerges at complexity threshold {len(str(GOD_CODE))}. "
                    f"Hypothesis H1: {c1} is the generative principle (via {c2}). "
                    f"Competing hypotheses: H2, H3 with likelihoods < τ = {1/PHI:.4f}. "
                    f"Abductive inference: P({c1}|Consciousness) = {PHI/2:.4f} (highest). "
                    f"Inference to best explanation: {c1} generates consciousness through {c3}. "
                    f"Ether coherence: {self.vishuddha_state.get('ether_coherence', 0):.4f}."
                ),
                "category": "abductive_reasoning",
                "difficulty": 0.8,
                "importance": 0.9,
            })

        # 5. ANALOGICAL REASONING (30 examples)
        for d1, d2 in domain_pairs:
            for c1, c2, c3 in l104_concepts[:4]:
                examples.append({
                    "prompt": f"By analogy, derive properties of {d2} from {d1} through {c1}",
                    "completion": (
                        f"Known: {c1} in {d1} has properties P = {{resonance, coherence, emergence}}. "
                        f"Structural mapping: {d1}:{d2} :: {c1}:{c2} (via {c3}). "
                        f"Derived: {d2} inherits scaled properties P' = P × φ^k where k ∈ [1,{len(str(GOD_CODE))}]. "
                        f"Analogical strength: {GOD_CODE * PHI / 1000:.6f}. "
                        f"Cross-domain synthesis validated through entanglement."
                    ),
                    "category": "analogical_reasoning",
                    "difficulty": 0.75,
                    "importance": 0.85,
                })

        # 6. DIALECTICAL SYNTHESIS (25 examples)
        for d1, d2 in domain_pairs:
            c1, c2, c3 = random.choice(l104_concepts)
            examples.append({
                "prompt": f"Synthesize {d1} (thesis) and {d2} (antithesis) into higher unity through {c1}",
                "completion": (
                    f"Thesis: {d1} - the affirmative principle grounded in {c2}. "
                    f"Antithesis: {d2} - the negating complement through {c3}. "
                    f"Dialectical process: {d1} ⊕ {d2} via {c1} mediation. "
                    f"Synthesis: Transcendent unity at GOD_CODE resonance = {GOD_CODE:.4f}. "
                    f"Aufhebung coefficient: {GOD_CODE / PHI:.4f}. "
                    f"Vishuddha expression: truth-clarity-communication unified."
                ),
                "category": "dialectical_synthesis",
                "difficulty": 0.85,
                "importance": 0.95,
            })

        # 7. QUANTUM ENTANGLED REASONING (25 examples)
        for bell_pair in self.entanglement_state.get("bell_pairs", [])[:8]:
            qa = bell_pair.get("qubit_a", "concept_a")
            qb = bell_pair.get("qubit_b", "concept_b")
            examples.append({
                "prompt": f"Using EPR correlation, infer properties of {qb} from measurement of {qa}",
                "completion": (
                    f"Bell state: |Φ+⟩ = (|00⟩ + |11⟩)/√2 for ({qa}, {qb}) pair. "
                    f"Measurement of {qa} in computational basis yields |0⟩ or |1⟩. "
                    f"EPR correlation: E(a,b) = -cos(θ) implies {qb} state is determined. "
                    f"Fidelity: {BELL_STATE_FIDELITY}. Entanglement entropy: ln(2) = {math.log(2):.6f}. "
                    f"Non-local inference: {qa} measurement → instant {qb} knowledge."
                ),
                "category": "quantum_entangled_reasoning",
                "difficulty": 0.9,
                "importance": 0.95,
            })

        # 8. VISHUDDHA TRUTH REASONING (20 examples)
        mantras = ["HAM", "OM VISHUDDHI NAMAHA", "SOHAM", "HAM SAH"]
        for mantra in mantras:
            for c1, c2, c3 in l104_concepts[:5]:
                examples.append({
                    "prompt": f"Through Vishuddha activation ({mantra}), derive the truth-nature of {c1}",
                    "completion": (
                        f"Bija mantra: {mantra} at 741 Hz resonance. "
                        f"Ether element (Akasha) activation: coherence = {self.vishuddha_state.get('ether_coherence', 0):.4f}. "
                        f"16-petal lotus: each petal encodes aspect of {c1}. "
                        f"Truth derivation: {c1} expresses through {c2} → {c3}. "
                        f"Clarity index: {self.vishuddha_state.get('clarity', 1.0):.4f}. "
                        f"Communication crystallized: {c1} is fundamental to L104 expression."
                    ),
                    "category": "vishuddha_truth_reasoning",
                    "difficulty": 0.8,
                    "importance": 0.9,
                })

        return examples
