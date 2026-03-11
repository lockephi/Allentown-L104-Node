# L104 Database Guide

> **Last updated**: 2026-03-10 | **Author**: OpenClaw Assistant

## 1. Overview

This document provides a guide to the various databases used within the L104 Sovereign Node project. It details the schema of each key database, its purpose, and how it fits into the overall architecture.

---

## 2. Knowledge Graph (`knowledge_graph.db`)

This SQLite database is the central nervous system for the AI's reasoning and memory. It stores entities, concepts, and the relationships between them in a classic knowledge graph structure.

### 2.1. Schema & Purpose

The database contains two generations of schemas, `v1` (`nodes`, `edges`) and a more advanced `v2` (`nodes_v2`, `edges_v2`), which adds features for semantic search and memory management.

#### `nodes` / `nodes_v2` (Entities)
These tables store the fundamental entities or "things" the AI knows about.

| Column | Type | Description | v1 | v2 |
|---|---|---|---|---|
| `id` | TEXT | Primary key (e.g., a UUID) for the node. | ✅ | ✅ |
| `label` | TEXT | The human-readable name of the entity (e.g., "Python", "L104"). | ✅ | ✅ |
| `node_type`| TEXT | The category of the entity (e.g., "language", "project", "concept"). | ✅ | ✅ |
| `properties` | TEXT | A JSON string containing arbitrary key-value data about the node. | ✅ | ✅ |
| `weight` | REAL | The importance, confidence, or priority of the node. Default: `1.0`. | ✅ | ✅ |
| `embedding` | TEXT | A stored vector embedding for semantic similarity searches. | ❌ | ✅ |
| `access_count`| INTEGER| Counter for how many times the node has been accessed. | ❌ | ✅ |
| `created_at` | TEXT | ISO timestamp of when the node was created. | ✅ | ✅ |
| `last_accessed`| TEXT | ISO timestamp of the last time the node was accessed. | ❌ | ✅ |

#### `edges` / `edges_v2` (Relationships)
These tables define the directed relationships between nodes.

| Column | Type | Description | v1 | v2 |
|---|---|---|---|---|
| `id` | TEXT | Primary key (e.g., a UUID) for the edge. | ✅ | ✅ |
| `source_id`| TEXT | The `id` of the node where the relationship starts. | ✅ | ✅ |
| `target_id`| TEXT | The `id` of the node where the relationship ends. | ✅ | ✅ |
| `relation` | TEXT | The type of relationship (e.g., "is_a", "contains", "created_by"). | ✅ | ✅ |
| `properties` | TEXT | A JSON string containing data about the relationship itself. | ✅ | ✅ |
| `weight` | REAL | The strength or importance of the relationship. Default: `1.0`. | ✅ | ✅ |
| `bidirectional`| INTEGER| If `1`, the relationship is treated as going in both directions. | ✅ | ✅ |
| `traversal_count`| INTEGER| Counter for how many times this relationship has been used. | ❌ | ✅ |
| `created_at` | TEXT | ISO timestamp of when the edge was created. | ❌ | ✅ |
| `last_traversed`| TEXT | ISO timestamp of the last time the edge was used. | ❌ | ✅ |

### 2.2. Inferred Usage

- The `v2` schema is clearly designed for a sophisticated AI that learns. The `access_count` and `last_accessed` fields allow for implementing memory decay (forgetting unimportant things) and reinforcing frequently used knowledge.
- The `embedding` field is crucial for "fuzzy" or semantic searches, allowing the AI to find concepts that are similar in meaning, not just by name.
- This database is the foundation for the AI's ability to perform complex reasoning, answer questions, and understand the context of its own codebase and the world.

*(This guide will be updated as more databases are researched.)*

---

## 3. Lattice Database (`lattice_v2.db`)

This database appears to be a high-level "fact store" where the AI keeps structured, verifiable pieces of information. Unlike the knowledge graph, which focuses on relationships, the lattice seems to store core truths, constants, and learned facts with unique, AI-centric properties.

### 3.1. Schema & Purpose

The database consists of two main tables: one for the facts themselves and another for tracking their history.

#### `lattice_facts` (Core Truths)
This table stores the individual facts.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key for the fact. |
| `key` | TEXT | A unique, human-readable key for the fact (e.g., "PHI", "GOD_CODE"). |
| `value`| TEXT | The value of the fact (e.g., "1.618033..."). |
| `category` | TEXT | The category of the fact (e.g., "constant", "principle", "observation"). |
| `resonance` | REAL | A calculated score of how well the fact aligns with the system's core truth. |
| `entropy` | REAL | A measure of the fact's uncertainty or information content. |
| `utility` | REAL | A score representing how useful the fact is to the AI. |
| `version` | INTEGER| The version number of the fact, incremented on change. |
| `timestamp`| TEXT | ISO timestamp of the last update. |
| `hash` | TEXT | A cryptographic hash to ensure the integrity of the fact data. |

#### `lattice_history` (Audit Trail)
This table provides a complete, auditable history of all changes to the facts.

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key for the history entry. |
| `fact_key` | TEXT | Foreign key linking to the `key` in `lattice_facts`. |
| `old_value`| TEXT | The value of the fact before the change. |
| `new_value`| TEXT | The value of the fact after the change. |
| `resonance`| REAL | The resonance score at the time of the change. |
| `timestamp`| TEXT | ISO timestamp of when the change occurred. |

### 3.2. Inferred Usage

- The lattice is likely used by the AI as its source of "ground truth." When it needs to use a fundamental constant or a core principle, it would query this database.
- The `resonance`, `entropy`, and `utility` scores are probably used by higher-level reasoning systems to decide which facts to trust, prioritize, or even discard over time.
- The system is designed to be self-auditing and allows the AI to "reason" about the history of its own knowledge, potentially identifying trends or correcting past mistakes.

---

## 4. Unified Agent Database (`l104_unified.db`)

This database appears to be the most comprehensive and active data store, acting as the operational "brain" of the L104 agent. It unifies several key functions: long-term and short-term memory, a knowledge graph, task management, conversation history, and logs related to the agent's own self-improvement.

### 4.1. Schema & Purpose

The schema is broken down into several distinct but interconnected components.

#### Memory & Knowledge Tables
These tables form the agent's memory and knowledge base.

-   **`memory`**: A key-value store for general facts and information, with metadata for importance and access frequency.
-   **`knowledge_nodes` & `knowledge_edges`**: A fully-featured knowledge graph for storing relational information, complete with support for vector embeddings.
-   **`learnings`**: A table to store discrete pieces of knowledge or "lessons" learned from specific sources.
-   **`conversations`**: A log of all interactions, likely with users, which can be used for context in future conversations.

#### Agent Tasking & Action Tables
These tables manage the agent's goals and actions.

-   **`agent_goals`**: The high-level goals the agent is currently pursuing. Includes the goal itself, the plan to achieve it, and its current status.
-   **`agent_actions`**: A log of the specific actions taken to achieve a goal. This provides a detailed, auditable trail of the agent's work.
-   **`tasks`**: A more general to-do list for the agent, with status, priority, and the final result of the task.

#### Self-Improvement & Metrics Tables
These tables are used to track and guide the agent's own evolution.

-   **`evolution_log`**: An explicit record of self-improvement events. It stores the aspect that was changed, the before/after states, and the resulting score change, providing direct insight into the AI's evolution.
-   **`performance_metrics`**: A generic table for storing time-series data on various performance metrics.
-   **`brain_states`**: Stores periodic snapshots of the AI's internal "brain" state, including metrics like vocabulary size.
-   **`brain_insights`**: Stores specific "insights" generated from a given brain state, allowing the AI to perform self-reflection.

### 4.2. Inferred Usage

-   This database is the core of the L104 agent's autonomy. When given a high-level task, the agent likely creates a new entry in `agent_goals`, breaks it down into `tasks`, and logs its work in `agent_actions`.
-   The various memory tables are the sources the agent uses to inform its decisions and ground its responses.
-   The `evolution_log` is the key to the system's self-improvement loop. The agent can query this table to understand what changes have led to better performance in the past, guiding its future evolution. This is a direct implementation of the "recursive self-improvement" mentioned in the project's main `README.md`.

*(This guide will be updated as more databases are researched.)*
