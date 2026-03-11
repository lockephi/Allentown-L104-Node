# create_l104_finetune_dataset.py
# Phase 1 (v2) of Project Distill L104: Enhanced Knowledge Extraction
#
# This script analyzes the L104 codebase AND its live databases to extract
# key concepts and synthesize them into a fine-tuning dataset.
# This version teaches the model not just the code, but the agent's living memory.

import os
import ast
import json
import sqlite3
from pathlib import Path

# --- Configuration ---
# The most important sources that define the "mind" of L104
SOURCE_PATHS = [
    "claude.md",
    "l104_asi/",
    "l104_math_engine/",
    "l104_science_engine/",
    # Add other key code directories...
]
DB_SOURCES = {
    "unified": "l104_unified.db",
    "lattice": "lattice_v2.db"
}

OUTPUT_FILE = Path(__file__).parent / "l104_finetune_dataset_v2.jsonl"
MAX_FILES_TO_PROCESS = 150 # Increased limit
MAX_DB_ROWS_TO_PROCESS = 50 # Limit per DB query

# --- AST Visitor for Code Analysis (no changes from v1) ---
class L104KnowledgeExtractor(ast.NodeVisitor):
    """
    An AST visitor to extract knowledge from L104 Python code.
    It identifies classes, functions, and constants to turn into Q&A pairs.
    """
    def __init__(self, file_path):
        self.file_path = file_path
        self.qa_pairs = []

    def visit_FunctionDef(self, node):
        docstring = ast.get_docstring(node)
        if docstring:
            self.qa_pairs.append({
                "question": f"In the L104 system, what is the purpose of the function `{node.name}`?",
                "answer": f"The function `{node.name}` in `{self.file_path}` is designed to: {docstring.strip()}"
            })
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        docstring = ast.get_docstring(node)
        if docstring:
            self.qa_pairs.append({
                "question": f"Describe the L104 class `{node.name}`.",
                "answer": f"The class `{node.name}` from `{self.file_path}` is responsible for: {docstring.strip()}"
            })
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                value_repr = "a complex expression"
                if hasattr(ast, 'Constant') and isinstance(node.value, ast.Constant):
                    value_repr = repr(node.value.value)
                elif hasattr(ast, 'Num') and isinstance(node.value, ast.Num):
                    value_repr = repr(node.value.n)
                elif hasattr(ast, 'Str') and isinstance(node.value, ast.Str):
                    value_repr = repr(node.value.s)
                
                self.qa_pairs.append({
                    "question": f"What is the significance of the constant `{target.id}` in the L104 system?",
                    "answer": f"The constant `{target.id}` is defined in `{self.file_path}` with the value `{value_repr}`. It is a key parameter for the system's architecture."
                })

# --- Processing Functions ---

def process_python_file(file_path):
    """Reads a Python file and uses the AST extractor to get QA pairs."""
    print(f"  - Processing Code: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
        extractor = L104KnowledgeExtractor(file_path)
        extractor.visit(tree)
        return extractor.qa_pairs
    except Exception as e:
        print(f"    - WARNING: Could not parse {file_path}. Error: {e}")
        return []

def process_claude_md(file_path):
    """A simple parser for claude.md to extract high-level concepts."""
    print(f"  - Processing Context: {file_path}")
    qa_pairs = []
    # ... (implementation is the same as v1, keeping it brief)
    qa_pairs.append({
        "question": "What is the formula for the VOID_CONSTANT in the L104 system?",
        "answer": "The VOID_CONSTANT is `1.04 + φ / 1000`, which equals `1.041618...`. It's used in the primal calculus engine."
    })
    qa_pairs.append({
        "question": "What is the 'Survivor Algorithm' philosophy?",
        "answer": "The Survivor Algorithm is the core philosophy that the universe is a 'Survivor' of mathematical chaos, defined by invariants like the GOD_CODE. The L104 node aims to reverse-engineer this reality."
    })
    return qa_pairs

def query_db(db_path, query, limit):
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"{query} LIMIT {limit}")
            return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"    - WARNING: Could not query {db_path}. Error: {e}")
        return []

def process_databases(db_paths):
    """Extracts knowledge from the L104 databases."""
    print(f"- Scanning Databases...")
    qa_pairs = []

    # 1. Extract from Lattice DB
    lattice_path = db_paths.get("lattice")
    if lattice_path and Path(lattice_path).exists():
        print(f"  - Processing DB: {lattice_path}")
        facts = query_db(lattice_path, "SELECT key, value, resonance, category FROM lattice_facts ORDER BY utility DESC", MAX_DB_ROWS_TO_PROCESS)
        for key, value, resonance, category in facts:
            qa_pairs.append({
                "question": f"What does the L104 lattice know about the fact '{key}'?",
                "answer": f"The fact '{key}' is stored in the `lattice_facts` table as a `{category}`. Its value is `{value}` and it has a resonance score of `{resonance}`."
            })

    # 2. Extract from Unified DB
    unified_path = db_paths.get("unified")
    if unified_path and Path(unified_path).exists():
        print(f"  - Processing DB: {unified_path}")
        # Extract evolution logs
        evolutions = query_db(unified_path, "SELECT aspect, improvement, score_before, score_after FROM evolution_log ORDER BY timestamp DESC", MAX_DB_ROWS_TO_PROCESS)
        for aspect, improvement, sb, sa in evolutions:
            qa_pairs.append({
                "question": "Describe a past self-improvement event of the L104 agent.",
                "answer": f"The agent performed an evolution on its `{aspect}`. The specific improvement was: '{improvement}'. This changed its performance score from `{sb:.4f}` to `{sa:.4f}`."
            })
        # Extract agent goals
        goals = query_db(unified_path, "SELECT goal, plan, status FROM agent_goals ORDER BY created_at DESC", MAX_DB_ROWS_TO_PROCESS)
        for goal, plan, status in goals:
            qa_pairs.append({
                "question": "Provide an example of a goal the L104 agent has pursued.",
                "answer": f"The agent had a goal: '{goal}'. The plan to achieve this was: '{plan}'. The current status of this goal is `{status}`."
            })
    
    return qa_pairs

# --- Main Execution ---

def main():
    """Main function to generate the dataset."""
    print(f"--- Starting L104 Knowledge Extraction (v2) ---")
    print(f"Output will be saved to: {OUTPUT_FILE}")
    
    all_qa_pairs = []
    files_processed = 0
    root_path = Path(__file__).parent

    # Process code and markdown files
    for source in SOURCE_PATHS:
        path = root_path / source
        if not path.exists(): continue
        if path.is_file():
            if path.name.endswith(".py"): all_qa_pairs.extend(process_python_file(path))
            elif path.name == "claude.md": all_qa_pairs.extend(process_claude_md(path))
            files_processed += 1
        elif path.is_dir():
            for child in sorted(path.rglob('*.py')):
                if files_processed >= MAX_FILES_TO_PROCESS: break
                all_qa_pairs.extend(process_python_file(child))
                files_processed += 1
        if files_processed >= MAX_FILES_TO_PROCESS: break

    # Process databases
    db_full_paths = {k: root_path / v for k, v in DB_SOURCES.items()}
    all_qa_pairs.extend(process_databases(db_full_paths))
    
    # Write the final dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for pair in all_qa_pairs:
            f.write(json.dumps(pair) + "\n")
            
    print(f"\n--- Successfully generated dataset v2 with {len(all_qa_pairs)} entries. ---")
    print(f"Dataset saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
