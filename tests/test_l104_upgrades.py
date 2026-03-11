# tests/test_l104_upgrades.py
# A test suite for the new autonomous upgrades (Overseer, Dataset Generator).
# This ensures the reliability and correctness of the new systems.

import pytest
import sqlite3
import os
from pathlib import Path
import time
import uuid
import json
from datetime import datetime, timedelta

# --- Import the scripts we are testing ---
# We need to add the project root to the path to import them as modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import l104_overseer
import create_l104_finetune_dataset

# --- Fixtures for creating temporary test environments ---

@pytest.fixture
def temp_unified_db(tmp_path):
    """
    Creates a temporary, in-memory unified database for testing the Overseer.
    """
    db_path = tmp_path / "l104_unified.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the necessary tables
    cursor.execute("""
        CREATE TABLE agent_goals (
            id TEXT PRIMARY KEY, goal TEXT, status TEXT, plan TEXT, progress TEXT,
            created_at REAL, completed_at REAL
        )
    """)
    cursor.execute("""
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY, title TEXT, description TEXT, status TEXT,
            priority INTEGER, created_at TIMESTAMP, completed_at TIMESTAMP, result TEXT
        )
    """)
    
    # Insert a stale goal (30 days old)
    stale_timestamp = (datetime.now() - timedelta(days=30)).timestamp()
    cursor.execute(
        "INSERT INTO agent_goals (id, goal, status, created_at) VALUES (?, ?, ?, ?)",
        ("goal_stale_123", "An old, forgotten goal", "in_progress", stale_timestamp)
    )
    conn.commit()
    conn.close()
    
    return db_path

@pytest.fixture
def temp_l104_project(tmp_path):
    """
    Creates a temporary, miniature L104 project for testing the dataset generator.
    """
    project_dir = tmp_path / "mini_l104"
    project_dir.mkdir()

    # Create a dummy python file
    dummy_code = '''
"""A docstring for a dummy class."""
class DummyClass:
    """A docstring for a dummy function."""
    def dummy_function():
        pass
DUMMY_CONSTANT = "hello"
'''
    (project_dir / "dummy_module.py").write_text(dummy_code)

    # Create a dummy claude.md
    (project_dir / "claude.md").write_text("VOID_CONSTANT Formula")
    
    return project_dir

# --- Tests for the L104 Overseer Agent ---

def test_overseer_identifies_stale_goal_and_creates_task(temp_unified_db):
    """
    Tests the core logic of the Overseer: identifying a stale goal and creating a task.
    """
    # Point the overseer to our temporary DB
    l104_overseer.DB_PATH = temp_unified_db
    l104_overseer.STALE_GOAL_THRESHOLD_HOURS = 1 # Lower threshold for testing

    # Run the function that checks for stale goals
    l104_overseer.check_stale_goals()

    # Now, check the database to see if a task was created
    conn = sqlite3.connect(temp_unified_db)
    cursor = conn.cursor()
    cursor.execute("SELECT title, priority, status FROM tasks WHERE title LIKE ?", ('%Review stale goal%',))
    tasks = cursor.fetchall()
    conn.close()

    # Assert that a new task was created
    assert len(tasks) == 1
    
    # Assert that the task has the correct properties
    task = tasks[0]
    assert "Review stale goal: goal_stale_123" in task[0]
    assert task[1] == 1  # Priority should be high
    assert task[2] == 'pending'


# --- Tests for the Dataset Generator ---

def test_dataset_generator_runs_and_creates_output(temp_l104_project, monkeypatch):
    """
    Tests that the dataset generator can run on a project and produce a valid output file.
    """
    # Change the current directory to our temp project so the script finds the files
    monkeypatch.chdir(temp_l104_project)

    # Point the script's configuration to the dummy files
    create_l104_finetune_dataset.SOURCE_PATHS = ["dummy_module.py", "claude.md"]
    create_l104_finetune_dataset.DB_SOURCES = {} # Disable DB scanning for this test
    output_file = temp_l104_project / "test_dataset.jsonl"
    create_l104_finetune_dataset.OUTPUT_FILE = output_file

    # Run the main function of the dataset generator
    create_l104_finetune_dataset.main()

    # Assert that the output file was created
    assert output_file.exists()

    # Assert that the file contains valid JSONL content
    with open(output_file, "r") as f:
        lines = f.readlines()
        assert len(lines) > 0
        # Check that each line is a valid JSON object
        for line in lines:
            data = json.loads(line)
            assert "question" in data
            assert "answer" in data
            
    # Check for specific content we expect
    content = output_file.read_text()
    assert "DUMMY_CONSTANT" in content
    assert "DummyClass" in content
    assert "VOID_CONSTANT" in content
