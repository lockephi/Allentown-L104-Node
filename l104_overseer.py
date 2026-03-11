# l104_overseer.py
# An autonomous daemon that monitors the L104 agent's health and performance.
# It links with the existing agent by creating tasks in the unified database.

import sqlite3
import time
import uuid
from pathlib import Path
from datetime import datetime, timedelta

# --- Configuration ---
DB_PATH = Path(__file__).parent / "l104_unified.db"
LOOP_INTERVAL_SECONDS = 300  # 5 minutes
STALE_GOAL_THRESHOLD_HOURS = 24
MAINTENANCE_SCHEDULE_HOURS = 24
PERFORMANCE_WINDOW_HOURS = 6

# --- Helper Functions ---
def print_log(message):
    """Prints a timestamped log message."""
    print(f"[{datetime.now().isoformat()}] [Overseer] {message}")

def execute_db_command(command, params=()):
    """Connects to the DB and executes a write command."""
    if not DB_PATH.exists():
        print_log(f"ERROR: Database file not found at {DB_PATH}")
        return False
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(command, params)
            conn.commit()
        return True
    except sqlite3.Error as e:
        print_log(f"ERROR: Database command failed: {e}")
        return False

def query_db(query, params=()):
    """Connects to the DB, executes a query, and returns results."""
    if not DB_PATH.exists():
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    except sqlite3.Error as e:
        print_log(f"ERROR: Database query failed: {e}")
        return None

def create_task(title, description, priority=2):
    """Creates a new task in the tasks table for the main agent."""
    task_id = f"task_{uuid.uuid4()}"
    print_log(f"Creating new task '{title}' with priority {priority}.")
    success = execute_db_command(
        "INSERT INTO tasks (id, title, description, priority, status) VALUES (?, ?, ?, ?, 'pending')",
        (task_id, title, description, priority)
    )
    if success:
        print_log(f"Successfully created task {task_id}.")
    else:
        print_log(f"Failed to create task '{title}'.")

# --- Overseer Modules ---

def check_stale_goals():
    """Identifies goals that haven't been updated and creates tasks to review them."""
    print_log("Checking for stale goals...")
    stale_threshold = (datetime.now() - timedelta(hours=STALE_GOAL_THRESHOLD_HOURS)).timestamp()
    
    stale_goals = query_db(
        "SELECT id, goal, progress FROM agent_goals WHERE status != 'completed' AND created_at < ?",
        (stale_threshold,)
    )
    
    if stale_goals:
        for goal_id, goal_text, progress in stale_goals:
            print_log(f"Found stale goal (ID: {goal_id}): '{goal_text}'")
            # Check if a task for this stale goal already exists
            task_exists = query_db("SELECT id FROM tasks WHERE title = ?", (f"Review stale goal: {goal_id}",))
            if not task_exists:
                create_task(
                    title=f"Review stale goal: {goal_id}",
                    description=f"The goal '{goal_text}' has not been updated in over {STALE_GOAL_THRESHOLD_HOURS} hours. Last known progress: '{progress}'. Please review, update, or close this goal.",
                    priority=1 # High priority
                )
            else:
                print_log(f"A review task for stale goal {goal_id} already exists. Skipping.")

def check_performance_metrics():
    """Monitors performance metrics for negative trends."""
    print_log("Checking performance metrics...")
    window_start = (datetime.now() - timedelta(hours=PERFORMANCE_WINDOW_HOURS)).timestamp()
    
    # Example check: Look for negative evolution scores in the last N hours
    negative_evolutions = query_db(
        "SELECT improvement, (score_after - score_before) as score_change FROM evolution_log WHERE timestamp > ? AND score_change < 0",
        (window_start,)
    )
    
    if negative_evolutions:
        for improvement, score_change in negative_evolutions:
            print_log(f"PERFORMANCE ALERT: Negative evolution detected (Score change: {score_change:.4f})")
            task_title = f"Investigate negative evolution: {improvement[:50]}..."
            task_exists = query_db("SELECT id FROM tasks WHERE title = ?", (task_title,))
            if not task_exists:
                create_task(
                    title=task_title,
                    description=f"A recent self-improvement cycle resulted in a negative score change of {score_change:.4f}. The improvement was: '{improvement}'. Please investigate the cause.",
                    priority=1
                )

def schedule_maintenance():
    """Schedules a routine database maintenance task."""
    print_log("Checking maintenance schedule...")
    
    # Check if a maintenance task was created in the last N hours
    last_maintenance_task = query_db(
        "SELECT created_at FROM tasks WHERE title = 'Perform routine database maintenance' ORDER BY created_at DESC LIMIT 1"
    )
    
    due_for_maintenance = True
    if last_maintenance_task:
        last_task_time = datetime.fromisoformat(last_maintenance_task[0][0])
        if datetime.now() - last_task_time < timedelta(hours=MAINTENANCE_SCHEDULE_HOURS):
            due_for_maintenance = False
            print_log("Maintenance task was recently created. Skipping.")

    if due_for_maintenance:
        create_task(
            title="Perform routine database maintenance",
            description=f"Perform routine VACUUM and re-indexing on the primary databases ({DB_PATH.name}, etc.) to ensure optimal performance.",
            priority=3 # Low priority
        )

def main_loop():
    """The main daemon loop."""
    print_log("L104 Overseer Agent starting up.")
    print_log(f"Database target: {DB_PATH}")
    print_log(f"Loop interval: {LOOP_INTERVAL_SECONDS} seconds.")
    
    while True:
        print_log("Starting new monitoring cycle.")
        check_stale_goals()
        check_performance_metrics()
        schedule_maintenance()
        print_log(f"Cycle complete. Sleeping for {LOOP_INTERVAL_SECONDS} seconds.")
        time.sleep(LOOP_INTERVAL_SECONDS)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        print_log("Overseer Agent shutting down.")
