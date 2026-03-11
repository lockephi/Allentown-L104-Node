# l104_dashboard.py
# A command-line utility to display the current status of the L104 agent.
# Connects to the unified database to pull real-time information on goals, tasks, and performance.

import sqlite3
import textwrap
from pathlib import Path

# --- Configuration ---
DB_PATH = Path(__file__).parent / "l104_unified.db"
SEPARATOR = "=" * 80

# --- Helper Functions ---
def print_header(title):
    """Prints a formatted header."""
    print("\n" + SEPARATOR)
    print(f" L104 Agent Dashboard: {title.upper()}")
    print(SEPARATOR)

def print_table(headers, rows):
    """Prints data in a formatted table."""
    if not rows:
        print(" | (No data found)")
        return

    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            cell_str = str(cell)
            if len(cell_str) > col_widths[i]:
                col_widths[i] = len(cell_str)

    # Print header
    header_line = " | ".join(headers[i].ljust(col_widths[i]) for i in range(len(headers)))
    print(f" | {header_line} |")
    print(f" |-{'-|-'.join('-' * w for w in col_widths)}-|")

    # Print rows
    for row in rows:
        row_line = " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(row)))
        print(f" | {row_line} |")

def query_db(query, params=()):
    """Connects to the DB, executes a query, and returns the results."""
    if not DB_PATH.exists():
        print(f"Error: Database file not found at {DB_PATH}")
        return None
    try:
        with sqlite3.connect(DB_PATH) as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

# --- Dashboard Sections ---

def show_active_goals():
    """Displays active goals from the agent_goals table."""
    print_header("Active Goals")
    rows = query_db("SELECT id, goal, status, progress FROM agent_goals WHERE status != 'completed' ORDER BY created_at DESC")
    if rows is not None:
        print_table(["ID", "Goal", "Status", "Progress"], rows)

def show_pending_tasks():
    """Displays pending tasks from the tasks table."""
    print_header("Pending Tasks")
    rows = query_db("SELECT id, title, priority, status FROM tasks WHERE status = 'pending' ORDER BY priority ASC, created_at DESC LIMIT 10")
    if rows is not None:
        print_table(["ID", "Title", "Priority", "Status"], rows)

def show_recent_performance():
    """Displays the last 5 performance metrics."""
    print_header("Recent Performance Metrics")
    rows = query_db("SELECT metric_name, value, context, DATETIME(timestamp, 'unixepoch') FROM performance_metrics ORDER BY timestamp DESC LIMIT 5")
    if rows is not None:
        print_table(["Metric", "Value", "Context", "Timestamp"], rows)

def show_last_evolution():
    """Displays the most recent evolution log entry."""
    print_header("Last Evolution")
    rows = query_db("SELECT aspect, improvement, score_before, score_after, DATETIME(timestamp, 'unixepoch') FROM evolution_log ORDER BY timestamp DESC LIMIT 1")
    if rows:
        row = rows[0]
        print(f"  Aspect:      {row[0]}")
        print(f"  Improvement: {textwrap.shorten(row[1], width=100)}")
        print(f"  Score:       {row[2]:.4f} -> {row[3]:.4f} (Change: {row[3] - row[2]:+.4f})")
        print(f"  Timestamp:   {row[4]}")
    else:
        print(" (No evolution events found)")


def main():
    """Main function to run the dashboard."""
    print(SEPARATOR)
    print(" L104 AGENT STATUS REPORT")
    
    show_active_goals()
    show_pending_tasks()
    show_recent_performance()
    show_last_evolution()
    
    print("\n" + SEPARATOR)
    print(" Report complete.")
    print(SEPARATOR)

if __name__ == "__main__":
    main()
