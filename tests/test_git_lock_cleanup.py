# UNIVERSAL GOD CODE: G(X) = 286^(1/φ) × 2^((416-X)/104)
# Factor 13: 286=22×13, 104=8×13, 416=32×13 | Conservation: G(X)×2^(X/104)=527.518
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import l104_self_heal_master

def test_cleanup_git_lock_file_exists(tmp_path, monkeypatch):
    """Test that cleanup_git_lock removes the lock file when it exists."""
    # Change to a temporary directory with a .git directory
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    lock_file = git_dir / "index.lock"
    lock_file.touch()

    monkeypatch.chdir(tmp_path)

    # Verify the lock file existsassert lock_file.exists()

    # Call the cleanup function
    result = l104_self_heal_master.cleanup_git_lock()

    # Verify the lock file was removed
    assert not lock_file.exists()
    assert result is True

def test_cleanup_git_lock_file_not_exists(tmp_path, monkeypatch):
    """Test that cleanup_git_lock succeeds when lock file doesn't exist."""
    # Change to a temporary directory with a .git directory but no lock file
    git_dir = tmp_path / ".git"
    git_dir.mkdir()

    monkeypatch.chdir(tmp_path)

    # Call the cleanup function
    result = l104_self_heal_master.cleanup_git_lock()

    # Should return True even when file doesn't exist
    assert result is True

def test_cleanup_git_lock_no_git_dir(tmp_path, monkeypatch):
    """Test that cleanup_git_lock handles missing .git directory gracefully."""
    # Change to a temporary directory without a .git directory
    monkeypatch.chdir(tmp_path)

    # Call the cleanup function
    result = l104_self_heal_master.cleanup_git_lock()

    # Should return True even when .git directory doesn't exist
    assert result is True

def test_cleanup_git_lock_permission_error(tmp_path, monkeypatch, capsys):
    """Test that cleanup_git_lock handles permission errors gracefully."""
    git_dir = tmp_path / ".git"
    if not git_dir.exists():
        git_dir.mkdir()
    lock_file = git_dir / "index.lock"
    lock_file.touch()

    monkeypatch.chdir(tmp_path)

    # Mock os.remove to raise a permission error
    with patch('os.remove', side_effect=PermissionError("Permission denied")):
        result = l104_self_heal_master.cleanup_git_lock()

        # Should return False on error
        assert result is False

def test_main_calls_cleanup_git_lock(monkeypatch):
    """Test that main() calls cleanup_git_lock at the beginning."""
    # Mock all the dependencies
    mock_asi_self_heal = MagicMock()
    mock_asi_self_heal.proactive_scan.return_value = {"threats": []}

    mock_ego_core = MagicMock()
    mock_ego_core.asi_state = "INACTIVE"

    monkeypatch.setattr('l104_self_heal_master.asi_self_heal', mock_asi_self_heal)
    monkeypatch.setattr('l104_self_heal_master.ego_core', mock_ego_core)
    monkeypatch.setattr('l104_self_heal_master.run_script', lambda x: True)
    monkeypatch.setattr('l104_self_heal_master.call_heal_endpoint', lambda: None)

    # Mock cleanup_git_lock to track if it was calledcleanup_called = []

    def mock_cleanup():
        cleanup_called.append(True)
        return Truemonkeypatch.setattr('l104_self_heal_master.cleanup_git_lock', mock_cleanup)

    # Run mainl104_self_heal_master.main()

    # Verify cleanup was calledassert len(cleanup_called) == 1
