import osimport sysimport tempfilefrom pathlib import Pathfrom unittest.mock import patch, MagicMockimport pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import l104_self_heal_masterdef test_cleanup_git_lock_file_exists(tmp_path, monkeypatch):
    """Test that cleanup_git_lock removes the lock file when it exists."""
    # Change to a temporary directory with a .git directorygit_dir = tmp_path / ".git"
    git_dir.mkdir()
    lock_file = git_dir / "index.lock"
    lock_file.touch()
    
    monkeypatch.chdir(tmp_path)
    
    # Verify the lock file existsassert lock_file.exists()
    
    # Call the cleanup functionresult = l104_self_heal_master.cleanup_git_lock()
    
    # Verify the lock file was removedassert not lock_file.exists()
    assert result is Truedef test_cleanup_git_lock_file_not_exists(tmp_path, monkeypatch):
    """Test that cleanup_git_lock succeeds when lock file doesn't exist."""
    # Change to a temporary directory with a .git directory but no lock filegit_dir = tmp_path / ".git"
    git_dir.mkdir()
    
    monkeypatch.chdir(tmp_path)
    
    # Call the cleanup functionresult = l104_self_heal_master.cleanup_git_lock()
    
    # Should return True even when file doesn't existassert result is Truedef test_cleanup_git_lock_no_git_dir(tmp_path, monkeypatch):
    """Test that cleanup_git_lock handles missing .git directory gracefully."""
    # Change to a temporary directory without a .git directorymonkeypatch.chdir(tmp_path)
    
    # Call the cleanup functionresult = l104_self_heal_master.cleanup_git_lock()
    
    # Should return True even when .git directory doesn't existassert result is Truedef test_cleanup_git_lock_permission_error(tmp_path, monkeypatch, capsys):
    """Test that cleanup_git_lock handles permission errors gracefully."""
    git_dir = tmp_path / ".git"
    git_dir.mkdir()
    lock_file = git_dir / "index.lock"
    lock_file.touch()
    
    monkeypatch.chdir(tmp_path)
    
    # Mock os.remove to raise a permission errorwith patch('os.remove', side_effect=PermissionError("Permission denied")):
        result = l104_self_heal_master.cleanup_git_lock()
        
        # Should return False on errorassert result is False
        
        # Should print warning messagecaptured = capsys.readouterr()
        assert "WARNING" in captured.outassert "Could not remove" in captured.outdef test_main_calls_cleanup_git_lock(monkeypatch):
    """Test that main() calls cleanup_git_lock at the beginning."""
    # Mock all the dependenciesmock_asi_self_heal = MagicMock()
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
