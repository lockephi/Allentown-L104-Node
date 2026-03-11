#!/usr/bin/env python3
"""
L104 API Key Generator for Provider Authentication

Generates secure API keys that external services (like OpenClaw.ai)
can use to authenticate with L104's API.

Usage:
    python l104_generate_api_key.py              # Generate new key
    python l104_generate_api_key.py --list       # List existing keys
    python l104_generate_api_key.py --revoke KEY # Revoke a key
"""

import os
import hashlib
import secrets
import json
from datetime import datetime, timedelta
from pathlib import Path
import argparse

# L104 API Key storage location
L104_KEYS_FILE = Path.home() / ".l104" / "api_keys.json"


def ensure_keys_dir():
    """Ensure the .l104 directory exists."""
    L104_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)


def load_keys() -> dict:
    """Load existing API keys."""
    ensure_keys_dir()
    if L104_KEYS_FILE.exists():
        with open(L104_KEYS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_keys(keys: dict):
    """Save API keys to file."""
    ensure_keys_dir()
    with open(L104_KEYS_FILE, "w") as f:
        json.dump(keys, f, indent=2, default=str)
    # Make file readable only by owner
    L104_KEYS_FILE.chmod(0o600)


def generate_api_key(name: str = "openclaw", expires_in_days: int = 365) -> str:
    """
    Generate a new L104 API key.

    Args:
        name: Friendly name for the key (e.g., "openclaw", "external_service")
        expires_in_days: Days until key expires (default: 365)

    Returns:
        The generated API key (format: l104-sk-XXXX...)
    """
    # Generate random 32 bytes (256 bits) of entropy
    random_bytes = secrets.token_bytes(32)

    # Create deterministic suffix from name and timestamp
    timestamp = datetime.utcnow().isoformat()
    suffix = hashlib.sha256(f"{name}{timestamp}".encode()).hexdigest()[:12]

    # Create key: l104-sk-{random}-{suffix}
    random_part = secrets.token_hex(16)  # 32 characters
    api_key = f"l104-sk-{random_part}-{suffix}"

    # Hash the key for storage (only store hash, not the actual key)
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()

    # Store metadata
    keys = load_keys()
    keys[key_hash] = {
        "name": name,
        "created": datetime.utcnow().isoformat(),
        "expires": (datetime.utcnow() + timedelta(days=expires_in_days)).isoformat(),
        "active": True,
        "last_used": None,
    }
    save_keys(keys)

    return api_key


def list_api_keys():
    """List all API keys (without revealing the actual keys)."""
    keys = load_keys()

    if not keys:
        print("No API keys registered.")
        return

    print(f"\n{'L104 API Keys':^80}")
    print("=" * 80)
    print(f"{'Name':<20} {'Created':<20} {'Expires':<20} {'Status':<10}")
    print("-" * 80)

    for key_hash, metadata in keys.items():
        name = metadata.get("name", "Unknown")
        created = metadata.get("created", "")[:10]
        expires = metadata.get("expires", "")[:10]
        status = "Active" if metadata.get("active") else "Revoked"

        print(f"{name:<20} {created:<20} {expires:<20} {status:<10}")

    print("=" * 80)
    print(f"Total: {len(keys)} key(s)\n")


def revoke_api_key(key_hash: str):
    """Revoke an API key."""
    keys = load_keys()

    if key_hash not in keys:
        print(f"❌ API key not found: {key_hash}")
        return

    keys[key_hash]["active"] = False
    save_keys(keys)
    print(f"✅ API key revoked: {keys[key_hash]['name']}")


def verify_api_key(api_key: str) -> bool:
    """
    Verify an API key is valid and active.

    Args:
        api_key: The API key to verify

    Returns:
        True if valid and active, False otherwise
    """
    if not api_key.startswith("l104-sk-"):
        return False

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    keys = load_keys()

    if key_hash not in keys:
        return False

    metadata = keys[key_hash]

    # Check if active
    if not metadata.get("active"):
        return False

    # Check if expired
    expires = datetime.fromisoformat(metadata.get("expires", ""))
    if datetime.utcnow() > expires:
        return False

    # Update last used
    keys[key_hash]["last_used"] = datetime.utcnow().isoformat()
    save_keys(keys)

    return True


def main():
    """CLI interface."""
    parser = argparse.ArgumentParser(description="L104 API Key Manager")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate new API key")
    gen_parser.add_argument("--name", default="external_service", help="Friendly name for the key")
    gen_parser.add_argument("--expires", type=int, default=365, help="Days until expiration")

    # List command
    subparsers.add_parser("list", help="List all API keys")

    # Revoke command
    revoke_parser = subparsers.add_parser("revoke", help="Revoke an API key")
    revoke_parser.add_argument("hash", help="Key hash to revoke")

    # Verify command
    verify_parser = subparsers.add_parser("verify", help="Verify an API key")
    verify_parser.add_argument("key", help="API key to verify")

    args = parser.parse_args()

    if args.command == "generate" or not args.command:
        # Default: generate with name from args or default
        name = getattr(args, "name", "external_service")
        expires = getattr(args, "expires", 365)

        api_key = generate_api_key(name=name, expires_in_days=expires)

        print(f"\n{'🔑 New L104 API Key Generated':^80}")
        print("=" * 80)
        print(f"Name:       {name}")
        print(f"Expires:    {expires} days")
        print(f"\n{'Your API Key (SAVE THIS NOW - you won\'t see it again!)':^80}")
        print("-" * 80)
        print(f"{api_key}")
        print("-" * 80)
        print(f"\nUsage:")
        print(f"  export L104_API_KEY='{api_key}'  # For L104 client")
        print(f"  Header: Authorization: Bearer {api_key}")
        print(f"\n⚠️  Keep this key secret! Do not commit to git or share publicly.\n")

    elif args.command == "list":
        list_api_keys()

    elif args.command == "revoke":
        revoke_api_key(args.hash)

    elif args.command == "verify":
        if verify_api_key(args.key):
            print(f"✅ API key is valid and active")
        else:
            print(f"❌ API key is invalid, revoked, or expired")


if __name__ == "__main__":
    main()
