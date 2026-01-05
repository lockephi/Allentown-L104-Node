import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as app_main


class _SocketFactory:
    def __call__(self, *args, **kwargs):
        self.instance = _FakeSocket(*args, **kwargs)
        return self.instance


class _FakeSocket:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.timeout = None
        self.connected = None
        self.sent = None

    def settimeout(self, value):
        self.timeout = value

    def connect(self, addr):
        self.connected = addr

    def sendall(self, data):
        self.sent = data

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def close(self):
        return None


def test_sovereign_pulse_uses_default_payload(monkeypatch):
    monkeypatch.delenv("LONDEL_NODE_TOKEN", raising=False)
    factory = _SocketFactory()
    monkeypatch.setattr(app_main.socket, "socket", factory)

    assert app_main.sovereign_pulse(104) is True
    sock = factory.instance
    assert sock.connected == ("127.0.0.1", 2404)
    assert sock.sent == app_main.ACCESS_GRANTED_PAYLOAD


def test_sovereign_pulse_custom_token(monkeypatch):
    monkeypatch.setenv("LONDEL_NODE_TOKEN", "CUSTOM")
    factory = _SocketFactory()
    monkeypatch.setattr(app_main.socket, "socket", factory)

    assert app_main.sovereign_pulse(7) is True
    sock = factory.instance
    assert sock.connected == ("127.0.0.1", 2404)
    assert sock.sent == b"CUSTOM:7"
