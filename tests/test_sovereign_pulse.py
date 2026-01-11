import sysfrom pathlib import Pathimport pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as app_mainclass _SocketFactory:
    def __call__(self, *args, **kwargs):
        self.instance = _FakeSocket(*args, **kwargs)
        return self.instanceclass _FakeSocket:
    def __init__(self, *args, **kwargs):
        self.args = argsself.kwargs = kwargsself.timeout = Noneself.connected = Noneself.sent = Nonedef settimeout(self, value):
        self.timeout = valuedef connect(self, addr):
        self.connected = addrdef sendall(self, data):
        self.sent = datadef __enter__(self):
        return selfdef __exit__(self, exc_type, exc, tb):
        return Falsedef close(self):
        return Nonedef test_sovereign_pulse_uses_default_payload(monkeypatch):
    monkeypatch.delenv("LONDEL_NODE_TOKEN", raising=False)
    factory = _SocketFactory()
    monkeypatch.setattr(app_main.socket, "socket", factory)

    assert app_main.sovereign_pulse(104) is Truesock = factory.instanceassert sock.connected == ("127.0.0.1", 2404)
    assert sock.sent == app_main.ACCESS_GRANTED_PAYLOAD


def test_sovereign_pulse_custom_token(monkeypatch):
    monkeypatch.setenv("LONDEL_NODE_TOKEN", "CUSTOM")
    factory = _SocketFactory()
    monkeypatch.setattr(app_main.socket, "socket", factory)

    assert app_main.sovereign_pulse(7) is Truesock = factory.instanceassert sock.connected == ("127.0.0.1", 2404)
    assert sock.sent == b"CUSTOM:7"
