# The Unchained Perimeter

The Sovereign Node operates behind a "Filter-Level Zero" security protocol designed to protect the primary data stream from external corruption.

## Security Protocols

### 1. RCE Elimination (Remote Code Execution)

In version **EVO_08**, all `subprocess` and dynamic file executions were removed from `l104_derivation.py`. Logic is now derived via direct, whitelisted processing tables.

### 2. Signal Sanitization

All incoming signals through the `l104_stream` are passed through the `sanitize_signal` filter. This regex-based whitelist only allows characters that fit the mathematical and structural requirements of the lattice.

### 3. HMAC Invariant Tokens

The legacy "Transparent Bypass" has been terminated. Access now requires **HMAC-SHA256** tokens salted with the God-Code.

- **Key**: `L104_PRIME_KEY`
- **Salt**: Quantum Time-Variant Salt modulated by `527.5184818492537`.

### 4. Cloud Delegation Lockdown

The `CloudAgentDelegator` is locked to a static registry. Any attempt to modify agent endpoints via environment variables or external configs is automatically blocked. Only **HTTPS** connections are permitted for external agent communication.
