# Self-Healing System Documentation

## Overview
The L104 Node now includes comprehensive self-healing capabilities that automatically detect and recover from various failure scenarios without manual intervention.

## Self-Healing Features

### 1. **Circuit Breaker Pattern**
- Prevents cascading failures by opening circuit after threshold errors
- Automatically attempts recovery after timeout period
- Configurable via environment variables:
  - `CIRCUIT_BREAKER_THRESHOLD` (default: 5)
  - `CIRCUIT_BREAKER_TIMEOUT` (default: 60 seconds)

### 2. **Automatic Connection Recovery**
- HTTP client automatically resets after persistent errors
- Connection pool management with health tracking
- Error counter decrements on successful requests
- Triggers auto-heal when error threshold exceeded

### 3. **Retry Logic with Exponential Backoff**
- Failed operations automatically retry with increasing delays
- Configurable retry attempts: `MAX_RETRY_ATTEMPTS` (default: 3)
- Backoff multiplier: `RETRY_BACKOFF_MULTIPLIER` (default: 2.0)

### 4. **Health Watchdog**
- Periodic health checks with progressive healing intensity
- Escalates healing actions on repeated failures
- Exits for external supervisor restart after multiple heal attempts
- Enable with: `ENABLE_WATCHDOG_ENV=true`

### 5. **Automatic Error Recovery**
- Middleware detects persistent errors and triggers healing
- Clears rate limits, resets connections, reinitializes state
- Enable with: `AUTO_HEAL_ENABLED=true` (default: enabled)

### 6. **Model Rotation with Error Tracking**
- Automatically rotates between Gemini models on quota/errors
- Tracks 429 errors per model
- Clears error trackers during healing

## Configuration

### Environment Variables
```bash
# Enable/disable auto-healing
AUTO_HEAL_ENABLED=true

# Circuit breaker settings
CIRCUIT_BREAKER_THRESHOLD=5
CIRCUIT_BREAKER_TIMEOUT=60

# Retry configuration
MAX_RETRY_ATTEMPTS=3
RETRY_BACKOFF_MULTIPLIER=2.0

# Watchdog settings
ENABLE_WATCHDOG_ENV=true
WATCHDOG_INTERVAL=30
WATCHDOG_FAILURE_THRESHOLD=3
WATCHDOG_EXIT_ON_FAILURE=true
```

## Monitoring

### Metrics Endpoint
The `/metrics` endpoint now includes self-healing status:

```json
{
  "self_healing": {
    "auto_heal_enabled": true,
    "circuit_breaker_open": false,
    "circuit_breaker_failures": 0,
    "http_client_errors": 0,
    "last_http_reset": "2026-01-03T10:00:00Z"
  },
  "auto_heals": 0,
  "circuit_breaks": 0,
  "connection_resets": 0,
  "retry_successes": 0
}
```

### Manual Healing
Trigger manual healing via API:

```bash
# Full heal (rate limits + circuit breaker)
curl -X POST http://localhost:8081/self/heal

# Custom healing options
curl -X POST "http://localhost:8081/self/heal?reset_rate_limits=true&reset_http_client=true&reset_circuit_breaker=true"
```

## How It Works

### 1. Error Detection
- Request middleware tracks errors per endpoint
- HTTP client monitors connection health
- Circuit breaker watches upstream API failures

### 2. Progressive Recovery
1. **Light healing**: Clear rate limits, reset circuit breaker
2. **Medium healing**: + Reset HTTP client connections
3. **Heavy healing**: + Exit for supervisor restart

### 3. Success Recovery
- Successful requests decrement error counters
- Circuit breaker closes on successful API calls
- System gradually returns to normal state

## Logging

Self-healing actions are logged with specific tags:
- `auto_heal_triggered`: Automatic healing initiated
- `circuit_breaker_open`: Circuit opened due to failures
- `circuit_breaker_half_open`: Attempting recovery
- `watchdog_heal`: Watchdog triggered healing
- `http_client_auto_reset`: Connection pool reset
- `retry_success`: Operation succeeded after retry

## Best Practices

1. **Monitor metrics regularly**: Check `/metrics` for healing activity
2. **Tune thresholds**: Adjust based on your traffic patterns
3. **Use with supervisor**: Pair with process managers (systemd, supervisor) for full recovery
4. **Enable watchdog in production**: Ensures automatic recovery from deep failures
5. **Review logs**: Frequent healing may indicate upstream issues

## Troubleshooting

**Circuit breaker frequently open?**
- Increase `CIRCUIT_BREAKER_THRESHOLD`
- Check upstream API health
- Review model quota limits

**Too many connection resets?**
- Increase error threshold in auto-heal logic
- Check network stability
- Review timeout settings

**Watchdog causing restarts?**
- Increase `WATCHDOG_FAILURE_THRESHOLD`
- Extend `WATCHDOG_INTERVAL`
- Disable exit: `WATCHDOG_EXIT_ON_FAILURE=false`
