# Caching Notes

Cache stampede occurs when many requests hit the backend at the same time after a cache entry expires.

Mitigations:
- request coalescing (single-flight / per-key lock)
- TTL jitter
- early refresh / refresh-ahead
- serving stale while revalidating
