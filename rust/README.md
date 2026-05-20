# rust/

Experimental Rust core for a minimal agent harness. Distils ideas from
Sandcastle (sandbox provider × branch strategy, two-phase prompt, completion
signal, structured output) and Flue (instance/harness/session hierarchy,
virtual sandbox default, role precedence, comptime-shaped result schemas)
into a build with a small dependency footprint.

## Status

M0 only. The actor primitive is live. Nothing else yet.

## Layout

- `actor/` — OS-thread actor primitive. `std` only, zero dependencies.

## Non-goals (for now)

- No async/await runtime.
- No supervision tree.
- No location-transparent addressing.
- No work-stealing scheduler.

Each of those lands when a real workload forces it, not before.

## Test

```
cd actor && cargo test
```
