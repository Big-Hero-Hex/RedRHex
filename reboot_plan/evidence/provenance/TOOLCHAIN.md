# P0 Simulator Toolchain Provenance

Status: **UNRESOLVED**

Static audit found the following candidate runtime facts; they are not yet accepted as a
clean reproducible toolchain:

| Fact | Observed value | Verification status |
|---|---|---|
| Behavioral source | `fix/review-2026-07@5cdc824` | recorded |
| Isaac Lab base | `v2.3.2` / `37ddf626871758333d6ed89cf64ad702aef127d0` | externally dirty |
| Isaac Sim build | `5.1.0-rc.19+release.26219.9c81211b.gl` | observed, checker pending |
| Python | `3.11.15` in intended Isaac environment | observed, checker pending |
| Flatdict | `4.1.0`; differs from external checkout requirement edit | policy unresolved |
| RSL-RL | `3.1.2` | observed, checker pending |

P0 must choose a clean dedicated Isaac checkout or explicitly pin/verify the required
external patch. Per-run device, seed/RNG, command, resolved config, and asset hashes do
not belong here; they belong in each P1/P2 run manifest referencing this toolchain ID.
