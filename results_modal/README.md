# Modal results (A100), pulled 2026-09-25

Three runs completed on Modal before the workspace hit its spend limit
("Workspace has exceeded its spend limit"), which is why the apps stopped —
not a code failure. All three are `state: done, exit_code: 0`.

| run | forgetting | acquisition | replay | coords/1e6 | wall min |
|---|---:|---:|---:|---:|---:|
| `yoked_full_lo-seed1337` | 0.1633 | 1.2524 | 1548 | 629.5 | 122.4 |
| `yoked_full_lo-seed2024` | 0.1295 | 1.2419 | 1614 | 623.2 | 139.2 |
| `trihope17-seed1337` (Qwen3-1.7B) | 0.1266 | **1.0395** | 618 | 5710.2 | 140.3 |

Forgetting recomputed from `retention.history` (a list of per-phase records):
per domain, the worst rise above the best loss seen so far, then averaged over
the four domains.

**`trihope17` cannot be read as a scale result yet.** Its control,
`random_commit17`, never ran — Modal ran out of credit first — so there is no
DP4 comparison at 1.7B. The one thing it does show is much stronger acquisition
(1.0395 vs 1.2490 for the 0.6B student), which is expected from a 2.8x larger
model and says nothing about the controller.

Only `run_summary.json`, `status.json` and `metrics.jsonl` were pulled; the
Volume's `events.jsonl` and checkpoints were left behind.
