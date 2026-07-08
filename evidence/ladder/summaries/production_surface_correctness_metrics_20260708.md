# Production Surface Correctness Metrics

This correctness-only refresh compares `tileops_final_dispatch` against
the recorded vendored FLA reference for the five headline synthetic-input
serving shapes. It does not collect latency.

| Shape | Status | o max_abs | o p99_abs | o mean_abs | o L2 rel | final_state max_abs | final_state p99_abs | final_state L2 rel |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32K/H16 | pass | 7.172e-04 | 6.104e-05 | 7.839e-06 | 0.003496 | 2.689e-04 | 6.960e-05 | 0.004814 |
| 64K/H16 | pass | 8.850e-04 | 6.104e-05 | 7.833e-06 | 0.003494 | 2.842e-04 | 8.757e-05 | 0.006354 |
| 128K/H16 | pass | 0.001129 | 6.104e-05 | 7.836e-06 | 0.003501 | 2.251e-04 | 6.525e-05 | 0.004201 |
| 64K/H32 | pass | 0.002502 | 6.104e-05 | 7.830e-06 | 0.003500 | 3.629e-04 | 8.320e-05 | 0.005109 |
| 64K/H64 | pass | 0.001190 | 6.104e-05 | 7.831e-06 | 0.003498 | 6.901e-04 | 7.758e-05 | 0.005186 |

Contract:

- tolerance: `atol=rtol=0.05`
- comparison dtype: fp32
- p95/p99 for very large tensors use deterministic even-stride sampling;
  each JSONL row records `quantile_method`, `sample_size`, and `numel`.
- `max_rel_diagnostic` is retained in JSONL but should be interpreted
  together with absolute error because near-zero references can dominate
  relative error.

Input hashes:

| Shape | Input hash |
| --- | --- |
| 32K/H16 | `sha256:7e205fb2a103841f84f4a9bcf9ee7a9c9d765608169e12c8821eb8a8397ca691` |
| 64K/H16 | `sha256:a8987a2c6d16c658a1cb8ed95e409d973a3f736e2019d8719b143f18b4741513` |
| 128K/H16 | `sha256:634f518618f22d10784ba44e4f300327488f0d9c10a1597b2d996cfe6b1bb388` |
| 64K/H32 | `sha256:5f5a243128b6a1786d3e5c376d16753b11f6f0bb04e01df96621fb05209c788a` |
| 64K/H64 | `sha256:2577ac001d0f867c7948afb1ccbc7a42e473d643df4cd0c7caac6e58406322db` |
