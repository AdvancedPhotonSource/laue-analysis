# Reflection simulation performance

The reflection-simulation benchmark is observational. It records call time and peak resident memory without setting a test threshold. Host load, filesystem cache state, Python version, and NumPy version can change the result.

Run the benchmark from the repository root:

```console
$ python tests/perf_testing/run_simulation_perf.py --case ni --warm-runs 5
```

The script starts a separate process for the cold measurement. `cold_total_seconds` includes package imports, fixture setup, and the first simulation. The warm measurement reuses imported modules and prepared package inputs.

## Baseline from 2026-08-28

This baseline used source revision `a8ecb503bac8ac6150ce2a1ea29a1a223527546a` with the uncommitted simulation-port changes in the working tree.

| Environment field | Value |
|---|---|
| CPU | Intel Xeon E5-2680 v2 at 2.80 GHz |
| Logical CPUs | 40 |
| Operating system | Linux 5.14.0-687.33.1.el9_8.x86_64 |
| Python | 3.13.7 |
| NumPy | 2.5.2 |

The benchmark used the tracked Ni case:

| Input field | Value |
|---|---|
| Crystal | Ni, space group 225 |
| Detector | `PHASE1-SYNTHETIC`, `(2048, 2048)` unbinned pixels |
| Detector size | `(409600, 409600)` µm |
| Detector translation | `(0, 0, 300000)` µm |
| Detector rotation vector | `(0, 0, 0)` rad |
| Depth | 0 µm |
| Energy interval | 6 to 30 keV, inclusive |
| Candidate limit | 100,000 |
| Returned reflections | 3 |

The row-basis reciprocal matrix in `1/nm` was:

```text
[[ 11.4367037948,  12.6082153872,   5.3074383238],
 [ 11.4367037948, -12.6082153872,   5.3074383238],
 [  7.5058512590,   0.0000000000, -16.1739416155]]
```

| Measurement | Result |
|---|---:|
| Cold import, setup, and first simulation | 0.6613 s |
| First simulation within the cold process | 0.4169 s |
| Cold-process peak resident memory | 59.70 MiB |
| Warm median, 5 calls | 0.3397 s |
| Warm minimum | 0.3372 s |
| Warm maximum | 0.4221 s |
| Warm-process peak resident memory | 61.69 MiB |

All five warm calls returned three reflections. Use the JSON output for detailed comparison because the table rounds timings to four decimal places.
