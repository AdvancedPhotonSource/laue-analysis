# CPU reconstruction reference

This directory holds the numerical acceptance reference for `reconstructN_cpu`
used by the build-system migration (`BUILD_DEPLOYMENT_PLAN.md`, Phase 0 and
Phase 3) and by `tests/test_reconstruct.py::TestReconstructCPUReference`.

| File | Purpose |
| --- | --- |
| `generate_reference.py` | Builds the deterministic synthetic wire scan, runs the program, and writes the two files below. |
| `cpu_reference.npz` | `images[51, 128, 128]` (float64) and `depth_um[51]` produced by the reference build. |
| `cpu_reference.json` | Provenance: commit, toolchain stamp, linked library names, input and output SHA-256, and the comparison tolerance. No machine paths. |

## What the input is

The input is not stored; the test regenerates it from a fixed seed and checks
its SHA-256 against the provenance. It is a 128 x 128 (16x binned) detector-0
scan of 81 images plus the intensity-map slice, with three Gaussian spots that
switch off once the wire's leading edge passes their assigned depths (0, -10,
and +12 micron). The on/off geometry replicates `pixel_to_point_xyz` and
`pixel_xyz_to_depth` from `WireScan.c`, so the reconstruction places each spot
within one or two depth bins of its assigned depth. It is a deterministic
exercise of the program's input, geometry, depth-binning, and output paths,
not a physical simulation.

## Comparison contract

Output pixels are written as `double` (`-t 5`) so the reference carries no
integer rounding. The recorded tolerance is `rtol=1e-12, atol=1e-9`; on the
same toolchain the output is bit-identical, and it is also identical between
`-N 1` and `-N 4`.

## Regenerating

Do not regenerate to make a build change pass. Regeneration is a scientific
decision that must be reviewed and recorded. When it is warranted:

```console
python tests/data/reconstruction/generate_reference.py
```

Run it from an environment where `lauelab` and its CPU executable are
installed, then commit the updated `.npz` and `.json` together with a note in
the commit message explaining why the reference changed.
