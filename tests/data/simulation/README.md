# Simulation baseline fixtures

These compressed NumPy fixtures capture raw portal/JZT output before the
backend was moved into `laueanalysis`. They deliberately preserve the portal
adapter's strict energy bounds, omitted occupancy, legacy harmonic hash, and
legacy ordering so later normalization differences can be reviewed.

The three cases are:

| Fixture | Material | Space group | Raw spots |
|---|---:|---:|---:|
| `portal_jzt_ni.npz` | Ni | 225 | 3 |
| `portal_jzt_cdte.npz` | CdTe | 225 | 32 |
| `portal_jzt_si.npz` | synthetic Si | 227 | 17 |

Each fixture contains `hkl`, `q`, `detector_xy`, `energy_kev`,
`relative_intensity`, and a JSON metadata scalar. The metadata records the
crystal, reciprocal matrix, detector, depth, energy interval, source commits,
and generation environment. Ni and CdTe use `tests/config/Ni.xml` and
`tests/config/CdTe.xml`; Si reproduces the material definition in the portal's
existing simulation test.

Generate the files from repository root with the reviewed sibling checkout:

```bash
python tests/data/simulation/generate_portal_goldens.py ../laue-portal
```

The generator refuses a portal revision other than `477e4be`. Repeated runs in
the capture environment produced byte-identical array payloads.

## Phase 2 normalized fixtures

`normalized_ni.npz`, `normalized_cdte.npz`, and `normalized_si.npz` pin the
maintained `SimulationResult` contract after occupancy propagation, inclusive
energy filtering, maintained detector projection, strongest-harmonic grouping,
and deterministic ordering. They use the same recorded inputs as the raw
fixtures and include normalization metadata and numerical tolerances.

After deliberately reviewing a scientific change, regenerate them with:

```bash
python tests/data/simulation/generate_normalized_goldens.py
```

The reviewed Phase 2 differences are intentionally small for these inputs.
Ni remains at three directions and Si at seventeen. CdTe changes from 32 raw
rows to 31 direction-distinct rows: legacy hashing retained both `(1, 1, 1)`
and `(3, 3, 3)`, while normalization keeps the stronger `(1, 1, 1)`
representative. All fixture occupancies are one and no reflection lies exactly
on an energy boundary, so occupancy propagation and inclusive bounds do not
change these three particular outputs; focused tests cover those semantics.
