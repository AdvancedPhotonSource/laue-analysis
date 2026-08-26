# Crystal indexing

Crystal indexing compares measured scattering-vector directions with reflections generated from a crystal description. It runs only when an `Indexer` has a crystal and the frame contains at least two detected peaks.

## Inputs

The stage receives:

- Unit `qhat` vectors from pixel-to-q conversion
- Unit-cell dimensions and angles
- An International Tables space-group number
- Atom sites and occupancies
- Energy limits, angular tolerance, search cone, preferred Miller-index direction, and input-count limit

Cell lengths are converted to angstroms before the native crystal model is constructed.

`IndexParams.max_data` limits the input to the first `max_data` detected peaks. At least two vectors must be available.

## Orientation search

The native orientation search uses `kev_max_calc` to calculate candidate reflections and `kev_max_test` when testing them. `angle_tolerance_deg` controls angular matching, `cone_deg` limits the search cone, and `hkl_prefer` supplies a preferred Miller-index direction.

The search can return no candidate pattern. That outcome produces a valid `FrameResult` with `indexed=False`.

For each candidate, the implementation refines Euler angles against matched measured vectors. A numerical failure during refinement raises `IndexingError`. An allocation failure raises `MemoryError`.

## Pattern output

Each returned `Pattern` contains:

- Euler angles and a rotation matrix
- A rotated reciprocal-lattice matrix
- A native goodness score
- Root-mean-square angular error
- Miller indices for assigned reflections
- Zero-based indices into `FrameResult.peaks`
- Per-assignment angular error, photon energy, and predicted intensity

The total `FrameResult.n_indexed` is the sum of assignment counts across patterns. Use `FrameResult.indexed_peak_indices` for the sorted unique peak indices assigned to any pattern.

## Energy calculation

For each assigned reciprocal-lattice vector `G`, the implementation calculates photon energy from its magnitude and the z component of the matched unit scattering direction:

```{math}
E = \frac{hc\lVert G\rVert}{4\pi\sin\theta}
```

with:

```{math}
\sin\theta = -\hat{q}_z
```

The result is stored in `Pattern.energy_kev`.

## Conventions requiring review

The source code and regression tests establish output shapes, units, and compatibility with current 34-ID-E reference results. They do not provide enough user-facing evidence to define the Euler-angle sequence, matrix multiplication convention, reciprocal-basis layout, or goodness-score interpretation safely.

Until those conventions receive domain review:

- Treat `euler_deg` as the native 34-ID-E representation.
- Preserve `rotation` as returned when exchanging orientation data.
- Do not compare `goodness` across different parameter sets without validation.
- Use `rms_error_deg` and per-assignment `err_deg` as reported angular errors, without assigning an undocumented acceptance threshold.