# Private JZT snapshot

This directory contains a private snapshot of the legacy JZT Laue simulator.
It is an implementation detail of `laueanalysis`; its modules and classes are
not a supported API.

## Provenance

- Source repository: `https://github.com/AdvancedPhotonSource/laue-portal`
- Reviewed source commit: `477e4beede63598cf8490f42574d350605f4f0da`
- Portal commit that introduced the snapshot and adapter:
  `083cb6f214edf8a4507612b445215b448ccbdf5e`
- Source path: `laue_portal/analysis/JZTLaueSim`
- Snapshot date in this package: 2026-08-28

The source files identify themselves as part of `pydiffract` by Jon Z.
Tischler, Argonne National Laboratory. The portal copy carries no separate
vendor license or more precise upstream version/revision. The repository-level
UChicago Argonne license in this repository covers this checked-in snapshot.
Original file headers have been retained; the containment edits below are
package modifications and do not imply new authorship of the legacy numerical
code.

## Deliberate differences from the portal snapshot

1. `LauePattern_allspots.py` was omitted. It was byte-for-byte identical to
   `LauePattern.py` at the reviewed commit and the only portal import of it was
   an eager, unused import in the adapter.
2. The portal's eager wildcard-importing `__init__.py` was replaced with an
   initializer that imports nothing.
3. Internal `JZTLaueSim.*` imports were converted to package-relative imports.
   This removes the portal adapter's `sys.path` mutation requirement.
4. `elementData.xml` is resolved with `importlib.resources` from this package,
   rather than from checkout-relative state.
5. The zero-scattering direction is rejected before the legacy energy
   division, avoiding a divide-by-zero warning without changing accepted
   reflections.
6. Three supported-Python compatibility warnings were corrected: two string
   identity comparisons and one invalid docstring escape.
7. Scalar extraction from NumPy matrices uses `item()` on the exercised
   symmetry paths, avoiding NumPy 1.25 array-to-scalar deprecations.
8. `LauePattern.calc()` records its complete pre-deduplication candidate list,
   accepted-candidate count, and limit state on private attributes. Its legacy
   return value is unchanged; the maintained wrapper needs these attributes to
   replace hash-based harmonic selection and reject truncated simulations.

No output/CIF modules were removed because `Lattice.py` imports them on the
exercised simulation dependency path. The snapshot otherwise intentionally
retains its legacy organization and style to keep parity review tractable.
