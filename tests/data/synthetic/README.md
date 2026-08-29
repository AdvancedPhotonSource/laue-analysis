# Synthetic Laue frames and LaueGo reference outputs

These fixtures replace measured beamline frames in the test suite. Nothing
here is measured data.

| Path | Contents |
| --- | --- |
| `generate.py` | Builds everything below. Deterministic (fixed seed and orientations). |
| `frames/*.h5` | Four `numpy.uint16` frames for detector `PE1621 723-3335` of `tests/data/geo/geoN_2022-03-29_14-15-05.xml`: one Ni grain (`grain_a`, `grain_b`), two Ni grains (`two_grains`), and a 256 x 256 sub-region with no spots (`empty`). Reflections come from `laueanalysis.analysis.simulate_reflections`, rendered as Gaussian spots on a constant background (no noise, so each frame compresses to a few tens of KB). |
| `baseline/peaks/`, `baseline/p2q/`, `baseline/index/` | Output of the LaueGo programs `peaksearch`, `pix2qs`, and `euler` run on those frames through `laueanalysis.indexing.lauego`. |
| `provenance.json` | Commit, package and `liblaue` versions, seed, settings, and per-frame SHA-256 and counts. |

The tests compare the in-process indexer (`liblaue.so`) against the LaueGo
outputs, so the two implementations check each other. The peak-search and
indexing settings used for the baseline are recorded in `provenance.json`
and repeated in the tests.

Regenerate only after a deliberate fixture change:

```console
$ python tests/data/synthetic/generate.py
```

Then update any counts or angles asserted in `tests/test_indexer.py`.
