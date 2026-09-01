# Reflection simulation

The public reflection-simulation API has two symbols. It accepts package crystal and detector models and returns backend-neutral NumPy arrays.

See [Simulate detector reflections](../guides/simulation.md) for input selection, numerical behavior, and examples.

```{eval-rst}
.. currentmodule:: lauelab.analysis

.. autoclass:: SimulationResult
   :members: missing_from

.. autofunction:: simulate_reflections
```

`SimulationResult` contains no backend objects or status flags. The implementation does not expose a backend choice or a fallback.
