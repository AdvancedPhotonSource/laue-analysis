# Configuration

Configuration objects are immutable. Use {func}`dataclasses.replace` to derive
new values and construct or replace an `Indexer` to validate them. See
{doc}`processing` for the processing API.

## Peak Search

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autoclass:: PeakParams
   :members:
```

## Orientation Indexing

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autoclass:: IndexParams
   :members:
```

## Frame Metadata

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autoclass:: FrameMetadata
   :members: as_dict
```
