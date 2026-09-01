# Processing

The preferred API either indexes one frame with `index_frame` or reuses an
`Indexer` across frames that share configuration.

## One-Off Indexing

```{eval-rst}
.. currentmodule:: laueanalysis.indexing

.. autofunction:: index_frame
```

## Reusable Indexer

```{eval-rst}
.. currentmodule:: laueanalysis.indexing

.. autoclass:: Indexer
   :members: index, index_many, replace, write_many_xml
```
