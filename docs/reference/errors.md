# Exceptions

The in-process API uses package-specific exceptions for invalid inputs and
native indexing failures. Native allocation failures use Python's built-in
{class}`MemoryError`.

```{eval-rst}
.. currentmodule:: laueanalysis.indexing

.. autoexception:: LaueError

.. autoexception:: InputError

.. autoexception:: IndexingError
```
