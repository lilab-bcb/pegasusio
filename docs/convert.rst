Convert to/from ``AnnData``
=============================

PegasusIO stores data as *MultimodalData* objects, and each *MultimodalData* has a default *UnimodalData* object to refer to. Each time, a format conversion is to convert the default *UnimodalData* to another format.

*AnnData* is the annotated data matrix object provided by anndata_ package.

Let ``mmdata`` be a PegasusIO *MultimodalData* object. Use the following code to convert its default *UnimodalData* to *AnnData*::

    >>> adata = mmdata.to_anndata()

And ``adata`` is the wanted *AnnData* object.

Now let ``adata`` be an *AnnData* object (See `anndata reading documentation`_ for how to load file to *AnnData*). Use the following code to convert it to PegasusIO *MultimodalData*::

    >>> import pegasusio as io
    >>> mmdata = io.MultimodalData(adata)

And ``mmdata`` is the wanted *MultimodalData* object.

.. _anndata: https://anndata.readthedocs.io/en/stable/anndata.AnnData.html
.. _anndata reading documentation: https://anndata.readthedocs.io/en/stable/api.html#reading
