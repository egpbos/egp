"""
Je kunt dit doen door gebruik te maken van het slice object!

Van http://docs.scipy.org/doc/numpy/reference/arrays.indexing.html#arrays-indexing:

Remember that a slicing tuple can always be constructed as obj and used in the
x[obj] notation. Slice objects can be used in the construction in place of the
[start:stop:step] notation. For example, x[1:10:5,::-1] can also be
implemented as obj = (slice(1,10,5), slice(None,None,-1)); x[obj] . This can
be useful for constructing generic code that works on arrays of arbitrary
dimension.
"""