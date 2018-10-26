import numpy as _np

def wrap_namespace(old, new):
    function_types = {_np.ufunc} # Create set of function types
    for name, obj in old.items():
        if type(obj) in function_types:
           new[name] = obj

wrap_namespace(_np.__dict__, globals())
