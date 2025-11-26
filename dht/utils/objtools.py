import inspect

def get_signature_kwargs(cls):
    '''Extracts the default keyword arguments from the `__init__` method of a class.

    Parameters
    ----------
    cls : class
        The class from which to extract the default keyword arguments.

    Returns
    -------
    kwargs : dict
        A dictionary where the keys are the names of the parameters with default 
        values in the `__init__` method of the class, and the values are the corresponding 
        default values.
    '''
    signature = inspect.signature(cls.__init__)
    kwargs = {
        name: param.default for name, param in signature.parameters.items() 
        if param.default is not inspect.Parameter.empty
    }
    return kwargs

def update_signature_kwargs(cls, **kwargs):
    '''Updates default keyword arguments with provided keyword arguments.

    Parameters
    ----------
    cls : class
        The class whose `__init__` method's default keyword arguments are to be updated.
    **kwargs : dict, optional
        The keyword arguments with which to update the default keyword arguments. 
        Only the keys present in the class's `__init__` method will be used.

    Returns
    -------
    signature_kwargs : dict
        A dictionary containing the updated default keyword arguments for the class's 
        `__init__` method.
    '''
    signature_kwargs = get_signature_kwargs(cls)
    updated_kwargs = {key: kwargs[key] for key in signature_kwargs if key in kwargs}
    signature_kwargs.update(updated_kwargs)
    return signature_kwargs