class FileExtensionError(FileNotFoundError):
    """Raise an exception for extension-related issues. """
    pass

class MRTrixError(IOError):
    """Raise an exception for MRPreproc-related issues. """
    pass