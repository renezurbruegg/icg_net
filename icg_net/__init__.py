def __getattr__(name):
    if name == "ICGNetModule":
        from .core import ICGNetModule
        return ICGNetModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")