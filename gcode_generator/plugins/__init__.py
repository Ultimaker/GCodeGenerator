import functools


def GeneratorPlugin(func):
    GeneratorPluginManager.register_plugin(func.__name__, func)
    return func


class GeneratorPluginManager:
    _plugins = {}

    def __init__(self, generator):
        self.generator = generator

    def __getattribute__(self, item):
        if item in GeneratorPluginManager._plugins:
            return functools.partial(GeneratorPluginManager._plugins[item], self.generator)
        return super().__getattribute__(item)

    @classmethod
    def register_plugin(cls, name, plugin):
        cls._plugins[name] = plugin
