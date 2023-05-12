import functools
from typing import Type, Callable


class GeneratorPlugin:
    _plugins = {}
    _sub_managers = {}
    _classpath = 'plugins'

    def __new__(cls, func: Callable) -> Callable:
        # Register argument as a plugin
        path = cls._classpath
        if path not in cls._plugins:
            cls._plugins[path] = {}
        cls._plugins[path][func.__name__] = func
        return func

    @classmethod
    def bind(cls, generator):
        # Actual constructor
        self = object.__new__(cls)
        self.__init__(generator)
        return self

    def __init__(self, generator):
        self.generator = generator
        if self._classpath in GeneratorPlugin._sub_managers:
            for name, Manager in GeneratorPlugin._sub_managers[self._classpath].items():
                setattr(self, name, Manager.bind(self.generator))

    def __init_subclass__(cls, **kwargs):
        if 'name' in kwargs:
            name = kwargs['name']
            for base in cls.__bases__:
                if issubclass(base, GeneratorPlugin):
                    cls._classpath = f'{base._classpath}.{name}'
                    if base._classpath not in GeneratorPlugin._sub_managers:
                        GeneratorPlugin._sub_managers[base._classpath] = {}
                    GeneratorPlugin._sub_managers[base._classpath][name] = cls
                    setattr(base, name, cls)

    def __getattr__(self, item: str):
        if self._classpath in GeneratorPlugin._plugins and item in GeneratorPlugin._plugins[self._classpath]:
            return functools.partial(GeneratorPlugin._plugins[self._classpath][item], self.generator)
        raise AttributeError

    @classmethod
    def category(cls, name) -> Type["GeneratorPlugin"]:
        if hasattr(cls, name):
            return getattr(cls, name)
        return type(name+'Plugins', (cls,), {}, name=name)      # noqa https://youtrack.jetbrains.com/issue/PY-46044
