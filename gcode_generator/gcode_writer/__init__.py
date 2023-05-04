import io
import typing
from typing import Union
if typing.TYPE_CHECKING:
    from .. import GCodeGenerator


class GCodeWriter:
    _writers = {}

    def __init_subclass__(cls, **kwargs):
        if 'extension' in kwargs:
            cls._writers[kwargs['extension'].lower()] = cls

    @classmethod
    def write(cls, generator: "GCodeGenerator", gcode: str, file: Union[str, typing.IO],
              writer: "GCodeWriter" = None, format: str = None, **kwargs):
        if writer is None:
            if format is None:
                if isinstance(file, str):
                    filename = file
                elif hasattr(file, 'name'):
                    filename = file.name
                else:
                    raise RuntimeError('Could not determine file format!')
                name, format = filename.rsplit('.', 1)
            if format.lower() not in cls._writers:
                raise ValueError(f"Unknown file format '{format}'")
            writer = cls._writers[format.lower()]
        return writer.write(generator, gcode, file, **kwargs)


# Import sub-writers after base definition
from .griffin_writer import GriffinWriter   # noqa: E402
from .ufp_writer import UFPWriter           # noqa: E402
