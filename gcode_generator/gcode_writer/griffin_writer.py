import io
import typing
from typing import Union

from . import GCodeWriter
if typing.TYPE_CHECKING:
    from .. import GCodeGenerator


class GriffinWriter(GCodeWriter, extension='gcode'):
    @classmethod
    def write(cls, generator: "GCodeGenerator", gcode: str, file: Union[str, typing.TextIO], **kwargs):
        if isinstance(file, str):
            # If 'file' is a filename [str], open file and re-run method
            with open(file, 'wt', newline='\n') as f:
                return cls.write(generator, gcode, f, **kwargs)
        else:
            file.write(cls.generate_header(generator, **kwargs))
            file.write('\n\n')
            file.write(gcode)

    @classmethod
    def generate_header(cls, generator, time_estimate=None, **kwargs):
        if time_estimate is None:
            time_estimate = generator.print_time
        bed_temps = [tool.material('heated bed temperature') for tool in generator.tools]
        header = {
            'HEADER_VERSION': 0.1,
            'FLAVOR': 'Griffin',
            'GENERATOR': {
                'NAME': 'GCodeGenerator',
                'VERSION': '2.0.0',
                'BUILD_DATE': '2023-04-05'
            },
            'TARGET_MACHINE': {'NAME': generator.machine_name},
            'PRINT': {
                'TIME': int(time_estimate),
                'SIZE': {
                    'MIN': generator.boundingbox[0].to_dict(),
                    'MAX': generator.boundingbox[1].to_dict(),
                }
            },
            'BUILD_PLATE': {
                'INITIAL_TEMPERATURE': min(temp for temp in bed_temps if temp is not None)
            },
            'EXTRUDER_TRAIN': {}
        }
        for idx, tool in enumerate(generator.tools):
            settings = {
                'MATERIAL': {
                    'VOLUME_USED': round((tool.material.usage + tool.position) * tool.material.area, 2),
                    'GUID': tool.material.guid
                },
                'NOZZLE': {
                    'DIAMETER': tool.nozzle.diameter,
                    'NAME': tool.nozzle.name
                }
            }
            if idx == generator.initial_tool:
                settings['INITIAL_TEMPERATURE'] = float(tool.material('print temperature', 200))
            else:
                settings['INITIAL_TEMPERATURE'] = float(tool.material('standby temperature', 100))
            header['EXTRUDER_TRAIN'][str(idx)] = settings

        build_volume_temps = [tool.material('build volume temperature') for tool in generator.tools]
        if any(temp is not None for temp in build_volume_temps):
            header['BUILD_VOLUME'] = {'TEMPERATURE': min(temp for temp in build_volume_temps if temp is not None)}

        if kwargs.get('emulate_cura', False):
            header['GENERATOR'] = {'NAME': 'Cura_SteamEngine', 'VERSION': '5.3.0', 'BUILD_DATE': '2023-03-07'}
        lines = ['START_OF_HEADER', *cls._dict2header(header), 'END_OF_HEADER']
        return '\n'.join(f';{line}' for line in lines)

    @classmethod
    def _dict2header(cls, data) -> list[str]:
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                for line in cls._dict2header(value):
                    lines.append(f'{key.upper()}.{line}')
            else:
                lines.append(f'{key.upper()}:{value}')
        return lines


__all__ = ['GriffinWriter']
