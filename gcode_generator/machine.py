import re
import math
import enum
from typing import Union, TextIO, Any, TYPE_CHECKING, Optional

import numpy as np

from .vector import Vector, Transform
from .fdm_material import FDMReader
if TYPE_CHECKING:
    from .gcode_generator import GCodeGenerator


class PrimeStrategy(enum.IntEnum):
    BLOB = 0
    NONE = 1


class NozzleOffset(Transform):
    def __new__(cls, dx=0, dy=0, dz=0):
        matrix = np.identity(5)
        matrix[0:3, -1] = (dx, dy, dz)
        return super().__new__(cls, matrix)


class Tool:
    """
    Tool class - Initialize with a hotend name, material name, machine name and line width to print with.
    """
    def __init__(self, hotend_name: str, material: Union[str, TextIO], machine_name: str,
                 line_width: float = None, offset: Vector = None):
        self.nozzle = Hotend(hotend_name, machine_name)
        self.material = Material(material, machine_name, self.nozzle.name)
        self.line_width = line_width if line_width is not None else self.nozzle.diameter
        self.position = 0.0
        if offset is None:
            offset = Vector(0,0,0)
        self.transform = NozzleOffset(offset['x'], offset['y'], offset['z'])


class Hotend:
    """
    Hotend class - Initialize with a hotend name and machine name
      hotend_name should be a print-core name ('AA 0.4' 'BB 0.8' 'CC 0.6' 'AA 0.25', ...)
       or a UM2+ nozzle size ('0.4 mm', '0.8mm', ...)
    """
    def __init__(self, hotend_name: str, machine: str):
        self.name = hotend_name
        match = re.match(r'(?:[A-Z]{2}\+? )?(\d+\.\d+)(?: ?mm)?', self.name)
        if match:
            self.diameter = float(match.group(1))


class Material:
    """
    Material class - Initialize with a fdm_material file, and optionally machine name and hotend name
      When a machine name and hotend name are specified, it will add the specific settings applicable
    """
    def __init__(self, material: Union[str, TextIO], machine=None, hotend=None):
        self.file = FDMReader(material)
        self.material = self.file.getroot()

        self.diameter = float(self.material.find('properties/diameter').text)
        self.guid = self.material.find('metadata/GUID').text.strip()
        self.usage = 0.0

        self.settings: dict[str, Any] = {}
        for element in self.material.findall('settings/setting'):
            self.settings[element.attrib['key']] = element.text
        self._update_settings_from_xml(self.material.find('./settings'))

        # Add machine-specific settings
        #  find XML 'machine' tag that has a `machine_identifier` child with product=[machine] attribute
        machine_settings = self.material.find(f"./settings/machine/machine_identifier[@product='{machine}']/..")
        if machine_settings is not None:
            self._update_settings_from_xml(machine_settings)
            hotend_settings = machine_settings.find(f"hotend[@id='{hotend}']")
            if hotend_settings:
                self._update_settings_from_xml(hotend_settings)

    def _update_settings_from_xml(self, element):
        for child in element.findall('./setting'):
            self.settings[child.attrib['key']] = child.text.strip()

    @property
    def area(self):
        """Cross-sectional area of material"""
        return math.pi * ((self.diameter / 2)**2)

    def __call__(self, key, default=None):
        return self.settings.get(key, default)

    def __getitem__(self, item):
        return self.settings[item]

    def __setitem__(self, key, value):
        self.settings[key] = value


class ToolManager(list):
    def __init__(self, generator: "GCodeGenerator"):
        super().__init__()
        self._generator = generator
        self._active_tool: Optional[int] = None
        self._initial_tool: Optional[int] = None

    def new(self, hotend_name, material: Union[str, TextIO], line_width=None, offset=None):
        """
        Add a tool to the machine.
          For hotend_name, see the Hotend class
          For material_name, see the Material class
          line_width is the extrusion width - set to the nozzle diameter if not specified
          offset is the XY(Z) offset of this nozzle compared to the printer position
        """
        tool = Tool(hotend_name, material, self._generator.machine_name, line_width, offset=offset)
        self.append(tool)

    def select(self, index: int):
        """Select a tool to print with, and set the extrusion position to 0"""
        if self._active_tool is None:
            self._initial_tool = index
        else:
            self._generator.set_temperature(int(self.current.material('standby temperature', 100)), tool=self._active_tool, wait=False)

        self._active_tool = index
        self._generator.writeline(f'T{index}')
        self._generator.set_position(e=0)
        # Set and wait new tool to print temperature
        self._generator.set_temperature(int(self.current.material('print temperature', 200)), tool=index, wait=True)

        for transform_index, transform in enumerate(self._generator.transform):
            if isinstance(transform, NozzleOffset):
                self._generator.transform[transform_index] = self.current.transform
                break
        else:
            self._generator.transform.insert(0, self.current.transform)

    @property
    def current(self) -> Tool:
        if len(self) == 0:
            raise RuntimeError('No tool created')
        if self._active_tool is None:
            self.select(0)
        return self[self._active_tool]

    @property
    def initial(self) -> int:
        if self._initial_tool is None:
            raise RuntimeError("No tool used yet!")
        return self._initial_tool
