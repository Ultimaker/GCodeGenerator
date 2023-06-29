import io
import re
import enum
import math
import typing
import tempfile
import warnings
from typing import Any, Union, Optional, Literal, TextIO

import numpy as np

from .fdm_material import FDMReader
from .gcode_writer import GCodeWriter
from .plugins import GeneratorPlugin


class GCodeWarning(RuntimeWarning):
    ...


class Vector3:
    """
    3D vector class
    The separate axes can be read or written as attributes (vector.x) or items (vector['x'])
    Addition and subtraction can be done with:
      - another Vector3: Vector3(1,2,3) + Vector3(1,0,0) == Vector3(2,2,3)
      - a list/tuple: Vector3(1,2,3) + (1,0,0) == Vector3(2,2,3)
      - a scalar/number: Vector3(1,2,3) + 1 == Vector3(2,3,4)
    """
    AXES = ('x', 'y', 'z')

    def __init__(self, x, y, z):
        self.array = np.array((x, y, z), dtype=float)

    @classmethod
    def from_array(cls, array):
        self = Vector3.__new__(cls)
        self.array = array
        return self

    def to_array(self: Union["Vector3", Any]) -> np.ndarray:
        if isinstance(self, Vector3):
            return self.array
        return self

    def to_dict(self):
        return {axis: self[axis] for axis in self.AXES}

    def __str__(self):
        return f'{self.__class__.__name__}({self.x}, {self.y}, {self.z})'

    def __repr__(self):
        return self.__str__()

    # Arithmetic methods
    def __add__(self, other):
        return Vector3.from_array(self.array + Vector3.to_array(other))

    def __iadd__(self, other):
        self.array += Vector3.to_array(other)
        return self

    def __sub__(self, other):
        return Vector3.from_array(self.array - Vector3.to_array(other))

    def __isub__(self, other):
        self.array -= Vector3.to_array(other)
        return self

    def __abs__(self):
        """The magnitude of this vector, ||Vector3||"""
        return np.linalg.norm(self.array)

    def distance(self, other: Union["Vector3", tuple[float, float, float]]):
        """Distance form this vector to other vector, ||a - b||"""
        return abs(self - other)

    @property
    def x(self):
        return self.array[0]

    @x.setter
    def x(self, value):
        self.array[0] = value

    @property
    def y(self):
        return self.array[1]

    @y.setter
    def y(self, value):
        self.array[1] = value

    @property
    def z(self):
        return self.array[2]

    @z.setter
    def z(self, value):
        self.array[2] = value

    def __getitem__(self, item):
        if isinstance(item, str) and (item.lower() in self.AXES):
            return getattr(self, item.lower())

    def __setitem__(self, item, value):
        if isinstance(item, str) and (item.lower() in self.AXES):
            setattr(self, item.lower(), value)

    def __iter__(self):
        return self.array.__iter__()

    def copy(self):
        return Vector3(self.x, self.y, self.z)

    def update(self, x: float = None, y: float = None, z: float = None):
        for axis, value in zip(self.AXES, (x, y, z)):
            if value is not None:
                self[axis] = value
        return self

class Arc:
    CW = 'CW'
    CCW = 'CCW'

    def __init__(self, start: Vector3, end: Vector3,
                 r: float = None, i: float = None, j: float = None, direction=CCW):
        self.start = start
        self.end = end
        self.radius = r
        self.direction = direction

        if math.isclose(self.start.x, self.end.x) and math.isclose(self.start.y, self.end.y):
            raise ValueError('Arc start and end have the same XY coordinates')
        assert self.direction in (Arc.CW, Arc.CCW)
        if self.radius is None and (i is None or j is None):
            raise ValueError('Specify radius(r) or dX(i) and dY(j)')
        if self.radius is None:
            self.radius = math.sqrt(i ** 2 + j ** 2)
        distance = self.start.distance((self.end.x, self.end.y, self.start.z))
        if (self.radius * 2) < distance:
            raise ValueError('Arc too small to move to new position')

        x0 = self.start.to_array()   # Start position
        x1 = self.end.to_array()     # Target position
        dx = x1 - x0
        if self.direction == Arc.CW:
            dx = -dx

        mid = (x0 + x1) / 2
        centercross = np.cross(dx, np.array([0, 0, 1]))
        bisector_direction = centercross / np.sqrt(centercross.dot(centercross))
        bisector_length = math.sqrt((r ** 2) - ((distance / 2) ** 2))
        self.circle_center = mid - bisector_direction * bisector_length
        self.i = self.circle_center[0] - x0[0]        # dX from start to center
        self.j = self.circle_center[1] - x0[1]        # dY from start to center

        x0_norm = x0 - self.circle_center
        x1_norm = x1 - self.circle_center
        arc_angle = math.acos(np.dot(x0_norm, x1_norm) / np.linalg.norm(x0_norm) / np.linalg.norm(x1_norm))
        self.arc_length = arc_angle * self.radius
        self.spiral_length = math.sqrt(self.arc_length**2 + (self.end.z - self.start.z)**2)

        self.start_angle = np.arctan2(x0_norm[1], x0_norm[0])
        self.end_angle = np.arctan2(x1_norm[1], x1_norm[0])
        if self.direction == Arc.CW:
            while self.end_angle > self.start_angle:
                self.end_angle -= 2*np.pi
        else:
            while self.end_angle < self.start_angle:
                self.end_angle += 2*np.pi


class Axis(enum.Flag):
    X = 1 << 0
    Y = 1 << 1
    Z = 1 << 2
    E = 1 << 3

    NONE = 0
    ALL = X | Y | Z | E


class PrimeStrategy(enum.IntEnum):
    BLOB = 0
    NONE = 1


class Tool:
    """
    Tool class - Initialize with a hotend name, material name, machine name and line width to print with.
    """
    def __init__(self, hotend_name: str, material: Union[str, TextIO], machine_name: str,
                 line_width: float = None, offset: Vector3 = None):
        self.nozzle = Hotend(hotend_name, machine_name)
        self.material = Material(material, machine_name, self.nozzle.name)
        self.line_width = line_width if line_width is not None else self.nozzle.diameter
        self.position = 0.0
        self.offset = offset if offset is not None else Vector3(0,0,0)


class Hotend:
    """
    Hotend class - Initialize with a hotend name and machine name
      hotend_name should be a print-core name ('AA 0.4' 'BB 0.8' 'CC 0.6' 'AA 0.25', ...)
       or a UM2+ nozzle size ('0.4 mm', '0.8mm', ...)
    """
    def __init__(self, hotend_name: str, machine: str):
        self.name = hotend_name
        match = re.match(r'(?:[A-Z]{2} )?(\d+\.\d+)(?: ?mm)?', self.name)
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


class GCodeGenerator:
    def __init__(self, machine_name, layer_height=None):
        self.buffer = io.StringIO(newline='\n')
        self.position = Vector3(0.0, 0.0, 0.0)
        self.layer_height = layer_height
        self.machine_name = machine_name
        self.tools: list[Tool] = []

        self.boundingbox = None
        self._active_tool = None
        self._initial_tool = None
        self._current_layer = -1
        self.current_feedrate = 0
        self.print_time = 0

        self.plugins = GeneratorPlugin.bind(self)

        self.writeline('M82')
        self.writeline()
        self.comment('start of gcode')

    def create_tool(self, hotend_name, material: Union[str, TextIO], line_width=None, offset=None):
        """
        Add a tool to the machine.
          For hotend_name, see the Hotend class
          For material_name, see the Material class
          line_width is the extrusion width - set to the nozzle diameter if not specified
          offset is the XY(Z) offset of this nozzle compared to the printer position
        """
        tool = Tool(hotend_name, material, self.machine_name, line_width, offset=offset)
        self.tools.append(tool)

    def select_tool(self, idx):
        """Select a tool to print with, and set the extrusion position to 0"""
        if idx < 0 or idx >= len(self.tools):
            raise ValueError('Tool index out of bounds')
        if self.initial_tool is None:
            # Keep track of first tool used
            self._initial_tool = idx
        if idx == self._active_tool:
            # Don't do anything if this tool is already active
            return
        if self._active_tool is not None:
            # Set previous tool to standby temperature
            self.set_temperature(int(self.tool.material('standby temperature', 100)), tool=self._active_tool, wait=False)

        self._active_tool = idx
        self.writeline(f'T{idx}')
        self.set_position(e=0)

        # Set and wait new tool to print temperature
        self.set_temperature(int(self.tool.material('print temperature', 200)), tool=idx, wait=True)

    @property
    def tool(self) -> Tool:
        """The currently selected tool"""
        if len(self.tools) == 0:
            raise RuntimeError('No tool created')
        if self._active_tool is None:
            if len(self.tools) == 1:
                self.select_tool(0)     # Automatically select tool if there is only 1 available
            else:
                raise RuntimeError('No tool selected')
        return self.tools[self._active_tool]

    @property
    def initial_tool(self) -> int:
        return self._initial_tool

    def xyz2args(self, x: float = None, y: float = None, z: float = None, e: float = None, f: float = None) -> list[str]:
        """Convert the axes into the correct GCode format (X12.34 Y5.6 ...)"""
        args = [f'{axis}{pos:.5f}'.rstrip('0').rstrip('.')
                for axis, pos in zip('XYZE', (x, y, z, e))
                if pos is not None]
        if f is not None:
            args.append(f'F{int(f)}')
        return args

    def rel2abs(self, x: float = None, y: float = None, z: float = None, e: float = None, mask: Union[Axis, bool] = Axis.ALL) -> list[Optional[float]]:
        """
        Calculate the absolute position of the relative distances from the current position.
          Use `mask` to only have a certain (set of) axis relative.
          If an axis is None, the return value for that axis is None.
        """
        if mask in (None, False):
            mask = Axis.NONE
        elif mask is True:
            mask = Axis.ALL
        positions = []
        for axis, pos in zip('xyz', (x, y, z)):
            if Axis[axis.upper()] in mask:
                positions.append(self.position[axis] + pos if pos is not None else None)
            else:
                positions.append(pos if pos is not None else None)
        if Axis.E in mask:
            positions.append(self.tool.position + e if e is not None else None)
        else:
            positions.append(e if e is not None else None)
        return positions

    def convert_feedrate(self, f, default=70):
        if f is None:
            f = default
        f = round(f*60)
        if f == self.current_feedrate:
            f = None
        else:
            self.current_feedrate = f
        return f

    def subtract_offset(self, x: float = None, y: float = None, z: float = None) -> list[Optional[float]]:
        """Subtract the current tool's nozzle offset"""
        return [position - offset if position is not None else None
                for (position, offset) in zip((x, y, z), self.tool.offset)]

    def move(self, x: float = None, y: float = None, z: float = None, e: float = None, f: float = None, relative=None, cmd='G0', extra_args=None):
        """
        Move the printhead in X,Y,Z and E axes, at a feedrate of F mm/s.
          Add `relative=Axis.ALL` to use relative positions (or `relative=Axis.E` to only have relative E axis...)
        """
        x, y, z, e = self.rel2abs(x, y, z, e, mask=relative)
        f = self.convert_feedrate(f, self.tool.material('travel speed', 150))
        args = [cmd] + self.xyz2args(*self.subtract_offset(x, y, z), e, f)

        old_position = self.position.copy()
        self.position.update(x, y, z)
        if e is not None:
            self.tool.position = e
        self.print_time += self.position.distance(old_position) / (self.current_feedrate / 60)

        if extra_args is not None:
            args += extra_args
        self.writeline(' '.join(args))

    def set_position(self, x: float = None, y: float = None, z: float = None, e: float = None, relative=None):
        """
        Set the current position of XYZ or E axis.
          It only keeps track of the position change for the E axis, so the bounding box in the header is probably
          going to be wrong when using this method on X,Y or Z axes.
        """
        x, y, z, e = self.rel2abs(x, y, z, e, mask=relative)
        args = ['G92'] + self.xyz2args(x, y, z, e)

        self.position.update(x, y, z)
        if e is not None:
            self.tool.material.usage += self.tool.position - e
            self.tool.position = e
        self.writeline(' '.join(args))

    def extrude(self, x: float = None, y: float = None, z: float = None, flowrate: float = 1.0, f: float = None, relative=None):
        """
        Move the printhead in X,Y,Z axes, and automatically calculate how much filament to extrude
        """
        x, y, z, _ = self.rel2abs(x, y, z, mask=relative)
        if f is None:
            f = self.tool.material('print speed', 70)

        # Calculate E axis
        new_position = self.position.copy().update(x, y, z)
        distance = self.position.distance(new_position)                         # pythagorean distance from old to new position
        material_volume = self.layer_height * self.tool.line_width * distance   # mm^3 of material for the line being extruded
        material_distance = material_volume / self.tool.material.area           # mm of filament to feed
        material_distance *= flowrate
        if new_position.z < self.layer_height:                                  # Automatically reduce flowrate if z < layer height
            warnings.warn(f'Reducing flowrate because Z position is less than layer height', GCodeWarning, stacklevel=2)
            material_distance *= (new_position.z / self.layer_height)

        self.update_bbox()                                                      # Update bounding box with start of line
        self.move(x, y, z, material_distance, f=f, relative=Axis.E, cmd='G1')
        self.update_bbox()                                                      # Update bounding box with end of line

    def extrude_polar(self, angle, length, flowrate: float = 1.0, f: float = None):
        """Extrude in polar coordinates: motion angle [radians] and length [mm]"""
        dx = length * np.cos(angle)
        dy = length * np.sin(angle)
        self.extrude(dx, dy, flowrate=flowrate, f=f, relative=True)

    def update_bbox(self, position=None):
        # Keep track of the bounding box of extruded moves
        if position is None:
            position = self.position
        if self.boundingbox is None:
            self.boundingbox = (self.position.copy(), self.position.copy())
        for axis in 'xyz':
            if position[axis] < self.boundingbox[0][axis]:
                self.boundingbox[0][axis] = position[axis]
            if position[axis] > self.boundingbox[1][axis]:
                self.boundingbox[1][axis] = position[axis]

    def arc(self,
            x: float = None, y: float = None, z: float = None, r: float = None, i: float = None, j: float = None,
            direction=Arc.CCW, f=None, extrude=True, relative=False, segments: Union[Literal[False], int] = False,
            flowrate=1.0):
        """
        Move in an arc shape
          Specify a radius r, or i and j for X and Y circle center offsets
          If `extrude` is True, also extrudes filament during the move
          When `segments` is False, it uses G2/G3 to make the arc
           Otherwise, it should be an integer of how many linear segments to split the arc into.
        """
        x, y, z, _ = self.rel2abs(x, y, z, mask=relative)
        if f is None:
            if extrude:
                f = self.tool.material('print speed', 70)
            else:
                f = self.tool.material('travel speed', 150)

        start = self.position
        end = start.copy().update(x, y, z)
        arc = Arc(start, end, r, i, j, direction)

        if segments:
            for a, z in np.linspace((arc.start_angle, start.z), (arc.end_angle, end.z), segments+1):
                x = arc.circle_center + np.cos(a) * arc.radius
                y = arc.circle_center + np.sin(a) * arc.radius
                if extrude:
                    self.extrude(x, y, z, flowrate=flowrate)
                else:
                    self.move(x, y, z)
        else:
            command = 'G2' if direction == Arc.CW else 'G3'
            args = [f'I{arc.i:.3f}'.rstrip('0').rstrip('.'), f'J{arc.j:.3f}'.rstrip('0').rstrip('.')]

            if extrude:
                material_volume = self.layer_height * self.tool.line_width * arc.spiral_length
                material_distance = material_volume / self.tool.material.area
                material_distance *= flowrate

                angles = sorted([arc.start_angle, arc.end_angle])
                for degree in (-360, -270, -180, -90, 0, 90, 180, 270, 360):
                    a = np.deg2rad(degree)
                    if angles[0] <= a <= angles[1]:
                        x = arc.circle_center[0] + arc.radius * np.cos(a)
                        y = arc.circle_center[1] + arc.radius * np.sin(a)
                        self.update_bbox(Vector3(x, y, start.z))
            else:
                material_distance = None
            self.move(x, y, z, material_distance, f, relative=Axis.E, cmd=command, extra_args=args)

    def set_temperature(self, target, tool: int = None, wait=False):
        """Set the hotend target temperature, and optionally wait for the temperature to reach this target"""
        args = ['M109' if wait else 'M104']
        if tool is not None:
            args.append(f'T{tool}')
        args.append(f'S{target:.0f}')
        self.writeline(' '.join(args))

    def set_bed_temperature(self, target, wait=False):
        """Set the bed target temperature, and optionally wait for the temperature to reach this target"""
        args = ['M190' if wait else 'M140', f'S{target:.0f}']
        self.writeline(' '.join(args))

    def set_fan(self, speed=None):
        """Set the object cooling fan speed, 0-100%. If no speed specified, use print profile speed, or 100%"""
        if speed is None:
            speed = float(self.tool.material('print cooling', 100))
        speed = max(0, min((speed * 255) // 100, 255))
        if speed > 0:
            self.writeline(f'M106 S{speed}')
        else:
            self.writeline(f'M107')

    def retract(self, distance=None, f=None):
        """Retract the material `distance` millimeters and `f` mm/s. If not specified, use print profile settings."""
        if distance is None:
            distance = float(self.tool.material('retraction amount', 4.5))
        if f is None:
            f = float(self.tool.material('retraction speed', 45))
        self.move(e=-distance, f=f, relative=Axis.E)

    def unretract(self, distance=None, f=None):
        """Unretract the material `distance` millimeters and `f` mm/s. If not specified, use print profile settings."""
        if distance is None:
            distance = float(self.tool.material('retraction amount', 4.5))
        if f is None:
            f = float(self.tool.material('retraction speed', 45))
        self.move(e=distance, f=f, relative=Axis.E)

    def prime(self, strategy=PrimeStrategy.BLOB):
        """Let the firmware prime the material of the currently active nozzle"""
        self.writeline(f'G280 S{strategy}')

    def pause(self, time=None):
        """Pause for `time` seconds, or until continued by user if no time specified"""
        if time is None:
            self.writeline('M0')
        else:
            self.wait(time)

    def wait(self, time):
        """Wait `time` seconds"""
        self.writeline(f'G4 S{time}')

    def wait_for_motion(self):
        """Wait for motion to complete"""
        self.writeline('M400')

    def comment(self, text):
        """Add a comment to the GCode"""
        self.writeline(f';{text}')

    def mark_layer(self, layer=None):
        if layer is None:
            layer = self._current_layer + 1
        self._current_layer = layer
        self.comment(f'LAYER:{self._current_layer}')

    def add_time_estimation(self, time=None):
        if time is None:
            time = self.print_time
        self.comment(f'TIME_ELAPSED:{time:.1f}')

    def writeline(self, line=''):
        """Write a line of GCode"""
        self.buffer.write(line + '\n')

    def save(self, file: Union[str, typing.IO], **kwargs):
        self.comment('end of gcode')
        self.writeline()
        self.writeline('M107')                  # Fan off
        for i, tool in enumerate(self.tools):
            self.writeline(f'M104 T{i} S0')     # Hotends off
        self.writeline('M140 S0')               # Bed off

        if self.boundingbox is None:
            self.update_bbox()                      # Make sure we have at least _something_ to put in the header...
        GCodeWriter.write(self, self.buffer.getvalue(), file, **kwargs)
