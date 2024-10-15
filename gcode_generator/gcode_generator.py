import contextlib
import io
import re
import enum
import math
import typing
import tempfile
import warnings
from typing import Any, Union, Optional, Literal, TextIO

import numpy as np

from .machine import Tool, PrimeStrategy, NozzleOffset, ToolManager
from .vector import Vector, Axis, TransformManager, Transform
from .fdm_material import FDMReader
from .gcode_writer import GCodeWriter
from .plugins import GeneratorPlugin


class GCodeWarning(RuntimeWarning):
    ...


class Arc:
    CW = 'CW'
    CCW = 'CCW'

    def __init__(self, start: Vector, end: Vector,
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
        distance = self.start.distance_to(end)
        if math.isclose(self.radius * 2, distance):
            self.radius = distance / 2
        elif (self.radius * 2) < distance:
            raise ValueError('Arc too small to move to new position')

        x0 = self.start.to_array("XYZ")   # Start position
        x1 = self.end.to_array("XYZ")     # Target position
        dx = x1 - x0
        if self.direction == Arc.CW:
            dx = -dx

        mid = (x0 + x1) / 2
        centercross = np.cross(dx, np.array([0, 0, 1]))
        bisector_direction = centercross / np.sqrt(centercross.dot(centercross))
        bisector_length = math.sqrt((self.radius ** 2) - ((distance / 2) ** 2))
        self.circle_center = mid - bisector_direction * bisector_length
        self.i = self.circle_center[0] - x0[0]        # dX from start to center
        self.j = self.circle_center[1] - x0[1]        # dY from start to center

        x0_norm = x0 - self.circle_center
        x1_norm = x1 - self.circle_center
        arc_angle = math.acos(round(np.dot(x0_norm, x1_norm) / np.linalg.norm(x0_norm) / np.linalg.norm(x1_norm), 10))
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


class GCodeGenerator:
    def __init__(self, machine_name, layer_height=None):
        self.transform = TransformManager(self)
        self.tools = ToolManager(self)

        self.buffer = io.StringIO(newline='\n')
        self._position = Vector(0.0, 0.0, 0.0)
        self.layer_height = layer_height
        self.machine_name = machine_name

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
        warnings.warn("GCodeGenerator.create_tool() has been deprecated. Use GCodeGenerator.tools.new() instead.", DeprecationWarning)
        self.tools.new(hotend_name=hotend_name, material=material, line_width=line_width, offset=offset)

    def select_tool(self, idx):
        warnings.warn("GCodeGenerator.select_tool() has been deprecated. Use GCodeGenerator.tools.select() instead.", DeprecationWarning)
        self.tools.select(idx)

    @property
    def tool(self) -> Tool:
        warnings.warn("GCodeGenerator.tool has been deprecated. Use GCodeGenerator.tools.current instead.", DeprecationWarning)
        return self.tools.current

    @property
    def initial_tool(self) -> int:
        warnings.warn("GCodeGenerator.initial_tool has been deprecated. Use GCodeGenerator.tools.initial instead.", DeprecationWarning)
        return self.tools.initial

    @property
    def raw_position(self) -> Vector:
        return self._position

    @property
    def position(self) -> Vector:
        return (self.transform.inverse @ self.raw_position).view(Vector)

    def apply_transform(self, vector: np.ndarray):
        return (self.transform @ vector).view(Vector)

    def push_transform(self, transform: Transform):
        self.transform.append(transform)
    def pop_transform(self):
        self.transform.pop(-1)

    @staticmethod
    def _format_float(number: float) -> str:
        return f"{number:.6f}".rstrip("0").rstrip(".")

    def move(self, x: float = None, y: float = None, z: float = None, e: float = None, f: float = None, relative=None, cmd='G0', extra_args=None):
        """
        Move the printhead in X,Y,Z and E axes, at a feedrate of F mm/s.
          Add `relative=Axis.ALL` to use relative positions (or `relative=Axis.E` to only have relative E axis...)
        """
        virtual_position = self.position.copy()
        virtual_position.update(x=x, y=y, z=z, e=e, relative=relative)

        old_position = self.raw_position.copy()
        new_position = self.apply_transform(virtual_position)

        args = [cmd]
        for axis in "XYZE":
            if not math.isclose(self.raw_position[axis], new_position[axis]):
                args.append(axis + self._format_float(new_position[axis]))
                self.raw_position[axis] = new_position[axis]
        if f is None:
            f = float(self.tools.current.material('travel speed', 150))
        if int(f*60) != int(self.current_feedrate*60):
            args.append(f"F{f*60:.0f}")
            self.current_feedrate = f

        self.print_time += old_position.distance_to(new_position) / self.current_feedrate
        self.tool.position = new_position['e']

        if extra_args is not None:
            args += extra_args
        self.writeline(" ".join(args))

    def set_position(self, x: float = None, y: float = None, z: float = None, e: float = None, relative=None):
        """
        Set the current position of XYZ or E axis.
          It only keeps track of the position change for the E axis, so the bounding box in the header is probably
          going to be wrong when using this method on X,Y or Z axes.
        """
        new_position = self.raw_position.copy()
        new_position.update(x=x, y=y, z=z, e=e, relative=relative)

        args = ["G92"]
        for axis in "XYZE":
            if not math.isclose(self.raw_position[axis], new_position[axis]):
                args.append(axis + self._format_float(new_position[axis]))
                if axis == "E":
                    self.tool.material.usage += self.tool.position - new_position[axis]
                    self.tool.position = new_position[axis]
                else:
                    warnings.warn("set_position() not recommended for XYZ axes! Use a transform instead.")

        self.writeline(" ".join(args))

    def extrude(self, x: float = None, y: float = None, z: float = None, flowrate: float = 1.0, f: float = None, relative=None):
        """
        Move the printhead in X,Y,Z axes, and automatically calculate how much filament to extrude
        """
        virtual_position = self.position.copy()
        virtual_position.update(x=x, y=y, z=z, relative=relative)

        new_position = self.apply_transform(virtual_position)
        distance = self.raw_position.distance_to(new_position)
        material_volume = self.layer_height * self.tool.line_width * distance  # mm^3 of material for the line being extruded
        material_distance = material_volume / self.tool.material.area  # mm of filament to feed
        material_distance *= flowrate
        if new_position.z < self.layer_height:  # Automatically reduce flowrate if z < layer height
            warnings.warn(f'Reducing flowrate because Z position is less than layer height', GCodeWarning, stacklevel=2)
            material_distance *= (new_position.z / self.layer_height)

        if f is None:
            f = float(self.tools.current.material('print speed', 70))

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
            position = self.raw_position
        if self.boundingbox is None:
            self.boundingbox = (self.raw_position.copy(), self.raw_position.copy())
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
        virtual_position = self.position.copy()
        virtual_position.update(x=x, y=y, z=z, relative=relative)

        start = self.raw_position.copy()
        end = self.apply_transform(virtual_position)

        if f is None:
            if extrude:
                f = self.tool.material('print speed', 70)
            else:
                f = self.tool.material('travel speed', 150)

        arc = Arc(self.position, virtual_position, r, i, j, direction)

        if segments:
            for a, z in np.linspace((arc.start_angle, start.z), (arc.end_angle, end.z), segments+1):
                x = arc.circle_center[0] + np.cos(a) * arc.radius
                y = arc.circle_center[1] + np.sin(a) * arc.radius
                if extrude:
                    self.extrude(x, y, z, flowrate=flowrate)
                else:
                    self.move(x, y, z)
        else:
            warnings.warn("GCodeGenerator.arc() called without segment count. Using G2/G3. Not recommended with transforms, and firmware support is not great!")
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
