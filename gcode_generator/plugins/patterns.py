import math
import typing

import numpy as np

from . import GeneratorPlugin
if typing.TYPE_CHECKING:
    from .. import GCodeGenerator


def remap(x, in_range, out_range):
    in_lo, in_hi = in_range
    out_lo, out_hi = out_range
    return (x - in_lo) / (in_hi - in_lo) * (out_hi - out_lo) + out_lo


@GeneratorPlugin
def circle(gcode: "GCodeGenerator",
           centerx, centery, radius, segments=100,
           startangle=0, endangle=2*math.pi, direction='CCW',
           f=None, extrude=True):

    gcode.move(centerx + math.cos(startangle) * radius,
               centery + math.sin(startangle) * radius)

    assert direction in ('CW', 'CCW')
    if direction == 'CW':
        # If counter-clockwise, move in negative angle direction, endangle should be smaller than startangle
        while endangle > startangle:
            endangle -= 2*math.pi

    for angle in np.linspace(startangle, endangle, segments):
        # angle = (i / segments) * 2 * math.pi + startangle
        x = centerx + math.cos(angle) * radius
        y = centery + math.sin(angle) * radius
        if extrude:
            gcode.extrude(x, y, f=f)
        else:
            gcode.move(x, y, f=f)


@GeneratorPlugin
def rectangle(gcode: "GCodeGenerator", centerx, centery, width, height, radius=0, flowrate=1.0, f=None):
    x0, y0 = centerx - width/2, centery - height/2
    x1, y1 = centerx + width/2, centery + height/2
    if radius > 0:
        if radius > (min(width, height) / 2):
            raise ValueError('Corner radius does not fit in rectangle')
        gcode.move(x0 + radius, y0)
        gcode.extrude(x1 - radius, y0, f=f, flowrate=flowrate)
        gcode.arc(x1, y0 + radius, radius, direction='CCW', f=f, flowrate=flowrate)
        gcode.extrude(x1, y1-radius, f=f, flowrate=flowrate)
        gcode.arc(x1 - radius, y1, radius, direction='CCW', f=f, flowrate=flowrate)
        gcode.extrude(x0 + radius, y1, f=f, flowrate=flowrate)
        gcode.arc(x0, y1 - radius, radius, direction='CCW', f=f, flowrate=flowrate)
        gcode.extrude(x0, y0 + radius, flowrate=flowrate)
        gcode.arc(x0 + radius, y0, radius, direction='CCW', f=f, flowrate=flowrate)
    else:
        gcode.move(x0, y0)
        gcode.extrude(x1, y0, f=f, flowrate=flowrate)
        gcode.extrude(x1, y1, f=f, flowrate=flowrate)
        gcode.extrude(x0, y1, f=f, flowrate=flowrate)
        gcode.extrude(x0, y0, f=f, flowrate=flowrate)


@GeneratorPlugin
def filled_rectangle(gcode: "GCodeGenerator", centerx, centery, width, height, flowrate=1.0, f=None):
    while True:
        if (width >= 2*gcode.tool.line_width) and (height >= 2*gcode.tool.line_width):
            rectangle(gcode, centerx, centery, width, height, flowrate=flowrate, f=f)
            width -= gcode.tool.line_width*2
            height -= gcode.tool.line_width*2
        else:
            break
