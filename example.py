import numpy as np
from gcode_generator import GCodeGenerator, Axis, Vector3

gcode = GCodeGenerator('Ultimaker S7', layer_height = 0.15)
gcode.create_tool('AA 0.4', 'git:generic_pla')
# gcode.create_tool('AA 0.4', 'git:generic_pla', offset=Vector3(22, 0, 0))
# gcode.select_tool(0)

centerx, centery = 150, 100
radius = 25

gcode.move(10, 10)
gcode.prime()
gcode.retract()
gcode.move(z=10, f=25)

gcode.move(centerx, centery, f=45)
gcode.writeline()

gcode.move(x=centerx + radius, y=centery, z=gcode.layer_height)
gcode.unretract()
for layer in range(100):
    gcode.mark_layer()
    gcode.move(z=(layer+1) * gcode.layer_height, f=10)
    gcode.move(centerx+radius, centery)

    for angle in np.linspace(0, 2 * np.pi, 100):
        x = np.cos(angle) * radius + centerx
        y = np.sin(angle) * radius + centery
        gcode.extrude(x, y)

gcode.save('test.ufp')
