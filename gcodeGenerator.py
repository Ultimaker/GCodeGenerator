"""
A simple generator for GCode. To assist in creation of simple GCode instructions.
This is not intended for advanced use or complex paths. The CuraEngine generates the real GCode instructions.
"""
__copyright__ = "Copyright (C) 2013 David Braam - Released under terms of the AGPLv3 License"
#       Added G2/3 and extrudesquare
#       o.vandeven@ultimaker.com 2016-02-10

import math
import numpy as np


class _ToolHead():
    __DEFAULT_PRINTING_TEMPERATURE = 200 # printing temperature for PLA

    # @param diameter the output diameter of the filament
    # @param filament_diameter an optional parameter for the diameter of the input material/feedstock
    def __init__(self, diameter, filament_diameter = 2.85):
        self.__diameter = diameter
        self.__material_used = 1
        self.__filament_diameter = filament_diameter
        self.__printing_temperature = self.__DEFAULT_PRINTING_TEMPERATURE

    # Returns output diameter of the hotend in mm.
    def getDiameter(self):
        return self.__diameter

    # Returns the material used in mm^3
    def getMaterialUsed(self):
        return self.__material_used

    # Logs amount of feedstock used.
    # @param mm_of_material Mm of feedstock/filament used.
    def usedMaterial(self, mm_of_material):
        self.__material_used += self.getMm3(mm_of_material)

    ## Generates the conversion factor between e mm and mm of travel in an extrusion move.
    # @param layer_height the height of the layer on which the conversion factor is used.
    def getExtrusionConversionFactor(self, layer_height = 0.3):
        return (self.__diameter * layer_height * extrusionMultiplier) / (1 / 4 * math.pi * self.__filament_diameter ** 2)

    ## translate mm of feedstock into mm^3 of material.
    # @param mm_of_material Mm of feedstock/filament used.
    def getMm3(self, mm_of_material):
        return mm_of_material * (1 / 4 * math.pi * self.__filament_diameter ** 2)

    ## Returns the tool printing temperature in degrees celsius.
    def getTemperature(self):
        return self.__printing_temperature

    ## sets the initial temperature
    # @param temp the temperature in degrees celsius.
    def setTemperature(self, temp):
        self.__printing_temperature = temp
        
        
class gcodeGenerator(object):
    """
    Generates a simple set of GCode commands for RepRap GCode firmware.
    Use the add* commands to build the GCode, and then use the list function to retrieve the resulting gcode.
    """
    def __init__(self):
        self._feedPrint = 50 * 60
        self._feedTravel = 150 * 60
        self._feedRetract = 25 * 60
        self._layerHeight = 0.1
        self._lineWidth = 0.4
        self._eValue = 0.0
        self._retract_amount = 4.5
        self._last_retract_amount = 0
        self._x = 0
        self._y = 0
        self._z = 0
        self._bedtemp = 60
        self._filamentdiameter = 2.85
        self._list = ['M110', 'G92 E0']
        self.__tools = []
        self.__machine_size = [200.0, 200.0, 200.0, float("inf")]
        self.__NOZZLE_SWITCH_RETRACT_LENGHT = 15 # 20
        self.__normal_retraction_lenght = 6.5
        self.__nr_of_heads=0
        self._preloaddistance=0
        self._Eunit='mm'

    def setPreloaddistance(self, preloaddistance, feedrate=None):
        if feedrate is None:
            feedrate = self._feedRetract
        if not preloaddistance == self._preloaddistance:
            self._eValue += preloaddistance-self._preloaddistance
            self._list += ['G1 E%f F%f' % (self._eValue, feedrate)]
            self._preloaddistance=preloaddistance
    
    def setEunit(self, Eunit):
        self._Eunit = Eunit                
                           
    def createTool(self, nozzle_diameter):
        self.__tools.append(_ToolHead(nozzle_diameter))
        self.__nr_of_heads += 1
    
    def SetToolStartTemperature(self, temperature):
        self.__tools[-1].setTemperature(temperature)
            
    def createHeader(self, time = 300):
        header = [
            ";START_OF_HEADER",
            ";HEADER_VERSION:0.1",
            ";FLAVOR:Griffin",
            ";GENERATOR.NAME:xy calibration script",
            ";GENERATOR.VERSION:2.1",
            ";GENERATOR.BUILD_DATE:2016-05-04",
            ";TARGET_MACHINE.NAME:Ultimaker Jedi"
        ]
        for tool_nr, tool in enumerate(self.__tools):
            header += [
                ";EXTRUDER_TRAIN.{0}.INITIAL_TEMPERATURE:{1}".format(tool_nr, tool.getTemperature()),
                ";EXTRUDER_TRAIN.{0}.MATERIAL.VOLUME_USED:{1}".format(tool_nr, tool.getMaterialUsed()),
                ";EXTRUDER_TRAIN.{0}.NOZZLE.DIAMETER:{1}".format(tool_nr, tool.getDiameter()),
            ]
        header += [
            ";BUILD_PLATE.INITIAL_TEMPERATURE:%d" % self._bedtemp,
            ";PRINT.TIME:%d" % time,
            ";PRINT.SIZE.MIN.X:0",
            ";PRINT.SIZE.MIN.Y:0",
            ";PRINT.SIZE.MIN.Z:0",
            ";PRINT.SIZE.MAX.X:{0}".format(*self.__machine_size),
            ";PRINT.SIZE.MAX.Y:{1}".format(*self.__machine_size),
            ";PRINT.SIZE.MAX.Z:{2}".format(*self.__machine_size),
            ";END_OF_HEADER",
            "",
            "T0",
            "G92 E0",
            "G0 F15000 X9 Y6 Z2",
            "G280",
            "G1 F1500 E-6.5",
#                "T0",
#                "G92 E0",
#                "G0 F7200.000000 X175.000 Y6.000 Z2.000",
#                "G280",
#                "G1 F1200.000 E-{}".format(self.__NOZZLE_SWITCH_RETRACT_LENGHT + self.__normal_retraction_lenght),
#                "G1 Z3.000",
#                "G92 E0",
#                "",
#                "T1",
#                "G92 E0",
#                "G0 F7200.000 X180.000 Y6.000 Z3.000",
#                "M109 S210.000",
#                "G280",
#                "G1 F1200.000 E-{}".format(self.__NOZZLE_SWITCH_RETRACT_LENGHT + self.__normal_retraction_lenght),
#                "M107",
#                "G1 Z3.000",
#                "G92 E0",
#                "T0",
#                "",
            ";Layer count: {}".format(1000),
            ";LAYER:0",
            "M107",
            ";TYPE:WALL-OUTER"
        ]
        return header
    
    def setFilamentDiameter(self, diameter):
        self._filamentdiameter = diameter

    def setPrintSpeed(self, speed):
        self._feedPrint = speed * 60
    
    def setTravelSpeed(self, speed):
        self._feedTravel = speed * 60  

    def setIroningSpeed(self, speed):
        self._feedIroning = speed * 60  
    
    def setExtrusionRate(self, lineWidth, layerHeight, extrusionMultiplier=1):
        self._layerHeight = layerHeight
        self._lineWidth = lineWidth
        self._extrusionMultiplier = extrusionMultiplier

    def setRetractAmount(self, amount):
            self._retract_amount = amount

    def home(self):
        self._x = -1
        self._y = -1
        self._z = -1
        self._list += ['G28']
    
    def wait(self, time):
        if time > 0:
            self._list += ['G4 P%d' % (time * 1000)]
    
    def temperature(self, temp):
        self._list += ['M104 T0 S%d' % (temp)]

    def bedTemperatureWait(self, temp):
        self._list += ['M190 S%d' % (temp)]

    def temperatureWait(self, temp):
        self._list += ['M109 T0 S%d' % (temp)]
    
    def setFanSpeed(self, speed):
        if speed > 0:
            self._list += ['M106 S%d' % (speed * 255 / 100)]
        else:
            self._list += ['M107']

    def move(self, x=None, y=None, z=None):
        cmd = "G0 "
        if x is not None:
            cmd += "X%0.3f " % (x)
            self._x = x
        if y is not None:
            cmd += "Y%0.3f " % (y)
            self._y = y
        if z is not None: # and z != self._z
            cmd += "Z%0.3f " % (z)
            self._z = z
        cmd += "F%d" % (self._feedTravel)
        self._list += [cmd]

    def iron(self, x=None, y=None):
        cmd = "G0 "
        if x is not None:
            cmd += "X%0.3f " % (x)
            self._x = x
        if y is not None:
            cmd += "Y%0.3f " % (y)
            self._y = y
        cmd += "F%d" % (self._feedIroning)
        self._list += [cmd]

    def prime(self, amount=None, feedrate=None):
        if feedrate is None:
            feedrate = self._feedRetract
        if amount is None:
            amount = self._last_retract_amount
        self._eValue += amount
        self._list += ['G1 E%f F%f' % (self._eValue, feedrate)]

    def retract(self, amount=None, feedrate=None):
        if feedrate is None:
            feedrate = self._feedRetract
        if amount is None:
            amount = self._retract_amount
        self._last_retract_amount = amount
        self._eValue -= amount
        self._list += ['G1 E%f F%f' % (self._eValue, feedrate)]

    def unretract(self, amount=None, feedrate=None):
        if feedrate is None:
            feedrate = self._feedRetract
        if amount is None:
            amount = self._retract_amount
        self._last_retract_amount = amount
        self._eValue += amount
        self._list += ['G1 E%f F%f' % (self._eValue, feedrate)]

    def extrude(self, x=None, y=None, z=None):
        cmd = "G1 "
        oldX = self._x
        oldY = self._y
        if x is not None:
            cmd += "X%0.3f " % (x)
            self._x = x
        if y is not None:
            cmd += "Y%0.3f " % (y)
            self._y = y
        if z is not None and z != self._z:
            cmd += "Z%0.3f " % (z)
            self._z = z
        filamentRadius = self._filamentdiameter / 2
        filamentArea = math.pi * filamentRadius * filamentRadius
        ePerMM = (self._lineWidth * self._layerHeight * self._extrusionMultiplier) / filamentArea
        self._eValue += math.sqrt((self._x - oldX) * (self._x - oldX) + (self._y - oldY) * (self._y - oldY)) * ePerMM
        cmd += "E%0.4f F%d" % (self._eValue, self._feedPrint)
        self._list += [cmd]

    def extrudeEComp(self, x=None, y=None, z=None,EComp=0):
        cmd = "G1 "
        oldX = self._x
        oldY = self._y
        if x is not None:
            cmd += "X%0.3f " % (x)
            self._x = x
        if y is not None:
            cmd += "Y%0.3f " % (y)
            self._y = y
        if z is not None and z != self._z:
            cmd += "Z%0.3f " % (z)
            self._z = z
        filamentRadius = self._filamentdiameter / 2
        filamentArea = math.pi * filamentRadius * filamentRadius
        ePerMM = (self._lineWidth * self._layerHeight) / filamentArea
        self._eValue += math.sqrt((self._x - oldX) * (self._x - oldX) + (self._y - oldY) * (self._y - oldY)) * ePerMM
        cmd += "E%0.4f F%d" % (self._eValue+EComp, self._feedPrint)
        self._list += [cmd]

    def extrudeXYZ(self, x=None, y=None, z=None):
        cmd = "G1 "
        oldX = self._x
        oldY = self._y
        oldZ = self._z
        if x is not None:
            cmd += "X%0.3f " % (x)
            self._x = x
        if y is not None:
            cmd += "Y%0.3f " % (y)
            self._y = y
        if z is not None and z != self._z:
            cmd += "Z%0.3f " % (z)
            self._z = z
        filamentRadius = self._filamentdiameter / 2
        filamentArea = math.pi * filamentRadius * filamentRadius
        ePerMM = (self._lineWidth * self._layerHeight * self._extrusionMultiplier) / filamentArea
        self._eValue += math.sqrt((self._x - oldX) * (self._x - oldX) + (self._y - oldY) * (self._y - oldY) + (self._z - oldZ) * (self._z - oldZ)) * ePerMM
        cmd += "E%0.4f F%d" % (self._eValue, self._feedPrint)
        self._list += [cmd]

    def arcextrude(self, x, y, r, rot='CW'):
        oldX = self._x
        oldY = self._y
        X1 = np.array([oldX,oldY,self._z])
        X2 = np.array([x,y,self._z])
        Xmean = (X1 + X2) / 2
        dX = X2 - X1
        if (np.linalg.norm(dX) >= 2 * r):
            self.comment("arc too small, converted to line")
            self.extrude(x,y)
        else:
            if rot == 'CW':          
                cmd = "G2 "
                dX =- dX
            elif rot == 'CCW':
                cmd = "G3 "
                
            cmd += "X%0.3f " % (x)
            self._x = x
            cmd += "Y%0.3f " % (y)
            self._y = y
            
            centercross = np.cross(dX,np.array([0,0,1]))
            
            centerdirection = centercross / np.sqrt(centercross.dot(centercross))
            centerdistance = math.sqrt(r**2 - (np.linalg.norm(dX / 2))**2)
            circlecenter = Xmean - centerdirection * centerdistance
              
            i = circlecenter[0]-X1[0]
            j = circlecenter[1]-X1[1]
            cmd += "I%0.3f " % (i)
            cmd += "J%0.3f " % (j)
            filamentRadius = self._filamentdiameter / 2

            filamentArea = math.pi * filamentRadius * filamentRadius
            ePerMM = (self._lineWidth * self._layerHeight) / filamentArea
            Xs1 = X1 - circlecenter
            Xs2 = X2 - circlecenter
            arcangle = math.acos(np.dot(Xs1,Xs2) / np.linalg.norm(Xs1) / np.linalg.norm(Xs2))
            arclength = arcangle * r
            self._eValue += arclength * ePerMM
            cmd += "E%0.4f F%d" % (self._eValue, self._feedPrint)
            self._list += [cmd]

    def arcextrude_segments(self, x, y, r, rot='CW'):
        oldX = self._x
        oldY = self._y
        X1 = p.array([oldX, oldY,self._z])
        X2 = np.array([x, y, self._z])
        Xmean = (X1 + X2) / 2
        dX = X2 - X1
        if np.linalg.norm(dX) >= 2 * r:
            self.comment("arc too small, converted to line")
            self.extrude(x,y)
        else:
            centercross = np.cross(dX,np.array([0, 0, 1]))
            centerdirection = centercross/np.sqrt(centercross.dot(centercross))
            centerdistance = math.sqrt(r**2 - (np.linalg.norm(dX / 2))**2)
            circlecenter = Xmean - centerdirection * centerdistance
            
            i = circlecenter[0] - X1[0]
            j = circlecenter[1] - X1[1]

            filamentRadius = self._filamentdiameter / 2

            Xs1 = X1 - circlecenter
            Xs2 = X2 - circlecenter
            arcangle = math.acos(np.dot(Xs1, Xs2) / np.linalg.norm(Xs1) / np.linalg.norm(Xs2))
          
            thstart = np.arctan2(-j, -i)
            th = np.linspace(thstart,thstart + arcangle,25)
            X = np.cos(th)* r +circlecenter[0]
            Y = np.sin(th)* r +circlecenter[1]
            for x, y in zip(X[1:], Y[1:]):
                self.extrude(x,y)

    def comment(self, comment):
        self._list += [';%s' % (comment)]

    def addCmd(self, cmd):
        self._list += [cmd]

    def list(self):
        return self._list

    def extrudeAreaX(self, xmin, xmax, ymin, ymax, z):
        w = self._lineWidth
        w2 = w / 2.0
        y = ymin + w2
        self.move(xmin + w2, y, z)
        self.prime()
        while y < ymax:
            self.extrude(xmin + w2, y, z)
            self.extrude(xmax - w2, y, z)
            y += w
            self.extrude(xmax - w2, y, z)
            self.extrude(xmin + w2, y, z)
            y += w
        self.retract()
    
    def extrudeAreaY(self, xmin, xmax, ymin, ymax, z):
        w = self._lineWidth
        w2 = w / 2.0
        x = xmin + w2
        self.move(x, ymin + w2, z)
        self.prime()
        while x < xmax:
            self.extrude(x, ymin + w2, z)
            self.extrude(x, ymax - w2, z)
            x += w
            self.extrude(x, ymax - w2, z)
            self.extrude(x, ymin + w2, z)
            x += w
        self.retract()        
            
    def extrudeSquare(self, center_x, center_y, cube_dimension_x, cube_dimension_y, r, dz):
        z = self._z
        z_step = dz / 4
        z = z + z_step             
        if cube_dimension_x > r * 2:
            self.extrude(center_x + (cube_dimension_x / 2) - r, center_y - (cube_dimension_y / 2), z)
        if r > 0:
            self.arcextrude(center_x + (cube_dimension_x / 2), center_y - (cube_dimension_y / 2) + r, r, 'CCW')
        z = z + z_step                        
        if (cube_dimension_y > r * 2):
            self.extrude(center_x + (cube_dimension_x / 2), center_y + (cube_dimension_y / 2)-r, z)
        if r > 0:                        
            self.arcextrude(center_x + (cube_dimension_x / 2) - r, center_y + (cube_dimension_y / 2), r, 'CCW')
        z = z + z_step                        
        if cube_dimension_x > r * 2:
            self.extrude(center_x - (cube_dimension_x / 2) + r, center_y + (cube_dimension_y / 2), z)
        if r > 0:                        
            self.arcextrude(center_x - (cube_dimension_x / 2), center_y + (cube_dimension_y / 2)-r, r, 'CCW')
        z = z + z_step
        if cube_dimension_y > r * 2:
            self.extrude(center_x - (cube_dimension_x / 2), center_y - (cube_dimension_y / 2) + r, z)
        if r > 0:                        
            self.arcextrude(center_x - (cube_dimension_x / 2) + r, center_y - (cube_dimension_y / 2), r, 'CCW')
        self._z = z
    
    def extrudeCircle(self, center_x,center_y, cube_dimension_x,cube_dimension_y,n,dz):
        z = self._z
        z = z + dz   
        th = np.linspace(0, 2 * np.pi, n + 1)
        x_ = center_x + cube_dimension_x / 2 * np.sin(th)        
        y_ = center_y + cube_dimension_y / 2 * np.cos(th)       
        z_ = z + dz * th / 2 /np.pi
        for n in range(n):
            self.extrude(x_[n+1], y_[n+1], z_[n+1])
        self._z = z
    
    def moveCircle(self, center_x,center_y, cube_dimension_x,cube_dimension_y,n,dz):
        z = self._z
        z = z + dz   
        th = np.linspace(0, 2 * np.pi, n + 1)
        x_ = center_x + cube_dimension_x / 2 * np.sin(th)        
        y_ = center_y + cube_dimension_y / 2 * np.cos(th)       
        z_ = z + dz * th / 2 / np.pi
        for n in range(n):
            self.move(x_[n+1], y_[n+1], z_[n+1])
        self._z = z