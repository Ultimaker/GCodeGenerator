# GCodeGenerator
Python library to generate gcode for Ultimaker 3D printers.

:warning: This repository is only for this library, not scripts that use this to create gcode. Store those scripts in your project specific repositories.  
You can install this library using `pip install git+https://github.com/Ultimaker/GCodeGenerator.git`

```python
from gcode_generator import GCodeGenerator
gcode = GCodeGenerator('Ultimaker S7', layer_height=0.15)
gcode.create_tool('AA 0.4', 'generic_pla')

gcode.move(100, 100, 0.15)
gcode.extrude(120, 120)     # Extrudes a diagonal line from (100,100) to (120,120)
...
gcode.save('test.gcode')    # Or 'test.ufp'
```

## Material profiles
This generator uses Ultimaker fdm_materials profile files to automatically use the print settings for the selected material. The settings currently used are:

| setting               | default | unit |
|-----------------------|---------|------|
| `print speed`         | 70      | mm/s |
| `travel speed`        | 150     | mm/s |
| `retraction amount`   | 4.5     | mm   |
| `retraction speed`    | 45      | mm/s |
| `print cooling`       | 100     | %    |
| `print temperature`   | 200     | °C   |
| `standby temperature` | 100     | °C   |

For settings that are not specified in the fdm_material file, the above default values are used.
Print profile settings can be overridden for each extruder separately using  
`generator.tools[0].material['print speed'] = 45`

A local `.xml.fdm_material` file can be used by specifying its file name/location.
A profile directly from the [GitHub repository](https://github.com/Ultimaker/fdm_materials) can also be used by specifying e.g. `git:ultimaker_pla_magenta` as the material name.

## File formats
The gcode can be saved using the `GCodeGenerator.save(file, **kwargs)` method.
`file` can either be a string (filename) or an open file-like object.  
Currently, saving as `.gcode` or `.ufp` are supported. Additionally, you can specify the following keyword arguments:
- `format`: [`gcode`|`ufp`] File format to save as. If not specified, automatically inferred from filename.
- `time_estimate`: number of seconds the print takes. If not specified, an automatic estimate is used.
- `image`: Image to include as thumbnail (UFP only)
- `name`: Name for the object in the metadata json (UFP only)

## Plugins
To keep the code organized, the generator only has basic commands built in. Functions for more complex patterns can be added to the plugins folder.
When a script needs these functions, it can import that module. This makes these functions available in the `GCodeGenerator.plugins` namespace.  
A function defined as:
```python
@GeneratorPlugin
def my_special_function(gcode: "GCodeGenerator", x, y, a, b):
    ...
```
can then be used as:
```python
gcode.plugins.my_special_function(x, y, a, b)
```
See `gcode_generator/plugins/patterns.py` for more examples.
