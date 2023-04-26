# Plugins

When creating a function that makes a more complex pattern than a single command, it should be written as a plugin to keep the code organized.  
A plugin is simply a function with a Plugin decorator. When called, the GCodeGenerator instance is automatically passed as the first parameter to the function.

An example plugin function is as follows:
```python
import typing

from gcode_generator.plugins import GeneratorPlugin
if typing.TYPE_CHECKING:
    from gcode_generator import GCodeGenerator

@GeneratorPlugin
def my_little_function(generator: "GCodeGenerator", param1, param2):
    generator.move(param1)
    generator.set_fan(param2)
    ...
```

This plugin can then be used in other code:
```python
from gcode_generator import GCodeGenerator

generator = GCodeGenerator('Ultimaker S5', layer_height=0.15)
generator.create_tool('AA 0.4', 'generic_pla')
generator.plugins.my_little_function(1, 2)
```
