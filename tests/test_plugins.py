import unittest

import gcode_generator
from gcode_generator.plugins import GeneratorPlugin


@GeneratorPlugin
def test1(gcode):
    assert isinstance(gcode, gcode_generator.GCodeGenerator)
    return 1

subplugins = GeneratorPlugin.category('subplugins')
@subplugins
def test2(gcode):
    assert isinstance(gcode, gcode_generator.GCodeGenerator)
    return 2

@subplugins.category('subsub')
def test3(gcode):
    assert isinstance(gcode, gcode_generator.GCodeGenerator)
    return 3

@subplugins.category('subsub')
def test4(gcode):
    assert isinstance(gcode, gcode_generator.GCodeGenerator)
    return 4


class TestPlugin(unittest.TestCase):
    def setUp(self):
        self.gcode = gcode_generator.GCodeGenerator('bla', 1)

    def test_plugin(self):
        self.assertEqual(self.gcode.plugins.test1(), 1)

    def test_subplugin(self):
        self.assertEqual(self.gcode.plugins.subplugins.test2(), 2)

    def test_nested_plugins(self):
        self.assertEqual(self.gcode.plugins.subplugins.subsub.test3(), 3)

    def test_not_found(self):
        with self.assertRaises(AttributeError):
            self.gcode.plugins.not_there()

    def test_same_category_name(self):
        self.assertIs(GeneratorPlugin.category('test'), GeneratorPlugin.category('test'))


if __name__ == '__main__':
    unittest.main()
