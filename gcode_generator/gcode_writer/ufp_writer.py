import os
import io
import sys
import json
import typing
import pathlib
import zipfile
import xml.etree.ElementTree

from . import GCodeWriter
from .griffin_writer import GriffinWriter
if typing.TYPE_CHECKING:
    from .. import GCodeGenerator


class UFPWriter(GCodeWriter, extension='ufp'):
    @classmethod
    def write(cls, generator: "GCodeGenerator", gcode: str, file: str, image=None, name=None, **kwargs):
        with zipfile.ZipFile(file, 'w') as ufp:
            # Add gcode file
            with ufp.open('/3D/model.gcode', 'w') as gcode_file:
                GriffinWriter.write(generator, gcode, io.TextIOWrapper(gcode_file, newline='\n'), **kwargs)

            # Add thumbnail image
            if image is None:
                image = os.path.join(os.path.dirname(__file__), '../thumbnail.png')
            cls.add_binary(ufp, image, '/Metadata/thumbnail.png')

            # Add UFP_Global metadata
            with ufp.open('/Metadata/UFP_Global.json', 'w') as ufp_global:
                if name is None:
                    name = os.path.basename(ufp.filename).rsplit('.', 1)[0]
                global_data = {'metadata': {'objects': [{'name': name}]}}
                json.dump(global_data, io.TextIOWrapper(ufp_global, newline='\n'))

            # Add fdm_material files
            material_files = []
            for tool in generator.tools:
                fn = f'/Materials/{os.path.basename(tool.material.file.filename)}'
                if fn not in material_files:
                    cls.add_binary(ufp, bytes(tool.material.file), fn)
                    material_files.append(fn)

            # Add xml files
            with ufp.open('/3D/_rels/model.gcode.rels', 'w') as file:
                UFPRelationships('/Metadata/thumbnail.png', *material_files).write(file)
            with ufp.open('/_rels/.rels', 'w') as file:
                UFPRelationships('/3D/model.gcode', '/Metadata/UFP_Global.json').write(file)
            with ufp.open('/[Content_Types].xml', 'w') as file:
                UFPContentTypes().write(file)

    @classmethod
    def add_binary(cls, ufp, file, target):
        if isinstance(file, str):
            with open(file, 'rb') as f:
                return cls.add_binary(ufp, f, target)
        else:
            if not isinstance(file, bytes):
                file = file.read()
            with ufp.open(target, 'w') as wfile:
                wfile.write(file)


class UFPXML(xml.etree.ElementTree.Element):
    def write(self, file, encoding='utf-8', xml_declaration=True):
        if isinstance(file, str):
            with open(file, 'wb') as f:
                return self.write(f)
        else:
            tree = xml.etree.ElementTree.ElementTree(self)
            if sys.version_info.major > 3 or (sys.version_info.major == 3 and sys.version_info.minor >= 9):
                xml.etree.ElementTree.indent(tree, ' '*2)
            tree.write(file, encoding=encoding, xml_declaration=xml_declaration)


class UFPRelationships(UFPXML):
    TYPES = {
        'thumbnail.png': 'http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail',
        '*.fdm_material': 'http://schemas.ultimaker.org/package/2018/relationships/material',
        '*.gcode': 'http://schemas.ultimaker.org/package/2018/relationships/gcode',
        'UFP_Global.json': 'http://schemas.ultimaker.org/package/2018/relationships/opc_metadata'
    }

    def __init__(self, *relationships: str):
        super().__init__('Relationships', {'xmlns': 'http://schemas.openxmlformats.org/package/2006/relationships'})
        for i, relationship in enumerate(relationships):
            path = pathlib.PurePath(relationship)
            rel_type = [value for key, value in self.TYPES.items() if path.match(key)][0]
            xml.etree.ElementTree.SubElement(self, 'Relationship', {'Target': relationship,
                                                                    'Type': rel_type,
                                                                    'Id': f'rel{i}'})


class UFPContentTypes(UFPXML):
    UFP_CONTENT_TYPES = {
        'rels': 'application/vnd.openxmlformats-package.relationships+xml',
        'gcode': 'text/x-gcode',
        'json': 'application/json',
        'png': 'image/png',
        'xml.fdm_material': 'application/x-ultimaker-material-profile'
    }

    def __init__(self, types=None):
        super().__init__('Types', {'xmlns': 'http://schemas.openxmlformats.org/package/2006/content-types'})
        if types is None:
            types = self.UFP_CONTENT_TYPES
        for ext, mime in types.items():
            xml.etree.ElementTree.SubElement(self, 'Default', {'Extension': ext, 'ContentType': mime})


__all__ = ['UFPWriter']
