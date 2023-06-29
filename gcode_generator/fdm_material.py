import re
import urllib.request
import xml.etree.ElementTree
from typing import Union, TextIO, Optional


class FDMElement(xml.etree.ElementTree.Element):
    def __init__(self, tag, attrib):
        match = re.match(r'\{.*}(.*)', tag)
        if match is not None:
            tag = match.group(1)
        super().__init__(tag, attrib)


class FDMReader:
    """
    Class to read fdm_material files.
     Can read local files, or download a profile from GitHub using 'git:generic_pla[@commit_id]'
    """
    EXTENSION = '.xml.fdm_material'

    def __init__(self, file: Union[str, TextIO]):
        if isinstance(file, str):
            if file.startswith('git:'):
                file = file.removeprefix('git:')
                commit = 'master'
                if '@' in file:
                    file, commit = file.split('@', 1)
                file = self.assert_suffix(file, self.EXTENSION)
                self.data = self.download_material(file, commit)
            else:
                file = self.assert_suffix(file, self.EXTENSION)
                with open(file, 'rb') as f:
                    self.data = f.read()
            self.filename = file
        else:
            self.data = file.read()
            if isinstance(self.data, str):
                self.data = self.data.encode('utf-8')
            self.filename: Optional[str] = None
            for attr in ['name', 'file', 'filename']:
                self.filename = getattr(file, attr, None)
                if isinstance(self.filename, str) and self.filename:
                    self.filename = self.assert_suffix(self.filename, self.EXTENSION)
                    break

        builder = xml.etree.ElementTree.TreeBuilder(element_factory=FDMElement)
        parser = xml.etree.ElementTree.XMLParser(target=builder)
        self.material = xml.etree.ElementTree.fromstring(self.data, parser=parser)

    def getroot(self):
        return self.material

    def __bytes__(self):
        return self.data

    @staticmethod
    def assert_suffix(string: str, suffix: str):
        if not string.endswith(suffix):
            string = string + suffix
        return string

    @classmethod
    def download_material(cls, filename, commit='master'):
        filename = cls.assert_suffix(filename, cls.EXTENSION)
        url = f'https://raw.githubusercontent.com/Ultimaker/fdm_materials/{commit}/{filename}'
        with urllib.request.urlopen(url) as file:
            return file.read()
