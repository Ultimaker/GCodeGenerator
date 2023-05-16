import re
import urllib.request
import xml.etree.ElementTree


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

    def __init__(self, filename: str):
        if filename.startswith('git:'):
            filename = filename.removeprefix('git:')
            commit = 'master'
            if '@' in filename:
                filename, commit = filename.split('@', 1)
            filename = self.assert_suffix(filename, self.EXTENSION)
            self.data = self.download_material(filename, commit)
        else:
            filename = self.assert_suffix(filename, self.EXTENSION)
            with open(filename, 'rb') as file:
                self.data = file.read()
        self.filename = filename

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
