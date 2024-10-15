import functools
from typing import Iterator, Type, TYPE_CHECKING, Union
import enum
import numpy as np

if TYPE_CHECKING:
    from .gcode_generator import GCodeGenerator


class Axis(enum.Flag):
    X = 1 << 0
    Y = 1 << 1
    Z = 1 << 2
    E = 1 << 3

    NONE = 0
    ALL = X | Y | Z | E

    def upper(self):
        return self.name.upper()


class Vector(np.ndarray):

    AXES = "XYZE"

    def __new__(cls, *args, **kwargs):
        array = np.zeros(len(cls.AXES) + 1)
        array[0:len(args)] = args
        array[-1] = 1
        return array.view(cls)

    def __getitem__(self, item):
        if isinstance(item, (str, Axis)) and item.upper() in self.AXES:
            item = self.AXES.index(item.upper())
        return super().__getitem__(item)

    def __setitem__(self, item, value):
        if isinstance(item, (str, Axis)) and item.upper() in self.AXES:
            item = self.AXES.index(item.upper())
        return super().__setitem__(item, value)

    def __getattr__(self, item):
        if item.upper() in self.AXES:
            return self[item]
        return super().__getattribute__(item)

    def __setattr__(self, item, value):
        if item.upper() in self.AXES:
            self[item] = value

    def to_array(self, axes = Axis.ALL):
        return np.array([self[axis] for axis in axes])

    def to_dict(self):
        return {axis: self[axis] for axis in self.AXES}

    def items(self) -> Iterator[tuple[str, float]]:
        for axis in self.AXES:
            yield axis, self[axis]

    def distance_to(self, other: "Vector") -> float:
        return np.linalg.norm(self - other)

    def update(self, *, relative: Union[bool, Axis] = Axis.NONE, **kwargs):
        if isinstance(relative, bool):
            relative = Axis.ALL if relative else Axis.NONE
        for key, value in kwargs.items():
            if value is not None and key.upper() in self.AXES:
                if relative is not None and Axis[key.upper()] in relative:
                    self[key] += value
                else:
                    self[key] = value


class Transform(np.ndarray):
    def __new__(cls, matrix: np.ndarray, generator: "GCodeGenerator" = None):
        if not isinstance(matrix, np.ndarray):
            matrix = np.array(matrix)
        if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Input has to be a 2D square matrix")
        self = matrix.view(cls)
        self.generator = generator
        return self

    def __enter__(self):
        self.generator.push_transform(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.generator is None:
            raise RuntimeError("Transform cannot be used as a context without a GCodeGenerator specified")
        self.generator.pop_transform()


def update_on_change(cls: Type):
    """Class decorator to call self._update() whenever the inherited list/dict/set is mutated."""
    names = {
        "__delitem__", "__iadd__", "__iand__", "__imul__", "__ior__", "__isub__", "__ixor__", "__setitem__",
        "add", "append", "clear", "difference_update", "discard", "extend", "insert", "intersection_update", "pop",
        "popitem", "remove", "reverse", "setdefault", "sort", "symmetric_difference_update", "update"
    }

    def create_wrapper(key):
        @functools.wraps(getattr(cls, key))
        def wrapper(self, *args, **kwargs):
            result = getattr(super(self.__class__, self), key)(*args, **kwargs)
            self._update()
            return result
        return wrapper

    return type(cls.__name__, (cls,), {name: create_wrapper(name) for name in names if hasattr(cls, name)})


@update_on_change
class TransformManager(list):
    def __init__(self, generator: "GCodeGenerator"):
        super().__init__()
        self._generator = generator
        self._total_transform: np.ndarray = np.identity(5)
        self._inverse_transform: np.ndarray = np.identity(5)

    def __call__(self, matrix: np.ndarray):
        return Transform(matrix, generator=self._generator)

    def __matmul__(self, other):
        return self._total_transform @ other

    @property
    def matrix(self):
        return self._total_transform

    @property
    def inverse(self):
        return self._inverse_transform

    def _update(self):
        print("updating total transform")
        total = np.identity(5)
        for matrix in self[::-1]:
            total = total @ matrix
        self._total_transform = total
        self._total_transform.setflags(write=False)
        self._inverse_transform = np.linalg.inv(total)
        self._inverse_transform.setflags(write=False)

    @classmethod
    def _translation_matrix(cls, dx=0, dy=0, dz=0):
        transform = np.identity(5)
        transform[0:3, -1] = [dx, dy, dz]
        return transform

    def translate(self, dx=0, dy=0, dz=0):
        return self(self._translation_matrix(dx=dx, dy=dy, dz=dz))

    @classmethod
    def _rotation_matrix(cls, angle, x=0, y=0):
        transform = np.identity(5)
        transform[0, 0] = np.cos(angle)
        transform[0, 1] = -np.sin(angle)
        transform[1, 0] = np.sin(angle)
        transform[1, 1] = np.cos(angle)
        return cls._translation_matrix(x, y) @ transform @ cls._translation_matrix(-x, -y)

    def rotate(self, angle, x=0, y=0):
        return self(self._rotation_matrix(angle=angle, x=x, y=y))


__all__ = ["Axis", "Vector", "Transform", "TransformManager"]

if __name__ == '__main__':
    a = np.array([1,2,3])
    v = a.view(Vector)
    print(v)
