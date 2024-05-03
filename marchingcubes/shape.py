import matplotlib.pyplot as plt
import numpy
import opensimplex
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from marchingcubes.constants import *


def plot(triangles: list):
    """Plot a list of triangles

    :param triangles: triangle vertices where first and last index is the same
    """

    figure = plt.figure()
    axis = figure.add_subplot(projection='3d')

    for triangle in triangles:
        collection = Poly3DCollection([triangle])
        axis.add_collection3d(collection)

    size = numpy.max(triangles)

    axis.set_xlim3d(0, size)
    axis.set_ylim3d(0, size)
    axis.set_zlim3d(0, size)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')

    plt.show()


def _construct_triangle(edge_indices: list[int], vertices: numpy.ndarray, noise: numpy.ndarray, interpolate: bool,
                        surface_level: float) -> list[numpy.ndarray]:
    triangle = []

    if interpolate:
        for edge_index in edge_indices:
            index_one, index_two = EDGE_INDEX_TO_VERTEX_INDICES[edge_index]
            x1, y1, z1 = vertices[index_one]
            x2, y2, z2 = vertices[index_two]
            edge = _interpolation(vertices[index_one], vertices[index_two], noise[x1, y1, z1], noise[x2, y2, z2],
                                  surface_level)

            triangle.append(edge)

    else:
        for edge_index in edge_indices:
            index_one, index_two = EDGE_INDEX_TO_VERTEX_INDICES[edge_index]
            edge = numpy.mean([vertices[index_one], vertices[index_two]], axis=0)

            triangle.append(edge)

    return triangle


def _interpolation(vertex_one: numpy.ndarray,
                   vertex_two: numpy.ndarray,
                   noise_value_one: numpy.ndarray,
                   noise_value_two: numpy.ndarray,
                   surface_level: float
                   ) -> numpy.ndarray:
    mu = (surface_level - noise_value_one) / (noise_value_two - noise_value_one)

    return vertex_one + mu * (vertex_two - vertex_one)


def _construct_voxel(noise: numpy.ndarray, vertices: numpy.ndarray, surface_mask: numpy.ndarray,
                     interpolate: bool, surface_level: float) -> list:
    triangulation_index = sum(
        2 ** index * surface_mask[zi, yi, xi] for index, (zi, yi, xi) in enumerate(vertices))

    if triangulation_index == 0 or triangulation_index == 255:
        return []

    edge_indices = TRIANGULATION_TABLE[triangulation_index]

    return [
        _construct_triangle(edge_indices[index:index + 3], vertices, noise, interpolate, surface_level) for index in
        range(0, len(edge_indices), 3)
    ]


def _get_noise(size: float, sample_points: int) -> numpy.ndarray:
    step = 1 / sample_points
    xi = numpy.arange(start=0, stop=size + step, step=step)
    yi = numpy.arange(start=0, stop=size + step, step=step)
    zi = numpy.arange(start=0, stop=size + step, step=step)

    return opensimplex.noise3array(xi, yi, zi)


def construct(size: float, surface_level: float = 0.0, sample_points: int = 1,
              interpolate: bool = True) -> list[list[numpy.ndarray]]:
    """Construct a triangulated shape on a surface level determined by opensimplex noise

    :param size: size of the shape (required)
    :param surface_level: level at which the surface is to be constructed (optional)
    :param sample_points: number of samples per voxel (optional)
    :param interpolate: enable linear interpolation between vertices (optional)
    :return: vertices where first and last index is the same
    """

    noise = _get_noise(size, sample_points)
    triangles = []
    surface_mask = numpy.less(noise, numpy.full(noise.shape, surface_level))

    for x, y, z in _iterator(noise):
        vertices = _get_vertices(x, y, z)
        triangles_in_cube = _construct_voxel(noise, vertices, surface_mask, interpolate, surface_level)
        scaled_triangles_in_cube = numpy.divide(triangles_in_cube, n)

        triangles.extend(scaled_triangles_in_cube)

    return triangles


def _get_vertices(x: float, y: float, z: float) -> numpy.ndarray:
    return numpy.array([
        [x, y + 1, z],  # 0
        [x + 1, y + 1, z],  # 1
        [x + 1, y, z],  # 2
        [x, y, z],  # 3
        [x, y + 1, z + 1],  # 4
        [x + 1, y + 1, z + 1],  # 5
        [x + 1, y, z + 1],  # 6
        [x, y, z + 1],  # 7
    ])


def _iterator(iter_array: numpy.ndarray):
    iter_shape = numpy.array(iter_array.shape) - 1

    for _ in (iter_object := numpy.nditer(numpy.empty(iter_shape), ["multi_index"])):
        xi, yi, zi = iter_object.multi_index

        yield xi, yi, zi
