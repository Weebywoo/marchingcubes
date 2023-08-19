import matplotlib.pyplot as plt
import numpy
import opensimplex
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from marchingcubes.constants import Constants


def plot(triangles: list, plot_size: tuple[float, float, float], path: str | None = None):
    """Plot a list of triangles

    :param triangles: triangle vertices where first and last index is the same
    :param plot_size: size of the plot
    :param path: Path of image of plot, if given won't plot. If not given won't save image to path.
    """
    figure = plt.figure()
    axis = figure.add_subplot(projection='3d')

    for triangle in triangles:
        collection = Poly3DCollection([triangle])
        axis.add_collection3d(collection)

    axis.set_xlim3d(0, plot_size[0] - Constants.NOISE_RESOLUTION)
    axis.set_ylim3d(0, plot_size[1] - Constants.NOISE_RESOLUTION)
    axis.set_zlim3d(0, plot_size[2] - Constants.NOISE_RESOLUTION)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')

    if path is None:
        plt.show()

    else:
        figure.savefig(path, dpi=300)


def _construct_triangle(edge_indices: list[int], vertices: numpy.ndarray, noise: numpy.ndarray) -> list[numpy.ndarray]:
    triangle = []

    if Constants.INTERPOLATE:
        for edge_index in edge_indices:
            index_one, index_two = Constants.EDGE_INDEX_TO_VERTEX_INDICES[edge_index]
            x1, y1, z1 = vertices[index_one]
            x2, y2, z2 = vertices[index_two]
            edge = _interpolation(vertices[index_one], vertices[index_two], noise[x1, y1, z1], noise[x2, y2, z2])

            triangle.append(edge * Constants.NOISE_RESOLUTION)

    else:
        for edge_index in edge_indices:
            index_one, index_two = Constants.EDGE_INDEX_TO_VERTEX_INDICES[edge_index]
            edge = numpy.mean([vertices[index_one], vertices[index_two]], axis=0)

            triangle.append(edge * Constants.NOISE_RESOLUTION)

    return triangle


def _interpolation(vertex_one: numpy.ndarray,
                   vertex_two: numpy.ndarray,
                   noise_value_one: numpy.ndarray,
                   noise_value_two: numpy.ndarray
                   ) -> numpy.ndarray:
    mu = (Constants.SURFACE_LEVEL - noise_value_one) / (noise_value_two - noise_value_one)

    return vertex_one + mu * (vertex_two - vertex_one)


def _construct_triangles(noise: numpy.ndarray, vertices: numpy.ndarray, surface_mask: numpy.ndarray) -> list:
    triangulation_index = sum(
        2 ** index * surface_mask[zi, yi, xi] for index, (zi, yi, xi) in enumerate(vertices))

    if triangulation_index == 0 or triangulation_index == 255:
        return []

    edge_indices = Constants.TRIANGULATION_TABLE[triangulation_index]

    return [
        _construct_triangle(edge_indices[index:index + 3], vertices, noise) for index in range(0, len(edge_indices), 3)
    ]


def _get_noise(noise_size: tuple[float, float, float]) -> numpy.ndarray:
    xi = numpy.arange(start=0, stop=noise_size[0], step=Constants.NOISE_RESOLUTION)
    yi = numpy.arange(start=0, stop=noise_size[1], step=Constants.NOISE_RESOLUTION)
    zi = numpy.arange(start=0, stop=noise_size[2], step=Constants.NOISE_RESOLUTION)

    return (opensimplex.noise3array(xi, yi, zi) + 1) / 2 / Constants.NOISE_SCALE


def construct(shape_size: tuple[float, float, float],
              noise: numpy.ndarray | None = None,
              surface_level: float = Constants.SURFACE_LEVEL,
              noise_scale: float = Constants.NOISE_SCALE,
              noise_resolution: float = Constants.NOISE_RESOLUTION,
              interpolate: bool = Constants.INTERPOLATE,
              seed: int | None = None
              ) -> list[list[numpy.ndarray]]:
    """Construct a triangulated shape on a surface level determined by opensimplex noise

    :param shape_size: size of the shape
    :param noise: noise values of shape size (optional)
    :param surface_level: level at which the surface is to be constructed (optional)
    :param noise_scale: value used to scale the noise (optional)
    :param noise_resolution: resolution at which noise is captured (optional)
    :param interpolate: enable linear interpolation between vertices (optional)
    :param seed: seed used to determine noise generation (optional)
    :return: vertices where first and last index is the same
    """
    Constants.SURFACE_LEVEL = surface_level
    Constants.NOISE_SCALE = noise_scale
    Constants.NOISE_RESOLUTION = noise_resolution
    Constants.INTERPOLATE = interpolate

    if seed is not None:
        opensimplex.seed(seed)

    noise = _get_noise(shape_size) if noise is None else noise

    triangles = []
    surface_mask = numpy.less(noise, numpy.full(noise.shape, Constants.SURFACE_LEVEL))

    for x, y, z in _iterator(noise):
        vertices = _get_vertices(x, y, z)
        triangles_in_cube = _construct_triangles(noise, vertices, surface_mask)

        triangles.extend(triangles_in_cube)

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
