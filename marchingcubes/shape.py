import numpy
import opensimplex
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from marchingcubes.constants import Constants


def plot(triangles: list, size: tuple[int, int, int]):
    """Plot a list of triangles

    :param triangles: triangle vertices where first and last index is the same
    :param size: size of the plot
    """
    figure = plt.figure()
    axis = figure.add_subplot(projection='3d')

    for triangle in triangles:
        for index in range(0, len(triangle), 3):
            collection = Poly3DCollection([triangle[index:index + 3]])
            axis.add_collection3d(collection)

    axis.set_xlim3d(0, size[0] - 1)
    axis.set_ylim3d(0, size[1] - 1)
    axis.set_zlim3d(0, size[2] - 1)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    plt.show()


def _get_opensimplex_3d_array(vectices: numpy.ndarray, noise_resolution: float, noise_scale: float) -> list[float]:
    return [
        opensimplex.noise3(x / noise_resolution, y / noise_resolution, z / noise_resolution) / noise_scale
        for x, y, z in vectices
    ]


def _get_edge_coordinate(edge_index: int,
                         vertices: numpy.ndarray,
                         noise_levels: numpy.ndarray,
                         surface_level: float,
                         interpolate: bool,
                         ) -> numpy.ndarray:
    index_one, index_two = Constants.EDGE_INDEX_TO_VERTEX_INDICES[edge_index]

    if interpolate:
        return _interpolation(
            vertices[index_one], vertices[index_two],
            noise_levels[index_one], noise_levels[index_two],
            surface_level
        )

    return numpy.mean([vertices[index_one], vertices[index_two]], axis=0)


def _interpolation(vertex_one: numpy.ndarray,
                   vertex_two: numpy.ndarray,
                   noise_value_one: numpy.ndarray,
                   noise_value_two: numpy.ndarray,
                   surface_level: float
                   ) -> numpy.ndarray:
    mu = (surface_level - noise_value_one) / (noise_value_two - noise_value_one)

    return vertex_one + mu * (vertex_two - vertex_one)


def _construct_triangles(noise_levels: numpy.ndarray,
                         vertices: numpy.ndarray,
                         surface_level: float,
                         interpolate: bool
                         ) -> list[numpy.ndarray] | None:
    above_surface_level_mask = [int(noise_level > surface_level) for noise_level in noise_levels]
    triangulation_index = sum(2 ** index * entry for index, entry in enumerate(above_surface_level_mask))

    if edge_indices := Constants.TRIANGULATION_TABLE[triangulation_index]:
        return [
            _get_edge_coordinate(edge_index, vertices, noise_levels, surface_level, interpolate)
            for edge_index in edge_indices
        ]

    return None


def _get_noiselevels(vertices: numpy.ndarray,
                     noise: numpy.ndarray | None,
                     noise_resolution: float,
                     noise_scale: float
                     ) -> numpy.ndarray | list:
    if noise is None:
        return _get_opensimplex_3d_array(vertices, noise_resolution, noise_scale)

    return [noise[z, y, x] for x, y, z in vertices]


def construct(size: tuple[int, int, int],
              noise: numpy.ndarray = None,
              surface_level: float = Constants.SURFACE_LEVEL,
              noise_scale: float = Constants.NOISE_SCALE,
              noise_resolution: float = Constants.NOISE_RESOLUTION,
              interpolate: bool = True,
              seed: int = None
              ) -> list[list[numpy.ndarray]]:
    """Construct a triangulated shape on a surface level determined by opensimplex noise

    :param size: size of the shape
    :param noise: noise values of shape size (optional)
    :param surface_level: level at which the surface is to be constructed (optional)
    :param noise_scale: value used to scale the noise (optional)
    :param noise_resolution: resolution at which noise is captured (optional)
    :param interpolate: enable linear interpolation between vertices (optional)
    :param seed: seed used to determine noise generation (optional)
    :return: vertices where first and last index is the same
    """
    if seed is not None:
        opensimplex.seed(seed)

    shape = []

    for x, y, z in _iterator(size):
        vertices = numpy.array([
            [x, y + 1, z],
            [x + 1, y + 1, z],
            [x + 1, y, z],
            [x, y, z],
            [x, y + 1, z + 1],
            [x + 1, y + 1, z + 1],
            [x + 1, y, z + 1],
            [x, y, z + 1],
        ])
        noise_levels = _get_noiselevels(vertices, noise, noise_resolution, noise_scale)

        if (triangles := _construct_triangles(noise_levels, vertices, surface_level, interpolate)) is not None:
            shape.append(triangles)

    return shape


def _iterator(shape: tuple[int, int, int]):
    for _ in (nditer := numpy.nditer(numpy.empty(shape), ["multi_index"])):
        xi, yi, zi = nditer.multi_index

        if numpy.not_equal((xi + 1, yi + 1, zi + 1), shape).all():
            yield xi, yi, zi


if __name__ == "__main__":
    size = (10, 10, 10)
    shape = construct(size)
    plot(shape, size)
