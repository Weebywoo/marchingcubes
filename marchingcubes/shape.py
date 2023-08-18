import numpy
import opensimplex
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from marchingcubes.constants import Constants


def plot(triangles: list,
         size: tuple[int, int, int],
         noise_resolution: float = Constants.NOISE_RESOLUTION
         ):
    """Plot a list of triangles

    :param noise_resolution: noise sampling rate
    :param triangles: triangle vertices where first and last index is the same
    :param size: size of the plot
    """
    figure = plt.figure()
    axis = figure.add_subplot(projection='3d')

    for triangle in triangles:
        for index in range(0, len(triangle), 3):
            collection = Poly3DCollection([triangle[index:index + 3]])
            axis.add_collection3d(collection)

    axis.set_xlim3d(0, size[0] - noise_resolution)
    axis.set_ylim3d(0, size[1] - noise_resolution)
    axis.set_zlim3d(0, size[2] - noise_resolution)
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    plt.show()


def _get_edge_coordinate(edge_index: int,
                         vertices: numpy.ndarray,
                         noise_levels: numpy.ndarray,
                         noise_resolution: float,
                         surface_level: float,
                         interpolate: bool,
                         ) -> numpy.ndarray:
    index_one, index_two = Constants.EDGE_INDEX_TO_VERTEX_INDICES[edge_index]

    if interpolate:
        return _interpolation(
            vertices[index_one] * noise_resolution, vertices[index_two] * noise_resolution,
            noise_levels[index_one], noise_levels[index_two],
            surface_level
        )

    return numpy.mean([vertices[index_one], vertices[index_two]], axis=0) * noise_resolution


def _interpolation(vertex_one: numpy.ndarray,
                   vertex_two: numpy.ndarray,
                   noise_value_one: numpy.ndarray,
                   noise_value_two: numpy.ndarray,
                   surface_level: float
                   ) -> numpy.ndarray:
    mu = (surface_level - noise_value_one) / (noise_value_two - noise_value_one)

    return vertex_one + mu * (vertex_two - vertex_one)


def _construct_triangles(noise: numpy.ndarray,
                         vertices: numpy.ndarray,
                         surface_level: float,
                         noise_resolution: float,
                         interpolate: bool
                         ) -> list[numpy.ndarray] | None:
    noise_levels = numpy.array([noise[zi, yi, xi] for xi, yi, zi in vertices])
    above_surface_level_mask = [int(noise_level > surface_level) for noise_level in noise_levels]
    triangulation_index = sum(2 ** index * entry for index, entry in enumerate(above_surface_level_mask))

    if edge_indices := Constants.TRIANGULATION_TABLE[triangulation_index]:
        return [
            _get_edge_coordinate(edge_index, vertices, noise_levels, noise_resolution, surface_level, interpolate)
            for edge_index in edge_indices
        ]

    return None


def _get_noise(size: tuple[int, int, int],
               noise_resolution: float,
               noise_scale: float
               ) -> numpy.ndarray | list:
    xi = numpy.arange(start=0, stop=size[0], step=noise_resolution)
    yi = numpy.arange(start=0, stop=size[1], step=noise_resolution)
    zi = numpy.arange(start=0, stop=size[2], step=noise_resolution)

    return opensimplex.noise3array(xi, yi, zi) / noise_scale


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

    if noise is None:
        noise = _get_noise(size, noise_resolution, noise_scale)

    shape = []

    for x, y, z in _iterator(noise):
        vertices = _get_vertices(x, y, z)
        triangles = _construct_triangles(noise, vertices, surface_level, noise_resolution, interpolate)

        if triangles is not None:
            shape.append(triangles)

    return shape


def _get_vertices(x: float, y: float, z: float) -> numpy.ndarray:
    return numpy.array([
        [x, y + 1, z],
        [x + 1, y + 1, z],
        [x + 1, y, z],
        [x, y, z],
        [x, y + 1, z + 1],
        [x + 1, y + 1, z + 1],
        [x + 1, y, z + 1],
        [x, y, z + 1],
    ])


def _iterator(array: numpy.ndarray):
    for _ in (nditer := numpy.nditer(array, ["multi_index"])):
        xi, yi, zi = nditer.multi_index

        if numpy.not_equal((xi + 1, yi + 1, zi + 1), array.shape).all():
            yield xi, yi, zi


if __name__ == "__main__":
    size = (3, 3, 3)
    shape = construct(size)
    plot(shape, size)
