# Marchingcubes

A Python package to create a shape based on opensimplex noise"

## Features

- Construct a shape from given noise values
- Construct a shape from generated noise values
- Plot a constructed shape

## Basic Usage

```py
from marchingcubes import shape

size = (10, 10, 10)
triangles = shape.construct(size)
shape.plot(triangles, size)
```

Should produce a plot like this:
![Figure_1.png](https://github.com/Weebywoo/marchingcubes/blob/main/docs/Figure_1.png?raw=true)

For more in depth information visit my [docs](https://marchingcubes.readthedocs.io/en/latest/).

## Helpful links

- [All marching cube triangulation cases in png](https://www.researchgate.net/profile/Zhongjie-Long/publication/282209849/figure/fig2/AS:362916613246979@1463537471898/Type-of-surface-combinations-for-the-marching-cube-algorithm-The-black-circles-means.png)
- [Polygonising a scalar field](http://paulbourke.net/geometry/polygonise/)
- [MARCHING CUBES: A HIGH RESOLUTION 3D SURFACE CONSTRUCTION ALGORITHM](https://people.eecs.berkeley.edu/~jrs/meshpapers/LorensenCline.pdf)
- [Generating Complex Procedural Terrains Using the GPU](https://developer.nvidia.com/gpugems/gpugems3/part-i-geometry/chapter-1-generating-complex-procedural-terrains-using-gpu)
- [Coding Adventure: Marching Cubes - Sebastian Lague](https://www.youtube.com/watch?v=M3iI2l0ltbE)
- [Vertex and Edge indices](http://paulbourke.net/geometry/polygonise/polygonise1.gif)