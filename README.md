# UMBRA: A Concurrent, Interactive 3D Viewer for Python

This repository contains an interactive 3D viewer for Python. I use variants of this viewer for real-time visualization in most of my research projects and as such, it has an extremely brittle API that can be subject to radical changes at any time. Also, the code is pretty messy. Therefore, use with caution. 

UMBRA may or may not stand for: **U**nderwhelmingly **M**ade, **B**ut **R**enders **A**mazingly.

*"Why a custom viewer if there is already XYZ?"*. 

This viewer is intended for visualizing experiments in computer graphics and computer vision research and there are two key requirements: 

- **concurrent**: some viewers control the main loop, which, for example, makes them unsuitable for visualization the progress of an ongoing iterative optimization. UMBRA runs in a separate thread concurrently to the host Python script or Jupyter notebook.
- **hackable**: most viewers are not implemented in Python but in C++ or JavaScript and are only exposed via bindings, which makes it difficult to modify or customize parts of the renderer. UMBRA is fully implemented in Python using moderngl. The main rendering loop can be easily modified, for example to [shade a mesh using a neural network with PyTorch](https://github.com/fraunhoferhhi/neural-deferred-shading).


This is a minimal example for rendering a moving point cloud:

```python
import numpy as np
import time

from umbra import MeshViewer

# Open a 3D viewer window ("Mesh" is misleading)
viewer = MeshViewer()

# Randomly generate n 3D points
n      = 16
points = np.random.rand(n, 3)

# Loop until the viewer window is closed
fps = 60
i   = 0
while viewer.is_open:
    time.sleep(1/fps) 

    # Move the points and show them in the viewer
    # The point cloud is referenced by its name ("my_points")
    points_offset = points + (i%fps) / fps
    viewer.set_points(points_offset, object_name="my_points")

    i += 1
```

### Points

```python
# A point cloud with default name 
# p.shape = (P,3)
viewer.set_points(p)

# A custom name can be used to identify different objects 
viewer.set_points(p, object_name="my_points")

# ... with colors
# c.shape = (P,3)
viewer.set_points(p, c=c, object_name="my_points")

# ... with custom size (in pixels)
viewer.set_points(p, c=c, point_size=10, object_name="my_points")
```

### Meshes

```python
# Triangle mesh as indexed face set
# v.shape = (V,3)
# f.shape = (F,3)
viewer.set_mesh(v, f, object_name="my_mesh")

# ... with vertex colors
# c.shape = (V,3)
viewer.set_mesh(v, f, c=c, object_name="my_mesh")

# ... and/or with normals
# n.shape = (V,3)
viewer.set_mesh(v, f, c=c, n=n, object_name="my_mesh")

# Change the material. The first parameter can be one of
# ['face', 'smooth', 'normal', 'flat', 'wireframe'] or a moderngl.Program
viewer.set_material('wireframe', index=0, object_name="my_mesh")

# An object can have multiple materials (e.g. to draw wireframe on top of the mesh)
viewer.set_material('smooth', index=0, object_name="my_mesh")
viewer.set_material('wireframe', index=1, object_name="my_mesh")
```

### Lines

```python
# A set of lines
# start.shape = (L,3)
# end.shape   = (L,3)
viewer.set_lines(start, end, object_name="my_lines")

# ... with color
# c.shape = (L,3)
viewer.set_lines(start, end, c=c, object_name="my_lines")
```

## Citation

The development of UMBRA started with the [Neural Deferred Shading](https://github.com/fraunhoferhhi/neural-deferred-shading) project, so please cite it if you are using this renderer or parts of the code for your research. 