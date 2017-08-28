import numpy as np

from openalea.image.serial.all import imread, imsave
from openalea.image.spatial_image import SpatialImage

from openalea.draco_stem.draco.draco import DracoMesh
from openalea.draco_stem.example_image import sphere_tissue_image

from openalea.oalab.colormap.colormap_def import load_colormaps

world.clear()

size = 100.
n_points = 11
n_layers = 2
img = sphere_tissue_image(size=size, n_points=n_points, n_layers=n_layers)

world.add(img,"segmented_image",colormap='glasbey',alphamap='constant',bg_id=1,alpha=0.25)

draco = DracoMesh(image=img)
draco.delaunay_adjacency_complex(surface_cleaning_criteria=[])

world.add(draco.triangulation_topomesh,'adjacency_complex')
world['adjacency_complex']['display_3'] = False
world['adjacency_complex']['display_0'] = True
world['adjacency_complex_vertices']['display_colorbar'] = False
world['adjacency_complex_vertices']['polydata_colormap'] = load_colormaps()['glasbey']
world['adjacency_complex_vertices']['point_radius'] = 1.5

draco.adjacency_complex_optimization(n_iterations=3)

world['adjacency_complex']['coef_3'] = 0.9
world['adjacency_complex']['display_3'] = True
world['adjacency_complex_cells']['display_colorbar'] = False
world['adjacency_complex_cells']['polydata_colormap'] = load_colormaps()['grey']
world['adjacency_complex_cells']['intensity_range'] = (-1,0)
world['adjacency_complex_cells']['preserve_faces'] = True
world['adjacency_complex_cells']['x_slice'] = (0,90)

triangular = ['star','remeshed','projected','flat']
image_dual_topomesh = draco.dual_reconstruction(reconstruction_triangulation = triangular, adjacency_complex_degree=3, maximal_edge_length=5.1)



from openalea.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces

compute_topomesh_property(image_dual_topomesh,'barycenter',2)
compute_topomesh_property(image_dual_topomesh,'normal',2,normal_method='orientation')

compute_topomesh_vertex_property_from_faces(image_dual_topomesh,'normal',adjacency_sigma=2,neighborhood=5)
compute_topomesh_property(image_dual_topomesh,'mean_curvature',2)
compute_topomesh_vertex_property_from_faces(image_dual_topomesh,'mean_curvature',adjacency_sigma=2,neighborhood=5)
compute_topomesh_vertex_property_from_faces(image_dual_topomesh,'gaussian_curvature',adjacency_sigma=2,neighborhood=5)

world.add(image_dual_topomesh ,'dual_reconstruction')
world['dual_reconstruction']['coef_3'] = 0.99
world['dual_reconstruction_cells']['display_colorbar'] = False
world['dual_reconstruction_cells']['x_slice'] = (0,40)
world['dual_reconstruction']['display_1'] = True
world['dual_reconstruction_edges']['display_colorbar'] = False
world['dual_reconstruction_edges']['polydata_alpha'] = 0.1
world['dual_reconstruction_edges']['x_slice'] = (0,40)


