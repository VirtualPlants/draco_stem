import numpy as np
from scipy import ndimage as nd

from openalea.deploy.shared_data import shared_data

import openalea.draco_stem.draco.draco
reload(openalea.draco_stem.draco.draco)
from openalea.draco_stem.draco.draco import DracoMesh

world.clear()

import vplants.meshing_data
filename = "p194-t2_imgSeg_SegExp_CellShapeCorr"
#filename = "segmentation"
dirname = shared_data(vplants.meshing_data)
meshing_dirname =  dirname.parent.parent

import os
if not os.path.exists(dirname+"output_meshes/"+filename):
    os.makedirs(dirname+"output_meshes/"+filename)

inputfile = dirname+"/segmented_images/"+filename+".inr.gz"
cell_vertex_file = dirname+"/output_meshes/"+filename+"/image_cell_vertex.dict"

draco = DracoMesh(image_file=inputfile, image_cell_vertex_file=cell_vertex_file)

world.add(draco.segmented_image,filename,colormap='glasbey',alphamap='constant',bg_id=1)
world.add(draco.point_topomesh,'image_cells')
#world.add(draco.layer_edge_topomesh['L1'],'L1_adjacency')
#world.add(draco.image_cell_vertex_topomesh,'image_cell_vertex')
raw_input()

#draco.delaunay_adjacency_complex()
draco.layer_adjacency_complex('L1')
#draco.construct_adjacency_complex()
#draco.adjacency_complex_optimization()
world.add(draco.triangulation_topomesh,'cell_adjacency_complex')
raw_input()

triangular = ['star','remeshed','projected','straight']
image_dual_topomesh = draco.dual_reconstruction(reconstruction_triangulation = triangular, adjacency_complex_degree=2)
#image_dual_topomesh = draco.draco_topomesh(reconstruction_triangulation = triangular)
world.add(image_dual_topomesh ,'dual_reconstuction')
raw_input()

DracoMesh()





