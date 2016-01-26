import numpy as np
from scipy import ndimage as nd

from openalea.deploy.shared_data import shared_data

from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh

import openalea.draco_stem.draco.dual_reconstruction
reload(openalea.draco_stem.draco.dual_reconstruction)

import openalea.draco_stem.draco.draco
reload(openalea.draco_stem.draco.draco)
from openalea.draco_stem.draco.draco import DracoMesh

#world.clear()

import vplants.meshing_data
#filename = "p194-t3_imgSeg_SegExp_CellShapeCorr"
#filename = "rs01_wt_t00_seg"
filename = "segmentation"
dirname = shared_data(vplants.meshing_data)
meshing_dirname =  dirname.parent.parent

import os
if not os.path.exists(dirname+"/output_meshes/"+filename):
    os.makedirs(dirname+"/output_meshes/"+filename)

inputfile = dirname+"/segmented_images/"+filename+".inr.gz"
cell_vertex_file = dirname+"/output_meshes/"+filename+"/image_cell_vertex.dict"
triangulation_file = dirname+"/output_meshes/"+filename+"/"+filename+"_draco_adjacency_complex.pkl"

draco = DracoMesh(image_file=inputfile, image_cell_vertex_file=cell_vertex_file, triangulation_file=triangulation_file)

world.add(draco.segmented_image,filename,colormap='glasbey',alphamap='constant',bg_id=1)
world.add(draco.point_topomesh,'image_cells')
#world.add(draco.layer_edge_topomesh['L1'],'L1_adjacency')
#world.add(draco.image_cell_vertex_topomesh,'image_cell_vertex')
raw_input()

#draco.delaunay_adjacency_complex()
#draco.layer_adjacency_complex('L1')
#draco.construct_adjacency_complex()
#draco.adjacency_complex_optimization(n_iterations=1)
world.add(draco.triangulation_topomesh,'cell_adjacency_complex')
raw_input()

from openalea.mesh.property_topomesh_io import save_property_topomesh
save_property_topomesh(draco.triangulation_topomesh,triangulation_file,original_pids=True)

#triangular = ['star','remeshed','projected','straight']
#triangular = ['star','remeshed','projected','flat']
#triangular = ['star','remeshed','projected','exact','flat']
triangular= ['star','flat']
image_dual_topomesh = draco.dual_reconstruction(reconstruction_triangulation = triangular, adjacency_complex_degree=3)
#image_dual_topomesh = draco.draco_topomesh(reconstruction_triangulation = triangular)


triangular_string = ""
for t in triangular:
    triangular_string += t+"_"
topomesh_filename = dirname+"/output_meshes/"+filename+"/"+filename+"_L1"+triangular_string+"_topomesh.ply"
save_ply_property_topomesh(image_dual_topomesh,topomesh_filename,color_faces=True)

world.add(image_dual_topomesh ,'dual_reconstuction')
raw_input()


from openalea.draco_stem.stem.tissue_mesh_quality import evaluate_topomesh_quality
quality_criteria=["Mesh Complexity","Triangle Area Deviation","Triangle Eccentricity","Cell Volume Error","Vertex Distance","Cell Convexity","Epidermis Cell Angle","Vertex Valence","Cell 2 Adjacency"]
draco_quality = evaluate_topomesh_quality(image_dual_topomesh,quality_criteria,image=draco.segmented_image,image_cell_vertex=draco.image_cell_vertex,image_labels=draco.image_labels,image_cell_volumes=draco.image_cell_volumes)

from vplants.meshing.cute_plot import spider_plot
import matplotlib.pyplot as plt

figure = plt.figure(2)
figure.clf()
spider_plot(figure,np.array([draco_quality[c] for c in quality_criteria]),color1=np.array([0.3,0.6,1.]),color2=np.array([1.,0.,0.]),xlabels=quality_criteria,ytargets=0.8 * np.ones_like(quality_criteria,float),n_points=100*len(quality_criteria),linewidth=2,smooth_factor=0.0,spline_order=1)
plt.show(block=False)
raw_input()


topomesh = read_ply_property_topomesh(topomesh_filename)
world.add(topomesh,"reopened_topomesh")








