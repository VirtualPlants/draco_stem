import numpy as np
from scipy import ndimage as nd

from openalea.deploy.shared_data import shared_data

import openalea.cgal_meshing.idra
reload(openalea.cgal_meshing.idra)
from openalea.cgal_meshing.idra import IdraMesh

import openalea.draco_stem.stem.tissue_mesh_optimization
reload(openalea.draco_stem.stem.tissue_mesh_optimization)
from openalea.draco_stem.stem.tissue_mesh_optimization import optimize_topomesh

import openalea.draco_stem.stem.tissue_mesh_quality
reload(openalea.draco_stem.stem.tissue_mesh_quality)
from openalea.draco_stem.stem.tissue_mesh_quality import evaluate_topomesh_quality

import pickle

world.clear()

import vplants.meshing_data
filename = "p194-t2_imgSeg_SegExp_CellShapeCorr"
#filename = "segmentation"
dirname = shared_data(vplants.meshing_data)
meshing_dirname =  dirname.parent.parent

inputfile = dirname+"/segmented_images/"+filename+".inr.gz"

idra = IdraMesh(image_file=inputfile)
world.add(idra.segmented_image,filename,colormap="glasbey",alphamap="constant",bg_id=1)
raw_input()
image_topomesh = idra.idra_topomesh(mesh_fineness=0.8)
world.add(image_topomesh,"CGAL_topomesh")

cell_vertex_file = dirname+"/output_meshes/"+filename+"/image_cell_vertex.dict"
image_cell_vertex = pickle.load(open(cell_vertex_file,'rb'))
image_cell_vertex = dict(zip(image_cell_vertex.keys(),np.array(image_cell_vertex.values())*idra.resolution))

quality_criteria=["Mesh Complexity","Triangle Area Deviation","Triangle Eccentricity","Cell Volume Error","Vertex Distance","Cell Convexity","Epidermis Cell Angle","Vertex Valence","Cell 2 Adjacency"]
cgal_quality = evaluate_topomesh_quality(image_topomesh,quality_criteria,image=idra.segmented_image,image_cell_vertex=image_cell_vertex,image_labels=idra.image_labels,image_cell_volumes=idra.image_cell_volumes)

from vplants.meshing.cute_plot import spider_plot
import matplotlib.pyplot as plt

figure = plt.figure(0)
figure.clf()
spider_plot(figure,np.array([cgal_quality[c] for c in quality_criteria]),color1=np.array([0.3,0.6,1.]),color2=np.array([1.,0.,0.]),xlabels=quality_criteria,ytargets=0.8 * np.ones_like(quality_criteria,float),n_points=100*len(quality_criteria),linewidth=2,smooth_factor=0.0,spline_order=1)
plt.show(block=False)
raw_input()

optimized_topomesh = optimize_topomesh(image_topomesh,omega_forces={'regularization':0.00,'laplacian':1.0,'planarization':0.27,'epidermis_planarization':0.27},cell_vertex_motion=True,image_cell_vertex=image_cell_vertex,edge_flip=True,iterations=20)
world.add(optimized_topomesh,"STEM_topomesh")
optimized_quality = evaluate_topomesh_quality(optimized_topomesh,quality_criteria,image=idra.segmented_image,image_cell_vertex=image_cell_vertex,image_labels=idra.image_labels,image_cell_volumes=idra.image_cell_volumes)

figure = plt.figure(1)
figure.clf()
spider_plot(figure,np.array([optimized_quality[c] for c in quality_criteria]),color1=np.array([0.3,0.6,1.]),color2=np.array([1.,0.,0.]),xlabels=quality_criteria,ytargets=0.8 * np.ones_like(quality_criteria,float),n_points=100*len(quality_criteria),linewidth=2,smooth_factor=0.0,spline_order=1)
plt.show(block=False)
raw_input()







