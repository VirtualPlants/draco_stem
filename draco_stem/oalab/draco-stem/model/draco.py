import numpy as np
from scipy import ndimage as nd

from openalea.deploy.shared_data import shared_data

from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh

from openalea.image.serial.all import imread
from openalea.image.spatial_image import SpatialImage

import openalea.draco_stem.draco.dual_reconstruction
reload(openalea.draco_stem.draco.dual_reconstruction)

import openalea.draco_stem.draco.draco
reload(openalea.draco_stem.draco.draco)
from openalea.draco_stem.draco.draco import DracoMesh

from openalea.oalab.colormap.colormap_def import load_colormaps

world.clear()

import vplants.meshing_data
#filename = "p194-t3_imgSeg_SegExp_CellShapeCorr"
#filename = "rs01_wt_t00_seg"
filename = "segmentation"
#filename = "olli01_lti6b_150421_sam01_t000_seg_hmin_2"
dirname = shared_data(vplants.meshing_data)
meshing_dirname =  dirname.parent.parent

import os
if not os.path.exists(dirname+"/output_meshes/"+filename):
    os.makedirs(dirname+"/output_meshes/"+filename)

inputfile = dirname+"/segmented_images/"+filename+".inr.gz"

#inputfile = "/Users/gcerutti/Developpement/openalea/openalea_marsalt/example/time_0_cut_seg_median.inr" 

img = imread(inputfile)
#img[img==0]=1
#img = SpatialImage(np.concatenate([img[:,:,35:],np.ones((img.shape[0],img.shape[1],5))],axis=2).astype(np.uint16),resolution=img.resolution)

cell_vertex_file = dirname+"/output_meshes/"+filename+"/image_cell_vertex.dict"
triangulation_file = dirname+"/output_meshes/"+filename+"/"+filename+"_draco_adjacency_complex.pkl"

from openalea.container import array_dict

# size = 50
# img = np.ones((size,size,size),np.uint8)

# points = {}
# points[11] = np.array([1,0,0],float)*size
# points[12] = np.array([0,1,0],float)*size
# points[31] = np.array([0,0,1],float)*size
# points[59] = np.array([1,1,1],float)*size
# points = array_dict(points)
# #img[10:50,10:50,30:50] = 59
# #img[10:30,10:50,10:30] = 12
# #img[30:50,10:30,10:30] = 11
# #img[30:50,30:50,10:30] = 31

# center = np.array([[size/2,size/2,size/2]],float)

# from scipy.cluster.vq import vq

# coords = np.transpose(np.mgrid[0:size,0:size,0:size],(1,2,3,0)).reshape((np.power(size,3),3))
# labels = points.keys()[vq(coords,points.values())[0]]

# ext_coords = coords[vq(coords,center)[1]>size/2.]

# img[tuple(np.transpose(coords))] = labels
# #img[tuple(np.transpose(ext_coords))] = 1
# img = SpatialImage(img,resolution=(1,1,1))


world.add(img,"segmented_image",colormap='glasbey',alphamap='constant',bg_id=1)

draco = DracoMesh(image=img, image_cell_vertex_file=cell_vertex_file, triangulation_file=triangulation_file)
#draco = DracoMesh(image_file=inputfile, image_cell_vertex_file=cell_vertex_file)
#draco = DracoMesh(image=img)

#world.add(draco.segmented_image-(draco.segmented_image==1),'segmented_image',colormap='glasbey',alphamap='constant',bg_id=0)
world.add(draco.point_topomesh,'image_cells')
world['image_cells_vertices'].set_attribute('point_radius',img.max())
#world.add(draco.layer_edge_topomesh['L1'],'L1_adjacency')
#world.add(draco.image_cell_vertex_topomesh,'image_cell_vertex')

# cube_points = {}
# cube_points[0] = np.array([0,0,0])*size
# cube_points[1] = np.array([1,0,0])*size
# cube_points[2] = np.array([0,1,0])*size
# cube_points[3] = np.array([0,0,1])*size
# cube_points[4] = np.array([1,1,0])*size
# cube_points[5] = np.array([0,1,1])*size
# cube_points[6] = np.array([1,0,1])*size
# cube_points[7] = np.array([1,1,1])*size

# cube_edges = []
# cube_edges += [[0,1]]
# cube_edges += [[0,2]]
# cube_edges += [[0,3]]
# cube_edges += [[1,4]]
# cube_edges += [[1,6]]
# cube_edges += [[2,4]]
# cube_edges += [[2,5]]
# cube_edges += [[3,5]]
# cube_edges += [[3,6]]
# cube_edges += [[4,7]]
# cube_edges += [[5,7]]
# cube_edges += [[6,7]]

# from openalea.mesh.property_topomesh_creation import edge_topomesh
# world.add(edge_topomesh(cube_edges,cube_points),"box")

raw_input()

from copy import deepcopy


#draco.delaunay_adjacency_complex(surface_cleaning_criteria = [])

draco.delaunay_adjacency_complex(surface_cleaning_criteria = ['surface','sliver','distance'])

triangulation_topomesh = deepcopy(draco.triangulation_topomesh)

from openalea.cellcomplex.property_topomesh.property_topomesh_extraction import epidermis_topomesh, clean_topomesh

triangulation_topomesh = epidermis_topomesh(triangulation_topomesh)

triangulation_triangles = np.array([list(triangulation_topomesh.borders(2,t,2)) for t in triangulation_topomesh.wisps(2)])
not_L1_triangles = np.array(list(triangulation_topomesh.wisps(2)))[np.sum(draco.cell_layer.values(triangulation_triangles) == 1,axis=1)<2]
for t in not_L1_triangles:
    triangulation_topomesh.remove_wisp(2,t)

clean_topomesh(triangulation_topomesh)

triangulation_cells = list(triangulation_topomesh.wisps(3))
for c in triangulation_cells:
    triangulation_topomesh.remove_wisp(3,c)

world.add(triangulation_topomesh,'cell_adjacency_complex')
world['cell_adjacency_complex'].set_attribute('display_3',False)
world['cell_adjacency_complex'].set_attribute('display_1',True)
world['cell_adjacency_complex_edges'].set_attribute('polydata_colormap',load_colormaps()['invert_grey'])
world['cell_adjacency_complex_edges'].set_attribute('intensity_range',(-1,0))
#world['cell_adjacency_complex'].set_attribute('coef_2',0.98)
#world['cell_adjacency_complex_faces'].set_attribute('x_slice',(30,70))
world['cell_adjacency_complex_edges'].set_attribute('display_colorbar',False)

triangular = ['star','split']

L1_draco = deepcopy(draco)
L1_draco.triangulation_topomesh = triangulation_topomesh
image_dual_topomesh = L1_draco.dual_reconstruction(reconstruction_triangulation = triangular, adjacency_complex_degree=2)
#image_dual_topomesh = draco.draco_topomesh(reconstruction_triangulation = triangular)

world.add(image_dual_topomesh ,'dual_reconstuction')
world['dual_reconstuction'].set_attribute('display_3',False)
world['dual_reconstuction'].set_attribute('display_2',True)


#draco.layer_adjacency_complex('L1')
#draco.construct_adjacency_complex()
#draco.adjacency_complex_optimization(n_iterations=2)


#L1_topomesh = deepcopy(draco.triangulation_topomesh)
world.add(L1_topomesh,'L1_adjacency_complex')
world['L1_adjacency_complex'].set_attribute('display_3',False)
world['L1_adjacency_complex'].set_attribute('display_2',True)
world['L1_adjacency_complex_faces'].set_attribute('polydata_colormap',load_colormaps()['grey'])
world['L1_adjacency_complex_faces'].set_attribute('intensity_range',(-1,0))
world['L1_adjacency_complex'].set_attribute('coef_2',0.98)
world['L1_adjacency_complex_faces'].set_attribute('x_slice',(30,70))
world['L1_adjacency_complex_faces'].set_attribute('display_colorbar',False)
world['L1_adjacency_complex_faces'].set_attribute('preserve_faces',True)
world['L1_adjacency_complex'].set_attribute('display_0',True)
world['L1_adjacency_complex_vertices'].set_attribute('point_radius',2.*draco.triangulation_topomesh.nb_wisps(0))
world['L1_adjacency_complex_vertices'].set_attribute('x_slice',(30,70))
world['L1_adjacency_complex_vertices'].set_attribute('display_colorbar',False)


draco.triangulation_topomesh = L1_topomesh
triangular= ['star']
image_dual_topomesh = draco.dual_reconstruction(reconstruction_triangulation = triangular, adjacency_complex_degree=2)
#image_dual_topomesh = draco.draco_topomesh(reconstruction_triangulation = triangular)

world.add(image_dual_topomesh ,'L1_dual_reconstuction')
raw_input()


triangular_string = ""
for t in triangular:
    triangular_string += t+"_"
topomesh_filename = dirname+"/output_meshes/"+filename+"/"+filename+"_draco_stem_L1_"+triangular_string+"topomesh.ply"

save_ply_property_topomesh(image_dual_topomesh,topomesh_filename,color_faces=True)


L1_file = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/output_meshes/p194-t3_imgSeg_SegExp_CellShapeCorr/p194-t3_imgSeg_SegExp_CellShapeCorr_triangulated_L1_star_remeshed_straight_temporal_properties_topomesh.pkl"

image_dual_topomesh = pickle.load(open(L1_file,'r'))


#L1_L2_topomesh = deepcopy(draco.triangulation_topomesh)
world.add(L1_L2_topomesh,'L1_L2_adjacency_complex')
world['L1_L2_adjacency_complex_cells'].set_attribute('polydata_colormap',load_colormaps()['grey'])
world['L1_L2_adjacency_complex_cells'].set_attribute('intensity_range',(-1,0))
world['L1_L2_adjacency_complex'].set_attribute('coef_3',0.95)
world['L1_L2_adjacency_complex_cells'].set_attribute('x_slice',(30,70))
world['L1_L2_adjacency_complex_cells'].set_attribute('display_colorbar',False)
world['L1_L2_adjacency_complex_cells'].set_attribute('preserve_faces',True)
world['L1_L2_adjacency_complex'].set_attribute('display_0',True)
world['L1_L2_adjacency_complex_vertices'].set_attribute('point_radius',1.5*L1_L2_topomesh.nb_wisps(0))
world['L1_L2_adjacency_complex_vertices'].set_attribute('x_slice',(30,70))
world['L1_L2_adjacency_complex_vertices'].set_attribute('display_colorbar',False)

draco.triangulation_topomesh = L1_L2_topomesh
triangular= ['star','remeshed','projected','exact','flat']
image_dual_topomesh = draco.dual_reconstruction(reconstruction_triangulation = triangular, adjacency_complex_degree=3)
#image_dual_topomesh = draco.draco_topomesh(reconstruction_triangulation = triangular)
world.add(image_dual_topomesh ,'L1_L2_dual_reconstuction')
raw_input()

from copy import deepcopy
triangulation_topomesh = deepcopy(draco.triangulation_topomesh)
world.add(triangulation_topomesh,'cell_adjacency_complex')
world['cell_adjacency_complex_cells'].set_attribute('polydata_colormap',load_colormaps()['grey'])
world['cell_adjacency_complex_cells'].set_attribute('intensity_range',(-1,0))
world['cell_adjacency_complex'].set_attribute('coef_3',0.95)
world['cell_adjacency_complex_cells'].set_attribute('x_slice',(50,100))
world['cell_adjacency_complex_cells'].set_attribute('display_colorbar',False)
world['cell_adjacency_complex_cells'].set_attribute('preserve_faces',True)
world['cell_adjacency_complex'].set_attribute('display_0',True)
world['cell_adjacency_complex_vertices'].set_attribute('point_radius',draco.triangulation_topomesh.nb_wisps(0))
world['cell_adjacency_complex_vertices'].set_attribute('x_slice',(50,100))
world['cell_adjacency_complex_vertices'].set_attribute('display_colorbar',False)

#delaunay_topomesh = deepcopy(draco.delaunay_topomesh)
world.add(delaunay_topomesh,'delaunay_complex')
world['delaunay_complex_cells'].set_attribute('polydata_colormap',load_colormaps()['grey'])
world['delaunay_complex_cells'].set_attribute('intensity_range',(-1,0))
world['delaunay_complex'].set_attribute('coef_3',0.95)
world['delaunay_complex_cells'].set_attribute('x_slice',(50,100))
world['delaunay_complex_cells'].set_attribute('display_colorbar',False)
world['delaunay_complex_cells'].set_attribute('preserve_faces',True)

#cleaned_delaunay_topomesh = deepcopy(draco.delaunay_topomesh)
world.add(cleaned_delaunay_topomesh,'cleaned_delaunay_complex')
world['cleaned_delaunay_complex_cells'].set_attribute('polydata_colormap',load_colormaps()['grey'])
world['cleaned_delaunay_complex_cells'].set_attribute('intensity_range',(-1,0))
world['cleaned_delaunay_complex'].set_attribute('coef_3',0.95)
world['cleaned_delaunay_complex_cells'].set_attribute('x_slice',(50,100))
world['cleaned_delaunay_complex_cells'].set_attribute('display_colorbar',False)
world['cleaned_delaunay_complex_cells'].set_attribute('preserve_faces',True)

import openalea.draco_stem.stem.tissue_mesh_optimization
reload(openalea.draco_stem.stem.tissue_mesh_optimization)
from openalea.draco_stem.stem.tissue_mesh_optimization import optimize_topomesh

from openalea.mesh.property_topomesh_io import save_property_topomesh
save_property_topomesh(draco.triangulation_topomesh,triangulation_file,original_pids=True)

#triangular = ['star','remeshed','straight']
#triangular = ['star','remeshed','projected','flat']
#triangular = ['star','remeshed','projected','exact','flat']

draco.triangulation_topomesh = triangulation_topomesh
triangular= ['star','flat']

triangular = ['star']
image_dual_topomesh = draco.dual_reconstruction(reconstruction_triangulation = triangular, adjacency_complex_degree=3)
#image_dual_topomesh = draco.draco_topomesh(reconstruction_triangulation = triangular)

world.add(image_dual_topomesh ,'dual_reconstuction')

import openalea.draco_stem.stem.tissue_mesh_quality
reload(openalea.draco_stem.stem.tissue_mesh_quality)
from openalea.draco_stem.stem.tissue_mesh_quality import evaluate_topomesh_quality

quality_criteria=["Mesh Complexity","Triangle Area Deviation","Triangle Eccentricity","Vertex Valence","Image Accuracy","Vertex Distance","Cell 2 Adjacency","Cell Convexity","Epidermis Cell Angle","Cell Cliques"]
dual_quality = evaluate_topomesh_quality(image_dual_topomesh,quality_criteria,image=draco.segmented_image,image_cell_vertex=draco.image_cell_vertex,image_labels=draco.image_labels,image_cell_volumes=draco.image_cell_volumes,image_graph=draco.image_graph)

from vplants.meshing.cute_plot import spider_plot
import matplotlib.pyplot as plt

figure = plt.figure(2)
figure.clf()
spider_plot(figure,np.array([dual_quality[c] for c in quality_criteria]),color1=np.array([0.3,0.6,1.]),color2=np.array([1.,0.,0.]),xlabels=quality_criteria,ytargets=0.8 * np.ones_like(quality_criteria,float),n_points=100*len(quality_criteria),linewidth=2,smooth_factor=0.0,spline_order=1)
plt.show(block=False)
raw_input()

triangular_string = ""
for t in triangular:
    triangular_string += t+"_"
topomesh_filename = dirname+"/output_meshes/"+filename+"/"+filename+"_"+triangular_string+"topomesh.ply"

save_ply_property_topomesh(image_dual_topomesh,topomesh_filename,color_faces=True)

from openalea.mesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces
compute_topomesh_property(image_dual_topomesh,'eccentricity',2)
compute_topomesh_vertex_property_from_faces(image_dual_topomesh,'eccentricity',weighting='uniform')


world.add(image_dual_topomesh ,'actual_dual_reconstuction')
raw_input()

# raw_file = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/nuclei_images/olli01_lti6b_150421_sam01_t000/olli01_lti6b_150421_sam01_t000_PIN.inr.gz"
# raw_img = imread(raw_file)
# raw_img = SpatialImage(np.concatenate([raw_img[:,:,35:],np.ones((raw_img.shape[0],raw_img.shape[1],5))],axis=2).astype(np.uint16),resolution=raw_img.resolution)
# world.add(raw_img,"raw_image")


from openalea.draco_stem.stem.tissue_mesh_quality import evaluate_topomesh_quality
quality_criteria=["Mesh Complexity","Triangle Area Deviation","Triangle Eccentricity","Cell Volume Error","Vertex Distance","Cell Convexity","Epidermis Cell Angle","Vertex Valence","Cell 2 Adjacency"]
draco_quality = evaluate_topomesh_quality(image_dual_topomesh,quality_criteria,image=draco.segmented_image,image_cell_vertex=draco.image_cell_vertex,image_labels=draco.image_labels,image_cell_volumes=draco.image_cell_volumes)

from vplants.meshing.cute_plot import spider_plot, histo_plot
import matplotlib.pyplot as plt

figure = plt.figure(0)
figure.clf()
histo_plot(figure,image_dual_topomesh.wisp_property('eccentricity',2).values(),np.array([0.8,0.4,0.2]),xlabel="Triangle Eccentricity",ylabel="Number of triangles (%)",cumul=False,bar=False)
plt.show(block=False)

figure = plt.figure(1)
figure.clf()
histo_plot(figure,image_dual_topomesh.wisp_property('area',2).values(),np.array([0.8,0.4,0.2]),xlabel="Triangle Area",ylabel="Number of triangles (%)",cumul=False,bar=False)
plt.show(block=False)

figure = plt.figure(2)
figure.clf()
spider_plot(figure,np.array([draco_quality[c] for c in quality_criteria]),color1=np.array([0.3,0.6,1.]),color2=np.array([1.,0.,0.]),xlabels=quality_criteria,ytargets=0.8 * np.ones_like(quality_criteria,float),n_points=100*len(quality_criteria),linewidth=2,smooth_factor=0.0,spline_order=1)
plt.show(block=False)
raw_input()

quality_file = open(dirname+"/output_meshes/"+filename+"/"+filename+"_draco_quality.csv","a+")
quality_file.seek(0)
if not quality_file.read(1):
    quality_file.write("Reconstruction Triangulation;")
    for q in quality_criteria:
        quality_file.write(q+";")
    quality_file.write("\n")
quality_file.write(triangular_string[:-1]+";")
for q in quality_criteria:
    quality_file.write(str(draco_quality[q])+";")
quality_file.write("\n")
quality_file.flush()
quality_file.close()

eccentricity_file = dirname+"/output_meshes/"+filename+"/"+filename+"_"+triangular_string+"eccentricities.csv"
np.savetxt(eccentricity_file,image_dual_topomesh.wisp_property('eccentricity',2).values(),delimiter=";")

area_file = dirname+"/output_meshes/"+filename+"/"+filename+"_"+triangular_string+"areas.csv"
np.savetxt(area_file,image_dual_topomesh.wisp_property('area',2).values(),delimiter=";")


topomesh = read_ply_property_topomesh(topomesh_filename)
world.add(topomesh,"reopened_topomesh")


compute_topomesh_property('normal',2)





