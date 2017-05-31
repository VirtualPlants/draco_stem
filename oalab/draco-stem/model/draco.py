import numpy as np
from scipy import ndimage as nd

from openalea.deploy.shared_data import shared_data

import openalea.mesh.property_topomesh_io
reload(openalea.mesh.property_topomesh_io)
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh

from openalea.image.serial.all import imread, imsave
from openalea.image.spatial_image import SpatialImage

import openalea.draco_stem.draco.dual_reconstruction
reload(openalea.draco_stem.draco.dual_reconstruction)

import openalea.draco_stem.draco.draco
reload(openalea.draco_stem.draco.draco)
from openalea.draco_stem.draco.draco import DracoMesh

from openalea.oalab.colormap.colormap_def import load_colormaps

from copy import deepcopy
from openalea.draco_stem.example_image import sphere_tissue_image

world.clear()

#import vplants.meshing_data
#filename = "p194-t3_imgSeg_SegExp_CellShapeCorr"
#filename = "rs01_wt_t00_seg"
#filename = "segmentation"
#filename = "olli01_lti6b_150421_sam01_t000_seg_hmin_2"
#filename = "sphere_cells"
filename = "yr02_t132_seg"


dirname = '/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/'
#dirname = shared_data(vplants.meshing_data)
#meshing_dirname =  dirname.parent.parent

import os
if not os.path.exists(dirname+"/output_meshes/"+filename):
    os.makedirs(dirname+"/output_meshes/"+filename)

inputfile = dirname+"/segmented_images/"+filename+".inr.gz"


img = imread(inputfile)
#img[img==0]=1
#img = SpatialImage(np.concatenate([img[:,:,35:],np.ones((img.shape[0],img.shape[1],5))],axis=2).astype(np.uint16),voxelsize=img.voxelsize)

cell_vertex_file = dirname+"/output_meshes/"+filename+"/image_cell_vertex.dict"
triangulation_file = dirname+"/output_meshes/"+filename+"/"+filename+"_draco_adjacency_complex.pkl"

from openalea.container import array_dict

world.add(img,"segmented_image",colormap='glasbey',alphamap='constant',bg_id=1,alpha=1.0)


#from openalea.cgal_meshing.idra import IdraMesh
#topomesh = IdraMesh(img,mesh_fineness=1.0).idra_topomesh()
#world.add(topomesh,'IDRA_mesh')

#idra_file = dirname+"/output_meshes/"+filename+"/"+filename+"_IDRA_mesh.ply"
#save_ply_property_topomesh(topomesh,idra_file,color_faces=True,colormap=load_colormaps()['glasbey'])

#from openalea.draco_stem.stem.tissue_mesh_optimization import optimize_topomesh

#draco = DracoMesh(image=img, image_cell_vertex_file=cell_vertex_file, triangulation_file=triangulation_file)
draco = DracoMesh(image_file=inputfile, image_cell_vertex_file=cell_vertex_file, triangulation_file=triangulation_file)
#draco = DracoMesh(image=img)
image_cell_vertex = draco.image_cell_vertex

#optimized_topomesh = optimize_topomesh(topomesh,omega_forces={'regularization':0.00,'neighborhood':0.0,'laplacian':1.0,'planarization':0.27,'epidermis_planarization':0.07,'convexity':0.02},omega_regularization_max=0.01,edge_flip=True,cell_vertex_motion=True,image_cell_vertex=image_cell_vertex)
#optimized_topomesh = optimize_topomesh(optimized_topomesh,omega_forces={'taubin_smoothing':0.65},cell_vertex_motion=True,image_cell_vertex=image_cell_vertex)
#world.add(optimized_topomesh,'IDRA_STEM_mesh')

#idra_stem_file = dirname+"/output_meshes/"+filename+"/"+filename+"_IDRA_STEM_mesh.ply"
#save_ply_property_topomesh(optimized_topomesh,idra_stem_file,color_faces=True,colormap=load_colormaps()['glasbey'])


#world.add(draco.segmented_image-(draco.segmented_image==1),'segmented_image',colormap='glasbey',alphamap='constant',bg_id=0)
#world.add(draco.point_topomesh,'image_cells')
#world['image_cells_vertices'].set_attribute('point_radius',img.max())
#world.add(draco.layer_edge_topomesh['L1'],'L1_adjacency')
#world.add(draco.image_cell_vertex_topomesh,'image_cell_vertex')

#draco.delaunay_adjacency_complex(surface_cleaning_criteria = [])
#draco.delaunay_adjacency_complex(surface_cleaning_criteria = ['surface','sliver','distance'])

#draco.adjacency_complex_optimization(n_iterations=2)


# from copy import deepcopy
# triangulation_topomesh = deepcopy(draco.triangulation_topomesh)
# world.add(triangulation_topomesh,'cell_adjacency_complex')
# world['cell_adjacency_complex_cells'].set_attribute('polydata_colormap',load_colormaps()['grey'])
# world['cell_adjacency_complex_cells'].set_attribute('intensity_range',(-1,0))
# world['cell_adjacency_complex'].set_attribute('coef_3',0.95)#
# #world['cell_adjacency_complex_cells'].set_attribute('x_slice',(50,100))
# world['cell_adjacency_complex_cells'].set_attribute('display_colorbar',False)

#triangular = ['star','remeshed','realistic','projected']
#triangular = ['star','remeshed','realistic','projected','flat']
triangular = ['star','remeshed']
image_dual_topomesh = draco.dual_reconstruction(reconstruction_triangulation = triangular, adjacency_complex_degree=3, maximal_edge_length=5.1)
#image_dual_topomesh = draco.draco_topomesh(reconstruction_triangulation = triangular)

from openalea.cellcomplex.property_topomesh.property_topomesh_optimization import property_topomesh_vertices_deformation
#property_topomesh_vertices_deformation(image_dual_topomesh,iterations=15)


world.add(image_dual_topomesh ,'dual_reconstuction')


draco_file = dirname+"/output_meshes/"+filename+"/"+filename+"_DRACO_mesh.ply"
save_ply_property_topomesh(image_dual_topomesh,draco_file,color_faces=True,colormap=load_colormaps()['glasbey'])

optimized_dual_topomesh = optimize_topomesh(image_dual_topomesh,omega_forces={'regularization':0.00,'neighborhood':0.65,'taubin_smoothing':0.65,'laplacian':0.7,'planarization':0.27,'epidermis_planarization':0.05,'convexity':0.06},omega_regularization_max=0.01,edge_flip=True,cell_vertex_motion=True,image_cell_vertex=image_cell_vertex)
optimized_dual_topomesh = optimize_topomesh(optimized_dual_topomesh,omega_forces={'taubin_smoothing':0.65},cell_vertex_motion=True,image_cell_vertex=image_cell_vertex)
world.add(optimized_dual_topomesh ,'dual_reconstuction')

draco_stem_file = dirname+"/output_meshes/"+filename+"/"+filename+"_DRACO_STEM_mesh.ply"
save_ply_property_topomesh(optimized_dual_topomesh,draco_stem_file,color_faces=True,colormap=load_colormaps()['glasbey'])


from openalea.mesh.property_topomesh_extraction import epidermis_topomesh



L1_topomesh = epidermis_topomesh(image_dual_topomesh)

world.add(L1_topomesh ,'dual_reconstuction')


center = np.array([size/2,size/2,size/2],float)*np.array(img.voxelsize)
    
positions = L1_topomesh.wisp_property('barycenter',0)
for p in L1_topomesh.wisps(0):
    positions[p] = center + (size/3.)*np.array(img.voxelsize)*(positions[p]-center)/np.linalg.norm(positions[p]-center)
    
L1_topomesh.update_wisp_property('barycenter',0,positions)
world.add(L1_topomesh ,'dual_reconstuction')


triangular_string = ""
for t in triangular:
    triangular_string += t+"_"
topomesh_filename = dirname+"/output_meshes/"+filename+"/"+filename+"_voronoi_L1_"+triangular_string+"topomesh.ply"

save_ply_property_topomesh(L1_topomesh,topomesh_filename,color_faces=True)



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
triangular= ['star','remeshed','projected','exact']
image_dual_topomesh = draco.dual_reconstruction(reconstruction_triangulation = triangular, adjacency_complex_degree=3)
#image_dual_topomesh = draco.draco_topomesh(reconstruction_triangulation = triangular)
world.add(image_dual_topomesh ,'L1_L2_dual_reconstuction')
raw_input()

from copy import deepcopy
triangulation_topomesh = deepcopy(draco.triangulation_topomesh)
world.add(triangulation_topomesh,'cell_adjacency_complex')
world['cell_adjacency_complex_cells'].set_attribute('polydata_colormap',load_colormaps()['grey'])
world['cell_adjacency_complex_cells'].set_attribute('intensity_range',(-1,0))
world['cell_adjacency_complex'].set_attribute('coef_3',0.95)#
#world['cell_adjacency_complex_cells'].set_attribute('x_slice',(50,100))
world['cell_adjacency_complex_cells'].set_attribute('display_colorbar',False)
#world['cell_adjacency_complex_cells'].set_attribute('preserve_faces',True)
#world['cell_adjacency_complex'].set_attribute('display_0',True)
#world['cell_adjacency_complex_vertices'].set_attribute('point_radius',draco.triangulation_topomesh.nb_wisps(0))
#world['cell_adjacency_complex_vertices'].set_attribute('x_slice',(50,100))
#world['cell_adjacency_complex_vertices'].set_attribute('display_colorbar',False)

#delaunay_topomesh = deepcopy(draco.delaunay_topomesh)
world.add(delaunay_topomesh,'delaunay_complex')
world['delaunay_complex_cells'].set_attribute('polydata_colormap',load_colormaps()['grey'])
world['delaunay_complex_cells'].set_attribute('intensity_range',(-1,0))
world['delaunay_complex'].set_attribute('coef_3',0.95)
world['delaunay_complex_cells'].set_attribute('x_slice',(50,100))
world['delaunay_complex_cells'].set_attribute('display_colorbar',False)
world['delaunay_complex_cells'].set_attribute('preserve_faces',True)

cleaned_delaunay_topomesh = deepcopy(draco.delaunay_topomesh)
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

triangular = ['star','remeshed','projected','exact','flat']
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
# raw_img = SpatialImage(np.concatenate([raw_img[:,:,35:],np.ones((raw_img.shape[0],raw_img.shape[1],5))],axis=2).astype(np.uint16),voxelsize=raw_img.voxelsize)
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





