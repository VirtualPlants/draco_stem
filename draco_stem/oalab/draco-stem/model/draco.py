import numpy as np
from scipy import ndimage as nd

from openalea.deploy.shared_data import shared_data

from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh

import openalea.draco_stem.draco.dual_reconstruction
reload(openalea.draco_stem.draco.dual_reconstruction)

import openalea.draco_stem.draco.draco
reload(openalea.draco_stem.draco.draco)
from openalea.draco_stem.draco.draco import DracoMesh

from openalea.oalab.colormap.colormap_def import load_colormaps

world.clear()

import vplants.meshing_data
#filename = "p194-t2_imgSeg_SegExp_CellShapeCorr"
#filename = "rs01_wt_t00_seg"
#filename = "segmentation"
filename = "olli01_lti6b_150421_sam01_t000_seg_hmin_2"
dirname = shared_data(vplants.meshing_data)
meshing_dirname =  dirname.parent.parent

import os
if not os.path.exists(dirname+"/output_meshes/"+filename):
    os.makedirs(dirname+"/output_meshes/"+filename)

inputfile = dirname+"/segmented_images/"+filename+".inr.gz"

from openalea.image.serial.all import imread
from openalea.image.spatial_image import SpatialImage
img = imread(inputfile)
img = SpatialImage(np.concatenate([img[:,:,35:],np.ones((img.shape[0],img.shape[1],5))],axis=2).astype(np.uint16),resolution=img.resolution)

cell_vertex_file = dirname+"/output_meshes/"+filename+"/image_cell_vertex.dict"
triangulation_file = dirname+"/output_meshes/"+filename+"/"+filename+"_draco_adjacency_complex.pkl"

world.add(img,filename,colormap='glasbey',alphamap='constant',bg_id=1)

#draco = DracoMesh(image_file=inputfile, image_cell_vertex_file=cell_vertex_file, triangulation_file=triangulation_file)
#draco = DracoMesh(image_file=inputfile, image_cell_vertex_file=cell_vertex_file)
draco = DracoMesh(image=img)

world.add(draco.segmented_image,filename,colormap='glasbey',alphamap='constant',bg_id=1)
world.add(draco.point_topomesh,'image_cells')
#world.add(draco.layer_edge_topomesh['L1'],'L1_adjacency')
#world.add(draco.image_cell_vertex_topomesh,'image_cell_vertex')
raw_input()

#draco.delaunay_adjacency_complex()
#draco.layer_adjacency_complex('L1')
draco.construct_adjacency_complex()
#draco.adjacency_complex_optimization(n_iterations=1)

world.add(draco.triangulation_topomesh,'cell_adjacency_complex')
world['cell_adjacency_complex_cells'].set_attribute('intensity_range',(-1,0))
world['cell_adjacency_complex_cells'].set_attribute('polydata_colormap',load_colormaps()['grey'])
world['cell_adjacency_complex'].set_attribute('coef_3',0.95)

from scipy.cluster.vq import vq
from copy import deepcopy
from openalea.container import array_dict
from openalea.mesh import PropertyTopomesh
from openalea.mesh.property_topomesh_creation import edge_topomesh, triangle_topomesh
from openalea.draco_stem.draco.adjacency_complex_optimization import triangles_from_adjacency_edges
from openalea.mesh.utils.array_tools import array_unique
from openalea.mesh.property_topomesh_analysis import compute_topomesh_property
from openalea.mesh.utils.intersection_tools import intersecting_triangle

triangulation_topomesh = deepcopy(draco.triangulation_topomesh)
cell_layer = deepcopy(draco.point_topomesh.wisp_property('layer',0))
positions = deepcopy(draco.point_topomesh.wisp_property('barycenter',0))
image_graph = deepcopy(draco.image_graph)
img_wall_surfaces = deepcopy(draco.image_wall_surfaces)
img_volumes = deepcopy(draco.image_cell_volumes)
image_cell_vertex = deepcopy(draco.image_cell_vertex)

L2_edges = np.array([[(c,n) for n in triangulation_topomesh.region_neighbors(0,c) if n>1 and cell_layer[n]==2] for c in triangulation_topomesh.wisps(0) if cell_layer[c]==2])
L2_edges = np.concatenate([e for e in L2_edges if len(e)>0])
L2_edges = L2_edges[L2_edges[:,1]>L2_edges[:,0]]
L2_edge_topomesh = edge_topomesh(L2_edges, positions)
#world.add(L2_edge_topomesh,"L2_neighborhood")

L2_L3_edges = np.array([[(c,n) for n in image_graph.vertices() if n>1 and cell_layer[n] in [0] and np.linalg.norm(positions[c]-positions[n])<20. ] for c in cell_layer.keys() if cell_layer[c] in [2]])
L2_L3_edges = array_unique(np.sort(np.concatenate([e for e in L2_L3_edges if len(e)>0])))
L2_L3_edge_topomesh = edge_topomesh(L2_L3_edges, positions)
world.add(L2_L3_edge_topomesh,"L2_L3_neighborhood")

L3_edges = np.array([[(c,n) for n in image_graph.neighbors(c) if n>1 and cell_layer[n] in [0]] for c in cell_layer.keys() if cell_layer[c] in [0]])
L3_edges = array_unique(np.sort(np.concatenate([e for e in L3_edges if len(e)>0])))
L3_edge_topomesh = edge_topomesh(L3_edges, positions)
world.add(L3_edge_topomesh,"L3_neighborhood")

L3_additional_edges = np.array([[(c,n) for n in np.unique(np.array(image_cell_vertex.keys())[np.where(np.array(image_cell_vertex.keys())==c)[0]])  if n>1 and n!=c and (n not in image_graph.neighbors(c)) and (cell_layer[n] in [0])] for c in cell_layer.keys() if cell_layer[c] in [0]])
L3_additional_edges = array_unique(np.sort(np.concatenate([e for e in L3_additional_edges if len(e)>0])))

layer_triangle_topomesh = triangle_topomesh(triangles_from_adjacency_edges(array_unique(np.concatenate([L2_edges,L2_L3_edges,L3_edges,L3_additional_edges]))),positions)
world.add(layer_triangle_topomesh,'L2_L3_neighborhood')
raw_input()

#omega_criteria = {'distance':1.0,'wall_surface':2.0,'layer':500.0,'clique':10.0}
omega_criteria = {'distance':1.0,'wall_surface':2.0,'layer':50.0,}

compute_topomesh_property(layer_triangle_topomesh,'length',1)
compute_topomesh_property(layer_triangle_topomesh,'borders',2)
compute_topomesh_property(layer_triangle_topomesh,'perimeter',2)

if omega_criteria.has_key('wall_surface'):
    #img_wall_surfaces = kwargs.get('wall_surfaces',None)
    #img_volumes = kwargs.get('cell_volumes',dict(zip(positions.keys(),np.ones_like(positions.keys()))))
    #assert layer_triangle_topomesh.has_wisp_property('wall_surface',1) or img_wall_surfaces is not None
    if not layer_triangle_topomesh.has_wisp_property('wall_surface',1):
        L2_L3_triangle_edge_vertices = np.array([np.sort([list(layer_triangle_topomesh.borders(1,e)) for e in layer_triangle_topomesh.borders(2,t)]) for t in layer_triangle_topomesh.wisps(2)])
        L2_L3_triangle_edge_wall_surface = np.array([[-1. if tuple(e) not in img_wall_surfaces.keys() else img_wall_surfaces[tuple(e)]/np.power(img_volumes.values(e).mean(),2./3.) for e in t] for t in L2_L3_triangle_edge_vertices])
        layer_triangle_topomesh.update_wisp_property('wall_surface',2,array_dict(L2_L3_triangle_edge_wall_surface.min(axis=1),list(layer_triangle_topomesh.wisps(2))))
        #layer_triangle_topomesh = layer_triangle_topomesh

triangle_weights = np.zeros(layer_triangle_topomesh.nb_wisps(2))
if omega_criteria.has_key('distance'):
    triangle_weights += omega_criteria['distance']*np.exp(-np.power(layer_triangle_topomesh.wisp_property('length',1).values(layer_triangle_topomesh.wisp_property('borders',2).values()).max(axis=1)/15.0,1))
if omega_criteria.has_key('wall_surface'):
    triangle_weights += omega_criteria['wall_surface']*layer_triangle_topomesh.wisp_property('wall_surface',2).values() 
if omega_criteria.has_key('layer'):
    triangle_weights += omega_criteria['layer']*np.sum(cell_layer.values(layer_triangle_topomesh.wisp_property('vertices',2).values())==2,axis=1)
triangle_weights = array_dict(triangle_weights,list(layer_triangle_topomesh.wisps(2)))

triangle_neighbor_edges = [np.concatenate([list(set(layer_triangle_topomesh.region_neighbors(1,e)))+[e] for e in layer_triangle_topomesh.borders(2,t)]) for t in layer_triangle_topomesh.wisps(2)]
triangle_neighbor_edge_triangles = [np.concatenate([list(layer_triangle_topomesh.regions(1,n_e)) for n_e in n_edges]) for n_edges in triangle_neighbor_edges]
triangle_tetrahedra_triangles = [np.unique(t)[nd.sum(np.ones_like(t),t,index=np.unique(t))>5] for t in triangle_neighbor_edge_triangles]

if omega_criteria.has_key('clique'):
    triangle_neighbor_weights = array_dict([triangle_weights.values(t).min() - omega_criteria['clique']*(len(t)-4) for t in triangle_tetrahedra_triangles],list(layer_triangle_topomesh.wisps(2)))
else:
    triangle_neighbor_weights = array_dict([triangle_weights.values(t).min() for t in triangle_tetrahedra_triangles],list(layer_triangle_topomesh.wisps(2)))
triangle_tetrahedra_triangles = array_dict(triangle_tetrahedra_triangles,list(layer_triangle_topomesh.wisps(2)))

tetrahedrization_triangles = np.array(list(layer_triangle_topomesh.wisps(2)))[np.array(map(len,triangle_tetrahedra_triangles))>=4]
    
constructed_triangulation_topomesh = PropertyTopomesh(3)

compute_topomesh_property(layer_triangle_topomesh,'vertices',2)
layer_triangle_L2_cells = (cell_layer.values(layer_triangle_topomesh.wisp_property('vertices',2).values())==2).sum(axis=1)
L2_triangles = np.array(list(layer_triangle_topomesh.wisps(2)))[layer_triangle_L2_cells==3]

free_triangles = list(L2_triangles[np.argsort(-triangle_neighbor_weights.values(L2_triangles))])
prev_n = 0

while len(free_triangles) > 0:
    fid_to_add = free_triangles.pop(0)
    print "--> Triangle",list(layer_triangle_topomesh.borders(2,fid_to_add))," : ",triangle_neighbor_weights[fid_to_add]
    
    triangle_vertex_edges = np.concatenate([list(set(layer_triangle_topomesh.regions(0,c)).difference(set(layer_triangle_topomesh.borders(2,fid_to_add)))) for c in layer_triangle_topomesh.borders(2,fid_to_add,2)])
    triangle_vertex_edge_vertices = np.concatenate([c*np.ones(layer_triangle_topomesh.nb_regions(0,c)-2) for c in layer_triangle_topomesh.borders(2,fid_to_add,2)])
    triangle_vertex_edge_neighbor_vertices = np.array([list(set(layer_triangle_topomesh.borders(1,e)).difference({v}))[0] for e,v in zip(triangle_vertex_edges,triangle_vertex_edge_vertices)])

    candidate_tetra_vertices = np.unique(triangle_vertex_edge_neighbor_vertices)[nd.sum(np.ones_like(triangle_vertex_edge_neighbor_vertices),triangle_vertex_edge_neighbor_vertices,index=np.unique(triangle_vertex_edge_neighbor_vertices))==3]
    candidate_tetra_edges = np.array([triangle_vertex_edges[triangle_vertex_edge_neighbor_vertices==c] for c in candidate_tetra_vertices])
    
    candidate_tetra_edge_triangles = [np.concatenate([list(set(layer_triangle_topomesh.regions(1,e)).difference({fid_to_add})) for e in candidate_edges]) for candidate_edges in candidate_tetra_edges]
    candidate_tetra_triangles = np.array([np.concatenate([[fid_to_add],np.unique(t)[nd.sum(np.ones_like(t),t,index=np.unique(t))==2]]) for t in candidate_tetra_edge_triangles])
    
    if len(candidate_tetra_triangles)>0:
        candidate_tetra_free_triangles = np.array([np.sum([t in free_triangles for t in tetra_triangles]) for tetra_triangles in candidate_tetra_triangles])
        candidate_tetra_triangle_weights = triangle_weights.values(candidate_tetra_triangles[:,1:]).min(axis=1)
        
        if (candidate_tetra_free_triangles == candidate_tetra_free_triangles.max()).sum() == 1:
            sorted_candidate_tetra_triangles = candidate_tetra_triangles[np.argsort(-candidate_tetra_free_triangles)]
        else:
            sorted_candidate_tetra_triangles = candidate_tetra_triangles[np.argsort(-candidate_tetra_triangle_weights)]
        
        for tetra_triangles in sorted_candidate_tetra_triangles:
            if np.all(np.array([0 if not constructed_triangulation_topomesh.has_wisp(2,t) else constructed_triangulation_topomesh.nb_regions(2,t) for t in tetra_triangles])<2):
                tetra_vertices = np.unique([list(layer_triangle_topomesh.borders(2,t,2)) for t in tetra_triangles])
                tetra_edges = np.unique([list(layer_triangle_topomesh.borders(2,t)) for t in tetra_triangles])
                if constructed_triangulation_topomesh.nb_wisps(3)!=1 or vq(np.sort([tetra_vertices]),np.sort([list(constructed_triangulation_topomesh.borders(3,t,3)) for t in constructed_triangulation_topomesh.wisps(3)]))[1][0]>0:
                    #if len(np.unique(cell_layer.values(tetra_vertices)))==2:
                    if True:
                        #tetra_triangle_tetras = np.unique([list(constructed_triangulation_topomesh.regions(2,t)) for t in tetra_triangles if constructed_triangulation_topomesh.has_wisp(2,t)])
                        tetra_triangle_tetras = np.array(list(constructed_triangulation_topomesh.wisps(3)))
                        if len(tetra_triangle_tetras)>0:
                            tetra_triangle_tetra_edges = np.unique([list(constructed_triangulation_topomesh.borders(3,t,2)) for t in tetra_triangle_tetras])
                            tetra_triangle_points = positions.values(np.array([list(layer_triangle_topomesh.borders(2,t,2)) for t in tetra_triangles]))
                            tetra_triangle_tetra_edge_points = positions.values(np.array([list(layer_triangle_topomesh.borders(1,e)) for e in tetra_triangle_tetra_edges]))
                            tetra_triangle_intersection = np.ravel([intersecting_triangle(edge_points,tetra_triangle_points) for edge_points in tetra_triangle_tetra_edge_points])
                        
                            tetra_triangle_edges = np.unique([list(layer_triangle_topomesh.borders(2,t)) for t in tetra_triangles])
                            tetra_triangle_tetra_triangles = np.unique([list(constructed_triangulation_topomesh.borders(3,t)) for t in tetra_triangle_tetras])
                            tetra_triangle_edge_points = positions.values(np.array([list(layer_triangle_topomesh.borders(1,e)) for e in tetra_triangle_edges]))
                            tetra_triangle_tetra_triangle_points = positions.values(np.array([list(layer_triangle_topomesh.borders(2,t,2)) for t in tetra_triangle_tetra_triangles]))
                            tetra_edge_intersection = np.ravel([intersecting_triangle(edge_points,tetra_triangle_tetra_triangle_points) for edge_points in tetra_triangle_edge_points])
                            tetra_triangle_intersection = np.concatenate([tetra_triangle_intersection,tetra_edge_intersection])
                        else:
                            tetra_triangle_intersection = [False]
                        if not np.any(tetra_triangle_intersection):
                            tid = constructed_triangulation_topomesh.add_wisp(3)
                            print "  --> Tetrahedron",tid,tetra_vertices," : ", triangle_weights.values(tetra_triangles[1:]).min()
                            for c in tetra_vertices:
                                if not constructed_triangulation_topomesh.has_wisp(0,c):
                                    constructed_triangulation_topomesh.add_wisp(0,c)
                            for e in tetra_edges:
                                if not constructed_triangulation_topomesh.has_wisp(1,e):
                                    constructed_triangulation_topomesh.add_wisp(1,e)
                                    for c in layer_triangle_topomesh.borders(1,e):
                                        constructed_triangulation_topomesh.link(1,e,c)
                            for t in tetra_triangles:
                                if not constructed_triangulation_topomesh.has_wisp(2,t):
                                    constructed_triangulation_topomesh.add_wisp(2,t)
                                    for e in layer_triangle_topomesh.borders(2,t):
                                        constructed_triangulation_topomesh.link(2,t,e)
                                constructed_triangulation_topomesh.link(3,tid,t)
                                
                                #if constructed_triangulation_topomesh.nb_regions(2,t)<2 and len(np.unique(cell_layer.values(list(constructed_triangulation_topomesh.borders(2,t,2)))))==2:
                                if constructed_triangulation_topomesh.nb_regions(2,t)<2:
                                    if not t in free_triangles:
                                        free_triangles.append(t)
                                    
                                    triangle_future_tetra_triangles = list(set(triangle_tetrahedra_triangles[t]).difference(set(constructed_triangulation_topomesh.wisps(2)).difference(set(free_triangles))))
                                    
                                    if omega_criteria.has_key('clique'):
                                        triangle_neighbor_weights[t] = np.min(triangle_weights.values(triangle_future_tetra_triangles)) - omega_criteria['clique']*(len(triangle_future_tetra_triangles)-4)
                                    else:
                                        triangle_neighbor_weights[t] = np.min(triangle_weights.values(triangle_future_tetra_triangles))
    
    if constructed_triangulation_topomesh.nb_wisps(3)>0:
        free_triangles = list(np.array(free_triangles)[np.argsort(-triangle_neighbor_weights.values(free_triangles))])
        constructed_triangulation_topomesh.update_wisp_property('barycenter',0,array_dict(positions.values(list(constructed_triangulation_topomesh.wisps(0))),list(constructed_triangulation_topomesh.wisps(0))))
    if constructed_triangulation_topomesh.nb_wisps(3)%100 == 1 and constructed_triangulation_topomesh.nb_wisps(3)!=prev_n:
        prev_n = constructed_triangulation_topomesh.nb_wisps(3)
        world.add(constructed_triangulation_topomesh,'triangulation_topomesh')
        #raw_input()


import openalea.draco_stem.stem.tissue_mesh_optimization
reload(openalea.draco_stem.stem.tissue_mesh_optimization)
from openalea.draco_stem.stem.tissue_mesh_optimization import optimize_topomesh

from openalea.mesh.property_topomesh_io import save_property_topomesh
save_property_topomesh(draco.triangulation_topomesh,triangulation_file,original_pids=True)

#triangular = ['star','remeshed','projected','straight']
triangular = ['star','remeshed','projected','flat']
#triangular = ['star','remeshed','projected','exact','flat']
#triangular= ['star','flat']
image_dual_topomesh = draco.dual_reconstruction(reconstruction_triangulation = triangular, adjacency_complex_degree=3)
#image_dual_topomesh = draco.draco_topomesh(reconstruction_triangulation = triangular)

triangular_string = ""
for t in triangular:
    triangular_string += t+"_"
topomesh_filename = dirname+"/output_meshes/"+filename+"/"+filename+"_L1"+triangular_string+"_topomesh.ply"
save_ply_property_topomesh(image_dual_topomesh,topomesh_filename,color_faces=True)

world.add(image_dual_topomesh ,'dual_reconstuction')
raw_input()

raw_file = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/nuclei_images/olli01_lti6b_150421_sam01_t000/olli01_lti6b_150421_sam01_t000_PIN.inr.gz"
raw_img = imread(raw_file)
raw_img = SpatialImage(np.concatenate([raw_img[:,:,35:],np.ones((raw_img.shape[0],raw_img.shape[1],5))],axis=2).astype(np.uint16),resolution=raw_img.resolution)
world.add(raw_img,"raw_image")


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








