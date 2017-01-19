import numpy as np
from scipy import ndimage as nd

from openalea.deploy.shared_data import shared_data

from openalea.container import array_dict
from openalea.mesh import TriangularMesh

import openalea.mesh.property_topomesh_io
reload(openalea.mesh.property_topomesh_io)
from openalea.mesh.property_topomesh_io import save_ply_property_topomesh, read_ply_property_topomesh, save_property_topomesh

from openalea.image.serial.all import imread
from openalea.image.spatial_image import SpatialImage

from openalea.mesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces, compute_topomesh_cell_property_from_faces
from openalea.mesh.property_topomesh_creation import vertex_topomesh, triangle_topomesh
from openalea.mesh.utils.delaunay_tools import delaunay_triangulation

import  openalea.mesh.property_topomesh_extraction
reload(openalea.mesh.property_topomesh_extraction)
from openalea.mesh.property_topomesh_extraction import clean_topomesh

from openalea.oalab.colormap.colormap_def import load_colormaps

import pickle
from copy import deepcopy

world.clear() 

#filename = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/output_meshes/p194-t5_imgSeg_SegExp_CellShapeCorr/p194-t5_imgSeg_SegExp_CellShapeCorr_triangulated_L1_L2_star_remeshed_projected_flat_temporal_properties_topomesh.pkl"
filename = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/output_meshes/p194-t5_imgSeg_SegExp_CellShapeCorr/p194-t5_imgSeg_SegExp_CellShapeCorr_triangulated_L1_star_remeshed_straight_temporal_properties_topomesh.pkl"
topomesh = pickle.load(open(filename,'r'))

topomesh = clean_topomesh(topomesh, clean_properties=True)

compute_topomesh_property(topomesh,'length',degree=1)
compute_topomesh_property(topomesh,'area',degree=2)
compute_topomesh_property(topomesh,'barycenter',degree=2)
compute_topomesh_property(topomesh,'normal',2,normal_method='orientation')
#compute_topomesh_property(topomesh,'normal',0)
compute_topomesh_vertex_property_from_faces(topomesh,'normal',weighting='area',adjacency_sigma=1.2,neighborhood=3)
#topomesh.update_wisp_property('normal',0,topomesh.wisp_property('barycenter',0).values()/ np.linalg.norm(topomesh.wisp_property('barycenter',0).values(),axis=1)[:,np.newaxis],list(topomesh.wisps(0)))
compute_topomesh_property(topomesh,'mean_curvature',2)
compute_topomesh_cell_property_from_faces(topomesh,'principal_curvature_tensor')

world.add(topomesh,'topomesh')


def topomesh_to_dataframe(topomesh, degree=3, properties=None):
    import pandas as pd
    
    if properties is None:
        properties = topomesh.wisp_properties(degree).keys()
    
    dataframe = pd.DataFrame()
    dataframe['id'] = np.array(list(topomesh.wisps(degree)))

    for property_name in properties:
        if np.array(topomesh.wisp_property(property_name,degree).values()[0]).ndim == 0:
            print "  --> Adding column ",property_name
            dataframe[property_name] = topomesh.wisp_property(property_name,degree).values(dataframe['id'].values)
    
    dataframe = dataframe.set_index('id')
    dataframe.index.name = None
        
    return dataframe

df = topomesh_to_dataframe(topomesh,3)
world.add(df,"topomesh_cell_data")


world.add(topomesh,"topomesh")

import vplants.meshing_data
filename = "p194-t5_imgSeg_SegExp_CellShapeCorr"
dirname = shared_data(vplants.meshing_data)
meshing_dirname =  dirname.parent.parent


inputfile = dirname+"/segmented_images/"+filename+".inr.gz"
img = imread(inputfile)
world.add(img,"segmented_image",colormap='glasbey',alphamap='constant',bg_id=1)

topomesh_filename = dirname+"/output_meshes/"+filename+"/"+filename+"_star_remeshed_projected_exact_flat_topomesh.ply"

filename = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/output_meshes/p194-t2_imgSeg_SegExp_CellShapeCorr/p194-t2_imgSeg_SegExp_CellShapeCorr_star_remeshed_projected_exact_flat_topomesh.ply"


world.clear() 

#filename = "/Users/gcerutti/Downloads/PIN1-GFP-CLV3-CH-MS-E1-LD-SAM5-mesh.ply"
#filename = "/Users/gcerutti/Downloads/PIN1-GFP-CLV3-CH-MS-E1-SD-SAM5-mesh.ply"
topomesh = read_ply_property_topomesh(filename,verbose=False)

world.add(topomesh,"topomesh")
#world['topomesh_faces'].set_attribute('intesity_range',(-1,0))



import pandas

curvatures = {}

for i,days in enumerate(['LD','SD']):
                         
    curvatures[days] = []
                         
    for n_sam in xrange(7):
        csv_filename = "/Users/gcerutti/Downloads/LD-SD-CLV3/PIN1-GFP-CLV3-CH-MS-E1-"+days+"-SAM"+str(n_sam)+"-CI-CLV3.csv"
        try:                     
            df = pandas.read_csv(csv_filename)
        except:
            print csv_filename,": File not found"
        else:
            filename = days+"-SAM"+str(n_sam)
            
            centers = np.transpose([df.Center_X.values,df.Center_Y.values,df.Center_Z.values])[1:-4]
            cell_barycenters = dict(zip(df.Label.values[1:-4].astype(int),centers))
            
            cell_signals = dict(zip(df.Label.values[1:-4].astype(int),df.Value.values[1:-4]))
            

            cell_flat_barycenters = deepcopy(cell_barycenters)
            for c in cell_barycenters.keys():
                cell_flat_barycenters[c][2] = 0.
    
            triangles = np.array(cell_barycenters.keys())[delaunay_triangulation(np.array([cell_flat_barycenters[c] for c in cell_barycenters.keys()]))]
            cell_topomesh = triangle_topomesh(triangles, cell_barycenters)
            
            cell_topomesh.update_wisp_property('signal',0,cell_signals)
            
            maximal_length = 15.
            
            compute_topomesh_property(cell_topomesh,'length',1)
            compute_topomesh_property(cell_topomesh,'triangles',1)
            
            boundary_edges = np.array(map(len,cell_topomesh.wisp_property('triangles',1).values()))==1
            distant_edges = cell_topomesh.wisp_property('length',1).values() > maximal_length
            edges_to_remove = np.array(list(cell_topomesh.wisps(1)))[boundary_edges & distant_edges]
            
            while len(edges_to_remove) > 0:
                triangles_to_remove = np.concatenate(cell_topomesh.wisp_property('triangles',1).values(edges_to_remove))
                for t in triangles_to_remove:
                    cell_topomesh.remove_wisp(2,t)
                
                clean_topomesh(cell_topomesh)
                
                compute_topomesh_property(cell_topomesh,'triangles',1)
            
                boundary_edges = np.array(map(len,cell_topomesh.wisp_property('triangles',1).values()))==1
                distant_edges = cell_topomesh.wisp_property('length',1).values() > maximal_length
                edges_to_remove = np.array(list(cell_topomesh.wisps(1)))[boundary_edges & distant_edges]
            
            compute_topomesh_property(cell_topomesh,'normal',2,normal_method='orientation')
            compute_topomesh_vertex_property_from_faces(cell_topomesh,'normal',adjacency_sigma=1.2,neighborhood=3)
            compute_topomesh_property(cell_topomesh,'mean_curvature',2)
            compute_topomesh_vertex_property_from_faces(cell_topomesh,'mean_curvature',adjacency_sigma=1.2,neighborhood=3)
            
            # world.add(cell_topomesh,filename+'cell_triangulation')
            # world[filename+'cell_triangulation'].set_attribute('property_degree_2',0)
            # world[filename+'cell_triangulation'].set_attribute('display_3',False)
            # world[filename+'cell_triangulation'].set_attribute('property_name_2','mean_curvature')
            # world[filename+'cell_triangulation'].set_attribute('display_2',True)
            # world[filename+'cell_triangulation_faces'].set_attribute('intensity_range',(-0.05,0.05))
            # world[filename+'cell_triangulation_faces'].set_attribute('polydata_colormap',load_colormaps()['curvature'])
            # world[filename+'cell_triangulation_faces'].set_attribute('display_colorbar',True)
            # world[filename+'cell_triangulation'].set_attribute('property_name_0','normal')
            # world[filename+'cell_triangulation'].set_attribute('display_0',True)
            # world[filename+'cell_triangulation_vertices'].set_attribute('point_radius',10)
            # world[filename+'cell_triangulation_vertices'].set_attribute('display_colorbar',False)
            
            print cell_topomesh.wisp_property('mean_curvature',0).values().mean()
            curvatures[days] += list(cell_topomesh.wisp_property('mean_curvature',0).values())
            #raw_input()

import matplotlib.pyplot as plt
from vplants.meshing.cute_plot import histo_plot, simple_plot

figure = plt.figure(0)
figure.clf()

colors = {}
colors['SD'] = np.array([0.1,0.5,0.2])
colors['LD'] = np.array([0.5,0.7,0.0])

    
for i,days in enumerate(['SD','LD']):
    
    simple_plot(figure,np.array([np.mean(curvatures[days]),np.mean(curvatures[days])+1e-7]),np.array([0,100]),colors[days],alpha=0.5,linewidth=1,linked=True)
    histo_plot(figure,np.array(curvatures[days]),colors[days],"Mean Curvature","Surface Elements (%)",bar=False,n_points=200,spline_order=4,smooth_factor=5,label=days)
    
    #plt.hist(curvatures[days],cumulative=False,bins=40,range=(0,0.04),histtype='step',lw=3,color=colors[days])
figure.gca().set_xlim(0,0.04)
figure.gca().set_xticklabels(figure.gca().get_xticks())
figure.gca().set_ylim(0,12)
figure.gca().set_yticklabels(figure.gca().get_yticks())
plt.legend()
#figure.gca().set_ylim(0,100)
plt.show(block=False)
    


import openalea.cellcomplex.property_topomesh.utils.image_tools
reload(openalea.cellcomplex.property_topomesh.utils.image_tools)
from openalea.cellcomplex.property_topomesh.utils.image_tools import compute_topomesh_image

polydata_img = compute_topomesh_image(topomesh,img)

world.add(polydata_img,"topomesh_image",colormap='glasbey',alphamap='constant',bg_id=1)

from copy import deepcopy

cells_img = deepcopy(img)
for c in set(np.unique(img)).difference(set(topomesh.wisps(3))):
    cells_img[img==c] = 1

true_positives = ((cells_img != 1)&(cells_img == polydata_img)).sum()
false_positives = ((cells_img == 1) & (polydata_img != 1)).sum() + ((cells_img != 1)&(polydata_img != 1)&(cells_img != polydata_img)).sum()
false_negatives = ((cells_img != 1) & (polydata_img == 1)).sum() + ((cells_img != 1)&(polydata_img != 1)&(cells_img != polydata_img)).sum()
true_negatives = ((cells_img == 1)&(cells_img == polydata_img)).sum()

estimators = {}
estimators['Precision'] = float(true_positives/float(true_positives+false_positives))
estimators['Recall'] = float(true_positives/float(true_positives+false_negatives))
estimators['Dice'] = float(2*true_positives/float(2*true_positives+false_positives+false_negatives))
estimators['Jaccard'] = float(true_positives/float(true_positives+false_positives+false_negatives))
estimators['Accuracy'] = float(true_positives+true_negatives)/float(true_positives+true_negatives+false_positives+false_negatives)
estimators['Identity'] = float((cells_img == polydata_img).sum())/np.prod(cells_img.shape)
print estimators


quality = [1,1,0.82,0.88,0.82,1,0.85,0.9,0.94,1]
quality_criteria = []
quality_criteria += ["Mesh Complexity"]
quality_criteria += ["Triangle Area Deviation"]
quality_criteria += ["Triangle Eccentricity"]
quality_criteria += ["Vertex Valence"]
quality_criteria += ["Image Accuracy"]
quality_criteria += ["Vertex Distance"]
quality_criteria += ["Cell 2 Adjacency"]
quality_criteria += ["Cell Convexity"]
quality_criteria += ["Epidermis Cell Angle"]
quality_criteria += ["Cell Cliques"]
quality = dict(zip(quality_criteria,quality))

from vplants.meshing.cute_plot import spider_plot
import matplotlib.pyplot as plt

figure = plt.figure(0)
figure.clf()
spider_plot(figure,np.array([quality[c] for c in quality_criteria]),color1=np.array([0.3,0.6,1.]),color2=np.array([1.,0.,0.]),xlabels=quality_criteria,ytargets=0.8 * np.ones_like(quality_criteria,float),n_points=100*len(quality_criteria),linewidth=2,smooth_factor=0.0,spline_order=1)
plt.show(block=False)
raw_input()

precision = {}
recall = {}
dice = {}
jaccard = {}
for c in topomesh.wisps(3):
    true_positives = ((img==c) & (polydata_img==c)).sum()
    false_positives = ((img!=c) & (polydata_img==c)).sum()
    false_negatives = ((img==c) & (polydata_img!=c)).sum()
    
    precision[c] = true_positives/float(true_positives+false_positives)
    recall[c] = true_positives/float(true_positives+false_negatives)
    dice[c] = 2*true_positives/float(2*true_positives+false_positives+false_negatives)
    jaccard[c] = true_positives/float(true_positives+false_positives+false_negatives)



filename = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/output_meshes/p194-t5_imgSeg_SegExp_CellShapeCorr/p194-t5_imgSeg_SegExp_CellShapeCorr_triangulated_L1_L2_star_remeshed_projected_flat_temporal_properties_topomesh.pkl"
topomesh = pickle.load(open(filename,'r'))

world.add(topomesh,"topomesh")

from openalea.mesh.property_topomesh_analysis import cut_surface_topomesh

topomesh = cut_surface_topomesh(topomesh,z_cut=55.,below=False)

world.add(topomesh,"topomesh")

compute_topomesh_property(topomesh,'vertices',2)
z_cut = 55.
triangle_below = array_dict(np.all(topomesh.wisp_property('barycenter',0).values(topomesh.wisp_property('vertices',2).values())[...,2]>z_cut,axis=1),list(topomesh.wisps(2)))
topomesh.update_wisp_property('below',2,triangle_below)

triangles_to_remove = [t for t in topomesh.wisps(2) if triangle_below[t]]
for t in triangles_to_remove:
    topomesh.remove_wisp(2,t)
    
cells_to_remove = [w for w in topomesh.wisps(3) if topomesh.nb_borders(3,w)==0]
for w in cells_to_remove:
    topomesh.remove_wisp(3,w)

edges_to_remove = [w for w in topomesh.wisps(1) if topomesh.nb_regions(1,w)==0]
for w in edges_to_remove:
    topomesh.remove_wisp(1,w)
    
vertices_to_remove = [w for w in topomesh.wisps(0) if topomesh.nb_regions(0,w)==0]
for w in vertices_to_remove:
    topomesh.remove_wisp(0,w)
    
world.add(topomesh,"topomesh")

compute_topomesh_property(topomesh,'triangles',1)
compute_topomesh_property(topomesh,'vertices',1)

topomesh.update_wisp_property('boundary',1,array_dict((np.array(map(len,topomesh.wisp_property('triangles',1).values()))==1).astype(int),list(topomesh.wisps(1))))

boundary_edges = np.array(list(topomesh.wisps(1)))[topomesh.wisp_property('boundary',1).values()==1]
boundary_vertices = np.unique(topomesh.wisp_property('vertices',1).values(boundary_edges))
#boundary_z = topomesh.wisp_property('barycenter',0).values(boundary_vertices)[:,2]

iso_z_positions = np.array([np.concatenate([topomesh.wisp_property('barycenter',0)[v][:2],[z_cut+2]]) if v in boundary_vertices else  topomesh.wisp_property('barycenter',0)[v] for v in topomesh.wisps(0)])
topomesh.update_wisp_property('barycenter',0,array_dict(iso_z_positions,list(topomesh.wisps(0))))

world.add(topomesh,"topomesh")

for degree in [0,1,2,3]:
    for property_name in topomesh.wisp_property_names(degree):
        topomesh.update_wisp_property(property_name,degree,array_dict(topomesh.wisp_property(property_name,degree).values(list(topomesh.wisps(degree))),list(topomesh.wisps(degree))))

topomesh_filename = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/output_meshes/p194-t5_imgSeg_SegExp_CellShapeCorr/p194-t5_imgSeg_SegExp_CellShapeCorr_triangulated_L1_L2_star_remeshed_projected_flat_temporal_properties_cut_iso_topomesh.pkl"
save_property_topomesh(topomesh, topomesh_filename, cells_to_save=None, properties_to_save=dict([(0,['barycenter']),(1,[]),(2,[]),(3,['layer','mother_cell','absolute_volumetric_growth_rate','volumetric_growth_rate','volumetric_strain_rate','stretch_tensor'])]))



from openalea.oalab.colormap.colormap_def import load_colormaps

filename = "/Users/gcerutti/Developpement/openalea/openalea_meshing_data/share/data/output_meshes/propTopomesh_withCurv_persoZones_T00/propTopomesh_withCurv_persoZones_T00_star_split_topomesh.pkl"
topomesh = pickle.load(open(filename,'r'))

compute_topomesh_property(topomesh,'triangles',1)
compute_topomesh_property(topomesh,'vertices',1)

topomesh.update_wisp_property('boundary',1,array_dict((np.array(map(len,topomesh.wisp_property('triangles',1).values()))==1).astype(int),list(topomesh.wisps(1))))

world.add(topomesh,"topomesh")
world['topomesh'].set_attribute('display_3',False)
world['topomesh'].set_attribute('property_name_1','boundary')
world['topomesh'].set_attribute('display_1',True)
world['topomesh_edges'].set_attribute('intensity_range',(0,1))
world['topomesh_edges'].set_attribute('polydata_colormap',load_colormaps()['Reds'])
world['topomesh_edges'].set_attribute('linewidth',3)
world['topomesh_edges'].set_attribute('polydata_alpha',0.5)
raw_input()

from scipy.cluster.vq import vq
from time import sleep
distance_threshold = 0.5

boundary_edges = np.array(list(topomesh.wisps(1)))[topomesh.wisp_property('boundary',1).values()==1]
boundary_edge_centers = topomesh.wisp_property('barycenter',0).values(topomesh.wisp_property('vertices',1).values(boundary_edges)).mean(axis=1)
sorted_boundary_edges = boundary_edges[np.argsort(-boundary_edge_centers[:,2])]
sorted_boundary_edge_centers = boundary_edge_centers[np.argsort(-boundary_edge_centers[:,2])]


boundary_vertices = list(np.unique(topomesh.wisp_property('vertices',1).values(boundary_edges)))

for e, edge_center in zip(sorted_boundary_edges, sorted_boundary_edge_centers)[:20]:
    edge_triangle = list(topomesh.regions(1,e))[0]
    edge_points = topomesh.wisp_property('barycenter',0).values(list(topomesh.borders(1,e)))
    edge_vector = edge_points[1] - edge_points[0]
    edge_length = np.linalg.norm(edge_vector)
    edge_vector = edge_vector/edge_length
    edge_opposite_vertex = list(set(topomesh.borders(2,edge_triangle,2)).difference(set(topomesh.borders(1,e))))[0]
    edge_opposite_point = topomesh.wisp_property('barycenter',0)[edge_opposite_vertex]
    edge_face_vector = edge_center - edge_opposite_point
    edge_face_vector = edge_face_vector - np.dot(edge_vector,edge_face_vector)*edge_vector
    edge_face_vector = (np.sqrt(3)*edge_length/2)*(edge_face_vector/np.linalg.norm(edge_face_vector))
    edge_face_point = edge_center + edge_face_vector
    
    boundary_points = topomesh.wisp_property('barycenter',0).values(boundary_vertices)
    
    candidate_point, candidate_distance = vq(np.array([edge_face_point]),boundary_points)    
    candidate_pid = boundary_vertices[candidate_point[0]]
    candidate_distance = candidate_distance[0]
    
    triangle_pids = []
    
    if candidate_distance > distance_threshold or candidate_pid in topomesh.borders(1,e):
        fid = topomesh.add_wisp(2)
        topomesh.link(2,fid,e)
        
        pid = topomesh.add_wisp(0)
        triangle_pids += [pid]
        
        positions = topomesh.wisp_property('barycenter',0).to_dict()
        positions[pid] = edge_face_point
        topomesh.update_wisp_property('barycenter',0,array_dict(positions))
        boundary_vertices += [pid]

        for v in topomesh.borders(1,e):
            triangle_pids += [v]
        
            eid = topomesh.add_wisp(1)
            topomesh.link(1,eid,pid)
            topomesh.link(1,eid,v)
            topomesh.link(2,fid,eid)
            

    else:
        fid = topomesh.add_wisp(2)
        topomesh.link(2,fid,e)
        
        triangle_pids += [candidate_pid]
        boundary_vertices += [candidate_pid]
        
        for v in topomesh.borders(1,e):
            triangle_pids += [v]
            
            candidate_eids = set(topomesh.regions(0,v)).intersection(set(topomesh.regions(0,candidate_pid)))
            if len(candidate_eids) == 1:
                candidate_eid = list(candidate_eids)[0]
                topomesh.link(2,fid,candidate_eid)
                if v in boundary_vertices:
                    boundary_vertices.remove(v)
            else:
                eid = topomesh.add_wisp(1)
                topomesh.link(1,eid,candidate_pid)
                topomesh.link(1,eid,v)
                topomesh.link(2,fid,eid)
                
    triangle_mesh = TriangularMesh()
    triangle_mesh.points = array_dict(topomesh.wisp_property('barycenter',0).values(triangle_pids),triangle_pids)
    triangle_mesh.triangles = dict([(0,triangle_pids)])
    triangle_mesh.triangle_data = dict([(0,1)])
    world.add(triangle_mesh,'triangle')
        
    compute_topomesh_property(topomesh,'triangles',1)
    compute_topomesh_property(topomesh,'vertices',1)
    topomesh.update_wisp_property('boundary',1,array_dict((np.array(map(len,topomesh.wisp_property('triangles',1).values()))==1).astype(int),list(topomesh.wisps(1))))
    
world.add(topomesh,"topomesh")
world['topomesh'].set_attribute('display_3',False)
world['topomesh'].set_attribute('property_name_1','boundary')
world['topomesh'].set_attribute('display_1',True)
world['topomesh_edges'].set_attribute('intensity_range',(0,1))
world['topomesh_edges'].set_attribute('polydata_colormap',load_colormaps()['Reds'])
world['topomesh_edges'].set_attribute('linewidth',3)
world['topomesh_edges'].set_attribute('polydata_alpha',0.5)


import vtk

dome_radius = 50.0
dome_scales = [2,2,2]
dome_axes = np.diag(np.ones(3))
dome_center = np.zeros(3)

ico = vtk.vtkPlatonicSolidSource()
ico.SetSolidTypeToIcosahedron()
#ico.SetSolidTypeToOctahedron()
ico.Update()

sphere = vtk.vtkSphereSource()
# #dome_sphere.SetCenter(meristem_model.shape_model['dome_center'])
sphere.SetRadius(1)
sphere.SetThetaResolution(16)
sphere.SetPhiResolution(16)
sphere.Update()

#subdivide = vtk.vtkLoopSubdivisionFilter()
#subdivide = vtk.vtkButterflySubdivisionFilter()
subdivide = vtk.vtkLinearSubdivisionFilter()
subdivide.SetNumberOfSubdivisions(2)
subdivide.SetInputConnection(sphere.GetOutputPort())
#subdivide.SetInputConnection(ico.GetOutputPort())
subdivide.Update()

decimate = vtk.vtkQuadricClustering()
decimate.SetInput(subdivide.GetOutput())
decimate.SetNumberOfDivisions(100,100,100)
decimate.SetFeaturePointsAngle(30.0)
decimate.CopyCellDataOn()
decimate.Update()

scale_transform = vtk.vtkTransform()
scale_factor = dome_radius/(np.sqrt(2)/2.)
scale_transform.Scale(scale_factor,scale_factor,scale_factor)

dome_sphere = vtk.vtkTransformPolyDataFilter()
#dome_sphere.SetInput(subdivide.GetOutput())
dome_sphere.SetInput(sphere.GetOutput())
#dome_sphere.SetInput(decimate.GetOutput())
dome_sphere.SetTransform(scale_transform)
dome_sphere.Update()
    
ellipsoid_transform = vtk.vtkTransform()
axes_transform = vtk.vtkLandmarkTransform()
source_points = vtk.vtkPoints()
source_points.InsertNextPoint([1,0,0])
source_points.InsertNextPoint([0,1,0])
source_points.InsertNextPoint([0,0,1])
target_points = vtk.vtkPoints()
target_points.InsertNextPoint(dome_axes[0])
target_points.InsertNextPoint(dome_axes[1])
target_points.InsertNextPoint(dome_axes[2])
axes_transform.SetSourceLandmarks(source_points)
axes_transform.SetTargetLandmarks(target_points)
axes_transform.SetModeToRigidBody()
axes_transform.Update()
ellipsoid_transform.SetMatrix(axes_transform.GetMatrix())
ellipsoid_transform.Scale(dome_scales[0],dome_scales[1],dome_scales[2])
center_transform = vtk.vtkTransform()
center_transform.Translate(dome_center[0],dome_center[1],dome_center[2])
center_transform.Concatenate(ellipsoid_transform)
dome_ellipsoid = vtk.vtkTransformPolyDataFilter()
dome_ellipsoid.SetInput(dome_sphere.GetOutput())
dome_ellipsoid.SetTransform(center_transform)
dome_ellipsoid.Update()

from vplants.meshing.vtk_tools import vtk_polydata_to_triangular_mesh
ellipsoid_mesh = vtk_polydata_to_triangular_mesh(dome_ellipsoid.GetOutput())

from openalea.mesh.property_topomesh_creation import triangle_topomesh
topomesh = triangle_topomesh(ellipsoid_mesh.triangles.values(),ellipsoid_mesh.points)

# world.add(topomesh,'topomesh')

# filename = "/Users/gcerutti/Downloads/ellipsoid_topomesh_curvature.pkl"
# save_property_topomesh(topomesh,filename)

# filename = "/Users/gcerutti/Downloads/savedPropertyTopomesh_step-1.pkl"
# topomesh = pickle.load(open(filename,'rb'))


# for c in list(topomesh.wisps(3))[::-1]:
#     if not topomesh.has_wisp(3,c+2):
#         topomesh.add_wisp(3,c+2)
#     for t in topomesh.borders(3,c):
#         topomesh.link(3,c+2,t)
#     topomesh.remove_wisp(3,c)
# topomesh.update_wisp_property('zones',3,array_dict(topomesh.wisp_property('zones',3).values(),topomesh.wisp_property('zones',3).keys()+2))

# compute_topomesh_property(topomesh,'normal',2)
# #compute_topomesh_property(topomesh,'normal',0)
# compute_topomesh_vertex_property_from_faces(topomesh,'normal',weighting='area',adjacency_sigma=1.2,neighborhood=3)
# compute_topomesh_property(topomesh,'mean_curvature',2)

# import openalea.mesh.property_topomesh_io
# reload( openalea.mesh.property_topomesh_io)
# from openalea.mesh.property_topomesh_io import save_property_topomesh, save_ply_property_topomesh, read_ply_property_topomesh


# from openalea.mesh.property_topomesh_creation import triangle_topomesh


# import openalea.mesh.triangular_mesh
# reload(openalea.mesh.triangular_mesh)
# from openalea.mesh.triangular_mesh import save_ply_triangular_mesh, save_ply_triangle_mesh

# mesh = world['topomesh_faces'].data
# ply_filename = "/Users/gcerutti/Desktop/test_tensor.ply"
# save_ply_triangle_mesh(ply_filename,mesh.points,mesh.triangles,triangle_properties=dict(curvature_tensor=mesh.triangle_data))
# #save_ply_triangle_mesh(ply_filename,mesh.points,mesh.triangles)


# topomesh = read_ply_property_topomesh(ply_filename)
# world.clear()
# world.add(topomesh,'topomesh')

topomesh_filename = "/Users/gcerutti/Downloads/pillshaped_topomesh_curvature.pkl"
save_property_topomesh(topomesh,topomesh_filename,properties_to_save=dict([(0,['barycenter','normal']),(1,[]),(2,['area','principal_curvature_tensor','mean_curvature','normal','stress','strain','deformationgradient','youngmoduli','plasticityspeed','strainthreshold','bases']),(3,['zones'])]),original_pids=True,original_eids=True,original_fids=True,original_cids=True)


# import numpy as np
# import pickle

filename = "/Users/gcerutti/Downloads/savedPropertyTopomesh_step-1.pkl"

# filename = "/Users/gcerutti/Downloads/savedPropertyTopomesh_sphere_curvature.pkl"
#filename = "/Users/gcerutti/Downloads/2DWall_Hole_mesh.ply"
filename = "/Users/gcerutti/Downloads/SAM_2D_microtubule.ply"
topomesh = read_ply_property_topomesh(filename)

topomesh.update_wisp_property('barycenter',0,topomesh.wisp_property('barycenter',0).values()*10,list(topomesh.wisps(0)))


from openalea.oalab.colormap.colormap_def import load_colormaps
from openalea.mesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces
compute_topomesh_property(topomesh,'area',2)
compute_topomesh_vertex_property_from_faces(topomesh,'CMFs',weighting='area',adjacency_sigma=1.2,neighborhood=3)

Q = topomesh.wisp_property('CMFs',0).values()
E = np.array([np.linalg.eigvalsh(q) for q in Q])
#CMF_anisotropies = np.array([np.sqrt(1- 4 * np.linalg.det(q[:2,:2]) / np.trace(q[:2,:2])**2) for q in Q])
CMF_anisotropies = np.array([100 *(e[-1] - e[-2]) / (e[-1] + e[-2]) for e in E])
topomesh.update_wisp_property('CMF_anisotropy', 0, array_dict(CMF_anisotropies,list(topomesh.wisps(0))))

world.add(topomesh,'topomesh')
world['topomesh'].set_attribute('property_degree_3',0)
world['topomesh'].set_attribute('display_3',False)

world['topomesh'].set_attribute('property_name_3','CMF_anisotropy')
world['topomesh'].set_attribute('display_3',True)
world['topomesh_cells'].set_attribute('polydata_colormap',load_colormaps()['sepia'])

world['topomesh_faces'].set_attribute('intensity_range',(0,1))

world['topomesh'].set_attribute('property_degree_2',2)
world['topomesh'].set_attribute('property_name_2','CMFs')
world['topomesh'].set_attribute('coef_2',1)
world['topomesh'].set_attribute('display_2', True)
world['topomesh_faces'].set_attribute('polydata_colormap',load_colormaps()['0RGB_red'])
world['topomesh_faces'].set_attribute('linewidth',2)
world['topomesh_faces'].set_attribute('point_radius',2)

#world['topomesh'].set_attribute('property_name_0','CMFs')


#world['topomesh'].set_attribute('display_1',True)
#world['topomesh_edges'].set_attribute('polydata_colormap',cmap_dict('grey'))
#world['topomesh_edges'].set_attribute('linewidth',3)

#world['topomesh'].set_attribute('property_name_3','main_stiffness')
#world['topomesh'].set_attribute('property_name_0','CMFs')
#world['topomesh'].set_attribute('display_0',True)
#world['topomesh_vertices'].set_attribute('point_radius',1e-5)
#world['topomesh_vertices'].set_attribute('polydata_colormap',load_colormaps()['0RGB_red'])
#world['topomesh_vertices'].set_attribute('linewidth',2)
#world['topomesh_vertices'].set_attribute('point_radius',50)


world['topomesh'].set_attribute('property_degree_2',2)
world['topomesh'].set_attribute('property_name_2','CMFs')
world['topomesh'].set_attribute('coef_2',1)
world['topomesh'].set_attribute('display_2', True)
world['topomesh_faces'].set_attribute('polydata_colormap',load_colormaps()['0RGB_red'])
world['topomesh_faces'].set_attribute('linewidth',2)
world['topomesh_faces'].set_attribute('point_radius',50)

#world['topomesh_faces'].set_attribute('polydata_colormap',load_colormaps()['inferno'])
#world['topomesh_faces'].set_attribute('polydata_alpha',0.57)
#world['topomesh_faces'].set_attribute('intensity_range',(0,30))

#from copy import deepcopy
#reverse_topomesh = deepcopy(topomesh)
#reverse_topomesh.update_wisp_property('main_stiffness',2,-topomesh.wisp_property('main_stiffness',2).values())

#world.add(reverse_topomesh,'reverse_topomesh')
#world['reverse_topomesh'].set_attribute('display_3',False)

#world['reverse_topomesh'].set_attribute('property_name_2','main_stiffness')
#world['reverse_topomesh'].set_attribute('display_2',True)
#world['reverse_topomesh_faces'].set_attribute('point_radius',0.5)
#world['reverse_topomesh_faces'].set_attribute('polydata_colormap',cmap_dict('invert_grey'))






triangular = ['star','remeshed','straight']
triangular_string = ""
for t in triangular:
    triangular_string += t+"_"
topomesh_filename = dirname+"/output_meshes/"+filename+"/"+filename+"_triangulated_L1_"+triangular_string+"temporal_properties_topomesh.pkl"

topomesh = pickle.load(open(topomesh_filename,'rb'))



from openalea.oalab.colormap.colormap_def import load_colormaps

import openalea.cellcomplex.property_topomesh.property_topomesh_analysis
reload(openalea.cellcomplex.property_topomesh.property_topomesh_analysis)
#import openalea.mesh.property_topomesh_analysis
#reload(openalea.mesh.property_topomesh_analysis)
from openalea.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces, compute_topomesh_cell_property_from_faces, topomesh_property_gaussian_filtering



compute_topomesh_vertex_property_from_faces(topomesh,'CMFs',weighting='area')

CMF_evalues = np.linalg.eigh(topomesh.wisp_property('CMFs',0).values())[0]
CMF_anisotropy = np.sort(np.abs(CMF_evalues))[:,2]/np.sort(np.abs(CMF_evalues))[:,1]
topomesh.update_wisp_property('CMF_anisotropy',0,array_dict(CMF_anisotropy,list(topomesh.wisps(0))))

from openalea.draco_stem.stem.tissue_mesh_optimization import optimize_topomesh

#topomesh = optimize_topomesh(topomesh,omega_forces=dict(laplacian_smoothing=0.65),iterations=10)
#topomesh_property_gaussian_filtering(topomesh ,'barycenter',0,adjacency_sigma=2.0,distance_sigma=100.0,neighborhood=3)
#topomesh.update_wisp_property('barycenter',0,dome_radius*topomesh.wisp_property('barycenter',0).values()/ np.linalg.norm(topomesh.wisp_property('barycenter',0).values(),axis=1)[:,np.newaxis],list(topomesh.wisps(0)))


print "Radius : ", np.linalg.norm(topomesh.wisp_property('barycenter',0).values() ,axis=1).mean()

compute_topomesh_property(topomesh,'length',degree=1)
compute_topomesh_property(topomesh,'area',degree=2)
compute_topomesh_property(topomesh,'barycenter',degree=2)
compute_topomesh_property(topomesh,'normal',2,normal_method='barycenter')
#compute_topomesh_property(topomesh,'normal',0)
compute_topomesh_vertex_property_from_faces(topomesh,'normal',weighting='area',adjacency_sigma=1.2,neighborhood=3)
#topomesh.update_wisp_property('normal',0,topomesh.wisp_property('barycenter',0).values()/ np.linalg.norm(topomesh.wisp_property('barycenter',0).values(),axis=1)[:,np.newaxis],list(topomesh.wisps(0)))
compute_topomesh_property(topomesh,'mean_curvature',2)


compute_topomesh_cell_property_from_faces(topomesh,'mean_curvature',weighting='area')
compute_topomesh_cell_property_from_faces(topomesh,'principal_curvature_tensor',weighting='area')

compute_topomesh_vertex_property_from_faces(topomesh,'area',weighting='area',adjacency_sigma=1.2,neighborhood=3)


world.add(topomesh,"topomesh")

world['topomesh'].set_attribute('property_degree_2',3)
#world['topomesh'].set_attribute('property_degree_3',2)
world['topomesh'].set_attribute('display_3',False)



world['topomesh'].set_attribute('property_name_2','volumetric_growth_rate')
world['topomesh'].set_attribute('display_2',True)
world['topomesh_faces'].set_attribute('polydata_colormap',load_colormaps()['inferno'])
world['topomesh_faces'].set_attribute('x_slice',(40,100))
world['topomesh_faces'].set_attribute('z_slice',(0,40))
world['topomesh_faces'].set_attribute('intensity_range',(0.05,0.25))
world['topomesh_faces'].set_attribute('polydata_alpha',0.9)

world['topomesh'].set_attribute('coef_3',1)
#world['topomesh'].set_attribute('property_name_3','mean_curvature')
#world['topomesh'].set_attribute('property_name_3','volumetric_growth_rate')
world['topomesh'].set_attribute('property_name_3','principal_curvature_tensor')
world['topomesh'].set_attribute('display_3',True)
world['topomesh_cells'].set_attribute('intensity_range',(-1,1))
world['topomesh_cells'].set_attribute('polydata_colormap',load_colormaps()['curvature'])
#world['topomesh_cells'].set_attribute('intensity_range',(0,0.3))
#world['topomesh_cells'].set_attribute('polydata_colormap',load_colormaps()['inferno'])
world['topomesh_cells'].set_attribute('polydata_alpha',0.9)
world['topomesh_cells'].set_attribute('point_radius',2.)
world['topomesh_cells'].set_attribute('linewidth',2)
world['topomesh_cells'].set_attribute('x_slice',(40,100))
world['topomesh_cells'].set_attribute('z_slice',(0,40))

#world['topomesh'].set_attribute('property_name_0','normal')
#world['topomesh'].set_attribute('display_0',True)
#world['topomesh_vertices'].set_attribute('point_radius',10)
#world['topomesh_vertices'].set_attribute('x_slice',(40,100))

world['topomesh'].set_attribute('display_1',True)
world['topomesh_edges'].set_attribute('polydata_alpha',0.5)
world['topomesh_edges'].set_attribute('linewidth',2)
world['topomesh_edges'].set_attribute('x_slice',(40,100))


import openalea.mesh.property_topomesh_optimization
reload(openalea.mesh.property_topomesh_optimization)

import openalea.draco_stem.stem.tissue_mesh_optimization
reload(openalea.draco_stem.stem.tissue_mesh_optimization)
from openalea.draco_stem.stem.tissue_mesh_optimization import optimize_topomesh

import openalea.mesh.property_topomesh_analysis
reload(openalea.mesh.property_topomesh_analysis)
from openalea.mesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_vertex_property_from_faces, topomesh_property_gaussian_filtering

from openalea.oalab.colormap.colormap_def import load_colormaps

filename = "/Users/gcerutti/Downloads/2DWall_mesh.ply"
topomesh = read_ply_property_topomesh(filename,verbose=True)


compute_topomesh_property(topomesh,'barycenter',2)
topomesh.update_wisp_property('zones',2,np.abs(topomesh.wisp_property('barycenter',2).values()).max(axis=1)<6.,list(topomesh.wisps(2)))

compute_topomesh_property(topomesh,'area',2)
compute_topomesh_vertex_property_from_faces(topomesh,'area')

world.add(topomesh,"topomesh")
#world['topomesh'].set_attribute('property_degree_2',0)
world['topomesh'].set_attribute('display_3',False)
world['topomesh'].set_attribute('property_name_2','area')
world['topomesh'].set_attribute('coef_2',0.95)
world['topomesh'].set_attribute('display_2',True)
world['topomesh_faces'].set_attribute('polydata_colormap',load_colormaps()['Greens'])
world['topomesh_faces'].set_attribute('intensity_range',(0,1))

print "Area : ",topomesh.wisp_property('area',2).values().mean()," (",topomesh.wisp_property('area',2).values().std(),")"
raw_input()

#target_areas = 1.5*topomesh.wisp_property('area',2).values().mean()*np.ones(topomesh.nb_wisps(2))
#target_areas = 0.7*topomesh.wisp_property('area',2).values().mean()*np.ones(topomesh.nb_wisps(2))
#target_areas = 2.*topomesh.wisp_property('area',2).values().mean() - topomesh.wisp_property('area',2).values()

target_areas = (1.2*topomesh.wisp_property('zones',2).values() + 1.0*(1-topomesh.wisp_property('zones',2).values()))*topomesh.wisp_property('area',2).values()
#target_areas = (1.2*topomesh.wisp_property('zones',2).values() + 1.0*(1-topomesh.wisp_property('zones',2).values()))*topomesh.wisp_property('area',2).values().mean()*np.ones(topomesh.nb_wisps(2))

area_filename = "/Users/gcerutti/Desktop/AreaOptimization/area0.jpg"
viewer.save_screenshot(area_filename)

for iteration in xrange(60):
    #topomesh = optimize_topomesh(topomesh,omega_forces=dict(area=0.15,regularization=0.001),iterations=1)
    topomesh = optimize_topomesh(topomesh,omega_forces=dict(area=0.01),iterations=100,target_areas=target_areas)
    compute_topomesh_vertex_property_from_faces(topomesh,'area')
    #world['topomesh'].data = topomesh
    world.add(topomesh,"topomesh")
    #world['topomesh'].set_attribute('property_degree_2',0)
    world['topomesh'].set_attribute('display_3',False)
    world['topomesh'].set_attribute('property_name_2','area')
    world['topomesh'].set_attribute('coef_2',0.95)
    world['topomesh'].set_attribute('display_2',True)
    world['topomesh_faces'].set_attribute('polydata_colormap',load_colormaps()['Greens'])
    world['topomesh_faces'].set_attribute('intensity_range',(0,1))
    print "Area : ",topomesh.wisp_property('area',2).values().mean()," (",topomesh.wisp_property('area',2).values().std(),")"
    
    area_filename = "/Users/gcerutti/Desktop/AreaOptimization/area"+str(iteration+1)+".jpg"
    viewer.save_screenshot(area_filename)
    #raw_input()


compute_topomesh_property(topomesh,'eccentricity',2)

import matplotlib.pyplot as plt
from vplants.meshing.cute_plot import simple_plot

figure = plt.figure(0)
figure.clf()
simple_plot(figure,topomesh.wisp_property('eccentricity',2).values(),topomesh.wisp_property('mean_curvature',2).values(),color1=np.array([0.2,0.8,0.4]),xlabel="Triangle Eccentricity",ylabel="Max Curvature",linked=False)
plt.show(block=False)

#filename = "p194-t5_imgSeg_SegExp_CellShapeCorr"
filename = "rs01_wt_t00_seg"
#filename = "segmentation"

import vplants.meshing_data
dirname = shared_data(vplants.meshing_data)
meshing_dirname =  dirname.parent.parent

import openalea.cellcomplex
dirname = shared_data(openalea.cellcomplex)

#triangular = ['star','remeshed','projected','straight']
triangular = ['star','flat']
triangular_string = ""
for t in triangular:
    triangular_string += t+"_"

#topomesh_filename = "/Users/gcerutti/Desktop/"+filename+"_L1"+triangular_string+"_topomesh.ply"
#topomesh_filename = dirname+"/output_meshes/"+filename+"/"+filename+"_L1"+triangular_string+"_topomesh.ply"
topomesh_filename = dirname+"/output_meshes/"+filename+"/"+filename+"_"+triangular_string+"topomesh.ply"
#topomesh_filename = dirname+"/output_meshes/"+filename+"/"+filename+"_L1_topomesh.ply"

topomesh = read_ply_property_topomesh(topomesh_filename)
world.add(topomesh,"topomesh")

compute_topomesh_property(topomesh,'cells',2)
np.array(map(len,topomesh.wisp_property('cells',2).values()))

compute_topomesh_property(topomesh,'triangles',1)
np.array(map(len,topomesh.wisp_property('triangles',1).values()))

import matplotlib.pyplot as plt
from vplants.meshing.cute_plot import histo_plot

figure = plt.figure(0)
figure.clf()
histo_plot(figure,topomesh.wisp_property('area',2).values(),color=np.array([1,0,0]),xlabel="Area",ylabel="Triangles (%)",cumul=False,bar=True)
plt.show(block=False)

compute_topomesh_property(topomesh,'normal',2)
compute_topomesh_vertex_property_from_faces(topomesh,'normal',weighting='area',adjacency_sigma=1.2,neighborhood=3)
compute_topomesh_property(topomesh,'mean_curvature',0)

world.add(topomesh,"topomesh")
world['topomesh'].set_attribute('property_degree_3',0)
world['topomesh'].set_attribute('display_3',False)
world['topomesh'].set_attribute('coef_3',0.99)
world['topomesh'].set_attribute('property_name_3','mean_curvature')
world['topomesh'].set_attribute('display_3',True)
world['topomesh_cells'].set_attribute('intensity_range',(-0.1,0.1))

world['topomesh_edges'].set_attribute('intensity_range',(0,10))


from openalea.image.serial.all import imread
inputfile = dirname+"/segmented_images/"+filename+".inr.gz"
img = imread(inputfile)
world.add(img,'segmented_image')


epidermis_triangles = np.array(list(topomesh.wisps(2)))[topomesh.wisp_property('epidermis',2).values(list(topomesh.wisps(2))).astype(bool)]
triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(epidermis_triangles)

positions = topomesh.wisp_property('barycenter',0)

triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])
triangle_edge_vertices = triangle_vertices[:,triangle_edge_list]
triangle_edge_points = positions.values(triangle_edge_vertices)
triangle_edge_vectors = triangle_edge_points[:,:,1]-triangle_edge_points[:,:,0]

triangle_vertex_normals = topomesh.wisp_property('normal',0).values(triangle_vertices)
triangle_barycenter_normals = triangle_vertex_normals.mean(axis=1)
#triangle_barycenter_normals = triangle_barycenter_normals/np.linalg.norm(triangle_barycenter_normals,axis=1)[:,np.newaxis]

triangle_barycenter_normal_derivatives = triangle_vertex_normals[:,triangle_edge_list] 
triangle_barycenter_normal_derivatives = triangle_barycenter_normal_derivatives[:,:,1] - triangle_barycenter_normal_derivatives[:,:,0]
triangle_barycenter_normal_derivatives = triangle_barycenter_normal_derivatives/np.linalg.norm(triangle_barycenter_normals,axis=1)[:,np.newaxis,np.newaxis]

triangle_barycenter_derivatives_projectors = np.transpose([np.einsum("...ij,...ij->...i",triangle_barycenter_normals,triangle_edge_vectors[:,k])[:,np.newaxis]*triangle_barycenter_normals for k in xrange(3)],(1,0,2))
triangle_projected_barycenter_derivatives = triangle_edge_vectors - triangle_barycenter_derivatives_projectors

E = np.einsum("...ij,...ij->...i",triangle_projected_barycenter_derivatives[:,1],triangle_projected_barycenter_derivatives[:,1])
F = np.einsum("...ij,...ij->...i",triangle_projected_barycenter_derivatives[:,1],triangle_projected_barycenter_derivatives[:,2])
G = np.einsum("...ij,...ij->...i",triangle_projected_barycenter_derivatives[:,2],triangle_projected_barycenter_derivatives[:,2])

L = -np.einsum("...ij,...ij->...i",triangle_barycenter_normal_derivatives[:,1],triangle_projected_barycenter_derivatives[:,1])
M1 = -np.einsum("...ij,...ij->...i",triangle_barycenter_normal_derivatives[:,1],triangle_projected_barycenter_derivatives[:,2])
M2 = -np.einsum("...ij,...ij->...i",triangle_barycenter_normal_derivatives[:,2],triangle_projected_barycenter_derivatives[:,1])
N = -np.einsum("...ij,...ij->...i",triangle_barycenter_normal_derivatives[:,2],triangle_projected_barycenter_derivatives[:,2])

weingarten_curvature_matrix = np.zeros((len(epidermis_triangles),2,2))
weingarten_curvature_matrix[:,0,0] = (L*G-M1*F)/(E*G-F*F)
weingarten_curvature_matrix[:,0,1] = (M2*G-N*F)/(E*G-F*F)
weingarten_curvature_matrix[:,1,0] = (M1*E-L*F)/(E*G-F*F)
weingarten_curvature_matrix[:,1,1] = (N*E-M2*F)/(E*G-F*F)

weingarten_curvature_matrix_eigenvalues, weingarten_curvature_matrix_eigenvectors = np.linalg.eig(weingarten_curvature_matrix)

weingarten_principal_curvature_min = weingarten_curvature_matrix_eigenvalues[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(weingarten_curvature_matrix_eigenvalues))[:,0]])].astype(float)
weingarten_principal_curvature_max = weingarten_curvature_matrix_eigenvalues[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(weingarten_curvature_matrix_eigenvalues))[:,1]])].astype(float)

weingarten_principal_vector_min = weingarten_curvature_matrix_eigenvectors[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(weingarten_curvature_matrix_eigenvalues))[:,0]])].astype(float)
weingarten_principal_vector_max = weingarten_curvature_matrix_eigenvectors[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(weingarten_curvature_matrix_eigenvalues))[:,1]])].astype(float)

weingarten_principal_direction_min = array_dict((weingarten_principal_vector_min[:,:,np.newaxis]*triangle_projected_barycenter_derivatives[:,1:3]).sum(axis=1),epidermis_triangles)
weingarten_principal_direction_max = array_dict((weingarten_principal_vector_max[:,:,np.newaxis]*triangle_projected_barycenter_derivatives[:,1:3]).sum(axis=1),epidermis_triangles)

P = np.transpose([weingarten_principal_direction_max.values(), weingarten_principal_direction_min.values(), triangle_barycenter_normals],(1,2,0))
D = np.array([np.diag(d) for d in np.transpose([weingarten_principal_curvature_max,weingarten_principal_curvature_min,np.zeros_like(epidermis_triangles)])])
P_i = np.array([np.linalg.pinv(p) for p in P])

face_curvature_tensor = np.einsum('...ij,...jk->...ik',-P,np.einsum('...ij,...jk->...ik',D,P_i))
face_curvature_matrix_eigenvalues, face_curvature_matrix_eigenvectors = np.linalg.eig(face_curvature_tensor)

face_principal_curvature_min = array_dict(face_curvature_matrix_eigenvalues[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(face_curvature_matrix_eigenvalues))[:,1]])].astype(float),epidermis_triangles)
face_principal_curvature_max = array_dict(face_curvature_matrix_eigenvalues[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(face_curvature_matrix_eigenvalues))[:,2]])].astype(float),epidermis_triangles)

face_principal_direction_min = array_dict(face_curvature_matrix_eigenvectors[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(face_curvature_matrix_eigenvalues))[:,1]])].astype(float),epidermis_triangles)
face_principal_direction_max = array_dict(face_curvature_matrix_eigenvectors[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(face_curvature_matrix_eigenvalues))[:,2]])].astype(float),epidermis_triangles)

face_principal_curvature_tensor = array_dict(face_curvature_tensor,epidermis_triangles)
topomesh.update_wisp_property('principal_curvature_tensor',2,np.array([face_principal_curvature_tensor[t] if t in epidermis_triangles else np.zeros((3,3)) for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))
topomesh.update_wisp_property('mean_curvature',2,np.array([(face_principal_curvature_min[t] + face_principal_curvature_max[t])/2. if t in epidermis_triangles else np.zeros((3,3)) for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))




import openalea.mesh.triangular_mesh
reload(openalea.mesh.triangular_mesh)
from openalea.mesh.triangular_mesh import topomesh_to_triangular_mesh
face_mesh,_ = topomesh_to_triangular_mesh(topomesh,2,property_name='principal_curvature_tensor',mesh_center=[0,0,0])
face_polydata = face_mesh._repr_vtk_()

sphere = vtk.vtkSphereSource()
sphere.SetThetaResolution(16)
sphere.SetPhiResolution(8)
sphere.Update()

glyph = vtk.vtkTensorGlyph()

glyph.SetSourceConnection(sphere.GetOutputPort())
glyph.SetInput(face_polydata)
glyph.ColorGlyphsOn()
glyph.ThreeGlyphsOff()
glyph.SetColorModeToEigenvalues()
glyph.SymmetricOff()
glyph.SetScaleFactor(20.)
glyph.ExtractEigenvaluesOn()
glyph.Update()

glyph_mesh = vtk_polydata_to_triangular_mesh(glyph.GetOutput())
world.add(glyph_mesh,"face_curvature_tensor_glyph")

face_principal_direction_max_mesh = TriangularMesh()
maximal_face_id = max(topomesh.wisps(2))+1
principal_direction_max_points = np.concatenate([np.array(list(topomesh.wisps(2))),np.array(list(topomesh.wisps(2)))+maximal_face_id])
#point_principal_direction_maxs = 10.*face_principal_curvature_max.values()[:,np.newaxis]*face_principal_direction_max.values()/np.linalg.norm(face_principal_direction_max.values(),axis=1)[:,np.newaxis]
point_principal_direction_maxs = face_principal_direction_max.values()/np.linalg.norm(face_principal_direction_max.values(),axis=1)[:,np.newaxis]
principal_direction_max_point_positions = np.concatenate([topomesh.wisp_property('barycenter',2).values()-point_principal_direction_maxs,topomesh.wisp_property('barycenter',2).values()+point_principal_direction_maxs])
face_principal_direction_max_mesh.points = array_dict(principal_direction_max_point_positions,principal_direction_max_points)
face_principal_direction_max_mesh.edges = array_dict(np.array([(c,c+maximal_face_id) for c in topomesh.wisps(2)]))
world.add(face_principal_direction_max_mesh,'face_max_principal_directions',colormap='Reds',linewidth=3)

face_principal_direction_min_mesh = TriangularMesh()
maximal_face_id = max(topomesh.wisps(2))+1
principal_direction_min_points = np.concatenate([np.array(list(topomesh.wisps(2))),np.array(list(topomesh.wisps(2)))+maximal_face_id])
#point_principal_direction_mins = 10.*face_principal_curvature_min.values()[:,np.newaxis]*face_principal_direction_min.values()/np.linalg.norm(face_principal_direction_min.values(),axis=1)[:,np.newaxis]
point_principal_direction_mins = face_principal_direction_min.values()/np.linalg.norm(face_principal_direction_min.values(),axis=1)[:,np.newaxis]
principal_direction_min_point_positions = np.concatenate([topomesh.wisp_property('barycenter',2).values()-point_principal_direction_mins,topomesh.wisp_property('barycenter',2).values()+point_principal_direction_mins])
face_principal_direction_min_mesh.points = array_dict(principal_direction_min_point_positions,principal_direction_min_points)
face_principal_direction_min_mesh.edges = array_dict(np.array([(c,c+maximal_face_id) for c in topomesh.wisps(2)]))
world.add(face_principal_direction_min_mesh,'face_min_principal_directions',colormap='Blues',linewidth=3)

topomesh.update_wisp_property('principal_curvature_min',2,np.array([face_principal_curvature_min[t] if t in epidermis_triangles else 0. for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))
topomesh.update_wisp_property('principal_curvature_max',2,np.array([face_principal_curvature_max[t] if t in epidermis_triangles else 0. for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))
topomesh.update_wisp_property('mean_curvature',2,(topomesh.wisp_property('principal_curvature_max',2).values(list(topomesh.wisps(2)))+topomesh.wisp_property('principal_curvature_min',2).values(list(topomesh.wisps(2))))/2.,np.array(list(topomesh.wisps(2))))

world.add(topomesh,"face_topomesh")
world['face_topomesh'].set_attribute('property_name_2','mean_curvature')
world['face_topomesh'].set_attribute('display_2',True)
world['face_topomesh'].set_attribute('display_3',False)
world['face_topomesh_faces'].set_attribute('intensity_range',(-0.1,0.1))






