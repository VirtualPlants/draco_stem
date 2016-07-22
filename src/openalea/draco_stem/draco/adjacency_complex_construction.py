import numpy as np
from scipy import ndimage as nd

from scipy.cluster.vq import kmeans, vq

from openalea.image.spatial_image import SpatialImage
from openalea.image.serial.all import imread, imsave

from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image
from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis

from openalea.container import array_dict, PropertyTopomesh

from openalea.mesh.utils.tissue_analysis_tools import cell_vertex_extraction

from openalea.mesh.property_topomesh_analysis import *

from openalea.mesh.utils.intersection_tools import inside_triangle, intersecting_segment, intersecting_triangle
from openalea.mesh.utils.evaluation_tools import jaccard_index
from openalea.mesh.utils.array_tools import array_unique
from openalea.mesh.utils.geometry_tools import tetra_geometric_features, triangle_geometric_features

from sys                                    import argv
from time                                   import time, sleep

tetra_triangle_edge_list  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
tetra_triangle_list  = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])


def layer_triangle_topomesh_construction(layer_edge_topomesh, positions, omega_criteria = {'distance':1.0,'wall_surface':2.0,'clique':10.0}, **kwargs):

    compute_topomesh_property(layer_edge_topomesh,'length',1)

    edge_weights = np.zeros(layer_edge_topomesh.nb_wisps(1))
    if omega_criteria.has_key('distance'):
        edge_weights += omega_criteria['distance']*np.exp(-np.power(layer_edge_topomesh.wisp_property('length',1).values()/15.0,1))
    
    if omega_criteria.has_key('wall_surface'):
        img_wall_surfaces = kwargs.get('wall_surfaces',None)
        img_volumes = kwargs.get('cell_volumes',dict(zip(positions.keys(),np.ones_like(positions.keys()))))
        assert layer_edge_topomesh.has_wisp_property('wall_surface',1) or img_wall_surfaces is not None
        if not layer_edge_topomesh.has_wisp_property('wall_surface',1):
            layer_edge_vertices = np.sort([list(layer_edge_topomesh.borders(1,e)) for e in layer_edge_topomesh.wisps(1)])
            layer_edge_wall_surface = np.array([img_wall_surfaces[tuple(e)]/np.power(img_volumes.values(e).mean(),2./3.) for e in layer_edge_vertices])
            layer_edge_topomesh.update_wisp_property('wall_surface',1,array_dict(layer_edge_wall_surface,list(layer_edge_topomesh.wisps(1))))
        
        edge_weights += omega_criteria['wall_surface']*layer_edge_topomesh.wisp_property('wall_surface',1).values() 
    
    edge_weights = array_dict(edge_weights,list(layer_edge_topomesh.wisps(1)))
    
    edge_neighbor_vertices = [np.concatenate([list(set(layer_edge_topomesh.region_neighbors(0,c)))+[c] for c in layer_edge_topomesh.borders(1,e)]) for e in layer_edge_topomesh.wisps(1)]
    edge_neighbor_vertices_edges = [np.concatenate([list(layer_edge_topomesh.regions(0,n_v)) for n_v in n_vertices]) for n_vertices in edge_neighbor_vertices]
    edge_triangle_edges = [np.unique(e)[nd.sum(np.ones_like(e),e,index=np.unique(e))>3] for e in edge_neighbor_vertices_edges]
    
    if omega_criteria.has_key('clique'):
        edge_neighbor_weights = array_dict([edge_weights.values(e).min() - omega_criteria['clique']*(len(e)>3) for e in edge_triangle_edges],list(layer_edge_topomesh.wisps(1)))
    else:
        edge_neighbor_weights = array_dict([edge_weights.values(e).min() for e in edge_triangle_edges],list(layer_edge_topomesh.wisps(1)))
    edge_triangle_edges = array_dict(edge_triangle_edges,list(layer_edge_topomesh.wisps(1)))
    
    triangulation_edges = np.array(list(layer_edge_topomesh.wisps(1)))[np.array(map(len,edge_triangle_edges))>=3]
    
    layer_triangulation_topomesh = PropertyTopomesh(3)
    layer_triangulation_topomesh.add_wisp(3,1)
    
    initial_edge = np.array(list(layer_edge_topomesh.wisps(1)))[triangulation_edges][np.argmax(edge_neighbor_weights.values(triangulation_edges))]
    free_edges = [initial_edge]
        
    while len(free_edges) > 0:
        eid_to_add = free_edges.pop(0)
        print "--> Edge",list(layer_edge_topomesh.borders(1,eid_to_add))," : ",edge_neighbor_weights[eid_to_add]
        
        edge_vertex_edges = np.concatenate([list(set(layer_edge_topomesh.regions(0,c)).difference({eid_to_add})) for c in layer_edge_topomesh.borders(1,eid_to_add)])
        edge_vertex_edge_vertices =  np.concatenate([c*np.ones(layer_edge_topomesh.nb_regions(0,c)-1) for c in layer_edge_topomesh.borders(1,eid_to_add)])
        edge_vertex_edge_neighbor_vertices = np.array([list(set(layer_edge_topomesh.borders(1,e)).difference({v}))[0] for e,v in zip(edge_vertex_edges,edge_vertex_edge_vertices)])
    
        candidate_triangle_vertices = np.unique(edge_vertex_edge_neighbor_vertices)[nd.sum(np.ones_like(edge_vertex_edge_neighbor_vertices),edge_vertex_edge_neighbor_vertices,index=np.unique(edge_vertex_edge_neighbor_vertices))==2]
        candidate_triangle_edges = np.array([np.concatenate([[eid_to_add],edge_vertex_edges[edge_vertex_edge_neighbor_vertices==c]]) for c in candidate_triangle_vertices])
        
        if len(candidate_triangle_edges)>0:
            candidate_triangle_free_edges = np.array([np.sum([e in free_edges for e in triangle_edges]) for triangle_edges in candidate_triangle_edges])
            candidate_triangle_edge_weights = edge_weights.values(candidate_triangle_edges[:,1:]).min(axis=1)
            
            if (candidate_triangle_free_edges ==candidate_triangle_free_edges.max()).sum() == 1:
                sorted_candidate_triangle_edges = candidate_triangle_edges[np.argsort(-candidate_triangle_free_edges)]
            else:
                sorted_candidate_triangle_edges = candidate_triangle_edges[np.argsort(-candidate_triangle_edge_weights)]
            
            for triangle_edges in sorted_candidate_triangle_edges:
                if np.all(np.array([0 if not layer_triangulation_topomesh.has_wisp(1,e) else layer_triangulation_topomesh.nb_regions(1,e) for e in triangle_edges])<2):
                    triangle_vertices = np.unique([list(layer_edge_topomesh.borders(1,e)) for e in triangle_edges])
                    if layer_triangulation_topomesh.nb_wisps(2)!=1 or vq(np.sort([triangle_vertices]),np.sort([list(layer_triangulation_topomesh.borders(2,t,2)) for t in layer_triangulation_topomesh.wisps(2)]))[1][0]>0:
                        fid = layer_triangulation_topomesh.add_wisp(2)
                        layer_triangulation_topomesh.link(3,1,fid)
                        print "  --> Triangle",fid,triangle_vertices," : ",edge_weights.values(triangle_edges[1:]).min()
                        for c in triangle_vertices:
                            if not layer_triangulation_topomesh.has_wisp(0,c):
                                layer_triangulation_topomesh.add_wisp(0,c)
                        for e in triangle_edges:
                            if not layer_triangulation_topomesh.has_wisp(1,e):
                                layer_triangulation_topomesh.add_wisp(1,e)
                                for c in layer_edge_topomesh.borders(1,e):
                                    layer_triangulation_topomesh.link(1,e,c)
                            layer_triangulation_topomesh.link(2,fid,e)
                        
                            if layer_triangulation_topomesh.nb_regions(1,e)<2:
                                if not e in free_edges:
                                    free_edges.append(e)
                                edge_future_triangle_edges = list(set(edge_triangle_edges[e]).difference(set(layer_triangulation_topomesh.wisps(1)).difference(set(free_edges))))
                                
                                if omega_criteria.has_key('clique'):
                                    edge_neighbor_weights[e] = np.min(edge_weights.values(edge_future_triangle_edges)) - omega_criteria['clique']*(len(edge_future_triangle_edges)>3)
                                else:
                                    edge_neighbor_weights[e] = np.min(edge_weights.values(edge_future_triangle_edges))
            
            print free_edges
            if len(free_edges)>0:
                free_edges = list(np.array(free_edges)[np.argsort(-edge_neighbor_weights.values(free_edges))])                    
        layer_triangulation_topomesh.update_wisp_property('barycenter',0,array_dict(positions.values(list(layer_triangulation_topomesh.wisps(0))),list(layer_triangulation_topomesh.wisps(0))))
    return layer_triangulation_topomesh


def layered_tetrahedra_topomesh_construction(layer_triangle_topomesh, positions, cell_layer, omega_criteria = {'distance':1.0,'wall_surface':2.0,'clique':10.0}, **kwargs):
    compute_topomesh_property(layer_triangle_topomesh,'length',1)
    compute_topomesh_property(layer_triangle_topomesh,'borders',2)
    compute_topomesh_property(layer_triangle_topomesh,'perimeter',2)
    
    if omega_criteria.has_key('wall_surface'):
        img_wall_surfaces = kwargs.get('wall_surfaces',None)
        img_volumes = kwargs.get('cell_volumes',dict(zip(positions.keys(),np.ones_like(positions.keys()))))
        assert layer_triangle_topomesh.has_wisp_property('wall_surface',1) or img_wall_surfaces is not None
        if not layer_triangle_topomesh.has_wisp_property('wall_surface',1):
            L1_L2_triangle_edge_vertices = np.array([np.sort([list(layer_triangle_topomesh.borders(1,e)) for e in layer_triangle_topomesh.borders(2,t)]) for t in layer_triangle_topomesh.wisps(2)])
            L1_L2_triangle_edge_wall_surface = np.array([[-1. if tuple(e) not in img_wall_surfaces.keys() else img_wall_surfaces[tuple(e)]/np.power(img_volumes.values(e).mean(),2./3.) for e in t] for t in L1_L2_triangle_edge_vertices])
            layer_triangle_topomesh.update_wisp_property('wall_surface',2,array_dict(L1_L2_triangle_edge_wall_surface.min(axis=1),list(layer_triangle_topomesh.wisps(2))))
            layer_triangle_topomesh = layer_triangle_topomesh

    triangle_weights = np.zeros(layer_triangle_topomesh.nb_wisps(2))
    if omega_criteria.has_key('distance'):
        triangle_weights += omega_criteria['distance']*np.exp(-np.power(layer_triangle_topomesh.wisp_property('length',1).values(layer_triangle_topomesh.wisp_property('borders',2).values()).max(axis=1)/15.0,1))
    if omega_criteria.has_key('wall_surface'):
        triangle_weights += omega_criteria['wall_surface']*layer_triangle_topomesh.wisp_property('wall_surface',2).values() 
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
    initial_triangle = np.array(list(layer_triangle_topomesh.wisps(2)))[tetrahedrization_triangles][np.argmax(triangle_neighbor_weights.values(tetrahedrization_triangles))]
    free_triangles = [initial_triangle]

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
                        if len(np.unique(cell_layer.values(tetra_vertices)))==2:
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
                                    
                                    if constructed_triangulation_topomesh.nb_regions(2,t)<2 and len(np.unique(cell_layer.values(list(constructed_triangulation_topomesh.borders(2,t,2)))))==2:
                                        if not t in free_triangles:
                                            free_triangles.append(t)
                                        
                                        triangle_future_tetra_triangles = list(set(triangle_tetrahedra_triangles[t]).difference(set(constructed_triangulation_topomesh.wisps(2)).difference(set(free_triangles))))
                                        
                                        if omega_criteria.has_key('clique'):
                                            triangle_neighbor_weights[t] = np.min(triangle_weights.values(triangle_future_tetra_triangles)) - omega_criteria['clique']*(len(triangle_future_tetra_triangles)-4)
                                        else:
                                            triangle_neighbor_weights[t] = np.min(triangle_weights.values(triangle_future_tetra_triangles))
        
        # print free_triangles
        if len(free_triangles)>0:
            free_triangles = list(np.array(free_triangles)[np.argsort(-triangle_neighbor_weights.values(free_triangles))])
        constructed_triangulation_topomesh.update_wisp_property('barycenter',0,array_dict(positions.values(list(constructed_triangulation_topomesh.wisps(0))),list(constructed_triangulation_topomesh.wisps(0))))

    return constructed_triangulation_topomesh


    
