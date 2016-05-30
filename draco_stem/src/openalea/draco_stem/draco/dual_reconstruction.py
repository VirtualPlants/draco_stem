# -*- coding: utf-8 -*-
# -*- python -*-
#
#       DRACO-STEM
#       Dual Reconstruction by Adjacency Complex Optimization
#       SAM Tissue Enhanced Mesh
#
#       Copyright 2014-2016 INRIA - CIRAD - INRA
#
#       File author(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       File contributor(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenaleaLab Website : http://virtualplants.github.io/
#
###############################################################################

import numpy as np
from scipy import ndimage as nd

from scipy.cluster.vq import kmeans, vq

from openalea.image.spatial_image import SpatialImage
from openalea.image.serial.all import imread, imsave

from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image
from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis
from openalea.mesh.utils.tissue_analysis_tools  import cell_vertex_extraction

from openalea.container.array_dict import array_dict
from openalea.container.property_topomesh import PropertyTopomesh
from openalea.mesh.property_topomesh_analysis import *
from openalea.mesh.utils.intersection_tools import inside_triangle, intersecting_segment, intersecting_triangle
from openalea.mesh.utils.array_tools  import array_unique
from openalea.mesh.utils.geometry_tools import triangle_geometric_features
from openalea.mesh.property_topomesh_optimization import topomesh_triangle_split

from time                                   import time, sleep


tetra_triangle_edge_list  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
tetra_triangle_list  = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])

def tetrahedra_dual_topomesh(image_tetrahedra, positions, voronoi=True, exterior=True, **kwargs):
    """
    Generates a PropertyTopomesh as the dual of a set of tetrahedra
    Tetrahedron -> Topomesh vertex
    Triangle -> Topomesh edge
    Edge -> Topomesh interface
    Vertex -> Topomesh cell
    Vertices are placed as the barycenters of tetrahedra (estimated for exterior vertices) or 
    """
    image_topomesh = PropertyTopomesh(3)
    vertex_positions = {}

    tetra_triangle_list  = np.array([[1,2,3],[0,2,3],[0,1,3],[0,1,2]])
    tetra_edge_list  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])
    triangle_adjacent_edge_list = np.array([[[0,1],[0,2]],[[1,0],[1,2]],[[2,0],[2,1]]])

    cell_tetrahedra = {}
    cell_neighbors = {}

    for c in np.unique(image_tetrahedra):
        cell_tetrahedra[c] = image_tetrahedra[np.where(image_tetrahedra == c)[0]]
        cell_neighbors[c] = np.unique(cell_tetrahedra[c])
        cell_neighbors[c] = cell_neighbors[c][np.where(cell_neighbors[c] != c)]

    tetrahedra_center = positions.values(image_tetrahedra[np.where(image_tetrahedra != 1)]).mean(axis=0)

    image_edges = np.concatenate(image_tetrahedra[:,tetra_edge_list])
    image_edges = image_edges[np.where(image_edges != 0)[0]]
    image_edge_lengths = np.linalg.norm(positions.values(image_edges[...,1])-positions.values(image_edges[...,0]),axis=1)
    neighbor_distance = image_edge_lengths.mean()

    anisotropic = False
    if kwargs.has_key('anisotropic_voronoi_tensors'):
        anisotropy_tensors = kwargs.get('anisotropic_voronoi_tensors')
        anisotropy_tensors[1] = np.identity(3)
        anisotropy_invert_tensors = array_dict(np.array([np.linalg.inv(anisotropy_tensors[c]) for c in anisotropy_tensors.keys()]),anisotropy_tensors.keys())
        # anisotropy_invert_tensors = array_dict(np.array([anisotropy_tensors[c] for c in anisotropy_tensors.keys()]),anisotropy_tensors.keys())
        anisotropic = True

    surface = False
    surface_topomesh = kwargs.get('surface_topomesh',None)
    if surface_topomesh is not None:

        compute_topomesh_property(surface_topomesh,'vertices',2)
        surface_triangle_points = surface_topomesh.wisp_property('barycenter',0).values(surface_topomesh.wisp_property('vertices',2).values())

        surface = True

    start_time = time()
    print "--> Creating vertices"

    for i,t in enumerate(image_tetrahedra):
        pid = image_topomesh.add_wisp(0)
        if 1 in t:
            if exterior:
                vertex_triangle = t[1:]
                triangle_matching = []
                for tetra_p in xrange(4):
                    triangle_matching.append(np.array(list(vq(np.concatenate([image_tetrahedra[:,:tetra_p],image_tetrahedra[:,tetra_p+1:]],axis=1),np.array([vertex_triangle])))))
                vertex_other_tetra = image_tetrahedra[np.where(np.array(triangle_matching)[:,1]==0)[1]]
                if len(vertex_other_tetra) > 0:
                    vertex_other_tetra = [other_tetra for other_tetra in vertex_other_tetra if not 1 in other_tetra][0]
                    vertex_other_tetra_point = [p for p in vertex_other_tetra if not p in t][0]
                vertex_triangle_barycenter = positions.values(t[1:]).mean(axis=0)

                vertex_triangle_normal = np.cross(positions[t[2]]-positions[t[1]],positions[t[3]]-positions[t[1]])/(np.linalg.norm(positions[t[2]]-positions[t[1]])*np.linalg.norm(positions[t[3]]-positions[t[1]]))
                if len(vertex_other_tetra)>0:
                    vertex_triangle_normal = np.sign(np.dot(vertex_triangle_normal,vertex_triangle_barycenter-positions[vertex_other_tetra_point]))*vertex_triangle_normal
                else:
                    vertex_triangle_normal = np.sign(np.dot(vertex_triangle_normal,vertex_triangle_barycenter-tetrahedra_center))*vertex_triangle_normal
                # exterior_position = vertex_triangle_barycenter + (vertex_triangle_barycenter - positions[vertex_other_tetra_point])
                vertex_triangle_normal = vertex_triangle_normal/np.linalg.norm(vertex_triangle_normal)

                positions[1] = vertex_triangle_barycenter + neighbor_distance*np.sqrt(3./2.)*vertex_triangle_normal
                # print np.linalg.norm(neighbor_distance*np.sqrt(3./2.)*vertex_triangle_normal)
                # positions[1] = vertex_triangle_barycenter + neighbor_distance*vertex_triangle_normal
                # positions[1] = vertex_triangle_barycenter + neighbor_distance*np.sqrt(2./3.)*vertex_triangle_normal
                if voronoi:
                    tetra_triangles = t[tetra_triangle_list]
                    tetra_triangle_edges = tetra_triangles[:,triangle_edge_list]
                    
                    # if False:
                    if anisotropic:
                        tetra_triangle_edge_vectors = positions.values(tetra_triangle_edges[...,1])-positions.values(tetra_triangle_edges[...,0])
                        tetra_triangle_edge_lengths = np.sqrt(np.einsum("...ij,...ij->...i",np.einsum("...i,...ij->...j",tetra_triangle_edge_vectors,anisotropy_invert_tensors.values(tetra_triangles)),tetra_triangle_edge_vectors))
                        
                        tetra_triangle_adjacent_edges = tetra_triangles[:,triangle_adjacent_edge_list]
                        tetra_triangle_adjacent_edge_vectors = positions.values(tetra_triangle_adjacent_edges[...,1])-positions.values(tetra_triangle_adjacent_edges[...,0])

                        tetra_triangle_combined_vectors = tetra_triangle_adjacent_edge_vectors.sum(axis=2)
                        tetra_triangle_combined_lengths = np.sqrt(np.einsum("...ij,...ij->...i",np.einsum("...i,...ij->...j",tetra_triangle_combined_vectors,anisotropy_invert_tensors.values(tetra_triangles)),tetra_triangle_combined_vectors))

                        tetra_triangle_edge_lengths = tetra_triangle_combined_lengths

                        tetra_triangle_perimeters = tetra_triangle_edge_lengths.sum(axis=1)
                        tetra_triangle_areas = np.sqrt((tetra_triangle_perimeters/2.0)*(tetra_triangle_perimeters/2.0-tetra_triangle_edge_lengths[...,0])*(tetra_triangle_perimeters/2.0-tetra_triangle_edge_lengths[...,1])*(tetra_triangle_perimeters/2.0-tetra_triangle_edge_lengths[...,2]))
                        tetra_triangle_areas[np.where(np.isnan(tetra_triangle_areas))] = 0
                        tetra_area = np.sum(tetra_triangle_areas)

                        vertex_positions[pid] = (positions.values(t)*tetra_triangle_areas[:,np.newaxis]).sum(axis=0)/tetra_area
                    else:
                        cayley_menger_edges = np.array([[(p1,p2) for p2 in t] for p1 in t])
                        cayley_menger_edge_lengths = np.linalg.norm(positions.values(cayley_menger_edges[...,1])-positions.values(cayley_menger_edges[...,0]),axis=2)

                        cayley_menger_matrix = np.array([[0,1,1,1,1],
                                                         [1]+list(np.power(cayley_menger_edge_lengths[0],2)),
                                                         [1]+list(np.power(cayley_menger_edge_lengths[1],2)),
                                                         [1]+list(np.power(cayley_menger_edge_lengths[2],2)),
                                                         [1]+list(np.power(cayley_menger_edge_lengths[3],2))])
                        
                        try:
                            center_weights = np.linalg.inv(cayley_menger_matrix)[0,1:]
                        except:
                            center_weights = np.array([-1,-1,-1,-1])

                        if center_weights.min()>0:
                            vertex_positions[pid] = (positions.values(t)*center_weights[:,np.newaxis]).sum(axis=0)
                        else:
                            opposite_triangle = tetra_triangles[np.argmin(center_weights)]
                            opposite_triangle_edges = opposite_triangle[triangle_edge_list]
                            opposite_triangle_sinuses = triangle_geometric_features(np.array([opposite_triangle]),positions,features=['sinus'])[0][0]
                            opposite_triangle_cosinuses = triangle_geometric_features(np.array([opposite_triangle]),positions,features=['cosinus'])[0][0]
                        
                            opposite_triangle_sincos = (opposite_triangle_sinuses*opposite_triangle_cosinuses)/(opposite_triangle_sinuses*opposite_triangle_cosinuses).sum()
                            if opposite_triangle_sincos.min()>0:
                                vertex_positions[pid] = (positions.values(t[1:])*opposite_triangle_sincos[:,np.newaxis]).sum(axis=0)
                            else:
                                projection_edge = opposite_triangle_edges[np.where(opposite_triangle_sincos<=0)][0]
                                vertex_positions[pid] = positions.values(projection_edge).mean(axis=0)
                else:
                    vertex_positions[pid] = positions.values(t).mean(axis=0)
            else:
                if voronoi:
                    try:
                        triangle_edges = t[1:][triangle_edge_list]
                        if anisotropic:
                            triangle_edge_vectors = positions.values(triangle_edges[...,1])-positions.values(triangle_edges[...,0])
                            triangle_edge_lengths = np.array([np.sqrt(np.einsum("...ij,...ij->...i",np.einsum("...i,...ij->...j",triangle_edge_vectors,anisotropy_invert_tensors.values([t[k],t[k],t[k]])),triangle_edge_vectors)) for k in xrange(1,4)])

                            triangle_adjacent_edges = t[1:][triangle_adjacent_edge_list]
                            triangle_adjacent_edge_vectors = positions.values(triangle_adjacent_edges[...,1])-positions.values(triangle_adjacent_edges[...,0])

                            triangle_combined_vectors = triangle_adjacent_edge_vectors.sum(axis=1)
                            triangle_combined_lengths = np.sqrt(np.einsum("...ij,...ij->...i",np.einsum("...i,...ij->...j",triangle_combined_vectors,anisotropy_invert_tensors.values(t[1:])),triangle_combined_vectors))

                            triangle_edge_lengths = triangle_combined_lengths
                        else:
                            triangle_edge_lengths = np.linalg.norm(positions.values(triangle_edges[...,1])-positions.values(triangle_edges[...,0]),axis=1)

                        triangle_perimeter = triangle_edge_lengths.sum()
                        vertex_positions[pid] = (positions.values(t[1:])*triangle_edge_lengths[:,np.newaxis]).sum(axis=0)/triangle_perimeter

                        triangle_sinuses = triangle_geometric_features(np.array([t[1:]]),positions,features=['sinus'])[0][0]
                        triangle_cosinuses = triangle_geometric_features(np.array([t[1:]]),positions,features=['cosinus'])[0][0]
                        
                        triangle_tangents = np.array(triangle_sinuses/triangle_cosinuses,np.float64)
                        # triangle_tangents[np.isinf(triangle_tangents)] = np.sign(triangle_tangents[np.isinf(triangle_tangents)])*np.power(10.,10)


                        # triangle_sinus_sums = triangle_sinuses[...,triangle_edge_list].sum(axis=1)/triangle_sinuses[...,triangle_edge_list].sum()
                        
                        triangle_tangent_sums = triangle_tangents[...,triangle_edge_list].sum(axis=1)/triangle_tangents[...,triangle_edge_list].sum()
                        # # print triangle_tangent_sums, np.maximum(triangle_tangent_sums,0)/np.maximum(triangle_tangent_sums,0).sum()
                        triangle_tangent_sums = np.maximum(triangle_tangent_sums,-1)/np.maximum(triangle_tangent_sums,-1).sum()

                        triangle_sincos = (triangle_sinuses*triangle_cosinuses)/(triangle_sinuses*triangle_cosinuses).sum()

                        # vertex_positions[pid] = (positions.values(t[1:])*triangle_sinus_sums[:,np.newaxis]).sum(axis=0)
                        # vertex_positions[pid] = (positions.values(t[1:])*triangle_tangent_sums[:,np.newaxis]).sum(axis=0)
                        vertex_positions[pid] = (positions.values(t[1:])*triangle_sincos[:,np.newaxis]).sum(axis=0)

                        if triangle_sincos.min()<0:
                            projection_edge = triangle_edges[np.where(triangle_sincos<0)][0]
                            vertex_positions[pid] = positions.values(projection_edge).mean(axis=0)
                            # projection_vector = positions[projection_edge[1]] - positions[projection_edge[0]]
                            # vertex_vector = vertex_positions[pid] - positions[projection_edge[0]]
                            # vertex_positions[pid] = positions[projection_edge[0]] + np.dot(vertex_vector,projection_vector)/(np.linalg.norm(projection_vector))*(projection_vector/(np.linalg.norm(projection_vector)))
                        
                        #vertex_positions[pid] = (5.*vertex_positions[pid] + positions.values(t[1:]).mean(axis=0))/6.
                        if surface:
                            vertex_triangle = t[1:]
                            triangle_matching = []
                            for tetra_p in xrange(4):
                                triangle_matching.append(np.array(list(vq(np.concatenate([image_tetrahedra[:,:tetra_p],image_tetrahedra[:,tetra_p+1:]],axis=1),np.array([vertex_triangle])))))
                            vertex_other_tetra = image_tetrahedra[np.where(np.array(triangle_matching)[:,1]==0)[1]]
                            if len(vertex_other_tetra) > 0:
                                vertex_other_tetra = [other_tetra for other_tetra in vertex_other_tetra if not 1 in other_tetra][0]
                                vertex_other_tetra_point = [p for p in vertex_other_tetra if not p in t][0]
                            vertex_triangle_barycenter = positions.values(t[1:]).mean(axis=0)

                            vertex_triangle_normal = np.cross(positions[t[2]]-positions[t[1]],positions[t[3]]-positions[t[1]])/(np.linalg.norm(positions[t[2]]-positions[t[1]])*np.linalg.norm(positions[t[3]]-positions[t[1]]))
                            if len(vertex_other_tetra)>0:
                                vertex_triangle_normal = np.sign(np.dot(vertex_triangle_normal,vertex_triangle_barycenter-positions[vertex_other_tetra_point]))*vertex_triangle_normal
                            else:
                                vertex_triangle_normal = np.sign(np.dot(vertex_triangle_normal,vertex_triangle_barycenter-tetrahedra_center))*vertex_triangle_normal
                            vertex_triangle_normal = vertex_triangle_normal/np.linalg.norm(vertex_triangle_normal)

                            surface_triangle_intersection = intersecting_triangle(np.array([vertex_positions[pid],vertex_positions[pid]+1000.*vertex_triangle_normal]),surface_triangle_points)[0]
                            print surface_triangle_intersection
                            print np.where(surface_triangle_intersection)
                            if surface_triangle_intersection.sum() > 0:
                                surface_triangle = np.where(surface_triangle_intersection)[0][0]
                                surface_triangle_center = surface_triangle_points[surface_triangle].mean(axis=0)
                                surface_distance = np.linalg.norm(surface_triangle_center-vertex_positions[pid])

                                vertex_positions[pid] = vertex_positions[pid] + 0.5*(1-np.tanh(2.0*(surface_distance-5.0)))*(surface_distance*vertex_triangle_normal)

                    except:
                        vertex_positions[pid] = positions.values(t[1:]).mean(axis=0)
                else:
                    vertex_positions[pid] = positions.values(t[1:]).mean(axis=0)
        else:
            if voronoi:
                tetra_triangles = t[tetra_triangle_list]
                tetra_triangle_edges = tetra_triangles[:,triangle_edge_list]
                if anisotropic:
                # if False:
                    tetra_triangle_edge_vectors = positions.values(tetra_triangle_edges[...,1])-positions.values(tetra_triangle_edges[...,0])
                    tetra_triangle_edge_lengths = np.sqrt(np.einsum("...ij,...ij->...i",np.einsum("...i,...ij->...j",tetra_triangle_edge_vectors,anisotropy_invert_tensors.values(tetra_triangles)),tetra_triangle_edge_vectors))
                
                    tetra_triangle_adjacent_edges = tetra_triangles[:,triangle_adjacent_edge_list]
                    tetra_triangle_adjacent_edge_vectors = positions.values(tetra_triangle_adjacent_edges[...,1])-positions.values(tetra_triangle_adjacent_edges[...,0])

                    tetra_triangle_combined_vectors = tetra_triangle_adjacent_edge_vectors.sum(axis=2)
                    tetra_triangle_combined_lengths = np.sqrt(np.einsum("...ij,...ij->...i",np.einsum("...i,...ij->...j",tetra_triangle_combined_vectors,anisotropy_invert_tensors.values(tetra_triangles)),tetra_triangle_combined_vectors))

                    tetra_triangle_edge_lengths = tetra_triangle_combined_lengths

                    tetra_triangle_perimeters = tetra_triangle_edge_lengths.sum(axis=1)
                    tetra_triangle_areas = np.sqrt((tetra_triangle_perimeters/2.0)*(tetra_triangle_perimeters/2.0-tetra_triangle_edge_lengths[...,0])*(tetra_triangle_perimeters/2.0-tetra_triangle_edge_lengths[...,1])*(tetra_triangle_perimeters/2.0-tetra_triangle_edge_lengths[...,2]))
                    tetra_triangle_areas[np.where(np.isnan(tetra_triangle_areas))] = 0
                    tetra_area = np.sum(tetra_triangle_areas)

                    vertex_positions[pid] = (positions.values(t)*tetra_triangle_areas[:,np.newaxis]).sum(axis=0)/tetra_area
                else:
                    try:
                        tetra_triangle_edge_lengths = np.linalg.norm(positions.values(tetra_triangle_edges[...,1])-positions.values(tetra_triangle_edges[...,0]),axis=2)

                        cayley_menger_edges = np.array([[(p1,p2) for p2 in t] for p1 in t])
                        cayley_menger_edge_lengths = np.linalg.norm(positions.values(cayley_menger_edges[...,1])-positions.values(cayley_menger_edges[...,0]),axis=2)

                        cayley_menger_matrix = np.array([[0,1,1,1,1],
                                                         [1]+list(np.power(cayley_menger_edge_lengths[0],2)),
                                                         [1]+list(np.power(cayley_menger_edge_lengths[1],2)),
                                                         [1]+list(np.power(cayley_menger_edge_lengths[2],2)),
                                                         [1]+list(np.power(cayley_menger_edge_lengths[3],2))])
            
                        center_weights = np.linalg.inv(cayley_menger_matrix)[0,1:]

                        if center_weights.min()>0:
                            vertex_positions[pid] = (positions.values(t)*center_weights[:,np.newaxis]).sum(axis=0)
                        else:
                            opposite_triangle = tetra_triangles[np.argmin(center_weights)]
                            opposite_triangle_edges = opposite_triangle[triangle_edge_list]
                            opposite_triangle_sinuses = triangle_geometric_features(np.array([opposite_triangle]),positions,features=['sinus'])[0][0]
                            opposite_triangle_cosinuses = triangle_geometric_features(np.array([opposite_triangle]),positions,features=['cosinus'])[0][0]
                        
                            opposite_triangle_sincos = (opposite_triangle_sinuses*opposite_triangle_cosinuses)/(opposite_triangle_sinuses*opposite_triangle_cosinuses).sum()
                            if opposite_triangle_sincos.min()>0:
                                vertex_positions[pid] = (positions.values(t[1:])*opposite_triangle_sincos[:,np.newaxis]).sum(axis=0)
                            else:
                                projection_edge = opposite_triangle_edges[np.where(opposite_triangle_sincos<=0)][0]
                                vertex_positions[pid] = positions.values(projection_edge).mean(axis=0)
                    except:
                        vertex_positions[pid] = positions.values(t).mean(axis=0)
            else:
                vertex_positions[pid] = positions.values(t).mean(axis=0)

    end_time = time()
    print "<-- Creating vertices      [",end_time-start_time,"]"

    start_time = time()
    print "--> Creating edges"
    image_triangles = np.concatenate(image_tetrahedra[:,tetra_triangle_list ],axis=0)
    _,unique_triangles = np.unique(np.ascontiguousarray(image_triangles).view(np.dtype((np.void,image_triangles.dtype.itemsize * image_triangles.shape[1]))),return_index=True)
    image_triangles = image_triangles[unique_triangles]

    tetrahedra_triangle_matching = np.array(np.concatenate([vq(image_tetrahedra[:,1:],image_triangles),
                                                            vq(np.concatenate([image_tetrahedra[:,0][:,np.newaxis],image_tetrahedra[:,2:]],axis=1),image_triangles),
                                                            vq(np.concatenate([image_tetrahedra[:,:2],image_tetrahedra[:,-1][:,np.newaxis]],axis=1),image_triangles),  
                                                            vq(image_tetrahedra[:,:-1],image_triangles)],axis=1),int)

    for i,t in enumerate(image_triangles):
        edge_vertices = np.unique((np.where((tetrahedra_triangle_matching [1]==0)&(tetrahedra_triangle_matching [0]==i))[0])%image_tetrahedra.shape[0])
        if len(edge_vertices)==2:
            eid = image_topomesh.add_wisp(1,i)
            image_topomesh.link(1,eid,edge_vertices[0])
            image_topomesh.link(1,eid,edge_vertices[1])
    end_time = time()
    print "<-- Creating edges         [",end_time-start_time,"]"

    start_time = time()
    print "--> Creating facets"
    image_edges = np.concatenate(image_triangles[:,triangle_edge_list ],axis=0)
    _,unique_edges = np.unique(np.ascontiguousarray(image_edges).view(np.dtype((np.void,image_edges.dtype.itemsize * image_edges.shape[1]))),return_index=True)
    image_edges = image_edges[unique_edges]

    triangle_edge_matching = np.array(np.concatenate([vq(image_triangles[np.array(list(image_topomesh.wisps(1))),1:],image_edges),
                                                      vq(np.concatenate([image_triangles[np.array(list(image_topomesh.wisps(1))),:1],image_triangles[np.array(list(image_topomesh.wisps(1))),2:]],axis=1),image_edges),
                                                      vq(image_triangles[np.array(list(image_topomesh.wisps(1))),:-1],image_edges)],axis=1),int)

    for i,t in enumerate(image_edges):
        # facet_edges = np.unique((np.where((triangle_edge_matching[1]==0)&(triangle_edge_matching[0]==i))[0])%image_triangles.shape[0])
        facet_edges = np.array(list(image_topomesh.wisps(1)))[np.unique((np.where((triangle_edge_matching[1]==0)&(triangle_edge_matching[0]==i))[0])%image_topomesh.nb_wisps(1))]
        fid = image_topomesh.add_wisp(2,i)
        for eid in facet_edges:
            image_topomesh.link(2,fid,eid)
    end_time = time()
    print "<-- Creating facets        [",end_time-start_time,"]"

    start_time = time()
    print "--> Creating cells"
    image_points = np.unique(image_tetrahedra)[1:]

    edge_point_matching = np.array(np.concatenate([vq(image_edges[:,0],image_points),vq(image_edges[:,1],image_points)],axis=1),int)
    for i,t in enumerate(image_points):
        cell_facets = np.unique((np.where((edge_point_matching[1]==0)&(edge_point_matching[0]==i))[0])%image_edges.shape[0])
        if len(cell_facets)>0:
            cid = image_topomesh.add_wisp(3,t)
            for fid in cell_facets:
                image_topomesh.link(3,cid,fid)
    end_time = time()
    print "<-- Creating cells         [",end_time-start_time,"]"

    image_topomesh.update_wisp_property('barycenter',0,values=vertex_positions)
    return image_topomesh


def tetrahedra_dual_triangular_topomesh(triangulation_topomesh,image_cell_vertex=None,triangular='star_split_regular_flat',**kwargs):

    compute_topomesh_property(triangulation_topomesh,'vertices',3)
    triangulation_tetrahedra = np.sort(triangulation_topomesh.wisp_property('vertices',3).values())
    positions = triangulation_topomesh.wisp_property('barycenter',0)

    voronoi = kwargs.get('voronoi',True)
    exterior = kwargs.get('exterior',False)

    surface_topomesh = kwargs.get('surface_topomesh',None)

    vertex_motion = kwargs.get('vertex_motion',False)
    resolution = kwargs.get('resolution',(1.0,1.0,1.0))

    image_topomesh = tetrahedra_dual_topomesh(triangulation_tetrahedra,positions,voronoi=voronoi,exterior=exterior,surface_topomesh=surface_topomesh)

    compute_topomesh_property(image_topomesh,'length',degree=1)
    compute_topomesh_property(image_topomesh,'vertices',degree=1)
    compute_topomesh_property(image_topomesh,'barycenter',degree=1)
    compute_topomesh_property(image_topomesh,'epidermis',degree=1)
    compute_topomesh_property(image_topomesh,'vertices',degree=3)
    compute_topomesh_property(image_topomesh,'cells',degree=2)
    compute_topomesh_property(image_topomesh,'barycenter',degree=3)
    compute_topomesh_property(image_topomesh,'epidermis',degree=0)
    compute_topomesh_property(image_topomesh,'vertices',degree=3)
    compute_topomesh_property(image_topomesh,'border_neighbors',degree=3)
    compute_topomesh_property(image_topomesh,'barycenter',degree=2)
    compute_topomesh_property(image_topomesh,'vertices',degree=2)
    compute_topomesh_property(image_topomesh,'borders',degree=2)
    compute_topomesh_property(image_topomesh,'regions',degree=2)

    if image_cell_vertex is not None and vertex_motion:
        compute_topomesh_property(image_topomesh,'cells',0)
        mesh_vertices = np.sort([image_topomesh.wisp_property('cells',0)[v] if (len(image_topomesh.wisp_property('cells',0)[v])==4) else np.concatenate([[1],image_topomesh.wisp_property('cells',0)[v]]) for v in image_topomesh.wisps(0) if len(image_topomesh.wisp_property('cells',0)[v]) in [3,4]])
        mesh_cell_vertex = array_dict([v for v in image_topomesh.wisps(0) if len(image_topomesh.wisp_property('cells',0)[v]) in [3,4]],keys=mesh_vertices).to_dict()
        
        cell_vertex_matching = vq(np.sort(np.array(image_cell_vertex.keys())),mesh_vertices)
        
        matched_image_index = np.where(cell_vertex_matching[1] == 0)[0]
        matched_mesh_index = cell_vertex_matching[0][matched_image_index]
        
        matched_image_cell_vertex = np.array(image_cell_vertex.values())[matched_image_index]
        matched_keys = np.sort(np.array(image_cell_vertex.keys()))[matched_image_index]
        
        matched_mesh_vertices = np.array([mesh_cell_vertex[tuple(p)] for p in mesh_vertices[matched_mesh_index]])
        matched_keys = np.sort(np.array(mesh_cell_vertex.keys()))[matched_mesh_index]
        
        initial_vertex_positions = array_dict(image_topomesh.wisp_property('barycenter',0).values(list(image_topomesh.wisps(0))),list(image_topomesh.wisps(0)))
        final_vertex_positions = {}
        
        for i,v in enumerate(matched_mesh_vertices):
            if not np.isnan(matched_image_cell_vertex[i]).any():
                final_vertex_positions[v] = matched_image_cell_vertex[i]
        matched_mesh_vertices = final_vertex_positions.keys()
        
        for v in matched_mesh_vertices:
            # print image_topomesh.wisp_property('barycenter',0)[v]," -> ",final_vertex_positions[v]
            image_topomesh.wisp_property('barycenter',0)[v] = final_vertex_positions[v]

    if len(triangular) == 0 or triangular is None:
        return image_topomesh
    else:
        from openalea.mesh.property_topomesh_optimization import property_topomesh_edge_split_optimization, property_topomesh_edge_flip_optimization
        from openalea.draco_stem.stem.tissue_mesh_optimization import optimize_topomesh


        if 'star' in triangular:
            image_triangular_topomesh = star_interface_topomesh(image_topomesh,inner_interfaces = True)
        elif 'delaunay' in triangular:
            interface_normals = {}
            for fid in image_topomesh.wisps(2):
                if image_topomesh.nb_regions(2,fid) == 2:
                    interface_cell_positions = positions.values(list(image_topomesh.regions(2,fid)))
                    interface_normals[fid] = interface_cell_positions[1] - interface_cell_positions[0]
                elif image_topomesh.nb_regions(2,fid) == 1:
                    interface_normals[fid] = triangulation_topomesh.wisp_property('normal',0)[list(image_topomesh.regions(2,fid))[0]]
                interface_normals[fid] = interface_normals[fid]/np.linalg.norm(interface_normals[fid])
            image_topomesh.update_wisp_property('normal',2,interface_normals)

            image_triangular_topomesh = delaunay_interface_topomesh(image_topomesh,inner_interfaces = True)
        
        # if 'flat' in triangular:
        #     image_triangular_topomesh = optimize_topomesh(image_triangular_topomesh,omega_forces=dict([('taubin_smoothing',0.65),('planarization',0.1),('epidermis_planarization',0.1)]),omega_regularization_max=0.0,gradient_derivatives=None,cell_vertex_motion=False,image_cell_vertex=image_cell_vertex,edge_flip=False,display=False,iterations=10,iterations_per_step=1)   

        vertex_preserving = 'exact' in triangular

        if 'split' in triangular:
            image_triangular_topomesh = topomesh_triangle_split(image_triangular_topomesh)
        elif 'remeshed' in triangular:
            n_flips = image_triangular_topomesh.nb_wisps(1)
            n_splits = image_triangular_topomesh.nb_wisps(1)
            maximal_length = kwargs.get('maximal_length',None)
            if maximal_length is None:
                compute_topomesh_property(image_triangular_topomesh,'length',1)
                target_length = np.percentile(image_triangular_topomesh.wisp_property('length',1).values(),50)
                maximal_length = 4./3. * target_length
            iterations = 0
            max_iterations = 10
            while (n_flips+n_splits > image_triangular_topomesh.nb_wisps(1)/100.) and (iterations<max_iterations):
                n_splits = property_topomesh_edge_split_optimization(image_triangular_topomesh, maximal_length=maximal_length, iterations=1)
                n_flips = property_topomesh_edge_flip_optimization(image_triangular_topomesh,omega_energies=dict([('neighborhood',0.65)]),simulated_annealing=False,iterations=3)
                image_triangular_topomesh = optimize_topomesh(image_triangular_topomesh,omega_forces=dict([('taubin_smoothing',0.33)]),omega_regularization_max=0.0,gradient_derivatives=None,image_resolution=resolution,cell_vertex_motion=vertex_preserving,image_cell_vertex=image_cell_vertex,edge_flip=False,display=False,iterations=10-iterations,iterations_per_step=1)
                iterations += 1
        
        if 'realistic' in triangular:
            image_triangular_topomesh = optimize_topomesh(image_triangular_topomesh,omega_forces=dict([('taubin_smoothing',0.65)]),omega_regularization_max=0.0,gradient_derivatives=None,image_resolution=resolution,cell_vertex_motion=vertex_preserving,image_cell_vertex=image_cell_vertex,edge_flip=False,display=False,iterations=20,iterations_per_step=1)   
        elif 'regular' in triangular:
            image_triangular_topomesh = optimize_topomesh(image_triangular_topomesh,omega_forces=dict([('taubin_smoothing',0.65),('convexity',0.02)]),omega_regularization_max=0.0,gradient_derivatives=None,image_resolution=resolution,cell_vertex_motion=False,image_cell_vertex=image_cell_vertex,edge_flip=False,display=False,iterations=50,iterations_per_step=1)   

        if 'projected' in triangular and surface_topomesh is not None:
            compute_topomesh_property(image_triangular_topomesh,'normal',2,object_positions=triangulation_topomesh.wisp_property('barycenter',0))
            compute_topomesh_property(image_triangular_topomesh,'normal',0)
            compute_topomesh_property(image_triangular_topomesh,'epidermis',0)

            surface_triangle_points = surface_topomesh.wisp_property('barycenter',0).values(surface_topomesh.wisp_property('vertices',2).values())
            vertex_positions = image_triangular_topomesh.wisp_property('barycenter',0)
            vertex_normals = image_triangular_topomesh.wisp_property('normal',0)

            projected_positions = {}
            surface_triangle_intersection = np.array([intersecting_triangle(np.array([vertex_positions[v]-vertex_normals[v],vertex_positions[v]+1000.*vertex_normals[v]]),surface_triangle_points)[0] for v in image_triangular_topomesh.wisps(0)])
            projectable_vertices, projected_triangles = np.where(surface_triangle_intersection)
            for v, t in zip(projectable_vertices, projected_triangles):
                surface_triangle_center = surface_triangle_points[t].mean(axis=0)
                surface_distance = np.linalg.norm(surface_triangle_center-vertex_positions[v])
                projected_positions[v] = vertex_positions[v] + 0.5*(1-np.tanh(2.0*(surface_distance-5.0)))*(surface_distance*vertex_normals[v])
            image_triangular_topomesh.update_wisp_property('barycenter',0,array_dict([projected_positions[v] if projected_positions.has_key(v) else vertex_positions[v] for v in image_triangular_topomesh.wisps(0)],list(image_triangular_topomesh.wisps(0))))

            image_triangular_topomesh = optimize_topomesh(image_triangular_topomesh,omega_forces=dict([('taubin_smoothing',0.65)]),omega_regularization_max=0.0,gradient_derivatives=None,image_resolution=resolution,cell_vertex_motion=vertex_preserving,image_cell_vertex=image_cell_vertex,edge_flip=False,display=False,iterations=10,iterations_per_step=1)   
        
        if 'flat' in triangular:
            image_triangular_topomesh = optimize_topomesh(image_triangular_topomesh,omega_forces=dict([('taubin_smoothing',0.65),('planarization',0.1),('epidermis_planarization',0.1)]),omega_regularization_max=0.0,gradient_derivatives=None,cell_vertex_motion=vertex_preserving,image_cell_vertex=image_cell_vertex,edge_flip=False,display=False,iterations=10,iterations_per_step=1)   
        elif 'straight' in triangular:
            image_triangular_topomesh = optimize_topomesh(image_triangular_topomesh,omega_forces=dict([('taubin_smoothing',0.65),('laplacian',0.5),('epidermis_planarization',0.1)]),omega_regularization_max=0.0,gradient_derivatives=None,cell_vertex_motion=True,image_cell_vertex=image_cell_vertex,edge_flip=False,display=False,iterations=10,iterations_per_step=1)   
            image_triangular_topomesh = optimize_topomesh(image_triangular_topomesh,omega_forces=dict([('taubin_smoothing',0.65)]),omega_regularization_max=0.0,gradient_derivatives=None,image_resolution=resolution,cell_vertex_motion=vertex_preserving,image_cell_vertex=image_cell_vertex,edge_flip=False,display=False,iterations=3,iterations_per_step=1)   

        try:
            return image_triangular_topomesh
        except:
            return image_topomesh


def triangulated_interface_topomesh(image_topomesh,maximal_length,inner_interfaces=True):
    from openalea.mesh.utils.delaunay_tools import delaunay_triangulation

    triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])

    image_triangular_topomesh = PropertyTopomesh(3)
    triangle_vertex_positions = {}
    triangle_edge_vertices = {}

    # maximal_length = np.power(img_volumes.mean(),1/3.)/np.pi

    for v in image_topomesh.wisps(0):
        image_triangular_topomesh.add_wisp(0,v)
        triangle_vertex_positions[v] = image_topomesh.wisp_property('barycenter',0)[v]

    for c in image_topomesh.wisps(3):
        image_triangular_topomesh.add_wisp(3,c)

    start_time = time()
    print "--> Triangulating Interfaces"
    for interface in image_topomesh.wisps(2):
        if interface%100 == 0:
            interface_start_time = time()

        interface_cells = image_topomesh.wisp_property('regions',2)[interface]
        interface_edges = image_topomesh.wisp_property('vertices',1).values(image_topomesh.wisp_property('borders',2)[interface])
        interface_vertices = np.unique(interface_edges)

        if (len(interface_vertices)>2) and (inner_interfaces or (len(interface_cells) == 1)):
        # if (len(interface_vertices)>2):

            interface_positions = array_dict(image_topomesh.wisp_property('barycenter',0).values(interface_vertices),interface_vertices)
        
            interface_center = interface_positions.values().mean(axis=0)
        
            interface_extension = 1*(image_topomesh.wisp_property('barycenter',0).values(interface_vertices) - interface_center)
            interface_pids = []

            for v in interface_vertices:
                interface_pids.append(v)


            interface_covariance = np.cov(np.transpose(interface_positions.values(interface_vertices)))
            interface_eig = np.linalg.eig(interface_covariance)
            sorted_values = np.argsort(interface_eig[0])
            interface_eigenvectors = np.transpose(interface_eig[1])[sorted_values]

            interface_vertices_vectors = interface_positions.values(interface_vertices) - interface_center
            interface_projectors = -np.einsum('ij,ij->i',interface_vertices_vectors,interface_eigenvectors[0][np.newaxis,:])[:,np.newaxis]*interface_eigenvectors[0][np.newaxis,:]
            
            interface_vertex_positions = array_dict(interface_positions.values(interface_vertices)+interface_projectors,interface_vertices)

            projected_vertex_positions = np.transpose([np.einsum('ij,ij->i',interface_vertices_vectors,interface_eigenvectors[2][np.newaxis,:]),np.einsum('ij,ij->i',interface_vertices_vectors,interface_eigenvectors[1][np.newaxis,:]),np.zeros_like(interface_vertices)])
            projected_vertex_positions = array_dict(projected_vertex_positions,interface_vertices)

            interface_points = list(interface_positions.values(interface_vertices))
            projected_points = list(projected_vertex_positions.values(interface_vertices)) 

            interface_edges = image_topomesh.wisp_property('vertices',1).values(image_topomesh.wisp_property('borders',2)[interface])
            interface_edge_lengths = np.linalg.norm(interface_positions.values(interface_edges[:,1]) - interface_positions.values(interface_edges[:,0]),axis=1)

            interface_triangulation_edges = []

            for e,l in zip(interface_edges,interface_edge_lengths):
                prev_p = interface_positions[e[0]]
                prev_proj_p = projected_vertex_positions[e[0]]
                prev_index = vq(np.array([e[0]]),interface_vertices)[0][0]
                end_reached = False
                while not end_reached:
                    next_p = prev_p + maximal_length*(interface_positions[e[1]]-interface_positions[e[0]])/l
                    next_proj_p = prev_proj_p + maximal_length*(projected_vertex_positions[e[1]]-projected_vertex_positions[e[0]])/l
                    if np.linalg.norm(interface_positions[e[1]]-next_p)>maximal_length/2. and np.linalg.norm(interface_positions[e[0]]-next_p)<l:
                        interface_triangulation_edges.append(np.sort([prev_index,len(interface_points)]))
                        
                        mesh_vertex_matching = vq(np.array([next_p]),np.array(triangle_vertex_positions.values()))
                        if mesh_vertex_matching[1][0] == 0:
                            interface_pids.append(triangle_vertex_positions.keys()[mesh_vertex_matching[0][0]])
                        else:
                            pid = image_triangular_topomesh.add_wisp(0)
                            triangle_vertex_positions[pid] = next_p
                            interface_pids.append(pid)

                        prev_index = len(interface_points)
                        interface_points.append(next_p)
                        prev_p = next_p
                        projected_points.append(next_proj_p)
                        prev_proj_p = next_proj_p
                    else:
                        interface_triangulation_edges.append(np.sort([prev_index,vq(np.array([e[1]]),interface_vertices)[0][0]]))
                        end_reached = True

            interface_border_edges = copy(interface_triangulation_edges)

            interface_projected_borders = np.array(projected_points)[:,:2][np.array(interface_border_edges)]
            intersecting_borders = False
            for b in interface_projected_borders:
                intersecting_borders = intersecting_borders or (intersecting_segment(b,interface_projected_borders).sum()>0)

            spherical_triangulation = False

            if intersecting_borders:
                spherical_triangulation = True

                c = -1
                while intersecting_borders and (c<len(interface_cells)-1):
                    c = c+1

                    cell_center = image_topomesh.wisp_property('barycenter',3)[interface_cells[c]]

                    sphere_z = (interface_center-cell_center)/np.linalg.norm(interface_center-cell_center)
                    sphere_x = np.array([sphere_z[2]-sphere_z[1],sphere_z[0]-sphere_z[2],sphere_z[1]-sphere_z[0]])
                    sphere_x = sphere_x/np.linalg.norm(sphere_x)
                    sphere_y = np.cross(sphere_x,sphere_z)

                    sphere_axes = np.array([sphere_z,sphere_y,sphere_x])

                    rho = np.linalg.norm(np.array(interface_points)-cell_center,axis=1)
                    phi = np.arccos(np.array(np.dot(np.array(interface_points)-cell_center,sphere_axes[2])/rho,np.float16))
                    theta = np.arccos(np.array(np.dot(np.array(interface_points)-cell_center,sphere_axes[1])/(rho*np.sin(phi)),np.float16))
                    theta = theta-np.pi/2.
                    theta[np.where(np.isnan(theta))] = 0.
                    phi = phi-np.pi/2

                    spherical_points = np.zeros_like(np.array(interface_points))
                    spherical_points[...,0] = 10*theta
                    spherical_points[...,1] = 10*np.log(np.tan(np.pi/4.+phi/2.))
                    spherical_points[...,2] = 0
                    spherical_points = list(spherical_points)

                    interface_projected_borders = np.array(spherical_points)[:,:2][np.array(interface_border_edges)]
                    intersecting_borders = False
                    for b in interface_projected_borders:
                        intersecting_borders = intersecting_borders or (intersecting_segment(b,interface_projected_borders).sum()>0)

            interface_triangles = np.array([np.concatenate([interface_vertex_positions.values(e),[interface_center]]) for e in interface_edges])
            inside_vertex_index = len(interface_points)

            points = [interface_center]
            proj_points = [[0.,0.,0.]]
            processed_points = []

            points_distance = [vq(np.array([p]),np.concatenate([np.array(proj_points[:i]).reshape(i,3),np.array(proj_points[i+1:]).reshape(len(points)-i-1,3)]))[1][0] for i,p in enumerate(proj_points)]
            points_order = np.argsort(-np.array(points_distance))
            points = list(np.array(points)[points_order])
            proj_points = list(np.array(proj_points)[points_order])

            for p,proj_p in zip(points,proj_points):
                vertex_matching = vq(np.array([p]),np.array(interface_points))
                if vertex_matching[1][0] > maximal_length/2.:
                    pid = image_triangular_topomesh.add_wisp(0)
                    triangle_vertex_positions[pid] = p
                    interface_pids.append(pid)
                    interface_points.append(p)
                    projected_points.append(proj_p)
                    if spherical_triangulation:
                        rho = np.linalg.norm(p-cell_center)
                        phi = np.arccos(np.array(np.dot(p-cell_center,sphere_axes[2])/rho,np.float16))
                        theta = np.arccos(np.array(np.dot(p-cell_center,sphere_axes[1])/(rho*np.sin(phi)),np.float16))
                        theta = theta-np.pi/2.
                        phi = phi-np.pi/2
                        sphere_p = np.array([10.*theta,10.*np.log(np.tan(np.pi/4.+phi/2.)),0.])
                        spherical_points.append(sphere_p)

            while len(points)>0:
                p = points.pop(0)
                proj_p = proj_points.pop(0)

                index = vq(np.array([proj_p]),np.array(projected_points))[0][0]

                for i in xrange(6):
                    new_p = p + maximal_length*(np.cos(i*np.pi/3.)*interface_eigenvectors[2] + np.sin(i*np.pi/3.)*interface_eigenvectors[1])
                    proj_new_p = proj_p + maximal_length*np.array([np.cos(i*np.pi/3.),np.sin(i*np.pi/3.),0.])

                    vertex_matching = vq(np.array([proj_new_p]),np.array(projected_points))
                    if inside_triangle(new_p,interface_triangles).sum()>0:
                        if vertex_matching[1][0] > maximal_length/2.:
                            interface_triangulation_edges.append(np.sort([index,len(interface_points)]))
                            pid = image_triangular_topomesh.add_wisp(0)
                            triangle_vertex_positions[pid] = new_p
                            interface_pids.append(pid)
                            interface_points.append(new_p)
                            projected_points.append(proj_new_p)

                            if spherical_triangulation:
                                rho = np.linalg.norm(new_p-cell_center)
                                phi = np.arccos(np.array(np.dot(new_p-cell_center,sphere_axes[2])/rho,np.float16))
                                theta = np.arccos(np.array(np.dot(new_p-cell_center,sphere_axes[1])/(rho*np.sin(phi)),np.float16))
                                theta = theta-np.pi/2.
                                phi = phi-np.pi/2
                                sphere_new_p = np.array([10.*theta,10.*np.log(np.tan(np.pi/4.+phi/2.)),0.])
                                spherical_points.append(sphere_new_p)
                            if (len(processed_points) == 0) or (np.float16(vq(np.array([new_p]),np.array(processed_points))[1][0]) > 0.) :
                                if (len(points) == 0) or (np.float16(vq(np.array([new_p]),np.array(points))[1][0]) > 0.) :
                                    points.append(new_p)
                                    proj_points.append(proj_new_p)
                        else:
                            interface_triangulation_edges.append(np.sort([index,vertex_matching[0][0]]))
                    else:
                        interface_triangulation_edges.append(np.sort([index,vertex_matching[0][0]]))

                points_distance = [vq(np.array([proj_p]),np.concatenate([np.array(proj_points[:i]).reshape(i,3),np.array(proj_points[i+1:]).reshape(len(points)-i-1,3)]))[1][0] for i,proj_p in enumerate(proj_points)]
                points_order = np.argsort(-np.array(points_distance))
                points = list(np.array(points)[points_order])
                proj_points = list(np.array(proj_points)[points_order])

                processed_points.append(p)

            interface_pids = np.array(interface_pids)

            if not spherical_triangulation:
                triangulation_points = projected_points
            else:
                triangulation_points = spherical_points

            interface_projected_triangulation = np.array(triangulation_points)[:,:2][np.array(interface_triangulation_edges)]
            interface_projected_borders = np.array(triangulation_points)[:,:2][np.array(interface_border_edges)]
            interface_triangulation_edges_intersecting_borders = []
            for i,b in enumerate(interface_projected_triangulation):
                if intersecting_segment(b,interface_projected_borders).sum()>0:
                    interface_triangulation_edges_intersecting_borders.append(i)

            suppressed_interface_triangulation_edges = np.array(interface_triangulation_edges)[interface_triangulation_edges_intersecting_borders]
            interface_triangulation_edges = np.delete(interface_triangulation_edges,interface_triangulation_edges_intersecting_borders,0)

            interface_delaunay_triangulation = delaunay_triangulation(triangulation_points)
            interface_triangulation_triangles = np.sort(np.array(interface_delaunay_triangulation))
            if len(interface_triangulation_triangles) == 0:
                interface_triangulation_triangles = np.array([[0,1,2]])

            interface_delaunay_triangulation_edges = np.sort(np.concatenate(interface_triangulation_triangles[:,triangle_edge_list ]))
            _,unique_triangulation_edges = np.unique(np.ascontiguousarray(interface_delaunay_triangulation_edges).view(np.dtype((np.void, interface_delaunay_triangulation_edges.dtype.itemsize * interface_delaunay_triangulation_edges.shape[1]))),return_index=True)
            interface_delaunay_triangulation_edges = interface_delaunay_triangulation_edges[unique_triangulation_edges]

            interface_missing_edges = np.array(interface_border_edges)[np.where(vq(np.array(interface_border_edges),interface_delaunay_triangulation_edges)[1]>0)]
            interface_triangulation_additional_edges = interface_delaunay_triangulation_edges[np.where(vq(interface_delaunay_triangulation_edges,np.array(interface_triangulation_edges))[1]>0)]
            interface_triangulation_edges_to_flip = interface_triangulation_additional_edges[np.sum(interface_triangulation_additional_edges>=inside_vertex_index,axis=1)>0]
                    
            interface_triangulation_flipped_edges = []
            if len(interface_triangulation_edges_to_flip)>0:
                edge_triangles = dict([(i,[]) for i,e in enumerate(interface_triangulation_edges_to_flip)])
                edge_matching = vq(interface_triangulation_triangles[:,1:],interface_triangulation_edges_to_flip)
                for i,t in enumerate(interface_triangulation_triangles):
                    if edge_matching[1][i] == 0:
                        edge_triangles[edge_matching[0][i]].append(i)
                edge_matching = vq(interface_triangulation_triangles[:,:2],interface_triangulation_edges_to_flip)
                for i,t in enumerate(interface_triangulation_triangles):
                    if edge_matching[1][i] == 0:
                        edge_triangles[edge_matching[0][i]].append(i)
                edge_matching = vq(np.concatenate([interface_triangulation_triangles[:,:1],interface_triangulation_triangles[:,2:]],axis=1),interface_triangulation_edges_to_flip)
                for i,t in enumerate(interface_triangulation_triangles):
                    if edge_matching[1][i] == 0:
                        edge_triangles[edge_matching[0][i]].append(i)
                edge_triangles = array_dict(edge_triangles)

                for i,e in enumerate(interface_triangulation_edges_to_flip):
                    edge_quadrilateral = np.unique(np.array([interface_triangulation_triangles[t] for t in edge_triangles[i]]))
                    edge_diagonal = edge_quadrilateral[np.where((edge_quadrilateral!=e[0])&(edge_quadrilateral!=e[1]))]

                    if len(edge_diagonal) == 2:
                        aligned_points = False
                        for j,t in enumerate(edge_triangles[i]):
                            aligned_points = aligned_points | (np.array(np.cross(interface_points[e[j]]-interface_points[edge_diagonal[0]],interface_points[e[j]]-interface_points[edge_diagonal[1]]),np.float16).sum()==0.)

                        intersecting_diagonals = intersecting_segment(np.array(triangulation_points)[:,:2][e],np.array(triangulation_points)[:,:2][edge_diagonal])

                        if (not aligned_points) and (intersecting_diagonals) and ((edge_diagonal>inside_vertex_index).sum() == 0) or (vq(np.array([edge_diagonal]),np.array(interface_triangulation_edges))[1] == 0):
                            for j,t in enumerate(edge_triangles[i]):
                                interface_triangulation_triangles[t] = np.sort(np.concatenate([edge_diagonal,[e[j]]]))
                                interface_triangulation_flipped_edges.append(edge_diagonal)

                            edge_triangles = dict([(i,[]) for i,e in enumerate(interface_triangulation_edges_to_flip)])
                            edge_matching = vq(interface_triangulation_triangles[:,1:],interface_triangulation_edges_to_flip)
                            for i,t in enumerate(interface_triangulation_triangles):
                                if edge_matching[1][i] == 0:
                                    edge_triangles[edge_matching[0][i]].append(i)
                            edge_matching = vq(interface_triangulation_triangles[:,:2],interface_triangulation_edges_to_flip)
                            for i,t in enumerate(interface_triangulation_triangles):
                                if edge_matching[1][i] == 0:
                                    edge_triangles[edge_matching[0][i]].append(i)
                            edge_matching = vq(np.concatenate([interface_triangulation_triangles[:,:1],interface_triangulation_triangles[:,2:]],axis=1),interface_triangulation_edges_to_flip)
                            for i,t in enumerate(interface_triangulation_triangles):
                                if edge_matching[1][i] == 0:
                                    edge_triangles[edge_matching[0][i]].append(i)
                            edge_triangles = array_dict(edge_triangles)

            interface_delaunay_triangulation_edges = np.sort(np.concatenate(interface_triangulation_triangles[:,triangle_edge_list ]))
            _,unique_triangulation_edges = np.unique(np.ascontiguousarray(interface_delaunay_triangulation_edges).view(np.dtype((np.void, interface_delaunay_triangulation_edges.dtype.itemsize * interface_delaunay_triangulation_edges.shape[1]))),return_index=True)
            interface_delaunay_triangulation_edges = interface_delaunay_triangulation_edges[unique_triangulation_edges]

            interface_missing_edges = np.array(interface_border_edges)[np.where(vq(np.array(interface_border_edges),interface_delaunay_triangulation_edges)[1]>0)]

            interface_triangulation_additional_edges = interface_delaunay_triangulation_edges[np.where(vq(interface_delaunay_triangulation_edges,np.array(interface_triangulation_edges))[1]>0)]
            
            interface_triangulation_exterior_edges = interface_triangulation_additional_edges[np.sum(interface_triangulation_additional_edges>inside_vertex_index,axis=1)==0]
            interface_triangles = np.array([np.concatenate([np.array(triangulation_points)[e],[np.array([0,0,0])]]) for e in np.transpose([vq(interface_edges[:,0],interface_pids)[0],vq(interface_edges[:,1],interface_pids)[0]])])
            
            if len(interface_triangulation_exterior_edges)>0:
                interface_triangulation_exterior_edges = interface_triangulation_exterior_edges[np.array([inside_triangle((triangulation_points[e[0]]+triangulation_points[e[1]])/2.,interface_triangles).sum()==0 for e in interface_triangulation_exterior_edges])]

            if len(interface_triangulation_exterior_edges)>0:
                edge_triangles = dict([(i,[]) for i,e in enumerate(interface_triangulation_exterior_edges)])
                edge_matching = vq(interface_triangulation_triangles[:,1:],interface_triangulation_exterior_edges)
                for i,t in enumerate(interface_triangulation_triangles):
                    if edge_matching[1][i] == 0:
                        edge_triangles[edge_matching[0][i]].append(i)
                edge_matching = vq(interface_triangulation_triangles[:,:2],interface_triangulation_exterior_edges)
                for i,t in enumerate(interface_triangulation_triangles):
                    if edge_matching[1][i] == 0:
                        edge_triangles[edge_matching[0][i]].append(i)
                edge_matching = vq(np.concatenate([interface_triangulation_triangles[:,:1],interface_triangulation_triangles[:,2:]],axis=1),interface_triangulation_exterior_edges)
                for i,t in enumerate(interface_triangulation_triangles):
                    if edge_matching[1][i] == 0:
                        edge_triangles[edge_matching[0][i]].append(i)
                edge_triangles = array_dict(edge_triangles)

                exterior_triangles = np.unique(np.concatenate(edge_triangles.values()))
                interface_triangulation_triangles = np.delete(interface_triangulation_triangles,exterior_triangles,0)

            if len(interface_triangulation_triangles)>0:
                interface_delaunay_triangulation_edges = np.sort(np.concatenate(interface_triangulation_triangles[:,triangle_edge_list ]))
                _,unique_triangulation_edges = np.unique(np.ascontiguousarray(interface_delaunay_triangulation_edges).view(np.dtype((np.void, interface_delaunay_triangulation_edges.dtype.itemsize * interface_delaunay_triangulation_edges.shape[1]))),return_index=True)
                interface_delaunay_triangulation_edges = interface_delaunay_triangulation_edges[unique_triangulation_edges]

                interface_triangulation_border_edges = interface_delaunay_triangulation_edges[np.sum(interface_delaunay_triangulation_edges>=inside_vertex_index,axis=1)==0]
                interface_triangulation_additional_border_edges = interface_triangulation_border_edges[np.where(vq(interface_triangulation_border_edges,np.array(interface_border_edges))[1]>0)]

                if len(interface_triangulation_additional_border_edges)>0:
                    edge_triangles = dict([(i,[]) for i,e in enumerate(interface_triangulation_additional_border_edges)])
                    edge_matching = vq(interface_triangulation_triangles[:,1:],interface_triangulation_additional_border_edges)
                    for i,t in enumerate(interface_triangulation_triangles):
                        if edge_matching[1][i] == 0:
                            edge_triangles[edge_matching[0][i]].append(i)
                    edge_matching = vq(interface_triangulation_triangles[:,:2],interface_triangulation_additional_border_edges)
                    for i,t in enumerate(interface_triangulation_triangles):
                        if edge_matching[1][i] == 0:
                            edge_triangles[edge_matching[0][i]].append(i)
                    edge_matching = vq(np.concatenate([interface_triangulation_triangles[:,:1],interface_triangulation_triangles[:,2:]],axis=1),interface_triangulation_additional_border_edges)
                    for i,t in enumerate(interface_triangulation_triangles):
                        if edge_matching[1][i] == 0:
                            edge_triangles[edge_matching[0][i]].append(i)
                    edge_triangles = array_dict(edge_triangles)

                    flat_triangles = []
                    for i,e in enumerate(interface_triangulation_additional_border_edges):
                        for t in edge_triangles[i]:
                            edge_triangulation_triangle = interface_triangulation_triangles[t]
                            triangle_vertex = edge_triangulation_triangle[np.where((edge_triangulation_triangle != e[0])&(edge_triangulation_triangle != e[1]))][0]
                            aligned_points = np.array(np.cross(interface_points[triangle_vertex]-interface_points[e[0]],interface_points[triangle_vertex]-interface_points[e[1]]),np.float16).sum()==0.
                            if aligned_points:
                                flat_triangles.append(t)

                    flat_triangles = np.unique(flat_triangles)
                    interface_triangulation_triangles = np.delete(interface_triangulation_triangles,flat_triangles,0)

                interface_pids_triangles = interface_pids[interface_triangulation_triangles]

                interface_pids_edges = np.sort(interface_pids_triangles[:,triangle_edge_list ])
                interface_fids = []

                for t in interface_pids_edges:
                    fid = image_triangular_topomesh.add_wisp(2)
                    interface_fids.append(fid)
                    for cid in interface_cells:
                        image_triangular_topomesh.link(3,cid,fid)
                    for e in t:
                        edge_matching = vq(np.array([e]),np.array(triangle_edge_vertices.values()).reshape(len(triangle_edge_vertices),2))
                        if edge_matching[1][0] == 0:
                            eid = triangle_edge_vertices.keys()[edge_matching[0][0]]
                            image_triangular_topomesh.link(2,fid,eid)
                        else:
                            eid = image_triangular_topomesh.add_wisp(1)
                            image_triangular_topomesh.link(1,eid,e[0])
                            image_triangular_topomesh.link(1,eid,e[1])
                            triangle_edge_vertices[eid] = e
                            image_triangular_topomesh.link(2,fid,eid)
                
                if interface%100 == 0:
                    interface_end_time = time()
                    # print "  --> Interface ",interface," / ",image_topomesh.nb_wisps(2),' ',interface_cells,'     [',interface_end_time-interface_start_time,'s]'
                    print "  --> Interface ",interface," / ",image_topomesh.nb_wisps(2),'     [',(interface_end_time-interface_start_time),'s]'

    end_time = time()
    print "--> Triangulating Interfaces  [",end_time-start_time,"s]"
    image_triangular_topomesh.update_wisp_property('barycenter',degree=0,values=triangle_vertex_positions)
    return image_triangular_topomesh


def star_interface_topomesh(image_topomesh,inner_interfaces=True):
    triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])

    image_triangular_topomesh = PropertyTopomesh(3)
    triangle_vertex_positions = {}

    for v in image_topomesh.wisps(0):
        image_triangular_topomesh.add_wisp(0,v)
        triangle_vertex_positions[v] = image_topomesh.wisp_property('barycenter',0)[v]

    for e in image_topomesh.wisps(1):
        image_triangular_topomesh.add_wisp(1,e)
        for v in image_topomesh.borders(1,e):
            image_triangular_topomesh.link(1,e,v)

    for c in image_topomesh.wisps(3):
        image_triangular_topomesh.add_wisp(3,c)

    start_time = time()
    print "--> Triangulating Interfaces"
    for interface in image_topomesh.wisps(2):
        if interface%100 == 0:
            interface_start_time = time()

        interface_cells = image_topomesh.wisp_property('regions',2)[interface]
        interface_edges = image_topomesh.wisp_property('vertices',1).values(image_topomesh.wisp_property('borders',2)[interface])
        interface_vertices = np.unique(interface_edges)

        if (len(interface_vertices)>2) and (inner_interfaces or (len(interface_cells) == 1)):

            interface_positions = array_dict(image_topomesh.wisp_property('barycenter',0).values(interface_vertices),interface_vertices)
            interface_center = interface_positions.values().mean(axis=0)

            center_pid = image_triangular_topomesh.add_wisp(0)
            triangle_vertex_positions[center_pid] = interface_center

            vertex_center_edges = {}
            for v in interface_vertices:
                eid = image_triangular_topomesh.add_wisp(1)
                image_triangular_topomesh.link(1,eid,v)
                image_triangular_topomesh.link(1,eid,center_pid)
                vertex_center_edges[v] = eid

            for e in image_topomesh.borders(2,interface):
                fid = image_triangular_topomesh.add_wisp(2)
                image_triangular_topomesh.link(2,fid,e)
                for v in image_topomesh.borders(1,e):
                    image_triangular_topomesh.link(2,fid,vertex_center_edges[v])
                for cid in interface_cells:
                    image_triangular_topomesh.link(3,cid,fid)

            if interface%100 == 0:
                interface_end_time = time()
                # print "  --> Interface ",interface," / ",image_topomesh.nb_wisps(2),' ',interface_cells,'     [',interface_end_time-interface_start_time,'s]'
                print "  --> Interface ",interface," / ",image_topomesh.nb_wisps(2),'     [',(interface_end_time-interface_start_time),'s]'

    end_time = time()
    print "--> Triangulating Interfaces  [",end_time-start_time,"s]"
    image_triangular_topomesh.update_wisp_property('barycenter',degree=0,values=triangle_vertex_positions)
    return image_triangular_topomesh

def delaunay_interface_topomesh(image_topomesh,inner_interfaces=True):    
    triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])

    image_triangular_topomesh = PropertyTopomesh(3)
    triangle_vertex_positions = {}

    for v in image_topomesh.wisps(0):
        image_triangular_topomesh.add_wisp(0,v)
        triangle_vertex_positions[v] = image_topomesh.wisp_property('barycenter',0)[v]

    edge_ids = {}
    for e in image_topomesh.wisps(1):
        image_triangular_topomesh.add_wisp(1,e)
        for v in image_topomesh.borders(1,e):
            image_triangular_topomesh.link(1,e,v)
        edge_ids[tuple(np.sort(list(image_topomesh.borders(1,e))))] = e

    for c in image_topomesh.wisps(3):
        image_triangular_topomesh.add_wisp(3,c)


    compute_topomesh_property(image_topomesh,'barycenter',2)
    compute_topomesh_property(image_topomesh,'vertices',2)
    compute_topomesh_property(image_topomesh,'regions',2)

    interface_vertices = image_topomesh.wisp_property('vertices',2).values()
    interface_points = np.array([image_topomesh.wisp_property('barycenter',0).values(v) for v in interface_vertices])
    interface_centers = image_topomesh.wisp_property('barycenter',2).values()
    interface_normals = image_topomesh.wisp_property('normal',2).values()

        
    def project(points,plane_center,normal_vector):
        import numpy as np
        vectors = points-plane_center
        projectors = -np.einsum('ij,ij->i',vectors,normal_vector[np.newaxis,:])[:,np.newaxis]*normal_vector[np.newaxis,:]
        projection_positions = points+projectors
        plane_vectors = {}
        plane_vectors[0] = np.cross(normal_vector,np.array([1,0,0]))
        plane_vectors[1] = np.cross(normal_vector,plane_vectors[0])
        projected_points = np.transpose([np.einsum('ij,ij->i',vectors,plane_vectors[0][np.newaxis,:]),np.einsum('ij,ij->i',vectors,plane_vectors[1][np.newaxis,:]),np.zeros_like(points[:,2])])
        return projected_points

    projected_interface_points = np.array(map(project,interface_points,interface_centers,interface_normals))

    def array_delaunay(points,indices):
        # from openalea.plantgl.algo import delaunay_triangulation
        from openalea.mesh.utils.delaunay_tools import delaunay_triangulation
        import numpy as np
        if len(indices)>3:
            triangulation = delaunay_triangulation(points)
            if len(triangulation)>0:
                return indices[np.array(triangulation)]
            else:
                return indices[:3][np.newaxis,:]
            
        else:
            return indices[np.newaxis,:]
    
    interface_triangulation = np.array(map(array_delaunay,projected_interface_points,interface_vertices))
    interface_triangulation = dict(zip(list(image_topomesh.wisps(2)),interface_triangulation))

    start_time = time()
    print "--> Triangulating Interfaces"
    for interface in image_topomesh.wisps(2):
        if interface%100 == 0:
            interface_start_time = time()

        interface_cells = image_topomesh.wisp_property('regions',2)[interface]
        interface_edges = image_topomesh.wisp_property('vertices',1).values(image_topomesh.wisp_property('borders',2)[interface])
        interface_vertices = np.unique(interface_edges)

        interface_triangles = interface_triangulation[interface]

        if (len(interface_vertices)>2) and (inner_interfaces or (len(interface_cells) == 1)):

            interface_triangle_edges = interface_triangles[:,triangle_edge_list]
            for triangle_edges in interface_triangle_edges:
                fid = image_triangular_topomesh.add_wisp(2)
                for edge in triangle_edges:
                    if not edge_ids.has_key(tuple(np.sort(edge))):
                        eid = image_triangular_topomesh.add_wisp(1)
                        edge_ids[tuple(np.sort(edge))] = eid 
                        image_triangular_topomesh.link(1,eid,edge[0])
                        image_triangular_topomesh.link(1,eid,edge[1])
                    else:
                        eid = edge_ids[tuple(np.sort(edge))]
                    image_triangular_topomesh.link(2,fid,eid)
                for cid in interface_cells:
                    image_triangular_topomesh.link(3,cid,fid)

            if interface%100 == 0:
                interface_end_time = time()
                # print "  --> Interface ",interface," / ",image_topomesh.nb_wisps(2),' ',interface_cells,'     [',interface_end_time-interface_start_time,'s]'
                print "  --> Interface ",interface," / ",image_topomesh.nb_wisps(2),'     [',(interface_end_time-interface_start_time),'s]'

    end_time = time()
    print "--> Triangulating Interfaces  [",end_time-start_time,"s]"
    image_triangular_topomesh.update_wisp_property('barycenter',degree=0,values=triangle_vertex_positions)
    return image_triangular_topomesh




