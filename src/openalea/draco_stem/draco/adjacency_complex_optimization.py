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

from openalea.container import array_dict, PropertyTopomesh

from openalea.mesh.utils.tissue_analysis_tools import cell_vertex_extraction

from openalea.mesh.property_topomesh_analysis import *
from openalea.mesh.property_topomesh_extraction import clean_topomesh

from openalea.mesh.utils.intersection_tools import inside_triangle, intersecting_segment, intersecting_triangle
from openalea.mesh.utils.evaluation_tools import jaccard_index
from openalea.mesh.utils.array_tools import array_unique
from openalea.mesh.utils.geometry_tools import tetra_geometric_features, triangle_geometric_features

from sys                                    import argv
from time                                   import time, sleep
from copy                                   import deepcopy

tetra_triangle_edge_list  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
tetra_triangle_list  = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])


def triangles_from_adjacency_edges(edges, positions=None):
    edges = np.sort(edges)

    edge_triangles = np.concatenate([[tuple(e) + (n,) for n in edges[:,1][np.where(edges[:,0]==e[0])]] for e in edges])

    edge_triangles = edge_triangles[np.where((edge_triangles[:,1]>edge_triangles[:,0])&(edge_triangles[:,2]>edge_triangles[:,1]))]
    corresponding_triangles_1 = vq(edge_triangles[:,1:],edges)
    corresponding_triangles_2 = vq(np.concatenate([edge_triangles[:,:1],edge_triangles[:,2:]],axis=1),edges)
    edge_triangles = edge_triangles[np.where((corresponding_triangles_1[1]==0)&(corresponding_triangles_2[1]==0))]
    
    return edge_triangles

def tetrahedra_from_triangulation(triangulation_triangles,positions,exterior=True):
    """
    Builds a list of tetrahedra from a list of triangles describing a valid tetrahedrization
    Handles the case of 5-cliques by removing appearing tetrahedra that are composed of smaller ones
    """
    triangulation_tetrahedra = np.concatenate([[tuple(c)+(n,) for n in triangulation_triangles[:,2][np.where((triangulation_triangles[:,0] == c[0])&(triangulation_triangles[:,1] == c[1]))]] for c in triangulation_triangles],axis=0)
    triangulation_tetrahedra = triangulation_tetrahedra[np.where((triangulation_tetrahedra[:,1]>triangulation_tetrahedra[:,0])&(triangulation_tetrahedra[:,2]>triangulation_tetrahedra[:,1])&(triangulation_tetrahedra[:,3]>triangulation_tetrahedra[:,2]))]
    corresponding_tetrahedra_1 = vq(triangulation_tetrahedra[:,1:],triangulation_triangles)
    corresponding_tetrahedra_2 = vq(np.concatenate([triangulation_tetrahedra[:,:1],triangulation_tetrahedra[:,2:]],axis=1),triangulation_triangles)
    corresponding_tetrahedra_3 = vq(np.concatenate([triangulation_tetrahedra[:,:2],triangulation_tetrahedra[:,3:]],axis=1),triangulation_triangles)
    triangulation_tetrahedra = triangulation_tetrahedra[np.where((corresponding_tetrahedra_1[1] == 0)&(corresponding_tetrahedra_2[1] == 0)&(corresponding_tetrahedra_3[1] == 0))]
    # print triangulation_tetrahedra.shape[0]," cell vertices"

    if triangulation_tetrahedra.shape[0]>0:
        triangulation_5_cliques = np.concatenate([[tuple(c)+(n,) for n in triangulation_tetrahedra[:,3][np.where((triangulation_tetrahedra[:,0] == c[0])&(triangulation_tetrahedra[:,1] == c[1])&(triangulation_tetrahedra[:,2] == c[2]))]] for c in triangulation_tetrahedra],axis=0)
        corresponding_5_cliques = vq(triangulation_5_cliques[:,1:],triangulation_tetrahedra)
        triangulation_5_cliques = triangulation_5_cliques[np.where(corresponding_5_cliques[1] == 0)]

        tetrahedra_triangle_matching = np.array(np.concatenate([vq(triangulation_tetrahedra[:,1:],triangulation_triangles),
                                                                vq(np.concatenate([triangulation_tetrahedra[:,0][:,np.newaxis],triangulation_tetrahedra[:,2:]],axis=1),triangulation_triangles),
                                                                vq(np.concatenate([triangulation_tetrahedra[:,:2],triangulation_tetrahedra[:,-1][:,np.newaxis]],axis=1),triangulation_triangles),  
                                                                vq(triangulation_tetrahedra[:,:-1],triangulation_triangles)],axis=1),int)
        n_5_cliques = 0
        for q in triangulation_5_cliques:
            clique_tetrahedra_vertices = np.array([q[:4],np.concatenate([q[:3],q[4:]]),np.concatenate([q[:2],q[3:]]),np.concatenate([q[:1],q[2:]]),q[1:]])
            
            tetra_matching = vq(clique_tetrahedra_vertices,triangulation_tetrahedra)
            clique_tetrahedra_vertices = clique_tetrahedra_vertices[np.where(tetra_matching[1]==0)]
            clique_tetrahedra_indices  = tetra_matching[0][np.where(tetra_matching[1]==0)]
            clique_tetrahedra = positions.values(clique_tetrahedra_vertices)
            clique_tetra_matrix = np.transpose(np.array([clique_tetrahedra[:,1],clique_tetrahedra[:,2],clique_tetrahedra[:,3]]) - clique_tetrahedra[:,0],axes=(1,2,0))
            clique_tetra_volume = abs(np.linalg.det(clique_tetra_matrix))/6.0
            
            if len(clique_tetra_volume) > 3:
                n_5_cliques += 1
                max_tetra = clique_tetrahedra_indices[np.argmax(clique_tetra_volume)]
                t0 = triangulation_tetrahedra[max_tetra]
                triangulation_tetrahedra = np.delete(triangulation_tetrahedra,max_tetra,0)
                triangulation_5_cliques = np.concatenate([[tuple(c)+(n,) for n in triangulation_tetrahedra[:,3][np.where((triangulation_tetrahedra[:,0] == c[0])&(triangulation_tetrahedra[:,1] == c[1])&(triangulation_tetrahedra[:,2] == c[2]))]] for c in triangulation_tetrahedra],axis=0)
                corresponding_5_cliques = vq(triangulation_5_cliques[:,1:],triangulation_tetrahedra)
                triangulation_5_cliques = triangulation_5_cliques[np.where(corresponding_5_cliques[1] == 0)]
        triangulation_5_cliques = np.concatenate([[tuple(c)+(n,) for n in triangulation_tetrahedra[:,3][np.where((triangulation_tetrahedra[:,0] == c[0])&(triangulation_tetrahedra[:,1] == c[1])&(triangulation_tetrahedra[:,2] == c[2]))]] for c in triangulation_tetrahedra],axis=0)
        corresponding_5_cliques = vq(triangulation_5_cliques[:,1:],triangulation_tetrahedra)
        triangulation_5_cliques = triangulation_5_cliques[np.where(corresponding_5_cliques[1] == 0)]
        # print n_5_cliques," 5-Cliques remaining"

        if exterior:
            tetrahedra_triangle_matching = np.array(np.concatenate([vq(triangulation_tetrahedra[:,1:],triangulation_triangles),
                                                                    vq(np.concatenate([triangulation_tetrahedra[:,0][:,np.newaxis],triangulation_tetrahedra[:,2:]],axis=1),triangulation_triangles),
                                                                    vq(np.concatenate([triangulation_tetrahedra[:,:2],triangulation_tetrahedra[:,-1][:,np.newaxis]],axis=1),triangulation_triangles),  
                                                                    vq(triangulation_tetrahedra[:,:-1],triangulation_triangles)],axis=1),int)

            matched_triangles = tetrahedra_triangle_matching[0][np.where(tetrahedra_triangle_matching[1] == 0)]
            triangle_tetrahedra = nd.sum(np.ones_like(matched_triangles),matched_triangles,index=np.arange(triangulation_triangles.shape[0]))
            exterior_triangles = triangulation_triangles[np.where(triangle_tetrahedra == 1)]
            triangulation_tetrahedra = np.concatenate([triangulation_tetrahedra,np.sort([np.concatenate([[1],t]) for t in exterior_triangles])])

    # print triangulation_tetrahedra.shape[0]," cell vertices (+exterior)"

    return triangulation_tetrahedra


def tetrahedrization_simulated_annealing_optimization(tetrahedra,positions,image_cell_vertex=None,omega_energies={'image':1.0,'adjacency':1.0},n_iterations=1,**kwargs):
    """
    """

    display = kwargs.get('display',False)
    screenshot_path = kwargs.get("screenshot_path",None)

    triangulation_tetrahedra = np.copy(tetrahedra)

    if image_cell_vertex != None:
        # print np.sort(image_cell_vertex.keys())
        # print triangulation_tetrahedra
        cell_vertex_jaccard = jaccard_index(np.sort(image_cell_vertex.keys()),triangulation_tetrahedra)
        print "Cell vertices Jaccard : ",cell_vertex_jaccard
    else:
        if omega_energies.has_key('image'):
            del  omega_energies['image']


    if display:
        pass

    cell_tetrahedra = {}
    cell_neighbors = {}

    for c in np.unique(triangulation_tetrahedra):
        cell_tetrahedra[c] = triangulation_tetrahedra[np.where(triangulation_tetrahedra == c)[0]]
        cell_neighbors[c] = np.unique(cell_tetrahedra[c])
        cell_neighbors[c] = cell_neighbors[c][np.where(cell_neighbors[c] != c)]

    epidermis_neighborhood = 9
    inner_neighborhood = 13

    simulated_annealing_initial_temperature = 1.4
    simulated_annealing_lambda = 0.9
    simulated_annealing_minimal_temperature = 0.2

    step = 0

    for iteration in xrange(n_iterations):
        print "_____________________________"
        print ""
        print "Iteration ",iteration
        print "_____________________________"
        print ""

        simulated_annealing_temperature = simulated_annealing_initial_temperature
        n_flips = 1


        while (simulated_annealing_temperature>simulated_annealing_minimal_temperature) and (n_flips>0):

            step = step+1

            simulated_annealing_temperature *= simulated_annealing_lambda
            n_flips = 0
            n_double_flips = 0
            n_annealing_flips = 0

            for c in np.unique(triangulation_tetrahedra)[1:]:

                if 1 in cell_tetrahedra[c]:
                    cell_exterior_triangles = cell_tetrahedra[c][np.where(cell_tetrahedra[c]==1)[0]][:,1:]
                    cell_exterior_triangle_barycenter = positions.values(cell_exterior_triangles).mean(axis=1)

                    cell_exterior_neighbors = np.unique(cell_exterior_triangles)
                    cell_exterior_neighbors = cell_exterior_neighbors[(cell_exterior_neighbors!=1)&(cell_exterior_neighbors!=c)]
                    cell_exterior_barycenter = positions.values(cell_exterior_neighbors).mean(axis=0)

                    cell_inner_neighbors = cell_neighbors[c][vq(cell_neighbors[c],cell_exterior_neighbors)[1]>0]
                    cell_inner_barycenter = positions.values(cell_inner_neighbors[np.where(cell_inner_neighbors != 1)]).mean(axis=0)

                    cell_exterior_triangle_normal = np.cross(positions.values(cell_exterior_triangles[:,1])-positions.values(cell_exterior_triangles[:,0]),positions.values(cell_exterior_triangles[:,2])-positions.values(cell_exterior_triangles[:,0]))
                    cell_exterior_triangle_normal = cell_exterior_triangle_normal/np.linalg.norm(cell_exterior_triangle_normal,axis=1)[:,np.newaxis]
                    cell_exterior_triangle_normal = np.sign(np.einsum('...ij,...j->...i',cell_exterior_triangle_normal,cell_exterior_barycenter-cell_inner_barycenter))[:,np.newaxis]*cell_exterior_triangle_normal
                   
                    cell_exterior_nromal = cell_exterior_triangle_normal.mean(axis=0)/np.linalg.norm(cell_exterior_triangle_normal.mean(axis=0))
                    cell_distance = np.linalg.norm(positions[c]-positions.values(cell_neighbors[c][np.where(cell_neighbors[c] != 1)]),axis=1).mean()

                    positions[1] = positions[c] + cell_distance*cell_exterior_nromal
                else:
                    positions[1] = positions[c] + (positions[c]-positions.values(cell_neighbors[c][np.where(cell_neighbors[c] != 1)]).mean(axis=0))


                cell_neighbor_tetras = nd.sum(np.ones_like(cell_tetrahedra[c]),cell_tetrahedra[c],index=cell_neighbors[c])
                cell_neighbor_distance = np.linalg.norm(positions.values(cell_neighbors[c])- positions[c],axis=1)

                cell_4_neighbors = cell_neighbors[c][np.where(cell_neighbor_tetras==4)]

                for neighbor in cell_4_neighbors:
                    cell_4_tetrahedra = cell_tetrahedra[c][np.where(cell_tetrahedra[c]==neighbor)[0]]
                    if (len(cell_4_tetrahedra) == 4) and (len(np.unique(cell_4_tetrahedra)) == 6):

                        cell_4_edges = np.concatenate(cell_4_tetrahedra[:,tetra_triangle_edge_list ])
                        cell_4_all_edges = np.concatenate([np.array([(p1,p2) for p2 in np.unique(cell_4_tetrahedra) if p1<p2]) for p1 in np.unique(cell_4_tetrahedra)[:-1]])
                        cell_4_diagonals = cell_4_all_edges[np.where(vq(cell_4_all_edges,cell_4_edges)[1]>0)]

                        for d in [cell_4_diagonals[np.random.rand()>0.5]]:
                            diagonal_edges = np.delete(cell_4_edges,np.where((cell_4_edges == d[0])|(cell_4_edges == d[1]))[0],0)
                            _,unique_edges = np.unique(np.ascontiguousarray(diagonal_edges).view(np.dtype((np.void,diagonal_edges.dtype.itemsize * diagonal_edges.shape[1]))),return_index=True)
                            diagonal_edges = diagonal_edges[unique_edges]
                            diagonal_edges = diagonal_edges[np.where(vq(diagonal_edges,np.sort([[c,neighbor]]))[1]>0)]
                            

                            cell_4_flipped_tetrahedra = np.sort([np.concatenate([d,e]) for e in diagonal_edges])

                            tetra_positions = positions.values(cell_4_tetrahedra)
                            cell_4_tetra_volumes = np.abs(np.sum((tetra_positions[:,0]-tetra_positions[:,3])*np.cross(tetra_positions[:,1]-tetra_positions[:,3],tetra_positions[:,2]-tetra_positions[:,3]),axis=1))/6.0
                        
                            tetra_positions = positions.values(cell_4_flipped_tetrahedra)
                            cell_4_flipped_tetra_volumes = np.abs(np.sum((tetra_positions[:,0]-tetra_positions[:,3])*np.cross(tetra_positions[:,1]-tetra_positions[:,3],tetra_positions[:,2]-tetra_positions[:,3]),axis=1))/6.0

                            cell_4_flipped_triangles = cell_4_flipped_tetrahedra[:,tetra_triangle_list ]
                            cell_4_flipped_triangle_edges = cell_4_flipped_triangles[...,triangle_edge_list ]
                            cell_4_flipped_triangle_edge_lengths = np.linalg.norm(positions.values(cell_4_flipped_triangle_edges[...,1]) - positions.values(cell_4_flipped_triangle_edges[...,0]),axis=3)
                            cell_4_flipped_max_distance = cell_4_flipped_triangle_edge_lengths.max()

                            if (abs(cell_4_tetra_volumes.sum() - cell_4_flipped_tetra_volumes.sum()) < 1./1000.) and ((vq(cell_4_flipped_tetrahedra,triangulation_tetrahedra)[1]==0).sum() == 0):
                                
                                diagonal_flipped_edges = np.concatenate([[(p1,p2) for p2 in np.unique(cell_4_diagonals) if p2 not in d] for p1 in d])

                                energy_variation = 0.0

                                if omega_energies.has_key('image'):
                                    corresponding_tetrahedra = (vq(cell_4_tetrahedra,np.sort(image_cell_vertex.keys()))[1]==0)
                                    corresponding_flipped_tetrahedra = (vq(cell_4_flipped_tetrahedra,np.sort(image_cell_vertex.keys()))[1]==0)
                                    image_energy_variation = 1+corresponding_tetrahedra.sum()-corresponding_flipped_tetrahedra.sum()
                                    energy_variation += omega_energies['image']*image_energy_variation

                                if omega_energies.has_key('adjacency'):
                                    adjacency_energy_variation = 0
                                    if (1 in cell_neighbors[c]):
                                        if (neighbor == 1):
                                            adjacency_energy_variation += np.power(len(cell_neighbors[c])-1-inner_neighborhood,2.0) - np.power(len(cell_neighbors[c])-1-epidermis_neighborhood,2.0)
                                        else:
                                            adjacency_energy_variation += np.power(len(cell_neighbors[c])-2-epidermis_neighborhood,2.0) - np.power(len(cell_neighbors[c])-1-epidermis_neighborhood,2.0)
                                    else:
                                        adjacency_energy_variation += np.power(len(cell_neighbors[c])-1-inner_neighborhood,2.0) - np.power(len(cell_neighbors[c])-inner_neighborhood,2.0)
                                    if neighbor != 1:
                                        if (1 in cell_neighbors[neighbor]):
                                            adjacency_energy_variation += np.power(len(cell_neighbors[neighbor])-2-epidermis_neighborhood,2.0) - np.power(len(cell_neighbors[neighbor])-1-epidermis_neighborhood,2.0)
                                        else:
                                            adjacency_energy_variation += np.power(len(cell_neighbors[neighbor])-1-inner_neighborhood,2.0) - np.power(len(cell_neighbors[neighbor])-inner_neighborhood,2.0)
                                    if d[0] != 1:
                                        if (1 in cell_neighbors[d[0]]):
                                            adjacency_energy_variation += np.power(len(cell_neighbors[d[0]])-epidermis_neighborhood,2.0) - np.power(len(cell_neighbors[d[0]])-1-epidermis_neighborhood,2.0)
                                        else:
                                            if (d[1] == 1):
                                                adjacency_energy_variation += np.power(len(cell_neighbors[d[0]])-epidermis_neighborhood,2.0) - np.power(len(cell_neighbors[d[0]])-inner_neighborhood,2.0)
                                            else:
                                                adjacency_energy_variation += np.power(len(cell_neighbors[d[0]])+1-inner_neighborhood,2.0) - np.power(len(cell_neighbors[d[0]])-inner_neighborhood,2.0)
                                    if d[1] != 1:
                                        if (1 in cell_neighbors[d[1]]):
                                            adjacency_energy_variation += np.power(len(cell_neighbors[d[1]])-epidermis_neighborhood,2.0) - np.power(len(cell_neighbors[d[1]])-1-epidermis_neighborhood,2.0)
                                        else:
                                            if (d[0] == 1):
                                                adjacency_energy_variation += np.power(len(cell_neighbors[d[1]])-epidermis_neighborhood,2.0) - np.power(len(cell_neighbors[d[1]])-inner_neighborhood,2.0)
                                            else:
                                                adjacency_energy_variation += np.power(len(cell_neighbors[d[1]])+1-inner_neighborhood,2.0) - np.power(len(cell_neighbors[d[1]])-inner_neighborhood,2.0)
                                    energy_variation += omega_energies['adjacency']*adjacency_energy_variation
                                
                                cell_flip_probability = np.exp(-energy_variation/simulated_annealing_temperature)
                                
                                if np.random.rand() < cell_flip_probability:
                                    n_flips += 2
                                    n_double_flips += 2
                                    n_annealing_flips += 2*(cell_flip_probability<1)

                                    for cell in np.unique(cell_4_tetrahedra):
                                        cell_tetra_matching = vq(cell_tetrahedra[cell],cell_4_tetrahedra)
                                        cell_tetrahedra[cell] = np.delete(cell_tetrahedra[cell],np.where(cell_tetra_matching[1]==0),0)
                                        cell_tetrahedra[cell] = np.concatenate([cell_tetrahedra[cell],cell_4_flipped_tetrahedra[np.where(cell_4_flipped_tetrahedra==cell)[0]]],axis=0)

                                    cell_neighbors[c] = np.delete(cell_neighbors[c],np.where(cell_neighbors[c]==neighbor),0)
                                    cell_neighbors[neighbor] = np.delete(cell_neighbors[neighbor],np.where(cell_neighbors[neighbor]==c),0)
                                    cell_neighbors[d[0]] = np.concatenate([cell_neighbors[d[0]],[d[1]]])
                                    cell_neighbors[d[1]] = np.concatenate([cell_neighbors[d[1]],[d[0]]])

                                    triangulation_tetrahedra = np.delete(triangulation_tetrahedra,np.where(vq(triangulation_tetrahedra,cell_4_tetrahedra)[1]==0),0)
                                    triangulation_tetrahedra = np.concatenate([triangulation_tetrahedra,cell_4_flipped_tetrahedra])

                cell_neighbor_tetras = nd.sum(np.ones_like(cell_tetrahedra[c]),cell_tetrahedra[c],index=cell_neighbors[c])
                cell_3_neighbors = cell_neighbors[c][np.where(cell_neighbor_tetras==3)]

                for neighbor in cell_3_neighbors:
                    neighbor_distance = cell_neighbor_distance[np.where(cell_neighbors[c]==neighbor)][0]

                    cell_3_tetrahedra = cell_tetrahedra[c][np.where(cell_tetrahedra[c]==neighbor)[0]]
                    if (len(cell_3_tetrahedra) == 3) and (len(np.unique(cell_3_tetrahedra)) == 5):

                        cell_3_triangle = np.unique(cell_3_tetrahedra[np.where((cell_3_tetrahedra != c)&(cell_3_tetrahedra != neighbor))])
                        cell_3_flipped_tetrahedra = np.sort([np.concatenate([cell_3_triangle,[c]]),np.concatenate([cell_3_triangle,[neighbor]])])

                        tetra_positions = positions.values(cell_3_tetrahedra)
                        cell_3_tetra_volumes = np.abs(np.sum((tetra_positions[:,0]-tetra_positions[:,3])*np.cross(tetra_positions[:,1]-tetra_positions[:,3],tetra_positions[:,2]-tetra_positions[:,3]),axis=1))/6.0
                    
                        tetra_positions = positions.values(cell_3_flipped_tetrahedra)
                        cell_3_flipped_tetra_volumes = np.abs(np.sum((tetra_positions[:,0]-tetra_positions[:,3])*np.cross(tetra_positions[:,1]-tetra_positions[:,3],tetra_positions[:,2]-tetra_positions[:,3]),axis=1))/6.0

                        cell_3_flipped_triangles = cell_3_flipped_tetrahedra[:,tetra_triangle_list ]
                        cell_3_flipped_triangle_edges = cell_3_flipped_triangles[...,triangle_edge_list ]
                        cell_3_flipped_triangle_edge_lengths = np.linalg.norm(positions.values(cell_3_flipped_triangle_edges[...,1]) - positions.values(cell_3_flipped_triangle_edges[...,0]),axis=3)
                        cell_3_flipped_max_distance = cell_3_flipped_triangle_edge_lengths.max()

                        if (abs(cell_3_tetra_volumes.sum() - cell_3_flipped_tetra_volumes.sum()) < 1./1000.) and ((vq(cell_3_flipped_tetrahedra,triangulation_tetrahedra)[1]==0).sum() == 0):
                            energy_variation = 0.0

                            if omega_energies.has_key('image'):
                                corresponding_tetrahedra = (vq(cell_3_tetrahedra,np.sort(image_cell_vertex.keys()))[1]==0)
                                corresponding_flipped_tetrahedra = (vq(cell_3_flipped_tetrahedra,np.sort(image_cell_vertex.keys()))[1]==0)
                                image_energy_variation = corresponding_tetrahedra.sum()-corresponding_flipped_tetrahedra.sum()
                                energy_variation += omega_energies['image']*image_energy_variation

                            if omega_energies.has_key('adjacency'):
                                adjacency_energy_variation = 0
                                if (1 in cell_neighbors[c]):
                                    if (neighbor == 1):
                                        adjacency_energy_variation += np.power(len(cell_neighbors[c])-1-inner_neighborhood,2.0) - np.power(len(cell_neighbors[c])-1-epidermis_neighborhood,2.0)
                                    else:
                                        adjacency_energy_variation += np.power(len(cell_neighbors[c])-2-epidermis_neighborhood,2.0) - np.power(len(cell_neighbors[c])-1-epidermis_neighborhood,2.0)
                                else:
                                    adjacency_energy_variation += np.power(len(cell_neighbors[c])-1-inner_neighborhood,2.0) - np.power(len(cell_neighbors[c])-inner_neighborhood,2.0)
                                if neighbor != 1:
                                    if (1 in cell_neighbors[neighbor]):
                                        adjacency_energy_variation += np.power(len(cell_neighbors[neighbor])-2-epidermis_neighborhood,2.0) - np.power(len(cell_neighbors[neighbor])-1-epidermis_neighborhood,2.0)
                                    else:
                                        adjacency_energy_variation += np.power(len(cell_neighbors[neighbor])-1-inner_neighborhood,2.0) - np.power(len(cell_neighbors[neighbor])-inner_neighborhood,2.0)
                                energy_variation += omega_energies['adjacency']*adjacency_energy_variation

                            cell_flip_probability = np.exp(-energy_variation/simulated_annealing_temperature)

                            if np.random.rand() < cell_flip_probability:
                                n_flips += 1
                                n_annealing_flips += (cell_flip_probability<1)

                                for cell in np.unique(cell_3_tetrahedra):
                                    cell_tetra_matching = vq(cell_tetrahedra[cell],cell_3_tetrahedra)
                                    cell_tetrahedra[cell] = np.delete(cell_tetrahedra[cell],np.where(cell_tetra_matching[1]==0),0)
                                    cell_tetrahedra[cell] = np.concatenate([cell_tetrahedra[cell],cell_3_flipped_tetrahedra[np.where(cell_3_flipped_tetrahedra==cell)[0]]],axis=0)

                                cell_neighbors[c] = np.delete(cell_neighbors[c],np.where(cell_neighbors[c]==neighbor),0)
                                cell_neighbors[neighbor] = np.delete(cell_neighbors[neighbor],np.where(cell_neighbors[neighbor]==c),0)

                                triangulation_tetrahedra = np.delete(triangulation_tetrahedra,np.where(vq(triangulation_tetrahedra,cell_3_tetrahedra)[1]==0),0)
                                triangulation_tetrahedra = np.concatenate([triangulation_tetrahedra,cell_3_flipped_tetrahedra])

                for cell_tetra in cell_tetrahedra[c]: 
                    if len(np.intersect1d(cell_tetra,cell_3_neighbors)) == 0:
                        cell_triangle = cell_tetra[np.where(cell_tetra!=c)]

                        cell_triangle_tetrahedra = []
                        for tetra_i in xrange(4):
                            tetra_matching = vq(np.concatenate([triangulation_tetrahedra[:,:tetra_i],triangulation_tetrahedra[:,(tetra_i+1):]],axis=1),np.array([cell_triangle]))
                            cell_triangle_tetrahedra.append(triangulation_tetrahedra[np.where(tetra_matching[1]==0)])
                        cell_triangle_tetrahedra = np.concatenate(cell_triangle_tetrahedra)

                        if (len(cell_triangle_tetrahedra) == 2) and (len(np.unique(cell_triangle_tetrahedra)) == 5):
                            cell_flip_neighbor = [p for p in np.unique(cell_triangle_tetrahedra) if not p in cell_tetra][0]

                            cell_triangle_flipped_tetrahedra = np.sort([np.concatenate([[c],[cell_flip_neighbor],e]) for e in cell_triangle[triangle_edge_list ]])
                            

                            cell_triangle_flipped_triangles = cell_triangle_flipped_tetrahedra[:,tetra_triangle_list ]
                            cell_triangle_flipped_triangle_edges = cell_triangle_flipped_triangles[...,triangle_edge_list ]
                            cell_triangle_flipped_triangle_edge_lengths = np.linalg.norm(positions.values(cell_triangle_flipped_triangle_edges[...,1]) - positions.values(cell_triangle_flipped_triangle_edges[...,0]),axis=3)
                            cell_triangle_flipped_max_distance = cell_triangle_flipped_triangle_edge_lengths.max()

                            if (cell_flip_neighbor != 1) and (intersecting_triangle(positions.values([c,cell_flip_neighbor]),positions.values(cell_triangle))) and  ((vq(cell_triangle_flipped_tetrahedra,triangulation_tetrahedra)[1]==0).sum() == 0):
                                energy_variation = 0.0

                                if omega_energies.has_key('image'):
                                    corresponding_tetrahedra = (vq(cell_triangle_tetrahedra,np.sort(image_cell_vertex.keys()))[1]==0)
                                    corresponding_flipped_tetrahedra = (vq(cell_triangle_flipped_tetrahedra,np.sort(image_cell_vertex.keys()))[1]==0)
                                    image_energy_variation = 1+corresponding_tetrahedra.sum()-corresponding_flipped_tetrahedra.sum()
                                    energy_variation += omega_energies['image']*image_energy_variation

                                if omega_energies.has_key('adjacency'):
                                    adjacency_energy_variation = 0
                                    if (1 in cell_neighbors[c]):
                                            adjacency_energy_variation += np.power(len(cell_neighbors[c])-epidermis_neighborhood,2.0) - np.power(len(cell_neighbors[c])-1-epidermis_neighborhood,2.0)
                                    else:
                                        if (cell_flip_neighbor == 1):
                                            adjacency_energy_variation += np.power(len(cell_neighbors[c])-epidermis_neighborhood,2.0) - np.power(len(cell_neighbors[c])-inner_neighborhood,2.0)
                                        else:
                                            adjacency_energy_variation += np.power(len(cell_neighbors[c])+1-inner_neighborhood,2.0) - np.power(len(cell_neighbors[c])-inner_neighborhood,2.0)
                                    if cell_flip_neighbor != 1:
                                        if (1 in cell_neighbors[neighbor]):
                                            adjacency_energy_variation += np.power(len(cell_neighbors[neighbor])-epidermis_neighborhood,2.0) - np.power(len(cell_neighbors[neighbor])-1-epidermis_neighborhood,2.0)
                                        else:
                                            adjacency_energy_variation += np.power(len(cell_neighbors[neighbor])-inner_neighborhood,2.0) - np.power(len(cell_neighbors[neighbor])-1-inner_neighborhood,2.0)
                                    energy_variation += omega_energies['adjacency']*adjacency_energy_variation

                                cell_flip_probability = np.exp(-energy_variation/simulated_annealing_temperature) 

                                if np.random.rand() < cell_flip_probability:
                                    n_flips += 1
                                    n_annealing_flips += (cell_flip_probability<1)

                                    for cell in np.unique(cell_triangle_tetrahedra):
                                        cell_tetra_matching = vq(cell_tetrahedra[cell],cell_triangle_tetrahedra)
                                        cell_tetrahedra[cell] = np.delete(cell_tetrahedra[cell],np.where(cell_tetra_matching[1]==0),0)
                                        cell_tetrahedra[cell] = np.concatenate([cell_tetrahedra[cell],cell_triangle_flipped_tetrahedra[np.where(cell_triangle_flipped_tetrahedra==cell)[0]]],axis=0)

                                    cell_neighbors[c] = np.concatenate([cell_neighbors[c],[cell_flip_neighbor]])
                                    cell_neighbors[cell_flip_neighbor] = np.concatenate([cell_neighbors[cell_flip_neighbor],[c]])

                                    triangulation_tetrahedra = np.delete(triangulation_tetrahedra,np.where(vq(triangulation_tetrahedra,cell_triangle_tetrahedra)[1]==0),0)
                                    triangulation_tetrahedra = np.concatenate([triangulation_tetrahedra,cell_triangle_flipped_tetrahedra])


            if display:
                pass

            print n_flips," Triangles flipped (",n_double_flips," double, ",n_annealing_flips," non-optimal)     [ T = ",simulated_annealing_temperature,"]"
            
            if image_cell_vertex != None:
                cell_vertex_jaccard = jaccard_index(np.sort(image_cell_vertex.keys()),triangulation_tetrahedra)
                print "Cell vertices Jaccard : ",cell_vertex_jaccard 
    return triangulation_tetrahedra


def tetrahedrization_topomesh_add_exterior(triangulation_topomesh):

    if not 1 in triangulation_topomesh.wisps(0):
        compute_topomesh_property(triangulation_topomesh,'cells',2)
        compute_topomesh_property(triangulation_topomesh,'triangles',1)
        compute_topomesh_property(triangulation_topomesh,'triangles',0)

        compute_topomesh_property(triangulation_topomesh,'epidermis',2)
        compute_topomesh_property(triangulation_topomesh,'epidermis',1)
        compute_topomesh_property(triangulation_topomesh,'epidermis',0)

        triangulation_topomesh.add_wisp(0,1)
        cell_exterior_edges = {}
        for c in triangulation_topomesh.wisps(0):
            if triangulation_topomesh.wisp_property('epidermis',0).has_key(c) and triangulation_topomesh.wisp_property('epidermis',0)[c]:
                e = triangulation_topomesh.add_wisp(1)
                triangulation_topomesh.link(1,e,c)
                triangulation_topomesh.link(1,e,1)
                cell_exterior_edges[c] = e 
        edge_exterior_triangles = {}
        for e in triangulation_topomesh.wisps(1):
            if triangulation_topomesh.wisp_property('epidermis',1).has_key(e) and triangulation_topomesh.wisp_property('epidermis',1)[e]:
                t = triangulation_topomesh.add_wisp(2)
                triangulation_topomesh.link(2,t,e)
                for c in triangulation_topomesh.borders(1,e):
                    triangulation_topomesh.link(2,t,cell_exterior_edges[c])
                edge_exterior_triangles[e] = t 
        for t in triangulation_topomesh.wisps(2):
            if triangulation_topomesh.wisp_property('epidermis',2).has_key(t) and triangulation_topomesh.wisp_property('epidermis',2)[t]:
                tet = triangulation_topomesh.add_wisp(3)
                triangulation_topomesh.link(3,tet,t)
                for e in triangulation_topomesh.borders(2,t):
                    triangulation_topomesh.link(3,tet,edge_exterior_triangles[e])  

        compute_topomesh_property(triangulation_topomesh,'triangles',0)
        compute_topomesh_property(triangulation_topomesh,'vertices',1)
        compute_topomesh_property(triangulation_topomesh,'regions',1)
        compute_topomesh_property(triangulation_topomesh,'triangles',1)
        compute_topomesh_property(triangulation_topomesh,'cells',1)
        compute_topomesh_property(triangulation_topomesh,'vertices',2)
        compute_topomesh_property(triangulation_topomesh,'cells',2)
        compute_topomesh_property(triangulation_topomesh,'regions',2)
        compute_topomesh_property(triangulation_topomesh,'vertices',3)
        compute_topomesh_property(triangulation_topomesh,'edges',3)
        compute_topomesh_property(triangulation_topomesh,'triangles',3)
        compute_topomesh_property(triangulation_topomesh,'vertices',3)
        compute_topomesh_property(triangulation_topomesh,'epidermis',2)


def tetrahedrization_topomesh_remove_exterior(triangulation_topomesh):
    if 1 in triangulation_topomesh.wisps(0):
        compute_topomesh_property(triangulation_topomesh,'vertices',1)
        compute_topomesh_property(triangulation_topomesh,'regions',1)
        compute_topomesh_property(triangulation_topomesh,'cells',1)
        compute_topomesh_property(triangulation_topomesh,'vertices',2)
        compute_topomesh_property(triangulation_topomesh,'regions',2)
        compute_topomesh_property(triangulation_topomesh,'cells',2)
        compute_topomesh_property(triangulation_topomesh,'vertices',3)
        compute_topomesh_property(triangulation_topomesh,'edges',3)
        compute_topomesh_property(triangulation_topomesh,'triangles',3)
        
        tetras_to_remove = []
        for t in triangulation_topomesh.wisps(3):
            if 1 in triangulation_topomesh.borders(3,t,3):
                tetras_to_remove.append(t)
        triangles_to_remove = []
        for t in triangulation_topomesh.wisps(2):
            if 1 in triangulation_topomesh.borders(2,t,2):
                triangles_to_remove.append(t)
        edges_to_remove = []
        for e in triangulation_topomesh.wisps(1):
            if 1 in triangulation_topomesh.borders(1,e):
                edges_to_remove.append(e)
        triangulation_topomesh.remove_wisp(0,1)
        for e in edges_to_remove:
            triangulation_topomesh.remove_wisp(1,e)
        for t in triangles_to_remove:
            triangulation_topomesh.remove_wisp(2,t)
        for t in tetras_to_remove:
            triangulation_topomesh.remove_wisp(3,t) 

        triangulation_topomesh.update_wisp_property('barycenter',0,triangulation_topomesh.wisp_property('barycenter',0).values(list(triangulation_topomesh.wisps(0))),list(triangulation_topomesh.wisps(0)))   

        compute_topomesh_property(triangulation_topomesh,'triangles',0)
        compute_topomesh_property(triangulation_topomesh,'vertices',1)
        compute_topomesh_property(triangulation_topomesh,'regions',1)
        compute_topomesh_property(triangulation_topomesh,'triangles',1)
        compute_topomesh_property(triangulation_topomesh,'cells',1)
        compute_topomesh_property(triangulation_topomesh,'vertices',2)
        compute_topomesh_property(triangulation_topomesh,'cells',2)
        compute_topomesh_property(triangulation_topomesh,'regions',2)
        compute_topomesh_property(triangulation_topomesh,'vertices',3)
        compute_topomesh_property(triangulation_topomesh,'edges',3)
        compute_topomesh_property(triangulation_topomesh,'triangles',3)
        compute_topomesh_property(triangulation_topomesh,'vertices',3)
        compute_topomesh_property(triangulation_topomesh,'epidermis',2)



def delaunay_tetrahedrization_topomesh(positions, image_cell_vertex=None, **kwargs):
    """
    Generate a simplicial complex of cell adjacency using Delaunay 3D

    Inputs:
        * positions : dictionary mapping cell ids to center point positions
        ** image_cell_vertex (optional) : dictionary of cell adjacency tetrahedra present in the segemented image
        ++ extension (kwargs) : transformation to apply to positions to enhance the performance of Delaunay

    The points in positions are 'triangulated' by a Delaunay tetrahedrization, that is later on optimized to remove exterior triangles
    of unlikely size, and uncover concave parts of the tissue. This optimization is performed based on a nuclei distance criterion.

    Outputs:
        * triangulation_topomesh : a PropertyTopomesh containing a valid complex of cell adjaceny tetrahedra (exterior not represented)
    """

    # from openalea.plantgl.algo import delaunay_triangulation3D, delaunay_triangulation
    from openalea.mesh.utils.delaunay_tools import delaunay_triangulation
    from openalea.mesh.utils.geometry_tools import tetra_geometric_features

    points = positions.keys()

    extension = kwargs.get('extension',np.array([1,1,0.7]))
    triangulation = delaunay_triangulation(positions.values(points)*extension)
    triangulation_triangles = np.sort(points[np.array(triangulation)[np.array(triangulation).sum(axis=1)>0]])

    triangulation_triangle_edges = triangulation_triangles[:,triangle_edge_list]
    triangulation_triangle_vectors = positions.values(triangulation_triangle_edges[...,1]) - positions.values(triangulation_triangle_edges[...,0])
    triangulation_triangle_lengths = np.linalg.norm(triangulation_triangle_vectors,axis=2)
    triangulation_triangle_perimeters = triangulation_triangle_lengths.sum(axis=1)

    triangulation_edges = np.concatenate(triangulation_triangles[:,triangle_edge_list ],axis=0)
    _,unique_edges = np.unique(np.ascontiguousarray(triangulation_edges).view(np.dtype((np.void,triangulation_edges.dtype.itemsize * triangulation_edges.shape[1]))),return_index=True)
    triangulation_edges = triangulation_edges[unique_edges]

    start_time = time()
    print "--> Generating triangulation topomesh"
    triangle_edges = np.concatenate(triangulation_triangles[:,triangle_edge_list],axis=0)
    triangle_edge_matching = vq(triangle_edges,triangulation_edges)[0]

    triangulation_tetrahedra = tetrahedra_from_triangulation(triangulation_triangles,positions,exterior=False)
    tetrahedra_triangles = np.concatenate(triangulation_tetrahedra[:,tetra_triangle_list])
    tetrahedra_triangle_matching = vq(tetrahedra_triangles,triangulation_triangles)[0]

    triangulation_topomesh = PropertyTopomesh(3)
    for c in np.unique(triangulation_triangles):
        triangulation_topomesh.add_wisp(0,c)
    for e in triangulation_edges:
        eid = triangulation_topomesh.add_wisp(1)
        for pid in e:
            triangulation_topomesh.link(1,eid,pid)
    for t in triangulation_triangles:
        fid = triangulation_topomesh.add_wisp(2)
        for eid in triangle_edge_matching[3*fid:3*fid+3]:
            triangulation_topomesh.link(2,fid,eid)
    for t in triangulation_tetrahedra:
        cid = triangulation_topomesh.add_wisp(3)
        for fid in tetrahedra_triangle_matching[4*cid:4*cid+4]:
            triangulation_topomesh.link(3,cid,fid)
    triangulation_topomesh.update_wisp_property('barycenter',0,positions)        

    end_time = time()
    print "<-- Generating triangulation topomesh [",end_time-start_time,"s]"

    clean_surface = kwargs.get('clean_surface',True)

    if not clean_surface:
        return triangulation_topomesh
    else:
        return tetrahedrization_clean_surface(triangulation_topomesh, image_cell_vertex=image_cell_vertex, **kwargs)


def tetrahedrization_clean_surface(initial_triangulation_topomesh, image_cell_vertex=None, **kwargs):
    
    triangulation_topomesh = deepcopy(initial_triangulation_topomesh)

    positions = triangulation_topomesh.wisp_property('barycenter',0)

    compute_tetrahedrization_geometrical_properties(triangulation_topomesh, normals=False)

    segmented_image = kwargs.get('segmented_image',None)
    binary_image = kwargs.get('binary_image',None)

    if binary_image is None and segmented_image is not None:
        size = np.array(segmented_image.shape)
        binary_image = SpatialImage(np.zeros(tuple(np.array(size*2,int)),np.uint8),voxelsize=segmented_image.voxelsize)
        binary_image[size[0]/2:3*size[0]/2,size[1]/2:3*size[1]/2,size[2]/2:3*size[2]/2][segmented_image>1] = 1

    if binary_image is not None:
        size = np.array(binary_image.shape)
        voxelsize = np.array(binary_image.voxelsize)
        point_radius = 0.6
        image_neighborhood = np.array(np.ceil(point_radius/np.array(binary_image.voxelsize)),int)
        structuring_element = np.zeros(tuple(2*image_neighborhood+1),np.uint8)

        neighborhood_coords = np.mgrid[-image_neighborhood[0]:image_neighborhood[0]+1,-image_neighborhood[1]:image_neighborhood[1]+1,-image_neighborhood[2]:image_neighborhood[2]+1]
        neighborhood_coords = np.concatenate(np.concatenate(np.transpose(neighborhood_coords,(1,2,3,0)))) + image_neighborhood
        neighborhood_coords = array_unique(neighborhood_coords)
            
        neighborhood_distance = np.linalg.norm(neighborhood_coords*voxelsize - image_neighborhood*voxelsize,axis=1)
        neighborhood_coords = neighborhood_coords[neighborhood_distance<=point_radius]
        neighborhood_coords = tuple(np.transpose(neighborhood_coords))
        structuring_element[neighborhood_coords] = 1

        binary_image = np.array(nd.binary_erosion(binary_image,structuring_element),np.uint8)
        print binary_image.shape, voxelsize
        binary_image = SpatialImage(binary_image,voxelsize=tuple(list(voxelsize)))

    surface_topomesh = kwargs.get('surface_topomesh',None)

    if surface_topomesh is None and binary_image is not None:
        from openalea.mesh.utils.implicit_surfaces import implicit_surface_topomesh
        from openalea.draco_stem.stem.tissue_mesh_optimization import optimize_topomesh
        
        grid_voxelsize = kwargs.get('grid_voxelsize',[8,8,8])
        grid_binary_image = binary_image[0:binary_image.shape[0]:grid_voxelsize[0],0:binary_image.shape[1]:grid_voxelsize[1],0:binary_image.shape[2]:grid_voxelsize[2]]

        surface_topomesh = implicit_surface_topomesh(grid_binary_image,np.array(grid_binary_image.shape),voxelsize*grid_voxelsize,center=True)
        surface_topomesh.update_wisp_property('barycenter',0,surface_topomesh.wisp_property('barycenter',0).values()+np.array(grid_binary_image.shape)*voxelsize*grid_voxelsize/4.)
        surface_topomesh = optimize_topomesh(surface_topomesh,omega_forces=dict(taubin_smoothing=0.65,neighborhood=1.0),edge_flip=True,iterations=10,iteration_per_step=2)


    surface_cleaning_criteria = kwargs.get('surface_cleaning_criteria',['surface','exterior','distance','sliver'])
    if surface_topomesh is None:
        if 'surface' in surface_cleaning_criteria:
            surface_cleaning_criteria.remove('surface')
    if binary_image is None:
        if 'exterior' in surface_cleaning_criteria:
            surface_cleaning_criteria.remove('exterior')

    compute_tetrahedrization_geometrical_properties(triangulation_topomesh, normals=False)

    triangulation_triangle_edges = triangulation_topomesh.wisp_property('borders',2).values()

    if 'exterior' in surface_cleaning_criteria or 'surface' in surface_cleaning_criteria:
        triangulation_edge_points = triangulation_topomesh.wisp_property('barycenter',0).values(triangulation_topomesh.wisp_property('vertices',1).values())
        compute_topomesh_property(surface_topomesh,'vertices',2)
        surface_triangle_points = surface_topomesh.wisp_property('barycenter',0).values(surface_topomesh.wisp_property('vertices',2).values())

    if 'exterior' in surface_cleaning_criteria:
        triangulation_points = triangulation_topomesh.wisp_property('barycenter',0).values()
        triangulation_point_coords = tuple(np.array(np.round(triangulation_points/voxelsize+size/2),np.uint16).transpose())
        exterior_point = array_dict(True-np.array(binary_image[triangulation_point_coords],bool),list(triangulation_topomesh.wisps(0)))
        triangulation_exterior_triangles = array_dict(np.any(exterior_point.values(triangulation_topomesh.wisp_property('vertices',2).values()),axis=1),list(triangulation_topomesh.wisps(2)))

    if 'surface' in surface_cleaning_criteria:
        triangulation_edge_surface_intersection =  array_dict([intersecting_triangle(e,surface_triangle_points).any() for e in triangulation_edge_points],list(triangulation_topomesh.wisps(1)))
        triangulation_triangle_surface_intersection = array_dict(np.any(triangulation_edge_surface_intersection.values(triangulation_triangle_edges),axis=1),list(triangulation_topomesh.wisps(2)))

    if 'distance' in surface_cleaning_criteria:
        triangulation_triangle_max_length = array_dict(np.max(triangulation_topomesh.wisp_property('length',1).values(triangulation_triangle_edges),axis=1),list(triangulation_topomesh.wisps(2)))

        maximal_distance = kwargs.get('maximal_distance',None)
        if maximal_distance is None:
            if image_cell_vertex is not None:
                image_edges = array_unique(np.concatenate(np.array(image_cell_vertex.keys())[:,tetra_triangle_edge_list]))
                image_edges = image_edges[image_edges.min(axis=1)!=1]
                image_distances = np.linalg.norm(positions.values(image_edges[:,0]) - positions.values(image_edges[:,1]),axis=1)
                #maximal_distance = image_distances.max()
                maximal_distance = np.percentile(image_distances,99)
            else:
                maximal_distance = 15.

    if 'sliver' in surface_cleaning_criteria:
        triangulation_tetrahedra_triangles = np.array([list(triangulation_topomesh.borders(3,t)) for t in triangulation_topomesh.wisps(3)])
        triangulation_tetrahedra_area = np.sum(triangulation_topomesh.wisp_property('area',2).values(triangulation_tetrahedra_triangles),axis=1)
        triangulation_tetrahedra_volume = triangulation_topomesh.wisp_property('volume',3).values()
        triangulation_tetrahedra_eccentricities = array_dict(1.0 - 216.*np.sqrt(3.)*np.power(triangulation_tetrahedra_volume,2.0)/np.power(triangulation_tetrahedra_area,3.0),list(triangulation_topomesh.wisps(3)))
        triangulation_topomesh.update_wisp_property('eccentricity',3,triangulation_tetrahedra_eccentricities.values(list(triangulation_topomesh.wisps(3))),list(triangulation_topomesh.wisps(3)))    
        triangulation_triangle_sliver = array_dict(map(np.mean,triangulation_tetrahedra_eccentricities.values(triangulation_topomesh.wisp_property('cells',2).values())),list(triangulation_topomesh.wisps(2)))

        maximal_eccentricity = kwargs.get('maximal_eccentricity',0.95)

    # triangulation_topomesh_triangle_to_delete = np.zeros_like(list(triangulation_topomesh.wisps(2)),bool)
    # if 'surface' in surface_cleaning_criteria:
    #     triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | triangulation_triangle_surface_intersection.values(list(triangulation_topomesh.wisps(2)))
    # if 'exterior' in surface_cleaning_criteria:
    #     triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | triangulation_exterior_triangles.values(list(triangulation_topomesh.wisps(2)))
    # if 'distance' in surface_cleaning_criteria:
    #     triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | (triangulation_triangle_max_length.values(list(triangulation_topomesh.wisps(2))) > maximal_distance)
    # if 'sliver' in surface_cleaning_criteria:
    #     triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | (triangulation_triangle_sliver.values(list(triangulation_topomesh.wisps(2))) > maximal_eccentricity)

    # triangulation_topomesh.update_wisp_property('to_delete',2,triangulation_topomesh_triangle_to_delete,list(triangulation_topomesh.wisps(2)))
    # triangulation_topomesh.update_wisp_property('eccentricity',3,triangulation_tetrahedra_eccentricities.values(list(triangulation_topomesh.wisps(3))),list(triangulation_topomesh.wisps(3)))    

    # triangulation_mesh,_,_ = topomesh_to_triangular_mesh(triangulation_topomesh,degree=3,coef=0.9,mesh_center=[0,0,0],property_name='to_delete',property_degree=2)
    # world.add(triangulation_mesh,'optimized_delaunay_triangulation',colormap='Oranges',intensity_range=(0,1),x_slice=(0,75))
    # raw_input()

    n_triangles_0 = triangulation_topomesh.nb_wisps(2)
    n_triangles_initial = triangulation_topomesh.nb_wisps(2)+1
    n_triangles = triangulation_topomesh.nb_wisps(2)
    print n_triangles,"Triangles"
    iteration = 0
  
    while n_triangles < n_triangles_initial:
        n_triangles_initial = n_triangles
        iteration = iteration+1

        exterior_triangles = [t for t in triangulation_topomesh.wisps(2) if len(list(triangulation_topomesh.regions(2,t))) < 2]

        compute_tetrahedrization_topological_properties(triangulation_topomesh)
        
        if 'sliver' in surface_cleaning_criteria:
            #print (np.array(map(len,triangulation_topomesh.wisp_property('cells',2).values()))==0).sum()
            #print (np.array(map(len,triangulation_topomesh.wisp_property('cells',2).values(list(triangulation_topomesh.wisps(2)))))==0).sum()
            #triangulation_triangle_sliver = array_dict(map(np.mean,triangulation_tetrahedra_eccentricities.values(triangulation_topomesh.wisp_property('cells',2).values())),list(triangulation_topomesh.wisps(2)))
            triangulation_triangle_sliver = array_dict([np.mean(triangulation_tetrahedra_eccentricities.values(c)) if len(c)>0 else 0 for c in triangulation_topomesh.wisp_property('cells',2).values()],list(triangulation_topomesh.wisps(2)))
            # triangulation_triangle_sliver = array_dict(map(np.mean,triangulation_tetrahedra_eccentricities.values(triangulation_topomesh.wisp_property('cells',2).values())),list(triangulation_topomesh.wisps(2)))
            # surface_faces = np.array(list(triangulation_topomesh.wisps(2)))[np.array([len(list(triangulation_topomesh.regions(2,t)))==1 for t in triangulation_topomesh.wisps(2)])]
            # print (triangulation_triangle_sliver.values(surface_faces) > maximal_eccentricity).sum(), "(",maximal_eccentricity,")"
            # raw_input()

        triangles_to_delete = set()

        for t in triangulation_topomesh.wisps(2):
            if len(list(triangulation_topomesh.regions(2,t)))==1:
                if 'exterior' in surface_cleaning_criteria and triangulation_exterior_triangles[t]:
                    triangles_to_delete.add(t)
                if 'surface' in surface_cleaning_criteria and triangulation_triangle_surface_intersection[t]:
                    triangles_to_delete.add(t)
                if 'distance' in surface_cleaning_criteria and triangulation_triangle_max_length[t] > maximal_distance:
                    triangles_to_delete.add(t)
                if 'sliver' in surface_cleaning_criteria and triangulation_triangle_sliver[t] > maximal_eccentricity:
                    triangles_to_delete.add(t)
            elif len(list(triangulation_topomesh.regions(2,t)))==0:
                    triangles_to_delete.add(t)
        
        for t in triangles_to_delete:
            for c in triangulation_topomesh.regions(2,t):
                triangulation_topomesh.remove_wisp(3,c)
            triangulation_topomesh.remove_wisp(2,t)
        
        lonely_edges = np.array(list(triangulation_topomesh.wisps(1)))[np.where(np.array(map(len,triangulation_topomesh.wisp_property('triangles',1).values(list(triangulation_topomesh.wisps(1)))))==0)[0]]
        for e in lonely_edges:
            triangulation_topomesh.remove_wisp(1,e)

        # triangulation_topomesh_triangle_to_delete = np.zeros_like(list(triangulation_topomesh.wisps(2)),bool)
        # if 'surface' in surface_cleaning_criteria:
        #     triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | triangulation_triangle_surface_intersection.values(list(triangulation_topomesh.wisps(2)))
        # if 'exterior' in surface_cleaning_criteria:
        #     triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | triangulation_exterior_triangles.values(list(triangulation_topomesh.wisps(2)))
        # if 'distance' in surface_cleaning_criteria:
        #     triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | (triangulation_triangle_max_length.values(list(triangulation_topomesh.wisps(2))) > maximal_distance)
        # if 'sliver' in surface_cleaning_criteria:
        #     triangulation_topomesh_triangle_to_delete = triangulation_topomesh_triangle_to_delete | (triangulation_triangle_sliver.values(list(triangulation_topomesh.wisps(2))) > maximal_eccentricity)

        # triangulation_topomesh.update_wisp_property('to_delete',2,triangulation_topomesh_triangle_to_delete,list(triangulation_topomesh.wisps(2)))
        # triangulation_topomesh.update_wisp_property('eccentricity',3,triangulation_tetrahedra_eccentricities.values(list(triangulation_topomesh.wisps(3))),list(triangulation_topomesh.wisps(3)))    

        # triangulation_mesh,_,_ = topomesh_to_triangular_mesh(triangulation_topomesh,degree=3,coef=0.9,mesh_center=[0,0,0],property_name='to_delete',property_degree=2)
        # #triangulation_mesh,_,_ = topomesh_to_triangular_mesh(triangulation_topomesh,degree=2,coef=0.999,mesh_center=[0,0,0],property_name='eccentricity')
        # world.add(triangulation_mesh,'optimized_delaunay_triangulation',colormap='Oranges',intensity_range=(0,1))

        n_triangles = triangulation_topomesh.nb_wisps(2)
        print n_triangles,"Triangles"

        triangulation_topomesh = clean_topomesh(triangulation_topomesh)

        # compute_topomesh_property(triangulation_topomesh,'epidermis',2)
            
        # edge_triangles = np.array([list(triangulation_topomesh.regions(1,e)) for e in triangulation_topomesh.wisps(1)])
        # edge_epidermis_triangles = np.array([triangulation_topomesh.wisp_property('epidermis',2).values(t) for t in edge_triangles])
        # edge_epidermis_triangle_number = np.array(map(np.sum,edge_epidermis_triangles))

        # while edge_epidermis_triangle_number.max()>2:
        #     edge_excess_epidermis_triangles = np.unique(np.concatenate([list(triangulation_topomesh.regions(1,e)) for i_e,e in enumerate(list(triangulation_topomesh.wisps(1))) if edge_epidermis_triangle_number[i_e]>2]))
        #     edge_excess_epidermis_triangles = edge_excess_epidermis_triangles[triangulation_topomesh.wisp_property('epidermis',2).values(edge_excess_epidermis_triangles)]
        #     for t in edge_excess_epidermis_triangles:
        #         for c in triangulation_topomesh.regions(2,t):
        #             triangulation_topomesh.remove_wisp(3,c)
        #         triangulation_topomesh.remove_wisp(2,t)
        #         print "removed triangle ",t
                
        #     lonely_triangles = np.array(list(triangulation_topomesh.wisps(2)))[np.where(np.array(map(len,[list(triangulation_topomesh.regions(2,t)) for t in triangulation_topomesh.wisps(2)]))==0)[0]]
        #     for t in lonely_triangles:
        #         triangulation_topomesh.remove_wisp(2,t)
        #         print "removed triangle ",t
                
        #     compute_topomesh_property(triangulation_topomesh,'epidermis',2)

        #     edge_triangles = np.array([list(triangulation_topomesh.regions(1,e)) for e in triangulation_topomesh.wisps(1)])
        #     edge_epidermis_triangles = np.array([triangulation_topomesh.wisp_property('epidermis',2).values(t) for t in edge_triangles])
        #     edge_epidermis_triangle_number = np.array(map(np.sum,edge_epidermis_triangles))
            
            
        # compute_topomesh_property(triangulation_topomesh,'regions',2)
        # compute_topomesh_property(triangulation_topomesh,'border_neighbors',degree=2)

        # compute_topomesh_property(triangulation_topomesh,'triangles',1)
        # lonely_edges = np.array(list(triangulation_topomesh.wisps(1)))[np.where(np.array(map(len,triangulation_topomesh.wisp_property('triangles',1).values(list(triangulation_topomesh.wisps(1)))))==0)[0]]
        # for e in lonely_edges:
        #     triangulation_topomesh.remove_wisp(1,e)

    return triangulation_topomesh


def tetrahedrization_topomesh_topological_optimization(input_triangulation_topomesh,omega_energies={'geometry':10.0,'adjacency':0.05},image_cell_vertex=None,image_graph=None,**kwargs):
    """
    Optimize a 3D simplicial complex of cell adjacency to make it fit the actual adjacencies in the image_topomesh

    Inputs:
        * input_triangulation_topomesh : a PropertyTopomesh containing the tetrahedrization to optimize
        * omega_energies : the weight affected to the different energies in the minimization process
        ** image_cell_vertex (optional) : dictionary of cell adjacency tetrahedra present in the segemented image
        ** image_graph (optional) : a PropertyGraph containing adjacencies between cells in the segmented image
        ++ n_iterations : number of S.A. temperature cycles
        ++ simulated_annealing_initial_temperature (kwargs) : start temperature of S.A. temperature cycles
        ++ simulated_annealing_minimal_temperature (kwargs) : end temperature of S.A. temperature cycles
        ++ simulated_annealing_lambda_temperature (kwargs) : temperature decrease factor of S.A. temperature cycles
        ++ epidermis_neighborhood (kwargs) : target value for the number of neighbors for cells on the surface of the tissue
        ++ inner_neighborhood (kwargs) : target value for the number of neighbors for cells inside the tissue

    Performs successive passes of topological operations (edge removals and triangle flips) in order to minimize an energy based on :
      - the actual adjacencies betwwen cells (if they are known) -> omega_energies['image']
      - the ideal number of neighbors for each cell -> omega_energies['adjacency']
      - the shape of the resulting adjacency tetrahedra -> omega_energies['geometry']
    The optimization is performed following a simulated-annealing process, with temperature dependent probabilities of performaing the
    topological operations, and a fixed number of temperature cycles.

    Outputs:
        * triangulation_topomesh : a PropertyTopomesh contianing the optimized tetrahedrization
    """

    from openalea.mesh.utils.array_tools import array_unique, array_difference
    from openalea.mesh.utils.geometry_tools import tetra_geometric_features
    from copy import deepcopy

    triangulation_topomesh = deepcopy(input_triangulation_topomesh)

    positions = kwargs.get('positions',deepcopy(input_triangulation_topomesh.wisp_property('barycenter',0)))

    simulated_annealing_initial_temperature = kwargs.get('simulated_annealing_initial_temperature', 2.5)
    simulated_annealing_lambda = kwargs.get('simulated_annealing_lambda', 0.86)
    simulated_annealing_minimal_temperature = kwargs.get('simulated_annealing_minimal_temperature', 0.25)

    epidermis_neighborhood = kwargs.get('epidermis_neighborhood', 9)
    inner_neighborhood = kwargs.get('inner_neighborhood', 13)

    n_iterations = kwargs.get('n_iterations', 1)

    for iteration in xrange(n_iterations):
        print "_____________________________"
        print ""
        print "Iteration ",iteration
        print "_____________________________"
        print ""

        simulated_annealing_temperature = simulated_annealing_initial_temperature
        n_flips = 1
        n_iteration_edge_removals = 0
        n_iteration_triangle_flips = 0
        
        while (simulated_annealing_temperature>simulated_annealing_minimal_temperature) and (n_flips>0):
            
            simulated_annealing_temperature *= simulated_annealing_lambda
            n_flips = 0
            
            start_time = time()

            tetrahedrization_topomesh_remove_exterior(triangulation_topomesh)
            tetrahedrization_topomesh_add_exterior(triangulation_topomesh)

            edge_tetras = triangulation_topomesh.wisp_property('vertices',3).values(triangulation_topomesh.wisp_property('cells',1).values())
            edge_tetra_cells = np.array(map(np.unique,edge_tetras))
            edge_cells = triangulation_topomesh.wisp_property('vertices',1).values()

            edge_neighbor_cells = np.array(map(array_difference,edge_tetra_cells,edge_cells))
            edge_neighbor_cell_number = np.array(map(len,edge_neighbor_cells))
            
            exterior_positions = np.zeros((triangulation_topomesh.nb_wisps(1),3),float)
            exterior_edges = (edge_cells.min(axis=1)==1)
            exterior_edge_cells = (edge_cells.max(axis=1))[exterior_edges]
            exterior_edge_neighbor_distances = np.array([np.linalg.norm(positions.values(neighbor_cells)-positions[cell],axis=1).mean(axis=0) for cell,neighbor_cells in zip(exterior_edge_cells,edge_neighbor_cells[exterior_edges])])
            #exterior_positions[exterior_edges] = positions.values(exterior_edge_cells) + exterior_edge_neighbor_distances[:,np.newaxis]*triangulation_topomesh.wisp_property('normal',0).values(exterior_edge_cells)
            exterior_positions[exterior_edges] = positions.values(exterior_edge_cells) + exterior_edge_neighbor_distances[:,np.newaxis]*triangulation_topomesh.wisp_property('normal',0).values(exterior_edge_cells)
            exterior_neighbor_edges = (np.array([cells.min() for cells in edge_neighbor_cells])==1)
            exterior_neighbor_edge_middles = positions.values(edge_cells[exterior_neighbor_edges]).mean(axis=1)
            exterior_neighbor_edge_normal = triangulation_topomesh.wisp_property('normal',0).values(edge_cells[exterior_neighbor_edges]).mean(axis=1)
            exterior_neighbor_edge_lengths = np.linalg.norm(positions.values(edge_cells[exterior_neighbor_edges][:,0])-positions.values(edge_cells[exterior_neighbor_edges][:,1]),axis=1)
            #exterior_positions[exterior_neighbor_edges] = exterior_neighbor_edge_middles + np.sqrt(2)/2. * exterior_neighbor_edge_lengths[:,np.newaxis]*exterior_neighbor_edge_normal
            exterior_positions[exterior_neighbor_edges] = exterior_neighbor_edge_middles + exterior_neighbor_edge_lengths[:,np.newaxis]*exterior_neighbor_edge_normal
            
            def edge_exterior_positions(positions,exterior_positions,edge_cell_array):
                import numpy as np
                cell_positions = positions.values(edge_cell_array)
                edge_cell_coords = np.where(edge_cell_array==1)
                if len(edge_cell_coords)>0:
                    cell_positions[edge_cell_coords] = exterior_positions[edge_cell_coords[0]]
                return cell_positions
            
            edge_middles = edge_exterior_positions(positions,exterior_positions,edge_cells).mean(axis=1)
            
            edge_vectors = edge_exterior_positions(positions,exterior_positions,edge_cells[:,1])
            edge_vectors -= edge_exterior_positions(positions,exterior_positions,edge_cells[:,0])
            edge_vectors = edge_vectors/np.linalg.norm(edge_vectors,axis=1)[:,np.newaxis]
            
            def project(points,plane_center,normal_vector):
                import numpy as np
                vectors = points-plane_center
                projectors = -np.einsum('ij,ij->i',vectors,normal_vector[np.newaxis,:])[:,np.newaxis]*normal_vector[np.newaxis,:]
                projection_positions = points+projectors
                plane_vectors = {}
                plane_vectors[0] = np.cross(normal_vector,np.array([1,0,0]))
                plane_vectors[1] = np.cross(normal_vector,plane_vectors[0])
                projected_points = np.transpose([np.einsum('ij,ij->i',vectors,plane_vectors[0][np.newaxis,:]),np.einsum('ij,ij->i',vectors,plane_vectors[1][np.newaxis,:]),np.zeros_like(points[:,2])])
                # print points,' -> ',projected_points
                return projected_points
                
            projected_edge_neighbors = np.array(map(project,edge_exterior_positions(positions,exterior_positions,edge_neighbor_cells),edge_middles,edge_vectors))
            
            def array_delaunay(points,indices):
                from openalea.mesh.utils.delaunay_tools import delaunay_triangulation
                #from openalea.plantgl.algo import delaunay_triangulation
                # import numpy as np
                if len(indices)>3:
                    triangulation = delaunay_triangulation(points)
                    if len(triangulation)>0:
                        return indices[np.array(triangulation)]
                    else:
                        return indices[:3][np.newaxis,:]  
                else:
                    return indices[np.newaxis,:]
            
            edge_neighbor_triangulation = np.array(map(array_delaunay,projected_edge_neighbors,edge_neighbor_cells))
            edge_flipped_tetras = np.array([np.sort(np.concatenate([[list(t)+[c] for c in cells] for t in triangles])) for cells,triangles in zip(edge_cells,edge_neighbor_triangulation)])
            
            edge_positions = [array_dict([pos]+list(positions.values()),[1]+list(positions.keys())) for pos in exterior_positions]
            
            edge_tetra_volumes = np.array([tetra_geometric_features(edge_tetras[e],edge_positions[e],features=['volume']) for e in xrange(triangulation_topomesh.nb_wisps(1))])
            edge_flipped_tetra_volumes = np.array([tetra_geometric_features(edge_flipped_tetras[e],edge_positions[e],features=['volume']) for e in xrange(triangulation_topomesh.nb_wisps(1))])
            
            edge_flip_tetra_volume_difference = np.abs(np.array(map(np.sum,edge_tetra_volumes)) - np.array(map(np.sum,edge_flipped_tetra_volumes)))
            #flippable_edges = (edge_flip_tetra_volume_difference<0.001) | (edge_cells.min(axis=1)==1)
            flippable_edges = (edge_flip_tetra_volume_difference<0.001)
            
            edge_energy_variation = np.zeros(triangulation_topomesh.nb_wisps(1),float)

            if len(flippable_edges) > 0:
            
                if omega_energies.has_key('image'):
                    matching_tetras = vq(np.concatenate(edge_tetras[flippable_edges]),np.sort(image_cell_vertex.keys()))[1]==0
                    matching_tetra_edges = np.concatenate([np.ones(tetras.shape[0])*e for e,tetras in zip(triangulation_topomesh.wisp_property('cells',1).keys()[flippable_edges],edge_tetras[flippable_edges])])
                    
                    matching_flipped_tetras = vq(np.concatenate(edge_flipped_tetras[flippable_edges]),np.sort(image_cell_vertex.keys()))[1] == 0
                    matching_flipped_tetra_edges = np.concatenate([np.ones(tetras.shape[0])*e for e,tetras in zip(triangulation_topomesh.wisp_property('cells',1).keys()[flippable_edges],edge_flipped_tetras[flippable_edges])])
                    
                    edge_matching_tetras = nd.sum(matching_tetras,matching_tetra_edges,index=triangulation_topomesh.wisp_property('cells',1).keys())
                    edge_matching_flipped_tetras = nd.sum(matching_flipped_tetras,matching_flipped_tetra_edges,index=triangulation_topomesh.wisp_property('cells',1).keys())
                    edge_image_energy_variation = edge_matching_tetras - edge_matching_flipped_tetras
                    edge_energy_variation += omega_energies['image']*edge_image_energy_variation
                
                if omega_energies.has_key('adjacency'):
                    cell_adjacencies = np.array([len(list(triangulation_topomesh.region_neighbors(0,c))) for c in triangulation_topomesh.wisps(0)])
                    cell_exterior_adjacencies = np.array([1 in triangulation_topomesh.region_neighbors(0,c) for c in triangulation_topomesh.wisps(0)])
                    cell_adjacencies = array_dict(cell_adjacencies-cell_exterior_adjacencies,list(triangulation_topomesh.wisps(0)))
                    edge_adjacency_energy_variation = np.zeros(triangulation_topomesh.nb_wisps(1),float)
                    flip_exterior = edge_cells.min(axis=1)==1
                    
                    edge_adjacency_energy_variation[flip_exterior] -= np.power(cell_adjacencies.values(edge_cells.max(axis=1)[flip_exterior])-epidermis_neighborhood,2.0)
                    edge_adjacency_energy_variation[flip_exterior] += np.power(cell_adjacencies.values(edge_cells.max(axis=1)[flip_exterior])-inner_neighborhood,2.0)
                    edge_adjacency_energy_variation[True - flip_exterior] -= np.power(cell_adjacencies.values(edge_cells[True - flip_exterior])-inner_neighborhood,2.0).sum(axis=1)
                    edge_adjacency_energy_variation[True - flip_exterior] += np.power(cell_adjacencies.values(edge_cells[True - flip_exterior])-1-inner_neighborhood,2.0).sum(axis=1)
                    
                    edge_tetra_edges = np.array([array_unique(np.sort(np.concatenate(t[:,tetra_triangle_edge_list]))) for t in edge_tetras])
                    edge_flipped_tetra_edges = np.array([array_unique(np.sort(np.concatenate(t[:,tetra_triangle_edge_list]))) for t in edge_flipped_tetras])
                    edge_created_edges = np.array([f_e[vq(f_e,e)[1]>0] for f_e,e in zip(edge_flipped_tetra_edges,edge_tetra_edges)])
                    edge_created_edge_edges = np.concatenate([e*np.ones(len(edge_created_edges[e])) for e in xrange(triangulation_topomesh.nb_wisps(1))])
                    edge_created_edges = np.concatenate(edge_created_edges)
                    
                    created_edge_adjacency_energy_variation = np.zeros(len(edge_created_edges),float)
                    created_exterior = edge_created_edges.min(axis=1)==1
                    created_edge_adjacency_energy_variation[created_exterior] -= np.power(cell_adjacencies.values(edge_created_edges.max(axis=1)[created_exterior])-inner_neighborhood,2.0)
                    created_edge_adjacency_energy_variation[created_exterior] += np.power(cell_adjacencies.values(edge_created_edges.max(axis=1)[created_exterior])-epidermis_neighborhood,2.0)
                    created_edge_adjacency_energy_variation[True - created_exterior] -= np.power(cell_adjacencies.values(edge_created_edges[True - created_exterior])-inner_neighborhood,2.0).sum(axis=1)
                    created_edge_adjacency_energy_variation[True - created_exterior] += np.power(cell_adjacencies.values(edge_created_edges[True - created_exterior])+1-inner_neighborhood,2.0).sum(axis=1)
                    
                    edge_creation_adjacency_energy_variation = nd.sum(created_edge_adjacency_energy_variation,edge_created_edge_edges,index=np.arange(triangulation_topomesh.nb_wisps(1)))
                    edge_adjacency_energy_variation += edge_creation_adjacency_energy_variation
                    
                    edge_energy_variation += omega_energies['adjacency']*edge_adjacency_energy_variation
                
                if omega_energies.has_key('geometry'):
                
                    edge_tetra_max_distance = np.array([tetra_geometric_features(edge_tetras[e],edge_positions[e],features=['max_distance']) for e in xrange(len(edge_tetras))])
                    edge_flipped_tetra_max_distance = np.array([tetra_geometric_features(edge_flipped_tetras[e],edge_positions[e],features=['max_distance']) for e in xrange(len(edge_tetras))])
                    
                    edge_tetra_eccentricity = np.array([tetra_geometric_features(edge_tetras[e],edge_positions[e],features=['eccentricity']) for e in xrange(len(edge_tetras))])
                    edge_flipped_tetra_eccentricity = np.array([tetra_geometric_features(edge_flipped_tetras[e],edge_positions[e],features=['eccentricity']) for e in xrange(len(edge_tetras))])
                    
                    edge_geometry_energy_variation = -np.array(map(np.mean,edge_tetra_max_distance)) + np.array(map(np.mean,edge_flipped_tetra_max_distance))
                    edge_geometry_energy_variation += 10.*(-np.array(map(np.max,edge_tetra_eccentricity)) + np.array(map(np.max,edge_flipped_tetra_eccentricity)))
                    edge_energy_variation += omega_energies['geometry']*edge_geometry_energy_variation
                
                
                edge_energy_variation[True-flippable_edges] = 1000.
                sorted_energy_edges = np.argsort(edge_energy_variation)
                
                end_time = time()
                print "--> Computing edge flip energy variations [",end_time-start_time,"s]"
                
                flip_start_time = time()
                flipped_edges = []
                suboptimal_flipped_edges = []
                modified_edges = []
                
                for edge_to_flip in sorted_energy_edges:
                    start_time = time()
                    edge_id = triangulation_topomesh.wisp_property('cells',1).keys()[edge_to_flip]
                    energy_variation = edge_energy_variation[edge_to_flip]
                    
                    cell_flip_probability = np.exp(-energy_variation/simulated_annealing_temperature)
                    if edge_id in modified_edges or energy_variation >= 1000:
                        cell_flip_probability = 0.
                    else:
                        tetras = edge_tetras[edge_to_flip]
                        flipped_tetras = edge_flipped_tetras[edge_to_flip]
                        cells = edge_cells[edge_to_flip]
                        neighbor_cells = edge_neighbor_cells[edge_to_flip]
                        
                        neighbor_edges = array_unique(np.sort(np.concatenate([[list(triangulation_topomesh.borders(1,e)) for e in triangulation_topomesh.regions(0,c)] for c in neighbor_cells])))
                        
                        existing_edges = np.sort([list(triangulation_topomesh.borders(1,e)) for e in list(triangulation_topomesh.region_neighbors(1,edge_id))])
                        existing_edge_count = np.array([len(list(triangulation_topomesh.regions(1,e,2))) for e in list(triangulation_topomesh.region_neighbors(1,edge_id))])
            
                        tetra_edges = np.concatenate(tetras[:,tetra_triangle_edge_list])
                        flipped_tetra_edges = np.concatenate(flipped_tetras[:,tetra_triangle_edge_list])
                        
                        created_edges = flipped_tetra_edges[vq(flipped_tetra_edges,tetra_edges)[1]>0]
                            
                        if (len(created_edges) > 0) and (vq(created_edges,neighbor_edges)[1].min() == 0):
                            cell_flip_probability = 0.
                        else:     
                            tetra_edge_match = vq(tetra_edges,existing_edges)[0][vq(tetra_edges,existing_edges)[1]==0]
                            tetra_edge_count = nd.sum(np.ones(tetra_edge_match.shape[0]),tetra_edge_match,index=np.arange(existing_edges.shape[0]))
                            
                            flipped_tetra_edge_match = vq(flipped_tetra_edges,existing_edges)[0][vq(flipped_tetra_edges,existing_edges)[1]==0]
                            flipped_tetra_edge_count = nd.sum(np.ones(flipped_tetra_edge_match.shape[0]),flipped_tetra_edge_match,index=np.arange(existing_edges.shape[0])) 
                                
                            if (existing_edge_count-tetra_edge_count+flipped_tetra_edge_count).min() < 3:
                                cell_flip_probability = 0.
                        
                    if np.random.rand() < cell_flip_probability:
                    
                        flipped_tetra_triangle_cells = array_unique(np.concatenate(np.sort(flipped_tetras[:,tetra_triangle_list])))
                        edge_tetra_triangles = np.unique(triangulation_topomesh.wisp_property('triangles',3).values(triangulation_topomesh.wisp_property('cells',1)[edge_id]))
                        edge_triangles = triangulation_topomesh.wisp_property('regions',1)[edge_id]
                        edge_neighbor_triangles = array_difference(edge_tetra_triangles,edge_triangles)
                        edge_neighbor_triangle_cells = np.sort(triangulation_topomesh.wisp_property('vertices',2).values(edge_neighbor_triangles))
                        flipped_tetra_triangle_matching = vq(flipped_tetra_triangle_cells,edge_neighbor_triangle_cells)
                        triangle_fids = {}
                        for t,match,distance in zip(flipped_tetra_triangle_cells,flipped_tetra_triangle_matching[0],flipped_tetra_triangle_matching[1]):
                            if distance == 0:
                                triangle_fids[tuple(t)] = edge_neighbor_triangles[match]
                        
                        flipped_tetra_edge_cells = array_unique(np.concatenate(np.sort(np.concatenate(flipped_tetras[:,tetra_triangle_list]))[:,triangle_edge_list]))
                        edge_tetra_edges = np.unique(triangulation_topomesh.wisp_property('edges',3).values(triangulation_topomesh.wisp_property('cells',1)[edge_id]))
                        edge_neighbor_edges = array_difference(edge_tetra_edges,np.array([edge_id]))
                        edge_neighbor_edge_cells = np.sort(triangulation_topomesh.wisp_property('vertices',1).values(edge_neighbor_edges))
                        flipped_tetra_edge_matching = vq(flipped_tetra_edge_cells,edge_neighbor_edge_cells)
                        edge_eids = {}
                        for e,match,distance in zip(flipped_tetra_edge_cells,flipped_tetra_edge_matching[0],flipped_tetra_edge_matching[1]):
                            if distance == 0:
                                edge_eids[tuple(e)] = edge_neighbor_edges[match]
                                
                        edge_flipped_tids = []
                        for tetra in flipped_tetras:
                            tid = triangulation_topomesh.add_wisp(3)
                            edge_flipped_tids.append(tid)
                            for t in np.sort(tetra[tetra_triangle_list]):
                                if triangle_fids.has_key(tuple(t)):
                                    fid = triangle_fids[tuple(t)]
                                else:
                                    fid = triangulation_topomesh.add_wisp(2)
                                    triangle_fids[tuple(t)] = fid
                                    for e in np.sort(t[triangle_edge_list]):
                                        if edge_eids.has_key(tuple(e)):
                                            eid = edge_eids[tuple(e)]
                                        else:
                                            eid = triangulation_topomesh.add_wisp(1)
                                            edge_eids[tuple(e)] = eid
                                            for c in e:
                                                triangulation_topomesh.link(1,eid,c)
                                        triangulation_topomesh.link(2,fid,eid)
                                triangulation_topomesh.link(3,tid,fid)
                        edge_flipped_tids = np.array(edge_flipped_tids)
                                    
                        for c in triangulation_topomesh.borders(1,edge_id):
                            triangulation_topomesh.unlink(1,edge_id,c)
                        triangles_to_remove = []
                        for fid in triangulation_topomesh.regions(1,edge_id):
                            triangles_to_remove.append(fid)
                        tetras_to_remove = []
                        for tid in triangulation_topomesh.regions(1,edge_id,2):
                            tetras_to_remove.append(tid)
                        for tid in tetras_to_remove:
                            triangulation_topomesh.remove_wisp(3,tid)
                        for fid in triangles_to_remove:
                            triangulation_topomesh.remove_wisp(2,fid)
                        triangulation_topomesh.remove_wisp(1,edge_id)
                        flipped_edges.append(edge_id)
                        if cell_flip_probability<1:
                            suboptimal_flipped_edges.append(edge_id)
                        
                        edge_tids = tetras_to_remove
                        
                        modified_edges += edge_eids.values()
                        #modified_edges += list(triangulation_topomesh.border_neighbors(1,edge_id))
                        #modified_edges += list(np.unique(np.concatenate([list(triangulation_topomesh.region_neighbors(1,e)) for e in edge_eids.values()])))
                        
                        end_time = time()
                        print "  --> Flipped edge ",edge_id," : ",edge_tids," -> ",edge_flipped_tids," (dE = ",energy_variation,") [",end_time-start_time,"s]" 
                        
                flip_end_time = time()
                print len(flipped_edges),' Edges Flipped (',len(suboptimal_flipped_edges),' non-optimal) [',flip_end_time-flip_start_time,'s]'
                n_flips += len(flipped_edges)
                
                
                n_iteration_edge_removals += len(flipped_edges)

            compute_topomesh_property(triangulation_topomesh,'vertices',1)
            compute_topomesh_property(triangulation_topomesh,'regions',1)
            compute_topomesh_property(triangulation_topomesh,'cells',1)
            compute_topomesh_property(triangulation_topomesh,'vertices',2)
            compute_topomesh_property(triangulation_topomesh,'regions',2)
            compute_topomesh_property(triangulation_topomesh,'cells',2)
            compute_topomesh_property(triangulation_topomesh,'vertices',3)
            compute_topomesh_property(triangulation_topomesh,'edges',3)
            compute_topomesh_property(triangulation_topomesh,'triangles',3)
            
            start_time = time()
            triangle_tetras = np.sort(triangulation_topomesh.wisp_property('vertices',3).values(triangulation_topomesh.wisp_property('cells',2).values()))
            triangle_tetra_cells = np.array(map(np.unique,triangle_tetras))
            triangle_cells = triangulation_topomesh.wisp_property('vertices',2).values()
            
            triangle_neighbor_cells = np.array(map(array_difference,triangle_tetra_cells,triangle_cells))
            triangle_neighbor_cell_number = np.array(map(len,triangle_neighbor_cells))
             
            exterior_positions = np.zeros((triangulation_topomesh.nb_wisps(2),3),float)
            exterior_triangles = (triangle_cells.min(axis=1)==1)
            exterior_triangle_cells = np.array(map(array_difference,triangle_cells[exterior_triangles],np.ones(triangulation_topomesh.nb_wisps(2),int)[exterior_triangles][:,np.newaxis]))
            exterior_triangle_lengths = np.linalg.norm(positions.values(exterior_triangle_cells[:,0])-positions.values(exterior_triangle_cells[:,1]),axis=1)
            exterior_triangle_middles = positions.values(exterior_triangle_cells).mean(axis=1)
            exterior_triangle_normal = triangulation_topomesh.wisp_property('normal',0).values(exterior_triangle_cells).mean(axis=1)
            #exterior_positions[exterior_triangles] = exterior_triangle_middles + np.sqrt(2)/2. * exterior_triangle_lengths[:,np.newaxis]*exterior_triangle_normal
            exterior_positions[exterior_triangles] = exterior_triangle_middles + exterior_triangle_lengths[:,np.newaxis]*exterior_triangle_normal
            exterior_neighbor_triangles = (triangle_neighbor_cells.min(axis=1)==1)
            exterior_neighbor_triangle_centers = positions.values(triangle_cells[exterior_neighbor_triangles]).mean(axis=1)
            exterior_neighbor_triangle_lengths = np.linalg.norm(positions.values(triangle_cells[exterior_neighbor_triangles][:,triangle_edge_list])[:,:,1] - positions.values(triangle_cells[exterior_neighbor_triangles][:,triangle_edge_list])[:,:,0],axis=2).mean(axis=1)
            exterior_neighbor_triangle_normal = triangulation_topomesh.wisp_property('normal',0).values(triangle_cells[exterior_neighbor_triangles]).mean(axis=1)
            #exterior_positions[exterior_neighbor_triangles] = exterior_neighbor_triangle_centers + np.sqrt(2)/2. * exterior_neighbor_triangle_lengths[:,np.newaxis]*exterior_neighbor_triangle_normal
            exterior_positions[exterior_neighbor_triangles] = exterior_neighbor_triangle_centers + exterior_neighbor_triangle_lengths[:,np.newaxis]*exterior_neighbor_triangle_normal
            
            triangle_flipped_tetras = np.sort([[list(e)+list(d) for e in t] for t,d in zip(triangle_cells[:,triangle_edge_list],triangle_neighbor_cells)])
            
            triangle_positions = [array_dict([pos]+list(positions.values()),[1]+list(positions.keys())) for pos in exterior_positions]
            
            triangle_tetra_volumes = np.array([tetra_geometric_features(triangle_tetras[t],triangle_positions[t],features=['volume']) for t in xrange(triangulation_topomesh.nb_wisps(2))])
            triangle_flipped_tetra_volumes = np.array([tetra_geometric_features(triangle_flipped_tetras[t],triangle_positions[t],features=['volume']) for t in xrange(triangulation_topomesh.nb_wisps(2))])
            
            triangle_flip_tetra_volume_difference = np.abs(np.array(map(np.sum,triangle_tetra_volumes)) - np.array(map(np.sum,triangle_flipped_tetra_volumes)))
            flippable_triangles = (triangle_flip_tetra_volume_difference<0.001) | (triangle_neighbor_cells.min(axis=1)==1)
            
            triangle_energy_variation = np.zeros(triangulation_topomesh.nb_wisps(2),float)
            
            if len(flippable_triangles)>0:

                if omega_energies.has_key('image'):
                    matching_tetras = vq(np.concatenate(triangle_tetras[flippable_triangles]),np.sort(image_cell_vertex.keys()))[1]==0
                    matching_tetra_triangles = np.concatenate([np.ones(tetras.shape[0])*t for t,tetras in zip(triangulation_topomesh.wisp_property('cells',2).keys()[flippable_triangles],triangle_tetras[flippable_triangles])])
                    
                    matching_flipped_tetras = vq(np.concatenate(triangle_flipped_tetras[flippable_triangles]),np.sort(image_cell_vertex.keys()))[1] == 0
                    matching_flipped_tetra_triangles = np.concatenate([np.ones(tetras.shape[0])*t for t,tetras in zip(triangulation_topomesh.wisp_property('cells',2).keys()[flippable_triangles],triangle_flipped_tetras[flippable_triangles])])
                    
                    triangle_matching_tetras = nd.sum(matching_tetras,matching_tetra_triangles,index=triangulation_topomesh.wisp_property('cells',2).keys())
                    triangle_matching_flipped_tetras = nd.sum(matching_flipped_tetras,matching_flipped_tetra_triangles,index=triangulation_topomesh.wisp_property('cells',2).keys())
                    triangle_image_energy_variation = 1 + triangle_matching_tetras - triangle_matching_flipped_tetras
                    triangle_energy_variation += omega_energies['image']*triangle_image_energy_variation
                
                if omega_energies.has_key('adjacency'):
                    cell_adjacencies = np.array([len(list(triangulation_topomesh.region_neighbors(0,c))) for c in triangulation_topomesh.wisps(0)])
                    cell_exterior_adjacencies = np.array([1 in triangulation_topomesh.region_neighbors(0,c) for c in triangulation_topomesh.wisps(0)])
                    cell_adjacencies = array_dict(cell_adjacencies-cell_exterior_adjacencies,list(triangulation_topomesh.wisps(0)))
                    for c in triangulation_topomesh.wisps(0):
                        if 1 in triangulation_topomesh.region_neighbors(0,c):
                            cell_adjacencies
                    triangle_adjacency_energy_variation = np.zeros(triangulation_topomesh.nb_wisps(2),float)
                    flip_exterior = triangle_neighbor_cells.min(axis=1)==1
                    
                    triangle_adjacency_energy_variation[flip_exterior] -= np.power(cell_adjacencies.values(triangle_neighbor_cells.max(axis=1)[flip_exterior])-inner_neighborhood,2.0)
                    triangle_adjacency_energy_variation[flip_exterior] += np.power(cell_adjacencies.values(triangle_neighbor_cells.max(axis=1)[flip_exterior])-epidermis_neighborhood,2.0)
                    triangle_adjacency_energy_variation[True - flip_exterior] -= np.power(cell_adjacencies.values(triangle_neighbor_cells[True - flip_exterior])-inner_neighborhood,2.0).sum(axis=1)
                    triangle_adjacency_energy_variation[True - flip_exterior] += np.power(cell_adjacencies.values(triangle_neighbor_cells[True - flip_exterior])+1-inner_neighborhood,2.0).sum(axis=1)
                    triangle_energy_variation += omega_energies['adjacency']*triangle_adjacency_energy_variation
                
                if omega_energies.has_key('geometry'):
                    from openalea.mesh.utils.geometry_tools import tetra_geometric_features
                    
                    triangle_tetra_max_distance = np.array([tetra_geometric_features(triangle_tetras[t],triangle_positions[t],features=['max_distance']) for t in xrange(len(triangle_tetras))])
                    triangle_flipped_tetra_max_distance = np.array([tetra_geometric_features(triangle_flipped_tetras[t],triangle_positions[t],features=['max_distance']) for t in xrange(len(triangle_tetras))])
                    
                    triangle_tetra_eccentricity = np.array([tetra_geometric_features(triangle_tetras[t],triangle_positions[t],features=['eccentricity']) for t in xrange(len(triangle_tetras))])
                    triangle_flipped_tetra_eccentricity = np.array([tetra_geometric_features(triangle_flipped_tetras[t],triangle_positions[t],features=['eccentricity']) for t in xrange(len(triangle_tetras))])
                    
                    triangle_geometry_energy_variation = -np.array(map(np.mean,triangle_tetra_max_distance)) + np.array(map(np.mean,triangle_flipped_tetra_max_distance))
                    triangle_geometry_energy_variation += 10.*(-np.array(map(np.max,triangle_tetra_eccentricity)) + np.array(map(np.max,triangle_flipped_tetra_eccentricity)))
                    triangle_energy_variation += omega_energies['geometry']*triangle_geometry_energy_variation
                
                    
                triangle_energy_variation[True-flippable_triangles] = 1000.
                sorted_energy_triangles = np.argsort(triangle_energy_variation)
                end_time = time()
                print "<-- Computing triangle flip energy variations [",end_time-start_time,"s]"
                            
                flipped_triangles = []
                suboptimal_flipped_triangles = []
                modified_triangles = []
                
                flip_start_time = time()
                for triangle_to_flip in sorted_energy_triangles:
                    start_time = time()
                    triangle_id = triangulation_topomesh.wisp_property('cells',2).keys()[triangle_to_flip]
                    energy_variation = triangle_energy_variation[triangle_to_flip]
                    
                    cell_flip_probability = np.exp(-energy_variation/simulated_annealing_temperature)
                    if triangle_id in modified_triangles or energy_variation >= 1000:
                        cell_flip_probability = 0.
                    else:
                        tetras = triangle_tetras[triangle_to_flip]
                        flipped_tetras = triangle_flipped_tetras[triangle_to_flip]
                        cells = triangle_cells[triangle_to_flip]
                        neighbor_cells = triangle_neighbor_cells[triangle_to_flip]
                    
                        if neighbor_cells[1] in triangulation_topomesh.region_neighbors(0,neighbor_cells[0]):
                            cell_flip_probability = 0.
                        else:
                            existing_edge_ids = np.unique([list(triangulation_topomesh.borders(3,t,2)) for t in triangulation_topomesh.regions(2,triangle_id)])
                            existing_edges = np.sort([list(triangulation_topomesh.borders(1,e)) for e in existing_edge_ids]) 
                            existing_edge_count = np.array([len(list(triangulation_topomesh.regions(1,e,2))) for e in existing_edge_ids]) 
                          
                            tetra_edges = np.concatenate(tetras[:,tetra_triangle_edge_list])
                            tetra_edge_match = vq(tetra_edges,existing_edges)[0][vq(tetra_edges,existing_edges)[1]==0]
                            tetra_edge_count = nd.sum(np.ones(tetra_edge_match.shape[0]),tetra_edge_match,index=np.arange(existing_edges.shape[0]))
                           
                            flipped_tetra_edges = np.concatenate(flipped_tetras[:,tetra_triangle_edge_list])
                            flipped_tetra_edge_match = vq(flipped_tetra_edges,existing_edges)[0][vq(flipped_tetra_edges,existing_edges)[1]==0]
                            flipped_tetra_edge_count = nd.sum(np.ones(flipped_tetra_edge_match.shape[0]),flipped_tetra_edge_match,index=np.arange(existing_edges.shape[0]))
                
                            if (existing_edge_count-tetra_edge_count+flipped_tetra_edge_count).min() < 3:
                                cell_flip_probability = 0.
                        
                    if np.random.rand() < cell_flip_probability:
                    
                        flipped_tetra_triangle_cells = array_unique(np.concatenate(np.sort(flipped_tetras[:,tetra_triangle_list])))
                        triangle_tetra_triangles = np.unique(triangulation_topomesh.wisp_property('triangles',3).values(triangulation_topomesh.wisp_property('cells',2)[triangle_id]))
                        triangle_neighbor_triangles = array_difference(triangle_tetra_triangles,np.array([triangle_id]))
                        triangle_neighbor_triangle_cells = np.sort(triangulation_topomesh.wisp_property('vertices',2).values(triangle_neighbor_triangles))
                        flipped_tetra_triangle_matching = vq(flipped_tetra_triangle_cells,triangle_neighbor_triangle_cells)
                        triangle_fids = {}
                        for t,match,distance in zip(flipped_tetra_triangle_cells,flipped_tetra_triangle_matching[0],flipped_tetra_triangle_matching[1]):
                            if distance == 0:
                                triangle_fids[tuple(t)] = triangle_neighbor_triangles[match]
                        
                        
                        flipped_tetra_edge_cells = array_unique(np.concatenate(np.sort(np.concatenate(flipped_tetras[:,tetra_triangle_list]))[:,triangle_edge_list]))
                        triangle_neighbor_edges = np.unique(triangulation_topomesh.wisp_property('edges',3).values(triangulation_topomesh.wisp_property('cells',2)[triangle_id]))
                        triangle_neighbor_edge_cells = np.sort(triangulation_topomesh.wisp_property('vertices',1).values(triangle_neighbor_edges))
                        flipped_tetra_edge_matching = vq(flipped_tetra_edge_cells,triangle_neighbor_edge_cells)
                        edge_eids = {}
                        for e,match,distance in zip(flipped_tetra_edge_cells,flipped_tetra_edge_matching[0],flipped_tetra_edge_matching[1]):
                            if distance == 0:
                                edge_eids[tuple(e)] = triangle_neighbor_edges[match]
                              
                        triangle_flipped_tids = []
                        for tetra in flipped_tetras:
                            tid = triangulation_topomesh.add_wisp(3)
                            triangle_flipped_tids.append(tid)
                            for t in np.sort(tetra[tetra_triangle_list]):
                                if triangle_fids.has_key(tuple(t)):
                                    fid = triangle_fids[tuple(t)]
                                else:
                                    fid = triangulation_topomesh.add_wisp(2)
                                    triangle_fids[tuple(t)] = fid
                                    for e in np.sort(t[triangle_edge_list]):
                                        if edge_eids.has_key(tuple(e)):
                                            eid = edge_eids[tuple(e)]
                                        else:
                                            eid = triangulation_topomesh.add_wisp(1)
                                            edge_eids[tuple(e)] = eid
                                            for c in e:
                                                triangulation_topomesh.link(1,eid,c)
                                        triangulation_topomesh.link(2,fid,eid)
                                triangulation_topomesh.link(3,tid,fid)
                        triangle_flipped_tids = np.array(triangle_flipped_tids)
                         
                        tetras_to_remove = []
                        for tid in triangulation_topomesh.regions(2,triangle_id):
                            tetras_to_remove.append(tid)
                        for tid in tetras_to_remove:
                            triangulation_topomesh.remove_wisp(3,tid)
                        triangulation_topomesh.remove_wisp(2,triangle_id)
                        flipped_triangles.append(triangle_id)
                        if cell_flip_probability<1:
                            suboptimal_flipped_triangles.append(triangle_id)
                        
                        triangle_tids = tetras_to_remove
                        
                        modified_triangles += triangle_fids.values()

                        end_time = time()
                        print "  --> Flipped triangle ",triangle_id," : ",triangle_tids," -> ",triangle_flipped_tids," (dE = ",energy_variation,") [",end_time-start_time,"s]" 

                flip_end_time = time()
                        
                print len(flipped_triangles),' Triangles Flipped (',len(suboptimal_flipped_triangles),' non-optimal) [',flip_end_time-flip_start_time,'s]'
                n_flips += len(flipped_triangles)    
                
                n_iteration_triangle_flips += len(flipped_triangles) 
            
            tetrahedrization_topomesh_remove_exterior(triangulation_topomesh)
            
            edge_triangles = np.array([list(triangulation_topomesh.regions(1,e)) for e in triangulation_topomesh.wisps(1)])
            edge_epidermis_triangles = np.array([triangulation_topomesh.wisp_property('epidermis',2).values(t) for t in edge_triangles])
            edge_epidermis_triangle_number = np.array(map(np.sum,edge_epidermis_triangles))
            
            while edge_epidermis_triangle_number.max()>2:
                edge_excess_epidermis_triangles = np.unique(np.concatenate([list(triangulation_topomesh.regions(1,e)) for i_e,e in enumerate(list(triangulation_topomesh.wisps(1))) if edge_epidermis_triangle_number[i_e]>2]))
                edge_excess_epidermis_triangles = edge_excess_epidermis_triangles[triangulation_topomesh.wisp_property('epidermis',2).values(edge_excess_epidermis_triangles)]
                for t in edge_excess_epidermis_triangles:
                    for c in triangulation_topomesh.regions(2,t):
                        triangulation_topomesh.remove_wisp(3,c)
                    triangulation_topomesh.remove_wisp(2,t)
                    print "removed triangle ",t
                    
                lonely_triangles = np.array(list(triangulation_topomesh.wisps(2)))[np.where(np.array(map(len,[list(triangulation_topomesh.regions(2,t)) for t in triangulation_topomesh.wisps(2)]))==0)[0]]
                for t in lonely_triangles:
                    triangulation_topomesh.remove_wisp(2,t)
                    print "removed triangle ",t
                    
                lonely_edges = np.array(list(triangulation_topomesh.wisps(1)))[np.where(np.array(map(len,[list(triangulation_topomesh.regions(1,e)) for e in triangulation_topomesh.wisps(1)]))==0)[0]]
                for e in lonely_edges:
                    triangulation_topomesh.remove_wisp(1,e)
                    print "removed edge ",e
                    
                compute_topomesh_property(triangulation_topomesh,'epidermis',2)
            
                edge_triangles = np.array([list(triangulation_topomesh.regions(1,e)) for e in triangulation_topomesh.wisps(1)])
                edge_epidermis_triangles = np.array([triangulation_topomesh.wisp_property('epidermis',2).values(t) for t in edge_triangles])
                edge_epidermis_triangle_number = np.array(map(np.sum,edge_epidermis_triangles))
        
            compute_topomesh_property(triangulation_topomesh,'vertices',3)
            tetra_features = tetra_geometric_features(triangulation_topomesh.wisp_property('vertices',3).values(),positions,features=['max_distance','eccentricity','min_dihedral_angle','max_dihedral_angle'])

            tetrahedrization_topomesh_add_exterior(triangulation_topomesh)
            
            if image_graph != None:
                cell_interface_jaccard = jaccard_index(np.sort([image_graph.edge_vertices(e) for e in image_graph.edges()]),np.sort(triangulation_topomesh.wisp_property('vertices',1).values()))
                print "Cell interface Jaccard (2-adjacency) : ",cell_interface_jaccard
            if image_cell_vertex != None:
                # cell_edge_jaccard = jaccard_index(np.sort(array_unique(np.concatenate(np.array(image_cell_vertex.keys())[:,tetra_triangle_list]))),np.sort(triangulation_topomesh.wisp_property('vertices',2).values()))
                # print "Cell edge Jaccard (3-adjacency) : ",cell_edge_jaccard
                cell_vertex_jaccard = jaccard_index(np.sort(image_cell_vertex.keys()),np.sort(triangulation_topomesh.wisp_property('vertices',3).values()))
                print "Cell vertices Jaccard  (4-adjacency) : ",cell_vertex_jaccard
            cell_adjacencies = np.array([len(list(triangulation_topomesh.region_neighbors(0,c))) for c in triangulation_topomesh.wisps(0)])
            cell_exterior_adjacencies = np.array([1 in triangulation_topomesh.region_neighbors(0,c) for c in triangulation_topomesh.wisps(0)])
            cell_adjacencies = array_dict(cell_adjacencies-cell_exterior_adjacencies,list(triangulation_topomesh.wisps(0)))
            cell_target_adjcencies = epidermis_neighborhood*cell_exterior_adjacencies + inner_neighborhood*(1-cell_exterior_adjacencies)
            cell_adjacency_error = np.power((cell_adjacencies.values()[1:]-cell_target_adjcencies[1:])/np.array(cell_target_adjcencies[1:],float),2)
            # print "Adjacency Error : ",cell_adjacency_error.mean()
            # print "Inner adjacency Error : ",cell_adjacency_error[True-cell_exterior_adjacencies[1:]].mean()
            # print "Epidermis adjacency Error : ",cell_adjacency_error[cell_exterior_adjacencies[1:]].mean()
            # print "Tetrahedra maximal length              : ",np.mean(tetra_features[:,0])
            # print "Tetrahedra eccentricity (shape factor) : ",np.mean(tetra_features[:,1])
            # print "Tetrahedra minimal dihedral angle      : ",np.mean(tetra_features[:,2])
            # print "Tetrahedra maximal dihedral angle      : ",np.mean(tetra_features[:,3])
            
    return triangulation_topomesh
    


def compute_tetrahedrization_topological_properties(triangulation_topomesh):

    compute_topomesh_property(triangulation_topomesh,'epidermis',degree=2)
    compute_topomesh_property(triangulation_topomesh,'cells',degree=2)
    compute_topomesh_property(triangulation_topomesh,'regions',degree=2)
    compute_topomesh_property(triangulation_topomesh,'borders',degree=2)
    compute_topomesh_property(triangulation_topomesh,'vertices',degree=2)
    compute_topomesh_property(triangulation_topomesh,'border_neighbors',degree=2)

    compute_topomesh_property(triangulation_topomesh,'vertices',1)
    compute_topomesh_property(triangulation_topomesh,'triangles',degree=1)
    compute_topomesh_property(triangulation_topomesh,'regions',1)
    compute_topomesh_property(triangulation_topomesh,'cells',1)

    compute_topomesh_property(triangulation_topomesh,'triangles',degree=0)
    compute_topomesh_property(triangulation_topomesh,'epidermis',degree=0)

    compute_topomesh_property(triangulation_topomesh,'vertices',degree=3)
    compute_topomesh_property(triangulation_topomesh,'edges',3)
    compute_topomesh_property(triangulation_topomesh,'triangles',3)


def compute_tetrahedrization_geometrical_properties(triangulation_topomesh, normals=True):

    compute_tetrahedrization_topological_properties(triangulation_topomesh)

    from openalea.mesh.utils.implicit_surfaces import point_spherical_density

        
    positions = triangulation_topomesh.wisp_property('barycenter',0)

    if normals:
        compute_topomesh_property(triangulation_topomesh,'normal',2)
        compute_topomesh_property(triangulation_topomesh,'barycenter',2)
        triangulation_triangle_exterior_density = point_spherical_density(positions,triangulation_topomesh.wisp_property('barycenter',2).values()+triangulation_topomesh.wisp_property('normal',2).values(),sphere_radius=10.,k=0.5)
        triangulation_triangle_interior_density = point_spherical_density(positions,triangulation_topomesh.wisp_property('barycenter',2).values()-triangulation_topomesh.wisp_property('normal',2).values(),sphere_radius=10,k=0.5)
        normal_orientation = 2*(triangulation_triangle_exterior_density<triangulation_triangle_interior_density)-1
        triangulation_topomesh.update_wisp_property('normal',2,normal_orientation[...,np.newaxis]*triangulation_topomesh.wisp_property('normal',2).values(),list(triangulation_topomesh.wisps(2)))

        compute_topomesh_property(triangulation_topomesh,'normal',0)

    tetra_features = tetra_geometric_features(triangulation_topomesh.wisp_property('vertices',3).values(),positions,features=['max_distance','eccentricity','min_dihedral_angle','max_dihedral_angle'])

    compute_topomesh_property(triangulation_topomesh,'barycenter',degree=3)

    compute_topomesh_property(triangulation_topomesh,'length',degree=1)
    compute_topomesh_property(triangulation_topomesh,'area',degree=2)
    compute_topomesh_property(triangulation_topomesh,'volume',degree=3)


def clean_tetrahedrization(triangulation_topomesh, clean_vertices=True, min_cell_neighbors=None):
    
    exterior = False
    if 1 in triangulation_topomesh.wisps(0):
        exterior = True
        tetrahedrization_topomesh_remove_exterior(triangulation_topomesh)

    compute_tetrahedrization_topological_properties(triangulation_topomesh)

    triangulation_triangle_n_cells = array_dict(map(len,triangulation_topomesh.wisp_property('cells',2).values()),list(triangulation_topomesh.wisps(2)))

    triangulation_edge_n_triangles = array_dict(map(len,[[t for t in triangulation_topomesh.regions(1,e) if triangulation_topomesh.wisp_property('epidermis',2)[t]] for e in triangulation_topomesh.wisps(1)]),list(triangulation_topomesh.wisps(1)))
    triangulation_triangle_edge_n_triangles = array_dict(triangulation_edge_n_triangles.values(triangulation_topomesh.wisp_property('borders',2).values(list(triangulation_topomesh.wisps(2)))).max(axis=1),list(triangulation_topomesh.wisps(2)))

    while triangulation_triangle_edge_n_triangles.values().max() > 2:
        triangles_to_delete = []
        for t in triangulation_topomesh.wisps(2):
            if triangulation_topomesh.wisp_property('epidermis',2)[t] and triangulation_triangle_edge_n_triangles[t]>2:
                triangles_to_delete.append(t)
        
        for t in triangles_to_delete:
            for c in triangulation_topomesh.regions(2,t):
                triangulation_topomesh.remove_wisp(3,c)
            triangulation_topomesh.remove_wisp(2,t)
        
        lonely_edges = np.array(list(triangulation_topomesh.wisps(1)))[np.where(np.array(map(len,[list(triangulation_topomesh.regions(1,e)) for e in triangulation_topomesh.wisps(1)]))==0)[0]]
        for e in lonely_edges:
            triangulation_topomesh.remove_wisp(1,e)

        compute_tetrahedrization_topological_properties(triangulation_topomesh)
        
        triangulation_edge_n_triangles = array_dict(map(len,[[t for t in triangulation_topomesh.regions(1,e) if triangulation_topomesh.wisp_property('epidermis',2)[t]] for e in triangulation_topomesh.wisps(1)]),list(triangulation_topomesh.wisps(1)))
        triangulation_triangle_edge_n_triangles = array_dict(triangulation_edge_n_triangles.values(triangulation_topomesh.wisp_property('borders',2).values(list(triangulation_topomesh.wisps(2)))).max(axis=1),list(triangulation_topomesh.wisps(2)))

    lonely_triangles = np.array(list(triangulation_topomesh.wisps(2)))[np.where(np.array(map(len,[list(triangulation_topomesh.regions(2,t)) for t in triangulation_topomesh.wisps(2)]))==0)[0]]
    for t in lonely_triangles:
        triangulation_topomesh.remove_wisp(2,t)
            
    lonely_edges = np.array(list(triangulation_topomesh.wisps(1)))[np.where(np.array(map(len,[list(triangulation_topomesh.regions(1,e)) for e in triangulation_topomesh.wisps(1)]))==0)[0]]
    for e in lonely_edges:
        triangulation_topomesh.remove_wisp(1,e)

    if clean_vertices:
        lonely_vertices = np.array(list(triangulation_topomesh.wisps(0)))[np.where(np.array(map(len,[list(triangulation_topomesh.regions(0,v,2)) for v in triangulation_topomesh.wisps(0)]))==0)[0]]
        for v in lonely_vertices:
            triangulation_topomesh.remove_wisp(0,v)  
        triangulation_topomesh.update_wisp_property('barycenter',0,array_dict(triangulation_topomesh.wisp_property('barycenter',0).values(list(triangulation_topomesh.wisps(0))),list(triangulation_topomesh.wisps(0))))

    compute_tetrahedrization_topological_properties(triangulation_topomesh)

    if min_cell_neighbors is not None:
        cell_neighbors = np.array([list(triangulation_topomesh.border_neighbors(3,c)) for c in triangulation_topomesh.wisps(3)])
        cell_neighborhood = array_dict(map(len,cell_neighbors),list(triangulation_topomesh.wisps(3)))
        cells_to_remove = np.array(list(triangulation_topomesh.wisps(3)))[cell_neighborhood.values()<min_cell_neighbors]

        while len(cells_to_remove) > 0:
            for c in cells_to_remove:
                triangles_to_remove = []
                for t in triangulation_topomesh.borders(3,c):
                    triangulation_topomesh.unlink(3,c,t)
                triangulation_topomesh.remove_wisp(3,c)

            triangles_to_remove = []
            for t in triangulation_topomesh.wisps(2):
                if len(list(triangulation_topomesh.regions(2,t))) == 0:
                    triangles_to_remove.append(t)
            for t in triangles_to_remove:
                triangulation_topomesh.remove_wisp(2,t)
                
            cell_neighbors = np.array([list(triangulation_topomesh.border_neighbors(3,c)) for c in triangulation_topomesh.wisps(3)])
            cell_neighborhood = array_dict(map(len,cell_neighbors),list(triangulation_topomesh.wisps(3)))
            cells_to_remove = np.array(list(triangulation_topomesh.wisps(3)))[cell_neighborhood.values()<min_cell_neighbors]
            
        edges_to_remove = []
        for e in triangulation_topomesh.wisps(1):
            if len(list(triangulation_topomesh.regions(1,e))) == 0:
                edges_to_remove.append(e)
        for e in edges_to_remove:
            triangulation_topomesh.remove_wisp(1,e)

        if clean_vertices:
            vertices_to_remove = []
            for v in triangulation_topomesh.wisps(0):
                if len(list(triangulation_topomesh.regions(0,v))) == 0:
                    vertices_to_remove.append(v)
            for v in vertices_to_remove:
                triangulation_topomesh.remove_wisp(0,v)
            triangulation_topomesh.update_wisp_property('barycenter',0,array_dict(triangulation_topomesh.wisp_property('barycenter',0).values(list(triangulation_topomesh.wisps(0))),list(triangulation_topomesh.wisps(0))))

        compute_tetrahedrization_topological_properties(triangulation_topomesh)

    if exterior:
        tetrahedrization_topomesh_add_exterior(triangulation_topomesh)
        compute_tetrahedrization_topological_properties(triangulation_topomesh)







        
