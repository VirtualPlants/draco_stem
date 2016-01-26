# -*- coding: utf-8 -*-
# -*- python -*-
#
#       DRACO-STEM
#       Dual Reconstruction by Adjacency Complex Optimization
#       SAM Tissue Enhanced Mesh
#
#       Copyright 2015-2016 INRIA - CIRAD - INRA
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
from scipy.cluster.vq import vq

from openalea.container import array_dict

from openalea.mesh import PropertyTopomesh
from openalea.mesh.property_topomesh_analysis import compute_topomesh_property, compute_topomesh_triangle_properties
from openalea.mesh.property_topomesh_optimization import property_topomesh_vertices_deformation, property_topomesh_edge_flip_optimization

from copy import deepcopy
from time import time


def optimize_topomesh(input_topomesh,omega_forces={'regularization':0.00,'laplacian':1.0,'planarization':0.27,'epidermis_planarization':0.07},omega_regularization_max=0.05,iterations=20,edge_flip=False,**kwargs):
    """
    """

    topomesh = deepcopy(input_topomesh)

    preparation_start_time = time()
    # topomesh.update_wisp_property('barycenter',degree=0,values=initial_vertex_positions,keys=np.array(list(topomesh.wisps(0))))

    compute_topomesh_property(topomesh,'valence',degree=0)
    compute_topomesh_property(topomesh,'borders',degree=1)
    compute_topomesh_property(topomesh,'vertices',degree=1)
    compute_topomesh_property(topomesh,'vertices',degree=2)
    compute_topomesh_property(topomesh,'vertices',degree=3)

    compute_topomesh_property(topomesh,'cells',degree=2)
    compute_topomesh_property(topomesh,'cells',degree=1)
    compute_topomesh_property(topomesh,'cells',degree=0)

    compute_topomesh_property(topomesh,'length',degree=1)

    compute_topomesh_property(topomesh,'barycenter',degree=3)
    compute_topomesh_property(topomesh,'barycenter',degree=2)
    compute_topomesh_property(topomesh,'barycenter',degree=1)
    
    triangular_mesh = kwargs.get('triangular_mesh',True)
    if triangular_mesh:
        compute_topomesh_triangle_properties(topomesh)
        compute_topomesh_property(topomesh,'normal',degree=2)
        compute_topomesh_property(topomesh,'angles',degree=2)

    compute_topomesh_property(topomesh,'epidermis',degree=0)
    compute_topomesh_property(topomesh,'epidermis',degree=1)
    # compute_topomesh_property(topomesh,'epidermis',degree=3)

    if omega_forces.has_key('planarization'):
        start_time = time()
        print "--> Computing interfaces"
        for cid in topomesh.wisps(3):
            for n_cid in topomesh.border_neighbors(3,cid):
                if (n_cid<cid) and (not (n_cid,cid) in topomesh._interface[3].values()):
                    iid = topomesh._interface[3].add((n_cid,cid),None)
        end_time = time()
        print "<-- Computing interfaces[",end_time-start_time,"s]"

    preparation_end_time = time()
    print "--> Preparing topomesh     [",preparation_end_time-preparation_start_time,"s]"

    display = kwargs.get('display',False)
    if display:
        pass

    optimization_start_time = time()

    if omega_forces.has_key('regularization'):
    	if omega_regularization_max is None:
        	omega_regularization_max = omega_forces['regularization']

    gradient_derivatives = kwargs.get("gradient_derivatives",[])

    cell_vertex_motion = kwargs.get("cell_vertex_motion",False)
    if cell_vertex_motion:
        image_cell_vertex = deepcopy(kwargs.get("image_cell_vertex",{}))
        for v in image_cell_vertex.keys():
            image_cell_vertex[v] = image_cell_vertex[v]*np.array(kwargs.get("image_resolution",(1.0,1.0,1.0)))
            #image_cell_vertex[v] = image_cell_vertex[v]
    
        compute_topomesh_property(topomesh,'cells',degree=0)
        vertex_cell_neighbours = topomesh.wisp_property('cells',degree=0)

        epidermis_vertices = np.array(list(topomesh.wisps(0)))[np.where(topomesh.wisp_property('epidermis',degree=0).values())]

        start_time = time()
        print "--> Computing mesh cell vertices"
        mesh_cell_vertex = {}
        for v in topomesh.wisps(0):
            if len(vertex_cell_neighbours[v]) == 5:
                for k in xrange(5):
                    vertex_cell_labels = tuple([c for c in vertex_cell_neighbours[v]][:k])+tuple([c for c in vertex_cell_neighbours[v]][k+1:])
                    if not mesh_cell_vertex.has_key(vertex_cell_labels):
                        mesh_cell_vertex[vertex_cell_labels] = v
            if len(vertex_cell_neighbours[v]) == 4:
                vertex_cell_labels = tuple([c for c in vertex_cell_neighbours[v]])
                mesh_cell_vertex[vertex_cell_labels] = v
                if v in epidermis_vertices: 
                # and count_l1_cells(vertex_cell_neighbours[v]) == 4:
                    for k in xrange(4):
                        vertex_cell_labels = (1,) + tuple([c for c in vertex_cell_neighbours[v]][:k])+tuple([c for c in vertex_cell_neighbours[v]][k+1:])
                        if not mesh_cell_vertex.has_key(vertex_cell_labels):
                            mesh_cell_vertex[vertex_cell_labels] = v
            if (len(vertex_cell_neighbours[v]) == 3) and (v in epidermis_vertices):
            # and (count_l1_cells(vertex_cell_neighbours[v]) == 3): 
                vertex_cell_labels = (1,) + tuple([c for c in vertex_cell_neighbours[v]])
                mesh_cell_vertex[vertex_cell_labels] = v
        end_time = time()
        print "<-- Computing mesh cell vertices [",end_time-start_time,"s]"

        cell_vertex_matching = vq(np.sort(array_dict(image_cell_vertex).keys()),np.sort(array_dict(mesh_cell_vertex).keys()))

        matched_image_index = np.where(cell_vertex_matching[1] == 0)[0]
        matched_mesh_index = cell_vertex_matching[0][np.where(cell_vertex_matching[1] == 0)[0]]

        matched_image_cell_vertex = np.array(image_cell_vertex.values())[matched_image_index]
        matched_keys = np.sort(np.array(image_cell_vertex.keys()))[matched_image_index]

        matched_mesh_vertices = np.array(mesh_cell_vertex.values())[cell_vertex_matching[0][np.where(cell_vertex_matching[1] == 0)[0]]]
        matched_keys = np.sort(np.array(mesh_cell_vertex.keys()))[matched_mesh_index]

        initial_vertex_positions = array_dict(topomesh.wisp_property('barycenter',0).values(list(topomesh.wisps(0))),list(topomesh.wisps(0)))

        final_vertex_positions = array_dict()
        fixed_vertex = array_dict(np.array([False for v in topomesh.wisps(0)]),np.array(list(topomesh.wisps(0))))
        for i,v in enumerate(matched_mesh_vertices):
            if not np.isnan(matched_image_cell_vertex[i]).any():
                final_vertex_positions[v] = matched_image_cell_vertex[i]
                print topomesh.wisp_property('barycenter',0)[v]," -> ",final_vertex_positions[v]
                fixed_vertex[v] = True
        matched_mesh_vertices = final_vertex_positions.keys()

    sigma_deformation_initial = kwargs.get("sigma_deformation",np.sqrt(3)/4.)
    sigma_deformation = sigma_deformation_initial*np.ones_like(np.array(list(topomesh.wisps(0))),float)

    if cell_vertex_motion:
        sigma_deformation[np.where(fixed_vertex.values())[0]] = 0.

    iterations_per_step = kwargs.get('iterations_per_step',1)

    for iteration in xrange(iterations/iterations_per_step+1):

        print "_____________________________"
        print ""
        print "       Iteration ",iteration
        print "_____________________________"
        start_time = time()
        
        gaussian_sigma = kwargs.get('gaussian_sigma',10.0)
        property_topomesh_vertices_deformation(topomesh,iterations=iterations_per_step,omega_forces=omega_forces,sigma_deformation=sigma_deformation,gradient_derivatives=gradient_derivatives,resolution=kwargs.get("image_resolution",(1.0,1.0,1.0)),gaussian_sigma=gaussian_sigma)

        if cell_vertex_motion:
            vertex_start_time = time()
            print "--> Moving cell vertices"
            if iteration <= (iterations/iterations_per_step+1)/1.:
                #topomesh.update_wisp_property('barycenter',degree=0,values=((iterations/iterations_per_step+1-(iteration+1))*initial_vertex_positions.values(matched_mesh_vertices) + (iteration+1)*final_vertex_positions.values(matched_mesh_vertices))/(iterations/iterations_per_step+1),keys=matched_mesh_vertices,erase_property=False)
                for v in matched_mesh_vertices:
                    topomesh.wisp_property('barycenter',degree=0)[v] = ((iterations/iterations_per_step+1-(iteration+1))*initial_vertex_positions[v] + (iteration+1)*final_vertex_positions[v])/(iterations/iterations_per_step+1)
            vertex_end_time = time()
            print "<-- Moving cell vertices     [",vertex_end_time-vertex_start_time,"s]"

        compute_topomesh_property(topomesh,'length',degree=1)
        compute_topomesh_property(topomesh,'barycenter',degree=3)
        compute_topomesh_property(topomesh,'barycenter',degree=2)

        if triangular_mesh:
            compute_topomesh_triangle_properties(topomesh)
            compute_topomesh_property(topomesh,'normal',degree=2)

        if edge_flip:
            # property_topomesh_edge_flip_optimization(topomesh,omega_energies=omega_forces,simulated_annealing=True,iterations=15,display=display)
            property_topomesh_edge_flip_optimization(topomesh,omega_energies=omega_forces,simulated_annealing=False,iterations=3,display=display)

            compute_topomesh_property(topomesh,'length',degree=1)
            compute_topomesh_property(topomesh,'barycenter',degree=3)
            compute_topomesh_property(topomesh,'barycenter',degree=2)
            if triangular_mesh:
                compute_topomesh_triangle_properties(topomesh)
                compute_topomesh_property(topomesh,'normal',degree=2)

        sigma_deformation = sigma_deformation_initial*np.power(0.95,(iteration+1)*iterations_per_step)*np.ones_like(np.array(list(topomesh.wisps(0))),float)
        if cell_vertex_motion:
            sigma_deformation[np.where(fixed_vertex.values())[0]] = 0.

        if omega_forces.has_key('regularization'):
            omega_forces['regularization'] = np.minimum(omega_forces['regularization']+(omega_regularization_max*iterations_per_step)/iterations,omega_regularization_max)

        if display:
            pass
            
        end_time = time()
        print "_____________________________"
        print ""
        print "      [",end_time-start_time,"s]"
    #raw_input()
    print "_____________________________"

    optimization_end_time = time()
    print "--> Optimizing Topomesh    [",optimization_end_time - optimization_start_time,"s]"

    return topomesh

