# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
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
from scipy import ndimage as nd

from scipy.cluster.vq import vq

from openalea.container.array_dict import array_dict
from openalea.container.property_topomesh import PropertyTopomesh
from openalea.mesh.property_topomesh_analysis import *
from openaela.mesh.utils.tissue_analysis_tools import cell_vertex_extraction
from openalea.mesh.utils.array_tools import array_unique
from openalea.container.topomesh_algo import is_collapse_topo_allowed, collapse_edge

from time                                   import time
from copy                                   import deepcopy


def property_topomesh_vertices_deformation(topomesh,iterations=1,omega_forces=dict([('gradient',0.1),('regularization',0.5)]),sigma_deformation=2*np.sqrt(3),gradient_derivatives=None,gaussian_sigma=10.0,resolution=(1.0,1.0,1.0),target_normal=None,distance_threshold=2*np.sqrt(3),edge_collapse=False):
    """todo"""

    for iteration in xrange(iterations):

        deformation_force = np.zeros_like(topomesh.wisp_property('barycenter',degree=0).values(),np.float32)

        if omega_forces.has_key('gradient') and omega_forces['gradient']!=0.0:
            start_time = time()
            print "--> Computing vertex force"
            assert gradient_derivatives != None
            gradient_force = property_topomesh_cell_vertex_force(topomesh,gradient_derivatives,resolution)
            print gradient_force
            deformation_force += omega_forces['gradient']*gradient_force
            end_time = time()
            print "<-- Computing vertex force   [",end_time-start_time,"s]"

        if omega_forces.has_key('regularization') and omega_forces['regularization']!=0.0:
            start_time = time()
            print "--> Computing regularization force"
            regularization_force = property_topomesh_triangle_regularization_force(topomesh)
            deformation_force += omega_forces['regularization']*regularization_force
            end_time = time()
            print "<-- Computing regularization force [",end_time-start_time,"s]"

        if omega_forces.has_key('area') and omega_forces['area']!=0.0:
            start_time = time()
            print "--> Computing area force"
            area_force = property_topomesh_area_smoothing_force(topomesh)
            deformation_force += omega_forces['area']*area_force
            end_time = time()
            print "<-- Computing area force [",end_time-start_time,"s]"

        if omega_forces.has_key('laplacian') and omega_forces['laplacian']!=0.0:
            start_time = time()
            print "--> Computing laplacian smoothing force"
            # laplacian_force = property_topomesh_laplacian_smoothing_force(topomesh)
            laplacian_force = property_topomesh_laplacian_epidermis_convexity_force(topomesh)
            deformation_force += omega_forces['laplacian']*laplacian_force
            end_time = time()
            print "<-- Computing laplacian smoothing force [",end_time-start_time,"s]"

        if omega_forces.has_key('laplacian_smoothing') and omega_forces['laplacian_smoothing']!=0.0:
            start_time = time()
            print "--> Computing laplacian smoothing force"
            laplacian_force = property_topomesh_laplacian_smoothing_force(topomesh)
            deformation_force += omega_forces['laplacian_smoothing']*laplacian_force
            end_time = time()
            print "<-- Computing laplacian smoothing force [",end_time-start_time,"s]"

        if omega_forces.has_key('gaussian_smoothing') and omega_forces['gaussian_smoothing']!=0.0:
            start_time = time()
            print "--> Computing gaussian smoothing force"
            gaussian_force = property_topomesh_gaussian_smoothing_force(topomesh,gaussian_sigma=gaussian_sigma)
            deformation_force += omega_forces['gaussian_smoothing']*gaussian_force
            end_time = time()
            print "<-- Computing gaussian smoothing force [",end_time-start_time,"s]"

        if omega_forces.has_key('curvature_flow_smoothing') and omega_forces['curvature_flow_smoothing']!=0.0:
            start_time = time()
            print "--> Computing curvature flow smoothing force"
            curvature_flow_force = property_topomesh_cotangent_laplacian_smoothing_force(topomesh)
            deformation_force += omega_forces['curvature_flow_smoothing']*curvature_flow_force
            end_time = time()
            print "<-- Computing curvature flow smoothing force [",end_time-start_time,"s]"

        if omega_forces.has_key('mean_curvature_smoothing') and omega_forces['mean_curvature_smoothing']!=0.0:
            start_time = time()
            print "--> Computing mean curvature smoothing force"
            mean_curvature_force = property_topomesh_mean_curvature_smoothing_force(topomesh)
            deformation_force += omega_forces['mean_curvature_smoothing']*mean_curvature_force
            end_time = time()
            print "<-- Computing mean curvature smoothing force [",end_time-start_time,"s]"

        if omega_forces.has_key('taubin_smoothing') and omega_forces['taubin_smoothing']!=0.0:
            start_time = time()
            print "--> Computing taubin smoothing force"
            taubin_force = property_topomesh_taubin_smoothing_force(topomesh,gaussian_sigma=gaussian_sigma)
            deformation_force += omega_forces['taubin_smoothing']*taubin_force
            end_time = time()
            print "<-- Computing taubin smoothing force [",end_time-start_time,"s]"

        if omega_forces.has_key('global_taubin_smoothing') and omega_forces['global_taubin_smoothing']!=0.0:
            start_time = time()
            print "--> Computing taubin smoothing force"
            taubin_force = property_topomesh_taubin_smoothing_force(topomesh,gaussian_sigma=20.0,cellwise_smoothing=False)
            deformation_force += omega_forces['global_taubin_smoothing']*taubin_force
            end_time = time()
            print "<-- Computing taubin smoothing force [",end_time-start_time,"s]"

        if omega_forces.has_key('planarization') and omega_forces['planarization']!=0.0:
            start_time = time()
            print "--> Computing planarization force"
            if target_normal is not None:
                planarization_force = property_topomesh_planarization_force(topomesh,target_normal)
            else:
                # planarization_force = property_topomesh_cell_interface_planarization_force(topomesh)
                planarization_force = property_topomesh_interface_planarization_force(topomesh) 
                # planarization_force = property_topomesh_interface_planarization_force(topomesh) + property_topomesh_epidermis_planarization_force(topomesh)
                # planarization_force = property_topomesh_interface_planarization_force(topomesh) + property_topomesh_epidermis_convexity_force(topomesh)
                # planarization_force = property_topomesh_epidermis_convexity_force(topomesh)
            print planarization_force
            deformation_force += omega_forces['planarization']*planarization_force
            end_time = time()
            print "<-- Computing planarization force [",end_time-start_time,"s]"
        
        if omega_forces.has_key('epidermis_planarization') and omega_forces['epidermis_planarization']!=0.0:
            start_time = time()
            print "--> Computing planarization force"
            planarization_force = property_topomesh_epidermis_planarization_force(topomesh)
            deformation_force += omega_forces['epidermis_planarization']*planarization_force
            end_time = time()
            print "<-- Computing planarization force [",end_time-start_time,"s]"

        if omega_forces.has_key('convexity') and omega_forces['convexity']!=0.0:
            start_time = time()
            print "--> Computing epidermis convexity force"
            convexity_force = property_topomesh_epidermis_convexity_force(topomesh)
            deformation_force += omega_forces['convexity']*convexity_force
            end_time = time()
            print "<-- Computing epidermis convexity force [",end_time-start_time,"s]"

        start_time = time()
        print "--> Applying Forces"
        deformation_force_amplitude = np.power(np.sum(np.power(deformation_force,2.0),axis=1),0.5)+np.power(10.,-8)
        print deformation_force
        #deformation_force[np.where(deformation_force_amplitude>sigma_deformation)[0]] = sigma_deformation*deformation_force[np.where(deformation_force_amplitude>sigma_deformation)[0]]/deformation_force_amplitude[np.where(deformation_force_amplitude>sigma_deformation)[0]][:,np.newaxis]
        
        deformation_force = np.minimum(1.0,sigma_deformation/deformation_force_amplitude)[:,np.newaxis] * deformation_force
        topomesh.update_wisp_property('barycenter',degree=0,values=topomesh.wisp_property('barycenter',degree=0).values(list(topomesh.wisps(0))) + deformation_force,keys=list(topomesh.wisps(0)))
        
        end_time = time()
        print "<-- Applying Forces          [",end_time-start_time,"s]"

        start_time = time()
        print "--> Updating distances"
        compute_topomesh_property(topomesh,'length',degree=1)
        end_time = time()
        print "<-- Updating distances       [",end_time-start_time,"s]"

        # if edge_collapse:

        #     start_time = time()
        #     print "--> Collapsing small edges"
            
        #     sorted_distances = np.array(sorted(enumerate(distances.values() ),key=lambda x:x[1]))

        #     edges_to_collapse = edge_vertices.values()[np.array(sorted_distances[:,0],int)[np.where(sorted_distances[:,1] < distance_threshold)]]
        #     edge_distances = distances.values()[np.array(sorted_distances[:,0],int)[np.where(sorted_distances[:,1] < distance_threshold)]]

        #     e_index = 0
        #     while e_index<len(edges_to_collapse):
        #         e_start_time = time()
        #         edge_start_time = time()
        #         edge_pids = edges_to_collapse[e_index]

        #         kept_pid,suppressed_pid = collapse_edge_with_graph(topomesh,vertex_graph,edge_pids)
        #         positions[kept_pid] = vertex_graph.vertex_property('barycenter')[kept_pid]
        #         del positions[suppressed_pid]
        #         edges_to_collapse = np.delete(edges_to_collapse,np.where(edges_to_collapse==suppressed_pid)[0],axis=0)
        #         e_index += 1
        #     print"  <-- ",e_index," edges collapsed" 
        #     end_time = time()
        #     print "<-- Collapsing small edges   [",end_time-start_time,"s]"

        print topomesh.nb_wisps(0)," Vertices, ",topomesh.nb_wisps(2)," Triangles, ",topomesh.nb_wisps(3)," Cells"

def property_topomesh_cell_vertex_force(topomesh,gradient_derivatives,resolution):
    """
    Compute for each vertex of the topomesh the force guiding its displacement towards a cell vertex
    """

    gradient_x = gradient_derivatives[0]
    gradient_y = gradient_derivatives[1]
    gradient_z = gradient_derivatives[2]

    vertices_coords = np.rollaxis(np.array(topomesh.wisp_property('barycenter',degree=0).values()/np.array(resolution),np.uint16),1)
    vertices_coords = np.maximum(np.minimum(vertices_coords,([gradient_x.shape[0]-1],[gradient_x.shape[1]-1],[gradient_x.shape[2]-1])),0)
    print vertices_coords
    gradient_force = np.rollaxis(np.array([gradient_x[tuple(vertices_coords)],gradient_y[tuple(vertices_coords)],gradient_z[tuple(vertices_coords)]]),1)*np.array(resolution)[np.newaxis,:]
    
    return gradient_force


def property_topomesh_triangle_regularization_force(topomesh):
    """todo"""

    if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'vertices',degree=2)
    if not topomesh.has_wisp_property('length',degree=1,is_computed=True):
        compute_topomesh_property(topomesh,'length',degree=1)

    triangle_vertices = topomesh.wisp_property('vertices',degree=2).values()
    rotated_triangle_vertices = np.transpose([triangle_vertices[:,2],triangle_vertices[:,0],triangle_vertices[:,1]])
    antirotated_triangle_vertices = np.transpose([triangle_vertices[:,1],triangle_vertices[:,2],triangle_vertices[:,0]])
    triangle_vertices = np.append(np.append(triangle_vertices,rotated_triangle_vertices,axis=0),antirotated_triangle_vertices,axis=0)

    edge_index_list = np.array([[1, 2],[0, 2],[0, 1]])
    triangle_edge_vertices = triangle_vertices[:,edge_index_list]

    triangle_edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,1]) - topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,0])
    #triangle_edge_lengths = np.power(np.sum(np.power(triangle_edge_vectors,2.0),axis=2),0.5)
    triangle_edge_lengths = np.linalg.norm(triangle_edge_vectors,axis=2)
    triangle_edge_directions = triangle_edge_vectors/triangle_edge_lengths[...,np.newaxis]

    triangle_perimeters = np.sum(triangle_edge_lengths,axis=1)
    triangle_areas = np.sqrt((triangle_perimeters/2.0)*(triangle_perimeters/2.0-triangle_edge_lengths[:,0])*(triangle_perimeters/2.0-triangle_edge_lengths[:,1])*(triangle_perimeters/2.0-triangle_edge_lengths[:,2]))
    triangle_areas[np.where(triangle_areas==0.0)] = 0.001

    average_area = np.mean(triangle_areas)

    triangle_sinuses = np.zeros_like(triangle_edge_lengths,np.float32)
    triangle_sinuses[:,0] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2],2.0),np.float16))
    triangle_sinuses[:,1] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0],2.0),np.float16))
    triangle_sinuses[:,2] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1],2.0),np.float16))
    triangle_sinuses[np.where(triangle_sinuses == 0.0)] = 0.001

    sinus_edge_1_force =  (((np.power(triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2,2.0)-np.power(triangle_edge_lengths[:,1],4.0))*triangle_edge_lengths[:,2])/(2.0*np.power(triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2],3.0)))/(2.0*triangle_sinuses[:,0])
    sinus_edge_1_force += ((2.*(triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2)*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2])/(2.0*np.power(triangle_edge_lengths[:,0]*triangle_edge_lengths[:,2],3.0)))/(2.0*triangle_sinuses[:,1])
    sinus_edge_1_force += (((np.power(triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2,2.0)-np.power(triangle_edge_lengths[:,1],4.0))*triangle_edge_lengths[:,0])/(2.0*np.power(triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1],3.0)))/(2.0*triangle_sinuses[:,2])

    sinus_edge_2_force =  (((np.power(triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,0]**2,2.0)-np.power(triangle_edge_lengths[:,2],4.0))*triangle_edge_lengths[:,1])/(2.0*np.power(triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2],3.0)))/(2.0*triangle_sinuses[:,0])
    sinus_edge_2_force += (((np.power(triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,0]**2,2.0)-np.power(triangle_edge_lengths[:,2],4.0))*triangle_edge_lengths[:,0])/(2.0*np.power(triangle_edge_lengths[:,0]*triangle_edge_lengths[:,2],3.0)))/(2.0*triangle_sinuses[:,1])
    sinus_edge_2_force += ((2.*(triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,2]**2)*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2])/(2.0*np.power(triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1],3.0)))/(2.0*triangle_sinuses[:,2])

    #sinus_unitary_force = -(sinus_edge_1_force[:,np.newaxis]*triangle_edge_directions[:,1] + sinus_edge_2_force[:,np.newaxis]*triangle_edge_directions[:,2])*triangle_areas[:,np.newaxis]
    sinus_unitary_force = -(sinus_edge_1_force[:,np.newaxis]*triangle_edge_vectors[:,1] + sinus_edge_2_force[:,np.newaxis]*triangle_edge_vectors[:,2])

    area_edge_1_force = -((triangle_areas - average_area)*(triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,1]**2)*triangle_edge_lengths[:,1])/(4.0*triangle_areas)
    area_edge_2_force = -((triangle_areas - average_area)*(triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2)*triangle_edge_lengths[:,2])/(4.0*triangle_areas)
    #area_edge_1_force = -((triangle_areas - average_area)*(triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,1]**2)*triangle_edge_lengths[:,1])/(4.0*average_area)
    #area_edge_2_force = -((triangle_areas - average_area)*(triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2)*triangle_edge_lengths[:,2])/(4.0*average_area)

    area_unitary_force = -(area_edge_1_force[:,np.newaxis]*triangle_edge_directions[:,1] + area_edge_2_force[:,np.newaxis]*triangle_edge_directions[:,2])

    #triangle_unitary_force = 8.0*sinus_unitary_force
    #triangle_unitary_force = sinus_unitary_force
    triangle_unitary_force = sinus_unitary_force*np.power(average_area,1)
    #triangle_unitary_force = sinus_unitary_force*np.power(average_area,1) + area_unitary_force
    #triangle_unitary_force = sinus_unitary_force + area_unitary_force
    #triangle_unitary_force = 0.02*area_unitary_force + 8.0*sinus_unitary_force
    # triangle_unitary_force = 0.02*area_unitary_force

    triangle_force = np.transpose([nd.sum(triangle_unitary_force[:,0],triangle_vertices[:,0],index=list(topomesh.wisps(0))),
                                   nd.sum(triangle_unitary_force[:,1],triangle_vertices[:,0],index=list(topomesh.wisps(0))),
                                   nd.sum(triangle_unitary_force[:,2],triangle_vertices[:,0],index=list(topomesh.wisps(0)))])

    return triangle_force

def property_topomesh_area_smoothing_force(topomesh):

    compute_topomesh_property(topomesh,'vertices',degree=2)
    compute_topomesh_property(topomesh,'length',degree=1)

    triangle_vertices = topomesh.wisp_property('vertices',degree=2).values()
    rotated_triangle_vertices = np.transpose([triangle_vertices[:,2],triangle_vertices[:,0],triangle_vertices[:,1]])
    antirotated_triangle_vertices = np.transpose([triangle_vertices[:,1],triangle_vertices[:,2],triangle_vertices[:,0]])
    triangle_vertices = np.append(np.append(triangle_vertices,rotated_triangle_vertices,axis=0),antirotated_triangle_vertices,axis=0)

    edge_index_list = np.array([[1, 2],[0, 1],[0, 2]])
    triangle_edge_vertices = triangle_vertices[:,edge_index_list]

    triangle_edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,1]) - topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,0])
    #triangle_edge_lengths = np.power(np.sum(np.power(triangle_edge_vectors,2.0),axis=2),0.5)
    triangle_edge_lengths = np.linalg.norm(triangle_edge_vectors,axis=2)
    triangle_edge_directions = triangle_edge_vectors/triangle_edge_lengths[...,np.newaxis]

    triangle_perimeters = np.sum(triangle_edge_lengths,axis=1)
    triangle_areas = np.sqrt((triangle_perimeters/2.0)*(triangle_perimeters/2.0-triangle_edge_lengths[:,0])*(triangle_perimeters/2.0-triangle_edge_lengths[:,1])*(triangle_perimeters/2.0-triangle_edge_lengths[:,2]))
    triangle_areas[np.where(triangle_areas==0.0)] = 0.001

    area_edge_1_force = -triangle_areas*(triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,1]**2)*triangle_edge_lengths[:,1]
    area_edge_2_force = -triangle_areas*(triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2)*triangle_edge_lengths[:,2]

    area_unitary_force = -(area_edge_1_force[:,np.newaxis]*triangle_edge_directions[:,1] + area_edge_2_force[:,np.newaxis]*triangle_edge_directions[:,2])

    triangle_force = np.transpose([nd.sum(area_unitary_force[:,d],triangle_vertices[:,0],index=list(topomesh.wisps(0))) for d in xrange(3)])

    return triangle_force


def property_topomesh_laplacian_smoothing_force(topomesh,cellwise_smoothing=False):

    if not topomesh.has_wisp_property('vertices',degree=1,is_computed=True):
        compute_topomesh_property(topomesh,'vertices',degree=1)
    edge_vertices = topomesh.wisp_property('vertices',degree=1).values()
    reversed_edge_vertices = np.transpose([edge_vertices[:,1],edge_vertices[:,0]])
    edge_vertices = np.append(edge_vertices,reversed_edge_vertices,axis=0)

    if not topomesh.has_wisp_property('valence',degree=0,is_computed=True):
        compute_topomesh_property(topomesh,'valence',degree=0)
    vertices_degrees = topomesh.wisp_property('valence',degree=0).values()

    if cellwise_smoothing:

        laplacian_force = np.zeros_like(topomesh.wisp_property('barycenter',degree=0).values(),np.float32)

        for c in topomesh.wisps(3):

            if not topomesh.has_wisp_property('vertices',degree=3,is_computed=True):
                compute_topomesh_property(topomesh,'vertices',degree=3)
            cell_vertices = topomesh.wisp_property('vertices',degree=3)[c]

            cell_edges = np.sum(nd.sum(np.ones_like(cell_vertices),cell_vertices,index=edge_vertices),axis=1)
            cell_edge_vertices = edge_vertices[np.where(cell_edges==2)[0]]
            cell_edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(cell_edge_vertices[:,1]) - topomesh.wisp_property('barycenter',degree=0).values(cell_edge_vertices[:,0])

            cell_laplacian_force = np.transpose([nd.sum(cell_edge_vectors[:,0],cell_edge_vertices[:,0],index=cell_vertices),
                                                 nd.sum(cell_edge_vectors[:,1],cell_edge_vertices[:,0],index=cell_vertices),
                                                 nd.sum(cell_edge_vectors[:,2],cell_edge_vertices[:,0],index=cell_vertices)])
            laplacian_force[cell_vertices] += cell_laplacian_force/vertices_degrees[cell_vertices,np.newaxis]

    else:
        edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(edge_vertices[:,1]) - topomesh.wisp_property('barycenter',degree=0).values(edge_vertices[:,0])

        laplacian_force = np.transpose([nd.sum(edge_vectors[:,d],edge_vertices[:,0],index=list(topomesh.wisps(0))) for d in [0,1,2]])
        laplacian_force = laplacian_force/vertices_degrees[:,np.newaxis]

    return laplacian_force

def property_topomesh_cotangent_laplacian_smoothing_force(topomesh):
    
    compute_topomesh_property(topomesh,'vertices',degree=2)
    compute_topomesh_property(topomesh,'length',degree=1)

    triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(list(topomesh.wisps(2)))
    rotated_triangle_vertices = np.transpose([triangle_vertices[:,2],triangle_vertices[:,0],triangle_vertices[:,1]])
    antirotated_triangle_vertices = np.transpose([triangle_vertices[:,1],triangle_vertices[:,2],triangle_vertices[:,0]])
    triangle_vertices = np.append(np.append(triangle_vertices,rotated_triangle_vertices,axis=0),antirotated_triangle_vertices,axis=0)

    edge_index_list = np.array([[1, 2],[0, 2],[0, 1]])
    triangle_edge_vertices = triangle_vertices[:,edge_index_list]

    triangle_edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,1]) - topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,0])
    #triangle_edge_lengths = np.power(np.sum(np.power(triangle_edge_vectors,2.0),axis=2),0.5)
    triangle_edge_lengths = np.linalg.norm(triangle_edge_vectors,axis=2)
    triangle_edge_directions = triangle_edge_vectors/triangle_edge_lengths[...,np.newaxis]

    triangle_perimeters = np.sum(triangle_edge_lengths,axis=1)
    triangle_areas = np.sqrt((triangle_perimeters/2.0)*(triangle_perimeters/2.0-triangle_edge_lengths[:,0])*(triangle_perimeters/2.0-triangle_edge_lengths[:,1])*(triangle_perimeters/2.0-triangle_edge_lengths[:,2]))

    triangle_cosines = np.zeros_like(triangle_edge_lengths,np.float32)
    triangle_cosines[:,0] = (triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2)/(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2])
    triangle_cosines[:,1] = (triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2)/(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0])
    triangle_cosines[:,2] = (triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2)/(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1])
    triangle_cosines[np.where(np.abs(triangle_cosines) < np.power(10.,-6))] = np.power(10.,-6)
    triangle_angles = np.arccos(triangle_cosines)

    triangle_sinuses = np.zeros_like(triangle_edge_lengths,np.float32)
    triangle_sinuses[:,0] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2],2.0),np.float16))
    triangle_sinuses[:,1] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0],2.0),np.float16))
    triangle_sinuses[:,2] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1],2.0),np.float16))
    triangle_sinuses[np.where(triangle_sinuses < np.power(10.,-6))] = np.power(10.,-6)

    triangle_cotangent_weights = (triangle_cosines/triangle_sinuses)
    #triangle_cotangent_weights = 1./triangle_edge_lengths
    #triangle_cotangent_weights = 0.5*(triangle_cosines/triangle_sinuses)/(triangle_edge_lengths + np.power(10.,-10))
    #triangle_cotangent_vectors = (triangle_cosines/triangle_sinuses)[...,np.newaxis] * triangle_edge_vectors/triangle_edge_lengths[:,np.newaxis]
    triangle_cotangent_vectors = triangle_cotangent_weights[...,np.newaxis] * triangle_edge_vectors
    #triangle_cotangent_vectors = triangle_cotangent_weights[...,np.newaxis] * triangle_edge_directions
    # triangle_cotangent_vectors = 1./np.tan(triangle_angles)[...,np.newaxis] * triangle_edge_vectors

    vertex_cotangent_force = np.transpose([nd.sum(triangle_cotangent_vectors[:,1,d]+triangle_cotangent_vectors[:,2,d],triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0)))) for d in xrange(3)])
    vertex_cotangent_sum = nd.sum(triangle_cotangent_weights[:,1] + triangle_cotangent_weights[:,2],triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0))))

    vertex_area = nd.sum(triangle_areas,triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0))))

    #laplacian_force = vertex_cotangent_force
    laplacian_force = vertex_cotangent_force/vertex_cotangent_sum[...,np.newaxis]
    #laplacian_force = vertex_cotangent_force/(4.*vertex_area[...,np.newaxis])
    laplacian_force[np.where(np.isnan(laplacian_force))] = 0.0
    laplacian_force[np.where(np.isinf(laplacian_force))] = 0.0

    return laplacian_force

def property_topomesh_gaussian_smoothing_force(topomesh,gaussian_sigma=10.0):

    compute_topomesh_property(topomesh,'vertices',degree=1)
    edge_vertices = topomesh.wisp_property('vertices',degree=1).values()

    reversed_edge_vertices = np.transpose([edge_vertices[:,1],edge_vertices[:,0]])
    edge_vertices = np.append(edge_vertices,reversed_edge_vertices,axis=0)


    edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(edge_vertices[:,1]) - topomesh.wisp_property('barycenter',degree=0).values(edge_vertices[:,0])
    edge_lengths = np.linalg.norm(edge_vectors,axis=1)
    gaussian_edge_lengths = np.exp(-np.power(edge_lengths,2.0)/np.power(gaussian_sigma,2.0))
    gaussian_edge_vectors = gaussian_edge_lengths[:,np.newaxis]*edge_vectors

    gaussian_force = np.transpose([nd.sum(gaussian_edge_vectors[:,d],edge_vertices[:,0],index=list(topomesh.wisps(0))) for d in [0,1,2]])
    vertices_weights = 1.+nd.sum(gaussian_edge_lengths,edge_vertices[:,0],index=list(topomesh.wisps(0)))
    gaussian_force = gaussian_force/vertices_weights[:,np.newaxis]

    return gaussian_force

def property_topomesh_taubin_smoothing_force(topomesh,gaussian_sigma=10.0,positive_factor=0.33,negative_factor=-0.34,cellwise_smoothing=True):

    if not topomesh.has_wisp_property('vertices',degree=1,is_computed=True):
        compute_topomesh_property(topomesh,'vertices',degree=1)

    if cellwise_smoothing:
        edge_vertices = topomesh.wisp_property('vertices',degree=1).values(np.concatenate([np.array(list(topomesh.borders(3,c,2)),int) for c in topomesh.wisps(3)]))
    else:
        edge_vertices = topomesh.wisp_property('vertices',degree=1).values()

    reversed_edge_vertices = np.transpose([edge_vertices[:,1],edge_vertices[:,0]])
    edge_vertices = np.append(edge_vertices,reversed_edge_vertices,axis=0)

    edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(edge_vertices[:,1]) - topomesh.wisp_property('barycenter',degree=0).values(edge_vertices[:,0])
    edge_lengths = np.linalg.norm(edge_vectors,axis=1)
    gaussian_edge_lengths = np.exp(-np.power(edge_lengths,2.0)/np.power(gaussian_sigma,2.0))
    gaussian_edge_vectors = gaussian_edge_lengths[:,np.newaxis]*edge_vectors

    taubin_force = np.transpose([nd.sum(gaussian_edge_vectors[:,0],edge_vertices[:,0],index=list(topomesh.wisps(0))),
                                 nd.sum(gaussian_edge_vectors[:,1],edge_vertices[:,0],index=list(topomesh.wisps(0))),
                                 nd.sum(gaussian_edge_vectors[:,2],edge_vertices[:,0],index=list(topomesh.wisps(0)))])
    #vertices_weights = 1.+nd.sum(gaussian_edge_lengths,edge_vertices[:,0],index=list(topomesh.wisps(0)))
    vertices_weights = nd.sum(gaussian_edge_lengths,edge_vertices[:,0],index=list(topomesh.wisps(0)))
    taubin_positive_force = positive_factor*taubin_force/vertices_weights[:,np.newaxis]

    taubin_positions = array_dict(topomesh.wisp_property('barycenter',degree=0).values(list(topomesh.wisps(0)))+taubin_positive_force, list(topomesh.wisps(0)))

    edge_vectors = taubin_positions.values(edge_vertices[:,1]) - taubin_positions.values(edge_vertices[:,0])
    edge_lengths = np.linalg.norm(edge_vectors,axis=1)
    gaussian_edge_lengths = np.exp(-np.power(edge_lengths,2.0)/np.power(gaussian_sigma,2.0))
    gaussian_edge_vectors = gaussian_edge_lengths[:,np.newaxis]*edge_vectors

    taubin_force = np.transpose([nd.sum(gaussian_edge_vectors[:,0],edge_vertices[:,0],index=list(topomesh.wisps(0))),
                                 nd.sum(gaussian_edge_vectors[:,1],edge_vertices[:,0],index=list(topomesh.wisps(0))),
                                 nd.sum(gaussian_edge_vectors[:,2],edge_vertices[:,0],index=list(topomesh.wisps(0)))])
    #vertices_weights = 1.+nd.sum(gaussian_edge_lengths,edge_vertices[:,0],index=list(topomesh.wisps(0)))
    vertices_weights = nd.sum(gaussian_edge_lengths,edge_vertices[:,0],index=list(topomesh.wisps(0)))
    taubin_negative_force = negative_factor*taubin_force/vertices_weights[:,np.newaxis]

    taubin_force = taubin_positive_force + taubin_negative_force

    return taubin_force

def property_topomesh_mean_curvature_smoothing_force(topomesh):
    """todo"""

    if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'vertices',degree=2)
    if not topomesh.has_wisp_property('cells',degree=0,is_computed=True):
        compute_topomesh_property(topomesh,'cells',degree=0)
    if not topomesh.has_wisp_property('length',degree=1,is_computed=True):
        compute_topomesh_property(topomesh,'length',degree=1)


    triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(list(topomesh.wisps(2)))
    rotated_triangle_vertices = np.transpose([triangle_vertices[:,2],triangle_vertices[:,0],triangle_vertices[:,1]])
    antirotated_triangle_vertices = np.transpose([triangle_vertices[:,1],triangle_vertices[:,2],triangle_vertices[:,0]])
    triangle_vertices = np.append(np.append(triangle_vertices,rotated_triangle_vertices,axis=0),antirotated_triangle_vertices,axis=0)

    edge_index_list = np.array([[1, 2],[0, 2],[0, 1]])
    triangle_edge_vertices = triangle_vertices[:,edge_index_list]

    triangle_edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,1]) - topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,0])
    #triangle_edge_lengths = np.power(np.sum(np.power(triangle_edge_vectors,2.0),axis=2),0.5)
    triangle_edge_lengths = np.linalg.norm(triangle_edge_vectors,axis=2)
    triangle_edge_directions = triangle_edge_vectors/triangle_edge_lengths[...,np.newaxis]

    triangle_perimeters = np.sum(triangle_edge_lengths,axis=1)
    triangle_areas = np.sqrt((triangle_perimeters/2.0)*(triangle_perimeters/2.0-triangle_edge_lengths[:,0])*(triangle_perimeters/2.0-triangle_edge_lengths[:,1])*(triangle_perimeters/2.0-triangle_edge_lengths[:,2]))

    triangle_cosines = np.zeros_like(triangle_edge_lengths,np.float32)
    triangle_cosines[:,0] = (triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2)/(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2])
    triangle_cosines[:,1] = (triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2)/(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0])
    triangle_cosines[:,2] = (triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2)/(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1])
    triangle_angles = np.arccos(triangle_cosines)

    triangle_sinuses = np.zeros_like(triangle_edge_lengths,np.float32)
    triangle_sinuses[:,0] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2],2.0),np.float16))
    triangle_sinuses[:,1] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0],2.0),np.float16))
    triangle_sinuses[:,2] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1],2.0),np.float16))
    triangle_sinuses[np.where(triangle_sinuses == 0.0)] = 0.001

    triangle_cotangent_vectors = (triangle_cosines/triangle_sinuses)[...,np.newaxis] * triangle_edge_vectors
    # triangle_cotangent_vectors = 1./np.tan(triangle_angles)[...,np.newaxis] * triangle_edge_vectors

    vertex_cotangent_sum = np.transpose([nd.sum(triangle_cotangent_vectors[:,1,0]+triangle_cotangent_vectors[:,2,0],triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0)))),
                                         nd.sum(triangle_cotangent_vectors[:,1,1]+triangle_cotangent_vectors[:,2,1],triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0)))),
                                         nd.sum(triangle_cotangent_vectors[:,1,2]+triangle_cotangent_vectors[:,2,2],triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0))))])
            

    vertex_area = nd.sum(triangle_areas,triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0))))
    # vertex_area = nd.sum(triangle_cotangent_square_lengths[:,1]+triangle_cotangent_square_lengths[:,2], triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0))))/8.

    # vertex_mean_curvature_vectors = vertex_cotangent_sum/(2.*vertex_area[:,np.newaxis])
    vertex_mean_curvature_vectors = (3.*vertex_cotangent_sum)/(4.*vertex_area[:,np.newaxis])

    return vertex_mean_curvature_vectors


def property_topomesh_laplacian_epidermis_convexity_force(topomesh):
    """todo"""

    if not topomesh.has_wisp_property('vertices',degree=1,is_computed=True):
        compute_topomesh_property(topomesh,'vertices',degree=1)

    if not topomesh.has_wisp_property('barycenter',degree=3,is_computed=True):
        compute_topomesh_property(topomesh,'barycenter',degree=3)

    if not topomesh.has_wisp_property('epidermis',degree=1,is_computed=True):
        compute_topomesh_property(topomesh,'epidermis',degree=1)

    if not topomesh.has_wisp_property('cells',degree=0,is_computed=True):
        compute_topomesh_property(topomesh,'cells',degree=0)
    if not topomesh.has_wisp_property('cells',degree=1,is_computed=True):
        compute_topomesh_property(topomesh,'cells',degree=1)


    epidermis_convexity_force = array_dict(np.zeros_like(topomesh.wisp_property('barycenter',degree=0).values(),np.float32),np.array(list(topomesh.wisps(0))))

    edge_cells_degree = np.array(map(len,topomesh.wisp_property('cells',degree=1).values()))
    # edge_vertices = topomesh.wisp_property('vertices',degree=1).values()[np.where((topomesh.wisp_property('epidermis',degree=1).values()))]
    # edge_vertices = topomesh.wisp_property('vertices',degree=1).values()[np.where((topomesh.wisp_property('epidermis',degree=1).values())&(edge_cells_degree>1))]
    edge_vertices = topomesh.wisp_property('vertices',degree=1).values()[np.where((edge_cells_degree>2)|((topomesh.wisp_property('epidermis',degree=1).values())&(edge_cells_degree>1)))]

    epidermis_vertices = np.unique(edge_vertices)
    vertices_degrees = np.array(nd.sum(np.ones_like(edge_vertices),edge_vertices,index=epidermis_vertices),int)

    reversed_edge_vertices = np.transpose([edge_vertices[:,1],edge_vertices[:,0]])
    edge_vertices = np.append(edge_vertices,reversed_edge_vertices,axis=0)

    edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(edge_vertices[:,1]) - topomesh.wisp_property('barycenter',degree=0).values(edge_vertices[:,0])

    laplacian_force = np.transpose([nd.sum(edge_vectors[:,0],edge_vertices[:,0],index=epidermis_vertices),
                                    nd.sum(edge_vectors[:,1],edge_vertices[:,0],index=epidermis_vertices),
                                    nd.sum(edge_vectors[:,2],edge_vertices[:,0],index=epidermis_vertices)])

    laplacian_force = laplacian_force/vertices_degrees[:,np.newaxis]

    # vertex_cell_barycenter = np.array([np.mean(topomesh.wisp_property('barycenter',degree=3).values(c),axis=0)  for c in topomesh.wisp_property('cells',0).values(epidermis_vertices)])
    # vertices_directions = topomesh.wisp_property('barycenter',degree=0).values(epidermis_vertices) - vertex_cell_barycenter
    # vertices_directions = vertices_directions/np.linalg.norm(vertices_directions,axis=1)[:,np.newaxis]
    # vertices_products = np.einsum('ij,ij->i',laplacian_force,vertices_directions)
    # convex_points = np.where(vertices_products<0.)[0]
    # laplacian_force[convex_points] -= vertices_directions[convex_points]*vertices_products[convex_points,np.newaxis]
    # laplacian_force[convex_points] = np.array([0.,0.,0.])

    epidermis_convexity_force.update(laplacian_force,keys=epidermis_vertices,erase_missing_keys=False)

    return epidermis_convexity_force.values()


def property_topomesh_interface_planarization_force(topomesh):

    if not topomesh.has_wisp_property('barycenter',degree=3,is_computed=True):
        compute_topomesh_property(topomesh,'barycenter',degree=3)

    if not topomesh.has_wisp_property('barycenter',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'barycenter',degree=2)
    if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'vertices',degree=2)
    if not topomesh.has_wisp_property('normal',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'normal',degree=2)

    if not topomesh.has_wisp_property('cell_interface',degree=2,is_computed=False):
        compute_topomesh_property(topomesh,'cell_interface',degree=2)

    if not topomesh.has_interface_property('interface',degree=3,is_computed=True):
        compute_topomesh_property(topomesh,'interface',degree=3)

    interface_planarization_force = np.zeros_like(topomesh.wisp_property('barycenter',degree=0).values(),np.float32)

    cell_interfaces = topomesh.interface_property('interface',degree=3).keys()

    interface_triangles = topomesh.wisp_property('cell_interface',degree=2).keys()
    triangle_cell_interface = topomesh.wisp_property('cell_interface',degree=2).values()
    triangle_cells = np.array(array_dict(topomesh._interface[3]).values(triangle_cell_interface))
    triangle_cell_barycenters = topomesh.wisp_property('barycenter',degree=3).values(triangle_cells)
    triangle_interface_directions = triangle_cell_barycenters[:,1] - triangle_cell_barycenters[:,0]

    triangle_normal_vectors = topomesh.wisp_property('normal',degree=2).values(interface_triangles)
    # reversed_normals = np.where(np.dot(triangle_normal_vectors,triangle_interface_directions) < 0)[0]
    reversed_normals = np.where(np.einsum('ij,ij->i',triangle_normal_vectors,triangle_interface_directions) < 0)[0]
    triangle_normal_vectors[reversed_normals] = -triangle_normal_vectors[reversed_normals]

    interface_normal_vectors = np.transpose([nd.mean(triangle_normal_vectors[:,0],triangle_cell_interface,index=cell_interfaces),
                                             nd.mean(triangle_normal_vectors[:,1],triangle_cell_interface,index=cell_interfaces),
                                             nd.mean(triangle_normal_vectors[:,2],triangle_cell_interface,index=cell_interfaces)])
    interface_normal_vectors = interface_normal_vectors/np.linalg.norm(interface_normal_vectors,axis=1)[:,np.newaxis]
    triangle_target_normals = interface_normal_vectors[triangle_cell_interface]

    triangle_barycenters = topomesh.wisp_property('barycenter',degree=2).values(interface_triangles)
    interface_barycenters = np.transpose([nd.mean(triangle_barycenters[:,0],triangle_cell_interface,index=cell_interfaces),
                                          nd.mean(triangle_barycenters[:,1],triangle_cell_interface,index=cell_interfaces),
                                          nd.mean(triangle_barycenters[:,2],triangle_cell_interface,index=cell_interfaces)])
    triangle_interface_barycenters = interface_barycenters[triangle_cell_interface]

    triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(interface_triangles)
    interface_vertices = np.unique(triangle_vertices)
    
    rotated_triangle_vertices = np.transpose([triangle_vertices[:,2],triangle_vertices[:,0],triangle_vertices[:,1]])
    antirotated_triangle_vertices = np.transpose([triangle_vertices[:,1],triangle_vertices[:,2],triangle_vertices[:,0]])
    triangle_vertices = np.append(np.append(triangle_vertices,rotated_triangle_vertices,axis=0),antirotated_triangle_vertices,axis=0)


    triangle_point_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_vertices[:,0]) - np.tile(triangle_interface_barycenters,(3,1))
    # triangle_point_projectors = -np.dot(triangle_point_vectors,triange_target_normals)[:,np.newaxis]*triangle_target_normals
    # print triangle_point_vectors.shape,triangle_target_normals.shape
    triangle_point_projectors = -np.einsum('ij,ij->i',triangle_point_vectors,np.tile(triangle_target_normals,(3,1)))[:,np.newaxis]*np.tile(triangle_target_normals,(3,1))
    # print triangle_point_projectors.shape
    triangle_projectors_norm = np.linalg.norm(triangle_point_projectors,axis=1)
    non_planar_points = np.where(triangle_projectors_norm > 1.)[0]
    triangle_point_projectors[non_planar_points] = triangle_point_projectors[non_planar_points]/triangle_projectors_norm[non_planar_points,np.newaxis]
            
            # planarization_unitary_force = normal_errors[:,np.newaxis]*point_projectors
    planarization_unitary_force = triangle_point_projectors

    interface_planarization_force = np.transpose([nd.sum(planarization_unitary_force[:,0],triangle_vertices[:,0],index=list(topomesh.wisps(0))),
                                                  nd.sum(planarization_unitary_force[:,1],triangle_vertices[:,0],index=list(topomesh.wisps(0))),
                                                  nd.sum(planarization_unitary_force[:,2],triangle_vertices[:,0],index=list(topomesh.wisps(0)))])
    interface_planarization_force[np.isnan(interface_planarization_force)] = 0.

    return interface_planarization_force


def property_topomesh_epidermis_convexity_force(topomesh):
    if not topomesh.has_wisp_property('barycenter',degree=3,is_computed=True):
        compute_topomesh_property(topomesh,'barycenter',degree=3)

    if not topomesh.has_wisp_property('barycenter',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'barycenter',degree=2)
    if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'vertices',degree=2)
    if not topomesh.has_wisp_property('normal',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'normal',degree=2)

    if not topomesh.has_wisp_property('cells',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'cells',degree=2)
    if not topomesh.has_wisp_property('epidermis',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'epidermis',degree=2)

    epidermis_triangles = np.array(list(topomesh.wisps(2)))[np.where(topomesh.wisp_property('epidermis',2).values())]
    triangle_cell = np.array([c[0] for c in topomesh.wisp_property('cells',2).values(epidermis_triangles)])
    epidermis_cells = np.unique(triangle_cell)

    triangle_cell_barycenters = topomesh.wisp_property('barycenter',degree=3).values(triangle_cell)

    triangle_barycenters = topomesh.wisp_property('barycenter',degree=2).values(epidermis_triangles)
    cell_epidermis_barycenters = np.transpose([nd.mean(triangle_barycenters[:,0],triangle_cell,index=list(topomesh.wisps(3))),
                                               nd.mean(triangle_barycenters[:,1],triangle_cell,index=list(topomesh.wisps(3))),
                                               nd.mean(triangle_barycenters[:,2],triangle_cell,index=list(topomesh.wisps(3)))])
    cell_epidermis_barycenters = array_dict(cell_epidermis_barycenters,keys=list(topomesh.wisps(3)))
    triangle_epidermis_barycenters = cell_epidermis_barycenters.values(triangle_cell)

    # print triangle_epidermis_barycenters.shape
    triangle_epidermis_directions = triangle_epidermis_barycenters - triangle_cell_barycenters
    # print triangle_epidermis_directions.shape


    triangle_normal_vectors = topomesh.wisp_property('normal',degree=2).values(epidermis_triangles)
    # reversed_normals = np.where(np.dot(triangle_normal_vectors,triangle_interface_directions) < 0)[0]
    reversed_normals = np.where(np.einsum('ij,ij->i',triangle_normal_vectors,triangle_epidermis_directions) < 0)[0]
    triangle_normal_vectors[reversed_normals] = -triangle_normal_vectors[reversed_normals]

    # cell_normal_vectors = np.transpose([nd.mean(triangle_normal_vectors[:,0],triangle_cell,index=list(topomesh.wisps(3))),
    #                                     nd.mean(triangle_normal_vectors[:,1],triangle_cell,index=list(topomesh.wisps(3))),
    #                                     nd.mean(triangle_normal_vectors[:,2],triangle_cell,index=list(topomesh.wisps(3)))])

    # cell_normal_vectors = cell_normal_vectors/np.linalg.norm(cell_normal_vectors,axis=1)[:,np.newaxis]
    # cell_normal_vectors = array_dict(cell_normal_vectors,keys=list(topomesh.wisps(3)))

    # triangle_target_normals = cell_normal_vectors.values(triangle_cell)

    triangle_radius_direction = triangle_barycenters - triangle_cell_barycenters
    triangle_target_normals = triangle_radius_direction/np.linalg.norm(triangle_radius_direction,axis=1)[:,np.newaxis]


    triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(epidermis_triangles)
    epidermis_vertices = np.unique(triangle_vertices)
    
    rotated_triangle_vertices = np.transpose([triangle_vertices[:,2],triangle_vertices[:,0],triangle_vertices[:,1]])
    antirotated_triangle_vertices = np.transpose([triangle_vertices[:,1],triangle_vertices[:,2],triangle_vertices[:,0]])
    triangle_vertices = np.append(np.append(triangle_vertices,rotated_triangle_vertices,axis=0),antirotated_triangle_vertices,axis=0)

    # triangle_point_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_vertices[:,0]) - np.tile(triangle_epidermis_barycenters,(3,1))
    triangle_point_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_vertices[:,0]) - np.tile(triangle_barycenters,(3,1))
    triangle_plane_products = np.einsum('ij,ij->i',triangle_point_vectors,np.tile(triangle_target_normals,(3,1)))
    triangle_point_projectors = -triangle_plane_products[:,np.newaxis]*np.tile(triangle_target_normals,(3,1))
    triangle_projectors_norm = np.linalg.norm(triangle_point_projectors,axis=1)
    non_planar_points = np.where(triangle_projectors_norm > 1.)[0]
    triangle_point_projectors[non_planar_points] = triangle_point_projectors[non_planar_points]/triangle_projectors_norm[non_planar_points,np.newaxis]
    convex_points = np.where(triangle_plane_products>0.)[0]
    triangle_point_projectors[convex_points] = np.array([0.,0.,0.])

    convexity_unitary_force = triangle_point_projectors

    epidermis_convexity_force = np.transpose([nd.sum(convexity_unitary_force[:,0],triangle_vertices[:,0],index=list(topomesh.wisps(0))),
                                                  nd.sum(convexity_unitary_force[:,1],triangle_vertices[:,0],index=list(topomesh.wisps(0))),
                                                  nd.sum(convexity_unitary_force[:,2],triangle_vertices[:,0],index=list(topomesh.wisps(0)))])

    return epidermis_convexity_force

def property_topomesh_epidermis_planarization_force(topomesh):
    if not topomesh.has_wisp_property('epidermis',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'epidermis',degree=2)
    if not topomesh.has_wisp_property('epidermis',degree=3,is_computed=True):
        compute_topomesh_property(topomesh,'epidermis',degree=3)
    if not topomesh.has_wisp_property('epidermis',degree=0,is_computed=True):
        compute_topomesh_property(topomesh,'epidermis',degree=0)
    
    if not topomesh.has_wisp_property('cells',degree=0,is_computed=True):
        compute_topomesh_property(topomesh,'cells',degree=0)
    
    if not topomesh.has_wisp_property('barycenter',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'barycenter',2)
    if not topomesh.has_wisp_property('barycenter',degree=3,is_computed=True):
        compute_topomesh_property(topomesh,'barycenter',3)

    if not topomesh.has_wisp_property('length',degree=1,is_computed=True):
        compute_topomesh_property(topomesh,'length',degree=1)
    if not topomesh.has_wisp_property('area',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'area',degree=2)
    if not topomesh.has_wisp_property('normal',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'normal',degree=2)

    epidermis_triangles = np.array(list(topomesh.wisps(2)))[topomesh.wisp_property('epidermis',2).values()]
    epidermis_cells = np.array(list(topomesh.wisps(3)))[topomesh.wisp_property('epidermis',3).values()]
    epidermis_vertices = np.array(list(topomesh.wisps(0)))[topomesh.wisp_property('epidermis',0).values()]
    
    triangle_cells = np.concatenate(topomesh.wisp_property('cells',2).values(epidermis_triangles))
    triangle_areas = topomesh.wisp_property('area',2).values(epidermis_triangles)
    triangle_centers = topomesh.wisp_property('barycenter',2).values(epidermis_triangles)
    triangle_weighted_normals = triangle_areas[:,np.newaxis]*topomesh.wisp_property('normal',2).values(epidermis_triangles)
    
    cell_areas = nd.sum(triangle_areas,triangle_cells,index=epidermis_cells)
    cell_epidermis_centers = np.transpose([nd.sum(triangle_areas*triangle_centers[:,k],triangle_cells,index=epidermis_cells) for k in xrange(3)])
    cell_epidermis_centers = cell_epidermis_centers/cell_areas[:,np.newaxis]
    cell_epidermis_centers = array_dict(cell_epidermis_centers,epidermis_cells)
    
    cell_weighted_normals = np.transpose([nd.sum(triangle_weighted_normals[:,k],triangle_cells,index=epidermis_cells) for k in xrange(3)])
    cell_normals = cell_weighted_normals/cell_areas[:,np.newaxis]
    cell_normals = array_dict(cell_normals/np.linalg.norm(cell_normals,axis=1)[:,np.newaxis],epidermis_cells)
    
    vertex_cells = np.concatenate([np.intersect1d(np.array(topomesh.wisp_property('cells',0)[v],int),epidermis_cells) for v in epidermis_vertices])
    vertex_cell_vertex = np.concatenate([v*np.ones_like(np.intersect1d(topomesh.wisp_property('cells',0)[v],epidermis_cells),int) for v in epidermis_vertices])
    
    vertex_cell_normals = cell_normals.values(vertex_cells)
    #vertex_cell_normals = np.array([cell_normals[c] if cell_normals.has_key(c) else np.array([0,0,1]) for c in vertex_cells])
    vertex_cell_centers = cell_epidermis_centers.values(vertex_cells)
    #vertex_cell_centers = np.array([cell_epidermis_centers[c] if cell_epidermis_centers.has_key(c) else topomesh.wisp_property('barycenter',3)[c]])
    vertex_cell_vectors = topomesh.wisp_property('barycenter',0).values(vertex_cell_vertex) - vertex_cell_centers
    vertex_cell_projectors = -np.einsum('ij,ij->i',vertex_cell_vectors,vertex_cell_normals)[:,np.newaxis]*vertex_cell_normals
    
    vertex_projectors = np.transpose([nd.sum(vertex_cell_projectors[:,k],vertex_cell_vertex,index=list(topomesh.wisps(0))) for k in xrange(3)])
    vertex_projectors[np.isnan(vertex_projectors)] = 0.
    return vertex_projectors


def property_topomesh_cell_interface_planarization_force(topomesh):

    if not topomesh.has_wisp_property('vertices',degree=3,is_computed=True):
        compute_topomesh_property(topomesh,'vertices',degree=3)
    if not topomesh.has_wisp_property('barycenter',degree=3,is_computed=True):
        compute_topomesh_property(topomesh,'barycenter',degree=3)
    if not topomesh.has_wisp_property('borders',degree=3,is_computed=True):
        compute_topomesh_property(topomesh,'borders',degree=3)
    if not topomesh.has_wisp_property('neighbors',degree=3,is_computed=True):
        compute_topomesh_property(topomesh,'border_neighbors',degree=3)

    planarization_force = np.zeros_like(topomesh.wisp_property('barycenter',degree=0).values(),np.float32)

    for c in topomesh.wisps(3):
        print "Cell",c,"interface planarization"
        for n in topomesh.wisp_property("neighbors",degree=3)[c]:
            normal_direction = topomesh.wisp_property("barycenter",degree=3)[n] - topomesh.wisp_property("barycenter",degree=3)[c]

            # triangle_vertices = np.array([list(topomesh.borders(2,t,2)) for t in topomesh.borders(3,c) if t in topomesh.borders(3,n)])
            interface_triangles = np.intersect1d(topomesh.wisp_property('borders',degree=3)[c],topomesh.wisp_property('borders',degree=3)[n])
            triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(interface_triangles)

            rotated_triangle_vertices = np.transpose([triangle_vertices[:,2],triangle_vertices[:,0],triangle_vertices[:,1]])
            antirotated_triangle_vertices = np.transpose([triangle_vertices[:,1],triangle_vertices[:,2],triangle_vertices[:,0]])
            triangle_vertices = np.append(np.append(triangle_vertices,rotated_triangle_vertices,axis=0),antirotated_triangle_vertices,axis=0)

            # interface_vertices = np.intersect1d(cell_vertices[c],cell_vertices[n])
            interface_vertices = np.unique(triangle_vertices)

            edge_index_list = np.array([[1, 2],[0, 1],[0, 2]])
            triangle_edge_vertices = triangle_vertices[:,edge_index_list]

            triangle_edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,1]) - topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,0])
            triangle_edge_lengths = np.power(np.sum(np.power(triangle_edge_vectors,2.0),axis=2),0.5)
            triangle_edge_directions = triangle_edge_vectors/triangle_edge_lengths[...,np.newaxis]

            normal_vectors = np.cross(triangle_edge_vectors[:,1],triangle_edge_vectors[:,2])
            # reversed_normals = np.where(np.dot(normal_vectors,normal_direction) < 0)[0]
            reversed_normals = np.where(numpy.core.umath_tests.inner1d(normal_vectors,normal_direction) < 0)[0]
            normal_vectors[reversed_normals] = np.cross(triangle_edge_vectors[reversed_normals,2],triangle_edge_vectors[reversed_normals,1])
            normal_vectors = normal_vectors/np.linalg.norm(normal_vectors,axis=1)[:,np.newaxis]
            if(np.isnan(normal_vectors).any()):
                print "Interface ",c,"->",n," planarization force  [",end_time-start_time,"]"

            normal_target = np.mean(normal_vectors,axis=0)
            normal_target = normal_target/np.linalg.norm(normal_target)
            if(np.isnan(normal_target).any()):
                print "Interface ",c,"->",n," planarization force  [",end_time-start_time,"]"
            # normal_errors = 1.0 - np.dot(normal_vectors,normal_target)

            barycenter = np.mean(topomesh.wisp_property('barycenter',degree=0).values(interface_vertices),axis=0)
            point_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_vertices[:,0]) - barycenter
            point_projectors = -np.dot(point_vectors,normal_target)[:,np.newaxis]*normal_target
            non_planar_points = np.where(abs(np.dot(point_vectors,normal_target)) > 1.)[0]
            point_projectors[non_planar_points] = point_projectors[non_planar_points]/abs(np.dot(point_vectors,normal_target))[non_planar_points,np.newaxis]
            
            # planarization_unitary_force = normal_errors[:,np.newaxis]*point_projectors
            planarization_unitary_force = point_projectors

            if(np.isnan(planarization_unitary_force).any()):
                print "Interface ",c,"->",n," planarization force  [",end_time-start_time,"]"

            interface_planarization_force = np.transpose([nd.sum(planarization_unitary_force[:,0],triangle_vertices[:,0],index=interface_vertices),
                                                nd.sum(planarization_unitary_force[:,1],triangle_vertices[:,0],index=interface_vertices),
                                                nd.sum(planarization_unitary_force[:,2],triangle_vertices[:,0],index=interface_vertices)])


            planarization_force[interface_vertices] += interface_planarization_force
    planarization_force[np.isnan(planarization_force)] = 0.

    return planarization_force

def topomesh_remove_vertex(topomesh,pid,kept_fid=None,triangulate=True):

    try:
        vertex_fids = list(topomesh.regions(0,pid,2))
        vertex_fid_cells = np.array([list(topomesh.regions(2,fid)) for fid in vertex_fids])

        if len(vertex_fids) > 0:
            assert array_unique(vertex_fid_cells).shape[0] == 1

            if kept_fid is None:
                kept_fid = vertex_fids[0]
            fid_to_keep = kept_fid
            fids_to_delete = list(set(vertex_fids).difference({fid_to_keep}))

            vertex_eids = list(topomesh.regions(0,pid))
            vertex_fid_eids = np.unique([list(topomesh.borders(2,fid)) for fid in vertex_fids])
            eids_to_keep = list(set(vertex_fid_eids).difference(set(vertex_eids)))

            for eid in vertex_eids:
                topomesh.remove_wisp(1,eid)

            for fid in fids_to_delete:
                topomesh.remove_wisp(2,fid)

            for eid in eids_to_keep:
                if not eid in topomesh.borders(2,fid_to_keep):
                    topomesh.link(2,fid_to_keep,eid)
        
        topomesh.remove_wisp(0,pid)
        del topomesh.wisp_property("barycenter",0)[pid]

        return True

    except AssertionError:
        print "Impossible to remove vertex : wrong configuration ( ",array_unique(vertex_fid_cells).shape[0]," cell memberships)"
        return False


def topomesh_collapse_edge(topomesh,eid,kept_pid=None,manifold=True):
    try:
        if manifold:
            assert len(list(topomesh.regions(1,eid))) == 2

        pid_to_keep, pid_to_delete = topomesh.borders(1,eid)

        print "--> Collapsing edge",eid," : ",pid_to_keep," ; ",pid_to_delete

        if kept_pid is not None and pid_to_keep != kept_pid:
            pid_to_delete, pid_to_keep = topomesh.borders(1,eid)

        edge_fids = list(topomesh.regions(1,eid))

        eids_to_keep = list(topomesh.regions(0,pid_to_keep))
        eids_to_delete = list(topomesh.regions(0,pid_to_delete))

        print "  --> Edge fids : ",edge_fids

        assert np.all([len(list(set(topomesh.borders(2,fid)).intersection(set(eids_to_keep)))) == 2 for fid in edge_fids])
        assert np.all([len(list(set(topomesh.borders(2,fid)).intersection(set(eids_to_delete)))) == 2 for fid in edge_fids])

        if manifold:
            assert np.all(np.concatenate([[len(list(topomesh.regions(1,e))) == 2 for e in topomesh.borders(2,fid)] for fid in edge_fids]))

        pids_to_link = np.unique([list(set(topomesh.borders(2,fid,2)).difference({pid_to_keep,pid_to_delete}))[0] for fid in edge_fids])

        print "  --> Face pids : ",pids_to_link

        assert len(pids_to_link) == len(edge_fids)
        # assert np.all(np.array(map(len,[list(topomesh.regions(0,pid)) for pid in pids_to_link])) > 3)
        assert np.all([np.all([len(list(topomesh.regions(1,e))) == 2 for e in topomesh.borders(2,fid) if e != eid]) for fid in edge_fids])

        for fid in edge_fids:
            print "    --> Face ",fid," : ",list(topomesh.borders(2,fid))," (",eids_to_keep,")"

            eid_to_keep = list(set(topomesh.borders(2,fid)).intersection(set(eids_to_keep)).difference({eid}))[0]
            eid_to_delete = list(set(topomesh.borders(2,fid)).intersection(set(eids_to_delete)).difference({eid}))[0]

            print "    --> Kept eid : ",eid_to_keep,list(topomesh.regions(1,eid_to_keep))
            print "    --> Deleted eid : ",eid_to_delete,list(topomesh.regions(1,eid_to_delete))

            topomesh.unlink(2,fid,eid)
            topomesh.unlink(2,fid,eid_to_keep)
            topomesh.unlink(2,fid,eid_to_delete)

            fid_to_link = list(topomesh.regions(1,eid_to_delete))[0]
            #for fid_to_link in topomesh.regions(1,eid_to_delete):
            topomesh.unlink(2,fid_to_link,eid_to_delete)
            topomesh.link(2,fid_to_link,eid_to_keep)
            
            topomesh.remove_wisp(1,eid_to_delete)
            topomesh.remove_wisp(2,fid)

        for eid_to_link in set(topomesh.regions(0,pid_to_delete)).difference({eid}):
            topomesh.unlink(1,eid_to_link,pid_to_delete)
            neighbor_pid = list(topomesh.borders(1,eid_to_link))[0]
            if neighbor_pid not in topomesh.region_neighbors(0,pid_to_keep):
                # topomesh.unlink(1,eid_to_link,pid_to_delete)
                topomesh.link(1,eid_to_link,pid_to_keep)
            else:
                for fid_to_link in topomesh.regions(1,eid_to_link):
                    if pid_to_keep in topomesh.borders(2,fid_to_link,2):
                        topomesh.remove_wisp(2,fid_to_link)
                    else:
                        eid_to_fuse = tuple(set(topomesh.regions(0,pid_to_keep)).intersection(set(topomesh.regions(0,neighbor_pid))))[0]
                        topomesh.unlink(2,fid_to_link,eid_to_link)
                        topomesh.link(2,fid_to_link,eid_to_fuse)
                topomesh.remove_wisp(1,eid_to_link)

        if kept_pid is not None:
            topomesh.wisp_property("barycenter",0)[pid_to_keep] = topomesh.wisp_property("barycenter",0)[kept_pid]
        else:
            topomesh.wisp_property("barycenter",0)[pid_to_keep] = (topomesh.wisp_property("barycenter",0).values([pid_to_keep,pid_to_delete]).sum(axis=0))/2.

        topomesh.remove_wisp(0,pid_to_delete)
        del topomesh.wisp_property("barycenter",0)[pid_to_delete]

        topomesh.remove_wisp(1,eid)

        print "<-- Collapsed edge",eid," : ",pid_to_keep

        return True

        # edge_vertices = np.sort(np.array([list(topomesh.borders(1,e)) for e in topomesh.wisps(1) if topomesh.nb_borders(1,e) == 2]))
        # edge_vertex_id = edge_vertices[:,0]*10000 + edge_vertices[:,1]
        # if edge_vertices.shape[0] != array_unique(edge_vertices).shape[0]:
        #     print eid," collapse error : (",pid_to_keep,pid_to_delete,")",np.array(list(topomesh.wisps(1)))[nd.sum(np.ones_like(edge_vertex_id),edge_vertex_id,index=edge_vertex_id)>1]
        #     raw_input()

    except AssertionError:
        print "Impossible to collapse edge : wrong configuration ( ",len(list(topomesh.regions(1,eid)))," regions)"
        return False


# def topomesh_collapse_edge(topomesh,eid,kept_pid=None):
#     try:
#         assert len(list(topomesh.regions(1,eid))) == 2

#         pid_to_keep, pid_to_delete = topomesh.borders(1,eid)

#         print "Collapsing edge eid : ",pid_to_keep, pid_to_delete

#         if kept_pid is not None and pid_to_keep != kept_pid:
#             pid_to_delete, pid_to_keep = topomesh.borders(1,eid)

#         edge_fids = list(topomesh.regions(1,eid))

#         for fid in edge_fids:
#             topomesh.remove_wisp(2,fid)

#         for pid in [pid_to_keep, pid_to_delete]:
#             topomesh.unlink(1,eid,pid)

#         neighbor_edges = list(topomesh.regions(0,pid_to_delete))
#         for neighbor_eid in neighbor_edges:
#             topomesh.unlink(1,neighbor_eid,pid_to_delete)
#             neighbor_pid = list(topomesh.borders(1,neighbor_eid))[0]
#             print neighbor_eid," : ", neighbor_pid, list(topomesh.region_neighbors(0,pid_to_keep))
#             if neighbor_pid not in topomesh.region_neighbors(0,pid_to_keep):
#                 topomesh.link(1,neighbor_eid,pid_to_keep)
#                 print neighbor_eid, list(topomesh.borders(1,neighbor_eid)),"[",list(topomesh.regions(1,neighbor_eid)),"]"
#             else:
#                 eid_to_fuse = tuple(set(topomesh.regions(0,pid_to_keep)).intersection(set(topomesh.regions(0,neighbor_pid))))[0]

#                 print neighbor_eid," -> ",tuple(set(topomesh.regions(0,pid_to_keep)).intersection(set(topomesh.regions(0,neighbor_pid))))

#                 topomesh.unlink(1,neighbor_eid,neighbor_pid)
#                 for fid in topomesh.regions(1,neighbor_eid):
#                     topomesh.unlink(2,fid,neighbor_eid)
#                     topomesh.link(2,fid,eid_to_fuse)
#                 topomesh.remove_wisp(1,neighbor_eid)
#                 print eid_to_fuse, list(topomesh.borders(1,eid_to_fuse)),"[",list(topomesh.regions(1,eid_to_fuse)),"]"

#         topomesh.wisp_property("barycenter",0)[pid_to_keep] = (topomesh.wisp_property("barycenter",0).values([pid_to_keep,pid_to_delete]).sum(axis=0))/2.

#         topomesh.remove_wisp(0,pid_to_delete)
#         del topomesh.wisp_property("barycenter",0)[pid_to_delete]

#         topomesh.remove_wisp(1,eid)

#         edge_borders = np.array(map(len,map(np.unique,[list(topomesh.borders(1,e)) for e in topomesh.wisps(1)])))
#         if np.min(edge_borders) == 1:
#             print eid," collapse error - borders : (",pid_to_keep,pid_to_delete,")",np.array(list(topomesh.wisps(1)))[edge_borders==1],neighbor_edges
#             raw_input()

#         edge_regions = np.array(map(len,map(np.unique,[list(topomesh.regions(1,e)) for e in topomesh.wisps(1)])))
#         if np.max(edge_regions) > 2:
#             print eid," collapse error - regions : (",pid_to_keep,pid_to_delete,")",np.array(list(topomesh.wisps(1)))[edge_regions>2],neighbor_edges
#             raw_input()

#     except AssertionError:
#         print "Impossible to collapse edge : wrong configuration ( ",len(list(topomesh.regions(1,eid)))," regions)"
#         return False


def topomesh_flip_edge(topomesh,eid):
    try:
        assert len(list(topomesh.regions(1,eid))) == 2
        edge_triangles = list(topomesh.regions(1,eid))
        assert set(list(topomesh.regions(2,edge_triangles[0]))) == set(list(topomesh.regions(2,edge_triangles[1])))
        edge_vertices = np.array(list(topomesh.borders(1,eid)))
        edge_triangle_edges = list(topomesh.region_neighbors(1,eid))
        edge_triangle_edge_vertices = [list(topomesh.borders(1,e)) for e in edge_triangle_edges]

        new_triangle_edges = {}
        for i_pid,pid in enumerate(edge_vertices):
            new_triangle_edges[i_pid] = []
            for e in edge_triangle_edges:
                if pid in topomesh.borders(1,e):
                    new_triangle_edges[i_pid].append(e)

        edge_triangle_vertices = np.unique(edge_triangle_edge_vertices)
        flipped_edge_vertices = [pid for pid in edge_triangle_vertices if not pid in topomesh.borders(1,eid)]
        assert len(flipped_edge_vertices) == 2
        for pid in topomesh.borders(1,eid):
            topomesh.unlink(1,eid,pid)
        for pid in flipped_edge_vertices:
            topomesh.link(1,eid,pid)

        for i_pid,fid in enumerate(topomesh.regions(1,eid)):
            for e in topomesh.borders(2,fid):
                if e != eid:
                    topomesh.unlink(2,fid,e)
            for e in new_triangle_edges[i_pid]:
                topomesh.link(2,fid,e)

        return True

    except AssertionError:
        print "Impossible to flip edge : wrong configuration ( ",len(list(topomesh.regions(1,eid)))," faces)"
        return False

def topomesh_split_edge(topomesh,eid):
    pid_to_keep, pid_to_unlink = topomesh.borders(1,eid)

    edge_fids = list(topomesh.regions(1,eid))
    eids_to_unlink = np.array([list(set(list(topomesh.borders(2,fid))).intersection(set(list(topomesh.regions(0,pid_to_unlink)))).difference({eid}))[0] for fid in edge_fids])
    pids_split = np.array([list(set(list(topomesh.borders(2,fid,2))).difference({pid_to_keep, pid_to_unlink}))[0] for fid in edge_fids])

    pid_to_add = topomesh.add_wisp(0)
    topomesh.unlink(1,eid,pid_to_unlink)
    topomesh.link(1,eid,pid_to_add)
    # print eid," : ",pid_to_keep,pid_to_add

    topomesh.wisp_property("barycenter",0)[pid_to_add] = (topomesh.wisp_property("barycenter",0).values([pid_to_keep,pid_to_unlink]).sum(axis=0))/2.

    eid_to_add = topomesh.add_wisp(1)
    topomesh.link(1,eid_to_add,pid_to_add)
    topomesh.link(1,eid_to_add,pid_to_unlink)
    # print "Split ",eid_to_add," : ",pid_to_add,pid_to_unlink

    for fid, eid_to_unlink, pid_split in zip(edge_fids,eids_to_unlink,pids_split):
        
        eid_split = topomesh.add_wisp(1)
        topomesh.link(1,eid_split,pid_to_add)
        topomesh.link(1,eid_split,pid_split)
        # print "Added ",eid_split," : ",pid_to_add,pid_split

        topomesh.unlink(2,fid,eid_to_unlink)
        topomesh.link(2,fid,eid_split)

        fid_to_add = topomesh.add_wisp(2)
        topomesh.link(2,fid_to_add,eid_to_add)
        topomesh.link(2,fid_to_add,eid_to_unlink)
        topomesh.link(2,fid_to_add,eid_split)

        for cid in topomesh.regions(2,fid):
            topomesh.link(3,cid,fid_to_add)
    
    # edge_borders = np.array(map(len,map(np.unique,[list(topomesh.borders(1,e)) for e in topomesh.wisps(1)])))
    # if np.min(edge_borders) == 1:
    # edge_vertices = np.sort(np.array([list(topomesh.borders(1,e)) for e in topomesh.wisps(1) if  topomesh.nb_borders(1,e) == 2]))
    # edge_vertex_id = edge_vertices[:,0]*10000 + edge_vertices[:,1]
    # if edge_vertices.shape[0] != array_unique(edge_vertices).shape[0]:
    #     print eid," split error : (",pid_to_keep,pid_to_unlink,")",np.array(list(topomesh.wisps(1)))[nd.sum(np.ones_like(edge_vertex_id),edge_vertex_id,index=edge_vertex_id)>1]
    #     raw_input()

    return True

def topomesh_remove_interface_vertex(topomesh, pid):
    try:
        vertex_fids = list(topomesh.regions(0,pid,2))

        if len(vertex_fids) > 0:
            fid_to_keep = np.min(vertex_fids)
            vertex_fid_cells = [list(topomesh.regions(2,fid)) for fid in vertex_fids]
            vertex_cells = np.unique(vertex_fid_cells)

            assert np.all([set(list(cids)) == set(list(vertex_cells)) for cids in vertex_fid_cells])

            face_eids = np.concatenate([[eid for eid in topomesh.borders(2,fid) if not pid in topomesh.borders(1,eid)] for fid in vertex_fids])
            oriented_face_eids = [face_eids[0]]
            oriented_face_eid_orientations = [1]
            candidate_eids = face_eids

            while len(oriented_face_eids) < len(face_eids) and (len(candidate_eids) > 0):
                current_eid = oriented_face_eids[-1]
                current_eid_orientation = oriented_face_eid_orientations[-1]
                if current_eid_orientation == 1:
                    start_pid, end_pid = topomesh.borders(1,current_eid)
                else:
                    end_pid, start_pid = topomesh.borders(1,current_eid)
                candidate_eids = list(set(list(topomesh.regions(0,end_pid))).intersection(set(list(face_eids))).difference({current_eid}))
                if len(candidate_eids)>0:
                    next_eid = candidate_eids[0]
                    oriented_face_eids += [next_eid]
                    if end_pid == list(topomesh.borders(1,next_eid))[0]:
                        oriented_face_eid_orientations += [1]
                    else:
                        oriented_face_eid_orientations += [-1]
            assert len(oriented_face_eids) == len(face_eids)

            eids_to_remove = []
            for fid in np.sort(vertex_fids):
                for eid in topomesh.borders(2,fid):
                    topomesh.unlink(2,fid,eid)
                    if pid in topomesh.borders(1,eid):                
                        eids_to_remove += [eid]
                    else:
                        topomesh.link(2,fid_to_keep,eid)
        else:
            eids_to_remove = list(topomesh.regions(0,pid))

            # print np.unique(eids_to_remove)
        for eid in np.unique(eids_to_remove):
            for pid_to_unlink in topomesh.borders(1,eid):
                topomesh.unlink(1,eid,pid_to_unlink)
            topomesh.remove_wisp(1,eid)
        for fid in vertex_fids:
            if fid != fid_to_keep:
                topomesh.remove_wisp(2,fid)
        assert topomesh.nb_regions(0,pid) == 0

        topomesh.remove_wisp(0,pid)

        return True
    except AssertionError:
        print "Impossible to remove vertex : wrong face definition"
        return False

def topomesh_remove_interface_edge(topomesh,eid):
    edge_fids = list(topomesh.regions(1,eid))
    fid_to_keep = np.min(edge_fids)

    edge_fid_cells = [list(topomesh.regions(2,fid)) for fid in edge_fids]
    edge_cells = np.unique(edge_fid_cells)

    assert np.all([set(list(cids)) == set(list(edge_cells)) for cids in edge_fid_cells])

    fids_to_remove = np.sort(edge_fids)[1:]

    topomesh.unlink(2,fid_to_keep,eid)
    for fid in fids_to_remove:
        face_eids = list(topomesh.borders(2,fid))
        for eid_to_link in face_eids:
            topomesh.unlink(2,fid,eid_to_link)
            if eid_to_link != eid and not eid_to_link in topomesh.borders(2,fid_to_keep):
                topomesh.link(2,fid_to_keep,eid_to_link)
        topomesh.remove_wisp(2,fid)

    edge_pids = list(topomesh.borders(1,eid))
    for pid in edge_pids:
        topomesh.unlink(1,eid,pid)
        if topomesh.nb_regions(0,pid) == 0:
            topomesh.remove_wisp(0,pid)
    topomesh.remove_wisp(1,eid)

    return True

def topomesh_remove_boundary_vertex(topomesh, pid):
    try:
        vertex_eids = list(topomesh.regions(0,pid))
        eid_to_keep = np.min(vertex_eids)

        vertex_eid_faces = [list(topomesh.regions(1,eid)) for eid in vertex_eids]
        vertex_faces = np.unique(vertex_eid_faces)

        assert len(vertex_eids) == 2
        assert np.all([set(list(fids)) == set(list(vertex_faces)) for fids in vertex_eid_faces])
        assert len(np.unique([list(topomesh.borders(1,eid)) for eid in vertex_eids])) == 3

        eids_to_remove = np.sort(vertex_eids)[1:]

        topomesh.unlink(1,eid_to_keep,pid)
        for eid in eids_to_remove:
            edge_fids = list(topomesh.regions(1,eid))
            for fid in edge_fids:
                topomesh.unlink(2,fid,eid)
            for pid_to_link in topomesh.borders(1,eid): 
                topomesh.unlink(1,eid,pid_to_link)
                if pid_to_link != pid:
                    topomesh.link(1,eid_to_keep,pid_to_link)
            topomesh.remove_wisp(1,eid)
        topomesh.remove_wisp(0,pid)

        # if np.array([list(topomesh.borders(1,eid)) for eid in topomesh.wisps(1)]).ndim != 2:
        #     print eid_to_keep, eids_to_remove
        #     print np.array(list(topomesh.wisps(1)))[np.array(map(len,[list(topomesh.borders(1,eid)) for eid in topomesh.wisps(1)]))!=2]
        #     raw_input()

        return True
    except AssertionError:
        print "Impossible to remove vertex : wrong edge definition"
        return False

def property_topomesh_edge_flip_optimization(topomesh,omega_energies=dict([('regularization',0.15),('neighborhood',0.65)]),simulated_annealing=True,display=False,**kwargs):
    from openalea.mesh.utils.geometry_tools import triangle_geometric_features

    projected = kwargs.get("projected_map_display",False)


    if simulated_annealing:
        iterations = kwargs.get('iterations',20)
        lambda_temperature = kwargs.get('lambda_temperature',0.85)
        minimal_temperature = kwargs.get('minimal_temperature',0.1)
        initial_temperature = minimal_temperature*np.power(1./lambda_temperature,iterations)
    else:
        iterations = kwargs.get('iterations',3)
        initial_temperature = 1.
        lambda_temperature = 0.9
        minimal_temperature = np.power(lambda_temperature,iterations)

    simulated_annealing_temperature = initial_temperature

    while simulated_annealing_temperature > minimal_temperature:

        simulated_annealing_temperature *= lambda_temperature

        compute_topomesh_property(topomesh,'triangles',0)

        compute_topomesh_property(topomesh,'borders',1)
        compute_topomesh_property(topomesh,'vertices',1)
        compute_topomesh_property(topomesh,'triangles',1)
        compute_topomesh_property(topomesh,'length',1)

        compute_topomesh_property(topomesh,'borders',2)
        compute_topomesh_property(topomesh,'vertices',2)

        compute_topomesh_triangle_properties(topomesh)

        # print "Area : ",topomesh.wisp_property('area',2).values().mean()," (",topomesh.wisp_property('area',2).values().std()," ) Eccentricity : ,",topomesh.wisp_property('eccentricity',2).values().mean(),"    [ T = ",simulated_annealing_temperature," ]"

        flippable_edges = topomesh.wisp_property('triangles',1).keys()[np.where(np.array(map(len,topomesh.wisp_property('triangles',1).values())) == 2)]
        
        flippable_edge_vertices = topomesh.wisp_property('vertices',1).values(flippable_edges)
        flippable_edge_triangle_vertices = topomesh.wisp_property('vertices',2).values(topomesh.wisp_property('triangles',1).values(flippable_edges))
        flippable_edge_flipped_vertices = np.array([list(set(list(np.unique(t))).difference(v)) for t,v in zip(flippable_edge_triangle_vertices,flippable_edge_vertices)])
        wrong_edges = np.where(np.array(map(len,flippable_edge_flipped_vertices)) != 2)[0]

        flippable_edges = np.delete(flippable_edges,wrong_edges,0)
        
        flippable_edge_vertices = np.delete(flippable_edge_vertices,wrong_edges,0)
        flippable_edge_triangle_vertices = np.delete(flippable_edge_triangle_vertices,wrong_edges,0)
        flippable_edge_flipped_vertices = np.array([e for e in np.delete(flippable_edge_flipped_vertices,wrong_edges,0)])
        
        flippable_edge_flipped_triangle_vertices = np.array([[np.concatenate([f,[v]]) for v in e] for (e,f) in zip(flippable_edge_vertices,flippable_edge_flipped_vertices)])

        flippable_edge_triangle_areas = np.concatenate([triangle_geometric_features(flippable_edge_triangle_vertices[:,e],topomesh.wisp_property('barycenter',0),features=['area']) for e in [0,1]],axis=1)
        flippable_edge_flipped_triangle_areas = np.concatenate([triangle_geometric_features(flippable_edge_flipped_triangle_vertices[:,e],topomesh.wisp_property('barycenter',0),features=['area']) for e in [0,1]],axis=1)
        flippable_edge_flipped_triangle_areas[np.isnan(flippable_edge_flipped_triangle_areas)] = 100.
        average_area = np.nanmean(topomesh.wisp_property('area',2).values())

        wrong_edges = np.where(np.abs(flippable_edge_triangle_areas.sum(axis=1)-flippable_edge_flipped_triangle_areas.sum(axis=1)) > average_area/10.)
        
        flippable_edges = np.delete(flippable_edges,wrong_edges,0)
        flippable_edge_vertices = np.delete(flippable_edge_vertices,wrong_edges,0)
        flippable_edge_triangle_vertices = np.delete(flippable_edge_triangle_vertices,wrong_edges,0)
        flippable_edge_flipped_vertices = np.delete(flippable_edge_flipped_vertices,wrong_edges,0)
        flippable_edge_flipped_triangle_vertices = np.delete(flippable_edge_flipped_triangle_vertices,wrong_edges,0)
        flippable_edge_triangle_areas = np.delete(flippable_edge_triangle_areas,wrong_edges,0)
        flippable_edge_flipped_triangle_areas =  np.delete(flippable_edge_flipped_triangle_areas,wrong_edges,0)
        
        flippable_edge_energy_variation = np.zeros_like(flippable_edges,np.float)

        if omega_energies.has_key('regularization'):
            flippable_edge_area_error = np.power(flippable_edge_triangle_areas-average_area,2.0).sum(axis=1)
            flippable_edge_area_flipped_error = np.power(np.maximum(flippable_edge_flipped_triangle_areas-average_area,0),2.0).sum(axis=1)
            flippable_edge_area_energy_variation = array_dict(0.02*(flippable_edge_area_flipped_error-flippable_edge_area_error)/np.power(4.0*average_area,2.0),flippable_edges)
            # flippable_edge_area_energy_variation = array_dict(flippable_edge_flipped_triangle_areas.sum(axis=1)-flippable_edge_triangle_areas.sum(axis=1),flippable_edges)

            flippable_edge_triangle_eccentricities = np.concatenate([triangle_geometric_features(flippable_edge_triangle_vertices[:,e],topomesh.wisp_property('barycenter',0),features=['sinus_eccentricity']) for e in [0,1]],axis=1)
            flippable_edge_flipped_triangle_eccentricities = np.concatenate([triangle_geometric_features(flippable_edge_flipped_triangle_vertices[:,e],topomesh.wisp_property('barycenter',0),features=['sinus_eccentricity']) for e in [0,1]],axis=1)
            flippable_edge_eccentricity_energy_variation = array_dict(8.0*(flippable_edge_flipped_triangle_eccentricities.sum(axis=1)-flippable_edge_triangle_eccentricities.sum(axis=1)),flippable_edges)

            flippable_edge_energy_variation += omega_energies['regularization']*flippable_edge_area_energy_variation.values(flippable_edges)
            flippable_edge_energy_variation += omega_energies['regularization']*flippable_edge_eccentricity_energy_variation.values(flippable_edges)

        if omega_energies.has_key('length'):
            flippable_edge_lengths = np.linalg.norm(topomesh.wisp_property('barycenter',0).values(flippable_edge_vertices[:,1]) - topomesh.wisp_property('barycenter',0).values(flippable_edge_vertices[:,0]),axis=1)
            flippable_edge_flipped_lengths = np.linalg.norm(topomesh.wisp_property('barycenter',0).values(flippable_edge_flipped_vertices[:,1]) - topomesh.wisp_property('barycenter',0).values(flippable_edge_flipped_vertices[:,0]),axis=1)
            flippable_edge_length_energy_variation = array_dict(flippable_edge_flipped_lengths-flippable_edge_lengths,flippable_edges)
            flippable_edge_energy_variation += omega_energies['length']*flippable_edge_length_energy_variation.values(flippable_edges)

        if omega_energies.has_key('neighborhood'):


            compute_topomesh_property(topomesh,'valence',0)

            nested_mesh = kwargs.get("nested_mesh",False)
            if nested_mesh:
                compute_topomesh_property(topomesh,'cells',0)
                compute_topomesh_property(topomesh,'epidermis',0)
                target_neighborhood = array_dict((np.array(map(len,topomesh.wisp_property('cells',0).values())) + topomesh.wisp_property('epidermis',0).values())*3,list(topomesh.wisps(0)))
            else:
                target_neighborhood = array_dict(np.ones_like(list(topomesh.wisps(0)))*6,list(topomesh.wisps(0)))
            
            flippable_edge_neighborhood_error = np.power(np.abs(topomesh.wisp_property('valence',0).values(flippable_edge_vertices)-target_neighborhood.values(flippable_edge_vertices)),2.0).sum(axis=1)
            flippable_edge_neighborhood_error += np.power(np.abs(topomesh.wisp_property('valence',0).values(flippable_edge_flipped_vertices)-target_neighborhood.values(flippable_edge_flipped_vertices)),2.0).sum(axis=1)
            flippable_edge_neighborhood_flipped_error = np.power(np.abs(topomesh.wisp_property('valence',0).values(flippable_edge_vertices)-1-target_neighborhood.values(flippable_edge_vertices)),2.0).sum(axis=1)
            flippable_edge_neighborhood_flipped_error += np.power(np.abs(topomesh.wisp_property('valence',0).values(flippable_edge_flipped_vertices)+1-target_neighborhood.values(flippable_edge_flipped_vertices)),2.0).sum(axis=1)
            flippable_edge_neighborhood_energy_variation = array_dict(flippable_edge_neighborhood_flipped_error-flippable_edge_neighborhood_error,flippable_edges)
            flippable_edge_energy_variation += omega_energies['neighborhood']*flippable_edge_neighborhood_energy_variation.values(flippable_edges)

        flippable_edge_energy_variation = array_dict(flippable_edge_energy_variation,flippable_edges)
        
        if display:
            pass

        flippable_edge_sorted_energy_variation = array_dict(np.sort(flippable_edge_energy_variation.values()),flippable_edges[np.argsort(flippable_edge_energy_variation.values())])

        start_time = time()
        print "--> Flipping mesh edges"
        flipped_edges = 0
        flipped_nonoptimal_edges = 0
        unflippable_edges = set()
        for eid in flippable_edge_sorted_energy_variation.keys():
            flip_probability = np.exp(-flippable_edge_sorted_energy_variation[eid]/simulated_annealing_temperature)
            if (simulated_annealing and (np.random.rand() < flip_probability)) or (1 < flip_probability):
                if not eid in unflippable_edges:
                    # print topomesh.wisp_property('vertices',1)[eid], flippable_edge_sorted_energy_variation[e]
                    # neighbor_edges = list(topomesh.region_neighbors(1,eid))
                    flipped_edges += 1
                    flipped_nonoptimal_edges += (flip_probability<1)
                    neighbor_edges = list(topomesh.border_neighbors(1,eid))
                    topomesh_flip_edge(topomesh,eid)
                    # for n_eid in neighbor_edges:
                    #     unflippable_edges.append(n_eid)
                    unflippable_edges = unflippable_edges.union(set(neighbor_edges))

        end_time = time()
        print "  --> Flipped ",flipped_edges," edges (",flipped_nonoptimal_edges," non-optimal)    [ T = ",simulated_annealing_temperature,"]"
        print "<-- Flipping mesh edges    [",end_time-start_time,"s]"

    return flipped_edges   


def property_topomesh_edge_split_optimization(topomesh, maximal_length=None, iterations=1):

    compute_topomesh_property(topomesh,'vertices',1)
    compute_topomesh_property(topomesh,'length',1)

    if maximal_length is None:
        target_length = np.percentile(topomesh.wisp_property('length',1).values(),70)
        maximal_length = 4./3. * target_length

    for iteration in xrange(iterations):
        sorted_edge_length_edges = np.array(list(topomesh.wisps(1)))[np.argsort(-topomesh.wisp_property('length',1).values(list(topomesh.wisps(1))))]
        sorted_edge_length_edges = sorted_edge_length_edges[topomesh.wisp_property('length',1).values(sorted_edge_length_edges) > maximal_length]
        modified_edges = set()
        n_splits = 0
        for eid in sorted_edge_length_edges:
            if not eid in modified_edges:
                #print "  <-- Splitting edge ",eid," [",np.min(map(len,map(np.unique,[list(topomesh.borders(1,e)) for e in topomesh.wisps(1)]))),"]"
                modified_edges = modified_edges.union(set(np.unique([np.array(list(topomesh.borders(2,fid))) for fid in topomesh.regions(1,eid)])))
                topomesh_split_edge(topomesh,eid)
                n_splits += 1
                #print "  <-- Splitted edge ",eid," [",np.min(map(len,map(np.unique,[list(topomesh.borders(1,e)) for e in topomesh.wisps(1)]))),"]"
        print "--> Splitted ",n_splits," edges"
        
        compute_topomesh_property(topomesh,'vertices',1)
        compute_topomesh_property(topomesh,'length',1)

    return n_splits


def optimize_topomesh(input_topomesh,omega_forces,iterations=20,edge_flip=False,display=False,**kwargs):
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

    if display:
        pass


    optimization_start_time = time()

    if omega_forces.has_key('regularization'):
        omega_regularization_max = kwargs.get('omega_regularization_max',omega_forces['regularization'])

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





