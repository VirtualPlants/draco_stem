# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
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

from scipy.cluster.vq                       import kmeans, vq
from array                                  import array

from openalea.container                     import PropertyTopomesh, array_dict

from copy                                   import copy, deepcopy
from time                                   import time
import os
import sys
import pickle

def compute_topomesh_property(topomesh,property_name,degree=0,positions=None,normal_method="density",object_positions=None,object_radius=10.,verbose=False):
    """Compute a property of a PropertyTopomesh

    The function compute and fills a property of a PropertyTopomesh passed as argument. The given
    property is computed for all elements of the specified degree and stored as a dictionary in
    the PropertyTopomesh structure

    :Parameters:
        - 'topomesh' (PropertyTopomesh) - the structure on which to compute the property
        - 'property_name' (string) - the name of the property to compute, among the following ones:
            * 'barycenter' (degree : [0, 1, 2, 3]) : the position of the center of the element
            * 'vertices' (degree : [0, 1, 2, 3])
            * 'triangles' (degree : [0, 1, 2, 3])
            * 'cells' (degree : [0, 1, 2, 3])
            * 'length' (degree : [1])
            * 'area' (degree : [2])
            * 'volume' (degree : [3])
            * 'normal' (degree : [0, 2])
        - 'degree' (int) - the degree of the elements on which to compute the property
        - 'positions' (dict) - [optional] a position dictionary if ('barycenter',0) is empty
        - 'object_positions' (dict) - [optional] position of the object(s) reprensented by the mesh
            * used only for property ('normal',2)
        - 'object_radius' (float) - [optional] radius of the object(s) represented by the mesh
            * used only for property ('normal',2)

        :Returns:
            None - PropertyTopomesh passed as argument is updated
    """

    if positions is None:
        positions = topomesh.wisp_property('barycenter',degree=0)

    start_time = time()
    if verbose:
        print "--> Computing ",property_name," property (",degree,")"

    if property_name == 'barycenter':
        if not isinstance(positions,array_dict):
            positions = array_dict(positions)
        if not 'barycenter' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('barycenter',degree=degree)
        if degree == 0:
            topomesh.update_wisp_property('barycenter',degree=degree,values=positions,keys=np.array(list(topomesh.wisps(degree))))
        else:
            if not topomesh.has_wisp_property('vertices',degree=degree,is_computed=True):
                compute_topomesh_property(topomesh,'vertices',degree=degree)
            element_vertices = topomesh.wisp_property('vertices',degree).values(np.array(list(topomesh.wisps(degree))))
            element_vertices_number = np.array(map(len,element_vertices))
            barycentrable_elements = np.array(list(topomesh.wisps(degree)))[np.where(element_vertices_number>0)]
            element_vertices = topomesh.wisp_property('vertices',degree).values(barycentrable_elements)
            vertices_positions = positions.values(element_vertices)
            if vertices_positions.dtype == np.dtype('O'):
                barycenter_positions = np.array([np.mean(p,axis=0) for p in vertices_positions])
            else:
                barycenter_positions = np.mean(vertices_positions,axis=1)
            topomesh.update_wisp_property('barycenter',degree=degree,values=barycenter_positions,keys=barycentrable_elements)

    if property_name == 'vertices':
        if not 'vertices' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('vertices',degree=degree)
        if degree == 0:
            topomesh.update_wisp_property('vertices',degree=degree,values=np.array(list(topomesh.wisps(degree))),keys=np.array(list(topomesh.wisps(degree))))
        else:
            topomesh.update_wisp_property('vertices',degree=degree,values=np.array([np.unique(list(topomesh.borders(degree,w,degree))) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))
        if degree == 1:
            topomesh.update_wisp_property('borders',degree=degree,values=np.array([np.unique(list(topomesh.borders(degree,w,degree))) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))
    
    if property_name == 'edges':
        if not 'edges' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('edges',degree=degree)
        if degree < 1:
            topomesh.update_wisp_property('edges',degree=degree,values=np.array([list(topomesh.regions(degree,w,1-degree)) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))
        elif degree > 1:
            topomesh.update_wisp_property('edges',degree=degree,values=np.array([list(topomesh.borders(degree,w,degree-1)) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))
        else:
            topomesh.update_wisp_property('edges',degree=degree,values=np.array(list(topomesh.wisps(degree))),keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'triangles':
        if not 'triangles' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('triangles',degree=degree)
        if degree < 2:
            topomesh.update_wisp_property('triangles',degree=degree,values=np.array([list(topomesh.regions(degree,w,2-degree)) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))
        elif degree > 2:
            topomesh.update_wisp_property('triangles',degree=degree,values=np.array([list(topomesh.borders(degree,w,degree-2)) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))
        else:
            topomesh.update_wisp_property('triangles',degree=degree,values=np.array(list(topomesh.wisps(degree))),keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'cells':
        if not 'cells' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('cells',degree=degree)
        if degree == topomesh.degree():
            topomesh.update_wisp_property('cells',degree=degree,values=np.array(list(topomesh.wisps(degree))),keys=np.array(list(topomesh.wisps(degree))))
        else:
            topomesh.update_wisp_property('cells',degree=degree,values=np.array([list(topomesh.regions(degree,w,topomesh.degree()-degree)) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'borders':
        assert degree>0
        if not 'borders' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('borders',degree=degree)
        topomesh.update_wisp_property('borders',degree=degree,values=np.array([list(topomesh.borders(degree,w)) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))
    
    if property_name == 'border_neighbors':
        assert degree>0
        if not 'neighbors' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('neighbors',degree=degree)
        topomesh.update_wisp_property('neighbors',degree=degree,values=np.array([np.unique(list(topomesh.border_neighbors(degree,w))) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'regions':
        assert degree<topomesh.degree()
        if not 'regions' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('regions',degree=degree)
        topomesh.update_wisp_property('regions',degree=degree,values=np.array([list(topomesh.regions(degree,w)) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'region_neighbors':
        assert degree<topomesh.degree()
        if not 'neighbors' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('neighbors',degree=degree)
        topomesh.update_wisp_property('neighbors',degree=degree,values=np.array([np.unique(list(topomesh.region_neighbors(degree,w))) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'valence':
        assert degree<topomesh.degree()
        if not 'valence' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('valence',degree=degree)
        # if not topomesh.has_wisp_property('neighbors',degree=degree,is_computed=True):
        #     compute_topomesh_property(topomesh,'region_neighbors',degree=degree)
        # topomesh.update_wisp_property('valence',degree=degree,values=np.array([len(topomesh.wisp_property('neighbors',degree)[w]) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))
        topomesh.update_wisp_property('valence',degree=degree,values=np.array([len(list(topomesh.region_neighbors(degree,w))) for w in topomesh.wisps(degree)]),keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'oriented_borders':
        if degree == 2:
            oriented_borders = []
            oriented_border_orientations = []
            for t,fid in enumerate(topomesh.wisps(2)):
                face_eids = np.array(list(topomesh.borders(2,fid)))
                #face_edges = np.array([list(topomesh.borders(1,eid)) for eid in face_eids])

                oriented_face_eids = [face_eids[0]]
                oriented_face_eid_orientations = [1]

                while len(oriented_face_eids) < len(face_eids):
                    current_eid = oriented_face_eids[-1]
                    current_eid_orientation = oriented_face_eid_orientations[-1]
                    if current_eid_orientation == 1:
                        start_pid, end_pid = topomesh.borders(1,current_eid)
                    else:
                        end_pid, start_pid = topomesh.borders(1,current_eid)
                    candidate_eids = set(list(topomesh.regions(0,end_pid))).intersection(set(list(face_eids))).difference({current_eid})
                    if len(oriented_face_eids)>0:
                        candidate_eids = candidate_eids.difference(set(oriented_face_eids))
                    if len(candidate_eids)>0:
                        next_eid = list(candidate_eids)[0]
                    else:
                        next_eid = list(set(list(face_eids)).difference(set(oriented_face_eids)))[0]
                    oriented_face_eids += [next_eid]
                    if end_pid == list(topomesh.borders(1,next_eid))[0]:
                        oriented_face_eid_orientations += [1]
                    else:
                        oriented_face_eid_orientations += [-1]
                # print oriented_face_eids," (",oriented_face_eid_orientations,")"
                oriented_borders += [oriented_face_eids]
                oriented_border_orientations += [oriented_face_eid_orientations]
                # raw_input()

            topomesh.update_wisp_property('oriented_borders',degree=2,values=np.array([(b,o) for b,o in zip(oriented_borders,oriented_border_orientations)]),keys=np.array(list(topomesh.wisps(2))))

        elif degree == 3:
            compute_topomesh_property(topomesh,'oriented_borders',2)
            compute_topomesh_property(topomesh,'oriented_vertices',2)
            compute_topomesh_property(topomesh,'barycenter',2)
            compute_topomesh_property(topomesh,'barycenter',3)

            triangle_edge_list  = np.array([[0, 1],[1, 2],[2, 0]])

            oriented_borders = []
            oriented_border_orientations = []
            oriented_border_components = []

            oriented_face_normals = []
            for f,fid in enumerate(topomesh.wisps(2)):
                face_vertices = topomesh.wisp_property('oriented_vertices',2)[fid]
                if len(face_vertices) == 3:
                    vertices_positions = topomesh.wisp_property('barycenter',0).values(face_vertices)
                    face_normal = np.cross(vertices_positions[1]-vertices_positions[0],vertices_positions[2]-vertices_positions[0])
                else:
                    face_edges = np.transpose(np.array([face_vertices,list(face_vertices[1:])+[face_vertices[0]]]))
                    face_barycenter =  topomesh.wisp_property('barycenter',2)[fid]
                    vertices_positions = np.array([np.concatenate([p,[face_barycenter]]) for p in topomesh.wisp_property('barycenter',0).values(face_edges)])
                    face_triangle_normals = np.cross(vertices_positions[:,1]-vertices_positions[:,0],vertices_positions[:,2]-vertices_positions[:,0])
                    face_triangle_normals = face_triangle_normals/np.linalg.norm(face_triangle_normals,axis=1)[:,np.newaxis]
                    face_triangle_lengths = np.linalg.norm(vertices_positions[...,triangle_edge_list,1] - vertices_positions[...,triangle_edge_list,0],axis=2)
                    face_triangle_perimeters = face_triangle_lengths.sum(axis=1)
                    face_triangle_areas = np.sqrt((face_triangle_perimeters/2.0)*(face_triangle_perimeters/2.0-face_triangle_lengths[...,0])*(face_triangle_perimeters/2.0-face_triangle_lengths[...,1])*(face_triangle_perimeters/2.0-face_triangle_lengths[...,2]))
                    face_normal = (face_triangle_normals*face_triangle_areas[:,np.newaxis]).sum(axis=0)/face_triangle_areas.sum()
                face_normal = face_normal/np.linalg.norm(face_normal)
                oriented_face_normals += [face_normal]
            topomesh.update_wisp_property('normal',degree=2,values=np.array(oriented_face_normals),keys=np.array(list(topomesh.wisps(2))))

            cell_face_component = 0
            for c,cid in enumerate(topomesh.wisps(3)):

                cell_fids = np.array(list(topomesh.borders(3,cid)))
                # print cell_fids

                cell_fid_normals = topomesh.wisp_property('normal',2).values(cell_fids)
                cell_fid_barycenters = topomesh.wisp_property('barycenter',2).values(cell_fids)
                cell_barycenter = topomesh.wisp_property('barycenter',3)[cid]

                cell_fid_dot_product = np.einsum('ij,ij->i',cell_fid_normals,cell_fid_barycenters-cell_barycenter[np.newaxis])
                cell_fid_orientation = array_dict(np.array(np.sign(cell_fid_dot_product),int),cell_fids)
                cell_fid_dot_product = array_dict(cell_fid_dot_product,cell_fids)

                start_fid = cell_fids[np.argmax(np.abs(cell_fid_dot_product.values(cell_fids)))]
                #start_fid = cell_fids[0]
                oriented_cell_fid_orientations = {}
                oriented_cell_fid_orientations[start_fid] = cell_fid_orientation[start_fid]
                
                cell_face_component = cell_face_component+1
                fid_components = {}
                fid_components[start_fid] = cell_face_component


                next_fid_queue = []
                current_fid_queue = []
                transition_eid_queue = []

                eid_orientations = array_dict(*topomesh.wisp_property('oriented_borders',2)[start_fid][::-1])
                for eid in topomesh.wisp_property('oriented_borders',2)[start_fid][0]:
                    candidate_fids = set(list(topomesh.regions(1,eid))).intersection(set(list(cell_fids))).difference(set(oriented_cell_fid_orientations.keys()))
                    if len(candidate_fids)>0:
                        for fid in candidate_fids:
                            next_fid_queue.insert(0,fid)
                            current_fid_queue.insert(0,start_fid)
                            transition_eid_queue.insert(0,eid)

                # print oriented_cell_fid_orientations
                # print next_fid_queue
                # print len(oriented_cell_fid_orientations.keys())," / ",len(cell_fids)," [",len(next_fid_queue),"]"
                # raw_input()

                while len(oriented_cell_fid_orientations.keys()) < len(cell_fids):

                    while len(next_fid_queue) > 0:
                        next_fid = next_fid_queue.pop()
                        current_fid = current_fid_queue.pop()
                        transition_eid = transition_eid_queue.pop()

                        current_face_eid_orientations = array_dict(*topomesh.wisp_property('oriented_borders',2)[current_fid][::-1])
                        current_fid_orientation = oriented_cell_fid_orientations[current_fid]
                        next_face_eid_orientations = array_dict(*topomesh.wisp_property('oriented_borders',2)[next_fid][::-1])

                        # print list(topomesh.borders(2,current_fid))
                        # print current_face_eid_orientations
                        # print list(topomesh.borders(2,next_fid))
                        # print next_face_eid_orientations
                        
                        next_fid_orientation = - current_face_eid_orientations[transition_eid]*current_fid_orientation*next_face_eid_orientations[transition_eid]
                        oriented_cell_fid_orientations[next_fid] = next_fid_orientation
                        fid_components[next_fid] = cell_face_component
                        # print list(topomesh.border_neighbors(2,next_fid))

                        for eid in topomesh.wisp_property('oriented_borders',2)[next_fid][0]:
                            candidate_fids = set(list(topomesh.regions(1,eid))).intersection(set(list(cell_fids)))
                            candidate_fids = candidate_fids.difference(set(oriented_cell_fid_orientations.keys()))
                            candidate_fids = candidate_fids.difference(set(next_fid_queue))

                            if len(candidate_fids)>0:
                                for fid in candidate_fids:
                                    next_fid_queue.insert(0,fid)
                                    current_fid_queue.insert(0,next_fid)
                                    transition_eid_queue.insert(0,eid)

                        # print oriented_cell_fid_orientations
                        # next_fid_queue
                        # print "Cell ",cid," : ", len(oriented_cell_fid_orientations.keys())," / ",len(cell_fids)," [",len(next_fid_queue),"]"
                        # raw_input()

                    if len(oriented_cell_fid_orientations.keys()) < len(cell_fids):
                        remaining_cell_fids = np.array(list(set(cell_fids).difference(set(oriented_cell_fid_orientations.keys()))))
                        start_fid = remaining_cell_fids[np.argmax(np.abs(cell_fid_dot_product.values(remaining_cell_fids)))]
                        oriented_cell_fid_orientations[start_fid] = cell_fid_orientation[start_fid]
                        
                        cell_face_component = cell_face_component+1
                        fid_components[start_fid] = cell_face_component

                        eid_orientations = array_dict(*topomesh.wisp_property('oriented_borders',2)[start_fid][::-1])
                        for eid in topomesh.wisp_property('oriented_borders',2)[start_fid][0]:
                            candidate_fids = set(list(topomesh.regions(1,eid))).intersection(set(list(cell_fids))).difference(set(oriented_cell_fid_orientations.keys()))
                            if len(candidate_fids)>0:
                                for fid in candidate_fids:
                                    next_fid_queue.insert(0,fid)
                                    current_fid_queue.insert(0,start_fid)
                                    transition_eid_queue.insert(0,eid)
                        
                        # print oriented_cell_fid_orientations
                        # print next_fid_queue
                        # print len(oriented_cell_fid_orientations.keys())," / ",len(cell_fids)," [",len(next_fid_queue),"]"
                        # raw_input()

                # oriented_cell_fids = [cell_fids[0]]
                # oriented_cell_fid_orientations = [cell_fid_orientation[cell_fids[0]]]

                # start_eid = topomesh.wisp_property('oriented_borders',2)[cell_fids[0]][0][0]
                # face_gaps = 0

                # while len(oriented_cell_fids) < len(cell_fids):
                #     current_fid = oriented_cell_fids[-1]
                #     current_fid_orientation = oriented_cell_fid_orientations[-1]

                #     face_eids = np.array(topomesh.wisp_property('oriented_borders',2)[current_fid][0])
                #     face_eid_orientation = array_dict(np.array(topomesh.wisp_property('oriented_borders',2)[current_fid][1]),face_eids)

                #     candidate_fids = set()
                #     current_eid = start_eid
                #     n_tries = 0

                #     while (len(list(candidate_fids))==0) and n_tries<len(face_eids):
                #         candidate_fids = set(list(topomesh.regions(1,current_eid))).intersection(set(list(cell_fids))).difference(set(oriented_cell_fids))
                #         # print current_eid,"(",face_eids,")  ->  ",candidate_fids
                #         previous_eid = current_eid
                #         current_eid = face_eids[(np.where(face_eids == current_eid)[0][0] + current_fid_orientation) % len(face_eids)]
                #         n_tries += 1
                #     if n_tries == len(face_eids):
                #         index = 0
                #         remaining_triangles = len(list(set(cell_fids).difference(oriented_cell_fids)))
                #         next_fid = list(set(cell_fids).difference(oriented_cell_fids))[index]
                #         edge_index = 0
                #         start_eid = topomesh.wisp_property('oriented_borders',2)[next_fid][0][edge_index]
                #         opposite_fid = list(set(topomesh.regions(1,start_eid)).difference({next_fid}))[0] if topomesh.nb_regions(1,start_eid)>1 else -1
                #         while (not opposite_fid in oriented_cell_fids) and (edge_index < 2):
                #             edge_index = edge_index+1
                #             start_eid = topomesh.wisp_property('oriented_borders',2)[next_fid][0][edge_index]
                #             opposite_fid = list(set(topomesh.regions(1,start_eid)).difference({next_fid}))[0] if topomesh.nb_regions(1,start_eid)>1 else -1
                #         while (not opposite_fid in oriented_cell_fids) and (index < remaining_triangles):
                #             next_fid = list(set(cell_fids).difference(oriented_cell_fids))[index]
                #             index = index+1
                #             edge_index = 0
                #             start_eid = topomesh.wisp_property('oriented_borders',2)[next_fid][0][edge_index]
                #             opposite_fid = list(set(topomesh.regions(1,start_eid)).difference({next_fid}))[0] if topomesh.nb_regions(1,start_eid)>1 else -1
                #             while (not opposite_fid in oriented_cell_fids) and (edge_index < 2):
                #                 edge_index = edge_index+1
                #                 start_eid = topomesh.wisp_property('oriented_borders',2)[next_fid][0][edge_index]
                #                 opposite_fid = list(set(topomesh.regions(1,start_eid)).difference({next_fid}))[0] if topomesh.nb_regions(1,start_eid)>1 else -1
                #         if index < len(list(set(cell_fids).difference(oriented_cell_fids))):
                #             #next_fid_orientation = cell_fid_orientation[next_fid]
                #             #next_fid_orientation = oriented_cell_fid_orientations
                #             opposite_face_eids = np.array(topomesh.wisp_property('oriented_borders',2)[opposite_fid][0])
                #             opposite_face_eid_orientation = array_dict(np.array(topomesh.wisp_property('oriented_borders',2)[opposite_fid][1]),opposite_face_eids)

                #             next_face_eids = np.array(topomesh.wisp_property('oriented_borders',2)[next_fid][0])
                #             next_face_eid_orientation = array_dict(np.array(topomesh.wisp_property('oriented_borders',2)[next_fid][1]),next_face_eids)
                            
                #             opposite_fid_orientation = np.array(oriented_cell_fid_orientations)[np.where(oriented_cell_fids==opposite_fid)]

                #             next_fid_orientation = - opposite_face_eid_orientation[start_eid] * current_fid_orientation * next_face_eid_orientation[start_eid]
                #         else:
                #             index = 0
                #             next_fid = list(set(cell_fids).difference(oriented_cell_fids))[index]
                #             start_eid = topomesh.wisp_property('oriented_borders',2)[next_fid][0][0]
                #             next_fid_orientation = cell_fid_orientation[next_fid]

                #         face_gaps += 1
                #     else:
                #         next_fid = int(list(candidate_fids)[0])                        
                #         start_eid = previous_eid
                #         next_face_eids = np.array(topomesh.wisp_property('oriented_borders',2)[next_fid][0])
                #         next_face_eid_orientation = array_dict(np.array(topomesh.wisp_property('oriented_borders',2)[next_fid][1]),next_face_eids)

                #         next_fid_orientation = - face_eid_orientation[start_eid]*current_fid_orientation * next_face_eid_orientation[start_eid]

                #     oriented_cell_fids += [next_fid]
                #     oriented_cell_fid_orientations += [next_fid_orientation]
                # # print cell_fids," -> ",oriented_cell_fids,"(",face_gaps,"face gaps)"
                # # print cell_fid_orientation.values(oriented_cell_fids) - np.array(oriented_cell_fid_orientations)

                oriented_borders += [oriented_cell_fid_orientations.keys()]
                oriented_border_orientations += [[oriented_cell_fid_orientations[b] for b in oriented_cell_fid_orientations.keys()]]
                oriented_border_components += [[fid_components[b] for b in oriented_cell_fid_orientations.keys()]]


            topomesh.update_wisp_property('oriented_borders',degree=3,values=np.array([(b,o) for b,o in zip(oriented_borders,oriented_border_orientations)]),keys=np.array(list(topomesh.wisps(3))))
            topomesh.update_wisp_property('oriented_border_components',degree=3,values=np.array([(b,c) for b,c in zip(oriented_borders,oriented_border_components)]),keys=np.array(list(topomesh.wisps(3))))


                # oriented_face_pids = [list(topomesh.borders(1,eid))[0 if ori==1 else 1] for eid,ori in zip(oriented_face_eids,oriented_face_eid_orientations)]
                # print oriented_face_pids
                # raw_input()

                # face_pids = np.array(list(topomesh.borders(2,fid,2)))
                # face_edges = np.array([list(topomesh.borders(1,eid)) for eid in topomesh.borders(2,fid)])

                # oriented_face_pids = [face_pids[0]]
                # while len(oriented_face_pids) < len(face_pids):
                #     current_pid = oriented_face_pids[-1]
                #     pid_edges = face_edges[np.where(face_edges==current_pid)[0]]
                #     candidate_pids = set(list(np.unique(pid_edges))).difference({current_pid})
                #     if len(oriented_face_pids)>1:
                #         candidate_pids = candidate_pids.difference({oriented_face_pids[-2]})
                #     oriented_face_pids += [list(candidate_pids)[0]]
                # print oriented_face_pids
                # raw_input()


    if property_name == 'oriented_vertices':
        if degree == 2:
            compute_topomesh_property(topomesh,'oriented_borders',2)
            oriented_vertices = []
            for t,fid in enumerate(topomesh.wisps(2)):
                oriented_face_eids = topomesh.wisp_property('oriented_borders',2)[fid][0]
                oriented_face_eid_orientations = topomesh.wisp_property('oriented_borders',2)[fid][1]
                oriented_face_pids = [list(topomesh.borders(1,eid))[0 if ori==1 else 1] for eid,ori in zip(oriented_face_eids,oriented_face_eid_orientations)]
                oriented_vertices += [oriented_face_pids]
            topomesh.update_wisp_property('oriented_vertices',degree=2,values=np.array(oriented_vertices),keys=np.array(list(topomesh.wisps(2))))

                # face_pids = np.array(list(topomesh.borders(2,fid,2)))
                # face_edges = np.array([list(topomesh.borders(1,eid)) for eid in topomesh.borders(2,fid)])

                # oriented_face_pids = [face_pids[0]]
                # while len(oriented_face_pids) < len(face_pids):
                #     current_pid = oriented_face_pids[-1]
                #     pid_edges = face_edges[np.where(face_edges==current_pid)[0]]
                #     candidate_pids = set(list(np.unique(pid_edges))).difference({current_pid})
                #     if len(oriented_face_pids)>1:
                #         candidate_pids = candidate_pids.difference({oriented_face_pids[-2]})
                #     oriented_face_pids += [list(candidate_pids)[0]]

    # if property_name == 'curvature':
    #     assert degree==0
    #     if not 'curvature' in topomesh.wisp_property_names(degree):
    #         topomesh.add_wisp_property('curvature',degree=degree)
    #     if not topomesh.has_wisp_property('vertices',degree=1,is_computed=True):
    #         compute_topomesh_property(topomesh,'vertices',degree=1)
    #     if not topomesh.has_wisp_property('vertices',degree=3,is_computed=True):
    #         compute_topomesh_property(topomesh,'vertices',degree=3)
    #     if not topomesh.has_wisp_property('edges',degree=3,is_computed=True):
    #         compute_topomesh_property(topomesh,'edges',degree=3)
    #     if not topomesh.has_wisp_property('cells',degree=0,is_computed=True):
    #         compute_topomesh_property(topomesh,'cells',degree=0)
    #     if not topomesh.has_wisp_property('neighbors',degree=0,is_computed=True):
    #         compute_topomesh_property(topomesh,'region_neighbors',degree=0)

    #     vertex_cells = topomesh.wisp_property('cells',0).values()
    #     vertex_cell_neighbors = np.array([[np.intersect1d(topomesh.wisp_property('vertices',3)[c],topomesh.wisp_property('neighbors',0)[v]) for c in topomesh.wisp_property('cells',0)[v]] for v in topomesh.wisps(0)])
    #     vertex_cell_valence = np.array([[len(n) for n in vertex_cell_neighbors[v]] for v in topomesh.wisps(0)])
    #     vertex_cell_edge_vectors = np.array([[topomesh.wisp_property('barycenter',degree=0).values(n) - topomesh.wisp_property('barycenter',degree=0)[v] for n in vertex_cell_neighbors[v]] for v in topomesh.wisps(0)])
    #     vertex_cell_laplacian = np.array([[np.sum(vectors,axis=0)/valence for vectors,valence in zip(vertex_cell_edge_vectors[v],vertex_cell_valence[v])] for v in topomesh.wisps(0)])
    #     vertex_cell_barycenters = np.array([topomesh.wisp_property('barycenter',degree=3).values(vertex_cells[v]) for v in topomesh.wisps(0)])
    #     vertex_cell_directions = np.array([topomesh.wisp_property('barycenter',degree=0)[v] - vertex_cell_barycenters[v] for v in topomesh.wisps(0)])
    #     vertex_cell_directions = np.array([vertex_cell_directions[v]/np.linalg.norm(vertex_cell_directions[v],axis=1)[:,np.newaxis] for v in topomesh.wisps(0)])
    #     vertex_cell_curvature_sign = np.array([-np.sign(np.einsum('ij,ij->i',vertex_cell_laplacian[v],vertex_cell_directions[v])) for v in topomesh.wisps(0)])
    #     vertex_cell_curvatures = np.array([vertex_cell_curvature_sign[v]*np.linalg.norm(vertex_cell_laplacian[v],axis=1) for v in topomesh.wisps(0)])
    #     topomesh.update_wisp_property('curvature',degree=degree,values=vertex_cell_curvatures,keys=np.array(list(topomesh.wisps(degree))))

    # if property_name == 'gaussian_curvature':
    #     assert degree==0
    #     if not 'gaussian_curvature' in topomesh.wisp_property_names(degree):
    #         topomesh.add_wisp_property('curvature',degree=degree)
    #     if not topomesh.has_wisp_property('borders',degree=3,is_computed=True):
    #         compute_topomesh_property(topomesh,'borders',degree=3)
    #     if not topomesh.has_wisp_property('vertices',degree=3,is_computed=True):
    #         compute_topomesh_property(topomesh,'vertices',degree=3)
    #     if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
    #         compute_topomesh_property(topomesh,'vertices',degree=2)
    #     if not topomesh.has_wisp_property('cells',degree=0,is_computed=True):
    #         compute_topomesh_property(topomesh,'cells',degree=0)
    #     if not topomesh.has_wisp_property('length',degree=1,is_computed=True):
    #         compute_topomesh_property(topomesh,'length',degree=1)

    #     vertex_cells = topomesh.wisp_property('cells',0).values()
    #     vertex_cell_curvatures = np.array([{} for v in topomesh.wisps(0)])

    #     cell_triangles = np.array(np.concatenate(topomesh.wisp_property('borders',degree=3).values()),int)
    #     cell_triangle_cells = np.array(np.concatenate([[c for t in topomesh.wisp_property('borders',degree=3)[c]] for c in topomesh.wisps(3)]),int)

    #     triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(cell_triangles)
    #     # rotated_triangle_vertices = np.transpose([triangle_vertices[:,2],triangle_vertices[:,0],triangle_vertices[:,1]])
    #     # antirotated_triangle_vertices = np.transpose([triangle_vertices[:,1],triangle_vertices[:,2],triangle_vertices[:,0]])
    #     # triangle_vertices = np.append(np.append(triangle_vertices,rotated_triangle_vertices,axis=0),antirotated_triangle_vertices,axis=0)

    #     edge_index_list = np.array([[1, 2],[0, 1],[0, 2]])
    #     triangle_edge_vertices = triangle_vertices[:,edge_index_list]

    #     triangle_edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,1]) - topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,0])
    #     #triangle_edge_lengths = np.power(np.sum(np.power(triangle_edge_vectors,2.0),axis=2),0.5)
    #     triangle_edge_lengths = np.linalg.norm(triangle_edge_vectors,axis=2)
    #     triangle_edge_directions = triangle_edge_vectors/triangle_edge_lengths[...,np.newaxis]

    #     triangle_perimeters = np.sum(triangle_edge_lengths,axis=1)
    #     triangle_areas = np.sqrt((triangle_perimeters/2.0)*(triangle_perimeters/2.0-triangle_edge_lengths[:,0])*(triangle_perimeters/2.0-triangle_edge_lengths[:,1])*(triangle_perimeters/2.0-triangle_edge_lengths[:,2]))

    #     triangle_cosines = np.zeros_like(triangle_edge_lengths,np.float32)
    #     triangle_cosines[:,0] = (triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2)/(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2])
    #     triangle_cosines[:,1] = (triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2)/(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0])
    #     triangle_cosines[:,2] = (triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2)/(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1])
    #     triangle_angles = np.arccos(triangle_cosines)

    #     # for v in topomesh.wisps(0):
    #     #     vertex_angle_sum = nd.sum(triangle_angles[np.where(triangle_vertices==v)],cell_triangle_cells[np.where(triangle_vertices==v)[0]],index=topomesh.wisp_property('cells',0)[v])
    #     #     vertex_area = nd.sum(triangle_areas[np.where(triangle_vertices==v)[0]]/3.,cell_triangle_cells[np.where(triangle_vertices==v)[0]],index=topomesh.wisp_property('cells',0)[v])
    #     #     vertex_curvature = (2*np.pi - vertex_angle_sum)/vertex_area
    #     #     vertex_cell_curvatures[v] = vertex_curvature
    #         #topomesh.wisp_property('curvature',degree=0)[v] = vertex_curvature

    #     for c in topomesh.wisps(3):
    #         vertex_angle_sum = nd.sum(triangle_angles[np.where(cell_triangle_cells==c)],triangle_vertices[np.where(cell_triangle_cells==c)],index=topomesh.wisp_property('vertices',degree=3)[c])
    #         vertex_area = nd.sum(triangle_areas[np.where(cell_triangle_cells==c)[0]][:,np.newaxis]/3.,triangle_vertices[np.where(cell_triangle_cells==c)],index=topomesh.wisp_property('vertices',degree=3)[c])
    #         vertex_curvature = array_dict(values=(2*np.pi - vertex_angle_sum)/vertex_area,keys=topomesh.wisp_property('vertices',degree=3)[c])
    #         # print vertex_curvature
    #         for v in vertex_curvature.keys():
    #             vertex_cell_curvatures[v][c] = vertex_curvature[v]

    #     vertex_cell_curvatures = np.array([np.array(vertex_cell_curvatures[v].values()) for v in topomesh.wisps(0)])

    #     topomesh.update_wisp_property('gaussian_curvature',degree=degree,values=vertex_cell_curvatures,keys=np.array(list(topomesh.wisps(degree))))

    # if property_name == 'mean_curvature':
    #     assert degree==0
    #     if not 'curvature' in topomesh.wisp_property_names(degree):
    #         topomesh.add_wisp_property('curvature',degree=degree)
    #     if not topomesh.has_wisp_property('borders',degree=3,is_computed=True):
    #         compute_topomesh_property(topomesh,'borders',degree=3)
    #     if not topomesh.has_wisp_property('vertices',degree=3,is_computed=True):
    #         compute_topomesh_property(topomesh,'vertices',degree=3)
    #     if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
    #         compute_topomesh_property(topomesh,'vertices',degree=2)
    #     if not topomesh.has_wisp_property('cells',degree=0,is_computed=True):
    #         compute_topomesh_property(topomesh,'cells',degree=0)
    #     if not topomesh.has_wisp_property('length',degree=1,is_computed=True):
    #         compute_topomesh_property(topomesh,'length',degree=1)
    #     if not topomesh.has_wisp_property('barycenter',degree=3,is_computed=True):
    #         compute_topomesh_property(topomesh,'barycenter',degree=3)
    #     if not topomesh.has_wisp_property('barycenter',degree=2,is_computed=True):
    #         compute_topomesh_property(topomesh,'barycenter',degree=2)
    #     if not topomesh.has_wisp_property('area',degree=2,is_computed=True):
    #         compute_topomesh_property(topomesh,'area',degree=2)
    #     if not topomesh.has_wisp_property('normal',degree=2,is_computed=True):
    #         compute_topomesh_property(topomesh,'normal',degree=2)

    #     triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(list(topomesh.wisps(2)))
    #     rotated_triangle_vertices = np.transpose([triangle_vertices[:,2],triangle_vertices[:,0],triangle_vertices[:,1]])
    #     antirotated_triangle_vertices = np.transpose([triangle_vertices[:,1],triangle_vertices[:,2],triangle_vertices[:,0]])
    #     triangle_vertices = np.append(np.append(triangle_vertices,rotated_triangle_vertices,axis=0),antirotated_triangle_vertices,axis=0)

    #     edge_index_list = np.array([[1, 2],[0, 1],[0, 2]])
    #     triangle_edge_vertices = triangle_vertices[:,edge_index_list]

    #     triangle_edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,1]) - topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,0])
    #     #triangle_edge_lengths = np.power(np.sum(np.power(triangle_edge_vectors,2.0),axis=2),0.5)
    #     triangle_edge_lengths = np.linalg.norm(triangle_edge_vectors,axis=2)
    #     triangle_edge_directions = triangle_edge_vectors/triangle_edge_lengths[...,np.newaxis]

    #     triangle_perimeters = np.sum(triangle_edge_lengths,axis=1)
    #     triangle_areas = np.sqrt((triangle_perimeters/2.0)*(triangle_perimeters/2.0-triangle_edge_lengths[:,0])*(triangle_perimeters/2.0-triangle_edge_lengths[:,1])*(triangle_perimeters/2.0-triangle_edge_lengths[:,2]))

    #     triangle_cosines = np.zeros_like(triangle_edge_lengths,np.float32)
    #     triangle_cosines[:,0] = (triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2)/(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2])
    #     triangle_cosines[:,1] = (triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2)/(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0])
    #     triangle_cosines[:,2] = (triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2)/(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1])
    #     triangle_angles = np.arccos(triangle_cosines)

    #     triangle_sinuses = np.zeros_like(triangle_edge_lengths,np.float32)
    #     triangle_sinuses[:,0] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2],2.0),np.float16))
    #     triangle_sinuses[:,1] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0],2.0),np.float16))
    #     triangle_sinuses[:,2] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2,2.0)/np.power(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1],2.0),np.float16))
    #     triangle_sinuses[np.where(triangle_sinuses == 0.0)] = 0.001

    #     triangle_cotangent_vectors = (triangle_cosines/triangle_sinuses)[...,np.newaxis] * triangle_edge_vectors
    #     # triangle_cotangent_vectors = 1./np.tan(triangle_angles)[...,np.newaxis] * triangle_edge_vectors

    #     vertex_cotangent_sum = np.transpose([nd.sum(triangle_cotangent_vectors[:,1,0]+triangle_cotangent_vectors[:,2,0],triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0)))),
    #                                          nd.sum(triangle_cotangent_vectors[:,1,1]+triangle_cotangent_vectors[:,2,1],triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0)))),
    #                                          nd.sum(triangle_cotangent_vectors[:,1,2]+triangle_cotangent_vectors[:,2,2],triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0))))])
                

    #     vertex_area = nd.sum(triangle_areas,triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0))))
    #     # vertex_area = nd.sum(triangle_cotangent_square_lengths[:,1]+triangle_cotangent_square_lengths[:,2], triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0))))/8.

    #     # vertex_mean_curvature_vectors = vertex_cotangent_sum/(2.*vertex_area[:,np.newaxis])
    #     vertex_mean_curvature_vectors = (3.*vertex_cotangent_sum)/(4.*vertex_area[:,np.newaxis])


    #     triangle_cotangent_vectors = 1./np.tan(triangle_angles)[...,np.newaxis] * triangle_edge_vectors
    #     triangle_cotangent_square_lengths = 1./np.tan(triangle_angles) * triangle_edge_lengths**2

    #     # triangle_cell_directions = topomesh.wisp_property('barycenter',degree=2).values(list(topomesh.wisps(2))) - topomesh.wisp_property('barycenter',degree=3).values([topomesh.wisp_property('cells',degree=2)[t][0] for t in topomesh.wisps(2)])
    #     # triangle_cell_directions = np.tile(triangle_cell_directions,(3,1))

    #     # triangle_normal_vectors = np.cross(triangle_edge_vectors[:,0],triangle_edge_vectors[:,1])
    #     # triangle_normal_vectors = triangle_normal_vectors/np.linalg.norm(triangle_normal_vectors,axis=1)[:,np.newaxis]

    #     # reversed_normals = np.where(np.einsum('ij,ij->i',triangle_normal_vectors,triangle_cell_directions) < 0)[0]
    #     # triangle_normal_vectors[reversed_normals] = -triangle_normal_vectors[reversed_normals]

    #     triangle_normal_vectors = np.tile(topomesh.wisp_property('normal',degree=2).values(list(topomesh.wisps(2))),(3,1))
    #     triangle_areas = np.tile(topomesh.wisp_property('area',degree=2).values(list(topomesh.wisps(2))),(3))

    #     vertex_normal_vectors = np.transpose([nd.sum(triangle_normal_vectors[:,0]*triangle_areas,triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0)))),
    #                                           nd.sum(triangle_normal_vectors[:,1]*triangle_areas,triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0)))),
    #                                           nd.sum(triangle_normal_vectors[:,2]*triangle_areas,triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0))))])

    #     vertex_mean_curvature_sign = -np.sign(np.einsum('ij,ij->i',vertex_mean_curvature_vectors,vertex_normal_vectors))
    #     vertex_mean_curvatures = array_dict(values=vertex_mean_curvature_sign*np.linalg.norm(vertex_mean_curvature_vectors,axis=1),keys=np.array(list(topomesh.wisps(0))))

    #     topomesh.update_wisp_property('mean_curvature',degree=degree,values=vertex_mean_curvatures,keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'length':
        assert degree == 1
        if not 'length' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('length',degree=degree)
        if not topomesh.has_wisp_property('borders',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'borders',degree=degree)
        vertices_positions = positions.values(topomesh.wisp_property('borders',degree).values(list(topomesh.wisps(degree))))
        edge_vectors = vertices_positions[:,1] - vertices_positions[:,0]
        edge_lengths = np.power(np.sum(np.power(edge_vectors,2.0),axis=1),0.5)
        topomesh.update_wisp_property('length',degree=degree,values=edge_lengths,keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'perimeter':
        assert degree == 2
        if not 'perimeter' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('perimeter',degree=degree)
        if not topomesh.has_wisp_property('borders',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'borders',degree=degree)
        if not topomesh.has_wisp_property('length',degree=1,is_computed=True):
            compute_topomesh_property(topomesh,'length',degree=1)
        edge_lengths = topomesh.wisp_property('length',degree=1).values(topomesh.wisp_property('borders',degree=degree).values(list(topomesh.wisps(degree))))
        triangle_perimeters = np.sum(edge_lengths,axis=1)
        topomesh.update_wisp_property('perimeter',degree=degree,values=triangle_perimeters,keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'area':
        assert degree == 2
        if not 'area' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('area',degree=degree)
        if not topomesh.has_wisp_property('borders',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'borders',degree=degree)
        if not topomesh.has_wisp_property('length',degree=1,is_computed=True):
            compute_topomesh_property(topomesh,'length',degree=1)
        edge_lengths = topomesh.wisp_property('length',degree=1).values(topomesh.wisp_property('borders',degree=degree).values(list(topomesh.wisps(degree))))
        triangle_perimeters = np.sum(edge_lengths,axis=1)
        triangle_areas = np.sqrt((triangle_perimeters/2.0)*(triangle_perimeters/2.0-edge_lengths[:,0])*(triangle_perimeters/2.0-edge_lengths[:,1])*(triangle_perimeters/2.0-edge_lengths[:,2]))
        topomesh.update_wisp_property('area',degree=degree,values=triangle_areas,keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'eccentricity':
        assert degree == 2
        if not 'eccentricity' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('eccentricity',degree=degree)
        if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
            compute_topomesh_property(topomesh,'vertices',degree=2)

        triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(list(topomesh.wisps(degree)))
        edge_index_list = np.array([[1, 2],[0, 1],[0, 2]])
        triangle_edge_vertices = triangle_vertices[:,edge_index_list]
        triangle_edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,1]) - topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,0])
        triangle_edge_lengths = np.linalg.norm(triangle_edge_vectors,axis=2)

        triangle_sinuses = np.zeros_like(triangle_edge_lengths,np.float32)
        triangle_sinuses[:,0] = np.power(np.array(1.0 - np.power((triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2)/(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2]),2.0),np.float16),0.5)
        triangle_sinuses[:,1] = np.power(np.array(1.0 - np.power((triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2)/(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1]),2.0),np.float16),0.5)
        triangle_sinuses[:,2] = np.power(np.array(1.0 - np.power((triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2)/(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0]),2.0),np.float16),0.5)

        triangle_sinus_eccentricities = 1.0 - (2.0*(triangle_sinuses[:,0]+triangle_sinuses[:,1]+triangle_sinuses[:,2]))/(3.*np.sqrt(3.))
        topomesh.update_wisp_property('eccentricity',degree=degree,values=triangle_sinus_eccentricities,keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'normal':
        if degree == 2:
            if not 'normal' in topomesh.wisp_property_names(degree):
                topomesh.add_wisp_property('normal',degree=degree)
            if not topomesh.has_wisp_property('vertices',degree=degree,is_computed=True):
                compute_topomesh_property(topomesh,'vertices',degree=degree)
            if not topomesh.has_wisp_property('epidermis',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'epidermis',2)
            if not topomesh.has_wisp_property('barycenter',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'barycenter',2)
            if not topomesh.has_wisp_property('barycenter',degree=3,is_computed=True):
                compute_topomesh_property(topomesh,'barycenter',3)
            

            if normal_method == "orientation":
                normal_cell = max(topomesh.wisps(3))+1
                topomesh.add_wisp(3,normal_cell)
                for t in topomesh.wisps(2):
                    topomesh.link(3,normal_cell,t)

                compute_topomesh_property(topomesh,'oriented_borders',2)
                compute_topomesh_property(topomesh,'oriented_borders',3)

                triangle_orientations = array_dict(topomesh.wisp_property('oriented_borders',3)[normal_cell][1],topomesh.wisp_property('oriented_borders',3)[normal_cell][0])
                topomesh.update_wisp_property('orientation',2,triangle_orientations.values(list(topomesh.wisps(2))),list(topomesh.wisps(2)))
                
                topomesh.remove_wisp(3,normal_cell)

                compute_topomesh_property(topomesh,'vertices',2)

                compute_topomesh_property(topomesh,'oriented_vertices',2)
                vertices_positions = topomesh.wisp_property('barycenter',0).values(topomesh.wisp_property('oriented_vertices',2).values())
                normal_vectors = np.cross(vertices_positions[:,1]-vertices_positions[:,0],vertices_positions[:,2]-vertices_positions[:,0])
                normal_norms = np.linalg.norm(normal_vectors,axis=1)
                normal_orientations = topomesh.wisp_property('orientation',2).values()

                face_vectors = vertices_positions.mean(axis=1) - topomesh.wisp_property('barycenter',0).values().mean(axis=0)
                global_normal_orientation = np.sign(np.einsum("...ij,...ij->...i",normal_orientations[:,np.newaxis]*normal_vectors,face_vectors).mean())
                topomesh.update_wisp_property('normal',2,global_normal_orientation*normal_orientations[:,np.newaxis]*normal_vectors/normal_norms[:,np.newaxis],list(topomesh.wisps(2)))

            elif normal_method == "density":
                vertices_positions = positions.values(topomesh.wisp_property('vertices',degree).values(list(topomesh.wisps(degree))))
                normal_vectors = np.cross(vertices_positions[:,1]-vertices_positions[:,0],vertices_positions[:,2]-vertices_positions[:,0])
                
                # reversed_normals = np.where(normal_vectors[:,2] < 0)[0]
            
                from openalea.cellcomplex.property_topomesh.utils.implicit_surfaces import point_spherical_density
                if object_positions is None:
                    object_positions = topomesh.wisp_property('barycenter',3)
                triangle_epidermis = topomesh.wisp_property('epidermis',2).values()

                triangle_exterior_density = point_spherical_density(object_positions,topomesh.wisp_property('barycenter',2).values()[triangle_epidermis]+normal_vectors[triangle_epidermis],sphere_radius=object_radius,k=0.5)
                triangle_interior_density = point_spherical_density(object_positions,topomesh.wisp_property('barycenter',2).values()[triangle_epidermis]-normal_vectors[triangle_epidermis],sphere_radius=object_radius,k=0.5)
                normal_orientation = 2*(triangle_exterior_density<triangle_interior_density)-1
                normal_vectors[triangle_epidermis] = normal_orientation[...,np.newaxis]*normal_vectors[triangle_epidermis]
                normal_norms = np.linalg.norm(normal_vectors,axis=1)
                # normal_norms[np.where(normal_norms==0)] = 0.001
                topomesh.update_wisp_property('normal',degree=degree,values=normal_vectors/normal_norms[:,np.newaxis],keys=np.array(list(topomesh.wisps(degree))))

            elif normal_method == "barycenter":
                vertices_positions = positions.values(topomesh.wisp_property('vertices',degree).values(list(topomesh.wisps(degree))))
                normal_vectors = np.cross(vertices_positions[:,1]-vertices_positions[:,0],vertices_positions[:,2]-vertices_positions[:,0])

                barycenter_vectors = topomesh.wisp_property('barycenter',2).values() - positions.values().mean(axis=0)
                normal_orientation = np.sign(np.einsum("...ij,...ij->...i",normal_vectors,barycenter_vectors))
                normal_vectors = normal_orientation[...,np.newaxis]*normal_vectors
                normal_norms = np.linalg.norm(normal_vectors,axis=1)

                topomesh.update_wisp_property('normal',degree=degree,values=normal_vectors/normal_norms[:,np.newaxis],keys=np.array(list(topomesh.wisps(degree))))

        

        elif degree == 0:
            if not topomesh.has_wisp_property('normal',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'normal',degree=2)
            if not topomesh.has_wisp_property('area',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'area',degree=2)
            if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'vertices',degree=2)
            if not topomesh.has_wisp_property('epidermis',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'epidermis',degree=2)

            epidermis_triangles = np.array(list(topomesh.wisps(2)))[topomesh.wisp_property('epidermis',2).values(list(topomesh.wisps(2)))]
            # print epidermis_triangles

            #triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(list(topomesh.wisps(2)))
            triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(epidermis_triangles)
            rotated_triangle_vertices = np.transpose([triangle_vertices[:,2],triangle_vertices[:,0],triangle_vertices[:,1]])
            antirotated_triangle_vertices = np.transpose([triangle_vertices[:,1],triangle_vertices[:,2],triangle_vertices[:,0]])
            triangle_vertices = np.append(np.append(triangle_vertices,rotated_triangle_vertices,axis=0),antirotated_triangle_vertices,axis=0)

            # triangle_areas = np.tile(topomesh.wisp_property('area',degree=2).values(list(topomesh.wisps(2))),(3))
            # triangle_normal_vectors = np.tile(topomesh.wisp_property('normal',degree=2).values(list(topomesh.wisps(2))),(3,1))
            triangle_areas = np.tile(topomesh.wisp_property('area',degree=2).values(epidermis_triangles),(3))
            triangle_normal_vectors = np.tile(topomesh.wisp_property('normal',degree=2).values(epidermis_triangles),(3,1))

            vertex_normal_vectors = np.transpose([nd.sum(triangle_normal_vectors[:,k]*triangle_areas,triangle_vertices[:,0],index=np.array(list(topomesh.wisps(0)))) for k in [0,1,2]])
            normal_norms = np.linalg.norm(vertex_normal_vectors,axis=1)
            normal_norms[np.where(normal_norms==0)] = 0.001
            topomesh.update_wisp_property('normal',degree=degree,values=vertex_normal_vectors/normal_norms[:,np.newaxis],keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'angles':
        assert degree == 2
        if not 'angles' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('angles',degree=degree)
        if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
            compute_topomesh_property(topomesh,'vertices',degree=2)

        triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(list(topomesh.wisps(degree)))
        edge_index_list = np.array([[1, 2],[0, 1],[0, 2]])
        triangle_edge_vertices = triangle_vertices[:,edge_index_list]
        triangle_edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,1]) - topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,0])
        triangle_edge_lengths = np.linalg.norm(triangle_edge_vectors,axis=2)

        triangle_cosines = np.zeros_like(triangle_edge_lengths,np.float32)
        triangle_cosines[:,0] = (triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2)/(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2])
        triangle_cosines[:,1] = (triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2)/(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1])
        triangle_cosines[:,2] = (triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2)/(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0])
        triangle_angles = np.arccos(triangle_cosines)
        topomesh.update_wisp_property('angles',degree=degree,values=triangle_angles,keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'incircle center':
        assert degree == 2
        if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
            compute_topomesh_property(topomesh,'vertices',degree=2)

        triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(list(topomesh.wisps(degree)))
        edge_index_list = np.array([[1, 2],[0, 2],[0, 1]])
        triangle_edge_vertices = triangle_vertices[:,edge_index_list]
        triangle_edge_vectors = topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,1]) - topomesh.wisp_property('barycenter',degree=0).values(triangle_edge_vertices[...,0])
        triangle_edge_lengths = np.linalg.norm(triangle_edge_vectors,axis=2)

        triangle_sinuses = np.zeros_like(triangle_edge_lengths,np.float32)

        triangle_sinuses[:,0] = np.sqrt(1.0 - np.power((triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2)/(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2]),2.0))
        triangle_sinuses[:,1] = np.sqrt(1.0 - np.power((triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2)/(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1]),2.0))
        triangle_sinuses[:,2] = np.sqrt(1.0 - np.power((triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2)/(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0]),2.0))

        # print triangle_sinuses.shape
        # print triangle_vertices.shape

        triangle_centers = ((topomesh.wisp_property('barycenter',degree=0).values(triangle_vertices)/triangle_sinuses[...,np.newaxis]).sum(axis=1))/((1./triangle_sinuses).sum(axis=1)[...,np.newaxis])
        # print triangle_centers
        # print triangle_centers.shape
        topomesh.update_wisp_property('incircle center',degree=degree,values=triangle_centers,keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'volume':
        assert degree == 3
        if not 'volume' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('volume',degree=degree)
        if not topomesh.has_wisp_property('borders',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'borders',degree=degree)
        if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
            compute_topomesh_property(topomesh,'vertices',degree=2)
        if not topomesh.has_wisp_property('barycenter',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'barycenter',degree=degree)

        cell_triangles = np.array(np.concatenate(topomesh.wisp_property('borders',degree=3).values(list(topomesh.wisps(degree)))),int)
        cell_tetrahedra_cells = np.array(np.concatenate([[c for t in topomesh.wisp_property('borders',degree=3)[c]] for c in topomesh.wisps(3)]),int)
        cell_triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(cell_triangles)
        cell_tetrahedra = np.concatenate([topomesh.wisp_property('barycenter',degree=3).values(cell_tetrahedra_cells)[:,np.newaxis], topomesh.wisp_property('barycenter',degree=0).values(cell_triangle_vertices)],axis=1)
    
        cell_tetra_matrix = np.transpose(np.array([cell_tetrahedra[:,1],cell_tetrahedra[:,2],cell_tetrahedra[:,3]]) - cell_tetrahedra[:,0],axes=(1,2,0))
        cell_tetra_volume = abs(np.linalg.det(cell_tetra_matrix))/6.0

        cell_volumes = nd.sum(cell_tetra_volume,cell_tetrahedra_cells,index=list(topomesh.wisps(3)))
        topomesh.update_wisp_property('volume',degree=degree,values=cell_volumes,keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'convexhull_volume':
        import openalea.plantgl.all as pgl
        assert degree == 3
        if not 'convexhull_volume' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('convexhull_volume',degree=degree)
        if not topomesh.has_wisp_property('vertices',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'vertices',degree=degree)
        for c in topomesh.wisps(3):
            cell_vertices = topomesh.wisp_property('vertices',degree)[c]
            if len(cell_vertices)>0:   
                convex_hull = pgl.Fit(pgl.Point3Array(positions.values(cell_vertices))).convexHull()
                topomesh.wisp_property('convexhull_volume',degree=degree)[c] = pgl.volume(convex_hull)

    if property_name == 'surface':
        assert degree == 3
        if not 'surface' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('surface',degree=degree)
        if not topomesh.has_wisp_property('borders',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'borders',degree=degree)
        if not topomesh.has_wisp_property('area',degree=2,is_computed=True):
            compute_topomesh_property(topomesh,'area',degree=2)

        cell_triangles = np.array(np.concatenate(topomesh.wisp_property('borders',degree=3).values(list(topomesh.wisps(degree)))),int)
        cell_triangle_cells = np.array(np.concatenate([[c for t in topomesh.wisp_property('borders',degree=3)[c]] for c in topomesh.wisps(3)]),int)

        cell_triangle_areas = topomesh.wisp_property('area',degree=2).values(cell_triangles)
        cell_surfaces = nd.sum(cell_triangle_areas,cell_triangle_cells,index=list(topomesh.wisps(3)))
        topomesh.update_wisp_property('surface',degree=degree,values=cell_surfaces,keys=np.array(list(topomesh.wisps(degree))))

    if property_name == 'convexhull_surface':
        import openalea.plantgl.all as pgl
        assert degree == 3
        if not 'convexhull_surface' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('convexhull_surface',degree=degree)
        if not topomesh.has_wisp_property('vertices',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'vertices',degree=degree)
        for c in topomesh.wisps(3):
            cell_vertices = topomesh.wisp_property('vertices',degree)[c]
            if len(cell_vertices)>0:   
                convex_hull = pgl.Fit(pgl.Point3Array(topomesh.wisp_property('barycenter',0).values(cell_vertices))).convexHull()
                topomesh.wisp_property('convexhull_surface',degree=degree)[c] = pgl.surface(convex_hull)

    if property_name == 'convexity':
        assert degree == 3
        if not 'convexity' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('convexity',degree=degree)
        if not topomesh.has_wisp_property('volume',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'volume',degree=degree)
        if not topomesh.has_wisp_property('convexhull_volume',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'convexhull_volume',degree=degree)
        topomesh.update_wisp_property('convexity',degree=degree,values=topomesh.wisp_property('volume',3).values(topomesh.wisp_property('convexhull_volume',3).keys())/topomesh.wisp_property('convexhull_volume',3).values(topomesh.wisp_property('convexhull_volume',3).keys()),keys=topomesh.wisp_property('convexhull_volume',3).keys())

    if property_name == 'epidermis':
        if not 'epidermis' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('epidermis',degree=degree)
        if degree != 2:
            if not topomesh.has_wisp_property('triangles',degree=degree,is_computed=True):
                compute_topomesh_property(topomesh,'triangles',degree=degree)
            if not topomesh.has_wisp_property('epidermis',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'epidermis',degree=2)
            epidermis = np.array([topomesh.wisp_property('epidermis',degree=2).values(topomesh.wisp_property('triangles',degree=degree)[w]).any() for w in topomesh.wisps(degree)])
            topomesh.update_wisp_property('epidermis',degree=degree,values=epidermis,keys=np.array(list(topomesh.wisps(degree))))
        else:
            if not topomesh.has_wisp_property('cells',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'cells',degree=2)
            topomesh.update_wisp_property('epidermis',degree=2,values=(np.array(map(len,topomesh.wisp_property('cells',2).values(list(topomesh.wisps(2))))) == 1),keys=np.array(list(topomesh.wisps(2))))

    # if property_name == 'epidermis_curvature':
    #     if degree == 0:
    #         if not topomesh.has_wisp_property('epidermis',degree=0,is_computed=True):
    #             compute_topomesh_property(topomesh,'epidermis',degree=0)
    #         if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
    #             compute_topomesh_property(topomesh,'vertices',degree=2)
    #         if not topomesh.has_wisp_property('barycenter',degree=2,is_computed=True):
    #             compute_topomesh_property(topomesh,'barycenter',degree=2)
    #         if not topomesh.has_wisp_property('barycenter',degree=3,is_computed=True):
    #             compute_topomesh_property(topomesh,'barycenter',degree=3)
    #         if not topomesh.has_wisp_property('area',degree=2,is_computed=True):
    #             compute_topomesh_property(topomesh,'area',degree=2)
    #         if not topomesh.has_wisp_property('normal',degree=2,is_computed=True):
    #             compute_topomesh_property(topomesh,'normal',degree=2)
            
    #         curvature_radius = kwargs.get('radius',15)

    #         epidermis_vertices = np.array(list(topomesh.wisps(0)))[topomesh.wisp_property('epidermis',0).values(list(topomesh.wisps(0)))]
    #         vertex_distances = np.array([vq(topomesh.wisp_property('barycenter',degree=0).values(epidermis_vertices),topomesh.wisp_property('barycenter',degree=0).values([v]))[1] for v in epidermis_vertices])

    #         vertex_neighborhood = np.array([np.array(list(topomesh.wisps(0)))[np.where(vertex_distances[v]<curvature_radius)] for v in xrange(epidermis_vertices.shape[0])])
    #         vertex_neighborhood_barycenter = np.array([np.mean(topomesh.wisp_property('barycenter',degree=0).values(vertex_neighborhood[v]),axis=0) for v in xrange(epidermis_vertices.shape[0])])
    #         vertex_vectors = topomesh.wisp_property('barycenter',degree=0).values(epidermis_vertices) - vertex_neighborhood_barycenter
    #         vertex_vector_norm = np.linalg.norm(vertex_vectors,axis=1)

    #         epidermis_triangles = np.array(list(topomesh.wisps(2)))[topomesh.wisp_property('epidermis',2).values(list(topomesh.wisps(2)))]
    #         triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(epidermis_triangles)
    #         rotated_triangle_vertices = np.transpose([triangle_vertices[:,2],triangle_vertices[:,0],triangle_vertices[:,1]])
    #         antirotated_triangle_vertices = np.transpose([triangle_vertices[:,1],triangle_vertices[:,2],triangle_vertices[:,0]])
    #         triangle_vertices = np.append(np.append(triangle_vertices,rotated_triangle_vertices,axis=0),antirotated_triangle_vertices,axis=0)

    #         triangle_normal_vectors = np.tile(topomesh.wisp_property('normal',degree=2).values(epidermis_triangles),(3,1))
    #         triangle_areas = np.tile(topomesh.wisp_property('area',degree=2).values(epidermis_triangles),(3))

    #         print triangle_normal_vectors.shape
    #         print triangle_areas.shape
    #         print triangle_vertices.shape

    #         vertex_normal_vectors = np.transpose([nd.sum(triangle_normal_vectors[:,0]*triangle_areas,triangle_vertices[:,0],index=epidermis_vertices),
    #                                               nd.sum(triangle_normal_vectors[:,1]*triangle_areas,triangle_vertices[:,0],index=epidermis_vertices),
    #                                               nd.sum(triangle_normal_vectors[:,2]*triangle_areas,triangle_vertices[:,0],index=epidermis_vertices)])

    #         vertex_curvature_sign = np.sign(np.einsum('ij,ij->i',vertex_vectors,vertex_normal_vectors))

    #         vertex_curvature = array_dict(4.*vertex_curvature_sign*vertex_vector_norm/curvature_radius,keys=epidermis_vertices)
    #         vertex_curvature_inner = np.array([vertex_curvature[v] if v in epidermis_vertices else 0.0 for v in topomesh.wisps(0)])

    #         topomesh.update_wisp_property('epidermis_curvature',0,values=vertex_curvature_inner,keys=np.array(list(topomesh.wisps(0))))
    #     if degree == 2:
    #         compute_topomesh_property(topomesh,'epidermis_curvature',degree=0,radius=kwargs.get('radius',60))
    #         triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(list(topomesh.wisps(2)))
    #         triangle_curvature = topomesh.wisp_property('epidermis_curvature',degree=0).values(triangle_vertices).mean(axis=1)
    #         topomesh.update_wisp_property('epidermis_curvature',2,values=triangle_curvature,keys=np.array(list(topomesh.wisps(2))))

    if property_name in ['principal_curvatures','principal_curvature_tensor','principal_direction_min','principal_direction_max','principal_curvature_min','principal_curvature_max','mean_curvature','gaussian_curvature']:
        if degree == 0:

            if not topomesh.has_wisp_property('barycenter',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'barycenter',degree=2)
            if not topomesh.has_wisp_property('barycenter',degree=3,is_computed=True):
                compute_topomesh_property(topomesh,'barycenter',degree=3)
            if not topomesh.has_wisp_property('area',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'area',degree=2)
            if not topomesh.has_wisp_property('normal',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'normal',degree=2)
            if not topomesh.has_wisp_property('normal',degree=0,is_computed=True):
                compute_topomesh_property(topomesh,'normal',degree=0)

            if not topomesh.has_wisp_property('neighors',degree=0,is_computed=True):
                compute_topomesh_property(topomesh,'region_neighbors',0)
            if not topomesh.has_wisp_property('epidermis',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'epidermis',2)
            if not topomesh.has_wisp_property('epidermis',degree=0,is_computed=True):
                compute_topomesh_property(topomesh,'epidermis',0)

            vertex_normal_vectors = topomesh.wisp_property('normal',0)

            epidermis_vertices = np.array(list(topomesh.wisps(0)))[topomesh.wisp_property('epidermis',0).values(list(topomesh.wisps(0)))]

            vertex_neighbor_vertices = np.concatenate([[n for n in topomesh.wisp_property('neighbors',0)[v] if n in epidermis_vertices] for v in epidermis_vertices])
            vertex_neighbor_vertex   = np.concatenate([[v for n in topomesh.wisp_property('neighbors',0)[v] if n in epidermis_vertices] for v in epidermis_vertices])

            vertex_neighbor_vectors = topomesh.wisp_property('barycenter',0).values(vertex_neighbor_vertex) - topomesh.wisp_property('barycenter',0).values(vertex_neighbor_vertices)

            vertex_neighbor_projected_vectors = vertex_neighbor_vectors - np.einsum('ij,ij->i',vertex_neighbor_vectors,vertex_normal_vectors.values(vertex_neighbor_vertex))[:,np.newaxis]*vertex_normal_vectors.values(vertex_neighbor_vertex)
            vertex_neighbor_projected_vectors = vertex_neighbor_projected_vectors/np.linalg.norm(vertex_neighbor_projected_vectors,axis=1)[:,np.newaxis]

            vertex_neighbor_directional_curvature = 2.*np.einsum('ij,ij->i',vertex_normal_vectors.values(vertex_neighbor_vertex),vertex_neighbor_vectors)/np.power(np.linalg.norm(vertex_neighbor_vectors,axis=1),2.0)

            # epidermis_triangles = np.array(list(topomesh.wisps(2)))[topomesh.wisp_property('epidermis',2).values(list(topomesh.wisps(2)))]
            compute_topomesh_property(topomesh,'triangles',0)
            vertex_neighbor_triangles = np.array([[t for t in topomesh.wisp_property('triangles',0)[v] if (t in topomesh.wisp_property('triangles',0)[n]) and (topomesh.wisp_property('epidermis',2)[t])] for (v,n) in zip(vertex_neighbor_vertex,vertex_neighbor_vertices)])
            vertex_neighbor_triangle_area = topomesh.wisp_property('area',2).values(vertex_neighbor_triangles)
            vertex_neighbor_triangle_area = np.array([a.sum() for a in vertex_neighbor_triangle_area])

            vertex_neighbor_projected_matrix = np.einsum('...ji,...ij->...ij',vertex_neighbor_projected_vectors[...,np.newaxis],vertex_neighbor_projected_vectors[...,np.newaxis])
            vertex_neighbor_curvature_matrix = vertex_neighbor_triangle_area[:,np.newaxis,np.newaxis]*vertex_neighbor_directional_curvature[:,np.newaxis,np.newaxis]*vertex_neighbor_projected_matrix

            vertex_curvature_matrix = np.zeros((epidermis_vertices.shape[0],3,3))
            for i in xrange(3):
                for j in xrange(3):
                       vertex_curvature_matrix[:,i,j] = nd.sum(vertex_neighbor_curvature_matrix[:,i,j],vertex_neighbor_vertex,index=epidermis_vertices)    
            vertex_curvature_matrix = vertex_curvature_matrix / (nd.sum(vertex_neighbor_triangle_area,vertex_neighbor_vertex,index=epidermis_vertices)[:,np.newaxis,np.newaxis]+np.power(10.,-10))

            vertex_curvature_tensor = array_dict(vertex_curvature_matrix,epidermis_vertices)

            try:
                vertex_curvature_matrix_eigenvalues, vertex_curvature_matrix_eigenvectors = np.linalg.eig(vertex_curvature_matrix)
            except np.linalg.LinAlgError:
            #except LinAlgError:
                ok_curvature_matrix = np.unique(np.where(1-np.isnan(vertex_curvature_matrix))[0])
                nan_curvature_matrix = np.unique(np.where(np.isnan(vertex_curvature_matrix))[0])
                vertex_curvature_matrix_trunc = np.delete(vertex_curvature_matrix,nan_curvature_matrix)
                vertex_curvature_matrix_eigenvalues_trunc, vertex_curvature_matrix_eigenvectors_trunc = np.linalg.eig(vertex_curvature_matrix_trunc)
                vertex_curvature_matrix_eigenvalues = np.zeros((vertex_curvature_matrix.shape[0],3))
                vertex_curvature_matrix_eigenvectors = np.zeros((vertex_curvature_matrix.shape[0],3,3))
                vertex_curvature_matrix_eigenvalues[ok_curvature_matrix] = vertex_curvature_matrix_eigenvalues_trunc
                vertex_curvature_matrix_eigenvectors[ok_curvature_matrix] = vertex_curvature_matrix_eigenvectors_trunc

            vertex_principal_curvature_min = array_dict(vertex_curvature_matrix_eigenvalues[tuple([np.arange(len(epidermis_vertices)),np.argsort(np.abs(vertex_curvature_matrix_eigenvalues))[:,1]])],epidermis_vertices)
            vertex_principal_curvature_max = array_dict(vertex_curvature_matrix_eigenvalues[tuple([np.arange(len(epidermis_vertices)),np.argsort(np.abs(vertex_curvature_matrix_eigenvalues))[:,2]])],epidermis_vertices)

            vertex_principal_direction_min = array_dict(vertex_curvature_matrix_eigenvectors[tuple([np.arange(len(epidermis_vertices)),np.argsort(np.abs(vertex_curvature_matrix_eigenvalues))[:,1]])],epidermis_vertices)
            vertex_principal_direction_max = array_dict(vertex_curvature_matrix_eigenvectors[tuple([np.arange(len(epidermis_vertices)),np.argsort(np.abs(vertex_curvature_matrix_eigenvalues))[:,2]])],epidermis_vertices)

            topomesh.update_wisp_property('principal_curvature_tensor',0,np.array([vertex_curvature_tensor[v] if v in epidermis_vertices else np.zeros((3,3),float) for v in topomesh.wisps(0)]),np.array(list(topomesh.wisps(0))))

            topomesh.update_wisp_property('principal_direction_min',0,np.array([vertex_principal_direction_min[v] if v in epidermis_vertices else np.array([0.,0.,0.]) for v in topomesh.wisps(0)]),np.array(list(topomesh.wisps(0))))
            topomesh.update_wisp_property('principal_direction_max',0,np.array([vertex_principal_direction_max[v] if v in epidermis_vertices else np.array([0.,0.,0.]) for v in topomesh.wisps(0)]),np.array(list(topomesh.wisps(0))))

            topomesh.update_wisp_property('principal_curvature_min',0,np.array([vertex_principal_curvature_min[v] if v in epidermis_vertices else 0. for v in topomesh.wisps(0)]),np.array(list(topomesh.wisps(0))))
            topomesh.update_wisp_property('principal_curvature_max',0,np.array([vertex_principal_curvature_max[v] if v in epidermis_vertices else 0. for v in topomesh.wisps(0)]),np.array(list(topomesh.wisps(0))))
            topomesh.update_wisp_property('mean_curvature',0,(topomesh.wisp_property('principal_curvature_max',0).values(list(topomesh.wisps(0)))+topomesh.wisp_property('principal_curvature_min',0).values(list(topomesh.wisps(0))))/2.,np.array(list(topomesh.wisps(0))))
            topomesh.update_wisp_property('gaussian_curvature',0,topomesh.wisp_property('principal_curvature_max',0).values(list(topomesh.wisps(0)))*topomesh.wisp_property('principal_curvature_min',0).values(list(topomesh.wisps(0))),np.array(list(topomesh.wisps(0))))
        
        elif degree == 2:
            # if not topomesh.has_wisp_property('principal_curvature_tensor',degree=0,is_computed=True):
            #     compute_topomesh_property(topomesh,'principal_curvature_tensor',degree=0)
            # if not topomesh.has_wisp_property('epidermis',degree=2,is_computed=True):
            #     compute_topomesh_property(topomesh,'epidermis',2)
            # if not topomesh.has_wisp_property('vertices',degree=2,is_computed=True):
            #     compute_topomesh_property(topomesh,'vertices',2)

            # epidermis_triangles = np.array(list(topomesh.wisps(2)))[topomesh.wisp_property('epidermis',2).values(list(topomesh.wisps(2)))]
            # triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(epidermis_triangles)

            # triangle_curvature_tensor = array_dict(topomesh.wisp_property('principal_curvature_tensor',degree=0).values(triangle_vertices).mean(axis=1),epidermis_triangles)
            # topomesh.update_wisp_property('principal_curvature_tensor',2,np.array([triangle_curvature_tensor[t] if t in epidermis_triangles else np.zeros((3,3),float) for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))
            # for direction in ['principal_direction_min','principal_direction_max']:
            #     triangle_direction = array_dict(topomesh.wisp_property(direction,degree=0).values(triangle_vertices).mean(axis=1),epidermis_triangles)
            #     topomesh.update_wisp_property(direction,2,np.array([triangle_direction[t] if t in epidermis_triangles else np.array([0.,0.,0.]) for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))
            # for curvature in ['principal_curvature_min','principal_curvature_max','mean_curvature','gaussian_curvature']:
            #     triangle_curvature = array_dict(topomesh.wisp_property(curvature,degree=0).values(triangle_vertices).mean(axis=1),epidermis_triangles)
            #     topomesh.update_wisp_property(curvature,2,np.array([triangle_curvature[t] if t in epidermis_triangles else 0. for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))
        
            if not topomesh.has_wisp_property('barycenter',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'barycenter',degree=2)
            if not topomesh.has_wisp_property('normal',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'normal',degree=2)
            if not topomesh.has_wisp_property('area',degree=2,is_computed=True):
                compute_topomesh_property(topomesh,'area',degree=2)
            if not topomesh.has_wisp_property('normal',degree=0,is_computed=True):
                compute_topomesh_property(topomesh,'normal',degree=0)

            epidermis_triangles = np.array(list(topomesh.wisps(2)))[topomesh.wisp_property('epidermis',2).values(list(topomesh.wisps(2))).astype(bool)]
            triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(epidermis_triangles)

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
            # weingarten_curvature_matrix[:,0,1] = (M1*E-L*F)/(E*G-F*F)
            weingarten_curvature_matrix[:,1,0] = (M1*E-L*F)/(E*G-F*F)
            # weingarten_curvature_matrix[:,1,0] = (M2*G-N*F)/(E*G-F*F)
            weingarten_curvature_matrix[:,1,1] = (N*E-M2*F)/(E*G-F*F)

            weingarten_curvature_matrix_eigenvalues, weingarten_curvature_matrix_eigenvectors = np.linalg.eig(weingarten_curvature_matrix)

            weingarten_curvature_matrix_eigenvalues = -weingarten_curvature_matrix_eigenvalues
            weingarten_curvature_matrix_eigenvectors = np.transpose(weingarten_curvature_matrix_eigenvectors,(0,2,1))

            weingarten_principal_curvature_min = weingarten_curvature_matrix_eigenvalues[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(weingarten_curvature_matrix_eigenvalues))[:,0]])].astype(float)
            weingarten_principal_curvature_max = weingarten_curvature_matrix_eigenvalues[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(weingarten_curvature_matrix_eigenvalues))[:,1]])].astype(float)

            weingarten_principal_vector_min = weingarten_curvature_matrix_eigenvectors[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(weingarten_curvature_matrix_eigenvalues))[:,0]])].astype(float)
            weingarten_principal_vector_max = weingarten_curvature_matrix_eigenvectors[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(weingarten_curvature_matrix_eigenvalues))[:,1]])].astype(float)

            weingarten_principal_direction_min = array_dict((weingarten_principal_vector_min[:,:,np.newaxis]*triangle_projected_barycenter_derivatives[:,1:3]).sum(axis=1),epidermis_triangles)
            weingarten_principal_direction_max = array_dict((weingarten_principal_vector_max[:,:,np.newaxis]*triangle_projected_barycenter_derivatives[:,1:3]).sum(axis=1),epidermis_triangles)

            P = np.transpose([weingarten_principal_direction_max.values(), weingarten_principal_direction_min.values(), triangle_barycenter_normals],(1,2,0))
            D = np.array([np.diag(d) for d in np.transpose([weingarten_principal_curvature_max, weingarten_principal_curvature_min, np.zeros_like(epidermis_triangles)])])
            P_i = np.array([np.linalg.pinv(p) for p in P])

            face_curvature_tensor = np.einsum('...ij,...jk->...ik',P,np.einsum('...ij,...jk->...ik',D,P_i))
            face_curvature_matrix_eigenvalues, face_curvature_matrix_eigenvectors = np.linalg.eig(face_curvature_tensor)

            face_principal_curvature_min = array_dict(face_curvature_matrix_eigenvalues[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(face_curvature_matrix_eigenvalues))[:,1]])].astype(float),epidermis_triangles)
            face_principal_curvature_max = array_dict(face_curvature_matrix_eigenvalues[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(face_curvature_matrix_eigenvalues))[:,2]])].astype(float),epidermis_triangles)

            face_curvature_matrix_eigenvectors = np.transpose(face_curvature_matrix_eigenvectors,(0,2,1))
            face_principal_direction_min = array_dict(face_curvature_matrix_eigenvectors[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(face_curvature_matrix_eigenvalues))[:,1]])].astype(float),epidermis_triangles)
            face_principal_direction_max = array_dict(face_curvature_matrix_eigenvectors[tuple([np.arange(len(epidermis_triangles)),np.argsort(np.abs(face_curvature_matrix_eigenvalues))[:,2]])].astype(float),epidermis_triangles)

            face_principal_curvature_tensor = array_dict(face_curvature_tensor,epidermis_triangles)
            topomesh.update_wisp_property('principal_curvature_tensor',2,np.array([face_principal_curvature_tensor[t] if t in epidermis_triangles else np.zeros((3,3)) for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))
            
            topomesh.update_wisp_property('principal_direction_min',2,np.array([face_principal_direction_min[t] if t in epidermis_triangles else np.zeros(3) for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))
            topomesh.update_wisp_property('principal_direction_max',2,np.array([face_principal_direction_max[t] if t in epidermis_triangles else np.zeros(3) for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))

            topomesh.update_wisp_property('principal_curvature_min',2,np.array([face_principal_curvature_min[t] if t in epidermis_triangles else 0. for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))
            topomesh.update_wisp_property('principal_curvature_max',2,np.array([face_principal_curvature_max[t] if t in epidermis_triangles else 0. for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))
            topomesh.update_wisp_property('mean_curvature',2,np.array([(face_principal_curvature_min[t] + face_principal_curvature_max[t])/2. if t in epidermis_triangles else 0. for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))
            topomesh.update_wisp_property('gaussian_curvature',2,np.array([(face_principal_curvature_min[t]*face_principal_curvature_max[t]) if t in epidermis_triangles else 0. for t in topomesh.wisps(2)]),np.array(list(topomesh.wisps(2))))


        elif degree == 3:
            if not topomesh.has_wisp_property('principal_curvature_tensor',degree=0,is_computed=True):
                compute_topomesh_property(topomesh,'principal_curvature_tensor',degree=0)
            if not topomesh.has_wisp_property('epidermis',degree=3,is_computed=True):
                compute_topomesh_property(topomesh,'epidermis',3)
            if not topomesh.has_wisp_property('vertices',degree=3,is_computed=True):
                compute_topomesh_property(topomesh,'vertices',3)
            if not topomesh.has_wisp_property('epidermis',degree=0,is_computed=True):
                compute_topomesh_property(topomesh,'epidermis',0)

            epidermis_cells = np.array(list(topomesh.wisps(3)))[topomesh.wisp_property('epidermis',3).values(list(topomesh.wisps(3)))]
            epidermis_vertices = np.array(list(topomesh.wisps(0)))[topomesh.wisp_property('epidermis',0).values(list(topomesh.wisps(0)))]
            cell_epidermis_vertices = np.array([np.intersect1d(topomesh.wisp_property('vertices',degree=3)[c],epidermis_vertices) for c in epidermis_cells])

            def row_mean(array):
                return np.mean(array,axis=0)
            cell_curvature_tensor = array_dict(map(row_mean,topomesh.wisp_property('principal_curvature_tensor',degree=0).values(cell_epidermis_vertices)),epidermis_cells)
            topomesh.update_wisp_property('principal_curvature_tensor',3,np.array([cell_curvature_tensor[c] if c in epidermis_cells else np.zeros((3,3),float) for c in topomesh.wisps(3)]),np.array(list(topomesh.wisps(3))))
            for direction in ['principal_direction_min','principal_direction_max']:
                cell_direction = array_dict(map(row_mean,topomesh.wisp_property(direction,degree=0).values(cell_epidermis_vertices)),epidermis_cells)
                topomesh.update_wisp_property(direction,3,np.array([cell_direction[c] if c in epidermis_cells else np.array([0.,0.,0.]) for c in topomesh.wisps(3)]),np.array(list(topomesh.wisps(3))))
            for curvature in ['principal_curvature_min','principal_curvature_max','mean_curvature','gaussian_curvature']:
                cell_curvature = array_dict(map(np.mean,topomesh.wisp_property(curvature,degree=0).values(cell_epidermis_vertices)),epidermis_cells)
                topomesh.update_wisp_property(curvature,3,np.array([cell_curvature[c] if c in epidermis_cells else 0. for c in topomesh.wisps(3)]),np.array(list(topomesh.wisps(3))))
        


    if property_name == 'cell_interface':
        assert degree==2
        if not 'cell_interface' in topomesh.wisp_property_names(degree):
            topomesh.add_wisp_property('cell_interface',degree=degree)
        if not topomesh.has_wisp_property('cells',degree=2,is_computed=True):
            compute_topomesh_property(topomesh,'cells',degree=2)
        interface_triangles = np.array(list(topomesh.wisps(2)))[np.where(np.array(map(len,topomesh.wisp_property('cells',2).values())) == 2)[0]]
        interface_triangle_cells = np.array([c for c in topomesh.wisp_property('cells',2).values(interface_triangles)])
        
        # def cell_interface(cid1,cid2):
        #     return topomesh.interface(3,cid1,cid2)
        # cell_interfaces = np.array(map(cell_interface,interface_triangle_cells[:,0],interface_triangle_cells[:,1])) 

        cell_interfaces = vq(np.sort(interface_triangle_cells),np.array(topomesh._interface[3].values()))[0]
        topomesh.update_wisp_property('cell_interface',degree=2,values=cell_interfaces,keys=interface_triangles)

    if property_name == 'interface':
        assert degree > 0
        if not 'interface' in topomesh.interface_property_names(degree):
            topomesh.add_interface_property('interface',degree=degree)
        if not topomesh.has_wisp_property('borders',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'borders',degree=degree)
        if not topomesh.has_wisp_property('neighbors',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'border_neighbors',degree=degree)

        neighbour_wids = np.array(topomesh._interface[degree].values())
        neighbour_borders = topomesh.wisp_property('borders',degree=degree).values(neighbour_wids)
        topomesh.update_interface_property('interface',degree=degree,values=np.array(map(np.intersect1d,neighbour_borders[:,0],neighbour_borders[:,1])),keys=np.array(list(topomesh.interfaces(degree))))

    if property_name == 'distance':
        assert degree > 0
        if not 'distance' in topomesh.interface_property_names(degree):
            topomesh.add_interface_property('distance',degree=degree)
        if not topomesh.has_wisp_property('barycenter',degree=degree,is_computed=True):
            compute_topomesh_property(topomesh,'barycenter',degree=degree)

        neighbour_wids = np.array(topomesh._interface[degree].values())
        neighbour_barycenters = topomesh.wisp_property('barycenter',degree=degree).values(neighbour_wids)
        neighbour_vectors = neighbour_barycenters[:,1] - neighbour_barycenters[:,0]
        topomesh.update_interface_property('interface',degree=degree,values=np.linalg.norm(neighbour_vectors,axis=1),keys=np.array(list(topomesh.interfaces(degree))))

    end_time = time()
    if verbose:
        print "<-- Computing",property_name,"property (",degree,") [",end_time-start_time,"s]"


def compute_topomesh_triangle_properties(topomesh,positions=None):
    """todo"""

    start_time = time()
    print "--> Computing triangle properties"

    if positions is None:
        positions = topomesh.wisp_property('barycenter',degree=0)
    if not topomesh.has_wisp_property('borders',degree=2,is_computed=True):
        compute_topomesh_property(topomesh,'borders',degree=2)
    if not topomesh.has_wisp_property('length',degree=1,is_computed=True):
        compute_topomesh_property(topomesh,'length',degree=1)

    edge_lengths = np.sort(topomesh.wisp_property('length',degree=1).values(topomesh.wisp_property('borders',degree=2).values()))

    triangle_perimeters = np.sum(edge_lengths,axis=1)
    if not 'perimeter' in topomesh.wisp_property_names(2):
        topomesh.add_wisp_property('perimeter',degree=2)
    topomesh.update_wisp_property('perimeter',degree=2,values=triangle_perimeters,keys=np.array(list(topomesh.wisps(2))))
    
    triangle_areas = np.sqrt((triangle_perimeters/2.0)*(triangle_perimeters/2.0-edge_lengths[:,0])*(triangle_perimeters/2.0-edge_lengths[:,1])*(triangle_perimeters/2.0-edge_lengths[:,2]))
    if not 'area' in topomesh.wisp_property_names(2):
        topomesh.add_wisp_property('area',degree=2)
    topomesh.update_wisp_property('area',degree=2,values=triangle_areas,keys=np.array(list(topomesh.wisps(2))))

    triangle_sinuses = np.zeros_like(edge_lengths,np.float32)
    triangle_sinuses[:,0] = np.sqrt(np.array(1.0 - np.power(edge_lengths[:,1]**2+edge_lengths[:,2]**2-edge_lengths[:,0]**2,2.0)/np.power(2.0*edge_lengths[:,1]*edge_lengths[:,2],2.0),np.float16))
    triangle_sinuses[:,1] = np.sqrt(np.array(1.0 - np.power(edge_lengths[:,2]**2+edge_lengths[:,0]**2-edge_lengths[:,1]**2,2.0)/np.power(2.0*edge_lengths[:,2]*edge_lengths[:,0],2.0),np.float16))
    triangle_sinuses[:,2] = np.sqrt(np.array(1.0 - np.power(edge_lengths[:,0]**2+edge_lengths[:,1]**2-edge_lengths[:,2]**2,2.0)/np.power(2.0*edge_lengths[:,0]*edge_lengths[:,1],2.0),np.float16))

    triangle_sinus_eccentricities = 1.0 - (2.0*(triangle_sinuses[:,0]+triangle_sinuses[:,1]+triangle_sinuses[:,2]))/(3*np.sqrt(3))
    if not 'eccentricity' in topomesh.wisp_property_names(2):
        topomesh.add_wisp_property('eccentricity',degree=2)

    #topomesh.update_wisp_property('eccentricity',degree=2,values=triangle_edge_ratios,keys=np.array(list(topomesh.wisps(2))))
    #topomesh.update_wisp_property('eccentricity',degree=2,values=triangle_eccentricities,keys=np.array(list(topomesh.wisps(2))))
    topomesh.update_wisp_property('eccentricity',degree=2,values=triangle_sinus_eccentricities,keys=np.array(list(topomesh.wisps(2))))

    end_time = time()
    print "<-- Computing triangle properties    [",end_time-start_time,"s]"


def compute_topomesh_vertex_property_from_faces(topomesh,property_name,weighting='area',adjacency_sigma=0.5,neighborhood=1):
    """
    """
    start_time = time()
    print "--> Computing vertex property from faces"

    assert topomesh.has_wisp_property(property_name,2,is_computed=True)
    assert weighting in ['uniform','area','angle','cotangent']
    if neighborhood > 1:
        try:
            assert weighting in ['uniform','area']
        except AssertionError:
            raise AssertionError("\""+weighting+"\" weighting can not be used with faces further than 1-ring (neighborhood > 1)")

    face_property = topomesh.wisp_property(property_name,2)

    assert not face_property.values().dtype == np.object

    if neighborhood == 1:
        vertex_faces = np.concatenate([list(topomesh.regions(0,v,2)) for v in topomesh.wisps(0)]).astype(np.uint16)
        vertex_face_vertices = np.concatenate([v*np.ones_like(list(topomesh.regions(0,v,2))) for v in topomesh.wisps(0)]).astype(np.uint16)
    else:
        vertex_face_adjacency_matrix = topomesh.nb_wisps(2)*np.ones((max(topomesh.wisps(0))+1,max(topomesh.wisps(2))+1))

        vertex_faces = np.concatenate([list(topomesh.regions(0,v,2)) for v in topomesh.wisps(0)]).astype(np.uint16)
        vertex_face_vertices = np.concatenate([v*np.ones_like(list(topomesh.regions(0,v,2))) for v in topomesh.wisps(0)]).astype(np.uint16)
    
        compute_topomesh_property(topomesh,'border_neighbors',2)

        vertex_face_adjacency_matrix[(vertex_face_vertices,vertex_faces)] = 0
        for dist in (np.arange(neighborhood)):
            neighbor_face_vertices, neighbor_faces = np.where(vertex_face_adjacency_matrix <= dist)

            neighbor_face_neighbors = np.concatenate(topomesh.wisp_property('neighbors',2).values(neighbor_faces))
            neighbor_face_neighbor_vertex = np.concatenate([v*np.ones_like(topomesh.wisp_property('neighbors',2)[f]) for f,v in zip(neighbor_faces,neighbor_face_vertices)])
            neighbor_face_neighbor_predecessor = np.concatenate([f*np.ones_like(topomesh.wisp_property('neighbors',2)[f]) for f in neighbor_faces])

            current_adjacency = vertex_face_adjacency_matrix[(neighbor_face_neighbor_vertex,neighbor_face_neighbors)]
            neighbor_adjacency = vertex_face_adjacency_matrix[(neighbor_face_neighbor_vertex,neighbor_face_neighbor_predecessor)] + 1

            current_adjacency = current_adjacency[np.argsort(-neighbor_adjacency)]
            neighbor_face_neighbors  = neighbor_face_neighbors [np.argsort(-neighbor_adjacency)]
            neighbor_face_neighbor_vertex = neighbor_face_neighbor_vertex[np.argsort(-neighbor_adjacency)]
            neighbor_adjacency = neighbor_adjacency[np.argsort(-neighbor_adjacency)]

            vertex_face_adjacency_matrix[(neighbor_face_neighbor_vertex,neighbor_face_neighbors)] = np.minimum(current_adjacency,neighbor_adjacency)

        vertex_face_vertices, vertex_faces = np.where(vertex_face_adjacency_matrix < neighborhood)
        vertex_adjacency_gaussian_weight = np.exp(-np.power(vertex_face_adjacency_matrix,2.0)/(2*np.power(adjacency_sigma,2.)))[(vertex_face_vertices, vertex_faces)]

    if weighting == 'uniform':
        vertex_face_weight = np.ones_like(vertex_faces)
    elif weighting == 'area':
        compute_topomesh_property(topomesh,'area',2)
        vertex_face_weight = topomesh.wisp_property('area',2).values(vertex_faces)
    elif weighting == 'angle':
        compute_topomesh_property(topomesh,'vertices',2)
        compute_topomesh_property(topomesh,'angles',2)
        vertex_face_face_vertices = topomesh.wisp_property('vertices',2).values(vertex_faces)
        
        vertex_face_face_angles = topomesh.wisp_property('angles',2).values(vertex_faces)
        vertex_face_vertex_angles = np.array([angles[vertices == v] for v,vertices,angles in zip(vertex_face_vertices,vertex_face_face_vertices,vertex_face_face_angles)])
        vertex_face_weight = vertex_face_vertex_angles[:,0]
    elif weighting == 'cotangent':
        compute_topomesh_property(topomesh,'vertices',2)
        compute_topomesh_property(topomesh,'angles',2)
        vertex_face_face_vertices = topomesh.wisp_property('vertices',2).values(vertex_faces)
        vertex_face_face_angles = topomesh.wisp_property('angles',2).values(vertex_faces)

        vertex_face_opposite_angles = np.array([angles[vertices != v] for v,vertices,angles in zip(vertex_face_vertices,vertex_face_face_vertices,vertex_face_face_angles)])
        vertex_face_cotangents = 1./np.tan(vertex_face_opposite_angles)
        vertex_face_weight = vertex_face_cotangents.sum(axis=1)

    if neighborhood > 1:
        vertex_face_weight = vertex_face_weight*vertex_adjacency_gaussian_weight

    vertex_face_property = face_property.values(vertex_faces)

    if vertex_face_property.ndim == 1:
        vertex_property = nd.sum(vertex_face_weight*vertex_face_property,vertex_face_vertices,index=list(topomesh.wisps(0)))/nd.sum(vertex_face_weight,vertex_face_vertices,index=list(topomesh.wisps(0)))
    elif vertex_face_property.ndim == 2:
        vertex_property = np.transpose([nd.sum(vertex_face_weight*vertex_face_property[:,k],vertex_face_vertices,index=list(topomesh.wisps(0)))/nd.sum(vertex_face_weight,vertex_face_vertices,index=list(topomesh.wisps(0))) for k in xrange(vertex_face_property.shape[1])])
    elif vertex_face_property.ndim == 3:
        vertex_property = np.transpose([[nd.sum(vertex_face_weight*vertex_face_property[:,j,k],vertex_face_vertices,index=list(topomesh.wisps(0)))/nd.sum(vertex_face_weight,vertex_face_vertices,index=list(topomesh.wisps(0))) for k in xrange(vertex_face_property.shape[2])] for j in xrange(vertex_face_property.shape[1])])

    if property_name in ['normal']:
        vertex_property_norm = np.linalg.norm(vertex_property,axis=1)
        vertex_property = vertex_property/vertex_property_norm[:,np.newaxis]
    vertex_property[np.isnan(vertex_property)] = 0

    topomesh.update_wisp_property(property_name,degree=0,values=array_dict(vertex_property,keys=list(topomesh.wisps(0))))
    
    end_time = time()
    print "<-- Computing vertex property from faces [",end_time-start_time,"s]"

def compute_topomesh_cell_property_from_faces(topomesh,property_name,weighting='area'):
    """
    """
    start_time = time()
    print "--> Computing cell property from faces"

    assert topomesh.has_wisp_property(property_name,2,is_computed=True)
    assert weighting in ['uniform','area']

    face_property = topomesh.wisp_property(property_name,2)

    cell_faces = np.concatenate([list(topomesh.borders(3,c)) for c in topomesh.wisps(3)]).astype(np.uint16)
    cell_face_cells = np.concatenate([c*np.ones(topomesh.nb_borders(3,c)) for c in topomesh.wisps(3)]).astype(np.uint16)

    if weighting == 'uniform':
        cell_face_weight = np.ones_like(cell_faces)
    elif weighting == 'area':
        compute_topomesh_property(topomesh,'area',2)
        cell_face_weight = topomesh.wisp_property('area',2).values(cell_faces)

    cell_face_property = face_property.values(cell_faces)

    if cell_face_property.ndim == 1:
        cell_property = nd.sum(cell_face_weight*cell_face_property,cell_face_cells,index=list(topomesh.wisps(3)))/nd.sum(cell_face_weight,cell_face_cells,index=list(topomesh.wisps(3)))
    elif cell_face_property.ndim == 2:
        cell_property = np.transpose([nd.sum(cell_face_weight*cell_face_property[:,k],cell_face_cells,index=list(topomesh.wisps(3)))/nd.sum(cell_face_weight,cell_face_cells,index=list(topomesh.wisps(3))) for k in xrange(cell_face_property.shape[1])])
    elif cell_face_property.ndim == 3:
        cell_property = np.transpose([[nd.sum(cell_face_weight*cell_face_property[:,j,k],cell_face_cells,index=list(topomesh.wisps(3)))/nd.sum(cell_face_weight,cell_face_cells,index=list(topomesh.wisps(3))) for k in xrange(cell_face_property.shape[2])] for j in xrange(cell_face_property.shape[1])])

    topomesh.update_wisp_property(property_name,degree=3,values=array_dict(cell_property,keys=list(topomesh.wisps(3))))
    
    end_time = time()
    print "<-- Computing cell property from faces [",end_time-start_time,"s]"


def filter_topomesh_property(topomesh,property_name,degree,coef=0.5,normalize=False):
    assert topomesh.has_wisp_property(property_name,degree=degree,is_computed=True)
    if degree>1:
        neighbors = np.array([[(w1,w2) for w2 in topomesh.border_neighbors(degree,w1) if not np.isnan(topomesh.wisp_property(property_name,degree=degree)[w2])] for w1 in topomesh.wisps(degree) if not np.isnan(topomesh.wisp_property(property_name,degree=degree)[w1])])
        neighbors = np.concatenate(neighbors[np.where(np.array(map(len,neighbors)) > 0)])
        # neighbors = topomesh.wisp_property('borders',degree-1).values()
    else:
        neighbors = np.array([[(w1,w2) for w2 in topomesh.region_neighbors(degree,w1) if np.isnan(topomesh.wisp_property(property_name,degree=degree)[w2]).sum() == 0] for w1 in topomesh.wisps(degree) if np.isnan(topomesh.wisp_property(property_name,degree=degree)[w1]).sum()==0])
        neighbors = np.concatenate(neighbors[np.where(np.array(map(len,neighbors)) > 0)])
        # neighbors = topomesh.wisp_property('borders',degree-1).values()
   
    wisp_neighbor_degree = nd.sum(np.ones_like(neighbors[:,0]),neighbors[:,0],index=np.array(list(topomesh.wisps(degree))))

    neighbor_properties = topomesh.wisp_property(property_name,degree=degree).values(neighbors[:,1])
    if neighbor_properties.ndim == 1:
        wisp_neighbor_property = nd.sum(neighbor_properties,neighbors[:,0],index=np.array(list(topomesh.wisps(degree))))
        wisp_neighbor_property[np.where(wisp_neighbor_degree>0)] = wisp_neighbor_property[np.where(wisp_neighbor_degree>0)]/wisp_neighbor_degree[np.where(wisp_neighbor_degree>0)]
    else:
        wisp_neighbor_property = np.transpose([nd.sum(neighbor_properties[:,d],neighbors[:,0],index=np.array(list(topomesh.wisps(degree)))) for d in xrange(neighbor_properties.shape[1])])
        wisp_neighbor_property[np.where(wisp_neighbor_degree>0)] = wisp_neighbor_property[np.where(wisp_neighbor_degree>0)]/wisp_neighbor_degree[np.where(wisp_neighbor_degree>0)][:,np.newaxis]

    filtered_property = (1.-coef)*topomesh.wisp_property(property_name,degree=degree).values(np.array(list(topomesh.wisps(degree)))) + coef*wisp_neighbor_property

    if neighbor_properties.ndim > 1 and normalize:
        filetered_property_norms = np.linalg.norm(filtered_property,axis=1)
        filtered_property = filtered_property/filetered_property_norms[:,np.newaxis]

    topomesh.update_wisp_property(property_name,degree=degree,values=filtered_property,keys=np.array(list(topomesh.wisps(degree))))


def topomesh_property_gaussian_filtering(topomesh,property_name,degree,adjacency_sigma=1.0,distance_sigma=1.0,neighborhood=3):
    assert topomesh.has_wisp_property(property_name,degree=degree,is_computed=True)

    if not topomesh.has_wisp_property('barycenter',degree=degree,is_computed=True):
        compute_topomesh_property(topomesh,'barycenter',degree=degree)

    adjacency_matrix = topomesh.nb_wisps(degree)*np.ones((topomesh.nb_wisps(degree),topomesh.nb_wisps(degree)),int)
    adjacency_matrix[tuple([np.arange(topomesh.nb_wisps(degree)),np.arange(topomesh.nb_wisps(degree))])] = 0

    wisp_index = array_dict(np.arange(topomesh.nb_wisps(degree)),list(topomesh.wisps(degree)))
    for i,wid in enumerate(topomesh.wisps(degree)):
        if degree>1:
            adjacency_matrix[tuple([i*np.ones(topomesh.nb_border_neighbors(degree,wid),int),wisp_index.values(list(topomesh.border_neighbors(degree,wid)))])] = 1
        else:
            adjacency_matrix[tuple([i*np.ones(topomesh.nb_region_neighbors(degree,wid),int),wisp_index.values(list(topomesh.region_neighbors(degree,wid)))])] = 1
       
    for dist in (np.arange(neighborhood)+1):
        for i,wid in enumerate(topomesh.wisps(degree)):
            neighbors = np.where(adjacency_matrix[i] <= dist)[0]
            neighbor_neighbors = np.where(adjacency_matrix[neighbors] <= dist)
            neighbor_neighbor_successor = neighbor_neighbors[1]
            neighbor_neighbor_predecessor = neighbors[neighbor_neighbors[0]]
            
            current_adjacency = adjacency_matrix[i][neighbor_neighbor_successor]
            neighbor_adjacency = adjacency_matrix[i][neighbor_neighbor_predecessor] + adjacency_matrix[neighbors][neighbor_neighbors]
            
            current_adjacency = current_adjacency[np.argsort(-neighbor_adjacency)]
            neighbor_neighbor_successor = neighbor_neighbor_successor[np.argsort(-neighbor_adjacency)]
            neighbor_adjacency = neighbor_adjacency[np.argsort(-neighbor_adjacency)]
            
            adjacency_matrix[i][neighbor_neighbor_successor] = np.minimum(current_adjacency,neighbor_adjacency)

    distance_matrix = np.array([vq(topomesh.wisp_property('barycenter',degree).values(),np.array([topomesh.wisp_property('barycenter',degree)[v]]))[1] for v in topomesh.wisps(degree)])

    vertex_adjacency_gaussian = np.exp(-np.power(adjacency_matrix,2.0)/(2*np.power(adjacency_sigma,2.)))
    vertex_distance_gaussian = np.exp(-np.power(distance_matrix,2.0)/(2*np.power(distance_sigma,2.)))
    vertex_gaussian = vertex_adjacency_gaussian*vertex_distance_gaussian

    properties = topomesh.wisp_property(property_name,degree=degree).values(list(topomesh.wisps(degree)))
    if properties.ndim == 1:
        filtered_properties = (vertex_gaussian*properties[:,np.newaxis]).sum(axis=0) / vertex_gaussian.sum(axis=0)
    else:
        filtered_properties = (vertex_gaussian[...,np.newaxis]*properties[:,np.newaxis]).sum(axis=0) / vertex_gaussian.sum(axis=0)[...,np.newaxis]

    topomesh.update_wisp_property(property_name,degree=degree,values=filtered_properties,keys=np.array(list(topomesh.wisps(degree))))

    




