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
from scipy.cluster.vq import vq

from openalea.container import array_dict, PropertyTopomesh
from openalea.mesh.utils.array_tools import array_unique

from time import time

tetra_triangle_edge_list  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
tetra_triangle_list  = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])

def tetrahedra_topomesh(tetrahedra, positions, **kwargs):

    tetrahedra = np.array(tetrahedra)
    positions = array_dict(positions)

    tetrahedra_triangles = array_unique(np.concatenate(np.sort(tetrahedra[:,tetra_triangle_list])))

    tetrahedra_triangle_edges = tetrahedra_triangles[:,triangle_edge_list]
    tetrahedra_triangle_vectors = positions.values(tetrahedra_triangle_edges[...,1]) - positions.values(tetrahedra_triangle_edges[...,0])
    tetrahedra_triangle_lengths = np.linalg.norm(tetrahedra_triangle_vectors,axis=2)
    tetrahedra_triangle_perimeters = tetrahedra_triangle_lengths.sum(axis=1)

    tetrahedra_edges = array_unique(np.concatenate(tetrahedra_triangles[:,triangle_edge_list],axis=0))

    start_time = time()
    print "--> Generating tetrahedra topomesh"
    triangle_edges = np.concatenate(tetrahedra_triangles[:,triangle_edge_list],axis=0)
    triangle_edge_matching = vq(triangle_edges,tetrahedra_edges)[0]

    tetrahedra_faces = np.concatenate(np.sort(tetrahedra[:,tetra_triangle_list]))
    tetrahedra_triangle_matching = vq(tetrahedra_faces,tetrahedra_triangles)[0]

    tetrahedra_topomesh = PropertyTopomesh(3)
    for c in np.unique(tetrahedra_triangles):
        tetrahedra_topomesh.add_wisp(0,c)
    for e in tetrahedra_edges:
        eid = tetrahedra_topomesh.add_wisp(1)
        for pid in e:
            tetrahedra_topomesh.link(1,eid,pid)
    for t in tetrahedra_triangles:
        fid = tetrahedra_topomesh.add_wisp(2)
        for eid in triangle_edge_matching[3*fid:3*fid+3]:
            tetrahedra_topomesh.link(2,fid,eid)
    for t in tetrahedra:
        cid = tetrahedra_topomesh.add_wisp(3)
        for fid in tetrahedra_triangle_matching[4*cid:4*cid+4]:
            tetrahedra_topomesh.link(3,cid,fid)
    tetrahedra_topomesh.update_wisp_property('barycenter',0,positions.values(np.unique(tetrahedra_triangles)),keys=np.unique(tetrahedra_triangles))        

    end_time = time()
    print "<-- Generating tetrahedra topomesh [",end_time-start_time,"s]"

    return tetrahedra_topomesh

def triangle_topomesh(triangles, positions, **kwargs):

    triangles = np.array(triangles)
    positions = array_dict(positions)

    edges = array_unique(np.concatenate(triangles[:,triangle_edge_list],axis=0))

    triangle_edges = np.concatenate(triangles[:,triangle_edge_list])

    start_time = time()
    print "--> Generating triangle topomesh"

    triangle_edge_matching = vq(triangle_edges,edges)[0]

    triangle_topomesh = PropertyTopomesh(3)
    for c in np.unique(triangles):
        triangle_topomesh.add_wisp(0,c)
    for e in edges:
        eid = triangle_topomesh.add_wisp(1)
        for pid in e:
            triangle_topomesh.link(1,eid,pid)
    for t in triangles:
        fid = triangle_topomesh.add_wisp(2)
        for eid in triangle_edge_matching[3*fid:3*fid+3]:
            triangle_topomesh.link(2,fid,eid)
    triangle_topomesh.add_wisp(3,0)
    for fid in triangle_topomesh.wisps(2):
        triangle_topomesh.link(3,0,fid)
    triangle_topomesh.update_wisp_property('barycenter',0,positions.values(np.unique(triangles)),keys=np.unique(triangles))    

    end_time = time()
    print "<-- Generating triangle topomesh [",end_time-start_time,"s]"

    return triangle_topomesh

def edge_topomesh(edges, positions, **kwargs):

    positions = array_dict(positions)

    start_time = time()
    print "--> Generating edge topomesh"

    edge_topomesh = PropertyTopomesh(3)
    for c in np.unique(edges):
        edge_topomesh.add_wisp(0,c)
    for e in edges:
        eid = edge_topomesh.add_wisp(1)
        for pid in e:
            edge_topomesh.link(1,eid,pid)
    edge_topomesh.update_wisp_property('barycenter',0,positions.values(np.unique(edges)),keys=np.unique(edges))    

    end_time = time()
    print "<-- Generating edge topomesh [",end_time-start_time,"s]"

    return edge_topomesh

def vertex_topomesh(positions, **kwargs):
    
    positions = array_dict(positions)

    start_time = time()
    print "--> Generating vertex topomesh"

    vertex_topomesh = PropertyTopomesh(3)
    for c in positions.keys():
        vertex_topomesh.add_wisp(0,c)
    vertex_topomesh.update_wisp_property('barycenter',0,positions.values(positions.keys()),keys=positions.keys())    

    end_time = time()
    print "<-- Generating vertex topomesh [",end_time-start_time,"s]"

    return vertex_topomesh