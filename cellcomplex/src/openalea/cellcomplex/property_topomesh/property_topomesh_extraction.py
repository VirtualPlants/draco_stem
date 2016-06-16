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

from copy import deepcopy

from openalea.container import array_dict

from openalea.mesh import PropertyTopomesh
from openalea.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property


def epidermis_topomesh(topomesh,cells=None):
    
    compute_topomesh_property(topomesh,'epidermis',3)
    compute_topomesh_property(topomesh,'epidermis',1)
    compute_topomesh_property(topomesh,'epidermis',0)
    
    if cells is None:
        epidermis_topomesh = deepcopy(topomesh)
    else:
        faces = np.array(np.unique(np.concatenate([np.array(list(topomesh.borders(3,c))) for c in cells])),int)
        edges = np.array(np.unique(np.concatenate([np.array(list(topomesh.borders(2,t))) for t in faces])),int)
        vertices = np.array(np.unique(np.concatenate([np.array(list(topomesh.borders(1,e))) for e in edges])),int)
        epidermis_topomesh = PropertyTopomesh(3)
        vertices_to_pids = {}
        for v in vertices:
            pid = epidermis_topomesh.add_wisp(0,v)
            vertices_to_pids[v] = pid
        edges_to_eids = {}
        for e in edges:
            eid = epidermis_topomesh.add_wisp(1,e)
            edges_to_eids[e] = eid
            for v in topomesh.borders(1,e):
                epidermis_topomesh.link(1,eid,vertices_to_pids[v])
        faces_to_fids = {}
        for f in faces:
            fid = epidermis_topomesh.add_wisp(2,f)
            faces_to_fids[f] = fid
            for e in topomesh.borders(2,f):
                epidermis_topomesh.link(2,fid,edges_to_eids[e])
        for c in cells:
            cid = epidermis_topomesh.add_wisp(3,c)
            for f in topomesh.borders(3,c):
                epidermis_topomesh.link(3,cid,faces_to_fids[f])
    
    vertices_to_remove = []
    for v in epidermis_topomesh.wisps(0):
        if not topomesh.wisp_property('epidermis',0)[v]:
            vertices_to_remove.append(v)
    for v in vertices_to_remove:
        epidermis_topomesh.remove_wisp(0,v)
    edges_to_remove = []
    for e in epidermis_topomesh.wisps(1):
        if not topomesh.wisp_property('epidermis',1)[e]:
            edges_to_remove.append(e)
    for e in edges_to_remove:
        epidermis_topomesh.remove_wisp(1,e)
    faces_to_remove = []
    for f in epidermis_topomesh.wisps(2):
        if not topomesh.wisp_property('epidermis',2)[f]:
            faces_to_remove.append(f)
    for f in faces_to_remove:
        epidermis_topomesh.remove_wisp(2,f)
    cells_to_remove = []
    for c in epidermis_topomesh.wisps(3):
        if not topomesh.wisp_property('epidermis',3)[c]:
            cells_to_remove.append(c)
    for c in cells_to_remove:
        epidermis_topomesh.remove_wisp(3,c)
    epidermis_topomesh.update_wisp_property('barycenter',0,topomesh.wisp_property('barycenter',0).values(list(epidermis_topomesh.wisps(0))),keys=np.array(list(epidermis_topomesh.wisps(0))))
    return epidermis_topomesh


def cut_surface_topomesh(input_topomesh, z_cut=0, below=True):

    topomesh = deepcopy(input_topomesh)    

    compute_topomesh_property(topomesh,'vertices',2)

    if below:
        triangle_below = array_dict(np.all(topomesh.wisp_property('barycenter',0).values(topomesh.wisp_property('vertices',2).values())[...,2] < z_cut,axis=1),list(topomesh.wisps(2)))
    else:
        triangle_below = array_dict(np.all(topomesh.wisp_property('barycenter',0).values(topomesh.wisp_property('vertices',2).values())[...,2] > z_cut,axis=1),list(topomesh.wisps(2)))
    topomesh.update_wisp_property('below',2,triangle_below)

    triangles_to_remove = [t for t in topomesh.wisps(2) if triangle_below[t]]
    for t in triangles_to_remove:
        topomesh.remove_wisp(2,t)

    topomesh = clean_topomesh(topomesh)

    compute_topomesh_property(topomesh,'triangles',1)
    compute_topomesh_property(topomesh,'vertices',1)

    topomesh.update_wisp_property('boundary',1,array_dict((np.array(map(len,topomesh.wisp_property('triangles',1).values()))==1).astype(int),list(topomesh.wisps(1))))

    boundary_edges = np.array(list(topomesh.wisps(1)))[topomesh.wisp_property('boundary',1).values()==1]
    boundary_vertices = np.unique(topomesh.wisp_property('vertices',1).values(boundary_edges))

    iso_z_positions = np.array([np.concatenate([topomesh.wisp_property('barycenter',0)[v][:2],[z_cut+(1-2*below)*2]]) if v in boundary_vertices else  topomesh.wisp_property('barycenter',0)[v] for v in topomesh.wisps(0)])
    topomesh.update_wisp_property('barycenter',0,array_dict(iso_z_positions,list(topomesh.wisps(0))))

    return topomesh


def clean_topomesh(input_topomesh):

    topomesh = deepcopy(input_topomesh)

    cells_to_remove = [w for w in topomesh.wisps(3) if topomesh.nb_borders(3,w)==0]
    for w in cells_to_remove:
        topomesh.remove_wisp(3,w)

    triangles_to_remove = [w for w in topomesh.wisps(2) if topomesh.nb_regions(2,w)==0]
    for w in triangles_to_remove:
        topomesh.remove_wisp(2,w)

    edges_to_remove = [w for w in topomesh.wisps(1) if topomesh.nb_regions(1,w)==0]
    for w in edges_to_remove:
        topomesh.remove_wisp(1,w)
        
    vertices_to_remove = [w for w in topomesh.wisps(0) if topomesh.nb_regions(0,w)==0]
    for w in vertices_to_remove:
        topomesh.remove_wisp(0,w)

    return topomesh
    

def cell_topomesh(input_topomesh, cells=None):
    start_time = time()

    #topomesh = PropertyTopomesh(topomesh=input_topomesh)
    topomesh = PropertyTopomesh(3)

    if cells is None:
        cells = set(topomesh.wisps(3))
    else:
        cells = set(cells)

    faces = set()
    for c in cells:
        topomesh._borders[3][c] = input_topomesh._borders[3][c]
        faces = faces.union(set(topomesh._borders[3][c]))

    edges = set()
    for f in faces:
        topomesh._borders[2][f] = input_topomesh._borders[2][f]
        topomesh._regions[2][f] = np.array(list(set(input_topomesh._regions[2][f]).intersection(cells)))
        edges = edges.union(set(topomesh._borders[2][f]))

    vertices = set()
    for e in edges:
        topomesh._borders[1][e] = input_topomesh._borders[1][e]
        topomesh._regions[1][e] = np.array(list(set(input_topomesh._regions[1][e]).intersection(faces)))
        vertices = vertices.union(set(topomesh._borders[1][e]))

    for v in vertices:
        topomesh._regions[0][v] = np.array(list(set(input_topomesh._regions[0][v]).intersection(edges)))

    topomesh.update_wisp_property('barycenter',0,array_dict(input_topomesh.wisp_property('barycenter',0).values(list(vertices)),list(vertices)))


    # cells_to_remove = [c for c in topomesh.wisps(3) if not c in cells]
    # cells_to_remove = list(set(topomesh.wisps(3)).difference(set(cells)))


    # for c in cells_to_remove:
    #     topomesh.remove_wisp(3,c)

    # faces_to_remove = [w for w in topomesh.wisps(2) if topomesh.nb_regions(2,w)==0]
    # for w in faces_to_remove:
    #     topomesh.remove_wisp(2,w)

    # edges_to_remove = [w for w in topomesh.wisps(1) if topomesh.nb_regions(1,w)==0]
    # for w in edges_to_remove:
    #     topomesh.remove_wisp(1,w)
        
    # vertices_to_remove = [w for w in topomesh.wisps(0) if topomesh.nb_regions(0,w)==0]
    # for w in vertices_to_remove:
    #     topomesh.remove_wisp(0,w)

    end_time = time()
    #end_time = time()
    print "<-- Extracting cell topomesh     [",end_time-start_time,"s]"

    return topomesh

