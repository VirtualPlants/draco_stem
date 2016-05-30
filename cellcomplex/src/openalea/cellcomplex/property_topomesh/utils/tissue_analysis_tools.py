# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
#
#       Copyright 2015-2016 INRIA - CIRAD - INRA
#
#       File author(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       File contributor(s): Guillaume Cerutti <guillaume.cerutti@inria.fr>,
#                            Jonathan Legrand <jonathan.legrand@inria.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenaleaLab Website : http://virtualplants.github.io/
#
###############################################################################

import numpy as np

def cell_vertex_extraction(img,**kwargs):
    from vplants.meshing.array_tools import array_unique
    from scipy.cluster.vq import vq
    
    shape = np.array(img.shape)
    neighborhood_img = []
    for x in np.arange(-1,1):
        for y in np.arange(-1,1):
            for z in np.arange(-1,1):
                neighborhood_img.append(img[1+x:shape[0]+x,1+y:shape[1]+y,1+z:shape[2]+z])
    neighborhood_img = np.sort(np.transpose(neighborhood_img,(1,2,3,0))).reshape((shape-1).prod(),8)
    neighborhoods = np.array(map(np.unique,neighborhood_img))
    neighborhood_size = np.array(map(len,neighborhoods)).reshape(shape[0]-1,shape[1]-1,shape[2]-1)
    neighborhoods = np.array(neighborhoods).reshape(shape[0]-1,shape[1]-1,shape[2]-1)

    vertex_coords = np.where(neighborhood_size==4)
    #vertex_coords = np.where(neighborhood_size >= 4)
    vertex_points = np.transpose(vertex_coords)+0.5
    vertex_cells = np.array([p for p in neighborhoods[vertex_coords]],int)
    #vertex_cells = np.array([p[:4] for p in neighborhoods[vertex_coords]],int)

    if (neighborhood_size>5).sum() > 0:
        clique_vertex_coords = np.where(neighborhood_size==5)
        clique_vertex_points = np.concatenate([(p,p) for p in np.transpose(clique_vertex_coords)])+0.5
        clique_vertex_cells = np.concatenate([[p[:4],np.concatenate([[p[0]],p[2:]])] for p in neighborhoods[clique_vertex_coords]]).astype(int)

        vertex_points = np.concatenate([vertex_points,clique_vertex_points])
        vertex_cells = np.concatenate([vertex_cells,clique_vertex_cells])

    unique_cell_vertices = array_unique(vertex_cells)
    vertices_matching = vq(vertex_cells,unique_cell_vertices)[0]
    unique_cell_vertex_points = np.array([np.mean(vertex_points[vertices_matching == v],axis=0) for v in xrange(len(unique_cell_vertices))])
    
    cell_vertex_dict = dict(zip([tuple(c) for c in unique_cell_vertices],list(unique_cell_vertex_points)))
    
    return cell_vertex_dict


def voxel_cell_vertex_extraction(img,**kwargs):
    from vplants.meshing.array_tools import array_unique
    from scipy.cluster.vq import vq
    
    shape = np.array(img.shape)
    neighborhood_img = []
    for x in np.arange(-1,2):
        for y in np.arange(-1,2):
            for z in np.arange(-1,2):
                neighborhood_img.append(img[1+x:shape[0]-1+x,1+y:shape[1]-1+y,1+z:shape[2]-1+z])
    neighborhood_img = np.sort(np.transpose(neighborhood_img,(1,2,3,0))).reshape((shape-2).prod(),27)
    neighborhoods = np.array(map(np.unique,neighborhood_img))
    neighborhood_size = np.array(map(len,neighborhoods)).reshape(shape[0]-2,shape[1]-2,shape[2]-2)
    neighborhoods = np.array(neighborhoods).reshape(shape[0]-2,shape[1]-2,shape[2]-2)
    
    vertex_coords = np.where(neighborhood_size==4)
    vertex_points = np.transpose(vertex_coords)+1
    vertex_cells = np.array([p for p in neighborhoods[vertex_coords]],int)

    unique_cell_vertices = array_unique(vertex_cells)
    vertices_matching = vq(vertex_cells,unique_cell_vertices)[0]
    unique_cell_vertex_points = np.array([np.mean(vertex_points[vertices_matching == v],axis=0) for v in xrange(len(unique_cell_vertices))])
    
    cell_vertex_dict = dict(zip([tuple(c) for c in unique_cell_vertices],list(unique_cell_vertex_points)))
    
    return cell_vertex_dict