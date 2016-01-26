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


def inside_triangle(point,triangles):
    v0 = triangles[:,2]-triangles[:,0]
    v1 = triangles[:,1]-triangles[:,0]
    v2 = point-triangles[:,0]

    dot00 = np.einsum('ij,ij->i',v0,v0)
    dot01 = np.einsum('ij,ij->i',v0,v1)
    dot02 = np.einsum('ij,ij->i',v0,v2)
    dot11 = np.einsum('ij,ij->i',v1,v1)
    dot12 = np.einsum('ij,ij->i',v1,v2)
    
    invDenom = 1./(dot00 * dot11-dot01*dot01)
    u = np.float16((dot11 * dot02 - dot01 * dot12)*invDenom)
    v = np.float16((dot00 * dot12 - dot01 * dot02)*invDenom)

    return (u>=0) & (v>=0) & (u+v<=1)

def intersecting_segment(segment,line_segments):
    if line_segments.ndim > 2:
        det_seg0 = np.array(np.linalg.det(np.transpose([segment[0]-line_segments[:,0],line_segments[:,1]-line_segments[:,0]],(1,2,0))),np.float16)
        det_seg1 = np.array(np.linalg.det(np.transpose([segment[1]-line_segments[:,0],line_segments[:,1]-line_segments[:,0]],(1,2,0))),np.float16)
        det_lin0 = np.array(np.linalg.det(np.transpose([line_segments[:,0]-segment[0],np.tile(segment[1]-segment[0],(line_segments.shape[0],1))],(1,2,0))),np.float16)
        det_lin1 = np.array(np.linalg.det(np.transpose([line_segments[:,1]-segment[0],np.tile(segment[1]-segment[0],(line_segments.shape[0],1))],(1,2,0))),np.float16)
    else:
        det_seg0 = np.array(np.linalg.det(np.transpose([segment[0]-line_segments[0],line_segments[1]-line_segments[0]])),np.float16)
        det_seg1 = np.array(np.linalg.det(np.transpose([segment[1]-line_segments[0],line_segments[1]-line_segments[0]])),np.float16)
        det_lin0 = np.array(np.linalg.det(np.transpose([line_segments[0]-segment[0],segment[1]-segment[0]])),np.float16)
        det_lin1 = np.array(np.linalg.det(np.transpose([line_segments[1]-segment[0],segment[1]-segment[0]])),np.float16)

    return ((det_seg0*det_seg1 < 0) & (det_lin0*det_lin1 < 0))


def intersecting_triangle(segment,triangles):
    if triangles.ndim <= 2:
        triangle_edge_1 = triangles[1] - triangles[0]
        triangle_edge_2 = triangles[2] - triangles[0]
        edge_rays_t = segment[0] - triangles[0]
        edge_rays_d = segment[1] - segment[0]

        triangle_p = np.cross(edge_rays_d,triangle_edge_2[np.newaxis,:])
        triangle_q = np.cross(edge_rays_t,triangle_edge_1[np.newaxis,:])

        triangle_norm = np.einsum('...ij,...ij->...i',triangle_p,triangle_edge_1[np.newaxis,:])
            
        triangle_distance = np.array(np.einsum('...ij,...ij->...i',triangle_q,triangle_edge_2[np.newaxis,:])/triangle_norm,np.float16)
        triangle_projection_u = np.array(np.einsum('...ij,...ij->...i',triangle_p,edge_rays_t[np.newaxis,:])/triangle_norm,np.float16)
        triangle_projection_v = np.array(np.einsum('...ij,...ij->...i',triangle_q,edge_rays_d[np.newaxis,:])/triangle_norm,np.float16)

        edge_triangle_intersection = (triangle_distance>0)&(triangle_distance<1)&(triangle_projection_u>0)&(triangle_projection_v>0)&(triangle_projection_u+triangle_projection_v<1)
    else:
        triangle_edge_1 = triangles[:,1] - triangles[:,0]
        triangle_edge_2 = triangles[:,2] - triangles[:,0]
        edge_rays_t = segment[0] - triangles[:,0]
        edge_rays_d = np.tile(segment[1] - segment[0],(triangles.shape[0],1))

        triangle_p = np.cross(edge_rays_d,triangle_edge_2)
        triangle_q = np.cross(edge_rays_t,triangle_edge_1)

        triangle_norm = np.einsum('...ij,...ij->...i',triangle_p,triangle_edge_1)
            
        triangle_distance = np.array(np.einsum('...ij,...ij->...i',triangle_q,triangle_edge_2)/triangle_norm,np.float16)
        triangle_projection_u = np.array(np.einsum('...ij,...ij->...i',triangle_p,edge_rays_t[np.newaxis,:])/triangle_norm,np.float16)
        triangle_projection_v = np.array(np.einsum('...ij,...ij->...i',triangle_q,edge_rays_d[np.newaxis,:])/triangle_norm,np.float16)

        edge_triangle_intersection = (triangle_distance>0)&(triangle_distance<1)&(triangle_projection_u>0)&(triangle_projection_v>0)&(triangle_projection_u+triangle_projection_v<1)

    return edge_triangle_intersection

