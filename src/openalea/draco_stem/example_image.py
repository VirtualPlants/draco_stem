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
from scipy.cluster.vq import vq

from openalea.container import array_dict

from openalea.image.spatial_image import SpatialImage

def sphere_tissue_image(size=100, n_points=12):

    center = np.array([size/2,size/2,size/2],float)
    radius = size/4.

    points = {}
    for p in range(n_points):
        theta = np.random.rand()*2.*np.pi
        phi = np.random.rand()*np.pi - np.pi/2.
        
        points[p+3] = center + radius*np.array([np.cos(theta)*np.cos(phi),np.sin(theta)*np.cos(phi),np.sin(phi)])
    points = array_dict(points)

    point_target_area = 4.*np.pi*np.power(radius,2.)/float(n_points)
    point_target_distance = np.power(point_target_area/np.pi,0.5)

    sigma_deformation = (size/100.)*(20./n_points)
    omega_forces = dict(distance=0.1*size/100., repulsion=100.0*np.power(size/100.,2))

    for iterations in xrange(100):
        point_vectors = np.array([points[p]- points.values() for p in points.keys()])
        point_distances = np.array([vq(points.values(),np.array([points[p]]))[1] for p in points.keys()])
        point_vectors = point_vectors/(point_distances[...,np.newaxis]+1e-7)

        point_distance_forces = omega_forces['distance']*((point_distances-point_target_distance)[...,np.newaxis]*point_vectors/point_target_distance).sum(axis=1)
        
        point_repulsion_forces = omega_forces['repulsion']*np.power(point_target_distance,2)*(point_vectors/(np.power(point_distances,2)+1e-7)[...,np.newaxis]).sum(axis=1)
        
        point_forces = np.zeros((len(points),3))
        point_forces += point_distance_forces
        point_forces += point_repulsion_forces
        
        point_forces = np.minimum(1.0,sigma_deformation/np.linalg.norm(point_forces,axis=1))[:,np.newaxis] * point_forces
        
        new_points = points.values() + point_forces
        
        new_points = center+ radius*((new_points-center)/np.linalg.norm((new_points-center),axis=1)[:,np.newaxis])
        
        points = array_dict(new_points,points.keys())
    points[2] = center

    coords = np.transpose(np.mgrid[0:size,0:size,0:size],(1,2,3,0)).reshape((np.power(size,3),3)).astype(int)
    labels = points.keys()[vq(coords,points.values())[0]]

    ext_coords = coords[vq(coords,np.array([center]))[1]>size/3.]

    img = np.ones((size,size,size),np.uint8)
    img[tuple(np.transpose(coords))] = labels
    img[tuple(np.transpose(ext_coords))] = 1
    img = SpatialImage(img,voxelsize=(60./size,60./size,60./size))

    return img


def cube_image(size=50):
    img = np.ones((size,size,size),np.uint8)

    points = {}
    points[11] = np.array([1,0,0],float)*size
    points[12] = np.array([0,1,0],float)*size
    points[31] = np.array([0,0,1],float)*size
    points[59] = np.array([1,1,1],float)*size
    points = array_dict(points)

    center = np.array([[size/2,size/2,size/2]],float)

    coords = np.transpose(np.mgrid[0:size,0:size,0:size],(1,2,3,0)).reshape((np.power(size,3),3))
    labels = points.keys()[vq(coords,points.values())[0]]

    ext_coords = coords[vq(coords,center)[1]>size/2.]

    img[tuple(np.transpose(coords))] = labels
    #img[tuple(np.transpose(ext_coords))] = 1
    img = SpatialImage(img,voxelsize=(1,1,1))

    return img

