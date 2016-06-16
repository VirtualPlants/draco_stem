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

def tetra_geometric_features(tetrahedra,positions,features=['volume','max_distance']):
    tetra_positions = positions.values(tetrahedra)

    tetra_features = {}

    tetra_edge_list  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    tetra_triangle_list  = np.array([[0,1,2],[0,1,3],[0,2,3],[1,2,3]])
    triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])

    tetra_triangles = tetrahedra[:,tetra_triangle_list ]
    tetra_edges = tetra_triangles[...,triangle_edge_list]
    tetra_edge_lengths = np.linalg.norm(positions.values(tetra_edges[...,1]) - positions.values(tetra_edges[...,0]),axis=3)
    
    tetra_sorted_edge_lengths = np.sort(np.linalg.norm(positions.values(tetrahedra[:,tetra_edge_list][...,1]) - positions.values(tetrahedra[:,tetra_edge_list][...,0]),axis=2))
    tetra_features['max_distance'] = tetra_sorted_edge_lengths[:,-1]
    tetra_features['min_distance'] = tetra_sorted_edge_lengths[:,0]

    if 'volume' in features or 'eccentricity' in features:
        tetra_volumes = np.abs(np.sum((tetra_positions[:,0]-tetra_positions[:,3])*np.cross(tetra_positions[:,1]-tetra_positions[:,3],tetra_positions[:,2]-tetra_positions[:,3]),axis=1))/6.0
        tetra_features['volume'] = tetra_volumes

    if 'area' in features or 'eccentricity' in features:
        tetra_triangle_perimeters = tetra_edge_lengths.sum(axis=2)
        tetra_triangle_areas = np.sqrt((tetra_triangle_perimeters/2.0)*(tetra_triangle_perimeters/2.0-tetra_edge_lengths[...,0])*(tetra_triangle_perimeters/2.0-tetra_edge_lengths[...,1])*(tetra_triangle_perimeters/2.0-tetra_edge_lengths[...,2]))
        tetra_areas = np.sum(tetra_triangle_areas,axis=1)

        tetra_triangle_eccentricities = 1. - (12.0*np.sqrt(3)*tetra_triangle_areas)/np.power(tetra_triangle_perimeters,2.0)
        tetra_max_eccentricities = tetra_triangle_eccentricities.max(axis=1)
        tetra_mean_eccentricities = tetra_triangle_eccentricities.mean(axis=1)

        tetra_triangle_sinuses = np.zeros_like(tetra_edge_lengths,np.float32)
        tetra_triangle_sinuses[...,0] = np.sqrt(np.array(1.0 - np.power(tetra_edge_lengths[...,1]**2+tetra_edge_lengths[...,2]**2-tetra_edge_lengths[...,0]**2,2.0)/np.power(2.0*tetra_edge_lengths[...,1]*tetra_edge_lengths[...,2],2.0),np.float16))
        tetra_triangle_sinuses[...,1] = np.sqrt(np.array(1.0 - np.power(tetra_edge_lengths[...,2]**2+tetra_edge_lengths[...,0]**2-tetra_edge_lengths[...,1]**2,2.0)/np.power(2.0*tetra_edge_lengths[...,2]*tetra_edge_lengths[...,0],2.0),np.float16))
        tetra_triangle_sinuses[...,2] = np.sqrt(np.array(1.0 - np.power(tetra_edge_lengths[...,0]**2+tetra_edge_lengths[...,1]**2-tetra_edge_lengths[...,2]**2,2.0)/np.power(2.0*tetra_edge_lengths[...,0]*tetra_edge_lengths[...,1],2.0),np.float16))
            # triangle_sinuses[...,1] = np.sqrt(np.array(1.0 - np.power(edge_lengths[:,2]**2+edge_lengths[:,0]**2-edge_lengths[:,1]**2,2.0)/np.power(2.0*edge_lengths[:,2]*edge_lengths[:,0],2.0),np.float16))
            # triangle_sinuses[:,2] = np.sqrt(np.array(1.0 - np.power(edge_lengths[:,0]**2+edge_lengths[:,1]**2-edge_lengths[:,2]**2,2.0)/np.power(2.0*edge_lengths[:,0]*edge_lengths[:,1],2.0),np.float16))

        tetra_triangle_sinus_eccentricities = 1.0 - (2.0*(tetra_triangle_sinuses[...,0]+tetra_triangle_sinuses[...,1]+tetra_triangle_sinuses[...,2]))/(3*np.sqrt(3))
        tetra_max_sinus_eccentricities = tetra_triangle_sinus_eccentricities.max(axis=1)
        tetra_mean_sinus_eccentricities = tetra_triangle_sinus_eccentricities.mean(axis=1)

        tetra_inscribed_sphere_radius = 3.*tetra_volumes/tetra_areas
        tetra_features['inscribed_sphere_radius'] = tetra_inscribed_sphere_radius

        tetra_eccentricities = 1.0 - 216.*np.sqrt(3.)*np.power(tetra_volumes,2.0)/np.power(tetra_areas,3.0)
        tetra_features['eccentricity'] = tetra_eccentricities

        tetra_edges = tetrahedra[:,tetra_edge_list]

        tetra_sorted_edge_lengths = np.sort(np.linalg.norm(positions.values(tetra_edges[...,1]) - positions.values(tetra_edges[...,0]),axis=2))
        tetra_max_distances = tetra_sorted_edge_lengths[:,-1]
        tetra_min_distances = tetra_sorted_edge_lengths[:,0]

        tetra_covariances =  np.array([np.cov(t,rowvar=False) for t in tetra_positions])
        tetra_lambdas = np.array([np.linalg.eig(c)[0] for c in tetra_covariances])

        tetra_max_lambdas = np.abs(tetra_lambdas).max(axis=1)
        tetra_min_lambdas = np.abs(tetra_lambdas).min(axis=1)
        tetra_lambda_ratios = 1.0 - tetra_min_lambdas/tetra_max_lambdas

        # maximal_tetra_area = np.sqrt(3)*np.power(maximal_distance,2.0)
        # maximal_tetra_volume = np.sqrt(2)*np.power(maximal_distance,3.0)/12.

        tetra_boxes = tetra_positions.max(axis=1) - tetra_positions.min(axis=1)
        tetra_features['area'] = tetra_areas

    if 'circumscribed_sphere_center' in features or 'circumscribed_sphere_radius' in features or 'radius_edge_ratio' in features:

        tetra_circumsphere_centers = []
        for t in tetrahedra:
            tetra_triangles = t[tetra_triangle_list]
            tetra_triangle_edges = tetra_triangles[:,triangle_edge_list]
            tetra_triangle_edge_lengths = np.linalg.norm(positions.values(tetra_triangle_edges[...,1])-positions.values(tetra_triangle_edges[...,0]),axis=2)

            try:
                cayley_menger_edges = np.array([[(p1,p2) for p2 in t] for p1 in t])
                cayley_menger_edge_lengths = np.linalg.norm(positions.values(cayley_menger_edges[...,1])-positions.values(cayley_menger_edges[...,0]),axis=2)
                cayley_menger_matrix = np.array([[0,1,1,1,1],
                                                 [1]+list(np.power(cayley_menger_edge_lengths[0],2)),
                                                 [1]+list(np.power(cayley_menger_edge_lengths[1],2)),
                                                 [1]+list(np.power(cayley_menger_edge_lengths[2],2)),
                                                 [1]+list(np.power(cayley_menger_edge_lengths[3],2))])

                center_weights = np.linalg.inv(cayley_menger_matrix)[0,1:]
                tetra_circumsphere_centers += [(positions.values(t)*center_weights[:,np.newaxis]).sum(axis=0)]
            except:
                print "LinAlgError..."
                tetra_circumsphere_centers += [positions.values(t).mean(axis=0)]
        tetra_circumsphere_centers = np.array(tetra_circumsphere_centers)
        tetra_circumsphere_radius = np.linalg.norm(tetra_positions[:,0] - tetra_circumsphere_centers,axis=1)

        tetra_features['circumscribed_sphere_center_x'] = tetra_circumsphere_centers[:,0]
        tetra_features['circumscribed_sphere_center_y'] = tetra_circumsphere_centers[:,1]
        tetra_features['circumscribed_sphere_center_z'] = tetra_circumsphere_centers[:,2]
        tetra_features['circumscribed_sphere_radius'] = tetra_circumsphere_radius
        tetra_features['radius_edge_ratio'] = tetra_circumsphere_radius/tetra_sorted_edge_lengths[:,-1]


    if 'max_dihedral_angle' in features or 'min_dihedral_angle' in features:
        tetra_triangles = tetrahedra[:,tetra_triangle_list ]
        tetra_edges = tetra_triangles[...,triangle_edge_list]
        tetra_triangle_edge_vectors = positions.values(tetra_edges[...,1]) - positions.values(tetra_edges[...,0])
        tetra_triangle_normals = np.cross(tetra_triangle_edge_vectors[:,:,2],tetra_triangle_edge_vectors[:,:,1])
        tetra_triangle_normals = tetra_triangle_normals/np.linalg.norm(tetra_triangle_normals,axis=2)[:,:,np.newaxis]

        tetra_centers = positions.values(tetrahedra).mean(axis=1)
        tetra_triangle_center = positions.values(tetra_triangles).mean(axis=2)
        tetra_triangle_tetra_center = np.repeat(tetra_centers,4,axis=0).reshape(len(tetra_centers),4,3)
        tetra_triangle_tetra_vector = tetra_triangle_tetra_center - tetra_triangle_center
        normal_orientation = np.sign(np.einsum('...ij,...ij->...i',tetra_triangle_normals,tetra_triangle_tetra_vector))
        tetra_triangle_normals = normal_orientation[...,np.newaxis]*tetra_triangle_normals

        tetra_dihedral_normals = tetra_triangle_normals[:,tetra_edge_list]
        tetra_dihedral_cosines = np.einsum('...ij,...ij->...i',tetra_dihedral_normals[:,:,0],tetra_dihedral_normals[:,:,1])
        tetra_dihedral_angles = 180. - 180.*np.arccos(tetra_dihedral_cosines)/np.pi
        tetra_sorted_dihedral_angles = np.sort(tetra_dihedral_angles)
        tetra_features['max_dihedral_angle'] = tetra_sorted_dihedral_angles[:,-1] 
        tetra_features['min_dihedral_angle'] = tetra_sorted_dihedral_angles[:,0] 

    return np.concatenate([tetra_features[f][:,np.newaxis] for f in features],axis=1)    


def triangle_geometric_features(triangles,positions,features=['area','max_distance']):
    triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])
    triangle_edges = triangles[...,triangle_edge_list]
    triangle_edge_lengths = np.linalg.norm(positions.values(triangle_edges[...,1]) - positions.values(triangle_edges[...,0]),axis=2)
    
    triangle_features={}

    triangle_features['edge_lengths'] = triangle_edge_lengths

    triangle_features['perimeter'] = triangle_edge_lengths.sum(axis=1)
    triangle_features['area'] = np.sqrt((triangle_features['perimeter']/2.0)*(triangle_features['perimeter']/2.0-triangle_edge_lengths[...,0])*(triangle_features['perimeter']/2.0-triangle_edge_lengths[...,1])*(triangle_features['perimeter']/2.0-triangle_edge_lengths[...,2]))
    triangle_features['eccentricity'] = 1. - (12.0*np.sqrt(3)*triangle_features['area'])/np.power(triangle_features['perimeter'],2.0)
    
    if ('max_distance' in features) or ('min_distance' in features):
        triangle_edge_lengths = np.sort(triangle_edge_lengths)
        triangle_features['max_distance'] = triangle_edge_lengths[:,-1]
        triangle_features['min_distance'] = triangle_edge_lengths[:,0]

    if 'sinus' in features or 'sinus_eccentricity' in features:
        triangle_features['sinus'] = np.zeros_like(triangle_edge_lengths,np.float16)
        triangle_features['sinus'][:,0] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[...,1]**2+triangle_edge_lengths[...,2]**2-triangle_edge_lengths[:,0]**2,2.0)/np.power(2.0*triangle_edge_lengths[...,1]*triangle_edge_lengths[...,2],2.0),np.float16))
        triangle_features['sinus'][:,1] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[...,2]**2+triangle_edge_lengths[...,0]**2-triangle_edge_lengths[:,1]**2,2.0)/np.power(2.0*triangle_edge_lengths[...,2]*triangle_edge_lengths[...,0],2.0),np.float16))
        triangle_features['sinus'][:,2] = np.sqrt(np.array(1.0 - np.power(triangle_edge_lengths[...,0]**2+triangle_edge_lengths[...,1]**2-triangle_edge_lengths[:,2]**2,2.0)/np.power(2.0*triangle_edge_lengths[...,0]*triangle_edge_lengths[...,1],2.0),np.float16))
    
    if 'cosinus' in features:
        triangle_features['cosinus'] = np.zeros_like(triangle_edge_lengths,np.float16)
        triangle_features['cosinus'][:,0] = (triangle_edge_lengths[...,1]**2+triangle_edge_lengths[...,2]**2-triangle_edge_lengths[:,0]**2)/(2.0*triangle_edge_lengths[...,1]*triangle_edge_lengths[...,2])
        triangle_features['cosinus'][:,1] = (triangle_edge_lengths[...,2]**2+triangle_edge_lengths[...,0]**2-triangle_edge_lengths[:,1]**2)/(2.0*triangle_edge_lengths[...,2]*triangle_edge_lengths[...,0])
        triangle_features['cosinus'][:,2] = (triangle_edge_lengths[...,0]**2+triangle_edge_lengths[...,1]**2-triangle_edge_lengths[:,2]**2)/(2.0*triangle_edge_lengths[...,0]*triangle_edge_lengths[...,1])

    if 'sinus_eccentricity' in features:
        triangle_features['sinus_eccentricity'] = 1.0 - (2.0*triangle_features['sinus'].sum(axis=1))/(3*np.sqrt(3))

    return np.concatenate([triangle_features[f][:,np.newaxis] for f in features],axis=1)

