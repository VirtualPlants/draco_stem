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

from openalea.container import array_dict

from openalea.mesh.property_topomesh_creation import tetrahedra_topomesh

from openalea.draco_stem.draco.adjacency_complex_optimization import tetrahedrization_topomesh_topological_optimization
from openalea.draco_stem.draco.adjacency_complex_optimization import tetrahedrization_topomesh_remove_exterior, tetrahedrization_topomesh_add_exterior
from openalea.draco_stem.draco.adjacency_complex_optimization import compute_tetrahedrization_topological_properties, compute_tetrahedrization_geometrical_properties

from openalea.mesh.utils.evaluation_tools import jaccard_index


def octahedron_tetrahedra(side_length=15.):
    points = {}
    points[2] = np.array([1,0,0]) + 0.1*np.random.rand(3)
    points[3] = np.array([0,1,0]) + 0.1*np.random.rand(3)
    points[4] = np.array([-1,0,0]) + 0.1*np.random.rand(3)
    points[5] = np.array([0,-1,0]) + 0.1*np.random.rand(3)
    points[6] = np.array([0,0,1]) + 0.1*np.random.rand(3)
    points[7] = np.array([0,0,-1]) + 0.1*np.random.rand(3)

    tetrahedra = []
    tetrahedra += [[2,3,4,6]]
    tetrahedra += [[2,4,5,6]]
    tetrahedra += [[2,3,4,7]]
    tetrahedra += [[2,4,5,7]]

    return tetrahedra_topomesh(tetrahedra,points)


def test_tetrahedrization_optimization():

    topomesh = octahedron_tetrahedra()

    compute_tetrahedrization_topological_properties(topomesh)
    compute_tetrahedrization_geometrical_properties(topomesh)

    tetrahedrization_topomesh_add_exterior(topomesh)

    target_tetrahedra = []
    target_tetrahedra += [(2,3,5,6)]
    target_tetrahedra += [(3,4,5,6)]
    target_tetrahedra += [(2,3,5,7)]
    target_tetrahedra += [(3,4,5,7)]

    target_tetrahedra += [(1,2,3,6)]
    target_tetrahedra += [(1,3,4,6)]
    target_tetrahedra += [(1,4,5,6)]
    target_tetrahedra += [(1,2,5,6)]

    target_tetrahedra += [(1,2,3,7)]
    target_tetrahedra += [(1,3,4,7)]
    target_tetrahedra += [(1,4,5,7)]
    target_tetrahedra += [(1,2,5,7)]


    target_tetrahedra = dict(zip(target_tetrahedra,np.zeros((4,3))))

    kwargs = {}
    kwargs['simulated_annealing_initial_temperature'] = 0.1
    kwargs['simulated_annealing_lambda'] =  0.9
    kwargs['simulated_annealing_minimal_temperature'] = 0.09
    kwargs['n_iterations'] = 1

    topomesh = tetrahedrization_topomesh_topological_optimization(topomesh,image_cell_vertex=target_tetrahedra,omega_energies=dict(image=1.0),**kwargs)
    tetrahedrization_topomesh_remove_exterior(topomesh)

    assert topomesh.nb_wisps(3) == 4

    for t in topomesh.wisps(3):
        print t, list(topomesh.borders(3,t,3))

    optimized_tetras = np.sort([list(topomesh.borders(3,t,3)) for t in topomesh.wisps(3)])
    target_tetras = np.sort(target_tetrahedra.keys())
    target_tetras = target_tetras[target_tetras[:,0] != 1]

    assert jaccard_index(optimized_tetras,target_tetras) == 1.0



