# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
#
#       Copyright 2016 INRIA - CIRAD - INRA
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

from openalea.mesh.property_topomesh_creation import vertex_topomesh, edge_topomesh, triangle_topomesh, tetrahedra_topomesh

def test_vertex_topomesh():
    n_points = 10
    points = dict(zip(range(n_points),np.random.rand(3*n_points).reshape((n_points,3))))

    topomesh = vertex_topomesh(points)

    assert topomesh.nb_wisps(0) == n_points
    assert topomesh.nb_wisps(1) == 0
    assert topomesh.nb_wisps(2) == 0
    assert topomesh.nb_wisps(3) == 0

    for p in points.keys():
        assert np.all(topomesh.wisp_property('barycenter',0)[p] == points[p])


def test_edge_topomesh():
    n_points = 10
    points = dict(zip(range(n_points),np.random.rand(3*n_points).reshape((n_points,3))))
    edges = np.transpose([np.arange(n_points),(np.arange(n_points)+1)%n_points])

    topomesh = edge_topomesh(edges, points)

    assert topomesh.nb_wisps(0) == n_points
    assert topomesh.nb_wisps(1) == n_points
    assert topomesh.nb_wisps(2) == 0
    assert topomesh.nb_wisps(3) == 0

    for p in points.keys():
        assert np.all(topomesh.wisp_property('barycenter',0)[p] == points[p])

    for eid,e in enumerate(edges):
        assert set(topomesh.borders(1,eid)) == set(e)



