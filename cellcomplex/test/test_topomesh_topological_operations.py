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

from openalea.mesh.property_topomesh_optimization import topomesh_flip_edge, topomesh_split_edge
from openalea.mesh.property_topomesh_analysis import compute_topomesh_property
from openalea.mesh.property_topomesh_creation import vertex_topomesh, edge_topomesh, triangle_topomesh, tetrahedra_topomesh

from openalea.container import array_dict

from example_topomesh import square_topomesh, hexagon_topomesh

def test_topomesh_edge_flip():
    side_length = 1.
    topomesh = square_topomesh(side_length)

    for fid in topomesh.wisps(2):
        assert set(topomesh.borders(2,fid,2)) == set([0,1,2]) or set(topomesh.borders(2,fid,2)) == set([1,2,3])
    compute_topomesh_property(topomesh,'area',2)
    assert np.all(np.isclose(topomesh.wisp_property('area',2).values(),np.power(side_length,2.)/2.,1e-7))

    topomesh_flip_edge(topomesh,2)

    for fid in topomesh.wisps(2):
        assert set(topomesh.borders(2,fid,2)) == set([0,2,3]) or set(topomesh.borders(2,fid,2)) == set([0,1,3])

    compute_topomesh_property(topomesh,'area',2)
    assert np.all(np.isclose(topomesh.wisp_property('area',2).values(),np.power(side_length,2.)/2.,1e-7))


def test_topomesh_edge_split():
    side_length = 1.
    topomesh = square_topomesh(side_length)

    topomesh_split_edge(topomesh,2)
    compute_topomesh_property(topomesh,'area',2)

    assert np.all(np.isclose(topomesh.wisp_property('area',2).values(),np.power(side_length,2.)/4.,1e-7))





