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

from openalea.mesh.property_topomesh_analysis import compute_topomesh_property
from openalea.mesh.property_topomesh_creation import vertex_topomesh, edge_topomesh, triangle_topomesh, tetrahedra_topomesh

from openalea.container import array_dict

from example_topomesh import square_topomesh, hexagon_topomesh


def test_length_property():
    side_length = 1.

    topomesh = square_topomesh(side_length)
    compute_topomesh_property(topomesh,'length',1)
    for eid in topomesh.wisps(1):
        if topomesh.nb_regions(1,eid) == 1:
            assert np.isclose(topomesh.wisp_property('length',1)[eid],side_length,1e-7)
        else:
            assert np.isclose(topomesh.wisp_property('length',1)[eid],np.sqrt(2)*side_length,1e-7)

    topomesh = hexagon_topomesh(side_length)
    compute_topomesh_property(topomesh,'length',1)
    for eid in topomesh.wisps(1):
        assert np.isclose(topomesh.wisp_property('length',1)[eid],side_length,1e-7)


def test_length_scaling():
    side_length = 1.
    scale_factor = 2.

    topomesh = hexagon_topomesh(side_length)
    compute_topomesh_property(topomesh,'length',1)

    topomesh.update_wisp_property('barycenter',0,array_dict(topomesh.wisp_property('barycenter',0).values()*scale_factor,list(topomesh.wisps(0))))
    side_length *= scale_factor

    compute_topomesh_property(topomesh,'length',1)
    for eid in topomesh.wisps(1):
        assert np.isclose(topomesh.wisp_property('length',1)[eid],side_length,1e-7)

def test_area_property():
    side_length = 1.

    topomesh = square_topomesh(side_length)
    compute_topomesh_property(topomesh,'area',2)
    for fid in topomesh.wisps(2):
        assert np.isclose(topomesh.wisp_property('area',2)[fid],np.power(side_length,2.)/2.,1e-7)

    topomesh = hexagon_topomesh(side_length)
    compute_topomesh_property(topomesh,'area',2)
    for fid in topomesh.wisps(2):
        assert np.isclose(topomesh.wisp_property('area',2)[fid],np.power(side_length,2.)*np.sqrt(3)/4.,1e-7)


def test_angle_property():
    side_length = 1.

    topomesh = square_topomesh(side_length)
    compute_topomesh_property(topomesh,'angles',2)
    for fid in topomesh.wisps(2):
        angles = topomesh.wisp_property('angles',2)[fid]
        assert (angles == np.pi/2.).sum() == 1
        assert (angles == np.pi/4.).sum() == 2

    topomesh = hexagon_topomesh(side_length)
    compute_topomesh_property(topomesh,'angles',2)
    for fid in topomesh.wisps(2):
        assert np.all(np.isclose(topomesh.wisp_property('angles',2)[fid],np.pi/3.,1e-7))

def test_eccentricity_property():
    side_length = 1.

    topomesh = hexagon_topomesh(side_length)
    compute_topomesh_property(topomesh,'eccentricity',2)
    print topomesh.wisp_property('eccentricity',2).values()
    print np.isclose(topomesh.wisp_property('eccentricity',2).values(),0.,1e1)
    assert np.all(np.isclose(1-topomesh.wisp_property('eccentricity',2).values(),1.,1e-3))





