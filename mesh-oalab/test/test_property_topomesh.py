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

from openalea.mesh import PropertyTopomesh
from openalea.mesh.property_topomesh_analysis import compute_topomesh_property

from openalea.mesh.testing import tetrahedron_example_topomesh
import numpy as np

def test_topomesh_creation():
    topomesh = tetrahedron_example_topomesh()

    assert topomesh.nb_wisps(0) == 4
    assert topomesh.nb_wisps(1) == 6
    assert topomesh.nb_wisps(2) == 4
    assert topomesh.nb_wisps(3) == 1


def test_topomesh_metrics():
    topomesh = tetrahedron_example_topomesh()

    compute_topomesh_property(topomesh,'length',1)
    compute_topomesh_property(topomesh,'area',2)
    compute_topomesh_property(topomesh,'volume',3)

    assert np.all((topomesh.wisp_property('length',1).values().astype(np.float16)==1) | (topomesh.wisp_property('length',1).values().astype(np.float16)==np.sqrt(2)))
    assert np.all((topomesh.wisp_property('area',2).values().astype(np.float16)==1/2.) | (topomesh.wisp_property('area',2).values().astype(np.float16)==np.sqrt(12)/4.))
    assert np.all((topomesh.wisp_property('volume',3).values().astype(np.float16)==1/6.))