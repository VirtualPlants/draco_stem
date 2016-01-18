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
from openalea.mesh.property_topomesh_creation import tetrahedra_topomesh


def tetrahedron_example_topomesh():
    points = {}
    points[0] = [0,0,0]
    points[1] = [1,0,0]
    points[2] = [0,1,0]
    points[3] = [0,0,1]

    topomesh = tetrahedra_topomesh([[0,1,2,3]],points)

    return topomesh