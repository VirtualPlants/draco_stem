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

from openalea.mesh.property_topomesh_creation import triangle_topomesh

def square_topomesh(side_length = 1):
    points = {}
    points[0] = [0,0,0]
    points[1] = [side_length,0,0]
    points[2] = [0,side_length,0]
    points[3] = [side_length,side_length,0]

    triangles = [[0,1,2],[3,2,1]]

    return triangle_topomesh(triangles, points)

def hexagon_topomesh(side_length = 1):
    points = {}
    points[0] = [0,0,0]
    for p in xrange(6):
        points[p+1] = [side_length*np.cos(p*np.pi/3.),side_length*np.sin(p*np.pi/3.),0]

    triangles = []
    for p in xrange(6):
        triangles += [[0,p + 1,(p+1)%6 + 1]]

    return triangle_topomesh(triangles, points)