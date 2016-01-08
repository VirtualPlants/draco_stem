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
from scipy.spatial import Delaunay

from openalea.mesh.utils.array_tools  import array_unique

tetra_triangle_edge_list  = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])

def delaunay_triangulation(points):
    if len(np.unique(points[:,2])) == 1:
        triangles = Delaunay(np.array(points)[:,:2]).simplices
    else:
        tetras = Delaunay(np.array(points)).simplices
        triangles = array_unique(np.sort(np.concatenate(tetras[:,tetra_triangle_list])))

    return triangles