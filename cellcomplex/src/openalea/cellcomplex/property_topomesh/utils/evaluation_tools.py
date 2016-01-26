# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
#
#       Copyright 2015-2016 INRIA - CIRAD - INRA
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
from scipy.cluster.vq                       import vq

def performance_measure(reference_set,experimental_set,measure='jaccard_index'):
    VP = (vq(experimental_set,reference_set)[1]==0).sum()
    FP = (vq(experimental_set,reference_set)[1]>0).sum()
    FN = (vq(reference_set,experimental_set)[1]>0).sum()

    if measure == 'true_positive':
        return VP
    elif measure == 'precision':
        return VP/float(VP+FP) 
    elif measure == 'recall':
        return VP/float(VP+FN) 
    elif measure == 'dice_index':
        return 2*VP / float(2*VP+FP+FN)
    elif measure == 'jaccard_index':
        return VP/float(VP+FP+FN)

def jaccard_index(reference_set,experimental_set):
    VP = (vq(experimental_set,reference_set)[1]==0).sum()
    FP = (vq(experimental_set,reference_set)[1]>0).sum()
    FN = (vq(reference_set,experimental_set)[1]>0).sum()
    return VP/float(VP+FP+FN)
