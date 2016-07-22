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
#       File contributor(s): Guillaume Baty <guillaume.baty@inria.fr>, 
#                            Guillaume Cerutti <guillaume.cerutti@inria.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       TissueLab Website : http://virtualplants.github.io/
#
###############################################################################

__revision__ = ""

from openalea.core.plugin import PluginDef
from openalea.core.authors import gcerutti


class AppletPlugin(object):
    name_conversion = PluginDef.DROP_PLUGIN


@PluginDef
class DracoMeshing(AppletPlugin):
    label = 'Draco'
    icon = 'draco.png'
    authors = [gcerutti]
    implement = 'IApplet'
    __plugin__ = True

    def __call__(self):
        # Load and instantiate graphical component that actually provide feature
        from openalea.draco_stem.draco_oalab.widget.draco_panel import DracoPanel
        return DracoPanel
