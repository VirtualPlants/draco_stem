# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
#
#       Copyright 2015 INRIA - CIRAD - INRA
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
class TopomeshControl(AppletPlugin):
    label = 'Topomesh Controls'
    icon = 'topomesh_control.png'
    authors = [gcerutti]
    implement = 'IApplet'
    __plugin__ = True

    def __call__(self):
        # Load and instantiate graphical component that actually provide feature
        from openalea.mesh_oalab.widget.property_topomesh_panel import TopomeshControlPanel
        return TopomeshControlPanel


@PluginDef
class DataframeControl(AppletPlugin):
    label = 'Dataframe Controls'
    icon = 'dataframe_control.png'
    authors = [gcerutti]
    implement = 'IApplet'
    __plugin__ = True

    def __call__(self):
        # Load and instantiate graphical component that actually provide feature
        from openalea.mesh_oalab.widget.dataframe_panel import DataframeControlPanel
        return DataframeControlPanel
