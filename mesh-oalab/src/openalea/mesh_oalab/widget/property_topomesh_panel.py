# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
#
#       Copyright 2015-2016 INRIA - CIRAD - INRA
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

# import weakref
from openalea.vpltk.qt import QtGui, QtCore
from openalea.core.observer import AbstractListener
from openalea.core.control import Control
from openalea.oalab.control.manager import ControlManagerWidget
from openalea.core.service.ipython import interpreter as get_interpreter

from openalea.oalab.service.drag_and_drop import add_drop_callback
from openalea.oalab.widget.world import WorldModel

from openalea.container import PropertyTopomesh, array_dict

try:
    from openalea.mesh.triangular_mesh import topomesh_to_triangular_mesh
except:
    print "Openalea.Cellcomplex must be installed to use TopomeshControls!"
    raise


import numpy as np

from tissuelab.gui.vtkviewer.vtkworldviewer import setdefault, world_kwargs


element_names = dict(zip(range(4),['vertices','edges','faces','cells']))

cst_proba = dict(step=0.01, min=0, max=1)
cst_degree = dict(step=1,min=0,max=3)

attribute_definition = {}
attribute_definition['topomesh'] = {}
for degree in xrange(4):
    attribute_definition['topomesh']["display_"+str(degree)] = dict(value=False,interface="IBool",constraints={},label="Display "+element_names[degree])
    attribute_definition['topomesh']["property_degree_"+str(degree)] = dict(value=degree,interface="IInt",constraints=cst_degree,label="Degree") 
    attribute_definition['topomesh']["property_name_"+str(degree)] = dict(value="",interface="IEnumStr",constraints=dict(enum=[""]),label="Property")     
    attribute_definition['topomesh']["coef_"+str(degree)] = dict(value=1,interface="IFloat",constraints=cst_proba,label="Coef") 


def _property_names(world_object, attr_name, property_name, **kwargs):
    degree = int(attr_name[-1])
    property_degree = world_object["property_degree_"+str(degree)]
    print "New property_names ! ",property_degree
    topomesh = world_object.data
    constraints = dict(enum=[""]+list(topomesh.wisp_property_names(property_degree)))
    print constraints
    if property_name in constraints['enum']:
        return dict(value=property_name, constraints=constraints)
    else:
        return dict(value="", constraints=constraints)

class TopomeshControlPanel(QtGui.QWidget, AbstractListener):
    StyleTableView = 0
    StylePanel = 1
    DEFAULT_STYLE = StylePanel

    element_names = dict(zip(range(4),['vertices','edges','faces','cells']))

    property_colormaps = {}
    property_colormaps['cells'] = 'glasbey'
    property_colormaps['volume'] = 'morocco'
    property_colormaps['eccentricity'] = 'jet'
    
    def __init__(self, parent=None, style=None):
        AbstractListener.__init__(self)
        QtGui.QWidget.__init__(self, parent=parent)

        self.world = None
        self.model = WorldModel()

        if style is None:
            style = self.DEFAULT_STYLE
        self.style = style

        # self._manager = {}

        # self._cb_world_object = QtGui.QComboBox()
        # p = QtGui.QSizePolicy
        # self._cb_world_object.setSizePolicy(p(p.Expanding, p.Maximum))
        # self._cb_world_object.currentIndexChanged.connect(self._selected_object_changed)
        self._mesh = {}
        self._mesh_matching = {}


        self._current = None
        # self._default_manager = self._create_manager()

        self.interpreter = get_interpreter()
        self.interpreter.locals['topomesh_control'] = self

        self._layout = QtGui.QVBoxLayout(self)
        # self._layout.addWidget(self._cb_world_object)

        if self.style == self.StyleTableView:
            self._view = None
            # self._view = ControlManagerWidget(manager=self._default_manager)
            # self._layout.addWidget(self._view)
        elif self.style == self.StylePanel:
            self._view = None
            # self._set_manager(self._default_manager)
        else:
            raise NotImplementedError('style %s' % self.style)


    def set_properties(self, properties):
        if self.style == self.StyleTableView:
            self._view.set_properties(properties)

    def properties(self):
        if self.style == self.StyleTableView:
            return self._view.properties()
        else:
            return []

    def set_style(self, style):
        if style == self.style:
            return

        world = self.world
        self.clear()
        if self.style == self.StyleTableView:
            view = self._view
        elif self.style == self.StylePanel:
            if self._view and self._view():
                view = self._view()
            else:
                return

        # Remove old view
        view.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self._layout.removeWidget(view)
        view.close()
        del view
        self._view = None

        self.style = style
        # if style == self.StyleTableView:
            # self._view = ControlManagerWidget(manager=self._default_manager)
            # self._layout.addWidget(self._view)

        self.set_world(world)

    # def __getitem__(self, key):
    #     return self._manager[self._current].control(name=key)

    def initialize(self):
        from openalea.core.world.world import World
        from openalea.core.service.ipython import interpreter
        world = World()
        world.update_namespace(interpreter())
        self.set_world(world)

    def set_world(self, world):
        self.clear()

        self.world = world
        self.world.register_listener(self)

        if self.style == self.StyleTableView:
            self.model.set_world(world)

        for object_name in world.keys():
            if isinstance(world[object_name].data,PropertyTopomesh):
                self.refresh_world_object(world[object_name])
    
    def notify(self, sender, event=None):
        signal, data = event
        if signal == 'world_changed':
            world, old_object, new_object = data
            if isinstance(new_object.data,PropertyTopomesh):
                self.refresh()
        elif signal == 'world_object_removed':
            world, old_object = data
            if isinstance(old_object.data,PropertyTopomesh):
                for degree in xrange(4):
                    if world.has_key(old_object.name+"_"+self.element_names[degree]):
                        world.remove(old_object.name+"_"+self.element_names[degree])
                self.refresh()
        elif signal == 'world_object_changed':
            world, old_object, world_object = data
            if isinstance(world_object.data,PropertyTopomesh):
                self.refresh_world_object(world_object)
        elif signal == 'world_object_item_changed':
            world, world_object, item, old, new = data
            if isinstance(world_object.data,PropertyTopomesh):
                # self.refresh_manager(world_object)
                if item == 'attribute':
                    self.update_topomesh_display(world_object, new)
        elif signal == 'world_sync':
            self.refresh()

    # def clear_managers(self):
    #     self._current = None
    #     self._cb_world_object.clear()
    #     for name, manager in self._manager.items():
    #         manager.clear_followers()
    #         del self._manager[name]
    #     self._set_manager(self._default_manager)

    def clear(self):
        # self.clear_managers()
        if self.world:
            self.world.unregister_listener(self)
            self.world = None

    def refresh_world_object(self, world_object):
        if world_object:
            dtype = 'topomesh'

            topomesh = world_object.data
            kwargs = world_kwargs(world_object)

            print "Set default attributes : ",world_object.name

            for degree in np.arange(4)[::-1]:
                setdefault(world_object, dtype, 'display_'+str(degree), attribute_definition=attribute_definition, **kwargs)
                world_object.silent = True
                setdefault(world_object, dtype, 'property_degree_'+str(degree), attribute_definition=attribute_definition, **kwargs)
                setdefault(world_object, dtype, 'property_name_'+str(degree), conv=_property_names, attribute_definition=attribute_definition, **kwargs)
                if degree>1:
                    setdefault(world_object, dtype, 'coef_'+str(degree), attribute_definition=attribute_definition, **kwargs)
                world_object.silent = False
            
            if not self._mesh.has_key(world_object.name):
                self._mesh[world_object.name] = dict([(0,None),(1,None),(2,None),(3,None)])
                self._mesh_matching[world_object.name] = dict([(0,None),(1,None),(2,None),(3,None)])

            world_object.set_attribute("display_"+str(max([degree for degree in xrange(4) if world_object.data.nb_wisps(degree)>0])),True)


    # def _fill_manager(self, manager, world_object):
    #     if world_object:
    #         dtype = 'topomesh'

    #         topomesh = world_object.data
    #         kwargs = world_kwargs(world_object)

    #         print "Set default attributes : ",world_object.name

    #         world_object.silent = True
    #         for degree in np.arange(4)[::-1]:
    #             setdefault(world_object, dtype, 'display_'+str(degree), **kwargs)
    #             setdefault(world_object, dtype, 'property_degree_'+str(degree), **kwargs)
    #             setdefault(world_object, dtype, 'property_name_'+str(degree), conv=_property_names, **kwargs)
    #             if degree>1:
    #                 setdefault(world_object, dtype, 'coef_'+str(degree), **kwargs)
    #         world_object.silent = False

                # for attribute in world_object.attributes:
                #     manager.add(
                #         attribute['name'],
                #         interface=attribute['interface'],
                #         value=attribute['value'],
                #         label=attribute['label'],
                #         constraints=attribute['constraints']
                #     )

                # if 'display' in attribute['name']:
                #     degree = int(attribute['name'][-1])
                #     manager.register_follower(attribute['name'], self._display_control_changed(world_object, degree))
                # if 'property_degree' in attribute['name']:
                #     manager.register_follower(attribute['name'], self._property_control_changed(world_object, degree))

    # def _get_manager(self, world_object):
    #     object_name = world_object.name
    #     if object_name not in self._manager:
    #         manager = self._create_manager(world_object)
    #         self._manager[object_name] = manager
    #         self._cb_world_object.addItem(object_name)
    #         world_object.set_attribute("display_"+str(world_object.data.degree()),True)
    #     return self._manager[object_name]

    # def _create_manager(self, world_object=None):
    #     from openalea.core.control.manager import ControlContainer
    #     manager = ControlContainer()
    #     self._fill_manager(manager, world_object)
    #     return manager

    # def _selected_object_changed(self, idx):
        # if idx != -1:
        #     self.select_world_object(self._cb_world_object.itemText(idx))

    # def _set_manager(self, manager):
    #     if self.style == self.StylePanel:
    #         view = self._view
    #         if self._view is not None:
    #             view = self._view()
    #         if view:
    #             self._layout.removeWidget(view)
    #             view.close()
    #             del view
    #         from openalea.oalab.service.qt_control import edit
    #         view = edit(manager)
    #         view.setAttribute(QtCore.Qt.WA_DeleteOnClose)
    #         self._view = weakref.ref(view)
    #         self._layout.addWidget(view)
    #         view.show()
    #         self.repaint()
    #     elif self.style == self.StyleTableView:
    #         self._view.model.set_manager(manager)
    #     else:
    #         raise NotImplementedError('style %s' % self.style)
    
    def select_world_object(self, object_name):
        if object_name != self._current:
            self._current = object_name
            # object_manager = self._manager[object_name]
            # object_manager.disable_followers()
            # self._set_manager(object_manager)
            # object_manager.enable_followers()

    def refresh_item(self, world_object, item, old, new):
        object_name = world_object.name
        # if item == 'attribute':
            # manager = self._get_manager(world_object)
            # attr_name = new['name']
            # attr_value = new['value']
            # control = manager.control(name=attr_name)
            # if control:
            #     control.value = attr_value
        # else:
        #     self.refresh_manager(world_object)

    # def refresh_manager(self, world_object):
    #     object_name = world_object.name
    #     object_manager = self._get_manager(world_object)

    #     manager_attr_names = [c.name for c in self._manager[object_name].controls()]
    #     object_attr_names = [a['name'] for a in world_object.attributes]
    #     if manager_attr_names != object_attr_names:
    #         object_manager.clear_followers()
    #         object_manager.clear()
    #         self._fill_manager(object_manager, world_object)
    #         if self._current == object_name:
    #             self._set_manager(object_manager)
    #             object_manager.enable_followers()
    #     else:
    #         for a in world_object.attributes:
    #             if a['value'] != self._manager[object_name].control(a['name']).value:
    #                 self._manager[object_name].control(a['name']).set_value(a['value'])

    def refresh(self):
        if self.world is not None:
            self.set_world(self.world)

    def update_topomesh_display(self, world_object, attribute):
        if world_object:
            if 'display_' in attribute['name'] or 'coef_' in attribute['name']:
                display_degree = int(attribute['name'][-1])
                if world_object['display_'+str(display_degree)]:
                    topomesh = world_object.data
                    property_name = world_object['property_name_'+str(display_degree)]
                    property_degree = world_object['property_degree_'+str(display_degree)]
                    if display_degree > 1:
                        coef = world_object['coef_'+str(display_degree)]
                    else:
                        coef = 1
                    print "Property : ",property_name," (",attribute['name'],")"
                    mesh, matching = topomesh_to_triangular_mesh(topomesh,degree=display_degree,coef=coef,mesh_center=[0,0,0],property_name=property_name,property_degree=property_degree)
                    
                    self._mesh[world_object.name][display_degree] = mesh
                    self._mesh_matching[world_object.name][display_degree] = matching

                    if self.world.has_key(world_object.name+"_"+self.element_names[display_degree]):
                        kwargs = world_kwargs(self.world[world_object.name+"_"+self.element_names[display_degree]])
                        if not 'coef_' in attribute['name']:
                            if kwargs.has_key('intensity_range'):
                                kwargs.pop('intensity_range')
                    else:
                        kwargs = {}
                        kwargs['colormap'] = 'glasbey' if (property_name == '') else self.property_colormaps.get(property_name,'grey')
                        # kwargs['position'] = world_object['position']

                    self.world.add(mesh,world_object.name+"_"+self.element_names[display_degree],**kwargs)

                else:
                    self.world.remove(world_object.name+"_"+self.element_names[display_degree])
            elif 'property_name_' in attribute['name']:
                display_degree = int(attribute['name'][-1])
                if world_object['display_'+str(display_degree)]:
                    topomesh = world_object.data
                    property_name = world_object['property_name_'+str(display_degree)]
                    property_degree = world_object['property_degree_'+str(display_degree)]
                    mesh_element_matching = self._mesh_matching[world_object.name][display_degree][property_degree]
                    if topomesh.has_wisp_property(property_name,property_degree):
                        property_data = array_dict(topomesh.wisp_property(property_name,property_degree).values(mesh_element_matching.values()),mesh_element_matching.keys())
                    else:
                        property_data = array_dict()
                    if property_degree == 0:
                        self._mesh[world_object.name][display_degree].point_data = property_data.to_dict()
                    else:
                        self._mesh[world_object.name][display_degree].triangle_data = property_data.to_dict()

                    self.world[world_object.name+"_"+self.element_names[display_degree]].data = self._mesh[world_object.name][display_degree]
                    if len(property_data)>1:
                        self.world[world_object.name+"_"+self.element_names[display_degree]].set_attribute('intensity_range',(property_data.values().min(),property_data.values().max()))

            elif 'property_degree_' in attribute['name']:
                dtype = 'topomesh'
                kwargs = world_kwargs(world_object)
                display_degree = int(attribute['name'][-1])
                print world_object['property_degree_'+str(display_degree)]
                world_object.silent = True
                setdefault(world_object, dtype, 'property_name_'+str(display_degree), conv=_property_names, attribute_definition=attribute_definition, **kwargs)
                world_object.silent = False
                # world_object.set_attribute("property_name_"+str(display_degree),"")


    # def _display_control_changed(self, world_object, display_degree):
    #     def _changed(old, new):
    #         self._display_changed(world_object, display_degree, old, new)
    #     return _changed

    # def _property_control_changed(self, world_object, display_degree):
    #     def _changed(old, new):
    #         self._property_degree_changed(world_object, display_degree, old, new)
    #     return _changed

    # def _display_changed(self, world_object, display_degree, old, new):
    #     if world_object:
    #         if self['display_'+str(display_degree)].value:
    #             topomesh = world_object.data
    #             property_name = self['property_'+str(display_degree)].value
    #             property_degree = self['property_degree_'+str(display_degree)].value
    #             if display_degree > 1:
    #                 coef = self['coef_'+str(display_degree)].value
    #             else:
    #                 coef = 1
    #             print "Property : ",property_name
    #             mesh,_,_ = topomesh_to_triangular_mesh(topomesh,degree=display_degree,coef=coef,mesh_center=[0,0,0],property_name=property_name,property_degree=property_degree)
    #             property_colormap = 'glasbey' if property_name == '' else self.property_colormaps.get(property_name,'grey')
    #             self.world.add(mesh,world_object.name+"_"+self.element_names[display_degree],colormap=property_colormap,position=world_object.get('position'))
    #         else:
    #             self.world.remove(world_object.name+"_"+self.element_names[display_degree])

    # def _property_degree_changed(self, world_object, display_degree, old, new):
    #     if world_object:





