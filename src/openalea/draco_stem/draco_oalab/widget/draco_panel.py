# -*- coding: utf-8 -*-
# -*- python -*-
#
#       DRACO-STEM
#       Dual Reconstruction by Adjacency Complex Optimization
#       SAM Tissue Enhanced Mesh
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
from openalea.core.control.manager import ControlContainer
from openalea.oalab.control.manager import ControlManagerWidget
from openalea.core.service.ipython import interpreter as get_interpreter
from openalea.deploy.shared_data import shared_data

from openalea.oalab.service.drag_and_drop import add_drop_callback
from openalea.oalab.widget.world import WorldModel
from openalea.oalab.colormap.colormap_def import load_colormaps

from copy import copy, deepcopy
import os

import openalea.draco_stem
from openalea.draco_stem.draco import DracoMesh
from openalea.draco_stem.draco.draco import draco_initialization
from openalea.draco_stem.draco.draco import draco_delaunay_adjacency_complex, draco_layer_adjacency_complex, draco_construct_adjacency_complex
from openalea.draco_stem.draco.draco import draco_adjacency_complex_optimization, draco_dual_reconstruction

from openalea.image.serial.all import imread


cst_adjacency = dict(enum=['','delaunay','L1','L2','L1-L2'])
cst_dualization = dict(enum=['star','star_split_regular','star_remeshed_projected_flat'])
cst_iterations = dict(step=1, min=0, max=10)

control_definition = {}
control_definition['img_filename'] = dict(value="",interface="IFileStr",constraints={},label="Image File")
control_definition['cell_vertex_filename'] = dict(value="",interface="IFileStr",constraints={},label="Cell Vertex File")
# control_definition['triangulation_filename'] = dict(value="",interface="IFileStr",constraints={},label="Triangulation File")

control_definition['initialize'] = dict(value=(lambda:None),interface="IAction",constraints={},label="Initialize Draco")
control_definition['display_img'] = dict(value=False,interface="IBool",constraints={},label="Display Image")
control_definition['display_cells'] = dict(value=False,interface="IBool",constraints={},label="Display Cells")

control_definition['adjacency_complex'] = dict(value='',interface='IEnumStr',constraints=cst_adjacency,label="Adjacency Complex")
control_definition['compute_adjacency'] = dict(value=(lambda:None),interface="IAction",constraints={},label="Compute Adjacency")
control_definition['display_adjacency'] = dict(value=False,interface="IBool",constraints={},label="Display Adjacency")

control_definition['n_iterations'] = dict(value=0,interface="IInt",constraints=cst_iterations,label="Optimization Iterations")
control_definition['optimize_adjacency'] = dict(value=(lambda:None),interface="IAction",constraints={},label="Optimize Adjacency")

control_definition['triangulation'] = dict(value='',interface='IEnumStr',constraints=cst_dualization,label="Dual Triangulation")
control_definition['dualize'] = dict(value=(lambda:None),interface="IAction",constraints={},label="Reconstruct Dual")
control_definition['display_dual'] = dict(value=False,interface="IBool",constraints={},label="Display Dual Mesh")



def set_default_control(control_list, control_name, control_definition=control_definition, value=None):
    control_names = [c['name'] for c in control_list]

    if control_name in control_definition:
        try:
            control = control_list[control_names.index(control_name)]
            for field in control_definition[control_name].keys():
                control[field] = control_definition[control_name][field]
            if value is not None:
                control['value'] = value
        except ValueError:
            control = copy(control_definition[control_name])
            control['name'] = control_name
            if value is not None:
                control['value'] = value
            control_list.append(control)



class DracoPanel(QtGui.QWidget, AbstractListener):

    def __init__(self, parent=None, style=None):
        AbstractListener.__init__(self)
        QtGui.QWidget.__init__(self, parent=parent)

        self.world = None

        self.name = ""
        self._controls = []

        self._manager = ControlContainer()

        self.interpreter = get_interpreter()
        self.interpreter.locals['draco_control'] = self

        self._layout = QtGui.QVBoxLayout(self)

        self._title_img = QtGui.QWidget()
        title_layout = QtGui.QHBoxLayout(self._title_img)

        p = QtGui.QSizePolicy
        pixmap_dirname = shared_data(openalea.draco_stem)

        icon_img = QtGui.QLabel()
        pixmap_icon = QtGui.QPixmap(os.path.join(pixmap_dirname,"../../src/openalea/draco_stem/draco_oalab/widget/draco_icon.png"))
        icon_img.setPixmap(pixmap_icon)
        icon_img.setScaledContents(True)
        icon_img.setFixedWidth(60)
        icon_img.setFixedHeight(60)
        # icon_img.setSizePolicy(p(p.Expanding, p.Maximum))
        title_layout.addWidget(icon_img)
        # title_layout.addSpacing(20)

        title_img = QtGui.QLabel()
        pixmap_title = QtGui.QPixmap(os.path.join(pixmap_dirname,"../../src/openalea/draco_stem/draco_oalab/widget/draco_title.png"))
        title_img.setPixmap(pixmap_title)
        title_img.setScaledContents(True)
        # title_img.setSizePolicy(p(p.Expanding, p.Maximum))
        title_img.setFixedWidth(140)
        title_img.setFixedHeight(60)
        title_layout.addWidget(title_img)
        # title_layout.addSpacing(20)

        title_label = QtGui.QLabel(u'Dual Reconstruction\nby Adjacency\nComplex Optimization')
        title_label.setAlignment(QtCore.Qt.AlignCenter)
        # title_label.setFixedWidth(150)
        title_layout.addWidget(title_label)


        self._title_img.setFixedHeight(75)
        self._title_img.setSizePolicy(p(p.Expanding, p.Maximum))
        self._layout.addWidget(self._title_img,0)

        self._view = None
        self._set_manager(self._manager)

        self.draco = None


    def initialize(self):
        from openalea.core.world.world import World
        from openalea.core.service.ipython import interpreter
        world = World()
        world.update_namespace(interpreter())
        self.set_world(world)

        set_default_control(self._controls,'img_filename')
        set_default_control(self._controls,'cell_vertex_filename')
        # set_default_control(self._controls,'triangulation_filename')

        set_default_control(self._controls,'initialize')

        self._fill_manager()

        self._set_manager(self._manager)


    def set_world(self, world):
        self.world = world
        # self.world.register_listener(self)

    def _set_manager(self, manager):
        view = self._view
        if self._view is not None:
            view = self._view()
        if view:
            self._layout.removeWidget(view)
            view.close()
            del view
        from openalea.oalab.service.qt_control import edit
        import weakref
        view = edit(manager)
        view.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self._view = weakref.ref(view)
        # self._view = view
        self._layout.addWidget(view)

        view.show()
        self.repaint()

    def _fill_manager(self):
        for control in self._controls:

            self._manager.add(
                control['name'],
                interface=control['interface'],
                value=control['value'],
                label=control['label'],
                constraints=control['constraints']
            )
            self._manager.register_follower(control['name'], self._control_changed(control['name']))

    def refresh_manager(self):
        manager_names = [c.name for c in self._manager.controls()]
        control_names = [c['name'] for c in self._controls]
        if manager_names != control_names:
            self._manager.clear_followers()
            self._manager.clear()
            self._fill_manager()
            self._set_manager(self._manager)
            self._manager.enable_followers()
        else:
            for c in self._controls:
                if c['value'] != self._manager.control(c['name']).value:
                    self._manager.control(c['name']).set_value(c['value'])


    def _control_changed(self, control_name):
        def _changed(old, new):
            self._draco_control_changed(control_name, old, new)
        return _changed


    def _draco_control_changed(self, control_name, old, new):
        print control_name," changed! : ",new

        control_names = [c['name'] for c in self._controls]
        self._controls[control_names.index(control_name)]['value'] = new

        if control_name == 'img_filename':
            img_file = new
            try:
                img = imread(img_file)
                assert img.ndim == 3
            except:
                print "Image type not recognized! Please choose a different file..."
                set_default_control(self._controls,'initialize')
            else:
                self.name = os.path.split(img_file)[-1].split('.')[0]
                set_default_control(self._controls,'initialize',value=self._init_button_pressed)
        
        elif control_name == 'display_img':
            if new:
                self.world.add(self.draco.segmented_image,self.name+'_segmented_image',colormap='glasbey',alphamap='constant',bg_id=1)
            else:
                if self.world.has_key(self.name+'_segmented_image'):
                    self.world.remove(self.name+'_segmented_image')

        elif control_name == 'display_cells':
            if new:
                self.world.add(self.draco.point_topomesh,self.name+'_image_cells')
                self.world[self.name+'_image_cells_vertices'].set_attribute('point_radius',self.draco.segmented_image.max())
                self.world[self.name+'_image_cells_vertices'].set_attribute('display_colorbar',False)
            else:
                if self.world.has_key(self.name+'_image_cells'):
                    self.world.remove(self.name+'_image_cells')

        elif control_name == 'display_adjacency':
            control_names = [c['name'] for c in self._controls]
            adjacency_complex = self._controls[control_names.index('adjacency_complex')]['value']
            if adjacency_complex in ['delaunay','L1-L2']:
                degree = 3
            else:
                degree = 2

            if new:
                self.world.add(self.draco.triangulation_topomesh,self.name+'_adjacency_complex')
                if degree == 3:
                    self.world[self.name+'_adjacency_complex_cells'].set_attribute('polydata_colormap',load_colormaps()['grey'])
                    self.world[self.name+'_adjacency_complex_cells'].set_attribute('intensity_range',(-1,0))
                    self.world[self.name+'_adjacency_complex'].set_attribute('coef_3',0.95)
                    self.world[self.name+'_adjacency_complex_cells'].set_attribute('display_colorbar',False)
                else:
                    self.world[self.name+'_adjacency_complex'].set_attribute('display_3',False)
                    self.world[self.name+'_adjacency_complex'].set_attribute('display_2',True)
                    self.world[self.name+'_adjacency_complex_faces'].set_attribute('polydata_colormap',load_colormaps()['grey'])
                    self.world[self.name+'_adjacency_complex_faces'].set_attribute('intensity_range',(-1,0))
                    self.world[self.name+'_adjacency_complex'].set_attribute('coef_2',0.98)
                    self.world[self.name+'_adjacency_complex_faces'].set_attribute('display_colorbar',False)
            else:
                if self.world.has_key(self.name+'_adjacency_complex'):
                    self.world.remove(self.name+'_adjacency_complex')

        elif control_name == 'display_dual':
            if new:
                self.world.add(self.draco.dual_reconstruction_topomesh,self.name+'_dual_reconstruction')
            else:
                if self.world.has_key(self.name+'_dual_reconstruction'):
                    self.world.remove(self.name+'_dual_reconstruction')


        elif control_name == 'adjacency_complex':
            if new == '':
                print "Please define a mode for adjacency complex computation"
                set_default_control(self._controls,'compute_adjacency')
            else:
                set_default_control(self._controls,'compute_adjacency',value=self._compute_adjacency_button_pressed)

        self.refresh_manager()


    def _init_button_pressed(self):
        control_names = [c['name'] for c in self._controls]
        img_file = self._controls[control_names.index('img_filename')]['value']
        cell_vertex_file = self._controls[control_names.index('cell_vertex_filename')]['value']
        print "Initializing Draco Mesh : ",img_file,"(",self.name,")" 

        if cell_vertex_file != "":
            self.draco = draco_initialization(image=None, image_file=img_file, image_cell_vertex_file=cell_vertex_file)
        else:
            self.draco = draco_initialization(image=None, image_file=img_file)

        set_default_control(self._controls,'display_img')
        set_default_control(self._controls,'display_cells')

        set_default_control(self._controls,'adjacency_complex')
        set_default_control(self._controls,'compute_adjacency')
        self.refresh_manager()


    def _compute_adjacency_button_pressed(self):
        control_names = [c['name'] for c in self._controls]
        adjacency_complex = self._controls[control_names.index('adjacency_complex')]['value']
        print "Computing Adjacency Complex : ",adjacency_complex,"(",self.name,")" 

        if adjacency_complex == 'delaunay':
            self.draco = draco_delaunay_adjacency_complex(self.draco, surface_cleaning_criteria=['surface','sliver','distance'])
        elif adjacency_complex == 'L1':
            self.draco = draco_layer_adjacency_complex(self.draco, 'L1')
        elif adjacency_complex == 'L2':
            self.draco = draco_layer_adjacency_complex(self.draco, 'L2')
        else:
            self.draco = draco_construct_adjacency_complex(self.draco)

        set_default_control(self._controls,'display_adjacency')

        set_default_control(self._controls,'n_iterations')
        set_default_control(self._controls,'optimize_adjacency',value=self._optimize_adjacency_button_pressed)
        self.refresh_manager()


    def _optimize_adjacency_button_pressed(self):
        control_names = [c['name'] for c in self._controls]
        n_iterations = self._controls[control_names.index('n_iterations')]['value']
        print "Optimizing Adjacency Complex : ",n_iterations,"(",self.name,")" 

        adjacency_complex = self._controls[control_names.index('adjacency_complex')]['value']

        if adjacency_complex == 'delaunay':
            self.draco = draco_adjacency_complex_optimization(self.draco, n_iterations=n_iterations)

        set_default_control(self._controls,'triangulation')
        set_default_control(self._controls,'dualize',value=self._dualize_button_pressed)
        self.refresh_manager()


    def _dualize_button_pressed(self):
        control_names = [c['name'] for c in self._controls]
        triangular = self._controls[control_names.index('triangulation')]['value']
        print "Reconstructing Dual : ",triangular,"(",self.name,")" 

        adjacency_complex = self._controls[control_names.index('adjacency_complex')]['value']

        if adjacency_complex in ['delaunay','L1-L2']:
            degree = 3
        else:
            degree = 2
        
        self.draco = draco_dual_reconstruction(self.draco, reconstruction_triangulation=triangular, adjacency_complex_degree=degree)

        set_default_control(self._controls,'display_dual')        
        self.refresh_manager()






