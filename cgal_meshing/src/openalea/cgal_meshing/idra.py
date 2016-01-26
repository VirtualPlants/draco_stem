# -*- coding: utf-8 -*-
# -*- python -*-
#
#       CGALMeshing - IDRA
#       Interfaces of Delaunay-Refinement tetrahedrA
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
from scipy import ndimage as nd
import os

from scipy.cluster.vq import kmeans, vq

from openalea.deploy.shared_data import shared_data
from openalea.image.spatial_image import SpatialImage
from openalea.image.serial.all import imread, imsave

from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image
from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis

from openalea.container import array_dict
from openalea.mesh import PropertyTopomesh
from openalea.mesh.property_topomesh_analysis import compute_topomesh_property

import openalea.cgal_meshing
from openalea.cgal_meshing import buildCGALMesh
from openalea.cgal_meshing.cgal_mesh import CGALMesh

from time import time
from copy import deepcopy

class IdraMesh(object):
    """
    IDRA - Interfaces of Delaunay-Refinement tetrahedrA
    """

    def __init__(self, image=None, image_file=None, mesh_fineness=1.0):
        """
        """

        start_time = time()
        print "--> Input Image"

        if image is not None:
            self.segmented_image = deepcopy(image)
        else:
            self.segmented_image = imread(image_file)
        self.segmented_image[self.segmented_image==0] = 1
        self.resolution = np.array(self.segmented_image.resolution)
        self.size = np.array(self.segmented_image.shape)

        self.mesh_fineness = mesh_fineness
        self.facet_distance = 2.0

        self.dirname = str(shared_data(openalea.cgal_meshing).parent.parent+'/')
        
        self.new_labels = None
        self.modified_image_file = ""
        self.modified_image = None

        self.cgal_mesh_file = ""
        self.cgal_mesh = CGALMesh()
        self.topomesh = None

        self.image_graph = None
        self.image_labels = None
        self.image_cell_volumes = None

        self.compute_image_graph()
        self.create_8bit_image()

        end_time = time()
        print "<-- Input Image              [",end_time-start_time,"s]"


    def compute_image_graph(self):
        self.image_graph = graph_from_image(self.segmented_image,spatio_temporal_properties=['volume','barycenter'],ignore_cells_at_stack_margins=False,property_as_real=True)
        self.image_labels = np.array(list(self.image_graph.vertices()))
        self.image_cell_volumes = np.array([self.image_graph.vertex_property('volume')[v] for v in self.image_graph.vertices()])
        print "  --> ",self.image_labels.shape[0]," Objects, ", self.image_cell_volumes[2:].mean()/np.prod(self.resolution)," voxels per object"


    def create_8bit_image(self):
        print "  --> Converting to 8bit"
        img_neighbors = np.array([list(self.image_graph.neighbors(v)) for v in self.image_graph.vertices()])

        count_labels = np.zeros(256)
        self.image_graph.add_vertex_property('8bit_labels')

        if self.image_labels.shape[0] < 255:
            for i,v in enumerate(self.image_graph.vertices()):
                self.image_graph.vertex_property('8bit_labels')[v] = i+2
        else:
            for v in self.image_graph.vertices():
                possible_labels = np.array([])
                select = 0
                while possible_labels.size == 0:
                    possible_labels = set(np.arange(2,256))
                    possible_labels = possible_labels.difference(set(np.where(count_labels>count_labels[2:].min()+select)[0]))
                    #possible_labels = possible_labels.difference(set(np.where(count_labels>count_labels[2:].min()+1)[0]))
                    neighbor_labels = set()
                    for n in self.image_graph.neighbors(v):
                        try:
                            neighbor_label = {self.image_graph.vertex_property('8bit_labels')[n]}
                            for n2 in self.image_graph.neighbors(n):
                                if n2 != v:
                                    try:
                                        neighbor_label = {self.image_graph.vertex_property('8bit_labels')[n2]}
                                    except KeyError:
                                        neighbor_label = set()
                                        pass
                                    neighbor_labels = neighbor_labels.union(neighbor_label)
                        except KeyError:
                            neighbor_label = set()
                            pass
                        neighbor_labels = neighbor_labels.union(neighbor_label)
                    possible_labels = np.array(list(possible_labels.difference(neighbor_labels)),np.uint8)
                    if possible_labels.size == 0:
                        select += 1
                        #print neighbor_labels
                new_label = possible_labels[np.random.randint(possible_labels.size)]
                self.image_graph.vertex_property('8bit_labels')[v] = new_label
                count_labels[new_label] += 1
                #print v, ' -> ', new_label

            self.new_labels = np.ones(self.image_labels.max()+1,np.uint8)
            self.new_labels[self.image_labels] = np.array([self.image_graph.vertex_property('8bit_labels')[v] for v in self.image_graph.vertices()],np.uint8)

            if np.unique(self.new_labels).shape[0]<255:
                label_index = np.ones(256)
                label_index[np.unique(self.new_labels)] = np.arange(self.new_labels.shape[0])+1
                for v in self.image_graph.vertices():
                    self.image_graph.vertex_property('8bit_labels')[v] = label_index[self.image_graph.vertex_property('8bit_labels')[v]]

        self.new_labels = np.ones(self.image_labels.max()+1,np.uint8)
        for v in self.image_graph.vertices():
            self.new_labels[v] = self.image_graph.vertex_property('8bit_labels')[v]
        print [v for v in self.image_graph.vertices()]
        print self.image_labels
        print self.new_labels[self.image_labels]
        print np.sort(np.unique(self.new_labels[self.image_labels]))
        print self.image_labels.shape, np.sort(np.unique(self.new_labels[self.image_labels])).shape
        raw_input()

        self.modified_image = np.array(self.new_labels[self.segmented_image],np.uint8)
        self.modified_image_file = self.dirname+"tmp/8bit_image.inr.gz"
        imsave(self.modified_image_file,SpatialImage(self.modified_image))


    def build_cgal_mesh(self, mesh_fineness=None):
        facet_angle = 30.0
        facet_size = 40.0
        edge_ratio = 4.0
        cell_size = 60.0

        if mesh_fineness is not None:
            self.mesh_fineness = mesh_fineness

        self.facet_distance = np.power(self.image_cell_volumes[2:].mean()/np.prod(self.resolution),1/3.)/(4.5*np.pi*self.mesh_fineness)
        print "  --> Facet Distance : ",self.facet_distance

        start_time = time()
        print "--> Building CGAL Mesh"

        self.cgal_mesh_file = self.dirname+"tmp/CGAL_output_mesh.mesh"
        buildCGALMesh(self.modified_image_file, self.cgal_mesh_file, facet_angle, facet_size, self.facet_distance, edge_ratio, cell_size)
        end_time = time()
        print "<-- Building CGAL Mesh       [",end_time-start_time,"s]"

        self.cgal_mesh = CGALMesh()
        self.cgal_mesh.createMesh(self.cgal_mesh_file)


    def reindex_cgal_mesh_components(self):
        start_time = time()
        print "--> Re-Indexing Components"
        self.cgal_mesh.components = np.unique(self.cgal_mesh.tri_subdomains)

        new_mesh = CGALMesh()
        new_mesh.createMesh(self.cgal_mesh_file)
        new_mesh.tri_subdomains = np.ones_like(self.cgal_mesh.tri_subdomains)
        new_mesh.tetra_subdomains = np.ones_like(self.cgal_mesh.tetra_subdomains)

        new_mesh.tri_subdomains[np.where(self.cgal_mesh.tri_subdomains == self.cgal_mesh.components[0])] = 1
        new_mesh.tetra_subdomains[np.where(self.cgal_mesh.tetra_subdomains == self.cgal_mesh.components[0])] = 1

        for c in self.cgal_mesh.components[1:]:
            cell_labels = np.where(self.new_labels == c)[0]
            n_cells = cell_labels.size

            if  n_cells > 0:
                # print "  --> Component ",c," -> ",n_cells," Cells"
                cell_tetrahedra = self.cgal_mesh.tetrahedra[np.where(self.cgal_mesh.tetra_subdomains==c)]
                
                if False:
                    print "  --> Component ",c," ->  1  Object (",n_cells," Cell) : ",cell_labels,"(",self.image_graph.vertex_property('8bit_labels')[cell_labels[0]],")"
                    new_mesh.tetra_subdomains[np.where(self.cgal_mesh.tetra_subdomains == c)] = cell_labels[0]
                    new_mesh.tri_subdomains[np.where(self.cgal_mesh.tri_subdomains == c)] = cell_labels[0]
                else:
                    cell_tetrahedra_components = np.zeros(cell_tetrahedra.shape[0],int)
                    tetrahedra_component_correspondance = {}

                    tetra_centers = np.mean(self.cgal_mesh.vertices[cell_tetrahedra],axis=1)

                    for t, tetra in enumerate(cell_tetrahedra):
                        if cell_tetrahedra_components[t] == 0:
                            neighbour_tetrahedra = np.unique(np.append(np.where(cell_tetrahedra==tetra[0])[0],np.append(np.where(cell_tetrahedra==tetra[1])[0],np.append(np.where(cell_tetrahedra==tetra[2])[0],np.where(cell_tetrahedra==tetra[3])[0]))))
                            if (cell_tetrahedra_components[neighbour_tetrahedra].max()>0):
                                neighbour_components = cell_tetrahedra_components[neighbour_tetrahedra][np.where(cell_tetrahedra_components[neighbour_tetrahedra]>0)]
                                min_component = np.array([tetrahedra_component_correspondance[component] for component in neighbour_components]).min()
                                for component in neighbour_components:
                                    tetrahedra_component_correspondance[tetrahedra_component_correspondance[component]] = min_component
                                    tetrahedra_component_correspondance[component] = min_component
                                cell_tetrahedra_components[neighbour_tetrahedra] = min_component
                            else:
                                tetrahedra_component_correspondance[cell_tetrahedra_components.max()+1] = int(cell_tetrahedra_components.max()+1)
                                cell_tetrahedra_components[neighbour_tetrahedra] = int(cell_tetrahedra_components.max()+1)

                    for component in tetrahedra_component_correspondance:
                        label = component
                        while label != tetrahedra_component_correspondance[label]:
                            label = tetrahedra_component_correspondance[label]
                            tetrahedra_component_correspondance[component] = tetrahedra_component_correspondance[label]

                    component_labels = np.unique([tetrahedra_component_correspondance[component] for component in cell_tetrahedra_components])
                    n_objects = component_labels.size

                    # if n_objects != n_cells:
                        # print tetrahedra_component_correspondance

                    for component in tetrahedra_component_correspondance:
                        tetrahedra_component_correspondance[component] = np.where(component_labels == tetrahedra_component_correspondance[component])[0][0]

                    for component in tetrahedra_component_correspondance:
                        cell_tetrahedra_components[np.where(cell_tetrahedra_components == component)] = tetrahedra_component_correspondance[component]

                    tetrahedra_object_centers = np.zeros((n_objects,3))

                    tetrahedra_object_centers[:,0] = nd.sum(tetra_centers[:,0],cell_tetrahedra_components,index=xrange(n_objects))/nd.sum(np.ones_like(cell_tetrahedra_components),cell_tetrahedra_components,index=xrange(n_objects))
                    tetrahedra_object_centers[:,1] = nd.sum(tetra_centers[:,1],cell_tetrahedra_components,index=xrange(n_objects))/nd.sum(np.ones_like(cell_tetrahedra_components),cell_tetrahedra_components,index=xrange(n_objects))
                    tetrahedra_object_centers[:,2] = nd.sum(tetra_centers[:,2],cell_tetrahedra_components,index=xrange(n_objects))/nd.sum(np.ones_like(cell_tetrahedra_components),cell_tetrahedra_components,index=xrange(n_objects))

                    img_points = np.array([self.image_graph.vertex_property('barycenter')[v]/self.resolution for v in cell_labels])
                    tetrahedra_labels = cell_labels[vq(tetrahedra_object_centers,img_points)[0]]

                    # img_points = np.array([self.image_graph.vertex_property('barycenter')[v]/self.resolution for v in self.image_labels])
                    # tetrahedra_labels = self.image_labels[vq(tetrahedra_object_centers,img_points)[0]]

                    cell_triangles = self.cgal_mesh.triangles[np.where(self.cgal_mesh.tri_subdomains==c)]
                    cell_triangles_components = np.array([np.unique(cell_tetrahedra_components[np.where(cell_tetrahedra==tri[0])[0]])[0] for tri in cell_triangles])

                    print "  --> Component ",c," -> ",n_objects," Objects (",n_cells," Cells) : ",tetrahedra_labels

                    new_mesh.tetra_subdomains[np.where(self.cgal_mesh.tetra_subdomains == c)] = tetrahedra_labels[cell_tetrahedra_components]
                    new_mesh.tri_subdomains[np.where(self.cgal_mesh.tri_subdomains == c)] = tetrahedra_labels[cell_triangles_components]

        self.cgal_mesh.tri_subdomains = new_mesh.tri_subdomains
        self.cgal_mesh.tetra_subdomains = new_mesh.tetra_subdomains
                
        self.cgal_mesh.components = np.unique(self.cgal_mesh.tri_subdomains)
        print self.cgal_mesh.vertices.shape[0],"Vertices, ",self.cgal_mesh.triangles.shape[0],"Triangles, ",self.cgal_mesh.tetrahedra.shape[0],"Tetrahedra, ",self.cgal_mesh.components.shape[0]," Components"
        end_time = time()
        print "<-- Re-Indexing Components   [",end_time-start_time,"s]"
        self.cgal_mesh.saveCGALfile(self.cgal_mesh_file)


    def create_topomesh(self):
        start_time = time()
        print "--> Creating Topomesh"
        self.cgal_mesh = CGALMesh()
        self.cgal_mesh.createMesh(self.cgal_mesh_file)
        self.cgal_mesh.generatePropertyTopomesh()
        self.topomesh = deepcopy(self.cgal_mesh.topo_mesh)

        positions = array_dict(self.cgal_mesh.vertex_positions.values()*self.resolution,keys=list(self.topomesh.wisps(0)))
        compute_topomesh_property(self.topomesh,'barycenter',degree=0,positions=positions)
        
        end_time = time()
        print "<-- Creating Topomesh        [",end_time-start_time,"s]"

        return self.topomesh

    def idra_topomesh(self, mesh_fineness=None):
        """
        Compute the cell interface topomesh following the standard IDRA process
        1/ Compute CGAL tetrahedral mesh of the 8bit image with the fineness settings
        2/ Re-label the tetrahedra with their original image labels
        3/ Return the interface mesh as a PropertyTopomesh
        """
        self.build_cgal_mesh(mesh_fineness)
        self.reindex_cgal_mesh_components()
        return self.create_topomesh()


def create_idra_topomesh(image, mesh_fineness=1.0):
    idra = IdraMesh(image,mesh_fineness)
    return idra.idra_topomesh()


