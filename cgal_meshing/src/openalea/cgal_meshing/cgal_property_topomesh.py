# -*- coding: utf-8 -*-
# -*- python -*-
#
#       CGALMeshing
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


def create_CGAL_topomesh(img, mesh_fineness=1.0):
    """
    """

    dirname = str(shared_data(openalea.cgal_meshing).parent.parent+'/')

    start_time = time()
    print "--> Input Image"

    img_graph = graph_from_image(img,spatio_temporal_properties=['volume','barycenter'],ignore_cells_at_stack_margins = False,property_as_real=False)
    img_labels = np.array(list(img_graph.vertices()))
    img_volumes = np.array([img_graph.vertex_property('volume')[v] for v in img_graph.vertices()])
    print "  --> ",img_labels.shape[0]," Objects, ", img_volumes[2:].mean()," microm3 per object"

    facet_distance = np.power(img_volumes[2:].mean(),1/3.)/(4.5*np.pi*mesh_fineness)
    print "  --> Facet Distance : ",facet_distance

    print "  --> Converting to 8bit"
    img_neighbors = np.array([list(img_graph.neighbors(v)) for v in img_graph.vertices()])

    count_labels = np.zeros(256)
    img_graph.add_vertex_property('8bit_labels')

    if img_labels.shape[0] < 255:
        for i,v in enumerate(img_graph.vertices()):
            img_graph.vertex_property('8bit_labels')[v] = i+2
    else:
        for v in img_graph.vertices():
            possible_labels = np.array([])
            select = 0
            while possible_labels.size == 0:
                possible_labels = set(np.arange(2,256))
                possible_labels = possible_labels.difference(set(np.where(count_labels>count_labels[2:].min()+select)[0]))
                #possible_labels = possible_labels.difference(set(np.where(count_labels>count_labels[2:].min()+1)[0]))
                neighbor_labels = set()
                for n in img_graph.neighbors(v):
                    try:
                        neighbor_label = {img_graph.vertex_property('8bit_labels')[n]}
                        for n2 in img_graph.neighbors(n):
                            if n2 != v:
                                try:
                                    neighbor_label = {img_graph.vertex_property('8bit_labels')[n2]}
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
            img_graph.vertex_property('8bit_labels')[v] = new_label
            count_labels[new_label] += 1
            #print v, ' -> ', new_label

        new_labels = np.ones(img.max()+1,np.uint8)
        new_labels[img_labels] = np.array([img_graph.vertex_property('8bit_labels')[v] for v in img_graph.vertices()],np.uint8)

        if np.unique(new_labels).shape[0]<255:
            label_index = np.ones(256)
            label_index[np.unique(new_labels)] = np.arange(new_labels.shape[0])+1
            for v in img_graph.vertices():
                img_graph.vertex_property('8bit_labels')[v] = label_index[img_graph.vertex_property('8bit_labels')[v]]

    new_labels = np.ones(img.max()+1,np.uint8)
    #new_labels[img_labels] = np.array([img_graph.vertex_property('8bit_labels')[v] for v in img_graph.vertices()],np.uint8)
    for v in img_graph.vertices():
        new_labels[v] = img_graph.vertex_property('8bit_labels')[v]

    mod_img = np.array(new_labels[img],np.uint8)
    inrfile = dirname+"tmp/8bit_image.inr.gz"
    imsave(inrfile,SpatialImage(mod_img))

    end_time = time()
    print "<-- Input Image              [",end_time-start_time,"s]"

    facet_angle = 30.0
    facet_size = 40.0
    edge_ratio = 4.0
    cell_size = 60.0

    start_time = time()
    print "--> Building CGAL Mesh"

    outputfile = dirname+"tmp/CGAL_output_mesh.mesh"
    buildCGALMesh(inrfile,outputfile,facet_angle,facet_size,facet_distance,edge_ratio,cell_size)
    end_time = time()
    print "<-- Building CGAL Mesh       [",end_time-start_time,"s]"

    mesh = CGALMesh()
    mesh.createMesh(outputfile)

    start_time = time()
    print "--> Re-Indexing Components"
    mesh.components = np.unique(mesh.tri_subdomains)

    new_mesh = CGALMesh()
    new_mesh.createMesh(outputfile)
    new_mesh.tri_subdomains = np.ones_like(mesh.tri_subdomains)
    new_mesh.tetra_subdomains = np.ones_like(mesh.tetra_subdomains)

    new_mesh.tri_subdomains[np.where(mesh.tri_subdomains == mesh.components[0])] = 1
    new_mesh.tetra_subdomains[np.where(mesh.tetra_subdomains == mesh.components[0])] = 1

    for c in mesh.components[1:]:
        cell_labels = np.where(new_labels == c)[0]
        n_cells = cell_labels.size

        if  n_cells > 0:
            # print "  --> Component ",c," -> ",n_cells," Cells"
            cell_tetrahedra = mesh.tetrahedra[np.where(mesh.tetra_subdomains==c)]

            #if n_cells == 1:
            if False:
                print "  --> Component ",c," ->  1  Object (",n_cells," Cell) : ",cell_labels,"(",img_graph.vertex_property('8bit_labels')[cell_labels[0]],")"
                new_mesh.tetra_subdomains[np.where(mesh.tetra_subdomains == c)] = cell_labels[0]
                new_mesh.tri_subdomains[np.where(mesh.tri_subdomains == c)] = cell_labels[0]
            else:
                cell_tetrahedra_components = np.zeros(cell_tetrahedra.shape[0],int)
                tetrahedra_component_correspondance = {}

                tetra_centers = np.mean(mesh.vertices[cell_tetrahedra],axis=1)

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

                img_points = np.array([img_graph.vertex_property('barycenter')[v] for v in cell_labels])
                tetrahedra_labels = cell_labels[vq(tetrahedra_object_centers,img_points)[0]]

                # img_points = np.array([img_graph.vertex_property('barycenter')[v] for v in img_labels])
                # tetrahedra_labels = img_labels[vq(tetrahedra_object_centers,img_points)[0]]

                cell_triangles = mesh.triangles[np.where(mesh.tri_subdomains==c)]
                cell_triangles_components = np.array([np.unique(cell_tetrahedra_components[np.where(cell_tetrahedra==tri[0])[0]])[0] for tri in cell_triangles])

                print "  --> Component ",c," -> ",n_objects," Objects (",n_cells," Cells) : ",tetrahedra_labels

                new_mesh.tetra_subdomains[np.where(mesh.tetra_subdomains == c)] = tetrahedra_labels[cell_tetrahedra_components]
                new_mesh.tri_subdomains[np.where(mesh.tri_subdomains == c)] = tetrahedra_labels[cell_triangles_components]

    mesh.tri_subdomains = new_mesh.tri_subdomains
    mesh.tetra_subdomains = new_mesh.tetra_subdomains
            
    mesh.components = np.unique(mesh.tri_subdomains)
    print mesh.vertices.shape[0],"Vertices, ",mesh.triangles.shape[0],"Triangles, ",mesh.tetrahedra.shape[0],"Tetrahedra, ",mesh.components.shape[0]," Components"
    end_time = time()
    print "<-- Re-Indexing Components   [",end_time-start_time,"s]"
    mesh.saveCGALfile(outputfile)

    start_time = time()
    print "--> Creating Topomesh"
    mesh = CGALMesh()
    mesh.createMesh(outputfile)
    mesh.generatePropertyTopomesh()

    positions = array_dict(mesh.vertex_positions.values()*np.array(img.resolution),keys=list(mesh.topo_mesh.wisps(0)))
    compute_topomesh_property(mesh.topo_mesh,'barycenter',degree=0,positions=positions)
    
    end_time = time()
    print "<-- Creating Topomesh        [",end_time-start_time,"s]"

    return mesh.topo_mesh


