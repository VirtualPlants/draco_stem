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

from scipy.cluster.vq import kmeans, vq

from openalea.container.array_dict import array_dict
from openalea.container.property_topomesh import PropertyTopomesh
from openalea.mesh.property_topomesh_analysis import *
from openalea.mesh.utils.tissue_analysis_tools import cell_vertex_extraction
from openalea.mesh.utils.array_tools import array_unique
from openalea.container.topomesh_algo import is_collapse_topo_allowed, collapse_edge
from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image


from time                                   import time
from copy                                   import deepcopy
import os
import sys
import pickle


def evaluate_topomesh_quality(topomesh,quality_criteria=["Mesh Complexity","Triangle Area Deviation","Triangle Eccentricity","Cell Volume Error","Vertex Distance","Cell Convexity","Epidermis Cell Angle","Vertex Valence","Cell 2 Adjacency"],image=None,image_cell_vertex=None,image_labels=None,image_cell_volumes=None,**kwargs):
    """
    """

    quality_data = {}

    if "Mesh Complexity" in quality_criteria:
        start_time = time()
        print "--> Computing mesh complexity"
        quality_data["Mesh Complexity"] = np.minimum(1.0,(152.*topomesh.nb_wisps(3))/np.sum([len(list(topomesh.border_neighbors(3,c))) for c in topomesh.wisps(3)]))
        end_time = time()
        print "<-- Computing mesh complexity          [",end_time-start_time,"s]"

    compute_topomesh_property(topomesh,'length',degree=1)

    triangular = kwargs.get('triangular',True)
    display = kwargs.get('display',False)

    if triangular:
        compute_topomesh_triangle_properties(topomesh)
        compute_topomesh_property(topomesh,'normal',degree=2)
        compute_topomesh_property(topomesh,'angles',degree=2)

    if "Triangle Area Deviation" in quality_criteria:
        start_time = time()
        print "--> Computing triangle area deviation"
        # area_deviation = np.nanmean(np.abs(topomesh.wisp_property('area',degree=2).values()-np.nanmean(topomesh.wisp_property('area',degree=2).values())))/np.nanmean(topomesh.wisp_property('area',degree=2).values())
        area_deviation = np.nanstd(topomesh.wisp_property('area',degree=2).values())/np.nanmean(topomesh.wisp_property('area',degree=2).values())
        # quality_data["Triangle Area Deviation"] = np.minimum(1.0,1.0-area_deviation/np.sqrt(2))
        # quality_data["Triangle Area Deviation"] = np.minimum(1.0,1.0-area_deviation/2.)
        quality_data["Triangle Area Deviation"] = np.minimum(1.0,np.sqrt(2)-area_deviation/np.sqrt(2))
        end_time = time()
        print "<-- Computing triangle area deviation  [",end_time-start_time,"s]"

    if "Triangle Eccentricity" in quality_criteria:
        start_time = time()
        print "--> Computing triangle eccentricity"
        quality_data["Triangle Eccentricity"] = 1.-2.*np.nanmean(topomesh.wisp_property('eccentricity',degree=2).values())
        end_time = time()
        print "<-- Computing triangle eccentricity    [",end_time-start_time,"s]"

    compute_topomesh_property(topomesh,'borders',degree=3)
    compute_topomesh_property(topomesh,'borders',degree=2)
    compute_topomesh_property(topomesh,'borders',degree=1)

    compute_topomesh_property(topomesh,'vertices',degree=3)
    compute_topomesh_property(topomesh,'vertices',degree=2)

    compute_topomesh_property(topomesh,'barycenter',degree=3)
    compute_topomesh_property(topomesh,'barycenter',degree=2)
    compute_topomesh_property(topomesh,'barycenter',degree=1)

    compute_topomesh_property(topomesh,'triangles',degree=0)

    compute_topomesh_property(topomesh,'cells',degree=0)
    compute_topomesh_property(topomesh,'cells',degree=1)
    compute_topomesh_property(topomesh,'cells',degree=2)

    compute_topomesh_property(topomesh,'epidermis',degree=0)
    compute_topomesh_property(topomesh,'epidermis',degree=1)
    compute_topomesh_property(topomesh,'epidermis',degree=3)


    if "Cell Volume Error" in quality_criteria or "Cell Convexity" in quality_criteria:
        compute_topomesh_property(topomesh,'volume',degree=3)
        compute_topomesh_property(topomesh,'convexhull_volume',degree=3)
    
    img_graph = kwargs.get('image_graph',None)

    if "Cell Volume Error" in quality_criteria:
        start_time = time()
        print "--> Computing cell volume error"

        from vplants.tissue_analysis.temporal_graph_from_image   import graph_from_image
        if (image_cell_volumes == None) or (image_labels == None) or (image_cell_vertex == None):
            img_graph = graph_from_image(image, spatio_temporal_properties=['volume','barycenter'],background=0,ignore_cells_at_stack_margins = False,property_as_real=True)
            image_labels = np.array(list(img_graph.vertices()))
            image_cell_volumes = np.array([img_graph.vertex_property('volume')[v] for v in image_labels])
        else:
            img_graph = None

        img_volumes = array_dict(image_cell_volumes,image_labels)

        if triangular:
            volume_error = array_dict([(c,(topomesh.wisp_property('volume',degree=3)[c] - img_volumes[c])/img_volumes[c]) for c in topomesh.wisps(3)])
        else:
            volume_error = array_dict([(c,(topomesh.wisp_property('convexhull_volume',degree=3)[c] - img_volumes[c])/img_volumes[c]) for c in topomesh.wisps(3)])


        quality_data["Cell Volume Error"] = np.maximum(1.-abs(volume_error.values()).mean(),0.0)
        end_time = time()
        print "<-- Computing cell volume error        [",end_time-start_time,"s]"

    if "Image Accuracy" in quality_criteria:
        start_time = time()
        print "--> Computing image accuracy"

        from openalea.mesh.utils.image_tools import compute_topomesh_image

        topomesh_img = compute_topomesh_image(topomesh,image)

        cells_img = deepcopy(image)
        for c in set(np.unique(image)).difference(set(topomesh.wisps(3))):
            cells_img[image==c] = 1

        true_positives = ((cells_img != 1)&(cells_img == topomesh_img)).sum()
        false_positives = ((cells_img == 1) & (topomesh_img != 1)).sum() + ((cells_img != 1)&(topomesh_img != 1)&(cells_img != topomesh_img)).sum()
        false_negatives = ((cells_img != 1) & (topomesh_img == 1)).sum() + ((cells_img != 1)&(topomesh_img != 1)&(cells_img != topomesh_img)).sum()
        true_negatives = ((cells_img == 1)&(cells_img == topomesh_img)).sum()

        estimators = {}
        estimators['Precision'] = float(true_positives/float(true_positives+false_positives))
        estimators['Recall'] = float(true_positives/float(true_positives+false_negatives))
        estimators['Dice'] = float(2*true_positives/float(2*true_positives+false_positives+false_negatives))
        estimators['Jaccard'] = float(true_positives/float(true_positives+false_positives+false_negatives))
        estimators['Accuracy'] = float(true_positives+true_negatives)/float(true_positives+true_negatives+false_positives+false_negatives)
        estimators['Identity'] = float((cells_img == topomesh_img).sum())/np.prod(cells_img.shape)
        print estimators

        quality_data["Image Accuracy"] = estimators['Dice']

        end_time = time()
        print "<-- Computing image accuracy           [",end_time-start_time,"s]"

    vertex_cell_neighbours = topomesh.wisp_property('cells',degree=0)

    vertex_cell_degree = np.array(map(len,vertex_cell_neighbours.values()))

    epidermis_vertices = np.array(list(topomesh.wisps(0)))[np.where(topomesh.wisp_property('epidermis',degree=0).values())]

    start_time = time()
    print "--> Computing mesh cell vertices"
    mesh_cell_vertex = {}
    for v in topomesh.wisps(0):
        if len(vertex_cell_neighbours[v]) == 5:
            for k in xrange(5):
                vertex_cell_labels = tuple([c for c in vertex_cell_neighbours[v]][:k])+tuple([c for c in vertex_cell_neighbours[v]][k+1:])
                if not mesh_cell_vertex.has_key(vertex_cell_labels):
                    mesh_cell_vertex[vertex_cell_labels] = v
        if len(vertex_cell_neighbours[v]) == 4:
            vertex_cell_labels = tuple([c for c in vertex_cell_neighbours[v]])
            mesh_cell_vertex[vertex_cell_labels] = v
            if v in epidermis_vertices: 
                for k in xrange(4):
                    vertex_cell_labels = (1,) + tuple([c for c in vertex_cell_neighbours[v]][:k])+tuple([c for c in vertex_cell_neighbours[v]][k+1:])
                    if not mesh_cell_vertex.has_key(vertex_cell_labels):
                        mesh_cell_vertex[vertex_cell_labels] = v
        if (len(vertex_cell_neighbours[v]) == 3) and (v in epidermis_vertices):
            vertex_cell_labels = (1,) + tuple([c for c in vertex_cell_neighbours[v]])
            mesh_cell_vertex[vertex_cell_labels] = v
    end_time = time()
    print "<-- Computing mesh cell vertices [",end_time-start_time,"s]"

    cell_vertex = np.unique(mesh_cell_vertex.values())
    mesh_cell_vertices = topomesh.wisp_property('barycenter',degree=0).values(cell_vertex)  

    if "Vertex Distance" in quality_criteria:
        start_time = time()
        print "--> Computing vertex distance"
        if image_cell_vertex is None:
            image_cell_vertex = cell_vertex_extraction(image,hollow_out=False,verbose=False)
            if img_graph is None:
                img_graph = graph_from_image(image, spatio_temporal_properties=['volume','barycenter'],background=0,ignore_cells_at_stack_margins = False,property_as_real=True)
            img_edges = np.array([img_graph.edge_vertices(e) for e in img_graph.edges()])

            tetra_edge_index_list = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
            img_vertex_edges = np.concatenate(np.sort(image_cell_vertex.keys())[:,tetra_edge_index_list])
            vertex_edge_matching = (vq(img_vertex_edges,img_edges)[1] > 0).reshape(len(image_cell_vertex),6).sum(axis=1)
            img_cell_vertex = np.array(image_cell_vertex.keys())[np.where(vertex_edge_matching==0)[0]]

            image_cell_vertex = array_dict(np.array([image_cell_vertex[tuple(vertex)] for vertex in img_cell_vertex]),np.sort(img_cell_vertex))
        else:
            image_cell_vertex = deepcopy(image_cell_vertex)
        # for v in image_cell_vertex.keys():
        #     image_cell_vertex[v] = image_cell_vertex[v]*np.array(image.resolution)
    
        image_cell_vertices = np.array(image_cell_vertex.values())
        print mesh_cell_vertices[np.where(1-np.isnan(mesh_cell_vertices)[:,0])]*image.resolution
        print image_cell_vertices*image.resolution
        vertex_distances_mesh = array_dict(vq(mesh_cell_vertices[np.where(1-np.isnan(mesh_cell_vertices)[:,0])]*image.resolution,image_cell_vertices*image.resolution)[1],cell_vertex)
        vertex_distances_image = vq(image_cell_vertices[np.where(1-np.isnan(image_cell_vertices)[:,0])],mesh_cell_vertices)[1]

        # quality_data["Vertex Distance"] = np.sqrt(3)/(np.maximum(np.sqrt(3),vertex_distances_image.mean()))
        quality_data["Vertex Distance"] = (0.25*np.sqrt(3))/(np.maximum((0.25*np.sqrt(3)),vertex_distances_image.mean()))
        end_time = time()
        print "<-- Computing vertex distance          [",end_time-start_time,"s]"


    if "Cell Convexity" in quality_criteria:
        start_time = time()
        print "--> Computing cell convexity"
       
        # quality_data["Cell Convexity"] = (topomesh.wisp_property('volume',degree=3).values()/topomesh.wisp_property('convexhull_volume',degree=3).values()).mean()
        quality_data["Cell Convexity"] = 1 - (((topomesh.wisp_property('convexhull_volume',3).values() - topomesh.wisp_property('volume',3).values()).mean())/topomesh.wisp_property('volume',3).values().mean())
        end_time = time()
        print "<-- Computing cell convexity           [",end_time-start_time,"s]"


    if "Epidermis Cell Angle" in quality_criteria:
        start_time = time()
        print "--> Computing epidermis cell angle"
        epidermis_vertex_cell_angles=np.array([])
        epidermis_vertex_angles = array_dict()
        angle_keys = np.concatenate([np.concatenate([[(t,v)] for v in topomesh.wisp_property('vertices',2)[t]],axis=0) for t in topomesh.wisps(2)],axis=0)
        angle_dict = dict([(tuple(k),0.) for k in angle_keys])

        for v in cell_vertex:
            if topomesh.wisp_property('epidermis',0)[v]:
                vertex_triangles = np.array([t for t in topomesh.wisp_property('triangles',0)[v] if topomesh.wisp_property('epidermis',2)[t]])

                vertex_triangle_cells = np.concatenate(topomesh.wisp_property('cells',degree=2).values(vertex_triangles))
                vertex_cells = np.unique(vertex_triangle_cells)

                vertex_triangle_cell_directions = topomesh.wisp_property('barycenter',2).values(vertex_triangles) - topomesh.wisp_property('barycenter',3).values(vertex_triangle_cells)
                vertex_triangle_normals = topomesh.wisp_property('normal',2).values(vertex_triangles)
                reversed_normals = np.where(np.einsum('ij,ij->i',vertex_triangle_normals,vertex_triangle_cell_directions) < 0)[0]
                vertex_triangle_normals[reversed_normals] = -vertex_triangle_normals[reversed_normals]
                vertex_normal = np.mean(vertex_triangle_normals,axis=0)
                vertex_normal = vertex_normal/np.linalg.norm(vertex_normal)

                triangle_vertices = topomesh.wisp_property('vertices',degree=2).values(vertex_triangles)
                triangle_positions = topomesh.wisp_property('barycenter',degree=0).values(triangle_vertices)

                triangle_proj_vectors = triangle_positions - topomesh.wisp_property('barycenter',degree=0)[v]
                triangle_proj_dot = np.einsum('...ij,...j->...i',triangle_proj_vectors,vertex_normal)
                triangle_proj_vectors = -triangle_proj_dot[...,np.newaxis]*vertex_normal

                triangle_proj_positions = triangle_positions + triangle_proj_vectors

                edge_index_list = np.array([[1, 2],[0, 1],[0, 2]])

                triangle_edge_positions = triangle_proj_positions[:,edge_index_list]
                triangle_edge_vectors = triangle_edge_positions[:,:,1] - triangle_edge_positions[:,:,0]

                triangle_edge_lengths = np.linalg.norm(triangle_edge_vectors,axis=2)

                triangle_cosines = np.zeros_like(triangle_edge_lengths,np.float32)
                triangle_cosines[:,0] = (triangle_edge_lengths[:,1]**2+triangle_edge_lengths[:,2]**2-triangle_edge_lengths[:,0]**2)/(2.0*triangle_edge_lengths[:,1]*triangle_edge_lengths[:,2])
                triangle_cosines[:,2] = (triangle_edge_lengths[:,2]**2+triangle_edge_lengths[:,0]**2-triangle_edge_lengths[:,1]**2)/(2.0*triangle_edge_lengths[:,2]*triangle_edge_lengths[:,0])
                triangle_cosines[:,1] = (triangle_edge_lengths[:,0]**2+triangle_edge_lengths[:,1]**2-triangle_edge_lengths[:,2]**2)/(2.0*triangle_edge_lengths[:,0]*triangle_edge_lengths[:,1])
                triangle_angles = 180.*np.arccos(triangle_cosines)/np.pi

                vertex_triangle_angles = triangle_angles[np.where(triangle_vertices==v)]

                vertex_cell_angles = array_dict(nd.sum(vertex_triangle_angles,vertex_triangle_cells,index=vertex_cells),vertex_cells)

                for t,c in zip(vertex_triangles,vertex_triangle_cells):
                    angle_dict[(t,v)] = vertex_cell_angles[c]
                
                epidermis_vertex_angles[v] = np.sum(vertex_cell_angles.values())
                if abs(epidermis_vertex_angles[v]-360)<10:
                    epidermis_vertex_cell_angles = np.concatenate([epidermis_vertex_cell_angles,vertex_cell_angles.values()])


        # quality_data["Epidermis Cell Angle"] = 1.0 - ((epidermis_vertex_cell_angles>180).sum()+(epidermis_vertex_cell_angles<90).sum())/(float(epidermis_vertex_cell_angles.shape[0]))
        
        quality_data["Epidermis Cell Angle"] = 1 - (np.sum(1 - np.exp(-np.power(np.maximum(100-epidermis_vertex_cell_angles,0.0)/20.,2.0))) + np.sum(1 - np.exp(-np.power(np.maximum(epidermis_vertex_cell_angles-170,0.0)/20.,2.0))))/(float(epidermis_vertex_cell_angles.shape[0]))
        # np.sqrt(np.sum(np.power(np.maximum(90-epidermis_vertex_cell_angles,0.0)/30.,2.0))) + np.sum(np.maximum(epidermis_vertex_cell_angles-180,0.0)/30.)
        # quality_data["Epidermis Cell Angle"] = np.minimum(1.0,3-np.sqrt(np.power(epidermis_vertex_cell_angles-120.,2.0).mean())/30.)
        # quality_data["Epidermis Cell Angle"] = np.minimum(1.0,1.0-np.sqrt(np.power(epidermis_vertex_cell_angles-120.,2.0).mean())/120.)
        # quality_data["Epidermis Cell Angle"] = 1.0-np.power(np.power(np.abs(epidermis_vertex_cell_angles-120.),1.0).mean(),1.0)/120.

        end_time = time()
        print "<-- Computing epidermis cell angle     [",end_time-start_time,"s]"


    if "Vertex Valence" in quality_criteria:
        start_time = time()
        print "--> Computing vertex degree"
        compute_topomesh_property(topomesh,'region_neighbors',degree=0)
        compute_topomesh_property(topomesh,'border_neighbors',degree=3)

        # interface_vertices = np.array(list(topomesh.wisps(0)))[np.where((np.array(map(len,topomesh.wisp_property('cells',0).values())) == 1) 
        #                                                          | ((np.array(map(len,topomesh.wisp_property('cells',0).values())) == 2) 
        #                                                           & (1- topomesh.wisp_property('epidermis',0).values())))]
        # interface_vertices_degree = np.array(map(len,topomesh.wisp_property('neighbors',0).values(interface_vertices)))

        target_neighborhood = array_dict((np.array(map(len,topomesh.wisp_property('cells',0).values())) + topomesh.wisp_property('epidermis',0).values())*3,list(topomesh.wisps(0)))
        vertices_neighborhood = array_dict(map(len,topomesh.wisp_property('neighbors',0).values()),list(topomesh.wisps(0)))
        
        interface_vertices = target_neighborhood.keys_where("==6")        
        # np.power(np.power(target_neighborhood.values(interface_vertices)-vertices_neighborhood.values(interface_vertices),2.0).mean(),0.5)/6.0

        quality_data["Vertex Valence"] = np.minimum(1.0,1.0-np.abs(target_neighborhood.values(interface_vertices)-vertices_neighborhood.values(interface_vertices)).mean()/6.0)
        # quality_data["Vertex Valence"] = np.minimum(1.0,1.0-np.nanmean(np.abs(interface_vertices_degree-6.0)))
        # quality_data["Vertex Valence"] = np.minimum(1.0,1.0-np.power(interface_vertices_degree-6.0,2.0).mean()/6.0)
        end_time = time()
        print "<-- Computing vertex degree            [",end_time-start_time,"s]"


    if "Cell 4 Adjacency" in quality_criteria:
        start_time = time()
        print "--> Computing cell adjacency"
        if image_cell_vertex==None:
            image_cell_vertex = cell_vertex_extraction(img,hollow_out=False,verbose=False)
            if img_graph == None:
                img_graph = graph_from_image(img, spatio_temporal_properties=['volume','barycenter'],background=0,ignore_cells_at_stack_margins = False,property_as_real=False,min_contact_surface=0.5)
            img_edges = np.array([img_graph.edge_vertices(e) for e in img_graph.edges()])

            tetra_edge_index_list = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
            img_vertex_edges = np.concatenate(np.sort(image_cell_vertex.keys())[:,tetra_edge_index_list])
            vertex_edge_matching = (vq(img_vertex_edges,img_edges)[1] > 0).reshape(len(image_cell_vertex),6).sum(axis=1)
            img_cell_vertex = np.array(image_cell_vertex.keys())[np.where(vertex_edge_matching==0)[0]]

            image_cell_vertex = array_dict(np.array([image_cell_vertex[tuple(vertex)] for vertex in img_cell_vertex]),np.sort(img_cell_vertex))


        cell_vertex_VP = (vq(np.sort(array_dict(mesh_cell_vertex).keys()),np.sort(array_dict(image_cell_vertex).keys()))[1]==0).sum()
        cell_vertex_FP = (vq(np.sort(array_dict(mesh_cell_vertex).keys()),np.sort(array_dict(image_cell_vertex).keys()))[1]>0).sum()
        cell_vertex_FN = (vq(np.sort(array_dict(image_cell_vertex).keys()),np.sort(array_dict(mesh_cell_vertex).keys()))[1]>0).sum()
        cell_vertex_jaccard = cell_vertex_VP/float(cell_vertex_VP+cell_vertex_FP+cell_vertex_FN)
        cell_vertex_dice = 2*cell_vertex_VP/float(2*cell_vertex_VP+cell_vertex_FP+cell_vertex_FN)


        quality_data["Cell 4 Adjacency"] = cell_vertex_jaccard
        end_time = time()
        print "<-- Computing cell adjacency           [",end_time-start_time,"s]"

    if "Cell 2 Adjacency" in quality_criteria:
        start_time = time()
        print "--> Computing cell adjacency"

        tetra_edge_index_list = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])

        if image_cell_vertex==None:
            image_cell_vertex = cell_vertex_extraction(img,hollow_out=False,verbose=False)
            if img_graph == None:
                img_graph = graph_from_image(img, spatio_temporal_properties=['volume','barycenter'],background=0,ignore_cells_at_stack_margins = False,property_as_real=False,min_contact_surface=0.5)
            img_edges = np.array([img_graph.edge_vertices(e) for e in img_graph.edges()])

            img_vertex_edges = np.concatenate(np.sort(image_cell_vertex.keys())[:,tetra_edge_index_list])
            vertex_edge_matching = (vq(img_vertex_edges,img_edges)[1] > 0).reshape(len(image_cell_vertex),6).sum(axis=1)
            img_cell_vertex = np.array(image_cell_vertex.keys())[np.where(vertex_edge_matching==0)[0]]

            image_cell_vertex = array_dict(np.array([image_cell_vertex[tuple(vertex)] for vertex in img_cell_vertex]),np.sort(img_cell_vertex))


        from vplants.meshing.array_tools import array_unique

        mesh_cell_edges = array_unique(np.concatenate(np.sort(array_dict(mesh_cell_vertex).keys())[:,tetra_edge_index_list]))
        image_cell_edges = array_unique(np.concatenate(np.sort(array_dict(image_cell_vertex).keys())[:,tetra_edge_index_list]))

        from vplants.meshing.evaluation_tools import jaccard_index

        cell_edge_jaccard = jaccard_index(image_cell_edges,mesh_cell_edges)

        quality_data["Cell 2 Adjacency"] = cell_edge_jaccard
        end_time = time()
        print "<-- Computing cell adjacency           [",end_time-start_time,"s]"

    if "Cell Cliques" in quality_criteria:

        vertex_cells = np.array([len(list(topomesh.regions(0,v,3))) for v in topomesh.wisps(0)])
        vertex_epidermis = topomesh.wisp_property('epidermis',degree=0).values()

        cell_vertices = ((vertex_cells>=3)*(vertex_epidermis)+(vertex_cells>=4)*(True-vertex_epidermis)).sum()
        clique_cell_vertices = ((vertex_cells>3)*(vertex_epidermis)+(vertex_cells>4)*(True-vertex_epidermis)).sum()

        quality_data["Cell Cliques"] = 1.0 - float(clique_cell_vertices)/float(cell_vertices)


    print quality_data

    # if display:
    #     spider_figure = plt.figure(kwargs.get('figure_title',"Topomesh Quality"))
    #     spider_figure.clf()
    #     spider_data = np.array([quality_data[c] for c in quality_criteria])
    #     spider_fields= quality_criteria
    #     # spider_targets = [0.8,0.7,0.8,0.8,0.9,0.8,0.7]
    #     spider_targets = 0.8 * np.ones_like(quality_criteria,float)
    #     spider_plot(spider_figure,spider_data,color1=np.array([0.3,0.6,1.]),color2=np.array([1.,0.,0.]),xlabels=spider_fields,ytargets=spider_targets,n_points=100*len(quality_criteria),linewidth=2,smooth_factor=0.0,spline_order=1)
    #     plt.show(block=False)

    return quality_data
