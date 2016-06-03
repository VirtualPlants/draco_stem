# -*- coding: utf-8 -*-
# -*- python -*-
#
#       TriangularMesh
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
from openalea.container import array_dict

from openalea.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property
from openalea.cellcomplex.triangular_mesh import TriangularMesh

from time import time

def topomesh_to_triangular_mesh(topomesh, degree=3, coef=1.0, mesh_center=None, epidermis=False, property_name=None, property_degree=None):


    start_time = time()
    print "--> Creating triangular mesh"

    triangular_mesh = TriangularMesh()


    if property_name is not None:
        if property_degree is None:
            property_degree = degree
        try:
            if not topomesh.has_wisp_property(property_name,degree=property_degree,is_computed=True):
                compute_topomesh_property(topomesh,property_name,degree=property_degree)
            assert len(topomesh.wisp_property(property_name,property_degree).keys()) == topomesh.nb_wisps(property_degree)
        except:
            property_name = None

    if mesh_center is None:
        mesh_center = topomesh.wisp_property('barycenter',0).values().mean(axis=0)
    else:
        mesh_center = np.array(mesh_center)

    if degree>1:
        compute_topomesh_property(topomesh,'vertices',degree=2)
        compute_topomesh_property(topomesh,'triangles',degree=3)
        compute_topomesh_property(topomesh,'cells',degree=2)

        cell_triangles = np.concatenate(topomesh.wisp_property('triangles',3).values(list(topomesh.wisps(3)))).astype(int)

    if degree == 3:
        if property_name is not None:
            property_data = topomesh.wisp_property(property_name,property_degree).values()
        else:
            property_data = np.array(list(topomesh.wisps(3)))


        vertices_positions = []
        triangle_vertices = []
        triangle_topomesh_cells = []
        vertices_topomesh_vertices = []
        # triangle_topomesh_triangles = []

        if property_data.ndim == 1 or property_degree<3:
            for c in topomesh.wisps(3):
                if len(list(topomesh.borders(3,c,3)))>0:
                    cell_center = topomesh.wisp_property('barycenter',0).values(list(topomesh.borders(3,c,3))).mean(axis=0)
                    cell_vertices_position = cell_center + coef*(topomesh.wisp_property('barycenter',0).values(list(topomesh.borders(3,c,3)))-cell_center) - mesh_center
                    cell_vertices_index = array_dict(len(vertices_positions) + np.arange(len(list(topomesh.borders(3,c,3)))),list(topomesh.borders(3,c,3)))
                    vertices_positions += list(cell_vertices_position)
                    vertices_topomesh_vertices += list(topomesh.borders(3,c,3))
                    triangle_vertices += list(cell_vertices_index.values(topomesh.wisp_property('vertices',2).values(topomesh.wisp_property('triangles',3)[c])))
                    triangle_topomesh_cells += list(c*np.ones_like(topomesh.wisp_property('triangles',3)[c]))
                    # triangle_topomesh_triangles += topomesh.wisp_property('triangles',3)[c]
            vertices_positions = array_dict(vertices_positions,np.arange(len(vertices_positions)))
            vertices_topomesh_vertices = array_dict(vertices_topomesh_vertices,np.arange(len(vertices_positions)))
            if epidermis:
                compute_topomesh_property(topomesh,'epidermis',2)
                epidermis_triangles = topomesh.wisp_property('epidermis',2).values(cell_triangles)
                triangle_vertices = array_dict(np.array(triangle_vertices)[epidermis_triangles],np.arange(len(cell_triangles[epidermis_triangles])))
                triangle_topomesh_cells = array_dict(np.array(triangle_topomesh_cells)[epidermis_triangles],np.arange(len(cell_triangles[epidermis_triangles])))
                triangle_topomesh_triangles = array_dict(cell_triangles[epidermis_triangles],np.arange(len(cell_triangles[epidermis_triangles])))
            else:
                triangle_vertices = array_dict(triangle_vertices,np.arange(len(cell_triangles)))
                triangle_topomesh_cells = array_dict(triangle_topomesh_cells,np.arange(len(cell_triangles)))
                triangle_topomesh_triangles = array_dict(cell_triangles,np.arange(len(cell_triangles)))
            edge_topomesh_edges = {}

            triangular_mesh.points = vertices_positions.to_dict()
            triangular_mesh.triangles = triangle_vertices.to_dict()
            if property_name is not None:  
                if property_degree == 2:
                    triangle_topomesh_triangle_property = array_dict(topomesh.wisp_property(property_name,property_degree).values(triangle_topomesh_triangles.values()),triangle_topomesh_triangles.keys())
                    triangular_mesh.triangle_data = triangle_topomesh_triangle_property.to_dict()
                elif property_degree == 0:
                    vertex_topomesh_vertex_property = array_dict(topomesh.wisp_property(property_name,property_degree).values(vertices_topomesh_vertices.values()),vertices_topomesh_vertices.keys())
                    triangular_mesh.point_data = vertex_topomesh_vertex_property.to_dict()
                elif property_degree == 3:
                    triangle_topomesh_cell_property = array_dict(topomesh.wisp_property(property_name,property_degree).values(triangle_topomesh_cells.values()),triangle_topomesh_cells.keys())
                    triangular_mesh.triangle_data = triangle_topomesh_cell_property.to_dict()
            else:
                triangular_mesh.triangle_data = triangle_topomesh_cells.to_dict()
        else:
            for c in topomesh.wisps(3):
                if len(list(topomesh.borders(3,c,3)))>0:
                    cell_center = topomesh.wisp_property('barycenter',0).values(list(topomesh.borders(3,c,3))).mean(axis=0)
                    vertices_positions += [cell_center]

            vertices_positions = array_dict(vertices_positions,np.array([c for c in topomesh.wisps(3) if len(list(topomesh.borders(3,c,3)))>0]))
            vertices_topomesh_vertices = {}
            edge_topomesh_edges = {}
            triangle_topomesh_triangles = {}
            triangle_topomesh_cells = {}

            cell_property = array_dict(topomesh.wisp_property(property_name,property_degree).values(vertices_positions.keys()),vertices_positions.keys())

            triangular_mesh.points = vertices_positions.to_dict()
            triangular_mesh.point_data = cell_property
            triangular_mesh.triangles = {}

    elif degree == 2:
        vertices_positions = []
        triangle_vertices = []
        vertices_topomesh_vertices = []
        for t in cell_triangles:
            triangle_center = topomesh.wisp_property('barycenter',0).values(list(topomesh.borders(2,t,2))).mean(axis=0)
            triangle_vertices_position = triangle_center + coef*(topomesh.wisp_property('barycenter',0).values(list(topomesh.borders(2,t,2)))-triangle_center) - mesh_center
            triangle_vertices_index = array_dict(len(vertices_positions) + np.arange(3),list(topomesh.borders(2,t,2)))
            vertices_positions += list(triangle_vertices_position)
            vertices_topomesh_vertices += list(topomesh.borders(2,t,2))
            triangle_vertices += list([triangle_vertices_index.values(topomesh.wisp_property('vertices',2)[t])])
        vertices_positions = array_dict(vertices_positions,np.arange(len(vertices_positions)))
        vertices_topomesh_vertices = array_dict(vertices_topomesh_vertices,np.arange(len(vertices_positions)))
        triangle_topomesh_cells = np.concatenate([c*np.ones_like(topomesh.wisp_property('triangles',3)[c]) for c in topomesh.wisps(3)]).astype(int)
        if epidermis:
            compute_topomesh_property(topomesh,'epidermis',2)
            epidermis_triangles = topomesh.wisp_property('epidermis',2).values(cell_triangles)
            triangle_vertices = array_dict(np.array(triangle_vertices)[epidermis_triangles],np.arange(len(cell_triangles[epidermis_triangles])))
            triangle_topomesh_cells = array_dict(np.array(triangle_topomesh_cells)[epidermis_triangles],np.arange(len(cell_triangles[epidermis_triangles])))
            triangle_topomesh_triangles = array_dict(cell_triangles[epidermis_triangles],np.arange(len(cell_triangles[epidermis_triangles])))
        else:
            triangle_vertices = array_dict(triangle_vertices,np.arange(len(cell_triangles)))
            triangle_topomesh_cells = array_dict(triangle_topomesh_cells,np.arange(len(cell_triangles)))
            triangle_topomesh_triangles = array_dict(cell_triangles,np.arange(len(cell_triangles)))
        edge_topomesh_edges = {}

        triangular_mesh.points = vertices_positions.to_dict()
        triangular_mesh.triangles = triangle_vertices.to_dict()
        if property_name is not None:
            if property_degree == 2:
                triangle_topomesh_triangle_property = array_dict(topomesh.wisp_property(property_name,property_degree).values(triangle_topomesh_triangles.values()),triangle_topomesh_triangles.keys())
                triangular_mesh.triangle_data = triangle_topomesh_triangle_property.to_dict()
            elif property_degree == 0:
                vertex_topomesh_vertex_property = array_dict(topomesh.wisp_property(property_name,property_degree).values(vertices_topomesh_vertices.values()),vertices_topomesh_vertices.keys())
                triangular_mesh.point_data = vertex_topomesh_vertex_property.to_dict()
            elif property_degree == 3:
                triangle_topomesh_cell_property = array_dict(topomesh.wisp_property(property_name,property_degree).values(triangle_topomesh_cells.values()),triangle_topomesh_cells.keys())
                triangular_mesh.triangle_data = triangle_topomesh_cell_property.to_dict()
        else:
            triangular_mesh.triangle_data = triangle_topomesh_cells.to_dict()

    elif degree == 1:
        compute_topomesh_property(topomesh,'vertices',degree=1)
        vertices_positions = topomesh.wisp_property('barycenter',0)
        edge_vertices = topomesh.wisp_property('vertices',1)
        triangular_mesh.points = vertices_positions.to_dict()
        triangular_mesh.edges = edge_vertices.to_dict()

        if property_name is not None:
            if property_degree == 1:
                triangular_mesh.edge_data = topomesh.wisp_property(property_name,property_degree).to_dict()
        triangle_topomesh_cells = {}
        triangle_topomesh_triangles = {}
        edge_topomesh_edges = dict(zip(triangular_mesh.edges.keys(),triangular_mesh.edges.keys()))
        vertices_topomesh_vertices = {}

    elif degree == 0:
        vertices_positions = topomesh.wisp_property('barycenter',0)
        triangular_mesh.points = vertices_positions.to_dict()

        if property_name is not None:
            if property_degree == 0:
                triangular_mesh.point_data = topomesh.wisp_property(property_name,property_degree).to_dict()
        triangle_topomesh_cells = {}
        triangle_topomesh_triangles = {}
        edge_topomesh_edges = {}
        vertices_topomesh_vertices = dict(zip(triangular_mesh.points.keys(),triangular_mesh.points.keys()))

    mesh_element_matching = {}
    mesh_element_matching[0] = vertices_topomesh_vertices
    mesh_element_matching[1] = edge_topomesh_edges
    mesh_element_matching[2] = triangle_topomesh_triangles
    mesh_element_matching[3] = triangle_topomesh_cells

    end_time = time()
    print "<-- Creating triangular mesh [",end_time - start_time,"s]"

    return triangular_mesh, mesh_element_matching