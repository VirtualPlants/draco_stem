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

def isiterable(obj):
    try:
        iter(obj)
        return True
    except:
        return False

class TriangularMesh(object):
    def __init__(self):
        self.points = {}
        self.point_data = {}
        self.edges = {}
        self.edge_data = {}
        self.triangles = {}
        self.triangle_data = {}

        self.point_radius = 2.0

    def _repr_geom_(self):
        import openalea.plantgl.all as pgl
        import vplants.plantgl.ext.color as color
        from openalea.container import array_dict
        
        scene = pgl.Scene()
        colormap = color.GlasbeyMap(0,256)

        if len(self.triangles) > 0:
            triangle_points = array_dict(self.points).values(self.triangles.values())
            triangle_normals = np.cross(triangle_points[:,1]-triangle_points[:,0],triangle_points[:,2]-triangle_points[:,0])
            mesh_center = np.mean(self.points.values(),axis=0)
            reversed_normals = np.array(self.triangles.keys())[np.where(np.einsum('ij,ij->i',triangle_normals,triangle_points[:,0]-mesh_center) < 0)[0]]
            for t in reversed_normals:
                self.triangles[t] = list(reversed(self.triangles[t]))
                                         

            points_index = array_dict(np.arange(len(self.points)),self.points.keys())

            if isiterable(self.triangle_data.values()[0]):
                colors = [pgl.Color4(colormap(self.triangle_data[t][0]%256).i3tuple()) if self.triangle_data.has_key(t) else pgl.Color4(colormap(0).i3tuple()) for t in self.triangles.keys()]
            else:
                colors = [pgl.Color4(colormap(self.triangle_data[t]%256).i3tuple()) if self.triangle_data.has_key(t) else pgl.Color4(colormap(0).i3tuple()) for t in self.triangles.keys()]

            # scene += pgl.Shape(pgl.FaceSet(self.points.values(),list(points_index.values(self.triangles.values()))),pgl.Material((255,255,255)))
            scene += pgl.Shape(pgl.FaceSet(self.points.values(),list(points_index.values(self.triangles.values())),colorList=colors,colorPerVertex=False))
            #for t in self.triangles.keys():
            #    scene += pgl.Shape(pgl.FaceSet([self.points[p] for p in self.triangles[t]],[list(range(3))]),pgl.Material(colormap(self.triangle_data[t]%256).i3tuple()),id=t)
        else:
            for p in self.points.keys():
                mat = pgl.Material(colormap(p%256).i3tuple(),transparency=0.0,name=p)
                scene += pgl.Shape(pgl.Translated(self.points[p],pgl.Sphere(self.point_radius,slices=16,stacks=16)),mat,id=p)
        return scene
    
    def _repr_vtk_(self):
        import vtk
        from openalea.container import array_dict

        if len(self.triangles) > 0:
            vtk_mesh = vtk.vtkPolyData()
            vtk_points = vtk.vtkPoints()
            vtk_point_data = vtk.vtkDoubleArray()
            vtk_triangles = vtk.vtkCellArray()
            vtk_triangle_data = vtk.vtkDoubleArray()
            
            mesh_points = []
            for p in self.points.keys():
                pid = vtk_points.InsertNextPoint(self.points[p])
                mesh_points.append(pid)
                if self.point_data.has_key(p):
                    if isiterable(self.point_data[p]):
                        vtk_point_data.InsertValue(pid,self.point_data[p][0])
                    else:
                        vtk_point_data.InsertValue(pid,self.point_data[p])
            mesh_points = array_dict(mesh_points,self.points.keys())
            if len(self.point_data) > 0:
                vtk_mesh.GetPointData().SetScalars(vtk_point_data)

            for t in self.triangles.keys():
                poly = vtk_triangles.InsertNextCell(3)
                for i in xrange(3):
                    vtk_triangles.InsertCellPoint(mesh_points[self.triangles[t][i]])
                if self.triangle_data.has_key(t):
                    if isiterable(self.triangle_data[t]):
                        vtk_triangle_data.InsertValue(poly,self.triangle_data[t][0])
                    else:
                        vtk_triangle_data.InsertValue(poly,self.triangle_data[t])
            vtk_mesh.SetPoints(vtk_points)
            vtk_mesh.SetPolys(vtk_triangles)
            if len(self.triangle_data) > 0:
                vtk_mesh.GetCellData().SetScalars(vtk_triangle_data)
            
            return vtk_mesh
        elif len(self.edges) > 0:
            vtk_mesh = vtk.vtkPolyData()
            vtk_points = vtk.vtkPoints()
            vtk_point_data = vtk.vtkDoubleArray()
            vtk_lines = vtk.vtkCellArray()
            vtk_line_data = vtk.vtkDoubleArray()

            mesh_points = []
            for p in self.points.keys():
                pid = vtk_points.InsertNextPoint(self.points[p])
                mesh_points.append(pid)
                if self.point_data.has_key(p):
                    if isiterable(self.point_data[p]):
                        vtk_point_data.InsertValue(pid,self.point_data[p][0])
                    else:
                        vtk_point_data.InsertValue(pid,self.point_data[p])
            mesh_points = array_dict(mesh_points,self.points.keys())
            if len(self.point_data) > 0:
                vtk_mesh.GetPointData().SetScalars(vtk_point_data)

            for e in self.edges.keys():
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0,mesh_points[self.edges[e][0]])
                line.GetPointIds().SetId(1,mesh_points[self.edges[e][1]])
                edge = vtk_lines.InsertNextCell(line)
                if self.edge_data.has_key(e):
                    if isiterable(self.edge_data[e]):
                        vtk_line_data.InsertValue(edge,self.edge_data[e][0])
                    else:
                        vtk_line_data.InsertValue(edge,self.edge_data[e])
                else:
                    vtk_line_data.InsertValue(edge,0)

            vtk_mesh.SetPoints(vtk_points)
            vtk_mesh.SetLines(vtk_lines)
            vtk_mesh.GetCellData().SetScalars(vtk_line_data)

            return vtk_mesh

        else:
            vtk_mesh = vtk.vtkPolyData()
            vtk_points = vtk.vtkPoints()
            vtk_cells = vtk.vtkDoubleArray()
            for p in self.points.keys():
                pid = vtk_points.InsertNextPoint(self.points[p])
                if self.point_data.has_key(p):
                    vtk_cells.InsertValue(pid,self.point_data[p])
                else:
                    vtk_cells.InsertValue(pid,p)
            vtk_mesh.SetPoints(vtk_points)
            vtk_mesh.GetPointData().SetScalars(vtk_cells)

            return vtk_mesh

    def min(self):
        if len(self.triangle_data) > 0:
            return np.min(self.triangle_data.values())
        elif len(self.point_data) > 0:
            return np.min(self.point_data.values())
        elif len(self.triangles) > 0:
            return np.min(self.triangles.keys())
        else:
            return np.min(self.points.keys())

    def max(self):
        if len(self.triangle_data) > 0:
            return np.max(self.triangle_data.values())
        elif len(self.point_data) > 0:
            return np.max(self.point_data.values())
        elif len(self.triangles) > 0:
            return np.max(self.triangles.keys())
        else:
            return np.max(self.points.keys())


def point_triangular_mesh(point_positions, point_data=None):
    points_mesh = TriangularMesh()
    points_mesh.points = point_positions
    if point_data is not None:
        points_mesh.point_data = point_data
    return points_mesh


def topomesh_to_triangular_mesh(topomesh, degree=3, coef=1.0, mesh_center=None, epidermis=False, property_name=None, property_degree=None):

    import numpy as np
    from openalea.container import array_dict
    
    from openalea.mesh.property_topomesh_analysis import compute_topomesh_property
    from openalea.mesh.triangular_mesh import TriangularMesh

    from time import time

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

        cell_triangles = np.concatenate(topomesh.wisp_property('triangles',3).values(list(topomesh.wisps(3))))

    if degree == 3:
        vertices_positions = []
        triangle_vertices = []
        triangle_topomesh_cells = []
        vertices_topomesh_vertices = []
        # triangle_topomesh_triangles = []
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
        triangle_topomesh_cells = np.concatenate([c*np.ones_like(topomesh.wisp_property('triangles',3)[c]) for c in topomesh.wisps(3)])
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
        vertices_topomesh_vertices = {}

    elif degree == 0:
        vertices_positions = topomesh.wisp_property('barycenter',0)
        triangular_mesh.points = vertices_positions.to_dict()

        if property_name is not None:
            if property_degree == 0:
                triangular_mesh.point_data = topomesh.wisp_property(property_name,property_degree).to_dict()
        triangle_topomesh_cells = {}
        triangle_topomesh_triangles = {}
        vertices_topomesh_vertices = dict(zip(triangular_mesh.points.keys(),triangular_mesh.points.keys()))

    mesh_element_matching = {}
    mesh_element_matching[0] = vertices_topomesh_vertices
    mesh_element_matching[1] = None
    mesh_element_matching[2] = triangle_topomesh_triangles
    mesh_element_matching[3] = triangle_topomesh_cells

    end_time = time()
    print "<-- Creating triangular mesh [",end_time - start_time,"s]"

    return triangular_mesh, mesh_element_matching



def save_ply_triangular_mesh(mesh,ply_filename,intensity_range=None):
    from time import time

    start_time =time()
    print "--> Saving .ply"

    ply_file = open(ply_filename,'w+')

    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex "+str(len(mesh.points))+"\n")
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    if len(mesh.point_data)>0 and np.ndim(mesh.point_data.values()[0])==0:
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        if intensity_range is None:
            point_data = array_dict(np.array(mesh.point_data.values(),int)%256,mesh.point_data.keys())
        else:
            point_data = array_dict((np.array(mesh.point_data.values())-intensity_range[0])/(intensity_range[1]-intensity_range[0]),mesh.point_data.keys())

    ply_file.write("element face "+str(len(mesh.triangles))+"\n")
    ply_file.write("property list uchar int vertex_indices\n")
    if len(mesh.triangle_data)>0 and np.ndim(mesh.triangle_data.values()[0])==0:
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        if intensity_range is None:
            triangle_data = array_dict(np.array(mesh.triangle_data.values(),int)%256,mesh.triangle_data.keys())
        else:
            triangle_data = array_dict((np.array(mesh.triangle_data.values())-intensity_range[0])/(intensity_range[1]-intensity_range[0]),mesh.triangle_data.keys())

    ply_file.write("end_header\n")

    vertex_index = {}
    for v,p in enumerate(mesh.points.keys()):
        ply_file.write(str(mesh.points[p][0])+" ")
        ply_file.write(str(mesh.points[p][1])+" ")
        ply_file.write(str(mesh.points[p][2]))
        if len(mesh.point_data)>0 and np.ndim(mesh.point_data.values()[0])==0:
            ply_file.write(" "+str(point_data[p]))
            ply_file.write(" "+str(point_data[p]))
            ply_file.write(" "+str(point_data[p]))
        ply_file.write("\n")
        vertex_index[p] = v

    for t in mesh.triangles.keys():
        ply_file.write("3 ")
        ply_file.write(str(vertex_index[mesh.triangles[t][0]])+" ")
        ply_file.write(str(vertex_index[mesh.triangles[t][1]])+" ")
        ply_file.write(str(vertex_index[mesh.triangles[t][2]]))
        if len(mesh.triangle_data)>0 and np.ndim(mesh.triangle_data.values()[0])==0:
            ply_file.write(" "+str(triangle_data[t]))
            ply_file.write(" "+str(triangle_data[t]))
            ply_file.write(" "+str(triangle_data[t]))
        ply_file.write("\n")

    ply_file.flush()
    ply_file.close()

    end_time = time()
    print "<-- Saving .ply        [",end_time-start_time,"s]"

