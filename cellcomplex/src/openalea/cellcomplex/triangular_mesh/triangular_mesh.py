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
        self.char_dimension = None

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
            
            if len(self.triangle_data)>0 and np.array(self.triangle_data.values()).ndim > 1:
                if np.array(self.triangle_data.values()).ndim==2:
                    vtk_point_data.SetNumberOfComponents(np.array(self.triangle_data.values()).shape[1])
                elif np.array(self.triangle_data.values()).ndim==3:
                    vtk_point_data.SetNumberOfComponents(np.array(self.triangle_data.values()).shape[1]*np.array(self.triangle_data.values()).shape[2])
                
                mesh_points = []
                positions = array_dict(self.points)
                for t in self.triangles.keys():
                    triangle_center = positions.values(self.triangles[t]).mean(axis=0)
                    tid = vtk_points.InsertNextPoint(triangle_center)
                    mesh_points.append(tid)

                    if self.triangle_data.has_key(t):
                        if isiterable(self.triangle_data[t]):
                            if np.array(self.triangle_data[t]).ndim==1:
                                vtk_point_data.InsertTuple(tid,self.triangle_data[t])
                            else:
                                vtk_point_data.InsertTuple(tid,np.concatenate(self.triangle_data[t]))
                mesh_points = array_dict(mesh_points,self.triangles.keys())
                
                vtk_mesh.SetPoints(vtk_points)

                if np.array(self.triangle_data.values()).ndim==2:
                    vtk_mesh.GetPointData().SetVectors(vtk_point_data)
                elif np.array(self.triangle_data.values()).ndim==3:
                    vtk_mesh.GetPointData().SetTensors(vtk_point_data)

            else:

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
                            if np.array(self.triangle_data[t]).ndim==1:
                                vtk_triangle_data.InsertTuple(poly,self.triangle_data[t])
                            else:
                                vtk_triangle_data.InsertTuple(poly,np.concatenate(self.triangle_data[t]))
                            # vtk_triangle_data.InsertValue(poly,self.triangle_data[t][0])
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
            if len(self.point_data)>0 and np.array(self.point_data.values()).ndim==2:
                vtk_cells.SetNumberOfComponents(np.array(self.point_data.values()).shape[1])
            elif len(self.point_data)>0 and np.array(self.point_data.values()).ndim==3:
                vtk_cells.SetNumberOfComponents(np.array(self.point_data.values()).shape[1]*np.array(self.point_data.values()).shape[2])
            for p in self.points.keys():
                pid = vtk_points.InsertNextPoint(self.points[p])
                if self.point_data.has_key(p):
                    if isiterable(self.point_data[p]):
                        if np.array(self.point_data[p]).ndim==1:
                            cell = vtk_cells.InsertNextTuple(self.point_data[p])
                        else:
                            cell = vtk_cells.InsertNextTuple(np.concatenate(self.point_data[p]))
                    else:
                        vtk_cells.InsertValue(pid,self.point_data[p])
                else:
                    vtk_cells.InsertValue(pid,p)
            vtk_mesh.SetPoints(vtk_points)
            if len(self.point_data)>0 and np.array(self.point_data.values()).ndim==2:
                vtk_mesh.GetPointData().SetVectors(vtk_cells)
            elif len(self.point_data)>0 and np.array(self.point_data.values()).ndim==3:
                vtk_mesh.GetPointData().SetTensors(vtk_cells)
            else:
                vtk_mesh.GetPointData().SetScalars(vtk_cells)


            return vtk_mesh

    def data(self):
        if len(self.triangle_data) > 0:
            data = np.array(self.triangle_data.values())
        elif len(self.point_data) > 0:
            data = np.array(self.point_data.values())
        elif len(self.triangles) > 0:
            data = np.array(self.triangles.keys())
        else:
            data = np.array(self.points.keys())
        return data

    def min(self):
        if self.data().ndim == 1:
            return np.min(self.data())
        elif self.data().ndim == 2:
            return np.min(np.linalg.norm(self.data(),axis=1))
        elif self.data().ndim == 3:
            return np.min(np.sqrt(np.trace(np.power(self.data(),2),axis1=1,axis2=2)))

    def max(self):
        if self.data().ndim == 1:
            return np.max(self.data())
        elif self.data().ndim == 2:
            return np.max(np.linalg.norm(self.data(),axis=1))
        elif self.data().ndim == 3:
            return np.max(np.sqrt(np.trace(np.power(self.data(),2),axis1=1,axis2=2)))

    def mean(self):
        if self.data().ndim == 1:
            return np.mean(self.data())
        elif self.data().ndim == 2:
            return np.mean(np.linalg.norm(self.data(),axis=1))
        elif self.data().ndim == 3:
            return np.mean(np.sqrt(np.trace(np.power(self.data(),2),axis1=1,axis2=2)))

    def bounding_box(self):
        if len(self.points)>0:
            extent_min = (np.min(self.points.values(),axis=0))
            extent_max = (np.max(self.points.values(),axis=0))
            return zip(extent_min,extent_max)
        else:
            return zip([0,0,0],[0,0,0])

    def characteristic_dimension(self):
        if self.char_dimension is None:
            if len(self.points)>1:
                if len(self.triangles)>0:
                    triangle_edge_list = [[1,2],[0,2],[0,1]]
                    triangle_edges = np.concatenate(np.array(self.triangles.values())[:,triangle_edge_list])
                    triangle_edge_points = array_dict(self.points).values(triangle_edges)
                    triangle_edge_vectors = triangle_edge_points[:,1] - triangle_edge_points[:,0]
                    triangle_edge_lengths = np.linalg.norm(triangle_edge_vectors,axis=1)
                    self.char_dimension = triangle_edge_lengths.mean()
                elif len(self.edges)>0:
                    edges = np.array(self.edges.values())
                    edge_points = array_dict(self.points).values(edges)
                    edge_vectors = edge_points[:,1] - edge_points[:,0]
                    edge_lengths = np.linalg.norm(edge_vectors,axis=1)
                    self.char_dimension = edge_lengths.mean()
                else:
                    #from scipy.cluster.vq import vq
                    #point_distances = np.sort([vq(np.array(self.points.values()),np.array([self.points[p]]))[1] for p in self.points.keys()])
                    # self.char_dimension = point_distances[:,1].mean()
                    bbox = np.array(self.bounding_box())
                    bbox_volume = np.prod(bbox[:,1] - bbox[:,0])
                    point_volume = bbox_volume/float(2.*len(self.points))
                    self.char_dimension = np.power(3.*point_volume/(4.*np.pi),1/3.)
                return self.char_dimension
            else:
                return 1.
        else:
            return self.char_dimension



def point_triangular_mesh(point_positions, point_data=None):
    points_mesh = TriangularMesh()
    points_mesh.points = point_positions
    if point_data is not None:
        points_mesh.point_data = point_data
    return points_mesh


def topomesh_to_triangular_mesh(topomesh, degree=3, coef=1.0, mesh_center=None, epidermis=False, property_name=None, property_degree=None):

    import numpy as np
    from openalea.container import array_dict
    
    from openalea.cellcomplex.property_topomesh.property_topomesh_analysis import compute_topomesh_property

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

        cell_triangles = np.concatenate(topomesh.wisp_property('triangles',3).values(list(topomesh.wisps(3)))).astype(int)

    if degree == 3:
        if property_name is not None:
            property_data = topomesh.wisp_property(property_name,property_degree).values()
        else:
            property_data = np.array(topomesh.wisps(3))


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



def save_ply_triangle_mesh(ply_filename, positions, triangles={}, edges={}, vertex_properties={}, triangle_properties={}, edge_properties={}):
    """
    """

    from time import time

    if isinstance(positions,list) or isinstance(positions,np.ndarray):
        positions = dict(zip(range(len(positions)),positions))
    if isinstance(triangles,list) or isinstance(triangles,np.ndarray):
        triangles = dict(zip(range(len(triangles)),triangles))
    if isinstance(edges,list) or isinstance(edges,np.ndarray):
        edges = dict(zip(range(len(edges)),edges))


    property_types = {}
    property_types['bool'] = "int"
    property_types['int'] = "int"
    property_types['int32'] = "int"
    property_types['int64'] = "int"
    property_types['float'] = "float"
    property_types['float32'] = "float"
    property_types['float64'] = "float"
    # property_types['object'] = "list"

    start_time =time()
    print "--> Saving .ply"

    ply_file = open(ply_filename,'w+')

    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")

    # Declaring vertices and vertex properties
    ply_file.write("element vertex "+str(len(positions))+"\n")
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")

    vertex_property_data = {}
    for property_name in vertex_properties.keys():
        property_data = vertex_properties[property_name]
        if isinstance(property_data,dict):
            property_data = np.array([property_data[p] for p in positions.keys()])
        elif isinstance(property_data,list) or isinstance(property_data,tuple):
            property_data = np.array(property_data)
        vertex_property_data[property_name] = property_data

        if property_data.ndim == 1:
            property_type = property_types[str(property_data.dtype)]
        elif property_data.ndim == 2:
            property_type = "list int "+property_types[str(property_data.dtype)]
        else:
            property_type = "tensor "
            for i in xrange(property_data.ndim-1):
                property_type += "int "
            property_type += property_types[str(property_data.dtype)]

        ply_file.write("property "+property_type+" "+property_name+"\n")

    ply_file.write("element face "+str(len(triangles))+"\n")
    ply_file.write("property list int int vertex_index\n")

    triangle_property_data = {}
    for property_name in triangle_properties.keys():
        property_data = triangle_properties[property_name]
        if isinstance(property_data,dict):
            property_data = np.array([property_data[t] for t in triangles.keys()])
        elif isinstance(property_data,list) or isinstance(property_data,tuple):
            property_data = np.array(property_data)
        triangle_property_data[property_name] = property_data

        if property_data.ndim == 1:
            property_type = property_types[str(property_data.dtype)]
        elif property_data.ndim == 2:       
            property_type = "list int "+property_types[str(property_data.dtype)]
        else:
            property_type = "tensor "
            for i in xrange(property_data.ndim-1):
                property_type += "int "
            property_type += property_types[str(property_data.dtype)]
        ply_file.write("property "+property_type+" "+property_name+"\n")
    ply_file.write("end_header\n")

    # Writing property data
    vertex_index = {}
    for pid, p in enumerate(positions.keys()):
        ply_file.write(str(positions[p][0])+" ")
        ply_file.write(str(positions[p][1])+" ")
        ply_file.write(str(positions[p][2]))
        for property_name in vertex_properties.keys():
            data = np.array(vertex_property_data[property_name][pid])
            if data.ndim == 0:
                ply_file.write(" "+str(data))
            else:
                ply_file.write(multidim_data_ply_string(data))
        ply_file.write("\n")
        vertex_index[p] = pid

    for tid, t in enumerate(triangles.keys()):
        ply_file.write("3 ")
        ply_file.write(str(vertex_index[triangles[t][0]])+" ")
        ply_file.write(str(vertex_index[triangles[t][1]])+" ")
        ply_file.write(str(vertex_index[triangles[t][2]]))
        for property_name in triangle_properties.keys():
            data = np.array(triangle_property_data[property_name][tid])
            if data.ndim == 0:
                ply_file.write(" "+str(data))
            else:
                ply_file.write(multidim_data_ply_string(data))
        ply_file.write("\n")
    ply_file.flush()
    ply_file.close()

    end_time = time()
    print "<-- Saving .ply        [",end_time-start_time,"s]"

def multidim_data_ply_string(data):
    data_string = ""
    for dim in xrange(data.ndim):
        data_string += " "+str(data.shape[dim])
    # print data," (",len(data),")"
    # data_string = " "+str(len(data))
    for d in data.ravel():  
        data_string += " "+str(d)
        # data_string += (" "+str(d)) if data.ndim==1 else multidim_data_ply_string(d)
    return data_string


