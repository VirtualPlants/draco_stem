# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
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

try:
    import vtk
except ImportError:
    print "VTK needs to be installed to use these functionalities"
    raise

try:
    from tissuelab.gui.vtkviewer.vtk_utils import matrix_to_image_reader
except ImportError:
    print "TissueLab should be installed to use these functionalities"
    raise


from openalea.container import array_dict

from openalea.image.spatial_image import SpatialImage

from openalea.mesh import PropertyTopomesh, TriangularMesh
from openalea.mesh.property_topomesh_analysis import cell_topomesh

from time import time

triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])

def SetInput(obj, _input):
    if vtk.VTK_MAJOR_VERSION <= 5:
        obj.SetInput(_input)
    else:
        obj.SetInputData(_input)

def compute_topomesh_image(topomesh, img):

    image_start_time = time()
    print "<-- Computing topomesh image"

    polydata_img = np.ones_like(img)
        
    for c in list(topomesh.wisps(3)):
        if len(list(topomesh.borders(3,c))) > 0:
            polydata_start_time = time()
            sub_topomesh = cell_topomesh(topomesh,cells=[c]) 
            
            start_time = time()
            bounding_box = np.array([[0,polydata_img.shape[0]],[0,polydata_img.shape[1]],[0,polydata_img.shape[2]]])
            bounding_box[:,0] = np.floor(sub_topomesh.wisp_property('barycenter',0).values().min(axis=0)/np.array(img.resolution)).astype(int)-1
            bounding_box[:,0] = np.maximum(bounding_box[:,0],0)
            bounding_box[:,1] = np.ceil(sub_topomesh.wisp_property('barycenter',0).values().max(axis=0)/np.array(img.resolution)).astype(int)+1
            bounding_box[:,1] = np.minimum(bounding_box[:,1],np.array(img.shape)-1)
            
            sub_polydata_img = polydata_img[bounding_box[0,0]:bounding_box[0,1],bounding_box[1,0]:bounding_box[1,1],bounding_box[2,0]:bounding_box[2,1]]
            #world.add(sub_polydata_img,"topomesh_image",colormap='glasbey',alphamap='constant',bg_id=1,intensity_range=(0,1))

            reader = matrix_to_image_reader('sub_polydata_image',sub_polydata_img,sub_polydata_img.dtype)
            #reader = matrix_to_image_reader('polydata_image',polydata_img,polydata_img.dtype)

            end_time = time()
            print "  --> Extracting cell sub-image      [",end_time-start_time,"s]"

            start_time = time()
            topomesh_center = bounding_box[:,0]*np.array(img.resolution)
            positions = sub_topomesh.wisp_property('barycenter',0)
            
            polydata = vtk.vtkPolyData()
            vtk_points = vtk.vtkPoints()
            vtk_triangles = vtk.vtkCellArray()
            
            for t in sub_topomesh.wisps(2):
                triangle_points = []
                for v in sub_topomesh.borders(2,t,2):
                    p = vtk_points.InsertNextPoint(positions[v]-topomesh_center)
                    triangle_points.append(p)
                triangle_points = array_dict(triangle_points,list(sub_topomesh.borders(2,t,2)))
                poly = vtk_triangles.InsertNextCell(3)
                for v in sub_topomesh.borders(2,t,2):
                    vtk_triangles.InsertCellPoint(triangle_points[v])

            polydata.SetPoints(vtk_points)
            polydata.SetPolys(vtk_triangles)
            
            end_time = time()
            print "  --> Creating VTK PolyData      [",end_time-start_time,"s]"
            
            start_time = time()
            pol2stenc = vtk.vtkPolyDataToImageStencil()
            pol2stenc.SetTolerance(0)
            pol2stenc.SetOutputOrigin((0,0,0))
            #pol2stenc.SetOutputOrigin(tuple(-bounding_box[:,0]))
            pol2stenc.SetOutputSpacing(img.resolution)
            SetInput(pol2stenc,polydata)
            pol2stenc.Update()
            end_time = time()
            print "  --> Cell ",c," polydata stencil   [",end_time-start_time,"s]"
            
            start_time = time()
            imgstenc = vtk.vtkImageStencil()
            if vtk.VTK_MAJOR_VERSION <= 5:
                imgstenc.SetInput(reader.GetOutput())
                imgstenc.SetStencil(pol2stenc.GetOutput())
            else:
                imgstenc.SetInputData(reader.GetOutput())
                imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
            imgstenc.ReverseStencilOn()
            imgstenc.SetBackgroundValue(c)
            imgstenc.Update()
            end_time = time()
            print "  --> Cell ",c," image stencil   [",end_time-start_time,"s]"

            start_time = time()
            dim = tuple((bounding_box[:,1]-bounding_box[:,0])[::-1])
            array = np.ones(dim, img.dtype)
            export = vtk.vtkImageExport()
            export.SetInputConnection(imgstenc.GetOutputPort())
            export.Export(array)
            end_time = time()
            print "  --> Exporting image       [",end_time-start_time,"s]"
            
            start_time = time()
            array = np.transpose(array,(2,1,0))
            polydata_img[bounding_box[0,0]:bounding_box[0,1],bounding_box[1,0]:bounding_box[1,1],bounding_box[2,0]:bounding_box[2,1]] = array
            end_time = time()
            print "  --> Inserting cell sub-image       [",end_time-start_time,"s]"
            
            polydata_end_time = time()
            print "--> Inserting topomesh cell ",c,"   [",polydata_end_time-polydata_start_time,"s]"
        
    image_end_time = time()
    print "<-- Computing topomesh image   [",image_end_time-image_start_time,"s]"

    return polydata_img


def vtk_polydata_to_triangular_mesh(polydata):
    mesh = TriangularMesh()

    start_time = time()
    print "--> Creating vertices"

    polydata_point_data = None
    if polydata.GetPointData().GetNumberOfComponents() > 0:
        polydata_point_data = polydata.GetPointData().GetArray(0)

    for v in xrange(polydata.GetPoints().GetNumberOfPoints()):
        mesh.points[v] = polydata.GetPoints().GetPoint(v)
        if polydata_point_data is not None:
            mesh.point_data[v] = polydata_point_data.GetTuple(v)

    end_time = time()
    print "<-- Creating vertices        [",end_time-start_time,"s]"

    start_time = time()
    print "--> Creating triangles"

    triangles = {}
    triangle_data = {}

    polydata_cell_data = None
    if polydata.GetCellData().GetNumberOfComponents() > 0:
        polydata_cell_data = polydata.GetCellData().GetArray(0)

    for t in xrange(polydata.GetNumberOfCells()):
        mesh.triangles[t] = np.sort([polydata.GetCell(t).GetPointIds().GetId(i) for i in xrange(3)])
        if polydata_cell_data is not None:
            mesh.triangle_data[t] = polydata_cell_data.GetTuple(t)

    end_time = time()
    print "<-- Creating triangles       [",end_time-start_time,"s]"

    return mesh


def vtk_polydata_to_cell_triangular_meshes(polydata):
    mesh = {} 

    polydata_cell_data = polydata.GetCellData().GetArray(0)
    triangle_cell_start_time = time()
    print "  --> Listing triangles"
    print "      - ",polydata.GetNumberOfCells()," triangles"
    polydata_triangles = np.sort([[polydata.GetCell(t).GetPointIds().GetId(i) for i in xrange(3)] for t in xrange(polydata.GetNumberOfCells())])   
    triangle_cell_end_time = time()
    print "  <-- Listing triangles            [",triangle_cell_end_time - triangle_cell_start_time,"s]"

    triangle_cell_start_time = time()
    print "  --> Listing triangle cells"
    triangle_cell = np.array([polydata_cell_data.GetTuple(t)[0] for t in xrange(polydata.GetNumberOfCells())],np.uint16)
    triangle_cell_end_time = time()
    print "  <-- Listing triangle cells     [",triangle_cell_end_time - triangle_cell_start_time,"s]"

    start_time = time()
    print "  --> Creating cell meshes"
    for c in np.unique(triangle_cell): 

        mesh[c] = TriangularMesh()
        cell_triangles = np.arange(polydata.GetNumberOfCells())[np.where(triangle_cell==c)]
        cell_triangle_points = np.sort([[polydata.GetCell(t).GetPointIds().GetId(i) for i in xrange(3)] for t in cell_triangles])
        cell_vertices = np.sort(np.unique(cell_triangle_points))


        mesh[c].points = array_dict(np.array([polydata.GetPoints().GetPoint(v) for v in cell_vertices]),cell_vertices).to_dict()
        mesh[c].triangles = array_dict(cell_triangle_points,cell_triangles).to_dict()
        mesh[c].triangle_data = array_dict(np.ones_like(cell_triangles)*c,cell_triangles).to_dict()
    end_time = time()
    print "  <-- Creating cell meshes   [",end_time-start_time,"s]"

    return mesh



def vtk_polydata_to_topomesh(polydata):
    topomesh = PropertyTopomesh(3)

    start_time = time()
    print "--> Creating vertices"

    # polydata_points = np.array([polydata.GetPoints().GetPoint(i) for i in xrange(polydata.GetPoints().GetNumberOfPoints())])
    # unique_points = array_unique(polydata_points)
    # n_points = unique_points.shape[0]
    # polydata_point_point_matching = (vq(polydata_points,unique_points)[0])

    vertex_positions = {}
    for v in xrange(polydata.GetPoints().GetNumberOfPoints()):
    # for v in xrange(n_points):
        vertex_positions[v] = polydata.GetPoints().GetPoint(v)
        # vertex_positions[v] = unique_points[v]
        topomesh.add_wisp(0,v)
    end_time = time()
    print "<-- Creating vertices        [",end_time-start_time,"s]"

    start_time = time()
    print "--> Creating edges"
    polydata_triangles = np.sort([[polydata.GetCell(t).GetPointIds().GetId(i) for i in xrange(3)] for t in xrange(polydata.GetNumberOfCells())])
    # polydata_triangles = np.sort(polydata_point_point_matching[polydata_triangles])
    polydata_edges = array_unique(np.concatenate(polydata_triangles[:,triangle_edge_list]))
    for e, edge_vertices in enumerate(polydata_edges):
        topomesh.add_wisp(1,e)
        for v in edge_vertices:
            topomesh.link(1,e,v)
    end_time = time()
    print "<-- Creating edges           [",end_time-start_time,"s]"

    start_time = time()
    print "--> Linking triangle to edges"
    unique_triangles = array_unique(polydata_triangles)
    n_triangles = unique_triangles.shape[0]
    polydata_triangle_triangle_matching = (vq(polydata_triangles,unique_triangles)[0])
    polydata_triangle_edge_matching = (vq(np.concatenate(unique_triangles[:,triangle_edge_list]),polydata_edges)[0]).reshape((n_triangles,3))
    end_time = time()
    print "<-- Linking triangle to edges[",end_time-start_time,"s]"

    start_time = time()
    print "--> Creating triangles"
    for t in xrange(n_triangles):
        topomesh.add_wisp(2,t)
        for e in polydata_triangle_edge_matching[t]:
            topomesh.link(2,t,e)
    end_time = time()
    print "<-- Creating triangles       [",end_time-start_time,"s]"

    start_time = time()
    print "--> Creating cells"
    polydata_cell_data = polydata.GetCellData().GetArray(0)
    polydata_cells = np.array(np.unique(np.concatenate([polydata_cell_data.GetTuple(t) for t in xrange(polydata.GetNumberOfCells())])),np.uint16)
    for c in polydata_cells:
        if c != 1:
            topomesh.add_wisp(3,c)
    for t in xrange(polydata.GetNumberOfCells()):
        for c in polydata_cell_data.GetTuple(t):
            if c != 1:
                topomesh.link(3,int(c),polydata_triangle_triangle_matching[t])
    end_time = time()
    print "<-- Creating cells           [",end_time-start_time,"s]"

    topomesh.update_wisp_property('barycenter',0,vertex_positions)

    return topomesh


def link_polydata_triangle_cells(polydata,img,img_graph=None):
    if img_graph is None:
        from openalea.image.algo.graph_from_image   import graph_from_image
        img_graph = graph_from_image(img,spatio_temporal_properties=['barycenter','volume'],ignore_cells_at_stack_margins = False,property_as_real=True)

    polydata_cell_data = polydata.GetCellData().GetArray(0)

    start_time = time()
    print "    --> Listing points"
    polydata_points = np.array([polydata.GetPoints().GetPoint(i) for i in xrange(polydata.GetPoints().GetNumberOfPoints())])
    end_time = time()
    print "    <-- Listing points               [",end_time - start_time,"s]"

    start_time = time()
    print "    --> Merging points"
    point_ids = {}
    for p in xrange(polydata.GetPoints().GetNumberOfPoints()):
        point_ids[tuple(polydata_points[p])] = []
    for p in xrange(polydata.GetPoints().GetNumberOfPoints()):
        point_ids[tuple(polydata_points[p])] += [p]

    unique_points = array_unique(polydata_points)
    n_points = unique_points.shape[0]

    point_unique_id = {}
    for p in xrange(n_points):
        for i in point_ids[tuple(unique_points[p])]:
            point_unique_id[i] = p
    end_time = time()
    print "    <-- Merging points               [",end_time - start_time,"s]"

    triangle_cell_start_time = time()
    print "    --> Listing triangles"
    print "      - ",polydata.GetNumberOfCells()," triangles"
    polydata_triangles = np.sort([[polydata.GetCell(t).GetPointIds().GetId(i) for i in xrange(3)] for t in xrange(polydata.GetNumberOfCells())])   
    print "      - ",array_unique(polydata_triangles).shape[0]," unique triangles"
    # polydata_triangle_points = [polydata.GetCell(t).GetPointIds() for t in xrange(polydata.GetNumberOfCells())]
    # polydata_triangles = np.sort([[triangle_points.GetId(i) for i in xrange(3)] for triangle_points in polydata_triangle_points])   
    polydata_triangles = np.sort(array_dict(point_unique_id).values(polydata_triangles)) 
    print "      - ",array_unique(polydata_triangles).shape[0]," unique triangles (merged vertices)"
    triangle_cell_end_time = time()
    print "    <-- Listing triangles            [",triangle_cell_end_time - triangle_cell_start_time,"s]"
    raw_input()

    triangle_cell_start_time = time()
    print "    --> Initializing triangle cells"
    triangle_cells = {}
    for t in xrange(polydata.GetNumberOfCells()):
        triangle_cells[tuple(polydata_triangles[t])] = []  
        for i in xrange(10):
            if t == (i*polydata.GetNumberOfCells())/10:
                print "     --> Initializing triangle cells (",10*i,"%)"
    triangle_cell_end_time = time()
    print "    <-- Initializing triangle cells  [",triangle_cell_end_time - triangle_cell_start_time,"s]"

    triangle_cell_start_time = time()
    for t in xrange(polydata.GetNumberOfCells()):
        triangle_cells[tuple(polydata_triangles[t])] += list(polydata_cell_data.GetTuple(t))
        for i in xrange(100):
            if t == (i*polydata.GetNumberOfCells())/100:
                triangle_cell_end_time = time()
                print "     --> Listing triangle cells (",i,"%) [",(triangle_cell_end_time-triangle_cell_start_time)/(polydata.GetNumberOfCells()/100.),"s]"
                triangle_cell_start_time = time()

    triangle_cell_start_time = time()
    print "    --> Cleaning triangle cells"
    for t in xrange(polydata.GetNumberOfCells()):
        triangle_cells[tuple(polydata_triangles[t])] = np.unique(triangle_cells[tuple(polydata_triangles[t])])  
        for i in xrange(10):
            if t == (i*polydata.GetNumberOfCells())/10:
                print "     --> Cleaning triangle cells (",10*i,"%)"
    triangle_cell_end_time = time()
    print "    <-- Cleaning triangle cells      [",triangle_cell_end_time - triangle_cell_start_time,"s]"


    # triangle_cell_start_time = time()
    # print "    --> Listing triangle cells"
    # triangle_cell = np.array([polydata_cell_data.GetTuple(t)[0] for t in xrange(polydata.GetNumberOfCells())],np.uint16)
    # triangle_cell_end_time = time()
    # print "    <-- Listing triangle cells     [",triangle_cell_end_time - triangle_cell_start_time,"s]"

    # triangle_cells = {}
    # for t in xrange(polydata.GetNumberOfCells()):
    #     triangle_cells[t] = []

    # for c in considered_cells:
    #     cell_triangles = np.where(triangle_cell == c)[0]
    #     for t in cell_triangles:
    #         triangle_cells[t] += [c]
    #     print "    Cell ",c," : ",cell_triangles.shape[0]," triangles"
    #     neighbor_triangles = where_list(triangle_cell,list(img_graph.neighbors(c)))[0]
    #     neighbor_triangle_matching = vq(polydata_triangles[cell_triangles],polydata_triangles[neighbor_triangles])
    #     cell_double_triangles = neighbor_triangles[neighbor_triangle_matching[0][np.where(neighbor_triangle_matching[1]==0)[0]]]
    #     for t in cell_double_triangles:
    #         triangle_cells[t] += [c]

    # print triangle_cells.values()
    # raw_input()

    # unique_triangles,unique_triangle_rows = array_unique(polydata_triangles,return_index=True)
    # n_triangles = unique_triangles.shape[0]

    # triangle_unique = array_dict(np.arange(n_triangles),unique_triangle_rows)

    # triangle_id = (polydata_triangles * np.array([contour.GetOutput().GetPoints().GetNumberOfPoints(),1,1./contour.GetOutput().GetPoints().GetNumberOfPoints()])).sum(axis=1)
    # unique_triangle_id = (unique_triangles * np.array([contour.GetOutput().GetPoints().GetNumberOfPoints(),1,1./contour.GetOutput().GetPoints().GetNumberOfPoints()])).sum(axis=1)

    # triangle_number = nd.sum(np.ones_like(triangle_id),triangle_id,index=triangle_id)
    # double_triangle_rows = np.where(triangle_number==2)[0]
    # print "    ",double_triangle_rows.shape[0]," double triangles / ",contour.GetOutput().GetNumberOfCells()
    # double_triangles = polydata_triangles[double_triangle_rows]

    # unique_triangle_number = nd.sum(np.ones_like(triangle_id),triangle_id,index=unique_triangle_id)
    # double_unique_triangle_rows = np.where(unique_triangle_number==2)[0]
    # double_unique_triangles = unique_triangles[double_unique_triangle_rows]

    # triangle_triangle_matching = (vq(double_triangles,double_unique_triangles)[0])
    # triangle_triangle_matching = array_dict(double_unique_triangle_rows[triangle_triangle_matching],double_triangle_rows)

    # unique_triangle_cells = {}
    # for t in xrange(n_triangles):
    #     unique_triangle_cells[t] = []
    # for t in xrange(polydata.GetNumberOfCells()):
    #     if triangle_number[t] == 1:
    #         unique_triangle_cells[triangle_unique[t]] += list(polydata_cell_data.GetTuple(t))
    #     elif triangle_number[t] == 2:
    #         unique_triangle_cells[triangle_triangle_matching[t]] += list(polydata_cell_data.GetTuple(t))
    # triangle_cells = {}
    # for t in xrange(polydata.GetNumberOfCells()):
    #     if triangle_number[t] == 1:
    #         triangle_cells[t] = np.array(unique_triangle_cells[triangle_unique[t]],np.uint16)
    #     elif triangle_number[t] == 2:
    #         triangle_cells[t] = np.array(unique_triangle_cells[triangle_triangle_matching[t]],np.uint16)

    cell_data = vtk.vtkFloatArray()
    cell_data.SetNumberOfComponents(2)
    cell_data.SetNumberOfTuples(polydata.GetNumberOfCells())
    for t in xrange(polydata.GetNumberOfCells()):
        triangle_key = tuple(polydata_triangles[t])
        if len(triangle_cells[triangle_key]) == 2:
            cell_data.SetTuple(t,np.sort(triangle_cells[triangle_key]))
        else:
            cell_data.SetTuple(t,np.concatenate([triangle_cells[triangle_key],[1]]))
    polydata.GetCellData().SetScalars(cell_data)

    return triangle_cells


def image_to_vtk_polydata(img,considered_cells=None,mesh_center=None,coef=1.0,mesh_fineness=1.0):
    start_time = time()
    print "--> Generating vtk mesh from image"

    vtk_mesh = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_triangles = vtk.vtkCellArray()
    vtk_cells = vtk.vtkLongArray()
    
    nx, ny, nz = img.shape
    data_string = img.tostring('F')

    reader = vtk.vtkImageImport()
    reader.CopyImportVoidPointer(data_string, len(data_string))
    if img.dtype == np.uint8:
        reader.SetDataScalarTypeToUnsignedChar()
    else:
        reader.SetDataScalarTypeToUnsignedShort()
    reader.SetNumberOfScalarComponents(1)
    reader.SetDataExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    reader.SetWholeExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    reader.SetDataSpacing(*img.resolution)

    if considered_cells is None:
        considered_cells = np.unique(img)[1:]

    if mesh_center is None:
        mesh_center = np.array(img.resolution)*np.array(img.shape)/2.

    marching_cube_start_time = time()
    print "  --> Marching Cubes"
    contour = vtk.vtkDiscreteMarchingCubes()
    SetInput(contour,reader.GetOutput())
    contour.ComputeNormalsOn()
    contour.ComputeGradientsOn()
    contour.ComputeScalarsOn()
    for i,label in enumerate(considered_cells):
        contour.SetValue(i,label)
    contour.Update()
    marching_cube_end_time = time()
    print "  <-- Marching Cubes : ",contour.GetOutput().GetPoints().GetNumberOfPoints()," Points,",contour.GetOutput().GetNumberOfCells()," Triangles, ",len(np.unique(img)[1:])," Cells [",marching_cube_end_time - marching_cube_start_time,"s]"

    marching_cubes = contour.GetOutput()

    marching_cubes_cell_data = marching_cubes.GetCellData().GetArray(0)

    triangle_cell_start_time = time()
    print "    --> Listing triangles"
    print "      - ",marching_cubes.GetNumberOfCells()," triangles"
    marching_cubes_triangles = np.sort([[marching_cubes.GetCell(t).GetPointIds().GetId(i) for i in xrange(3)] for t in xrange(marching_cubes.GetNumberOfCells())])   
    triangle_cell_end_time = time()
    print "    <-- Listing triangles            [",triangle_cell_end_time - triangle_cell_start_time,"s]"

    triangle_cell_start_time = time()
    print "    --> Listing triangle cells"
    triangle_cell = np.array([marching_cubes_cell_data.GetTuple(t)[0] for t in xrange(marching_cubes.GetNumberOfCells())],np.uint16)
    triangle_cell_end_time = time()
    print "    <-- Listing triangle cells     [",triangle_cell_end_time - triangle_cell_start_time,"s]"

    triangle_cell_start_time = time()
    print "    --> Updating marching cubes mesh"
    vtk_mesh = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_triangles = vtk.vtkCellArray()
    vtk_cells = vtk.vtkLongArray()

    for label in considered_cells:

        # cell_start_time = time()

        cell_marching_cubes_triangles = marching_cubes_triangles[np.where(triangle_cell == label)]

        marching_cubes_point_ids = np.unique(cell_marching_cubes_triangles)

        marching_cubes_points = np.array([marching_cubes.GetPoints().GetPoint(p) for p in marching_cubes_point_ids])
        marching_cubes_center = marching_cubes_points.mean(axis=0)
        marching_cubes_points = marching_cubes_center + coef*(marching_cubes_points-marching_cubes_center) - mesh_center

        cell_points = []
        for p in xrange(marching_cubes_points.shape[0]):
            pid = vtk_points.InsertNextPoint(marching_cubes_points[p])
            cell_points.append(pid)
        cell_points = array_dict(cell_points,marching_cubes_point_ids)

        for t in xrange(cell_marching_cubes_triangles.shape[0]):
            poly = vtk_triangles.InsertNextCell(3)
            for i in xrange(3):
                pid = cell_marching_cubes_triangles[t][i]
                vtk_triangles.InsertCellPoint(cell_points[pid])
            vtk_cells.InsertValue(poly,label)

        # cell_end_time = time()
        # print "  --> Cell",label,":",cell_marching_cubes_triangles.shape[0],"triangles [",cell_end_time-cell_start_time,"s]"

    vtk_mesh.SetPoints(vtk_points)
    vtk_mesh.SetPolys(vtk_triangles)
    vtk_mesh.GetCellData().SetScalars(vtk_cells)

    triangle_cell_end_time = time()
    print "    <-- Updating marching cubes mesh [",triangle_cell_end_time - triangle_cell_start_time,"s]"

    decimation_start_time = time()
    print "  --> Decimation"
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    SetInput(smoother,vtk_mesh)
    smoother.SetFeatureAngle(30.0)
    smoother.SetPassBand(0.05)
    smoother.SetNumberOfIterations(25)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()

    decimate = vtk.vtkQuadricClustering()
    SetInput(decimate,smoother.GetOutput())
    decimate.SetNumberOfDivisions(*tuple(mesh_fineness*np.array(np.array(img.shape)*np.array(img.resolution)/2.,np.uint16)))
    decimate.SetFeaturePointsAngle(30.0)
    decimate.CopyCellDataOn()
    decimate.Update()

    decimation_end_time = time()
    print "  <-- Decimation     : ",decimate.GetOutput().GetPoints().GetNumberOfPoints()," Points,",decimate.GetOutput().GetNumberOfCells()," Triangles, ",len(considered_cells)," Cells [",decimation_end_time - decimation_start_time,"s]"

    end_time = time()
    print "<-- Generating vtk mesh from image      [",end_time-start_time,"s]"

    return decimate.GetOutput()


def image_to_vtk_cell_polydata(img,considered_cells=None,mesh_center=None,coef=1.0,mesh_fineness=1.0,smooth_factor=1.0):

    start_time = time()
    print "--> Generating vtk mesh from image"

    vtk_mesh = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_triangles = vtk.vtkCellArray()
    vtk_cells = vtk.vtkLongArray()
    
    nx, ny, nz = img.shape
    data_string = img.tostring('F')

    reader = vtk.vtkImageImport()
    reader.CopyImportVoidPointer(data_string, len(data_string))
    if img.dtype == np.uint8:
        reader.SetDataScalarTypeToUnsignedChar()
    else:
        reader.SetDataScalarTypeToUnsignedShort()
    reader.SetNumberOfScalarComponents(1)
    reader.SetDataExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    reader.SetWholeExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    reader.SetDataSpacing(*img.resolution)
    reader.Update()

    if considered_cells is None:
        considered_cells = np.unique(img)[1:]

    if mesh_center is None:
        #mesh_center = np.array(img.resolution)*np.array(img.shape)/2.
        mesh_center = np.array([0,0,0])

    for label in considered_cells:

        cell_start_time = time()

        cell_volume = (img==label).sum()*np.array(img.resolution).prod()

        # mask_data = vtk.vtkImageThreshold()
        # mask_data.SetInputConnection(reader.GetOutputPort())
        # mask_data.ThresholdBetween(label, label)
        # mask_data.ReplaceInOn()
        # mask_data.SetInValue(label)
        # mask_data.SetOutValue(0)
        contour = vtk.vtkDiscreteMarchingCubes()
        # contour.SetInput(mask_data.GetOutput())
        SetInput(contour,reader.GetOutput())
        contour.ComputeNormalsOn()
        contour.ComputeGradientsOn()
        contour.SetValue(0,label)
        contour.Update()

        # print "    --> Marching Cubes : ",contour.GetOutput().GetPoints().GetNumberOfPoints()," Points,",contour.GetOutput().GetNumberOfCells()," Triangles,  1 Cell"

        # decimate = vtk.vtkDecimatePro()
        # decimate.SetInputConnection(contour.GetOutputPort())
        # # decimate.SetTargetReduction(0.75)
        # decimate.SetTargetReduction(0.66)
        # # decimate.SetTargetReduction(0.5)
        # # decimate.SetMaximumError(2*np.sqrt(3))
        # decimate.Update()

        smooth_iterations = int(np.ceil(smooth_factor*8.))

        smoother = vtk.vtkWindowedSincPolyDataFilter()
        SetInput(smoother,contour.GetOutput())
        smoother.BoundarySmoothingOn()
        # smoother.BoundarySmoothingOff()
        smoother.FeatureEdgeSmoothingOn()
        # smoother.FeatureEdgeSmoothingOff()
        smoother.SetFeatureAngle(120.0)
        # smoother.SetPassBand(1)
        smoother.SetPassBand(0.01)
        smoother.SetNumberOfIterations(smooth_iterations)
        smoother.NonManifoldSmoothingOn()
        smoother.NormalizeCoordinatesOn()
        smoother.Update()

        divisions = int(np.ceil(np.power(cell_volume,1/3.)*mesh_fineness))

        decimate = vtk.vtkQuadricClustering()
        # decimate = vtk.vtkQuadricDecimation()
        # decimate = vtk.vtkDecimatePro()
        # decimate.SetInput(contour.GetOutput())
        SetInput(decimate,smoother.GetOutput())
        # decimate.SetTargetReduction(0.95)
        # decimate.AutoAdjustNumberOfDivisionsOff()
        decimate.SetNumberOfDivisions(divisions,divisions,divisions)
        decimate.SetFeaturePointsAngle(120.0)
        # decimate.AttributeErrorMetricOn()
        # decimate.ScalarsAttributeOn()
        # decimate.PreserveTopologyOn()
        # decimate.CopyCellDataOn()
        # decimate.SetMaximumCost(1.0)
        # decimate.SetMaximumCollapsedEdges(10000.0)
        decimate.Update()

        # print "    --> Decimation     : ",decimate.GetOutput().GetPoints().GetNumberOfPoints()," Points,",decimate.GetOutput().GetNumberOfCells()," Triangles,  1 Cell"

        cell_polydata = decimate.GetOutput()
        # cell_polydata = smoother.GetOutput()

        polydata_points = np.array([cell_polydata.GetPoints().GetPoint(p) for p in xrange(cell_polydata.GetPoints().GetNumberOfPoints())])
        polydata_center = polydata_points.mean(axis=0)
        polydata_points = polydata_center + coef*(polydata_points-polydata_center) - mesh_center

        cell_points = []
        for p in xrange(cell_polydata.GetPoints().GetNumberOfPoints()):
            pid = vtk_points.InsertNextPoint(polydata_points[p])
            cell_points.append(pid)
        cell_points = array_dict(cell_points,np.arange(cell_polydata.GetPoints().GetNumberOfPoints()))

        for t in xrange(cell_polydata.GetNumberOfCells()):
            poly = vtk_triangles.InsertNextCell(3)
            for i in xrange(3):
                pid = cell_polydata.GetCell(t).GetPointIds().GetId(i)
                vtk_triangles.InsertCellPoint(cell_points[pid])
                vtk_cells.InsertValue(poly,label)

        cell_end_time = time()
        print "  --> Cell",label,":",decimate.GetOutput().GetNumberOfCells(),"triangles (",cell_volume," microm3 ) [",cell_end_time-cell_start_time,"s]"

    vtk_mesh.SetPoints(vtk_points)
    vtk_mesh.SetPolys(vtk_triangles)
    vtk_mesh.GetCellData().SetScalars(vtk_cells)

    print "  <-- Cell Mesh      : ",vtk_mesh.GetPoints().GetNumberOfPoints()," Points,",vtk_mesh.GetNumberOfCells()," Triangles, ",len(considered_cells)," Cells"

    end_time = time()
    print "<-- Generating vtk mesh from image      [",end_time-start_time,"s]"

    return vtk_mesh


def topomesh_to_vtk_polydata(topomesh,degree=2,positions=None,topomesh_center=None,coef=1):
    import numpy as np
    import vtk
    from time import time
    from openalea.container import array_dict

    if positions is None:
        positions = topomesh.wisp_property('barycenter',0)
        
    if topomesh_center is None:
        topomesh_center = np.mean(positions.values(),axis=0)
#        topomesh_center = np.array([0,0,0])
        print topomesh_center

    vtk_mesh = vtk.vtkPolyData()
    vtk_points = vtk.vtkPoints()
    vtk_edges = vtk.vtkCellArray()
    vtk_triangles = vtk.vtkCellArray()
    vtk_cells = vtk.vtkLongArray()

    start_time = time()
    print "--> Creating VTK PolyData"
    
    if degree == 3:
        for c in topomesh.wisps(3):
            cell_points = []
            cell_center = np.mean([positions[v] for v in topomesh.borders(3,c,3)],axis=0)
            for v in topomesh.borders(3,c,3):
                position = cell_center + coef*(positions[v]-cell_center) - topomesh_center
                position[np.where(np.abs(position)<.001)] =0.
                p = vtk_points.InsertNextPoint(position)
                cell_points.append(p)
            cell_points = array_dict(cell_points,list(topomesh.borders(3,c,3)))

            for t in topomesh.borders(3,c):
                poly = vtk_triangles.InsertNextCell(3)
                for v in topomesh.borders(2,t,2):
                    vtk_triangles.InsertCellPoint(cell_points[v])
                vtk_cells.InsertValue(poly,c)

    elif degree == 2:
        for t in topomesh.wisps(2):
            triangle_points = []
            triangle_center = np.mean([positions[v] for v in topomesh.borders(2,t,2)],axis=0)
            for v in topomesh.borders(2,t,2):
                position = triangle_center + coef*(positions[v]-triangle_center) - topomesh_center
                position[np.where(np.abs(position)<.001)] =0.
                p = vtk_points.InsertNextPoint(position)
                triangle_points.append(p)
            triangle_points = array_dict(triangle_points,list(topomesh.borders(2,t,2)))
            poly = vtk_triangles.InsertNextCell(3)
            for v in topomesh.borders(2,t,2):
                vtk_triangles.InsertCellPoint(triangle_points[v])
            vtk_cells.InsertValue(poly,list(topomesh.regions(2,t))[0])

    elif degree == 1:
        points = []
        for v in topomesh.wisps(0):
            position = positions[v]
            position[np.where(np.abs(position)<.001)] =0.
            p = vtk_points.InsertNextPoint(position)
            points.append(p)
        points = array_dict(points,list(topomesh.wisps(0)))

        for e in topomesh.wisps(1):
            # if topomesh.wisp_property('epidermis',1)[e]:
            # if True:
            if len(list(topomesh.regions(1,e))) > 2:
                c = vtk_edges.InsertNextCell(2)
                for v in topomesh.borders(1,e):
                    vtk_edges.InsertCellPoint(points[v])

    print"  --> Linking Mesh"
    vtk_mesh.SetPoints(vtk_points)
    vtk_mesh.SetPolys(vtk_triangles)
    vtk_mesh.SetLines(vtk_edges)
    vtk_mesh.GetCellData().SetScalars(vtk_cells)

    end_time = time()
    print "<-- Creating VTK PolyData      [",end_time-start_time,"s]"

    return vtk_mesh


def image_to_pgl_mesh(img,sampling=4,cell_coef=1.0,mesh_fineness=1.0,smooth=1.0,colormap=None):
    try:
        import openalea.plantgl.all as pgl
    except ImportError:
        print "PlantGL needs to be installed to use this functionality"
        raise

    try:
        resolution = img.resolution
    except:
        resolution = (1.0,1.0,1.0)

    img = SpatialImage(img[0:-1:sampling,0:-1:sampling,0:-1:sampling],resolution=tuple(np.array(resolution)*sampling))

    img_polydata = image_to_vtk_cell_polydata(img,coef=cell_coef,mesh_fineness=mesh_fineness,smooth_factor=smooth)
    # img_mesh = vtk_polydata_to_triangular_mesh(img_polydata)
    # return img_mesh._repr_geom_()
    img_mesh = vtk_polydata_to_cell_triangular_meshes(img_polydata)
    start_time = time()
    print "--> Constructing geometry (pGL)"
    scene = pgl.Scene()
    for c in img_mesh.keys():
        scene += draw_triangular_mesh(img_mesh[c],mesh_id=c,colormap=colormap)
    end_time = time()
    print "<-- Constructing geometry (pGL) [",end_time - start_time,"s]"
    return scene


def image_to_triangular_mesh(img,sampling=4,cell_coef=1.0,mesh_fineness=1.0,smooth=1.0,resolution=None):
    from vplants.meshing.triangular_mesh import TriangularMesh
    from time import time

    if resolution is None:
        try:
            resolution = img.resolution
        except:
            resolution = (1.0,1.0,1.0)
    img = SpatialImage(img[0:-1:sampling,0:-1:sampling,0:-1:sampling],resolution=tuple(np.array(resolution)*sampling))

    img_polydata = image_to_vtk_cell_polydata(img,coef=cell_coef,mesh_fineness=mesh_fineness,smooth_factor=smooth)
    img_mesh = vtk_polydata_to_triangular_mesh(img_polydata)
    return img_mesh





