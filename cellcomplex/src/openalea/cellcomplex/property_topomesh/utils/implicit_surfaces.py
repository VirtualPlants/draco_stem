# -*- coding: utf-8 -*-
# -*- python -*-
#
#       PropertyTopomesh
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
from copy import deepcopy

def implicit_surface(density_field,size,resolution,iso=0.5):
    import numpy as np
    from scipy.cluster.vq                       import kmeans, vq
    from openalea.container import array_dict
   
    from skimage.measure import marching_cubes
    surface_points, surface_triangles = marching_cubes(density_field,iso)
    
    surface_points = (np.array(surface_points))*(size*resolution/np.array(density_field.shape)) - size*resolution/2.

    points_ids = np.arange(len(surface_points))
    points_to_delete = []
    for p,point in enumerate(surface_points):
        matching_points = np.sort(np.where(vq(surface_points,np.array([point]))[1] == 0)[0])
        if len(matching_points) > 1:
            points_to_fuse = matching_points[1:]
            for m_p in points_to_fuse:
                surface_triangles[np.where(surface_triangles==m_p)] = matching_points[0]
                points_to_delete.append(m_p)

    points_to_delete = np.unique(points_to_delete)
    print len(points_to_delete),"points deleted"
    surface_points = np.delete(surface_points,points_to_delete,0)
    points_ids = np.delete(points_ids,points_to_delete,0)
    surface_triangles = array_dict(np.arange(len(surface_points)),points_ids).values(surface_triangles)

    for p,point in enumerate(surface_points):
        matching_points = np.where(vq(surface_points,np.array([point]))[1] == 0)[0]
        if len(matching_points) > 1:
            print p,point
            raw_input()

    triangles_to_delete = []
    for t,triangle in enumerate(surface_triangles):
        if len(np.unique(triangle)) < 3:
            triangles_to_delete.append(t)
        # elif triangle.max() >= len(surface_points):
        #     triangles_to_delete.append(t)
    surface_triangles = np.delete(surface_triangles,triangles_to_delete,0)

    return surface_points, surface_triangles

def marching_cubes(field,iso=0.5):
    try:
        from skimage.measure import marching_cubes
        surface_points, surface_triangles = marching_cubes(density_field,iso)

    except ImportError:
        print "Please try to install SciKit-Image!"

        from mayavi import mlab
        from mayavi.mlab import contour3d

        mlab.clf()
        surface = mlab.contour3d(field,contours=[iso])

        my_actor=surface.actor.actors[0] 
        poly_data_object=my_actor.mapper.input 
        surface_points = (np.array(poly_data_object.points) - np.array([abs(grid_points/2.),abs(grid_points/2.),abs(grid_points/2.)])[np.newaxis,:])*(grid_max/abs(grid_points/2.))
        surface_triangles = poly_data_object.polys.data.to_array().reshape([-1,4]) 
        surface_triangles = surface_triangles[:,1:]

    return surface_points, surface_triangles

def vtk_marching_cubes(field,iso=0.5):

    import vtk

    int_field = (np.minimum(field*255,255)).astype(np.uint8)
    nx, ny, nz = int_field.shape
    data_string = int_field.tostring('F')

    reader = vtk.vtkImageImport()
    reader.CopyImportVoidPointer(data_string, len(data_string))
    reader.SetDataScalarTypeToUnsignedChar()
    reader.SetNumberOfScalarComponents(1)
    reader.SetDataExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    reader.SetWholeExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    reader.Update()

    contour = vtk.vtkImageMarchingCubes()
    if vtk.VTK_MAJOR_VERSION <= 5:
        contour.SetInput(reader.GetOutput())
    else:
        contour.SetInputData(reader.GetOutput())   
    contour.ComputeNormalsOn()
    contour.ComputeGradientsOn()
    contour.SetValue(0,int(iso*255))
    contour.Update()

    field_polydata = contour.GetOutput()

    polydata_points = np.array([field_polydata.GetPoints().GetPoint(p) for p in xrange(field_polydata.GetPoints().GetNumberOfPoints())])
    polydata_triangles =  np.array([[field_polydata.GetCell(t).GetPointIds().GetId(i) for i in xrange(3)] for t in xrange(field_polydata.GetNumberOfCells())])

    return polydata_points, polydata_triangles

def implicit_surface_topomesh(density_field,size,resolution,iso=0.5,center=True):
    import numpy as np
    from scipy.cluster.vq                       import kmeans, vq
    from openalea.container import array_dict, PropertyTopomesh

    surface_points, surface_triangles = vtk_marching_cubes(density_field,iso)

    surface_points = (np.array(surface_points))*(size*resolution/np.array(density_field.shape)) 
    if center:
        surface_points -= np.array(density_field.shape)*resolution/2.

    # points_ids = np.arange(len(surface_points))
    # points_to_delete = []
    # for p,point in enumerate(surface_points):
    #     matching_points = np.sort(np.where(vq(surface_points,np.array([point]))[1] == 0)[0])
    #     if len(matching_points) > 1:
    #         points_to_fuse = matching_points[1:]
    #         for m_p in points_to_fuse:
    #             surface_triangles[np.where(surface_triangles==m_p)] = matching_points[0]
    #             points_to_delete.append(m_p)

    # points_to_delete = np.unique(points_to_delete)
    # print len(points_to_delete),"points deleted"
    # surface_points = np.delete(surface_points,points_to_delete,0)
    # points_ids = np.delete(points_ids,points_to_delete,0)
    # surface_triangles = array_dict(np.arange(len(surface_points)),points_ids).values(surface_triangles)

    # for p,point in enumerate(surface_points):
    #     matching_points = np.where(vq(surface_points,np.array([point]))[1] == 0)[0]
    #     if len(matching_points) > 1:
    #         print p,point
    #         raw_input()

    # triangles_to_delete = []
    # for t,triangle in enumerate(surface_triangles):
    #     if len(np.unique(triangle)) < 3:
    #         triangles_to_delete.append(t)
    #     # elif triangle.max() >= len(surface_points):
    #     #     triangles_to_delete.append(t)
    # surface_triangles = np.delete(surface_triangles,triangles_to_delete,0)

    surface_topomesh = PropertyTopomesh(3)

    for p in surface_points:
        pid = surface_topomesh.add_wisp(0)

    triangle_edge_list  = np.array([[1, 2],[0, 2],[0, 1]])
    surface_edges = np.sort(np.concatenate(surface_triangles[:,triangle_edge_list]))
    _,unique_edges = np.unique(np.ascontiguousarray(surface_edges).view(np.dtype((np.void,surface_edges.dtype.itemsize * surface_edges.shape[1]))),return_index=True)
    surface_edges = surface_edges[unique_edges]

    for e in surface_edges:
        eid = surface_topomesh.add_wisp(1)
        for pid in e:
            surface_topomesh.link(1,eid,pid)

    surface_triangle_edges = np.sort(np.concatenate(surface_triangles[:,triangle_edge_list]))
    surface_triangle_edge_matching = vq(surface_triangle_edges,surface_edges)[0].reshape(surface_triangles.shape[0],3)

    for t in surface_triangles:
        fid = surface_topomesh.add_wisp(2)
        for eid in surface_triangle_edge_matching[fid]:
            surface_topomesh.link(2,fid,eid)

    cid = surface_topomesh.add_wisp(3)
    for fid in surface_topomesh.wisps(2):
        surface_topomesh.link(3,cid,fid)

    surface_topomesh.update_wisp_property('barycenter',0,array_dict(surface_points),keys=list(surface_topomesh.wisps(0)))

    return surface_topomesh


def spherical_density_function(positions,sphere_radius,k=0.1):
    import numpy as np
    
    def density_func(x,y,z):
        density = np.zeros_like(x+y+z,float)
        max_radius = sphere_radius
        # max_radius = 0.

        for p in positions.keys():
            cell_distances = np.power(np.power(x-positions[p][0],2) + np.power(y-positions[p][1],2) + np.power(z-positions[p][2],2),0.5)
            density += 1./2. * (1. - np.tanh(k*(cell_distances - (sphere_radius+max_radius)/2.)))
        return density
    return density_func


def point_spherical_density(positions,points,sphere_radius=5,k=1.0):
    return spherical_density_function(positions,sphere_radius=sphere_radius,k=k)(points[:,0],points[:,1],points[:,2])

