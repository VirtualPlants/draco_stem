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
from scipy import ndimage as nd

from scipy.cluster.vq                       import kmeans, vq
from array                                  import array

from openalea.container                     import PropertyTopomesh, array_dict

from copy                                   import copy
from time                                   import time
import os
import sys
import pickle


def save_property_topomesh(topomesh, path, cells_to_save=None, properties_to_save=dict([(0,['barycenter']),(1,[]),(2,[]),(3,[])]),**kwargs):
    if cells_to_save is None:
        cells_to_save = np.array(list(topomesh.wisps(3)))
    triangles_to_save = np.array(np.unique(np.concatenate([np.array(list(topomesh.borders(3,c))) for c in cells_to_save])),int)
    edges_to_save = np.array(np.unique(np.concatenate([np.array(list(topomesh.borders(2,t))) for t in triangles_to_save])),int)
    vertices_to_save = np.array(np.unique(np.concatenate([np.array(list(topomesh.borders(1,e))) for e in edges_to_save])),int)

    original_pids = kwargs.get('original_pids',False)
    original_eids = kwargs.get('original_eids',False)
    original_fids = kwargs.get('original_fids',False)
    original_cids = kwargs.get('original_cids',True)

    sub_topomesh = PropertyTopomesh(3)

    vertices_to_pids = {}
    for v in vertices_to_save:
        if original_pids:
            pid = sub_topomesh.add_wisp(0,v)
        else:
            pid = sub_topomesh.add_wisp(0)
        vertices_to_pids[v] = pid

    edges_to_eids = {}
    for e in edges_to_save:
        if original_eids:
            eid = sub_topomesh.add_wisp(1,e)
        else:
            eid = sub_topomesh.add_wisp(1)
        edges_to_eids[e] = eid
        for v in topomesh.borders(1,e):
            sub_topomesh.link(1,eid,vertices_to_pids[v])

    triangles_to_fids = {}
    for t in triangles_to_save:
        if original_fids:
            fid = sub_topomesh.add_wisp(2,t)
        else:
            fid = sub_topomesh.add_wisp(2)
        triangles_to_fids[t] = fid
        for e in topomesh.borders(2,t):
            sub_topomesh.link(2,fid,edges_to_eids[e])

    cells_to_cids = {}
    for c in cells_to_save:
        if original_cids:
            cid = sub_topomesh.add_wisp(3,c)
        else:
            cid = sub_topomesh.add_wisp(3,c)
        cells_to_cids[c] = cid
        for t in topomesh.borders(3,c):
            sub_topomesh.link(3,cid,triangles_to_fids[t])

    wisps_to_save = {}
    wisps_to_save[0] = vertices_to_save
    wisps_to_save[1] = edges_to_save
    wisps_to_save[2] = triangles_to_save
    wisps_to_save[3] = cells_to_save

    wisps_to_wids = {}
    wisps_to_wids[0] = vertices_to_pids 
    wisps_to_wids[1] = edges_to_eids 
    wisps_to_wids[2] = triangles_to_fids 
    wisps_to_wids[3] = cells_to_cids 

    if not 0 in properties_to_save.keys():
        properties_to_save[0] = []
    if not 'barycenter' in properties_to_save[0]:
        properties_to_save[0].append('barycenter')

    for degree in properties_to_save.keys():
        for property_name in properties_to_save[degree]:
            print "Property ",property_name,'(',degree,')'
            if topomesh.has_wisp_property(property_name,degree=degree,is_computed=True):
                wids_to_save = array_dict(wisps_to_wids[degree]).values(wisps_to_save[degree])
                sub_topomesh.update_wisp_property(property_name,degree,array_dict(topomesh.wisp_property(property_name,degree).values(wisps_to_save[degree]),wids_to_save))

    pickle.dump(sub_topomesh,open(path,"wb"))


def save_tissue_property_topomesh(topomesh,tissue_filename='tissue.zip'):
    from openalea.celltissue import Tissue, TissueDB, topen, Config, ConfigItem, ConfigFormat

    tissue = Tissue()

    ptyp = tissue.add_type("point")
    etyp = tissue.add_type("edge")
    ftyp = tissue.add_type("face")
    ctyp = tissue.add_type("cell")

    mesh_id = tissue.add_relation("mesh",(ptyp,etyp,ftyp,ctyp))
    mesh = tissue.relation(mesh_id)

    cells = {}
    for c in topomesh.wisps(3):
        if len(list(topomesh.borders(3,c)))>0:
            cid = tissue.add_element(ctyp)
            mesh.add_wisp(3,cid)
            cells[c] = cid

    faces = {}
    for t in topomesh.wisps(2):
        fid = tissue.add_element(ftyp)
        mesh.add_wisp(2,fid)
        faces[t] = fid

    edges = {}
    for e in topomesh.wisps(1):
        eid = tissue.add_element(etyp)
        mesh.add_wisp(1,eid)
        edges[e] = eid

    points = {}
    for v in topomesh.wisps(0):
        pid = tissue.add_element(ptyp)
        mesh.add_wisp(0,pid)
        points[v] = pid

    for e in topomesh.wisps(1):
        for v in topomesh.borders(1,e):
            mesh.link(1,edges[e],points[v])

    for t in topomesh.wisps(2):
        for e in topomesh.borders(2,t):
            mesh.link(2,faces[t],edges[e])

    for c in topomesh.wisps(3):
        if len(list(topomesh.borders(3,c)))>0:
            for t in topomesh.borders(3,c):
                mesh.link(3,cells[c],faces[t])

    positions = {}
    for v in topomesh.wisps(0):
        positions[points[v]] = tuple(topomesh.wisp_property('barycenter',0)[v])

    cfg = ConfigFormat("config")
    cfg.add_section("topology")
    cfg.add_item(ConfigItem("cell",ctyp) )
    cfg.add_item(ConfigItem("face",ftyp) )
    cfg.add_item(ConfigItem("edge",etyp) )
    cfg.add_item(ConfigItem("point",ptyp) )
    cfg.add_item(ConfigItem("mesh_id",mesh_id))

    tissue_db = TissueDB()
    tissue_db.set_tissue(tissue)
    tissue_db.set_property("position",positions)
    tissue_db.set_description("position","position of points")
    tissue_db.set_config("config",cfg.config())

    tissue_db.write(tissue_filename)

def save_ply_cellcomplex_topomesh(topomesh,ply_filename,color_faces=False,colormap=None,oriented=True):
    """
    Implementing the PLY standard defined at Saisnbury Computational Workshop 2015
    """

    start_time =time()
    print "--> Saving .ply"
    ply_file = open(ply_filename,'w+')

    if oriented:
        compute_topomesh_property(topomesh,'oriented_borders',2)
        compute_topomesh_property(topomesh,'oriented_vertices',2)
        compute_topomesh_property(topomesh,'oriented_borders',3)

    property_types = {}
    property_types['bool'] = "int"
    property_types['int'] = "int"
    property_types['int32'] = "int"
    property_types['int64'] = "int"
    property_types['float'] = "float"
    property_types['float32'] = "float"
    property_types['float64'] = "float"
    property_types['object'] = "list"

    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex "+str(topomesh.nb_wisps(0))+"\n")
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("element face "+str(topomesh.nb_wisps(2))+"\n")
    ply_file.write("property list int int vertex_index\n")
    if color_faces:
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        triangle_data = array_dict(np.array([(list(topomesh.regions(2,t))[0])%256 for t in topomesh.wisps(2)]),list(topomesh.wisps(2)))
    ply_file.write("element edge "+str(topomesh.nb_wisps(1))+"\n")
    ply_file.write("property int source\n")
    ply_file.write("property int target\n")
    ply_file.write("element volume "+str(topomesh.nb_wisps(3))+"\n")
    ply_file.write("property list int int face_index\n")
    ply_file.write("property int label\n")
    ply_file.write("end_header\n")   

    vertex_index = {}
    for v,pid in enumerate(topomesh.wisps(0)):
        ply_file.write(str(topomesh.wisp_property('barycenter',0)[pid][0])+" ")
        ply_file.write(str(topomesh.wisp_property('barycenter',0)[pid][1])+" ")
        ply_file.write(str(topomesh.wisp_property('barycenter',0)[pid][2])+" ")
        ply_file.write("\n")
        vertex_index[pid] = v

    face_index = {}
    for t,fid in enumerate(topomesh.wisps(2)):

        if oriented:
            oriented_face_pids = topomesh.wisp_property('oriented_vertices',2)[fid]
        else:
            face_pids = np.array(list(topomesh.borders(2,fid,2)))
            face_edges = np.array([list(topomesh.borders(1,eid)) for eid in topomesh.borders(2,fid)])

            print fid," : ",face_pids, face_edges

            oriented_face_pids = [face_pids[0]]
            while len(oriented_face_pids) < len(face_pids):
                current_pid = oriented_face_pids[-1]
                pid_edges = face_edges[np.where(face_edges==current_pid)[0]]
                candidate_pids = set(list(np.unique(pid_edges))).difference({current_pid})
                if len(oriented_face_pids)>1:
                    candidate_pids = candidate_pids.difference({oriented_face_pids[-2]})
                if len(list(candidate_pids))==0:
                    candidate_pids = set(list(face_pids)).difference(list(oriented_face_pids))
                oriented_face_pids += [list(candidate_pids)[0]]
            print fid," : ",oriented_face_pids

        ply_file.write(str(len(list(topomesh.borders(2,fid,2))))+" ")
        for pid in oriented_face_pids:
            ply_file.write(str(vertex_index[pid])+" ")
        if color_faces:
            if colormap is None:
                ply_file.write(str(triangle_data[fid])+" ")
                ply_file.write(str(triangle_data[fid])+" ")
                ply_file.write(str(triangle_data[fid])+" ")
            else:
                color = colormap._color_points.values()[int(triangle_data[fid])]
                ply_file.write(str(int(255*color[0]))+" ")
                ply_file.write(str(int(255*color[1]))+" ")
                ply_file.write(str(int(255*color[2]))+" ")
        ply_file.write("\n")
        face_index[fid] = t

    edge_index = {}
    for e,eid in enumerate(topomesh.wisps(1)):
        ply_file.write(str(vertex_index[list(topomesh.borders(1,eid))[0]])+" ")
        ply_file.write(str(vertex_index[list(topomesh.borders(1,eid))[1]])+" ")
        ply_file.write("\n")
        edge_index[eid] = e

    for c, cid in enumerate(topomesh.wisps(3)):
        if oriented:
            oriented_cell_fids = topomesh.wisp_property('oriented_borders',3)[cid][0]
            oriented_cell_fid_orientations = topomesh.wisp_property('oriented_borders',3)[cid][1]
            ply_file.write(str(len(oriented_cell_fids))+" ")
            for fid,orientation in zip(oriented_cell_fids,oriented_cell_fid_orientations):
                ply_file.write(str(orientation*(face_index[fid]+1))+" ")
        else:
            for fid in topomesh.borders(3,cid):
                ply_file.write(str(face_index[fid]+1)+" ")
        ply_file.write(str(cid)+" ")
        ply_file.write("\n")

    ply_file.flush()
    ply_file.close()

    end_time = time()
    print "<-- Saving .ply        [",end_time-start_time,"s]"


def save_ply_property_topomesh(topomesh,ply_filename,properties_to_save=dict([(0,[]),(1,['length']),(2,['area','epidermis']),(3,[])]),color_faces=False, coordinatepropname = 'barycenter', verbose = True):
    if verbose:
        start_time =time()
        print "--> Saving .ply"
    ply_file = open(ply_filename,'w+')

    property_types = {}
    property_types['bool'] = "int"
    property_types['int'] = "int"
    property_types['int32'] = "int"
    property_types['int64'] = "int"
    property_types['float'] = "float"
    property_types['float32'] = "float"
    property_types['float64'] = "float"
    property_types['object'] = "list"

    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex "+str(topomesh.nb_wisps(0))+"\n")
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    for property_name in properties_to_save[0]:
        if property_name != coordinatepropname and topomesh.has_wisp_property(property_name,0,is_computed=True):
            property_type = property_types[str(topomesh.wisp_property(property_name,0).values().dtype)]
            ply_file.write("property "+property_type+" "+property_name+"\n")

    ply_file.write("element face "+str(topomesh.nb_wisps(2))+"\n")
    ply_file.write("property list int int vertex_index\n")
    if color_faces:
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        triangle_data = array_dict(np.array([list(topomesh.regions(2,t))[0]%256 for t in topomesh.wisps(2)]),list(topomesh.wisps(2)))
    for property_name in properties_to_save[2]:
        if topomesh.has_wisp_property(property_name,2,is_computed=True):
            property_type = property_types[str(topomesh.wisp_property(property_name,2).values().dtype)]
            ply_file.write("property "+property_type+" "+property_name+"\n")
    ply_file.write("element edge "+str(topomesh.nb_wisps(1))+"\n")
    ply_file.write("property int source\n")
    ply_file.write("property int target\n")
    ply_file.write("property list int int face_index\n")
    for property_name in properties_to_save[1]:
        if topomesh.has_wisp_property(property_name,1,is_computed=True):
            property_type = property_types[str(topomesh.wisp_property(property_name,1).values().dtype)]
            ply_file.write("property "+property_type+" "+property_name+"\n")
    ply_file.write("element volume "+str(topomesh.nb_wisps(3))+"\n")
    ply_file.write("property list int int face_index\n")
    ply_file.write("property int label\n")

    ply_file.write("end_header\n")    
    vertex_index = {}
    for v,pid in enumerate(topomesh.wisps(0)):
        ply_file.write(str(topomesh.wisp_property(coordinatepropname,0)[pid][0])+" ")
        ply_file.write(str(topomesh.wisp_property(coordinatepropname,0)[pid][1])+" ")
        ply_file.write(str(topomesh.wisp_property(coordinatepropname,0)[pid][2])+" ")
        for property_name in properties_to_save[0] :
            if property_name != coordinatepropname and topomesh.has_wisp_property(property_name,0,is_computed=True):
                property_type = property_types[str(topomesh.wisp_property(property_name,0).values().dtype)]
                if property_type == 'int':
                    ply_file.write(str(int(topomesh.wisp_property(property_name,0)[pid]))+" ")
                else:
                    ply_file.write(str(topomesh.wisp_property(property_name,0)[pid])+" ")
        ply_file.write("\n")
        vertex_index[pid] = v        

    face_index = {}
    for t,fid in enumerate(topomesh.wisps(2)):
        ply_file.write(str(len(list(topomesh.borders(2,fid,2))))+" ")
        for pid in topomesh.borders(2,fid,2):
            ply_file.write(str(vertex_index[pid])+" ")
        # ply_file.write(str(len(list(topomesh.borders(2,fid))))+" ")
        # for eid in topomesh.borders(2,fid):
        #     ply_file.write(str(edge_index[eid])+" ")
        if color_faces:
            ply_file.write(str(triangle_data[fid])+" ")
            ply_file.write(str(triangle_data[fid])+" ")
            ply_file.write(str(triangle_data[fid])+" ")
        for property_name in properties_to_save[2]:
            if topomesh.has_wisp_property(property_name,2,is_computed=True):
                property_type = property_types[str(topomesh.wisp_property(property_name,2).values().dtype)]
                if property_type == 'int':
                    ply_file.write(str(int(topomesh.wisp_property(property_name,2)[fid]))+" ")
                else:
                    ply_file.write(str(topomesh.wisp_property(property_name,2)[fid])+" ")
        ply_file.write("\n")
        face_index[fid] = t

    edge_index = {}
    for e,eid in enumerate(topomesh.wisps(1)):
        ply_file.write(str(vertex_index[list(topomesh.borders(1,eid))[0]])+" ")
        ply_file.write(str(vertex_index[list(topomesh.borders(1,eid))[1]])+" ")
        ply_file.write(str(len(list(topomesh.regions(1,eid))))+" ")
        for fid in topomesh.regions(1,eid):
            ply_file.write(str(face_index[fid])+" ")
        for property_name in properties_to_save[1]:
            if topomesh.has_wisp_property(property_name,1,is_computed=True):
                ply_file.write(str(topomesh.wisp_property(property_name,1)[eid])+" ")
        ply_file.write("\n")
        edge_index[eid] = e

    for c, cid in enumerate(topomesh.wisps(3)):
        ply_file.write(str(len(list(topomesh.borders(3,cid))))+" ")
        for fid in topomesh.borders(3,cid):
            ply_file.write(str(face_index[fid])+" ")
        ply_file.write(str(cid)+" ")
        ply_file.write("\n")

    ply_file.flush()
    ply_file.close()

    if verbose:
        end_time = time()
        print "<-- Saving .ply        [",end_time-start_time,"s]"


def read_ply_property_topomesh(ply_filename, verbose = True):
    """
    """
    import re
    from openalea.cellcomplex.property_topomesh.utils.array_tools import array_unique

    if verbose:
        start_time =time()
        print "--> Reading .ply"

    property_types = {}
    property_types['int'] = 'int'
    property_types['int32'] = 'int'
    property_types['uint'] = 'int'
    property_types['uint8'] = 'int'
    property_types['uchar'] = 'int'
    property_types['float'] = 'float'
    property_types['float32'] = 'float'
    property_types['list'] = 'list'
    property_types['tensor'] = 'tensor'

    ply_file = open(ply_filename,'rU')
    ply_stream = enumerate(ply_file,1)
    assert "ply" in ply_stream.next()[1]
    assert "ascii" in ply_stream.next()[1]

    n_wisps = {}
    properties = {}
    properties_types = {}
    properties_list_types = {}
    properties_tensor_dims = {}
    element_name = ""
    property_name = ""
    elements = []

    lineno, line = ply_stream.next()
    while not 'end_header' in line:
        try:
            if re.split(' ',line)[0] == 'element':
                element_name = re.split(' ',line)[1]
                elements.append(element_name)
                n_wisps[element_name] = int(re.split(' ',line)[2])
                properties[element_name] = []
                properties_types[element_name] = {}
                properties_list_types[element_name] = {}
                properties_tensor_dims[element_name] = {}
                
            if re.split(' ',line)[0] == 'property':
                print line
                property_name = re.split(' ',line)[-1][:-1]
                properties[element_name].append(property_name)
                properties_types[element_name][property_name] = re.split(' ',line)[1]
                if properties_types[element_name][property_name] == 'list':
                    list_type = re.split(' ',line)[-2]
                    properties_list_types[element_name][property_name] = list_type
                elif properties_types[element_name][property_name] == 'tensor':
                    properties_tensor_dims[element_name][property_name] = (np.array(re.split(' ',line))[:-2]=='int').sum()
                    list_type = re.split(' ',line)[-2]
                    properties_list_types[element_name][property_name] = list_type
        except Exception, e:
                raise ValueError(ply_filename, lineno, line, e)
            
        lineno, line = ply_stream.next()

    if verbose: print n_wisps
    if verbose: print properties
    if verbose: print properties_list_types

    element_properties = {}

    for element_name in elements:
        element_properties[element_name] = {}
        for wid in xrange(n_wisps[element_name]):
            lineno, line = ply_stream.next()
            line = line.strip()
            line_props = {}
            prop_index = 0
            try:
                for prop in properties[element_name]:
                    prop_type = properties_types[element_name][prop]
                    if property_types[prop_type] == 'float':
                        line_props[prop] = float(re.split(' ',line)[prop_index])
                        prop_index += 1
                    elif property_types[prop_type] == 'int':
                        line_props[prop] = int(re.split(' ',line)[prop_index])
                        prop_index += 1
                    elif property_types[prop_type] == 'list':
                        list_length = int(re.split(' ',line)[prop_index])
                        prop_index += 1
                        list_type =  properties_list_types[element_name][prop]
                        if property_types[list_type] == 'float':
                            line_props[prop] = [float(p) for p in re.split(' ',line)[prop_index:prop_index+list_length]]
                        elif property_types[list_type] == 'int':
                            line_props[prop] = [int(p) for p in re.split(' ',line)[prop_index:prop_index+list_length]]
                        prop_index += list_length
                    elif property_types[prop_type] == 'tensor':
                        n_dims = properties_tensor_dims[element_name][prop]
                        tensor_dims = tuple(np.array(re.split(' ',line)[prop_index:prop_index+n_dims]).astype(int))
                        prop_index += n_dims
                        list_type =  properties_list_types[element_name][prop]
                        line_props[prop] = np.array(re.split(' ',line)[prop_index:prop_index+np.prod(tensor_dims)]).astype(property_types[list_type]).reshape(tensor_dims)
                        prop_index += np.prod(tensor_dims)
            except Exception, e:
                raise ValueError(ply_filename, lineno, line, e)

            element_properties[element_name][wid] = line_props
    ply_file.close()

    if verbose:
        print "<-- Parsing .ply        [",time()-start_time,"s]"

    element_matching = {}

    point_positions = {}
    for pid in xrange(n_wisps['vertex']):
        point_positions[pid] = np.array([element_properties['vertex'][pid][dim] for dim in ['x','y','z']])

    face_vertices = {}
    for fid in xrange(n_wisps['face']):
        if element_properties['face'][fid].has_key('vertex_index'):
            face_vertices[fid] = element_properties['face'][fid]['vertex_index']
        elif element_properties['face'][fid].has_key('vertex_indices'):
            face_vertices[fid] = element_properties['face'][fid]['vertex_indices']

    timecheck = verbose

    if timecheck: check_time =time()
    if verbose: print len(point_positions)," Points, ", len(face_vertices), " Faces"

    unique_points = array_unique(np.array(point_positions.values()))
    if len(unique_points) == len(point_positions):
        unique_points = np.array(point_positions.values())
        point_matching = array_dict(point_positions.keys(),point_positions.keys())
    else:
        point_matching = array_dict(vq(np.array(point_positions.values()),unique_points)[0],point_positions.keys())

    element_matching['vertex'] = point_matching
    if verbose: print len(unique_points)," Unique Points"
    if timecheck: print 'point unicity test:',time()-check_time 

    if timecheck: check_time =time()
    faces = np.array(face_vertices.values())
    if faces.ndim == 2:
        if verbose: print "Faces = Triangles !"
        triangular = True
        triangles = np.sort(point_matching.values(faces))
        unique_triangles = array_unique(triangles)
        if len(unique_triangles) == len(triangles):
            unique_triangles = triangles
            triangle_matching = array_dict(face_vertices.keys(),face_vertices.keys())
        else:
            triangle_matching = array_dict(vq(triangles,unique_triangles)[0],face_vertices.keys())
    else:
        triangular = False
        if len(faces)>0:
            unique_triangles = point_matching.values(faces)
            triangle_matching = array_dict(face_vertices.keys(),face_vertices.keys())
        else:
            unique_triangles = np.array([])
            triangle_matching = array_dict()
    element_matching['face'] = triangle_matching
    if verbose: print len(unique_triangles)," Unique Faces"
    if timecheck: print 'face unicity test:',time()-check_time 

    if timecheck: check_time =time()
    if n_wisps.has_key('edge'):
        edge_vertices = {}
        edge_faces = {}
        for eid in xrange(n_wisps['edge']):
            edge_vertices[eid] = point_matching.values(np.array([element_properties['edge'][eid][dim] for dim in ['source','target']]))
            if element_properties['edge'][eid].has_key('face_index'):
                edge_faces[eid] = element_properties['edge'][eid]['face_index']
        #print element_properties['edge']
        if not 'face_index' in properties['edge']:
            face_edge_vertices = np.sort(np.concatenate([np.transpose([v,list(v[1:])+[v[0]]]) for v in unique_triangles]))
            face_edge_faces = np.concatenate([fid*np.ones_like(unique_triangles[fid]) for fid in xrange(len(unique_triangles))])
            face_edge_matching = vq(face_edge_vertices,np.sort(edge_vertices.values()))
            for eid in xrange(n_wisps['edge']):
                edge_faces[eid] = []
            for eid, fid in zip(face_edge_matching[0], face_edge_faces):
                edge_faces[eid] += [fid]
    else:
        edge_vertices = dict(zip(range(3*len(unique_triangles)),np.sort(np.concatenate([np.transpose([v,list(v[1:])+[v[0]]]) for v in unique_triangles]))))
        edge_faces = dict(zip(range(3*len(unique_triangles)),np.concatenate([fid*np.ones_like(unique_triangles[fid]) for fid in xrange(len(unique_triangles))])))
    if verbose: print len(edge_vertices)," Edges" 

    if len(edge_vertices)>0:
        unique_edges = array_unique(np.sort(edge_vertices.values()))
        if len(unique_edges) == len(edge_vertices.values()):
            unique_edges = edge_vertices.values()
            edge_matching = array_dict(edge_vertices.keys(),edge_vertices.keys())
        else:
            edge_matching = array_dict(vq(np.sort(edge_vertices.values()),unique_edges)[0],edge_vertices.keys())
    else:
        unique_edges = np.array([])
        edge_matching = array_dict()
    element_matching['edge'] = edge_matching
    if verbose: print len(unique_edges)," Unique Edges"
    if timecheck: print 'edge unicity test:',time()-check_time 
    
    if timecheck: check_time =time()
    face_cells = {}
    for fid in xrange(len(unique_triangles)):
        face_cells[fid] = set()

    cell_matching = {}
    if n_wisps.has_key('volume'):
        if 'label' in properties['volume']:
            for cid in xrange(n_wisps['volume']):
                cell_matching[cid] = element_properties['volume'][cid]['label']
        else:
            cell_matching = dict(zip(range(n_wisps['volume']),range(n_wisps['volume'])))

        if 'face_index' in properties['volume']:
            for c in xrange(n_wisps['volume']):
                for f in element_properties['volume'][c]['face_index']:
                    face_cells[triangle_matching[f]] = face_cells[triangle_matching[f]].union({cell_matching[c]})
        else:
            for f in xrange(len(unique_triangles)):
                face_cells[triangle_matching[f]] = {0}
    else:
        cell_matching[0] = 0
        for f in xrange(len(unique_triangles)):
            face_cells[triangle_matching[f]] = {0}
    element_matching['volume'] = cell_matching
    if verbose: print len(cell_matching)," Cells"
    if timecheck: print 'cell unicity test:',time()-check_time 


    if timecheck: check_time =time()
    topomesh = PropertyTopomesh(3)

    for pid in xrange(len(unique_points)):
        topomesh.add_wisp(0,pid)
    
    for fid in xrange(len(unique_triangles)):
        topomesh.add_wisp(2,fid)
        for cid in face_cells[fid]:           
            if not topomesh.has_wisp(3, cid):
                topomesh.add_wisp(3,cid)
            topomesh.link(3,cid,fid)

    for eid,e in enumerate(unique_edges):
        topomesh.add_wisp(1,eid)
        for pid in e:
            topomesh.link(1,eid,pid)

    for e in edge_vertices.keys():
        eid = edge_matching[e]
        if isinstance(edge_faces[e],list):
            for f in edge_faces[e]:
                fid = triangle_matching[f]
                topomesh.link(2,fid,eid)
        else:
            fid = edge_faces[e]
            topomesh.link(2,fid,eid)
    if timecheck: print 'topomesh topo creation:',time()-check_time 

    topomesh.update_wisp_property('barycenter',0,array_dict(unique_points,np.arange(len(unique_points))))

    native_properties = {}
    native_properties['vertex'] = ['x','y','z']
    native_properties['edge'] = ['source','target','face_index']
    native_properties['face'] = ['vertex_index','vertex_indices','red','green','blue']
    native_properties['volume'] = ['face_index','label']

    for property_degree, element_name in zip([0,1,2,3],['vertex','edge','face','volume']):
        if n_wisps.has_key(element_name):
            for property_name in properties[element_name]:
                if property_name not in native_properties[element_name]:
                    property_dict = {}
                    for w in element_properties[element_name].keys():
                        property_dict[element_matching[element_name][w]] = element_properties[element_name][w][property_name]
                    topomesh.update_wisp_property(property_name,property_degree,array_dict(property_dict))
    if timecheck: print 'topomesh creation:',time()-check_time 

    if verbose:
        end_time = time()
        print "<-- Reading .ply        [",end_time-start_time,"s]"

    return topomesh








