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
    wisps_to_save[1] = triangles_to_save
    wisps_to_save[2] = edges_to_save
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
                sub_topomesh.update_wisp_property(property_name,degree,array_dict(topomesh.wisp_property(property_name,degree).values(wisps_to_save[degree]),array_dict(wisps_to_wids[degree]).values(wisps_to_save[degree])))

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

    # tissue_file = topen(tissue_filename,'w')
    # tissue_file.write(tissue
    # tissue_file.write(positions,"position","position of points"))
    # tissue_file.write_config(cfg,"config")
    # tissue_file.close()

    tissue_db = TissueDB()
    tissue_db.set_tissue(tissue)
    tissue_db.set_property("position",positions)
    tissue_db.set_description("position","position of points")
    tissue_db.set_config("config",cfg.config())

    tissue_db.write(tissue_filename)

def save_ply_cellcomplex_topomesh(topomesh,ply_filename,color_faces=False,colormap=None,oriented=True):
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

def save_ply_property_topomesh(topomesh,ply_filename,properties_to_save=dict([(0,[]),(1,['length']),(2,['area','epidermis']),(3,[])]),color_faces=False):
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
        if topomesh.has_wisp_property(property_name,0,is_computed=True):
            property_type = property_types[str(topomesh.wisp_property(property_name,0).values().dtype)]
            ply_file.write("property "+property_type+" "+property_name+"\n")
    ply_file.write("element line "+str(topomesh.nb_wisps(1))+"\n")
    ply_file.write("property int vertex_0\n")
    ply_file.write("property int vertex_1\n")
    for property_name in properties_to_save[1]:
        if topomesh.has_wisp_property(property_name,1,is_computed=True):
            property_type = property_types[str(topomesh.wisp_property(property_name,1).values().dtype)]
            ply_file.write("property "+property_type+" "+property_name+"\n")
    ply_file.write("element face "+str(topomesh.nb_wisps(2))+"\n")
    ply_file.write("property list uchar int vertex_indices\n")
    ply_file.write("property list uchar int edge_indices\n")
    if color_faces:
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        triangle_data = array_dict(np.array([list(topomesh.regions(2,t))[0]%256 for t in topomesh.wisps(2)]),list(topomesh.wisps(2)))
    for property_name in properties_to_save[2]:
        if topomesh.has_wisp_property(property_name,2,is_computed=True):
            property_type = property_types[str(topomesh.wisp_property(property_name,2).values().dtype)]
            ply_file.write("property "+property_type+" "+property_name+"\n")
    ply_file.write("element cell "+str(topomesh.nb_wisps(3))+"\n")
    ply_file.write("property list uchar int triangle_indices\n")

    ply_file.write("end_header\n")    
    vertex_index = {}
    for v,pid in enumerate(topomesh.wisps(0)):
        ply_file.write(str(topomesh.wisp_property('barycenter',0)[pid][0])+" ")
        ply_file.write(str(topomesh.wisp_property('barycenter',0)[pid][1])+" ")
        ply_file.write(str(topomesh.wisp_property('barycenter',0)[pid][2])+" ")
        ply_file.write("\n")
        vertex_index[pid] = v

    edge_index = {}
    for e,eid in enumerate(topomesh.wisps(1)):
        ply_file.write(str(vertex_index[list(topomesh.borders(1,eid))[0]])+" ")
        ply_file.write(str(vertex_index[list(topomesh.borders(1,eid))[1]])+" ")
        for property_name in properties_to_save[1]:
            if topomesh.has_wisp_property(property_name,1,is_computed=True):
                ply_file.write(str(topomesh.wisp_property(property_name,1)[eid])+" ")
        ply_file.write("\n")
        edge_index[eid] = e

    face_index = {}
    for t,fid in enumerate(topomesh.wisps(2)):
        ply_file.write(str(len(list(topomesh.borders(2,fid,2))))+" ")
        for pid in topomesh.borders(2,fid,2):
            ply_file.write(str(vertex_index[pid])+" ")
        ply_file.write(str(len(list(topomesh.borders(2,fid))))+" ")
        for eid in topomesh.borders(2,fid):
            ply_file.write(str(edge_index[eid])+" ")
        if color_faces:
            ply_file.write(" "+str(triangle_data[fid]))
            ply_file.write(" "+str(triangle_data[fid]))
            ply_file.write(" "+str(triangle_data[fid]))
        for property_name in properties_to_save[2]:
            if topomesh.has_wisp_property(property_name,2,is_computed=True):
                property_type = property_types[str(topomesh.wisp_property(property_name,2).values().dtype)]
                if property_type == 'int':
                    ply_file.write(str(int(topomesh.wisp_property(property_name,2)[fid]))+" ")
                else:
                    ply_file.write(str(topomesh.wisp_property(property_name,2)[fid])+" ")
        ply_file.write("\n")
        face_index[fid] = t

    for c, cid in enumerate(topomesh.wisps(3)):
        ply_file.write(str(len(list(topomesh.borders(3,cid))))+" ")
        for fid in topomesh.borders(3,cid):
            ply_file.write(str(face_index[fid])+" ")
        ply_file.write("\n")

    ply_file.flush()
    ply_file.close()

    end_time = time()
    print "<-- Saving .ply        [",end_time-start_time,"s]"



