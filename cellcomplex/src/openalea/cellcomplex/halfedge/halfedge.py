from openalea.container.utils.id_generator import IdGenerator


class HalfEdge:
    def __init__(self, id, vertex, face, opposite, previous = None, next = None):
        self.id = id
        self.vertex = vertex
        self.face   = face
        self.previous = previous
        self.next     = next
        self.opposite = opposite

    def __repr__(self):
        return 'HalfEdge('+', '.join([k+'='+v for k,v in zip(['id', 'vertex','vertextail' ,'face', 'opposite', 'previous', 'next'],
                                       map(str,[self.id, self.vertex, self.vertextail(), self.face, 
                                               self.opposite.id if self.opposite else None, 
                                               self.previous.id if self.previous else None, 
                                               self.next.id if self.next else None]))])+')'


    def vertextail(self):
        if self.opposite:   return self.opposite.vertex
        elif self.previous: return self.previous.vertex
        return None

class HalfEdgeMesh:
    def __init__(self):
        self._edges     = {}
        self._vertices  = {}
        self._faces     = {}
        self._halfedges = {}

        self._idgen = IdGenerator()

        self.verticesproperties = {}
        self.facesproperties = {}
        self.edgeproperties = {}

        self.verticesproperties['position'] = {}

    def vertices(self):
        return self._vertices.keys()

    def edges(self):
        return self._edges.keys()

    def faces(self):
        return self._faces.keys()

    def dart(self, did):
        return self._edges[did]

    def edge_face(self, eid): # EF
        edge = self._edges[eid]
        return (edge.face, edge.opposite.face if edge.opposite else None)

    def edge_vertices(self, eid): # EV
        edge = self._edges[eid]
        return (edge.vertex, edge.vertextail())

    def face_edges(self, fid): # FE

        firstedge = self._faces[fid]
        if firstedge is None: return
        yield firstedge

        currentedge = firstedge 

        while True:
            currentedge = currentedge.next
            if (currentedge is None) or (currentedge == firstedge):   break
            yield currentedge

    def vertex_edges(self, vid): # VE

        return [h.id for h in self._vertices.get(vid,[])]

        # firstedge = self._vertices.get(vid)
        # if firstedge is None: return
        # yield firstedge.id

        # currentedge = firstedge 

        # while True:

        #     currentedge = currentedge.next.opposite
        #     if (currentedge is None) or (currentedge == firstedge):  
        #         break
        #     yield currentedge.id


    def face_vertices(self, fid): # FV

        firstedge = self._faces[fid]
        if firstedge is None: return
        currentedge = firstedge 
        yield currentedge.vertex

        while True:
            currentedge = currentedge.next
            if (currentedge is None or currentedge == firstedge): break
            yield currentedge.vertex    


    def vertex_faces(self, vid): # VF to be done !!!!

        return [h.face for h in self._vertices.get(vid,[])]

        # firstedge = self._vertices[vid]
        # if firstedge is None: return
        # yield firstedge.face

        # currentedge = firstedge 

        # while True:
        #     currentedge = currentedge.next.opposite
        #     if (currentedge is None) or (currentedge == firstedge):   break
        #     yield currentedge.face

    def face_neighbors(self, fid):
        for currentedge in self.face_edges(fid):
            yield edge.opposite.face

    def edge_neighbors(self, eid):
        edge = self._edges[eid]
        for vertex in (edge.vertex, edge.opposite.vertex):
            for neid in self.vertex_edges(vertex):
                if neid != eid:
                    yield neid

    def vertex_neighbors(self, vid):
        for edge in self.vertex_edges(vid):
            yield edge.opposite.vertex

    def add_vertex(self, position = None, vid = None):
        vid = self._idgen.get_id(vid)
        if position: 
            self.set_position(vid, position)
        return vid

    def set_position(self, vid, position):
        self.verticesproperties['position'][vid] = position

    def get_position(self, vid):
        return self.verticesproperties['position'][vid]

    def add_polygon(self, vids, fid = None):
        fid = self._idgen.get_id(fid)
        edges = []
        print 'add_polygon', vids
        for vid1, vid2 in zip(vids,vids[1:]+[vids[0]]):
            print 'add edge', (vid1,vid2),
            for leid in self.vertex_edges(vid1):
                if self._halfedges[leid].vertextail() == vid2:
                    oppedge  = self._edges[leid]
                    break
            else:
                oppedge  = None

            eid = self._idgen.get_id() if oppedge is None else -oppedge.id
            newedge = HalfEdge(eid, vid2,  fid, opposite = oppedge)
            if oppedge: 
                oppedge.opposite = newedge
            else:
                self._edges[eid] = newedge
            self._halfedges[eid] = newedge

            print '-->', eid
            edges.append(newedge)

            if not self._vertices.has_key(vid2):
                self._vertices[vid2] = [newedge]
            else: 
                self._vertices[vid2].append(newedge)


        for i, edge in enumerate(edges):
            edge.previous = edges[(i-1)%len(edges)]
            edge.next = edges[(i+1)%len(edges)]



        self._faces[fid] = edges[0]
        return fid    

    def topglscene(self):
        from openalea.plantgl.all import Scene, Shape, FaceSet
        normalizedpoints = self.verticesproperties['position'].values()
        normpid = dict([(vid, nid) for nid,vid in enumerate(self.verticesproperties['position'].keys())])
        indexlist = []
        for fid in self.faces():
            index = list(self.face_vertices(fid))
            index = [normpid[pid] for pid in index]
            indexlist.append(index)
        return Scene([Shape(FaceSet(normalizedpoints, indexlist), id = fid )])

    def print_info(self):
        print 'vertices :'
        for vid, edges in self._vertices.items():
            print vid, ':', [e.id for e in edges], self.get_position(vid)

        print 'edges :'
        for eid, edge in self._edges.items():
            print eid, ':', edge.id, '-->', [edge.id, edge.opposite.id if edge.opposite else None]

        print 'faces :'
        for fid, edge in self._faces.items():
            print fid, ':', edge.id, '-->', [h.id for h in self.face_edges(fid)]

        print 'half edges :'
        for hid, hedge in self._halfedges.items():
            print hid, ':', hedge