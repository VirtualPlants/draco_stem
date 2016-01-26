from openalea.container.utils.id_generator import IdGenerator


class WingedEdge:
    def __init__(self, id, vertexhead, vertextail, faceleft, edgetailleft, edgeheadleft, faceright, edgetailright, edgeheadright):
        self.id = id
        self.vertexhead = vertexhead
        self.vertextail = vertextail
        self.faceleft   = faceleft
        self.edgeheadleft  = edgeheadleft
        self.edgetailleft  = edgetailleft
        self.faceright     = faceright
        self.edgeheadright = edgeheadright
        self.edgetailright = edgetailright

    def __repr__(self):
        return 'WingedEdge('+', '.join(map(str,[self.id, self.vertexhead, self.vertextail, self.faceleft, 
                                               self.edgetailleft.id if self.edgetailleft else None, 
                                               self.edgeheadleft.id if self.edgeheadleft else None, 
                                               self.faceright, 
                                               self.edgetailright.id if self.edgetailright else None, 
                                               self.edgeheadright.id if self.edgeheadright else None]))+')'

    
    def opposite_vertex(self, vid):
        if vid == self.vertexhead: return self.vertextail
        else:
            assert vid == self.vertextail
            return self.vertexhead


class WingedEdgeMesh:
    def __init__(self):
        self._vertices  = {}
        self._edges     = {}
        self._faces     = {}

        self._idgen = IdGenerator()

        self.verticesproperties = {}
        self.facesproperties = {}
        self.edgeproperties = {}

        self.verticesproperties['position'] = {}

    # FV all vertices of a face
    # EV both vertices of an edge
    # VF all faces sharing a vertex
    # EF all faces sharing an edge
    # FE all edges of a  face
    # VE all edges sharing a vertex

    def print_info(self):
        print 'vertices :'
        for vid, edge in self._vertices.items():
            print vid, ':', edge.id, self.get_position(vid)

        print 'edges :'
        for eid, edge in self._edges.items():
            print eid, ':', edge

        print 'faces :'
        for fid, edge in self._faces.items():
            print fid, ':', edge.id

    def vertices(self):
        return self._vertices.keys()

    def edges(self):
        return self._edges.keys()

    def faces(self):
        return self._faces.keys()

    def edge_face(self, eid): # EF
        edge = self._edges[eid]
        return (edge.faceleft, edge.faceright)

    def edge_vertices(self, eid): # EV
        edge = self._edges[eid]
        return (edge.vertexhead, edge.vertextail)

    def face_edges(self, fid): # FE
        def nexedge(currentedge, fid):
            isleft = (currentedge.faceleft == fid)
            assert isleft or (currentedge.faceright == fid)
            if isleft : return currentedge.edgeheadleft
            else : return currentedge.edgetailright

        firstedge = self._faces[fid]
        if firstedge is None: return
        yield firstedge

        currentedge = firstedge 

        while True:
            currentedge = nexedge(currentedge, fid)
            if (currentedge is None) or (currentedge == firstedge):   break
            yield currentedge

    def vertex_edge(self, vid): # VE
        def nexedge(currentedge, vid):
            ishead = (currentedge.vertexhead == vid)
            assert ishead or (currentedge.vertextail == vid)
            if ishead : return currentedge.edgeheadright
            else : return currentedge.edgetailleft

        firstedge = self._vertices.get(vid)
        if firstedge is None: return
        yield firstedge

        currentedge = firstedge 

        while True:
            currentedge = nexedge(currentedge, vid)
            if (currentedge is None) or (currentedge == firstedge):   break
            yield currentedge

    def face_vertex(self, fid): # FV
        def nexedge(currentedge, fid):
            isleft = (currentedge.faceleft == fid)
            assert isleft or (currentedge.faceright == fid)
            if isleft : return currentedge.edgeheadleft, currentedge.vertexhead
            else : return currentedge.edgetailright, currentedge.vertextail

        firstedge = self._faces[fid]
        if firstedge is None: return
        currentedge = firstedge 

        while True:
            currentedge, currentvertex = nexedge(currentedge, fid)
            if (currentedge is None): break
            if (currentedge == firstedge):  
                yield currentvertex
                break
            yield currentvertex


    def vertex_face(self, vid): # VF to be done !!!!
        def nexedge(currentedge, vid):
            ishead = (currentedge.vertexhead == vid)
            assert ishead or (currentedge.vertextail == vid)
            if ishead : return currentedge.edgeheadright, currentedge.faceleft
            else : return currentedge.edgetailleft, currentedge.faceright

        firstedge = self._vertices[vid]
        if firstedge is None: return
        yield firstedge

        currentedge = firstedge 

        while True:
            currentedge, currentface = nexedge(currentedge, vid)
            if (currentedge is None) or (currentedge == firstedge):   break
            yield currentface


    def face_neighbors(self, fid):
        for currentedge in self.faceedges(fid):
            isleft = (currentedge.faceleft == fid)
            if isleft : yield edge.faceright
            else: yield edge.faceleft

    def edge_neighbors(self, eid):
        edge = self._edges[eid]
        for vertex in (edge.vertexhead, edge.vertextail):
            for neid in self.vertex_edge(vertex):
                if neid != eid:
                    yield neid


    def vertex_neighbors(self, vid):
        for edge in self.vertex_edge(vid):
            if edge.vertexhead == vid: yield edge.vertextail
            else: yield edge.vertexhead

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
        for vid1, vid2 in zip(vids,vids[1:]+[vids[0]]):
            for edge in self.vertex_edge(vid1):
                if edge.opposite_vertex(vid1) == vid2:
                    edges.append(edge)
                    if edge.vertexhead == vid2:
                        assert edge.faceleft is None
                        edge.faceleft = fid
                    else:                        
                        assert edge.faceright is None
                        edge.faceright = fid
                    break
            else:
                eid = self._idgen.get_id()
                newedge = WingedEdge(eid, vid2, vid1, fid, None, None, None, None, None)
                self._edges[eid] = newedge
                edges.append(newedge)

        for i, edge in enumerate(edges):
            if edge.faceright == fid:
                edge.edgeheadright = edges[(i-1)%len(edges)]
                edge.edgetailright = edges[(i+1)%len(edges)]
            else:
                assert edge.faceleft == fid
                edge.edgetailleft = edges[(i-1)%len(edges)]
                edge.edgeheadleft = edges[(i+1)%len(edges)]

            if not self._vertices.has_key(edge.vertexhead):
                self._vertices[edge.vertexhead] = edge
            if not self._vertices.has_key(edge.vertextail):
                self._vertices[edge.vertextail] = edge

        self._faces[fid] = edges[0]
        return fid


    def topglscene(self):
        from openalea.plantgl.all import Scene, Shape, FaceSet
        normalizedpoints = self.verticesproperties['position'].values()
        normpid = dict([(vid, nid) for nid,vid in enumerate(self.verticesproperties['position'].keys())])
        indexlist = []
        for fid in self._faces.keys():
            index = list(self.face_vertex(fid))
            index = [normpid[pid] for pid in index]
            indexlist.append(index)
        return Scene([Shape(FaceSet(normalizedpoints, indexlist), id = fid )])
