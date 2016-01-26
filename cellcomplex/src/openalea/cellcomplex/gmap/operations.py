from gmap import GMap, pointvalence

def topo_add_edge(gmap, dart1, dart2):
    assert gmap.is_free(dart1,1) == gmap.is_free(dart2,1)
    d1,d2 = gmap.add_darts(2)
    gmap.link_darts(0,d1,d2)
    if not gmap.is_free(dart1,1):
        d11,d21 = gmap.alpha(1,dart1), gmap.alpha(1,dart2)
        d1b,d2b = gmap.add_darts(2)
        gmap.link_darts(0,d1b,d2b)
        gmap.link_darts(2,d1,d1b)
        gmap.link_darts(2,d2,d2b)
        gmap.overlink_darts(1,d2b,d21)
        gmap.overlink_darts(1,d1b,d11)
    gmap.overlink_darts(1,dart1,d1)
    gmap.overlink_darts(1,dart2,d2)

def topo_split_edge(gmap, dart):

    orbit1 = [d for d,a in gmap.orbit_iter(dart,range(2,gmap.degree+1))]
    orbit2 = [d for d,a in gmap.orbit_iter(gmap.alpha(0,dart),range(2,gmap.degree+1))]

    ndart1 = [gmap.add_dart() for d in orbit1]
    ndart2 = [gmap.add_dart() for d in orbit2]

    for nd,d in zip(ndart1, orbit1):
        gmap.overlink_darts(0,nd, d)

    for nd,d in zip(ndart1, orbit1):
        for i in xrange(2,gmap.degree+1):
            gmap.set_alpha(i,nd, gmap.alpha(0,i,d))

    for nd,d in zip(ndart2, orbit2):
        gmap.overlink_darts(0,nd, d)

    for nd,d in zip(ndart2, orbit2):
        for i in xrange(2,gmap.degree+1):
            gmap.set_alpha(i,nd, gmap.alpha(0,i,d))

    for nd1, nd2 in zip(ndart1, ndart2):
        gmap.link_darts(1, nd1, nd2)

    return ndart1+ndart2

def topo_quadrangulation(gmap):
    ngmap = gmap.copy()

    edgevertex = {}
    for d in gmap.iterate_over_each_i_cell(1):
        resd = topo_split_edge(ngmap, d)
        edgevertex[d] = set(resd)

    facevertex = {}
    for d in gmap.iterate_over_each_i_cell(2):
        facedartset = set(ngmap.iterate_over_each_dart_of_a_i_cell(d,2))
        for de in gmap.iterate_over_each_incident_cell(d,2,1):
            for nde in gmap.iterate_over_each_dart_of_a_i_cell(de,1):
                if nde in edgevertex: break
            npoint = edgevertex[nde]
            npdarts =  tuple(facedartset & npoint)
            assert len(npdarts) in [2,4]

            phi = {}
            for npdart in npdarts:
                npdartb =  ngmap.add_dart()
                npdartc =  ngmap.add_dart()
                ngmap.link_darts(0, npdartb, npdartc)
                phi[npdart] = npdartb
                preva1 = ngmap.alpha(1, npdart)
                ngmap.overlink_darts(1, npdart,npdartb)
                if phi.has_key(preva1):
                    ngmap.link_darts(2, npdartb, phi[preva1])
                    ngmap.link_darts(2, npdartc, ngmap.alpha(0,phi[preva1]))
                for deg in xrange(3, ngmap.degree+1):
                    if phi.has_key(ngmap.alpha(deg,npdart)):
                        ngmap.link_darts(deg, npdartb, phi[ngmap.alpha(deg,npdart)])
                        ngmap.link_darts(deg, npdartc, ngmap.alpha(0,phi[ngmap.alpha(deg,npdart)]))
                mapdartc = ngmap.alpha(0,1,0,1,0,1,npdartb)
                if ngmap.is_free(mapdartc, 1): 
                    ngmap.link_darts(1, npdartc, mapdartc)
            facevertex[d] = npdartc

    edgevertex = dict([(d,v.pop()) for d,v in edgevertex.items()])
    return ngmap, edgevertex, facevertex

def quadrangulation(gmap):

    newfacepoints = dict([(d,gmap.cell_center(d,2)) for d in gmap.iterate_over_each_i_cell(2)])
    newedgepoints = dict([(d, gmap.cell_center(d,1))  for d in gmap.iterate_over_each_i_cell(1)])

    ngmap, edgevertex, facevertex = topo_quadrangulation(gmap)


    for de, dp in edgevertex.items():
        ngmap.set_position(dp,newedgepoints[de])

    for df, dp in facevertex.items():
        ngmap.set_position(dp,newfacepoints[df])

    return ngmap

def catmullclark_subdivision(gmap):
    initialpoints = list()
    pos = lambda d : gmap.get_position(d)


    newfacepoints = dict([(d,gmap.cell_center(d,2)) for d in gmap.iterate_over_each_i_cell(2)])
    def getfacepoint(df): return gmap.property(newfacepoints,2,df)

    def edgepoint(ef):
        return (pos(ef)+ pos(gmap.alpha(0,ef)) + getfacepoint(ef) + getfacepoint(gmap.alpha(2,ef)))/4.

    newedgepoints = dict([(d, edgepoint(d))  for d in gmap.iterate_over_each_i_cell(1)])
    def getedgepoint(de): return gmap.property(newedgepoints,1,de)

    pointvalences = dict([(d,pointvalence(gmap,d)) for d in gmap.iterate_over_each_i_cell(0)])

    def newpointposition(d):
        n = pointvalences[d]
        n2 = float(n * n)
        npos = (n-3.)/float(n)*pos(d)
        npos += (2./n2)*(sum(getedgepoint(de) for de in gmap.iterate_over_each_incident_cell(d,0,1)))
        npos += (1./n2)*(sum(getfacepoint(df) for df in gmap.iterate_over_each_incident_cell(d,0,2)))  
        return npos


    newpointpos = dict([(d,newpointposition(d)) for d in pointvalences.keys()])

    ngmap, edgevertex, facevertex = topo_quadrangulation(gmap)

    for d, npos in newpointpos.items():
        ngmap.set_position(d,npos)

    for de, dp in edgevertex.items():
        ngmap.set_position(dp,newedgepoints[de])

    for df, dp in facevertex.items():
        ngmap.set_position(dp,newfacepoints[df])

    return ngmap


def doosabin_subdivision(gmap):
    from basicshapes import polygon
    ngmap = GMap(gmap.degree)
    edgecenter = dict([(d,gmap.cell_center(d,1)) for d in gmap.iterate_over_each_i_cell(1)])

    def getedgecenter(de): return gmap.property(edgecenter,1,de)

    facepolygon = dict()
    for df in gmap.iterate_over_each_i_cell(2):
        pointlist = []
        facecenter = gmap.cell_center(df,2)
        dp = df
        while True:
            pointlist.append((facecenter + gmap.get_position(dp) + getedgecenter(dp) + getedgecenter(gmap.alpha(1,dp)))/4.)
            dp = gmap.alpha(1,0,dp)
            if dp == df : break

        ngmap, pdarts = polygon(pointlist, ngmap)
        facepolygon[df] = pdarts[0]


    for de in gmap.iterate_over_each_i_cell(1):
        oppde = gmap.alpha(2,de)
        face1, face2 = gmap.get_embedding_dart(facepolygon,2,de) , gmap.get_embedding_dart(facepolygon, 2, oppde) 
        posi = gmap.orbit(face1,(0,1)).index(de)
        posj = gmap.orbit(face2,(0,1)).index(oppde)

        di = ngmap.orbit(facepolygon[face1],(0,1))[posi]
        dj = ngmap.orbit(facepolygon[face2],(0,1))[posj]

        di1, dio1 = ngmap.add_dart(),ngmap.add_dart()
        ngmap.link_darts(0, di1, dio1)
        ngmap.link_darts(2, di, di1)
        ngmap.link_darts(2, ngmap.alpha(0,di), dio1)

        dj1, djo1 = ngmap.add_dart(),ngmap.add_dart()
        ngmap.link_darts(0, dj1, djo1)
        ngmap.link_darts(2, dj, dj1)
        ngmap.link_darts(2, ngmap.alpha(0,dj), djo1)

        dki, dkj = ngmap.add_dart(),ngmap.add_dart()
        ngmap.link_darts(0, dki, dkj)
        ngmap.link_darts(1, di1, dki)
        ngmap.link_darts(1, dj1, dkj)

        dkoi, dkoj = ngmap.add_dart(),ngmap.add_dart()
        ngmap.link_darts(0, dkoi, dkoj)
        ngmap.link_darts(1, dio1, dkoi)
        ngmap.link_darts(1, djo1, dkoj)

    ngmap.iclosure(2)
        #ngmap.link_darts(2, di, dj )
        #ngmap.link_darts(2, ngmap.alpha(0,di), ngmap.alpha(0,dj) )


    return ngmap


def topo_triangulation(gmap, degree, dart):
    phi = dict([(deg,{}) for deg in xrange(self.degree+1)])

    nbgs = map(self.dart,self.iterate_over_each_dart_of_a_i_cell(dart, degree))

    for ndart in nbgs:
        for j in xrange(1, self.degree):
            phi[j][ndart] = self.add_sdart()
        phi[j][ndart] = ndart

    for ndart in nbgs:
        phi[0][ndart].set_alpha(0,phi[1][ndart])

        for j in xrange(1, degree+1):
            phi[0][ndart].set_alpha(j,phi[0][ndart.alpha(j-1)])

        for j in xrange(degree+1, gmap.degree):
            phi[0][ndart].set_alpha(j,phi[0][ndart.alpha(j)])


        for k in xrange(1,degree):
            for j in xrange(0, k-1):
                phi[k][ndart].set_alpha(j,phi[k][ndart.alpha(j)])

            phi[k][ndart].set_alpha(k-1,phi[k-1][ndart])
            phi[k][ndart].set_alpha(k,phi[k+1][ndart])

            for j in xrange(k+1, degree+1):
                phi[k][ndart].set_alpha(j,phi[k][ndart.alpha(j-1)])

            for j in xrange(degree+1, gmap.degree+1):
                phi[k][ndart].set_alpha(j,phi[k][ndart.alpha(j)])

    for ndart in nbgs:
        ndart.set_alpha(degree-1, phi[degree-1][ndart])
    return phi[0][dart]

