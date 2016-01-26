from gmap import GMap

def topopolygon(nbpoints = 3, gmap = None):
    if gmap is None:   gmap = GMap(2)

    darts = [gmap.add_dart() for i in xrange(nbpoints*2)]
    for i in xrange(nbpoints):
        gmap.link_darts(0, darts[2*i], darts[2*i+1])
    for i in xrange(nbpoints):
        gmap.link_darts(1, darts[2*i+1], darts[(2*i+2) % (2*nbpoints)])
    return gmap, darts

def topopolyline(nbpoints = 3, gmap = None):
    if gmap is None:   gmap = GMap(1)

    darts = [gmap.add_dart() for i in xrange(nbpoints*2)]
    for i in xrange(nbpoints):
        gmap.link_darts(0, darts[2*i], darts[2*i+1])
    for i in xrange(nbpoints-1):
        gmap.link_darts(1, darts[2*i+1], darts[2*i+2])
    return gmap, darts


def toposquare(gmap = None):
    return topopolygon(4,gmap)

def toposquares(nb = 5, gmap = None):
    darts = []
    for i in xrange(nb):
        gmap, cdarts = toposquare(gmap)
        darts.append(cdarts)

    return gmap, darts

def square(p1, p2, p3, p4, gmap = None):
    gmap, darts = toposquare(gmap)
    for i,p in enumerate([p1,p2,p3,p4]):
        gmap.set_position(darts[2*i], p)
    return gmap, darts

def polygon(pointlist, gmap = None):
    assert len(pointlist) > 2
    gmap, darts = topopolygon(len(pointlist), gmap)
    for i,p in enumerate(pointlist):
        gmap.set_position(darts[2*i], p)
    return gmap, darts

def polyline(pointlist, gmap = None):
    assert len(pointlist) > 1
    gmap, darts = topopolyline(len(pointlist), gmap)
    for i,p in enumerate(pointlist):
        gmap.set_position(darts[2*i], p)
    return gmap, darts


def tetrahedra(p1, p2, p3, p4, gmap):
    gmap, tr1 = topopolygon(3,gmap)
    gmap, tr2 = topopolygon(3,gmap)
    gmap, tr3 = topopolygon(3,gmap)
    gmap, tr4 = topopolygon(3,gmap)

    gmap.sew_dart(tr1[0],tr2[0])
    gmap.sew_dart(tr1[2],tr3[0])
    gmap.sew_dart(tr1[4],tr4[0])

    gmap.sew_dart(tr2[2],tr3[5])
    gmap.sew_dart(tr3[2],tr4[5])
    gmap.sew_dart(tr4[2],tr2[5])

    for i,p in enumerate([p1,p2,p3]):
        gmap.set_position(tr1[2*i], p)
    
    gmap.set_position(tr2[3], p4)

    return gmap, sum([tr1,tr2,tr3,tr4],[])

def cube(xsize = 5, ysize  = 5 , zsize = 5, center = [0,0,0], gmap = None):
    from numpy import array
    center = array(center)
    gmap, msquares = toposquares(6, gmap)

    # sew top square to lateral squares
    gmap.sew_dart( msquares[0][0], msquares[1][1] )
    gmap.sew_dart( msquares[0][2], msquares[4][1] )
    gmap.sew_dart( msquares[0][4], msquares[3][1] )
    gmap.sew_dart( msquares[0][6], msquares[2][1] )

    # sew bottom square to lateral squares
    gmap.sew_dart( msquares[5][0], msquares[1][5] )
    gmap.sew_dart( msquares[5][2], msquares[2][5] )
    gmap.sew_dart( msquares[5][4], msquares[3][5] )
    gmap.sew_dart( msquares[5][6], msquares[4][5] )

    # sew lateral squares between each other
    gmap.sew_dart( msquares[1][2], msquares[2][7] )
    gmap.sew_dart( msquares[2][2], msquares[3][7] )
    gmap.sew_dart( msquares[3][2], msquares[4][7] )
    gmap.sew_dart( msquares[4][2], msquares[1][7] )

    for darti, position in zip([0,2,4,6],[ [xsize, ysize, zsize], [xsize, -ysize, zsize] , [-xsize, -ysize, zsize], [-xsize, ysize, zsize]]):
        dart = msquares[0][darti]
        gmap.set_position(dart, center+position)
    
    for darti, position in zip([0,2,4,6],[ [xsize, -ysize, -zsize], [xsize, ysize, -zsize] , [-xsize, +ysize, -zsize], [-xsize, -ysize, -zsize]]):
        dart = msquares[5][darti]
        gmap.set_position(dart, center+position)

    return gmap, sum(msquares,[])



def holeshape(xsize = 5, ysize = 5, zsize = 5, internalratio = 0.5, gmap = None):
    assert 0 < internalratio < 1

    gmap, msquares = toposquares(16, gmap)

    # sew upper squares between each other
    gmap.sew_dart( msquares[0][2], msquares[1][1] )
    gmap.sew_dart( msquares[1][4], msquares[2][3] )
    gmap.sew_dart( msquares[2][6], msquares[3][5] )
    gmap.sew_dart( msquares[3][0], msquares[0][7] )

    # sew upper squares with external lateral
    gmap.sew_dart( msquares[0][0], msquares[8][1] )
    gmap.sew_dart( msquares[1][2], msquares[9][1] )
    gmap.sew_dart( msquares[2][4], msquares[10][1] )
    gmap.sew_dart( msquares[3][6], msquares[11][1] )

    # # sew upper squares with internal lateral
    gmap.sew_dart( msquares[0][5], msquares[12][0] )
    gmap.sew_dart( msquares[1][7], msquares[13][0] )
    gmap.sew_dart( msquares[2][1], msquares[14][0] )
    gmap.sew_dart( msquares[3][3], msquares[15][0] )

    # sew lower squares between each other
    gmap.sew_dart( msquares[4][6], msquares[5][1] )
    gmap.sew_dart( msquares[5][4], msquares[6][7] )
    gmap.sew_dart( msquares[6][2], msquares[7][5] )
    gmap.sew_dart( msquares[7][0], msquares[4][3] )

    # sew lower squares with external lateral
    gmap.sew_dart( msquares[4][0], msquares[8][5] )
    gmap.sew_dart( msquares[5][6], msquares[9][5] )
    gmap.sew_dart( msquares[6][4], msquares[10][5] )
    gmap.sew_dart( msquares[7][2], msquares[11][5] )

    # sew lower squares with internal lateral
    gmap.sew_dart( msquares[4][5], msquares[12][4] )
    gmap.sew_dart( msquares[5][3], msquares[13][4] )
    gmap.sew_dart( msquares[6][1], msquares[14][4] )
    gmap.sew_dart( msquares[7][7], msquares[15][4] )

    # sew external lateral squares between each other
    gmap.sew_dart( msquares[8][7], msquares[9][2] )
    gmap.sew_dart( msquares[9][7], msquares[10][2] )
    gmap.sew_dart( msquares[10][7], msquares[11][2] )
    gmap.sew_dart( msquares[11][7], msquares[8][2] )

    # sew internal lateral squares between each other
    gmap.sew_dart( msquares[12][2], msquares[13][7] )
    gmap.sew_dart( msquares[13][2], msquares[14][7] )
    gmap.sew_dart( msquares[14][2], msquares[15][7] )
    gmap.sew_dart( msquares[15][2], msquares[12][7] )

    pos = { 
            (0,0) : [xsize,  ysize,  zsize] ,
            (1,2) : [xsize,  -ysize, zsize] ,
            (2,4) : [-xsize, -ysize, zsize] ,
            (3,6) : [-xsize, ysize,  zsize] ,

            (0,5) : [xsize*internalratio,  ysize*internalratio,  zsize] ,
            (1,7) : [xsize*internalratio,  -ysize*internalratio, zsize] ,
            (2,1) : [-xsize*internalratio, -ysize*internalratio, zsize] ,
            (3,3) : [-xsize*internalratio, ysize*internalratio,  zsize] ,

            (4,1) : [xsize,  ysize,  -zsize] ,
            (5,7) : [xsize,  -ysize, -zsize] ,
            (6,5) : [-xsize, -ysize, -zsize] ,
            (7,3) : [-xsize, ysize,  -zsize] ,

            (4,4) : [xsize*internalratio,  ysize*internalratio,  -zsize] ,
            (5,2) : [xsize*internalratio,  -ysize*internalratio, -zsize] ,
            (6,0) : [-xsize*internalratio, -ysize*internalratio, -zsize] ,
            (7,6) : [-xsize*internalratio, ysize*internalratio,  -zsize] ,
          }

    for darti, position in pos.items():
        sqid, dartid = darti
        dart = msquares[sqid][dartid]
        gmap.set_position(dart, position)

    return gmap, sum(msquares,[])

def mobiusband(gmap = None):
    from math import pi, cos, sin
    from numpy import array
    nbsq = 16
    if gmap is None: gmap = GMap(2)
    gmap, msquares = toposquares(nbsq, gmap)
    for i in xrange(nbsq-1):
        gmap.sew_dart(msquares[i][5],msquares[i+1][0])    
    gmap.sew_dart(msquares[nbsq-1][4],msquares[0][0])    

    dalpha = 2 * pi / nbsq
    radius1, radius2 = 10, 5 
    length = 1
    alpha = 0
    ralpha = pi/8
    for sqid in xrange(nbsq):
        dart1, dart2 = msquares[sqid][0], msquares[sqid][1]
        centralpos = array([0,radius1 * cos(alpha), radius2 * sin(alpha)])
        #u,v = [1,0,0], [0,cos(alpha),sin(alpha)]
        #deltapos = [length*cos(alpha), length*sin(alpha)]
        position1 = centralpos+[length*cos(ralpha), length*sin(ralpha)*cos(alpha), length*sin(ralpha)*sin(alpha)]
        gmap.set_position(dart1, position1)

        position2 = centralpos+[length*cos(pi+ralpha), length*sin(pi+ralpha)*cos(alpha), length*sin(pi+ralpha)*sin(alpha)]
        gmap.set_position(dart2, position2)
        alpha += dalpha
        ralpha += dalpha/2

    return gmap, sum(msquares,[])




def crossshape():
    gmap = GMap(3)
    gmap, cube1 = cube(gmap=gmap)
    
    gmap, cube2 = cube(gmap=gmap, center = [10,0,0])
    gmap.sew_dart(cube1[8],cube2[8*3+1])
    
    gmap, cube3 = cube(gmap=gmap, center = [0,10,0])
    gmap.sew_dart(cube1[8*2],cube3[8*4+1])
    
    gmap, cube4 = cube(gmap=gmap, center = [0,0,10])
    gmap.sew_dart(cube1[0],cube4[8*5+1])
    
    gmap, cube5 = cube(gmap=gmap, center = [-10,0,0])
    gmap.sew_dart(cube1[8*3],cube5[8+1])
    
    gmap, cube6 = cube(gmap=gmap, center = [0,-10,0])
    gmap.sew_dart(cube1[8*4],cube6[8*2+1])
    
    gmap, cube7 = cube(gmap=gmap, center = [0,0,-10])
    gmap.sew_dart(cube1[8*5],cube7[1])

    return gmap, sum([cube1,cube2,cube3,cube4,cube5,cube6,cube7],[])



def pgl2gmap(geometry, gmap = None):
    from openalea.plantgl.all import *
    d = Discretizer()
    geometry.apply(d)
    res = d.result
    if not res is None:
        if gmap is None: gmap = GMap(2 if not res.solid else 3)
        indexlist = res.indexList

        alldarts = []
        pointdart = dict()
        eij2dij = dict()
        for fi, ind in enumerate(indexlist):
            prevdj = None
            firstdi = None
            ind = tuple(ind)
            for i,j in zip(ind,ind[1:]+ind[:1]):
                di, dj = gmap.add_dart(), gmap.add_dart()
                alldarts += [di,dj]
                if firstdi is None: firstdi = di
                gmap.link_darts(0,di,dj)
                pointdart.setdefault(i,di)
                pointdart.setdefault(j,dj)
                if not prevdj is None: gmap.link_darts(1,prevdj,di)
                if eij2dij.has_key((i,j)) or eij2dij.has_key((j,i)):
                    try:
                        odi, odj = eij2dij[(i,j)]
                        del eij2dij[(i,j)]
                    except:
                        odj, odi = eij2dij[(j,i)]
                        del eij2dij[(j,i)]
                    gmap.link_darts(2,odj,dj)
                    gmap.link_darts(2,odi,di)
                else: 
                    eij2dij[(i,j)] = (di,dj)
                prevdj = dj
            gmap.link_darts(1,prevdj,firstdi)

        for pi, di in pointdart.items():
            gmap.set_position(di, res.pointList[pi])
        return gmap, alldarts


