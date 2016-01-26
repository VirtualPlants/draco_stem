class GMap:
  def __init__(self):
    """ constructor """
    self.maxdartid = 0
    self.alphas = { 0 : {}, 1 : {}, 2 : {} }
    self.positions = {}
 
  def darts(self):
    """ Return a list of id representing the darts of the structure """
    return self.alphas[0].keys()
 
  def alpha(self, degree, dart):
    return self.alphas[degree][dart]
 
  def is_free(self, degree, dart):
    """ Test if dart is free for alpha_degree (if it is a fixed point) """
    return self.alpha(degree,dart) == dart
 
  def add_dart(self):
    """ Create a new dart and return its id. 
        Set its alpha_i to itself (fixed points) """
    dart = self.maxdartid
    self.maxdartid += 1
    for alpha in self.alphas.values():
        alpha[dart] = dart # fixed point
    return dart
 
  def is_valid(self):
    """ Test the validity of the structure.
        Check if there is pending dart for alpha_0 and alpha_1 (fixed point) """
    for dart, alpha_0_of_dart in self.alphas[0].items():
        if dart == alpha_0_of_dart : return False # no fixed point
        if dart != self.alpha(0,alpha_0_of_dart) : return False # alpha_0 is an involution
 
    for dart, alpha_1_of_dart in self.alphas[1].items():
        if dart == alpha_1_of_dart : return False # no fixed point
        if dart != self.alpha(1,alpha_1_of_dart) : return False # alpha_1 is an involution
 
    for dart, alpha_2_of_dart in self.alphas[2].items(): # alpha_0 alpha_2 is an involution
        if self.alpha(0,self.alpha(2,self.alpha(0,alpha_2_of_dart))) != dart: return False
 
    return True
 
  def link_dart(self,degree, dart1, dart2): # should be link_darts
     """ Link the two darts with a relation alpha_degree """
     assert self.is_free(degree, dart1)
     assert self.is_free(degree, dart2)
     alpha_i = self.alphas[degree]
     alpha_i[dart1] = dart2
     alpha_i[dart2] = dart1
 
  def print_alphas(self):
     """ print for each dart, the value of the different alpha applications. """ 
     for d in self.darts():
        print d,self.alpha(0,d),self.alpha(1,d),self.alpha(2,d), self.positions.get(d)
 
  def orbit(self,dart,list_of_alpha_value):
    """ Return the orbit of dart using a list of alpha relation.
         Example of use: gmap.orbit(0,[0,1]).
         In python, you can use the set structure to process only once all darts of the orbit.  """
    orbit = []
    toprocess = [dart]
    marked = set()
 
    while len(toprocess) > 0 :
        d = toprocess.pop(0)
        if not d in marked:
            orbit.append(d)
            marked.add(d)
            for alpha_v in list_of_alpha_value:
                alpha_i_of_d = self.alpha(alpha_v,d)
                toprocess.append(alpha_i_of_d)
    return orbit
 
  def elements(self, degree):
        """ 
        Return one dart per element of degree. For this, consider all darts 
        as initial set S. Take the first dart d, remove from the set 
        all darts of the orbit starting from d and corresponding to element 
        of degree degree. Take then next element from set S
        and do the same until S is empty. Return all darts d that where use 
        """
        elements = []
        darts = set(self.darts())
 
        list_of_alpha_value = range(3)
        list_of_alpha_value.remove(degree)
 
        while len(darts) > 0:
            dart = darts.pop()
            elementi = self.orbit(dart, list_of_alpha_value)
            darts -= set(elementi)
            elements.append(dart)
 
        return elements

  def adjacent_cells(self, dart, degree):
        """ Return all the elements of degree degree
        that are adjacent to the element dart with respect
        to the alpha relation of degree degree.
        (Typically all points sharing an edge with a point)
        For this iterate over all the dart of the orbit of (dart, degree).
        For each dart d of this orbit, get its neighbor n (alpha degree)
        and remove its orbit (n, degree) from the set of darts
        to consider.
        See function incident_cells for inspiration.
        """

        neighbors = set()

        alphas = range(3)
        alphas.remove(degree) 

        # orbit = set(self.orbit(dart, list_of_alpha_value))

        marked = set()

        # while len(orbit) > 0:
        #   d = orbit.pop()
        #   neighbor_dart = self.alpha(degree, d)
        #   neighbors |= {neighbor_dart}
        #   orbit -= set(self.orbit(neighbor_dart,))
        
        for d in self.orbit(dart, alphas):
            neighbor_dart = self.alpha(degree, d)
            if neighbor_dart not in marked:
              neighbors |= {neighbor_dart}
              marked |= set(self.orbit(neighbor_dart, alphas))

        return list(neighbors)
 


  def incident_cells(self, dart, degree, incidentdegree):
        """ Return all the element of degree incidentdegree
        that are incident to the element dart of degree degree.
        (Typically all edges around a point)
        For this iterate over all the dart of the orbit of (dart, degree).
        For each dart d of this orbit, get all the darts coresponding
        to the orbit of the element (d, incidentdegree) and remove them
        from the original set.
        See function elements for inspiration.
        """
        results = []

        alphas = range(3)
        alphas.remove(degree) 

        incidentalphas = range(3)
        incidentalphas.remove(incidentdegree) 

        marked = set()

        for d in self.orbit(dart, alphas):
            if not d in marked:
                results.append(d)
                marked |= set(self.orbit(d, incidentalphas))

        return results

  def pointvalence(self, pointdart):
      """ Return the valence of a vertex """
      return len(self.incident_cells(pointdart,0,1))
  


  def cell_center(self, dart, degree):
     """ Generic function to compute the center of any elements of any degree """
     import numpy as np
     alphas = range(3)
     alphas.remove(degree)
     points = []
     darts = set(self.orbit(dart,alphas))
     pointdarts = []
     while len(darts) > 0:
        d = darts.pop()
        pointdarts.append(d)
        if degree != 1 : darts.remove(self.alpha(1,d))
     points = [self.get_position(d) for d in pointdarts]
     return sum(points, np.array([0,0,0]))/len(points)

  def insert_edge(self, dart):
        """ Insert an edge at the point represented by dart.
            Return a dart corresponding to the dandling edge end.
        """
        dart1 = self.alpha(1, dart)
        newdarts = [self.add_dart() for i in xrange(4)]

        self.link_dart(0, newdarts[0], newdarts[1])
        self.link_dart(0, newdarts[3], newdarts[2])

        self.link_dart(2, newdarts[0], newdarts[3])
        self.link_dart(2, newdarts[1], newdarts[2])

        self.alphas[1][dart] = newdarts[0]
        self.alphas[1][newdarts[0]] = dart

        self.alphas[1][dart1] = newdarts[3]
        self.alphas[1][newdarts[3]] = dart1

        return newdarts[1]

  def splitface(self, dart1, dart2):
       """ split face by inserting an edge between dart1 and dart2 """
       dedge = self.insert_edge(dart1)

       dart2a1 = self.alpha(1,dart2)
       dedgea2 = self.alpha(2, dedge)

       self.alphas[1][dart2] = dedge
       self.alphas[1][dedge] = dart2

       
       self.alphas[1][dart2a1] = dedgea2
       self.alphas[1][dedgea2] = dart2a1

  def splitedge(self, dart):
      """ Operator to split an edge. 
          Return a dart corresponding to the new points
      """
      orbit1 = self.orbit(dart,[2])
      orbit2 = self.orbit(self.alpha(0,dart),[2])

      newdart1 = [self.add_dart() for i in orbit1]
      newdart2 = [self.add_dart() for i in orbit2]

      for d, nd in zip(orbit1+orbit2, newdart1+newdart2):
        self.alphas[0][d] = nd
        self.alphas[0][nd] = d

      for nd1, nd2 in zip(newdart1, newdart2):
        self.link_dart(1, nd1, nd2)

      for nd in newdart1+newdart2:
        if self.is_free(2, nd) and not self.is_free(2, self.alpha(0, nd)):
            self.link_dart(2,nd, self.alpha(0,self.alpha(2,self.alpha(0,nd))))

      return newdart1[0]

  def incident_cells(self, dart, degree, incidentdegree):
        """ Return all the element of degree incidentdegree
        that are incident to the element dart of degree degree.
        (Typically all edges around a point)
        For this iterate over all the dart of the orbit of (dart, degree).
        For each dart d of this orbit, get all the darts coresponding
        to the orbit of the element (d, incidentdegree) and remove them
        from the original set.
        See function elements for inspiration.
        """
        results = []
 
        alphas = range(3)
        alphas.remove(degree) 
 
        incidentalphas = range(3)
        incidentalphas.remove(incidentdegree) 
 
        marked = set()
 
        for d in self.orbit(dart, alphas):
            if not d in marked:
                results.append(d)
                marked |= set(self.orbit(d, incidentalphas))
 
        return results

  def get_embedding_dart(self, dart, degree, propertydict ):
    """ Check if a dart of the orbit representing the element 
         of degree degree has already been associated with a value in propertydict. 
         If yes, return this dart, else return the dart passed as argument """
    alphas = range(3)
    alphas.remove(degree)
    for d in self.orbit(dart, alphas ):
        if propertydict.has_key(d): return d
    return dart
 
  def get_position(self, dart):
      """ Retrieve the coordinates associated to the vertex &lt;alpha_1, alpha_2>(dart) """
      return self.positions.get(self.get_embedding_dart(dart,0,self.positions))
 
  def set_position(self, dart, position) :
     """ Associate coordinates with the vertex &lt;alpha_1,alpha_2>(dart) """
     import numpy as np
     self.positions[self.get_embedding_dart(dart,0,self.positions)] = np.array(position)
 
 
  def sew_dart(self,degree, dart1, dart2, merge_attribute = True):
    """ Sew two elements of degree 'degree' that start at dart1 and dart2.
        Determine first the orbits of dart to sew.
        Check if they are compatible
        Sew pairs of corresponding darts.
    """
    if degree == 1:
        self.link_dart(1, dart1, dart2)
 
    else:
        alpha_v = [0,2]
        alpha_v.remove(degree)
        orbit1 = self.orbit(dart1,alpha_v)
        orbit2 = self.orbit(dart2,alpha_v)
        if len(orbit1) != len(orbit2):
            raise ValueError('Incompatible orbits', orbit1, orbit2)
        for d1,d2 in zip(orbit1, orbit2):
 
            if merge_attribute:
                d1e = self.get_embedding_dart(d1, 0, self.positions)
                d2e = self.get_embedding_dart(d2, 0, self.positions)
                if d1e in self.positions and d2e in self.positions:
                    pos = (self.positions[d1e] + self.positions[d2e]) / 2
                    del self.positions[d2e]
                    self.positions[d1e] = pos
            
            self.link_dart(degree, d1, d2)
 
 
  def orderedorbit(self,dart,list_of_alpha_value):
    """ Return the orbit of dart using a list of alpha relation.
        Example of use. gmap.orbit(0,[0,1]).
        Warning: No fixed point for the given alpha should be contained.
         """
    orbit = []
 
    cdart = dart
    alpha_v = 0
    nbalpha = len(list_of_alpha_value)
    while True :
        orbit.append(cdart)
        ncdart = self.alpha(list_of_alpha_value[alpha_v],cdart)
        alpha_v = (alpha_v + 1) % nbalpha
        if ncdart == cdart or ncdart == dart:
            break 
        cdart = ncdart
    return orbit
    
  def display(self, color = [205,205,205], add = False):
    from openalea.plantgl.all import Scene, Shape, Material, FaceSet, Viewer
    from random import randint
    s = Scene()
    for facedart in self.elements(2):
        lastdart = facedart
        positions = []
        for dart in self.orderedorbit(facedart,[0,1]):
            if self.alpha(0, dart) != lastdart:
                positions.append(self.get_position(dart))
            lastdart = dart
        if color is None:
            mat = Material((randint(0,255),randint(0,255),randint(0,255)))
        else:
            mat = Material(tuple(color),diffuse=0.25)
        s.add(Shape(FaceSet(positions, [range(len(positions))]) , mat ))
    if add : Viewer.add(s)
    else : Viewer.display(s)
 
  # def display(self, add = False, color = [190,205,205]):
  #   from openalea.plantgl.all import Scene, Shape, Material, FaceSet, Viewer
  #   from random import randint
  #   s = Scene()
  #   for facedart in self.elements(2):
  #       lastdart = facedart
  #       positions = []
  #       # positions.append(self.get_position(facedart))
  #       for dart in self.orderedorbit(facedart,[0,1]):
  #           if self.alpha(0, dart) != lastdart:
  #               positions.append(self.get_position(dart))
  #           lastdart = dart
  #       if color is None:
  #           mat = Material((randint(0,255),randint(0,255),randint(0,255)))
  #       else:
  #           mat = Material(tuple(color),diffuse=0.25)
  #       s.add(Shape(FaceSet(positions, [range(len(positions))]) , mat ))
  #   if add : Viewer.add(s)
  #   else : Viewer.display(s)

  def darts2pglscene(self, textsize = None):
            from openalea.plantgl.all import Scene, Shape, Material, FaceSet, Polyline, PointSet, Translated, Text, Font
            from numpy import array
            s = Scene()
            th = 0.1
            matdart = Material((0,0,0))
            matalphai = [Material((100,100,100)),Material((0,200,0)),Material((0,0,200)),Material((200,0,0))]
            matlabel =  Material((0,0,0))
            alphaipos = [{} for i in xrange(3)]
            vertex = []
            font = Font(size=textsize if textsize else 10)
 
            def process_edge(dart, facecenter = None, fshift = [0.,0.,0.]):
                try:
                    edgecenter = self.cell_center(dart,1)
                except:
                    print 'Cannot display dart', dart,': no coordinates.'
                    return
                eshift = array(fshift)
                if not facecenter is None : 
                    eshift += (facecenter-edgecenter)*2*th 
 
                for cdart in [dart, self.alpha(0,dart)]:
                    pi = self.get_position(cdart) 
                    pdir = (edgecenter-pi)
                    pi =  pi + (pdir*(th*3) + eshift)
                    vertex.append(pi)
                    dartlengthratio = 0.6
                    s.add(Shape(Polyline([pi,pi+pdir*dartlengthratio]) , matdart, id = cdart))
                    s.add(Shape(Translated(pi+pdir*0.5, Text(str(cdart),fontstyle=font)), matlabel, id=cdart))
                    for i in xrange(0,3):
                        oppidart = self.alpha( i,cdart)
                        if oppidart != cdart:
                            if i == 0 : 
                                alphaidartpos = pi+pdir* dartlengthratio
                            else:
                                alphaidartpos = pi+pdir*(0.1*i)
                            if alphaipos[i].has_key(oppidart):
                                s.add(Shape(Polyline([alphaipos[i][oppidart],alphaidartpos],width=5) , matalphai[i]))
                            else:
                                alphaipos[i][cdart] = alphaidartpos
 
            def process_face(dart, cellcenter = None, cshift = [0.,0.,0.]):
                try:
                    facecenter = self.cell_center(fid,2)
                except:
                    facecenter = None
                
                fshift = cshift 
                if not cellcenter is None and not facecenter is None: fshift = (cellcenter-facecenter)*th
                for dart in self.incident_cells(fid,2,1):
                    process_edge(dart, facecenter, fshift)         
                       
            for fid in self.elements(2):
                    process_face(fid)
 
            s.add(Shape(PointSet(vertex, width=5) ,Material((0,0,0)) ))
            return s
 
  def dartdisplay(self, add = False, textsize = None):
        from openalea.plantgl.all import Viewer
        s = self.darts2pglscene(textsize)
        if add : Viewer.add(s)
        else : Viewer.display(s)
 
  def nb_elements(self, degree): 
     return len(self.elements(degree))
 
  def eulercharacteristic (gmap): 
     return gmap.nb_elements(0) - gmap.nb_elements(1) + gmap.nb_elements(2)







