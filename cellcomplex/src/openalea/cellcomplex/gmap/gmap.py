

class DartProxy:
    def __init__(self, gmap, dartid):
        self.gmap = gmap
        self.dartid = dartid

    def alpha(self, degree):
        return DartProxy(self.gmap,self.gmap.alpha(degree,self.dartid))

    def set_alpha(self, degree, dart):
        if isinstance(self, DartProxy): dart = dart.dartid
        self.gmap.set_alpha(degree,self.dartid, dart)
    
    def get_position(self):
        try:
            return self.gmap.get_position(self.dartid)
        except:
            return None

    def set_position(self, pos):
        self.gmap.set_position(self.dartid, pos)

    position = property(get_position, set_position)

    def __hash__(self):
        return hash(self.dartid)

class GMap:
    def __init__(self, degree=2):
        """ constructor """
        self.degree = degree
        self.maxdartid = 0
        self.__alphas = {}
        self.positions = {}
        self.properties = dict([(i,{}) for i in xrange(degree+1)])


    def darts(self):
        """ Return a list of id representing the darts of the structure """
        return self.__alphas.keys()

    def dart(dartid):
        return DartProxy(self, dartid)

    def nb_darts(self):
        return len(self.darts())
     
    def add_dart(self, dartid = None):
        """ Create a new dart and return its id """
        if (dartid is None) or (dartid in self.__alphas):
            dartid = self.maxdartid

        if dartid >=  self.maxdartid:
            self.maxdartid = dartid + 1

        self.__alphas[dartid] = [dartid for i in xrange(self.degree+1)] # fixed point
        return dartid

    def add_darts(self, nb):
        return [self.add_dart() for i in xrange(nb)]

    def add_sdart(self, dartid = None):
        return DartProxy(self, self.add_dart(dartid))

    def remove_dart(self, dart):
        if self.positions.has_key(dart):
            pos = self.positions[dart]
            del self.positions[dart]
            for d in self.iterate_over_each_dart_of_a_i_cell(dart,0):
                if d != dart:
                    self.positions[d] =  pos 

        for deg, alphai in enumerate(self.__alphas[dart]):
            if alphai != dart and self.__alphas[alphai][deg] == dart:
                self.__alphas[alphai][deg] = alphai
        
        del self.__alphas[dart]

    def alpha(self, *degrees_and_dart):
        """ Return the alpha_i(alpha_j(...(dart)) """
        assert len(degrees_and_dart) >= 2
        dart = degrees_and_dart[-1]
        for deg in reversed(degrees_and_dart[:-1]):
            assert deg <= self.degree
            dart = self.__alphas[dart][deg]
        return dart

    def set_alpha(self, degree, dart, oppdart):
        self.__alphas[dart][degree] = oppdart

    def is_free(self, dart, degree):
        """ Test if dart is free for alpha_degree (if it is a fixed point) """
        return self.alpha(degree,dart) == dart
     
    def min_free_degree(self, dart):
        """ Return mninimum degree for which a dart is free """
        for i in xrange(self.degree+1):
            if self.alpha(i, dart) == dart : return i
        return None

     
    def compatible_min_free_degree(self,dart1, dart2):
        for i in xrange(self.degree+1):
            if self.is_free(dart1,i) and  self.is_free(dart2,i) : return i
        return None

    def link_darts(self,degree, dart1, dart2):
            """ Link the two darts with a relation alpha_degree """
            assert self.is_free(dart1,degree)
            assert self.is_free(dart2,degree)
            self.set_alpha(degree, dart1, dart2)
            self.set_alpha(degree, dart2, dart1)
     
    def overlink_darts(self,degree, dart1, dart2):
            """ Link the two darts with a relation alpha_degree. Do not check if they are free """
            self.set_alpha(degree, dart1, dart2)
            self.set_alpha(degree, dart2, dart1)
     
    def unlink_darts(self, degree, dart1, dart2 = None):
            """ Unlink the two darts with the relation alpha_degree. 
                If dart2 is not given, it will uses the dart linked with dart1. 
                If it is given, it check if it is the good one. """
            if dart2:
                assert self.alpha(degree,dart1) == dart2
            else:
                dart2 = self.alpha(degree,dart1)
                assert dart1 != dart2
            self.set_alpha(degree, dart1, dart1)
            self.set_alpha(degree, dart2, dart2)

    def are_linked_darts(self,degree, dart1, dart2):
         """ Are the two darts linked with a relation alpha_degree """
         return self.alpha(degree,dart1) == dart2
     
    def degree_of_link(self,dart1, dart2):
         """ Find the degree of the link between the two darts  """
         for deg in xrange(self.degree):
            if self.alpha(deg,dart1) == dart2: return deg

    def check_validity(self):
        """ Test the validity of the structure.
            Check if there is pending dart for alpha_0 and alpha_1 (fixed point) """
        for i in xrange(0, self.degree):
            for dart in self.darts():
                if dart == self.alpha(i,dart) : raise ValueError('Fixed points', dart,i)
                if dart != self.alpha(i,i,dart) : raise ValueError('Not involution', i, dart)

        for i in xrange(0, self.degree - 2):
            for dart in self.darts(): # alpha_i alpha_i+2 is an involution
                if self.alpha(i,i+2,i,i+2,dart) != dart: raise ValueError('Not involution', i, i+2, dart)
        
        #tolerance = 1e-5
        #from numpy.linalg import norm
        # check each dart of same point has same coordinate
        #for pid in self.iterate_over_each_i_cell(0):
        #    pos = self.get_position(pid)
        #    for d in self.iterate_over_each_dart_of_a_i_cell(pid,0):
        #        if norm(self.get_position(d) - pos) > tolerance : raise ValueError('Position problem', pid, d, pos, self.get_position(d))


    def is_valid(self):
        """ Test the validity of the structure.
            Check if there is pending dart for alpha_0 and alpha_1 (fixed point) """
        try:
            self.check_validity()
            return True
        except ValueError,e :
            return False

    def set_position(self,dart, position) :
        """ Associate coordinates with the vertex <alpha_1,...,alpha_i>(dart) """
        self.set_property(self.positions, 0, dart, position)
     
    def get_position(self,dart) :
        """ Return the coordinates with the vertex <alpha_1,...,alpha_i>(dart) """
        return self.property(self.positions, 0, dart)
     
    def has_position(self,dart) :
        """ Return the coordinates with the vertex <alpha_1,...,alpha_i>(dart) """
        return self.has_property(self.positions, 0, dart)

    def property(self, propnameordict, degree, dart):
        d, val = self.get_embedding_dart_and_value( propnameordict, degree, dart)
        if val is None : raise KeyError('No property associated with dart %i' % dart)
        return val
        

    def set_property(self, propnameordict, degree, dart, value):
        if type(propnameordict) == str:
            propvals = self.properties[degree]
        else:
            propvals = propnameordict
        for d in self.iterate_over_each_dart_of_a_i_cell(dart,degree):
            if propvals.has_key(d): 
                propvals[d] = value
            else: propvals[dart] = value

    def has_property(self, propnameordict, degree, dart):
        d, val = self.get_embedding_dart_and_value( propnameordict, degree, dart)
        return not val is None
    
    def get_embedding_dart_and_value(self, propnameordict, degree, dart):
        """ Return the dart that contains the embedding.
            If no dart still exists, return dart """
        if type(propnameordict) == str:
            propvals = self.properties[degree]
        else:
            propvals = propnameordict
        for d in self.iterate_over_each_dart_of_a_i_cell(dart,degree):
            if propvals.has_key(d):  return d, propvals[d]
        return dart, None
     
    def get_embedding_dart(self, propnameordict, degree, dart):
        """ Return the dart that contains the embedding.
            If no dart still exists, return dart """
        return self.get_embedding_dart_and_value(propnameordict, degree, dart)[0]
     
    def orbit_alphas_for_elements(self,degree, maxdegree = None):
        """ Return the alpha value for the orbit giving an element of a given degree.
            For instance a vertex is given by the orbit <alpha_1, ...., alpha_gmap_degree>. """ 
        if maxdegree is None: maxdegree = self.degree
        if maxdegree < degree: raise ValueError('Orbit for element of degree %i not possible in a structure of degree %i', degree, self.degree)
        alphas = range(maxdegree+1)
        del alphas[degree]
        return alphas

    def orbit_alphas_to_sew(self, degree):
        """ 
           Return the alpha value for the orbit to sew: <0, degree -2, degree +2, ..., gmap.degree 
           alphas_for_orbits_to_sew = { 2 : { 0 : [2],   1 : None , 2 : [0] },
                                        3 : { 0 : [2,3], 1:  [3], 2 : [0], 3 : [0,1] } }
        """ 
        alphas = range(self.degree+1)
        if degree < self.degree: del alphas[degree+1]
        del alphas[degree]
        if degree > 0: del alphas[degree-1]
        return alphas


    def orbit_iter(self, dart, list_of_alpha_value):
        """ Return the orbit of dart using a list of alpha relation.
             Example of use: gmap.orbit(0,[0,1]).
             In python, you can use the set structure to process only once all darts of the orbit.  """
        toprocess = [(dart, None)]
        marked = set()

        while len(toprocess) > 0 :
            d, a = toprocess.pop()
            if not d in marked:
                yield d, a
                marked.add(d)
                for alpha_v in list_of_alpha_value:
                    alpha_i_of_d = self.alpha(alpha_v,d)
                    if not alpha_i_of_d in marked:
                        toprocess.append((alpha_i_of_d,alpha_v) )

    def orbit(self,dart,list_of_alpha_value):
        """ Return the orbit of dart using a list of alpha relation.
             Example of use: gmap.orbit(0,[0,1]).
             In python, you can use the set structure to process only once all darts of the orbit.  """
        return [dart for dart, alphai in self.orbit_iter(dart,list_of_alpha_value)]

    def orderedorbit_iter(self,dart,list_of_alpha_value):
        """ Return the orbit of dart using a list of alpha relation.
            Example of use. gmap.orbit(0,[0,1]).
            Warning: No fixed point for the given alpha should be contained.
             """
        cdart = dart
        nbalpha = len(list_of_alpha_value)
        if nbalpha == 3:
            toprocess = [dart]
            marked = set()
            lastalpha = list_of_alpha_value.pop(-1)
            firstalpha = list_of_alpha_value[0]
            while len(toprocess) > 0:
                first = True
                ndart = topprocess.pop(0)
                for d,a in orderedorbit_iter(self,ndart,list_of_alpha_value):
                    if a == firstalpha:
                        ld = self.alpha(lastalpha, d)
                        if  ld != d and ld not in marked:
                            toprocess.append(ld)
                    marked.add(d)
                    if a is None: 
                        if first: 
                            first = False
                            yield d,a
                        else : yield d,lastalpha
                    yield d, a 

        elif nbalpha == 2:
            alpha_v = 0
            yield cdart, None
            while True :
                ncdart = self.alpha(list_of_alpha_value[alpha_v],cdart)
                if ncdart == cdart or ncdart == dart:
                    break 
                yield ncdart, list_of_alpha_value[alpha_v]
                alpha_v = (alpha_v + 1) % nbalpha
                cdart = ncdart
        else:
            raise ValueError('Cannot process list_of_alpha_value',list_of_alpha_value)

    def orderedorbit(self,dart,list_of_alpha_value):
        """ Return the orbit of dart using a list of alpha relation.
            Example of use. gmap.orbit(0,[0,1]).
            Warning: No fixed point for the given alpha should be contained.
             """
        return [dart for dart, alphai in self.orderedorbit_iter(dart,list_of_alpha_value)]
     
    def are_sewable_dart(self, dart1, dart2, degree = None):
        """ Check if two elements of degree 'degree' that start at dart1 and dart2 are sewable.

        """
        if degree is None:
            degree = self.compatible_min_free_degree(dart1,dart2)
            assert not degree is None

        orbit_alphas = self.orbit_alphas_to_sew(degree)
        if orbit_alphas is []:
            return True
        else:
            orbit1 = self.orbit(dart1,orbit_alphas)
            orbit2 = self.orbit(dart2,orbit_alphas)
            map = dict()
            if len(orbit1) != len(orbit2):
                raise ValueError('Incompatible orbits', orbit1, orbit2)
            for d1,d2 in zip(orbit1, orbit2):            
                map[d1] = d2
                for deg in orbit_alphas:
                    try:
                        if map[self.alpha(deg,d1)] != self.alpha(deg,d2): return False
                    except KeyError, e: pass
            return True

    def sew_dart(self, dart1, dart2, degree = None, verbose = False):
        """ Sew two elements of degree 'degree' that start at dart1 and dart2.
            Determine first the orbits of dart to sew.
            Sew pairs of corresponding darts.
        """
        if degree is None:
            degree = self.compatible_min_free_degree(dart1,dart2)
            assert not degree is None

        assert self.are_sewable_dart(dart1, dart2, degree)

        orbit_alphas = self.orbit_alphas_to_sew(degree)
        if orbit_alphas is []:
            self.link_darts(degree, dart1, dart2)
        else:
            orbit1 = self.orbit(dart1,orbit_alphas)
            orbit2 = self.orbit(dart2,orbit_alphas)
            if len(orbit1) != len(orbit2):
                raise ValueError('Incompatible orbits', orbit1, orbit2)
            for d1,d2 in zip(orbit1, orbit2):
                if verbose: print 'link',d1,d2
                self.link_darts(degree, d1, d2)


    def unsew_dart(self, dart1, degree = None):
        """ Sew two elements of degree 'degree' that start at dart1 and dart2.
            Determine first the orbits of dart to sew.
            Check if they are compatible
            Sew pairs of corresponding darts.
        """
        if degree is None:
            degree = self.degree_of_link(dart1,dart2)
            assert not degree is None

        orbit_alphas = self.orbit_alphas_to_sew(degree)
        if orbit_alphas is []:
            self.unlink_darts(degree, dart1, dart2)
        else:
            orbit1 = self.orderedorbit(dart1,orbit_alphas)
            for d1 in  self.orderedorbit(dart1,orbit_alphas):
                self.unlink_darts(degree, d1)

    def detect_sewable_cell(self, dart1, dart2, degree):
        from numpy.linalg import norm
        pos1 = [(d,self.get_position(d)) for d in self.iterate_over_each_incident_cell(dart1,degree,0)]
        pos2 = [(d,self.get_position(d)) for d in self.iterate_over_each_incident_cell(dart2,degree,0)]
        pos1.sort(cmp=lambda a,b: cmp(a[1][0],b[1][0]))
        pos2.sort(cmp=lambda a,b: cmp(a[1][0],b[1][0]))
        fp2 = 0
        epsilon = 1e-3
        mapping = []
        mapped1, mapped2 = set(), set()
        for d1, p1 in pos1:
            if not d1 in mapped1:
                for d2, p2 in pos2[fp2:]:
                    if not d2 in mapped2:
                        if p2[0] < p1[0]: fp2 += 1
                        if norm(p1-p2) < epsilon: 
                            segcandidate1 = [self.alpha(0,d) for d in self.iterate_over_each_incident_cell(d1,0,1)]
                            segcandidate2 = [self.alpha(0,d) for d in self.iterate_over_each_incident_cell(d2,0,1)]
                            candidates = None
                            for cand1 in segcandidate1:
                                for cand2 in segcandidate2:
                                    if norm(self.get_position(cand1)-self.get_position(cand2)) < epsilon:
                                        candidates = (self.alpha(0,cand1),self.alpha(0,cand2))
                                        break
                                else:
                                    continue
                                break
                            if candidates:
                                c1, c2 = candidates
                                for lc1 in self.orbit(c1,range(2,self.degree+1)):
                                    if not lc1 in mapped1:
                                        mapped = False
                                        for lc2 in self.orbit(c2,range(2,self.degree+1)):
                                            if not lc2 in mapped2:
                                                orbit1 = list(self.iterate_over_each_dart_of_a_i_cell(lc1, degree-1))
                                                orbit2 = list(self.iterate_over_each_dart_of_a_i_cell(lc2, degree-1))
                                                if len(orbit1) != len(orbit2): continue
                                                for od1, od2 in zip(orbit1, orbit2):
                                                    if norm(self.get_position(od1)-self.get_position(od2)) > epsilon:
                                                        break
                                                else:
                                                    mapping.append((lc1,lc2))
                                                    mapped1 |= set(orbit1)
                                                    mapped2 |= set(orbit2)
                                                    mapped = True
                                                    break
                                        if mapped: break

        return mapping

    def intuitive_sew(self,dart1, dart2, degree):
        mapping = self.detect_sewable_cell( dart1, dart2, degree)
        for d1,d2 in mapping:
            self.sew_dart(d1,d2,degree)
        return mapping


    def iterate_over_each_i_cell(self, degree):
            darts = set(self.darts())

            list_of_alpha_value = self.orbit_alphas_for_elements(degree)

            while len(darts) > 0:
                dart = darts.pop()
                yield dart
                darts -= set(self.orbit(dart, list_of_alpha_value))

    def iterate_over_each_dart_of_a_i_cell(self, dart, degree):
            for d, deg in self.orbit_iter(dart, self.orbit_alphas_for_elements(degree)):
                yield d


    def iterate_over_each_connex_components(self):
            darts = set(self.darts())

            list_of_alpha_value = range(self.degree+1)

            while len(darts) > 0:
                dart = darts.pop()
                yield dart
                darts -= set(self.orbit(dart, list_of_alpha_value))

    def iterate_over_each_incident_cell(self, dart, degree, incidentdegree):
            results = []
            maxdegree = degree if degree > incidentdegree else self.degree
            list_of_alpha_value = self.orbit_alphas_for_elements(incidentdegree, maxdegree)
            marked = set()

            for d in self.orbit(dart, self.orbit_alphas_for_elements(degree,maxdegree)):
                if not d in marked:
                    yield d
                    marked |= set(self.orbit(d, list_of_alpha_value))

    def iterate_over_each_adjacent_cell(self, dart, degree):
            results = []
            list_of_alpha_value = self.orbit_alphas_for_elements(degree)
            marked = set()

            for d in self.orbit(dart, list_of_alpha_value):
                nd = self.alpha( degree,d)
                if not nd in marked:
                    yield nd
                    marked |= set(self.orbit(nd, list_of_alpha_value))


    def nb_cells(self, degree): 
            return len(list(self.iterate_over_each_i_cell(degree)))

    def nb_connex_components(self): 
            return len(list(self.iterate_over_each_connex_components()))

    def copy(self):
        from copy import deepcopy
        return deepcopy(self)
     
    def decrease_degree(self):
        for dart, dartalphas in self.__alphas.items():
            del dartalphas[self.degree]
        self.degree -= 1

    def increase_degree(self):
        self.degree += 1
        for dart, dartalphas in self.__alphas.items():
            dartalphas.append(dart)

    def add(self, gmap):
        """ add topology of gmap into self. Return mapping of element of gmap into self. """
        mapping = dict()
        for dart in gmap.darts():
            mapping[dart] = self.add_dart(dart)

        degree = min(gmap.degree,self.degree)
        for olddart, newdart in mapping:
            for deg in xrange(degree):
                self.set_alpha( deg, newdart, mapping[gmap.alpha(deg,olddart)])
            if gmap.has_position(olddart):
                self.set_position(newdart, gmap.get_position(olddart))

        return mapping

    def restriction(self, darts):
        """ return a copy of self restricted to the given set of darts """
        gmap = GMap(self.degree)

        for dart in darts:
            d = gmap.add_dart(dart)
            assert d == dart

        for dart in darts:
            for deg in xrange(self.degree):
                opdart = self.alpha(deg,dart)
                if opdart in darts : gmap.set_alpha( deg,dart, opdart)
            if self.has_position(dart):
                gmap.set_position(dart, self.get_position(dart))

        return gmap
      

    def iboundary(self, degree = None):
        if degree is None: degree = self.degree
        boundarydarts = [dart for dart in self.darts() if self.is_free(dart,degree)]

        gmap = GMap(self.degree)

        for dart in boundarydarts:
            d = gmap.add_dart(dart)
            assert d == dart

        alphas = self.orbit_alphas_to_sew(degree)

        for dart in boundarydarts:
            for deg in alphas:
                opdart = self.alpha(deg,dart)
                gmap.set_alpha( deg,dart, opdart)
            # computing for degree - 1
            nextdart = lambda d : self.alpha(degree, degree-1, d)
            cdart = dart
            while not self.is_free(self.alpha( degree -1, cdart),degree):
                cdart = nextdart(cdart)

            cdart = self.alpha(degree-1,cdart)
            gmap.set_alpha( degree-1,dart, cdart)
            if self.has_position(dart):
                gmap.set_position(dart,self.get_position(dart))

        if degree == self.degree: gmap.decrease_degree()
        return gmap
      
    def dual(self):
        from numpy import array
        gmap = self.copy()
        newpositions = dict()
        for dart in self.iterate_over_each_i_cell(self.degree):
            celldarts = list(self.orbit_iter(dart,self.orbit_alphas_for_elements(self.degree)))
            points = [self.get_position(d) for d, deg in celldarts if deg != 1]
            cellcenter = sum(points,array([0,0,0]))/len(points)
            newpositions[dart] = cellcenter
        gmap.__alphas = dict( [(dart,list(reversed(alphas))) for dart, alphas in gmap.__alphas.items()] )
        gmap.positions = dict()
        for dart, pos in newpositions.items():
            gmap.set_position(dart,pos)
        return gmap

    def is_orientable(self):
        E1, E2 = set([]),set([])
        dart0 = self.darts()[0]
        for dart0 in self.iterate_over_each_connex_components():
            toprocess = [dart0]
            E1.add(dart0)

        while len(toprocess) > 0:
            dart = toprocess.pop(0)
            if dart in E1:    e1,e2 = E1,E2
            elif dart in E2:  e1,e2 = E2,E1
            else:             raise ValueError('Problem in orientation computation')

            for deg in xrange(self.degree+1):
                oppdart = self.alpha(deg,dart)
                if dart != oppdart:
                    if oppdart in e1: return False
                    elif not oppdart in e2:
                        e2.add(oppdart)
                        toprocess.append(oppdart)
        return True

    def orientation(self):
        E1, E2 = set([]),set([])
        dart0 = self.darts()[0]
        for dart0 in self.iterate_over_each_connex_components():
            toprocess = [dart0]
            E1.add(dart0)

        while len(toprocess) > 0:
            dart = toprocess.pop(0)
            if dart in E1:    e1,e2 = E1,E2
            elif dart in E2:  e1,e2 = E2,E1
            else:             raise ValueError('Problem in orientation computation')

            for deg in xrange(self.degree+1):
                oppdart = self.alpha(deg,dart)
                if dart != oppdart:
                    if oppdart in e1: raise ValueError('Not orientable')
                    elif not oppdart in e2:
                        e2.add(oppdart)
                        toprocess.append(oppdart)
        return E1


    def iclosure(self, degree):
        for dart in self.darts():
            if self.is_free(dart, degree):
                ndart = self.add_dart()
                self.link_darts(degree, dart, ndart)

                for deg in self.orbit_alphas_to_sew(degree):
                    njdart = self.alpha( deg,dart)
                    if not self.is_free(dart, deg) and not self.is_free(njdart, degree):
                        self.link_darts(deg, ndart, self.alpha( degree,njdart))

                if degree > 0:
                    nidart = self.alpha( degree -1,dart)

                    nextdart = lambda d : self.alpha(degree-1,self.alpha( degree,d))
                    while not self.is_free(nidart,degree) and not self.is_free(self.alpha( degree,nidart),degree-1):
                        nidart = nextdart(nidart)

                    if not self.is_free(nidart, degree):
                        self.link_darts(degree-1,ndart,self.alpha(degree,nidart))

      
    def closure(self):
        for deg in xrange(self.degree+1):
            self.iclosure(deg)
      
    def is_removable_cell(self, dart, degree):
        if degree == self.degree : return False
        if degree == self.degree - 1 : return True
        for d in self.iterate_over_each_dart_of_a_i_cell(dart, degree):
            if self.alpha( degree+1, degree+2, d) != self.alpha(degree+2, degree+1, d): return False
        return True
      
    def remove_cell(self, dart, degree):
        assert self.is_removable_cell(dart,degree)

        marked = set(self.iterate_over_each_dart_of_a_i_cell(dart, degree))
        for d in self.iterate_over_each_dart_of_a_i_cell(dart, degree):
            d1 = self.alpha( degree,d)
            if not d1 in marked:
                d2 = self.alpha(degree,degree+1,d)
                while d2 in marked:
                    d2 = self.alpha(degree,degree+1,d2)
                self.set_alpha( degree,d1,d2)

        toremove = list(self.iterate_over_each_dart_of_a_i_cell(dart, degree))
        for d in toremove:
            self.remove_dart(d)

    def is_contractible(self, dart, degree):
        if degree == 0: return False
        if degree == 1: return True
        for d in self.iterate_over_each_dart_of_a_i_cell(dart, degree):
            if self.alpha(degree-1, degree-2, d) != self.alpha(degree-2, degree-1, d): return False
        return True

    def contract_cell(self, dart, degree):
        assert self.is_contractible(dart,degree)

        marked = set(self.iterate_over_each_dart_of_a_i_cell(dart, degree))
        for d in self.iterate_over_each_dart_of_a_i_cell(dart, degree):
            d1 = self.alpha( degree,d)
            if not d1 in marked:
                d2 = self.alpha(degree,degree-1,d)
                while d2 in marked:
                    d2 = self.alpha(degree,degree-1,d2)
                self.set_alpha( degree,d1,d2) 

        toremove = list(self.iterate_over_each_dart_of_a_i_cell(dart, degree))
        for d in toremove:
            self.remove_dart(d)

    def is_insertable(self, gmap, degree, mapping):
        """
        Test if gmap can be inserted into self.
        links between gmap and self are specified using 'degree' and 'mapping'.
        'mapping' gives a mapping between darts of gmap to darts of self.
        """
        icell = list(gmap.iterate_over_each_i_cell(degree)) 
        if len(icell) != 1: return False
        icell = icell[0]
        if not gmap.is_removable_cell(icell, degree): return False
        invmap = dict([(v,k) for k,v in mapping.items()])
        for dart, ndart in mapping.items():
            if not gmap.is_free(dart, degree):  return False
            for deg in self.orbit_alphas_to_sew(degree):
                if gmap.alpha(deg,dart) in mapping and mapping[gmap.alpha(deg,dart)] != self.alpha(deg,ndart): return False
                if self.alpha(deg,ndart) in invmap  and invmap[self.alpha(deg,ndart)] != gmap.alpha(deg,invmap[ndart]): return False
            dart2 = gmap.alpha( degree+1,dart)
            while not dart2 in mapping:
                dart2 = gmap.alpha(degree+1,degree,dart2,)
            if mapping[dart2] != self.alpha(degree,ndart): return False
        return True
      
    def insert(self, gmap, degree, mapping):
        """
        Insert gmap into self.
        links between gmap and self are specified using 'degree' and 'mapping'.
        'mapping' gives a mapping between darts of gmap to darts of self.
        """
        assert self.is_insertable(gmap, degree, mapping)
        idmap = dict()
        for dart in gmap.darts():
            idmap[dart] = self.add_dart(dart)

        #print idmap
        for dart in gmap.darts():
            ndart = idmap[dart]
            for deg in xrange(gmap.degree+1):
                if not gmap.is_free(dart,deg):
                    self.set_alpha( deg,ndart, idmap[gmap.alpha(deg,dart)])
            if gmap.positions.has_key(dart): self.set_position(ndart, gmap.get_position(dart))
            if dart in mapping:
                self.set_alpha( degree,ndart, mapping[dart])
                self.set_alpha( degree,mapping[dart], ndart)


    def is_expansible(self, gmap, degree, mapping):
        """
        Test if gmap can be inserted into self.
        links between gmap and self are specified using 'degree' and 'mapping'.
        'mapping' gives a mapping between darts of gmap to darts of self.
        """
        icell = list(gmap.iterate_over_each_i_cell(degree)) 
        if len(icell) != 1: return False
        icell = icell[0]
        if not gmap.is_contractible_cell(icell): return False
        invmap = dict([(v,k) for k,v in mapping.items()])
        for dart, ndart in mapping.items():
            if not gmap.is_free(dart, degree):  return False
            for deg in self.orbit_alphas_to_sew(degree):
                if gmap.alpha(deg,dart) in mapping and mapping[gmap.alpha(deg,dart)] != self.alpha(deg,ndart): return False
                if self.alpha(deg,ndart) in invmap  and invmap[self.alpha(deg,ndart)] != gmap.alpha(deg,invmap[ndart]): return False
            dart2 = gmap.alpha( degree-1,dart)
            while not dart2 in mapping:
                dart2 = gmap.alpha(degree+1,degree,dart2)
            if mapping[dart2] != self.alpha(degree,ndart): return False
        return True

    def expand(self, gmap, degree, mapping):
        """
        Insert gmap into self.
        links between gmap and self are specified using 'degree' and 'mapping'.
        'mapping' gives a mapping between darts of gmap to darts of self.
        """
        assert self.is_expansible(gmap, degree, mapping)
        idmap = dict()
        for dart in gmap.darts():
            idmap[dart] = self.add_dart(dart)

        for dart in gmap.darts():
            ndart = idmap[dart]
            for deg in xrange(gmap.degree+1):
                if not gmap.is_free(dart,deg):
                    self.set_alpha( deg,ndart, idmap[gmap.alpha(deg,dart)])
            if gmap.positions.has_key(dart): self.set_position(ndart, gmap.get_position(dart))
            if dart in mapping:
                self.set_alpha( degree,dart, mapping[dart])
                self.set_alpha( degree,mapping[dart], ndart)

    def ichamfering(self, dart, degree):
        phi = dict([(deg,{}) for deg in xrange(self.degree+1)])
        nbgs = list(self.iterate_over_each_dart_of_a_i_cell(dart, degree))

        for ndart in nbgs:
            phi[degree][ndart] = ndart
            for j in xrange(degree+1, self.degree+1):
                phi[j][ndart] = self.add_dart()

        for ndart in nbgs:
            for j in xrange(degree+1,self.degree+1):
                for k in xrange(0, degree):
                    self.set_alpha(k,phi[j][ndart],phi[j][self.alpha(k,ndart)])

                for k in xrange(degree, j):
                    self.set_alpha(k,phi[j][ndart], phi[j][self.alpha(k+1,ndart)])

                self.set_alpha(j,phi[j][ndart], phi[j-1][ndart])

                if j < self.degree:
                    self.set_alpha(j+1,phi[j][ndart], phi[j+1][ndart])

                for k in xrange(j+2,self.degree+1):
                    self.set_alpha(k,phi[j][ndart], phi[j][ndart])

        for ndart in nbgs:
            self.set_alpha(degree+1,ndart, phi[degree+1][ndart])


    def chamfering(self, dart, degree, geomratio = 0.3):
        posdict = dict()

        if degree == 0:
            if self.has_position(dart):
                pos = self.get_position(dart)
                for ndart, deg in self.orderedorbit_iter(dart,[1,2]):
                    if deg != 2 :
                        posdict[ndart] = pos*(1-geomratio)+self.get_position(self.alpha(0,ndart))*geomratio
        elif degree == 1:

            # for cdart in [dart, self.alpha(0,dart)]:
            #     pos =  self.get_position(cdart)
            #     for ndart, deg in self.orderedorbit_iter(cdart,[1,2]):
            #         if deg ==  1 and ndart != cdart:
            #             posdict[ndart] = pos*(1-geomratio)+self.get_position(self.alpha(0,ndart))*geomratio
            pass


        self.ichamfering(dart, degree)

        #print posdict
        for ndart, pos in posdict.items():
             self.set_position(ndart,pos)


    def print_alphas(self, darts = None):
        if darts is None: darts = self.darts()

        for d in darts:
            print d,'\t',
            for i in xrange(self.degree+1):
               print self.alpha(i,d),'\t',
            print self.get_position(d) if self.positions.has_key(d) else None

    def eulercharacteristic (self): 
        return sum([pow(-1,i)*self.nb_cells(i) for i in xrange(self.degree+1)])

    def cell_center(self, dart, degree):
        from numpy import array
        points = [self.get_position(d) for d, deg in self.orbit_iter(dart,self.orbit_alphas_for_elements(degree)) if deg != 1]
        return sum(points,array([0.,0.,0.]))/len(points)

    def translate(self, translation):
        newpositions = {}
        for dart,pos in self.positions.items():
            newpositions[dart] = pos + translation
        self.positions = newpositions

    def scale(self, scaling):
        newpositions = {}
        for dart,pos in self.positions.items():
            newpositions[dart] = pos * scaling
        self.positions = newpositions

    def display(self, degree = None, add = False, randomcolor = True):
        from openalea.plantgl.all import Scene, Shape, Material, FaceSet, Polyline, PointSet, Viewer
        from random import randint
        s = Scene()
        m = Material()
        if degree is None: degree = self.degree
        if degree >= 2:
            try:
                ccw = self.orientation()
            except ValueError:
                ccw = set()

            for fid in self.iterate_over_each_i_cell(2):
                if fid in ccw: ofid = self.alpha(0,fid)
                else: ofid = fid
                positions = []
                for dart, deg in self.orderedorbit_iter(ofid,[0,1]):
                    if deg != 1:
                        positions.append(self.get_position(dart))
                s.add(Shape(FaceSet(positions, [range(len(positions))]) , Material((randint(0,255),randint(0,255),randint(0,255))) if randomcolor else m))
        elif degree == 1:
            for eid in self.iterate_over_each_i_cell(1):
                s.add(Shape(Polyline([self.get_position(eid),self.get_position(self.alpha(0,eid))]) ,Material((randint(0,255),randint(0,255),randint(0,255))) ))
        elif degree == 0:
                s.add(Shape(PointSet([self.get_position(pid) for pid in self.iterate_over_each_i_cell(0)]) ,Material((randint(0,255),randint(0,255),randint(0,255))) ))
        
        
        if add : Viewer.add(s)
        else   : Viewer.display(s)

    def darts2pglscene(self, textsize = None):
            from openalea.plantgl.all import Scene, Shape, Material, FaceSet, Polyline, PointSet, Translated, Text, Font
            from numpy import array
            s = Scene()
            th = 0.1
            matdart = Material((0,0,0))
            matalphai = [Material((100,100,100)),Material((0,200,0)),Material((0,0,200)),Material((200,0,0))]
            matlabel =  Material((0,0,0))
            alphaipos = [{} for i in xrange(self.degree+1)]
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
                    for i in xrange(0,self.degree+1):
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
                for dart in self.iterate_over_each_incident_cell(fid,2,1):
                    process_edge(dart, facecenter, fshift)         
                       
            if self.degree == 3:
                for cellid in self.iterate_over_each_i_cell(3):
                    try:
                        cellcenter = self.cell_center(cellid,3)
                    except:
                        cellcenter = None
                    for fid in self.iterate_over_each_incident_cell(cellid,3,2):
                        process_face(fid, cellcenter)

            elif self.degree == 2:
                for fid in self.iterate_over_each_i_cell(2):
                    process_face(fid)

            elif self.degree == 1:
                for eid in self.iterate_over_each_i_cell(1):
                    process_edge(eid)


            s.add(Shape(PointSet(vertex, width=5) ,Material((0,0,0)) ))
            return s

    def dartdisplay(self, add = False, textsize = None):
        from openalea.plantgl.all import Viewer
        s = self.darts2pglscene(textsize)
        if add : Viewer.add(s)
        else : Viewer.display(s)

def pointvalence(gmap, pointdart):
    return len(list(gmap.iterate_over_each_incident_cell(pointdart,0,1)))

GMap.pointvalence = pointvalence




