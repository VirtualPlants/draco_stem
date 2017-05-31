# -*- coding: utf-8 -*-
# -*- python -*-
#
#       DRACO-STEM
#       Dual Reconstruction by Adjacency Complex Optimization
#       SAM Tissue Enhanced Mesh
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
from scipy.cluster.vq import vq
from copy import deepcopy
import pickle

from openalea.deploy.shared_data import shared_data

from openalea.image.spatial_image import SpatialImage
from openalea.image.serial.all import imread, imsave

from vplants.tissue_analysis.temporal_graph_from_image import graph_from_image
from vplants.tissue_analysis.spatial_image_analysis import SpatialImageAnalysis

from openalea.container import array_dict
from openalea.mesh import PropertyTopomesh, TriangularMesh
from openalea.mesh.property_topomesh_creation import vertex_topomesh, edge_topomesh, triangle_topomesh, tetrahedra_topomesh
from openalea.mesh.property_topomesh_analysis import compute_topomesh_property

from openalea.mesh.utils.tissue_analysis_tools import cell_vertex_extraction
from openalea.mesh.utils.intersection_tools import intersecting_triangle

from openalea.draco_stem.draco.adjacency_complex_optimization import delaunay_tetrahedrization_topomesh, clean_tetrahedrization, compute_tetrahedrization_geometrical_properties, triangles_from_adjacency_edges
from openalea.draco_stem.draco.adjacency_complex_optimization import tetrahedrization_topomesh_topological_optimization, tetrahedrization_topomesh_add_exterior, tetrahedrization_topomesh_remove_exterior
from openalea.draco_stem.draco.adjacency_complex_construction import layer_triangle_topomesh_construction, layered_tetrahedra_topomesh_construction

from openalea.draco_stem.draco.dual_reconstruction import tetrahedra_dual_triangular_topomesh


class DracoMesh(object):
    """DRACO - Dual Reconstruction by Adjacency Complex Optimization.

    Class constructing a PropertyTopomesh representing a segmented tissue SpatialImage by the dualization of an optimized adjacency complex
        --> Initialize with an image (file)
        --> After initialization pick an adjacency complex generation method (and/or optimization)
        --> Generate a tissue mesh by dual reconstruction with the desired properties
    """

    def __init__(self, image=None, image_file=None, image_cell_vertex_file=None, triangulation_file=None, reconstruction_triangulation=['star','split','projected','flat']):
        """Initialize the DRACO object by providing a segmented image.

        Image can be passed either as an object of a filename. Cell adjacency information will be extracted from image 
        at initialization, still, some time can always be saved: if previously extracted, it can be read from existing
        file; if not the extrcted information will be saved in the specified files.

        Args:
            image (SpatialImage): a (non-eroded) segmented label image
            image_file (str): a valid path to an image of readable type (.inr, .inr.gz, .tiff...) : 
            image_cell_vertex_file (str): file to read from if cell-vertices have already been extracted
            triangulation_file (str): file to load from if an already existing adjacency complex is to be used 
            reconstruction_triangulation (list): default values for dual reconstruction triangulation (see dual_reconstruction for more details)

        Returns:
            None
        """

        if image is not None:
            self.segmented_image = deepcopy(image)
        else:
            self.segmented_image = imread(image_file)
        self.segmented_image[self.segmented_image==0] = 1
        self.voxelsize = np.array(self.segmented_image.voxelsize)
        self.size = np.array(self.segmented_image.shape)
        
        self.triangulation_topomesh = None

        self.image_graph = None
        
        if image_cell_vertex_file is not None:
            try:
                self.image_cell_vertex = pickle.load(open(image_cell_vertex_file,"rb"))
            except:
                self.image_cell_vertex = cell_vertex_extraction(self.segmented_image)
                pickle.dump(self.image_cell_vertex,open(image_cell_vertex_file,"wb"))
        else:
            self.image_cell_vertex = cell_vertex_extraction(self.segmented_image)
        self.image_cell_vertex_topomesh = None

        self.positions = None
        self.cell_layer = None
        self.image_labels = None
        self.image_cell_volumes = None
        self.image_wall_surfaces = None
        self.point_topomesh = None

        self.delaunay_topomesh = None
        self.optimized_delaunay_topomesh = None

        self.surface_topomesh = None
        self.reconstruction_triangulation = reconstruction_triangulation
        self.dual_reconstruction_topomesh = None

        self.compute_image_adjacency()

        if triangulation_file is not None:
            try:
                topomesh = pickle.load(open(triangulation_file,'rb'))
                self.triangulation_topomesh = topomesh
            except:
                print "Impossible to load adjacency complex topomesh : FileNotFound ",triangulation_file


    def compute_image_adjacency(self):
        """Compute the adjacency relationships between cells in the tissue image.

        By default, the DRACO adjacency complex is set to the cell_vertices tetrahedra (!!NOT A VALID CELL COMPLEX!!)
        !!STRONG RISK OF TOPOLOGICAL ERRORS IF DUALIZING AT THIS STAGE!!

        Updates:
            image_graph (PropertyGraph): adjacency graph connecting adjacent cells by edges
            image_cell_vertex (dict): cell vertices representing 4 co-adjacent cells by the point where they meet
            cell_layer (int): information whether the cell belong to the first layer of cells (L1), the second one (L2) or lower layers (0)
            layer_edge_topomesh (PropertyTopomesh): adjacency graphs restricted to the L1 and L2 layers
            layer_triangle_topomesh (PropertyTopomesh): adjacency triangles between L1 and L2 layers (**TO BE COMPLETED**)
            triangulation_topomesh (PropertyTopomesh): the tetrahedra representing adjacencies between cells 

        Returns:
            None
        """

        self.image_graph = graph_from_image(self.segmented_image, spatio_temporal_properties=['volume','barycenter','L1'], ignore_cells_at_stack_margins=False, property_as_real=True)   
        self.image_labels = np.array(list(self.image_graph.vertices()))
        self.image_cell_volumes = array_dict([self.image_graph.vertex_property('volume')[v] for v in self.image_labels],self.image_labels)
        img_center = np.nanmean(self.image_graph.vertex_property('barycenter').values(),axis=0)

        self.positions = array_dict(self.image_graph.vertex_property('barycenter'))
        self.point_topomesh = vertex_topomesh(self.positions)

        img_analysis = SpatialImageAnalysis(self.segmented_image)
        exterior_cells = np.array(list(img_analysis.neighbors(1)))
        self.image_wall_surfaces = img_analysis.wall_areas(real=True)

        self.image_graph.add_vertex(1)
        for c in exterior_cells:
            self.image_graph.add_edge(1,c)

        for v in self.image_cell_vertex.keys():
            self.image_cell_vertex[v] = self.image_cell_vertex[v]*self.voxelsize
        image_cell_vertex_tetrahedra = np.sort(self.image_cell_vertex.keys())
        image_cell_vertex_tetrahedra = np.delete(image_cell_vertex_tetrahedra,np.where(image_cell_vertex_tetrahedra[:,0]==1)[0],axis=0)
        self.image_cell_vertex_topomesh = tetrahedra_topomesh(image_cell_vertex_tetrahedra,self.positions)
        self.triangulation_topomesh = deepcopy(self.image_cell_vertex_topomesh)

        truncated_image = self.segmented_image[:,:,:]
        truncated_image_graph = graph_from_image(truncated_image, spatio_temporal_properties=['barycenter','L1'], background=1, ignore_cells_at_stack_margins=True, property_as_real=True)

        self.cell_layer = array_dict(np.zeros_like(self.positions.keys()),self.positions.keys())
        for c in truncated_image_graph.vertices():
            if c>1 and truncated_image_graph.vertex_property('L1')[c]:
                self.cell_layer[c] = 1
        for c in self.cell_layer.keys():
            if c>1 and self.cell_layer[c] == 1:
                for n in truncated_image_graph.neighbors(c):
                    if n>1 and self.cell_layer[n]!=1:
                        self.cell_layer[n] = 2
        self.point_topomesh.update_wisp_property('layer',0,self.cell_layer.values(list(self.point_topomesh.wisps(0))),list(self.point_topomesh.wisps(0)))

        self.layer_edge_topomesh = {}

        if (self.cell_layer.values() == 1).sum() > 1:
            L1_edges = np.array([[(c,n) for n in self.image_graph.neighbors(c) if n>1 and self.cell_layer[n]==1] for c in self.cell_layer.keys() if self.cell_layer[c]==1])
            L1_edges = np.concatenate([e for e in L1_edges if len(e)>0])
            L1_edges = L1_edges[L1_edges[:,1]>L1_edges[:,0]]
            L1_edge_topomesh = edge_topomesh(L1_edges, self.positions)
            self.layer_edge_topomesh['L1'] = L1_edge_topomesh

        if (self.cell_layer.values() == 2).sum() > 1:
            L2_edges = np.array([[(c,n) for n in self.image_graph.neighbors(c) if n>1 and self.cell_layer[n]==2] for c in self.cell_layer.keys() if self.cell_layer[c]==2])
            L2_edges = np.concatenate([e for e in L2_edges if len(e)>0])
            L2_edges = L2_edges[L2_edges[:,1]>L2_edges[:,0]]
            L2_edge_topomesh = edge_topomesh(L2_edges, self.positions)
            self.layer_edge_topomesh['L2'] = L2_edge_topomesh

        self.layer_triangle_topomesh = {}

        if (self.cell_layer.values() == 1).sum() > 1 and (self.cell_layer.values() == 2).sum() > 0:
            L1_L2_edges = np.array([[(c,n) for n in self.image_graph.neighbors(c) if n>1 and self.cell_layer[n] in [1,2]] for c in self.cell_layer.keys() if self.cell_layer[c] in [1,2]])
            L1_L2_edges = np.concatenate([e for e in L1_L2_edges if len(e)>0])
            L1_L2_edges = L1_L2_edges[L1_L2_edges[:,1]>L1_L2_edges[:,0]]

            L1_L2_additional_edges = np.array([[(c,n) for n in np.unique(np.array(self.image_cell_vertex.keys())[np.where(np.array(self.image_cell_vertex.keys())==c)[0]])  if n>1 and n!=c and (n not in self.image_graph.neighbors(c)) and (self.cell_layer[n] in [1,2])] for c in self.cell_layer.keys() if self.cell_layer[c] in [1,2]])
            if len([e for e in L1_L2_additional_edges if len(e)>0])>0:
                L1_L2_additional_edges = np.concatenate([e for e in L1_L2_additional_edges if len(e)>0])
                L1_L2_additional_edges = L1_L2_additional_edges[L1_L2_additional_edges[:,1]>L1_L2_additional_edges[:,0]]
                L1_L2_edges = np.concatenate([L1_L2_edges,L1_L2_additional_edges])

            self.layer_triangle_topomesh['L1_L2'] =  triangle_topomesh(triangles_from_adjacency_edges(L1_L2_edges),self.positions)


    def delaunay_adjacency_complex(self, surface_cleaning_criteria = ['surface','exterior','distance','sliver']):
        """Estimate the adjacency complex by the Delaunay tetrahedrization of the cell barycenters.

        Since Delaunay applied on the cell barycenters would produce a convex simplicial complex, it is necessary 
        to carve out the complex to keep only the actual relevant simplices.

        Args:
            surface_cleaning_criteria (list): the criteria used during the surface carving phase of the Delaunay complex :
                • 'exterior' : remove surface simplices that lie entirely outside the tissue
                • 'surface' : remove surface simplices that intersect the surface of the tissue
                • 'distance' : remove surface simplices that link cells too far apart
                • 'sliver' : remove surface simplices that create flat tetrahedra (slivers)

        Updates:
            triangulation_topomesh (PropertyTopomesh) : the DracoMesh adjacency complex is set to this Delaunay complex.

        Returns:
            None
        """     

        clean_surface = len(surface_cleaning_criteria)>0
        self.delaunay_topomesh = delaunay_tetrahedrization_topomesh(self.positions, image_cell_vertex=self.image_cell_vertex, segmented_image=self.segmented_image, clean_surface=clean_surface, surface_cleaning_criteria=surface_cleaning_criteria)
        clean_tetrahedrization(self.delaunay_topomesh, clean_vertices=False)
        discarded_cells = np.array(list(self.delaunay_topomesh.wisps(0)))[np.where(np.array(map(len,[list(self.delaunay_topomesh.regions(0,v,2)) for v in self.delaunay_topomesh.wisps(0)]))==0)[0]]
        for v in discarded_cells:
            self.delaunay_topomesh.remove_wisp(0,v)  
        self.triangulation_topomesh = deepcopy(self.delaunay_topomesh)


    def adjacency_complex_optimization(self, n_iterations = 1, omega_energies = {'image':10.0,'geometry':0.1,'adjacency':0.01}):
        """Optimize the adjacency complex to match better the actual cell adjacencies in the tissue.

        The optimization is performed as an iterative energy minimization process of local topological transformations 
        (edge flips and triangle swaps) following a simulated annealing heuristic.

        Args:
            n_iterations (int): number of iterations (cycles of simulated annealing) to perform
            omega_energies (dict): weights of the different terms of the energy functional :
                • 'image' : energy measuring the difference between the adjacency complex simplices and the ones extracted from the image
                • 'geometry' : energy penalizing irregular tetrahedra in the adjacency complex
                • 'adjacency' : energy pulling the number of neighbors of each cell to an empirical optimal value according to its layer

        Updates:
            triangulation_topomesh (PropertyTopomesh) : the DracoMesh adjacency complex is set to this optimized complex.

        Returns:
            None
        """

        self.optimized_delaunay_topomesh = deepcopy(self.delaunay_topomesh)
        compute_tetrahedrization_geometrical_properties(self.optimized_delaunay_topomesh)
        tetrahedrization_topomesh_add_exterior(self.optimized_delaunay_topomesh)
        self.optimized_delaunay_topomesh = tetrahedrization_topomesh_topological_optimization(self.optimized_delaunay_topomesh,image_cell_vertex=self.image_cell_vertex,omega_energies=omega_energies,image_graph=self.image_graph,n_iterations=n_iterations)
        tetrahedrization_topomesh_remove_exterior(self.optimized_delaunay_topomesh)
        clean_tetrahedrization(self.optimized_delaunay_topomesh,min_cell_neighbors=2)
        self.triangulation_topomesh = deepcopy(self.optimized_delaunay_topomesh)


    def layer_adjacency_complex(self, layer_name='L1', omega_criteria = {'distance':1.0,'wall_surface':2.0,'clique':10.0}):
        """Estimate a 2-D adjacency complex of a single cell layer by optimal simplex aggregation.

        Args:
            layer_name (str): the cell layer considered ('L1' or 'L2')
            omega_criteria (dict): weights of the criteria used to compute the weights of triangles in the aggregation process :
                • 'distance' : penalize distant cells
                • 'wall_surface' : give weight to large cell interfaces
                • 'clique' : process uncertain configurations first

        Updates:
            triangulation_topomesh (PropertyTopomesh) : the DracoMesh adjacency complex is set to this constructed complex of degree 2.

        Returns:
            None
        """
        layer_triangulation_topomesh = layer_triangle_topomesh_construction(self.layer_edge_topomesh[layer_name], self.positions, omega_criteria=omega_criteria, wall_surfaces=self.image_wall_surfaces, cell_volumes=self.image_cell_volumes)
        self.layer_triangle_topomesh[layer_name] = deepcopy(layer_triangulation_topomesh)
        self.triangulation_topomesh = deepcopy(self.layer_triangle_topomesh[layer_name])


    def construct_adjacency_complex(self, omega_criteria = {'distance':1.0,'wall_surface':2.0,'clique':10.0}):
        """Estimate a layered adjacency complex containing L1 and L2 cells by optimal simplex aggregation.

        !!ONLY IMPLEMENTED FOR THE L1_L2 ADJACENCY LAYER!! **TO BE COMPLETED**

        Args:
            omega_criteria (dict): weights of the criteria used to compute the weights of triangles in the aggregation process :
                • 'distance' : penalize distant cells
                • 'wall_surface' : give weight to large cell interfaces
                • 'clique' : process uncertain configurations first
        
        Updates:
            triangulation_topomesh (PropertyTopomesh) : the DracoMesh adjacency complex is set to this constructed complex of degree 2.

        Returns:
            None
        """

        constructed_triangulation_topomesh = layered_tetrahedra_topomesh_construction(self.layer_triangle_topomesh['L1_L2'], self.positions, self.cell_layer, omega_criteria=omega_criteria, wall_surfaces=self.image_wall_surfaces, cell_volumes=self.image_cell_volumes)
        self.triangulation_topomesh = deepcopy(constructed_triangulation_topomesh)


    def mesh_image_surface(self, layers=[], voxelsize = 8):
        """Compute a surface mesh of the tissue object in the image.

        Args:
            layers (list): values of the cell layers to consider when computing the surface (1, 2 or 0)
            voxelsize (int): sampling step of the produced mesh
        
        Returns:
            surface_topomesh (PropertyTopomesh): triangular mesh with no cell information representing the surface of the tissue
        """

        grid_voxelsize = [voxelsize, voxelsize, voxelsize]
        binary_img = np.zeros(tuple(np.array(self.size*2,int)),np.uint8)
        if len(layers) == 0:
            binary_img[self.size[0]/2:3*self.size[0]/2,self.size[1]/2:3*self.size[1]/2,self.size[2]/2:3*self.size[2]/2][self.segmented_image>1] = 1
        else:
            layer_img = deepcopy(self.segmented_image)
            layer_img[(np.any(np.array([self.cell_layer.values(self.segmented_image) == l for l in layers]),axis=0))|(self.segmented_image==1)] = 1
            binary_img[self.size[0]/2:3*self.size[0]/2,self.size[1]/2:3*self.size[1]/2,self.size[2]/2:3*self.size[2]/2][layer_img>1] = 1

        binary_img = binary_img[0:binary_img.shape[0]:grid_voxelsize[0],0:binary_img.shape[1]:grid_voxelsize[1],0:binary_img.shape[2]:grid_voxelsize[2]]

        from openalea.mesh.utils.implicit_surfaces import implicit_surface_topomesh
        self.surface_topomesh = implicit_surface_topomesh(binary_img,binary_img.shape,self.voxelsize*grid_voxelsize)
        self.surface_topomesh.update_wisp_property('barycenter',0,self.surface_topomesh.wisp_property('barycenter',0).values()+np.array(binary_img.shape)*self.voxelsize*grid_voxelsize/4.)

        return self.surface_topomesh


    def dual_reconstruction(self, reconstruction_triangulation=None, adjacency_complex_degree=3, cell_vertex_constraint=True, maximal_edge_length=None):
        """Compute the dual geometry of the DRACO adjacency complex as a 3D interface mesh.

        Several options are possible for the interface triangulation:
            - 'star' : a vertex is inserted at the center of the interface, creating star-arranged triangles
            - 'delaunay' : the interface polygon is triangulated using Delaunay (!!ASSUMES INTERFACE IS CONVEX!!)
            - 'split' : all the triangles are split into 4 new triangles with vertices inserted at the middle of the edges
            - 'remeshed' : an isotropic remeshing algorithm is performed on the whole mesh
            - 'regular' : optimize the quality of the triangles as much as possible (!!CELL SHAPES WON'T BE PRESERVED!!)
            - 'realistic' : optimize the quality of the triangles while keeping plausible cell shapes (STEM optimization)
            - 'projected' : project the exterior mesh vertices onto the actual object surface
            - 'flat' : flatten all cell interfaces by projecting vertices on the interface median plane (3D complex required)
            - 'straight' : straighten all cell boundaries by local laplacian operator (best for 2D complex)
            - 'exact' : ensure cell vertices are well preserved during all geometrical optimization processes
            !!IF "reconstruction_triangulation" IS SET AS EMPTY A POLYGONAL INTERFACE MESH WILL BE RETURNED!!

        Args:
            reconstruction_triangulation (list): parameters of the interface triangulation method :
                • initial tirangulation (mandatory) : 'star' or 'delaunay'
                • triangulation refinement (optional) : 'split' or 'remeshed'
                • geometrical optimization (optional, multiple choice) : 'regular', 'realistic', 'projected', 'flat', 'straight'

            adjacency_complex_degree (int): 2 or 3, whether the adjacency complex is made of single layer triangles or tetrahedra
            cell_vertex_constraint (bool): whether the cell corners should be constrained to their position in the image or not
            maximal_edge_length (float): in micrometers, the maximal length for the remeshing algorithm

        Returns:
            dual_reconstruction_topomesh (PropertyTopomesh): triangular mesh representing the cell geometry
        """
        
        if reconstruction_triangulation is not None:
            self.reconstruction_triangulation = reconstruction_triangulation

        self.mesh_image_surface()

        if adjacency_complex_degree == 3:
            tetrahedrization_topomesh_add_exterior(self.triangulation_topomesh)
        elif adjacency_complex_degree == 2:
            tetrahedrization_topomesh_remove_exterior(self.triangulation_topomesh)
            if not self.triangulation_topomesh.has_wisp(3,1):
                self.triangulation_topomesh.add_wisp(3,1)
                for t in self.triangulation_topomesh.wisps(2):
                    self.triangulation_topomesh.link(3,1,t)
            tetrahedrization_topomesh_add_exterior(self.triangulation_topomesh)
            if self.triangulation_topomesh.has_wisp(3,1):
                self.triangulation_topomesh.remove_wisp(3,1)

        self.dual_reconstruction_topomesh = tetrahedra_dual_triangular_topomesh(self.triangulation_topomesh,triangular=self.reconstruction_triangulation,image_cell_vertex=self.image_cell_vertex, voronoi=True, vertex_motion=cell_vertex_constraint, surface_topomesh=self.surface_topomesh, maximal_length=maximal_edge_length)
        
        if adjacency_complex_degree == 3:
            tetrahedrization_topomesh_remove_exterior(self.triangulation_topomesh)
        
        return self.dual_reconstruction_topomesh


    def adjacency_complex_topomesh(self):
        """
        Returns the DRACO adjacency complex as a PropertyTopomesh
        """
        return self.triangulation_topomesh


    def dual_reconstructed_topomesh(self):
        """
        Returns the DRACO dual reconstruction as a PropertyTopomesh
        """
        if self.dual_reconstruction_topomesh is not None:
            return self.dual_reconstruction_topomesh
        else:
            return self.dual_reconstruction()

    def draco_topomesh(self, n_iterations=3, reconstruction_triangulation=None):
        """
        Compute the dual reconstruction following the standard DRACO algorithm 
        1/ Compute Delaunay adjacency complex with default surface carving parameters
        2/ Perform the adjacency complex optimization with default energy weights
        3/ Return the dual reconstruction with the initial triangulation settings
        """
        self.delaunay_adjacency_complex()
        if n_iterations>0:
            self.adjacency_complex_optimization(n_iterations=3)
        return self.dual_reconstruction(reconstruction_triangulation)


def create_draco_topomesh(image, n_iterations=3, reconstruction_triangulation=None):
    """DRACO - Dual Reconstruction by Adjacency Complex Optimization.

    Generate a PropertyTopomesh representing the cell tissue contained in a
    segmented images by the dualization of an optimized adjacency complex.

    Args:
        image (:class:`openalea.image.SpatialImage`):
            A segmented 3D image stack where tissue cells are represented as
            connected labelled regions. Label 1 cooresponds to the exterior.
        n_iterations (int):
            The number of optimization passes on the adjacency complex.
        reconstruction_triangulation (str, *optional*):
            A string (or list of strings) containing the options for the interface triangulation among:
                * *star* : a vertex is inserted at the center of the interface, creating star-arranged triangles
                * *delaunay* : the interface polygon is triangulated using Delaunay (!!ASSUMES INTERFACE IS CONVEX!!)
                * *split* : all the triangles are split into 4 new triangles with vertices inserted at the middle of the edges
                * *remeshed* : an isotropic remeshing algorithm is performed on the whole mesh
                * *regular* : optimize the quality of the triangles as much as possible (!!CELL SHAPES WON'T BE PRESERVED!!)
                * *realistic* : optimize the quality of the triangles while keeping plausible cell shapes (STEM optimization)
                * *projected* : project the exterior mesh vertices onto the actual object surface
                * *flat* : flatten all cell interfaces by projecting vertices on the interface median plane (3D complex required)
                * *straight* : straighten all cell boundaries by local laplacian operator (best for 2D complex)
                * *exact* : ensure cell vertices are well preserved during all geometrical optimization processes

    Returns:
        dual_reconstruction_topomesh (:class:`openalea.mesh.PropertyTopomesh`):
            The PropertyTopomesh representing the geometry of the tissue.

    """
    draco = DracoMesh(image)
    return draco.draco_topomesh(n_iterations, reconstruction_triangulation)


def draco_initialization(image=None, image_file=None, cell_vertex_file=None):
    """Initialize the DRACO object by providing a segmented image.

        Image can be passed either as an object of a filename. Cell adjacency information will be extracted from image 
        at initialization, still, some time can always be saved: if previously extracted, it can be read from existing
        file; if not the extrcted information will be saved in the specified files.

        Args:
            image (:class:`openalea.image.SpatialImage`): 
                A (non-eroded) segmented label image.
            image_file (str): 
                A valid path to an image of readable type (.inr, .inr.gz, .tiff...).
            image_cell_vertex_file (str): 
                File to read from if cell-vertices have already been extracted.
            triangulation_file (str)
                File to load from if an already existing adjacency complex is to be used.
            reconstruction_triangulation (str): 
                Default values for dual reconstruction triangulation (see draco_dual_reconstruction for more details)

        Returns:
            draco (:class:`openalea.draco_stem.draco.DracoMesh`):
                A DracoMesh containing adjacency information on the image cells.
        """
    return DracoMesh(image, image_file, cell_vertex_file)


def draco_delaunay_adjacency_complex(input_draco, surface_cleaning_criteria = ['surface','exterior','distance','sliver']):
    draco = deepcopy(input_draco)
    draco.delaunay_adjacency_complex(surface_cleaning_criteria)
    return draco


def draco_layer_adjacency_complex(input_draco, layer_name='L1', omega_criteria = {'distance':1.0,'wall_surface':2.0,'clique':10.0}):
    draco = deepcopy(input_draco)
    draco.layer_adjacency_complex(layer_name, omega_criteria)
    return draco


def draco_construct_adjacency_complex(input_draco, omega_criteria = {'distance':1.0,'wall_surface':2.0,'clique':10.0}):
    draco = deepcopy(input_draco)
    draco.construct_adjacency_complex(omega_criteria)
    return draco


def draco_adjacency_complex_optimization(input_draco, n_iterations = 1, omega_energies = {'image':10.0,'geometry':0.1,'adjacency':0.01}):
    draco = deepcopy(input_draco)
    draco.adjacency_complex_optimization(n_iterations, omega_energies)
    return draco


def draco_dual_reconstruction(input_draco, reconstruction_triangulation=None, adjacency_complex_degree=3, cell_vertex_constraint=True, maximal_edge_length=None):
    draco = deepcopy(input_draco)
    draco.dual_reconstruction(reconstruction_triangulation, adjacency_complex_degree, cell_vertex_constraint, maximal_edge_length)
    return draco

def draco_segmented_image(draco):
    return draco.segmented_image








    



