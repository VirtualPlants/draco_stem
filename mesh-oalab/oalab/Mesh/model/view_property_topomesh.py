import openalea.mesh

from openalea.mesh import PropertyTopomesh
from openalea.mesh.property_topomesh_io import read_ply_property_topomesh
from openalea.mesh.property_topomesh_analysis import compute_topomesh_property

from openalea.deploy.shared_data import shared_data
dirname = shared_data(openalea.mesh)
filename = dirname + "/p194-t4_L1_topomesh.ply"

topomesh = read_ply_property_topomesh(filename)
world.add(topomesh, "topomesh")
