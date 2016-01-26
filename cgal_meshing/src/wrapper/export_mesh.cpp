#include "export_mesh.h"


void class_mesh()
{
	Py_Initialize();
	//import_array()

	def("buildCGALMesh",&buildCGALMesh);
}