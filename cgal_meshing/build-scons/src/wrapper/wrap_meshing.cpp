#include "export_mesh.h"

#include <boost/python.hpp>
#include <boost/version.hpp>

using namespace boost::python;


BOOST_PYTHON_MODULE(_cgal_meshing)
{
	class_mesh();
}