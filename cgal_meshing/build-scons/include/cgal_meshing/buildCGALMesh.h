#ifndef __BUILD_MESH_H__
#define __BUILD_MESH_H__

#ifndef _MSC_VER
#ifdef NDEBUG
#warning CGAL subdomain labeling in 3D mesh generation does not work correctly with gcc 4.4, sconsX, and if NDEBUG is defined
#undef NDEBUG
#endif
#endif
/*
#define CGAL_MESH_3_NO_DEPRECATED_SURFACE_INDEX
#define CGAL_MESH_3_NO_DEPRECATED_C3T3_ITERATORS
#define CGAL_USE_ZLIB
 */
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Simple_cartesian.h>

#include <CGAL/Mesh_triangulation_3.h>
#include <CGAL/Mesh_complex_3_in_triangulation_3.h>
#include <CGAL/Mesh_criteria_3.h>

#include <CGAL/Labeled_image_mesh_domain_3.h>
#include <CGAL/make_mesh_3.h>
#include <CGAL/Image_3.h>
#include <CGAL/Triangle_3.h>


#include <CGAL/ImageIO.h>

#include <iostream>
#include <string>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <time.h>



// Domain
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
//typedef CGAL::Simple_cartesian<double> K;

typedef CGAL::Labeled_image_mesh_domain_3<CGAL::Image_3,K> Mesh_domain;

// Triangulation
typedef CGAL::Triangle_3<K> Triangle;
typedef CGAL::Mesh_triangulation_3<Mesh_domain>::type Tr;
typedef CGAL::Mesh_complex_3_in_triangulation_3<Tr> C3t3;

// Criteria
typedef CGAL::Mesh_criteria_3<Tr> Mesh_criteria;

// To avoid verbose function and named parameters call
using namespace CGAL::parameters;

void buildCGALMesh(char* filename, char* outputfilename, double f_a, double f_s, double f_d, double c_r_r, double c_s);

#endif