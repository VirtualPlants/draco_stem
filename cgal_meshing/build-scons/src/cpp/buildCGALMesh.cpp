/* -*-c++-*- 
 *------------------------------------------------------------------------------
 *                                                                              
 *        vplant.vfruit: 3D virtual fruit package                                     
 *                                                                              
 *        Copyright 2013 INRIA - CIRAD - INRA                      
 *                                                                              
 *        File author(s): Mik Cieslak
 *                        Guillaume Cerutti
 *                                                                              
 *        Distributed under the Cecill-C License.                               
 *        See accompanying file LICENSE.txt or copy at                          
 *            http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html       
 *                                                                              
 *        OpenAlea WebSite : http://openalea.gforge.inria.fr                    
 *       
 *                                                                       
 *-----------------------------------------------------------------------------*/

#include "buildCGALMesh.h"

using namespace std;

void buildCGALMesh(char* filename, char* outputfilename, double f_a, double f_s, double f_d, double c_r_r, double c_s)
{

  // Load image
  CGAL::Image_3 image;
  clock_t start_time, end_time;

  start_time = clock();
  cerr <<"--> Reading Image"<<endl;
  image.read(filename);
  cout<<image.xdim()<<" "<<image.ydim()<<" "<<image.zdim()<<endl;
  end_time = clock();
  cerr <<"<-- Reading Image            [ "<<setprecision(10)<<(float)(end_time - start_time)/CLOCKS_PER_SEC<<" s]"<<endl;

  // Domain
  Mesh_domain domain(image);

  // Mesh criteria
  Mesh_criteria criteria (facet_angle=f_a,
                          facet_size=f_s,
                          facet_distance=f_d,
                          cell_radius_edge_ratio=c_r_r,
                          cell_size=c_s);


  C3t3 c3t3;

  // cerr<< "--> Initial Points"<<endl;
  // vector<vector<vector<int> > > surface_points(256);

  // vector<int> count_components(256);

  // for (int x=0;x<image.xdim();x++)
  // {
  //   for (int y=0;y<image.ydim();y++)
  //   {
  //     for (int z=0;z<image.zdim();z++)
  //     {
  //       unsigned char label = CGAL::IMAGEIO::static_evaluate<unsigned char>(image.image(),x,y,z);
  //       //cout<<"Voxel "<<x<<";"<<y<<";"<<z<<" : "<<label<<endl;
  //       if (label != 0)
  //       {
  //         count_components[label]++;

  //         if (count_components[label]%1000 == 0)
  //         {
  //           C3t3::Vertex_handle v = c3t3.triangulation().insert(Mesh_domain::Point_3(x,y,z));
  //           if ( v != C3t3::Vertex_handle() )
  //           {
  //             c3t3.set_dimension(v, 3);
  //             c3t3.set_index(v, label);
  //           }
  //         }

  //         bool surface = false;
  //         if ((!surface)&&(x<image.xdim()-1)) surface = (CGAL::IMAGEIO::static_evaluate<unsigned char>(image.image(),x+1,y,z) != label);
  //         if ((!surface)&&(x>0))              surface = (CGAL::IMAGEIO::static_evaluate<unsigned char>(image.image(),x-1,y,z) != label);
  //         if ((!surface)&&(y<image.ydim()-1)) surface = (CGAL::IMAGEIO::static_evaluate<unsigned char>(image.image(),x,y+1,z) != label);
  //         if ((!surface)&&(y>0))              surface = (CGAL::IMAGEIO::static_evaluate<unsigned char>(image.image(),x,y-1,z) != label);
  //         if ((!surface)&&(z<image.zdim()-1)) surface = (CGAL::IMAGEIO::static_evaluate<unsigned char>(image.image(),x,y,z+1) != label);
  //         if ((!surface)&&(z>0))              surface = (CGAL::IMAGEIO::static_evaluate<unsigned char>(image.image(),x,y,z-1) != label);

  //         if (surface)
  //         {
  //           vector<int> point(3);
  //           point[0] = x;
  //           point[1] = y;
  //           point[2] = z;
  //           surface_points[label].push_back(point);
  //         }
  //       }
  //     }
  //   }
  // }


  // for (int c=1;c<256;c++)
  // {
  //     if (count_components[c]>0)
  //     {
  //         /*int point_index_prev = -1;

  //         for (int p=0;p<100;p++)
  //         {
  //             int point_index = p*surface_points[c].size()/100;
  //             if (point_index != point_index_prev)
  //             {
  //                 C3t3::Vertex_handle v = c3t3.triangulation().insert(Mesh_domain::Point_3(surface_points[c][point_index][0],surface_points[c][point_index][1],surface_points[c][point_index][2]));
  //                 if ( v != C3t3::Vertex_handle() )
  //                 {
  //                   c3t3.set_dimension(v, 3);
  //                   c3t3.set_index(v, c);
  //                 }

  //                 point_index_prev = point_index;
  //             }
  //         }*/

  //         /*vector<double> centroid(3);
  //         for (int p=0;p<(int)surface_points[c].size();p++)
  //         {
  //           centroid[0] += surface_points[c][p][0];
  //           centroid[1] += surface_points[c][p][1];
  //           centroid[2] += surface_points[c][p][2];
  //         }
  //         centroid[0] = (int)(centroid[0]/surface_points[c].size());
  //         centroid[1] = (int)(centroid[1]/surface_points[c].size());
  //         centroid[2] = (int)(centroid[2]/surface_points[c].size());
  //         C3t3::Vertex_handle v = c3t3.triangulation().insert(Mesh_domain::Point_3(centroid[0],centroid[1],centroid[2]));
  //         if ( v != C3t3::Vertex_handle() )
  //         {
  //           c3t3.set_dimension(v, 3);
  //           c3t3.set_index(v, c);
  //         }*/

  //         cout<<"Component "<<c<<" : "<<count_components[c]<<" (Surface : "<<surface_points[c].size()<<")"<<endl;
  //     }
  // }

  // CGAL_assertion( c3t3.triangulation().dimension() == 3 );

  // ofstream init_file("test/initial.mesh");
  // c3t3.output_to_medit(init_file);

  // cerr<< "<-- Initial Points"<<endl;

  // Build mesher and launch refinement process
  // Don't reset c3t3 as we just created it

  // Meshing
  start_time = clock();
  cerr <<"--> Making Mesh"<<endl;
  c3t3 = CGAL::make_mesh_3<C3t3>(domain, criteria, no_perturb(), no_exude(), odt()); 
  //CGAL::refine_mesh_3<C3t3>(c3t3, domain, criteria, no_perturb(), no_exude(), odt(), no_reset_c3t3());   
  end_time = clock();
  cerr <<"<-- Making Mesh              [ "<<setprecision(10)<<(float)(end_time - start_time)/CLOCKS_PER_SEC<<" s]"<<endl;

  
  /*cerr << "--> Triangulation "<<endl;
  Tr triangulation = c3t3.triangulation(); 
  cerr << "<-- Triangulation "<<endl;

  int components = 0;

  for (Tr::Finite_cells_iterator it = triangulation.finite_cells_begin(); it != triangulation.finite_cells_end(); ++it)
  {
    int component = it->subdomain_index();
    components = component>components ? component : components;
  }
  components++;

  vector<int> count_tetrahedra(components);
  int n_cells = 0;
  for (Tr::Finite_cells_iterator it = triangulation.finite_cells_begin(); it != triangulation.finite_cells_end(); ++it)
  { 
    //cout<<"Cell #"<<n_cells<<" : Component "<<it->subdomain_index()<<endl;
    
    int component = (int) it->subdomain_index();
    if (component>0) n_cells++;
    count_tetrahedra[component] ++;
  }
  for (int c=0;c<components;c++)
  {
    if (count_tetrahedra[c]>0)
    {
      cout<<"Component "<<c<<" : "<<count_tetrahedra[c]<<" Cells"<<endl;
    }
  } 

  int n_vertices = 0;
  for (Tr::Finite_vertices_iterator it = triangulation.finite_vertices_begin(); it != triangulation.finite_vertices_end(); ++it)
  { 
    //cout<<"Vertex #"<<n_vertices<<" : "<<it->point()<<endl;
    n_vertices++;
  }
  cout<<n_vertices<<" Vertices, "<<n_cells<<" Cells"<<endl;
  getchar();

  vector<vector<int> > index_triangles;
  vector<vector<int> > count_triangles(components);

  n_cells = 0;
  int total_triangles = 0;

  for (Tr::Finite_cells_iterator it = triangulation.finite_cells_begin(); it != triangulation.finite_cells_end(); ++it)
  {
    //cout<<"Cell #"<<n_cells<<" : Component "<<it->subdomain_index()<<endl;
    //n_cells++;
    int component = (int) it->subdomain_index();

    if (component>0)
    //if (true)
    {
      for (int t=0;t<4;t++)
      {
        vector<int> indices(3);
        for (int v=0;v<3;v++)
        { 
          indices[v] = std::distance(triangulation.all_vertices_begin(), it->vertex((t+v)%4));
        }

        //cout<<"Triangle "<<n_cells<<"-"<<t<<" : ";
        //cout<<" [ "<<indices[0]<<" ; "<<indices[1]<<" ; "<<indices[2]<<" ]";

        bool deja_vu = false;
        int existing_triangle_index = -1;
        int triangle = 0;

        while ((!deja_vu)&&(triangle<index_triangles.size()))
        {
          for (int p=0;p<3;p++)
          {
            deja_vu = deja_vu || ((index_triangles[triangle][p%3]==indices[0])&&(index_triangles[triangle][(p+1)%3]==indices[1])&&(index_triangles[triangle][(p+2)%3]==indices[2]));
          }
          for (int p=0;p<3;p++)
          {
            deja_vu = deja_vu || ((index_triangles[triangle][p%3]==indices[0])&&(index_triangles[triangle][(p-1)%3]==indices[1])&&(index_triangles[triangle][(p-2)%3]==indices[2]));
          }

          if (deja_vu)
          {
              existing_triangle_index = triangle;
          }

          triangle++;
        }

        if (!deja_vu)
        {
          //cout<<" -> Triangle "<<index_triangles.size();
          //cout<<" [ "<<indices[0]<<" ; "<<indices[1]<<" ; "<<indices[2]<<" ]";
          //cout<<endl;

          index_triangles.push_back(indices);

          for (int c=0;c<components;c++)
          {
            if (c == component)
            {
              count_triangles[c].push_back(1);
            }
            else
            {
              count_triangles[c].push_back(0);
            }
          }
        }
        else
        {
          //cout<<" -> Existing Triangle "<<existing_triangle_index;
          count_triangles[component][existing_triangle_index]++;
        }
        //cout<<endl;
        total_triangles ++;
      }
    }
  }

  int n_triangles = 0;
  for (int triangle=0;triangle<index_triangles.size();triangle++)
  {
    for (int c=1;c<components;c++)
    {
      if (count_triangles[c][triangle] == 1)
      {
        n_triangles ++;
      }
    }
  }
  cout<<n_triangles<<"/"<<index_triangles.size()<<" ("<<count_triangles[0].size()<<") Triangles"<<endl;*/


  // Output
  ofstream medit_file(outputfilename);
  //ofstream medit_file("test/compare.mesh");
  c3t3.output_to_medit(medit_file);

  /*//ofstream mesh_file("test/test.mesh",ios::out|ios::trunc);
  ofstream mesh_file(outputfilename,ios::out|ios::trunc);
  mesh_file<<"MeshVersionFormatted 1"<<endl;
  mesh_file<<"Dimension 3"<<endl;
  mesh_file<<"Vertices"<<endl;
  mesh_file<<n_vertices<<endl;
  for (Tr::Finite_vertices_iterator it = triangulation.finite_vertices_begin(); it != triangulation.finite_vertices_end(); ++it)
  { 
    mesh_file<<it->point()[0]<<" "<<it->point()[1]<<" "<<it->point()[2]<<" "<<"1"<<endl;
  }
  mesh_file<<"Triangles"<<endl;
  mesh_file<<n_triangles<<endl;
  for (int triangle=0;triangle<index_triangles.size();triangle++)
  {
    for (int c=1;c<components;c++)
    {
      if (count_triangles[c][triangle] == 1)
      {
        mesh_file<<index_triangles[triangle][0]<<" "<<index_triangles[triangle][1]<<" "<<index_triangles[triangle][2]<<" "<<c<<endl;
      }
    }
  }
  mesh_file.close();*/


  return;
}

/*int main(int argc, char* argv[])
{
  double param[7];
  string str_perturb ("-no_perturb");
  string str_exude ("-no_exude");

  if (argc >= 8)
  {
    param[0] = atof(argv[3]);
    param[1] = atof(argv[4]);
    param[2] = atof(argv[5]);
    param[3] = atof(argv[6]);
    param[4] = atof(argv[7]);
    param[5] = 0.0;
    param[6] = 0.0;
    
    if (argc == 9)
    {
      if (str_perturb.compare(argv[8]) == 0)
        param[5] = 1.0;
      else if (str_exude.compare(argv[8]) == 0)
        param[6] = 1.0;
      else
        cerr << "Warning: unrecognized command " << argv[8] << " will be ignored.";
    }
    else if (argc == 10)
    {
      if (str_perturb.compare(argv[8]) == 0)
        param[5] = 1.0;
      else if (str_exude.compare(argv[8]) == 0)
        param[6] = 1.0;
      else
        cerr << "Warning: unrecognized command " << argv[8] << " will be ignored.";
      if (str_perturb.compare(argv[9]) == 0)
        param[5] = 1.0;
      else if (str_exude.compare(argv[9]) == 0)
        param[6] = 1.0;
      else
        cerr << "Warning: unrecognized command " << argv[9] << " will be ignored.";
    }
  }
  else
  {
    cerr << "Usage: " << argv[0] << " ";
    cerr << "input_file output_file facet_angle facet_size facet_distance cell_radius_edge cell_size ";
    cerr << "perturb exude" << endl;
    cerr << "e.g., " << argv[0] << " out.inr out.mesh 30 2.0 0.5 2.0 3.0 -no_perturb -no_exude" << endl;
    return (0);
  }


#ifndef _MSC_VER
#warning These parameter values must be printed otherwise there is a bug in CGAL for subdomain labeling with gcc 4.4 and sconsX
  cerr << "params: " << param[0] << ", " << param[1] << ", " << param[2] << ", " << param[3] << ", " << param[4] << ", " << param[5] << ", " << param[6] << endl;
#endif

  buildCGALMesh(argv[1],argv[2],param[0],param[1],param[2],param[3],param[4],param[5]==1,param[6]==1);

  return 0;
}*/
