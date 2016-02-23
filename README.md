# OpenAlea Meshing Libraries

Data structures, algorithms and generation pipelines for meshes in OpenAlea

## Contents

### CellComplex

Package implementing data structures for the representation of cellular complexes. The main structure is an implementation of Incidence Graphs in a class named **PropertyTopomesh**.

####Dependencies : `openalea`, `numpy`, `scipy`, `openalea-container`

### Mesh-OALab

Plugins and visual components for the integration of mesh structures in TissueLab

####Dependencies : `cellcomplex`, `vtk`, `tissuelab`

### DRACO-STEM

Generating high-quality meshes of cell tissue from 3D segmented images :
* **D**ual **R**econstruction by **A**djacency **C**omplex **O**ptimization
* **S**AM **T**issue **E**nhanced **M**esh

####Dependencies : `cellcomplex`, `openalea-image`, `tissue-analysis`

### CGAL-Meshing

Generating coarse meshes of cell tissue from 3D segmented images, and enhance them using the STEM functionalities

####Dependencies : `cellcomplex`, `CGAL`, `openalea-image`, `tissue-analysis`
