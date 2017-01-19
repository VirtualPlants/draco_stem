import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

quality_data = pd.read_csv('/Users/gcerutti/Desktop/DRACO-STEM/draco_stem_quaity.csv',delimiter=';')
world.add(quality_data,'quality')

color_points = {}
color_points[0] = [248,218,217]
color_points[0.5] = [255,255,255]
color_points[1] = [227,240,223]

color_dict = dict(red=[],green=[],blue=[])
for p in np.sort(color_points.keys()):
    for k,c in enumerate(['red','green','blue']):
        color_dict[c] += [(p,color_points[p][k]/255.,color_points[p][k]/255.)]
for c in ['red','green','blue']:
    color_dict[c] = tuple(color_dict[c])
# print color_dict
import matplotlib as mpl
mpl_cmap = mpl.colors.LinearSegmentedColormap('RdGn', color_dict)

methods = ['Idra','Idra-Stem','Voronoi','Draco','Draco-Stem+']
estimators = ['Cell Convexity','Epidermis Cell Angle','Cell Cliques','Image Accuracy','Vertex Distance','Cell 2 Adjacency','Triangle Area Deviation','Triangle Eccentricity','Vertex Valence','Mesh Complexity']

quality_data['Mean'] = np.mean([quality_data[e] for e in estimators],axis=0)
quality_data['Min'] = np.min([quality_data[e] for e in estimators],axis=0)

estimators = estimators+['Mean','Min']

mean_quality = np.zeros((len(methods),len(estimators)))
for i_m,m in enumerate(methods):
    for i_e,e in enumerate(estimators):
        mean_quality[i_m,i_e] = quality_data[quality_data['Method']==m].mean()[e]


figure = plt.figure(1)
figure.clf()
figure.gca().imshow(mean_quality,cmap=mpl_cmap,vmin=0.5,vmax=1.0,interpolation='nearest')
for i_m,m in enumerate(methods):
    for i_e,e in enumerate(estimators):
        std_quality = quality_data[quality_data['Method']==m].std()[e]
        figure.gca().text(i_e,i_m,"$\mathrm{"+str(np.round(mean_quality[i_m,i_e],3))+"}$",horizontalalignment='center',size=40,alpha=1 - 8.*std_quality)
        figure.gca().text(i_e-0.1,i_m+0.2,"$\pm \mathrm{"+str(np.round(std_quality,3))+"}$",horizontalalignment='left',size=24,alpha=1 - 8.*std_quality)

figure.gca().set_yticks(np.arange(len(methods)))
figure.gca().set_yticklabels(methods)
figure.gca().set_xticks(np.arange(len(estimators)))
figure.gca().set_xticklabels(estimators,rotation='vertical')
figure.set_size_inches(32,24)
figure.gca().axis('off')
figure.savefig('/Users/gcerutti/Desktop/DRACO-STEM/Table.png')



