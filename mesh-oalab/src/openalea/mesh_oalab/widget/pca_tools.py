# -*- coding: utf-8 -*-
# -*- python -*-
#
#       Meshing
#
#       Copyright 2015 INRIA - CIRAD - INRA
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

from scipy.cluster.vq                       import kmeans, vq

from openalea.container.array_dict             import array_dict

from sklearn.decomposition import PCA
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import matplotlib
matplotlib.use( "MacOSX" )
import matplotlib.pyplot as plt

from vplants.meshing.cute_plot              import simple_plot, density_plot, smooth_plot, histo_plot, bar_plot, violin_plot, spider_plot

import matplotlib.font_manager as fm
import matplotlib.patches as patch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

try:
    from sklearn.decomposition import PCA
    from sklearn.cross_validation import train_test_split
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
except:
    print "SciKit-Learn should be installed to use PCA analysis tools!"

    def pca_analysis(**kwargs):
        return None,None

else:

    def pca_analysis(data,data_classes,class_colors,class_labels=None,features=None,n_components=2,linewidth=1,marker_size=50,alpha=1,**kwargs):
        if data_classes is None:
            data_classes = np.zeros(data.shape[0],np.uint8)

        classes = np.unique(data_classes)
        n_classes = len(classes)
        print classes

        if np.array(class_colors).ndim == 1:
            class_colors = np.array([class_colors for c in classes])

        if class_labels is None:
            class_labels = classes

        class_effectives = dict(zip(classes,[np.sum(data_classes == c) for c in classes]))
            
        # classification_data = np.concatenate([data_classes[:,np.newaxis],data],axis=1)
        classification_data = data

        if features == None:
            features = np.array(["class"]+["feature "+str(f+1) for f in xrange(data.shape[1])])
        else:
            features = np.concatenate([["class"],features])

        balance_classes = kwargs.get('balance_classes',False)
        if balance_classes:
            balanced_classes_selection = np.array([np.random.rand()<=(class_effectives.values().min()/float(class_effectives[c])) for c in data_classes])
            classification_data = classification_data[balanced_classes_selection]
            data_classes = data_classes[balanced_classes_selection]
            class_effectives = dict(zip(classes,[np.sum(data_classes == c) for c in classes]))

        features_mean = np.mean(classification_data,axis=0)
        features_std = np.std(classification_data,axis=0)

        centered_data = (classification_data - features_mean)
        center_reduced_data = (classification_data - features_mean)/features_std
        center_reduced_data[np.isnan(center_reduced_data)] = 0.

        pca_data = center_reduced_data

        pca = PCA(n_components=n_components)
        pca.fit(pca_data)

        print "  --> Principal Component Analysis on",n_components,"components : ",pca.explained_variance_ratio_
        projected_data = pca.transform(pca_data)

        from vplants.plantgl.ext.color import GlasbeyMap, CurvMap

        #cmap = kwargs.get("colormap",GlasbeyMap(0,255))
        #cmap = kwargs.get("colormap",CurvMap(classes.min(),classes.max() if classes.max()>classes.min() else classes.max()+1))

        classes_to_display = kwargs.get("classes_to_display",classes)

        data_axes = np.diagflat(np.ones(pca_data.shape[1]))
        projected_axes = pca.transform(data_axes)

        # cmap = CurvMap(classes.min(),classes.max())

        pca_figure = kwargs.get('pca_figure',plt.figure("Data Points"))
        pca_figure.clf()
        pca_figure.patch.set_facecolor('white')
        # font = fm.FontProperties(family = 'CenturyGothic',fname = '/Library/Fonts/Microsoft/Century Gothic', weight ='light')
        font = fm.FontProperties(family = 'Trebuchet', weight ='light')
        pca_figure.patch.set_facecolor('white')

        draw_points = kwargs.get('draw_points',100)
        draw_classes = kwargs.get('draw_classes',True)
        draw_ellipses = kwargs.get('draw_ellipses',draw_classes)
        fireworks = kwargs.get('fireworks',draw_ellipses)


        axes = pca_figure.add_subplot(111)
        for i_class,c in enumerate(classes_to_display):
            if class_effectives[c]>1:

                # class_color = np.array(cmap(c%254+1).i3tuple())/255.
                # class_color = np.array(cmap(c).i3tuple())/255.
                class_color = class_colors[i_class]
                class_center = projected_data[np.where(data_classes==c)].mean(axis=0)



                _,figure_labels = axes.get_legend_handles_labels()
                print class_labels, figure_labels
                plot_label = str(class_labels[i_class]) if str(class_labels[i_class]) not in figure_labels else None

                if draw_points > 0:
                    points_to_draw = np.where(np.random.rand(class_effectives[c]) <= draw_points/100.)
                    axes.scatter(projected_data[:,0][np.where(data_classes==c)][points_to_draw],projected_data[:,1][np.where(data_classes==c)][points_to_draw],c=class_color,linewidth=linewidth,s=marker_size,alpha=alpha,label=plot_label)
                    if draw_classes and fireworks:
                        [axes.plot([p[0],class_center[0]],[p[1],class_center[1]],color=class_color,alpha=alpha/5.,linewidth=linewidth) for p in projected_data[np.where(data_classes==c)][points_to_draw]]
                
        for i_class,c in enumerate(classes_to_display):
            if class_effectives[c]>1:
                # class_color = np.array(cmap(c%254+1).i3tuple())/255.
                # class_color = np.array(cmap(c).i3tuple())/255.
                class_color = class_colors[i_class]
                class_center = projected_data[np.where(data_classes==c)].mean(axis=0)
                if draw_classes:

                    if draw_ellipses:
                        class_covariance = np.cov(projected_data[np.where(data_classes==c)],rowvar=False)
                        vals, vecs = np.linalg.eigh(class_covariance)
                        order = vals.argsort()[::-1]
                        vals = vals[order]
                        vecs = vecs[:,order]
                        theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))
                        width, height = 2.*np.sqrt(2)*np.sqrt(vals)
                        class_ellipse = patch.Ellipse(xy=class_center, width=width, height=height, angle=theta,color=class_color,alpha=alpha/10., linewidth=2.*linewidth)
                        
                        axes.add_patch(class_ellipse)

                    axes.scatter([class_center[0]],[class_center[1]],marker=u'o',c=class_color,s=2.*marker_size,linewidth=1.5*linewidth,alpha=(alpha+1.)/2.)
                    # axes.text(class_center[0], class_center[1],str(c), size=20, ha="center", va="center",bbox = dict(boxstyle="round",ec=class_color,fc=0.75+class_color/4.))


        # x_min,x_max = np.percentile(projected_data[:,0],5)-4,np.percentile(projected_data[:,0],95)+4
        # y_min,y_max = np.percentile(projected_data[:,1],5)-4,np.percentile(projected_data[:,1],95)+4


        x_min,x_max = np.percentile(projected_data[:,0],1)-4,np.percentile(projected_data[:,0],99)+4
        y_min,y_max = np.percentile(projected_data[:,1],1)-4,np.percentile(projected_data[:,1],99)+4
        # xx, yy = np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
        # # proba = gnb.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
        # proba = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
        # proba = proba.reshape(xx.shape)
        # my_color_map = matplotlib.colors.LinearSegmentedColormap.from_list('RdGr',['#ff0000','#ffffff','#00ff00'])
        # axes.contourf(xx, yy, proba, cmap=my_color_map, alpha=.25)

        axes.set_xlim(x_min,x_max)
        axes.set_xlabel("PCA Component 1",fontproperties=font, size=10, style='italic')
        axes.set_xticklabels(axes.get_xticks(),fontproperties=font, size=12)
        axes.set_ylim(y_min,y_max )
        axes.set_ylabel("PCA Component 2", fontproperties=font, size=10, style='italic')
        axes.set_yticklabels(axes.get_yticks(),fontproperties=font, size=12)
        # axes.axis('equal')
        # plt.show()

        # correlation_figure = kwargs.get('correlation_figure',plt.figure("PCA Axes"))
        # correlation_figure.clf()
        # correlation_figure.patch.set_facecolor('white')
        # axes = correlation_figure.add_subplot(111)
        axins = zoomed_inset_axes(axes,2.)
        circle = patch.Circle((0, 0), 1, facecolor='none',edgecolor=(0.8,0.8,0.8), linewidth=3, alpha=0.5)
        axins.add_patch(circle)
        for i,a in enumerate(projected_axes):
            axins.text(a[0], a[1],features[1:][i], size=10, ha="center", va="center",bbox = dict(boxstyle="round",ec=(0.1,0.1,0.1),fc=(0.5,0.5,0.5),alpha=0.2))
            # axins.plot([0,a[0]],[0,a[1]],color=np.array(cmap(24%254+1).i3tuple())/255.,linewidth=2.*np.linalg.norm(a),alpha=0.2)
            axins.plot([0,a[0]],[0,a[1]],color=(0.5,0.5,0.5),linewidth=2.*np.linalg.norm(a),alpha=0.2)
        axins.set_xlim(-1,1)
        axins.set_xlabel("")
        axins.set_xticklabels(["" for t in axins.get_xticks()])
        axins.set_ylim(-1,1)
        axins.axis('equal')
        axins.set_ylabel("")
        axins.set_yticklabels(["" for t in axins.get_yticks()])

        pca_figure.sca(axes)

            # plt.show(block=False)

        # projected_data_train, projected_data_test, classes_train, classes_test = train_test_split(projected_data,classes,test_size=.33)

        # gnb = GaussianNB()
        # classifier = gnb.fit(projected_data_train,classes_train)
        # predicitons_train = classifier.predict(projected_data_train)
        # predicitons_test = classifier.predict(projected_data_test)
        # good_classification_train = (classes_train == predicitons_train).sum()/float(classes_train.shape[0])
        # good_classification_test = (classes_test == predicitons_test).sum()/float(classes_test.shape[0])
        # random_classification_test = (classes_test == (np.random.rand(classes_test.shape[0])>0.5)).sum()/float(classes_test.shape[0])
        # print "  --> Naive Bayes on "+n_components+" components : ",good_classification_test,"( ",good_classification_train,") good classification    (Random :",random_classification_test,")"

        # svm = SVC(kernel="linear", C=0.025)
        # classifier = svm.fit(projected_data_train,classes_train)
        # predicitons_train = classifier.predict(projected_data_train)
        # predicitons_test = classifier.predict(projected_data_test)
        # good_classification_train = (classes_train == predicitons_train).sum()/float(classes_train.shape[0])
        # good_classification_test = (classes_test == predicitons_test).sum()/float(classes_test.shape[0])
        # random_classification_test = (classes_test == (np.random.rand(classes_test.shape[0])>0.5)).sum()/float(classes_test.shape[0])
        # print "  --> Linear SVM on "+n_components+" components : ",good_classification_test,"( ",good_classification_train,") good classification    (Random :",random_classification_test,")"


        return pca, projected_data
