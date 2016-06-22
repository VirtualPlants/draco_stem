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

import matplotlib
matplotlib.use( "MacOSX" )
#matplotlib.use( "cairo" )
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from matplotlib.patches                     import Rectangle

from scipy.cluster.vq                       import kmeans, vq
from scipy.interpolate                      import interp1d, spline, splrep, splev
from scipy.stats                            import gaussian_kde



def simple_plot(figure,X,Y,colors,xlabel="",ylabel="",linked=True,n_points=400,marker_size=10,linewidth=3,alpha=1.0,label=None):
    """
    """

    if np.array(colors).ndim == 1:
        colors = np.array([colors for i in xrange(len(X))])

    #font = fm.FontProperties(family = 'CenturyGothic',fname = '/Library/Fonts/Microsoft/Century Gothic', weight ='light')
    font = fm.FontProperties(family = 'Trebuchet', weight ='light')
    figure.patch.set_facecolor('white')
    axes = figure.add_subplot(111)

    if linked:
        X_smooth = np.linspace(X.min(),X.max(),n_points)
        interpolator = interp1d(X,Y)
        Y_smooth = np.array([interpolator(x) for x in X_smooth])

        color_interpolator = [interp1d(X,colors[:,k]) for k in [0,1,2]]
        colors_smooth = np.transpose([[color_interpolator[k](x) for x in X_smooth] for k in [0,1,2]])

        for i in xrange(100):
            # color = tuple(color1*(1.0-i/100.0) + color2*(i/100.0))
            color = colors_smooth[(i*n_points/100):((i+1)*n_points)/100+1].mean(axis=0)

            if i == 0:
                axes.plot(X_smooth[(i*n_points/100):((i+1)*n_points)/100+1],Y_smooth[(i*n_points/100):((i+1)*n_points)/100+1],linewidth=linewidth,color=color,alpha=alpha,label=label if marker_size==0 else None)
            else:
                axes.plot(X_smooth[(i*n_points/100):((i+1)*n_points)/100+1],Y_smooth[(i*n_points/100):((i+1)*n_points)/100+1],linewidth=linewidth,color=color,alpha=alpha)
        # axes.plot(X,Y,linewidth=linewidth,color=color,alpha=alpha,label=label)
    # ratios = (Y-Y.min())/(Y.max()-Y.min())
    axes.scatter(X,Y,c=colors,label=label if marker_size>0 else None,s=marker_size,alpha=alpha,linewidth=linewidth)
    # colors = 
    # for i in xrange(100):
    #   color = tuple(color1*(1.0-ratios[i]) + color2*ratios[i])
        
        # axes.plot(X_smooth[(i*n_points/100):((i+1)*n_points)/100+1],Y_smooth[(i*n_points/100):((i+1)*n_points)/100+1],linewidth=linewidth,color=color,alpha=1.0)
    axes.set_xlabel(xlabel,fontproperties=font, size=10, style='italic')
    # axes.set_xlim(X.min(),X.max())
    axes.set_xticklabels(axes.get_xticks(),fontproperties=font, size=12)
    axes.set_ylabel(ylabel, fontproperties=font, size=10, style='italic')
    # axes.set_ylim(Y.min(),Y.max())
    axes.set_yticklabels(axes.get_yticks(),fontproperties=font, size=12)


def surface_plot(figure,X,Y,Z,color1,color2=None,xlabel="",ylabel="",zlabel="",alpha=1.0,linewidth=3,label=""):
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib.colors import LinearSegmentedColormap

    if color2 is None:
        color2 = color1

    cdict = {'red':   ((0.0, color1[0], color1[0]),(1.0, color2[0], color2[0])),
             'green': ((0.0, color1[1], color1[1]),(1.0, color2[1], color2[1])),
             'blue':  ((0.0, color1[2], color1[2]),(1.0, color2[2], color2[2]))}

    cmap = LinearSegmentedColormap('CMap', cdict)


    font = fm.FontProperties(family = 'Trebuchet', weight ='light')
    figure.patch.set_facecolor('white')
    axes = figure.add_subplot(111,projection='3d')

    if X.ndim<2:
        X = np.tile(np.array(X),(Y.shape[-1],1)).transpose()
    if Y.ndim<2:
        Y = np.tile(np.array(Y),(X.shape[0],1))

    axes.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=cmap,alpha=alpha,linewidth=linewidth,label=label)

    axes.set_xlabel(xlabel,fontproperties=font, size=10, style='italic')
    axes.set_xlim(X.min(),X.max())
    axes.set_xticklabels(axes.get_xticks(),fontproperties=font, size=12)
    axes.set_ylabel(ylabel, fontproperties=font, size=10, style='italic')
    axes.set_ylim(Y.min(),Y.max())
    axes.set_yticklabels(axes.get_yticks(),fontproperties=font, size=12)
    axes.set_zlabel(zlabel, fontproperties=font, size=10, style='italic')
    axes.set_zlim(Z.min(),Z.max())
    axes.set_zticklabels(axes.get_zticks(),fontproperties=font, size=12)


def density_plot(figure,X,Y,color,xlabel="",ylabel="",n_points=10,linewidth=1,marker_size=40.,alpha=1.0,label=""):
    font = fm.FontProperties(family = 'Trebuchet', weight ='light')
    #font = fm.FontProperties(family = 'CenturyGothic',fname = '/Library/Fonts/Microsoft/Century Gothic', weight ='light')
    figure.patch.set_facecolor('white')
    axes = figure.add_subplot(111)
    # axes.plot(X,Y,linewidth=1,color=tuple(color2),alpha=0.2)
    # ratios = (Y-Y.min())/(Y.max()-Y.min())
    # X_min = X.mean()-3*X.std()
    # X_max = X.mean()+3*X.std()
    X_min = np.percentile(X,100/n_points)
    X_max = np.percentile(X,100 - 100/n_points)
    Y_min = np.percentile(Y,100/n_points)
    # Y_min = Y.mean()-3*Y.std()
    Y_max = np.percentile(Y,100 - 100/n_points)

    X_grid = np.linspace(X_min,X_max,n_points)
    Y_grid = np.linspace(Y_min,Y_max,n_points)

    X_sampled = X_grid[vq(X,X_grid)[0]]
    Y_sampled = Y_grid[vq(Y,Y_grid)[0]]

    point_density = {}
    for x in np.unique(X_sampled):
        point_count = nd.sum(np.ones_like(np.where(X_sampled==x)),Y_sampled[np.where(X_sampled==x)],index=np.unique(Y_sampled))
        for i,y in enumerate(np.unique(Y_sampled)):
            point_density[(x,y)] = point_count[i]/len(Y)

    point_area = np.array([np.pi*10.0*marker_size*point_density[(x,y)]/np.array(point_density.values()).max() for x,y in zip(X_sampled,Y_sampled)])
    #colors = np.random.rand(len(X))
    colors = np.array([point_density[(x,y)]/np.array(point_density.values()).max() * color for x,y in zip(X_sampled,Y_sampled)])
    colors += np.array([(1-point_density[(x,y)]/np.array(point_density.values()).max()) * np.ones(3) for x,y in zip(X_sampled,Y_sampled)])

    axes.scatter(X_sampled,Y_sampled,s=point_area,c=colors,linewidth=linewidth,alpha=alpha,label=label)
    axes.set_xlim(X_min,X_max)
    axes.set_xlabel(xlabel,fontproperties=font, size=10, style='italic')
    axes.set_xticklabels(axes.get_xticks(),fontproperties=font, size=12)
    axes.set_ylim(Y_min,Y_max)
    axes.set_ylabel(ylabel, fontproperties=font, size=10, style='italic')
    axes.set_yticklabels(axes.get_yticks(),fontproperties=font, size=12)


def density_contour_plot(figure,X,Y,color,XY_range=None,xlabel="",ylabel="",n_points=100,n_contours=10,smooth_factor=1.0,linewidth=1,marker_size=40.,alpha=1.0,label=""):
    font = fm.FontProperties(family = 'Trebuchet', weight ='light')
    figure.patch.set_facecolor('white')
    axes = figure.add_subplot(111)

    if XY_range is None:
        XY_range = [[X.min(),X.max()],[Y.min(),Y.max()]]

    range_x = np.linspace(XY_range[0][0],XY_range[0][1],n_points)
    range_y = np.linspace(XY_range[1][0],XY_range[1][1],n_points)
    xx, yy = np.meshgrid(range_x,range_y)

    range_x_cr = (range_x - range_x.mean())/range_x.std()
    range_y_cr = (range_y - range_y.mean())/range_y.std()
    xx_cr, yy_cr = np.meshgrid(range_x_cr,range_y_cr)


    def density_function(positions,radius=1,k=0.1):
        def density_func(x,y):
            points = np.array(positions.values())
           
            if len((x+y).shape) == 1:
                distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis],2),0.5)
            elif len((x+y).shape) == 2:
                distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis,np.newaxis],2),0.5)

            density_potential = 1./2. * (1. - np.tanh(k*(distances - radius)))
            density = density_potential.sum(axis=0)

            return density
        return density_func

    positions = dict(zip(range(len(X)),np.transpose([X,Y])))
    positions_cr = dict(zip(range(len(X)),np.transpose([(X-range_x.mean())/range_x.std(),(Y-range_y.mean())/range_y.std()])))
    
    # radius = np.sqrt((X.std() * Y.std()))
    radius = 10.0/n_points
    density_k = 100.*np.exp(-smooth_factor/2.)
    data_density = density_function(positions_cr,radius,density_k)(xx_cr,yy_cr)

    levels = np.linspace(0,data_density.max(),n_contours)
    colors = np.array([(lev*color + (n_contours-1-lev)*np.ones(3))/(n_contours-1) for lev in xrange(n_contours)])

    axes.scatter(X,Y,s=marker_size,c=color,linewidth=0,alpha=alpha/2.,label=label)
    axes.contour(xx,yy,data_density,linewidths=linewidth,levels=levels,colors=colors,alpha=(alpha+1)/2.)

    axes.set_xlim(*tuple(XY_range[0]))
    axes.set_xlabel(xlabel,fontproperties=font, size=10, style='italic')
    axes.set_xticklabels(axes.get_xticks(),fontproperties=font, size=12)
    axes.set_ylim(*tuple(XY_range[1]))
    axes.set_ylabel(ylabel, fontproperties=font, size=10, style='italic')
    axes.set_yticklabels(axes.get_yticks(),fontproperties=font, size=12)


def map_plot(figure,X,Y,Z,colormap,XY_range=None,xlabel="",ylabel="",n_points=100,n_contours=20,smooth_factor=1.0,linewidth=1,alpha=1.0,label=""):
    font = fm.FontProperties(family = 'Trebuchet', weight ='light')
    figure.patch.set_facecolor('white')
    axes = figure.add_subplot(111)

    if XY_range is None:
        XY_range = [[X.min(),X.max()],[Y.min(),Y.max()]]

    range_x = np.linspace(XY_range[0][0],XY_range[0][1],n_points)
    range_y = np.linspace(XY_range[1][0],XY_range[1][1],n_points)
    xx, yy = np.meshgrid(range_x,range_y)

    range_x_cr = (range_x - range_x.mean())/range_x.std()
    range_y_cr = (range_y - range_y.mean())/range_y.std()
    xx_cr, yy_cr = np.meshgrid(range_x_cr,range_y_cr)

    def density_function(positions,radius=1,k=0.1):
        def density_func(x,y,return_potential=False):
            points = np.array(positions.values())
           
            if len((x+y).shape) == 1:
                distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis],2),0.5)
            elif len((x+y).shape) == 2:
                distances = np.power(np.power(x[np.newaxis] - points[:,0,np.newaxis,np.newaxis],2) +  np.power(y[np.newaxis] - points[:,1,np.newaxis,np.newaxis],2),0.5)

            density_potential = 1./2. * (1. - np.tanh(k*(distances - radius)))
            density = density_potential.sum(axis=0)

            if return_potential:
                return density, density_potential
            else:
                return density
        return density_func

    positions = dict(zip(range(len(X)),np.transpose([X,Y])))
    positions_cr = dict(zip(range(len(X)),np.transpose([(X-range_x.mean())/range_x.std(),(Y-range_y.mean())/range_y.std()])))
    
    radius = 10.0/n_points
    density_k = 100.*np.exp(-smooth_factor/2.)
    data_density, data_potential = density_function(positions_cr,radius,density_k)(xx_cr,yy_cr,return_potential=True)
    data_potential = np.transpose(data_potential,(1,2,0))

    data_membership = data_potential/data_density[...,np.newaxis]

    zz_map = np.sum(data_membership*Z[np.newaxis,np.newaxis,:],axis=2)


    levels = np.linspace(0,1,n_contours)

    # axes.contourf(xx,yy,zz_map,cmap=colormap)
    # axes.contourf(xx,yy,zz_map,levels,cmap=colormap,alpha=alpha,antialiased=True,vmin=0,vmax=1)
    axes.pcolormesh(xx,yy,zz_map,cmap=colormap,alpha=alpha,antialiased=True,shading='gouraud',linewidth=0,vmin=0,vmax=1)
    axes.contourf(xx,yy,zz_map,levels,cmap=colormap,alpha=alpha/3.,antialiased=True,vmin=0,vmax=1)
    axes.contour(xx,yy,zz_map,levels,cmap=colormap,alpha=alpha,linewidth=linewidth,antialiased=True,vmin=0,vmax=1)

    axes.set_xlim(*tuple(XY_range[0]))
    axes.set_xlabel(xlabel,fontproperties=font, size=10, style='italic')
    axes.set_xticklabels(axes.get_xticks(),fontproperties=font, size=12)
    axes.set_ylim(*tuple(XY_range[1]))
    axes.set_ylabel(ylabel, fontproperties=font, size=10, style='italic')
    axes.set_yticklabels(axes.get_yticks(),fontproperties=font, size=12)



def smooth_plot(figure,X,Y,color1,color2,xlabel="",ylabel="",filled=False,n_points=400,smooth_factor=1.0,spline_order=3,linewidth=3,alpha=1.0,label=""):
    """
    """
    X_smooth = np.linspace(X.min(),X.max(),n_points)
    tck = splrep(X,Y,s=smooth_factor,k=spline_order)
    Y_smooth = splev(X_smooth,tck,der=0)

    font = fm.FontProperties(family = 'Trebuchet', weight ='light')
    #font = fm.FontProperties(family = 'CenturyGothic',fname = '/Library/Fonts/Microsoft/Century Gothic', weight ='light')
    figure.patch.set_facecolor('white')
    axes = figure.add_subplot(111)
    axes.plot(X,Y,linewidth=1,color=tuple(color2),alpha=0.2)
    if filled:
        axes.fill_between(X_smooth,Y_smooth,0,color=color2,alpha=0.1)
    for i in xrange(100):
        color = tuple(color1*(1.0-i/100.0) + color2*(i/100.0))
        if i == 0:
            axes.plot(X_smooth[(i*n_points/100):((i+1)*n_points)/100+1],Y_smooth[(i*n_points/100):((i+1)*n_points)/100+1],linewidth=linewidth,color=color,alpha=alpha,label=label)
        else:
            axes.plot(X_smooth[(i*n_points/100):((i+1)*n_points)/100+1],Y_smooth[(i*n_points/100):((i+1)*n_points)/100+1],linewidth=linewidth,color=color,alpha=alpha)
    axes.set_xlim(X.min(),X.max())
    axes.set_xlabel(xlabel,fontproperties=font, size=10, style='italic')
    axes.set_xticklabels(axes.get_xticks(),fontproperties=font, size=12)
    if '%' in ylabel:
        axes.set_ylim(0,np.minimum(2*Y.max(),100))
    axes.set_ylabel(ylabel, fontproperties=font, size=10, style='italic')
    axes.set_yticklabels(axes.get_yticks(),fontproperties=font, size=12)


def bar_plot(figure,X,Y,color1,color2,xlabel="",ylabel="",label=""):
    font = fm.FontProperties(family = 'Trebuchet', weight ='light')
    #font = fm.FontProperties(family = 'CenturyGothic',fname = '/Library/Fonts/Microsoft/Century Gothic', weight ='light')
    figure.patch.set_facecolor('white')
    axes = figure.add_subplot(111)
    width = X[1]-X[0]
    axes.plot(X,Y,linewidth=1,color=tuple(color2),alpha=0.0,label=label)
    for x in xrange(X.size):
        i = (Y[x]-Y.min())/(Y.max()-Y.min())
        color = tuple(color1*(1.0-i) + color2*(i))
        axes.add_patch(Rectangle((X[x] - width/2, 0), width, Y[x], facecolor=color, edgecolor=(0.8,0.8,0.8), alpha = 0.5))
    axes.set_xlim(X.min(),X.max())
    axes.set_xlabel(xlabel,fontproperties=font, size=10, style='italic')
    axes.set_xticklabels(axes.get_xticks(),fontproperties=font, size=12)
    if '%' in ylabel:
        axes.set_ylim(0,np.minimum(2*Y.max(),100))
    axes.set_ylabel(ylabel, fontproperties=font, size=10, style='italic')
    axes.set_yticklabels(axes.get_yticks(),fontproperties=font, size=12)


def histo_plot(figure,X,color,xlabel="",ylabel="",cumul=False,bar=True,n_points=400,smooth_factor=0.1,spline_order=3,linewidth=3,alpha=1.0,label=""):
    if '%' in xlabel:
        magnitude = 100
        X_values = np.array(np.minimum(np.around(X),101),int)
    else:
        # magnitude = np.power(10,np.around(4*np.log10(X.mean()))/4+0.5)
        magnitude = np.power(10,np.around(4*np.log10(np.nanmean(X)+np.nanstd(X)+1e-7))/4+0.5)
        magnitude = np.around(magnitude,int(-np.log10(magnitude))+1)
        # print magnitude
        #magnitude = X.mean()+5.0*X.std()
        X_values = np.array(np.minimum(np.around(100*X[True-np.isnan(X)]/magnitude),101),int)
    X_histo = np.zeros(101,float)
    for x in xrange(101):
        X_histo[x] = nd.sum(np.ones_like(X_values,float),X_values,index=x)
        if '%' in ylabel:
            X_histo[x] /= X_values.size/100.0
        if cumul:
            X_histo[x] += X_histo[x-1]

    if bar:
        bar_plot(figure,np.linspace(0,magnitude,101),X_histo,np.array([1,1,1]),color,xlabel,ylabel,label=label)
    else:
        smooth_plot(figure,np.linspace(0,magnitude,101),X_histo,color,color,xlabel,ylabel,n_points=n_points,smooth_factor=smooth_factor,spline_order=spline_order,linewidth=linewidth,alpha=alpha,label=label)


def violin_plot(figure,X,data,colors,xlabel="",ylabel="",n_points=400,violin_width=None,linewidth=3,marker_size=20):
    font = fm.FontProperties(family = 'Trebuchet', weight ='light')
    #font = fm.FontProperties(family = 'CenturyGothic',fname = '/Library/Fonts/Microsoft/Century Gothic', weight ='light')
    figure.patch.set_facecolor('white')
    axes = figure.add_subplot(111)
    if violin_width is None:
        if len(X)>1:
            violin_width = ((np.array(X)[1:] - np.array(X)[:-1]).mean())/3.
        else:
            violin_width = 0.33
    for x in xrange(len(X)):
        color = colors[x]

        Y = gaussian_kde(data[x])
        D_smooth = np.linspace(np.percentile(Y.dataset,1),np.percentile(Y.dataset,99),n_points)
        Y_smooth = Y.evaluate(D_smooth)
        Y_smooth = violin_width*Y_smooth/Y_smooth.max()
        axes.fill_betweenx(D_smooth,X[x],X[x]+Y_smooth,facecolor=color,alpha=0.1)
        axes.fill_betweenx(D_smooth,X[x],X[x]-Y_smooth,facecolor=color,alpha=0.1)
        axes.plot(X[x]+Y_smooth,D_smooth,color=color,linewidth=linewidth,alpha=0.8)
        axes.plot(X[x]-Y_smooth,D_smooth,color=color,linewidth=linewidth,alpha=0.8)
        axes.plot([X[x]-Y_smooth[0],X[x]+Y_smooth[0]],[D_smooth[0],D_smooth[0]],color=color,linewidth=linewidth,alpha=0.8)
        axes.plot([X[x]-Y_smooth[-1],X[x]+Y_smooth[-1]],[D_smooth[-1],D_smooth[-1]],color=color,linewidth=linewidth,alpha=0.8)
        axes.plot(X[x]-Y_smooth,D_smooth,color=color,linewidth=linewidth,alpha=0.8)
        axes.plot(X[x],np.percentile(data[x],50),'o',markersize=marker_size,markeredgewidth=linewidth,color=color)
        axes.plot([X[x],X[x]],[np.percentile(data[x],25),np.percentile(data[x],75)],color=color,linewidth=2*linewidth,alpha=0.5)
    axes.set_xlim(min(X)-1,max(X)+1)
    axes.set_xlabel(xlabel,fontproperties=font, size=10, style='italic')
    axes.set_xticklabels(axes.get_xticks(),fontproperties=font, size=12)
    axes.set_ylabel(ylabel, fontproperties=font, size=10, style='italic')
    axes.set_yticklabels(axes.get_yticks(),fontproperties=font, size=12)


def box_plot(figure,X,data,colors,xlabel="",ylabel="",box_width=None,linewidth=3,marker_size=20,alpha=1):
    font = fm.FontProperties(family = 'Trebuchet', weight ='light')
    #font = fm.FontProperties(family = 'CenturyGothic',fname = '/Library/Fonts/Microsoft/Century Gothic', weight ='light')
    figure.patch.set_facecolor('white')
    axes = figure.add_subplot(111)
    
    if box_width is None:
        if len(X)>1:
            box_width = ((np.array(X)[1:] - np.array(X)[:-1]).mean())/3.
        else:
            box_width = 0.33
    for x in xrange(len(X)):
        color = colors[x]
        box_color = np.array(color)/2.

        axes.fill_between([X[x]-box_width,X[x]+box_width],[np.percentile(data[x],75),np.percentile(data[x],75)],[np.percentile(data[x],25),np.percentile(data[x],25)],facecolor=color,alpha=alpha/2.)
        axes.plot([X[x]-box_width,X[x]+box_width],[np.percentile(data[x],50),np.percentile(data[x],50)],color=box_color,linewidth=linewidth,alpha=(alpha+1)/2)
        axes.plot([X[x]-box_width,X[x]+box_width],[np.percentile(data[x],25),np.percentile(data[x],25)],color=box_color,linewidth=linewidth/2.,alpha=alpha)
        axes.plot([X[x]-box_width,X[x]+box_width],[np.percentile(data[x],75),np.percentile(data[x],75)],color=box_color,linewidth=linewidth/2.,alpha=alpha)
        axes.plot([X[x]-box_width,X[x]-box_width],[np.percentile(data[x],25),np.percentile(data[x],75)],color=box_color,linewidth=linewidth/2.,alpha=alpha)
        axes.plot([X[x]+box_width,X[x]+box_width],[np.percentile(data[x],25),np.percentile(data[x],75)],color=box_color,linewidth=linewidth/2.,alpha=alpha)
        axes.plot([X[x],X[x]],[np.percentile(data[x],25),np.percentile(data[x],10)],color=box_color,linewidth=linewidth/3.,alpha=alpha)
        axes.plot([X[x],X[x]],[np.percentile(data[x],75),np.percentile(data[x],90)],color=box_color,linewidth=linewidth/3.,alpha=alpha)
        axes.plot([X[x]-box_width/4.,X[x]+box_width/5.],[np.percentile(data[x],10),np.percentile(data[x],10)],color=box_color,linewidth=linewidth/2.,alpha=alpha)
        axes.plot([X[x]-box_width/4.,X[x]+box_width/5.],[np.percentile(data[x],90),np.percentile(data[x],90)],color=box_color,linewidth=linewidth/2.,alpha=alpha)
        # axes.plot(X[x],np.percentile(data[x],50),'o',markersize=np.pi*np.power(box_width/2.,2.),markeredgewidth=linewidth/2.,color=color)

        outliers = [d for d in data[x] if (d < np.percentile(data[x],10)) or (d > np.percentile(data[x],90))]
        x_outliers = [X[x] for d in outliers]
        axes.scatter(x_outliers, outliers, s=marker_size/3.,c=color,linewidth=0,alpha=alpha/2.)

    axes.set_xlim(min(X)-1,max(X)+1)
    axes.set_xlabel(xlabel,fontproperties=font, size=10, style='italic')
    axes.set_xticklabels(axes.get_xticks(),fontproperties=font, size=12)
    axes.set_ylabel(ylabel, fontproperties=font, size=10, style='italic')
    axes.set_yticklabels(axes.get_yticks(),fontproperties=font, size=12)

def spider_plot(figure,Y,color1,color2=None,xlabels=[],ytargets=None,y_std=None,n_points=400,smooth_factor=0.1,filled=True,spline_order=1,linewidth=3,label=""):
    
    if color2 is None:
        color2 = color1

    font = fm.FontProperties(family = 'Trebuchet', weight ='light')
    #font = fm.FontProperties(family = 'CenturyGothic',fname = '/Library/Fonts/Microsoft/Century Gothic', weight ='light')
    figure.patch.set_facecolor('white')
    axes = figure.add_subplot(111,polar=True)

    n_axes = Y.shape[0]

    X = (np.arange(n_axes)*2*np.pi)/n_axes
    X = np.concatenate([X,[2*np.pi]])
    Y = np.concatenate([Y,[Y[...,0]]])

    X_smooth = np.linspace(X.min(),X.max(),n_points)
    tck = splrep(X,Y,s=smooth_factor,k=spline_order)
    Y_smooth = splev(X_smooth,tck,der=0)

    if y_std is not None:
        y_std = np.concatenate([y_std,[y_std[...,0]]])
        tck_std = splrep(X,y_std,s=smooth_factor,k=spline_order)
        Y_std_smooth = splev(X_smooth,tck_std,der=0)

    color = color1

    axes.set_ylim(0.,1.)
    axes.autoscale(False)
    if filled:
        axes.fill_between(X,Y,color=color,alpha=0.1)
    axes.plot(X,Y,linewidth=1,color=color,alpha=0.2)

    if ytargets is None:
        ytargets = [0.5 for a in xrange(n_axes)]

    for a in xrange(n_axes):
        points = np.array(np.arange((a-0.5)*(n_points/n_axes),(a+0.5)*(n_points/n_axes)),int)%n_points
        theta = X_smooth[points[1:-1]]
        rect1 = ytargets[a]*np.ones_like(theta)
        rect2 = np.ones_like(theta)
        axes.fill(np.concatenate((theta,theta[::-1])),np.concatenate((rect1,(0.5*rect1+0.5*rect2)[::-1])),color='green',alpha=0.1)
        axes.fill(np.concatenate((theta,theta[::-1])),np.concatenate(((0.5*rect1+0.5*rect2),rect2[::-1])),color='green',alpha=0.3)

        for i, p in enumerate(points[1:]):
            value = np.minimum(Y_smooth[p]/(0.5+ytargets[a]/2),1.0)
            color = value*color1 + (1.-value)*color2
            if filled:
                axes.fill_between(X_smooth[p-1:p+1],Y_smooth[p-1:p+1],color=color,alpha=0.1)
            if a == 0 and i == 0:
                axes.plot(X_smooth[p-1:p+1],Y_smooth[p-1:p+1],color=color,linewidth=linewidth,label=label)
            else:
                axes.plot(X_smooth[p-1:p+1],Y_smooth[p-1:p+1],color=color,linewidth=linewidth)

            if y_std is not None:
                axes.plot(X_smooth[p-1:p+1],Y_smooth[p-1:p+1]+Y_std_smooth[p-1:p+1],color=color,linewidth=1,alpha=0.5)
                axes.plot(X_smooth[p-1:p+1],Y_smooth[p-1:p+1]-Y_std_smooth[p-1:p+1],color=color,linewidth=1,alpha=0.5)

    plt.thetagrids(X*180/np.pi,labels=None,frac=1.2)
    plt.rgrids(np.arange(1,11)/10., labels=None, angle=22.5)

    axes.set_xticklabels(xlabels,fontproperties=font,size=12)
    # axes.set_xticklabels(xlabels,fontproperties=font,size=12,rotation=0,rotation_mode="anchor",ha="center",y=1.5)
    axes.set_yticklabels([],fontproperties=font,size=12)




