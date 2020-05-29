#%%READ AND MODIFY DATA
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.ticker import FormatStrFormatter
import pylab
import numpy as np
import seaborn as sns
from scipy.integrate import odeint
import itertools
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
from operator import itemgetter
from scipy import stats

#PLOT CORRELATION MATRICES
def corrplot(R,Y,ax,fig,title):
    m,n=R.shape
    I=ax.imshow(R, vmin=0, vmax=1, cmap='RdYlGn', origin='upper',
                extent=[Y[0]-0.5,Y[-1]+0.5,Y[-1]+0.5,Y[0]-0.5],
                aspect='equal')
    cbar=fig.colorbar(I, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.set_yticklabels(cbar.ax.get_yticks())
    cbar.ax.set_ylabel('correlation')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylabel('Year')
    ax.set_xlabel('Year')
    ax.set_xticks(Y)
    ax.set_yticks(Y)
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.set_ylim([Y[-1]+0.5,Y[0]-0.5])
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.set_title(title)
    return I

#PLOT MIC DISTRIBUTION
def micdistplot(pY,gmm_x,Z,fig,ax,title):
    cmap=plt.cm.get_cmap('Purples')
    #HEATMAP
    npY=np.array(pY+[2020])
    X, Y=np.meshgrid(npY, gmm_x)
    H=ax.pcolormesh(X,Y,np.transpose(Z), edgecolors='face', cmap=cmap)
    cbar=fig.colorbar(H, ax=ax)
    cbar.ax.set_yticklabels(cbar.ax.get_yticks())
    cbar.ax.set_ylabel('frequency')
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.set_ylabel('log$_2$ MIC')
    ax.set_xlabel('Year')
    ax.set_xticks(npY+0.5)
    ax.set_yticklabels(ax.get_yticks())
    ax.set_xticklabels(ax.get_xticks(), rotation=90)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_title(title)     
    ax.set_xlim([min(pY), max(pY)+1])
    ax.set_ylim([-10,10])
    return fig,ax

#PLOT MIC TRENDS OVER HEATMAP
def mictrends_heatmap(fig,ax,pY,DFT,sp,dr):
    DFc=DFT[(DFT.Species==sp) & (DFT.Antibiotic==dr)]
    xp=np.linspace(2003,2020,100)
    yp=[0 for z in xp]
    ax.plot(xp,yp,'w', lw=2)
    #global trend
    Gi=DFc.Intercept.mean()
    Gs=DFc.Trend.mean()
    yp=[Gi+z*Gs for z in xp]
    ax.plot(xp,yp,'k', lw=4)
    #resistant trend
    Ri=DFc.Rintercept.mean()
    Rs=DFc.Rtrend.mean()
    yp=[Ri+z*Rs for z in xp]
    ax.plot(xp,yp,'--r', lw=2)
    #rest
    Si=DFc.Sintercept.mean()
    Ss=DFc.Strend.mean()
    yp=[Si+z*Ss for z in xp]
    ax.plot(xp,yp,'--b', lw=2)       
    ax.set_xlim([min(pY), max(pY)+1])
    ax.set_ylim([-10,10])
    return fig,ax

#GAUSSIAN MIXTURE AND EXTRACT R CLUSTER
def extract_rcluster(sp,dr,pY,pM,sM,DF):
    from sklearn import mixture
    newDFM=pd.DataFrame({'Country': [],'Year': [], 'MIC':[]})
    newDFm=pd.DataFrame({'Country': [],'Year': [], 'MIC':[]})  
    #GAUSSIAN MIXTURE FOR EACH YEAR
    for i,(y,smoothmic) in enumerate(zip(pY,sM)):
        mic=pM[i]
        n_components=np.arange(1,10)
        minAIC=1e32
        minNC=0
        for n in n_components:
            gmm=mixture.GaussianMixture(n_components=n, max_iter=1000)
            gmm.fit(X=np.expand_dims(smoothmic, 1))
            AIC=gmm.bic(X=np.expand_dims(smoothmic, 1))
            if AIC>minAIC:
                minNC=n-1
                break
            minAIC=AIC
        gmm=mixture.GaussianMixture(n_components=minNC, max_iter=1000)
        X=np.expand_dims(smoothmic, 1)
        gmm.fit(X)
        labels = gmm.predict(X)    
        #EXTRACT MOST RESISTANT COMPONENT
        maxV=-20
        imax=0
        for n in range(minNC):  
            a1=smoothmic.iloc[labels==n]  
            newM=a1.max()
            if newM>maxV:
                imax=a1.index
                maxV=newM
        maxMIC=mic[imax].tolist()
        yearL=[y for z in maxMIC]
        cL=(DF.iloc[imax,:]['Country']).tolist()
        DFmax=pd.DataFrame({'Country': cL, 'Year': yearL, 'MIC': maxMIC})
        newDFM=newDFM.append(DFmax, ignore_index=True, sort=False)
        #EXTRACT LEAST RESISTANT COMPONENT
        imin=smoothmic.index.difference(imax)
        minMIC=mic[imin].tolist()
        yearL=[y for z in minMIC]
        cL=(DF.iloc[imin,:]['Country']).tolist()
        DFmin=pd.DataFrame({'Country': cL, 'Year': yearL, 'MIC': minMIC})   
        newDFm=newDFm.append(DFmin, ignore_index=True, sort=False)
    #SAVE DATAFRAMES           
    newDFM.to_csv('results/Rcluster/'+sp+'_'+dr+'_Rcluster.csv', index=False)
    newDFm.to_csv('results/Rcluster/'+sp+'_'+dr+'_Scluster.csv', index=False)

#GAUSSIAN MIXTURE AND EXTRACT R CLUSTER
def gaussian_mixture(pY,sM):
    from sklearn import mixture
    #SMOOTH AND MIXTURE
    N=1000
    Z=np.zeros((len(pY), N))
    NC=[]
    ycount=0
    for y,smoothmic in zip(pY,sM):
        n_components=np.arange(1,10)
        minAIC=1e32
        minNC=0
        for n in n_components:
            gmm=mixture.GaussianMixture(n_components=n, max_iter=1000)
            gmm.fit(X=np.expand_dims(smoothmic, 1))
            AIC=gmm.bic(X=np.expand_dims(smoothmic, 1))
            if AIC>minAIC:
                minNC=n-1
                break
            minAIC=AIC
        NC.append(minNC)
        gmm=mixture.GaussianMixture(n_components=minNC, max_iter=1000)
        X=np.expand_dims(smoothmic, 1)
        gmm.fit(X)
        # Evaluate GMM
        gmm_x = np.linspace(-10,10,N)
        gmm_y = np.exp(gmm.score_samples(gmm_x.reshape(-1, 1)))
        for j,z in enumerate(gmm_y):
            Z[ycount,j]=z
        ycount+=1
    return Z,gmm_x

#GET BREAKPOINTS FROM DATABASE
def get_breakpoints(df,pL,drugs):
    bP={}
    for sp in pL:#OBTAIN CLINICAL BREAKPOINTS
        bP[sp]={}
        DFsp=df[df.Species==sp]
        for dr in drugs:
            br=dr+'_I'
            if br not in DFsp: continue
            DFnew=DFsp[[dr,br]].dropna()
            if DFnew.empty: continue
            DFS=DFnew[DFnew[br]=='Susceptible']
            DFR=DFnew[DFnew[br]=='Resistant']
            S1=DFS[dr].max()
            R1=DFR[dr].min()
            R=max(DFnew[dr])
            S=min(DFnew[dr])
            if len(DFS)>0:
                S=S1
            if len(DFR)>0:
                R=R1
            bP[sp][dr]=[S,R]
    return bP

#GET KEY FROM DATAFRAME
def get_key(df, key):
    kL=df[key].tolist()
    kM={}
    for k in kL:
        kM[k]=1
    kL=[]
    for k in kM:
        kL.append(k)
    return sorted(kL)

#MIC CHANGE
def mic_change(sp,dr,DF):
    DFc=pd.DataFrame({'Year': DF['Year'], dr: DF[dr]})
    DFc=DFc.dropna()
    if DFc.empty: 
        return []
    yL=get_key(DFc, 'Year')
    if len(yL)<2: 
        return []#CAN'T COMPUTE TREND!
    x1=DFc['Year'].tolist()
    y1=DFc[dr].tolist()
    slope, intercept, r_value, p_value, std_err=stats.linregress(x1,y1)
    if p_value>0.05:
        slope=0.0
        std_err=0.0
        intercept=DFc[dr].mean()
    return [sp,dr,slope,std_err,intercept]

#MIC DISTRIBUTION
def mic_dist(sp,dr,DF,BPsp):
    Year=range(2004,2018)
    #GET DATA
    DFsp=DF[DF.Species==sp]
    pY=[]
    pM=[]
    sM=[]
    for y1,y in enumerate(Year):
        DFy=DFsp[DFsp['Year']==y]
        newDF=pd.DataFrame({"Country": DFy['Country'], dr: DFy[dr]})
        newDF=newDF.dropna()
        mic=newDF[dr]-BPsp
        smoothmic=mic+np.random.normal(0.0,1.0,len(mic))
        if len(mic)<10: continue
        pY.append(y)    
        pM.append(mic)
        sM.append(smoothmic)
    return pY, pM, sM

#READ DATASET
def read_dataset(dir):
    #DRUGS LIST
    drugs=read_key('Drugs', dir)
    
    #READ FILE
    DF=pd.read_excel(dir+'atlas_treated.xlsx', dtype=str)
    for dr in drugs:
        DF[dr]=DF[dr].astype('float64')
    DF['Year']=DF['Year'].astype('int64')
  
    #RESTRICTED DATABASE
    #we only want those species with more than 500 entries in ATLAS
    spC=pd.read_csv(dir+'key_Species.txt', '\t', header=None)    
    spC=spC.sort_values([1],ascending=False)
    pL=[]
    for index, row in spC.iterrows():
        if row[1]>500:
            pL.append(row[0])
    DF=DF[DF.Species.isin(pL)]
    
    #GET CLINICAL BREAKPOINTS
    bP=get_breakpoints(DF,pL,drugs)
    return DF, pL, bP, drugs

#READ CORRELATION MATRICES FROM FILE
def read_corr(sp,dr):
    R=[]
    with open('results/correlations/'+sp+'_'+dr+'.txt','r') as f:
        for line in f:
            R.append([float(x) for x in line.split()])
    Y=[]
    with open('results/correlations/'+sp+'_'+dr+'_years.txt','r') as f:
        for line in f:
            w=line.split()
            Y.append(int(w[0]))
    return [np.array(R),Y]

#READ KEY FROM FILE
def read_key(key, folder):
    pL=pd.read_csv(folder+'key_'+key+'.txt','\t',header=None)
    return pL[0].tolist()

#TAU TEST
def tautest(R):
    n,m=R.shape
    tau=(np.sum(R**2)-n)/(n**2-n)
    tauN=np.sum(R**2)
    return tau,tauN

#WATERFALL PLOT
def waterfall(Y,X,M):
    from matplotlib.collections import PolyCollection, LineCollection
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    verts = []
    evenly_spaced_interval=np.linspace(0, 1,len(Y))
    colors=[cm.viridis(x) for x in evenly_spaced_interval]
    for i,y in enumerate(Y):
        verts.append(list(zip(X,M[i])))
    poly=PolyCollection(verts, facecolors=colors)
    line=LineCollection(verts, colors='k')
    line.set_alpha(0.4)
    ax.add_collection3d(poly, zdir='y', zs=Y)# zs=Y, zdir='y')
    ax.add_collection3d(line, zdir='y', zs=Y)# zs=Y, zdir='y')
    ax.set_xlabel('log$_2$ MIC')
    ax.set_xlim3d(-10, 10)
    ax.set_ylabel('Year', rotation=45)
    ax.set_ylim3d(Y[0], Y[-1])
    ax.set_zlabel('probability estimate', rotation=90)
    # ax.view_init(30, 110)
    return fig

#WORLDMAPS
def worldmaps():
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    DF=pd.read_csv(dir+'average_mic_per_year.csv')
    pC=get_key(DF, 'Country')
    Year=list(range(2004,2018))
    fR={}
    for y in Year:
        print(y)
        fR[y]={}
        for country in pC:
            DFc=DF[(DF.Country==country) & (DF.Year==y)]
            m=DFc['Average MIC'].dropna()
            #        print(country, m)
            if m.empty: continue
        #        m=m.mean()
            fR[y][country]=m.mean()
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    shapename = 'admin_0_countries'
    countries_shp = shpreader.natural_earth(resolution='110m',
                                        category='cultural', name=shapename)
    cL=[z.attributes['NAME_LONG'] for z in shpreader.Reader(countries_shp).records()]
    # some nice "earthy" colors
    cmap=sns.cubehelix_palette(n_colors=8, start=.5, rot=-.4, as_cmap=True)
    cmap=matplotlib.cm.get_cmap('hot_r')
    cmap=matplotlib.cm.get_cmap('seismic')
    cmin=-4.0
    cmax=0.0
    norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax)
    for y in Year:
        print(y)
        fig=plt.figure(figsize=(40,40))
#    regions=[11, 'CEU', 'S. Europe/Mediterranean']
    # choose a good projection for regional maps
        proj=ccrs.LambertConformal(central_longitude=15)
        proj=ccrs.PlateCarree()
#    # do the plot
#    ax = regionmask.defined_regions.srex.plot(regions=regions, add_ocean=False, resolution='50m',
#                                          proj=proj, label='abbrev')
    # fine tune the extent
#    proj=ccrs.PlateCarree(central_longitude=-35,
#                            central_latitude=40,
#                            standard_parallels=(0, 80)))
        ax=plt.axes(projection=proj)#ccrs.PlateCarree())
#    ax.set_extent([-15, 45, 28, 75], crs=ccrs.PlateCarree())
#    ax.add_feature(cartopy.feature.LAND)
#    ax.add_feature(cartopy.feature.LAKES, alpha=0.95)
        ax.stock_img()
    #    ax2=fig.add_subplot(1,2,2)
        for i,country in enumerate(shpreader.Reader(countries_shp).records()):
            cN=country.attributes['NAME_LONG']
        if cN=='Republic of Korea':
            cN='Korea, South'
        if cN=='Russian Federation':
            cN='Russia'
        if cN=='Slovakia':
            cN='Slovak Republic'
        if cN not in fR[y]:
            color='dimgrey'
        else:
            color=cmap((fR[y][cN]-cmin)/(cmax-cmin))
            print(cN, fR[y][cN], (fR[y][cN]-cmin)/(cmax-cmin))
#            color=cmap(fR[y][cN])
#        print(country.attributes['NAME_LONG'])
        ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                      facecolor=color,
                      label=country.attributes['NAME_LONG'])
    ax.set_title(y, fontproperties=prop, fontsize=30)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS)#, linestyle='-', linewidth=10)#, alpha=.5)
#    cbar=plt.colorbar(fig, cax=ax)
#    divider = make_axes_locatable(ax)
#    cax = divider.append_axes("right", size="5%", pad=0.05)
    cax = fig.add_axes([0.95, 0.4, 0.01, 0.05])
    cb=matplotlib.colorbar.ColorbarBase(ax=cax, cmap=cmap, 
                                        orientation="vertical")
    Lab=list(np.linspace(cmin,cmax,5))
#    yLab=[cmin+(cmax-cmin)*z for z in [0, 0.25, 0.5, 0.75, 1.0]]
#    yLab=[np.floor(z*100)/100 for z in yLab]
    yLab=[str(z) for z in Lab]
    cb.set_ticks(np.linspace(0., 1.0, 5))
    cb.ax.set_yticklabels(yLab, fontproperties=prop)
    cb.ax.tick_params(labelsize=20)
    fig.savefig(dir+'figures/worldmap_average_mic_'+str(y)+'.pdf', bbox_inches='tight')
#    fig.savefig(dir+'figures/resistance_'+str(y)+'.png', bbox_inches='tight')
    plt.close('all')
