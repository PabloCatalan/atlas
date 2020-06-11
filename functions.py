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
import statsmodels.api as sm

#CLUSTERING
def clustermap_clustering(X,Y,SC):
    Z=np.zeros((len(X), len(Y)))
    for i,x in enumerate(X):
        for j,y in enumerate(Y):
            key=x+'+'+y
            if key in SC:
                Z[i,j]=SC[key]
            else:
                Z[i,j]=0
    
    import scipy.cluster.hierarchy as sch
    method='ward'
    #LEFT DENDROGRAM
    Ylink1 = sch.linkage(Z, method=method)
    # Z1 = sch.dendrogram(Ylink1, orientation='left',link_color_func=lambda k: 'k')
    #TOP-SIDE DENDROGRAM
    Ylink2 = sch.linkage(np.transpose(Z), method=method)
    # Z2 = sch.dendrogram(Ylink2, link_color_func=lambda k: 'k')
    return Z,Ylink1,Ylink2
    
#GAUSSIAN MIXTURE AND EXTRACT R CLUSTER
def extract_rcluster(sp,dr,pY,pM,sM,DF):
    from sklearn import mixture
    newDFM=pd.DataFrame({'Country': [],'Year': [], 'MIC':[]})
    newDFm=pd.DataFrame({'Country': [],'Year': [], 'MIC':[]})  
    #GAUSSIAN MIXTURE FOR EACH YEAR
    for i,(y,smoothmic) in enumerate(zip(pY,sM)):
        mic=pM[i]
        n_components=np.arange(1,6)
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
        X=np.expand_dims(mic,1)
        labels=gmm.predict(X)    
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
        yearL=[int(y) for z in maxMIC]
        cL=DF.loc[imax,:]['Country']
        cL=cL.tolist()
        DFmax=pd.DataFrame({'Country': cL, 'Year': yearL, 'MIC': maxMIC})
        DFmax['Year']=DFmax['Year'].astype(int)
        newDFM=newDFM.append(DFmax, ignore_index=True, sort=False)
        #EXTRACT LEAST RESISTANT COMPONENT
        imin=smoothmic.index.difference(imax)
        minMIC=mic[imin].tolist()
        yearL=[int(y) for z in minMIC]
        cL=DF.loc[imin,:]['Country']
        cL=cL.tolist()
        DFmin=pd.DataFrame({'Country': cL, 'Year': yearL, 'MIC': minMIC})   
        DFmin['Year']=DFmin['Year'].astype(int)
        newDFm=newDFm.append(DFmin, ignore_index=True, sort=False)
    #SAVE DATAFRAMES           
    newDFM.to_csv('results/Rcluster/'+sp+'_'+dr+'_Rcluster.csv', float_format='%.3f', index=False)
    newDFm.to_csv('results/Rcluster/'+sp+'_'+dr+'_Scluster.csv', float_format='%.3f', index=False)

#GAUSSIAN MIXTURE AND EXTRACT R CLUSTER
def extract_rclusternowrite(sp,dr,pY,pM,sM,DF):
    from sklearn import mixture
    newDFM=pd.DataFrame({'Country': [],'Year': [], 'MIC':[]})
    newDFm=pd.DataFrame({'Country': [],'Year': [], 'MIC':[]})  
    #GAUSSIAN MIXTURE FOR EACH YEAR
    for i,(y,smoothmic) in enumerate(zip(pY,sM)):
        mic=pM[i]
        n_components=np.arange(1,6)
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
        X=np.expand_dims(mic,1)
        labels=gmm.predict(X)    
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
        yearL=[int(y) for z in maxMIC]
        cL=DF.loc[imax,:]['Country']
        cL=cL.tolist()
        DFmax=pd.DataFrame({'Country': cL, 'Year': yearL, 'MIC': maxMIC})
        DFmax['Year']=DFmax['Year'].astype(int)
        newDFM=newDFM.append(DFmax, ignore_index=True, sort=False)
        #EXTRACT LEAST RESISTANT COMPONENT
        imin=smoothmic.index.difference(imax)
        minMIC=mic[imin].tolist()
        yearL=[int(y) for z in minMIC]
        cL=DF.loc[imin,:]['Country']
        cL=cL.tolist()
        DFmin=pd.DataFrame({'Country': cL, 'Year': yearL, 'MIC': minMIC})   
        DFmin['Year']=DFmin['Year'].astype(int)
        newDFm=newDFm.append(DFmin, ignore_index=True, sort=False)
    #SAVE DATAFRAMES           
    return newDFM, newDFm

#FIND CHANGEPOINT
def find_changepoint(data):
    #from: http://www.claudiobellei.com/2016/11/15/changepoint-frequentist/
    n = len(data)
    tau = np.arange(1,n)
    lmbd = 2*np.log(n) #Bayesian Information Criterion
    eps = 1.e-8 #to avoid zeros in denominator
    mu0 = np.mean(data)
    s0 = np.sum((data-mu0)**2)
    s1 = np.asarray([np.sum((data[0:i]-np.mean(data[0:i]))**2) for i in tau])
    s2 = np.asarray([np.sum((data[i:]-np.mean(data[i:]))**2) for i in tau])
    R  = s0-s1-s2
    G  = np.max(R)
    taustar = list(R).index(G) + 1
    sd1 = np.std(data[0:taustar])
    sd2 = np.std(data[taustar:])
    mu1 = np.mean(data[0:taustar])
    mu2 = np.mean(data[taustar:])
    #use pooled standard deviation
    var = ( taustar*sd1**2 + (n-taustar)*sd2**2 ) / n
    teststat=2*G
    criterion=var*lmbd
    if teststat > criterion:
        if mu2>mu1:
            return taustar
        else:
            return -1
    else:
        return -1#len(data)
    
#GAUSSIAN MIXTURE
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

#CUSTOM LINEAR REGRESSION WITH CONFIDENCE INTERVALS
def LR(x,y):
    mask = ~np.isnan(x) & ~np.isnan(y)
    x1=np.array(sorted(x[mask]))
    y1=np.array([z2 for z1,z2 in sorted(zip(x[mask],y[mask]))])
    x2 = sm.add_constant(x1) 
    fitted = sm.OLS(y1, x2).fit() 
    y_hat = fitted.predict(x2) # x is an array from line 12 above
    y_err = y1 - y_hat
    mean_x = x2.T[1].mean()
    #COMPUTE CONFIDENCE INTERVALS
    n = len(x1)
    dof = n - fitted.df_model - 1
    t = stats.t.ppf(1-0.025, df=dof)
    s_err = np.sum(np.power(y_err, 2))
    conf = t * np.sqrt((s_err/(n-2))*(1.0/n + (np.power((x2-mean_x),2) / 
           ((np.sum(np.power(x2,2))) - n*(np.power(mean_x,2))))))
    upper = y_hat + abs(conf[:,1])
    lower = y_hat - abs(conf[:,1])
    
    #COMPUTE R SQUARED
    eT=y1-y1.mean()
    ssT=np.sum(eT**2)
    eR=y1-y_hat
    ssE=np.sum(eR**2)
    R2=1-ssE/ssT
    
    #GET PARAMETERS
    slope, intercept, r_value, p_value, std_err=stats.linregress(x1,y1)    
    return x1,y_hat,upper,lower,R2, slope, intercept

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
        pM.append(np.array(mic, dtype=float))
        sM.append(np.array(smoothmic, dtype=float))
    return pY, pM, sM

#PLOT CLUSTERMAPS
def plot_clustermap(X,Y,Z,Ylink1,Ylink2,xp,yp,zp):
    import scipy.cluster.hierarchy as sch
    fig = plt.figure(figsize=(8,8))
    # for the letf dendrogram
    # Add an axes at position rect [left, bottom, width, height]
    ax1 = fig.add_axes([0.09, 0.1, 0.2, 0.6])
    Z1 = sch.dendrogram(Ylink1, orientation='left',link_color_func=lambda k: 'k')
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.axis('off')

    # top side dendogram
    ax2 = fig.add_axes([0.3, 0.71, 0.6, 0.2])
    Z2 = sch.dendrogram(Ylink2, link_color_func=lambda k: 'k')
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.axis('off')

    # main heat-map
    axmatrix = fig.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    #ADAPT XP AND YP
    newX=list(X[idx1])
    newY=list(Y[idx2])
    xp=[newX.index(z) for z in xp]
    yp=[newY.index(z) for z in yp]
    Z = Z[idx1, :]
    Z = Z[:, idx2]
    # the actual heat-map    
    cmap=matplotlib.cm.get_cmap('seismic')
    im = axmatrix.imshow(Z, aspect='auto', cmap=cmap, vmin=-1e6, vmax=1e6)
    size=55
    P=axmatrix.scatter(yp, xp, c=zp, s=size, cmap=cmap, vmin=-1, vmax=1, edgecolors='k', linewidths=0.5)

    # xticks to the right (x-axis)
    axmatrix.set_xticks(range(len(Y)))
    axmatrix.set_xticklabels(Y[idx2], minor=False)
    axmatrix.xaxis.set_label_position('bottom')
    axmatrix.xaxis.tick_bottom()

    plt.xticks(rotation=90)

    # xticks to the right (y-axis)
    axmatrix.set_yticks(range(len(X)))
    axmatrix.set_yticklabels(X[idx1], minor=False)
    axmatrix.yaxis.set_label_position('right')
    axmatrix.yaxis.tick_right()

    # to add the color bar
    axcolor=fig.add_axes([0.09, 0.71, 0.02, 0.1])
    cbar=fig.colorbar(P,cax=axcolor)
    cbar.ax.set_yticklabels(['-1.0 or less', '0.0', '1.0 or more'])

    return fig,axmatrix

#PLOT CORRELATION MATRICES
def plot_corr(R,Y,ax,fig,title):
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

#PLOT MIC DISTRIBUTIONS
def plot_micdist(Z,pY,fig,ax,title):
    cmap=plt.cm.get_cmap('Purples')
    #HEATMAP
    npY=np.array(pY+[2020])
    N=1000
    gmm_x=np.linspace(-10,10,N)
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
    return ax

#PLOT MIC HISTOGRAMS
def plot_michistograms(pY,oH,sH,gM,title):
    nrows=1
    ncols=len(pY)
    if len(pY)>6:
        nrows=2
        ncols=int((len(pY)+1)/2)
    fig=plt.figure(figsize=(3*ncols,4*nrows))
    N=1000
    X=np.linspace(-10,10,N)
    for i,y in enumerate(pY):
        ax=fig.add_subplot(nrows,ncols,i+1)
        #BINS
        B1=np.arange(-10,10.5,0.5)#for sH
        B2=np.arange(-10,10.5,1)#for oH
        #plot
        ax.bar(B1[:-1], sH[i], width=0.5, alpha=0.5, facecolor='navy')
        ax.bar(B2[:-1], oH[i], width=1.0, alpha=0.5, facecolor='orangered')
        ax.plot(X, gM[i], color="crimson", lw=2, label=y)
        yp=np.linspace(0,1,100)
        ax.plot([0 for z in yp],yp,'--k')
        ax.set_xlabel('MIC')
        ax.set_ylabel('probability')
        ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.set_xticks([-5,0,5])
        ax.set_xticklabels(['-5', '0', '5'])
        ax.set_xlim([-10,10])
        ax.set_ylim([0,1])   
        if i==0:         
            ax.set_title(title)
        ax.text(0.05,0.9,y,transform=ax.transAxes)
    fig.tight_layout()
    return fig

#WATERFALL PLOT
def plot_waterfall(Z,pY,fig,title):
    from matplotlib.collections import PolyCollection, LineCollection
    #READ MIC DATA FROM FILES
    N=1000
    X=np.linspace(-10,10,N)
    ax = fig.gca(projection='3d')
    verts = []
    evenly_spaced_interval=np.linspace(0, 1,len(pY))
    colors=[cm.viridis(x) for x in evenly_spaced_interval]
    for i,y in enumerate(pY):
        verts.append(list(zip(X,Z[i])))
    poly=PolyCollection(verts, facecolors=colors)
    line=LineCollection(verts, colors='k')
    line.set_alpha(0.4)
    ax.add_collection3d(poly, zdir='y', zs=pY)# zs=Y, zdir='y')
    ax.add_collection3d(line, zdir='y', zs=pY)# zs=Y, zdir='y')
    ax.set_xlabel('log$_2$ MIC', rotation=45)
    ax.set_xlim3d(-10, 10)
    ax.set_ylabel('Year', rotation=45)
    ax.set_ylim3d(pY[0], pY[-1])
    ax.set_zlabel('probability estimate', rotation=90)
    ax.view_init(30, 290)
    ax.set_title(title)
    return ax

#READ DATASET
def read_dataset(dir, complete=False):
    #DRUGS LIST
    drugs=read_key('Drugs', dir)
    
    #READ FILE
    DF=pd.read_excel(dir+'atlas_treated.xlsx', dtype=str)
    for dr in drugs:
        DF[dr]=DF[dr].astype('float64')
    DF['Year']=DF['Year'].astype('int64')
      
    
    if complete==False:#RESTRICTED DATABASE
        #we only want those species with more than 500 entries in ATLAS
        spC=pd.read_csv(dir+'key_Species.txt', '\t', header=None)    
        spC=spC.sort_values([1],ascending=False)
        pL=[]
        for index, row in spC.iterrows():
            if row[1]>500:
                pL.append(row[0])
        DF=DF[DF.Species.isin(pL)]
    else:#COMPLETE DATABASE
        spC=pd.read_csv(dir+'key_Species.txt', '\t', header=None)    
        spC=spC.sort_values([1],ascending=False)
        pL=spC[0].tolist()
    
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

#READ CORRELATION MATRICES FROM FILE
def read_hist(sp,dr):
    oH=[]
    with open('results/mic_distributions/'+sp+'_'+dr+'_original.txt','r') as f:
        for line in f:
            oH.append([float(x) for x in line.split()])
    sH=[]
    with open('results/mic_distributions/'+sp+'_'+dr+'_smooth.txt','r') as f:
        for line in f:
            sH.append([float(x) for x in line.split()])
    return [np.array(oH),np.array(sH)]

#READ CORRELATION MATRICES FROM FILE
def read_micdist(sp,dr):
    Z=[]
    with open('results/mic_distributions/'+sp+'_'+dr+'.txt','r') as f:
        for line in f:
            Z.append([float(x) for x in line.split()])
    pY=[]
    with open('results/mic_distributions/'+sp+'_'+dr+'_years.txt','r') as f:
        for line in f:
            w=line.split()
            pY.append(int(w[0]))
    return [Z,pY]

#READ KEY FROM FILE
def read_key(key, folder):
    pL=pd.read_csv(folder+'key_'+key+'.txt','\t',header=None)
    return pL[0].tolist()

#TAU TEST
def tautest(R):
    n,m=R.shape
    tau=(np.sum(R**2)-n)/2/(n**2-n)
    tauN=(np.sum(R**2)-n)/2
    return tau,tauN

#WORLDMAPS
def worldmaps(M,Y,Mmin,Mmax):
    #M is a dict that holds the value that we want to plot
    #for each country in ATLAS
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    #PREPARE MAP
    shapename = 'admin_0_countries'
    countries_shp = shpreader.natural_earth(resolution='110m',
                                        category='cultural', name=shapename)
    cmap=matplotlib.cm.get_cmap('seismic')
    pC=[]
    for key,value in M.items():
        pC.append(key)
    cmin=np.floor(Mmin)
    cmax=np.ceil(Mmax)
    norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax)
    #START PLOT
    fig=plt.figure(figsize=(40,40))
    proj=ccrs.PlateCarree()
    ax=plt.axes(projection=proj)#ccrs.PlateCarree())
    ax.stock_img()
    for i,country in enumerate(shpreader.Reader(countries_shp).records()):
        cN=country.attributes['NAME_LONG']
        if cN=='Republic of Korea':
            cN='Korea, South'
        if cN=='Russian Federation':
            cN='Russia'
        if cN=='Slovakia':
            cN='Slovak Republic'
        if cN not in M:
            color='dimgrey'
        else:
            color=cmap((M[cN]-cmin)/(cmax-cmin))
            # color=cmap(M[cN])
        ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                      facecolor=color,
                      label=country.attributes['NAME_LONG'])
    ax.set_title(Y, fontsize=40)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS)
    cax = fig.add_axes([0.95, 0.4, 0.01, 0.05])
    cb=matplotlib.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm,
                                        orientation="vertical")
    cb.ax.set_yticklabels(cb.ax.get_yticks())
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return fig,ax

#WORLDMAPS
def worldmaps_europe(M,Y,Mmin,Mmax):
    #M is a dict that holds the value that we want to plot
    #for each country in ATLAS
    import cartopy
    import cartopy.crs as ccrs
    import cartopy.io.shapereader as shpreader
    #PREPARE MAP
    shapename = 'admin_0_countries'
    countries_shp = shpreader.natural_earth(resolution='110m',
                                        category='cultural', name=shapename)
    cmap=matplotlib.cm.get_cmap('seismic')
    pC=[]
    for key,value in M.items():
        pC.append(key)
    cmin=np.floor(Mmin)
    cmax=np.ceil(Mmax)
    norm = matplotlib.colors.Normalize(vmin=cmin, vmax=cmax)
    #START PLOT
    fig=plt.figure(figsize=(40,40))
    # choose a good projection for regional maps
    regions=[11, 'CEU', 'S. Europe/Mediterranean']
    proj=ccrs.LambertConformal(central_longitude=15)
    ax=plt.axes(projection=proj)#ccrs.PlateCarree())
    # ax.stock_img()
    ax.set_extent([-15, 45, 28, 75], crs=ccrs.PlateCarree())
    for i,country in enumerate(shpreader.Reader(countries_shp).records()):
        cN=country.attributes['NAME_LONG']
        if cN=='Republic of Korea':
            cN='Korea, South'
        if cN=='Russian Federation':
            cN='Russia'
        if cN=='Slovakia':
            cN='Slovak Republic'
        if cN not in M:
            color='dimgrey'
        else:
            color=cmap((M[cN]-cmin)/(cmax-cmin))
            # color=cmap(M[cN])
        ax.add_geometries(country.geometry, ccrs.PlateCarree(),
                      facecolor=color,
                      label=country.attributes['NAME_LONG'])
    ax.set_title(Y, fontsize=40)
    ax.add_feature(cartopy.feature.OCEAN)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS)
    cax = fig.add_axes([0.95, 0.4, 0.01, 0.05])
    cb=matplotlib.colorbar.ColorbarBase(ax=cax, cmap=cmap, norm=norm,
                                        orientation="vertical")
    cb.ax.set_yticklabels(cb.ax.get_yticks())
    cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    return fig,ax