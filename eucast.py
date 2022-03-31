#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 11:38:50 2021

@author: emily
"""

import os
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas
import numpy



## Species and drug 
AllSpecies = pd.read_csv('Species_counts.csv', sep=',',header=0)
sp = AllSpecies.Species
Drugs = pd.read_csv('Key_drugs.csv', sep=',',header=0)
dr = Drugs.Antibiotic
FinalDF= pd.DataFrame(columns=['Species']) 


NormFinal = pd.DataFrame(columns=['Species'])
value1 = "Acinetobacter anitratus"
NormFinal = NormFinal.append({'Species': value1}, ignore_index=True)


for j in range (0, len(dr)):

  if os.path.isfile('EUCAST/' + dr[j] + '.csv'): #Check file exists 
          pass
  else:
          continue
      
  for i in range (0, len(sp)):

     
     ds = pd.read_csv('atlas_treated.csv', sep=',',header=0, low_memory=False)
     
     DS = ds[ds['Species'].str.contains(sp[i])]
     DS = DS[["Year",dr[j]]]
     DS = DS.dropna(subset=[dr[j]]) #Drop drugs that are all NA
     years = DS['Year'].unique()
     
     if years.size >= 3: 

    
      counts = DS[dr[j]].value_counts()
      counts.index.name = dr[j]
      counts = counts.reset_index(name='Total')
    
      Prob = counts.Total
      ProbTotal = np.sum(counts.Total)
      ProbFinal = Prob.divide(ProbTotal, fill_value=0)
    
      counts['Prob'] = pd.Series(ProbFinal, index=counts.index)
      DS1 = counts
      
      
      outname = (sp[i] + '_' + dr[j] + '.csv')
    
      outdir = './PASS' #Just for future use really...
      if not os.path.exists(outdir):
          os.mkdir(outdir)
    
      fullname = os.path.join(outdir, outname)    
    
      DS1.to_csv(fullname)
      
      ds2 = pd.read_csv(('EUCAST/' + dr[j] + '.csv'), sep=',', header=0, low_memory=False) 
      ds2.rename(columns={ ds2.columns[0]: "Species" }, inplace = True)
      
      sp_count=ds2['Species'].str.contains(sp[i]).sum()
      
      if sp_count>0:
          pass
      else:
          continue

      
      
      DS2 = ds2[ds2[ds2.columns[0]].str.contains(sp[i])]
      DS2 = DS2.iloc[:, 0:20]


        
      MIC = [-8.96578, -7.96578, -6.96578, -5.96578, -5.05889, -4.05889, -3, 
             -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
      old_names = DS2.columns[1:20]  
      DS2.rename(columns=dict(zip(old_names, MIC)), inplace=True)
      DS2 = DS2.transpose()   
      DS2 = DS2.iloc[1:]
      DS2.index.name = dr[j]
      DS2 = DS2.rename(columns={ DS2.columns[0]: "Counts" })
     
      Prob = DS2.Counts
      ProbTotal = np.sum(Prob)
      ProbFinal = Prob.divide(ProbTotal, fill_value=0)
    
      DS2['Prob'] = pd.Series(ProbFinal, index=DS2.index)
      DS2 = DS2.reset_index(drop=False)

    
      outname = outname = (sp[i] + '_' + dr[j] + '_' + 'EUCAST' + '.csv')
      fullname = os.path.join(outdir, outname) 
      DS2.to_csv(fullname)
      
      New = pandas.concat([DS1[dr[j]],DS2[dr[j]]]).drop_duplicates().reset_index(drop=True)
      New = pd.DataFrame({dr[j]:New})
      New = New.sort_values(by=[dr[j]], ascending=False)
      DS1New = New.merge(DS1, how='left')
      DS2New = New.merge(DS2, how='left')
      
      ### Histograms ###
      fig, ax = plt.subplots()
      p1 = plt.bar(DS1New[dr[j]], DS1New.Prob, color = "skyblue", alpha=0.5, 
                 label="ATLAS")
      p2 = plt.bar(DS2New[dr[j]], DS2New.Prob, color = "forestgreen", alpha=0.5, 
                 label="EUCAST")
      plt.xlim([-10,10])
      plt.ylim([0,1])
      plt.axvline(x=0, linestyle='--', color = '0.5')
    
    
      plt.xlabel('log$_{2}$ MIC')
      plt.ylabel('Probability')
      plt.suptitle((sp[i] + ' ' + 'and' + ' ' + dr[j]))
      plt.legend()
      plt.legend(loc = "upper left")
      plt.show()
      
      outname = outname = (sp[i] + '_' + dr[j] + '.pdf')
    
      outdir = './Histograms'
      if not os.path.exists(outdir):
          os.mkdir(outdir)
    
      fullname = os.path.join(outdir, outname)    
      fig.savefig(fullname, format='pdf', dpi=1200)
      
      AData = numpy.array(DS1New.Prob)
      AData = numpy.nan_to_num(AData)
      
      EData = numpy.array(DS2New.Prob)
      EData = numpy.nan_to_num(EData)
      
      DataNorm = numpy.linalg.norm((AData - EData), ord=1) 
      
      Norm = pd.DataFrame()
      Norm['Species'] = pd.Series(sp[i])
      Norm[dr[j]] = pd.Series(DataNorm)

      
      NormFinal = pd.merge(NormFinal, Norm, how='outer')
      

    
     
     else:
         
      continue
  
     
  outdir = './FinalData'
  if not os.path.exists(outdir):
          os.mkdir(outdir)
             
NormFinal = NormFinal.replace('', np.nan).groupby('Species').first().replace(np.nan, '')  
outname = outname = ('Final_output' + '.csv')
fullname = os.path.join(outdir, outname) 
NormFinal.to_csv(fullname)

### Heatmap ###

NormFinal = NormFinal.replace(r'\s+',np.nan,regex=True).replace('',np.nan)
NormFinal2 = NormFinal.reset_index(drop=False)
plt.subplots(figsize=(20,15))
ax = sns.heatmap(NormFinal, linewidth=0.5)
plt.yticks(np.arange(72)+0.5, NormFinal2.Species)
plt.xticks(rotation=90) 
plt.show()

ax.figure.savefig('NormHeatmap.pdf', dpi=1200)

## Reorder heatmap ##

rows_index=NormFinal.max(axis=1).sort_values(ascending=False).index
col_index=NormFinal.max().sort_values(ascending=False).index
new_df=NormFinal.loc[rows_index,col_index]


plt.subplots(figsize=(20,15))
ax = sns.heatmap(new_df, linewidth=0.5)
plt.xticks(rotation=90) 

plt.show()

ax.figure.savefig('NormHeatmapReorder.pdf', dpi=1200)