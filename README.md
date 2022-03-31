# ATLAS database manipulation

ATLAS is a publicly available antibiotic resistance database, the focus of the following paper:

**Seeking patterns of antibiotic resistance in ATLAS, an open, raw
MIC database with patient metadata* (2022) *Pablo Catalán, Jessica M.A. Blair, Ivana Gudelj, Jonathan R. Iredell and Robert E. Beardmore (accepted in Nature Communications).*

This repository holds the Python code needed to reproduce the analyses in that manuscript, as well as all figures (both in the main text as well as the Supplementary Material). Scripts are included as Python notebook (hopefully to help the reader follow the thread) and a separate *functions.py* file holds all necessary routines. All code was written by Pablo Catalán, except file *eucast.py* which was written by Emily Wood.

First, you should run *prepare_data.ipynb*. This will generate all data files that will be used to create the figures. It can take some time. 

Then, you can safely run *main_figures.ipynb* and *supp_figures.ipynb*.

Also, the notbeook *mic_distributions.ipynb* allows you to plot MIC distributions in ATLAS for every pathogen-antibiotic pair for which there are clinically defined breakpoints.

Hopefully, the notebooks are self-explanatory, but if there is something you don't understand (or something that doesn't work!), don't hesitate to contact me at pcatalan [at] math [dot] uc3m [dot] es

