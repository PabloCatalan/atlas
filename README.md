# ATLAS database manipulation

ATLAS is a publicly available antibiotic resistance database, the focus of the following paper:

**Clinical Antibiotic Resistance Patterns Across 70 Countries** (2020) *Pablo Catalán, Carlos Reding, Jessica Blair, Ivana Gudelj, Jon Iredell and Robert Beardmore (to be published).*

This repository holds the Python code needed to reproduce the analyses in that manuscript, as well as all figures (both in the main text as well as the Supplementary Material). Scripts are included as Python notebook (hopefully to help the reader follow the thread) and a separate *functions.py* file holds all necessary routines.

First, you should run *prepare_data.ipynb*. This will generate all data files that will be used to create the figures. It can take some time. 

Then, you can safely run *main_figures.ipynb* and *supp_figures.ipynb*.

Hopefully, the notebooks are self-explanatory, but if there is something you don't understand, don't hesitate to contact me at pcatalan [at] math [dot] uc3m [dot] es

