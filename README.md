# Code repository for the paper _Efficient numerical methods for the Maxey-Riley equations with Basset History Term_

[![DOI](https://zenodo.org/badge/750261240.svg)](https://zenodo.org/doi/10.5281/zenodo.10839776)

This repository contains the Python scripts implemented for the development of the paper _Efficient numerical methods for the Maxey-Riley equations with Basset History Term_.

Here one can therefore find the scripts that generate the figures and tables in the following publication:

> _Urizarna-Carasa, J., Schlegel, L., Ruprecht, D. (2023), Efficient numerical methods for the Maxey-Riley equations with Basset history term. To appear in International Journal of Computational Fluid Dynamics._

For questions, contact [M. Sc. Julio Urizarna](https://www.linkedin.com/in/julio-urizarna/) or [Prof. Dr. Daniel RUPRECHT](https://www.mat.tuhh.de/home/druprecht/?homepage_id=druprecht).

## How to run the code

> :mega: Note that the file `00_2mm_Faraday_50Hz_40ms_1_6g.mat` that contains the data with the experimental field is not provided, since we do not hold ownership on it. This may rise an error when trying to reproduce some figures and tables. Please contact [Prof. Dr. Alexandra VON KAMEKE](https://www.haw-hamburg.de/hochschule/beschaeftigte/detail/person/person/show/alexandra-von-kameke/) if interested in using it.

Within each Figure or Table folder, e.g. Figure-01 or Table-4-upper, there is a file with the either prefixes "a02" (a02_PLOTTR_) or "a06" (either a06_CONVRG_ or a06_PRECSN_ or a06_ERRORS_). These are the scripts that must be run in order to obtain the figures/tables:

- a02_PLOTTR_ files produce trajectory data,
- a06_CONVRG_ files produce Convergence plots or tables,
- a06_PRECSN_ files generate Work-Precision plots and
- a06_ERRORS_ file is only used once to obtain the error for different values of c for a particle in the Vortex. 

Within the files above, some parameters can be modified, such as time steps, initial and final times, densities, particle radius, etc.

All other files are supporting material used to run the scripts:

- a00_PMTERS_CONST.py defines the constants of the MRE (R, S, alpha, etc.),
- a00_MAT and a00_TSTEP define the matrices and time grid associated to Prasath et al.'s solver (enable a speed-up of the calculations),
- a03_FIELD0_ files define the velocity fields,
- a09_PRTCLE_ files hold the numerical or analytical solver (in case an analytical solution is available).

Within the OUTPUT folders, one can find the data associated to each Plot and Table as well as the Plot and Table themselves. Generating a single Figure or Table takes between 1 to 5 hours. The reader may therefore prefer to copy the data from the .txt file and generate the figures from it.

The toolbox currently depends on classical Python packages such as [Numpy](https://numpy.org/), [SciPy](https://scipy.org/) and [Matplotlib](https://matplotlib.org/).

Additional libraries like [progressbar2](https://pypi.org/project/progressbar2/) may also require installation.

## Script naming convenction

Each .py file is made up of a code with the following structure *z99_AAAAAA_BBBBB.py*. Prefix *z99* is linked to the code *AAAAAA*, but enables an appropiate sorting within the folder different from Alphabetical sorting. Code *AAAAAA* summarizes what the code does:

 - either defines parameters (PMTERS),
 - plots trajectories (PLOTTR),
 - creates Convergence or Work-precision plots (CONVRG or PRECSN),
 - defines the velocity fields (FIELD0) or
 - the particle classes associated to each solver (PRTCLE)

 The second part of the code, i.e. *BBBBB*, provides more specific information about the file, such as the type of solver or type of velocity field:

 - CONST stands for the fact that the file defines Constant parameters,
 - ANALY stands for the Analytical solution of the Vortex (at the time of developing this code we were only aware of one analytical solution),
 - OSCLT stands for oscillatory background, either its analytical solution (PRTCLE_OSCLT) or the field (FIELD0_OSCLT),
 - QSCNT stands for quiescent flow, either its analytical solution (PRTCLE_QSCNT) or the zero field (FIELD0_QSCNT),
 - VORTX stands for vortex described by Candelier et al. (2004), either its analytical solution (PRTCLE_VORTX) or the field (FIELD0_VORTX),
 - BICKL stands for Bickley Jet,
 - DATA1 stands for the Faraday flow, obtained from experimental Data,
 - 00000 stands for the abstract particle class,
 - TRAPZ holds the Trapezoidal Rule solver,
 - DTCHE holds Daitche's 1st, 2nd, 3rd order schemes,
 - PRSTH holds Prasath et al.'s solver,
 - DIRK4 holds the solver with the 4th order DIRK method,
 - IMEX4 holds the solver with either the 1st, 2nd, 3rd and 4th order IMEX.

## Repository structure

- [Code-for-Figures](./Code-for-Figures/) : Scripts to generate the figures from the papers as well as some additional ones that were excluded from it.
- [Code-for-Tables](./Code-for-Tables/) : Scripts to generate the Tables from the papers as well as some additional ones that were excluded from it.

## Aditional considerations

It is important to note that the version of the MRE used here corresponds to its nondimensional form and therefore all velocities, distances and times taken as an input are considered to be nondimensional.

In the case of realistic experiments in which units play a key role, the user of the toolbox will require to make sure that the velocity input from the field as well as the lengths and times are nondimensional. This matter is worked in a paper that will soon be published between Daniel Ruprecht, Alexandra von Kameke, Kathrin Padberg-gehle and me. 

## Acknowledgements

This repository is a result of the work carried out by 
[ M. Sc. Julio URIZARNA-CARASA](https://www.mat.tuhh.de/home/jurizarna_en) and [Prof. Dr. Daniel RUPRECHT](https://www.mat.tuhh.de/home/druprecht/?homepage_id=druprecht), from Hamburg University of Technology.

<p align="center">
  <img src="./Logos/tuhh-logo.png" height="55"/> &nbsp;&nbsp;&nbsp;&nbsp;
</p>

We are grateful to [Prof. Dr. Alexandra VON KAMEKE](https://www.haw-hamburg.de/hochschule/beschaeftigte/detail/person/person/show/alexandra-von-kameke/) for providing us with the data from the experimental
measurements of the Faraday flow and to [Prof. Dr. Kathrin PADBERG-GEHLE](https://www.leuphana.de/institute/imd/personen/kathrin-padberg-gehle.html) for providing the Bickley jet example.

This project is funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) - SFB 1615 - 503850735.

<p align="center">
  <img src="./Logos/tu_SMART_LOGO_02.jpg" height="105"/> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
</p>
