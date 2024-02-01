Dear reader,

in order to obtain Figure 05 in the paper, just open "a02_PLOTTR_DATA1.py" and run it.

The files that can be found in this folder are:

- 00_2mm_Faraday_50Hz_40ms_1_6g.mat: MATLAB file with the variables of the experimentally obtained velocity field.
		       
- a00_PMTERS_CONST.py: File that calculates some of the parameters of the paper, such as alpha, gamma, R, S, etc.
                       This file is used by the PRTCLE files.
                       
- a02_PLOTTR_DATA1.py: Script that should be run in order to obtain Figure 05 in our paper.
		       This plot defines some of the parameters of the particles as well as the numerical schemes, calculates the solutions and plots them.
		       
- a03_FIELD0_00000.py: File with the abstract class of a generic flow field.

- a03_FIELD0_DATA1.py: File with the flow field of the Faraday flow and the parameters of our paper.

- a09_PRTCLE_DTCHE.py: Script that has the class of a particle that given any field can calculate its trajectory using Forward Euler, Adam Bashford 2nd order and 3rd order methods, described in Daitche's paper.

- a09_PRTCLE_IMEX4.py: Script that has the class of a particle that given any field can calculate its trajectory using the IMEX methods defined in our paper. The update method within this class may calculate the trajectory of the particle with a first, second, third or fourth order method.

- a09_PRTCLE_TRAPZ.py: Script that has the class of a particle that given any field can calculate its trajectory using the Trapezoidal Rule. The update method within this class calculates the trajectory of the particle with a fourth order method, respectively.

The resulting plots are saved within the folder VISUAL_OUTPUT.
