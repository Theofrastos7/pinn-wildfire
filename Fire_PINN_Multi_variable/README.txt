PINN fire dispersion multi-variable problem

Files:
functions.py
main.py
create_fire_dataset.py
mypinn.py
mypinn_inverse.py


The file main.py runs a simulation of fire dispersion (non-dimensional) using the functions of functions.py.
For a multi-variable problem we need to seperate the grid according to the number of parameters we want. For this example,
I am using four different values of Lambda and Beta parameters, therefore the grid is seperated in four quadrants. The top right is
Q1, top left Q2, bottom left Q3 and bottom right Q4.

Main.py uses scaled version of the original lambda and beta parameters (Lines in main.py 36-66), for the lamb1-4 and beta1-4 
used in each quadrant. (lamb1 + beta1 -> Q1, lamb2 + beta2 -> Q2, etc). 
This is used because the lambdas and betas parameters in each quadrant should not diverge to much from the original values. 
These betas and lambdas variables are used in the Diffusion-Reaction step (functions.py, lines 82-103) to produce the simulation 
with each quadrant having its own values. 

The create_fire_dataset.py produces and saves a dataset in the folder graph, using the same functions and logic as main.py.
It also saves all the betas and lambdas values in the csv, so they can be accessed from mypinn.py upon reading the file.
There is already a dataset file in the folder data, which can be replaced by a new one upon running the create_fire_dataset.py.

The mypinn.py now uses this information in the pdes calculation. For each collocation point, depending on its position, or in 
other words the quadrant it is located in, the according values of beta and lambda are used (mypinn.py, lines 223-240).
The rest of the pinn is the exact same as before.

The mypinn_inverse.py does not change much from mypinn.py. The major difference is that the lambdas and betas for each quadrant, 
are now not known but rather they are trainables parameters of the pinn (mypinn_inverse.py, lines 115-122 + 143-146). Each time the 
PINN's loss is calucalted, the new values of the betas and lamdas are appended in their history (lines 294-301). Finally, a plot is 
produced at the end of the programm that displays the divergence of each lambda and beta. 