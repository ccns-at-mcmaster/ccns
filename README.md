# Data Reduction at CCNS

### This repository is a work in progress. It is being made public for purposes of collaboration and user feedback. In its current state, reduction results have not been validated, and it should not be relied upon.

This is the repository for data analysis and control system monitoring at the Canadian Centre for Neutron Science 
(CCNS) currently under construction
at the McMaster University Nuclear Reactor.

This repo contains the python package ccns which contains all data reduction algorithms necessary to reduce small angle
neutron scattering data obtained at the MacSANS laboratory.

To install this package simply install it via pip from the top directory:

`python setup.py install`

The ccns package is designed to load NXsas nexus files from our instrument control client. Data is extracted from
these files and used with methods found in ccns/reduction to reduce the data and output in NXcanSAS format compatible
with SASview.

For analysis using this package or in a Jupyter notebook, we provide the methods found in ccns/analysis and ccns/masking 
to mask or analyze data. Analysis methods exist to automatically generate common SANS data plots utilizing reduced data 
dictionaries loaded into a xarray. The xarray format was chosen because of its label-based selection capabilities. We 
are interested in seeing how intuitive this feels for our users.

We suggest we use reduction methods from ccns/reduction/gaussiandq.py which reduce SANS data via a Gaussian
approximation. An exact numerical method has been attempted in ccns/reduction/numericaldq.py however as of time of
writing, the computational load is too great to be practical and further development is on hold.

Any feedback on this repository may be directed to the authors:
Devin Burke (burkeds@mcmaster.ca)
Canadian Centre for Neutron Science (ccns@mcmaster.ca)
