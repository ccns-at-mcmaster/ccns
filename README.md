# Data Reduction at CCNS

### This repository is a work in progress. It is being made public for purposes of collaboration and user feedback. In its current state, reduction results have not been validated, and it should not be relied upon.

This is the repository for data analysis and control system monitoring at the Canadian Centre for Neutron Science 
(CCNS) currently under construction
at the McMaster University Nuclear Reactor.

This package contains all data reduction algorithms necessary to reduce small angle
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

Users should use data reduction methods found in ccns/reduction/__init__.py. These are all the methods you need to 
produce a reduced data structure which can be saved to NXcanSAS format. You can import these methods using:

`from ccns.reduction import *`

I suggest you use reduction methods from ccns/reduction/gaussiandq.py which reduce SANS data via a Gaussian
approximation. An exact numerical method has been attempted in ccns/reduction/numericaldq.py however as of time of
writing, the computational load is too great to be practical and further development is on hold.

Reduced data returned by ccns.reduction.reduce_data is combined with the raw data by 
ccns.writers.nxcansas_writer.get_sasentry to return a dictionary which can be saved to NXcanSAS format.

ccns.writers contains class definitions for datawriters. Currently only a nexusformat writer exists. You can initialize
an nxcansas_writer and using its pre-defined methods, open a nexus file and add an entry by using an appropriately 
structured dictionary. 
```
writer = NXcanSASWriter()
writer.set_filename = "filename"
writer.open()
writer.add_entry(dictionary)
writer.close()
```

Any feedback on this repository may be directed to the authors:<br />
Devin Burke (burkeds@mcmaster.ca)<br />
Canadian Centre for Neutron Science (ccns@mcmaster.ca)
