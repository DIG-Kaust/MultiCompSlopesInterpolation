![LOGO](https://github.com/DIG-Kaust/Project_Template/blob/master/logo.png)

``mcslopes`` is a Python library providing the fundamental building blocks to perform local slopes estimation and slope-assisted
data reconstruction of multi-component seismic data. It is primarily built on top of PyLops and contains both CPU and GPU versions
of each code (see ``notebooks`` and ``scripts`` folder). 

Note that the ``USE_CUPY`` variable can be used to switch between the CPU and GPU versions of each code. Also note that the CPU 
version of the ``overthrust3d`` example will be very slow compared to its GPU equivalent!

For more details refer to the accompanying paper **Multi-component local slopes and thier application to wavefield reconstruction problems
Ravasi M., Vasconcelos I.** submitted to EAGE 2023.

## Project structure
This repository is organized as follows:

* :open_file_folder: **mcslopes**: python library containing routines for local slopes estimation and slope-assisted
data reconstruction of multi-component seismic data;
* :open_file_folder: **data**: folder containing data;
* :open_file_folder: **notebooks**: set of jupyter notebooks reproducing the experiments in the paper (see below for more details);
* :open_file_folder: **scripts**: set of python scripts used to run the slope-assisted 
  data reconstruction algorithm on any data of choise.

## Notebooks
The following notebooks are provided:

- :orange_book: ``others/Slopes_comparison_hyperbolic.ipynb``: notebook comparing different slope estimation methods
  on a set of hyperbolic events;
- :orange_book: ``others/NMO_gradient.ipynb``: notebook computing NMO of gradient data as in Robertsson et al., 2008;
- :orange_book: ``gom/Create_data.ipynb``: notebook showing how to synthetically create multi-channel, subsampled data to be fed to
  our reconstruction notebooks ans scripts;
- :orange_book: ``gom/Interpolation_GOM.ipynb``: notebook performing single-channel, multi-channel and slope-assisted 
  multi-channel data reconstruction on a 2D shot gather from the Missisipi Canyon data;
- :orange_book: ``overthrust3d/Interpolation_Ove3D.ipynb``: notebook performing single-channel, multi-channel and slope-assisted 
  multi-channel data reconstruction on a 3D shot gather modelled from the SEG/EAGE Overthrust model;


## Scripts
The following scripts are provided:

- :orange_book: ``2d/Interpolation_slopes.ipynb``: script performing multi-channel interpolation with local slopes of 2d shot gathers;

Both scripts require an input ``.npz`` file containing the following fields:

- :card_index: ``data``: 2-dimensional pressure data to interpolate of size ``nt x nx``.
- :card_index: ``grad1``: 2-dimensional first-order gradient data of size ``nt x nx``.
- :card_index: ``grad2``: 2-dimensional second-order gradient data of size ``nt x nx``.
- :card_index: ``t``: time axis of size ``nt``.
- :card_index: ``x``: sparsely sampled spatial axis of size ``nx``.
- :card_index: ``xorig``: finely sampled spatial axis of size ``nx_orig``. Note that ``subfactor = nx_orig / nx``, which is
  the subsampling factor of the data.

In synthetic examples one can also provide a second ``.npz`` file containing the following fields (which will be used to compare the
true wavefield with the reconstructed one):

- :card_index: ``data``: 2-dimensional pressure data to interpolate of size ``nt x nx_orig``.
- :card_index: ``grad1``: 2-dimensional first-order gradient data of size ``nt x nx_orig``.
- :card_index: ``grad2``: 2-dimensional second-order gradient data of size ``nt x nx_orig``.
- :card_index: ``t``: time axis of size ``nt``.
- :card_index: ``x``: finely sampled spatial axis of size ``nx_orig``.


## Getting started :space_invader: :robot:
To ensure reproducibility of the results, we suggest using the `environment.yml` file when creating an environment.

Simply run:
```
./install_env.sh
```
It will take some time, if at the end you see the word `Done!` on your terminal you are ready to go. After that you can simply install your package:
```
pip install .
```
or in developer mode:
```
pip install -e .
```

Remember to always activate the environment by typing:
```
conda activate mcslopes
```

Finally, to run tests simply type:
```
pytest
```

**Disclaimer:** All experiments have been carried on a Intel(R) Xeon(R) CPU @ 2.10GHz equipped with a single NVIDIA GEForce RTX 3090 GPU. Different environment 
configurations may be required for different combinations of workstation and GPU.
