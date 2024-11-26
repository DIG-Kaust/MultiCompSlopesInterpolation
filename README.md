![LOGO](https://github.com/DIG-Kaust/MultiCompSlopesInterpolation/blob/main/asset/logo.png)

``mcslopes`` is a Python library providing the fundamental building blocks to perform local slopes estimation and slope-assisted
data reconstruction of multi-component seismic data. It is primarily built on top of PyLops and contains both CPU and GPU versions
of each code (see ``notebooks`` and ``scripts`` folders). 

Note that the ``USE_CUPY`` variable can be used to switch between the CPU and GPU versions of each code. Also note that the CPU 
version of the ``overthrust3d`` example will be very slow compared to its GPU equivalent!

For more details refer to the accompanying paper **Multichannel wavefield reconstruction using smooth slopes information from multicomponent data -
Ravasi M., Ruan, J, and Vasconcelos I.** submitted to EAGE 2023.

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
- :orange_book: ``gom/Create_data.ipynb``: notebook showing how to synthetically create multi-channel, subsampled data to be fed to our reconstruction notebooks ans scripts;
- :orange_book: ``gom/Interpolation_GOM.ipynb``: notebook performing single-channel, multi-channel and slope-assisted 
  multi-channel data reconstruction on a 2D shot gather from the Missisipi Canyon data;
- :orange_book: ``overthrust3d/Create_data.ipynb``: notebook saving FD model data and correcting the particle velocity measurement to a pressure spatial gradient data;
- :orange_book: ``overthrust3d/Derivativechecking_Ove3D.ipynb``: notebook comparing 2 approaches to compute spatial derivatives of pressure data;
- :orange_book: ``overthrust3d/Interpolation_Ove3D.ipynb``: notebook performing single-channel, multi-channel and slope-assisted 
  multi-channel data reconstruction on a 3D shot gather modelled from the SEG/EAGE Overthrust model;
- :orange_book: ``overthrust3d/Interpolationwin_Ove3D.ipynb``: notebook performing single-channel, multi-channel and slope-assisted multi-channel data reconstruction with fk sliding windows on a 3D shot gather modelled from the SEG/EAGE Overthrust model;


## Scripts
The following scripts are provided:

- :orange_book: ``2d/Interpolation_slopes.py``: script performing multi-channel interpolation with local slopes of a 2d shot gather;
- :orange_book: ``3d/Interpolation_slopes.py``: script performing multi-channel interpolation with local slopes of a 3d shot gather.

Both scripts require an input ``.npz`` file containing the following fields. 

For 2d case:

- :card_index: ``data``: 2-dimensional pressure data to interpolate of size ``nt x nx``.
- :card_index: ``grad1``: 2-dimensional first-order gradient data of size ``nt x nx``.
- :card_index: ``grad2``: 2-dimensional second-order gradient data of size ``nt x nx``.
- :card_index: ``t``: time axis of size ``nt``.
- :card_index: ``x``: sparsely sampled spatial axis of size ``nx``.
- :card_index: ``xorig``: finely sampled spatial axis of size ``nx_orig``. Note that ``subfactor = nx_orig / nx``, which is the subsampling factor of the data.

For 3d case:

- :card_index: ``data``: 3-dimensional pressure data to interpolate of size ``ny x nx x nt``
- :card_index: ``grad1``: 3-dimensional first-order gradient data to interpolate of size ``ny x nx x nt``
- :card_index: ``t``: time axis of size ``nt``.
- :card_index: ``x``: finely sampled inline axis of size ``nx``.
- :card_index: ``y``: sparsely sampled crossline axis of size ``ny``
- :card_index: ``yorig``: finely sampled crossline axis of size ``ny_orig``. Note that ``subfactor = ny_orig / ny``, which is the subsampling factor of the data.


In synthetic examples one can also provide a second ``.npz`` file containing the following fields (which will be used to compare the
true wavefield with the reconstructed one). 

For 2d case:

- :card_index: ``data``: 2-dimensional pressure data to interpolate of size ``nt x nx_orig``.
- :card_index: ``grad1``: 2-dimensional first-order gradient data of size ``nt x nx_orig``.
- :card_index: ``grad2``: 2-dimensional second-order gradient data of size ``nt x nx_orig``.
- :card_index: ``t``: time axis of size ``nt``.
- :card_index: ``x``: finely sampled spatial axis of size ``nx_orig``.

For 3d case:

- :card_index: ``data``: 3-dimensional pressure data to interpolate of size ``ny_orig x nx x nt``.
- :card_index: ``grad1``: 2-dimensional first-order gradient data of size ``ny_orig x nx x nt``.
- :card_index: ``t``: time axis of size ``nt``.
- :card_index: ``x``: finely sampled inline axis of size ``nx``.
- :card_index: ``y``: finely sampled crossline axis of size ``ny``.


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
