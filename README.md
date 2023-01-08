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

- :orange_book: ``gom/Interpolation_GOM.ipynb``: notebook performing single-channel, multi-channel and slope-assisted 
  multi-channel data reconstruction on a 2D shot gather from the Missisipi Canyon data;
- :orange_book: ``overthrust3d/Interpolation_Ove3D.ipynb``: notebook performing single-channel, multi-channel and slope-assisted 
  multi-channel data reconstruction on a 3D shot gather modelled from the SEG/EAGE Overthrust model;


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
