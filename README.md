# MERRCh

## Installation

create an "external" folder, and download there the Eigen library https://gitlab.com/libeigen/eigen/-/releases/3.4.0
You can also install it in another folder, but you need to change CMakeLists accordingly.

Then compile with 

mkdir build
cmake --build build

Note that you must compile on a similar architecture as you will run (for example, on a cluster, start an interacting session and compile there).

## Use

./build/MERRCh --config MYCONFIG.cfg --ncores NSLOTS --data DATASET.csv --name MYNAME --additional_results

the cfg file contains most of the parameters, the dataset is given in argument, name corresponds to the name of the outputs (the date is automatically added). Additional_results allows to stor transformations and cognate apparitions

NSLOTS should be equal to the number of CPU cores.

## Output

Several text files. The console will print the ESS at each step and the elapsed time.

the NAME+date file contains a text file that includes the whole particle state at the end.

.nex file is only the set of trees as nexus file.

NAME+date_cfg.txt is the configuration file used for the simulation and the seed that was used.
