#!/bin/bash


module purge 

module load LANG/PYTHON/3.5.2-SYSTEM
module load TOOLS/THEANO/0.8.2-CPU-PY352
module load LIB/BOOST/1.65.1-PY352
module load LIB/OPENCV/2.4.13-CPUONLY


setenv ROBOSCHOOL_PATH /local-scratch/playground/roboschool

### Terrain RL stuff
setenv TERRAINRL_PATH /local-scratch/playground/TerrainRL
setenv LD_LIBRARY_PATH ${LD_LIBRARY_PATH}:/local-scratch/playground/TerrainRL/lib

