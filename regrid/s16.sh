#!/bin/bash
#
#SBATCH -J CON_regrid
#SBATCH -p general
#SBATCH -o %j.txt
#SBATCH -e %j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gongg@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH -A r00599

# load Conda config
source /N/u/gongg/Quartz/anaconda3/etc/profile.d/conda.sh


# activate gongg
conda activate xesmf


# makesure env activate
which python
python --version

#Run your program
srun python CONUS404_regrid.py
