#!/bin/bash
#SBATCH --job-name=job1                      # Specify job name
#SBATCH --partition=helio                    # Specify partition name
#SBATCH --account=helio                      # Use assigned project account
#SBATCH --qos=helio_default                  # Select QOS assigned to user
#SBATCH --ntasks=1                           # Specify max. number of tasks to be invoked 
#SBATCH --cpus-per-task=10 		     # Specify number of CPUs per task
#SBATCH --time=05:00:00                      # Set expected time limit for job
#SBATCH --mail-type=START,END,FAIL           # Notify user by email upon job start, end and failure
#SBATCH --mail-user=oliveira@mps.mpg.de      # your e-mail address
##SBATCH --error=serial_%s.e                 # Capture errors to file 

# Environment preparation
export MODULESHOME='/usr/share/Modules/3.2.10'

. $MODULESHOME/init/bash

export OMP_NUM_THREADS=1

# run job
srun python3 create_dataset_batch_job.py
