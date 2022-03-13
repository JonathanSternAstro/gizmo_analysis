#!/bin/bash                                                                        

#SBATCH --job-name=movieFrame
#SBATCH --partition=cosmoshimem
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=2:00:00
#SBATCH --output=output/movieFrame_%j.out
#SBATCH --error=output/movieFrame_%j.err
#SBATCH --mail-user=jonathan.stern@northwestern.edu
#SBATCH --mail-type=fail
#SBATCH --mail-type=end
#SBATCH --account=b1026
                                                                                   

export PYTHONPATH=$PYTHONPATH:/opt/apps/intel18/impi18_0/python3/3.7.0/lib/python3.7/site-packages
                                                                                   
########################################################################           
# Input Arguments                                                                  
########################################################################           

# example usage
# sbatch movieFrame_script fixedZ_transonic5 100
                                                                                   
python ../pysrc/profiles_and_projections_script.py $1 $2

