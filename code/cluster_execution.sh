
module load new gcc/4.8.2 python/2.7.14

bsub -n 12 -W 24:00 -J "Cournot[1-16]" "python -W ignore model.py \$LSB_JOBINDEX "