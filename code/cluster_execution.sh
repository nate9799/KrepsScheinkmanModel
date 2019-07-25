
module load new gcc/4.8.2 python/2.7.14

bsub -n 6 -W 24:00 -R "rusage[mem=6000]" -J "Cournot[1-22]" "python -W ignore model.py \$LSB_JOBINDEX "
