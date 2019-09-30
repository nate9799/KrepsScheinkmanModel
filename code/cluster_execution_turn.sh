
module load new gcc/4.8.2 python/2.7.14

bsub -n 1 -W 4:00 -R "rusage[mem=6000]" -J "Cournot_Turn [1-1700]" "python -W ignore turn_model.py \$LSB_JOBINDEX "
