
module load new gcc/4.8.2 python/2.7.14

bsub -n 6 -W 24:00 -R "rusage[mem=6000]" -J "Cournot_Turn [1-20]" "python -W ignore turn_model.py \$LSB_JOBINDEX "