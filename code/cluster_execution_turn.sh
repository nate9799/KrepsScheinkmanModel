
module load new gcc/4.8.2 python/2.7.14

_bsub -n 6 -W 24:00 -R "rusage[mem=6000]" -J "Cournot_Turn" "python -W ignore model_turn.py"
