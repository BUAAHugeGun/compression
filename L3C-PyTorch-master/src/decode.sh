work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=1 \
python -u l3c.py ./logs 0524_0001 dec out.l3c out.png

