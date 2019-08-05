work_path=$(dirname $0)
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:1 --ntasks-per-node=1 --cpus-per-task=1 \
python -u l3c.py ./logs 0524_0001 enc /mnt/lustre/zhengyaoyan/conpression/test.jpeg out.l3c

