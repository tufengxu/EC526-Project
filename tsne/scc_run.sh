#! /bin/bash -l
#$ -P paralg
#$ -pe omp 16
#$ -N tsne_16c

OMP_NUM_THREADS=$NSLOTS

module load gcc
module load anaconda3

exec >  ${SGE_O_WORKDIR}/${JOB_NAME}-${JOB_ID}.scc.out 2>&1

make

exit