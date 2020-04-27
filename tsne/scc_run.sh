#! /bin/bash -l
#$ -P paralg
#$ -pe omp 36
#$ -N tsne_36c

OMP_NUM_THREADS=$NSLOTS

module load gcc
module load anaconda3

exec >  ${SGE_O_WORKDIR}/${JOB_NAME}-${JOB_ID}.scc.out 2>&1

make scc
# make scc_no_parallel

exit