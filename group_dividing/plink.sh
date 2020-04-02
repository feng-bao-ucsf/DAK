#!/bin/bash

#SBATCH -p hsph
#SBATCH -n 1
#SBATCH --mem 40000
#SBATCH -t 9999

module load plink/1.90-fasrc01

for i in $(cat list.txt)
do plink --bfile ../aftimp_Schi_AA_QC100 --extract $i --range --out Schi_pathway_$i --recodeA
done
