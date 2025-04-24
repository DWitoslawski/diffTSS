#!/bin/bash

print_usage(){
	echo "Usage: bash get_results_from_slurm.sh [-t ('cell' or 'rna')] [-r rna_slurm.out] [-n no_rna_slurm.out] [-c cell_type]"
}

if [[ "$#" -lt 6 ]]; then
    print_usage
    exit 1
fi

while getopts "t:r:n:c:k:g:" arg; do
        case "$arg" in
		t ) eval_type="$OPTARG";;
		r ) rna="$OPTARG";;
                n ) no_rna="$OPTARG";;
		c ) cell="$OPTARG";;
		k ) k562="$OPTARG";;
		g ) gm12878="$OPTARG";;
        esac
done

if [ $eval_type = "rna" ]
then
	grep 'PearsonR' $rna | awk '{ print $2 }' > ${cell}_rna_encoding_training.txt
	grep 'PearsonR' $no_rna | awk '{ print $2 }' > ${cell}_no_rna_encoding_training.txt
	echo "Average Pearson R for rna_encoding: $(awk 'BEGIN{s=0;}{s=s+$1;}END{print s/NR}' ${cell}_rna_encoding_training.txt)"
	echo "Average Pearson R for no_rna_encoding: $(awk 'BEGIN{s=0;}{s=s+$1;}END{print s/NR}' ${cell}_no_rna_encoding_training.txt)"
fi

if [ $eval_type = "cell" ]
then	
    grep 'PearsonR' $k562 | awk '{ print $2 }' > k562_rna_encoding_training.txt
    grep 'PearsonR' $gm12878 | awk '{ print $2 }' > gm12878_rna_encoding_training.txt
    echo "Average Pearson R for k652 rna_encoding: $(awk 'BEGIN{s=0;}{s=s+$1;}END{print s/NR}' k562_rna_encoding_training.txt)"
    echo "Average Pearson R for gm12878 rna_encoding: $(awk 'BEGIN{s=0;}{s=s+$1;}END{print s/NR}' gm12878_rna_encoding_training.txt)"
fi


#paste -d',' ${cell}_rna_encoding_training.txt ${cell}_no_rna_encoding_training.txt > ${cell}_rna_encoding_v_no_encoding.txt
