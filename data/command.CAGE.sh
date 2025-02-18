#!/bin/bash
set -eux
export LC_ALL=C
#
#data_path='/data8/han_lab/mhan/diffTSS/data/'
#bam_folder=${data_path}/CAGE.fantom.bam
#bw_folder=${data_path}/bw
#w5_folder=${data_path}/w5
#
#mkdir -p ${bw_folder}
#mkdir -p ${w5_folder}
# 
#for file in ${bam_folder}/*.bam
#do
#  bam=`basename ${file}`
#  echo ${bam}
#  bamCoverage --binSize 1 --normalizeUsin
#


awk -F"\t" '{print $1"\t"($11-64)"\t"($11+64)"\t"$5"\t"$14"\t"$4}' multiTSS.refTSS.ENSEMBL.merged.newID.bed > TSS.bed

computeMatrix reference-point -S K562+.bw -R TSS.bed -a 384 -b 256 --binSize 128 --averageTypeBins sum --referencePoint center -o K562_TSS.gz --outFileNameMatrix K562_values_TSS.tab --outFileSortedRegions K562.bed
plotProfile -m K562_TSS.gz -out K562_TSS_profile.png
plotHeatmap  -m K562_TSS.gz       -out K562_TSS_Heatmap.png

computeMatrix reference-point -S K562.rep1+.bw -R TSS.bed -a 384 -b 256 --binSize 128 --averageTypeBins sum --referencePoint center -o K562.rep1_TSS.gz --outFileNameMatrix K562.rep1_values_TSS.tab --outFileSortedRegions K562.rep1.bed
plotProfile -m K562.rep1_TSS.gz -out K562.rep1_TSS_profile.png
plotHeatmap  -m K562.rep1_TSS.gz       -out K562.rep1_TSS_Heatmap.png

computeMatrix reference-point -S K562.rep2+.bw -R TSS.bed -a 384 -b 256 --binSize 128 --averageTypeBins sum --referencePoint center -o K562.rep2_TSS.gz --outFileNameMatrix K562.rep2_values_TSS.tab --outFileSortedRegions K562.rep2.bed
plotProfile -m K562.rep2_TSS.gz -out K562.rep2_TSS_profile.png
plotHeatmap  -m K562.rep2_TSS.gz       -out K562.rep2_TSS_Heatmap.png

computeMatrix reference-point -S K562.rep3+.bw -R TSS.bed -a 384 -b 256 --binSize 128 --averageTypeBins sum --referencePoint center -o K562.rep3_TSS.gz --outFileNameMatrix K562.rep3_values_TSS.tab --outFileSortedRegions K562.rep3.bed
plotProfile -m K562.rep3_TSS.gz -out K562.rep3_TSS_profile.png
plotHeatmap  -m K562.rep3_TSS.gz       -out K562.rep3_TSS_Heatmap.png




computeMatrix reference-point -S GM12878.rep1+.bw -R TSS.bed -a 384 -b 256 --binSize 128 --averageTypeBins sum --referencePoint center -o GM12878.rep1_TSS.gz --outFileNameMatrix GM12878.rep1_values_TSS.tab --outFileSortedRegions GM12878.rep1.bed
plotProfile -m GM12878.rep1_TSS.gz -out GM12878.rep1_TSS_profile.png
plotHeatmap  -m GM12878.rep1_TSS.gz       -out GM12878.rep1_TSS_Heatmap.png

computeMatrix reference-point -S GM12878.rep2+.bw -R TSS.bed -a 384 -b 256 --binSize 128 --averageTypeBins sum --referencePoint center -o GM12878.rep2_TSS.gz --outFileNameMatrix GM12878.rep2_values_TSS.tab --outFileSortedRegions GM12878.rep2.bed
plotProfile -m GM12878.rep2_TSS.gz -out GM12878.rep2_TSS_profile.png
plotHeatmap  -m GM12878.rep2_TSS.gz       -out GM12878.rep2_TSS_Heatmap.png

computeMatrix reference-point -S GM12878.rep3+.bw -R TSS.bed -a 384 -b 256 --binSize 128 --averageTypeBins sum --referencePoint center -o GM12878.rep3_TSS.gz --outFileNameMatrix GM12878.rep3_values_TSS.tab --outFileSortedRegions GM12878.rep3.bed
plotProfile -m GM12878.rep3_TSS.gz -out GM12878.rep3_TSS_profile.png
plotHeatmap  -m GM12878.rep3_TSS.gz       -out GM12878.rep3_TSS_Heatmap.png



# K562 rep3 and GM12878 rep2
awk '{print $3+$4+$5}' K562.rep3_values_TSS.tab | sed '1,2d' > K562.rep3.sum
awk '{print $4}' K562.rep3.bed > /tmp/ID
paste /tmp/ID K562.rep3.sum | sort > /tmp/K562_values.sort
awk '{print $3+$4+$5}' GM12878.rep2_values_TSS.tab | sed '1,2d'  > GM12878.rep2.sum
awk '{print $4}' GM12878.rep2.bed > /tmp/ID
paste /tmp/ID GM12878.rep2.sum | sort > /tmp/GM12878_values.sort
awk -F"\t" '{print $5"\t"$6}' multiTSS.refTSS.ENSEMBL.merged.newID.bed | sort > /tmp/ENSG.sort

join -a 1 -e nan /tmp/ENSG.sort /tmp/K562_values.sort | awk '{print $3}' > /tmp/K562_values.txt
join -a 1 -e nan /tmp/ENSG.sort /tmp/GM12878_values.sort | awk '{print $3}' > /tmp/GM12878_values.txt
paste /tmp/ENSG.sort /tmp/K562_values.txt /tmp/GM12878_values.txt > /tmp/K562_GM12878_CAGE.txt
echo "ENSID	K562_CAGE_128*3_sum	GM12878_CAGE_128*3_sum" > K562_GM12878_CAGE.txt
cut -f2- /tmp/K562_GM12878_CAGE.txt | sed 1d >> K562_GM12878_CAGE.txt



