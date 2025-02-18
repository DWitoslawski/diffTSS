export LC_ALL=C
#
#zcat Roadmap_FantomAnnotations.InputData.pM10Kb.K562expr.txt.gz | awk '{print $1}' > Roadmap_FantomAnnotations.ENSID.txt
#
#singularity exec /data2/han_lab/singularity/singularity.ensembl/ensembl.sif perl /opt/ensembl-tools/scripts/id_history_converter/IDmapper.pl -s human -f Roadmap_FantomAnnotations.ENSID.txt > /tmp/Xpresso.newID
#awk -F"." '{print $1"\t"$2}' /tmp/Xpresso.newID  | awk '{print $1"\t"$3}' | sort | uniq > Xpresso.mapping
#
#zcat Roadmap_FantomAnnotations.InputData.pM10Kb.K562expr.txt.gz | awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10}' > Roadmap_FantomAnnotations.K562.txt
#zcat Roadmap_FantomAnnotations.InputData.pM10Kb.GM12878expr.txt.gz | awk '{print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10}' > Roadmap_FantomAnnotations.GM12878.txt
#python replaceID.Xpresso.py Roadmap_FantomAnnotations.K562.txt Xpresso.mapping  Roadmap_FantomAnnotations.K562.newID
#python replaceID.Xpresso.py Roadmap_FantomAnnotations.GM12878.txt Xpresso.mapping  Roadmap_FantomAnnotations.GM12878.newID
#
sort Roadmap_FantomAnnotations.K562.newID > /tmp/RNA.newID
awk '{print $1}' /data8/han_lab/mhan/diffTSS/data/bw/K562_GM12878_CAGE.txt | awk -F"_" '{print $1}' > /tmp/ID
paste /tmp/ID bw/K562_GM12878_CAGE.txt > /tmp/CAGE.txt
sort /tmp/CAGE.txt > /tmp/CAGE.sort
join -t "	"  -a 1 /tmp/CAGE.sort /tmp/RNA.newID > /tmp/RNA_CAGE.txt
#awk '{print }' /tmp/RNA_CAGE.txt > RNA_CAGE.txt

awk '{ 
  if (NF == 4) 
    print $2"\t"$3"\t"$4"\t0\t0\t0\t0\t0\t0\t0\t0\t0"; 
  else if (NF == 14) 
    print $2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$7"\t"$8"\t"$9"\t"$10"\t"$12"\t"$13"\t"$14; 
}' /tmp/RNA_CAGE.txt  > RNA_CAGE.txt

