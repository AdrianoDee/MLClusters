#!/bin/sh

key=$(($1))
subkey=$(($2))
thepath=${3:"/lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/"}

export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch/
source $VO_CMS_SW_DIR/cmsset_default.sh
export SCRAM_ARCH=slc6_amd64_gcc530

X509_USER_PROXY=/lustre/home/adrianodif/proxyCMS

source /lustre/home/adrianodif/cms.sh

cd /lustre/home/adrianodif/CMSSW_9_0_0_pre4/src/
cmsenv
cd /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/

#eosPath="root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_9_0_0_pre4/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2017_realistic_v6_PU50-v1/10000/"



echo "Starting dataset production with $key key"

fileName="${key}_${subkey}_runs"

mkdir $fileName
cd $fileName
mkdir DataFiles
#	mkdir RootFiles/Doublets/
#	mkdir RootFiles/Datasets/
#        inFile="$eosPath$fileNameR"

slept=$((1 + RANDOM % 10))
sleep $slept
echo "Slept for $slept s for randomness"

echo "Running STEP1"
cmsRun /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/TTbar_13TeV_TuneCUETP8M1_cfi_GEN_SIM_PU.py > step1.log 2>&1

wait
#sleep 60
echo "Running STEP2"
cmsRun /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/step2_DIGI_L1_DIGI2RAW_HLT_PU.py > step2.log 2>&1

wait
echo "Running STEP3"
cmsRun /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/step3_RAW2DIGI_L1Reco_RECO_EI_PAT_VALIDATION_DQM_PU.py > step3.log 2>&1

wait

sleep 30

echo "Sorting, matching, hFiving, zipping ..."

cd /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/$fileName/DataFiles
echo $PWD
dub=$(ls *doublets*.txt)

for d in $dub
do
  f=${d%".txt"}
  f="${f}C.txt"
  sort -u $d > $f
  echo $f
done

wait

echo " - sorted"

cd /lustre/home/adrianodif/CMSSW_9_0_0_pre4/src/
cmsenv
cd /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/$fileName/DataFiles
echo $PWD

mat=$(ls *matchC.txt)

for m in $mat
do
  /lustre/home/adrianodif/Git/MLClustersDebug/cmssw/DoubletsMatcher -p $m
done

wait

echo " - matched"

cd /lustre/home/adrianodif/CMSSW_9_2_7/src/
cmsenv
cd /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/$fileName/DataFiles/

echo "-h5"
python2.7 /lustre/home/adrianodif/Git/MLClustersDebug/cmssw/cmsswToH5.py --read /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/$fileName/DataFiles/

wait

cd /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/$fileName/DataFiles
echo $PWD

gzip -rf -9 *.txt

echo "- zipped "

echo "All done. Good luck!"

cd
