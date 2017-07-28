#!/bin/sh

export VO_CMS_SW_DIR=/cvmfs/cms.cern.ch/
source $VO_CMS_SW_DIR/cmsset_default.sh
export SCRAM_ARCH=slc6_amd64_gcc530

X509_USER_PROXY=/lustre/home/adrianodif/proxyCMS

source /lustre/home/adrianodif/cms.sh

cd /lustre/home/adrianodif/CMSSW_9_0_0_pre4/src/
cmsenv
cd /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/

#eosPath="root://eoscms.cern.ch//eos/cms/store/relval/CMSSW_9_0_0_pre4/RelValTTbar_13/GEN-SIM-DIGI-RAW/PU25ns_90X_upgrade2017_realistic_v6_PU50-v1/10000/"

noEvents=$(($1))

echo "Starting $noRuns dataset production with $noEvents events"

        fileName="${1}_${2}_runs"
		
	mkdir $fileName
	cd $fileName
	mkdir DataFiles 
#	mkdir RootFiles/Doublets/
#	mkdir RootFiles/Datasets/
#        inFile="$eosPath$fileNameR" 
	
	slept=$((1 + RANDOM % 10))
	sleep $slept
	echo "Slept for $slept s"
	
	echo "Unzipping,hFiving, . . ."
	
	cd /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/$fileName/DataFiles/
	
	echo "-unzipping"
	gzip -d *.gz
		

	cd /lustre/home/adrianodif/CMSSW_9_2_7/src/
	cmsenv
        cd /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/
  	
	cd /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/$fileName/DataFiles
	
	echo "-h5"
        python2.7 /lustre/home/adrianodif/Git/MLClustersDebug/cmssw/cmsswToH5.py 
	
	wait	
	
        cd /lustre/home/adrianodif/HTCondor/doubletsPU_RND_2/$fileName/DataFiles
        echo $PWD

        gzip -rf -9 *.txt

        echo "- zipped "
	
        echo "All done. Good luck!"


 



