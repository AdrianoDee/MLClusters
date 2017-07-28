# ML Clusters

Under cmssw/ are stored the scripts and the CMSSW_9_0_0_pre4 addons to run the generation
of doublets with TTbar_13TeV_TuneCUETP8M1_cfi recipe [adjustable to any recipe] 

- **ml_cmsswPU_data** the core dataset producer bash script from cmssw. It outputs an hdf file with the doublets datasets for each event. Takes at least two arguments that are the jobId and the taskId for HTCondor;
- **jobCreator** just a simple script to create an HTC condor job to run the production on HTC;
- **DoubletsMatcher** the script that matches doublets with recoTracks. Takes as input the txt file of doublets and matched hits coming out from customised CMSSW_9_0_0_pre4 cmsRun;
- **cmsswToH5** a simple script that converts datasets from txt to hdf;
