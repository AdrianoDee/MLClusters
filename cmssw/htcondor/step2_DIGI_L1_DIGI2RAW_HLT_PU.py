# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: step2 --nThreads 4 --conditions auto:phase1_2017_realistic --pileup AVE_35_BX_25ns --pileup_input das:/RelValMinBias_13/CMSSW_8_1_0-81X_upgrade2017_realistic_v26-v1/GEN-SIM -s DIGI:pdigi_valid,L1,DIGI2RAW,HLT:@relval2016 --datatier GEN-SIM-DIGI-RAW -n 1 --geometry DB:Extended --eventcontent FEVTDEBUGHLT --filein file:step1.root --fileout file:step2.root --geometry DB:Extended --era Run2_2017_trackingPhase1CA
import FWCore.ParameterSet.Config as cms

from Configuration.StandardSequences.Eras import eras

process = cms.Process('HLT',eras.Run2_2017_trackingPhase1CA)

#randomizer for HTCondor

import time
import random

millis = int(round(time.time())*random.random()*random.random()*random.uniform(2,5))

while millis > 900000000:
        millis = millis * random.uniform(0.0,0.5)

millis = int(millis)
print millis

#

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Digi_cff')
process.load('Configuration.StandardSequences.SimL1Emulator_cff')
process.load('Configuration.StandardSequences.DigiToRaw_cff')
process.load('HLTrigger.Configuration.HLT_GRun_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(-1)
)

# Input source
process.source = cms.Source("PoolSource",
    dropDescendantsOfDroppedBranches = cms.untracked.bool(False),
    fileNames = cms.untracked.vstring('file:step1.root'),
    inputCommands = cms.untracked.vstring('keep *',
        'drop *_genParticles_*_*',
        'drop *_genParticlesForJets_*_*',
        'drop *_kt4GenJets_*_*',
        'drop *_kt6GenJets_*_*',
        'drop *_iterativeCone5GenJets_*_*',
        'drop *_ak4GenJets_*_*',
        'drop *_ak7GenJets_*_*',
        'drop *_ak8GenJets_*_*',
        'drop *_ak4GenJetsNoNu_*_*',
        'drop *_ak8GenJetsNoNu_*_*',
        'drop *_genCandidatesForMET_*_*',
        'drop *_genParticlesForMETAllVisible_*_*',
        'drop *_genMetCalo_*_*',
        'drop *_genMetCaloAndNonPrompt_*_*',
        'drop *_genMetTrue_*_*',
        'drop *_genMetIC5GenJs_*_*'),
    secondaryFileNames = cms.untracked.vstring()
)

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('step2 nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGHLToutput = cms.OutputModule("PoolOutputModule",
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM-DIGI-RAW'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(10485760),
    fileName = cms.untracked.string('file:step2.root'),
    outputCommands = process.FEVTDEBUGHLTEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements
process.RandomNumberGeneratorService.generator.initialSeed = millis
process.mix.input.nbPileupEvents.averageNumber = cms.double(35.000000)
process.mix.bunchspace = cms.int32(25)
process.mix.minBunch = cms.int32(-12)
process.mix.maxBunch = cms.int32(3)
process.mix.input.fileNames = cms.untracked.vstring(['root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/085B613E-A0BA-E611-98FF-0CC47A4D7628.root', 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/0AAC07F8-9CBA-E611-848B-0025905A6094.root', 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/503780F3-9DBA-E611-8CB7-0025905A60F8.root', 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/6E56E66E-9DBA-E611-A01B-0CC47A4C8E3C.root', 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/9C3F223C-A0BA-E611-83B6-0CC47A4D769A.root', 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/A662BA6F-9DBA-E611-9F01-0025905A4964.root', 'root://cms-xrd-global.cern.ch//store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/B65871F9-9CBA-E611-BCBF-0025905A6134.root'])
process.mix.digitizers = cms.PSet(process.theDigitizersValid)
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

# Path and EndPath definitions
process.digitisation_step = cms.Path(process.pdigi_valid)
process.L1simulation_step = cms.Path(process.SimL1Emulator)
process.digi2raw_step = cms.Path(process.DigiToRaw)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGHLToutput_step = cms.EndPath(process.FEVTDEBUGHLToutput)

# Schedule definition
process.schedule = cms.Schedule(process.digitisation_step,process.L1simulation_step,process.digi2raw_step)
process.schedule.extend(process.HLTSchedule)
process.schedule.extend([process.endjob_step,process.FEVTDEBUGHLToutput_step])

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(4)
process.options.numberOfStreams=cms.untracked.uint32(0)

# customisation of the process.

# Automatic addition of the customisation function from HLTrigger.Configuration.customizeHLTforMC
from HLTrigger.Configuration.customizeHLTforMC import customizeHLTforFullSim

#call to customisation function customizeHLTforFullSim imported from HLTrigger.Configuration.customizeHLTforMC
process = customizeHLTforFullSim(process)

# End of customisation functions

# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
