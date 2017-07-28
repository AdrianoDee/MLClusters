# Auto generated configuration file
# using:
# Revision: 1.19
# Source: /local/reps/CMSSW/CMSSW/Configuration/Applications/python/ConfigBuilder.py,v
# with command line options: TTbar_13TeV_TuneCUETP8M1_cfi --pileup_input das:/RelValMinBias_13/CMSSW_8_1_0-81X_upgrade2017_realistic_v26-v1/GEN-SIM --pileup AVE_35_BX_25ns --nThreads 4 --conditions auto:phase1_2017_realistic -n 1 --era Run2_2017 --eventcontent FEVTDEBUG --relval 9000,50 -s GEN,SIM --datatier GEN-SIM --beamspot Realistic50ns13TeVCollision --geometry DB:Extended --fileout file:step1.root
import FWCore.ParameterSet.Config as cms

#randomizer for HTCondor

import time
import random

millis = int(round(time.time())*random.random()*random.random()*random.uniform(2,5))

while millis > 900000000:
	millis = millis * random.uniform(0.0,0.5)

millis = int(millis)
print millis

from Configuration.StandardSequences.Eras import eras

process = cms.Process('SIM',eras.Run2_2017)

# import of standard configurations
process.load('Configuration.StandardSequences.Services_cff')
process.load('SimGeneral.HepPDTESSource.pythiapdt_cfi')
process.load('FWCore.MessageService.MessageLogger_cfi')
process.load('Configuration.EventContent.EventContent_cff')
process.load('SimGeneral.MixingModule.mix_POISSON_average_cfi')
process.load('Configuration.StandardSequences.GeometryRecoDB_cff')
process.load('Configuration.StandardSequences.GeometrySimDB_cff')
process.load('Configuration.StandardSequences.MagneticField_cff')
process.load('Configuration.StandardSequences.Generator_cff')
process.load('IOMC.EventVertexGenerators.VtxSmearedRealistic50ns13TeVCollision_cfi')
process.load('GeneratorInterface.Core.genFilterSummary_cff')
process.load('Configuration.StandardSequences.SimIdeal_cff')
process.load('Configuration.StandardSequences.EndOfProcess_cff')
process.load('Configuration.StandardSequences.FrontierConditions_GlobalTag_cff')

process.maxEvents = cms.untracked.PSet(
    input = cms.untracked.int32(1)
)

# Input source
process.source = cms.Source("EmptySource")

process.options = cms.untracked.PSet(

)

# Production Info
process.configurationMetadata = cms.untracked.PSet(
    annotation = cms.untracked.string('TTbar_13TeV_TuneCUETP8M1_cfi nevts:1'),
    name = cms.untracked.string('Applications'),
    version = cms.untracked.string('$Revision: 1.19 $')
)

# Output definition

process.FEVTDEBUGoutput = cms.OutputModule("PoolOutputModule",
    SelectEvents = cms.untracked.PSet(
        SelectEvents = cms.vstring('generation_step')
    ),
    dataset = cms.untracked.PSet(
        dataTier = cms.untracked.string('GEN-SIM'),
        filterName = cms.untracked.string('')
    ),
    eventAutoFlushCompressedSize = cms.untracked.int32(5242880),
    fileName = cms.untracked.string('file:step1.root'),
    outputCommands = process.FEVTDEBUGEventContent.outputCommands,
    splitLevel = cms.untracked.int32(0)
)

# Additional output definition

# Other statements

process.RandomNumberGeneratorService.generator.initialSeed = millis
process.mix.input.nbPileupEvents.averageNumber = cms.double(35.000000)
process.mix.bunchspace = cms.int32(25)
process.mix.minBunch = cms.int32(-12)
process.mix.maxBunch = cms.int32(3)
process.mix.input.fileNames = cms.untracked.vstring(['root://cms-xrd-global.cern.ch///store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/085B613E-A0BA-E611-98FF-0CC47A4D7628.root', '/store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/0AAC07F8-9CBA-E611-848B-0025905A6094.root', '/store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/503780F3-9DBA-E611-8CB7-0025905A60F8.root', '/store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/6E56E66E-9DBA-E611-A01B-0CC47A4C8E3C.root', '/store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/9C3F223C-A0BA-E611-83B6-0CC47A4D769A.root', '/store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/A662BA6F-9DBA-E611-9F01-0025905A4964.root', '/store/relval/CMSSW_8_1_0/RelValMinBias_13/GEN-SIM/81X_upgrade2017_realistic_v26-v1/10000/B65871F9-9CBA-E611-BCBF-0025905A6134.root'])
process.XMLFromDBSource.label = cms.string("Extended")
process.genstepfilter.triggerConditions=cms.vstring("generation_step")
from Configuration.AlCa.GlobalTag import GlobalTag
process.GlobalTag = GlobalTag(process.GlobalTag, 'auto:phase1_2017_realistic', '')

process.generator = cms.EDFilter("Pythia8GeneratorFilter",
    PythiaParameters = cms.PSet(
        parameterSets = cms.vstring('pythia8CommonSettings',
            'pythia8CUEP8M1Settings',
            'processParameters'),
        processParameters = cms.vstring('Top:gg2ttbar = on ',
            'Top:qqbar2ttbar = on ',
            '6:m0 = 175 '),
        pythia8CUEP8M1Settings = cms.vstring('Tune:pp 14',
            'Tune:ee 7',
            'MultipartonInteractions:pT0Ref=2.4024',
            'MultipartonInteractions:ecmPow=0.25208',
            'MultipartonInteractions:expPow=1.6'),
        pythia8CommonSettings = cms.vstring('Tune:preferLHAPDF = 2',
            'Main:timesAllowErrors = 10000',
            'Check:epTolErr = 0.01',
            'Beams:setProductionScalesFromLHEF = off',
            'SLHA:keepSM = on',
            'SLHA:minMassSM = 1000.',
            'ParticleDecays:limitTau0 = on',
            'ParticleDecays:tau0Max = 10',
            'ParticleDecays:allowPhotonRadiation = on')
    ),
    comEnergy = cms.double(13000.0),
    filterEfficiency = cms.untracked.double(1.0),
    maxEventsToPrint = cms.untracked.int32(0),
    pythiaHepMCVerbosity = cms.untracked.bool(False),
    pythiaPylistVerbosity = cms.untracked.int32(0)
)


process.ProductionFilterSequence = cms.Sequence(process.generator)

# Path and EndPath definitions
process.generation_step = cms.Path(process.pgen)
process.simulation_step = cms.Path(process.psim)
process.genfiltersummary_step = cms.EndPath(process.genFilterSummary)
process.endjob_step = cms.EndPath(process.endOfProcess)
process.FEVTDEBUGoutput_step = cms.EndPath(process.FEVTDEBUGoutput)

# Schedule definition
process.schedule = cms.Schedule(process.generation_step,process.genfiltersummary_step,process.simulation_step,process.endjob_step,process.FEVTDEBUGoutput_step)

#Setup FWK for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(4)
process.options.numberOfStreams=cms.untracked.uint32(0)
# filter all path with the production filter sequence
for path in process.paths:
	getattr(process,path)._seq = process.ProductionFilterSequence * getattr(process,path)._seq


# Customisation from command line

# Add early deletion of temporary data products to reduce peak memory need
from Configuration.StandardSequences.earlyDeleteSettings_cff import customiseEarlyDelete
process = customiseEarlyDelete(process)
# End adding early deletion
