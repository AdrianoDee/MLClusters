#include "RecoTracker/TkHitPairs/interface/HitPairGeneratorFromLayerPair.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"

#include "RecoTracker/TkTrackingRegions/interface/HitRZCompatibility.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/TrackingRegionBase.h"
#include "RecoTracker/TkHitPairs/interface/OrderedHitPairs.h"
#include "RecoTracker/TkHitPairs/src/InnerDeltaPhi.h"

#include "FWCore/Framework/interface/Event.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/SiPixelDetId/interface/PXBDetId.h"
#include "DataFormats/SiPixelDetId/interface/PXFDetId.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHit.h"

#include <iostream>
#include <string>
#include <fstream>

#include <TH2F.h>

using namespace GeomDetEnumerators;
using namespace std;

static bool doubsProduction = true;
// static std::map<std::pair < std::pair < std::pair<float,float>,std::pair<float,float> >, std::pair < std::pair< float, std::pair <float ,float>> , std::pair < float, std::pair <float ,float> > > >, std::vector<float>>  hitPairCache;
// typedef std::map<std::pair < std::pair < std::pair<float,float>,std::pair<float,float> >, std::pair < std::pair< float, std::pair <float ,float>> , std::pair < float, std::pair <float ,float> > > >, std::vector<float>>::iterator hitPairCacheIterator;


int detOnArr[10] = {0,1,2,3,14,15,16,29,30,31};
std::vector<int> detOn(detOnArr,detOnArr+sizeof(detOnArr)/sizeof(int));
// std::map<std::pair <int,std::pair<float,float> >, int> cacheMap;

typedef PixelRecoRange<float> Range;

namespace {
  template<class T> inline T sqr( T t) {return t*t;}
}


#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHitBuilder.h"
#include "TrackingTools/Records/interface/TransientRecHitRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"

HitPairGeneratorFromLayerPair::HitPairGeneratorFromLayerPair(
  unsigned int inner,
  unsigned int outer,
  LayerCacheType* layerCache,
  unsigned int max)
  : theLayerCache(layerCache), theOuterLayer(outer), theInnerLayer(inner), theMaxElement(max)
  {
  }

  HitPairGeneratorFromLayerPair::~HitPairGeneratorFromLayerPair() {}

  // devirtualizer
  #include<tuple>
  namespace {

    template<typename Algo>
    struct Kernel {
      using  Base = HitRZCompatibility;
      void set(Base const * a) {
        assert( a->algo()==Algo::me);
        checkRZ=reinterpret_cast<Algo const *>(a);
      }

      void operator()(int b, int e, const RecHitsSortedInPhi & innerHitsMap, bool * ok) const {
        constexpr float nSigmaRZ = 3.46410161514f; // std::sqrt(12.f);
        for (int i=b; i!=e; ++i) {
          Range allowed = checkRZ->range(innerHitsMap.u[i]);
          float vErr = nSigmaRZ * innerHitsMap.dv[i];
          Range hitRZ(innerHitsMap.v[i]-vErr, innerHitsMap.v[i]+vErr);
          Range crossRange = allowed.intersection(hitRZ);
          ok[i-b] = ! crossRange.empty() ;
        }
      }
      Algo const * checkRZ;

    };


    template<typename ... Args> using Kernels = std::tuple<Kernel<Args>...>;

  }


  void HitPairGeneratorFromLayerPair::hitPairs(
    const TrackingRegion & region, OrderedHitPairs & result,
    const edm::Event& iEvent, const edm::EventSetup& iSetup, Layers layers) {

      auto const & ds = doublets(region, iEvent, iSetup, layers);
      for (std::size_t i=0; i!=ds.size(); ++i) {
        result.push_back( OrderedHitPair( ds.hit(i,HitDoublets::inner),ds.hit(i,HitDoublets::outer) ));
      }
      if (theMaxElement!=0 && result.size() >= theMaxElement){
        result.clear();
        edm::LogError("TooManyPairs")<<"number of pairs exceed maximum, no pairs produced";
      }
    }

    HitDoublets HitPairGeneratorFromLayerPair::doublets( const TrackingRegion& region,
      const edm::Event & iEvent, const edm::EventSetup& iSetup, const Layer& innerLayer, const Layer& outerLayer,
      LayerCacheType& layerCache) {

        const RecHitsSortedInPhi & innerHitsMap = layerCache(innerLayer, region, iSetup);
        if (innerHitsMap.empty()) return HitDoublets(innerHitsMap,innerHitsMap);

        const RecHitsSortedInPhi& outerHitsMap = layerCache(outerLayer, region, iSetup);
        if (outerHitsMap.empty()) return HitDoublets(innerHitsMap,outerHitsMap);
        HitDoublets result(innerHitsMap,outerHitsMap); result.reserve(std::max(innerHitsMap.size(),outerHitsMap.size()));
        doublets(region,
          *innerLayer.detLayer(),*outerLayer.detLayer(),
          innerHitsMap,outerHitsMap,iSetup,theMaxElement,result);

          int eveNumber = iEvent.id().event();
          int runNumber = iEvent.id().run();
          int lumNumber = iEvent.id().luminosityBlock();

          int detSeqIn = innerLayer.detLayer()->seqNum();
          int detSeqOut = outerLayer.detLayer()->seqNum();

          float padHalfSize = 4.0;

          std::vector<int>::iterator detOnItOne = find(detOn.begin(),detOn.end(),innerLayer.detLayer()->seqNum());
          std::vector<int>::iterator detOnItTwo = find(detOn.begin(),detOn.end(),outerLayer.detLayer()->seqNum());

          // std::cout<<"clusters"<<innerLayer.name()<<" "<<outerLayer.name()<<std::endl;
          // std::pair<float,float> runEvt(runNumber,eveNumber);
          // std::pair<float,float> detPair(innerLayer.detLayer()->seqNum(),outerLayer.detLayer()->seqNum());
          // std::pair<int,std::pair<float,float> > evDetPair(eveNumber,detPair);
          // std::pair< std::pair<float,float> ,std::pair<float,float> > evtRunDets(runEvt,detPair);

          // std::map< std::pair<int,std::pair<float,float> >  ,int>::iterator itDets = cacheMap.find(evDetPair);

          // if(itDets!=cacheMap.end()) std::cout<<"Pair Det In : "<<innerLayer.detLayer()->seqNum()<<" Det Out: "<<outerLayer.detLayer()->seqNum()<<" already done. Skipping"<<std::endl;

          std::string fileName = "./DataFiles/" + std::to_string(lumNumber) +"_"+std::to_string(runNumber) +"_"+std::to_string(eveNumber) + "_doublets.txt";
          std::ofstream fDoublets(fileName, std::ofstream::app);

          // int dataSize = 24;
          // std::vector<float> zeros(dataSize,0.0);

          if(detOnItOne!=detOn.end() && detOnItTwo!=detOn.end() && result.size()!=0 && doubsProduction)
          {

            float inX, inY, inZ, outX, outY, outZ;
            int layerIn = 0, ladderIn = 0, moduleIn = 0, sideIn = 0, diskIn = 0, panelIn = 0, bladeIn = 0;
            int layerOut = 0, ladderOut = 0, moduleOut = 0, sideOut = 0, diskOut = 0, panelOut = 0, bladeOut = 0;
            int detCounterIn = -1, detCounterOut = -1;
            bool isBigIn = false, isEdgIn = false,isBigOut = false, isEdgOut = false,isBadIn = false,isBadOut = false,isBarrelIn = false,isBarrelOut = false;

            TH2F *innerCluster = 0, *outerCluster = 0;

            detCounterIn= (Int_t)(detOnItOne - detOn.begin());
            detCounterOut = (Int_t)(detOnItTwo - detOn.begin());

            // std::string fileName = "./RootFiles/Doublets/" + std::to_string(lumNumber) +"_"+std::to_string(runNumber) +"_"+std::to_string(eveNumber) + "_doublets.txt";
            // std::ofstream fDoublets(fileName, std::ofstream::out);

            for (size_t i = 0; i < result.size(); i++) {

              // cacheMap[evDetPair] = 1;

              int inId = result.innerHitId(i);
              int outId = result.outerHitId(i);

              layerIn  = 0;
              ladderIn = 0;
              moduleIn = 0;
              sideIn   = 0;
              diskIn   = 0;
              panelIn  = 0;
              bladeIn  = 0;

              layerOut  = 0;
              ladderOut = 0;
              moduleOut = 0;
              sideOut   = 0;
              diskOut   = 0;
              panelOut  = 0;
              bladeOut  = 0;

              RecHitsSortedInPhi::Hit innerHit = result.hit(i, HitDoublets::inner);
              RecHitsSortedInPhi::Hit outerHit = result.hit(i, HitDoublets::outer);

              inX = (innerHit->hit()->globalState()).position.x();
              outX = (outerHit->hit()->globalState()).position.x();
              inY = (innerHit->hit()->globalState()).position.y();
              outY = (outerHit->hit()->globalState()).position.y();
              inZ = (innerHit->hit()->globalState()).position.z();
              outZ = (outerHit->hit()->globalState()).position.z();

              // std::pair< float, float > xyIn((float)inX,(float)inY);
              // std::pair< float, float > xyOut((float)outX,(float)outY);
              // std::pair< float, std::pair< float, float > > zxyIn((float)inZ,xyIn);
              // std::pair< float, std::pair< float, float > > zxyOut((float)outZ,xyOut);
              // std::pair< std::pair< float, std::pair< float, float > >, std::pair< float, std::pair< float, float > > > xyzInOut (zxyIn,zxyOut);
              // std::pair< std::pair< std::pair<float,float> ,std::pair<float,float> > ,std::pair< std::pair< float, std::pair< float, float > >, std::pair< float, std::pair< float, float > > >> hitsID(evtRunDets,xyzInOut);
              DetId outerDetId = outerHit->hit()->geographicalId();
              DetId innerDetId = innerHit->hit()->geographicalId();


              unsigned int subidIn=innerDetId.subdetId();
              unsigned int subidOut=outerDetId.subdetId();

              if (! (((subidIn==1) || (subidOut==2)) && ((subidOut==1) || (subidOut==2)))) continue;

              PXBDetId pbdetIdIn(0), pbdetIdOut(0);
              PXFDetId pfdetIdIn(0), pfdetIdOut(0);

              //Barrel
              if(subidIn==1)
              {
                isBarrelIn = true;
                pbdetIdIn = PXBDetId(innerDetId);

                layerIn  = pbdetIdIn.layer();
                ladderIn = pbdetIdIn.ladder();
                moduleIn = pbdetIdIn.module();

              }
              if(subidOut==1)
              {
                isBarrelOut = true;
                pbdetIdOut = PXBDetId(outerDetId);

                layerOut  = pbdetIdOut.layer();
                ladderOut = pbdetIdOut.ladder();
                moduleOut = pbdetIdOut.module();

              }

              //Forward
              if(subidIn==2)
              {
                isBarrelIn = false;
                pfdetIdIn = PXFDetId(innerDetId);

                sideIn = pfdetIdIn.side();
                diskIn = pfdetIdIn.disk();
                bladeIn = pfdetIdIn.blade();
                panelIn = pfdetIdIn.panel();
                moduleIn = pfdetIdIn.module();


              }
              if(subidOut==2)
              {
                isBarrelOut = false;
                pfdetIdOut = PXFDetId(outerDetId);

                sideOut = pfdetIdOut.side();
                diskOut = pfdetIdOut.disk();
                bladeOut = pfdetIdOut.blade();
                panelOut = pfdetIdOut.panel();
                moduleOut = pfdetIdOut.module();


              }

              const SiPixelRecHit* siHitIn = dynamic_cast<const SiPixelRecHit*>((innerHit));
              const SiPixelRecHit* siHitOut = dynamic_cast<const SiPixelRecHit*>((outerHit));

              SiPixelRecHit::ClusterRef const& clusterIn = siHitIn->cluster();
              SiPixelRecHit::ClusterRef const& clusterOut = siHitOut->cluster();

              isBigIn = siHitIn->spansTwoROCs();
              isBadIn = siHitIn->hasBadPixels();
              isEdgIn = siHitIn->isOnEdge();

              isBigOut = siHitOut->spansTwoROCs();
              isBadOut = siHitOut->hasBadPixels();
              isEdgOut = siHitOut->isOnEdge();

              innerCluster = new TH2F("innerCluster->","innerCluster->",(Int_t)(padHalfSize*2),floor(clusterIn->x())-padHalfSize,floor(clusterIn->x())+padHalfSize,(Int_t)(padHalfSize*2),floor(clusterIn->y())-padHalfSize,floor(clusterIn->y())+padHalfSize);
              outerCluster = new TH2F("hitclusterOut","hitclusterOut",(Int_t)(padHalfSize*2),floor(clusterOut->x())-padHalfSize,floor(clusterOut->x())+padHalfSize,(Int_t)(padHalfSize*2),floor(clusterOut->y())-padHalfSize,floor(clusterOut->y())+padHalfSize);

              // std::cout<<innerCluster->GetNbinsX()<<outerCluster->GetNbinsX()<<std::endl;

              for (int nx = 0; nx < innerCluster->GetNbinsX(); ++nx)
              for (int ny = 0; ny < innerCluster->GetNbinsY(); ++ny)
              {
                innerCluster->SetBinContent(nx,ny,0.0);
                outerCluster->SetBinContent(nx,ny,0.0);
              }
              for (int k = 0; k < clusterIn->size(); ++k)
              innerCluster->SetBinContent(innerCluster->FindBin((double)clusterIn->pixel(k).x, (double)clusterIn->pixel(k).y),clusterIn->pixel(k).adc);

              for (int k = 0; k < clusterOut->size(); ++k)
              outerCluster->SetBinContent(outerCluster->FindBin((double)clusterOut->pixel(k).x, (double)clusterOut->pixel(k).y),clusterOut->pixel(k).adc);

              //	fileName = "./RootFiles/Doublets/" + std::to_string(lumNumber) +"_"+std::to_string(runNumber) +"_"+std::to_string(eveNumber) + "_doubletmap.txt";
              //    	std::ofstream fClustersMap(fileName, std::ofstream::out);
              std::vector<float> clustVec;

              clustVec.push_back(runNumber); //0
              clustVec.push_back(eveNumber); //1

              clustVec.push_back(detSeqIn); //2
              clustVec.push_back(detSeqOut); //3

              clustVec.push_back((float)(floor(inZ*1000.))/1000.); //4
              clustVec.push_back((float)(floor(inX*1000.))/1000.);
              clustVec.push_back((float)(floor(inY*1000.))/1000.);

              clustVec.push_back((float)(floor(outZ*1000.))/1000.); //7
              clustVec.push_back((float)(floor(outX*1000.))/1000.);
              clustVec.push_back((float)(floor(outY*1000.))/1000.);

              clustVec.push_back(detCounterIn);
              clustVec.push_back(detCounterOut);
              clustVec.push_back(isBarrelIn);
              clustVec.push_back(isBarrelOut);
              clustVec.push_back(layerIn);
              clustVec.push_back(ladderIn);
              clustVec.push_back(moduleIn);
              clustVec.push_back(sideIn);
              clustVec.push_back(diskIn);
              clustVec.push_back(panelIn);
              clustVec.push_back(bladeIn);
              clustVec.push_back(layerOut);
              clustVec.push_back(ladderOut);
              clustVec.push_back(moduleOut);
              clustVec.push_back(sideOut);
              clustVec.push_back(diskOut);
              clustVec.push_back(panelOut);
              clustVec.push_back(bladeOut);
              clustVec.push_back(inId);
              clustVec.push_back(outId);
              clustVec.push_back(isBigIn);
              clustVec.push_back(isEdgIn);
              clustVec.push_back(isBadIn);
              clustVec.push_back(isBigOut);
              clustVec.push_back(isEdgOut);
              clustVec.push_back(isBadOut);

              for (int nx = 0; nx < innerCluster->GetNbinsX(); ++nx)
              for (int ny = 0; ny < innerCluster->GetNbinsY(); ++ny)
              clustVec.push_back(innerCluster->GetBinContent(nx,ny));

              for (int nx = 0; nx < outerCluster->GetNbinsX(); ++nx)
              for (int ny = 0; ny < outerCluster->GetNbinsY(); ++ny)
              clustVec.push_back(outerCluster->GetBinContent(nx,ny));

              for (size_t i = 0; i < clustVec.size(); ++i)
                fDoublets << clustVec[i] << "\t";

              fDoublets << -5421369.3478427;

              // for (int i = 0; i < dataSize - 1; ++i)
              //   fDoublets << infos[i] << "\t";
              //
              // fDoublets << infos[dataSize-1];

              fDoublets << std::endl;//<< std::endl;

              clustVec.clear();


            }
          }

          // std::cout<<"Map size : "<<hitPairCache.size()<<std::endl;
/*
          for (hitPairCacheIterator it=hitPairCache.begin(); it!=hitPairCache.end(); ++it)
          {

            std::vector<float> theVec = it->second;

            // //		std::cout<<"HERE!"<<std::endl;
            fClustersMap << it->first.first.first.first << "\t" << it->first.first.first.second<< "\t";
            //hitsID.evtRunDets.runEvt.runNumber \t hitsID.evtRunDets.runEvt.eveNumber
            fClustersMap << it->first.first.second.first << "\t" << it->first.first.second.second<< "\t" ;
            //hitsID.evtRunDets.detPair.innerSeq \t hitsID.evtRunDets.detPair.outerSeq
            fClustersMap << it->first.second.first.first << "\t" << it->first.second.first.second.first << "\t" << it->first.second.first.second.second<< "\t";
            //hitsID.xyzInOut.zxyIn.inZ \t hitsID.xyzInOut.zxyIn.xyIn.inX \t hitsID.xyzInOut.zxyIn.xyIn.inY \t
            fClustersMap << it->first.second.second.first << "\t" << it->first.second.second.second.first << "\t" << it->first.second.second.second.second<< "\t";
            //hitsID.xyzInOut.zxyOut.outZ \t hitsID.xyzInOut.zxyOut.xyOut.outX \t hitsID.xyzInOut.zxyOut.xyOut.outY \t
            // //
            // // fDoublets << it->first.first.first.first << "\t" << it->first.first.first.second<< "\t";
            // // fDoublets << it->first.first.second.first << "\t" << it->first.first.second.second<< "\t";
            // // fDoublets << it->first.second.first.first << "\t" << it->first.second.first.second.first << "\t" << it->first.second.first.second.second<< "\t";
            // // fDoublets << it->first.second.second.first << "\t" << it->first.second.second.second.first << "\t" << it->first.second.second.second.second<< "\t";
            // // // fDoublets << std::endl << std::endl;
            //
            // fClustersMap<<detCounterIn<<"\t";
            // fClustersMap<<detCounterOut<<"\t";
            //
            // fClustersMap<<isBarrelIn<<"\t";
            // fClustersMap<<isBarrelOut<<"\t";
            //
            // fClustersMap<<layerIn<<"\t";
            // fClustersMap<<ladderIn<<"\t";
            // fClustersMap<<moduleIn<<"\t";
            // fClustersMap<<sideIn<<"\t";
            // fClustersMap<<diskIn<<"\t";
            // fClustersMap<<panelIn<<"\t";
            // fClustersMap<<bladeIn<<"\t";
            //
            // fClustersMap<<layerOut<<"\t";
            // fClustersMap<<ladderOut<<"\t";
            // fClustersMap<<moduleOut<<"\t";
            // fClustersMap<<sideOut<<"\t";
            // fClustersMap<<diskOut<<"\t";
            // fClustersMap<<panelOut<<"\t";
            // fClustersMap<<bladeOut<<"\t";
            //
            // fClustersMap<<inId<<"\t";
            // fClustersMap<<outId<<"\t";
            //
            // fClustersMap<<isBigIn<<"\t";
            // fClustersMap<<isEdgIn<<"\t";
            // fClustersMap<<isBadIn<<"\t";
            // fClustersMap<<isBigOut<<"\t";
            // fClustersMap<<isEdgOut<<"\t";
            // fClustersMap<<isBadOut<<"\t";

            for (size_t i = 0; i < theVec.size(); i++) {
              fClustersMap<<theVec[i]<<"\t";
            }

            //		std::cout << "CIAO" << std::endl;
            fClustersMap << -5421369.3478427;
            fClustersMap << std::endl;//<< std::endl;
          }

          // std::cout<<"Doublets : "<<hitPairCache.size()<<std::endl;



          */
          //	std::pair < std::pair < std::pair<float,float>,std::pair<float,float> >, std::pair < std::pair< float, std::pair <float ,float>> , std::pair < float, std::pair <float ,float> > > >
          //	<< it->first->first->first->first

          fDoublets.clear();
          fDoublets.close();




    //      if(itDets==cacheMap.end())
    //
    //     else
    //     {
    //     auto & buffer = hitPairCache[evDetPair];
    //     //HitDoublets buffer(&(*hitPairCache[evDetPair]));
    //
    //     for (std::size_t i=0; i!=result.size(); ++i) {
    //     buffer->add(  result.innerHitId(i), result.outerHitId(i));
    //   }
    //   hitPairCache[evDetPair] = buffer;
    //
    // }


    //  ifstream::pos_type size = sizeof(hitPairCache);
    // ofstream save1;
    //  save1.open("saved", ios::binary);


    return result;

  }

  void HitPairGeneratorFromLayerPair::doublets(const TrackingRegion& region,
    const DetLayer & innerHitDetLayer,
    const DetLayer & outerHitDetLayer,
    const RecHitsSortedInPhi & innerHitsMap,
    const RecHitsSortedInPhi & outerHitsMap,
    const edm::EventSetup& iSetup,
    const unsigned int theMaxElement,
    HitDoublets & result){

      //  HitDoublets result(innerHitsMap,outerHitsMap); result.reserve(std::max(innerHitsMap.size(),outerHitsMap.size()));
      typedef RecHitsSortedInPhi::Hit Hit;
      InnerDeltaPhi deltaPhi(outerHitDetLayer, innerHitDetLayer, region, iSetup);

      // std::cout << "layers " << theInnerLayer.detLayer()->seqNum()  << " " << outerLayer.detLayer()->seqNum() << std::endl;

      // constexpr float nSigmaRZ = std::sqrt(12.f);
      constexpr float nSigmaPhi = 3.f;
      for (int io = 0; io!=int(outerHitsMap.theHits.size()); ++io) {
        if (!deltaPhi.prefilter(outerHitsMap.x[io],outerHitsMap.y[io])) continue;
        Hit const & ohit =  outerHitsMap.theHits[io].hit();
        PixelRecoRange<float> phiRange = deltaPhi(outerHitsMap.x[io],
          outerHitsMap.y[io],
          outerHitsMap.z[io],
          nSigmaPhi*outerHitsMap.drphi[io]
        );

        if (phiRange.empty()) continue;

        const HitRZCompatibility *checkRZ = region.checkRZ(&innerHitDetLayer, ohit, iSetup, &outerHitDetLayer,
          outerHitsMap.rv(io),outerHitsMap.z[io],
          outerHitsMap.isBarrel ? outerHitsMap.du[io] :  outerHitsMap.dv[io],
          outerHitsMap.isBarrel ? outerHitsMap.dv[io] :  outerHitsMap.du[io]
        );
        if(!checkRZ) continue;

        Kernels<HitZCheck,HitRCheck,HitEtaCheck> kernels;

        auto innerRange = innerHitsMap.doubleRange(phiRange.min(), phiRange.max());
        LogDebug("HitPairGeneratorFromLayerPair")<<
        "preparing for combination of: "<< innerRange[1]-innerRange[0]+innerRange[3]-innerRange[2]
        <<" inner and: "<< outerHitsMap.theHits.size()<<" outter";
        for(int j=0; j<3; j+=2) {
          auto b = innerRange[j]; auto e=innerRange[j+1];
          bool ok[e-b];
          switch (checkRZ->algo()) {
            case (HitRZCompatibility::zAlgo) :
            std::get<0>(kernels).set(checkRZ);
            std::get<0>(kernels)(b,e,innerHitsMap, ok);
            break;
            case (HitRZCompatibility::rAlgo) :
            std::get<1>(kernels).set(checkRZ);
            std::get<1>(kernels)(b,e,innerHitsMap, ok);
            break;
            case (HitRZCompatibility::etaAlgo) :
            std::get<2>(kernels).set(checkRZ);
            std::get<2>(kernels)(b,e,innerHitsMap, ok);
            break;
          }
          for (int i=0; i!=e-b; ++i) {
            if (!ok[i]) continue;
            if (theMaxElement!=0 && result.size() >= theMaxElement){
              result.clear();
              edm::LogError("TooManyPairs")<<"number of pairs exceed maximum, no pairs produced";
              delete checkRZ;
              return;
            }
            result.add(b+i,io);
          }
        }
        delete checkRZ;
      }
      LogDebug("HitPairGeneratorFromLayerPair")<<" total number of pairs provided back: "<<result.size();
      result.shrink_to_fit();

    }
