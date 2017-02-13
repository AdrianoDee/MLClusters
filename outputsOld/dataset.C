#include <iostream>
#include <fstream>
#include <algorithm>

#include <string>
#include <sstream>

#include <iterator>

#include "TGraph.h"
#include "TFile.h"
#include "TCanvas.h"
#include "TString.h"
#include "TH2D.h"
#include "TKey.h"
#include "TCollection.h"

//#define LEVELS 65536
#define LEVELS 1

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, elems);
    return elems;
}

void datasets(int numberOfFiles, float limit = 10E7, float fakeFrac = 0.4)
{

    std::vector< std::vector< int > > vectorClusters;
    std::vector< int > vectorLabels;

    std::ofstream clusters;
    std::ofstream clusterslabels;

    clusters.open("clusterstrain.txt", std::ofstream::out);
    clusterslabels.open("clusterstrainlabels.txt", std::ofstream::out);

    std::vector< int > randomIndices;

    float fileprogress = 0.0;
    int barWidth = 50;

    int trueCounter = 0, fakeCounter = 0;
    int trueMax = (int) limit*(1.0-fakeFrac);
    int fakeMax = (int) limit*(fakeFrac);

    for (int i = 0; i <= numberOfFiles; ++i) {

      std::string fileName = "clustersfile";
      std::string labelsName = "clusterslabel";
      fileName += std::to_string(i);
      fileName += ".txt";

      labelsName += std::to_string(i);
      labelsName += ".txt";

      ifstream datasTxt(fileName);
      ifstream labelTxt(labelsName);

      int trueOrFake;

      if (datasTxt.good() && labelTxt.good()){

        std::string labels;
        labelTxt>>labels;

        int labelCounter = 0;

        //std::cout<<labels<<"  -  "<<std::endl<<std::endl;
        // std::cout<<labels.at(9);
        // std::cout<<labels[1];

        while((fakeCounter<fakeMax || trueCounter<trueMax) && labelCounter<labels.size()-1 && labelCounter<limit)
        //while(false)
        {
          std::string label;
          std::vector< int > cluster;
          //std::cout<<std::endl<<"Label  = "<<labels.at(labelCounter)<<" - Cluster :";//<<std::endl;

          label = labels.at(labelCounter);
          labelCounter++;

           for (int j = 0; j < 8*8*2; ++j) {
             int buffer;
             datasTxt>>buffer;
             //std::cout<<buffer<<" ";//<<std::endl;
             cluster.push_back(buffer);
           }

          //std::cout<<trueOrFake<<"  -  ";
          trueOrFake = atoi(label.data());
          //std::cout<<trueOrFake<<"  -  "<<trueOrFake+11<<std::endl;
          // for (size_t i = 0; i < cluster.size(); i++) {
          //   std::cout<<cluster[i]<<" ";
          // }
          // std::cout<<" - "<<cluster.size()<<std::endl;
          //
           if(trueOrFake==0) {
             fakeCounter++;
             //std::cout<<fakeCounter<<" "<<label<<" "<<std::endl;
           }

           if(trueOrFake==1 && labelCounter%30==0) trueCounter++;

           if((trueOrFake==0 && fakeCounter<=fakeMax)||(trueOrFake==1 && trueCounter<=trueMax && labelCounter%30==0))
           {
             //std::cout<<fakeCounter<<" "<<fakeMax<<" "<<trueCounter<<" "<<trueMax<<std::endl;
             vectorClusters.push_back(cluster);
             vectorLabels.push_back(trueOrFake);
           }

        }

        std::cout<<"File "<<fileName<<" - Clusters : "<<vectorClusters.size()<<" - "<<vectorLabels.size()<<" fake : "<<fakeCounter<<" true  "<<trueCounter<<std::endl;
        //std::cout<<fakeMax<<" "<<trueMax<<std::endl;
      }

    }

    for (size_t i = 0; i < vectorLabels.size(); i++) {
      if(i==0) clusterslabels<<vectorLabels[i];
      else clusterslabels<<" "<<vectorLabels[i];
    }

    for (size_t i = 0; i < vectorClusters.size(); ++i) {
      for (size_t j = 0; j < vectorClusters[i].size(); ++j) {
        if(i==0 && j==0) clusters<<vectorClusters[i][j];
        else clusters<<" "<<vectorClusters[i][j];
      }

    }

   }
