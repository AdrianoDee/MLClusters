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

void label(int numberOfFiles, float limit = 10E7, float fakeFrac = 0.4)
{

    std::vector< std::vector< int > > vectorClusters;
    std::vector< int > vectorLabels;

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

        while(fakeCounter<fakeMax || trueCounter<trueMax)
        {

          trueOrFake = atoi(labels[labelCounter]);
          labelCounter++;
          std::vector< int > cluster;

          for (int j = 0; j < 8*8*2; ++j) {
            int buffer;
            datasTxt>>buffer;
            //std::cout<<buffer<<std::endl;
            cluster.push_back(buffer);
          }

          std::cout<<trueOrFake<<"  -  ";
          for (size_t i = 0; i < cluster.size(); i++) {
            std::cout<<cluster[i]<<" ";
          }
          std::cout<<" - "<<cluster.size()<<std::endl;

          if((trueOrFake==0 && (fakeCounter++)<=fakeMax)||(trueOrFake==1 && (trueCounter++)<=trueMax))
          {
            vectorClusters.push_back(cluster);
            vectorLabels.push_back(trueOrFake);
          }

        }

      }
    }


   }
