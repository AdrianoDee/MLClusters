#include <iostream>
#include <string>
#include <sys/times.h>
#include <sys/time.h>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <algorithm>
#include <tuple>
#include <map>

int detOnArr[10] = {0,1,2,3,14,15,16,29,30,31};
std::vector<int> detOn(detOnArr,detOnArr+sizeof(detOnArr)/sizeof(int));

//x,y,z,cX,cY,adc,zeroADC,sizex,sizey,size,
// typedef clusterKey std::tuple <float,float,float,float,float,float,float,float,float,float>
//typedef std::tuple <float,float,float,float> clusterKey;
//layer ladder module side disk panel blade hitC
typedef std::tuple <float,float,float,float,float,float,float,float,float,float,float> clusterKey;
// typedef std::tuple <float,float,float,float,float,float> clusterKey; //,float,float,float,float>
typedef std::map< clusterKey , std::vector<float>> HitInfos;
typedef HitInfos::iterator HitInfosIt;
typedef std::map< clusterKey , float> HitIn;

float tFact = 1.0;

bool debugging=false;

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
  std::stringstream ss;
  ss.str(s);
  std::string item;
  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }
}

void debug(int line) {
  if (debugging)
    std::cout <<"Debugging on line " <<line <<std::endl;
}

#define DEBUGLINE debug(__LINE__);

int main(int argc, char** argv)
{

    int lum = 1;
    int run = 1;
    int evt = 1;
    std::string read = "./";
    std::string path = ".";
  for (int i = 1; i < argc; ++i)
	{
        std::string arg = argv[i];
        if (arg == "-p")
        {
          if (i + 1 < argc) // Make sure we aren't at the end of argv!
          {
            i++;
            read = argv[i];
          }
        }

  }

  std::vector <std::string > tokens; tokens.clear();
  split(read,'/',tokens);
  std::string nameFile = tokens[tokens.size()-1];
  std::string jobName = tokens[0];

  for(int i=0;i<tokens.size()-1;i++)
      path += "/" + tokens[i];

  tokens.clear();

  split(jobName,'_',tokens);

  int jobOne = atof(tokens[0].data());
  int jobTwo = atof(tokens[1].data());

  tokens.clear();
  split(nameFile,'_',tokens);

  lum = atof(tokens[0].data());
  run = atof(tokens[1].data());
  evt = atof(tokens[2].data());

  std::cout << "==========================================" << std::endl;
  std::cout << "Doublets matching for : " << std::endl;
  std::cout << " - Lum : " << lum << std::endl;
  std::cout << " - Run : " << run << std::endl;
  std::cout << " - Evt : " << evt << std::endl;
  std::cout << "==========================================" << std::endl;

  HitInfos hitInfo;

  int dataSize = 24;

  std::string line;
  std::vector<float> infoVec; infoVec.clear();
  std::vector<float> doubVec; doubVec.clear();
  std::vector<float> doubInf; doubInf.clear();

  std::vector< std::vector< float> > doublets; doublets.clear();

  std::vector<float> zeros(dataSize,-1.0);

  //std::string fileName = path + std::to_string(lum) +"_"+std::to_string(run) +"_"+std::to_string(evt) + "_doubletsmatch.txt";
  std::ifstream fMatched(read);
  std::cout << "Path : " << path << " File : " << read << std::endl;
  std::cout << " - Hits mapping" << std::endl;
  int totd = 0, matd = 0, dhit=0, uhm = 0,same=0;
  while(fMatched.good() && std::getline(fMatched,line))
  {


    // std::cout<<"KEY : "<<++key<<" "<<line<<std::endl;

    split(line,'\t',tokens);
    //std::cout << tokens.size() << std::endl;

    for (size_t i = 18; i < tokens.size(); i++)
    {
      infoVec.push_back(atof(tokens[i].data()));
      // std::cout<< i <<" - "<< tokens[i].data()<<std::endl;
    }

    // std::cout<< "Key no. : "<<key++<<std::endl;

    float Z = atof(tokens[0].data());
    float X = atof(tokens[1].data());
    float Y = atof(tokens[2].data());
    float hId = atof(tokens[3].data());
    //
    float layer = atof(tokens[4].data());
    float ladder = atof(tokens[5].data());
    float module = atof(tokens[6].data());
    float side = atof(tokens[7].data());
    float disk = atof(tokens[8].data());
    float panel = atof(tokens[9].data());
    float blade = atof(tokens[10].data());
    //
    float cX = atof(tokens[11].data());
    float cY = atof(tokens[12].data());
    float adc = atof(tokens[13].data());
    float zeroADC = atof(tokens[14].data());
    float sizeX = atof(tokens[15].data());
    float sizeY = atof(tokens[16].data());
    float size = atof(tokens[17].data());
    float tId = atof(tokens[18].data());

    Z =(float)((int)(atof(tokens[0].data())*tFact))/tFact;
    X =(float)((int)(atof(tokens[1].data())*tFact))/tFact;
    Y =(float)((int)(atof(tokens[2].data())*tFact))/tFact;

    // std::cout<<"Test: "<<  X << " - " << (int)(atof(tokens[1].data())) << std::endl;

    // std::pair<float,float> xy(X,Y);
    // std::pair < float, std::pair <float ,float> > zxy(Z,xy);
    // clusterKey cKey(X,Y,Z,cX,cY,adc);//, tId,cX,cY,adc,zeroADC,sizeX,sizeY,size);
    // clusterKey cKey(dId,cX,cY,adc);
    clusterKey cKey (layer,ladder,module,side,disk,panel,blade,cX,cY,adc,zeroADC);
    // std::cout << X << " - " << Y << " - "  << Z << " - " << adc << " - "<< cX << " - "<< cY <<std::endl;

    //std::cout<<(float)(floor(atof(tokens[1].data())*1000.)/1000.0)<<"-"<<(float)((floor(atof(tokens[2].data())*1000.0)/1000.0))<<"-"<<(float)(floor(atof(tokens[0].data())*1000.)/1000.)<<std::endl;
    hitInfo[cKey] = infoVec;
    infoVec.clear();
    line.clear();
    tokens.clear();

  }

  fMatched.clear();
//  fMatched.seekg(0, ios::beg);
  fMatched.close();

  HitIn countHit;
  std::string fileName = path + "/" + std::to_string(lum) +"_"+std::to_string(run) +"_"+std::to_string(evt) + "_doubletsC.txt";
  std::ifstream fDoublets(fileName);

  std::cout << "Path : " << path << " File : " << fileName << std::endl;
  std::cout << " - Doublets matching" << std::endl;

  fileName = path + "/" + std::to_string(jobOne) +"_" + std::to_string(jobTwo) +"_" + std::to_string(lum) +"_" + std::to_string(run) +"_"+std::to_string(evt) + "_dataset.txt";
  std::ofstream fDataset(fileName);


  while(fDoublets.good() && std::getline(fDoublets,line))
  {
    ++totd;
    doubInf.clear();
    doubVec.clear();

    split(line,'\t',tokens);
    //std::cout << tokens.size() << std::endl;

    for (size_t i = 0; i < tokens.size(); i++)
      {
        doubVec.push_back(atof(tokens[i].data()));
        // std::cout<< i <<" - "<< tokens[i].data()<<std::endl;
      }
   // std::cout<<doubVec<<std::endl;
    //std::cout<<line<<std::endl;
    float iZ = doubVec[4], iX = doubVec[5], iY = doubVec[6];
    float ilayer = 0., iladder = 0., imodule = 0., iside = 0., idisk = 0., ipanel = 0., iblade = 0.;
    float iADC, iCX, iCY, iZeroADC;

    iZ =(float)((int)(doubVec[4]*tFact))/tFact;
    iX =(float)((int)(doubVec[5]*tFact))/tFact;
    iY =(float)((int)(doubVec[6]*tFact))/tFact;

    ilayer = doubVec[18];
    iladder = doubVec[19];
    imodule = doubVec[20];
    iside = doubVec[21];
    idisk = doubVec[22];
    ipanel = doubVec[23];
    iblade = doubVec[24];

    iADC = doubVec[43];
    iCX = doubVec[41];
    iCY = doubVec[42];
    iZeroADC = doubVec[44];

    // std::pair <float ,float> ixy(iX,iY);
    // std::pair < float, std::pair <float ,float> > ixyz(iZ,ixy);
    // clusterKey iKey(iX,iY,iZ,iCX,iCY,iADC);
    // clusterKey iKey(iDet,iCX,iCY,iADC);
    clusterKey iKey (ilayer,iladder,imodule,iside,idisk,ipanel,iblade,iCX,iCY,iADC,iZeroADC);

    float oZ = doubVec[7], oX = doubVec[8], oY = doubVec[9];
    float olayer = 0, oladder = 0, omodule = 0, oside = 0, odisk = 0, opanel = 0, oblade = 0;
    float oADC, oCX, oCY, oZeroADC;

    oZ =(float)((int)(doubVec[7]*tFact))/tFact;
    oX =(float)((int)(doubVec[8]*tFact))/tFact;
    oY =(float)((int)(doubVec[9]*tFact))/tFact;

    olayer = doubVec[25];
    oladder = doubVec[26];
    omodule = doubVec[27];
    oside = doubVec[28];
    odisk = doubVec[29];
    opanel = doubVec[30];
    oblade = doubVec[31];

    oADC = doubVec[54];
    oCX = doubVec[52];
    oCY = doubVec[53];
    oZeroADC = doubVec[55];

    // std::cout << oX << " - " << oY << " - "  << oZ << " - " << oADC << " - "<< oCX << " - "<< oCY <<std::endl;

    // clusterKey oKey(oX,oY,oZ,oCX,oCY,oADC);
    // clusterKey oKey(oDet,oCX,oCY,oADC);
    clusterKey oKey (olayer,oladder,omodule,oside,odisk,opanel,oblade,oCX,oCY,oADC,oZeroADC);
    // std::pair <float ,float> oxy(oX,oY);
    // std::pair < float, std::pair <float ,float> > oxyz(oZ,oxy);

    HitInfosIt itHitIn = hitInfo.find(iKey);
    HitInfosIt itHitOut = hitInfo.find(oKey);

    if(itHitIn!=hitInfo.end() )
	{
		// std::cout<<iX<<" - "<<iY<<" - "<<iZ<<" - "<<std::endl;
		// std::cout<<itHitIn->first.second.first<<" - "<<itHitIn->first.second.second<<" - "<<itHitIn->first.first<<std::endl;

		countHit[iKey] = tFact;
		++dhit;
	}

	if(itHitOut!=hitInfo.end() )
        {
		// std::cout<<oX<<" - "<<oY<<" - "<<oZ<<" - "<<std::endl;
    //             std::cout<<itHitOut->first.second.first<<" - "<<itHitOut->first.second.second<<" - "<<itHitOut->first.first<<std::endl;
                countHit[oKey] = tFact;
                ++dhit;
        }
    if(itHitIn!=hitInfo.end() && itHitOut!=hitInfo.end())
    {
      ++same;
      std::cout << "===§===§===§===§===§===§===§===§===§===§==="<< std::endl;
      bool twoHitsTrack = true;
      for(int i=12;i<dataSize;++i)
        if((itHitIn->second)[i]!=(itHitOut->second)[i])
          {
            ++uhm;
            // std::cout<<i<<" - "<<(itHitIn->second)[i]<<" -> "<< (itHitOut->second)[i]<<std::endl;
            twoHitsTrack = false;
          }
      if(twoHitsTrack)
      {
	       ++matd;
         doubInf = itHitIn->second;
         //	std::cout<<itHitIn->first.first<<std::endl;
         // std::cout<<iZ<<std::endl;
         // std::cout<<itHitOut->first.first<<std::endl;
         //  std::cout<<oZ<<std::endl;
	       std::cout<< "Very, very good! : " << std::endl;
       }
      else
     {

//	std::cout<<(itHitIn->second)[11]<<std::endl;
//	std::cout<<(itHitOut->second)[11]<<std::endl;
	// std::cout<<"UHM"<<std::endl;
	doubInf = zeros;
    }}
    else
	doubInf = zeros;
    if((int)doubInf.size()!=dataSize) std::cout<<"WARNINCAZZO: "<<doubInf.size()<<" "<<dataSize<<std::endl;

    doubVec.insert(doubVec.end(),doubInf.begin(),doubInf.end());

    int vecsize = doubVec.size();
    //std::cout<< vecsize <<std::endl;
    for(int j = 0;j<vecsize - 1;++j)
      fDataset << doubVec[j]<<"\t";

    fDataset << doubVec[vecsize - 1];

    fDataset << std::endl;
    doubInf.clear();
    tokens.clear();
    doubVec.clear();
 //   doublets.push_back(doubVec);

    line.clear();

  }
/*
  fileName = path + std::to_string(lum) +"_"+std::to_string(run) +"_"+std::to_string(evt) + "_dataset.txt";
  std::ofstream fDataset(fileName);

  std::cout << " - Doublets matching" << std::endl;
  for (size_t i = 0; i < doublets.size(); i++) {

    int vecsize = doublets[i].size();

    for(int j = 0;j<vecsize - 1;++j)
      fDataset << doublets[i][j]<<"\t";

    fDataset << doublets[i][vecsize - 1];

    fDataset << std::endl;
  }

*/

std::cout << matd <<" on "<<totd<<" and "<<dhit<<" ... and " << countHit.size() << " with uhm " << uhm << " and same "<<same<<std::endl;

}
