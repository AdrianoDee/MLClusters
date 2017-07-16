#include <iostream>
#include <string>
#include <fstream>
#include <sys/times.h>
#include <sys/time.h>
#include <fstream>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <sstream>
#include <algorithm>

int detOnArr[10] = {0,1,2,3,14,15,16,29,30,31};
std::vector<int> detOn(detOnArr,detOnArr+sizeof(detOnArr)/sizeof(int));

typedef std::map< std::pair < float, std::pair <float ,float> > , std::vector<float>> HitInfos;
typedef HitInfos::iterator HitInfosIt;



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

  for (int i = 1; i < argc; ++i)
	{
		std::string arg = argv[i];

    int lum = 1;
    int run = 1;
    int evt = 1;


    if (arg == "-e")
		{
			if (i + 1 < argc) // Make sure we aren't at the end of argv!
			{
				i++;
				std::istringstream ss(argv[i]);
				if (!(ss >> evt))
				{
					std::cerr << "Invalid number " << argv[i] << '\n';
					exit(1);

				}
			}
		}

    if (arg == "-r")
    {
      if (i + 1 < argc) // Make sure we aren't at the end of argv!
      {
        i++;
        std::istringstream ss(argv[i]);
        if (!(ss >> run))
        {
          std::cerr << "Invalid number " << argv[i] << '\n';
          exit(1);

        }
      }
    }

    if (arg == "-l")
    {
      if (i + 1 < argc) // Make sure we aren't at the end of argv!
      {
        i++;
        std::istringstream ss(argv[i]);
        if (!(ss >> lum))
        {
          std::cerr << "Invalid number " << argv[i] << '\n';
          exit(1);

        }
      }
    }

  }


  std::cout << "==========================================" << std::endl;
  std::cout << "Doublets matching for : " << std::endl;
  std::cout << " - Lum : " << lum << std::endl;
  std::cout << " - Run : " << run << std::endl;
  std::cout << " - Evt : " << run << std::endl;
  std::cout << "==========================================" << std::endl;

  HitInfos hitInfo;

  int dataSize = 24;

  std::string line;
  std::vector<float> infoVec; infoVec.clear();
  std::vector<float> doubVec; doubVec.clear();
  std::vector<float> doubInf; doubInf.clear();

  std::vector< std::vector< float> > doublets; doublets.clear();

  std::vector<std::string > tokens; tokens.clear();
  std::vector<float> zeros(dataSize,0.0);

  std::string fileName = std::to_string(lum) +"_"+std::to_string(run) +"_"+std::to_string(evt) + "_doubletsmatch.txt";
  std::ifstream fMatched(fileName, std::ofstream::app);

  while(fMatched.good() && std::getline(fMatched,line))
  {



    // std::cout<<"KEY : "<<++key<<" "<<line<<std::endl;

    split(line,'\t',tokens);

    for (size_t i = 3; i < tokens.size(); i++)
      infoVec.push_back(atof(tokens[i].data()));

    // std::cout<< "Key no. : "<<key++<<std::endl;


    std::pair<float,float> xy(atof(tokens[1].data()),atof(tokens[2].data()));
    std::pair < float, std::pair <float ,float> zxy(atof(tokens[0].data()),xy)

    hitInfo[zxy] = infoVec;
    infoVec.clear();
    line.clear();
    tokens.clear();

  }

  fMatched.clear();
  fMatched.seekg(0, ios::beg);
  fMatched.close();


  std::string fileName = std::to_string(lum) +"_"+std::to_string(run) +"_"+std::to_string(evt) + "_doublets.txt";
  std::ifstream fDoublets(fileName, std::ofstream::app);

  while(fDoublets.good() && std::getline(fDoublets,line))
  {

    doubInf.clear();
    doubVec.clear();

    split(line,'\t',tokens);

    for (size_t i = 0; i < tokens.size(); i++)
      doubVec.push_back(atof(tokens[i].data()));

    float iZ = doubVec[4], iX = doubVec[5], iY = doubVec[6];
    std::pair <float ,float> ixy(iX,iY);
    std::pair < float, std::pair <float ,float> > ixyz(iZ,ixy);

    float oZ = doubVec[7], oX = doubVec[8], oY = doubVec[9];
    std::pair <float ,float> ixy(oX,oY);
    std::pair < float, std::pair <float ,float> > oxyz(oZ,oxy);

    HitInfosIt itHitIn = hitInfo.find(ixyz);
    HitInfosIt itHitOut = hitInfo.find(oxyz);

    if(itHitIn!=hitInfos.end() && itHitOut!=hitInfos.end())
    {
      if((itHitIn->second)[0]==(itHitOut->second)[0])
        doubInf = itHitIn->second;
    }
    else
      doubInf = zeros;

    if((int)doubInf.size()!=dataSize) std::cout<<"WARNINCAZZO: "<<infos.size()<<" "<<dataSize<<std::endl;

    doubVec.insert(clust.end(),doubInf.begin(),doubInf.end());

    doublets.push_back(doubVec);

    line.clear();

  }

  fileName = std::to_string(lum) +"_"+std::to_string(run) +"_"+std::to_string(evt) + "_dataset.txt";
  std::ifstream fDataset(fileName);

  for (size_t i = 0; i < doublets.size(); i++) {

    int vecsize = doublets[i].size();

    for(int j = 0;j<vecsize - 1;++j)
      fDataset << doublets[i][j]<<"\t";

    fDataset << doublets[i][vecsize - 1];

    fDataset << std::endl;
  }


}
