// The comments in this code are removed based on the license agreements
// This code optimizes a sophisticated model with topology optimization approach for highly non-convex non-linear problem with 30,000 variables

#define NDEBUG

//#define TraceFollow
//#define TraceFollowPrint
//#define InterpoleXolds			//Interpoles MMA xoldes in x2newx function
//#define x2newxRoPrint
//#define ChangeSIMPPower			//If SIMP Power would change in each mdgeo change, have to be active
#define FreezeInnerSolidX			//Freezes Inner solid x
#define CutXTails					//Rounds X values up

//#define DebugFD					//Prints Finite difference df0dx out
//#define InnerFDdfDebug
//#define FDdfFiltering
//#define FDdfElFiltering
//#define DebugDF					//Prints df0dx

//#define OptimizationisDynamic
#define Dynf0valFTU						//have to be uncommented whenever f0val is not UTU even in SBS
//#define Dynf0valUTKHatU				// f0val = Trans(u)*KHat*u
//#define Dynf0valUTKU					// f0val = Trans(u)*K*u
//#define Dynf0valUTMU					// f0val = Trans(u)*Mass*u
//#define Dynf0valUTSU					// f0val = Trans(u)*(K+s*M)*u
#define Dynf0valUTKDU					// f0val = Trans(u)*K*Delta(u)
//#define MPISensitivityAnalisys
//#define MPIInnerSensitivityAnalisys		//If is used MPISensitivityAnalisys has to be uncommented as MPI initialization is done with it at main()
//#define MTMakestiffness
//#define DynMTMakestiffness
//#define MassModificationInnerDynOpt
//#define MassModificationDynOpt

//#define DynDf0dxDebug			 	//signs to print infos (General and SBS meth.)
//#define DynMassDebug
//#define DynFDdf0dx
//#define DynInnerFDdf0dx
#define DynRoPrint
#define DynInnerRoPrint
//#define DynEarthQuake				//Important note: Uncomment if and if just analysis is for EQ, Uncomment for Earthquake and reads ground acceleration in Dynloads vector
//#define DynUEA					//Use of equilibrim acceleration is activated (Beta-Newmark algorithm)
#define DynDamping					//Activates damping on structure

//#define DynInnerDispPrint
#define DynDispPrint

//#define FilterDynOpt				//Filtering techniques
//#define DynRoFilterPrint
//#define DynFDFiltering
//#define ElFilterDynOpt
//#define DynRoEFilterPrint
//#define DynFDElFiltering

#define DynSBSFillStorySolid		//SBS
//#define DynSBSRigidStory
//#define DynSBSEarthQuake			//Important note: Uncomment if and if just analysis is for EQ, uncomment for Earthquake in SBS and reads in Dynloads vector
//#define DynSBSf0valDispStories	//Displacements norm in all stories for f0val : if macro Dynf0valFTU is activated works
//#define DynSBSf0valUTKHatU
//#define DynSBSf0valUTKU
//#define DynSBSf0valUTMU
#define DynSBSf0valUTKDU

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <boost/numeric/ublas/matrix_sparse.hpp>
//#include <boost/numeric/ublas/symmetric.hpp>
//#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/numeric/ublas/operation.hpp>
#include <boost/iterator.hpp>
#include <boost/timer.hpp>
#include "pdsp_defs.h"

#ifdef _OPENMP
	#include <omp.h>
#endif

#include <mpi.h>
#ifdef MPISensitivityAnalisys
	//#include <mpi.h>
	//double TimeMPI;
	int MyMPIRank;
	int OurMPISize;
#else
	int MyMPIRank=0;
	int OurMPISize=1;
#endif

#include "MatCal.h"
//#include "matrix_market_interface.h"

extern "C"
{
	#include <taucs.h>
}

typedef boost::numeric::ublas::matrix< double > Matrix;
typedef boost::numeric::ublas::matrix< size_t > SizMatrix;
typedef boost::numeric::ublas::compressed_matrix < double,boost::numeric::ublas::row_major,0, boost::numeric::ublas::unbounded_array<int> > CSMatrix;
typedef boost::numeric::ublas::compressed_matrix < double,boost::numeric::ublas::row_major,0, boost::numeric::ublas::unbounded_array<int> > CSMatrix;
//typedef boost::numeric::ublas::compressed_matrix < double, boost::numeric::ublas::column_major, 0, boost::numeric::ublas::unbounded_array<int> > CCMatrix;
typedef boost::numeric::ublas::compressed_matrix < double > CMatrix;

typedef boost::numeric::ublas::vector< double > Vector;
typedef boost::numeric::ublas::vector< size_t > SizVector;
typedef boost::numeric::ublas::compressed_vector < double > CVector;
typedef boost::numeric::ublas::compressed_vector < std::vector<double> > CVVector;
//typedef boost::numeric::ublas::zero_vector <double> Zvector;
//typedef CMatrix::iterator1 iterator1;
//typedef CMatrix::iterator2 iterator2;

using namespace std;

double DensityVal = 1.00;
double MassPower  = 1.00;

double MassModifyP= 6.00;
double RoModifyVal= 0.1;

double InnerSolidUV=0.999;
double InnerSolidLV=0.002;

double sf0val=4.00/.0001/.0001;
double ImpFac=1.00;

double Thickness=1.00;									//Thickness of used plane

double E;
double Nu;
double TimeMPI;
int Pstress;											//Pstress:1 , Ptrain:0

double Length = 0;										//length of plate
double Width  = 0;										//width of plate									//number of elements in width
double Elemlen;											//element length
double Elemwid; 										//element width

size_t Lenpart = 0;									//number of elements in length
size_t Widpart = 0;
size_t NodesNum;										//number of nodes
size_t ElemsNum;										//number of elements
size_t UNum;

Matrix D(3, 3);											//Material stiffness matrix									//connectivity matrix
Matrix Nodes;											//coordinates of nodes matrix
Matrix IntegPoints(4,3);								//integration points for elements
Matrix EDisps;

SizMatrix Connectivity;

//CMatrix Stiffness;									  //(101*101*2,101*101*2,4*((Lenpart-1)*(Widpart-1)*9+12*(Lenpart+Widpart-2)+16));
//#ifdef OptimizationisDynamic
	CMatrix KuuT;
	#ifdef DynDamping
		CMatrix CuuT;
		CSMatrix CuuS;
	#endif
//#endif
CSMatrix KuuS;
//boost::numeric::ublas::symmetric_adaptor <CSMatrix, boost::numeric::ublas::upper> Kuu(KuuS);
//CMatrix Kub;
CMatrix Kb;

CVector Loads;
Vector Disps;
Vector Fu;
Vector Du;

SizVector Index;
SizVector IndexRev;

void make()
{
	ifstream inmp("MaterialProp.txt");					//reading material properties
	inmp >> E >> Nu >> Pstress;

	//ifstream inconn("Connectivity.txt");		  	//reading connectivity matrix if necessary
	//ifstream innode("Nodes.txt");						//Nodes coordinates
	ifstream inSimpleGeo("SimpleGeo.txt");
	inSimpleGeo >> Length >> Width >> Lenpart >> Widpart;

	if (Length)											//checking the simple geometry
	{
		NodesNum =(Lenpart+1)*(Widpart+1);			  //initializing Matrixes , vectors & values
		ElemsNum =Lenpart*Widpart;
		Elemlen =Length/Lenpart;
		Elemwid =Width/Widpart;

		Connectivity.resize(ElemsNum,8);
		Nodes.resize(NodesNum,2);
		//Stiffness.resize(NodesNum*2,NodesNum*2,false);
		Loads.resize(NodesNum*2);
		Disps.resize(NodesNum*2);
		Loads.clear();
		Disps.clear();
		for (size_t elem=0;elem < ElemsNum;elem++)	//building connectivity matrix
		{
			long int n=(elem-elem % Lenpart)/Lenpart;
			Connectivity(elem,0)=2*(elem+n);
			Connectivity(elem,1)=2*(elem+n)+1;
			Connectivity(elem,2)=2*(elem+n+1);
			Connectivity(elem,3)=2*(elem+n+1)+1;
			Connectivity(elem,4)=2*(elem+Lenpart+n+2);
			Connectivity(elem,5)=2*(elem+Lenpart+n+2)+1;
			Connectivity(elem,6)=2*(elem+Lenpart+n+1);
			Connectivity(elem,7)=2*(elem+Lenpart+n+1)+1;
		}
		for (size_t node=0;node < NodesNum;node++)	//building nodes coordinates matrix
		{
			long int mn=node%(Lenpart+1);
			long int nn=(node-mn)/(Lenpart+1);
			Nodes(node,0)=mn*Elemlen;
			Nodes(node,1)=nn*Elemwid;
		}
	}

	double ii;
	size_t i;
	ifstream inLoads("Loads.txt");					  //Reading exerted loads
	while (inLoads>>i)
	{
		if (i >= 2*NodesNum)
		{
			cout<<"Exerted load number "<<i<<" is out of bound"<<endl;
			exit(1);
		}
		inLoads>>ii;
		Loads(i)=ii;
	}


	ifstream inDisps("Displacements.txt");			  //Reading exerted diplacements
	ii=0;
	while (inDisps>>i)
	{
		EDisps.resize(EDisps.size1()+1,2,(ii ? true : false));
		EDisps(EDisps.size1()-1,0)=i;
		inDisps>>EDisps(EDisps.size1()-1,1);
		if (i >= 2*NodesNum)
		{
			cout<<"Exerted displacement number "<<i<<" is out of bound"<<endl;
			exit(1);
		}

		Disps(i)=EDisps(EDisps.size1()-1,1);
		ii++;
		//inDisps>>ii;
		//EDisps(i)=ii;
	}
	//cout<<"--------------->>>  "<<EDisps<<endl;
	UNum=2*NodesNum-EDisps.size1();

	KuuS.resize(UNum,UNum,false);
#ifdef OptimizationisDynamic
	KuuT.resize(UNum,UNum);
	#ifdef DynDamping
		CuuT.resize(UNum,UNum);
		CuuS.resize(UNum,UNum,false);
	#endif
#endif
	Kb.resize(EDisps.size1(),2*NodesNum,false);
	Fu.resize(UNum);
	Du.resize(UNum);
	Fu.clear();
	Du.clear();
	//cout<<Loads<<endl;
	//cout<<Disps<<endl;

	{
		IntegPoints(0,0)=-sqrt(3.00)/3.00;				  //Gausian integration points
		IntegPoints(1,0)=sqrt(3.00)/3.00;
		IntegPoints(2,0)=sqrt(3.00)/3.00;
		IntegPoints(3,0)=-sqrt(3.00)/3.00;

		IntegPoints(0,1)=-sqrt(3.00)/3.00;
		IntegPoints(1,1)=-sqrt(3.00)/3.00;
		IntegPoints(2,1)=sqrt(3.00)/3.00;
		IntegPoints(3,1)=sqrt(3.00)/3.00;

		IntegPoints(0,2)=1; 								//Impact factor
		IntegPoints(1,2)=1;
		IntegPoints(2,2)=1;
		IntegPoints(3,2)=1;
	}

	if (Pstress)
	{
		D(0, 0) = E / (1.00 - pow(Nu, 2.00)) * 1.00;		//Initializing material stiffness properties
		D(0, 1) = E / (1.00 - pow(Nu, 2.00)) * Nu;
		D(0, 2) = E / (1.00 - pow(Nu, 2.00)) * 0.00;
		D(1, 0) = E / (1.00 - pow(Nu, 2.00)) * Nu;
		D(1, 1) = E / (1.00 - pow(Nu, 2.00)) * 1.00;
		D(1, 2) = E / (1.00 - pow(Nu, 2.00)) * 0.00;
		D(2, 0) = E / (1.00 - pow(Nu, 2.00)) * 0.00;
		D(2, 1) = E / (1.00 - pow(Nu, 2.00)) * 0.00;
		D(2, 2) = E / (1.00 - pow(Nu, 2.00)) * (1.00 - Nu) / 2.00;
	}
	else
	{
		D(0 , 0) = E / (1.00 + Nu) / (1.00 - 2.00 * Nu) * (1.00 - Nu);
		D(0 , 1) = E / (1.00 + Nu) / (1.00 - 2.00 * Nu)* Nu;
		D(0 , 2) = E / (1.00 + Nu) / (1.00 - 2.00 * Nu)* 0.00;
		D(1 , 0) = E / (1.00 + Nu) / (1.00 - 2.00 * Nu)* Nu;
		D(1 , 1) = E / (1.00 + Nu) / (1.00 - 2.00 * Nu)* (1.00 - Nu);
		D(1 , 2) = E / (1.00 + Nu) / (1.00 - 2.00 * Nu)* 0.00;
		D(2 , 0) = E / (1.00 + Nu) / (1.00 - 2.00 * Nu)* 0.00;
		D(2 , 1) = E / (1.00 + Nu) / (1.00 - 2.00 * Nu)* 0.00;
		D(2 , 2) = E / (1.00 + Nu) / 2.00;
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////////
class element
{
	double x1;											   //x dir. coordinate value of node:1 ...
	double y1;
	double x2;
	double y2;
	double x3;
	double y3;
	double x4;
	double y4;

	Matrix kIntP1;
	Matrix kIntP2;
	Matrix kIntP3;
	Matrix kIntP4;

	Matrix mIntP1;
	Matrix mIntP2;
	Matrix mIntP3;
	Matrix mIntP4;

	Matrix helper38;
	void kElemBuild();
	void mElemBuild();
public:

	element(double X1, double Y1,double X2,double Y2,double X3, double Y3,double X4,double Y4): x1(X1), y1(Y1), x2(X2), y2(Y2), x3(X3), y3(Y3), x4(X4), y4(Y4)
	{
		k.resize(8,8);
		m.resize(8,8);
		//kIntP.resize(8,8*IntegPoints.size1());
		Jacob.resize(2,2);
		InvJ.resize(2,2);
		B.resize(3,8);

		kIntP1.resize(8,8);
		kIntP2.resize(8,8);
		kIntP3.resize(8,8);
		kIntP4.resize(8,8);

		mIntP1.resize(8,8);
		mIntP2.resize(8,8);
		mIntP3.resize(8,8);
		mIntP4.resize(8,8);

		helper38.resize(3,8);
		//kbuild();
		kElemBuild();
#ifdef OptimizationisDynamic
		mElemBuild();
#endif
	}
	Matrix Jacob;												//Jacobian matrix of element
	double DetJ;
	Matrix InvJ;
	Matrix N;
	Matrix B;
	Matrix k;													//Stiffness matrix of element
	Matrix m;
	//Matrix kIntP;
	Matrix& JBuild(const float &ks,const float &et);
	void NBuild(const float &ks,const float &et);
	Matrix& BBuild(const float &ks,const float &et);
	Matrix& kBuild(double *Ro,Matrix &kelem);
	void mbuild(double *Ro,Matrix &melem);
};

Matrix& element::JBuild(const float &ks,const float &et)
{
	Jacob(0,0)= x1*-1.00/4.00*(1.00-et)+ x2*1.00/4.00*(1.00-et)+ x3*1.00/4.00*(1.00+et)+ x4*-1.00/4.00*(1.00+et);
	Jacob(0,1)= y1*-1.00/4.00*(1.00-et)+ y2*1.00/4.00*(1.00-et)+ y3*1.00/4.00*(1.00+et)+ y4*-1.00/4.00*(1.00+et);
	Jacob(1,0)= x1*-1.00/4.00*(1.00-ks)+ x2*-1.00/4.00*(1.00+ks)+ x3*1.00/4.00*(1.00+ks)+ x4*1.00/4.00*(1.00-ks);
	Jacob(1,1)= y1*-1.00/4.00*(1.00-ks)+ y2*-1.00/4.00*(1.00+ks)+ y3*1.00/4.00*(1.00+ks)+ y4*1.00/4.00*(1.00-ks);
	//cout<<Jacob<<"=jacobian"<<endl;
	DetJ=Jacob(0,0)*Jacob(1,1)-Jacob(0,1)*Jacob(1,0);

	InvJ(0,0) = 1.00/DetJ*Jacob(1,1);
	InvJ(0,1) = 1.00/DetJ*-Jacob(0,1);
	InvJ(1,0) = 1.00/DetJ*-Jacob(1,0);
	InvJ(1,1) = 1.00/DetJ*Jacob(0,0);
	return Jacob;
}

void element::NBuild(const float &ks,const float &et)
{
	N(0,0)=N(1,1)=1.00/4.00*(1.00-ks)*(1.00-et);
	N(0,1)=N(1,0)=0.00;

	N(0,2)=N(1,3)=1.00/4.00*(1.00+ks)*(1.00-et);
	N(0,3)=N(1,2)=0.00;

	N(0,4)=N(1,5)=1.00/4.00*(1.00+ks)*(1.00+et);
	N(0,5)=N(1,4)=0.00;

	N(0,6)=N(1,7)=1.00/4.00*(1.00-ks)*(1.00+et);
	N(0,7)=N(1,6)=0.00;
}

Matrix& element::BBuild(const float &ks,const float &et)
{
	JBuild(ks,et);
	B(0,0)= -1/4.00*(1-et)*InvJ(0,0)-1/4.00*(1-ks)*InvJ(0,1);
	B(0,1)= 0.00;
	B(1,0)= 0.00;
	B(1,1)= -1/4.00*(1-et)*InvJ(1,0)-1/4.00*(1-ks)*InvJ(1,1);
	B(2,0)= -1/4.00*(1-et)*InvJ(1,0)-1/4.00*(1-ks)*InvJ(1,1);
	B(2,1)= -1/4.00*(1-et)*InvJ(0,0)-1/4.00*(1-ks)*InvJ(0,1);

	B(0,2)= 1/4.00*(1-et)*InvJ(0,0)-1/4.00*(1+ks)*InvJ(0,1);
	B(0,3)= 0.00;
	B(1,2)= 0.00;
	B(1,3)= 1/4.00*(1-et)*InvJ(1,0)-1/4.00*(1+ks)*InvJ(1,1);
	B(2,2)= 1/4.00*(1-et)*InvJ(1,0)-1/4.00*(1+ks)*InvJ(1,1);
	B(2,3)= 1/4.00*(1-et)*InvJ(0,0)-1/4.00*(1+ks)*InvJ(0,1);

	B(0,4)= 1/4.00*(1+et)*InvJ(0,0)+1/4.00*(1+ks)*InvJ(0,1);
	B(0,5)= 0.00;
	B(1,4)= 0.00;
	B(1,5)= 1/4.00*(1+et)*InvJ(1,0)+1/4.00*(1+ks)*InvJ(1,1);
	B(2,4)= 1/4.00*(1+et)*InvJ(1,0)+1/4.00*(1+ks)*InvJ(1,1);
	B(2,5)= 1/4.00*(1+et)*InvJ(0,0)+1/4.00*(1+ks)*InvJ(0,1);

	B(0,6)= -1/4.00*(1+et)*InvJ(0,0)+1/4.00*(1-ks)*InvJ(0,1);
	B(0,7)= 0.00;
	B(1,6)= 0.00;
	B(1,7)= -1/4.00*(1+et)*InvJ(1,0)+1/4.00*(1-ks)*InvJ(1,1);
	B(2,6)= -1/4.00*(1+et)*InvJ(1,0)+1/4.00*(1-ks)*InvJ(1,1);
	B(2,7)= -1/4.00*(1+et)*InvJ(0,0)+1/4.00*(1-ks)*InvJ(0,1);
	//cout<<B<<"=B"<<endl;
	return B;
}

void element::kElemBuild()
{
	/*for (size_t i=0;i<8;i++)
		for (size_t j=0;j<8;j++)
			kIntP1(i,j)=kIntP2(i,j)=kIntP3(i,j)=kIntP4(i,j)=0.00;*/

	BBuild(IntegPoints(0,0),IntegPoints(0,1));
	axpy_prod(trans(B),axpy_prod(D,B,helper38),kIntP1);
	kIntP1 *= (IntegPoints(0,2)*DetJ);

	BBuild(IntegPoints(1,0),IntegPoints(1,1));
	axpy_prod(trans(B),axpy_prod(D,B,helper38),kIntP2);
	kIntP2 *= (IntegPoints(1,2)*DetJ);

	BBuild(IntegPoints(2,0),IntegPoints(2,1));
	axpy_prod(trans(B),axpy_prod(D,B,helper38),kIntP3);
	kIntP3 *= (IntegPoints(2,2)*DetJ);

	BBuild(IntegPoints(3,0),IntegPoints(3,1));
	axpy_prod(trans(B),axpy_prod(D,B,helper38),kIntP4);
	kIntP4 *= (IntegPoints(3,2)*DetJ);

}

void element::mElemBuild()
{
	JBuild(IntegPoints(0,0),IntegPoints(0,1));
	NBuild(IntegPoints(0,0),IntegPoints(0,1));
	axpy_prod(trans(N),N,mIntP1);
	mIntP1 *= (IntegPoints(0,2)*DetJ);

	JBuild(IntegPoints(1,0),IntegPoints(1,1));
	NBuild(IntegPoints(1,0),IntegPoints(1,1));
	axpy_prod(trans(N),N,mIntP2);
	mIntP2 *= (IntegPoints(1,2)*DetJ);

	JBuild(IntegPoints(2,0),IntegPoints(2,1));
	NBuild(IntegPoints(2,0),IntegPoints(2,1));
	axpy_prod(trans(N),N,mIntP3);
	mIntP3 *= (IntegPoints(2,2)*DetJ);

	JBuild(IntegPoints(3,0),IntegPoints(3,1));
	NBuild(IntegPoints(3,0),IntegPoints(3,1));
	axpy_prod(trans(N),N,mIntP4);
	mIntP4 *= (IntegPoints(3,2)*DetJ);
}

Matrix& element::kBuild(double *Ro,Matrix &kelem)					//Computing stiffness matrix of element
{
	/*for (size_t i=0;i<8;i++)
		for (size_t j=0;j<8;j++)
			k(i,j)=0;*/
	kelem=(kIntP1*Ro[0]+kIntP2*Ro[1]+kIntP3*Ro[2]+kIntP4*Ro[3])*Thickness;
	return kelem;
}

void element::mbuild(double *Ro,Matrix &melem)
{
	melem=(mIntP1*Ro[0]+mIntP2*Ro[1]+mIntP3*Ro[2]+mIntP4*Ro[3])*DensityVal;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
////computes Kuu & Fu & Kb

void BuildIndex()
{
	Index.resize(2*NodesNum);
	IndexRev.resize(UNum);

	size_t ii=0;
	size_t jj=0;

	for (size_t i=0;i<2*NodesNum;i++)
	{
		bool exist=false;
		for (size_t j=0;j<EDisps.size1();j++)
			if (i==EDisps(j,0))
			{
				Index(i)=2*NodesNum-1-ii;
				exist=true;
				ii++;
				break;
			}
		if (exist==false)
		{
			Index(i)=jj;
			IndexRev(jj)=i;
			jj++;
		}
	}
}

void BuildStiffness(element& TestElem)
{
	BuildIndex();
	/*for (long int i=0; i<NodesNum; i++)
			for(long int elem=0;elem<ElemsNum;elem++)
				for (int j=0;j<4;j++)
					if (i==Connectivity(elem,j))
					{

								Stiffness(2*i,2*Connectivity(elem,0))		 = 0.00;
								Stiffness(2*i+1,2*Connectivity(elem,0))	   = 0.00;
								Stiffness(2*i,2*Connectivity(elem,0)+1)	   = 0.00;
								Stiffness(2*i+1,2*Connectivity(elem,0)+1)	 = 0.00;

								Stiffness(2*i,2*Connectivity(elem,1))		 = 0.00;
								Stiffness(2*i+1,2*Connectivity(elem,1))	   = 0.00;
								Stiffness(2*i,2*Connectivity(elem,1)+1)	   = 0.00;
								Stiffness(2*i+1,2*Connectivity(elem,1)+1)	 = 0.00;

								Stiffness(2*i,2*Connectivity(elem,2))		 = 0.00;
								Stiffness(2*i+1,2*Connectivity(elem,2))	   = 0.00;
								Stiffness(2*i,2*Connectivity(elem,2)+1)	   = 0.00;
								Stiffness(2*i+1,2*Connectivity(elem,2)+1)	 = 0.00;

								Stiffness(2*i,2*Connectivity(elem,3))		 = 0.00;
								Stiffness(2*i+1,2*Connectivity(elem,3))	   = 0.00;
								Stiffness(2*i,2*Connectivity(elem,3)+1)	   = 0.00;
								Stiffness(2*i+1,2*Connectivity(elem,3)+1)	 = 0.00;
					}*/

	cout<<"entring Kuu building"<<endl;
	////////////////////////////////////////////////////////////////////////////
	//building Kuu , Fu=Lu-Kub*Db :. Kuu*Du=Fu

	size_t ii(0),jj(0);

	for (size_t elem=0; elem<ElemsNum; elem++)
		for (size_t i=0;i<8;i++)
			for (size_t j=0;j<8;j++)
			{
				ii = Connectivity(elem,i);
				jj = Connectivity(elem,j);
				if ( Loads(ii) && Index(ii)<UNum && (!Fu(Index(ii))))
					Fu(Index(ii))=Loads(ii);
				/*if (Loads(2*ii+1) && Index(2*ii+1)<UNum && (!Fu(Index(2*ii+1))))
					Fu(Index(2*ii+1))=Loads(2*ii+1);
				*/
				////////////////////////////////////////////////////

				if (Index(ii) < UNum && Index(jj) < UNum && Index(ii) <= Index(jj))

					KuuS(Index(ii),Index(jj)) 	 += TestElem.k(i,j);

				else if (Index(ii) < UNum && Index(jj) >= UNum)

					Fu(Index(ii)) -= TestElem.k(i,j)*Disps(jj) ;

				else

					Kb(Index(ii)-UNum,Index(jj)) += TestElem.k(i,j);



				/*if(Index(2*ii+1) < UNum && Index(2*jj) < UNum)

					Kuu(Index(2*ii+1),Index(2*jj)) 	 += TestElem.k(2*i+1,2*j);

				else if (Index(2*ii+1) < UNum && Index(2*jj) >= UNum)

					Fu(Index(2*ii+1)) -= TestElem.k(2*i+1,2*j)*Disps(2*jj) ;

				else

					Kb(Index(2*ii+1)-UNum,Index(2*jj)) += TestElem.k(2*i+1,2*j);



				if(Index(2*ii) < UNum && Index(2*jj+1) < UNum)

					Kuu(Index(2*ii),Index(2*jj+1)) 	 += TestElem.k(2*i,2*j+1);

				else if (Index(2*ii) < UNum && Index(2*jj+1) >= UNum)

					Fu(Index(2*ii)) -= TestElem.k(2*i,2*j+1)*Disps(2*jj+1) ;

				else

					Kb(Index(2*ii)-UNum,Index(2*jj+1)) += TestElem.k(2*i,2*j+1);



				if(Index(2*ii+1) < UNum && Index(2*jj+1) < UNum)

					Kuu(Index(2*ii+1),Index(2*jj+1)) 	 += TestElem.k(2*i+1,2*j+1);

				else if (Index(2*ii+1) < UNum && Index(2*jj+1) >= UNum)

					Fu(Index(2*ii+1)) -= TestElem.k(2*i+1,2*j+1)*Disps(2*jj+1) ;

				else

					Kb(Index(2*ii+1)-UNum,Index(2*jj+1)) += TestElem.k(2*i+1,2*j+1);
				*/
				/////////////////////////////////////////////////////////////////////////////////////////

				/*Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j))		 += TestElem.k(2*i,2*j);
				Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j))	   += TestElem.k(2*i+1,2*j);
				Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j)+1)	   += TestElem.k(2*i,2*j+1);
				Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j)+1)	 += TestElem.k(2*i+1,2*j+1);*/
			}

	cout<<"Kuu & Fu building is done"<<endl;
}
////////////////////////////////////////// Conjugate gradient solver
Vector& CGSolve(const CMatrix &A,const Vector &B,Vector &X,bool cond=true)
{
	size_t XNum=X.size();

	if (XNum==2)  //For A:2*2 Matrix solve is in expilicit form
	{
		CMatrix InvA(XNum,XNum);
		double DetA=A(0,0)*A(1,1)-A(0,1)*A(1,0);
		InvA(0,0)=(1.00/DetA)*A(1,1);
		InvA(1,0)=(-1.00/DetA)*A(1,0);
		InvA(1,1)=(1.00/DetA)*A(0,0);

		axpy_prod(InvA,B,X);
		return X;
	}

	/*CMatrix Kuu(UNum,UNum);
	CMatrix Kb(EDisps.size1(),2*NodesNum);
	Vector Du(UNum);
	Vector Index(2*NodesNum);
	long int ii=0;
	long int jj=0;
	for (long int i=0;i<2*NodesNum;i++)
		{
			bool exist=false;
			for (long int j=0;j<EDisps.size1();j++)
				if (i==EDisps(j,0))
				{
					Index(2*NodesNum-1-ii)=i;
					exist=true;
					ii++;
					break;
				}
			if (exist==false)
			{
				Index(jj)=i;
				jj++;
			}
		}
	cout<<"entring Kuu building"<<endl;
	////////////////////////////////////////////////////////////////////////////
	//building Kuu , Fu=Lu-Kub*Db :. Kuu*Du=Fu
	for (iterator1 iterow=Stiffness.begin1();iterow != Stiffness.end1(); ++iterow)
		{
				if (Loads(iterow.index1()) && Index(iterow.index1())<UNum)
					Fu(Index(iterow.index1()))=Loads(iterow.index1());
			for(iterator2 itecol=iterow.begin();itecol != iterow.end(); ++itecol)
				{
					if(Index(itecol.index1()) < UNum && Index(itecol.index2()) < UNum)
					{
						Kuu(Index(itecol.index1()),Index(itecol.index2()))=Stiffness(itecol.index1(),itecol.index2());
					}
					else if(Index(itecol.index1()) < UNum && Index(itecol.index2()) >= UNum )
					{
						Fu(itecol.index2()) -= Stiffness(itecol.index1(),itecol.index2())*Disps(itecol.index2());
					}
					else if(Index(itecol.index1()) >= UNum)
					{
						Kb(Index(itecol.index1())-UNum,Index(itecol.index2()))=Stiffness(itecol.index1(),itecol.index2());
					}
					//cout<<Kuu(itecol.index1(),itecol.index2())<<endl;
				}
		}

	cout<<"Kuu & Fu building is done"<<endl<<"Entering solving"<<endl;*/
	/////////////////////////////////////////////////////////////////////////////////////////////////
	//zero the Du vector
	//Computation of Du with iterative method: Conjugational Gradient

	/*for (size_t i=0;i<XNum;i++)
		X(i)=0;*/

	X.clear();														//should be uncommented when A,B got changed too much
	Vector residual(XNum);
	Vector P(XNum);
	Vector helper(XNum);
	Vector z(XNum);
	residual=B-axpy_prod(A,X,helper);
	P=residual;
	double alfa;
	double beta;
	double inprod;
	double inprod2;
	double norm2B;
	inprod=prec_inner_prod(residual,residual);
	double norm2res=norm2B=sqrt(inprod);
	long int j=0;
	cout << "just before loop" << endl;
	while ((norm2res/norm2B)>0.0000001)
	{
		//inprod=prec_inner_prod(residual,residual);
		alfa=inprod/(prec_inner_prod(axpy_prod(A,P,helper),P));
		X += alfa*P;
		residual -= alfa*helper;
		/*axpy_prod(A, X, z);
		z = z - B;
		//norm2res=norm_2(z);
		norm2res=sqrt(prec_inner_prod(z,z));*/
		//cout << j+1 << "  " << norm2res << endl;
		inprod2=prec_inner_prod(residual,residual);
		beta=inprod2/inprod;
		inprod=inprod2;
		norm2res=sqrt(inprod2);
		/*if(j%10==0)
			cout<<"norm2res in "<<j<<" = "<<norm2res<<endl;*/
		P=residual+beta*P;

		j++;
	}
	cout<<"number of iteration= "<<j<<endl;
	axpy_prod(A, X, z);
	z -= B;
	cout<<"norm infinity= "<<norm_inf(z)<<"  ,norm 2= "<<norm_2(z)<<endl;
	if (cond == true)
	{
		for (size_t i=0;i<UNum;i++)								//Locating Du in Disps vector
			Disps(IndexRev(i))=X(i);
	}
	//cout<<Disps<<endl;
	/*double maxz=0;
	double minz=0;
	for (long int i=0;i<z.size();i++)
		{
			if(z(i)>maxz)
				maxz=z(i);
			if(z(i)<minz)
				minz=z(i);
		}
	cout<<minz<<"  ---- sum of z  "<<maxz<<endl;*/

	//WriteMatrixMarketMatrix("A.mm", Kuu, false);
	//WriteMatrixMarketVector("B.mm", Fu);
	//WriteMatrixMarketVector("X.mm", Du);

	/*cout<<"Entering Solving equations"<<endl;
	lu_factorize(Kuu,Pm);
	Du=Fu;
	//ofstream out("out.txt");
	//out<<Kuu;
	lu_substitute(Kuu,Pm,Du);*/
	return X;
}
///////////// TAUCS Solver. A assumed to be symmetric & double
int TAUCS_Solve(CSMatrix &A,Vector &B,Vector &X,bool cond=true)
{
	size_t XNum=A.size1();
	taucs_ccs_matrix K;
	K.n=K.m=XNum;
	K.flags= TAUCS_DOUBLE | TAUCS_SYMMETRIC  | TAUCS_LOWER;
	K.values.d=A.value_data().begin();
	K.colptr=A.index1_data().begin();
	K.rowind=A.index2_data().begin();

	boost::timer t;
	const char* options[] = {"taucs.factor.LLT=true", "taucs.factor.ordering=amd", "taucs.factor.ll=true", NULL };
	int Result=taucs_linsolve(&K,NULL,1,X.data().begin(),B.data().begin(),(char**)options,NULL);
	if (Result != TAUCS_SUCCESS)
		cout<<"Error in Solving : ";

	switch (Result)
	{
	case 0:
	{
		/*boost::numeric::ublas::symmetric_adaptor <CSMatrix, boost::numeric::ublas::upper> AS(A);
		Vector z(X.size());
		axpy_prod(AS, X, z);
		z -= B;
		cout<<"norm infinity= "<<norm_inf(z)<<"  ,norm 2= "<<norm_2(z)<<endl;*/
		if (MyMPIRank == 0)
			cout<<"TAUCS Solve is done , "<<t.elapsed()<<endl;
		break;
	}
	case -1:
		cout<<"TAUCS_ERROR"<<endl;
		exit(1);
	case -2:
		cout<<"TAUCS_ERROR_NOMEM"<<endl;
		exit(1);
	case -3:
		cout<<"TAUCS_ERROR_BADARGS"<<endl;
		exit(1);
	}

	if (cond)
	{
		for (size_t i=0;i<UNum;i++)								//Locating Du in Disps vector
			Disps(IndexRev(i))=X(i);
	}
	return Result;
}
//TAUCS_Solve 2////
int TAUCS_Solve2(CSMatrix &A,Vector &B1,Vector &X1,Vector &B2,Vector &X2,bool cond=true)
{
	size_t XNum=A.size1();
	taucs_ccs_matrix K;
	K.n=K.m=XNum;
	/////////////////////////////////////// Putting B1 and B2 in array B
	//cout<<"here "<<XNum<<endl;
	double *B = new double[XNum*2], *X = new double[XNum*2];
	//cout<<"here2"<<endl;
	for (size_t i=0;i<XNum;i++)
	{
		B[i]=B1(i);
	}

	for (size_t i=0;i<XNum;i++)
	{
		B[i+XNum]=B2(i);
	}

	///////////////////////////////////////
	K.flags= TAUCS_DOUBLE | TAUCS_SYMMETRIC  | TAUCS_LOWER;
	K.values.d=A.value_data().begin();
	K.colptr=A.index1_data().begin();
	K.rowind=A.index2_data().begin();

	cout<<"TAUCS Solve is started"<<endl;

	boost::timer t;

	const char* options[] = {"taucs.factor.LLT=true", "taucs.factor.ordering=amd", "taucs.factor.ll=true", NULL };
	int Result=taucs_linsolve(&K,NULL,2,X,B,(char**)options,NULL);
	if (Result != TAUCS_SUCCESS)
		cout<<"Error in Solving : ";

	switch (Result)
	{
	case 0:
	{
		/*boost::numeric::ublas::symmetric_adaptor <CSMatrix, boost::numeric::ublas::upper> AS(A);
		Vector z(X.size());
		axpy_prod(AS, X, z);
		z -= B;
		cout<<"norm infinity= "<<norm_inf(z)<<"  ,norm 2= "<<norm_2(z)<<endl;*/
		cout<<"TAUCS Solve is done, "<< t.elapsed() <<endl;
		break;
	}
	case -1:
		cout<<"TAUCS_ERROR"<<endl;
		exit(1);
	case -2:
		cout<<"TAUCS_ERROR_NOMEM"<<endl;
		exit(1);
	case -3:
		cout<<"TAUCS_ERROR_BADARGS"<<endl;
		exit(1);
	}
	/////////////////////////////////////// Putting X in X1 and X2 back
	for (size_t i=0;i<XNum;i++)
	{
		X1(i)=X[i];
	}
	for (size_t i=0;i<XNum;i++)
	{
		X2(i)=X[i+XNum];
	}
	///////////////////////////////////////
	if (cond)
	{
		for (size_t i=0;i<UNum;i++)								//Locating Du in Disps vector
			Disps(IndexRev(i))=X[i];
	}

	delete[] B;
	delete[] X;

	return Result;
}
//Taucs factorization or/and afterward solve
int Taucs_Factor_Solve(CSMatrix &A,void **F,int whattodo=1,Vector &B=Du,Vector &X=Du,bool cond=false)
{
	size_t XNum=A.size1();
	int Result=-10;
	taucs_ccs_matrix K;
	K.n=K.m=XNum;
	K.flags= TAUCS_DOUBLE | TAUCS_SYMMETRIC  | TAUCS_LOWER;
	K.values.d=A.value_data().begin();
	K.rowind=A.index2_data().begin();
	if (whattodo == 1)
	{
		const char* options[] = {"taucs.factor.LLT=true", "taucs.factor.ordering=amd", "taucs.factor.ll=true", NULL };
		Result=taucs_linsolve(&K,F,0,NULL,NULL,(char**)options,NULL);
	}
	else if (whattodo == 2)
	{
		const char* options[] = {"taucs.factor=false", NULL };
		Result=taucs_linsolve(&K,F,1,X.data().begin(),B.data().begin(),(char**)options,NULL);
	}
	if (Result != TAUCS_SUCCESS)
		cout<<"Error in Solving : ";

	switch (Result)
	{
	case 0:
	{
		/*boost::numeric::ublas::symmetric_adaptor <CSMatrix, boost::numeric::ublas::upper> AS(A);
		Vector z(X.size());
		axpy_prod(AS, X, z);
		z -= B;
		cout<<"norm infinity= "<<norm_inf(z)<<"  ,norm 2= "<<norm_2(z)<<endl;*/
		//if (MyMPIRank == 0)
			//cout<<"TAUCS Factor/Solve is done , "<<endl;
		break;
	}
	case -1:
		cout<<"TAUCS_ERROR"<<endl;
		exit(1);
	case -2:
		cout<<"TAUCS_ERROR_NOMEM"<<endl;
		exit(1);
	case -3:
		cout<<"TAUCS_ERROR_BADARGS"<<endl;
		exit(1);
	}
	if (cond)
	{
		for (size_t i=0;i<UNum;i++)								//Locating Du in Disps vector
			Disps(IndexRev(i))=X(i);
	}
	return Result;
}

int Taucs_FreeFactor(void **F)
{
	int result=-10;
	const char* options[] = {"taucs.factor.LLT=true", "taucs.factor.ordering=amd", "taucs.factor.ll=true", NULL };
	result=taucs_linsolve(NULL,F,0,NULL,NULL,(char**)options,NULL);

	return result;
}
// Super LU solve  //////////////////////////////////////////////////////////////////////////////////
class SLUSolve
{
	SuperMatrix A,B;
	SuperMatrix AC, L, U;
	DNformat Bstore;
	NCformat Astore;
	superlumt_options_t superlumt_options;
	fact_t fact;
	trans_t trans;
	yes_no_t refact, usepr;
	double u, drop_tol;
	//int perm_c[bcm.size1()]; /* column permutation vector */
	//int perm_r[bcm.size1()]; /* row permutations from partial pivoting */
	void *work;
	size_t n;
	int *perm_c, *perm_r; 				/* column and row permutation vectors */
	Gstat_t Gstat;
	flops_t flopcnt;
public:
	SLUSolve(const size_t nsize): n(nsize)
	{
		fact  = EQUILIBRATE;
		trans = NOTRANS;
		panel_size = sp_ienv(1);
		relax = sp_ienv(2);
		u = 1.0;
		usepr = NO;
		drop_tol = 0.0;
		work = NULL;
		lwork = 0;
		superlumt_options.SymmetricMode=YES;
		perm_c=new int[n];
		perm_r=new int[n];
	}
	~SLUSolve()
	{
		delete []perm_c;
		delete []perm_r;
	}
	bool FTSP ,LTSP;
	int dPrint_CompCol_Matrix(SuperMatrix *A);
	int dPrint_Dense_Matrix(SuperMatrix *A);

};

int SLUSolve::dPrint_CompCol_Matrix(SuperMatrix *A)
{
	NCformat	 *Astore;
	register int i;
	double	   *dp;

	printf("\nCompCol matrix: ");
	printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);
	Astore = (NCformat *) A->Store;
	dp = (double *) Astore->nzval;
	printf("nrow %d, ncol %d, nnz %d\n", A->nrow,A->ncol,Astore->nnz);
	printf("\nnzval: ");
	for (i = 0; i < Astore->nnz; ++i) printf("%f  ", dp[i]);
	printf("\nrowind: ");
	for (i = 0; i < Astore->nnz; ++i) printf("%d  ", Astore->rowind[i]);
	printf("\ncolptr: ");
	for (i = 0; i <= A->ncol; ++i) printf("%d  ", Astore->colptr[i]);
	printf("\nend CompCol matrix.\n");

	return 0;
}

int SLUSolve::dPrint_Dense_Matrix(SuperMatrix *A)
{
	DNformat	 *Astore;
	register int i;
	double	   *dp;

	printf("\nDense matrix: ");
	printf("Stype %d, Dtype %d, Mtype %d\n", A->Stype,A->Dtype,A->Mtype);
	Astore = (DNformat *) A->Store;
	dp = (double *) Astore->nzval;
	printf("nrow %d, ncol %d, lda %d\n", A->nrow,A->ncol,Astore->lda);
	printf("\nnzval: ");
	for (i = 0; i < A->nrow; ++i) printf("%f  ", dp[i]);
	printf("\nend Dense matrix.\n");

	return 0;
}

bool SLUSolve::Solve(const int nrhs,CSMatrix &Aref,Vector &B1,Vector &X1,Vector &B2=Fu,Vector &X2=Fu,const int &nprocs=2)
{
	n = Aref.size1();
	//SuperMatrix AC;
	superlumt_options.nprocs=2;
//// A definition ////////////////////
	A.nrow = n;
	A.ncol = n;
	A.Dtype = SLU_D;
	A.Mtype = SLU_GE;
	A.Stype = SLU_NC;

	Astore.colptr = Aref.index1_data().begin();
	Astore.rowind = Aref.index2_data().begin();
	Astore.nzval  = Aref.value_data().begin();
	Astore.nnz 	  = Aref.nnz();

	A.Store=&Astore;
	//dPrint_CompCol_Matrix(&A);
/////B definition ////////////////////
	B.nrow = n;
	B.ncol = nrhs;
	B.Dtype = SLU_D;
	B.Mtype = SLU_GE;
	B.Stype = SLU_DN;
	Bstore.lda = n;
	double Bnzval[nrhs*Aref.size1()];
//#pragma omp parallel for
	for (size_t ivec=0;ivec<n;ivec++)
		Bnzval[ivec]=B1(ivec);
	if (nrhs==2)
	{
//#pragma omp parallel for
		for (size_t ivec=0;ivec<n;ivec++)
			Bnzval[ivec+n]=B2(ivec);
	}
	Bstore.nzval=Bnzval;
	B.Store = &Bstore;
	//dPrint_Dense_Matrix(&B);
///////////////////////////////////////
	if (FTSP)
	{
		//first time of solving process to generate the perm_c
		/********************************
		 * THE FIRST TIME FACTORIZATION *
		 ********************************/
		cout<<"Entering first time"<<endl;
		/* ------------------------------------------------------------
		   Allocate storage and initialize statistics variables.
		   ------------------------------------------------------------*/
		StatAlloc(n, nprocs, panel_size, relax, &Gstat);
		StatInit(n, nprocs, &Gstat);

		/* ------------------------------------------------------------
		   Get column permutation vector perm_c[], according to permc_spec:
		   permc_spec = 0: natural ordering
		   permc_spec = 1: minimum degree ordering on structure of A'*A
		   permc_spec = 2: minimum degree ordering on structure of A'+A
		   permc_spec = 3: approximate minimum degree for unsymmetric matrices
		   ------------------------------------------------------------*/
		permc_spec = 1;
		/* ------------------------------------------------------------
		   Initialize the option structure superlumt_options using the
		   user-input parameters;
		   Apply perm_c to the columns of original A to form AC.
		   ------------------------------------------------------------*/
		refact= NO;
		pdgstrf_init(nprocs, fact, trans, refact, panel_size, relax,
					 u, usepr, drop_tol, perm_c, perm_r,
					 work, lwork, &A, &AC, &superlumt_options, &Gstat);

		/* ------------------------------------------------------------
		   Compute the LU factorization of A.
		   The following routine will create nprocs threads.
		   ------------------------------------------------------------*/
		pdgstrf(&superlumt_options, &AC, perm_r, &L, &U, &Gstat, &info);

		flopcnt = 0;
		for (i = 0; i < nprocs; ++i) flopcnt += Gstat.procstat[i].fcops;
		cout<<"flopcnt1= "<<flopcnt<<endl;
		Gstat.ops[FACT] = flopcnt;

		/* ------------------------------------------------------------
		   Solve the system A*X=B, overwriting B with X.
		   ------------------------------------------------------------*/
		dgstrs(trans, &L, &U, perm_r, perm_c, &B, &Gstat, &info);
		//dPrint_Dense_Matrix(&B);
#ifdef DebugSLU
		cout<<Bnzval[0]<<" , "<<Bnzval[1]<<" , "<<Bnzval[2]<<endl;
		cout<<Bnzval[3]<<" , "<<Bnzval[4]<<" , "<<Bnzval[5]<<endl;
		cout<<Bnzval[0]*Aref(0,0)+Aref(0,1)*Bnzval[1]+Aref(0,2)*Bnzval[2]<<" , "<<Bnzval[0]*Aref(1,0)+Aref(1,1)*Bnzval[1]+Aref(1,2)*Bnzval[2]<<" , "<<Bnzval[0]*Aref(2,0)+Aref(2,1)*Bnzval[1]+Aref(2,2)*Bnzval[2]<<endl;
		cout<<Bnzval[3]*Aref(0,0)+Aref(0,1)*Bnzval[4]+Aref(0,2)*Bnzval[5]<<" , "<<Bnzval[3]*Aref(1,0)+Aref(1,1)*Bnzval[4]+Aref(1,2)*Bnzval[5]<<" , "<<Bnzval[3]*Aref(2,0)+Aref(2,1)*Bnzval[4]+Aref(2,2)*Bnzval[5]<<endl;
#endif
		//Putting B in X1,X2 back///////////
		//#pragma omp parallel for num_threads(2)
		for (size_t ivec=0;ivec<n;ivec++)
			X1(ivec)=Bnzval[ivec];
		if (nrhs==2)
		{
			//#pragma omp parallel for num_threads(2)
			for (size_t ii=0;ii<UNum;ii++)								//Locating Du in Disps vector
				Disps(IndexRev(ii))=X1(ii);
			for (size_t ivec=0;ivec<n;ivec++)
				X2(ivec)=Bnzval[ivec+n];
		}
		////////////////////////////////////
		return true;
	}
	/*********************************
	 * THE SUBSEQUENT FACTORIZATIONS *
	 *********************************/
	cout<<"entering second part of SLU solver"<<endl;
	usepr=YES;
	//fact=FACTORED;
	/* ------------------------------------------------------------
	   Re-initialize statistics variables and options used by the
	   factorization routine pdgstrf().
	   ------------------------------------------------------------*/
	StatAlloc(n, nprocs, panel_size, relax, &Gstat);
	StatInit(n, nprocs, &Gstat);
	refact= YES;
	pdgstrf_init(nprocs, fact, trans, refact, panel_size, relax,
				 u, usepr, drop_tol, perm_c, perm_r,
				 work, lwork, &A, &AC, &superlumt_options, &Gstat);

	/* ------------------------------------------------------------
	   Compute the LU factorization of A.
	   The following routine will create nprocs threads.
	   ------------------------------------------------------------*/
	pdgstrf(&superlumt_options, &AC, perm_r, &L, &U, &Gstat, &info);

	flopcnt = 0;
	for (i = 0; i < nprocs; ++i) flopcnt += Gstat.procstat[i].fcops;
	cout<<"flpcnt= "<<flopcnt<<endl;
	Gstat.ops[FACT] = flopcnt;

	/* ------------------------------------------------------------
	   Re-generate right-hand side B, then solve A*X= B.
	   ------------------------------------------------------------*/
	dgstrs(trans, &L, &U, perm_r, perm_c, &B, &Gstat, &info);


	/* ------------------------------------------------------------
	  Deallocate storage after factorization.
	  ------------------------------------------------------------*/
	if (LTSP)
		pxgstrf_finalize(&superlumt_options, &AC);
	fflush(stdout);
	StatFree(&Gstat);
	//Putting B in X1,X2 back///////////
//#pragma omp parallel for
	for (size_t ivec=0;ivec<n;ivec++)
		X1(ivec)=Bnzval[ivec];
	if (nrhs==2)
	{
//#pragma omp parallel for
		for (size_t ii=0;ii<UNum;ii++)								//Locating Du in Disps vector
			Disps(IndexRev(ii))=X1(ii);
		for (size_t ivec=0;ivec<n;ivec++)
			X2(ivec)=Bnzval[ivec+n];
	}
	////////////////////////////////////
	return true;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////
class mma
{
	double a0;
	double epsimin;

	size_t n1;
	size_t n2;

	Vector a;
	Vector b;
	Vector c;
	Vector d;


protected:				//If is used as nested object should be changed to public
	Vector xmin;
	Vector xmax;

	Vector alfa;
	Vector beta;
	Vector p0;
	Vector q0;

	CMatrix P;
	CMatrix Q;
public:
	mma():a0(1.00),m(1),iter(0),maxite(40)
	{
		//cout<<"entering constructor:mma"<<endl;
		inoptimize>>n>>xPrimitive;

		inoptimize>>n1>>n2;									// added to mma to get n by multiplied numbers: n1 , n2
		if ((n1+1)*(n2+1) != n)
		{
			if (MyMPIRank == 0)
				cout<<"n != n1*n2  in continue n=n1*n2 was done"<<endl<<endl;
			n=(n1+1)*(n2+1);
		}

		xval.resize(n);
		xold1.resize(n);
		xold2.resize(n);
		low.resize(n);
		upp.resize(n);
		xmin.resize(n);
		xmax.resize(n);
		alfa.resize(n);
		beta.resize(n);
		p0.resize(n);
		q0.resize(n);

		fval.resize(m);
		a.resize(m);
		b.resize(m);
		c.resize(m);
		d.resize(m);

		df0dx.resize(n);
		dfdx.resize(m,n);
		df0dx2.resize(n,false);
		P.resize(m,n,false);
		Q.resize(m,n,false);

		for (size_t i=0;i<n;i++)
		{
			xval(i) =1.00*xPrimitive;
			xold1(i)=1.00*xPrimitive;
			xold2(i)=1.00*xPrimitive;

			low(i)=xmin(i)=0.001;
			upp(i)=xmax(i)=1.00;
		}

		for (size_t i=0;i<m;i++)
		{
			a(i)=0.00;
			d(i)=0.00;
			c(i)=1000.00;
		}
		//cout<<"exiting constructor:mma"<<endl;


	}
	Vector low;/////////////////////////////////////!!!!!!!!!!!!
	Vector upp;/////////////////////////////////////!!!!!!!!!!!!

	size_t m;
	size_t n;

	double xPrimitive;
	double f0val;

	long int iter;
	long int maxite;
	long int counter;

	Vector fval;
	Vector xval;
	Vector xold1;
	Vector xold2;
	Vector df0dx;

	CVector df0dx2;

	Matrix dfdx;
	CMatrix dfdx2;

	//void init();
	void mmasub();
	void subsolve();
};

void mma::mmasub()
{
	if (MyMPIRank == 0)
		cout<<"Entering mmasub"<<endl;
	epsimin=sqrt(m+n)*pow(10.00,-9.00);
	double feps = 0.000001;
	double asyinit=0.50;
	double asyincr=1.20;
	double asydecr=0.70;
	double albefa=0.10;

	Vector een(n),eem(m),zeron(n),zerom(m),zzz(n),factor(n),helperm(m),helperm2(m),helpern(n),helpern2(n);
	CMatrix helpercnn(n,n),helpercmn(m,n);
	Matrix zeromn(m,n);

	for (size_t i=0;i<n;i++)
	{
		for (size_t j=0;j<m;j++)
			zeromn(j,i)=0.00;
		een(i)=1.00;
		zeron(i)=0.00;
	}

	for (size_t i=0;i<m;i++)
		eem(i)=1.00;
	// Calculation of the asymptotes low and upp :
	if (iter < 2.50)
	{
		low = xval - asyinit*(xmax-xmin);
	}
	else
	{
		OboProd(xval-xold1,xold1-xold2,zzz);
		factor=een;
		for (size_t i=0;i<n;i++)
		{
			if (zzz(i)>0.00)
				factor(i)=asyincr;
			else if (zzz(i)<0.00)
				factor(i)=asydecr;
		}
		OboProd(factor,xold1-low,helpern);
		low=xval-helpern;
		OboProd(factor,upp-xold1,helpern);
		upp=xval+helpern;
		//cout<<"low="<<low<<endl;
		//cout<<"upp="<<upp<<endl;
		//cout<<"factor="<<factor<<endl;
	}

	// Calculation of the bounds alfa and beta :
	zzz=low+albefa*(xval-low);
	OboMax(zzz,xmin,alfa);
	zzz=upp-albefa*(upp-xval);
	OboMin(zzz,xmax,beta);

	// Calculations of p0, q0, P, Q and b.
	Vector ux1(n) , ux2(n) , ux3(n) , xl1(n) , xl2(n) , xl3(n) , ul1(n) , ulinv1(n) , uxinv1(n)
	, xlinv1(n) , uxinv3(n) ,xlinv3(n);
	ux1=upp-xval;
	OboProd(ux1,ux1,ux2);
	xl1=xval-low;
	OboProd(xl1,xl1,xl2);
	OboProd(xl2,xl1,xl3);
	ul1=upp-low;
	OboDivid(een,ul1,ulinv1);
	OboDivid(een,ux1,uxinv1);
	OboDivid(een,xl1,xlinv1);
	OboDivid(een,ux3,uxinv3);
	OboDivid(een,xl3,xlinv3);

	Vector diap(n),diaq(n),dg0dx2(n),del0(n),delpos0(n);

	OboProd(ux3,xl1,helpern);
	OboDivid(helpern,2.00*ul1,diap);
	OboDivid(helpern,2.00*ul1,diaq);

	/*cout<<ux3<<"ux3"<<endl;
	cout<<xl1<<"xl1"<<endl;
	cout<<ul1<<"ul1"<<endl;
	cout<<upp<<"upp"<<endl;
	cout<<low<<"low"<<endl;
	cout<<xval<<"xval"<<endl;
	cout<<diap<<"diap"<<endl;
	cout<<diaq<<"diaq"<<endl;
	cout<<p0<<"p0"<<endl;
	cout<<zzz<<"zzz"<<endl;
	cout<<xval<<"xval"<<endl;
	cout<<xold1<<"xold1"<<endl;
	cout<<xold2<<"xold2"<<endl;*/

	delpos0=p0=q0=zeron;
	//cout<<p0<<"p01"<<endl;
	for (size_t i=0;i<n;i++)
	{
		if (df0dx(i)>0.00)
			p0(i)=df0dx(i);
		else
			q0(i)=-df0dx(i);
	}
	//cout<<df0dx<<"=df0dx"<<endl;

	OboAbs(df0dx,helpern);
	p0 += 0.001*helpern+feps*ulinv1;
	OboProd(p0,ux2,p0);

	OboAbs(df0dx,helpern);
	q0 += 0.001*helpern+feps*ulinv1;
	OboProd(q0,xl2,q0);
	OboDivid(p0,ux3,helpern);
	OboDivid(q0,xl3,helpern2);
	dg0dx2=2.00*(helpern+helpern2);
	del0=df0dx2-dg0dx2;
	for (size_t i=0;i<n;i++)
		if (del0(i)>0)
			delpos0(i)=del0(i);
	OboProd(delpos0,diap,helpern);
	p0 +=helpern;
	OboProd(delpos0,diaq,helpern);
	q0 +=helpern;

	P=zeromn;
	Q=zeromn;
	CMatrix dgdx2(m,n);

	for (size_t i=0;i<m;i++)
		for (size_t j=0;j<n;j++)
		{
			if (dfdx(i,j)>0)
				P(i,j)=dfdx(i,j);
			else if (dfdx(i,j)<0)
				Q(i,j)=-dfdx(i,j);
		}


	SpDiag(ux2,helpercnn);
	P=axpy_prod(P,helpercnn,helpercmn);
	SpDiag(xl2,helpercnn);
	Q=axpy_prod(Q,helpercnn,helpercmn);

	SpDiag(uxinv3,helpercnn);
	dgdx2+= 2.00*axpy_prod(Q,helpercnn,helpercmn);

	Matrix del(m,n);
	CMatrix delpos(m,n);
	del=dfdx2-dgdx2;

	for (size_t i=0;i<m;i++)
		for (size_t j=0;j<n;j++)
			if (del(i,j)>0)
				delpos(i,j)=del(i,j);

	SpDiag(diap,helpercnn);
	axpy_prod(delpos,helpercnn,helpercmn);
	P += helpercmn;
	SpDiag(diaq,helpercnn);
	axpy_prod(delpos,helpercnn,helpercmn);
	Q += helpercmn;
	//cout<<P<<"=P"<<endl;
	//cout<<Q<<"=Q"<<endl;
	b=axpy_prod(P,uxinv1,helperm)+axpy_prod(Q,xlinv1,helperm2)-fval;

	/*cout<<b<<"=b"<<endl;
	cout<<alfa<<"=alfa"<<endl;
	cout<<beta<<"=beta"<<endl;
	cout<<a<<"=a"<<endl;
	cout<<p0<<"=p0"<<endl;
	cout<<xmax<<"=xmax"<<endl;
	cout<<zzz<<"=zzz"<<endl;
	cout<<low<<"=low"<<endl;
	cout<<upp<<"=upp"<<endl;
	cout<<factor<<"=factor"<<endl;
	cout<<albefa<<"=albefa"<<endl;*/

	subsolve();
}

void mma::subsolve()
{
	if (MyMPIRank == 0)
		cout<<"entering subsolve"<<endl;
	Vector een(n);
	Vector eem(m);
	for (size_t i=0;i<n;i++)
		een(i)=1.00;
	for (size_t i=0;i<m;i++)
		eem(i)=1.00;

	double epsi=1.00;
	Vector epsvecn(n),epsvecm(m),x(n),y(m),lam(m),xsi(n),eta(n),mu(m),s(m);
	epsvecn=epsi*een;
	epsvecm=epsi*eem;

	x=0.50*(alfa+beta);
	y=eem;
	double z=1.00;

	lam=eem;
	OboDivid(een,x-alfa,xsi);
	OboMax(xsi,een,xsi);
	OboDivid(een,beta-x,eta);
	OboMax(eta,een,eta);

	OboMax(eem,0.50*c,mu);
	double zet=1.00;
	s=eem;
	long int itera=0,ittt=0;

	/*cout<<mu<<"=mu"<<endl;
	cout<<eta<<"=eta"<<endl;
	cout<<xsi<<"=xsi"<<endl;
	cout<<lam<<"=lam"<<endl;*/

	CMatrix helpercmn(m,n),helpercmm(m,m),helpercnn(n,n);
	Vector ux1(n),xl1(n),ux2(n),xl2(n),uxinv1(n),xlinv1(n),uxinv2(n),xlinv2(n),plam(n),qlam(n),gvec(m)
	,dpsidx(n),rex(n),rey(m),helpern(n),helperm(m),helpern2(n),helperm2(m),relam(m),rexsi(n)
	,reeta(n),remu(m),res(m),residu(3*n+4*m+2);

	double rez,rezet,residunorm,residumax;
	counter=0;
	while (epsi>epsimin)
	{
		epsvecn=epsi*een;
		epsvecm=epsi*eem;
		ux1=upp-x;
		xl1=x-low;
		OboProd(ux1,ux1,ux2);
		OboProd(xl1,xl1,xl2);

		OboDivid(een,ux1,uxinv1);
		OboDivid(een,xl1,xlinv1);

		axpy_prod(trans(P),lam,helpern);
		plam=p0+helpern;
		axpy_prod(trans(Q),lam,helpern);
		qlam=q0+helpern;

		axpy_prod(P,uxinv1,gvec);
		axpy_prod(Q,xlinv1,helperm);
		gvec += helperm;

		OboDivid(plam,ux2,dpsidx);
		OboDivid(qlam,xl2,helpern);
		dpsidx -= helpern;

		rex=dpsidx-xsi+eta;
		OboProd(d,y,helperm);
		rey=c+helperm-mu-lam;
		rez=a0-zet-prec_inner_prod(a,lam);
		relam=gvec-z*a-y+s-b;

		OboProd(xsi,x-alfa,rexsi);
		rexsi -= epsvecn;
		OboProd(eta,beta-x,reeta);
		reeta -= epsvecn;
		OboProd(mu,y,remu);
		remu -= epsvecm;
		rezet=zet*z-epsi;
		OboProd(lam,s,res);
		res -= epsvecm;

		///////////////////////////////////////////////
		for (size_t i=0;i<n;i++)
		{
			residu(i)=rex(i);
			residu(i+n+2*m+1)=rexsi(i);
			residu(i+2*n+2*m+1)=reeta(i);
		}
		for (size_t i=0;i<m;i++)
		{
			residu(i+n)=rey(i);
			residu(i+n+m+1)=relam(i);
			residu(i+3*n+2*m+1)=remu(i);
			residu(i+3*n+3*m+2)=res(i);
		}
		residu(n+m)=rez;
		residu(3*m+3*n+1)=rezet;
		////////////////////////////////////////////////
		residunorm=norm_2(residu);
		residunorm=sqrt(prec_inner_prod(residu,residu));
		residumax=norm_inf(residu);
		//cout<<residumax<<" =residumax"<<endl;
		//cout<<residunorm<<" =residunorm"<<endl;
		ittt=0;
		while ((residumax>(0.90*epsi) )&& ittt<100)
		{
			ittt++;
			itera++;
			ux1=upp-x;
			xl1=x-low;
			OboProd(ux1,ux1,ux2);
			OboProd(xl1,xl1,xl2);

			Vector ux3(n),xl3(n),dlam(m);

			OboProd(ux1,ux2,ux3);
			OboProd(xl1,xl2,xl3);
			OboDivid(een,ux1,uxinv1);
			OboDivid(een,xl1,xlinv1);
			OboDivid(een,ux2,uxinv2);
			OboDivid(een,xl2,xlinv2);

			plam=p0+axpy_prod(trans(P),lam,helpern);
			qlam=q0+axpy_prod(trans(Q),lam,helpern);
			gvec=axpy_prod(P,uxinv1,helperm)+axpy_prod(Q,xlinv1,helperm2);
			/*cout<<gvec<<" =gvec"<<endl;
			cout<<uxinv1<<" =uxinv1"<<endl;
			cout<<P<<"=P"<<endl;*/

			CMatrix GG(m,n);

			SpDiag(uxinv2,helpercnn);
			axpy_prod(P,helpercnn,GG);
			SpDiag(xlinv2,helpercnn);
			axpy_prod(Q,helpercnn,helpercmn);
			GG -= helpercmn;

			OboDivid(plam,ux2,dpsidx);
			OboDivid(qlam,xl2,helpern);
			dpsidx -= helpern;

			//cout<<dpsidx<<" =dpsidx"<<endl;

			Vector delx(n),dely(m),dellam(m),diagx(n),diagxinv(n),diagy(m),diagyinv(m),diaglam(m),
			diaglamyi(m);
			delx = dpsidx - OboDivid(epsvecn,(x-alfa),helpern) + OboDivid(epsvecn,(beta-x),helpern2);
			dely = c + OboProd(d,y,helperm) - lam - OboDivid(epsvecm,y,helperm2);
			delz = a0 - prec_inner_prod(a,lam) - epsi/z;
			dellam=gvec - z*a - y - b + OboDivid(epsvecm,lam,helperm);
			diagx=OboDivid(plam,ux3,helpern)+OboDivid(qlam,xl3,helpern2);
			diagx = 2.00*diagx + OboDivid(xsi,(x-alfa),helpern) + OboDivid(eta,(beta-x),helpern2);
			OboDivid(een,diagx,diagxinv);
			diagy=d+OboDivid(mu,y,helperm);
			OboDivid(eem,diagy,diagyinv);
			OboDivid(s,lam,diaglam);
			diaglamyi = diaglam+diagyinv;

			Vector dx(n);
			double dz;
			/*cout<<diagyinv<<"diagyinv"<<endl;
			cout<<diagx<<"diagx"<<endl;
			cout<<diaglamyi<<"diaglamyi"<<endl;
			cout<<plam<<"plam"<<endl;
			cout<<qlam<<"qlam"<<endl;
			cout<<ux2<<"ux2"<<endl;
			cout<<xl2<<"xl2"<<endl;
			cout<<p0<<"p0"<<endl;
			cout<<P<<"P"<<endl;
			cout<<lam<<"lam"<<endl;
			cout<<ux1<<"ux1"<<endl;
			cout<<x<<"x"<<endl;
			cout<<upp<<"upp"<<endl;*/
			if (m<n)
			{
				Vector blam(m),bb(m+1),solut(m+1);
				CMatrix Alam(m,m),AA(m+1,m+1);

				blam=dellam+OboDivid(dely,diagy,helperm2)-axpy_prod(GG,OboDivid(delx,diagx,helpern),helperm);

				for (size_t i=0;i<m;i++)
					bb(i)=blam(i);
				bb(m)=delz;

				SpDiag(diaglamyi,Alam);
				Alam += axpy_prod(axpy_prod(GG,SpDiag(diagxinv,helpercnn),helpercmn),trans(GG),helpercmm);

				for (size_t i=0;i<m;i++)
				{
					AA(m,i)=AA(i,m)=a(i);
					for (size_t j=0;j<m;j++)
						AA(i,j)=Alam(i,j);
				}
				AA(m,m)=-zet/z;
				CGSolve(AA,bb,solut,false);
				for (size_t i=0;i<m;i++)
					dlam(i)=solut(i);
				dz=solut(m);
				OboDivid(-delx,diagx,dx);
				axpy_prod(trans(GG),dlam,helpern);
				dx -= OboDivid(helpern,diagx,helpern2);
			}
			else
			{
				Vector diaglamyiinv(m),dellamyi(m),axz(n),bx(n),bb(n+1),solut(n+1);
				double azz,bz;
				CMatrix Axx(n,n),AA(n+1,n+1);

				OboDivid(eem,diaglamyi,diaglamyiinv);
				dellamyi=dellam+OboDivid(dely,diagy,helperm);

				SpDiag(diagx,Axx);
				Axx += axpy_prod(trans(GG),axpy_prod(SpDiag(diaglamyiinv,helpercmm),GG,helpercmn),helpercnn);
				azz=zet/z+prec_inner_prod(a,OboDivid(a,diaglamyi,helperm));
				axpy_prod(-trans(GG),OboDivid(a,diaglamyi,helperm),axz);
				bx=delx+axpy_prod(trans(GG),OboDivid(dellamyi,diaglamyi,helperm),helpern);
				bz=delz-prec_inner_prod(a,OboDivid(dellamyi,diaglamyi,helperm));

				for (size_t i=0;i<n;i++)
				{
					AA(n,i)=AA(i,n)=axz(i);
					for (size_t j=0;j<n;j++)
						AA(i,j)=Axx(i,j);
				}
				AA(n,n)=azz;

				for (size_t i=0;i<n;i++)
					bb(i)=-bx(i);
				bb(n)=-bz;

				CGSolve(AA,bb,solut,false);
				for (size_t i=0;i<n;i++)
					dx(i)=solut(i);
				dz=solut(n);

				OboDivid(axpy_prod(GG,dx,helperm),diaglamyi,dlam);
				dlam += (-dz*OboDivid(a,diaglamyi,helperm)+OboDivid(dellamyi,diaglamyi,helperm2));
			}
			/*cout<<dlam<<"=dlam"<<endl;
			cout<<dx<<"=dx"<<endl;
			Vector dy(m),dxsi(n),deta(n),dmu(m),ds(m);
			double dzet;
			dy = OboDivid(-dely,diagy,helperm) + OboDivid(dlam,diagy,helperm2);
			dxsi = -xsi + OboDivid(epsvecn,(x-alfa),helpern);
			dxsi -= OboDivid(OboProd(xsi,dx,helpern),(x-alfa),helpern2);

			deta = -eta + OboDivid(epsvecn,(beta-x),helpern);
			deta += OboDivid(OboProd(eta,dx,helpern2),(beta-x),helpern);

			dmu  = -mu + OboDivid(epsvecm,y,helperm);
			dmu -= OboDivid(OboProd(mu,dy,helperm2),y,helperm);

			dzet = -zet + epsi/z - zet*dz/z;

			ds   = -s + OboDivid(epsvecm,lam,helperm);
			ds -= OboDivid(OboProd(s,dlam,helperm2),lam,helperm);

			Vector xx(2*n+4*m+2),dxx(2*n+4*m+2);
			for (size_t i=0;i<n;i++)
			{
				xx(2*m+1+i)=xsi(i);
				xx(2*m+n+1+i)=eta(i);

				dxx(2*m+1+i)=dxsi(i);
				dxx(2*m+n+1+i)=deta(i);
			}
			for (size_t i=0;i<m;i++)
			{
				xx(i)=y(i);
				xx(m+1+i)=lam(i);
				xx(2*n+2*m+1+i)=mu(i);
				xx(2*n+3*m+2+i)=s(i);

				dxx(i)=dy(i);
				dxx(m+1+i)=dlam(i);
				dxx(2*n+2*m+1+i)=dmu(i);
				dxx(2*n+3*m+2+i)=ds(i);
			}
			xx(m)=z;
			xx(2*n+3*m+1)=zet;

			dxx(m)=dz;
			dxx(2*n+3*m+1)=dzet;

			Vector stepxx(2*n+4*m+2);
			double stmxx,stmalfa,stmbeta,stmalbe,stmalbexx,stminv,steg;

			OboDivid(-1.01*dxx,xx,stepxx);
			VMax(stepxx,stmxx);
			VMax(OboDivid(-1.01*dx,(x-alfa),helpern),stmalfa);
			VMax(OboDivid(1.01*dx,(beta-x),helpern),stmbeta);
			stmalbe=max(stmalfa,stmbeta);
			stmalbexx=max(stmalbe,stmxx);
			stminv=max(stmalbexx,1.00);
			steg=1.00/stminv;

			Vector xold(n),yold(m),lamold(m),xsiold(n),etaold(n),muold(m),sold(m);
			double zold(z),zetold(zet);

			xold   =   x;
			yold   =   y;
			zold   =   z;
			lamold =  lam;
			xsiold =  xsi;
			etaold =  eta;
			muold  =  mu;
			zetold =  zet;
			sold   =   s;

			long int itto=0;
			double resinew=2.00*residunorm;
			//cout<<xx<<"=xx"<<endl;
			//cout<<dxx<<"=dxx"<<endl;

			while ((resinew>residunorm) && (itto<50))
			{
				counter++;
				itto++;
				x = xold+steg*dx;
				y = yold+steg*dy;
				z = zold+steg*dz;
				lam = lamold+steg*dlam;
				xsi = xsiold + steg*dxsi;
				eta = etaold + steg*deta;
				mu  = muold  + steg*dmu;
				zet = zetold + steg*dzet;
				s   =   sold + steg*ds;
				ux1=upp-x;
				xl1=x-low;
				OboProd(ux1,ux1,ux2);
				OboProd(xl1,xl1,xl2);
				OboDivid(een,ux1,uxinv1);
				OboDivid(een,xl1,xlinv1);
				plam=p0+axpy_prod(trans(P),lam,helpern);
				qlam=q0+axpy_prod(trans(Q),lam,helpern);
				gvec=axpy_prod(P,uxinv1,helperm)+axpy_prod(Q,xlinv1,helperm2);
				dpsidx=OboDivid(plam,ux2,helpern)-OboDivid(qlam,xl2,helpern2);

				/*cout<<steg<<" =steg"<<endl;
				cout<<dx<<" =dx"<<endl;
				cout<<dpsidx<<" =dpsidx"<<endl;
				cout<<gvec<<" =gvec"<<endl;
				cout<<plam<<" =plam"<<endl;
				cout<<qlam<<" =qlam"<<endl;
				cout<<x<<" =x"<<endl;
				cout<<xsi<<" =xsi"<<endl;
				cout<<ux1<<" =ux1"<<endl;
				cout<<ux2<<" =ux2"<<endl;*/

				rex = dpsidx-xsi+eta;
				rey = c + OboProd(d,y,helperm)- mu - lam;
				rez = a0 - zet - prec_inner_prod(a,lam);
				relam = gvec - a*z - y + s - b;
				rexsi = OboProd(xsi,(x-alfa),helpern) - epsvecn;
				reeta = OboProd(eta,(beta-x),helpern) - epsvecn;
				remu = OboProd(mu,y,helperm) - epsvecm;
				rezet = zet*z - epsi;
				res = OboProd(lam,s,helperm) - epsvecm;
				///////////////////////////////////////////////
				for (size_t i=0;i<n;i++)
				{
					residu(i)=rex(i);
					residu(i+n+2*m+1)=rexsi(i);
					residu(i+2*n+2*m+1)=reeta(i);
				}
				for (size_t i=0;i<m;i++)
				{
					residu(i+n)=rey(i);
					residu(i+n+m+1)=relam(i);
					residu(i+3*n+2*m+1)=remu(i);
					residu(i+3*n+3*m+2)=res(i);
				}
				residu(n+m)=rez;
				residu(3*m+3*n+1)=rezet;
				////////////////////////////////////////////////
				//resinew=norm_2(residu);
				resinew=sqrt(prec_inner_prod(residu,residu));
				//cout<<resinew<<" =resinew"<<endl;
				steg /= 2.00;

			}
			//cout<<residu<<"=reside"<<endl;
			residunorm=resinew;
			residumax=norm_inf(residu);
			/*cout<<residu<<"=residu"<<endl;
			cout<<residumax<<" =residumax2"<<endl;
			cout<<residunorm<<" =residunorm2"<<endl;*/
			steg *= 2.00;
		}
		epsi *= 0.10;
	}
	xold2 = xold1;
	xold1 = xval;
	xval = x;
	if (MyMPIRank == 0)
		cout<<counter<<" =Number of trys in subsolve"<<endl;
}


/////////////////////////////////////////////////////////////////////////////////////////////////////

class Optimization : public mma
{
	//Vector ro;
	//Vector Landa;
	Vector df0du;
	//Vector helperUNum;
	Vector helperm;

protected:
	bool FTEIC;
	size_t MDLenpart;
	size_t MDWidpart;
	size_t MDElemLenNum;
	size_t MDElemWidNum;

	//First time of each inner convergence
	//int OptTyp;

public:

	Optimization():Element(0,0,Elemlen,0,Elemlen,Elemwid,0,Elemwid)/*,slusolve(UNum)*/,inMDGeo("MDGeo.txt"),InSimpPowers("ChangedSimpPowers.txt")
	{
		inMDGeo>>SimpPower>>MDLenpart>>MDWidpart;

		if (n !=(MDLenpart+1.00)*(MDWidpart+1.00))
		{
			cout<<"number of variables in mma and MD are not equal"<<endl<<"If element based filtering technique is used decrease mdlenpart and mdwidpart equal to 1"<<endl;
			exit(1);
		}

		//ro.resize(4);
		Ro.resize(n);
		xvalDif.resize(n);
		//Landa.resize(UNum);
		df0du.resize(UNum);
		//helperUNum.resize(UNum);
		helperm.resize(m);
		for (size_t i=0;i<m;i++)
			helperm(i)=1.00;
		Ro=xval;

		MDElemLenNum=Lenpart/MDLenpart;
		MDElemWidNum=Widpart/MDWidpart;
		BuildIndex();
		xAllowDifRate=0.01;
		fAllowDifRate=0.001;
	}
	element Element;
	//SLUSolve slusolve;
	ifstream inMDGeo;
	ifstream InSimpPowers;				//If ChangeSIMPPower is defined

	Vector Ro;
	Vector xvalDif;
	SizVector InnerMapIndex;			//To use in inner optimization freezing solids (mapping)
	SizVector InnerMapIndexRev;			//To use in inner optimization freezing solids (reverse mapping)

	double f0valOld;
	double xvalMaxDifRate;
	double f0valDifRate;
	double xAllowDifRate;
	double fAllowDifRate;

	double SimpPower;

	double* roBuilder(size_t &elem ,double *ro,int condition,double *Massro);
	double* roBuilderIn(size_t &elem,double *ro,vector<double> *EIPI,double *Massro);
	void InSensitvtroBuilder(double* ro,size_t mprime,size_t nprime,size_t i,size_t DPN,vector<double> *EIPI,double *Massro);
	bool MakeStiffness(int OptTyp,vector<double> *EIPI);
	bool MakeStiffnessMT(int OptTyp,vector<double> *EIPI);
	void SensitvtAnalyzMD();
	void SensitvtAnalyz();
	void SensitvtAnalyzUTU();
	void InSensitvtAnalyz(vector<double> *EIPI);
	void ElFSensitvtAnalyz();
	bool X2NewX(size_t NStory,double *SHeights);
	void FollowTrace();
	void CutXTail();
	void CalResidue();												//A func. to calculate percentage of xval and f0val difference
	void FilterSenstvts(double& FRadius);
	void ElFilterSenstvts(double &FRadius);
	bool Optimize();
	bool InnerOptimize();
	bool FilterOptimize();
	bool ElFilterOptimize();

};

double* Optimization::roBuilder(size_t &elem ,double *ro,int condition=0 ,double *Massro=NULL)
{
	size_t i=elem%MDElemLenNum;
	size_t j=((elem%(Lenpart*MDElemWidNum))-(elem%(Lenpart*MDElemWidNum))%Lenpart)/Lenpart;
	size_t mprime=((elem-elem%MDElemLenNum)/MDElemLenNum)%MDLenpart;
	size_t nprime=(elem-elem%(Lenpart*MDElemWidNum))/(Lenpart*MDElemWidNum);

	double ro1,ro2,ro3,ro4;

	ro1=Ro(mprime+nprime*(MDLenpart+1));
	ro2=Ro(mprime+nprime*(MDLenpart+1)+1);
	ro3=Ro(mprime+(nprime+1)*(MDLenpart+1)+1);
	ro4=Ro(mprime+(nprime+1)*(MDLenpart+1));


	double ks,et;
	Vector N(4);
	for (size_t ii=0;ii<IntegPoints.size1();ii++)
	{
		ks=(2.00*i+IntegPoints(ii,0)-MDElemLenNum+1.00)/MDElemLenNum;
		et=(2.00*j+IntegPoints(ii,1)-MDElemWidNum+1.00)/MDElemWidNum;

		N(0)=1.00/4.00*(1.00-ks)*(1.00-et);
		N(1)=1.00/4.00*(1.00+ks)*(1.00-et);
		N(2)=1.00/4.00*(1.00+ks)*(1.00+et);
		N(3)=1.00/4.00*(1.00-ks)*(1.00+et);

		ro[ii]=(condition ? N(condition-1)*SimpPower : 1.00)*pow(N(0)*ro1+	 //to conclude both sensitivity and analyse mode
				N(1)*ro2 + N(2)*ro3 + N(3)*ro4 , SimpPower-(condition != 0));

		if (Massro != NULL)
		{
			Massro[ii]=/*(condition ? */MassPower*N(condition-1)*pow(N(0)*ro1+
						N(1)*ro2 + N(2)*ro3 + N(3)*ro4 , MassPower-1.00) /*: (N(0)*ro1 + N(1)*ro2 + N(2)*ro3 + N(3)*ro4))*/; //Because it is used just in dynnamic sensitivity analysis, is commented
#ifdef MassModificationDynOpt
			if ((N(0)*ro1 + N(1)*ro2 + N(2)*ro3 + N(3)*ro4) <= RoModifyVal)
				Massro[ii]=/*(condition ? */MassModifyP*N(condition-1)*pow((N(0)*ro1 + N(1)*ro2 + N(2)*ro3 + N(3)*ro4),MassModifyP-1.00)/* : pow(Massro[ii],6)*/;
#endif
		}
	}
	return ro;
}

double* Optimization::roBuilderIn(size_t &elem,double *ro,vector<double> *EIPI=0,double *Massro=NULL)
{
	/*/****************************************
	 **  Builds ro vector of an element by   **
	 **  interpolation of each integration 	 **
	 **point and store the data of it in EIPI**
	 *****************************************/

	size_t m1 , n1 , mprime , nprime;
	m1=elem%Lenpart;
	n1=(elem-m1)/Lenpart;

	double ks,et,N[4],ro1,ro2,ro3,ro4;

	for (size_t i=0;i<IntegPoints.size1();i++)
	{
		mprime=size_t(((IntegPoints(i,0)+1.00)/2.00+m1)*Elemlen/(Length/MDLenpart));
		nprime=size_t(((IntegPoints(i,1)+1.00)/2.00+n1)*Elemwid/(Width/MDWidpart));

		ro1=Ro(mprime+nprime*(MDLenpart+1));
		ro2=Ro(mprime+nprime*(MDLenpart+1)+1);
		ro3=Ro(mprime+(nprime+1)*(MDLenpart+1)+1);
		ro4=Ro(mprime+(nprime+1)*(MDLenpart+1));

		ks=fmod(((IntegPoints(i,0)+1.00)/2.00+m1)*Elemlen,(Length/MDLenpart))*2.00/(Length/MDLenpart)-1.00;
		et=fmod(((IntegPoints(i,1)+1.00)/2.00+n1)*Elemwid,(Width/MDWidpart))*2.00/(Width/MDWidpart)-1.00;
		if (FTEIC)
		{
			EIPI[mprime+nprime*MDLenpart].push_back(i);
			EIPI[mprime+nprime*MDLenpart].push_back(elem);
			EIPI[mprime+nprime*MDLenpart].push_back(ks);
			EIPI[mprime+nprime*MDLenpart].push_back(et);
		}
		N[1]=1.00/4.00*(1.00+ks)*(1.00-et);
		N[2]=1.00/4.00*(1.00+ks)*(1.00+et);
		N[3]=1.00/4.00*(1.00-ks)*(1.00+et);

		ro[i]=pow(N[0]*ro1+N[1]*ro2 + N[2]*ro3 + N[3]*ro4 , SimpPower);

		if (Massro != NULL)
		{
			Massro[i]=pow(N[0]*ro1 + N[1]*ro2 + N[2]*ro3 + N[3]*ro4,MassPower);
#ifdef MassModificationInnerDynOpt
			if ((N[0]*ro1 + N[1]*ro2 + N[2]*ro3 + N[3]*ro4) <= RoModifyVal)
				Massro[i] = pow(N[0]*ro1 + N[1]*ro2 + N[2]*ro3 + N[3]*ro4,MassModifyP);
#endif
		}
	}
	return ro;
}

void Optimization::InSensitvtroBuilder(double *ro,size_t mprime,size_t nprime,size_t i,size_t DPN,vector<double> *EIPI,double *Massro=NULL)
{
	double ks,et,N[4],ro1,ro2,ro3,ro4;
	ro[0]=ro[1]=ro[2]=ro[3]=0.00;

	ro1=Ro(mprime+nprime*(MDLenpart+1));
	ro2=Ro(mprime+nprime*(MDLenpart+1)+1);
	ro3=Ro(mprime+(nprime+1)*(MDLenpart+1)+1);
	ro4=Ro(mprime+(nprime+1)*(MDLenpart+1));

	ks=EIPI[mprime+nprime*MDLenpart][2+i*4];
	et=EIPI[mprime+nprime*MDLenpart][3+i*4];

	N[0]=1.00/4.00*(1.00-ks)*(1.00-et);
	N[1]=1.00/4.00*(1.00+ks)*(1.00-et);
	N[2]=1.00/4.00*(1.00+ks)*(1.00+et);
	N[3]=1.00/4.00*(1.00-ks)*(1.00+et);

	ro[size_t(EIPI[mprime+nprime*MDLenpart][i*4])]=(SimpPower*N[DPN-1])*pow(N[0]*ro1+N[1]*ro2 + N[2]*ro3 + N[3]*ro4 , SimpPower-1.00);

	if (Massro != NULL)
	{
		Massro[0]=Massro[1]=Massro[2]=Massro[3]=0.00;
		Massro[size_t(EIPI[mprime+nprime*MDLenpart][i*4])]=MassPower*N[DPN-1]*pow(N[0]*ro1 + N[1]*ro2 + N[2]*ro3 + N[3]*ro4,MassPower-1.00);
#ifdef MassModificationInnerDynOpt
		if ((N[0]*ro1+N[1]*ro2 + N[2]*ro3 + N[3]*ro4) <= RoModifyVal)
			Massro[size_t(EIPI[mprime+nprime*MDLenpart][i*4])]=MassModifyP*N[DPN-1]*pow((N[0]*ro1+N[1]*ro2 + N[2]*ro3 + N[3]*ro4) , MassModifyP-1.00);
#endif
	}
}

bool Optimization::MakeStiffnessMT(int OptTyp=1,vector<double> *EIPI=0)
{
	cout<<"Entering stiffness maker"<<endl;
	KuuS.clear();
	Fu.clear();
	//Kb.clear();
	boost::timer t;
	CSMatrix KuuS2(UNum,UNum);
#ifdef OptimizationisDynamic
	CMatrix KuuT2(UNum,UNum);
#endif
	Vector Fu2(UNum),Fu0(UNum);
	Fu2.clear();
	Fu0.clear();
	size_t mthl=(ElemsNum-fmod(ElemsNum,2.00))/2;

#pragma omp parallel sections num_threads(2)
	{
#pragma omp section
		for (size_t elem=0;elem<mthl;elem++)
		{
			//printf("number0 %d",elem);
			size_t ii(0),jj(0);
			Matrix kelem(8,8);
			double ro[IntegPoints.size1()];

			if (OptTyp == 1)
				roBuilder(elem,ro);
			else if (OptTyp == 0)
				roBuilderIn(elem,ro,EIPI);
			else if (OptTyp == 2)
				for (unsigned int i=0;i<IntegPoints.size1();i++)
					ro[i]=pow(Ro(elem),SimpPower);

			//cout<<ro<<" =ro"<<endl;
			Element.kBuild(ro,kelem);
			//cout<<Element.k<<"  =k"<<endl;
			//cout<<Index<<" =Index"<<endl;
			for (size_t i=0;i<8;i++)
				for (size_t j=0;j<8;j++)
				{
					//cout<<i<<" , "<<j<<endl;
					ii = Connectivity(elem,i);
					jj = Connectivity(elem,j);
					if ( Loads(ii) && Index(ii)<UNum && (!Fu0(Index(ii))))
						//#pragma omp critical
						Fu0(Index(ii))=Loads(ii);
					/*if ( Loads(2*ii+1) && Index(2*ii+1)<UNum && (!Fu(Index(2*ii+1))))
						Fu(Index(2*ii+1))=Loads(2*ii+1);
					*/
					////////////////////////////////////////////////////


					if (Index(ii) < UNum && Index(jj) < UNum && Index(ii) <= Index(jj))
					{
						//#pragma omp critical
						/*size_t iii = Index(ii), jjj = Index(jj);
						#pragma omp critical
						{
							double V = KuuS(iii, jjj);
							V += kelem(i,j);
							KuuS(iii, jjj) = V;
						}*/
#ifndef OptimizationisDynamic
						/*KuuS(Index(jj),Index(ii)) =*/
						KuuS(Index(ii),Index(jj)) += kelem(i,j);
#else
						KuuT(Index(jj),Index(ii))=KuuT(Index(ii),Index(jj))=KuuS(Index(ii),Index(jj)) += kelem(i,j);
#endif
					}

					else if (Index(ii) < UNum && Index(jj) >= UNum)
						//#pragma omp critical
						Fu(Index(ii)) -= kelem(i,j)*Disps(jj) ;

					//else

					//	Kb(Index(ii)-UNum,Index(jj)) += Element.k(i,j);



					/*if(Index(2*ii+1) < UNum && Index(2*jj) < UNum)

						Kuu(Index(2*ii+1),Index(2*jj)) 	 += Element.k(2*i+1,2*j);

					else if (Index(2*ii+1) < UNum && Index(2*jj) >= UNum)

						Fu(Index(2*ii+1)) -= Element.k(2*i+1,2*j)*Disps(2*jj) ;

					//else

						//Kb(Index(2*ii+1)-UNum,Index(2*jj)) += Element.k(2*i+1,2*j);



					if(Index(2*ii) < UNum && Index(2*jj+1) < UNum)

						Kuu(Index(2*ii),Index(2*jj+1)) 	 += Element.k(2*i,2*j+1);

					else if (Index(2*ii) < UNum && Index(2*jj+1) >= UNum)

						Fu(Index(2*ii)) -= Element.k(2*i,2*j+1)*Disps(2*jj+1) ;

					//else

						//Kb(Index(2*ii)-UNum,Index(2*jj+1)) += Element.k(2*i,2*j+1);



					if(Index(2*ii+1) < UNum && Index(2*jj+1) < UNum)

						Kuu(Index(2*ii+1),Index(2*jj+1)) 	 += Element.k(2*i+1,2*j+1);

					else if (Index(2*ii+1) < UNum && Index(2*jj+1) >= UNum)

						Fu(Index(2*ii+1)) -= Element.k(2*i+1,2*j+1)*Disps(2*jj+1) ;

					//else

						//Kb(Index(2*ii+1)-UNum,Index(2*jj+1)) += Element.k(2*i+1,2*j+1);
					*/
					/////////////////////////////////////////////////////////////////////////////////////////

					/*Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j))		 += TestElem.k(2*i,2*j);
					Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j))	   += TestElem.k(2*i+1,2*j);
					Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j)+1)	   += TestElem.k(2*i,2*j+1);

				}
			//cout<<"Kuu : elem="<<elem<<endl;
		}
#pragma omp section
		for (size_t elem=mthl;elem<ElemsNum;elem++)
		{
			//printf("number1 %d",elem);
			size_t ii(0),jj(0);
			Matrix kelem(8,8);
			double ro[IntegPoints.size1()];

			if (OptTyp == 1)
				roBuilder(elem,ro);
			else if (OptTyp == 0)
				roBuilderIn(elem,ro,EIPI);
			else if (OptTyp == 2)
				for (unsigned int i=0;i<IntegPoints.size1();i++)
					ro[i]=pow(Ro(elem),SimpPower);

			//cout<<ro<<" =ro"<<endl;
			Element.kBuild(ro,kelem);
			//cout<<Element.k<<"  =k"<<endl;
			//cout<<Index<<" =Index"<<endl;
			for (size_t i=0;i<8;i++)
				for (size_t j=0;j<8;j++)
				{
					//cout<<i<<" , "<<j<<endl;
					ii = Connectivity(elem,i);
					jj = Connectivity(elem,j);
					if ( Loads(ii) && Index(ii)<UNum && (!Fu0(Index(ii))))

						Fu0(Index(ii))=Loads(ii);
					/*if ( Loads(2*ii+1) && Index(2*ii+1)<UNum && (!Fu(Index(2*ii+1))))
						Fu(Index(2*ii+1))=Loads(2*ii+1);
					*/
					////////////////////////////////////////////////////


					if (Index(ii) < UNum && Index(jj) < UNum && Index(ii) <= Index(jj))
					{

						/*size_t iii = Index(ii), jjj = Index(jj);
						#pragma omp critical
						{
							double V = KuuS(iii, jjj);
							V += kelem(i,j);
							KuuS(iii, jjj) = V;
						}*/
#ifndef OptimizationisDynamic
						/*KuuS(Index(jj),Index(ii)) =*/
						KuuS2(Index(ii),Index(jj)) += kelem(i,j);
#else
						KuuT2(Index(jj),Index(ii))=KuuT2(Index(ii),Index(jj))=KuuS2(Index(ii),Index(jj)) += kelem(i,j);
#endif
					}

					else if (Index(ii) < UNum && Index(jj) >= UNum)

						Fu2(Index(ii)) -= kelem(i,j)*Disps(jj) ;
				}
		}
	}
	KuuS += KuuS2;
#ifdef OptimizationisDynamic
	KuuT+=KuuT2;
#endif
	Fu += (Fu2 + Fu0);
	cout<<"stiffness is built, "<< t.elapsed() << endl;
	return true;

}

bool Optimization::MakeStiffness(int OptTyp=1,vector<double> *EIPI=0)
{
	if (MyMPIRank == 0)
		cout<<"Entering stiffness maker"<<endl;
	KuuS.clear();
	//#ifdef OptimizationisDynamic
	//KuuT.clear();
	//#endif
	Fu.clear();
	//Kb.clear();

	boost::timer t;
//#pragma omp parallel for schedule(dynamic)
	for (size_t elem=0;elem<ElemsNum;elem++)
	{
		double ro[IntegPoints.size1()];
		Matrix kelem(8,8);
		size_t ii(0),jj(0);
		if (OptTyp == 1)
			roBuilder(elem,ro);
		else if (OptTyp == 0)
			roBuilderIn(elem,ro,EIPI);
		else if (OptTyp == 2)
			for (unsigned int i=0;i<IntegPoints.size1();i++)
				ro[i]=pow(Ro(elem),SimpPower);

		//cout<<ro<<" =ro"<<endl;
		Element.kBuild(ro,kelem);
		//cout<<Element.k<<"  =k"<<endl;
		//cout<<Index<<" =Index"<<endl;
		for (size_t i=0;i<8;i++)
			for (size_t j=0;j<8;j++)
			{
				//cout<<i<<" , "<<j<<endl;
				ii = Connectivity(elem,i);
				jj = Connectivity(elem,j);
				if ( Loads(ii) && Index(ii)<UNum && (!Fu(Index(ii))))
					//#pragma omp critical
					Fu(Index(ii))=Loads(ii);
				/*if ( Loads(2*ii+1) && Index(2*ii+1)<UNum && (!Fu(Index(2*ii+1))))

					Fu(Index(2*ii+1))=Loads(2*ii+1);
				*/
				////////////////////////////////////////////////////

				if (Index(ii) < UNum && Index(jj) < UNum && Index(ii)<=Index(jj))
				{
					//#pragma omp critical
#ifndef OptimizationisDynamic
					/*KuuS(Index(jj),Index(ii)) = */
					KuuS(Index(ii),Index(jj)) += kelem(i,j);
#else
					KuuS(Index(ii),Index(jj)) += kelem(i,j);
					KuuT(Index(jj),Index(ii))=KuuT(Index(ii),Index(jj))=KuuS(Index(ii),Index(jj));
#endif
				}

				else if (Index(ii) < UNum && Index(jj) >= UNum)
					//#pragma omp critical
					Fu(Index(ii)) -= kelem(i,j)*Disps(jj) ;

				//else

				//	Kb(Index(ii)-UNum,Index(jj)) += Element.k(i,j);



				/*if(Index(2*ii+1) < UNum && Index(2*jj) < UNum)

					Kuu(Index(2*ii+1),Index(2*jj)) 	 += Element.k(2*i+1,2*j);

				else if (Index(2*ii+1) < UNum && Index(2*jj) >= UNum)

					Fu(Index(2*ii+1)) -= Element.k(2*i+1,2*j)*Disps(2*jj) ;

				//else

					//Kb(Index(2*ii+1)-UNum,Index(2*jj)) += Element.k(2*i+1,2*j);



				if(Index(2*ii) < UNum && Index(2*jj+1) < UNum)

					Kuu(Index(2*ii),Index(2*jj+1)) 	 += Element.k(2*i,2*j+1);

				else if (Index(2*ii) < UNum && Index(2*jj+1) >= UNum)

					Fu(Index(2*ii)) -= Element.k(2*i,2*j+1)*Disps(2*jj+1) ;

				//else

					//Kb(Index(2*ii)-UNum,Index(2*jj+1)) += Element.k(2*i,2*j+1);

				else if (Index(2*ii+1) < UNum && Index(2*jj+1) >= UNum)

					Fu(Index(2*ii+1)) -= Element.k(2*i+1,2*j+1)*Disps(2*jj+1) ;

				//else

					//Kb(Index(2*ii+1)-UNum,Index(2*jj+1)) += Element.k(2*i+1,2*j+1);
				*/
				/////////////////////////////////////////////////////////////////////////////////////////

				/*Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j))		 += TestElem.k(2*i,2*j);
				Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j))	   += TestElem.k(2*i+1,2*j);
				Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j)+1)	   += TestElem.k(2*i,2*j+1);
				Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j)+1)	 += TestElem.k(2*i+1,2*j+1);*/

			}
		//cout<<"Kuu : elem="<<elem<<endl;
	}
	if (MyMPIRank == 0)
		cout<<"stiffness is built , "<<t.elapsed()<<endl;
	return true;

}

void Optimization::SensitvtAnalyzMD()
{
	cout<<"Entering Sensitivity Analysis"<<endl;
	Vector Landa(UNum);

	if (iter==0)										// needs to be modified if loads number grows too much
	{
		for (CVector::iterator iterat=Loads.begin();iterat != Loads.end();iterat++)
			if (Index(iterat.index())<UNum)
				df0du(Index(iterat.index())) = *iterat;
		//cout<<df0du<<" =df0du"<<endl;
	}
	double ro[IntegPoints.size1()];
	Matrix kelem(8,8);

	TAUCS_Solve(KuuS,df0du,Landa,false);
	Landa *= -1.00;
	for (size_t xiter=0;xiter<n;xiter++)
	{
		int lhs,rhs,uhs,bhs;
		size_t iro=xiter%(MDLenpart+1);
		size_t jro=(xiter-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		KuuS.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		for (size_t nprime=bhs+jro-1;nprime<2-uhs+jro-1;nprime++)
			for (size_t mprime=lhs+iro-1;mprime<2-rhs+iro-1;mprime++)
				for (size_t elem=0;elem<MDElemLenNum*MDElemWidNum;elem++)
				{
					size_t ielem,jelem;
					ielem=elem%MDElemLenNum;
					jelem=(elem-ielem)/MDElemLenNum;
					size_t elemnum=(nprime*MDElemWidNum*Lenpart+mprime*MDElemLenNum)+Lenpart*jelem+ielem;

					size_t ii,jj;
					roBuilder(elemnum,ro,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1)); //to show the function which edge ro is being derived in a prime mesh desgine
					Element.kBuild(ro,kelem);
					//cout<<Element.k<<endl;
					for (size_t i=0;i<8;i++)
						for (size_t j=0;j<8;j++)
						{

							ii = Connectivity(elemnum,i);
							jj = Connectivity(elemnum,j);

							if (Index(ii) < UNum && Index(jj) < UNum)

								KuuS(Index(ii),Index(jj)) 	 += kelem(i,j);

							/*if(Index(2*ii+1) < UNum && Index(2*jj) < UNum)

								Kuu(Index(2*ii+1),Index(2*jj)) 	 += Element.k(2*i+1,2*j);

							if(Index(2*ii) < UNum && Index(2*jj+1) < UNum)

								Kuu(Index(2*ii),Index(2*jj+1)) 	 += Element.k(2*i,2*j+1);

							if(Index(2*ii+1) < UNum && Index(2*jj+1) < UNum)

								Kuu(Index(2*ii+1),Index(2*jj+1)) += Element.k(2*i+1,2*j+1);*/

							/////////////////////////////////////////////////////////////////////////////////////////

							/*Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j))		 += TestElem.k(2*i,2*j);
							Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j))	   += TestElem.k(2*i+1,2*j);
							Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j)+1)	   += TestElem.k(2*i,2*j+1);
							Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j)+1)	 += TestElem.k(2*i+1,2*j+1);*/
						}
				}
		Vector helperUNum(UNum);
		df0dx(xiter)=prec_inner_prod(axpy_prod(KuuS,Du,helperUNum),Landa);
		if (iter==0)																			//dfdx is equal for this purpose
			dfdx(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs); 		//for edge ro modifications
	}
	cout<<"Sensitivity analysis is done"<<endl;

}

void Optimization::SensitvtAnalyz()
{
	cout<<"Entering Sensitivity Analysis"<<endl;
	Vector Landa(UNum);

	if (iter==0)										// needs to be modified if loads number grows too much
	{
		for (CVector::iterator iterat=Loads.begin();iterat != Loads.end();iterat++)
			if (Index(iterat.index())<UNum)
				df0du(Index(iterat.index())) = *iterat;
		//cout<<df0du<<" =df0du"<<endl;
	}
	double ro[IntegPoints.size1()];

	//slusolve.Solve(2,KuuS,Fu,Du,df0du,Landa);

	//TAUCS_Solve(KuuS,Fu,Du);
	//Landa =-1.00*Du;								//if all constraints displacement values be 0 Fu = df0du so Du = Landa;

	//TAUCS_Solve(KuuS,df0du,Landa,false);

	TAUCS_Solve2(KuuS,Fu,Du,df0du,Landa);
	f0val=prec_inner_prod(Disps,Loads);
	Landa *= -1.00;

	//cout<<norm_2(Fu-df0du)<<" =residual"<<endl;
	//f0val=prec_inner_prod(Disps,Loads);

#pragma omp parallel for private(ro) num_threads(2) schedule(guided)
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		CMatrix Kuu(UNum,UNum);
		int lhs,rhs,uhs,bhs;
		size_t iro=xiter%(MDLenpart+1);
		size_t jro=(xiter-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		size_t mLeft=(lhs+iro-1)*MDElemLenNum;
		size_t mRight=(2-rhs+iro-1)*MDElemLenNum;
		size_t nBottom=(bhs+jro-1)*MDElemWidNum;
		size_t nUp=(2-uhs+jro-1)*MDElemWidNum;

		for (size_t jelem=nBottom;jelem<nUp;jelem++)
			for (size_t ielem=mLeft;ielem<mRight;ielem++)
			{
				size_t elemnum = jelem*Lenpart+ielem;
				size_t ii,jj;
				size_t mprime=(ielem-ielem%MDElemLenNum)/MDElemLenNum;
				size_t nprime=(jelem-jelem%MDElemWidNum)/MDElemWidNum;
				roBuilder(elemnum,ro,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1)); //to show the function which edge ro is being derived in a prime mesh desgine
				Element.kBuild(ro,kelem);

				for (size_t i=0;i<8;i++)
					for (size_t j=0;j<8;j++)
					{
						ii = Connectivity(elemnum,i);
						jj = Connectivity(elemnum,j);

						if (Index(ii) < UNum && Index(jj) < UNum)

							Kuu(Index(ii),Index(jj)) += kelem(i,j);

						/*if(Index(2*ii+1) < UNum && Index(2*jj) < UNum)


						if(Index(2*ii) < UNum && Index(2*jj+1) < UNum)

							Kuu(Index(2*ii),Index(2*jj+1)) 	 += Element.k(2*i,2*j+1);

						if(Index(2*ii+1) < UNum && Index(2*jj+1) < UNum)

							Kuu(Index(2*ii+1),Index(2*jj+1)) += Element.k(2*i+1,2*j+1);*/

						/////////////////////////////////////////////////////////////////////////////////////////

						/*Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j))		+= TestElem.k(2*i,2*j);
						Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j))		+= TestElem.k(2*i+1,2*j);
						Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j)+1)		+= TestElem.k(2*i,2*j+1);
						Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j)+1)	+= TestElem.k(2*i+1,2*j+1);*/
					}
			}
		Vector helperUNum(UNum);
		df0dx(xiter)=prec_inner_prod(axpy_prod(Kuu,Du,helperUNum),Landa);
		if (iter==0)																			//dfdx is equal for this purpose
			dfdx(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs); 		//for edge ro modifications
	}
	cout<<"Sensitivity analysis is done"<<endl;

}

void Optimization::SensitvtAnalyzUTU()
{
	cout<<"Entering Sensitivity Analysis"<<endl;
	Vector Landa(UNum);

	/*if (iter==0)										// needs to be modified if loads number grows too much
	{
		for (CVector::iterator iterat=Loads.begin();iterat != Loads.end();iterat++)
			if (Index(iterat.index())<UNum)
				df0du(Index(iterat.index())) = *iterat;
		//cout<<df0du<<" =df0du"<<endl;
	}*/
	double ro[IntegPoints.size1()];

	//slusolve.Solve(2,KuuS,Fu,Du,df0du,Landa);

	TAUCS_Solve(KuuS,Fu,Du);
	//Landa =-1.00*Du;								//if all constraints displacement values be 0 Fu = df0du so Du = Landa;
	f0val=prec_inner_prod(Disps,Disps);
	TAUCS_Solve(KuuS,Du,Landa,false);

	//TAUCS_Solve2(KuuS,Fu,Du,df0du,Landa);
	Landa *= -2.00;

	//cout<<norm_2(Fu-df0du)<<" =residual"<<endl;
	//f0val=prec_inner_prod(Disps,Loads);

#pragma omp parallel for private(ro) num_threads(2) schedule(guided)
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		CMatrix Kuu(UNum,UNum);
		int lhs,rhs,uhs,bhs;
		size_t iro=xiter%(MDLenpart+1);
		size_t jro=(xiter-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		size_t mLeft=(lhs+iro-1)*MDElemLenNum;
		size_t mRight=(2-rhs+iro-1)*MDElemLenNum;
		size_t nBottom=(bhs+jro-1)*MDElemWidNum;
		size_t nUp=(2-uhs+jro-1)*MDElemWidNum;

		for (size_t jelem=nBottom;jelem<nUp;jelem++)
			for (size_t ielem=mLeft;ielem<mRight;ielem++)
			{
				size_t elemnum = jelem*Lenpart+ielem;
				size_t ii,jj;
				size_t mprime=(ielem-ielem%MDElemLenNum)/MDElemLenNum;
				size_t nprime=(jelem-jelem%MDElemWidNum)/MDElemWidNum;
				roBuilder(elemnum,ro,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1)); //to show the function which edge ro is being derived in a prime mesh desgine
				Element.kBuild(ro,kelem);

				for (size_t i=0;i<8;i++)
					for (size_t j=0;j<8;j++)
					{
						ii = Connectivity(elemnum,i);
						jj = Connectivity(elemnum,j);

						if (Index(ii) < UNum && Index(jj) < UNum)

							Kuu(Index(ii),Index(jj)) += kelem(i,j);

						/*if(Index(2*ii+1) < UNum && Index(2*jj) < UNum)

							Kuu(Index(2*ii+1),Index(2*jj)) 	 += Element.k(2*i+1,2*j);

						if(Index(2*ii) < UNum && Index(2*jj+1) < UNum)

							Kuu(Index(2*ii),Index(2*jj+1)) 	 += Element.k(2*i,2*j+1);

						if(Index(2*ii+1) < UNum && Index(2*jj+1) < UNum)

							Kuu(Index(2*ii+1),Index(2*jj+1)) += Element.k(2*i+1,2*j+1);*/

						/////////////////////////////////////////////////////////////////////////////////////////

						/*Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j))		+= TestElem.k(2*i,2*j);
						Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j))		+= TestElem.k(2*i+1,2*j);
						Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j)+1)		+= TestElem.k(2*i,2*j+1);
						Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j)+1)	+= TestElem.k(2*i+1,2*j+1);*/
					}
			}
		Vector helperUNum(UNum);
		df0dx(xiter)=prec_inner_prod(axpy_prod(Kuu,Du,helperUNum),Landa);
		if (iter==0)																			//dfdx is equal for this purpose
			dfdx(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs); 		//for edge ro modifications
	}
	cout<<"Sensitivity analysis is done"<<endl;

}

void Optimization::InSensitvtAnalyz(vector<double> *EIPI)
{
	cout<<"Entering Inner Sensitivity Analysis"<<endl;
	Vector Landa(UNum);

	if (iter==0)										// needs to be modified if loads number grows too much
	{
		for (CVector::iterator iterat=Loads.begin();iterat != Loads.end();iterat++)
			if (Index(iterat.index())<UNum)
				df0du(Index(iterat.index())) = *iterat;
		//cout<<df0du<<" =df0du"<<endl;
	}
	double ro[IntegPoints.size1()];

	//TAUCS_Solve(KuuS,Fu,Du);
	//Landa =-1.00*Du;								//if all constraints displacement values be 0 Fu = df0du so Du = Landa;

	//TAUCS_Solve(KuuS,df0du,Landa,false);

	TAUCS_Solve2(KuuS,Fu,Du,df0du,Landa);
	Landa *= -1.00;

	f0val=prec_inner_prod(Disps,Loads);

#pragma omp parallel for private(ro) num_threads(2) schedule(guided)
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		CMatrix Kuu(UNum,UNum);

		int lhs,rhs,uhs,bhs;
		size_t iro=InnerMapIndexRev(xiter)%(MDLenpart+1);
		size_t jro=(InnerMapIndexRev(xiter)-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		for (size_t nprime=bhs+jro-1;nprime<2-uhs+jro-1;nprime++)
			for (size_t mprime=lhs+iro-1;mprime<2-rhs+iro-1;mprime++)
				for (size_t iii=0;iii<EIPI[mprime+nprime*MDLenpart].size()/4;iii++)
				{
					size_t ii,jj;
					InSensitvtroBuilder(ro,mprime,nprime,iii,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1),EIPI); //5th argument shows the number of corner ro to be driven
					Element.kBuild(ro,kelem);
					//cout<<Element.k<<endl;
					size_t elemnum=size_t(EIPI[mprime+nprime*MDLenpart][4*iii+1]);

					for (size_t i=0;i<8;i++)
						for (size_t j=0;j<8;j++)
						{

							ii = Connectivity(elemnum,i);
							jj = Connectivity(elemnum,j);

							if (Index(ii) < UNum && Index(jj) < UNum)

								Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);

							/*if(Index(2*ii+1) < UNum && Index(2*jj) < UNum)

								Kuu(Index(2*ii+1),Index(2*jj)) 	 += Element.k(2*i+1,2*j);

							if(Index(2*ii) < UNum && Index(2*jj+1) < UNum)

								Kuu(Index(2*ii),Index(2*jj+1)) 	 += Element.k(2*i,2*j+1);

							if(Index(2*ii+1) < UNum && Index(2*jj+1) < UNum)

								Kuu(Index(2*ii+1),Index(2*jj+1)) += Element.k(2*i+1,2*j+1);*/

							/////////////////////////////////////////////////////////////////////////////////////////

							/*Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j))		 += TestElem.k(2*i,2*j);
							Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j))	   += TestElem.k(2*i+1,2*j);
							Stiffness(2*Connectivity(elem,i),2*Connectivity(elem,j)+1)	   += TestElem.k(2*i,2*j+1);
							Stiffness(2*Connectivity(elem,i)+1,2*Connectivity(elem,j)+1)	 += TestElem.k(2*i+1,2*j+1);*/
						}
				}
		Vector helperUNum(UNum);
		df0dx(xiter)=prec_inner_prod(axpy_prod(Kuu,Du,helperUNum),Landa);
		if (iter==0)																			//dfdx is equal for this purpose
	}

	cout<<"Inner Sensitivity analysis is done"<<endl;

}

void Optimization::ElFSensitvtAnalyz()
{
	cout<<"Entering filtering sensitivity analysis"<<endl;
	Vector Landa(UNum);

	if (iter==0)										// needs to be modified if loads number grows too much
	{
		for (CVector::iterator iterat=Loads.begin();iterat != Loads.end();iterat++)
			if (Index(iterat.index())<UNum)
				df0du(Index(iterat.index())) = *iterat;
		//cout<<df0du<<" =df0du"<<endl;
	}
	double ro[IntegPoints.size1()];


	TAUCS_Solve(KuuS,df0du,Landa,false);
	Landa *= -1.00;
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		CMatrix Kuu(UNum,UNum);
		Kuu.clear();
		for (unsigned int roi=0;roi<IntegPoints.size1();roi++)
			ro[roi]=SimpPower*pow(Ro(xiter),SimpPower-1.00);
		Element.kBuild(ro,kelem);
		size_t ii,jj;
		//cout<<Element.k<<endl;
		for (size_t i=0;i<8;i++)
			for (size_t j=0;j<8;j++)
			{

				ii = Connectivity(xiter,i);
				jj = Connectivity(xiter,j);

				if (Index(ii) < UNum && Index(jj) < UNum)

					Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);

			}
		Vector helperUNum(UNum);
		df0dx(xiter)=prec_inner_prod(axpy_prod(Kuu,Du,helperUNum),Landa);
		if (iter==0)																			//dfdx is equal for this purpose
			dfdx(0,xiter)=Elemlen*Elemwid; 		//for edge ro modifications
		//cout<<xiter<<endl;
	}
	cout<<"Filtering Sensitivity analysis is done"<<endl;

}

bool Optimization::X2NewX(size_t NStory=0 ,double *SHeights=NULL)						//This function converts every thing to new M.D.
{
	size_t MDLenpartNew,MDWidpartNew,MDElemLenNumNew,MDElemWidNumNew;
	double MDElemLength , MDElemWidth , MDElemLengthNew , MDElemWidthNew;

	if (!(inMDGeo>>MDLenpartNew>>MDWidpartNew))
		return false;

	#ifdef InterpoleXolds
		Vector xold1_(n);
		Vector xold2_(n);
		Vector low_(n);
		Vector upp_(n);
		xold1_=xold1;
		xold2_=xold2;
		low_=low;
		upp_=upp;
		xold1.resize((MDLenpartNew+1)*(MDWidpartNew+1));
		xold2.resize((MDLenpartNew+1)*(MDWidpartNew+1));
		low.resize((MDLenpartNew+1)*(MDWidpartNew+1));
		upp.resize((MDLenpartNew+1)*(MDWidpartNew+1));
	#endif

	MDElemLength=Length/MDLenpart;
	MDElemWidth=Width/MDWidpart;
	MDElemLengthNew=Length/MDLenpartNew;
	MDElemWidthNew=Width/MDWidpartNew;

	MDElemLenNumNew=Lenpart/MDLenpartNew;
	MDElemWidNumNew=Widpart/MDWidpartNew;

	n =(MDLenpartNew+1)*(MDWidpartNew+1);
////////////////////////////////////////////////////////////////////////////////////
	xval.resize(n);								// Ro was equal to xval now from Ro, xval will be interpolated
	double ks , et, I , II;
	size_t mprime , nprime ;

	for (size_t xiter=0;xiter<n;xiter++)
	{
		size_t iro=xiter%(MDLenpartNew+1);
		size_t jro=(xiter-iro)/(MDLenpartNew+1);

		I=fmod((iro*MDElemLengthNew),MDElemLength);
		II=fmod((jro*MDElemWidthNew),MDElemWidth);

		ks=(I*2.00)/MDElemLength-1.00;
		et=(II*2.00)/MDElemWidth-1.00;

		mprime=size_t((iro*MDElemLengthNew-I)/MDElemLength);
		nprime=size_t((jro*MDElemWidthNew-II)/MDElemWidth);

		if (mprime > MDLenpart-1)
		{
			mprime -= 1;
			ks = 1.00;
		}
		if (nprime > MDWidpart-1)
		{
			nprime -= 1;
			et = 1.00;
		}

		double ro1=Ro(mprime+nprime*(MDLenpart+1));
		double ro2=Ro(mprime+nprime*(MDLenpart+1)+1);
		double ro3=Ro(mprime+(nprime+1)*(MDLenpart+1)+1);
		double ro4=Ro(mprime+(nprime+1)*(MDLenpart+1));

		xval(xiter)=1.00/4.00*(1.00-ks)*(1.00-et)*ro1+1.00/4.00*(1.00+ks)*(1.00-et)*ro2+
					1.00/4.00*(1.00+ks)*(1.00+et)*ro3+1.00/4.00*(1.00-ks)*(1.00+et)*ro4;

		#ifdef InterpoleXolds
			double roold1_2=xold1_(mprime+nprime*(MDLenpart+1)+1);
			double roold1_3=xold1_(mprime+(nprime+1)*(MDLenpart+1)+1);
			double roold1_4=xold1_(mprime+(nprime+1)*(MDLenpart+1));
			xold1(xiter)=1.00/4.00*(1.00-ks)*(1.00-et)*roold1_1+1.00/4.00*(1.00+ks)*(1.00-et)*roold1_2+
					1.00/4.00*(1.00+ks)*(1.00+et)*roold1_3+1.00/4.00*(1.00-ks)*(1.00+et)*roold1_4;

			double roold2_1=xold2_(mprime+nprime*(MDLenpart+1));
			double roold2_2=xold2_(mprime+nprime*(MDLenpart+1)+1);
			double roold2_3=xold2_(mprime+(nprime+1)*(MDLenpart+1)+1);
			double roold2_4=xold2_(mprime+(nprime+1)*(MDLenpart+1));
			xold2(xiter)=1.00/4.00*(1.00-ks)*(1.00-et)*roold2_1+1.00/4.00*(1.00+ks)*(1.00-et)*roold2_2+
					1.00/4.00*(1.00+ks)*(1.00+et)*roold2_3+1.00/4.00*(1.00-ks)*(1.00+et)*roold2_4;
		// here used again as are not more needed

			roold1_1=upp_(mprime+nprime*(MDLenpart+1));
			roold1_2=upp_(mprime+nprime*(MDLenpart+1)+1);
			roold1_3=upp_(mprime+(nprime+1)*(MDLenpart+1)+1);
			roold1_4=upp_(mprime+(nprime+1)*(MDLenpart+1));
			upp(xiter)=1.00/4.00*(1.00-ks)*(1.00-et)*roold1_1+1.00/4.00*(1.00+ks)*(1.00-et)*roold1_2+
					1.00/4.00*(1.00+ks)*(1.00+et)*roold1_3+1.00/4.00*(1.00-ks)*(1.00+et)*roold1_4;

			roold2_1=low_(mprime+nprime*(MDLenpart+1));
			roold2_2=low_(mprime+nprime*(MDLenpart+1)+1);
			roold2_3=low_(mprime+(nprime+1)*(MDLenpart+1)+1);
			roold2_4=low_(mprime+(nprime+1)*(MDLenpart+1));
			low(xiter)=1.00/4.00*(1.00-ks)*(1.00-et)*roold2_1+1.00/4.00*(1.00+ks)*(1.00-et)*roold2_2+
					1.00/4.00*(1.00+ks)*(1.00+et)*roold2_3+1.00/4.00*(1.00-ks)*(1.00+et)*roold2_4;
		#endif
	}

////////////////////////////////////////////////////////////////////////////////////
	MDLenpart=MDLenpartNew;
	MDWidpart=MDWidpartNew;
	MDElemLenNum=MDElemLenNumNew;
	MDElemWidNum=MDElemWidNumNew;
	#ifdef x2newxRoPrint
		ofstream OutXInterpoled("X2newxRoInterpoled.txt");
		OutXInterpoled<<"Mdgeo: "<<MDLenpart<<" , "<<MDWidpart<<endl<<xval<<endl;
	#endif

	#ifndef InterpoleXolds
		xold1.resize(n);
		xold2.resize(n);

		low.resize(n);
		upp.resize(n);
	#endif

	iter=0;
	Ro.resize(n);

	xmin.resize(n);
	xmax.resize(n);
	alfa.resize(n);
	beta.resize(n);
	p0.resize(n);
	q0.resize(n);

	df0dx.resize(n);
	dfdx.resize(m,n);
	df0dx2.resize(n,false);
	dfdx2.resize(m,n,false);
	P.resize(m,n,false);
	Q.resize(m,n,false);
#ifndef InterpoleXolds
	Ro=xold2=xold1=xval;
#else
	Ro=xval;
#endif

	for (size_t i=0;i<n;i++)
	{
		xmin(i)=0.001;
		xmax(i)=1.00;
		#ifndef InterpoleXolds
			upp(i)=xmax(i);
			low(i)=xmin(i);
		#endif
	}

#ifdef FreezeInnerSolidX
	if (MDLenpart > Lenpart || MDWidpart > Widpart)
	{
		InnerMapIndex.resize(n);
		FreezeSolidX(NStory,SHeights);
	}
#else
	if (MDLenpart > Lenpart || MDWidpart > Widpart)
	{
		InnerMapIndex.resize(n);
		InnerMapIndexRev.resize(n);
		for (size_t i=0;i<n;i++)
			InnerMapIndexRev(i)=InnerMapIndex(i)=i;
	}
#endif

	/*for (size_t i=0;i<m;i++)
	{
		a(i)=0.00;
		d(i)=0.00;
		c(i)=1000.00;
	}*/

#ifdef ChangeSIMPPower
	InSimpPowers>>SimpPower;
#endif
	return true;

}

void Optimization::FreezeSolidX(size_t NStory=0 ,double *SHeights=NULL)				//This function freezes solid x
{
	size_t NonSolidCount=0;
	InnerMapIndex.clear();
	for (size_t xiter=0;xiter<n;xiter++)
	{
		int lhs,rhs,uhs,bhs;
		size_t iro=xiter%(MDLenpart+1);
		size_t jro=(xiter-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		int WanderMore=1;
		for (size_t iprime=iro+lhs-1;iprime<iro+2-rhs;iprime++)
		{
			for(size_t jprime=jro+bhs-1;jprime<jro+2-uhs;jprime++)
			{
				size_t xiterprime=iprime+jprime*(MDLenpart+1);
				if ((InnerSolidUV>xval(xiter) && xval(xiter)>InnerSolidLV) || (InnerSolidUV>xval(xiterprime) && xval(xiterprime)>InnerSolidLV) || abs(xval(xiter)-xval(xiterprime))>InnerSolidLV)
				{
					#ifdef DynSBSFillStorySolid
					for (size_t istory=0;istory<NStory;istory++)
					{
						if ((xiter>=((SHeights[istory]/(Width/MDWidpart))*(MDLenpart+1))) && (xiter<=((SHeights[istory]/(Width/MDWidpart))*(MDLenpart+1)+MDLenpart)))
						{
							WanderMore = 0;
							break;
						}
					}
					if (WanderMore == 0)
						break;
					#endif
					NonSolidCount++;
					InnerMapIndex(xiter)=NonSolidCount;
					WanderMore = 0;
					//cout<<"haha "<<xiter<<"  "<<xval(xiter)<<"  "<<xval(xiterprime)<<endl;
					break;
				}
				//else
					//cout<<"	OUT"<<xiter<<"  "<<xval(xiter)<<"  "<<xval(xiterprime)<<endl;
			}
			if (WanderMore == 0)
				break;
		}

	}
	xval.resize(NonSolidCount);
	InnerMapIndexRev.resize(NonSolidCount);
	for (size_t i=0;i<n;i++)
		if (InnerMapIndex(i) != 0)
		{
			xval(InnerMapIndex(i)-1)=Ro(i);
			InnerMapIndexRev(InnerMapIndex(i)-1)=i;
		}

	n=NonSolidCount;

	//Ro.resize(n);
	low.resize(n);
	upp.resize(n);
	xmin.resize(n);
	xmax.resize(n);
	alfa.resize(n);
	beta.resize(n);
	p0.resize(n);
	q0.resize(n);

	df0dx.resize(n);
	dfdx.resize(m,n);
	df0dx2.resize(n,false);
	dfdx2.resize(m,n,false);
	P.resize(m,n,false);
	Q.resize(m,n,false);

	/*Ro=*/xold2=xold1=xval;

	for (size_t i=0;i<n;i++)
	{
		low(i)=xmin(i)=0.001;
		upp(i)=xmax(i)=1.00;

		//Ro(InnerMapIndexRev(i))=xval(i);
	}
	#ifdef FreezeInnerSolidX
		ofstream OutXFrozen("FrozenSolidX.txt");
		OutXFrozen<<"Mdgeo: "<<MDLenpart<<" , "<<MDWidpart<<endl<<Ro<<endl<<InnerMapIndex<<endl<<InnerMapIndexRev<<endl<<xval<<endl;
	#endif
}

void Optimization::FollowTrace()
{
	ifstream FollowDataIn("Following/FollowTraceDataIn.txt");
	ifstream FollowXIn("Following/FollowTraceXIn.txt");
	FollowDataIn>>iter>>f0val>>f0valDifRate>>xvalMaxDifRate;
	for (size_t i=0;i<(MDLenpart+1)*(MDWidpart+1);i++)
	{
		FollowXIn>>xval(i);
		Ro(i)=xval(i);
	}
	#ifndef OptimizationisDynamic
		for (size_t i=0;i<(MDLenpart+1)*(MDWidpart+1);i++)
			FollowXIn>>xold1(i);
		for (size_t i=0;i<(MDLenpart+1)*(MDWidpart+1);i++)
			FollowXIn>>xold2(i);
	#else
		for (size_t i=0;i<(MDLenpart+1)*(MDWidpart+1);i++)
			xold1(i)=xold2(i)=xval(i);
	#endif

	#ifdef TraceFollowPrint
		ofstream FollowXOut("Following/FollowXOut.txt");
		FollowXOut<<xval<<endl<<xold1<<endl<<xold2<<endl;
	#endif
	#ifndef OptimizationisDynamic
		for (CVector::iterator iterat=Loads.begin();iterat != Loads.end();iterat++)
				if (Index(iterat.index())<UNum)
					df0du(Index(iterat.index())) = *iterat;
	#endif
	for (size_t xiter=0;xiter<n;xiter++)
	{
		int lhs,rhs,uhs,bhs;
		size_t jro=(xiter-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;
		dfdx(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
	}
}

void Optimization::CutXTail()
{
	double TailVal = 1e3;
	for (size_t i = 0; i < n; i++)
	{
		double tail;
		tail = xval(i)*TailVal-floor(xval(i)*TailVal);
		if (tail >= 0.50)
		{
			xval(i) *= TailVal;
			xval(i) = ceil(xval(i));
			xval(i) /= TailVal;
		}
		else
		{
			xval(i) *= TailVal;
			xval(i) = floor(xval(i));
			xval(i) /= TailVal;
		}
	}
}

void Optimization::CalResidue()
{
	f0valDifRate=(f0val-f0valOld)/f0valOld;
	xvalMaxDifRate=norm_inf(xval-xold1);
}

void Optimization::FilterSenstvts(double &FRadius)
{
	if (MyMPIRank == 0)
		cout<<"Entering filtering sensitivities"<<endl;
	Vector Fdf0dx(n);
	double a(Length/MDLenpart),b(Width/MDWidpart);
	for (size_t xiter=0;xiter<n;xiter++)
	{
		size_t mprimeLeft(0),mprimeRight(MDLenpart),nprimeBottom(0),nprimeUp(MDWidpart);
		double X,Y;
		//size_t iro=xiter%(MDLenpart+1);
		//size_t jro=(xiter-iro)/(MDLenpart+1);
		X=xiter%(MDLenpart+1)*a;
		Y=(xiter-X/a)/(MDLenpart+1)*b;
		if (X-FRadius>0.00)
			mprimeLeft=(X-FRadius+fmod(FRadius,a))/a;
		if (X+FRadius<Length)
			mprimeRight=(X+FRadius-fmod(FRadius,a))/a;
		if (Y-FRadius>0.00)
			nprimeBottom=(Y-FRadius+fmod(FRadius,b))/b;
		if (Y+FRadius<Width)
			nprimeUp=(Y+FRadius-fmod(FRadius,b))/b;

		double radius,sigma1(0),sigma2(0);

		for (size_t nprime=nprimeBottom;nprime<nprimeUp+1;nprime++)
			for (size_t mprime=mprimeLeft;mprime<mprimeRight+1;mprime++)
			{
				radius=sqrt(pow(X-mprime*a,2.00)+pow(Y-nprime*b,2.00));
				if (radius <= FRadius)
				{
					size_t xiter2=mprime+nprime*(MDLenpart+1);
					sigma1 += (FRadius-radius)*df0dx(xiter2)*xval(xiter2);
					sigma2 += (FRadius-radius);
				}
			}
		Fdf0dx(xiter)=sigma1/(sigma2*xval(xiter));
		/*if(xiter<30)
			cout<<Fdf0dx(xiter)<<" , "<<df0dx(xiter)<<endl;*/
	}
	df0dx=Fdf0dx;
	if (MyMPIRank == 0)
		cout<<"Filtering is done"<<endl;
}

void Optimization::ElFilterSenstvts(double &FRadius)
{
	if (MyMPIRank == 0)
		cout<<"Entering element based filtering sensitivities"<<endl;
	Vector Fdf0dx(n);
	for (size_t xiter=0;xiter<n;xiter++)
	{
		size_t mLeft(0),mRight(Lenpart-1),nBottom(0),nUp(Widpart-1);
		double X,Y;
		//size_t iro=xiter%(MDLenpart+1);
		//size_t jro=(xiter-iro)/(MDLenpart+1);
		X=(xiter%Lenpart)*Elemlen;
		Y=(xiter-xiter%Lenpart)/Lenpart*Elemwid;
		if (X-FRadius > 0.00)
			mLeft=(X-FRadius+fmod(FRadius,Elemlen))/Elemlen;
		if (X+FRadius < Length)
			mRight=(X+FRadius-fmod(FRadius,Elemlen))/Elemlen;
		if (Y-FRadius > 0.00)
			nBottom=(Y-FRadius+fmod(FRadius,Elemwid))/Elemwid;
		if (Y+FRadius<Width)
			nUp=(Y+FRadius-fmod(FRadius,Elemwid))/Elemwid;

		double radius,sigma1(0),sigma2(0);

		for (size_t nprime=nBottom;nprime<nUp+1;nprime++)
			for (size_t mprime=mLeft;mprime<mRight+1;mprime++)
			{
				radius=sqrt(pow(X-mprime*Elemlen,2.00)+pow(Y-nprime*Elemwid,2.00));
				if (radius <= FRadius)
				{
					size_t xiter2=mprime+nprime*Lenpart;
					sigma1 += (FRadius-radius)*df0dx(xiter2)*xval(xiter2);
					sigma2 += (FRadius-radius);
				}
			}
		Fdf0dx(xiter)=sigma1/(sigma2*xval(xiter));
		/*if(xiter<30)
			cout<<Fdf0dx(xiter)<<" , "<<df0dx(xiter)<<endl;*/
	}
	df0dx=Fdf0dx;
	if (MyMPIRank == 0)
		cout<<"Element based filtering is done"<<endl;
}

bool Optimization::Optimize()
{
	//slusolve.FTSP=true;
	ofstream outoptimize("OutOptimize.txt");
	ofstream OutXOptimize("OutXOptimize.txt");
	ofstream OutdfOptimize("OutdfOptimize.txt");
	ofstream OutdfFDOptimize("OutdfFDOptimize.txt");
	ofstream outk("OutK.txt");

#ifdef TraceFollow
	FollowTrace();
#endif
	bool Resume=true;
	do
	{
		if ((MDLenpart == Lenpart) && (MDWidpart == Widpart))
			Resume=false;
		while (/*(iter<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
			//OutXOptimize<<"mdgeo: "<<MDLenpart<<" , "<<MDWidpart<<endl;
			//OutXOptimize<<"Iter num "<<iter<<endl;
			//OutXOptimize<<"xval= "<<xval<<endl<<endl<<endl<<endl;

			OutXOptimize<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
			for (size_t roiter=0;roiter<n;roiter++)
			{
				OutXOptimize<<xval(roiter);
				if (roiter<n-1)
					OutXOptimize<<",";
			}
			OutXOptimize<<"};"<<endl;

			//cout<<"xval= "<<xval<<endl;
			////////////////////////////////////////////////////////////////////////////////
#ifdef MTMakestiffness
			MakeStiffnessMT();
#else
			MakeStiffness();
#endif
			//cout<<KuuS.nnz()<<endl;
			//cout<<"index="<<Index<<endl<<"indexrev="<<IndexRev<<endl;
			/*if(iter==1)
			{
				double differ=0.00;
				Kuu=trans(Kuu)-Kuu;
				for (CMatrix::iterator1 iii=Kuu.begin1(); iii!=Kuu.end1();iii++)
					for (CMatrix::iterator2 jjj=iii.begin();jjj!=iii.end();jjj++)
						differ += abs(*jjj);
				cout<<"difference= "<<differ<<endl;
			}*/
#ifdef DebugK
			if (iter==0)
			{
				ofstream Kijvout("Knew.ijv");

				for (CSMatrix::iterator1 i1=KuuS.begin1();i1 != KuuS.end1();i1++)
					for (CSMatrix::iterator2 i2=i1.begin();i2 != i1.end(); i2++)
						Kijvout<<i2.index1()+1<<" "<<i2.index2()+1<<"  "<<*i2<<endl;
				for (int i=0;i<62;i++)
				{
					Kijvout<<KuuS.value_data().begin()[i]<<"\t\t";
					Kijvout<<KuuS.index1_data().begin()[i]<<"\t\t";
					Kijvout<<KuuS.index2_data().begin()[i]<<endl;
				}
			}
#endif
			//cout<<norm_inf(trans(Kuu)-Kuu)<<endl;
			//cout<<Fu<<endl;
			//cout<<Disps<<endl;
			//if (iter==1)
			//outk<<"Kuu= "<<Kuu<<endl;
			/*		if (iter==1)
					{
						ofstream Kijvout("K.ijv");
						ofstream Fout("F.vec");

						for(CMatrix::iterator1 i1=Kuu.begin1();i1 != Kuu.end1();i1++)
							for(CMatrix::iterator2 i2=i1.begin();i2 != i1.end(); i2++)
								Kijvout<<i2.index1()+1<<" "<<i2.index2()+1<<"  "<<*i2<<endl;
						for (size_t i=0;i<UNum;i++)
							Fout<<Fu(i)<<endl;
					}
			*/
			//cout<<KuuS<<endl;
			//cout<<Fu<<endl;
			//TAUCS_Solve(KuuS,Fu,Du);
			//cout<<"disps="<<Disps<<endl<<endl;

			/*if (iter==0)
			{
				cout<<Du<<endl;
				cout<<IndexRev<<endl;
			}*/
			//f0val=prec_inner_prod(Disps,Loads);

			//if(iter!=0)
			f0valOld=f0val;
			SensitvtAnalyz();
			CalResidue();
			//slusolve.FTSP=false;
#ifdef DebugDF
			//if (iter != 0)
			{
				OutdfOptimize<<"iter= "<<iter<<endl;
				OutdfOptimize<<df0dx<<endl;
			}
#endif
#ifdef DebugFD
			//if (iter == 2)
			{
				OutdfFDOptimize<<"df0dx adjoint m. iter= "<<iter<<endl;
				Vector df0dxFD(n);
				//Vector RoFD(n);
				//RoFD = Ro;
				for (size_t iiii=0;iiii<n;iiii++)
				{
					Ro(iiii) += pow(10.00,-6);
					MakeStiffness();
					TAUCS_Solve(KuuS,Fu,Du);
					//df0dxFD(iiii)=(prec_inner_prod(Disps,Loads)-f0val)/pow(10.00,-6);
					df0dxFD(iiii)=(prec_inner_prod(Disps,Disps)-f0val)/pow(10.00,-6);
					Ro(iiii) -= pow(10.00,-6);
					OutdfFDOptimize<<df0dxFD(iiii)<<" , "<<df0dx(iiii)<<endl;
				}
				//OutdfFDOptimize<<Du<<endl;
				Resume=false;

			}
#endif
			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
			if (/*(iter+1<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				//outoptimize<<"low in iter number "<<iter<<" is equal to: "<<low<<endl;
				//outoptimize<<"upp in iter number "<<iter<<" is equal to: "<<upp<<endl;
				mmasub();
				#ifdef CutXTails
					CutXTail();
				#endif
				Ro=xval;
			}
			iter++;
			////////////////////////////////////////////////////////////////////////////////
			cout<<"mdgeo: "<<MDLenpart<<" , "<<MDWidpart<<endl;
			cout<<"f0val in iteration Num "<<iter<<" = "<<f0val<<endl;
			cout<<"fval in iteration Num "<<iter<<" = "<<fval<<endl;
			/*cout<<"df0dx= "<<df0dx<<endl;
			cout<<"dfdx= "<<dfdx<<endl;*/
			//cout<<"Number of trys in mma: "<<counter<<endl;
			cout<<"Iteration number "<<iter<<" is ended"<<endl;
			cout<<"-------------------------------------------------------------------"<<endl;
			outoptimize<<"mdgeo: "<<MDLenpart<<" , "<<MDWidpart<<endl;
			outoptimize<<"f0val and fval in iteration number "<<iter<<" = "<<f0val<<" , "<<fval<<endl;
			//outoptimize<<"low in iter"<<iter<<" ="<<low<<endl;
			//outoptimize<<"upp in iter"<<iter<<" ="<<upp<<endl;
			//outoptimize<<"dfdx in iter"<<iter<<" ="<<dfdx<<endl;
			if (iter==1)
			{

				OutdfOptimize<<"df0dx in first iteration = "<<df0dx<<endl<<endl;
				OutdfOptimize<<"dfdx in first iteration = "<<dfdx<<endl<<endl;
				//outoptimize<<"Du="<<Du<<endl;
			}
			//outoptimize<<"df0dx in iter number "<<iter<<" is equal to: "<<df0dx<<endl;

			outoptimize<<"Number of trys in mma: "<<counter<<endl;
			//outoptimize<<"norm2 difference= "<<norm_2(xval-xold1)<<endl;
			outoptimize<<"-----------------------------------------------------------------------"<<endl;
		}
		//outoptimize<<"f0val= "<<f0val<<endl;
		//outoptimize<<"fval= "<<fval<<endl;
		//outoptimize<<"df0dx= "<<df0dx<<endl<<endl;
		//outoptimize<<"dfdx= "<<dfdx<<endl;
		//OutXOptimize<<"xval= "<<xval<<endl<<endl;
		outoptimize<<"Number of trys in mma: "<<counter<<endl;
		outoptimize<<"norm2 difference= "<<norm_2(xval-xold1)<<endl<<endl;
		X2NewX();
	}
	while (Resume);

	return true;
}

bool Optimization::InnerOptimize()
{
	/*///////////////////////////////////////////////////////
	// OptTyp specifies the type of optimization		   //
	// OptTyp=0		:Ordinary Inner Optimization		   //
	// OptTyp=1		:Ordinary Opt. or node based filter op.//
	// OptTyp=2		:Element based filtering Optimization  //
	///////////////////////////////////////////////////////*/

	int OptTyp=0;													// Shows Kind of optimization is inner
	bool Resume=true;
#ifdef FreezeInnerSolidX
	Matrix EfArea(m,Ro.size());
#endif
	ofstream outoptimize("InnerOutOptimize.txt");
	ofstream OutXOptimize("InnerXOutOpt.txt");
	ofstream OutdfOptimize("InnerdfOutOpt.txt");
	//ofstream outk("InnerOutK.txt");
	do
	{
		FTEIC=true;														// first time of each inner convergence
		vector<double> EIPI[MDLenpart*MDWidpart];						//M.D. Element Integration Points Information matrix
		while (/*(iter<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
			//OutXOptimize<<"mdgeo: "<<MDLenpart<<" , "<<MDWidpart<<endl;
			//OutXOptimize<<"Inner iter "<<iter<<endl;
			//OutXOptimize<<"Innerxval= "<<Ro<<endl<<endl<<endl<<endl;
			OutXOptimize<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
			for (size_t roiter=0;roiter<(MDLenpart+1)*(MDWidpart+1);roiter++)
			{
				OutXOptimize<<Ro(roiter);
				if (roiter<(MDLenpart+1)*(MDWidpart+1)-1)
					OutXOptimize<<",";
			}
			OutXOptimize<<"};"<<endl;
			//cout<<"xval= "<<xval<<endl;
			////////////////////////////////////////////////////////////////////////////////
			MakeStiffness(OptTyp,EIPI);
			FTEIC=false;
			/*if(iter==1)
			{
				double differ=0.00;
				Kuu=trans(Kuu)-Kuu;
				for (CMatrix::iterator1 iii=Kuu.begin1(); iii!=Kuu.end1();iii++)
					for (CMatrix::iterator2 jjj=iii.begin();jjj!=iii.end();jjj++)
						differ += abs(*jjj);
				cout<<"difference= "<<differ<<endl;
			}*/


			//cout<<norm_inf(trans(Kuu)-Kuu)<<endl;
			//cout<<Fu<<endl;
			//cout<<Disps<<endl;
			//if (iter==1)
			//outk<<"Kuu= "<<Kuu<<endl;
			/*		if (iter==1)
					{
						ofstream Kijvout("K.ijv");
						ofstream Fout("F.vec");

						for(CMatrix::iterator1 i1=Kuu.begin1();i1 != Kuu.end1();i1++)
								Kijvout<<i2.index1()+1<<" "<<i2.index2()+1<<"  "<<*i2<<endl;
						for (size_t i=0;i<UNum;i++)
							Fout<<Fu(i)<<endl;
					}
			*/

			//TAUCS_Solve(KuuS,Fu,Du);
			//cout<<"disps="<<Disps<<endl<<endl;

			/*if (iter==0)
			{
				cout<<Du<<endl;
				cout<<IndexRev<<endl;
			}*/
			//f0val=prec_inner_prod(Disps,Loads);
			f0valOld=f0val;
			InSensitvtAnalyz(EIPI);
			CalResidue();
#ifdef InnerFDdfDebug
			ofstream OutFDDebug("InnerFDDebug.txt");
			double f0val2=f0val;
			for (size_t FDiter=0;FDiter<n;FDiter++)
			{
				Ro(FDiter) += 1e-6;
				MakeStiffness(OptTyp,EIPI);
				TAUCS_Solve(KuuS,Fu,Du);
				f0val2=prec_inner_prod(Disps,Loads);
				OutFDDebug<<df0dx(FDiter)<<" , ";
				df0dx(FDiter)=(f0val2-f0val)/1e-6;
				Ro(FDiter) -= 1e-6;
				cout<<"df0dx"<<df0dx(FDiter)<<endl;
				OutFDDebug<<df0dx(FDiter)<<endl;
			}
			OutFDDebug<<"iter num "<<iter<<" is done"<<endl;
			OutFDDebug<<" MD= "<<MDLenpart<<" , "<<MDWidpart<<endl;
			OutFDDebug<<"___________________________________________________________"<<endl;
#endif
#ifdef FreezeInnerSolidX

			if (iter==0)
			{
				for (size_t xiter=0; xiter<Ro.size(); xiter++)
				{
					int lhs,rhs,uhs,bhs;
					size_t iro=xiter%(MDLenpart+1);
					size_t jro=(xiter-iro)/(MDLenpart+1);

					lhs = (iro == 0) ? 1 : 0;
					rhs = (iro == MDLenpart) ? 1 : 0;
					bhs = (jro == 0) ? 1 : 0;
					uhs = (jro == MDWidpart) ? 1 : 0;
					EfArea(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
				}
			}
			axpy_prod(EfArea,Ro,fval);
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;

#else
			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
#endif
			if (/*(iter+1<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				//outoptimize<<"low in iter number "<<iter<<" is equal to: "<<low<<endl;
				//outoptimize<<"upp in iter number "<<iter<<" is equal to: "<<upp<<endl;
				mmasub();
				#ifdef CutXTails
					CutXTail();
				#endif
				#ifndef FreezeInnerSolidX
					Ro=xval;
				#else
					for (size_t i=0;i<InnerMapIndexRev.size();i++)
						Ro(InnerMapIndexRev(i))=xval(i);
				#endif
			}
			iter++;
			////////////////////////////////////////////////////////////////////////////////
			cout<<"Inner f0val in iteration Num "<<iter<<" = "<<f0val<<endl;
			cout<<"Inner fval in iteration Num "<<iter<<" = "<<fval<<endl;
			/*cout<<"df0dx= "<<df0dx<<endl;
			cout<<"dfdx= "<<dfdx<<endl;*/
			//cout<<"Number of trys in mma: "<<counter<<endl;
			cout<<"Inner iteration number "<<iter<<" is ended"<<endl;
			cout<<"-------------------------------------------------------------------"<<endl;
			cout<<"mdgeo: "<<MDLenpart<<" , "<<MDWidpart<<endl;
			outoptimize<<"mdgeo: "<<MDLenpart<<" , "<<MDWidpart<<endl;
			outoptimize<<"Inner f0val and fval in iteration number "<<iter<<" = "<<f0val<<" , "<<fval<<endl;
			//outoptimize<<"low in iter"<<iter<<" ="<<low<<endl;
			//outoptimize<<"upp in iter"<<iter<<" ="<<upp<<endl;
			//outoptimize<<"dfdx in iter"<<iter<<" ="<<dfdx<<endl;
			if (iter==1)
			{

				OutdfOptimize<<"Inner df0dx in first iteration = "<<df0dx<<endl<<endl;
				OutdfOptimize<<"Inner dfdx in first iteration = "<<dfdx<<endl<<endl;
				//outoptimize<<"Du="<<Du<<endl;
			}
			//outoptimize<<"df0dx in iter number "<<iter<<" is equal to: "<<df0dx<<endl;

			outoptimize<<"Number of trys in mma: "<<counter<<endl;
			//outoptimize<<"norm2 difference= "<<norm_2(xval-xold1)<<endl;
			outoptimize<<"-----------------------------------------------------------------------"<<endl;
		}
		//outoptimize<<"f0val= "<<f0val<<endl;
		//outoptimize<<"fval= "<<fval<<endl;
		//outoptimize<<"df0dx= "<<df0dx<<endl<<endl;
		//outoptimize<<"dfdx= "<<dfdx<<endl;
		//OutXOptimize<<"xval= "<<xval<<endl<<endl;
		outoptimize<<"Number of trys in mma: "<<counter<<endl;
		//outoptimize<<"norm2 difference= "<<norm_2(xval-xold1)<<endl<<endl;
		Resume=X2NewX();
	}
	while (Resume);
	return true;
}

bool Optimization::FilterOptimize()
{
	ifstream InFilter("InFilter.txt");
	ofstream OutFilter("OutFilter.txt");
	ofstream OutXFilter("OutXFilter.txt");
	ofstream OutdfFilter("OutdfFilter.txt");
#ifdef FDdfFiltering
	ofstream FDFilterOut("FDFilterOut.txt");
#endif
	double FilterRadius;
	while (InFilter>>FilterRadius)
	{
		iter=0;
		while (/*(iter<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
			cout<<"Filter Radius="<< FilterRadius<<endl;
			OutFilter<<"Filter Radius="<< FilterRadius<<endl;
			//OutXFilter<<"Filter Radius="<< FilterRadius<<endl;

			cout<<"iter number"<<iter<<endl;
			OutFilter<<"iter number"<<iter<<endl;
			//OutXFilter<<"iter number"<<iter<<endl;

			MakeStiffness();
			TAUCS_Solve(KuuS,Fu,Du);

			f0valOld=f0val;
			f0val=prec_inner_prod(Disps,Loads);
			SensitvtAnalyz();
			CalResidue();
#ifdef FDdfFiltering
			double f0val2;
			for (size_t FD_iter=0;FD_iter<n;FD_iter++)
			{
				Ro(FD_iter) += 1e-9;
				MakeStiffness();
				TAUCS_Solve(KuuS,Fu,Du);
				FDFilterOut<<"i= "<<FD_iter<<" : "<<df0dx(FD_iter)<<" , "<<(f0val2-f0val)/1e-9<<endl;
				Ro(FD_iter) -= 1e-9;
			}
#endif
			OutdfFilter<<"before df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl;
			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
			if (/*(iter+1<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				FilterSenstvts(FilterRadius);
				OutdfFilter<<" after df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl<<endl<<endl;
				mmasub();
				#ifdef CutXTails
					CutXTail();
				#endif
				Ro=xval;
			}
			cout<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			OutFilter<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			OutFilter<<"__________________________________________________________________"<<endl<<endl;
			cout<<"__________________________________________________________________"<<endl<<endl;
			OutFilter<<"__________________________________________________________________"<<endl<<endl;
			//OutXFilter<<xval<<endl<<endl<<endl;
			//OutXFilter<<"__________________________________________________________________"<<endl<<endl;

			OutXFilter<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
			for (size_t roiter=0;roiter<n;roiter++)
			{
				OutXFilter<<xval(roiter);
				if (roiter<n-1)
					OutXFilter<<",";
			}
			OutXFilter<<"};"<<endl;

			iter++;
		}
	}
	return true;
}

bool Optimization::ElFilterOptimize()
{
	int OptTyp=2;
	ifstream InFilter("InEFilter.txt");
	ofstream OutFilter("OutEFilter.txt");
	ofstream OutXFilter("OutEXFilter.txt");
	ofstream OutdfFilter("OutdfEFilter.txt");
#ifdef FDdfElFiltering
	ofstream FDElFilterOut("FDElFilterOut.txt");
#endif
	double FilterRadius;
	while (InFilter>>FilterRadius)
	{
		iter=0;
		while (/*(iter<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
			cout<<"Filter Radius="<< FilterRadius<<endl;
			OutFilter<<"Filter Radius="<< FilterRadius<<endl;
			//OutXFilter<<"Filter Radius="<< FilterRadius<<endl;

			cout<<"iter number"<<iter<<endl;
			OutFilter<<"iter number"<<iter<<endl;
			//OutXFilter<<"iter number"<<iter<<endl;

			MakeStiffness(OptTyp);
			TAUCS_Solve(KuuS,Fu,Du);

			f0valOld=f0val;
			f0val=prec_inner_prod(Disps,Loads);
			ElFSensitvtAnalyz();
			CalResidue();
#ifdef FDdfElFiltering
			double f0val2;
			for (size_t FD_EFilter=0;FD_EFilter<n;FD_EFilter++)
			{
				Ro(FD_EFilter) += 1e-9;
				MakeStiffness(OptTyp);
				TAUCS_Solve(KuuS,Fu,Du);
				f0val2=prec_inner_prod(Disps,Loads);
				FDElFilterOut<<"i= "<<FD_EFilter<<" : "<<df0dx(FD_EFilter)<<" , "<<(f0val2-f0val)/1e-9<<endl;
				Ro(FD_EFilter) -= 1e-9;
			}
			FDElFilterOut<<"iter num "<<iter<<" is done"<<endl<<"___________________________________"<<endl;
#endif

			OutdfFilter<<"before df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl;
			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
			if (/*(iter+1<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				ElFilterSenstvts(FilterRadius);
				OutdfFilter<<" after df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl<<endl<<endl;
				mmasub();
				#ifdef CutXTails
					CutXTail();
				#endif
				Ro=xval;
			}
			cout<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			OutFilter<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			OutFilter<<"__________________________________________________________________"<<endl<<endl;
			cout<<"__________________________________________________________________"<<endl<<endl;
			OutFilter<<"__________________________________________________________________"<<endl<<endl;
			//OutXFilter<<xval<<endl<<endl<<endl;
			//OutXFilter<<"__________________________________________________________________"<<endl<<endl;
			OutXFilter<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
			for (size_t roiter=0;roiter<n;roiter++)
			{
				OutXFilter<<xval(roiter);
				if (roiter<n-1)
					OutXFilter<<",";
			}
			OutXFilter<<"};"<<endl;

			iter++;
		}
	}
	return true;
}

class DynamicOptimization : public Optimization
{
	ifstream InDynData;
	// bool FTEIC;

public:
	DynamicOptimization () : InDynData("DynamicOpt/Dyn_InData.txt")
	{
		InDynData>>Dt>>StartT>>EndT;
		#ifdef DynDamping
			InDynData>>CoefDK>>CoefDM;
		#endif
		DudotInit.resize(UNum);
		DuInit.resize(UNum);
		//Du2dot.resize(UNum);
		TimeIterN=(EndT-StartT)/Dt+1;
		DynLoad.resize(TimeIterN);
		MassS.resize(UNum,UNum);
		MassT.resize(UNum,UNum);

		#ifdef DynSBSEarthQuake
			du2dotG.resize(UNum);
		#else
			#ifdef DynEarthQuake
				du2dotG.resize(UNum);
			#endif
		#endif

		DuInit.clear();
		DudotInit.clear();
		//Du2dot.clear();

		if (!DynInitialize())
		{
			printf("First Displacements or velocities could not be read in dynamic optimization\n");
			exit(1);
		}
	}
	float Dt;
	double StartT;
	double EndT;
	#ifdef DynDamping
		double CoefDK;							//Coefficient of Stiffness part of damping matrix
		double CoefDM;							//Coefficient of Mass part of damping matrix
	#endif
	size_t TimeIterN;
	size_t LEP;
	Vector DudotInit;
	//Vector Du2dot;
	Vector DuInit;
	Vector DynLoad;
	#ifdef DynSBSEarthQuake
		Vector du2dotG;
	#else
		#ifdef DynEarthQuake
			Vector du2dotG;
		#endif
	#endif
	CSMatrix MassS;
	CMatrix MassT;

	bool DynInitialize();

	void MassroBuilder(size_t &elem ,double *ro,int condition);
	void MassroBuilderIn(size_t &elem,double *ro,vector<double> *EIPI);
	//void MassInSensitvtroBuilder(double *ro,size_t mprime,size_t nprime,size_t i,size_t DPN,vector<double> *EIPI);

	void DynAnalyse(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal);
	bool DynSensitvtAnalyse(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal);
	bool DynSensitvtAnalyseMPI(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal);
	bool DynSensitvtAnalyseFTU(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal);
	bool DynSensitvtAnalyseFTUMPI(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal);

	void DynInSensitvtAnalyz(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal);
	void DynInSensitvtAnalyzFTU(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal);
	void DynInSensitvtAnalyzMPI(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal);
	void DynInSensitvtAnalyzFTUMPI(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal);
	void BuildMassMatrix(int OptTyp,vector<double> *EIPI);

	bool DynOptimize();
	bool DynInnerOptimize();

	bool DynFilterOptimize();
	bool DynElFSensitvtAnalyz(Vector *DuTotal/*,Vector *DudotTotal*/,Vector *Du2dotTotal);
	bool DynElFSensitvtAnalyzMPI(Vector *DuTotal/*,Vector *DudotTotal*/,Vector *Du2dotTotal);
	bool DynElFilterOptimize();

	void BuildSBSMassMatrix(size_t &NStory,double *SMasses,double *SHeights);
	void DynModifySBSStiffness(size_t &NStory,double *SHeights);
	void DynSBSAnalyse(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights);
	bool DynSBSSensitvtAnalyse(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights);
	bool DynSBSSensitvtAnalyseMPI(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights);
	bool DynSBSOptimize();
	void DynSBSInSensitvtAnalyz(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights);
	void DynSBSInSensitvtAnalyzMPI(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights);
	bool DynSBSInnerOptimize();

	bool DynSBSFilterOptimize();
	bool DynSBSElFSensitvtAnalyz(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights);
	bool DynSBSElFSensitvtAnalyzMPI(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights);
	bool DynSBSElFilterOptimize();
};

bool DynamicOptimization::DynInitialize()
{
	size_t i=0;
	double d=0;
	ifstream InDynInitDisps("DynamicOpt/DynInitDisps.txt");
	ifstream InDynInitVelocs("DynamicOpt/DynInitVelocs.txt");
	#ifdef DynSBSEarthQuake
		ifstream InDynInitLoads("DynamicOpt/DynEQAcceleration.txt");
	#else
		#ifdef DynEarthQuake
			ifstream InDynInitLoads("DynamicOpt/DynEQAcceleration.txt");
		#else
			ifstream InDynInitLoads("DynamicOpt/DynInitLoads.txt");
		#endif
	#endif

	while (InDynInitDisps>>i)
	{
		InDynInitDisps>>d;
		DuInit(IndexRev(i))=d;
	}
	while (InDynInitVelocs>>i)
	{
		InDynInitVelocs>>d;
		DudotInit(IndexRev(i))=d;
	}
	InDynInitLoads>>LEP;
	for (i=0;i<TimeIterN;i++)
		InDynInitLoads>>DynLoad(i);
	//cout<<DynLoad<<endl;
	return true;
}

void DynamicOptimization::MassroBuilder(size_t &elem ,double *ro,int condition=0)
{
	size_t i=elem%MDElemLenNum;
	size_t j=((elem%(Lenpart*MDElemWidNum))-(elem%(Lenpart*MDElemWidNum))%Lenpart)/Lenpart;
	size_t mprime=((elem-elem%MDElemLenNum)/MDElemLenNum)%MDLenpart;
	size_t nprime=(elem-elem%(Lenpart*MDElemWidNum))/(Lenpart*MDElemWidNum);

	double ro1,ro2,ro3,ro4;

	ro1=Ro(mprime+nprime*(MDLenpart+1));
	ro2=Ro(mprime+nprime*(MDLenpart+1)+1);
	ro3=Ro(mprime+(nprime+1)*(MDLenpart+1)+1);
	ro4=Ro(mprime+(nprime+1)*(MDLenpart+1));


	double ks,et;
	Vector N(4);
	if (condition != 0)
	{
		cout<<"error because of line 3373"<<endl;
		exit(1);
	}
	for (size_t ii=0;ii<IntegPoints.size1();ii++)
	{
		ks=(2.00*i+IntegPoints(ii,0)-MDElemLenNum+1.00)/MDElemLenNum;
		et=(2.00*j+IntegPoints(ii,1)-MDElemWidNum+1.00)/MDElemWidNum;

		N(1)=1.00/4.00*(1.00+ks)*(1.00-et);
		N(2)=1.00/4.00*(1.00+ks)*(1.00+et);
		N(3)=1.00/4.00*(1.00-ks)*(1.00+et);
		ro[ii]=/*(condition ? N(condition-1) :*/ pow((N(0)*ro1+N(1)*ro2 + N(2)*ro3 + N(3)*ro4),MassPower)/*)*/;	 //to conclude both sensitivity and analyse mode

#ifdef MassModificationDynOpt
		if ((N(0)*ro1+N(1)*ro2 + N(2)*ro3 + N(3)*ro4) <= RoModifyVal) 								/**Because it is only used when condition = 0 else should be (N(0)*ro1+N(1)*ro2 + N(2)*ro3 + N(3)*ro4)**/

			ro[ii] = /*(condition ? (6.00*N(condition-1)*pow((N(0)*ro1+N(1)*ro2 + N(2)*ro3 + N(3)*ro4),5.00)) :*/ pow((N[0]*ro1 + N[1]*ro2 + N[2]*ro3 + N[3]*ro4),MassModifyP)/*)*/;
#endif
	}
}

void DynamicOptimization::MassroBuilderIn(size_t &elem,double *ro,vector<double> *EIPI=0)
{
	size_t m2 , n2 , mprime , nprime;
	m2=elem%Lenpart;
	n2=(elem-m2)/Lenpart;

	double ks,et,N[4],ro1,ro2,ro3,ro4;

	for (size_t i=0;i<IntegPoints.size1();i++)
	{
		mprime=size_t(((IntegPoints(i,0)+1.00)/2.00+m2)*Elemlen/(Length/MDLenpart));
		nprime=size_t(((IntegPoints(i,1)+1.00)/2.00+n2)*Elemwid/(Width/MDWidpart));

		ro1=Ro(mprime+nprime*(MDLenpart+1));
		ro2=Ro(mprime+nprime*(MDLenpart+1)+1);
		ro3=Ro(mprime+(nprime+1)*(MDLenpart+1)+1);
		ro4=Ro(mprime+(nprime+1)*(MDLenpart+1));

		ks=fmod(((IntegPoints(i,0)+1.00)/2.00+m2)*Elemlen,(Length/MDLenpart))*2.00/(Length/MDLenpart)-1.00;
		et=fmod(((IntegPoints(i,1)+1.00)/2.00+n2)*Elemwid,(Width/MDWidpart))*2.00/(Width/MDWidpart)-1.00;
		if (FTEIC)
		{
			EIPI[mprime+nprime*MDLenpart].push_back(i);
			EIPI[mprime+nprime*MDLenpart].push_back(elem);
			EIPI[mprime+nprime*MDLenpart].push_back(ks);
			EIPI[mprime+nprime*MDLenpart].push_back(et);
		}
		N[0]=1.00/4.00*(1.00-ks)*(1.00-et);
		N[1]=1.00/4.00*(1.00+ks)*(1.00-et);
		N[2]=1.00/4.00*(1.00+ks)*(1.00+et);
		N[3]=1.00/4.00*(1.00-ks)*(1.00+et);

		ro[i]=pow((N[0]*ro1 + N[1]*ro2 + N[2]*ro3 + N[3]*ro4),MassPower);
#ifdef MassModificationInnerDynOpt
		if ((N[0]*ro1 + N[1]*ro2 + N[2]*ro3 + N[3]*ro4) <= RoModifyVal)
			ro[i] = pow((N[0]*ro1 + N[1]*ro2 + N[2]*ro3 + N[3]*ro4),MassModifyP);
#endif
	}
}

/*void DynamicOptimization::MassInSensitvtroBuilder(double *ro,size_t mprime,size_t nprime,size_t i,size_t DPN,vector<double> *EIPI)
{
	double ks,et,N[4],ro1,ro2,ro3,ro4;
	ro[0]=ro[1]=ro[2]=ro[3]=0.00;

	ro1=Ro(mprime+nprime*(MDLenpart+1));
	ro2=Ro(mprime+nprime*(MDLenpart+1)+1);
	ro3=Ro(mprime+(nprime+1)*(MDLenpart+1)+1);
	ro4=Ro(mprime+(nprime+1)*(MDLenpart+1));

	ks=EIPI[mprime+nprime*MDLenpart][2+i*4];
	et=EIPI[mprime+nprime*MDLenpart][3+i*4];

	N[0]=1.00/4.00*(1.00-ks)*(1.00-et);
	N[1]=1.00/4.00*(1.00+ks)*(1.00-et);
	N[2]=1.00/4.00*(1.00+ks)*(1.00+et);
	N[3]=1.00/4.00*(1.00-ks)*(1.00+et);

	ro[size_t(EIPI[mprime+nprime*MDLenpart][i*4])]=N[DPN-1];
}*/

void DynamicOptimization::BuildMassMatrix(int OptTyp=1,vector<double> *EIPI=0)
{
	if (MyMPIRank == 0)
		cout<<"Entering Mass matrix builder"<<endl;
	MassS.clear();
	//#ifdef OptimizationisDynamic
	//MassT.clear();
	//#endif

//#pragma omp parallel for schedule(dynamic)
	for (size_t elem=0;elem<ElemsNum;elem++)
	{
		double ro[IntegPoints.size1()];
		//double Massro[IntegPoints.size1()];

		Matrix melem(8,8);
		size_t ii(0),jj(0);
		if (OptTyp == 1)
			MassroBuilder(elem,ro);
		else if (OptTyp == 0)
			MassroBuilderIn(elem,ro,EIPI);
		else if (OptTyp == 2)
			for (unsigned int i=0;i<IntegPoints.size1();i++)
				ro[i]=pow(Ro(elem),MassPower);

		//cout<<ro<<" =ro"<<endl;
		Element.mbuild(ro,melem);
		//cout<<Element.k<<"  =k"<<endl;
		//cout<<Index<<" =Index"<<endl;
		for (size_t i=0;i<8;i++)
			for (size_t j=0;j<8;j++)
			{
				//cout<<i<<" , "<<j<<endl;
				ii = Connectivity(elem,i);
				jj = Connectivity(elem,j);

				////////////////////////////////////////////////////

				if (Index(ii) < UNum && Index(jj) < UNum && Index(ii)<=Index(jj))
				{
					//#pragma omp critical
					//cout<<i<<" , "<<j<<" , "<<melem(i,j);
#ifndef OptimizationisDynamic
					/*KuuS(Index(jj),Index(ii)) = */
					MassS(Index(ii),Index(jj)) += melem(i,j);
#else
					MassS(Index(ii),Index(jj)) += melem(i,j);
					MassT(Index(jj),Index(ii))=MassT(Index(ii),Index(jj))=MassS(Index(ii),Index(jj));
#endif
				}
			}
	}
	if (MyMPIRank == 0)
		cout<<"Mass Matrix is built"<<endl;
}

void DynamicOptimization::DynAnalyse(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal)
{
	DuTotal[0]=DuInit;
	#ifdef DynDamping
		DudotTotal[0]=DudotInit;
	#endif
	//DudotTotal[0]=Dudot;
//#ifdef OptimizationisDynamic
	axpy_prod(KuuT,-DuTotal[0],Du);										//Du used just as a vector no meaning
//#endif
	Du(Index(LEP)) += DynLoad(0);										//Because of no settelments
	TAUCS_Solve(MassS,Du,Du2dotTotal[0],false);
	//cout<<"shetab"<<Du2dotTotal[0]<<endl<<Du<<endl<<Index(LEP)<<endl<<MassT<<endl; exit(1);

	CSMatrix KHat(UNum,UNum);

	#ifndef DynDamping
		KHat=KuuS+(4.00/Dt/Dt)*MassS;
		#ifdef Dynf0valUTKHatU
			CMatrix KHatT(UNum,UNum);
			KHatT=KuuT+(4.00/Dt/Dt)*MassT;
		#endif
	#else
		CuuS=CoefDK*KuuS+CoefDM*MassS;
		CuuT=CoefDK*KuuT+CoefDM*MassT;
		KHat=KuuS+2.00/Dt*CuuS+(4.00/Dt/Dt)*MassS;
		#ifdef Dynf0valUTKHatU
			CMatrix KHatT(UNum,UNum);
			KHatT=KuuT+2.00/Dt*CuuT+(4.00/Dt/Dt)*MassT;
		#endif
	#endif

	#ifdef Dynf0valUTSU
		CMatrix ST(UNum,UNum);
		ST=KuuT+sf0val*MassT;
	#endif
	void *F=NULL;
	Taucs_Factor_Solve(KHat,&F);
	//ofstream outu2dot("DynamicOpt/DynU2dotOut.txt");
	#ifdef DynUEA
		void *FMass=NULL;
		Taucs_Factor_Solve(MassS,&FMass);
	#endif
	#ifdef DynEarthQuake
		if (iter == 0)
		{
			du2dotG.clear();
			for (size_t inode=0;inode<NodesNum;inode++)
				{
					if (Index(inode*2)<UNum)
						du2dotG(Index(2*inode))=1.00;
				}
			//cout<<du2dotG<<endl;
		}
	#endif
	f0val=0.00;
	Vector Dudot(UNum);
	Dudot=DudotInit;
	if (MyMPIRank == 0)
		cout<<"Entering Analysis"<<endl;
	size_t iterj=0;

	for (double timei=StartT+Dt;timei <= EndT;timei += Dt)
	{
		//if (MyMPIRank == 0)
		//	printf ("%5.2f %%",timei/EndT*100.00);
		iterj++;
		#ifndef DynEarthQuake
			axpy_prod(MassT,(4.00/Dt*Dudot+2.00*Du2dotTotal[iterj-1]),Fu);	//Here Fu is used just as a vector no meaning
			//cout<<Fu<<endl; exit(1);
			Fu(Index(LEP)) += (DynLoad(iterj)-DynLoad(iterj-1));
		#else
			Vector Ddu2dotG(UNum);
			Ddu2dotG = (DynLoad(iterj)-DynLoad(iterj-1))*du2dotG;
			axpy_prod(MassT,(4.00/Dt*Dudot+2.00*Du2dotTotal[iterj-1]-Ddu2dotG),Fu);	//Here Fu is used just as a vector no meaning
		#endif
		#ifdef DynDamping
			axpy_prod(CuuT,2.00*Dudot,Fu,false);
		#endif
		Taucs_Factor_Solve(KHat,&F,2,Fu,Du);
		DuTotal[iterj] =DuTotal[iterj-1]+Du;							//Du used here as Delta Displacement
#ifndef DynUEA
		Du2dotTotal[iterj]=4.00/Dt/Dt*Du-4.00/Dt*Dudot-Du2dotTotal[iterj-1];
#else
		/* Acceleration of equilibrium equation*/
		axpy_prod(-KuuT,DuTotal[iterj],Fu);
		Fu(Index(LEP)) += DynLoad(iterj);
		Taucs_Factor_Solve(MassS,&FMass,2,Fu,Du2dotTotal[iterj]);
#endif
		//outu2dot<<Du2dotTotal[iterj]<<endl;
		Dudot=2.00/Dt*Du-Dudot;
		#ifdef DynDamping
			DudotTotal[iterj]=Dudot;
		#endif
		//cout<<"du= "<<DuTotal[iterj]<<endl/*<<Du<<endl<<Du2dotTotal[iterj]<<endl*/;
#ifndef Dynf0valFTU
		f0val += prec_inner_prod(DuTotal[iterj],DuTotal[iterj])*ImpFac;
#else
	#ifdef Dynf0valUTKHatU
		f0val += prec_inner_prod(axpy_prod(KHatT,DuTotal[iterj],Fu),DuTotal[iterj])*ImpFac;//Here Fu used as a Vector no meaning
	#else
		#ifdef Dynf0valUTKU
			f0val += prec_inner_prod(axpy_prod(KuuT,DuTotal[iterj],Fu),DuTotal[iterj])*ImpFac;
		#else
			#ifdef Dynf0valUTMU
				f0val += prec_inner_prod(axpy_prod(MassT,DuTotal[iterj],Fu),DuTotal[iterj])*ImpFac;
			#else
				#ifdef Dynf0valUTSU
					f0val += prec_inner_prod(axpy_prod(ST,DuTotal[iterj],Fu),DuTotal[iterj])*ImpFac;
				#else
					#ifdef Dynf0valUTKDU
						f0val += prec_inner_prod(axpy_prod(KuuT,DuTotal[iterj],Fu),Du)*ImpFac;
					#else
						f0val += DuTotal[iterj](Index(LEP))*pow(1.00,2.00)*DuTotal[iterj](Index(LEP))*ImpFac;
					#endif
				#endif
			#endif
		#endif
	#endif
#endif
		//if (MyMPIRank == 0)
			//cout<<"Time analysis = "<<timei<<endl;
		//if (MyMPIRank == 0)
		//	printf ("\b\b\b\b\b\b\b");
	}
	/*if (iter==1)
	{
		ofstream testout("DynamicOpt/testout.txt");
		for (size_t itest=0;itest<TimeIterN;itest++)
		{
			testout<<DuTotal[itest]<<endl;
		}
		testout<<endl;
		for (size_t itest=0;itest<TimeIterN;itest++)
		{
			testout<<Du2dotTotal[itest]<<endl;
		}
		exit(1);
	}*/
	/** Free the factorization F **/
	//const char* options[] = {"taucs.factor.LLT=true", "taucs.factor.ll=true", "taucs.factor.ordering=amd", NULL };
	//taucs_linsolve(NULL,&F,0,NULL,NULL,(char**)options,NULL);
	Taucs_FreeFactor(&F);
	if (MyMPIRank == 0)
		cout<<"Analysis is done"<<endl;
}

bool DynamicOptimization::DynSensitvtAnalyse(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	void *F=NULL;
	df0dx.clear();
	Taucs_Factor_Solve(KHat,&F);
	cout<<"Building Landas"<<endl;
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		Fu += 2.00*DuTotal[iteri]*ImpFac;
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	Taucs_FreeFactor(&F);
	cout<<"Landas are built"<<endl;

	double ro[IntegPoints.size1()];
	double Massro[IntegPoints.size1()];
	cout<<"Entering sensitivity analysis"<<endl;

#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		CMatrix dMassdx(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif

		int lhs,rhs,uhs,bhs;
		size_t iro=xiter%(MDLenpart+1);
		size_t jro=(xiter-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		size_t mLeft=(lhs+iro-1)*MDElemLenNum;
		size_t nBottom=(bhs+jro-1)*MDElemWidNum;
		size_t nUp=(2-uhs+jro-1)*MDElemWidNum;

		for (size_t jelem=nBottom;jelem<nUp;jelem++)
			for (size_t ielem=mLeft;ielem<mRight;ielem++)
			{
				size_t elemnum = jelem*Lenpart+ielem;
				size_t ii,jj;
				size_t mprime=(ielem-ielem%MDElemLenNum)/MDElemLenNum;
				size_t nprime=(jelem-jelem%MDElemWidNum)/MDElemWidNum;
				roBuilder(elemnum,ro,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1),Massro); // Third arg. is to show the function which edge ro is being derived in a prime mesh desgine

				Element.kBuild(ro,kelem);
				//cout<<"massro= "<<Massro[0]<<" , "<<Massro[1]<<" , "<<Massro[2]<<" , "<<Massro[3]<<endl;
				Element.mbuild(Massro,melem);

				for (size_t i=0;i<8;i++)
					for (size_t j=0;j<8;j++)
					{
						ii = Connectivity(elemnum,i);
						jj = Connectivity(elemnum,j);

						if (Index(ii) < UNum && Index(jj) < UNum)
						{
							Kuu(Index(ii),Index(jj)) += kelem(i,j);
							dMassdx(Index(ii),Index(jj)) += melem(i,j);
						}

					}
			}
		//df0dx(xiter)=0.00;
		Vector helper1(UNum);
		Vector helper2(UNum);
		#ifdef DynDamping
			Cuu=Kuu*CoefDK+dMassdx*CoefDM;
		#endif
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]-pow(-1,jiter+1)*DuTotal[0],helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			#ifndef DynEarthQuake
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#else
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]+du2dotG*DynLoad(jiter+1)-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#endif
			df0dx(xiter) += prec_inner_prod(helper1,Landa[jiter]);
		}
		if (iter==0)														//dfdx is equal for this purpose
			dfdx(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
	}

	cout<<"sensitivity analysis is done"<<endl;
	delete[] Landa;
	return true;
}

bool DynamicOptimization::DynSensitvtAnalyseFTU(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	#ifdef Dynf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		KHatT=KuuT+(4.00/Dt/Dt)*MassT;
		#ifdef DynDamping
			KHatT+=2.00/Dt*CuuT;
		#endif
	#endif
	#ifdef Dynf0valUTSU
		CMatrix ST(UNum,UNum);
		ST=KuuT+sf0val*MassT;
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	void *F=NULL;
	df0dx.clear();
	Taucs_Factor_Solve(KHat,&F);
	cout<<"Building Landas"<<endl;
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		//Fu += 2.00*DuTotal[iteri];
		#ifdef Dynf0valUTKHatU
			Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
		#else
			#ifdef Dynf0valUTKU
				Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
			#else
				#ifdef Dynf0valUTMU
					Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					#ifdef Dynf0valUTSU
						Fu += 2.00*axpy_prod(ST,DuTotal[iteri],HelperUNum)*ImpFac;
					#else
						#ifdef Dynf0valUTKDU
							Fu += axpy_prod(KuuT,2.00*DuTotal[iteri]-DuTotal[iteri-1],HelperUNum)*ImpFac;
							if (iteri != TimeIterN-1)
								Fu += axpy_prod(KuuT,-DuTotal[iteri+1],HelperUNum)*ImpFac;
						#else
							Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
						#endif
					#endif
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	Taucs_FreeFactor(&F);
	cout<<"Landas are built"<<endl;

	double ro[IntegPoints.size1()];
	double Massro[IntegPoints.size1()];
	cout<<"Entering sensitivity analysis"<<endl;

#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		CMatrix dMassdx(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif

		int lhs,rhs,uhs,bhs;
		size_t iro=xiter%(MDLenpart+1);
		size_t jro=(xiter-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		size_t mLeft=(lhs+iro-1)*MDElemLenNum;
		size_t mRight=(2-rhs+iro-1)*MDElemLenNum;
		size_t nBottom=(bhs+jro-1)*MDElemWidNum;
		size_t nUp=(2-uhs+jro-1)*MDElemWidNum;

		for (size_t jelem=nBottom;jelem<nUp;jelem++)
			for (size_t ielem=mLeft;ielem<mRight;ielem++)
			{
				size_t elemnum = jelem*Lenpart+ielem;
				size_t ii,jj;
				size_t mprime=(ielem-ielem%MDElemLenNum)/MDElemLenNum;
				size_t nprime=(jelem-jelem%MDElemWidNum)/MDElemWidNum;
				roBuilder(elemnum,ro,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1),Massro); // Third arg. is to show the function which edge ro is being derived in a prime mesh desgine

				Element.kBuild(ro,kelem);
				//cout<<"massro= "<<Massro[0]<<" , "<<Massro[1]<<" , "<<Massro[2]<<" , "<<Massro[3]<<endl;
				Element.mbuild(Massro,melem);

				for (size_t i=0;i<8;i++)
					for (size_t j=0;j<8;j++)
					{
						ii = Connectivity(elemnum,i);
						jj = Connectivity(elemnum,j);

						if (Index(ii) < UNum && Index(jj) < UNum)
						{
							Kuu(Index(ii),Index(jj)) += kelem(i,j);
							dMassdx(Index(ii),Index(jj)) += melem(i,j);
						}

					}
			}
		//df0dx(xiter)=0.00;
		#ifdef DynDamping
			Cuu=Kuu*CoefDK+dMassdx*CoefDM;
		#endif
		Vector helper1(UNum);
		Vector helper2(UNum);
		#ifdef Dynf0valUTKHatU
			#ifndef DynDamping
				dKHatdx=Kuu+dMassdx*4.00/Dt/Dt;
			#else
				dKHatdx=Kuu+2.00/Dt*Cuu+dMassdx*4.00/Dt/Dt;
			#endif
		#endif
		#ifdef Dynf0valUTSU
			CMatrix dSTdx(UNum,UNum);
			dSTdx=Kuu+dMassdx*sf0val;
		#endif
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]-pow(-1,jiter+1)*DuTotal[0],helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			#ifndef DynEarthQuake
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#else
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]+du2dotG*DynLoad(jiter+1)-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#endif
			df0dx(xiter) += prec_inner_prod(helper1,Landa[jiter]);
			#ifdef Dynf0valUTKHatU
				df0dx(xiter) += prec_inner_prod(axpy_prod(dKHatdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
			#else
				#ifdef Dynf0valUTKU
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					#ifdef Dynf0valUTMU
						df0dx(xiter) += prec_inner_prod(axpy_prod(dMassdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
					#else
						#ifdef Dynf0valUTSU
							df0dx(xiter) += prec_inner_prod(axpy_prod(dSTdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
						#else
							#ifdef Dynf0valUTKDU
								df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1]-DuTotal[jiter])*ImpFac;
							#endif
						#endif
					#endif
				#endif
			#endif
		}
		if (iter==0)																			//dfdx is equal for this purpose
			dfdx(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
	}

	cout<<"sensitivity analysis is done"<<endl;
	delete[] Landa;
	return true;
}

bool DynamicOptimization::DynSensitvtAnalyseMPI(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	void *F=NULL;
	Vector df0dxMPI(n);
	Matrix dfdxMPI(m,n);

	df0dxMPI.clear();
	df0dx.clear();
	dfdxMPI.clear();

	Taucs_Factor_Solve(KHat,&F);
	if (MyMPIRank == 0)
		cout<<"Building Landas"<<endl;
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		Fu += 2.00*DuTotal[iteri]*ImpFac;
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	Taucs_FreeFactor(&F);
	if (MyMPIRank == 0)
		cout<<"Landas are built"<<endl;

	double ro[IntegPoints.size1()];
	double Massro[IntegPoints.size1()];
	if (MyMPIRank == 0)
		cout<<"Entering sensitivity analysis"<<endl;

//#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=MyMPIRank;xiter<n;xiter+=OurMPISize)
	{
		if (MyMPIRank == 0)
			printf ("%5.2f %%",xiter/(n-1.00)*100.00);
		Matrix kelem(8,8);
		Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		CMatrix dMassdx(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif

		int lhs,rhs,uhs,bhs;
		size_t iro=xiter%(MDLenpart+1);
		size_t jro=(xiter-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		size_t mLeft=(lhs+iro-1)*MDElemLenNum;
		size_t mRight=(2-rhs+iro-1)*MDElemLenNum;
		size_t nBottom=(bhs+jro-1)*MDElemWidNum;
		size_t nUp=(2-uhs+jro-1)*MDElemWidNum;

		for (size_t jelem=nBottom;jelem<nUp;jelem++)
			for (size_t ielem=mLeft;ielem<mRight;ielem++)
			{
				size_t elemnum = jelem*Lenpart+ielem;
				size_t ii,jj;
				size_t mprime=(ielem-ielem%MDElemLenNum)/MDElemLenNum;
				size_t nprime=(jelem-jelem%MDElemWidNum)/MDElemWidNum;
				roBuilder(elemnum,ro,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1),Massro); // Third arg. is to show the function which edge ro is being derived in a prime mesh desgine

				Element.kBuild(ro,kelem);
				//cout<<"massro= "<<Massro[0]<<" , "<<Massro[1]<<" , "<<Massro[2]<<" , "<<Massro[3]<<endl;
				Element.mbuild(Massro,melem);

				for (size_t i=0;i<8;i++)
					for (size_t j=0;j<8;j++)
					{
						ii = Connectivity(elemnum,i);
						jj = Connectivity(elemnum,j);

						if (Index(ii) < UNum && Index(jj) < UNum)
						{
							Kuu(Index(ii),Index(jj)) += kelem(i,j);
							dMassdx(Index(ii),Index(jj)) += melem(i,j);
						}

					}
			}
		//df0dx(xiter)=0.00;
		Vector helper1(UNum);
		Vector helper2(UNum);
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]-pow(-1,jiter+1)*DuTotal[0],helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			#ifndef DynEarthQuake
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#else
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]+du2dotG*DynLoad(jiter+1)-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#endif
			df0dxMPI(xiter) += prec_inner_prod(helper1,Landa[jiter]);
		}
		if (iter==0)																			//dfdx is equal for this purpose
			dfdxMPI(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
		//printf ("I am num %d and my loop is %d\n",MyMPIRank,xiter);
		if (MyMPIRank == 0)
			printf ("\b\b\b\b\b\b\b");
	}
	//printf ("I am num: %d \n",MyMPIRank);
	//cout<<df0dxMPI<<endl;
	//MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(df0dxMPI.data().begin(),df0dx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if (iter==0)
		MPI_Allreduce(dfdxMPI.data().begin(),dfdx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	//cout<<"df0dx="<<df0dx<<endl;
	//MPI_Barrier(MPI_COMM_WORLD);

	if (MyMPIRank == 0)
		cout<<"\bsensitivity analysis is done"<<endl;
	delete[] Landa;
	return true;
}

bool DynamicOptimization::DynSensitvtAnalyseFTUMPI(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	#ifdef Dynf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		KHatT=KuuT+(4.00/Dt/Dt)*MassT;
		#ifdef DynDamping
			KHatT+=2.00/Dt*CuuT;
		#endif
	#endif
	#ifdef Dynf0valUTSU
		CMatrix ST(UNum,UNum);
		ST=KuuT+sf0val*MassT;
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	void *F=NULL;
	Vector df0dxMPI(n);
	Matrix dfdxMPI(m,n);

	df0dxMPI.clear();
	dfdxMPI.clear();

	Taucs_Factor_Solve(KHat,&F);
	if (MyMPIRank == 0)
		cout<<"Building Landas"<<endl;
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		//Fu += 2.00*DuTotal[iteri];
		#ifdef Dynf0valUTKHatU
			Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
		#else
			#ifdef Dynf0valUTKU
				Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
			#else
				#ifdef Dynf0valUTMU
					Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					#ifdef Dynf0valUTSU
						Fu += 2.00*axpy_prod(ST,DuTotal[iteri],HelperUNum)*ImpFac;
					#else
						#ifdef Dynf0valUTKDU
							Fu += axpy_prod(KuuT,2.00*DuTotal[iteri]-DuTotal[iteri-1],HelperUNum)*ImpFac;
							if (iteri != TimeIterN-1)
								Fu += axpy_prod(KuuT,-DuTotal[iteri+1],HelperUNum)*ImpFac;

						#else
							Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
						#endif
					#endif
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	Taucs_FreeFactor(&F);
	if (MyMPIRank == 0)
		cout<<"Landas are built"<<endl;

	double ro[IntegPoints.size1()];
	double Massro[IntegPoints.size1()];
	if (MyMPIRank == 0)
		cout<<"Entering sensitivity analysis"<<endl;

//#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=MyMPIRank;xiter<n;xiter+=OurMPISize)
	{
		if (MyMPIRank == 0)
			printf ("%5.2f %%",xiter/(n-1.00)*100.00);
		Matrix kelem(8,8);
		Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		CMatrix dMassdx(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif

		int lhs,rhs,uhs,bhs;
		size_t iro=xiter%(MDLenpart+1);
		size_t jro=(xiter-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		size_t mLeft=(lhs+iro-1)*MDElemLenNum;
		size_t mRight=(2-rhs+iro-1)*MDElemLenNum;
		size_t nUp=(2-uhs+jro-1)*MDElemWidNum;

		for (size_t jelem=nBottom;jelem<nUp;jelem++)
			for (size_t ielem=mLeft;ielem<mRight;ielem++)
			{
				size_t elemnum = jelem*Lenpart+ielem;
				size_t ii,jj;
				size_t mprime=(ielem-ielem%MDElemLenNum)/MDElemLenNum;
				size_t nprime=(jelem-jelem%MDElemWidNum)/MDElemWidNum;
				roBuilder(elemnum,ro,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1),Massro); // Third arg. is to show the function which edge ro is being derived in a prime mesh desgine

				Element.kBuild(ro,kelem);
				//cout<<"massro= "<<Massro[0]<<" , "<<Massro[1]<<" , "<<Massro[2]<<" , "<<Massro[3]<<endl;
				Element.mbuild(Massro,melem);

				for (size_t i=0;i<8;i++)
					for (size_t j=0;j<8;j++)
					{
						ii = Connectivity(elemnum,i);
						jj = Connectivity(elemnum,j);

						if (Index(ii) < UNum && Index(jj) < UNum)
						{
							Kuu(Index(ii),Index(jj)) += kelem(i,j);
							dMassdx(Index(ii),Index(jj)) += melem(i,j);
						}

					}
			}
		//df0dx(xiter)=0.00;
		#ifdef DynDamping
			Cuu=Kuu*CoefDK+dMassdx*CoefDM;
		#endif
		Vector helper1(UNum);
		Vector helper2(UNum);
		#ifdef Dynf0valUTKHatU
			CMatrix dKHatdx(UNum,UNum);
			#ifndef DynDamping
				dKHatdx=Kuu+dMassdx*4.00/Dt/Dt;
			#else
				dKHatdx=Kuu+2.00/Dt*Cuu+dMassdx*4.00/Dt/Dt;
			#endif
		#endif
		#ifdef Dynf0valUTSU
			CMatrix dSTdx(UNum,UNum);
			dSTdx=Kuu+dMassdx*sf0val;
		#endif
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]-pow(-1,jiter+1)*DuTotal[0],helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			#ifndef DynEarthQuake
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#else
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]+du2dotG*DynLoad(jiter+1)-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#endif
			df0dxMPI(xiter) += prec_inner_prod(helper1,Landa[jiter]);
			#ifdef Dynf0valUTKHatU
				df0dxMPI(xiter) += prec_inner_prod(axpy_prod(dKHatdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
			#else
				#ifdef Dynf0valUTKU
					df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					#ifdef Dynf0valUTMU
						df0dxMPI(xiter) += prec_inner_prod(axpy_prod(dMassdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
					#else
						#ifdef Dynf0valUTSU
							df0dxMPI(xiter) += prec_inner_prod(axpy_prod(dSTdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
						#else
							#ifdef Dynf0valUTKDU
								df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1]-DuTotal[jiter])*ImpFac;
							#endif
						#endif
					#endif
				#endif
			#endif
		}
		if (iter==0)																			//dfdx is equal for this purpose
			dfdxMPI(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
		//printf ("I am num %d and my loop is %d\n",MyMPIRank,xiter);
		if (MyMPIRank == 0)
			printf ("\b\b\b\b\b\b\b");
	}
	//printf ("I am num: %d \n",MyMPIRank);
	//cout<<df0dxMPI<<endl;
	//MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(df0dxMPI.data().begin(),df0dx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if (iter==0)
		MPI_Allreduce(dfdxMPI.data().begin(),dfdx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	//cout<<"df0dx="<<df0dx<<endl;
	//MPI_Barrier(MPI_COMM_WORLD);

	if (MyMPIRank == 0)
		cout<<"\bsensitivity analysis is done"<<endl;
	delete[] Landa;
	return true;
}

bool DynamicOptimization::DynOptimize()
{
	bool Resume=true;
	iter =0;
	Vector helperm(m);
	helperm(0)=1.00;

	Vector* DuTotal=new Vector[TimeIterN];
	Vector* DudotTotal=new Vector[TimeIterN];
	Vector* Du2dotTotal=new Vector[TimeIterN];

	ofstream DynOutOpt("DynamicOpt/DynOptOut.txt");
#ifdef DynDf0dxDebug
	ofstream DynOutdf("DynamicOpt/DyndfOut.txt");
#endif

#ifdef DynMassDebug
	ofstream DynOutMass("DynamicOpt/DynMassOut.txt");
#endif

#ifdef DynFDdf0dx
	ofstream DynOutFD("DynamicOpt/DynFDdfOut.txt");
#endif

#ifdef DynRoPrint
	ofstream DynOutXval("DynamicOpt/DynXvalOut.txt");
#endif

#ifdef DynDispPrint
	ofstream DynOutDisps("DynamicOpt/DynDispsOut.txt");
#endif

	for (size_t i=0;i<TimeIterN;i++)
	{
		DuTotal[i].resize(UNum);
		#ifdef DynDamping
			DudotTotal[i].resize(UNum);
		#endif
		Du2dotTotal[i].resize(UNum);
	}
	#ifdef TraceFollow
		FollowTrace();
	#endif
	do
	{
		if ((MDLenpart == Lenpart) && (MDWidpart == Widpart))
			Resume=false;
		while (/*(iter<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
#ifdef DynMTMakestiffness
			MakeStiffnessMT();
#else
			MakeStiffness();
#endif

			BuildMassMatrix();
#ifdef DynMassDebug
			DynOutMass<<MassT<<endl;
			DynOutMass<<KuuT<<endl;
#endif

			f0valOld=f0val;
			DynAnalyse(DuTotal,DudotTotal,Du2dotTotal);
			CalResidue();
#ifdef MPISensitivityAnalisys
	#ifndef Dynf0valFTU
			DynSensitvtAnalyseMPI(DuTotal,DudotTotal,Du2dotTotal);
	#else
			DynSensitvtAnalyseFTUMPI(DuTotal,DudotTotal,Du2dotTotal);
	#endif
#else
	#ifdef Dynf0valFTU
			DynSensitvtAnalyseFTU(DuTotal,DudotTotal,Du2dotTotal);
	#else
			DynSensitvtAnalyse(DuTotal,DudotTotal,Du2dotTotal);
	#endif
#endif

#ifdef DynDf0dxDebug
			DynOutdf<<"iter num"<<iter<<endl;
			DynOutdf<<"MD="<<MDLenpart<<" , "<<MDWidpart<<endl;
			DynOutdf<<"f0val , fval= "<<f0val<<" , "<<fval<<endl;
			DynOutdf<<"df0dx= "<<df0dx<<endl;
#endif

#ifdef DynDispPrint
			//if (iter == maxite-1)
			//{
			DynOutDisps<<"iter= "<<iter<<endl<<endl;
				for (size_t jiter=0;jiter<TimeIterN;jiter++)
				{
					//for (size_t iDisp=0;iDisp<UNum;iDisp++)
						//Disps(IndexRev(iDisp))=DuTotal[jiter](iDisp);
					//DynOutDisps<<"iter= "<<iter<<endl<<", Disps "<<jiter<<" = "<<Disps<<endl;
					DynOutDisps<<DuTotal[jiter](IndexRev(LEP))<<" , ";
				}
				DynOutDisps<<endl<<endl;
			//}
#endif

#ifdef DynFDdf0dx
			//if(iter==2)
			{
				DynOutFD<<"iter= "<<iter<<endl;
				double f0val2=f0val;
				for (size_t FDiter=0;FDiter<n;FDiter++)
				{
					//cout << (DynOutFD << "TEST" << endl);
					Ro(FDiter) += 1e-6;
					MakeStiffness();
					BuildMassMatrix();
					DynAnalyse(DuTotal,DudotTotal,Du2dotTotal);
					DynOutFD<<"i= "<<FDiter<<" : "<<(df0dx(FDiter)) << " , ";
					df0dx(FDiter)=(f0val-f0val2)/1e-6;
					DynOutFD << (df0dx(FDiter)) << endl;
					//DynOutFD.flush();
					//cout<<df0dx(FDiter)<<" , ";
					Ro(FDiter) -= 1e-6;
				}
				DynOutFD<<endl<<"Md= "<<MDLenpart<<" , "<<MDWidpart<<endl;
				DynOutFD<<"________________________________________________"<<endl;
			}
#endif

			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
			if (/*(iter+1<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				mmasub();
				//cout<<"Ro= "<<Ro<<endl;
			#ifdef CutXTails
				CutXTail();
			#endif
			#ifdef DynRoPrint
				DynOutXval<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
				for (size_t roiter=0;roiter<n;roiter++)
				{
					DynOutXval<<xval(roiter);
					if (roiter<n-1)
						DynOutXval<<",";
				}
				DynOutXval<<"};"<<endl;

				//DynOutXval<<"iter= "<<iter<<" Md="<<MDLenpart<<" , "<<MDWidpart<<" f0val ,fval= "<<f0val<<" , "<<fval<<endl<<MDLenpart<<" , "<<MDWidpart<<endl<<xval<<endl<<endl;
			#endif

				Ro=xval;
			}
			iter++;
			if (MyMPIRank == 0)
			{
				cout<<"f0val= "<<f0val<<" ,fval= "<<fval<<endl;
				cout<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
				cout<<MDLenpart<<" , "<<MDWidpart<<endl;
			}
			DynOutOpt<<MDLenpart<<" , "<<MDWidpart<<endl;
			DynOutOpt<<"f0val= "<<f0val<<" ,fval= "<<fval<<endl;
			DynOutOpt<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
			DynOutOpt<<"Number of trys in mma: "<<counter<<endl;
			DynOutOpt<<"Iteration number "<<iter<<" is done"<<endl<<"_________________________________________"<<endl;
			if (MyMPIRank == 0)
				printf("Iteration number %ld is done\n_____________________________________________________________\n",iter);
		}
		X2NewX();
	}
	while (Resume);
	delete[] DuTotal;
	delete[] DudotTotal;
	delete[] Du2dotTotal;
	return true;
}

void DynamicOptimization::DynInSensitvtAnalyz(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);
	df0dx.clear();
	void *F=NULL;
	Taucs_Factor_Solve(KHat,&F);
	cout<<"Building Landas Inner opt."<<endl;
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		Fu += 2.00*DuTotal[iteri]*ImpFac;
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	/*for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)
	{
		Du.clear();
		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		axpy_prod(MassT,Du,Fu);
		Fu += 2.00*DuTotal[iteri];
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
	}*/
	Taucs_FreeFactor(&F);
	cout<<"Landas are built, Inner Optimization"<<endl;
	double ro[IntegPoints.size1()];
	double Massro[IntegPoints.size1()];
#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		CMatrix dMassdx(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif

		int lhs,rhs,uhs,bhs;
		size_t iro=InnerMapIndexRev(xiter)%(MDLenpart+1);
		size_t jro=(InnerMapIndexRev(xiter)-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		for (size_t nprime=bhs+jro-1;nprime<2-uhs+jro-1;nprime++)
			for (size_t mprime=lhs+iro-1;mprime<2-rhs+iro-1;mprime++)
				for (size_t iii=0;iii<EIPI[mprime+nprime*MDLenpart].size()/4;iii++)
				{
					size_t ii,jj;
					InSensitvtroBuilder(ro,mprime,nprime,iii,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1),EIPI,Massro); //5th argument shows the number of corner ro to be driven
					Element.kBuild(ro,kelem);
					Element.mbuild(Massro,melem);
					//cout<<Element.k<<endl;
					size_t elemnum=size_t(EIPI[mprime+nprime*MDLenpart][4*iii+1]);

					for (size_t i=0;i<8;i++)
						for (size_t j=0;j<8;j++)
						{

							ii = Connectivity(elemnum,i);
							jj = Connectivity(elemnum,j);

							if (Index(ii) < UNum && Index(jj) < UNum)
							{
								Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);
								dMassdx(Index(ii),Index(jj)) += melem(i,j);
							}
						}
				}
		//df0dx(xiter)=0.00;
		Vector helper1(UNum);
		Vector helper2(UNum);
		#ifdef DynDamping
			Cuu=Kuu*CoefDK+dMassdx*CoefDM;
		#endif
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]-pow(-1,jiter+1)*DuTotal[0],helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			#ifndef DynEarthQuake
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#else
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]+du2dotG*DynLoad(jiter+1)-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#endif
			df0dx(xiter) += prec_inner_prod(helper1,Landa[jiter]);
		}
		if (iter==0)																			//dfdx is equal for this purpose
			dfdx(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
	}
	cout<<"Inner sensitivity analysis is done"<<endl;
	//Taucs_FreeFactor(&F);
	delete[] Landa;
}

void DynamicOptimization::DynInSensitvtAnalyzFTU(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	#ifdef Dynf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		KHatT=KuuT+(4.00/Dt/Dt)*MassT;
		#ifdef DynDamping
			KHatT+=2.00/Dt*CuuT;
		#endif
	#endif
	#ifdef Dynf0valUTSU
		CMatrix ST(UNum,UNum);
		ST=KuuT+sf0val*MassT;
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);
	df0dx.clear();
	void *F=NULL;
	Taucs_Factor_Solve(KHat,&F);
	cout<<"Building Landas Inner opt."<<endl;
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		//Fu += 2.00*DuTotal[iteri];
		#ifdef Dynf0valUTKHatU
			Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
		#else
			#ifdef Dynf0valUTKU
				Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
			#else
				#ifdef Dynf0valUTMU
					Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					#ifdef Dynf0valUTSU
						Fu += 2.00*axpy_prod(ST,DuTotal[iteri],HelperUNum)*ImpFac;
					#else
						#ifdef Dynf0valUTKDU
							Fu += axpy_prod(KuuT,2.00*DuTotal[iteri]-DuTotal[iteri-1],HelperUNum)*ImpFac;
							if (iteri != TimeIterN-1)
								Fu += axpy_prod(KuuT,-DuTotal[iteri+1],HelperUNum)*ImpFac;
						#else
							Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
						#endif
					#endif
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	/*for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)
	{
		Du.clear();
		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		axpy_prod(MassT,Du,Fu);
		Fu += 2.00*DuTotal[iteri];
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
	}*/
	Taucs_FreeFactor(&F);
	cout<<"Landas are built, Inner Optimization"<<endl;
	double ro[IntegPoints.size1()];
	double Massro[IntegPoints.size1()];
#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		CMatrix dMassdx(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif

		int lhs,rhs,uhs,bhs;
		size_t iro=InnerMapIndexRev(xiter)%(MDLenpart+1);
		size_t jro=(InnerMapIndexRev(xiter)-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		for (size_t nprime=bhs+jro-1;nprime<2-uhs+jro-1;nprime++)
			for (size_t mprime=lhs+iro-1;mprime<2-rhs+iro-1;mprime++)
				for (size_t iii=0;iii<EIPI[mprime+nprime*MDLenpart].size()/4;iii++)
				{
					size_t ii,jj;
					InSensitvtroBuilder(ro,mprime,nprime,iii,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1),EIPI,Massro); //5th argument shows the number of corner ro to be driven
					Element.kBuild(ro,kelem);
					Element.mbuild(Massro,melem);
					//cout<<Element.k<<endl;
					size_t elemnum=size_t(EIPI[mprime+nprime*MDLenpart][4*iii+1]);

					for (size_t i=0;i<8;i++)
						for (size_t j=0;j<8;j++)
						{

							ii = Connectivity(elemnum,i);
							jj = Connectivity(elemnum,j);

							if (Index(ii) < UNum && Index(jj) < UNum)
							{
								Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);
								dMassdx(Index(ii),Index(jj)) += melem(i,j);
							}
						}
				}
		//df0dx(xiter)=0.00;
		#ifdef DynDamping
			Cuu=Kuu*CoefDK+dMassdx*CoefDM;
		#endif
		Vector helper1(UNum);
		Vector helper2(UNum);
		#ifdef Dynf0valUTKHatU
			CMatrix dKHatdx(UNum,UNum);
			#ifndef DynDamping
				dKHatdx=Kuu+dMassdx*4.00/Dt/Dt;
			#else
				dKHatdx=Kuu+2.00/Dt*Cuu+dMassdx*4.00/Dt/Dt;
			#endif
		#endif
		#ifdef Dynf0valUTSU
			CMatrix dSTdx(UNum,UNum);
			dSTdx=Kuu+dMassdx*sf0val;
		#endif
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]-pow(-1,jiter+1)*DuTotal[0],helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			#ifndef DynEarthQuake
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#else
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]+du2dotG*DynLoad(jiter+1)-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#endif
			df0dx(xiter) += prec_inner_prod(helper1,Landa[jiter]);
			#ifdef Dynf0valUTKHatU
				df0dx(xiter) += prec_inner_prod(axpy_prod(dKHatdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
			#else
				#ifdef Dynf0valUTKU
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					#ifdef Dynf0valUTMU
						df0dx(xiter) += prec_inner_prod(axpy_prod(dMassdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
					#else
						#ifdef Dynf0valUTSU
							df0dx(xiter) += prec_inner_prod(axpy_prod(dSTdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
						#else
							#ifdef Dynf0valUTKDU
								df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1]-DuTotal[jiter])*ImpFac;
							#endif
						#endif
					#endif
				#endif
			#endif
		}
		if (iter==0)																			//dfdx is equal for this purpose
			dfdx(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
	}
	cout<<"Inner sensitivity analysis is done"<<endl;
	//Taucs_FreeFactor(&F);
	delete[] Landa;
}

void DynamicOptimization::DynInSensitvtAnalyzMPI(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal)
{
	if (MyMPIRank == 0)
		cout<<"Entering MPI sensitivity analysis"<<endl;
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	Vector df0dxMPI(n);
	Matrix dfdxMPI(m,n);

	df0dxMPI.clear();
	df0dx.clear();
	dfdxMPI.clear();

	void *F=NULL;
	Taucs_Factor_Solve(KHat,&F);
	if (MyMPIRank == 0)
		cout<<"Building Landas Inner opt."<<endl;
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		Fu += 2.00*DuTotal[iteri]*ImpFac;
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	/*for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)
	{
		Du.clear();
		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		axpy_prod(MassT,Du,Fu);
		Fu += 2.00*DuTotal[iteri];
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
	}*/
	Taucs_FreeFactor(&F);
	if (MyMPIRank == 0)
		cout<<"Landas are built, Inner Optimization"<<endl;
	double ro[IntegPoints.size1()];
	double Massro[IntegPoints.size1()];
//#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=MyMPIRank;xiter<n;xiter +=OurMPISize)
	{
		if (MyMPIRank == 0)
			printf ("%5.2f %%",xiter/(n-1.00)*100.00);
		Matrix kelem(8,8);
		Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		CMatrix dMassdx(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif

		int lhs,rhs,uhs,bhs;
		size_t iro=InnerMapIndexRev(xiter)%(MDLenpart+1);
		size_t jro=(InnerMapIndexRev(xiter)-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		for (size_t nprime=bhs+jro-1;nprime<2-uhs+jro-1;nprime++)
			for (size_t mprime=lhs+iro-1;mprime<2-rhs+iro-1;mprime++)
				for (size_t iii=0;iii<EIPI[mprime+nprime*MDLenpart].size()/4;iii++)
				{
					size_t ii,jj;
					InSensitvtroBuilder(ro,mprime,nprime,iii,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1),EIPI,Massro); //5th argument shows the number of corner ro to be driven
					Element.kBuild(ro,kelem);
					Element.mbuild(Massro,melem);
					//cout<<Element.k<<endl;
					size_t elemnum=size_t(EIPI[mprime+nprime*MDLenpart][4*iii+1]);

					for (size_t i=0;i<8;i++)
						for (size_t j=0;j<8;j++)
						{

							ii = Connectivity(elemnum,i);
							jj = Connectivity(elemnum,j);

							if (Index(ii) < UNum && Index(jj) < UNum)
							{
								Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);
								dMassdx(Index(ii),Index(jj)) += melem(i,j);
							}
						}
				}
		//df0dx(xiter)=0.00;
		Vector helper1(UNum);
		Vector helper2(UNum);
		#ifdef DynDamping
			Cuu=Kuu*CoefDK+dMassdx*CoefDM;
		#endif
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]-pow(-1,jiter+1)*DuTotal[0],helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			#ifndef DynEarthQuake
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#else
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]+du2dotG*DynLoad(jiter+1)-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#endif
			df0dxMPI(xiter) += prec_inner_prod(helper1,Landa[jiter]);
		}
		if (iter==0)																			//dfdx is equal for this purpose
			dfdxMPI(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
		if (MyMPIRank == 0)
			printf ("\b\b\b\b\b\b\b");
	}
	MPI_Allreduce(df0dxMPI.data().begin(),df0dx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if (iter==0)
		MPI_Allreduce(dfdxMPI.data().begin(),dfdx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if (MyMPIRank == 0)
		cout<<"\bInner sensitivity analysis is done"<<endl;
	//Taucs_FreeFactor(&F);
	delete[] Landa;
}

void DynamicOptimization::DynInSensitvtAnalyzFTUMPI(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal)
{
	if (MyMPIRank == 0)
		cout<<"Entering MPI sensitivity analysis"<<endl;
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	#ifdef Dynf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		KHatT=KuuT+(4.00/Dt/Dt)*MassT;
		#ifdef DynDamping
			KHatT+=2.00/Dt*CuuT;
		#endif
	#endif
	#ifdef Dynf0valUTSU
		CMatrix ST(UNum,UNum);
		ST=KuuT+sf0val*MassT;
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	Vector df0dxMPI(n);
	Matrix dfdxMPI(m,n);

	df0dxMPI.clear();
	df0dx.clear();
	dfdxMPI.clear();

	void *F=NULL;
	Taucs_Factor_Solve(KHat,&F);
	if (MyMPIRank == 0)
		cout<<"Building Landas Inner opt."<<endl;
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		//Fu += 2.00*DuTotal[iteri];
		#ifdef Dynf0valUTKHatU
			Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
		#else
			#ifdef Dynf0valUTKU
				Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
			#else
				#ifdef Dynf0valUTMU
					Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					#ifdef Dynf0valUTSU
						Fu += 2.00*axpy_prod(ST,DuTotal[iteri],HelperUNum)*ImpFac;
					#else
						#ifdef Dynf0valUTKDU
							Fu += axpy_prod(KuuT,2.00*DuTotal[iteri]-DuTotal[iteri-1],HelperUNum)*ImpFac;
							if (iteri != TimeIterN-1)
								Fu += axpy_prod(KuuT,-DuTotal[iteri+1],HelperUNum)*ImpFac;
						#else
							Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
						#endif
					#endif
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	/*for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)
	{
		Du.clear();
		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		axpy_prod(MassT,Du,Fu);
		Fu += 2.00*DuTotal[iteri];
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
	}*/
	Taucs_FreeFactor(&F);
	if (MyMPIRank == 0)
		cout<<"Landas are built, Inner Optimization"<<endl;
	double ro[IntegPoints.size1()];
	double Massro[IntegPoints.size1()];
//#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=MyMPIRank;xiter<n;xiter +=OurMPISize)
	{
		if (MyMPIRank == 0)
			printf ("%5.2f %%",xiter/(n-1.00)*100.00);
		Matrix kelem(8,8);
		Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		CMatrix dMassdx(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif

		int lhs,rhs,uhs,bhs;
		size_t iro=InnerMapIndexRev(xiter)%(MDLenpart+1);
		size_t jro=(InnerMapIndexRev(xiter)-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		for (size_t nprime=bhs+jro-1;nprime<2-uhs+jro-1;nprime++)
			for (size_t mprime=lhs+iro-1;mprime<2-rhs+iro-1;mprime++)
				for (size_t iii=0;iii<EIPI[mprime+nprime*MDLenpart].size()/4;iii++)
				{
					size_t ii,jj;
					InSensitvtroBuilder(ro,mprime,nprime,iii,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1),EIPI,Massro); //5th argument shows the number of corner ro to be driven
					Element.kBuild(ro,kelem);
					Element.mbuild(Massro,melem);
					//cout<<Element.k<<endl;
					size_t elemnum=size_t(EIPI[mprime+nprime*MDLenpart][4*iii+1]);

					for (size_t i=0;i<8;i++)
						for (size_t j=0;j<8;j++)
						{

							ii = Connectivity(elemnum,i);
							jj = Connectivity(elemnum,j);

							if (Index(ii) < UNum && Index(jj) < UNum)
							{
								Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);
								dMassdx(Index(ii),Index(jj)) += melem(i,j);
							}
						}
				}
		//df0dx(xiter)=0.00;
		#ifdef DynDamping
			Cuu=Kuu*CoefDK+dMassdx*CoefDM;
		#endif
		Vector helper1(UNum);
		Vector helper2(UNum);
		#ifdef Dynf0valUTKHatU
			CMatrix dKHatdx(UNum,UNum);
			#ifndef DynDamping
				dKHatdx=Kuu+dMassdx*4.00/Dt/Dt;
			#else
				dKHatdx=Kuu+2.00/Dt*Cuu+dMassdx*4.00/Dt/Dt;
			#endif
		#endif
		#ifdef Dynf0valUTSU
			CMatrix dSTdx(UNum,UNum);
			dSTdx=Kuu+dMassdx*sf0val;
		#endif
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]-pow(-1,jiter+1)*DuTotal[0],helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			#ifndef DynEarthQuake
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#else
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]+du2dotG*DynLoad(jiter+1)-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#endif
			df0dxMPI(xiter) += prec_inner_prod(helper1,Landa[jiter]);
			#ifdef Dynf0valUTKHatU
				df0dxMPI(xiter) += prec_inner_prod(axpy_prod(dKHatdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
			#else
				#ifdef Dynf0valUTKU
					df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					#ifdef Dynf0valUTMU
						df0dxMPI(xiter) += prec_inner_prod(axpy_prod(dMassdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
					#else
						#ifdef Dynf0valUTSU
							df0dxMPI(xiter) += prec_inner_prod(axpy_prod(dSTdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
						#else
							#ifdef Dynf0valUTKDU
								df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1]-DuTotal[jiter])*ImpFac;
							#endif
						#endif
					#endif
				#endif
			#endif
		}
		if (iter==0)																			//dfdx is equal for this purpose
			dfdxMPI(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
		if (MyMPIRank == 0)
			printf ("\b\b\b\b\b\b\b");
	}
	MPI_Allreduce(df0dxMPI.data().begin(),df0dx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if (iter==0)
		MPI_Allreduce(dfdxMPI.data().begin(),dfdx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if (MyMPIRank == 0)
		cout<<"\bInner sensitivity analysis is done"<<endl;
	//Taucs_FreeFactor(&F);
	delete[] Landa;
}

bool DynamicOptimization::DynInnerOptimize()
{
	/*///////////////////////////////////////////////////////
	// OptTyp specifies the type of optimization		   //
	// OptTyp=0		:Ordinary Inner Optimization		   //
	// OptTyp=1		:Ordinary Opt. or node based filter op.// ?????????????????
	// OptTyp=2		:Element based filtering Optimization  //
	///////////////////////////////////////////////////////*/

	int OptTyp=0;													// Shows Kind of optimization is inner
	bool Resume=true;
	iter=0;

	Vector* DuTotal=new Vector[TimeIterN];
	Vector* DudotTotal=new Vector[TimeIterN];
	Vector* Du2dotTotal=new Vector[TimeIterN];

	for (size_t i=0;i<TimeIterN;i++)
	{
		DuTotal[i].resize(UNum);
		#ifdef DynDamping
			DudotTotal[i].resize(UNum);
		#endif
		Du2dotTotal[i].resize(UNum);
	}
	Vector helperm(m);
	helperm(0)=1.00;

#ifdef FreezeInnerSolidX
	Matrix EfArea(m,Ro.size());
#endif

	ofstream DynOutInnerOpt("DynamicOpt/DynInnerOptOut.txt");

#ifdef DynInnerFDdf0dx
	ofstream DynOutInnerFD("DynamicOpt/DynInnerFDdfOut.txt");
#endif

#ifdef DynInnerRoPrint
	ofstream DynOutInnerXval("DynamicOpt/DynInnerXvalOut.txt");
#endif

#ifdef DynInnerDispPrint
	ofstream DynOutInnerDisps("DynamicOpt/DynInnerDispsOut.txt");
#endif
	do
	{
		FTEIC=true;														// first time of each inner convergence
		vector<double> EIPI[MDLenpart*MDWidpart];						//M.D. Element Integration Points Information matrix
		while (/*(iter<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
			MakeStiffness(OptTyp,EIPI);
			FTEIC=false;
			BuildMassMatrix(OptTyp,EIPI);

			f0valOld=f0val;
			DynAnalyse(DuTotal,DudotTotal,Du2dotTotal);
			CalResidue();
#ifdef MPIInnerSensitivityAnalisys
	#ifndef Dynf0valFTU
			DynInSensitvtAnalyzMPI(EIPI,DuTotal,DudotTotal,Du2dotTotal);
	#else
			DynInSensitvtAnalyzFTUMPI(EIPI,DuTotal,DudotTotal,Du2dotTotal);
	#endif
#else
	#ifndef Dynf0valFTU
			DynInSensitvtAnalyz(EIPI,DuTotal,DudotTotal,Du2dotTotal);
	#else
			DynInSensitvtAnalyzFTU(EIPI,DuTotal,DudotTotal,Du2dotTotal);
	#endif
#endif

#ifdef DynInnerDispPrint

			if (iter == maxite-1)
			{
				for (size_t jiter=0;jiter<TimeIterN;jiter++)
				{
					for (size_t iDisp=0;iDisp<UNum;iDisp++)
						Disps(IndexRev(iDisp))=DuTotal[jiter](iDisp);
					DynOutInnerDisps<<"iter= "<<iter<<endl<<"Disps "<<jiter<<" = "<<Disps<<endl;
				}
			}

#endif

#ifdef DynInnerFDdf0dx
			//if(iter==2)
			{
				DynOutInnerFD<<"iter= "<<iter<<endl;
				double f0val2=f0val;
				for (size_t FDiter=0;FDiter<n;FDiter++)
				{
					//cout << (DynOutFD << "TEST" << endl);
					Ro(InnerMapIndexRev(FDiter)) += 1.00e-6;
					MakeStiffness(OptTyp,EIPI);
					BuildMassMatrix(OptTyp,EIPI);
					DynAnalyse(DuTotal,DudotTotal,Du2dotTotal);
					DynOutInnerFD<<"i= "<<FDiter<<" : "<<(df0dx(FDiter))<< " , "<<(f0val-f0val2)/1.00e-6 << endl;
					//df0dx(FDiter)=(f0val-f0val2)/1.00e-9;
					//DynOutInnerFD << (df0dx(FDiter)) << endl;
					//DynOutFD.flush();
					//cout<<df0dx(FDiter)<<" , ";
					Ro(InnerMapIndexRev(FDiter)) -= 1.00e-6;
				}
				f0val=f0val2;
				DynOutInnerFD<<endl<<"Md= "<<MDLenpart<<" , "<<MDWidpart<<endl;
				DynOutInnerFD<<"________________________________________________"<<endl;
			}
#endif

#ifdef FreezeInnerSolidX

			if (iter==0)
			{
				for (size_t xiter=0; xiter<Ro.size(); xiter++)
				{
					int lhs,rhs,uhs,bhs;
					size_t iro=xiter%(MDLenpart+1);
					size_t jro=(xiter-iro)/(MDLenpart+1);

					lhs = (iro == 0) ? 1 : 0;
					rhs = (iro == MDLenpart) ? 1 : 0;
					bhs = (jro == 0) ? 1 : 0;
					uhs = (jro == MDWidpart) ? 1 : 0;
					EfArea(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
				}
			}
			axpy_prod(EfArea,Ro,fval);
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;

#else
			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
#endif
			if (/*(iter+1<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				mmasub();
				#ifdef CutXTails
					CutXTail();
				#endif
				#ifndef FreezeInnerSolidX
					Ro=xval;
				#else
					for (size_t i=0;i<InnerMapIndexRev.size();i++)
						Ro(InnerMapIndexRev(i))=xval(i);
				#endif
				#ifdef DynInnerRoPrint
					DynOutInnerXval<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
					for (size_t roiter=0;roiter<(MDLenpart+1)*(MDWidpart+1);roiter++)
					{
						DynOutInnerXval<<Ro(roiter);
						if (roiter<(MDLenpart+1)*(MDWidpart+1)-1)
							DynOutInnerXval<<",";
					}
					DynOutInnerXval<<"};"<<endl;

					//DynOutInnerXval<<"iter= "<<iter<<" Md="<<MDLenpart<<" , "<<MDWidpart<<" f0val ,fval= "<<f0val<<" , "<<fval<<endl<<MDLenpart<<" , "<<MDWidpart<<endl<<xval<<endl<<endl;
				#endif
			}
			iter++;

/*#ifdef DynInnerRoPrint
			DynOutInnerXval<<"iter= "<<iter<<" Md="<<MDLenpart<<" , "<<MDWidpart<<" f0val ,fval= "<<f0val<<" , "<<fval<<endl<<MDLenpart<<" , "<<MDWidpart<<endl<<xval<<endl<<endl;
#endif*/
			if (MyMPIRank == 0)
			{
				cout<<"f0val= "<<f0val<<" ,fval= "<<fval<<endl;
				cout<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
				cout<<MDLenpart<<" , "<<MDWidpart<<endl;
			}
			DynOutInnerOpt<<MDLenpart<<" , "<<MDWidpart<<endl;
			DynOutInnerOpt<<"f0val= "<<f0val<<" ,fval= "<<fval<<endl;
			DynOutInnerOpt<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
			DynOutInnerOpt<<"Number of trys in mma: "<<counter<<endl;
			DynOutInnerOpt<<"Inner Iteration number "<<iter<<" is done"<<endl<<"_________________________________________"<<endl;
			if (MyMPIRank == 0)
				printf("Iteration number %ld is done\n_____________________________________________________________\n",iter);
		}
		Resume=X2NewX();
	}
	while (Resume);

	delete[] DuTotal;
	delete[] DudotTotal;
	delete[] Du2dotTotal;

	return true;
}

////// Filtering techniques //////////////////////////////////////////////////////////////////////////////////////
/** Yet other objective functions are not added to Element based filtering tech. **/
bool DynamicOptimization::DynFilterOptimize()
{
	Vector helperm(1);
	helperm(0)=1.00;
	Vector* DuTotal=new Vector[TimeIterN];
	Vector* DudotTotal=new Vector[TimeIterN];
	Vector* Du2dotTotal=new Vector[TimeIterN];

	for (size_t i=0;i<TimeIterN;i++)
	{
		DuTotal[i].resize(UNum);
		#ifdef DynDamping
			DudotTotal[i].resize(UNum);
		#endif
		Du2dotTotal[i].resize(UNum);
	}

	ifstream DynInFilter("DynamicOpt/DynFiltering/DynInFilter.txt");
	ofstream DynOutFilter("DynamicOpt/DynFiltering/DynOutFilter.txt");
#ifdef DynRoFilterPrint
	ofstream DynOutXFilter("DynamicOpt/DynFiltering/DynOutXFilter.txt");
#endif
#ifdef DyndfFilterDebug
	ofstream DynOutdfFilter("DynamicOpt/DynFiltering/DynOutdfFilter.txt");
#endif
#ifdef DynFDFiltering
	ofstream DynFDdfFilOut("DynamicOpt/DynFiltering/DynOutFDdfFilter.txt");
#endif
	double FilterRadius;
	while (DynInFilter>>FilterRadius)
	{
		iter=0;
		while (/*(iter<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
			if (MyMPIRank == 0)
				cout<<"Filter Radius="<< FilterRadius<<endl;
			DynOutFilter<<"Filter Radius="<< FilterRadius<<endl;
/*#ifdef DynRoFilterPrint
			DynOutXFilter<<"Filter Radius="<< FilterRadius<<endl;
#endif*/

			if (MyMPIRank == 0)
				cout<<"iter number"<<iter<<endl;
			DynOutFilter<<"iter number"<<iter<<endl;
/*#ifdef DynRoFilterPrint
			DynOutXFilter<<"iter number"<<iter<<endl;
#endif*/
			MakeStiffness();
			BuildMassMatrix();

			f0valOld=f0val;
			DynAnalyse(DuTotal,DudotTotal,Du2dotTotal);
			//DynSensitvtAnalyse(DuTotal,Du2dotTotal);
			#ifdef MPISensitivityAnalisys
				#ifndef Dynf0valFTU
						DynSensitvtAnalyseMPI(DuTotal,DudotTotal,Du2dotTotal);
				#else
						DynSensitvtAnalyseFTUMPI(DuTotal,DudotTotal,Du2dotTotal);
				#endif
			#else
				#ifdef Dynf0valFTU
						DynSensitvtAnalyseFTU(DuTotal,DudotTotal,Du2dotTotal);
				#else
						DynSensitvtAnalyse(DuTotal,DudotTotal,Du2dotTotal);
				#endif
			#endif
			CalResidue();
#ifdef DynFDFiltering
			double f0val2=f0val;
			for (size_t FD_DF=0;FD_DF<n;FD_DF++)
			{
				Ro(FD_DF) += 1e-9;
				MakeStiffness();
				BuildMassMatrix();
				DynAnalyse(DuTotal,DudotTotal,Du2dotTotal);
				DynFDdfFilOut<<"i = "<<FD_DF<<" : "<<df0dx(FD_DF)<<" , "<<(f0val-f0val2)/1e-9<<endl;
				Ro(FD_DF) -= 1e-9;
			}
			f0val=f0val2;
			DynFDdfFilOut<<"iter num "<<iter<<endl<<"_____________________________________"<<endl;
#endif

#ifdef DyndfFilterDebug
			DynOutdfFilter<<"before df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl;
#endif
			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
			FilterSenstvts(FilterRadius);
#ifdef DyndfFilterDebug
			DynOutdfFilter<<" after df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl<<endl<<endl;
#endif
			if (/*(iter+1<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				mmasub();
				#ifdef CutXTails
					CutXTail();
				#endif
				Ro=xval;
			}
			if (MyMPIRank == 0)
				cout<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			DynOutFilter<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			DynOutFilter<<"__________________________________________________________________"<<endl<<endl;
			if (MyMPIRank == 0)
				cout<<"__________________________________________________________________"<<endl<<endl;
			DynOutFilter<<"__________________________________________________________________"<<endl<<endl;
#ifdef DynRoFilterPrint
			DynOutXFilter<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
			for (size_t roiter=0;roiter<n;roiter++)
			{
				DynOutXFilter<<xval(roiter);
				if (roiter<n-1)
					DynOutXFilter<<",";
			}
			DynOutXFilter<<"};"<<endl;
#endif
			iter++;
		}
	}
	delete[] DuTotal;
	delete[] DudotTotal;
	delete[] Du2dotTotal;
	return true;
}

bool DynamicOptimization::DynElFSensitvtAnalyz(Vector *DuTotal/*,Vector *DudotTotal*/,Vector *Du2dotTotal)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef Dynf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		KHatT=KuuT+(4.00/Dt/Dt)*MassT;
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	void *F=NULL;
	df0dx.clear();
	Taucs_Factor_Solve(KHat,&F);
	cout<<"Building Landas"<<endl;
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		//Fu += 2.00*DuTotal[iteri];
		#ifdef Dynf0valUTKHatU
			Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
		#else
			#ifdef Dynf0valUTKU
				Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
			#else
				#ifdef Dynf0valUTMU
					Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	cout<<"Landas are built"<<endl;

	double ro[IntegPoints.size1()];
	double Massro[IntegPoints.size1()];
	cout<<"Entering filtering sensitivity analysis"<<endl;
#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		CMatrix dMassdx(UNum,UNum);

		Kuu.clear();
		dMassdx.clear();
		for (unsigned int roi=0;roi<IntegPoints.size1();roi++)
		{
			ro[roi]=SimpPower*pow(Ro(xiter),SimpPower-1.00);
			Massro[roi]=MassPower*pow(Ro(xiter),MassPower-1.00);
		}
		Element.kBuild(ro,kelem);
		Element.mbuild(Massro,melem);
		size_t ii,jj;
		//cout<<Element.k<<endl;
		for (size_t i=0;i<8;i++)
			for (size_t j=0;j<8;j++)
			{

				ii = Connectivity(xiter,i);
				jj = Connectivity(xiter,j);

				if (Index(ii) < UNum && Index(jj) < UNum)
				{
					Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);
					dMassdx(Index(ii),Index(jj)) += melem(i,j);
				}
			}
		Vector helper1(UNum);
		Vector helper2(UNum);
		#ifdef Dynf0valUTKHatU
			CMatrix dKHatdx(UNum,UNum);
			dKHatdx=Kuu+dMassdx*4.00/Dt/Dt;
		#endif
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			#ifndef DynEarthQuake
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#else
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]+du2dotG*DynLoad(jiter+1)-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#endif
			df0dx(xiter) += prec_inner_prod(helper1,Landa[jiter]);
			#ifdef Dynf0valUTKHatU
				df0dx(xiter) += prec_inner_prod(axpy_prod(dKHatdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
			#else
				#ifdef Dynf0valUTKU
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					#ifdef Dynf0valUTMU
						df0dx(xiter) += prec_inner_prod(axpy_prod(dMassdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
					#endif
				#endif
			#endif
		}
		if (iter==0)							//dfdx is equal for this purpose
			dfdx(0,xiter)=Elemlen*Elemwid; 		//for edge ro modifications
		//cout<<xiter<<endl;
	}
	delete[] Landa;
	Taucs_FreeFactor(&F);
	cout<<"Filtering Sensitivity analysis is done"<<endl;
	return true;

}

bool DynamicOptimization::DynElFSensitvtAnalyzMPI(Vector *DuTotal/*,Vector *DudotTotal*/,Vector *Du2dotTotal)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef Dynf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		KHatT=KuuT+(4.00/Dt/Dt)*MassT;
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	void *F=NULL;

	Vector df0dxMPI(n);
	Matrix dfdxMPI(m,n);

	df0dxMPI.clear();
	df0dx.clear();
	dfdxMPI.clear();

	Taucs_Factor_Solve(KHat,&F);
	if (MyMPIRank==0)
		cout<<"Building Landas"<<endl;
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		//Fu += 2.00*DuTotal[iteri];
		#ifdef Dynf0valUTKHatU
			Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
		#else
			#ifdef Dynf0valUTKU
				Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
			#else
				#ifdef Dynf0valUTMU
					Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	if (MyMPIRank==0)
		cout<<"Landas are built"<<endl;

	double ro[IntegPoints.size1()];
	double Massro[IntegPoints.size1()];
	if (MyMPIRank==0)
		cout<<"Entering filtering sensitivity analysis"<<endl;
//#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=MyMPIRank;xiter<n;xiter += OurMPISize)
	{
		if (MyMPIRank == 0)
			printf ("%5.2f %%",xiter/(n-1.00)*100.00);
		Matrix kelem(8,8);
		Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		CMatrix dMassdx(UNum,UNum);

		Kuu.clear();
		dMassdx.clear();
		for (unsigned int roi=0;roi<IntegPoints.size1();roi++)
		{
			ro[roi]=SimpPower*pow(Ro(xiter),SimpPower-1.00);
			Massro[roi]=MassPower*pow(Ro(xiter),MassPower-1.00);
		}
		Element.kBuild(ro,kelem);
		Element.mbuild(Massro,melem);
		size_t ii,jj;
		//cout<<Element.k<<endl;
		for (size_t i=0;i<8;i++)
			for (size_t j=0;j<8;j++)
			{

				ii = Connectivity(xiter,i);
				jj = Connectivity(xiter,j);

				if (Index(ii) < UNum && Index(jj) < UNum)
				{
					Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);
					dMassdx(Index(ii),Index(jj)) += melem(i,j);
				}
			}
		Vector helper1(UNum);
		Vector helper2(UNum);
		#ifdef Dynf0valUTKHatU
			CMatrix dKHatdx(UNum,UNum);
			dKHatdx=Kuu+dMassdx*4.00/Dt/Dt;
		#endif
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]-pow(-1,jiter+1)*DuTotal[0],helper1);
			#ifndef DynEarthQuake
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#else
				helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]+du2dotG*DynLoad(jiter+1)-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			#endif
			df0dxMPI(xiter) += prec_inner_prod(helper1,Landa[jiter]);
			#ifdef Dynf0valUTKHatU
				df0dxMPI(xiter) += prec_inner_prod(axpy_prod(dKHatdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
			#else
				#ifdef Dynf0valUTKU
					df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					#ifdef Dynf0valUTMU
						df0dxMPI(xiter) += prec_inner_prod(axpy_prod(dMassdx,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
					#endif
				#endif
			#endif
		}
		if (iter==0)							//dfdx is equal for this purpose
			dfdxMPI(0,xiter)=Elemlen*Elemwid; 		//for edge ro modifications
		if (MyMPIRank == 0)
			printf ("\b\b\b\b\b\b\b");
	}

	MPI_Allreduce(df0dxMPI.data().begin(),df0dx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if (iter==0)
		MPI_Allreduce(dfdxMPI.data().begin(),dfdx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	delete[] Landa;
	Taucs_FreeFactor(&F);
	if (MyMPIRank==0)
		cout<<"Filtering Sensitivity analysis is done"<<endl;
	return true;

}

bool DynamicOptimization::DynElFilterOptimize()
{
	int OptTyp=2;
	Vector helperm(1);
	helperm(0)=1.00;
	Vector* DuTotal=new Vector[TimeIterN];
	Vector* DudotTotal=new Vector[TimeIterN];
	Vector* Du2dotTotal=new Vector[TimeIterN];

	for (size_t i=0;i<TimeIterN;i++)
	{
		DuTotal[i].resize(UNum);
		#ifdef DynDamping
			DudotTotal[i].resize(UNum);
		#endif
		Du2dotTotal[i].resize(UNum);
	}
	ifstream DynInFilter("DynamicOpt/DynFiltering/DynInFilter.txt");
	ofstream DynOutFilter("DynamicOpt/DynFiltering/DynOutEFilter.txt");
#ifdef DynRoEFilterPrint
	ofstream DynOutXFilter("DynamicOpt/DynFiltering/DynOutEXFilter.txt");
#endif
#ifdef DyndfEFilterDebug
	ofstream DynOutdfFilter("DynamicOpt/DynFiltering/DynOutdfEFilter.txt");
#endif
#ifdef DynFDElFiltering
	ofstream DynFDdfElFOut("DynamicOpt/DynFiltering/DynOutFDdfEFilter.txt");
#endif
	double FilterRadius;
	while (DynInFilter>>FilterRadius)
	{
		iter=0;
		while (/*(iter<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
			if (MyMPIRank == 0)
				cout<<"Filter Radius="<< FilterRadius<<endl;
			DynOutFilter<<"Filter Radius="<< FilterRadius<<endl;
/*#ifdef DynRoEFilterPrint
			DynOutXFilter<<"Filter Radius="<< FilterRadius<<endl;
#endif*/
			if (MyMPIRank == 0)
				cout<<"iter number"<<iter<<endl;
			DynOutFilter<<"iter number"<<iter<<endl;
/*#ifdef DynRoEFilterPrint
			DynOutXFilter<<"iter number"<<iter<<endl;
#endif*/
			MakeStiffness(OptTyp);
			BuildMassMatrix(OptTyp);
			f0valOld=f0val;
			DynAnalyse(DuTotal,DudotTotal,Du2dotTotal);
			//DynSensitvtAnalyse(DuTotal,Du2dotTotal);
#ifdef MPISensitivityAnalisys
			DynElFSensitvtAnalyzMPI(DuTotal,Du2dotTotal);
#else
			DynElFSensitvtAnalyz(DuTotal,Du2dotTotal);
#endif
			CalResidue();
#ifdef DyndfEFilterDebug
			DynOutdfFilter<<"before df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl;
#endif
#ifdef DynFDElFiltering
			double f0val2=f0val;
			for (size_t FD_DEF=0;FD_DEF<n;FD_DEF++)
			{
				Ro(FD_DEF) += 1.00e-9;
				MakeStiffness(OptTyp);
				BuildMassMatrix(OptTyp);
				DynAnalyse(DuTotal/*,DudotTotal*/,Du2dotTotal);
				DynFDdfElFOut<<"i = "<<FD_DEF<<" : "<<df0dx(FD_DEF)<<" , "<<(f0val-f0val2)/1e-9<<endl;
				Ro(FD_DEF) -= 1e-9;
			}
			DynFDdfElFOut<<"iter num "<<iter<<endl<<"_____________________________________"<<endl;
			f0val=f0val2;
#endif

			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
			ElFilterSenstvts(FilterRadius);
#ifdef DyndfEFilterDebug
			DynOutdfFilter<<" after df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl<<endl<<endl;
#endif
			if (/*(iter+1<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				mmasub();
				#ifdef CutXTails
					CutXTail();
				#endif
				Ro=xval;
			}
			if (MyMPIRank == 0)
				cout<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			DynOutFilter<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			DynOutFilter<<"__________________________________________________________________"<<endl<<endl;
			if (MyMPIRank == 0)
				cout<<"__________________________________________________________________"<<endl<<endl;
			DynOutFilter<<"__________________________________________________________________"<<endl<<endl;
#ifdef DynRoEFilterPrint
			DynOutXFilter<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
			for (size_t roiter=0;roiter<n;roiter++)
			{
				DynOutXFilter<<xval(roiter);
				if (roiter<n-1)
					DynOutXFilter<<",";
			}
			DynOutXFilter<<"};"<<endl;
#endif

			iter++;
		}
	}
	delete[] DuTotal;
	delete[] DudotTotal;
	delete[] Du2dotTotal;
	return true;
}

////// Hirarchichal Structures (Story based structures) ///////////////////////////////////////////////////////

void DynamicOptimization::BuildSBSMassMatrix(size_t &NStory,double *SMasses,double *SHeights)
{
	if (MyMPIRank == 0)
		cout<<"Entering Mass matrix builder"<<endl;
	MassS.clear();
	//#ifdef OptimizationisDynamic
	//MassT.clear();
	//#endif
	size_t NSElements[NStory];
	double SNodeMass[NStory];

	Matrix melem(8,8);
	double ro[IntegPoints.size1()];
	for (size_t istory=0;istory<NStory;istory++)
	{
		if (fmod(SHeights[istory]/Elemwid,1.00) != 0.00)
		{
			cout<<"Error in Heights of structures in story "<<istory+1<<" that is not dividable"<<endl;
			//cout<<SHeights[istory]<<" , "<<NStory<<endl;
			exit(1);
		}
		else
		{
			NSElements[istory]=SHeights[istory]/Elemwid;
			//SNodeMass[istory]=SMasses[istory]/(Lenpart*4.00);
			SNodeMass[istory]=SMasses[istory]/Length/Elemwid;
			ro[0]=ro[1]=ro[2]=ro[3]=SNodeMass[istory]/DensityVal;
			Element.mbuild(ro,melem);
		}
		size_t ii(0) , jj(0);
		for (size_t elem = (NSElements[istory]-1)*Lenpart;elem<NSElements[istory]*Lenpart;elem++)
		{
			for (size_t i=0;i<8;i++)
				for (size_t j=0;j<8;j++)
				{
					//cout<<i<<" , "<<j<<endl;
					ii = Connectivity(elem,i);

					////////////////////////////////////////////////////

					if (Index(ii) < UNum && Index(jj) < UNum && Index(ii)<=Index(jj))
					{
						//#pragma omp critical
						//cout<<i<<" , "<<j<<" , "<<melem(i,j);
#ifndef OptimizationisDynamic
						/*KuuS(Index(jj),Index(ii)) = */
						MassS(Index(ii),Index(jj)) += melem(i,j);
#else
						MassS(Index(ii),Index(jj)) += melem(i,j);
						MassT(Index(jj),Index(ii))=MassT(Index(ii),Index(jj))=MassS(Index(ii),Index(jj));
#endif
					}
				}
		}
	}

	if (MyMPIRank == 0)
		cout<<"Mass Matrix is built"<<endl;
}

void DynamicOptimization::DynModifySBSStiffness(size_t &NStory,double *SHeights)
{
	size_t NSElements[NStory];

	for (size_t istory=0;istory<NStory;istory++)
	{
		//cout<<"here"<<istory<<" , "<<SHeights[istory]<<endl;
		NSElements[istory]=SHeights[istory]/Elemwid;

		for (size_t elem = (NSElements[istory]-1)*Lenpart;elem<NSElements[istory]*Lenpart;elem++)
		{
			//cout<<"		elem= "<<elem<<endl;
			double ro[IntegPoints.size1()];
			Matrix kelem(8,8);
			size_t ii(0),jj(0);
			for (unsigned int i=0;i<IntegPoints.size1();i++)
				ro[i]=0.50e7;

			//cout<<ro<<" =ro"<<endl;
			Element.kBuild(ro,kelem);
			for (size_t i=0;i<8;i++)
				for (size_t j=0;j<8;j++)
				{
					ii = Connectivity(elem,i);
					jj = Connectivity(elem,j);

					if (Index(ii) < UNum && Index(jj) < UNum && Index(ii)<=Index(jj))
					{
						//#pragma omp critical
#ifndef OptimizationisDynamic
						/*KuuS(Index(jj),Index(ii)) = */
						KuuS(Index(ii),Index(jj)) += kelem(i,j);
#else
						KuuS(Index(ii),Index(jj)) += kelem(i,j);
						KuuT(Index(jj),Index(ii))=KuuT(Index(ii),Index(jj))=KuuS(Index(ii),Index(jj));
#endif
					}

				}
		}
	}

}

void DynamicOptimization::DynSBSAnalyse(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights)
{
	DuTotal[0]=DuInit;
	#ifdef DynDamping
		DudotTotal[0]=DudotInit;
	#endif
//#ifdef OptimizationisDynamic
	axpy_prod(KuuT,-DuTotal[0],Du);										//Du used just as a vector no meaning
//#endif
	//Du(Index(LEP)) += DynLoad(0);										//Because of no settelments
	//TAUCS_Solve(MassS,Du,Du2dotTotal[0],false);
	//cout<<"shetab"<<Du2dotTotal[0]<<endl<<Du<<endl<<Index(LEP)<<endl<<MassT<<endl; exit(1);
	Du2dotTotal[0].clear();

	CSMatrix KHat(UNum,UNum);
	#ifndef DynDamping
		KHat=KuuS+(4.00/Dt/Dt)*MassS;
		#ifdef DynSBSf0valUTKHatU
			CMatrix KHatT(UNum,UNum);
			KHatT=KuuT+(4.00/Dt/Dt)*MassT;
		#endif
	#else
		CuuS=CoefDK*KuuS+CoefDM*MassS;
		CuuT=CoefDK*KuuT+CoefDM*MassT;
		KHat=KuuS+2.00/Dt*CuuS+(4.00/Dt/Dt)*MassS;
		#ifdef DynSBSf0valUTKHatU
			CMatrix KHatT(UNum,UNum);
			KHatT=KuuT+2.00/Dt*CuuT+(4.00/Dt/Dt)*MassT;
		#endif
	#endif
	void *F=NULL;
	Taucs_Factor_Solve(KHat,&F);
	/*#ifdef DynUEA
		void *FMass=NULL;
		Taucs_Factor_Solve(MassS,&FMass);
	#endif*/
	#ifdef DynSBSEarthQuake
		if (iter == 0)
		{
			du2dotG.clear();											//Used for delta du2dot ground
			for (size_t ist=0;ist<NStory;ist++)							//Here DynLoad contains ground accelerations
				for (size_t isti=(SHeights[ist]/Elemwid-1)*(Lenpart+1)*2;isti<(SHeights[ist]/Elemwid)*(Lenpart+1)*2;isti+=2)
					du2dotG(Index(isti)) = du2dotG(Index(isti+(Lenpart+1)*2)) = 1.00;
		}
	#endif
	f0val=0.00;
	Vector Dudot(UNum);
	Dudot=DudotInit;
	if (MyMPIRank == 0)
		cout<<"Entering Analysis"<<endl;
	size_t iterj=0;

	for (double timei=StartT+Dt;timei <= EndT;timei += Dt)
	{
		iterj++;
		#ifdef DynSBSEarthQuake
			Vector Ddu2dotG(UNum);
			Ddu2dotG = (DynLoad(iterj)-DynLoad(iterj-1))*du2dotG;
			axpy_prod(MassT,(4.00/Dt*Dudot+2.00*Du2dotTotal[iterj-1]-Ddu2dotG),Fu);	//Here Fu is used just as a vector no meaning
		#else
			axpy_prod(MassT,(4.00/Dt*Dudot+2.00*Du2dotTotal[iterj-1]),Fu);	//Here Fu is used just as a vector no meaning
			//cout<<Fu<<endl; exit(1);
			Fu(Index(LEP)) += (DynLoad(iterj)-DynLoad(iterj-1));
		#endif
		#ifdef DynDamping
			axpy_prod(CuuT,2.00*Dudot,Fu,false);
		#endif
		Taucs_Factor_Solve(KHat,&F,2,Fu,Du);
		DuTotal[iterj] =DuTotal[iterj-1]+Du;								//Du is used here as Delta Displacement

		//#ifndef DynUEA
			Du2dotTotal[iterj]=4.00/Dt/Dt*Du-4.00/Dt*Dudot-Du2dotTotal[iterj-1];
			Dudot=2.00/Dt*Du-Dudot;
			#ifdef DynDamping
				DudotTotal[iterj]=Dudot;
			#endif
		/*#else
			// Acceleration of equilibrium equation
			Dudot=2.00/Dt*Du-Dudot;
			axpy_prod(-KuuT,DuTotal[iterj],Fu);
			#ifdef DynSBSEarthQuake
				for (size_t ist=0;ist<NStory;ist++)							//Here DynLoad contains ground accelerations
					for (size_t isti=(SHeights[ist]/Elemwid-1)*(Lenpart+1)*2;isti<(SHeights[ist]/Elemwid)*(Lenpart+1)*2;isti+=2)
						Ddu2dotG(Index(isti)) = Ddu2dotG(Index(isti+(Lenpart+1)*2)) = DynLoad(iterj);

				Fu += axpy_prod(MassT,-Ddu2dotG,Du); //Du used here just as a helper vector
			#else
				Fu(Index(LEP)) += DynLoad(iterj);
			#endif
			Taucs_Factor_Solve(MassS,&FMass,2,Fu,Du2dotTotal[iterj]);*/
		//#endif

		#ifndef Dynf0valFTU
			f0val += prec_inner_prod(DuTotal[iterj],DuTotal[iterj]);
		#else
			//for(size_t istory=0;istory<NStory;istory++)
			#ifdef DynSBSf0valDispStories
				for (size_t istory=0;istory<NStory;istory++)
					f0val += pow(DuTotal[iterj](Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart)),2.00);
			#else
				#ifdef DynSBSf0valUTKHatU
					f0val += prec_inner_prod(axpy_prod(KHatT,DuTotal[iterj],Fu),DuTotal[iterj])*ImpFac;//Here Fu used as a Vector no meaning
				#else
					#ifdef DynSBSf0valUTKU
						f0val += prec_inner_prod(axpy_prod(KuuT,DuTotal[iterj],Fu),DuTotal[iterj])*ImpFac;
					#else
						#ifdef DynSBSf0valUTMU
							f0val += prec_inner_prod(axpy_prod(MassT,DuTotal[iterj],Fu),DuTotal[iterj])*ImpFac;
						#else
							#ifdef DynSBSf0valUTKDU
								f0val += prec_inner_prod(axpy_prod(KuuT,DuTotal[iterj],Fu),Du)*ImpFac;
							#else
								f0val += DuTotal[iterj](Index(LEP))*pow(1.00,2.00)*DuTotal[iterj](Index(LEP))*ImpFac;
							#endif
						#endif
					#endif
					//f0val += DuTotal[iterj](Index(LEP))*DuTotal[iterj](Index(LEP))*ImpFac;
				#endif
			#endif
		#endif
		//cout<<"du= "<<DuTotal[iterj]<<endl/*<<Du<<endl<<Du2dotTotal[iterj]<<endl*/;

		if (MyMPIRank == 0)
			cout<<"Time analysis = "<<timei<<endl;
	}
	/*if (iter==1)
	{
		ofstream testout("DynamicOpt/testout.txt");
		for (size_t itest=0;itest<TimeIterN;itest++)
		{
			testout<<DuTotal[itest]<<endl;
		}
		testout<<endl;
		for (size_t itest=0;itest<TimeIterN;itest++)
		{
			testout<<Du2dotTotal[itest]<<endl;
		}
		exit(1);
	}*/
	/** Free the factorization F **/
	//const char* options[] = {"taucs.factor.LLT=true", "taucs.factor.ll=true", "taucs.factor.ordering=amd", NULL };
	//taucs_linsolve(NULL,&F,0,NULL,NULL,(char**)options,NULL);
	if (MyMPIRank == 0)
		cout<<"Analysis is done"<<endl;
}

bool DynamicOptimization::DynSBSSensitvtAnalyse(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	#ifdef DynSBSf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		KHatT=KuuT+(4.00/Dt/Dt)*MassT;
		#ifdef DynDamping
			KHatT+=2.00/Dt*CuuT;
		#endif
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		#ifndef Dynf0valFTU
			Fu += 2.00*DuTotal[iteri];
		#else
			#ifdef DynSBSf0valDispStories
				for (size_t istory=0;istory<NStory;istory++)
					Fu(Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart)) += 2.00*DuTotal[iteri](Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart));
			#else
				#ifdef DynSBSf0valUTKHatU
					Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					#ifdef DynSBSf0valUTKU
						Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
					#else
						#ifdef DynSBSf0valUTMU
							Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
						#else
							#ifdef DynSBSf0valUTKDU
								Fu += axpy_prod(KuuT,2.00*DuTotal[iteri]-DuTotal[iteri-1],HelperUNum)*ImpFac;
								if (iteri != TimeIterN-1)
									Fu += axpy_prod(KuuT,-DuTotal[iteri+1],HelperUNum)*ImpFac;
							#else
								Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
							#endif
						#endif
					#endif
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	Taucs_FreeFactor(&F);
	cout<<"Landas are built"<<endl;

	double ro[IntegPoints.size1()];
	//double Massro[IntegPoints.size1()];
	cout<<"Entering sensitivity analysis"<<endl;

#pragma omp parallel for private(ro) num_threads(2) schedule(guided)
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		//Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif
		//CMatrix dMassdx(UNum,UNum);

		int lhs,rhs,uhs,bhs;
		size_t iro=xiter%(MDLenpart+1);
		size_t jro=(xiter-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		//dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		size_t mLeft=(lhs+iro-1)*MDElemLenNum;
		size_t mRight=(2-rhs+iro-1)*MDElemLenNum;
		size_t nBottom=(bhs+jro-1)*MDElemWidNum;
		size_t nUp=(2-uhs+jro-1)*MDElemWidNum;

		for (size_t jelem=nBottom;jelem<nUp;jelem++)
			for (size_t ielem=mLeft;ielem<mRight;ielem++)
			{
				size_t elemnum = jelem*Lenpart+ielem;
				size_t ii,jj;
				size_t mprime=(ielem-ielem%MDElemLenNum)/MDElemLenNum;
				size_t nprime=(jelem-jelem%MDElemWidNum)/MDElemWidNum;
				roBuilder(elemnum,ro,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1)/*,Massro*/); // Third arg. is to show the function which edge ro is being derived in a prime mesh desgine

				Element.kBuild(ro,kelem);
				//cout<<"massro= "<<Massro[0]<<" , "<<Massro[1]<<" , "<<Massro[2]<<" , "<<Massro[3]<<endl;
				//Element.mbuild(Massro,melem);

				for (size_t i=0;i<8;i++)
					for (size_t j=0;j<8;j++)
					{
						ii = Connectivity(elemnum,i);
						jj = Connectivity(elemnum,j);

						if (Index(ii) < UNum && Index(jj) < UNum)
						{
							Kuu(Index(ii),Index(jj)) += kelem(i,j);
							//dMassdx(Index(ii),Index(jj)) += melem(i,j);
						}

					}
			}
		//df0dx(xiter)=0.00;
		#ifdef DynDamping
			Cuu=Kuu*CoefDK;
		#endif
		Vector helper1(UNum);
		//Vector helper2(UNum);
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]/*-pow(-1,jiter+1)*DuTotal[0]*/,helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			//helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			df0dx(xiter) += prec_inner_prod(helper1,Landa[jiter]);
			#ifdef DynSBSf0valUTKHatU
				#ifdef DynDamping
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu*(1.00+2.00/Dt*CoefDK),DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#endif
			#else
				#ifdef DynSBSf0valUTKU
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					#ifdef DynSBSf0valUTKDU
						df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1]-DuTotal[jiter])*ImpFac;
					#endif
				#endif
			#endif
		}
		if (iter==0)													//dfdx is equal for this purpose
			dfdx(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
	}

	cout<<"sensitivity analysis is done"<<endl;
	delete[] Landa;
	return true;
}

bool DynamicOptimization::DynSBSSensitvtAnalyseMPI(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	#ifdef DynSBSf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		KHatT=KuuT+(4.00/Dt/Dt)*MassT;
		#ifdef DynDamping
			KHatT+=2.00/Dt*CuuT;
		#endif
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	void *F=NULL;
	Vector df0dxMPI(n);
	Matrix dfdxMPI(m,n);

	df0dxMPI.clear();
	df0dx.clear();
	dfdxMPI.clear();

	Taucs_Factor_Solve(KHat,&F);
	if (MyMPIRank == 0)
		cout<<"Building Landas"<<endl;
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		#ifndef Dynf0valFTU
			Fu += 2.00*DuTotal[iteri];
		#else
			#ifdef DynSBSf0valDispStories
				for (size_t istory=0;istory<NStory;istory++)
					Fu(Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart)) += 2.00*DuTotal[iteri](Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart))*ImpFac;
			#else
				#ifdef DynSBSf0valUTKHatU
					Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					#ifdef DynSBSf0valUTKU
						Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
					#else
						#ifdef DynSBSf0valUTMU
							Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
						#else
							#ifdef DynSBSf0valUTKDU
								Fu += axpy_prod(KuuT,2.00*DuTotal[iteri]-DuTotal[iteri-1],HelperUNum)*ImpFac;
								if (iteri != TimeIterN-1)
									Fu += axpy_prod(KuuT,-DuTotal[iteri+1],HelperUNum)*ImpFac;
							#else
								Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
							#endif
						#endif
					#endif
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	Taucs_FreeFactor(&F);
	if (MyMPIRank == 0)
		cout<<"Landas are built"<<endl;

	double ro[IntegPoints.size1()];
	//double Massro[IntegPoints.size1()];
	if (MyMPIRank == 0)
		cout<<"Entering sensitivity analysis"<<endl;

//#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=MyMPIRank;xiter<n;xiter+=OurMPISize)
	{
		if (MyMPIRank == 0)
			printf ("%5.2f %%",xiter/(n-1.00)*100.00);
		Matrix kelem(8,8);
		//Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif
		//CMatrix dMassdx(UNum,UNum);

		int lhs,rhs,uhs,bhs;
		size_t iro=xiter%(MDLenpart+1);
		size_t jro=(xiter-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		//dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		size_t mLeft=(lhs+iro-1)*MDElemLenNum;
		size_t mRight=(2-rhs+iro-1)*MDElemLenNum;
		size_t nBottom=(bhs+jro-1)*MDElemWidNum;
		size_t nUp=(2-uhs+jro-1)*MDElemWidNum;

		for (size_t jelem=nBottom;jelem<nUp;jelem++)
			for (size_t ielem=mLeft;ielem<mRight;ielem++)
			{
				size_t elemnum = jelem*Lenpart+ielem;
				size_t ii,jj;
				size_t mprime=(ielem-ielem%MDElemLenNum)/MDElemLenNum;
				size_t nprime=(jelem-jelem%MDElemWidNum)/MDElemWidNum;
				roBuilder(elemnum,ro,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1)/*,Massro*/); // Third arg. is to show the function which edge ro is being derived in a prime mesh desgine

				Element.kBuild(ro,kelem);
				//cout<<"massro= "<<Massro[0]<<" , "<<Massro[1]<<" , "<<Massro[2]<<" , "<<Massro[3]<<endl;
				//Element.mbuild(Massro,melem);

				for (size_t i=0;i<8;i++)
					for (size_t j=0;j<8;j++)
					{
						ii = Connectivity(elemnum,i);
						jj = Connectivity(elemnum,j);

						if (Index(ii) < UNum && Index(jj) < UNum)
						{
							Kuu(Index(ii),Index(jj)) += kelem(i,j);
							//dMassdx(Index(ii),Index(jj)) += melem(i,j);
						}

					}
			}
		//df0dx(xiter)=0.00;
		#ifdef DynDamping
			Cuu=Kuu*CoefDK;
		#endif
		Vector helper1(UNum);
		//Vector helper2(UNum);
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]/*-pow(-1,jiter+1)*DuTotal[0]*/,helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			//helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			
		}
		if (iter==0)																			//dfdx is equal for this purpose
			dfdxMPI(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
		if (MyMPIRank == 0)
			printf ("\b\b\b\b\b\b\b");
	}
	//printf ("I am num: %d \n",MyMPIRank);
	//cout<<df0dxMPI<<endl;
	//MPI_Barrier(MPI_COMM_WORLD);
	MPI_Allreduce(df0dxMPI.data().begin(),df0dx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if (iter==0)
		MPI_Allreduce(dfdxMPI.data().begin(),dfdx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	//cout<<"df0dx="<<df0dx<<endl;
	//MPI_Barrier(MPI_COMM_WORLD);

	if (MyMPIRank == 0)
		cout<<"\bsensitivity analysis is done"<<endl;
	delete[] Landa;
	return true;
}

bool DynamicOptimization::DynSBSOptimize()
{
	bool Resume=true;

	size_t NStory;
	ifstream DynInSBS("DynamicOpt/DynSBS/DynInSBS.txt");
	DynInSBS>>NStory;
	double SHeights[NStory];
	double SMasses[NStory];
	for (size_t IterInput=0;IterInput<NStory;IterInput++)
	{
		DynInSBS>>SHeights[IterInput];
		DynInSBS>>SMasses[IterInput];
		//cout<<IterInput<<" : "<<SHeights[IterInput]<<" , "<<SMasses[IterInput]<<endl;
	}

	iter =0;
	Vector helperm(m);
	helperm(0)=1.00;

	BuildSBSMassMatrix(NStory,SMasses,SHeights);

	Vector* DuTotal=new Vector[TimeIterN];
	Vector* DudotTotal=new Vector[TimeIterN];
	Vector* Du2dotTotal=new Vector[TimeIterN];

	ofstream DynOutOpt("DynamicOpt/DynSBS/DynSBSOptOut.txt");
#ifdef DynDf0dxDebug
	ofstream DynOutdf("DynamicOpt/DynSBS/DynSBSdfOut.txt");
#endif

#ifdef DynMassDebug
	ofstream DynOutMass("DynamicOpt/DynSBS/DynSBSMassOut.txt");
#endif

#ifdef DynFDdf0dx
	ofstream DynOutFD("DynamicOpt/DynSBS/DynFDSBSdfOut.txt");
#endif

#ifdef DynRoPrint
	ofstream DynOutXval("DynamicOpt/DynSBS/DynSBSXvalOut.txt");
#endif

#ifdef DynDispPrint
	ofstream DynOutDisps("DynamicOpt/DynSBS/DynSBSDispsOut.txt");
#endif

	for (size_t i=0;i<TimeIterN;i++)
	{
		DuTotal[i].resize(UNum);
		#ifdef DynDamping
			DudotTotal[i].resize(UNum);
		#endif
		Du2dotTotal[i].resize(UNum);
	}
	#ifdef TraceFollow
		FollowTrace();
	#endif
	do
	{
		if ((MDLenpart == Lenpart) && (MDWidpart == Widpart))
			Resume=false;
		while (/*(iter < maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
#ifdef DynSBSFillStorySolid
			for (size_t istory=0;istory<NStory;istory++)
			{
				for (size_t jstory=0;jstory<MDLenpart+1;jstory++)
					Ro(size_t(SHeights[istory]/(Width/MDWidpart))*(MDLenpart+1)+jstory)=xval(size_t(SHeights[istory]/(Width/MDWidpart))*(MDLenpart+1)+jstory)=1.00;
			}
			//DynOutXval<<" before iter= "<<iter<<" Md="<<MDLenpart<<" , "<<MDWidpart<<" f0val ,fval= "<<f0val<<" , "<<fval<<endl<<MDLenpart<<" , "<<MDWidpart<<endl<<xval<<endl<<endl;

#endif

#ifdef DynMTMakestiffness
			MakeStiffnessMT();
#else
			MakeStiffness();
#endif
#ifdef DynSBSRigidStory
			DynModifySBSStiffness(NStory,SHeights);
#endif
			//BuildSBSMassMatrix(NStory,SMasses,SHeights);
#ifdef DynMassDebug
			DynOutMass<<MassT<<endl;
			DynOutMass<<KuuT<<endl;
#endif

			f0valOld=f0val;
			DynSBSAnalyse(DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
#ifdef MPISensitivityAnalisys
			DynSBSSensitvtAnalyseMPI(DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
#else
			DynSBSSensitvtAnalyse(DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
#endif
			CalResidue();
#ifdef DynDf0dxDebug
			DynOutdf<<"iter num"<<iter<<endl;
			DynOutdf<<"MD="<<MDLenpart<<" , "<<MDWidpart<<endl;
			DynOutdf<<"f0val , fval= "<<f0val<<" , "<<fval<<endl;
			DynOutdf<<"df0dx= "<<df0dx<<endl;
#endif

#ifdef DynDispPrint
			//if (iter == maxite-1)
			//{
				for (size_t jiter=0;jiter<TimeIterN;jiter++)
				{
					//for (size_t iDisp=0;iDisp<UNum;iDisp++)
						//Disps(IndexRev(iDisp))=DuTotal[jiter](iDisp);
					DynOutDisps/*<<"iter= "<<iter<<"Disps "<<jiter<<" = "*/<<DuTotal[jiter](IndexRev(LEP))<<" , "<<endl;
				}
			//}
#endif

#ifdef DynFDdf0dx
			//if(iter==2)
			//{
				DynOutFD<<"iter= "<<iter<<endl;
				double f0val2=f0val;
				BuildSBSMassMatrix(NStory,SMasses,SHeights);
				for (size_t FDiter=0;FDiter<n;FDiter++)
				{
					//cout << (DynOutFD << "TEST" << endl);
					Ro(FDiter) += 1e-6;
					MakeStiffness();
					#ifdef DynSBSRigidStory
						DynModifySBSStiffness(NStory,SHeights);
					#endif

					DynSBSAnalyse(DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
					DynOutFD<<"i= "<<FDiter<<" : "<<(df0dx(FDiter)) << " , ";
					df0dx(FDiter)=(f0val-f0val2)/1e-6;
					DynOutFD << (df0dx(FDiter)) << endl;
					//DynOutFD.flush();
					cout<<df0dx(FDiter)<<" , ";
					Ro(FDiter) -= 1e-6;
				}
				f0val=f0val2;
				DynOutFD<<endl<<"Md= "<<MDLenpart<<" , "<<MDWidpart<<endl;
				DynOutFD<<"________________________________________________"<<endl;
			//}
#endif

			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
			if (/*(iter+1 < maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				mmasub();
				#ifdef CutXTails
					CutXTail();
				#endif
				//cout<<"Ro= "<<Ro<<endl;
				#ifdef DynRoPrint
					DynOutXval<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
					for (size_t roiter=0;roiter<n;roiter++)
					{
						DynOutXval<<xval(roiter);
						if (roiter<n-1)
							DynOutXval<<",";
					}
					DynOutXval<<"};"<<endl;
					//DynOutXval<<"iter= "<<iter<<" Md="<<MDLenpart<<" , "<<MDWidpart<<" f0val ,fval= "<<f0val<<" , "<<fval<<endl<<MDLenpart<<" , "<<MDWidpart<<endl<<xval<<endl<<endl;
				#endif
				Ro=xval;
			}
			iter++;
			if (MyMPIRank == 0)
			{
				cout<<"f0val= "<<f0val<<" ,fval= "<<fval<<endl;
				cout<<MDLenpart<<" , "<<MDWidpart<<endl;
				cout<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
			}
			DynOutOpt<<MDLenpart<<" , "<<MDWidpart<<endl;
			DynOutOpt<<"f0val= "<<f0val<<" ,fval= "<<fval<<endl;
			DynOutOpt<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
			DynOutOpt<<"Number of trys in mma: "<<counter<<endl;
			DynOutOpt<<"Iteration number "<<iter<<" is done"<<endl<<"_________________________________________"<<endl;
			if (MyMPIRank == 0)
				printf("Iteration number %ld is done\n_____________________________________________________________\n",iter);
		}
		X2NewX(NStory,SHeights);
	}
	while (Resume);
	delete[] DuTotal;
	delete[] DudotTotal;
	delete[] Du2dotTotal;
	return true;
}

void DynamicOptimization::DynSBSInSensitvtAnalyz(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	#ifdef DynSBSf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		#ifdef DynDamping
			KHatT+=2.00/Dt*CuuT;
		#endif
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);
	df0dx.clear();
	void *F=NULL;
	Taucs_Factor_Solve(KHat,&F);
	cout<<"Building Landas Inner opt."<<endl;
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning
		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		#ifndef Dynf0valFTU
			Fu += 2.00*DuTotal[iteri];
		#else
			#ifdef DynSBSf0valDispStories
				for (size_t istory=0;istory<NStory;istory++)
					Fu(Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart)) += 2.00*DuTotal[iteri](Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart));
			#else
				#ifdef DynSBSf0valUTKHatU
					Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					#ifdef DynSBSf0valUTKU
						Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
					#else
						#ifdef DynSBSf0valUTMU
							Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
						#else
							#ifdef DynSBSf0valUTKDU
								Fu += axpy_prod(KuuT,2.00*DuTotal[iteri]-DuTotal[iteri-1],HelperUNum)*ImpFac;
								if (iteri != TimeIterN-1)
									Fu += axpy_prod(KuuT,-DuTotal[iteri+1],HelperUNum)*ImpFac;
							#else
								Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
							#endif
						#endif
					#endif
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	/*for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)
	{
		Du.clear();
		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		axpy_prod(MassT,Du,Fu);
		Fu += 2.00*DuTotal[iteri];
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
	}*/
	Taucs_FreeFactor(&F);
	cout<<"Landas are built, Inner Optimization"<<endl;
	double ro[IntegPoints.size1()];
	//double Massro[IntegPoints.size1()];
#pragma omp parallel for private(ro) num_threads(2) schedule(guided)
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		//Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif
		//CMatrix dMassdx(UNum,UNum);

		int lhs,rhs,uhs,bhs;
		size_t iro=InnerMapIndexRev(xiter)%(MDLenpart+1);
		size_t jro=(InnerMapIndexRev(xiter)-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		bhs = (jro == 0) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		//dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		for (size_t nprime=bhs+jro-1;nprime<2-uhs+jro-1;nprime++)
			for (size_t mprime=lhs+iro-1;mprime<2-rhs+iro-1;mprime++)
				for (size_t iii=0;iii<EIPI[mprime+nprime*MDLenpart].size()/4;iii++)
				{
					size_t ii,jj;
					InSensitvtroBuilder(ro,mprime,nprime,iii,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1),EIPI/*,Massro*/); //5th argument shows the number of corner ro to be driven
					Element.kBuild(ro,kelem);
					//Element.mbuild(Massro,melem);
					//cout<<Element.k<<endl;
					size_t elemnum=size_t(EIPI[mprime+nprime*MDLenpart][4*iii+1]);

					for (size_t i=0;i<8;i++)
						for (size_t j=0;j<8;j++)
						{

							ii = Connectivity(elemnum,i);
							jj = Connectivity(elemnum,j);

							if (Index(ii) < UNum && Index(jj) < UNum)
							{
								Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);
								//dMassdx(Index(ii),Index(jj)) += melem(i,j);
							}
						}
				}
		//df0dx(xiter)=0.00;
		
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]/*-pow(-1,jiter+1)*DuTotal[0]*/,helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			//helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			df0dx(xiter) += prec_inner_prod(helper1,Landa[jiter]);
			#ifdef DynSBSf0valUTKHatU
				#ifdef DynDamping
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu*(1.00+2.00/Dt*CoefDK),DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#endif
			#else
				#ifdef DynSBSf0valUTKU
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					#ifdef DynSBSf0valUTKDU
						df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1]-DuTotal[jiter])*ImpFac;
					#endif
				#endif
			#endif
		}
		if (iter==0)																			//dfdx is equal for this purpose
			dfdx(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
	}
	cout<<"Inner sensitivity analysis is done"<<endl;
	//Taucs_FreeFactor(&F);
	delete[] Landa;
}

void DynamicOptimization::DynSBSInSensitvtAnalyzMPI(vector<double> *EIPI,Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights)
{
	if (MyMPIRank == 0)
		cout<<"Entering MPI sensitivity analysis"<<endl;
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	#ifdef DynSBSf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		KHatT=KuuT+(4.00/Dt/Dt)*MassT;
		#ifdef DynDamping
			KHatT+=2.00/Dt*CuuT;
		#endif
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	Vector df0dxMPI(n);
	Matrix dfdxMPI(m,n);

	df0dxMPI.clear();
	df0dx.clear();
	dfdxMPI.clear();

	void *F=NULL;
	Taucs_Factor_Solve(KHat,&F);
	if (MyMPIRank == 0)
		cout<<"Building Landas Inner opt."<<endl;
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		#ifndef Dynf0valFTU
			Fu += 2.00*DuTotal[iteri];
		#else
			#ifdef DynSBSf0valDispStories
				for (size_t istory=0;istory<NStory;istory++)
					Fu(Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart)) += 2.00*DuTotal[iteri](Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart))*ImpFac;
			#else
				#ifdef DynSBSf0valUTKHatU
					Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					#ifdef DynSBSf0valUTKU
						Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
					#else
						#ifdef DynSBSf0valUTMU
							Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
						#else
							#ifdef DynSBSf0valUTKDU
								Fu += axpy_prod(KuuT,2.00*DuTotal[iteri]-DuTotal[iteri-1],HelperUNum)*ImpFac;
								if (iteri != TimeIterN-1)
									Fu += axpy_prod(KuuT,-DuTotal[iteri+1],HelperUNum)*ImpFac;
							#else
								Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
							#endif
						#endif
					#endif
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	/*for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)
	{
		Du.clear();
		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		axpy_prod(MassT,Du,Fu);
		Fu += 2.00*DuTotal[iteri];
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
	}*/
	Taucs_FreeFactor(&F);
	if (MyMPIRank == 0)
		cout<<"Landas are built, Inner Optimization"<<endl;
	double ro[IntegPoints.size1()];
	//double Massro[IntegPoints.size1()];
//#pragma omp parallel for private(ro,Massro) num_threads(2) schedule(guided)
	for (size_t xiter=MyMPIRank;xiter<n;xiter +=OurMPISize)
	{
		if (MyMPIRank == 0)
			printf ("%5.2f %%",xiter/(n-1.00)*100.00);
		Matrix kelem(8,8);
		//Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif
		//CMatrix dMassdx(UNum,UNum);

		int lhs,rhs,uhs,bhs;
		size_t iro=InnerMapIndexRev(xiter)%(MDLenpart+1);
		size_t jro=(InnerMapIndexRev(xiter)-iro)/(MDLenpart+1);

		lhs = (iro == 0) ? 1 : 0;
		rhs = (iro == MDLenpart) ? 1 : 0;
		uhs = (jro == MDWidpart) ? 1 : 0;

		Kuu.clear();
		//dMassdx.clear();
		//size_t elembegin=((jro-1.00)*MDElemWidNum*MDLenpart+(iro-1.00)*MDElemLenNum)+lhs*MDElemLenNum+bhs*MDElemWidNum*MDLenpart;
		//size_t elemend=
		for (size_t nprime=bhs+jro-1;nprime<2-uhs+jro-1;nprime++)
			for (size_t mprime=lhs+iro-1;mprime<2-rhs+iro-1;mprime++)
				for (size_t iii=0;iii<EIPI[mprime+nprime*MDLenpart].size()/4;iii++)
				{
					size_t ii,jj;
					InSensitvtroBuilder(ro,mprime,nprime,iii,3+(mprime-iro+1)-(nprime-jro+1)-2*(mprime-iro+1)*(nprime-jro+1),EIPI/*,Massro*/); //5th argument shows the number of corner ro to be driven
					Element.kBuild(ro,kelem);
					//Element.mbuild(Massro,melem);
					//cout<<Element.k<<endl;
					size_t elemnum=size_t(EIPI[mprime+nprime*MDLenpart][4*iii+1]);

					for (size_t i=0;i<8;i++)
						for (size_t j=0;j<8;j++)
						{

							ii = Connectivity(elemnum,i);
							jj = Connectivity(elemnum,j);

							if (Index(ii) < UNum && Index(jj) < UNum)
							{
								Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);
								//dMassdx(Index(ii),Index(jj)) += melem(i,j);
							}
						}
				}
		//df0dx(xiter)=0.00;
		#ifdef DynDamping
			Cuu=Kuu*CoefDK;
		#endif
		Vector helper1(UNum);
		//Vector helper2(UNum);
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1]/*-pow(-1,jiter+1)*DuTotal[0]*/,helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			//helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1]-pow(-1,jiter+1)*Du2dotTotal[0],helper2);
			df0dxMPI(xiter) += prec_inner_prod(helper1,Landa[jiter]);
			#ifdef DynSBSf0valUTKHatU
				#ifdef DynDamping
					df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu*(1.00+2.00/Dt*CoefDK),DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#endif
			#else
				#ifdef DynSBSf0valUTKU
					df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					#ifdef DynSBSf0valUTKDU
						df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1]-DuTotal[jiter])*ImpFac;
					#endif
				#endif
			#endif
		}
		if (iter==0)																			//dfdx is equal for this purpose
			dfdxMPI(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
		if (MyMPIRank == 0)
			printf ("\b\b\b\b\b\b\b");
	}
	MPI_Allreduce(df0dxMPI.data().begin(),df0dx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if (iter==0)
		MPI_Allreduce(dfdxMPI.data().begin(),dfdx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if (MyMPIRank == 0)
		cout<<"\bInner sensitivity analysis is done"<<endl;
	//Taucs_FreeFactor(&F);
	delete[] Landa;
}

bool DynamicOptimization::DynSBSInnerOptimize()
{
	/*///////////////////////////////////////////////////////
	// OptTyp specifies the type of optimization		   //
	// OptTyp=0		:Ordinary Inner Optimization		   //
	// OptTyp=1		:Ordinary Opt. or node based filter op.//
	// OptTyp=2		:Element based filtering Optimization  //
	///////////////////////////////////////////////////////*/

	int OptTyp=0;													// Shows kind of optimization is inner
	bool Resume=true;
	iter=0;
	size_t NStory;
	ifstream DynInSBS("DynamicOpt/DynSBS/DynInSBS.txt");
	DynInSBS>>NStory;
	double SHeights[NStory];
	double SMasses[NStory];
	for (size_t IterInput=0;IterInput<NStory;IterInput++)
	{
		DynInSBS>>SHeights[IterInput];
		DynInSBS>>SMasses[IterInput];
	}

	Vector* DuTotal=new Vector[TimeIterN];
	Vector* DudotTotal=new Vector[TimeIterN];
	Vector* Du2dotTotal=new Vector[TimeIterN];

	for (size_t i=0;i<TimeIterN;i++)
	{
		DuTotal[i].resize(UNum);
		#ifdef DynDamping
			DudotTotal[i].resize(UNum);
		#endif
		Du2dotTotal[i].resize(UNum);
	}
	Vector helperm(m);
	helperm(0)=1.00;

#ifdef FreezeInnerSolidX
	Matrix EfArea(m,Ro.size());
#endif

	ofstream DynOutInnerOpt("DynamicOpt/DynSBS/DynSBSInnerOptOut.txt");

#ifdef DynInnerFDdf0dx
	ofstream DynOutInnerFD("DynamicOpt/DynSBS/DynSBSInnerFDdfOut.txt");
#endif

#ifdef DynInnerRoPrint
	ofstream DynOutInnerXval("DynamicOpt/DynSBS/DynSBSInnerXvalOut.txt");
#endif

#ifdef DynInnerDispPrint
	ofstream DynOutInnerDisps("DynamicOpt/DynSBS/DynSBSInnerDispsOut.txt");
#endif
	do
	{
		FTEIC=true;														// first time of each inner convergence
		vector<double> EIPI[MDLenpart*MDWidpart];						//M.D. Element Integration Points Information matrix
		while (/*(iter<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
			MakeStiffness(OptTyp,EIPI);
			FTEIC=false;
#ifdef DynSBSRigidStory
			DynModifySBSStiffness(NStory,SHeights);
#endif
			//BuildSBSMassMatrix(NStory,SMasses,SHeights);

			f0valOld=f0val;
			DynSBSAnalyse(DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
#ifdef MPIInnerSensitivityAnalisys
			DynSBSInSensitvtAnalyzMPI(EIPI,DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
#else
			DynSBSInSensitvtAnalyz(EIPI,DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
#endif
			CalResidue();
#ifdef DynInnerDispPrint

			if (iter == maxite-1)
			{
				for (size_t jiter=0;jiter<TimeIterN;jiter++)
				{
					for (size_t iDisp=0;iDisp<UNum;iDisp++)
						Disps(IndexRev(iDisp))=DuTotal[jiter](iDisp);
					DynOutInnerDisps<<"iter= "<<iter<<endl<<"Disps "<<jiter<<" = "<<Disps<<endl;
				}
			}

#endif

#ifdef DynInnerFDdf0dx
			//if(iter==2)
			{
				DynOutInnerFD<<"iter= "<<iter<<endl;
				double f0val2=f0val;
				for (size_t FDiter=0;FDiter<n;FDiter++)
				{
					//cout << (DynOutFD << "TEST" << endl);
					Ro(InnerMapIndexRev(FDiter)) += 1.00e-6;
					MakeStiffness(OptTyp,EIPI);
					#ifdef DynSBSRigidStory
						DynModifySBSStiffness(NStory,SHeights);
					#endif
					BuildSBSMassMatrix(NStory,SMasses,SHeights);
					//BuildMassMatrix(OptTyp,EIPI);
					DynSBSAnalyse(DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
					DynOutInnerFD<<"i= "<<FDiter<<" : "<<(df0dx(FDiter))<< " , "<<(f0val-f0val2)/1.00e-6 << endl;
					//df0dx(FDiter)=(f0val-f0val2)/1.00e-9;
					//DynOutInnerFD << (df0dx(FDiter)) << endl;
					//DynOutFD.flush();
					//cout<<df0dx(FDiter)<<" , ";
					Ro(InnerMapIndexRev(FDiter)) -= 1.00e-6;
				}
				f0val=f0val2;
				DynOutInnerFD<<endl<<"Md= "<<MDLenpart<<" , "<<MDWidpart<<endl;
				DynOutInnerFD<<"________________________________________________"<<endl;
			}
#endif
#ifdef FreezeInnerSolidX

			if (iter==0)
			{
				for (size_t xiter=0; xiter<Ro.size(); xiter++)
				{
					int lhs,rhs,uhs,bhs;
					size_t iro=xiter%(MDLenpart+1);
					size_t jro=(xiter-iro)/(MDLenpart+1);

					lhs = (iro == 0) ? 1 : 0;
					rhs = (iro == MDLenpart) ? 1 : 0;
					bhs = (jro == 0) ? 1 : 0;
					uhs = (jro == MDWidpart) ? 1 : 0;
					EfArea(0,xiter)=(Length/MDLenpart*Width/MDWidpart)/pow(2.00,lhs+rhs+bhs+uhs);
				}
			}
			axpy_prod(EfArea,Ro,fval);
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;

#else
			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
#endif
			if (/*(iter+1<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				mmasub();
				#ifdef CutXTails
					CutXTail();
				#endif
				#ifndef FreezeInnerSolidX
					Ro=xval;
				#else
					for (size_t i=0;i<InnerMapIndexRev.size();i++)
						Ro(InnerMapIndexRev(i))=xval(i);
				#endif
			}
			iter++;

#ifdef DynInnerRoPrint
			DynOutInnerXval<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
			for (size_t roiter=0;roiter<Ro.size();roiter++)
			{
				DynOutInnerXval<<Ro(roiter);
				if (roiter<Ro.size()-1)
					DynOutInnerXval<<",";
			}
			DynOutInnerXval<<"};"<<endl;
			//DynOutXval<<"iter= "<<iter<<" Md="<<MDLenpart<<" , "<<MDWidpart<<" f0val ,fval= "<<f0val<<" , "<<fval<<endl<<MDLenpart<<" , "<<MDWidpart<<endl<<xval<<endl<<endl;
			//DynOutInnerXval<<"iter= "<<iter<<" Md="<<MDLenpart<<" , "<<MDWidpart<<" f0val ,fval= "<<f0val<<" , "<<fval<<endl<<MDLenpart<<" , "<<MDWidpart<<endl<<Ro<<endl<<endl;
#endif
			if (MyMPIRank == 0)
			{
				cout<<"f0val= "<<f0val<<" ,fval= "<<fval<<endl;
				cout<<MDLenpart<<" , "<<MDWidpart<<endl;
				cout<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
			}
			DynOutInnerOpt<<MDLenpart<<" , "<<MDWidpart<<endl;
			DynOutInnerOpt<<"f0val= "<<f0val<<" ,fval= "<<fval<<endl;
			DynOutInnerOpt<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
			DynOutInnerOpt<<"Number of trys in mma: "<<counter<<endl;
			DynOutInnerOpt<<"Inner Iteration number "<<iter<<" is done"<<endl<<"_________________________________________"<<endl;
			if (MyMPIRank == 0)
				printf("Iteration number %ld is done\n_____________________________________________________________\n",iter);
		}
		Resume=X2NewX(NStory,SHeights);
	}
	while (Resume);

	delete[] DuTotal;
	delete[] DudotTotal;
	delete[] Du2dotTotal;

	return true;
}

///////SBS Filtering techniques

bool DynamicOptimization::DynSBSFilterOptimize()
{
	size_t NStory;
	ifstream DynInSBS("DynamicOpt/DynSBS/DynInSBS.txt");
	DynInSBS>>NStory;
	double SHeights[NStory];
	double SMasses[NStory];
	for (size_t IterInput=0;IterInput<NStory;IterInput++)
	{
		DynInSBS>>SHeights[IterInput];
		DynInSBS>>SMasses[IterInput];
		//cout<<IterInput<<" : "<<SHeights[IterInput]<<" , "<<SMasses[IterInput]<<endl;
	}

	iter =0;
	Vector helperm(m);
	helperm(0)=1.00;

	BuildSBSMassMatrix(NStory,SMasses,SHeights);

	Vector* DuTotal=new Vector[TimeIterN];
	Vector* Du2dotTotal=new Vector[TimeIterN];
	for (size_t i=0;i<TimeIterN;i++)
	{
		DuTotal[i].resize(UNum);
		#ifdef DynDamping
			DudotTotal[i].resize(UNum);
		#endif
		Du2dotTotal[i].resize(UNum);
	}

	ifstream DynInFilter("DynamicOpt/DynSBS/DynSBSFiltering/DynSBSInFilter.txt");
	ofstream DynOutFilter("DynamicOpt/DynSBS/DynSBSFiltering/DynSBSOutFilter.txt");
#ifdef DynRoFilterPrint
	ofstream DynOutXFilter("DynamicOpt/DynSBS/DynSBSFiltering/DynSBSOutXFilter.txt");
#endif
#ifdef DyndfFilterDebug
	ofstream DynOutdfFilter("DynamicOpt/DynSBS/DynSBSFiltering/DynSBSOutdfFilter.txt");
#endif
#ifdef DynFDFiltering
	ofstream DynFDdfFilOut("DynamicOpt/DynSBS/DynSBSFiltering/DynSBSOutFDdfFilter.txt");
#endif
	double FilterRadius;
	while (DynInFilter>>FilterRadius)
	{
		iter=0;
		while (/*(iter<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
#ifdef DynSBSFillStorySolid
			for (size_t istory=0;istory<NStory;istory++)
			{
				for (size_t jstory=0;jstory<MDLenpart+1;jstory++)
					Ro(size_t(SHeights[istory]/(Width/MDWidpart))*(MDLenpart+1)+jstory)=xval(size_t(SHeights[istory]/(Width/MDWidpart))*(MDLenpart+1)+jstory)=1.00;
			}
			//DynOutXval<<" before iter= "<<iter<<" Md="<<MDLenpart<<" , "<<MDWidpart<<" f0val ,fval= "<<f0val<<" , "<<fval<<endl<<MDLenpart<<" , "<<MDWidpart<<endl<<xval<<endl<<endl;

#endif
			if (MyMPIRank == 0)
				cout<<"Filter Radius="<< FilterRadius<<endl;
			DynOutFilter<<"Filter Radius="<< FilterRadius<<endl;
/*#ifdef DynRoFilterPrint
			DynOutXFilter<<"Filter Radius="<< FilterRadius<<endl;
#endif*/

			if (MyMPIRank == 0)
				cout<<"iter number"<<iter<<endl;
			DynOutFilter<<"iter number"<<iter<<endl;
/*#ifdef DynRoFilterPrint
			DynOutXFilter<<"iter number"<<iter<<endl;
#endif*/
			MakeStiffness();
			//BuildMassMatrix();
#ifdef DynSBSRigidStory
			DynModifySBSStiffness(NStory,SHeights);
#endif

			f0valOld=f0val;
			DynSBSAnalyse(DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
#ifdef MPISensitivityAnalisys
			DynSBSSensitvtAnalyseMPI(DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
#else
			DynSBSSensitvtAnalyse(DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
#endif
			CalResidue();
#ifdef DynFDFiltering
			double f0val2=f0val;
			for (size_t FD_DF=0;FD_DF<n;FD_DF++)
			{
				Ro(FD_DF) += 1e-6;
				MakeStiffness();
				//BuildMassMatrix();
				#ifdef DynSBSRigidStory
					DynModifySBSStiffness(NStory,SHeights);
				#endif
				DynFDdfFilOut<<"i = "<<FD_DF<<" : "<<df0dx(FD_DF)<<" , "<<(f0val-f0val2)/1e-6<<endl;
				Ro(FD_DF) -= 1e-6;
			}
			f0val=f0val2;
			DynFDdfFilOut<<"iter num "<<iter<<endl<<"_____________________________________"<<endl;
#endif

#ifdef DyndfFilterDebug
			DynOutdfFilter<<"before df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl;
#endif
			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
			FilterSenstvts(FilterRadius);
#ifdef DyndfFilterDebug
			DynOutdfFilter<<" after df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl<<endl<<endl;
#endif
			if (/*(iter+1<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				mmasub();
				#ifdef CutXTails
					CutXTail();
				#endif
				Ro=xval;
			}
			if (MyMPIRank == 0)
				cout<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			DynOutFilter<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			DynOutFilter<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
			if (MyMPIRank == 0)
				cout<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
			DynOutFilter<<"__________________________________________________________________"<<endl<<endl;
			if (MyMPIRank == 0)
				cout<<"__________________________________________________________________"<<endl<<endl;
			DynOutFilter<<"__________________________________________________________________"<<endl<<endl;
#ifdef DynRoFilterPrint
			DynOutXFilter<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
			for (size_t roiter=0;roiter<n;roiter++)
			{
				DynOutXFilter<<xval(roiter);
				if (roiter<n-1)
					DynOutXFilter<<",";
			}
			DynOutXFilter<<"};"<<endl;
#endif
			iter++;
		}
	}
	delete[] DuTotal;
	delete[] DudotTotal;
	delete[] Du2dotTotal;
	return true;
}

bool DynamicOptimization::DynSBSElFSensitvtAnalyz(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	#ifdef DynSBSf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		KHatT=KuuT+(4.00/Dt/Dt)*MassT;
		#ifdef DynDamping
			KHatT+=2.00/Dt*CuuT;
		#endif
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	void *F=NULL;
	df0dx.clear();
	Taucs_Factor_Solve(KHat,&F);
	cout<<"Building Landas"<<endl;
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		#ifndef Dynf0valFTU
			Fu += 2.00*DuTotal[iteri];
		#else
			#ifdef DynSBSf0valDispStories
				for (size_t istory=0;istory<NStory;istory++)
					Fu(Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart)) += 2.00*DuTotal[iteri](Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart))*ImpFac;
			#else
				#ifdef DynSBSf0valUTKHatU
					Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					#ifdef DynSBSf0valUTKU
						Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
					#else
						#ifdef DynSBSf0valUTMU
							Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
						#else
							Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
						#endif
					#endif
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	cout<<"Landas are built"<<endl;

	double ro[IntegPoints.size1()];
	//double Massro[IntegPoints.size1()];
	cout<<"Entering filtering sensitivity analysis"<<endl;
#pragma omp parallel for private(ro) num_threads(2) schedule(guided)
	for (size_t xiter=0;xiter<n;xiter++)
	{
		Matrix kelem(8,8);
		//Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif
		//CMatrix dMassdx(UNum,UNum);

		Kuu.clear();
		//dMassdx.clear();
		for (unsigned int roi=0;roi<IntegPoints.size1();roi++)
		{
			ro[roi]=SimpPower*pow(Ro(xiter),SimpPower-1.00);
			//Massro[roi]=1.00;
		}
		Element.kBuild(ro,kelem);
		//Element.mbuild(Massro,melem);
		size_t ii,jj;
		//cout<<Element.k<<endl;
		for (size_t i=0;i<8;i++)
			for (size_t j=0;j<8;j++)
			{

				ii = Connectivity(xiter,i);
				jj = Connectivity(xiter,j);

				if (Index(ii) < UNum && Index(jj) < UNum)
				{
					Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);
					//dMassdx(Index(ii),Index(jj)) += melem(i,j);
				}
			}
		Vector helper1(UNum);
		Vector helper2(UNum);
		#ifdef DynDamping
			Cuu=Kuu*CoefDK;
		#endif
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1],helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			//helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1],helper2);
			df0dx(xiter) += prec_inner_prod(helper1,Landa[jiter]);
			#ifdef DynSBSf0valUTKHatU
				#ifdef DynDamping
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu*(1.00+2.00/Dt*CoefDK),DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#endif
			#else
				#ifdef DynSBSf0valUTKU
					df0dx(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#endif
			#endif
		}
		if (iter==0)							//dfdx is equal for this purpose
			dfdx(0,xiter)=Elemlen*Elemwid; 		//for edge ro modifications
		//cout<<xiter<<endl;
	}
	delete[] Landa;
	Taucs_FreeFactor(&F);
	cout<<"Filtering Sensitivity analysis is done"<<endl;
	return true;

}

bool DynamicOptimization::DynSBSElFSensitvtAnalyzMPI(Vector *DuTotal,Vector *DudotTotal,Vector *Du2dotTotal,size_t &NStory,double *SHeights)
{
	CSMatrix KHat(UNum,UNum);
	KHat=KuuS+4.00/Dt/Dt*MassS;
	#ifdef DynDamping
		KHat+=2.00/Dt*CuuS;
	#endif
	#ifdef DynSBSf0valUTKHatU
		CMatrix KHatT(UNum,UNum);
		KHatT=KuuT+(4.00/Dt/Dt)*MassT;
		#ifdef DynDamping
			KHatT+=2.00/Dt*CuuT;
		#endif
	#endif
	Vector *Landa=new Vector[TimeIterN-1];
	for (size_t i=0;i<TimeIterN-1;i++)
		Landa[i].resize(UNum);

	void *F=NULL;

	Vector df0dxMPI(n);
	Matrix dfdxMPI(m,n);

	df0dxMPI.clear();
	df0dx.clear();
	dfdxMPI.clear();

	Taucs_Factor_Solve(KHat,&F);
	if (MyMPIRank == 0)
		cout<<"Building Landas"<<endl;
	Vector HelperUNum(UNum);
	for (size_t iteri=TimeIterN-1;iteri > 0;iteri--)			//Calculating Landa coefficients
	{
		//cout<<"Landa "<<iteri<<" is started"<<endl;
		Du.clear();												//Du used here as a vector no meaning

		for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
		{
			//cout<<"j="<<j<<endl;
			Du += (-16.00/Dt/Dt*pow(-1,j)*j*Landa[j+iteri-1]);
		}
		axpy_prod(MassT,Du,Fu);
		#ifdef DynDamping
			Du.clear();
			for (size_t j = 1 ; j < TimeIterN-iteri ; j++)
			{
				//cout<<"j="<<j<<endl;
				Du += (-4.00/Dt*pow(-1,j)*Landa[j+iteri-1]);
			}
			axpy_prod(CuuT,Du,Fu,false);
		#endif
		//cout<<"Dutotal "<<iteri<<" ="<<DuTotal[iteri]<<endl;
		#ifndef Dynf0valFTU
			Fu += 2.00*DuTotal[iteri];
		#else
			#ifdef DynSBSf0valDispStories
				for (size_t istory=0;istory<NStory;istory++)
					Fu(Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart)) += 2.00*DuTotal[iteri](Index((SHeights[istory]/Elemwid+1)*(Lenpart+1)*2-2-Lenpart))*ImpFac;
			#else
				#ifdef DynSBSf0valUTKHatU
					Fu += 2.00*axpy_prod(KHatT,DuTotal[iteri],HelperUNum)*ImpFac;
				#else
					#ifdef DynSBSf0valUTKU
						Fu += 2.00*axpy_prod(KuuT,DuTotal[iteri],HelperUNum)*ImpFac;
					#else
						#ifdef DynSBSf0valUTMU
							Fu += 2.00*axpy_prod(MassT,DuTotal[iteri],HelperUNum)*ImpFac;
						#else
							Fu(Index(LEP)) += 2.00*pow(1.00,2.00)*DuTotal[iteri](Index(LEP))*ImpFac;
						#endif
					#endif
				#endif
			#endif
		#endif
		//cout<<"Fu="<<Fu<<endl;
		Taucs_Factor_Solve(KHat,&F,2,Fu,Landa[iteri-1]);
		//cout<<"Landa "<<iteri<<" is done"<<endl;
		//cout<<Landa[iteri-1]<<endl;
	}
	if (MyMPIRank == 0)
		cout<<"Landas are built"<<endl;

	double ro[IntegPoints.size1()];
	//double Massro[IntegPoints.size1()];
	if (MyMPIRank == 0)
		cout<<"Entering filtering sensitivity analysis"<<endl;
//#pragma omp parallel for private(ro) num_threads(2) schedule(guided)
	for (size_t xiter=MyMPIRank;xiter<n;xiter += OurMPISize)
	{
		if (MyMPIRank == 0)
			printf ("%5.2f %%",xiter/(n-1.00)*100.00);
		Matrix kelem(8,8);
		//Matrix melem(8,8);
		CMatrix Kuu(UNum,UNum);
		#ifdef DynDamping
			CMatrix Cuu(UNum,UNum);
		#endif
		//CMatrix dMassdx(UNum,UNum);

		Kuu.clear();
		//dMassdx.clear();
		for (unsigned int roi=0;roi<IntegPoints.size1();roi++)
		{
			ro[roi]=SimpPower*pow(Ro(xiter),SimpPower-1.00);
			//Massro[roi]=1.00;
		}
		Element.kBuild(ro,kelem);
		//Element.mbuild(Massro,melem);
		size_t ii,jj;
		//cout<<Element.k<<endl;
		for (size_t i=0;i<8;i++)
			for (size_t j=0;j<8;j++)
			{

				ii = Connectivity(xiter,i);
				jj = Connectivity(xiter,j);

				if (Index(ii) < UNum && Index(jj) < UNum)
				{
					Kuu(Index(ii),Index(jj)) 	 += kelem(i,j);
					//dMassdx(Index(ii),Index(jj)) += melem(i,j);
				}
			}
		Vector helper1(UNum);
		Vector helper2(UNum);
		#ifdef DynDamping
			Cuu=Kuu*CoefDK;
		#endif
		for (size_t jiter=0;jiter<TimeIterN-1;jiter++)
		{
			axpy_prod(-Kuu,DuTotal[jiter+1],helper1);
			#ifdef DynDamping
				axpy_prod(-Cuu,DudotTotal[jiter+1],helper1,false);
			#endif
			//helper1 -= axpy_prod(dMassdx,Du2dotTotal[jiter+1],helper2);
			df0dxMPI(xiter) += prec_inner_prod(helper1,Landa[jiter]);
			#ifdef DynSBSf0valUTKHatU
				#ifdef DynDamping
					df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu*(1.00+2.00/Dt*CoefDK),DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#else
					df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#endif
			#else
				#ifdef DynSBSf0valUTKU
					df0dxMPI(xiter) += prec_inner_prod(axpy_prod(Kuu,DuTotal[jiter+1],helper1),DuTotal[jiter+1])*ImpFac;
				#endif
			#endif
		}
		if (iter==0)							//dfdx is equal for this purpose
			dfdxMPI(0,xiter)=Elemlen*Elemwid; 		//for edge ro modifications
		//cout<<xiter<<endl;
		if (MyMPIRank == 0)
			printf ("\b\b\b\b\b\b\b");
	}

	MPI_Allreduce(df0dxMPI.data().begin(),df0dx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
	if (iter==0)
		MPI_Allreduce(dfdxMPI.data().begin(),dfdx.data().begin(),n,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	delete[] Landa;
	Taucs_FreeFactor(&F);
	if (MyMPIRank == 0)
		cout<<"\bFiltering Sensitivity analysis is done"<<endl;
	return true;

}

bool DynamicOptimization::DynSBSElFilterOptimize()
{
	int OptTyp=2;

	size_t NStory;
	ifstream DynInSBS("DynamicOpt/DynSBS/DynInSBS.txt");
	DynInSBS>>NStory;
	double SHeights[NStory];
	double SMasses[NStory];
	for (size_t IterInput=0;IterInput<NStory;IterInput++)
	{
		DynInSBS>>SHeights[IterInput];
		DynInSBS>>SMasses[IterInput];
		//cout<<IterInput<<" : "<<SHeights[IterInput]<<" , "<<SMasses[IterInput]<<endl;
	}

	BuildSBSMassMatrix(NStory,SMasses,SHeights);

	Vector helperm(1);
	helperm(0)=1.00;
	Vector* DudotTotal=new Vector[TimeIterN];
	Vector* Du2dotTotal=new Vector[TimeIterN];

	for (size_t i=0;i<TimeIterN;i++)
	{
		DuTotal[i].resize(UNum);
		#ifdef DynDamping
			DudotTotal[i].resize(UNum);
		#endif
		Du2dotTotal[i].resize(UNum);
	}
	ifstream DynInFilter("DynamicOpt/DynSBS/DynSBSFiltering/DynSBSInFilter.txt");
	ofstream DynOutFilter("DynamicOpt/DynSBS/DynSBSFiltering/DynSBSOutEFilter.txt");
#ifdef DynRoEFilterPrint
	ofstream DynOutXFilter("DynamicOpt/DynSBS/DynSBSFiltering/DynSBSOutEXFilter.txt");
#endif
#ifdef DyndfEFilterDebug
	ofstream DynOutdfFilter("DynamicOpt/DynSBS/DynSBSFiltering/DynSBSOutdfEFilter.txt");
#endif
#ifdef DynFDElFiltering
	ofstream DynFDdfElFOut("DynamicOpt/DynSBS/DynSBSFiltering/DynSBSOutFDdfEFilter.txt");
#endif
	double FilterRadius;
	while (DynInFilter>>FilterRadius)
	{
		iter=0;
		while (/*(iter<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
		{
			if (MyMPIRank == 0)
				cout<<"Filter Radius= "<< FilterRadius<<endl;
			DynOutFilter<<"Filter Radius= "<< FilterRadius<<endl;
/*#ifdef DynRoEFilterPrint
			DynOutXFilter<<"Filter Radius= "<< FilterRadius<<endl;
#endif*/
			if (MyMPIRank == 0)
				cout<<"iter number"<<iter<<endl;
			DynOutFilter<<"iter number"<<iter<<endl;
/*#ifdef DynRoEFilterPrint
			DynOutXFilter<<"iter number"<<iter<<endl;
#endif*/
			#ifdef DynSBSFillStorySolid
				for (size_t istory=0;istory<NStory;istory++)
				{
					for (size_t jstory=0;jstory<MDLenpart+1;jstory++)
						Ro(size_t(SHeights[istory]/(Width/(MDWidpart+1))-1)*(MDLenpart+1)+jstory)=xval(size_t(SHeights[istory]/(Width/(MDWidpart+1))-1)*(MDLenpart+1)+jstory)=1.00;
				}
				//DynOutXval<<" before iter= "<<iter<<" Md="<<MDLenpart<<" , "<<MDWidpart<<" f0val ,fval= "<<f0val<<" , "<<fval<<endl<<MDLenpart<<" , "<<MDWidpart<<endl<<xval<<endl<<endl;
			#endif
			MakeStiffness(OptTyp);
#ifdef DynSBSRigidStory
			DynModifySBSStiffness(NStory,SHeights);
#endif

			f0valOld=f0val;
			DynSBSAnalyse(DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);

			CalResidue();

#ifdef DyndfEFilterDebug
			DynOutdfFilter<<"before df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl;
#endif
#ifdef DynFDElFiltering
			double f0val2=f0val;
			for (size_t FD_DEF=0;FD_DEF<n;FD_DEF++)
			{
				Ro(FD_DEF) += 1.00e-7;
				MakeStiffness(OptTyp);
				#ifdef DynSBSRigidStory
					DynModifySBSStiffness(NStory,SHeights);
				#endif
				//BuildMassMatrix(OptTyp);
				DynSBSAnalyse(DuTotal,DudotTotal,Du2dotTotal,NStory,SHeights);
				DynFDdfElFOut<<"i = "<<FD_DEF<<" : "<<df0dx(FD_DEF)<<" , "<<(f0val-f0val2)/1e-7<<endl;
				Ro(FD_DEF) -= 1e-7;
			}
			DynFDdfElFOut<<"iter num "<<iter<<endl<<"_____________________________________"<<endl;
			f0val=f0val2;
#endif

			axpy_prod(dfdx,Ro,fval);											//in our case dfdx=area of each Ro
			fval -= xPrimitive*Lenpart*Widpart*Elemlen*Elemwid*helperm;
			ElFilterSenstvts(FilterRadius);
#ifdef DyndfEFilterDebug
			DynOutdfFilter<<" after df0dx in iter "<<iter<<" = "<<df0dx<<endl<<endl<<endl<<endl;
#endif
			if (/*(iter+1<maxite)||*/(f0valDifRate < -fAllowDifRate) || (xvalMaxDifRate > xAllowDifRate) || (iter<=1))
			{
				mmasub();
				#ifdef CutXTails
					CutXTail();
				#endif
				Ro=xval;
			}
			if (MyMPIRank == 0)
				cout<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			DynOutFilter<<"f0val and fval = "<<f0val<<" , "<<fval<<endl;
			if (MyMPIRank == 0)
				cout<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
			DynOutFilter<<"f0val residue= "<<f0valDifRate<<", Xval Residue= "<<xvalMaxDifRate<<endl;
			DynOutFilter<<"__________________________________________________________________"<<endl<<endl;
			if (MyMPIRank == 0)
				cout<<"__________________________________________________________________"<<endl<<endl;
			DynOutFilter<<"__________________________________________________________________"<<endl<<endl;
#ifdef DynRoEFilterPrint
			DynOutXFilter<<"(*"<<MDLenpart<<","<<MDWidpart<<","<<iter<<"*) a={";
			for (size_t roiter=0;roiter<n;roiter++)
			{
				DynOutXFilter<<xval(roiter);
				if (roiter<n-1)
					DynOutXFilter<<",";
			}
			DynOutXFilter<<"};"<<endl;
#endif

			iter++;
		}
	}
	delete[] DuTotal;
	delete[] DudotTotal;
	delete[] Du2dotTotal;
	return true;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
#ifdef MPISensitivityAnalisys
	MPI_Init(&argc, &argv);
	TimeMPI=MPI_Wtime();
	//int MyMPIRank;
	//int OurMPISize;
	MPI_Comm_rank(MPI_COMM_WORLD, &MyMPIRank);
	MPI_Comm_size(MPI_COMM_WORLD, &OurMPISize);
	//printf ("MMMMYYYYY IIISSSSS %d va SSSSIIIIZZZZEEEE=%d\n",MyMPIRank,OurMPISize);
#else
	TimeMPI=MPI_Wtime();
#endif
	make();
#ifndef OptimizationisDynamic
	//cout<<Connectivity<<endl;
	Optimization opt;
	opt.Optimize();
	opt.InnerOptimize();
	//opt.FilterOptimize();
	//opt.ElFilterOptimize();
#else
	DynamicOptimization DynOpt;
	#ifdef FilterDynOpt
		//DynOpt.DynFilterOptimize();
		DynOpt.DynSBSFilterOptimize();
	#endif
	#ifdef ElFilterDynOpt
		//DynOpt.DynElFilterOptimize();
		DynOpt.DynSBSElFilterOptimize();
	#endif
	DynOpt.DynOptimize();
	DynOpt.DynInnerOptimize();

	//DynOpt.DynSBSOptimize();
	//DynOpt.DynSBSInnerOptimize();

#endif
	printf("[%u]: Execution time: %f\n", MyMPIRank, MPI_Wtime() - TimeMPI);
#ifdef MPISensitivityAnalisys
	MPI_Finalize();
#endif
	/*for(boost::numeric::ublas::symmetric_matrix<double,boost::numeric::ublas::lower>.iterator1 i1;i1=a.begin1();i1 != a.end1())
		for(boost::numeric::ublas::symmetric_matrix<double,boost::numeric::ublas::lower>.iterator2 i2;i2=i1.begin();i1 != i1.end())
			cout<<*i2<<endl;*/


	/*CSMatrix a(4,4);
	Vector b(4);
	b.clear();
	b(0)=10;
	a(0,0)=1;
	//a(0,1)=2;
	//a(1,0)=4;
	a(1,1)=2;
	//a(1,2)=5;
	a(2,2)=3;
	a(3,3)=4;
	cout<<b.data()[0]<<endl;
	for (size_t i=0 ;i<7;i++)
	{
		cout<<"value= "<<(a.value_data())[i]<<"  index1="<<(a.index1_data())[i]<<" index2="<<(a.index2_data())[i]<<endl;
	}

	void *F=NULL;
	Taucs_Factor_Solve(a,&F);
	Vector x(4);
	Taucs_Factor_Solve(a,&F,2,b,x);
	Vector help(4);
	axpy_prod(a,x,help);
	cout<<help<<endl;*/
	/*element TestElem(0,0,Elemlen,0,Elemlen,Elemwid,0,Elemwid);
	cout<<"size1  "<<NodesNum*2<<endl;
	//cout<<"""""""BBBBBBBEEEEEEEFFFFFFFFOOOOOOOOORRRRRRRRRRR"""""""<<endl;
	Vector testro(4);
	for (size_t i=0;i<4;i++)
		testro(i)=0.027;
	TestElem.kBuild(testro);
	BuildStiffness(TestElem);
	CGSolve(Kuu,Fu,Du);
	cout<<"Du="<<Du<<endl;
	cout<<IndexRev<<" =Indexrev"<<endl;
	cout<<Index<<" =Index"<<endl;
	cout<<Connectivity<<"=connectivity"<<endl;
	cout<<Nodes<<"=nodes"<<endl;
	cout<<TestElem.k<<endl;
	cout<<Fu<<"=Fu"<<endl;
	cout<<"Kuu="<<Kuu<<endl;
	cout<<Stiffness<<" =stiffness"<<endl;
	cout<<D<<"=D"<<endl;*/
	/*ofstream test("test.txt");


	cout<<opt.Ro<<endl;
	for (size_t i=0;i<opt.n;i++)
	{
		opt.Ro(i)=(i+1.00)/(opt.n+1.00);
	}
	opt.xval=opt.Ro;
	//opt.Optimize();

	opt.MakeStiffness();
	cout<<norm_inf(Kuu-trans(Kuu))<<" =norm"<<endl;
	CGSolve(Kuu,Fu,Du);
	double f1=prec_inner_prod(Disps,Loads);
	opt.SensitvtAnalyz();
	test<<opt.df0dx<<endl<<endl;
	test<<"-------------------------------------------------------------------------"<<endl<<endl;
	//opt.Optimize();
	for (size_t i=0;i<opt.n;i++)
	{
		opt.Ro=opt.xval;

		opt.Ro(i)+=.000001;
		Kuu.clear();
		opt.MakeStiffness();
		CGSolve(Kuu,Fu,Du);
		double f2=prec_inner_prod(Disps,Loads);
		test<<"df0dx("<<i<<")= "<<(f2-f1)/.000001<<endl;
	}*/

	//opt.MakeStiffness();
	//cout<<Kuu<<endl;
	//ofstream out("stiffness.txt");
	//out<<Stiffness;
	/*for(iterator1 iter1 = Stiffness.begin1(); iter1 != Stiffness.end1(); ++iter1)
		//#ifndef BOOST_UBLAS_NO_NESTED_CLASS_RELATION
		for (iterator2 iter2 = iter1.begin(); iter2 != iter1.end(); ++iter2)
		//#else
		//for (iterator2 iter2 = begin(iter1, boost::numeric::ublas::iterator1_tag()); iter1 != end(iter1, boost::numeric::ublas::iterator1_tag()); ++iter2)
		//#endif
			cout << "K(" << iter2.index1() << ", " << iter2.index2() << ") = " << *iter2 << endl;*/

	//CGSolve(Kuu,Fu,Du);
	//cout<<Du<<endl;
	//////////////////////////////////////////////////////////////////////////////////////////////
	/*CMatrix test(100,100);
	for (int i=0;i<100;++i)

		for (int j=0;j<100;j++)
			{
				if(i==j)
					test(i,i)=i;
				test(i,j)=(i+1)*(j+1);
			}
	test(0,0)=4;
	test(1,1)=10;
	test(2,2)=12;
	test(3,3)=13;
	boost::numeric::ublas::permutation_matrix<double> pm(4);
	Vector testinv(4);
	testinv(0)=2;
	testinv(2)=3;
	cout<<test<<endl;
	cout<<pm<<endl;
	lu_factorize(test,pm);
	cout<<"test = "<<test<<endl;
	cout<<"pm=  "<<pm<<endl;
	//testinv.assign(boost::numeric::ublas::identity_matrix<double>(4));
	lu_substitute(test,pm,testinv);
	cout<<testinv<<endl;*/
	/////////////////////////////////////////////////////////////////////////////
	cout<<"-------------------------------------------------"<<endl;
	/*int jj=0;
	for(int j=0;j<2*NodesNum;j++)

		if(Index(i)<UNum & Index(j)<UNum)
		{
			if (jj%10==0)
				cout<<endl;
			cout<<Kuu(Index(i),Index(j))-Stiffness(i,j);

			jj++;
		}*/

	/*mma ma;
	//cout<<ma.df0dx<<" =df0dx"<<endl;
	for (size_t i=0;i<ma.n;i++)
	{
		ma.df0dx(i)=ma.df0dx2(i)=i-2.00;
		for(size_t j=0;j<ma.m;j++)
			ma.dfdx(j,i)=ma.dfdx2(j,i)=i*j;
		if(i<ma.m)
			ma.fval(i)=i-2.00;
	}

	ma.f0val=2;
	ma.mmasub();*/



	//cout<<ma.xval<<" =xval"<<endl;

	/*ifstream testmat("testmat.txt");
	size_t sz1,sz2;
	double help;
	testmat>>sz1>>sz2;
	CMatrix tmatrix(sz1,sz2);
	for (size_t i=0;i<sz1;i++)
		for (size_t j=0;j<sz2;j++)
		{
			testmat>>help;
			tmatrix(i,j)=help;
		}
	ifstream testvec("testvec.txt");
	size_t sz;
	testvec>>sz;
	Vector tvector(sz);
	for (size_t i=0;i<sz;i++)
		{
			testvec>>help;
			tvector(i)=help;
		}
	Vector result(sz1);
	CGSolve(tmatrix,tvector,result,false);
	cout<<"CGSolve= "<<result<<endl;*/
	/*CMatrix testmat(5,5);
	Vector testvec(5);
	Vector result(5);
	for (size_t i=0;i<5;i++)
		{
			testmat(i,i)=i+1.00;
			testvec(i)=i+1.00;
		}
	CGSolve(testmat,testvec,result,false);
	cout<<result<<endl;*/

	//cout<<"Execution lasted "<<taucs_ctime()<<" seconds"<<endl;
	return 0;
}
