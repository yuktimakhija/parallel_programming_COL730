#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "stdlib.h"
#include "mpi.h"
#include <unordered_map>
#include <math.h>
using namespace std;

bool compare(vector<double> I, vector<double> I_prev, int start, int end, int nprocs);
int endpoint(int rank, int nprocs, int N);
void mapper_function(int my_rank, void *ms_ptr);
void reducer_function(void *ms_ptr);

struct mr_struct{
	int num_page, my_rank, nprocs, assigned_start_r, assigned_end_r;
	vector <double> I,I_prev;
	float **stochastic_matrix;
	// unordered_map<int, vector<int>> conn_list;
};

double beta = 0.15;

int main(int narg, char *args[])
{
	if (narg<2){
	throw invalid_argument("Boo hoo");
	}
	MPI_Init(&narg,&args);

	int my_rank, nprocs ;
	MPI_Comm_rank(MPI_COMM_WORLD , &my_rank );
	MPI_Comm_size(MPI_COMM_WORLD , &nprocs );

	string arg_name = args[1];  
	string fname = "test/"+arg_name+".txt";
	int num_page=-1; 
	fstream infile(fname);
	int x,y,ymax=0;
	// unordered_map<int, vector<int>> conn_list;
	vector<int> from, to, num_connections;
	while (infile>>x>>y){ // x -> y
		from.push_back(x);
		to.push_back(y);
		// if (conn_list.find(x) == conn_list.end()){
		// 	vector<int> temp_vector;
		// 	conn_list[x] = temp_vector;
		// }
		// conn_list[x].push_back(y);
		if (x>num_page){
			for (int i = num_page+1;i<x;i++)
			num_connections.push_back(0);
			num_connections.push_back(1);
			num_page = x;
		}
		else{
			num_connections[x]++;
		}
		if (y>ymax){
			ymax = y;
		}    
	}
	infile.close();
	if (num_page>ymax){
		num_page++;
	}
	else{
	for (int i=num_page+1;i<=ymax;i++){
		num_connections.push_back(0);
	}
	num_page = ymax+1;
	}
	
	float *stochastic[num_page];
	int total_connections = to.size();
	vector<double> I_prev;
	vector<double> I;
	for (int i = 0; i < num_page; i++)
	{
		stochastic[i] = new float[num_page];
		I.push_back(0.0);
		I_prev.push_back(0.0);
	}
	I[0] = 1.0;
	I_prev[0] = 1.0;
	// stochastic[my_rank*num_page/nprocs] to stochastic[(my_rank+1)*num_page/nprocs]
	for (int i=endpoint(my_rank,nprocs,num_page);i<endpoint(my_rank+1,nprocs,num_page);i++){
		if (num_connections[i] == 0){
			for (int j=0;j<num_page;j++) 
				stochastic[i][j] = 1.0/num_page;
		}
		else{
			for (int j=0;j<num_page;j++) 
				stochastic[i][j] = 0.0;
		}
	}
	
	for (int i=endpoint(my_rank,nprocs,total_connections);i<endpoint(my_rank+1,nprocs,total_connections);i++){
		if ((from[i]>=endpoint(my_rank,nprocs,num_page)) && (from[i]<endpoint(my_rank+1,nprocs,num_page))){
			stochastic[from[i]][to[i]] = 1.0/num_connections[from[i]];
		}
	}


	MPI_Barrier(MPI_COMM_WORLD);
	mr_struct *ms = new  mr_struct{.num_page=num_page, .my_rank = my_rank, .nprocs = nprocs, .I = I, .I_prev = I_prev };
//   mr_struct *ms = (mr_struct *) malloc(sizeof(mr_struct));
//   ms->num_page = num_page;
//   ms->my_rank = my_rank;
//   ms->nprocs = nprocs;
//   for (int i = 0; i < num_page; i++)
//   {
//     ms->I.push_back(I[i]);
//     ms->I_prev.push_back(I_prev[i]);
//   }
	// ms->I = I;
	// ms->I_prev = I_prev;
	// ms->assigned_start_r = num_page;
	// ms->assigned_end_r = 0;
	ms->stochastic_matrix = (float**) malloc(num_page*sizeof(float *));
	for (int i = 0; i < num_page; i++)
	{
		// ms->stochastic_matrix[i] = (double*) malloc(num_page*sizeof(double));
		ms->stochastic_matrix[i] = stochastic[i];
	}
	double tstart = MPI_Wtime();
	int k = 0;
	int assigned_starts[nprocs], assigned_counts[nprocs];
	while (k==0 || !compare(ms->I,ms->I_prev,endpoint(my_rank,nprocs,num_page), endpoint(my_rank+1,nprocs,num_page), nprocs)){
		k++;
		ms->I_prev = ms->I;
		mapper_function(my_rank, ms);
		// mapreduce->map(nprocs, mapper_function, ms);
		// mapreduce->collate(NULL);
		// mapreduce->reduce(reducer_function, ms);
		// mapreduce->gather(1);
		reducer_function(ms);
		// int assigned_count = ms->assigned_end_r - ms->assigned_start_r + 1;
		// MPI_Allgather(&ms->assigned_start_r, 1, MPI_INT, assigned_starts, 1, MPI_INT, MPI_COMM_WORLD);
		// MPI_Allgather(&assigned_count, 1, MPI_INT, assigned_counts, 1, MPI_INT, MPI_COMM_WORLD);
				
		// MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, &((ms->I)[0]), assigned_counts, assigned_starts, MPI_DOUBLE, MPI_COMM_WORLD);

	}
	
	double tstop = MPI_Wtime();
	cout<<endl<<"Taken: "<<k<<" iterations (time: "<<(tstop-tstart)<<" s)"<<endl;
	
	MPI_Finalize();

	return 0;
}

bool compare(vector<double> I, vector<double> I_prev, int start, int end, int nprocs){
	bool flag = true;
	for (int i=start;i<end && i<I.size();i++){
		if (I[i] - I_prev[i] > 1e-7){
			flag = false;
			break;
		}
	}
	bool buffer[nprocs];
	MPI_Allgather(&flag, 1, MPI_C_BOOL, buffer, 1, MPI_C_BOOL, MPI_COMM_WORLD);
	for (int i = 0; i < nprocs; i++){
		flag = flag && buffer[i];
	}
	return flag;
}

int endpoint(int rank, int nprocs, int N){
	return min( ceil(((float)rank)/nprocs*N), (float)N);
}

void mapper_function(int my_rank, void *ms_ptr){
	mr_struct *ms = (mr_struct*) ms_ptr;
	fill(ms->I.begin(), ms->I.end(), 0);
	for (int i = endpoint(my_rank, ms->nprocs, ms->num_page); i < endpoint(my_rank+1, ms->nprocs, ms->num_page); i++){
		// for (auto it = conn_list[i].begin(); it != conn_list[i].end(); it++){
		// 	if (*it != find())
		// 	 val_at_key += beta/num_page * I_prev[*it] + (1-beta)*stochastic_matrix[*it][i]*I_prev[*it]; 
		// }
		for (int j = 0; j < ms->num_page; j++){
			ms->I[j] += ::beta/ms->num_page * ms->I_prev[i] + (1- ::beta)*ms->stochastic_matrix[i][j]*ms->I_prev[i];
		}
	}
}

void reducer_function(void *ms_ptr){
	mr_struct *ms = (mr_struct*) ms_ptr;
	MPI_Allreduce(MPI_IN_PLACE, &(ms->I[0]), ms->num_page, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}