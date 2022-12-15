#include <iostream>
#include <sstream>
#include <vector>
#include <fstream>
#include <string>
#include <chrono>
#include "stdlib.h"
#include "stdio.h"
#include "mapreduce/include/mapreduce.hpp"
#include <iomanip>
#include <unordered_map>

using namespace std;
using namespace std::chrono;

int num_page;
vector<double> I, I_prev;
float **stochastic_matrix;
// unordered_map<int, vector<int>> conn_list;
double beta=0.15;

bool compare(vector<double> I, vector<double> I_prev, int start, int end);

template<typename MapTask>
class initial_data_source : mapreduce::detail::noncopyable{
	int counter;
	int N;
 	public:
		bool const setup_key(typename MapTask::key_type &key){
			key=counter;
			counter+=1;
			return key<N;
		}
		bool const get_data(typename MapTask::key_type const &key,typename MapTask::value_type &value_at_key){
			value_at_key=I[key];
			return true;
		}
		initial_data_source(int num_page){
			counter=0;
			N=num_page;
		}
};

struct custom_map_task:public mapreduce::map_task<int,double>{
	template<typename Runtime>
	void operator()(Runtime &runtime,key_type const &key,value_type const &value) const{
		double val_at_key=0;
		// for (auto it = conn_list[key].begin(); it != conn_list[key].end(); it++){
		// 	if (*it != find())
		// 	 val_at_key += beta/num_page * I_prev[j] + (1-beta)*stochastic_matrix[j][key]*I_prev[j]; 
		// }
		for (int j = 0; j < num_page; j++){
			val_at_key += beta/num_page * I_prev[j] + (1-beta)*stochastic_matrix[j][key]*I_prev[j]; 
		}
		runtime.emit_intermediate(key,val_at_key);
	}
};

struct custom_reduce_task:public mapreduce::reduce_task<int,double>{
	template<typename Runtime,typename It>
	void operator()(Runtime &runtime,key_type const &key,It it,It ite) const{
		double reduced_val=0;
		for(;it!=ite;it++){
			reduced_val+=(*it);
		}
		runtime.emit(key,reduced_val);
	}
};

typedef mapreduce::job<custom_map_task, 
	custom_reduce_task, 
	mapreduce::null_combiner, 
	initial_data_source<custom_map_task>> custom_mr_job;


int main(int narg, char **args)
{
	if (narg<2){
		throw invalid_argument("Boo hoo");
	}
	
	string arg_name = args[1];  
	string fname = "test/"+arg_name+".txt";
	num_page=-1; 
	fstream infile(fname);
	int x,y,ymax=0;
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
	
	// double *stochastic[num_page];
	stochastic_matrix = (float**) malloc(num_page*sizeof(float *));
	int total_connections = to.size();
	vector<double> empty_init_vec(num_page, 0.0);
	// vector<double> I;
	I = empty_init_vec;
	I_prev = empty_init_vec;
	for (int i = 0; i < num_page; i++)
	{
		stochastic_matrix[i] = (float*) malloc(num_page*sizeof(float));
		// I.push_back(0.0);
		// I_prev.push_back(0.0);
	}
	I[0] = 1.0;
	I_prev[0] = 1.0;
	// stochastic[my_rank*num_page/nprocs] to stochastic[(my_rank+1)*num_page/nprocs]
	for (int i=0;i<num_page;i++){
		if (num_connections[i] == 0){
			for (int j=0;j<num_page;j++) 
				stochastic_matrix[i][j] = 1.0/num_page;
		}
		else{
			for (int j=0;j<num_page;j++) 
				stochastic_matrix[i][j] = 0.0;
		}
	}
	
	// for (int i=endpoint(my_rank,nprocs,total_connections);i<endpoint(my_rank+1,nprocs,total_connections);i++){
	for (int i=0;i<total_connections;i++){
		stochastic_matrix[from[i]][to[i]] = 1.0/num_connections[from[i]];
	}

	int k = 0;

	mapreduce::specification mr_spec;
    mapreduce::results mr_res;

	auto tstart = high_resolution_clock::now();

	while (k==0 || !compare(I,I_prev,0,num_page)){
		k++;
		I_prev = I;

		custom_mr_job::datasource_type init(num_page);
		custom_mr_job job(init,mr_spec);
		// job.run<mapreduce::schedule_policy::sequential<custom_mr_job>>(mr_res);
		job.run<mapreduce::schedule_policy::cpu_parallel<custom_mr_job>>(mr_res);

		for(auto iter=job.begin_results();iter!=job.end_results();iter++)
			I[iter->first]=iter->second;

	}

	auto tend = high_resolution_clock::now();
	fstream output_file;
	output_file.open("outputs/"+arg_name+"-pr-cpp.txt", fstream::out);
	double sum = 0.0;
	output_file<<setprecision(12);
	for (int i = 0; i < num_page; i++) {
		output_file<<i<<" = "<<I[i]<<endl;
		sum += I[i];
	}
	output_file<<"s = "<<sum<<endl;
	output_file.close();
	// cout<<endl<<"Took "<<iters_taken<<" iters in "<<tend - tstart<<" s"<<endl;
	float time_taken_microsecs = duration_cast<microseconds>(tend - tstart).count();
	cout<<endl<<"Taken: "<<k<<" iterations (time: "<<time_taken_microsecs/1000000.0<<" s)"<<endl;
	// float time_taken_microsecs = duration_cast<microseconds>(tend - tstart).count();
	// cout<<endl<<"Taken: "<<k<<" iterations (time: "<<time_taken_microsecs/1000000.0<<" s)"<<endl;
}

bool compare(vector<double> I, vector<double> I_prev, int start, int end){
	for (int i=start;i<end && i<I.size();i++){
		if (I[i] - I_prev[i] > 1e-7){
			return false;
		}
	}
	return true;
}