/* *****************************************************************************
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *  
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 ******************************************************************************
 */

/*  
 *  Test of the MultiPipe construct
 *  
 *  Composition: Source(1) -> Filter(*) -> FlatMap(*) -> Map(*) -> WMR_GPU_TB(*) -> Sink(1)
 */ 

// includes
#include <string>
#include <iostream>
#include <random>
#include <math.h>
#include <ff/ff.hpp>
#include <windflow.hpp>
#include <windflow_gpu.hpp>

using namespace ff;
using namespace std;

// defines
#define RATIO 0.46566128e-9

// global variable for the result
long global_sum; 

// generation of pareto-distributed pseudo-random numbers
double pareto(double alpha, double kappa)
{
	double u;
	long seed = random();
	u = (seed) * RATIO;
	return (kappa / pow(u, (1. / alpha)));
}

// struct of the input tuple
struct tuple_t
{
	size_t key;
	uint64_t id;
	uint64_t ts;
	uint64_t value;

	// constructor
	tuple_t(size_t _key, uint64_t _id, uint64_t _ts, uint64_t _value): key(_key), id(_id), ts(_ts), value(_value) {}

	// default constructor
	tuple_t(): key(0), id(0), ts(0), value(0) {}

	// getInfo method
	tuple<size_t, uint64_t, uint64_t> getInfo() const
	{
		return tuple<size_t, uint64_t, uint64_t>(key, id, ts);
	}

	// setInfo method
	void setInfo(size_t _key, uint64_t _id, uint64_t _ts)
	{
		key = _key;
		id = _id;
		ts = _ts;
	}
};

// struct of the output data type
struct output_t
{
	size_t key;
	uint64_t id;
	uint64_t ts;
	uint64_t value;

	// default constructor
	output_t(): key(0), id(0), ts(0), value(0) {}

	// getInfo method
	tuple<size_t, uint64_t, uint64_t> getInfo() const
	{
		return tuple<size_t, uint64_t, uint64_t>(key, id, ts);
	}

	// setInfo method
	void setInfo(size_t _key, uint64_t _id, uint64_t _ts)
	{
		key = _key;
		id = _id;
		ts = _ts;
	}
};

// source functor
class Source_Functor
{
private:
	size_t len; // stream length per key
	size_t keys; // number of keys
	vector<uint64_t> next_ts;

public:
	// constructor
	Source_Functor(size_t _len, size_t _keys): len(_len), keys(_keys), next_ts(_keys, 0)
	{
		srand(0);
	}

	// operator()
	void operator()(Shipper<tuple_t> &shipper)
	{
		// generation of the input stream
		for (size_t i=0; i<len; i++) {
			for (size_t k=0; k<keys; k++) {
				tuple_t t(k, i, next_ts[k], i);
				double x = (1000 * 0.05) / 1.05;
				next_ts[k] += ceil(pareto(1.05, x));
				//next_ts[k] += 1000;
				shipper.push(t);
			}
		}
	}
};

// filter functor
class Filter_Functor
{
public:
	// operator()
	bool operator()(tuple_t &t)
	{
		// drop odd numbers
		if (t.value % 2 == 0)
			return true;
		else
			return false;
	}
};

// flatmap functor
class FlatMap_Functor
{
public:
	// operator()
	void operator()(const tuple_t &t, Shipper<tuple_t> &shipper)
	{
		// generate three items per input
		for (size_t i=0; i<3; i++) {
			tuple_t t2 = t;
			t2.value = t.value + i;
			shipper.push(t2);
		}
	}
};

// map functor
class Map_Functor
{
public:
	// operator()
	void operator()(tuple_t &t)
	{
		// double the value
		t.value = t.value * 2;
	}
};

// sink functor
class Sink_Functor
{
private:
	size_t received; // counter of received results
	long totalsum;
	size_t keys;
	vector<size_t> check_counters;

public:
	// constructor
	Sink_Functor(size_t _keys): received(0), totalsum(0), keys(_keys), check_counters(_keys, 0) {}

	// operator()
	void operator()(optional<output_t> &out)
	{
		if (out) {
			received++;
			totalsum += (*out).value;
			// check the ordering of results
			//if (check_counters[(*out).key] != (*out).id)
				//cout << "Results received out-of-order!" << endl;
			//else cout << "Received result window " << *out->id << " of key " << out->key << " with value " << (*out).value << endl;
			check_counters[(*out).key]++;	
		}
		else {
			LOCKED_PRINT("Received " << received << " window results, total sum " << totalsum << endl;)
			global_sum = totalsum;
		}
	}
};

// main
int main(int argc, char *argv[])
{
	int option = 0;
	size_t runs = 1;
	size_t stream_len = 0;
	size_t win_len = 0;
	size_t win_slide = 0;
	size_t n_keys = 1;
	size_t batch_len = 1;
	// initalize global variable
	global_sum = 0;
	// arguments from command line
	if (argc != 13) {
		cout << argv[0] << " -r [runs] -l [stream_length] -k [n_keys] -w [win length usec] -s [win slide usec] -b [batch len]" << endl;
		exit(EXIT_SUCCESS);
	}
	while ((option = getopt(argc, argv, "r:l:k:w:s:b:")) != -1) {
		switch (option) {
			case 'r': runs = atoi(optarg);
					 break;
			case 'l': stream_len = atoi(optarg);
					 break;
			case 'k': n_keys = atoi(optarg);
					 break;
			case 'w': win_len = atoi(optarg);
					 break;
			case 's': win_slide = atoi(optarg);
					 break;
			case 'b': batch_len = atoi(optarg);
					 break;
			default: {
				cout << argv[0] << " -r [runs] -l [stream_length] -k [n_keys] -w [win length usec] -s [win slide usec] -b [batch len]" << endl;
				exit(EXIT_SUCCESS);
			}
        }
    }
    // set random seed
    mt19937 rng;
    rng.seed(std::random_device()());
    size_t min = 1;
    size_t max = 10;
    std::uniform_int_distribution<std::mt19937::result_type> dist6(min, max);
    int filter_degree, flatmap_degree, map_degree, wmap_degree, reduce_degree;
    size_t source_degree = 1;
    long last_result = 0;
    // executes the runs
    for (size_t i=0; i<runs; i++) {
    	filter_degree = dist6(rng);
    	flatmap_degree = dist6(rng);
    	map_degree = dist6(rng);
    	wmap_degree = dist6(rng);
    	reduce_degree = dist6(rng);
    	if (wmap_degree == 1)
    		wmap_degree = 2; // map stage of the WMR must always be parallel
    	cout << "Run " << i << " Source(" << source_degree <<")->Filter(" << filter_degree << ")->FlatMap(" << flatmap_degree << ")->Map(" << map_degree << ")->Win_MapReduce_GPU_TB(" << wmap_degree << "," << reduce_degree << ")->Sink(1)" << endl;
	    // prepare the test
	    MultiPipe application("test_wmr_tb_gpu");
	    // source
	    Source_Functor source_functor(stream_len, n_keys);
	    auto *source = Source_Builder<decltype(source_functor)>(source_functor).withName("test_wmr_tb_gpu_source").withParallelism(source_degree).build_ptr();
	    application.add_source(*source);
	    // filter
	    Filter_Functor filter_functor;
	    auto *filter = Filter_Builder<decltype(filter_functor)>(filter_functor).withName("test_wmr_tb_gpu_filter").withParallelism(filter_degree).build_ptr();
	    application.add(*filter);
	    // flatmap
	    FlatMap_Functor flatmap_functor;
	    auto *flatmap = FlatMap_Builder<decltype(flatmap_functor)>(flatmap_functor).withName("test_wmr_tb_gpu_flatmap").withParallelism(flatmap_degree).build_ptr();
	    application.add(*flatmap);
	    // map
	    Map_Functor map_functor;
	    auto *map = Map_Builder<decltype(map_functor)>(map_functor).withName("test_wmr_tb_gpu_map").withParallelism(map_degree).build_ptr();
	    application.add(*map);
		// user-defined map function (Non-Incremental Query on GPU)
		auto map_function = [] __host__ __device__ (size_t key, size_t pid, const tuple_t *data, output_t *res, size_t size, char *memory) {
			long sum = 0;
			for (size_t i=0; i<size; i++) {
				sum += data[i].value;
			}
			res->key = key;
			res->id = pid;
			res->value = sum;
			return 0;
		};
		// user-defined reduce function (Non-Incremental Query)
		auto reduce_function = [](size_t key, size_t wid, Iterable<output_t> &input, output_t &win_result) {
			long sum = 0;
			for (auto t: input) {
				int val = t.value;
				sum += val;
			}
			win_result.key = key;
			win_result.id = wid;
			win_result.value = sum;
			return 0;
		};
	    // wmr
	    auto *wmr = WinMapReduceGPU_Builder<decltype(map_function), decltype(reduce_function)>(map_function, reduce_function).withName("test_wmr_tb_wmr").withParallelism(wmap_degree, reduce_degree).withTBWindow(microseconds(win_len), microseconds(win_slide)).withBatch(batch_len).build_ptr();
	    application.add(*wmr);
	    // sink
	    Sink_Functor sink_functor(n_keys);
	    auto *sink = Sink_Builder<decltype(sink_functor)>(sink_functor).withName("test_wmr_tb_gpu_sink").withParallelism(1).build_ptr();
	    application.add_sink(*sink);
	   	// run the application
	   	application.run_and_wait_end();
	   	if (i == 0) {
	   		last_result = global_sum;
	   		cout << "Result is --> " << GREEN << "OK" << "!!!" << DEFAULT << endl;
	   	}
	   	else {
	   		if (last_result == global_sum) {
	   			cout << "Result is --> " << GREEN << "OK" << "!!!" << DEFAULT << endl;
	   		}
	   		else {
	   			cout << "Result is --> " << RED << "FAILED" << "!!!" << DEFAULT << endl;
	   		}
	   	}
    }

	return 0;
}
