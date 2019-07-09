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
 *  Test Program of the nesting between Win_Farm_GPU and Win_MapReduce_GPU (CB windows with
 *  MAP phase on the GPU)
 *  
 *  Test program of the nesting between the Win_Farm_GPU and the Win_MapReduce_GPU pattern
 *  instantiated with a non-incremental REDUCE phase on the CPU while the MAP phase in on GPU.
 *  The query computes the sum of the value attribute of all the tuples in the window.
 *  The sliding window specification uses the count-based model.
 */

// includes
#include <string>
#include <iostream>
#include <ff/ff.hpp>
#include <windflow.hpp>
#include <windflow_gpu.hpp>
#include <sum_cb.hpp>

using namespace ff;
using namespace std;

// main
int main(int argc, char *argv[])
{
	int option = 0;
	size_t stream_len = 0;
	size_t win_len = 0;
	size_t win_slide = 0;
	size_t num_keys = 1;
	size_t wf_degree = 1;
	size_t map_degree = 1;
	size_t reduce_degree = 1;
	size_t batch_len = 1;
	// arguments from command line
	if (argc != 17) {
		cout << argv[0] << " -l [stream_length] -k [num keys] -w [win length] -s [win slide] -r [MAP pardegree] -n [REDUCE pardegree] -m [WLQ pardegree] -b [batch_len]" << endl;
		exit(EXIT_SUCCESS);
	}
	while ((option = getopt(argc, argv, "l:k:w:s:r:n:m:b:")) != -1) {
		switch (option) {
			case 'l': stream_len = atoi(optarg);
					 break;
			case 'k': num_keys = atoi(optarg);
					 break;
			case 'w': win_len = atoi(optarg);
					 break;
			case 's': win_slide = atoi(optarg);
					 break;
			case 'r': wf_degree = atoi(optarg);
					 break;					 
			case 'n': map_degree = atoi(optarg);
					 break;
			case 'm': reduce_degree = atoi(optarg);
					 break;
			case 'b': reduce_degree = atoi(optarg);
					 break;
			default: {
				cout << argv[0] << " -l [stream_length] -k [num keys] -w [win length] -s [win slide] -r [MAP pardegree] -n [REDUCE pardegree] -m [WLQ pardegree] -b [batch_len]" << endl;
				exit(EXIT_SUCCESS);
			}
        }
    }
	// user-defined map function (Non-Incremental Query on GPU)
	auto F = [] __host__ __device__ (size_t wid, const tuple_t *data, tuple_t *res, size_t size, char *memory) {
		long sum = 0;
		for (size_t i=0; i<size; i++) {
			sum += data[i].value;
		}
		res->value = sum;
	};
	// user-defined reduce function (Non-Incremental Query)
	auto G = [](size_t wid, Iterable<tuple_t> &input, tuple_t &win_result) {
		long sum = 0;
		for (auto t: input) {
			int val = t.value;
			sum += val;
		}
		win_result.value = sum;
	};
	// creation of the Win_MapReduce_GPU pattern
	auto *wm_gpu = WinMapReduceGPU_Builder<decltype(F), decltype(G)>(F, G).withCBWindows(win_len, win_slide)
													     .withParallelism(map_degree, reduce_degree)
													     .withBatch(batch_len)
													     .withName("test_sum")
													     .withOptLevel(LEVEL)
													     .build_ptr();
	// creation of the Win_Farm_GPU pattern
	auto *wf_gpu = WinFarmGPU_Builder<decltype(*wm_gpu)>(*wm_gpu).withParallelism(wf_degree)
													             .withName("test_sum")
													             .withOptLevel(LEVEL)
													             .build_ptr();
	// creation of the pipeline
	Generator generator(stream_len, num_keys);
	Consumer consumer(num_keys);
	ff_Pipe<tuple_t, tuple_t> pipe(generator, *wf_gpu, consumer);
	cout << "Starting ff_pipe with cardinality " << pipe.cardinality() << "..." << endl;
	if (pipe.run_and_wait_end() < 0) {
		cerr << "Error execution of ff_pipe" << endl;
		return -1;
	}
	else {
		cout << "...end ff_pipe" << endl;
		return 0;
	}
	return 0;
}
