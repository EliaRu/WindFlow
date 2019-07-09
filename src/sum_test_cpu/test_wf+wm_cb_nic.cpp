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
 *  Test Program of the nesting between Win_Farm and Win_MapReduce (CB windows with a
 *  Non-Incremental Query)
 *  
 *  Test program of the nesting between the Win_Farm and the Win_MapReduce pattern
 *  instantiated with a non-incremental query. The query computes the sum of the
 *  value attribute of all the tuples in the window. The sliding window specification
 *  uses the count-based model.
 */ 

// includes
#include <string>
#include <iostream>
#include <ff/ff.hpp>
#include <windflow.hpp>
#include "sum_cb.hpp"

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
	// arguments from command line
	if (argc != 15) {
		cout << argv[0] << " -l [stream_length] -k [num keys] -w [win length] -s [win slide] -r [WF pardegree] -n [MAP pardegree] -m [REDUCE pardegree]" << endl;
		exit(EXIT_SUCCESS);
	}
	while ((option = getopt(argc, argv, "l:k:w:s:r:n:m:")) != -1) {
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
			default: {
				cout << argv[0] << " -l [stream_length] -k [num keys] -w [win length] -s [win slide] -r [WF pardegree] -n [MAP pardegree] -m [REDUCE pardegree]" << endl;
				exit(EXIT_SUCCESS);
			}
        }
    }
	// user-defined map and reduce functions (Non-Incremental Query)
	auto F = [](size_t wid, Iterable<tuple_t> &input, tuple_t &win_result) {
		long sum = 0;
		// print the window content
		for (auto t : input) {
			int val = t.value;
			sum += val;
		}
		win_result.value = sum;
	};
	// creation of the Win_MapReduce and Win_Farm patterns
	Win_MapReduce wm = WinMapReduce_Builder(F, F).withCBWindows(win_len, win_slide)
									.withParallelism(map_degree, reduce_degree)
									.withName("test_sum")
									.withOptLevel(LEVEL)
									.build();
	Win_Farm wf = WinFarm_Builder(wm).withParallelism(wf_degree)
									.withName("test_sum")
									.withOptLevel(LEVEL)
									.build();
	// creation of the pipeline
	Generator generator(stream_len, num_keys);
	Consumer consumer(num_keys);
	ff_Pipe<tuple_t, tuple_t> pipe(generator, wf, consumer);
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
