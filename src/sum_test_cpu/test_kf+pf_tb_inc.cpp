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
 *  Test Program of the nesting between Key_Farm and Pane_Farm (TB windows with an
 *  Incremental Query)
 *  
 *  Test program of the nesting between the Key_Farm and the Pane_Farm pattern
 *  instantiated with an incremental query. The query computes the sum of the
 *  value attribute of all the tuples in the window. The sliding window specification
 *  uses the time-based model.
 */ 

// includes
#include <string>
#include <iostream>
#include <ff/ff.hpp>
#include <windflow.hpp>
#include "sum_tb.hpp"

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
	size_t kf_degree = 1;
	size_t plq_degree = 1;
	size_t wlq_degree = 1;
	// arguments from command line
	if (argc != 15) {
		cout << argv[0] << " -l [stream_length] -k [num keys] -w [win length usec] -s [win slide usec] -r [KF pardegree] -n [PLQ pardegree] -m [WLQ pardegree]" << endl;
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
			case 'r': kf_degree = atoi(optarg);
					 break;					 
			case 'n': plq_degree = atoi(optarg);
					 break;
			case 'm': wlq_degree = atoi(optarg);
					 break;
			default: {
				cout << argv[0] << " -l [stream_length] -k [num keys] -w [win length usec] -s [win slide usec] -r [KF pardegree] -n [PLQ pardegree] -m [WLQ pardegree]" << endl;
				exit(EXIT_SUCCESS);
			}
        }
    }
    // user-defined pane function (Incremental Query)
	auto F = [](size_t key, size_t pid, const tuple_t &t, output_t &pane_result) {
		pane_result.key = key;
		pane_result.id = pid;
		pane_result.value += t.value;
		return 0;
	};
    // user-defined window function (Incremental Query)
	auto G = [](size_t key, size_t wid, const output_t &r, output_t &win_result) {
		win_result.key = key;
		win_result.id = wid;
		win_result.value += r.value;
		return 0;
	};
	// creation of the Pane_Farm and Key_Farm patterns
	Pane_Farm pf = PaneFarm_Builder(F, G).withTBWindow(microseconds(win_len), microseconds(win_slide))
									.withParallelism(plq_degree, wlq_degree)
									.withName("test_sum")
									.withOpt(LEVEL)
									.build();
	Key_Farm kf = KeyFarm_Builder(pf).withParallelism(kf_degree)
									.withName("test_sum")
									.withOpt(LEVEL)
									.build();
	// creation of the pipeline
	Generator generator(stream_len, num_keys);
	Consumer consumer(num_keys);
	ff_Pipe<tuple_t, output_t> pipe(generator, kf, consumer);
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
