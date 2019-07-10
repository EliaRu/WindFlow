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
 *  Test Program of the Win_Seq Pattern (CB windows with an Incremental Query)
 *  
 *  Test program of the Win_Seq pattern instantiated with an incremental query.
 *  The query computes the sum of the value attribute of all the tuples in the window.
 *  The sliding window specification uses the count-based model.
 */ 

// includes
#include <string>
#include <iostream>
#include <ff/ff.hpp>
#include <windflow.hpp>
#include <builders.hpp>
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
	// arguments from command line
	if (argc != 9) {
		cout << argv[0] << " -l [stream_length] -k [num keys] -w [win length] -s [win slide]" << endl;
		exit(EXIT_SUCCESS);
	}
	while ((option = getopt(argc, argv, "l:k:w:s:")) != -1) {
		switch (option) {
			case 'l': stream_len = atoi(optarg);
					 break;
			case 'k': num_keys = atoi(optarg);
					 break;
			case 'w': win_len = atoi(optarg);
					 break;
			case 's': win_slide = atoi(optarg);
					 break;
			default: {
				cout << argv[0] << " -l [stream_length] -k [num keys] -w [win length] -s [win slide]" << endl;
				exit(EXIT_SUCCESS);
			}
        }
    }
	// user-defined window function (Incremental Query)
	auto F = [](size_t wid, const tuple_t &t, output_t &result) {
		result.value += t.value;
	};
        auto L = [] (size_t key, uint64_t id, const tuple_t& t, output_t& res ) {
            res.key = key;
            res.id = id;
            res.value = t.value;
            return 0;
        };
        auto C = [] ( size_t key, uint64_t id, const output_t& a, const output_t &b, output_t &res ) {
            res.key = key;
            res.id = id;
            res.value = a.value + b.value;
            return 0;
        };
	// creation of the Win_Seq pattern
        using winseq_t = Win_Seq<tuple_t, output_t>;
	Win_Seq seq = WinSeq_Builder(F).withCBWindows(win_len, win_slide)
								   .withName("test_sum")
								   .build();	

	// creation of the pipeline
        cout << "Incremental Win_Seq without Flat FAT" << endl;
	Generator generator(stream_len, num_keys);
	Consumer consumer(num_keys);
	ff_Pipe<tuple_t, output_t> pipe(generator, seq, consumer);
	cout << "Starting ff_pipe with cardinality " << pipe.cardinality() << "..." << endl;
	if (pipe.run_and_wait_end() < 0) {
		cerr << "Error execution of ff_pipe" << endl;
		return -1;
	}
	else {
		cout << "...end ff_pipe" << endl;
	}
    
        cout << "Incremental Win_Seq with Flat FAT" << endl;
        winseq_t seq1 = winseq_t( 
            L, C, true, win_len, win_slide, "test_sum" 
        );
	Generator generator1(stream_len, num_keys);
	Consumer consumer1(num_keys);
	ff_Pipe<tuple_t, output_t> pipe1(generator1, seq1, consumer1);
	cout << "Starting ff_pipe with cardinality " << pipe1.cardinality() << "..." << endl;
	if (pipe1.run_and_wait_end() < 0) {
		cerr << "Error execution of ff_pipe" << endl;
		return -1;
	}
	else {
		cout << "...end ff_pipe" << endl;
	}
        return 0;
}
