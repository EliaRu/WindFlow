/******************************************************************************
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

/** 
 *  @file    standard.hpp
 *  @author  Gabriele Mencagli
 *  @date    12/02/2019
 *  
 *  @brief Standard nodes used by the WindFlow library
 *  
 *  @section Standard Nodes (Description)
 *  
 *  This file implements a set of nodes used by some of the patterns in the library in their
 *  default configuration.
 */ 

#ifndef STANDARD_H
#define STANDARD_H

// includes
#include <ff/multinode.hpp>

using namespace ff;

// class of the standard emitter
template<typename tuple_t>
class standard_emitter: public ff_monode_t<tuple_t>
{
private:
    // function type to map the key onto an identifier starting from zero to pardegree-1
    using f_routing_t = function<size_t(size_t, size_t)>;
    // friendships with other classes in the library
    template<typename T1>
    friend class Sink;
    template<typename T1, typename T2>
    friend class Map;
    template<typename T1>
    friend class Filter;
    template<typename T1, typename T2>
    friend class FlatMap;
    friend class MultiPipe;
    bool isKeyed; // flag stating whether the key-based distribution is used or not
    f_routing_t routing; // routing function

    // private constructor I
    standard_emitter(): isKeyed(false) {}

    // private constructor II
    standard_emitter(f_routing_t _routing): isKeyed(true), routing(_routing) {}

    // svc_init method (utilized by the FastFlow runtime)
    int svc_init()
    {
        return 0;
    }

    // svc method (utilized by the FastFlow runtime)
    tuple_t *svc(tuple_t *t)
    {
    	if (isKeyed) { // keyed-based distribution enabled
	        // extract the key from the input tuple
	        size_t key = std::get<0>(t->getInfo()); // key
	        // evaluate the routing function
	        size_t dest_w = routing(key, this->get_num_outchannels());
	        // sent the tuple
	        this->ff_send_out_to(t, dest_w);
	        return this->GO_ON;
    	}
    	else // default distribution
    		return t;
    }

    // svc_end method (FastFlow runtime)
    void svc_end() {}
};

// class of the standard collector
class standard_collector: public ff_minode
{
    void *svc(void *t) { return t; }
};

#endif
