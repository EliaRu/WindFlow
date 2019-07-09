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
 *  @file    wf_nodes.hpp
 *  @author  Gabriele Mencagli
 *  @date    01/10/2018
 *  
 *  @brief Emitter and Collector nodes of the Win_Farm and Win_Farm_GPU patterns
 *  
 *  @section Win_Farm_Nodes (Description)
 *  
 *  This file implements the Emitter and the Collector nodes used in the Win_Farm
 *  and Win_Farm_GPU patterns in the library.
 */ 

#ifndef WF_NODES_H
#define WF_NODES_H

// includes
#include <vector>
#include <ff/multinode.hpp>

using namespace ff;
using namespace std;

// class WF_Emitter
template<typename tuple_t, typename input_t=tuple_t>
class WF_Emitter: public ff_monode_t<input_t, wrapper_tuple_t<tuple_t>>
{
private:
    // type of the wrapper of input tuples
    using wrapper_in_t = wrapper_tuple_t<tuple_t>;
    tuple_t tmp; // never used
    // key data type
    using key_t = typename remove_reference<decltype(std::get<0>(tmp.getControlFields()))>::type;
    win_type_t winType; // type of the windows (CB or TB)
    uint64_t win_len; // window length (in no. of tuples or in time units)
    uint64_t slide_len; // window slide (in no. of tuples or in time units)
    size_t pardegree; // parallelism degree (number of inner patterns)
    size_t id_outer; // identifier in the outermost pattern
    size_t n_outer; // parallelism degree in the outermost pattern
    uint64_t slide_outer; // sliding factor utilized by the outermost pattern
    role_t role; // role of the innermost pattern
    vector<size_t> to_workers; // vector of identifiers used for scheduling purposes
    // struct of a key descriptor
    struct Key_Descriptor
    {
        uint64_t rcv_counter; // number of tuples received of this key
        tuple_t last_tuple; // copy of the last tuple received of this key

        // Constructor
        Key_Descriptor(): rcv_counter(0) {}
    };
    unordered_map<key_t, Key_Descriptor> keyMap; // hash table that maps a descriptor for each key
    bool isCombined; // true if this node is used within a treeComb node
    vector<pair<wrapper_in_t *, int>> output_queue; // used in case of treeComb mode

public:
    // Constructor
    WF_Emitter(win_type_t _winType, uint64_t _win_len, uint64_t _slide_len, size_t _pardegree, size_t _id_outer, size_t _n_outer, uint64_t _slide_outer, role_t _role):
               winType(_winType),
               win_len(_win_len),
               slide_len(_slide_len),
               pardegree(_pardegree),
               id_outer(_id_outer),
               n_outer(_n_outer),
               slide_outer(_slide_outer),
               role(_role),
               to_workers(pardegree),
               isCombined(false) {}

    // svc_init method (utilized by the FastFlow runtime)
    int svc_init()
    {
        return 0;
    }

    // svc method (utilized by the FastFlow runtime)
    wrapper_in_t *svc(input_t *wt)
    {
        // extract the key and id/timestamp fields from the input tuple
        tuple_t *t = extractTuple<tuple_t, input_t>(wt);
        auto key = std::get<0>(t->getControlFields()); // key
        size_t hashcode = hash<decltype(key)>()(key); // compute the hashcode of the key
        uint64_t id = (winType == CB) ? std::get<1>(t->getControlFields()) : std::get<2>(t->getControlFields()); // identifier or timestamp
        // access the descriptor of the input key
        auto it = keyMap.find(key);
        if (it == keyMap.end()) {
            // create the descriptor of that key
            keyMap.insert(make_pair(key, Key_Descriptor()));
            it = keyMap.find(key);
        }
        Key_Descriptor &key_d = (*it).second;
        // check duplicate or out-of-order tuples
        if (key_d.rcv_counter == 0) {
            key_d.rcv_counter++;
            key_d.last_tuple = *t;
        }
        else {
            // tuples can be received only ordered by id/timestamp
            uint64_t last_id = (winType == CB) ? std::get<1>((key_d.last_tuple).getControlFields()) : std::get<2>((key_d.last_tuple).getControlFields());
            if (id < last_id) {
                // the tuple is immediately deleted
                deleteTuple<tuple_t, input_t>(wt);
                return this->GO_ON;
            }
            else {
                key_d.rcv_counter++;
                key_d.last_tuple = *t;
            }
        }
        // gwid of the first window of that key assigned to this Win_Farm
        uint64_t first_gwid_key = (id_outer - (hashcode % n_outer) + n_outer) % n_outer;
        // initial identifer/timestamp of the keyed sub-stream arriving at this Win_Farm
        uint64_t initial_id = first_gwid_key * slide_outer;
        // special cases: role is WLQ or REDUCE
        if (role == WLQ || role == REDUCE)
            initial_id = 0;
        // if the id/timestamp of the tuple is smaller than the initial one, it must be discarded
        if (id < initial_id) {
            deleteTuple<tuple_t, input_t>(wt);
            return this->GO_ON;
        }
        // determine the range of local window identifiers that contain t
        long first_w = -1;
        long last_w = -1;
        // sliding or tumbling windows
        if (win_len >= slide_len) {
            if (id+1-initial_id < win_len)
                first_w = 0;
            else
                first_w = ceil(((double) (id + 1 - win_len - initial_id))/((double) slide_len));
            last_w = ceil(((double) id + 1 - initial_id)/((double) slide_len)) - 1;
        }
        // hopping windows
        else {
            uint64_t n = floor((double) (id-initial_id) / slide_len);
            // if the tuple belongs to at least one window of this Win_Farm
            if (id-initial_id >= n*(slide_len) && id-initial_id < (n*slide_len)+win_len) {
                first_w = last_w = n;
            }
            else {
                // delete the received tuple
                deleteTuple<tuple_t, input_t>(wt);
                return this->GO_ON;
            }
        }
        // determine the set of internal patterns that will receive the tuple
        uint64_t countRcv = 0;
        uint64_t i = first_w;
        // the first window of the key is assigned to worker startDstIdx
        size_t startDstIdx = hashcode % pardegree;
        while ((i <= last_w) && (countRcv < pardegree)) {
            to_workers[countRcv] = (startDstIdx + i) % pardegree;
            countRcv++;
            i++;
        }
        // prepare the wrapper to be sent
        wrapper_in_t *out = prepareWrapper<input_t, wrapper_in_t>(wt, countRcv);
        // for each destination we send the same wrapper
        for (size_t i = 0; i < countRcv; i++) {
            if (!isCombined)
                this->ff_send_out_to(out, to_workers[i]);
            else
                output_queue.push_back(make_pair(out, to_workers[i]));
        }
        return this->GO_ON;
    }

    // method to manage the EOS (utilized by the FastFlow runtime)
    void eosnotify(ssize_t id)
    {
        // iterate over all the keys
        for (auto &k: keyMap) {
            Key_Descriptor &key_d = k.second;
            if (key_d.rcv_counter > 0) {
                // send the last tuple to all the internal patterns as an EOS marker
                tuple_t *t = new tuple_t();
                *t = key_d.last_tuple;
                wrapper_in_t *wt = new wrapper_in_t(t, pardegree, true); // eos marker enabled
                for (size_t i=0; i < pardegree; i++) {
                    if (!isCombined)
                        this->ff_send_out_to(wt, i);
                    else
                        output_queue.push_back(make_pair(wt, i));
                }
            }
        }
    }

    // svc_end method (utilized by the FastFlow runtime)
    void svc_end() {}

    // get the number of destinations
    size_t getNDestinations()
    {
        return pardegree;
    }

    // set/unset the treeComb mode
    void setTreeCombMode(bool _val)
    {
        isCombined = _val;
    }

    // method to get a reference to the internal output queue (used in treeComb mode)
    vector<pair<wrapper_in_t *, int>> &getOutputQueue()
    {
        return output_queue;
    }
};

// class WF_Collector
template<typename result_t>
class WF_Collector: public ff_minode_t<result_t, result_t>
{
private:
    result_t tmp; // never used
    // key data type
    using key_t = typename remove_reference<decltype(std::get<0>(tmp.getControlFields()))>::type;
    // inner struct of a key descriptor
    struct Key_Descriptor
    {
        uint64_t next_win; // next window to be transmitted of that key
        deque<result_t *> resultsSet; // deque of buffered results of that key

        // Constructor
        Key_Descriptor(): next_win(0) {}
    };
    // hash table that maps key identifiers onto key descriptors
    unordered_map<key_t, Key_Descriptor> keyMap;

public:
    // Constructor
    WF_Collector() {}

    // svc_init method (utilized by the FastFlow runtime)
    int svc_init()
    {
        return 0;
    }

    // svc method (utilized by the FastFlow runtime)
    result_t *svc(result_t *r)
    {
        // extract key and identifier from the result
        auto key = std::get<0>(r->getControlFields()); // key
        uint64_t wid = std::get<1>(r->getControlFields()); // identifier
        // find the corresponding key descriptor
        auto it = keyMap.find(key);
        if (it == keyMap.end()) {
            // create the descriptor of that key
            keyMap.insert(make_pair(key, Key_Descriptor()));
            it = keyMap.find(key);
        }
        Key_Descriptor &key_d = (*it).second;
        uint64_t &next_win = key_d.next_win;
        deque<result_t *> &resultsSet = key_d.resultsSet;
        // add the new result at the correct place
        if ((wid - next_win) >= resultsSet.size()) {
            size_t new_size = (wid - next_win) + 1;
            resultsSet.resize(new_size, nullptr);
        }
        resultsSet[wid - next_win] = r;
        // scan all the buffered results and emit the ones in order
        auto itr = resultsSet.begin();
        for (; itr < resultsSet.end(); itr++) {
            if (*itr != nullptr) {
                this->ff_send_out(*itr);
                next_win++;
            }
            else break;
        }
        // delete the entries of the emitted results
        resultsSet.erase(resultsSet.begin(), itr);
        return this->GO_ON;
    }

    // svc_end method (utilized by the FastFlow runtime)
    void svc_end() {}
};

#endif
