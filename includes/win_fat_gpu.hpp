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
 *  @file    
 *  @author  
 *  @date    
 *  
 *  @brief 
 *  
 *  @section 
 *  
 *  
 *  
 *  
 *  
 *  
 *  
 *  
 */ 

#ifndef WIN_FAT_GPU_H
#define WIN_FAT_GPU_H

// includes
#include <list>
#include <string>
#include <unordered_map>
#include <math.h>
#include <cassert>
#include <iostream>

#include <ff/node.hpp>
#include <window.hpp>
#include <stream_archive.hpp>
#include <batchedfat.hpp>

using namespace ff;


/** 
 *  \class 
 *  
 *  \brief 
 *  
 *  
 * 
 * 
 */ 
template<typename tuple_t, typename result_t, typename win_F_t, typename input_t>
class Win_FAT_GPU: public ff_node_t<input_t, result_t>
{
private:
    // const iterator type for accessing tuples
    using const_input_iterator_t = typename vector<tuple_t>::const_iterator;
    // type of the stream archive used by the Win_Seq_GPU pattern
    using archive_t = StreamArchive<result_t, vector<result_t>>;
    // window type used by the Win_Seq_GPU pattern
    using win_t = Window<tuple_t, result_t>;
    // function type to compare two tuples
    using f_compare_t = function<bool(const tuple_t &, const tuple_t &)>;
    using f_winlift_t =
        function<int( size_t, uint64_t, const tuple_t &, result_t & )>;
    // friendships with other classes in the library
    template<typename T1, typename T2, typename T3, typename T4>
    friend class Win_Farm_GPU;
    template<typename T1, typename T2, typename T3, typename T4>
    friend class Key_Farm_GPU;
    template<typename T1, typename T2, typename T3, typename T4>
    friend class Pane_Farm_GPU;
    template<typename T1, typename T2, typename T3, typename T4>
    friend class Win_MapReduce_GPU;
    // struct of a key descriptor
    struct Key_Descriptor
    {
        list<result_t> tuples;
        BatchedFAT<tuple_t, result_t, win_F_t> bfat;
        result_t acc;
        uint64_t cb_id;
        uint64_t last_quantum;
        list<win_t> wins; // open windows of this key
        uint64_t emit_counter; // progressive counter (used if role is PLQ or MAP)
        uint64_t rcv_counter; // number of tuples received of this key
        tuple_t last_tuple; // copy of the last tuple received of this key
        uint64_t next_lwid; // next window to be opened of this key (lwid)
        size_t batchedWin; // number of batched windows of the key
        vector<size_t> start, end; // vectors of initial/final positions of each window in the current micro-batch
        vector<uint64_t> gwids; // vector of gwid of the windows in the current micro-batch
        vector<uint64_t> tsWin; // vector of the final timestamp of the windows in the current micro-batch
        optional<tuple_t> start_tuple; // optional to the first tuple of the current micro-batch

        // constructor

        Key_Descriptor( 
            f_winlift_t _winLift,
            win_F_t _combine, 
            size_t _batchSize, 
            size_t _numWindows,
            size_t _windowSize, 
            size_t _slide,
            f_compare_t _compare,
            uint64_t _emit_counter = 0 )
        : tuples( ),
          bfat( _winLift, _combine, _batchSize, _numWindows,  _windowSize, _slide ),
          cb_id( 0 ),
          last_quantum( 0 ),
          emit_counter(_emit_counter),
          rcv_counter(0),
          next_lwid(0),
          batchedWin(0) 
        { 
            tuple_t tmp;
            _winLift( 0, 0, tmp, acc );
            
        }

        // move constructor
        Key_Descriptor(Key_Descriptor &&_k):
                       tuples( move( _k.tuples ) ),
                       bfat( move( _k.bfat ) ),
                       acc( move( _k.acc ) ),
                       cb_id( move( _k.cb_id ) ),
                       last_quantum( move ( _k.last_quantum ) ),
                       wins(move(_k.wins)),
                       emit_counter(_k.emit_counter),
                       rcv_counter(_k.rcv_counter),
                       last_tuple(_k.last_tuple),
                       next_lwid(_k.next_lwid),
                       batchedWin(_k.batchedWin),
                       start(_k.start),
                       end(_k.end),
                       gwids(_k.gwids),
                       tsWin(_k.tsWin),
                       start_tuple(_k.start_tuple) {}
    };
    // CPU variables
    f_compare_t compare; // function to compare two tuples
    uint64_t win_len; // window length (no. of tuples or in time units)
    uint64_t slide_len; // slide length (no. of tuples or in time units)
    win_type_t winType; // window type (CB or TB)
    string name; // string of the unique name of the pattern
    PatternConfig config; // configuration structure of the Win_Seq_GPU pattern
    role_t role; // role of the Win_Seq_GPU instance
    unordered_map<size_t, Key_Descriptor> keyMap; // hash table that maps a descriptor for each key
    pair<size_t, size_t> map_indexes = make_pair(0, 1); // indexes useful is the role is MAP
    size_t batch_len; // length of the micro-batch in terms of no. of windows (i.e. 1 window mapped onto 1 CUDA thread)
    size_t n_thread_block; // number of threads per block
    size_t tuples_per_batch; // number of tuples per batch (only for CB windows)
    win_F_t winFunction; // function to be executed per window
    f_winlift_t winLift;
    bool rebuildFAT;
    bool isFirstBatch;
    bool timebasedBFAT;
    uint64_t quantum;
    result_t *host_results = nullptr; // array of results copied back from the GPU
    // GPU variables
    cudaStream_t cudaStream; // CUDA stream used by this Win_Seq_GPU instance
    size_t no_thread_block; // number of CUDA threads per block
    tuple_t *Bin = nullptr; // array of tuples in the micro-batch (allocated on the GPU)
    result_t *Bout = nullptr; // array of results of the micro-batch (allocated on the GPU)
    size_t *gpu_start, *gpu_end = nullptr; // arrays of the starting/ending positions of each window in the micro-batch (allocated on the GPU)
    uint64_t *gpu_gwids = nullptr; // array of the gwids of the windows in the microbatch (allocated on the GPU)
    size_t scratchpad_size = 0; // size of the scratchpage memory area on the GPU (one per CUDA thread)
    char *scratchpad_memory = nullptr; // scratchpage memory area (allocated on the GPU, one per CUDA thread)
#if defined(LOG_DIR)
    bool isTriggering = false;
    unsigned long rcvTuples = 0;
    unsigned long rcvTuplesTriggering = 0; // a triggering tuple activates a new batch
    double avg_td_us = 0;
    double avg_ts_us = 0;
    double avg_ts_triggering_us = 0;
    double avg_ts_non_triggering_us = 0;
    volatile unsigned long startTD, startTS, endTD, endTS;
    ofstream *logfile = nullptr;
#endif

    Win_FAT_GPU(f_winlift_t _winLift,
                win_F_t _winFunction,
                uint64_t _win_len,
                uint64_t _slide_len,
                size_t _batch_len,
                bool _rebuildFAT,
                string _name,
                PatternConfig _config,
                role_t _role)
                :
                winLift( _winLift ),
                winFunction(_winFunction),
                win_len(_win_len),
                slide_len(_slide_len),
                winType( CB ),
                rebuildFAT( _rebuildFAT ),
                isFirstBatch( true ),
                timebasedBFAT( false ),
                batch_len(_batch_len),
                n_thread_block( 0 ),
                name(_name),
                scratchpad_size( 0 ),
                config(_config),
                role(_role)
    {
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the batch length
        if (_batch_len == 0) {
            cerr << RED << "WindFlow Error: batch length cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // create the CUDA stream
        if (cudaStreamCreate(&cudaStream) != cudaSuccess) {
            cerr << RED << "WindFlow Error: cudaStreamCreate() returns error code" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // define the compare function depending on the window type
        compare = [](const tuple_t &t1, const tuple_t &t2) {
            return std::get<1>(t1.getInfo()) < std::get<1>(t2.getInfo());
        };
    }

    Win_FAT_GPU(f_winlift_t _winLift,
                win_F_t _winFunction,
                uint64_t _win_len,
                uint64_t _slide_len,
                uint64_t _quantum,
                size_t _batch_len,
                bool _rebuildFAT,
                string _name,
                PatternConfig _config,
                role_t _role)
                :
                winLift( _winLift ),
                winFunction(_winFunction),
                win_len(_win_len),
                slide_len(_slide_len),
                quantum( _quantum ),
                winType( CB ),
                rebuildFAT( _rebuildFAT ),
                isFirstBatch( true ),
                timebasedBFAT( true ),
                batch_len(_batch_len),
                n_thread_block( 0 ),
                name(_name),
                scratchpad_size( 0 ),
                config(_config),
                role(_role)
    {
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the batch length
        if (_batch_len == 0) {
            cerr << RED << "WindFlow Error: batch length cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // create the CUDA stream
        if (cudaStreamCreate(&cudaStream) != cudaSuccess) {
            cerr << RED << "WindFlow Error: cudaStreamCreate() returns error code" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // define the compare function depending on the window type
        compare = [](const tuple_t &t1, const tuple_t &t2) {
            return std::get<1>(t1.getInfo()) < std::get<1>(t2.getInfo());
        };
    }

    // method to set the indexes useful if role is MAP
    void setMapIndexes(size_t _first, size_t _second) {
        map_indexes.first = _first;
        map_indexes.second = _second;
    }

public:
    Win_FAT_GPU(f_winlift_t _winLift,
                win_F_t _winFunction,
                uint64_t _win_len,
                uint64_t _slide_len,
                size_t _batch_len,
                bool _rebuildFAT,
                string _name )
                :
                Win_FAT_GPU( _winLift, _winFunction, _win_len, _slide_len, _batch_len, _rebuildFAT, _name, PatternConfig( 0, 1, _slide_len, 0, 1, _slide_len ), SEQ ) { }

    Win_FAT_GPU(f_winlift_t _winLift,
                win_F_t _winFunction,
                uint64_t _win_len,
                uint64_t _slide_len,
                uint64_t _quantum,
                size_t _batch_len,
                bool _rebuildFAT,
                string _name )
                :
                Win_FAT_GPU( _winLift, _winFunction, _win_len, _slide_len, _quantum, _batch_len, _rebuildFAT, _name, PatternConfig( 0, 1, _slide_len, 0, 1, _slide_len ), SEQ ) {  }

//@cond DOXY_IGNORE

    // svc_init method (utilized by the FastFlow runtime)
    int svc_init()
    {
        // initialization with count-based windows

        // compute the fixed number of tuples per batch
        if (slide_len <= win_len) // sliding or tumbling windows
            tuples_per_batch = (batch_len - 1) * slide_len + win_len;
        else // hopping windows
            tuples_per_batch = win_len * batch_len;
#if defined(LOG_DIR)
        logfile = new ofstream();
        name += "_seq_" + to_string(ff_node_t<input_t, result_t>::get_my_id()) + ".log";
        string filename = string(STRINGIFY(LOG_DIR)) + "/" + name;
        logfile->open(filename);
#endif
        return 0;
    }

    result_t *countbasedSvc( input_t *wt )
    {
#if defined (LOG_DIR)
        startTS = current_time_nsecs();
        if (rcvTuples == 0)
            startTD = current_time_nsecs();
        rcvTuples++;
#endif
        // extract the key and id/timestamp fields from the input tuple
        tuple_t *t = extractTuple<tuple_t, input_t>(wt);
        size_t key = std::get<0>(t->getInfo()); // key
        uint64_t id = std::get<1>(t->getInfo()); // identifier 
        // access the descriptor of the input key
        auto it = keyMap.find(key);
        if (it == keyMap.end()) {
            // create the descriptor of that key
            keyMap.insert(
                make_pair(
                    key, 
                    Key_Descriptor( 
                        winLift,
                        winFunction, 
                        tuples_per_batch, 
                        batch_len,
                        win_len, 
                        slide_len, 
                        compare,
                        role == MAP ? map_indexes.first : 0 
                    ) 
                )
            );
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
            uint64_t last_id = std::get<1>((key_d.last_tuple).getInfo());
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
        // gwid of the first window of that key assigned to this Win_Seq_GPU instance
        uint64_t first_gwid_key = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) * config.n_outer + (config.id_outer - (key % config.n_outer) + config.n_outer) % config.n_outer;
        // initial identifer/timestamp of the keyed sub-stream arriving at this Win_Seq_GPU instance
        uint64_t initial_outer = ((config.id_outer - (key % config.n_outer) + config.n_outer) % config.n_outer) * config.slide_outer;
        uint64_t initial_inner = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) * config.slide_inner;
        uint64_t initial_id = initial_outer + initial_inner;
        // special cases: if role is WLQ or REDUCE
        if (role == WLQ || role == REDUCE)
            initial_id = initial_inner;
        // if the id/timestamp of the tuple is smaller than the initial one, it must be discarded
        if (id < initial_id) {
            deleteTuple<tuple_t, input_t>(wt);
            return this->GO_ON;
        }
        // determine the local identifier of the last window containing t
        long last_w = -1;
        // sliding or tumbling windows
        if (win_len >= slide_len)
            last_w = ceil(((double) id + 1 - initial_id)/((double) slide_len)) - 1;
        // hopping windows
        else {
            uint64_t n = floor((double) (id-initial_id) / slide_len);
            last_w = n;
            // if the tuple does not belong to at least one window assigned to this Win_Seq instance
            if ((id-initial_id < n*(slide_len)) || (id-initial_id >= (n*slide_len)+win_len)) {
                // if it is not an EOS marker, we delete the tuple immediately
                if (!isEOSMarker<tuple_t, input_t>(*wt)) {
                    // delete the received tuple
                    deleteTuple<tuple_t, input_t>(wt);
                    return this->GO_ON;
                }
            }
        }

        auto &wins = key_d.wins;
        // create all the new windows that need to be opened by the arrival of t
        for (long lwid = key_d.next_lwid; lwid <= last_w; lwid++) {
            // translate the lwid into the corresponding gwid
            uint64_t gwid = first_gwid_key + (lwid * config.n_outer * config.n_inner);
            wins.push_back(win_t(key, lwid, gwid, Triggerer_CB(win_len, slide_len, lwid, initial_id), CB, win_len, slide_len));
            key_d.next_lwid++;
        }
        
        // last received tuple already in the archive
        auto& win = wins.front( );
        if( win.onTuple( *t ) == FIRED ) {
            key_d.batchedWin++;
            key_d.gwids.push_back( win.getGWID( ) );
            wins.pop_front( );
            if( key_d.batchedWin == batch_len ) {
                assert( key_d.gwids.size( ) == batch_len );
#if defined(LOG_DIR)
                rcvTuplesTriggering++;
                isTriggering = true;
#endif

                vector<result_t> batchedTuples( 
                    key_d.tuples.begin( ), 
                    key_d.tuples.end( )
                );
                if( rebuildFAT ) {
                    key_d.bfat.build( move( batchedTuples ), key, 0, 0 );
                    auto it = key_d.tuples.begin( );
                    for( size_t i = 0; i < batch_len * slide_len; i++ ) {
                        it++;
                    }
                    key_d.tuples.erase( 
                        key_d.tuples.begin( ),
                        it
                    );
                } else {
                    if( isFirstBatch ) {
                        key_d.bfat.build( move( batchedTuples ), key, 0, 0 );
                        isFirstBatch = false;
                    } else {
                        key_d.bfat.update( move( batchedTuples ), key, 0, 0 );
                    }
                    key_d.tuples.clear( );
                }
                auto results = key_d.bfat.getResults( );
                for( size_t i = 0; i < batch_len; i++ ) {
                    result_t *res = new result_t( results[i] );
                    res->key = key;
                    res->id = key_d.gwids[i];
                    if (role == MAP) {
                        res->setInfo(key, key_d.emit_counter, std::get<2>(res->getInfo()));
                        key_d.emit_counter += map_indexes.second;
                    }
                    else if (role == PLQ) {
                        uint64_t new_id = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) + (key_d.emit_counter * config.n_inner);
                        res->setInfo(key, new_id, std::get<2>(res->getInfo()));
                        key_d.emit_counter++;
                    }
                    this->ff_send_out(res);
                }
                key_d.batchedWin = 0;
                (key_d.gwids).clear();
            }
        }
        if (!isEOSMarker<tuple_t, input_t>(*wt)) {
            result_t res;
            winLift( key, t->id, *t, res );
            (key_d.tuples).push_back( res );
        }

        // delete the received tuple
        deleteTuple<tuple_t, input_t>(wt);
#if defined(LOG_DIR)
        endTS = current_time_nsecs();
        endTD = current_time_nsecs();
        double elapsedTS_us = ((double) (endTS - startTS)) / 1000;
        avg_ts_us += (1.0 / rcvTuples) * (elapsedTS_us - avg_ts_us);
        if (isTriggering)
            avg_ts_triggering_us += (1.0 / rcvTuplesTriggering) * (elapsedTS_us - avg_ts_triggering_us);
        else
            avg_ts_non_triggering_us += (1.0 / (rcvTuples - rcvTuplesTriggering)) * (elapsedTS_us - avg_ts_non_triggering_us);
        isTriggering = false;
        double elapsedTD_us = ((double) (endTD - startTD)) / 1000;
        avg_td_us += (1.0 / rcvTuples) * (elapsedTD_us - avg_td_us);
        startTD = current_time_nsecs();
#endif
        return this->GO_ON;
    }

    result_t *ManageWindow( Key_Descriptor& key_d, input_t *wt, tuple_t *t )
    {
        size_t key = std::get<0>(t->getInfo()); // key
        uint64_t id =  std::get<1>(t->getInfo()); // identifier or timestamp
        // gwid of the first window of that key assigned to this Win_Seq_GPU instance
        uint64_t first_gwid_key = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) * config.n_outer + (config.id_outer - (key % config.n_outer) + config.n_outer) % config.n_outer;
        // initial identifer/timestamp of the keyed sub-stream arriving at this Win_Seq_GPU instance
        uint64_t initial_outer = ((config.id_outer - (key % config.n_outer) + config.n_outer) % config.n_outer) * config.slide_outer;
        uint64_t initial_inner = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) * config.slide_inner;
        uint64_t initial_id = initial_outer + initial_inner;
        // special cases: if role is WLQ or REDUCE
        if (role == WLQ || role == REDUCE)
            initial_id = initial_inner;
        // if the id/timestamp of the tuple is smaller than the initial one, it must be discarded
        if (id < initial_id) {
            return this->GO_ON;
        }
        // determine the local identifier of the last window containing t
        long last_w = -1;
        // sliding or tumbling windows
        if (win_len >= slide_len)
            last_w = ceil(((double) id + 1 - initial_id)/((double) slide_len)) - 1;
        // hopping windows
        else {
            uint64_t n = floor((double) (id-initial_id) / slide_len);
            last_w = n;
            // if the tuple does not belong to at least one window assigned to this Win_Seq instance
            if ((id-initial_id < n*(slide_len)) || (id-initial_id >= (n*slide_len)+win_len)) {
                // if it is not an EOS marker, we delete the tuple immediately
                if (!isEOSMarker<tuple_t, input_t>(*wt)) {
                    // delete the received tuple
                    return this->GO_ON;
                }
            }
        }

        auto &wins = key_d.wins;
        // create all the new windows that need to be opened by the arrival of t
        for (long lwid = key_d.next_lwid; lwid <= last_w; lwid++) {
            // translate the lwid into the corresponding gwid
            uint64_t gwid = first_gwid_key + (lwid * config.n_outer * config.n_inner);
            wins.push_back(win_t(key, lwid, gwid, Triggerer_CB(win_len, slide_len, lwid, initial_id), CB, win_len, slide_len));
            key_d.next_lwid++;
        }
        
        // last received tuple already in the archive
        auto& win = wins.front( );
        if( win.onTuple( *t ) == FIRED ) {
            key_d.batchedWin++;
            key_d.gwids.push_back( win.getGWID( ) );
            wins.pop_front( );
            if( key_d.batchedWin == batch_len ) {
                assert( key_d.gwids.size( ) == batch_len );
#if defined(LOG_DIR)
                rcvTuplesTriggering++;
                isTriggering = true;
#endif

                vector<result_t> batchedTuples( 
                    key_d.tuples.begin( ), 
                    key_d.tuples.end( )
                );
                if( rebuildFAT ) {
                    key_d.bfat.build( move( batchedTuples ), key, 0, 0 );
                    auto it = key_d.tuples.begin( );
                    for( size_t i = 0; i < batch_len * slide_len; i++ ) {
                        it++;
                    }
                    key_d.tuples.erase( 
                        key_d.tuples.begin( ),
                        it
                    );
                } else {
                    if( isFirstBatch ) {
                        key_d.bfat.build( move( batchedTuples ), key, 0, 0 );
                        isFirstBatch = false;
                    } else {
                        key_d.bfat.update( move( batchedTuples ), key, 0, 0 );
                    }
                    key_d.tuples.clear( );
                }
                auto results = key_d.bfat.getResults( );
                for( size_t i = 0; i < batch_len; i++ ) {
                    result_t *res = new result_t( results[i] );
                    res->key = key;
                    res->id = key_d.gwids[i];
                    if (role == MAP) {
                        res->setInfo(key, key_d.emit_counter, std::get<2>(res->getInfo()));
                        key_d.emit_counter += map_indexes.second;
                    }
                    else if (role == PLQ) {
                        uint64_t new_id = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) + (key_d.emit_counter * config.n_inner);
                        res->setInfo(key, new_id, std::get<2>(res->getInfo()));
                        key_d.emit_counter++;
                    }
                    this->ff_send_out(res);
                }
                key_d.batchedWin = 0;
                (key_d.gwids).clear();
            }
        }
        if (!isEOSMarker<tuple_t, input_t>(*wt)) {
            (key_d.tuples).push_back( *reinterpret_cast<result_t*>( t ) );
        }
        return this->GO_ON;
    }

    result_t *timebasedSvc( input_t *wt )
    {
#if defined (LOG_DIR)
        startTS = current_time_nsecs();
        if (rcvTuples == 0)
            startTD = current_time_nsecs();
        rcvTuples++;
#endif
        // extract the key and id/timestamp fields from the input tuple
        tuple_t *t = extractTuple<tuple_t, input_t>(wt);
        size_t key = std::get<0>(t->getInfo()); // key
        static uint64_t base_ts = std::get<2>(t->getInfo()); // identifier or timestamp
        uint64_t ts =  std::get<2>(t->getInfo()) - base_ts; // identifier or timestamp
        // access the descriptor of the input key
        auto it = keyMap.find(key);
        if (it == keyMap.end()) {
            // create the descriptor of that key
            keyMap.insert(
                make_pair(
                    key,
                    Key_Descriptor( 
                        winLift,
                        winFunction, 
                        tuples_per_batch, 
                        batch_len,
                        win_len, 
                        slide_len, 
                        compare,
                        role == MAP ? map_indexes.first : 0 
                    ) 
                )
            );
            it = keyMap.find(key);
        }
        Key_Descriptor &key_d = (*it).second;
        if (key_d.rcv_counter == 0) {
            key_d.rcv_counter++;
            key_d.last_tuple = *t;
        }
        else {
            // tuples can be received only ordered by id/timestamp
            uint64_t last_ts = std::get<2>((key_d.last_tuple).getInfo()) - base_ts;
            if ( ts < last_ts ) {
                // the tuple is immediately deleted
                deleteTuple<tuple_t, input_t>(wt);
                return this->GO_ON;
            }
            else {
                key_d.rcv_counter++;
                key_d.last_tuple = *t;
            }
        }

        uint64_t which_quantum = ts / quantum;
        int64_t distance = which_quantum - key_d.last_quantum;
        for( int64_t i = 0; i < distance; i++ ) {
            key_d.acc.key = key;
            key_d.acc.ts = key_d.last_tuple.ts;
            key_d.acc.id = ( key_d.cb_id )++;
            ( void ) ManageWindow( 
                key_d, wt, reinterpret_cast<tuple_t*>( &( key_d.acc ) ) 
            );
            ( key_d.last_quantum )++;
            tuple_t tmp;
            winLift( key, 0, tmp, key_d.acc );
        }
        result_t tmp;
        winLift( key, 0, *t, tmp );
        winFunction( key, 0, key_d.acc, tmp, key_d.acc );

#if defined(LOG_DIR)
        endTS = current_time_nsecs();
        endTD = current_time_nsecs();
        double elapsedTS_us = ((double) (endTS - startTS)) / 1000;
        avg_ts_us += (1.0 / rcvTuples) * (elapsedTS_us - avg_ts_us);
        double elapsedTD_us = ((double) (endTD - startTD)) / 1000;
        avg_td_us += (1.0 / rcvTuples) * (elapsedTD_us - avg_td_us);
        startTD = current_time_nsecs();
#endif
        // delete the received tuple
        deleteTuple<tuple_t, input_t>(wt);
        return this->GO_ON;
    }

    // svc method (utilized by the FastFlow runtime)
    result_t *svc(input_t *wt)
    {
        if( timebasedBFAT ) {
            return timebasedSvc( wt );
        } else {
            return countbasedSvc( wt );
        }
    }

    void countbasedEosNotify( ssize_t id )
    {

        for( auto &k: keyMap ) {
            size_t key = k.first;
            Key_Descriptor &key_d = k.second;
            auto &wins = key_d.wins;
            list<result_t> tuples;
            if( !rebuildFAT && !isFirstBatch ) {
                tuples = key_d.bfat.getBatchedTuples( );
                for( size_t i = 0; i < batch_len * slide_len; i++ ) {
                    tuples.pop_front( );
                }
            }
            tuples.insert( 
                tuples.end( ), 
                key_d.tuples.begin( ), 
                key_d.tuples.end( ) 
            );
            for( auto gwid : key_d.gwids ) {
                
                result_t *res = new result_t( );
                auto it = tuples.begin( );
                for( size_t i = 0; i < win_len; it++, i++ ) 
                {
                    winFunction( key, gwid, *it, *res, *res );
                }
                res->key = key;
                res->id = gwid;
                if (role == MAP) {
                    res->setInfo(key, key_d.emit_counter, std::get<2>(res->getInfo()));
                    key_d.emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    uint64_t new_id = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) + (key_d.emit_counter * config.n_inner);
                    res->setInfo(key, new_id, std::get<2>(res->getInfo()));
                    key_d.emit_counter++;
                }
                for( size_t i = 0; i < slide_len; i++ ) {
                    tuples.pop_front( );
                }
                this->ff_send_out(res);
            }
            for( auto &win : wins ) {
                auto gwid = win.getGWID( );

                result_t *res = new result_t( );
                for( auto it = tuples.begin( );
                     it != tuples.end( ); 
                     it++ ) 
                {
                    winFunction( key, gwid, *it, *res, *res );
                }
                res->key = key;
                res->id = gwid;
                if (role == MAP) {
                    res->setInfo(key, key_d.emit_counter, std::get<2>(res->getInfo()));
                    key_d.emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    uint64_t new_id = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) + (key_d.emit_counter * config.n_inner);
                    res->setInfo(key, new_id, std::get<2>(res->getInfo()));
                    key_d.emit_counter++;
                }
                for( size_t i = 0; i < slide_len && tuples.size( ) > 0; i++ ) {
                    tuples.pop_front( );
                }
                this->ff_send_out(res);
            }
        }
    }

    void timebasedEosNotify( ssize_t id )
    {
        for( auto &k: keyMap ) {
            size_t key = k.first;
            Key_Descriptor &key_d = k.second;
            auto &wins = key_d.wins;

            key_d.acc.key = k.first;
            key_d.acc.ts = key_d.last_tuple.ts;
            key_d.acc.id = ( key_d.cb_id )++;
            tuple_t tmp;
            wrapper_tuple_t<tuple_t> t( &tmp );
            ( void ) ManageWindow( 
                key_d, nullptr, reinterpret_cast<tuple_t*>( &( key_d.acc ) ) 
            );

            list<result_t> tuples;
            if( !rebuildFAT && !isFirstBatch ) {
                tuples = key_d.bfat.getBatchedTuples( );
                for( size_t i = 0; i < batch_len * slide_len; i++ ) {
                    tuples.pop_front( );
                }
            }
            tuples.insert( 
                tuples.end( ), 
                key_d.tuples.begin( ), 
                key_d.tuples.end( ) 
            );
            for( auto gwid : key_d.gwids ) {
                
                result_t *res = new result_t( );
                auto it = tuples.begin( );
                for( size_t i = 0; i < win_len; it++, i++ ) 
                {
                    winFunction( key, gwid, *it, *res, *res );
                }
                res->key = key;
                res->id = gwid;
                if (role == MAP) {
                    res->setInfo(key, key_d.emit_counter, std::get<2>(res->getInfo()));
                    key_d.emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    uint64_t new_id = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) + (key_d.emit_counter * config.n_inner);
                    res->setInfo(key, new_id, std::get<2>(res->getInfo()));
                    key_d.emit_counter++;
                }
                for( size_t i = 0; i < slide_len; i++ ) {
                    tuples.pop_front( );
                }
                this->ff_send_out(res);
            }
            for( auto &win : wins ) {
                auto gwid = win.getGWID( );

                result_t *res = new result_t( );
                for( auto it = tuples.begin( );
                     it != tuples.end( ); 
                     it++ ) 
                {
                    winFunction( key, gwid, *it, *res, *res );
                }
                res->key = key;
                res->id = gwid;
                if (role == MAP) {
                    res->setInfo(key, key_d.emit_counter, std::get<2>(res->getInfo()));
                    key_d.emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    uint64_t new_id = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) + (key_d.emit_counter * config.n_inner);
                    res->setInfo(key, new_id, std::get<2>(res->getInfo()));
                    key_d.emit_counter++;
                }
                for( size_t i = 0; i < slide_len && tuples.size( ) > 0; i++ ) {
                    tuples.pop_front( );
                }
                this->ff_send_out(res);
            }
        }
    }

    // method to manage the EOS (utilized by the FastFlow runtime)
    void eosnotify(ssize_t id)
    {
        if( timebasedBFAT ) {
            return timebasedEosNotify( id );
        } else {
            return countbasedEosNotify( id );
        }
    }

    // svc_end method (utilized by the FastFlow runtime)
    void svc_end()
    {
        // deallocate data structures allocated on the GPU
#if defined (LOG_DIR)
        ostringstream stream;
        stream << "************************************LOG************************************\n";
        stream << "No. of received tuples: " << rcvTuples << "\n";
        stream << "No. of received tuples (triggering): " << rcvTuplesTriggering << "\n";
        stream << "Average service time: " << avg_ts_us << " usec \n";
        stream << "Average service time (triggering): " << avg_ts_triggering_us << " usec \n";
        stream << "Average service time (non triggering): " << avg_ts_non_triggering_us << " usec \n";
        stream << "Average inter-departure time: " << avg_td_us << " usec \n";
        stream << "***************************************************************************\n";
        *logfile << stream.str();
        logfile->close();
        delete logfile;
#endif
    }

//@endcond

    /** 
     *  \brief Get the window type (CB or TB) utilized by the pattern
     *  \return adopted windowing semantics (count- or time-based)
     */
    win_type_t getWinType() { return winType; }

    /// Method to start the pattern execution asynchronously
    virtual int run(bool)
    {
        return ff_node::run();
    }

    /// Method to wait the pattern termination
    virtual int wait()
    {
        return ff_node::wait();
    }
};

#endif
