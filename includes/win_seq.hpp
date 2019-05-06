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
 *  @file    win_seq.hpp
 *  @author  Gabriele Mencagli
 *  @date    30/06/2017
 *  
 *  @brief Win_Seq pattern executing a windowed transformation on a multi-core CPU
 *  
 *  @section Win_Seq (Description)
 *  
 *  This file implements the Win_Seq pattern able to execute windowed queries on a
 *  multicore. The pattern executes streaming windows in a serial fashion on a CPU
 *  core and supports both a non-incremental and an incremental query definition.
 *  
 *  The template arguments tuple_t and result_t must be default constructible, with a copy constructor
 *  and copy assignment operator, and they must provide and implement the setInfo() and
 *  getInfo() methods.
 */ 

#ifndef WIN_SEQ_H
#define WIN_SEQ_H

// includes
#include <vector>
#include <string>
#include <unordered_map>
#include <math.h>
#include <ff/node.hpp>
#include <window.hpp>
#include <builders.hpp>
#include <stream_archive.hpp>
#include <fat.hpp>

using namespace ff;

/** 
 *  \class Win_Seq
 *  
 *  \brief Win_Seq pattern executing a windowed transformation on a multi-core CPU
 *  
 *  This class implements the Win_Seq pattern executing windowed queries on a multicore
 *  in a serial fashion.
 */ 
template<typename tuple_t, typename result_t, typename input_t>
class Win_Seq: public ff_node_t<input_t, result_t>
{
public:
    /// function type of the non-incremental window processing
    using f_winfunction_t = function<int(size_t, uint64_t, Iterable<tuple_t> &, result_t &)>;
    /// function type of the incremental window processing
    using f_winupdate_t = function<int(size_t, uint64_t, const tuple_t &, result_t &)>;

    /// type of the functions that insert an element in the Flat FAT
    using f_winlift_t = 
        function<int(size_t, uint64_t, const tuple_t&, result_t&)>;
    /// function type of the incremental window processing used in the Flat FAT
    using f_wincombine_t =
        function<int(size_t, uint64_t, const result_t&, const result_t&, result_t&)>;
private:
    // const iterator type for accessing tuples
    using const_input_iterator_t = typename deque<tuple_t>::const_iterator;
    // type of the stream archive used by the Win_Seq pattern
    using archive_t = StreamArchive<tuple_t, deque<tuple_t>>;
    /// type of the Flat FAT
    using fat_t = FlatFAT<tuple_t, result_t>;
    // window type used by the Win_Seq pattern
    using win_t = Window<tuple_t, result_t>;
    // function type to compare two tuples
    using f_compare_t = function<bool(const tuple_t &, const tuple_t &)>;
    // friendships with other classes in the library
    template<typename T1, typename T2, typename T3>
    friend class Win_Farm;
    template<typename T1, typename T2, typename T3>
    friend class Key_Farm;
    template<typename T1, typename T2, typename T3>
    friend class Pane_Farm;
    template<typename T1, typename T2, typename T3>
    friend class Win_MapReduce;
    template<typename T1, typename T2, typename T3, typename T4>
    friend class Pane_Farm_GPU;
    template<typename T1, typename T2, typename T3, typename T4>
    friend class Win_MapReduce_GPU;
    // struct of a key descriptor
    struct Key_Descriptor
    {
        archive_t archive; // archive of tuples of this key
        fat_t fat; // Flat FAT of this key
        vector<win_t> wins; // open windows of this key
        uint64_t emit_counter; // progressive counter (used if role is PLQ or MAP)
        uint64_t rcv_counter; // number of tuples received of this key
        tuple_t last_tuple; // copy of the last tuple received of this key
        uint64_t next_lwid; // next window to be opened of this key (lwid)

        // constructor
        Key_Descriptor(f_compare_t _compare, uint64_t _emit_counter=0):
                       archive(_compare),
                       emit_counter(_emit_counter),
                       rcv_counter(0),
                       next_lwid(0)
        {
            wins.reserve(DEFAULT_VECTOR_CAPACITY);
        }

        Key_Descriptor(
            f_winlift_t _winLift,
            f_wincombine_t _winCombine,
            size_t _win_len,
            uint64_t _emit_counter = 0 
        ) : 
            fat( _winLift, _winCombine, _win_len ),
            emit_counter(_emit_counter),
            rcv_counter(0),
            next_lwid(0)
        {
            wins.reserve(DEFAULT_VECTOR_CAPACITY);
        }

        // move constructor
        Key_Descriptor(Key_Descriptor &&_k):
                       archive(move(_k.archive)),
                       fat(move(_k.fat)),
                       wins(move(_k.wins)),
                       emit_counter(_k.emit_counter),
                       rcv_counter(_k.rcv_counter),
                       next_lwid(_k.next_lwid) {}
    };
    f_winfunction_t winFunction; // function of the non-incremental window processing
    f_winupdate_t winUpdate; // function of the incremental window processing
    f_winlift_t winLift; // function for inserting an element in the Flat FAT
    f_wincombine_t winCombine; // function of the incremental window processing in the Flat FAT
    f_compare_t compare; // function to compare two tuples
    uint64_t win_len; // window length (no. of tuples or in time units)
    uint64_t slide_len; // slide length (no. of tuples or in time units)
    win_type_t winType; // window type (CB or TB)
    string name; // string of the unique name of the pattern
    bool isNIC; // this flag is true if the pattern is instantiated with a non-incremental query function
    bool useFlatFAT; // it is true if the object uses a Flat FAT 
    PatternConfig config; // configuration structure of the Win_Seq pattern
    role_t role; // role of the Win_Seq instance
    unordered_map<size_t, Key_Descriptor> keyMap; // hash table that maps a descriptor for each key
    pair<size_t, size_t> map_indexes = make_pair(0, 1); // indexes useful is the role is MAP
#if defined(LOG_DIR)
    bool isTriggering = false;
    unsigned long rcvTuples = 0;
    unsigned long rcvTuplesTriggering = 0;
    double avg_td_us = 0;
    double avg_ts_us = 0;
    double avg_ts_triggering_us = 0;
    double avg_ts_non_triggering_us = 0;
    volatile unsigned long startTD, startTS, endTD, endTS;
    ofstream *logfile = nullptr;
#endif

    // private constructor I (non-incremental queries)
    Win_Seq(f_winfunction_t _winFunction,
            uint64_t _win_len,
            uint64_t _slide_len,
            win_type_t _winType,
            string _name,
            PatternConfig _config,
            role_t _role)
            :
            winFunction(_winFunction),
            win_len(_win_len),
            slide_len(_slide_len),
            winType(_winType),
            isNIC(true),
            useFlatFAT( false ),
            name(_name),
            config(_config),
            role(_role)
    {
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // define the compare function depending on the window type
        if (winType == CB) {
            compare = [](const tuple_t &t1, const tuple_t &t2) {
                return std::get<1>(t1.getInfo()) < std::get<1>(t2.getInfo());
            };
        }
        else {
            compare = [](const tuple_t &t1, const tuple_t &t2) {
                return std::get<2>(t1.getInfo()) < std::get<2>(t2.getInfo());
            };
        }
    }

    // private constructor II (incremental queries)
    Win_Seq(f_winupdate_t _winUpdate,
            uint64_t _win_len,
            uint64_t _slide_len,
            win_type_t _winType,
            string _name,
            PatternConfig _config,
            role_t _role)
            :
            winUpdate(_winUpdate),
            win_len(_win_len),
            slide_len(_slide_len),
            winType(_winType),
            isNIC(false),
            useFlatFAT( false ),
            name(_name),
            config(_config),
            role(_role)
    {
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // define the compare function depending on the window type
        if (winType == CB) {
            compare = [](const tuple_t &t1, const tuple_t &t2) {
                return std::get<1>(t1.getInfo()) < std::get<1>(t2.getInfo());
            };
        }
        else {
            compare = [](const tuple_t &t1, const tuple_t &t2) {
                return std::get<2>(t1.getInfo()) < std::get<2>(t2.getInfo());
            };
        }
    }

    //private constructor III (incremental queries, it uses a Flat FAT)
    Win_Seq(f_winlift_t _winLift,
            f_wincombine_t _winCombine,
            uint64_t _win_len,
            uint64_t _slide_len,
            string _name,
            PatternConfig _config,
            role_t _role)
            :
            winLift( _winLift ),
            winCombine( _winCombine ),
            win_len(_win_len),
            slide_len(_slide_len),
            winType( CB ),
            isNIC(false),
            useFlatFAT( true ),
            name(_name),
            config(_config),
            role(_role)
    {
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        if( _win_len <= _slide_len ) {
            cerr << RED << "WindFlow Error: "
            << "FlatFAT implementation supports only sliding windows" 
            << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // define the compare function depending on the window type
        compare = [](const tuple_t &t1, const tuple_t &t2) {
            return std::get<1>(t1.getInfo()) < std::get<1>(t2.getInfo());
        };
    }

    // method to set the indexes useful if role is MAP
    void setMapIndexes(size_t _first, size_t _second) {
        map_indexes.first = _first; // id
        map_indexes.second = _second; // pardegree
    }

public:
    /** 
     *  \brief Constructor I (Non-Incremental Queries)
     *  
     *  \param _winFunction the non-incremental window processing function
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _name string with the unique name of the pattern
     */ 
    Win_Seq(f_winfunction_t _winFunction,
            uint64_t _win_len,
            uint64_t _slide_len,
            win_type_t _winType,
            string _name)
            :
            Win_Seq(_winFunction, _win_len, _slide_len, _winType, _name, PatternConfig(0, 1, _slide_len, 0, 1, _slide_len), SEQ) {}

    /** 
     *  \brief Constructor II (Incremental Queries)
     *  
     *  \param _winUpdate the incremental window processing function
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _name string with the unique name of the pattern
     */ 
    Win_Seq(f_winupdate_t _winUpdate,
            uint64_t _win_len,
            uint64_t _slide_len,
            win_type_t _winType,
            string _name)
            :
            Win_Seq(_winUpdate, _win_len, _slide_len, _winType, _name, PatternConfig(0, 1, _slide_len, 0, 1, _slide_len), SEQ) {}

    /** 
     *  \brief Constructor III (Incremental Queries and Flat FAT)
     *  
     *  \param _winLift the function that inserts a tuple in the FAT
     *  \param _winCombine the incremental window processing function
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _name string with the unique name of the pattern
     */ 
    Win_Seq(f_winlift_t _winLift,
            f_wincombine_t _winCombine,
            uint64_t _win_len,
            uint64_t _slide_len,
            string _name)
            :
            Win_Seq(_winLift, _winCombine, _win_len, _slide_len, _name, PatternConfig(0, 1, _slide_len, 0, 1, _slide_len), SEQ) {}
//@cond DOXY_IGNORE

    // svc_init method (utilized by the FastFlow runtime)
    int svc_init()
    {
#if defined(LOG_DIR)
        logfile = new ofstream();
        name += "_seq_" + to_string(ff_node_t<input_t, result_t>::get_my_id()) + ".log";
        string filename = string(STRINGIFY(LOG_DIR)) + "/" + name;
        logfile->open(filename);
#endif
        return 0;
    }

    result_t *defaultSvc( input_t *wt ) {
#if defined (LOG_DIR)
        startTS = current_time_nsecs();
        if (rcvTuples == 0)
            startTD = current_time_nsecs();
        rcvTuples++;
#endif
        // extract the key and id/timestamp fields from the input tuple
        tuple_t *t = extractTuple<tuple_t, input_t>(wt);
        size_t key = std::get<0>(t->getInfo()); // key
        uint64_t id = (winType == CB) ? std::get<1>(t->getInfo()) : std::get<2>(t->getInfo()); // identifier or timestamp
        // access the descriptor of the input key
        auto it = keyMap.find(key);
        if (it == keyMap.end()) {
            // create the descriptor of that key
            keyMap.insert(
                make_pair(
                    key, 
                    Key_Descriptor(
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
            uint64_t last_id = (winType == CB) ? std::get<1>((key_d.last_tuple).getInfo()) : std::get<2>((key_d.last_tuple).getInfo());
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
        // gwid of the first window of that key assigned to this Win_Seq instance
        uint64_t first_gwid_key = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) * config.n_outer + (config.id_outer - (key % config.n_outer) + config.n_outer) % config.n_outer;
        // initial identifer/timestamp of the keyed sub-stream arriving at this Win_Seq instance
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
        // copy the tuple into the archive of the corresponding key
        if (!isEOSMarker<tuple_t, input_t>(*wt) && isNIC)
            (key_d.archive).insert(*t);
        auto &wins = key_d.wins;
        // create all the new windows that need to be opened by the arrival of t
        for (long lwid = key_d.next_lwid; lwid <= last_w; lwid++) {
            // translate the lwid into the corresponding gwid
            uint64_t gwid = first_gwid_key + (lwid * config.n_outer * config.n_inner);
            if (winType == CB)
                wins.push_back(win_t(key, lwid, gwid, Triggerer_CB(win_len, slide_len, lwid, initial_id), CB, win_len, slide_len));
            else
                wins.push_back(win_t(key, lwid, gwid, Triggerer_TB(win_len, slide_len, lwid, initial_id), TB, win_len, slide_len));
            key_d.next_lwid++;
        }
        // evaluate all the open windows
        size_t cnt_fired = 0;
        for (auto &win: wins) {
            if (win.onTuple(*t) == CONTINUE) { // window is not fired yet
                if (!isNIC && !isEOSMarker<tuple_t, input_t>(*wt)) {
                    // incremental query -> call winUpdate
                    if (winUpdate(key, win.getGWID(), *t, *(win.getResult())) < 0) {
                        cerr << RED << "WindFlow Error: winUpdate() call error (negative return value)" << DEFAULT << endl;
                        exit(EXIT_FAILURE);
                    }
                }
            }
            else { // window is fired
                // acquire from the archive the optionals to the first and the last tuple of the window
                optional<tuple_t> t_s = win.getFirstTuple();
                optional<tuple_t> t_e = win.getFiringTuple();
                // non-incremental query -> call winFunction
                if (isNIC) {
#if defined(LOG_DIR)
                    rcvTuplesTriggering++;
                    isTriggering = true;
#endif
                    pair<const_input_iterator_t, const_input_iterator_t> its;
                    // empty window
                    if (!t_s) {
                        its.first = (key_d.archive).end();
                        its.second = (key_d.archive).end();
                    }
                    // non-empty window
                    else
                        its = (key_d.archive).getWinRange(*t_s, *t_e);
                    Iterable<tuple_t> iter(its.first, its.second);
                    if (winFunction(key, win.getGWID(), iter, *(win.getResult())) < 0) {
                        cerr << RED << "WindFlow Error: winFunction() call error (negative return value)" << DEFAULT << endl;
                        exit(EXIT_FAILURE);
                    }
                }
                // purge the tuples from the archive (if the window is not empty)
                if (t_s)
                    (key_d.archive).purge(*t_s);
                cnt_fired++;
                // send the result of the fired window
                result_t *out = win.getResult();
                // special cases: role is PLQ or MAP
                if (role == MAP) {
                    out->setInfo(key, key_d.emit_counter, std::get<2>(out->getInfo()));
                    key_d.emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    uint64_t new_id = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) + (key_d.emit_counter * config.n_inner);
                    out->setInfo(key, new_id, std::get<2>(out->getInfo()));
                    key_d.emit_counter++;
                }
                this->ff_send_out(out);
            }
        }
        // purge the fired windows
        wins.erase(wins.begin(), wins.begin() + cnt_fired);
        // delete the received tuple
        deleteTuple<tuple_t, input_t>(wt);
#if defined(LOG_DIR)
        endTS = current_time_nsecs();
        endTD = current_time_nsecs();
        double elapsedTS_us = ((double) (endTS - startTS)) / 1000;
        avg_ts_us += (1.0 / rcvTuples) * (elapsedTS_us - avg_ts_us);
        if (isNIC) {
            if (isTriggering)
                avg_ts_triggering_us += (1.0 / rcvTuplesTriggering) * (elapsedTS_us - avg_ts_triggering_us);
            else
                avg_ts_non_triggering_us += (1.0 / (rcvTuples - rcvTuplesTriggering)) * (elapsedTS_us - avg_ts_non_triggering_us);
            isTriggering = false;
        }
        double elapsedTD_us = ((double) (endTD - startTD)) / 1000;
        avg_td_us += (1.0 / rcvTuples) * (elapsedTD_us - avg_td_us);
        startTD = current_time_nsecs();
#endif
        return this->GO_ON;
    }

    result_t *flatFATSvc( input_t *wt ) {
#if defined (LOG_DIR)
        startTS = current_time_nsecs();
        if (rcvTuples == 0)
            startTD = current_time_nsecs();
        rcvTuples++;
#endif
        // extract the key and id/timestamp fields from the input tuple
        tuple_t *t = extractTuple<tuple_t, input_t>(wt);
        size_t key = std::get<0>(t->getInfo()); // key
        uint64_t id =  std::get<1>(t->getInfo()); // identifier or timestamp
        // access the descriptor of the input key
        auto it = keyMap.find(key);
        if (it == keyMap.end()) {
            // create the descriptor of that key
            keyMap.insert(
                make_pair(
                    key,
                    Key_Descriptor( 
                        winLift,
                        winCombine, 
                        win_len,
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
        // gwid of the first window of that key assigned to this Win_Seq instance
        uint64_t first_gwid_key = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) * config.n_outer + (config.id_outer - (key % config.n_outer) + config.n_outer) % config.n_outer;
        // initial identifer/timestamp of the keyed sub-stream arriving at this Win_Seq instance
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

        bool hasBeenInserted = false;
        size_t cnt_fired = 0;
        // evaluate all the open windows
        for (auto &win: wins) {
            uint64_t gwid = win.getGWID( );
            if( win.onTuple(*t) == CONTINUE ) {
                // the first time a window doesn't fire it inserts the
                // tuple in the FlatFAT once
                if( !isEOSMarker<tuple_t, input_t>(*wt) && !hasBeenInserted ) {
                    if( key_d.fat.insert( key, gwid, *t ) < 0 ) {
                        cerr << RED 
                        << "WindFlow Error: " 
                        << "FlatFAT insert() call error (negative return value)" 
                        << DEFAULT << endl;
                        exit(EXIT_FAILURE);
                    }
                    hasBeenInserted = true;
                }
            } else {
                cnt_fired++;

                // send the result of the fired window
                result_t *out;
                out = key_d.fat.getResult( key, gwid );
                // purge the tuples from Flat FAT
                for( uint64_t i = 0; i < slide_len; i++ ) {
                    key_d.fat.removeOldestTuple( key, gwid );
                }

                // special cases: role is PLQ or MAP
                if (role == MAP) {
                    out->setInfo(key, key_d.emit_counter, std::get<2>(out->getInfo()));
                    key_d.emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    uint64_t new_id = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) + (key_d.emit_counter * config.n_inner);
                    out->setInfo(key, new_id, std::get<2>(out->getInfo()));
                    key_d.emit_counter++;
                }
                this->ff_send_out(out);
            }
        }
        // purge the fired windows
        wins.erase(wins.begin(), wins.begin() + cnt_fired);
        // delete the received tuple
        deleteTuple<tuple_t, input_t>(wt);
#if defined(LOG_DIR)
        endTS = current_time_nsecs();
        endTD = current_time_nsecs();
        double elapsedTS_us = ((double) (endTS - startTS)) / 1000;
        avg_ts_us += (1.0 / rcvTuples) * (elapsedTS_us - avg_ts_us);
        double elapsedTD_us = ((double) (endTD - startTD)) / 1000;
        avg_td_us += (1.0 / rcvTuples) * (elapsedTD_us - avg_td_us);
        startTD = current_time_nsecs();
#endif
        return this->GO_ON;
    }

    // svc method (utilized by the FastFlow runtime)
    result_t *svc(input_t *wt) {
        // it checks for which implementation the object is using
        if( useFlatFAT ) {
            return flatFATSvc( wt );
        } else {
            return defaultSvc( wt );
        }
    }

    void defaultEosNotifiy( ssize_t id ) {
        // iterate over all the keys
        for (auto &k: keyMap) {
            auto &wins = (k.second).wins;
            // iterate over all the existing windows of the key
            for (auto &win: wins) {
                // non-incremental query
                if (isNIC) {
                    // acquire from the archive the optional to the first tuple of the window
                    optional<tuple_t> t_s = win.getFirstTuple();
                    pair<const_input_iterator_t,const_input_iterator_t> its;
                    // empty window
                    if (!t_s) {
                        its.first = ((k.second).archive).end();
                        its.second = ((k.second).archive).end();
                    }
                    // non-empty window
                    else
                        its = ((k.second).archive).getWinRange(*t_s);
                    Iterable<tuple_t> iter(its.first, its.second);
                    if (winFunction(k.first, win.getGWID(), iter, *(win.getResult())) < 0) {
                        cerr << RED << "WindFlow Error: winFunction() call error (negative return value)" << DEFAULT << endl;
                        exit(EXIT_FAILURE);
                    }
                }
                // send the result of the window
                result_t *out = win.getResult();
                // special cases: role is PLQ or MAP
                if (role == MAP) {
                    out->setInfo(k.first, (k.second).emit_counter, std::get<2>(out->getInfo()));
                    (k.second).emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    uint64_t new_id = ((config.id_inner - (k.first % config.n_inner) + config.n_inner) % config.n_inner) + ((k.second).emit_counter * config.n_inner);
                    out->setInfo(k.first, new_id, std::get<2>(out->getInfo()));
                    (k.second).emit_counter++;
                }
                this->ff_send_out(out);
            }
        }
    }
    
    void flatFATEosNotify( ssize_t id ) {
        // iterate over all the keys
        for (auto &k: keyMap) {
            auto &wins = (k.second).wins;
            // iterate over all the existing windows of the key
            for (auto &win: wins) {
                size_t k = k.first;
                uint64_t gwid = win.getGWID( );
                //send out the result of the window
                result_t *out;
                out = k.second.fat.getResult( k, gwid );
                // purge the tuples from Flat FAT
                for( int i = 0; i < slide_len; i++ ) {
                    k.second.fat.removeOldestTuple( k, gwid );
                }
                // special cases: role is PLQ or MAP
                if (role == MAP) {
                    out->setInfo(k.first, (k.second).emit_counter, std::get<2>(out->getInfo()));
                    (k.second).emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    uint64_t new_id = ((config.id_inner - (k.first % config.n_inner) + config.n_inner) % config.n_inner) + ((k.second).emit_counter * config.n_inner);
                    out->setInfo(k.first, new_id, std::get<2>(out->getInfo()));
                    (k.second).emit_counter++;
                }
                this->ff_send_out(out);
            }
        }
    }

    // method to manage the EOS (utilized by the FastFlow runtime)
    void eosnotify(ssize_t id)
    {
        // it checks for which implementation the object is using
        if( useFlatFAT ) {
            flatFATEosNotify( id );
        } else {
            defaultEosNotifiy( id );
        }
    }

    // svc_end method (utilized by the FastFlow runtime)
    void svc_end()
    {
#if defined (LOG_DIR)
        ostringstream stream;
        if (!isNIC) {
            stream << "************************************LOG************************************\n";
            stream << "No. of received tuples: " << rcvTuples << "\n";
            stream << "Average service time: " << avg_ts_us << " usec \n";
            stream << "Average inter-departure time: " << avg_td_us << " usec \n";
            stream << "***************************************************************************\n";
        }
        else {
            stream << "************************************LOG************************************\n";
            stream << "No. of received tuples: " << rcvTuples << "\n";
            stream << "No. of received tuples (triggering): " << rcvTuplesTriggering << "\n";
            stream << "Average service time: " << avg_ts_us << " usec \n";
            stream << "Average service time (triggering): " << avg_ts_triggering_us << " usec \n";
            stream << "Average service time (non triggering): " << avg_ts_non_triggering_us << " usec \n";
            stream << "Average inter-departure time: " << avg_td_us << " usec \n";
            stream << "***************************************************************************\n";
        }
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
