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
 *  The template parameters tuple_t and result_t must be default constructible, with
 *  a copy Constructor and copy assignment operator, and they must provide and implement
 *  the setControlFields() and getControlFields() methods.
 */ 

#ifndef WIN_SEQ_H
#define WIN_SEQ_H

/// includes
#include <vector>
#include <list>
#include <string>
#include <unordered_map>
#include <math.h>
#include <ff/node.hpp>
#include <window.hpp>
#include <context.hpp>
#include <iterable.hpp>
#include <stream_archive.hpp>
#include <fat.hpp>
#include <cassert>

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
    /// type of the non-incremental window processing function
    using win_func_t = function<void(uint64_t, Iterable<tuple_t> &, result_t &)>;
    /// type of the rich non-incremental window processing function
    using rich_win_func_t = function<void(uint64_t, Iterable<tuple_t> &, result_t &, RuntimeContext &)>;
    /// type of the incremental window processing function
    using winupdate_func_t = function<void(uint64_t, const tuple_t &, result_t &)>;
    /// type of the rich incremental window processing function
    using rich_winupdate_func_t = function<void(uint64_t, const tuple_t &, result_t &, RuntimeContext &)>;
    /// type of the closing function
    using closing_func_t = function<void(RuntimeContext &)>;

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
    using compare_func_t = function<bool(const tuple_t &, const tuple_t &)>;
    tuple_t tmp; // never used
    // key data type
    using key_t = typename remove_reference<decltype(std::get<0>(tmp.getControlFields()))>::type;
    // friendships with other classes in the library
    template<typename T1, typename T2, typename T3>
    friend class Win_Farm;
    template<typename T1, typename T2>
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
        vector<result_t> received_tuples;
        result_t acc;
        uint64_t cb_id;
        uint64_t last_quantum;
        list<win_t> wins; // open windows of this key
        uint64_t emit_counter; // progressive counter (used if role is PLQ or MAP)
        uint64_t rcv_counter; // number of tuples received of this key
        uint64_t slide_counter;
        uint64_t ts_rcv_counter;
        tuple_t last_tuple; // copy of the last tuple received of this key
        uint64_t next_lwid; // next window to be opened of this key (lwid)

        // Constructor
        Key_Descriptor(compare_func_t _compare_func, uint64_t _emit_counter=0):
                       archive(_compare_func),
                       emit_counter(_emit_counter),
                       cb_id( 0 ),
                       last_quantum( 0 ),
                       rcv_counter(0),
                       slide_counter( 0 ),
                       ts_rcv_counter( 0 ),
                       next_lwid(0)
        {
        }

        Key_Descriptor(
            f_winlift_t _winLift,
            f_wincombine_t _winCombine,
            bool _isCommutative,
            size_t _win_len,
            uint64_t _emit_counter = 0 
        ) : 
            fat( _winLift, _winCombine, _win_len, _isCommutative ),
            emit_counter(_emit_counter),
            cb_id( 0 ),
            last_quantum( 0 ),
            rcv_counter(0),
            slide_counter( 0 ),
            ts_rcv_counter( 0 ),
            next_lwid(0)
        {
            tuple_t t;
            _winLift( 0, 0, t, acc );
        }

        // move Constructor
        Key_Descriptor(Key_Descriptor &&_k):
                       archive(move(_k.archive)),
                       fat(move(_k.fat)),
                       received_tuples( move( _k.received_tuples ) ),
                       acc( move( _k.acc ) ),
                       cb_id( move( _k.cb_id ) ),
                       last_quantum( move ( _k.last_quantum ) ),
                       wins(move(_k.wins)),
                       emit_counter(_k.emit_counter),
                       rcv_counter(_k.rcv_counter),
                       slide_counter( _k.slide_counter ),
                       ts_rcv_counter( _k.ts_rcv_counter ),
                       next_lwid(_k.next_lwid) {}
    };
    win_func_t win_func; // function for the non-incremental window processing
    rich_win_func_t rich_win_func; // rich function for the non-incremental window processing
    winupdate_func_t winupdate_func; // function for the incremental window processing
    rich_winupdate_func_t rich_winupdate_func; // rich function for the incremental window processing
    closing_func_t closing_func; // closing function
    compare_func_t compare_func; // function to compare two tuples
    f_winlift_t winLift; // function for inserting an element in the Flat FAT
    f_wincombine_t winCombine; // function of the incremental window processing in the Flat FAT
    bool isWinCombineCommutative;
    bool timebasedFAT;
    uint64_t quantum;
    uint64_t win_len; // window length (no. of tuples or in time units)
    uint64_t slide_len; // slide length (no. of tuples or in time units)
    win_type_t winType; // window type (CB or TB)
    string name; // string of the unique name of the pattern
    bool isNIC; // this flag is true if the pattern is instantiated with a non-incremental query function
    bool useFlatFAT; // it is true if the object uses a Flat FAT 
    bool isRich; // flag stating whether the function to be used is rich
    RuntimeContext context; // RuntimeContext
    PatternConfig config; // configuration structure of the Win_Seq pattern
    role_t role; // role of the Win_Seq
    unordered_map<key_t, Key_Descriptor> keyMap; // hash table that maps a descriptor for each key
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

    // private initialization method
    void init()
    {
        // check the validity of the windowing parameters
        if (win_len == 0 || slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // define the compare function depending on the window type
        if (winType == CB) {
            compare_func = [](const tuple_t &t1, const tuple_t &t2) {
                return std::get<1>(t1.getControlFields()) < std::get<1>(t2.getControlFields());
            };
        }
        else {
            compare_func = [](const tuple_t &t1, const tuple_t &t2) {
                return std::get<2>(t1.getControlFields()) < std::get<2>(t2.getControlFields());
            };
        }
    }

    //private constructor III (incremental queries, it uses a Flat FAT)
    Win_Seq(f_winlift_t _winLift,
            f_wincombine_t _winCombine,
            bool _isWinCombineCommutative,
            uint64_t _win_len,
            uint64_t _slide_len,
            string _name,
            PatternConfig _config,
            role_t _role)
            :
            winLift( _winLift ),
            winCombine( _winCombine ),
            isWinCombineCommutative( _isWinCombineCommutative ),
            win_len(_win_len),
            slide_len(_slide_len),
            winType( CB ),
            timebasedFAT( false ),
            isNIC(false),
            useFlatFAT( true ),
            isRich( false ),
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
        compare_func = [](const tuple_t &t1, const tuple_t &t2) {
            return std::get<1>(t1.getControlFields()) < std::get<1>(t2.getControlFields());
        };

        closing_func = []( RuntimeContext& r ) { };
    }

    Win_Seq(f_winlift_t _winLift,
            f_wincombine_t _winCombine,
            bool _isWinCombineCommutative,
            uint64_t _win_len,
            uint64_t _slide_len,
            uint64_t _quantum,
            string _name,
            PatternConfig _config,
            role_t _role)
            :
            winLift( _winLift ),
            winCombine( _winCombine ),
            isWinCombineCommutative( _isWinCombineCommutative ),
            win_len(_win_len),
            slide_len(_slide_len),
            quantum( _quantum * 1000 ),
            winType( CB ),
            isNIC(false),
            useFlatFAT( false ),
            timebasedFAT( true ),
            isRich( false ),
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
        compare_func = [](const tuple_t &t1, const tuple_t &t2) {
            return std::get<1>(t1.getControlFields()) < std::get<1>(t2.getControlFields());
        };

        closing_func = []( RuntimeContext& r ) { };
    }

    // method to set the indexes useful if role is MAP
    void setMapIndexes(size_t _first, size_t _second) {
        map_indexes.first = _first; // id
        map_indexes.second = _second; // pardegree
    }

public:
    /** 
     *  \brief Constructor I
     *  
     *  \param _win_func the non-incremental window processing function
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _name string with the unique name of the pattern
     *  \param _closing_func closing function
     *  \param _context RuntimeContext object to be used
     *  \param _config configuration of the pattern
     *  \param _role role of the pattern
     */ 
    Win_Seq(win_func_t _win_func,
            uint64_t _win_len,
            uint64_t _slide_len,
            win_type_t _winType,
            string _name,
            closing_func_t _closing_func,
            RuntimeContext _context,
            PatternConfig _config,
            role_t _role)
            :
            win_func(_win_func),
            win_len(_win_len),
            slide_len(_slide_len),
            winType(_winType),
            useFlatFAT( false ),
            timebasedFAT( false ),
            name(_name),
            closing_func(_closing_func),
            context(_context),
            config(_config),
            role(_role),
            isNIC(true),
            isRich(false)
    {
        init();
    }


    /** 
     *  \brief Constructor II
     *  
     *  \param _rich_win_func the rich non-incremental window processing function
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _name string with the unique name of the pattern
     *  \param _closing_func closing function
     *  \param _context RuntimeContext object to be used
     *  \param _config configuration of the pattern
     *  \param _role role of the pattern
     */ 
    Win_Seq(rich_win_func_t _rich_win_func,
            uint64_t _win_len,
            uint64_t _slide_len,
            win_type_t _winType,
            string _name,
            closing_func_t _closing_func,
            RuntimeContext _context,
            PatternConfig _config,
            role_t _role)
            :
            rich_win_func(_rich_win_func),
            win_len(_win_len),
            slide_len(_slide_len),
            winType(_winType),
            name(_name),
            closing_func(_closing_func),
            context(_context),
            config(_config),
            role(_role),
            isNIC(true),
            isRich(true),
            useFlatFAT( false ),
            timebasedFAT( false )
    {
        init();
    }

    /** 
     *  \brief Constructor III
     *  
     *  \param _winupdate_func the incremental window processing function
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _name string with the unique name of the pattern
     *  \param _closing_func closing function
     *  \param _context RuntimeContext object to be used
     *  \param _config configuration of the pattern
     *  \param _role role of the pattern
     */ 
    Win_Seq(winupdate_func_t _winupdate_func,
            uint64_t _win_len,
            uint64_t _slide_len,
            win_type_t _winType,
            string _name,
            closing_func_t _closing_func,
            RuntimeContext _context,
            PatternConfig _config,
            role_t _role)
            :
            winupdate_func(_winupdate_func),
            win_len(_win_len),
            slide_len(_slide_len),
            winType(_winType),
            name(_name),
            closing_func(_closing_func),
            context(_context),
            config(_config),
            role(_role),
            isNIC(false),
            isRich(false),
            useFlatFAT( false ),
            timebasedFAT( false )
    {
        init();
    }

    /** 
     *  \brief Constructor IV
     *  
     *  \param _rich_winupdate_func the rich incremental window processing function
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _name string with the unique name of the pattern
     *  \param _closing_func closing function
     *  \param _context RuntimeContext object to be used
     *  \param _config configuration of the pattern
     *  \param _role role of the pattern
     */ 
    Win_Seq(rich_winupdate_func_t _rich_winupdate_func,
            uint64_t _win_len,
            uint64_t _slide_len,
            win_type_t _winType,
            string _name,
            closing_func_t _closing_func,
            RuntimeContext _context,
            PatternConfig _config,
            role_t _role)
            :
            rich_winupdate_func(_rich_winupdate_func),
            win_len(_win_len),
            slide_len(_slide_len),
            winType(_winType),
            name(_name),
            closing_func(_closing_func),
            context(_context),
            config(_config),
            role(_role),
            isNIC(false),
            isRich(true),
            useFlatFAT( false ),
            timebasedFAT( false )
    {
        init();
    }

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
            bool _isWinCombineCommutative,
            uint64_t _win_len,
            uint64_t _slide_len,
            string _name)
            :
            Win_Seq(_winLift, _winCombine, _isWinCombineCommutative, _win_len, _slide_len, _name, PatternConfig(0, 1, _slide_len, 0, 1, _slide_len), SEQ) {}

    Win_Seq(f_winlift_t _winLift,
            f_wincombine_t _winCombine,
            bool _isWinCombineCommutative,
            uint64_t _win_len,
            uint64_t _slide_len,
            uint64_t _quantum,
            string _name)
            :
            Win_Seq(_winLift, _winCombine, _isWinCombineCommutative, _win_len, _slide_len, _quantum, _name, PatternConfig(0, 1, _slide_len, 0, 1, _slide_len), SEQ) {}

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
        auto key = std::get<0>(t->getControlFields()); // key
        size_t hashcode = hash<decltype(key)>()(key); // compute the hashcode of the key
        uint64_t id = (winType == CB) ? std::get<1>(t->getControlFields()) : std::get<2>(t->getControlFields()); // identifier or timestamp
        // access the descriptor of the input key
        auto it = keyMap.find(key);
        if (it == keyMap.end()) {
            // create the descriptor of that key
            keyMap.insert(
                make_pair(
                    key, 
                    Key_Descriptor(
                        compare_func, 
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
        // gwid of the first window of that key assigned to this Win_Seq
        uint64_t first_gwid_key = ((config.id_inner - (hashcode % config.n_inner) + config.n_inner) % config.n_inner) * config.n_outer + (config.id_outer - (hashcode % config.n_outer) + config.n_outer) % config.n_outer;
        // initial identifer/timestamp of the keyed sub-stream arriving at this Win_Seq
        uint64_t initial_outer = ((config.id_outer - (hashcode % config.n_outer) + config.n_outer) % config.n_outer) * config.slide_outer;
        uint64_t initial_inner = ((config.id_inner - (hashcode % config.n_inner) + config.n_inner) % config.n_inner) * config.slide_inner;
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
            // if the tuple does not belong to at least one window assigned to this Win_Seq
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
                    // incremental query -> call rich_/winupdate_func
                    if (!isRich)
                        winupdate_func(win.getGWID(), *t, *(win.getResult()));
                    else
                        rich_winupdate_func(win.getGWID(), *t, *(win.getResult()), context);
                }
            }
            else { // window is fired
                // acquire from the archive the optionals to the first and the last tuple of the window
                optional<tuple_t> t_s = win.getFirstTuple();
                optional<tuple_t> t_e = win.getFiringTuple();
                // non-incremental query -> call win_func
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
                    // non-incremental query -> call rich_/win_func
                    if (!isRich)
                        win_func(win.getGWID(), iter, *(win.getResult()));
                    else
                        rich_win_func(win.getGWID(), iter, *(win.getResult()), context);
                }
                // purge the tuples from the archive (if the window is not empty)
                if (t_s)
                    (key_d.archive).purge(*t_s);
                cnt_fired++;
                // send the result of the fired window
                result_t *out = win.getResult();
                // special cases: role is PLQ or MAP
                if (role == MAP) {
                    out->setControlFields(key, key_d.emit_counter, std::get<2>(out->getControlFields()));
                    key_d.emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    uint64_t new_id = ((config.id_inner - (hashcode % config.n_inner) + config.n_inner) % config.n_inner) + (key_d.emit_counter * config.n_inner);
                    out->setControlFields(key, new_id, std::get<2>(out->getControlFields()));
                    key_d.emit_counter++;
                }
                this->ff_send_out(out);
            }
        }
        // purge the fired windows
        size_t i = 0;
        auto jt = wins.begin( );
        for( ; i < cnt_fired; i++, jt++ ) {
        }
        wins.erase(wins.begin(), jt );
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
        size_t key = std::get<0>(t->getControlFields()); // key
        uint64_t id =  std::get<1>(t->getControlFields()); // identifier or timestamp
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
                        isWinCombineCommutative,
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
            key_d.slide_counter++;
            key_d.last_tuple = *t;
        }
        else {
            // tuples can be received only ordered by id/timestamp
            uint64_t last_id = std::get<1>((key_d.last_tuple).getControlFields());
            if (id < last_id) {
                // the tuple is immediately deleted
                deleteTuple<tuple_t, input_t>(wt);
                return this->GO_ON;
            }
            else {
                key_d.rcv_counter++;
                key_d.slide_counter++;
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

        if (!isEOSMarker<tuple_t, input_t>(*wt)) {
            result_t res;
            winLift( key, t->id, *t, res );
            (key_d.received_tuples).push_back( res );
        }

        size_t cnt_fired = 0;
        uint64_t gwid;
        if( key_d.rcv_counter == win_len )
        {
            cnt_fired = 1;
            uint64_t lwid = key_d.next_lwid;
            gwid = first_gwid_key + (lwid * config.n_outer * config.n_inner);
            key_d.next_lwid++;
            key_d.slide_counter = 0;
        } else if( ( key_d.rcv_counter > win_len ) && key_d.slide_counter % slide_len == 0 ) {
            cnt_fired = 1;
            uint64_t lwid = key_d.next_lwid;
            gwid = first_gwid_key + (lwid * config.n_outer * config.n_inner);
            key_d.next_lwid++;
            key_d.slide_counter = 0;
        }

        if( cnt_fired == 1 ) {
            if( key_d.fat.insert( key, gwid, key_d.received_tuples ) < 0 ) {
                cout << RED << "Errore nella fat.insert "
                << DEFAULT << endl;
                exit(EXIT_FAILURE);
            }
            key_d.received_tuples.clear( );
            // send the result of the fired window
            result_t *out;
            out = key_d.fat.getResult( key, gwid );
            // purge the tuples from Flat FAT
            key_d.fat.remove( key, gwid, slide_len );
            out->setControlFields( key, gwid, out->ts );
            // special cases: role is PLQ or MAP    
            if (role == MAP) {
                out->setControlFields(key, key_d.emit_counter, std::get<2>(out->getControlFields()));
                key_d.emit_counter += map_indexes.second;
            }
            else if (role == PLQ) {
                uint64_t new_id = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) + (key_d.emit_counter * config.n_inner);
                out->setControlFields(key, new_id, std::get<2>(out->getControlFields()));
                key_d.emit_counter++;
            }
            this->ff_send_out(out);
        }

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

result_t *ManageWindow( Key_Descriptor& key_d, input_t *wt, tuple_t *t )
{
    size_t key = std::get<0>(t->getControlFields()); // key
    uint64_t id =  std::get<1>(t->getControlFields()); // identifier or timestamp
    // check duplicate or out-of-order tuples
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
        return this->GO_ON;
    }

    key_d.ts_rcv_counter++;
    key_d.slide_counter++;
    if( !isEOSMarker<tuple_t, input_t>(*wt) ) {
        key_d.received_tuples.push_back( *reinterpret_cast<result_t*>( t ) );
    }
    //cout << key_d.ts_rcv_counter << endl;

    size_t cnt_fired = 0;
    uint64_t gwid;
    if( key_d.ts_rcv_counter == win_len )
    {
        cnt_fired = 1;
        uint64_t lwid = key_d.next_lwid;
        gwid = first_gwid_key + (lwid * config.n_outer * config.n_inner);
        key_d.next_lwid++;
        key_d.slide_counter = 0;
    } else if( ( key_d.ts_rcv_counter > win_len ) && key_d.slide_counter % slide_len == 0 ) {
        cnt_fired = 1;
        uint64_t lwid = key_d.next_lwid;
        gwid = first_gwid_key + (lwid * config.n_outer * config.n_inner);
        key_d.next_lwid++;
        key_d.slide_counter = 0;
    }

    if( cnt_fired == 1 ) {
        if( key_d.fat.insert( key, gwid, key_d.received_tuples ) < 0 ) {
            cout << RED << "Errore nella fat.insert "
            << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        key_d.received_tuples.clear( );
        // send the result of the fired window
        result_t *out;
        out = key_d.fat.getResult( key, gwid );
        // purge the tuples from Flat FAT
        key_d.fat.remove( key, gwid, slide_len );
        out->setControlFields( key, gwid, out->ts );
        // special cases: role is PLQ or MAP    
        if (role == MAP) {
            out->setControlFields(key, key_d.emit_counter, std::get<2>(out->getControlFields()));
            key_d.emit_counter += map_indexes.second;
        }
        else if (role == PLQ) {
            uint64_t new_id = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) + (key_d.emit_counter * config.n_inner);
            out->setControlFields(key, new_id, std::get<2>(out->getControlFields()));
            key_d.emit_counter++;
        }
        this->ff_send_out(out);
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
        size_t key = std::get<0>(t->getControlFields()); // key
        static uint64_t base_ts = std::get<2>(t->getControlFields()); // identifier or timestamp
        uint64_t ts =  std::get<2>(t->getControlFields()) - base_ts; // identifier or timestamp
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
                        isWinCombineCommutative,
                        win_len,
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
            uint64_t last_ts = std::get<2>((key_d.last_tuple).getControlFields()) - base_ts;
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
        result_t *res = this->GO_ON;
        for( int64_t i = 0; i < distance; i++ ) {
            key_d.acc.key = key;
            key_d.acc.ts = key_d.last_tuple.ts;
            key_d.acc.id = ( key_d.cb_id )++;
            res = ManageWindow( 
                key_d, wt, reinterpret_cast<tuple_t*>( &( key_d.acc ) ) 
            );
            ( key_d.last_quantum )++;
            tuple_t tmp;
            winLift( key, 0, tmp, key_d.acc );
        }
        result_t tmp;
        winLift( key, 0, *t, tmp );
        winCombine( key, 0, key_d.acc, tmp, key_d.acc );

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
        return res;
    }

    // svc method (utilized by the FastFlow runtime)
    result_t *svc(input_t *wt) {
        // it checks for which implementation the object is using
        if( timebasedFAT ) {
            return timebasedSvc( wt );
        } else if( useFlatFAT ) {
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
                    // non-incremental query -> call rich_/win_func
                    if (!isRich)
                        win_func(win.getGWID(), iter, *(win.getResult()));
                    else
                        rich_win_func(win.getGWID(), iter, *(win.getResult()), context);
                }
                // send the result of the window
                result_t *out = win.getResult();
                // special cases: role is PLQ or MAP
                if (role == MAP) {
                    out->setControlFields(k.first, (k.second).emit_counter, std::get<2>(out->getControlFields()));
                    (k.second).emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    size_t hashcode = hash<key_t>()(k.first); // compute the hashcode of the key
                    uint64_t new_id = ((config.id_inner - (hashcode % config.n_inner) + config.n_inner) % config.n_inner) + ((k.second).emit_counter * config.n_inner);
                    out->setControlFields(k.first, new_id, std::get<2>(out->getControlFields()));
                    (k.second).emit_counter++;
                }
                this->ff_send_out(out);
            }
        }
    }
    
    void flatFATEosNotify( ssize_t id ) {
        // iterate over all the keys
        for (auto &k: keyMap) {
            // iterate over all the existing windows of the key
            size_t key = k.first;
            auto &tuples = k.second.received_tuples;
            auto &key_d = k.second;

            k.second.fat.insert( 
                k.first, 
                0, 
                k.second.received_tuples 
            );

            while( !key_d.fat.isEmpty( ) ) {
                uint64_t first_gwid_key = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) * config.n_outer + (config.id_outer - (key % config.n_outer) + config.n_outer) % config.n_outer;
                uint64_t lwid = key_d.next_lwid;
                uint64_t gwid = first_gwid_key + (lwid * config.n_outer * config.n_inner);
                key_d.next_lwid++;
                //send out the result of the window
                result_t *out;
                out = k.second.fat.getResult( key, gwid );
                // purge the tuples from Flat FAT
                k.second.fat.remove( key, gwid, slide_len );
                out->setControlFields( key, gwid, out->ts );
                // special cases: role is PLQ or MAP
                if (role == MAP) {
                    out->setControlFields(k.first, (k.second).emit_counter, std::get<2>(out->getControlFields()));
                    (k.second).emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    uint64_t new_id = ((config.id_inner - (k.first % config.n_inner) + config.n_inner) % config.n_inner) + ((k.second).emit_counter * config.n_inner);
                    out->setControlFields(k.first, new_id, std::get<2>(out->getControlFields()));
                    (k.second).emit_counter++;
                }
                this->ff_send_out(out);
            }
        }
    }

    void timebasedEosNotify( ssize_t id ) {
        // iterate over all the keys
        for (auto &k: keyMap) {
            size_t key = k.first;
            auto &wins = (k.second).wins;
            auto &key_d = k.second;

            key_d.acc.key = k.first;
            key_d.acc.ts = key_d.last_tuple.ts;
            key_d.acc.id = ( key_d.cb_id )++;
            ( void ) ManageWindow( 
                key_d, nullptr, reinterpret_cast<tuple_t*>( &( key_d.acc ) ) 
            );
            ( key_d.last_quantum )++;

            // iterate over all the existing windows of the key
            k.second.fat.insert( 
                k.first, 
                0, 
                k.second.received_tuples 
            );
            while( !key_d.fat.isEmpty( ) ) {
                uint64_t first_gwid_key = ((config.id_inner - (key % config.n_inner) + config.n_inner) % config.n_inner) * config.n_outer + (config.id_outer - (key % config.n_outer) + config.n_outer) % config.n_outer;
                uint64_t lwid = key_d.next_lwid;
                uint64_t gwid = first_gwid_key + (lwid * config.n_outer * config.n_inner);
                key_d.next_lwid++;
                //send out the result of the window
                result_t *out;
                out = k.second.fat.getResult( key, gwid );
                // purge the tuples from Flat FAT
                k.second.fat.remove( key, gwid, slide_len );
                out->setControlFields( key, gwid, out->ts );
                // special cases: role is PLQ or MAP
                if (role == MAP) {
                    out->setControlFields(k.first, (k.second).emit_counter, std::get<2>(out->getControlFields()));
                    (k.second).emit_counter += map_indexes.second;
                }
                else if (role == PLQ) {
                    uint64_t new_id = ((config.id_inner - (k.first % config.n_inner) + config.n_inner) % config.n_inner) + ((k.second).emit_counter * config.n_inner);
                    out->setControlFields(k.first, new_id, std::get<2>(out->getControlFields()));
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
        if( timebasedFAT ) {
            return timebasedEosNotify( id );
        } else if( useFlatFAT ) {
            flatFATEosNotify( id );
        } else {
            defaultEosNotifiy( id );
        }
    }

    // svc_end method (utilized by the FastFlow runtime)
    void svc_end()
    {
        // call the closing function
        closing_func(context);
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
