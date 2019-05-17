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
 *  @file    win_farm.hpp
 *  @author  Gabriele Mencagli
 *  @date    03/10/2017
 *  
 *  @brief Win_Farm pattern executing a windowed transformation in parallel on multi-core CPUs
 *  
 *  @section Win_Farm (Description)
 *  
 *  This file implements the Win_Farm pattern able to executes windowed queries on a
 *  multicore. The pattern executes streaming windows in parallel on the CPU cores
 *  and supports both a non-incremental and an incremental query definition.
 *  
 *  The template arguments tuple_t and result_t must be default constructible, with a copy constructor
 *  and copy assignment operator, and they must provide and implement the setInfo() and
 *  getInfo() methods.
 */ 

#ifndef WIN_FARM_H
#define WIN_FARM_H

// includes
#include <ff/farm.hpp>
#include <ff/optimize.hpp>
#include <win_seq.hpp>
#include <wf_nodes.hpp>
#include <pane_farm.hpp>
#include <win_mapreduce.hpp>

/** 
 *  \class Win_Farm
 *  
 *  \brief Win_Farm pattern executing a windowed transformation in parallel on multi-core CPUs
 *  
 *  This class implements the Win_Farm pattern executing windowed queries in parallel on
 *  a multicore.
 */ 
template<typename tuple_t, typename result_t, typename input_t>
class Win_Farm: public ff_farm
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
    /// type of the Pane_Farm passed to the proper nesting constructor
    using pane_farm_t = Pane_Farm<tuple_t, result_t>;
    /// type of the Win_MapReduce passed to the proper nesting constructor
    using win_mapreduce_t = Win_MapReduce<tuple_t, result_t>;
private:
    // type of the wrapper of input tuples
    using wrapper_in_t = wrapper_tuple_t<tuple_t>;
    // type of the WF_Emitter node
    using wf_emitter_t = WF_Emitter<tuple_t, input_t>;
    // type of the WF_Collector node
    using wf_collector_t = WF_Collector<result_t>;
    // type of the Win_Seq to be created within the regular constructor
    using win_seq_t = Win_Seq<tuple_t, result_t, wrapper_in_t>;
    // friendships with other classes in the library
    template<typename T1, typename T2, typename T3>
    friend class Pane_Farm;
    template<typename T1, typename T2, typename T3>
    friend class Win_MapReduce;
    template<typename T1, typename T2, typename T3, typename T4>
    friend class Pane_Farm_GPU;
    template<typename T1, typename T2, typename T3, typename T4>
    friend class Win_MapReduce_GPU;
    template<typename T>
    friend auto get_WF_nested_type(T);
    // flag stating whether the Win_Farm has been instantiated with complex workers (Pane_Farm or Win_MapReduce instances)
    bool hasComplexWorkers;
    // optimization level of the Win_Farm
    opt_level_t opt_level;
    // window type (CB or TB)
    win_type_t winType;
    // number of Win_Farm emitters
    size_t num_emitters;

    // private constructor I (stub)
    Win_Farm() {}

    // private constructor II (non-incremental queries)
    Win_Farm(f_winfunction_t _winFunction,
             uint64_t _win_len,
             uint64_t _slide_len,
             win_type_t _winType,
             size_t _emitter_degree,
             size_t _pardegree,
             string _name,
             bool _ordered,
             opt_level_t _opt_level,
             PatternConfig _config,
             role_t _role): hasComplexWorkers(false), opt_level(_opt_level), winType(_winType), num_emitters(_emitter_degree)
    {
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the emitter degree
        if (_emitter_degree == 0) {
            cerr << RED << "WindFlow Error: at least one emitter is needed" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the parallelism degree
        if (_pardegree == 0) {
            cerr << RED << "WindFlow Error: parallelism degree cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the optimization level
        if (_opt_level != LEVEL0) {
            cerr << YELLOW << "WindFlow Warning: optimization level has no effect" << DEFAULT << endl;
            opt_level = LEVEL0;
        }
        // vector of Win_Seq instances
        vector<ff_node *> w;
        // private sliding factor of each Win_Seq instance
        uint64_t private_slide = _slide_len * _pardegree;
        // standard case: one Emitter node
        if (_emitter_degree == 1) {
            // create the Win_Seq instances
            for (size_t i = 0; i < _pardegree; i++) {
                // configuration structure of the Win_Seq instances
                PatternConfig configSeq(_config.id_inner, _config.n_inner, _config.slide_inner, i, _pardegree, _slide_len);
                auto *seq = new win_seq_t(_winFunction, _win_len, private_slide, _winType, _name + "_wf", configSeq, _role);
                w.push_back(seq);
            }
        }
        // advanced case: multiple Emitter nodes
        else {
            ff_a2a *a2a = new ff_a2a();
            // create the Emitter nodes
            vector<ff_node *> emitters(_emitter_degree);
            for (size_t i = 0; i < _emitter_degree; i++) {
                auto *emitter = new wf_emitter_t(_winType, _win_len, _slide_len, _pardegree, _config.id_inner, _config.n_inner, _config.slide_inner, _role);
                emitters[i] = emitter;
            }
            a2a->add_firstset(emitters, 0, true);
            // create the Win_Seq nodes composed with an orderingNodes
            vector<ff_node *> seqs(_pardegree);
            for (size_t i = 0; i < _pardegree; i++) {
                auto *ord = new OrderingNode<tuple_t, wrapper_in_t>(((_winType == CB) ? ID : TS));
                // configuration structure of the Win_Seq instances
                PatternConfig configSeq(_config.id_inner, _config.n_inner, _config.slide_inner, i, _pardegree, _slide_len);
                auto *seq = new win_seq_t(_winFunction, _win_len, private_slide, _winType, _name + "_wf", configSeq, _role);
                auto *comb = new ff_comb(ord, seq, true, true);
                seqs[i] = comb;
            }
            a2a->add_secondset(seqs, true);
            w.push_back(a2a);
        }
        ff_farm::add_workers(w);
        // create the Emitter and Collector nodes
        if (_emitter_degree == 1)
            ff_farm::add_emitter(new wf_emitter_t(_winType, _win_len, _slide_len, _pardegree, _config.id_inner, _config.n_inner, _config.slide_inner, _role));
        if (_ordered)
            ff_farm::add_collector(new wf_collector_t());
        else
            ff_farm::add_collector(nullptr);
        // when the Win_Farm will be destroyed we need aslo to destroy the emitter, workers and collector
        ff_farm::cleanup_all();
    }

    // private constructor III (incremental queries)
    Win_Farm(f_winupdate_t _winUpdate,
             uint64_t _win_len,
             uint64_t _slide_len,
             win_type_t _winType,
             size_t _emitter_degree,
             size_t _pardegree,
             string _name,
             bool _ordered,
             opt_level_t _opt_level,
             PatternConfig _config,
             role_t _role): hasComplexWorkers(false), opt_level(_opt_level), winType(_winType), num_emitters(_emitter_degree)
    {
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the emitter degree
        if (_emitter_degree == 0) {
            cerr << RED << "WindFlow Error: at least one emitter is needed" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the parallelism degree
        if (_pardegree == 0) {
            cerr << RED << "WindFlow Error: parallelism degree cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the optimization level
        if (_opt_level != LEVEL0) {
            cerr << YELLOW << "WindFlow Warning: optimization level has no effect" << DEFAULT << endl;
            opt_level = LEVEL0;
        }
        // vector of Win_Seq instances
        vector<ff_node *> w;
        // private sliding factor of each Win_Seq instance
        uint64_t private_slide = _slide_len * _pardegree;
        // standard case: one Emitter node
        if (_emitter_degree == 1) {
            // create the Win_Seq instances
            for (size_t i = 0; i < _pardegree; i++) {
                // configuration structure of the Win_Seq instances
                PatternConfig configSeq(_config.id_inner, _config.n_inner, _config.slide_inner, i, _pardegree, _slide_len);
                auto *seq = new win_seq_t(_winUpdate, _win_len, private_slide, _winType, _name + "_wf", configSeq, _role);
                w.push_back(seq);
            }
        }
        // advanced case: multiple Emitter nodes
        else {
            ff_a2a *a2a = new ff_a2a();
            // create the Emitter nodes
            vector<ff_node *> emitters(_emitter_degree);
            for (size_t i = 0; i < _emitter_degree; i++) {
                auto *emitter = new wf_emitter_t(_winType, _win_len, _slide_len, _pardegree, _config.id_inner, _config.n_inner, _config.slide_inner, _role);
                emitters[i] = emitter;
            }
            a2a->add_firstset(emitters, 0, true);
            // create the Win_Seq nodes composed with an orderingNodes
            vector<ff_node *> seqs(_pardegree);
            for (size_t i = 0; i < _pardegree; i++) {
                auto *ord = new OrderingNode<tuple_t, wrapper_in_t>(((_winType == CB) ? ID : TS));
                // configuration structure of the Win_Seq instances
                PatternConfig configSeq(_config.id_inner, _config.n_inner, _config.slide_inner, i, _pardegree, _slide_len);
                auto *seq = new win_seq_t(_winUpdate, _win_len, private_slide, _winType, _name + "_wf", configSeq, _role);
                auto *comb = new ff_comb(ord, seq, true, true);
                seqs[i] = comb;
            }
            a2a->add_secondset(seqs, true);
            w.push_back(a2a);
        }
        ff_farm::add_workers(w);
        // create the Emitter and Collector nodes
        if (_emitter_degree == 1)
            ff_farm::add_emitter(new wf_emitter_t(_winType, _win_len, _slide_len, _pardegree, _config.id_inner, _config.n_inner, _config.slide_inner, _role));
        if (_ordered)
            ff_farm::add_collector(new wf_collector_t());
        else
            ff_farm::add_collector(nullptr);
        // when the Win_Farm will be destroyed we need aslo to destroy the emitter, workers and collector
        ff_farm::cleanup_all();
    }

    //private constructor IV (incremental queries with FAT )
    Win_Farm( f_winlift_t _winLift,
              f_wincombine_t _winCombine,
              uint64_t _win_len,
              uint64_t _slide_len,
              size_t _emitter_degree,
              size_t _pardegree,
              string _name,
              bool _ordered,
              opt_level_t _opt_level,
              PatternConfig _config,
              role_t _role 
    ) : hasComplexWorkers( false ), opt_level( _opt_level ), winType( CB ),
        num_emitters( _emitter_degree )
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
        // check the validity of the emitter degree
        if (_emitter_degree == 0) {
            cerr << RED << "WindFlow Error: at least one emitter is needed" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the parallelism degree
        if (_pardegree == 0) {
            cerr << RED << "WindFlow Error: parallelism degree cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the optimization level
        if (_opt_level != LEVEL0) {
            cerr << YELLOW << "WindFlow Warning: optimization level has no effect" << DEFAULT << endl;
            opt_level = LEVEL0;
        }
        // vector of Win_Seq instances
        vector<ff_node *> w;
        // private sliding factor of each Win_Seq instance
        uint64_t private_slide = _slide_len * _pardegree;
        // standard case: one Emitter node
        if (_emitter_degree == 1) {
            // create the Win_Seq instances
            for (size_t i = 0; i < _pardegree; i++) {
                // configuration structure of the Win_Seq instances
                PatternConfig configSeq(_config.id_inner, _config.n_inner, _config.slide_inner, i, _pardegree, _slide_len);
                auto *seq = new win_seq_t(
                    _winLift, _winCombine, _win_len, private_slide, 
                    _name + "_wf", configSeq, _role
                );
                w.push_back(seq);
            }
        }
        // advanced case: multiple Emitter nodes
        else {
            ff_a2a *a2a = new ff_a2a();
            // create the Emitter nodes
            vector<ff_node *> emitters(_emitter_degree);
            for (size_t i = 0; i < _emitter_degree; i++) {
                auto *emitter = new wf_emitter_t(winType, _win_len, _slide_len, _pardegree, _config.id_inner, _config.n_inner, _config.slide_inner, _role);
                emitters[i] = emitter;
            }
            a2a->add_firstset(emitters, 0, true);
            // create the Win_Seq nodes composed with an orderingNodes
            vector<ff_node *> seqs(_pardegree);
            for (size_t i = 0; i < _pardegree; i++) {
                auto *ord = new OrderingNode<tuple_t, wrapper_in_t>(((winType == CB) ? ID : TS));
                // configuration structure of the Win_Seq instances
                PatternConfig configSeq(_config.id_inner, _config.n_inner, _config.slide_inner, i, _pardegree, _slide_len);
                auto *seq = new win_seq_t(
                    _winLift, _winCombine, _win_len, private_slide, 
                    _name + "_wf", configSeq, _role
                );
                auto *comb = new ff_comb(ord, seq, true, true);
                seqs[i] = comb;
            }
            a2a->add_secondset(seqs, true);
            w.push_back(a2a);
        }
        ff_farm::add_workers(w);
        // create the Emitter and Collector nodes
        if (_emitter_degree == 1)
            ff_farm::add_emitter(new wf_emitter_t(winType, _win_len, _slide_len, _pardegree, _config.id_inner, _config.n_inner, _config.slide_inner, _role));
        if (_ordered)
            ff_farm::add_collector(new wf_collector_t());
        else
            ff_farm::add_collector(nullptr);
        // when the Win_Farm will be destroyed we need aslo to destroy the emitter, workers and collector
        ff_farm::cleanup_all();
    }

    // method to optimize the structure of the Win_Farm pattern
    void optimize_WinFarm(opt_level_t opt)
    {
        if (opt == LEVEL0) // no optimization
            return;
        else if (opt == LEVEL1 || opt == LEVEL2) // optimization level 1
            remove_internal_collectors(*this); // remove all the default collectors in the Win_Farm
        else { // optimization level 2
            cerr << YELLOW << "WindFlow Warning: optimization level not supported yet" << DEFAULT << endl;
            assert(false);
        }
    }

public:
    /** 
     *  \brief Constructor I (Non-Incremental Queries)
     *  
     *  \param _winFunction the non-incremental window processing function
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _emitter_degree number of replicas of the emitter node
     *  \param _pardegree parallelism degree of the Win_Farm pattern
     *  \param _name string with the unique name of the pattern
     *  \param _ordered true if the results of the same key must be emitted in order, false otherwise
     *  \param _opt_level optimization level used to build the pattern
     */ 
    Win_Farm(f_winfunction_t _winFunction,
             uint64_t _win_len,
             uint64_t _slide_len,
             win_type_t _winType,
             size_t _emitter_degree,
             size_t _pardegree,
             string _name,
             bool _ordered=true,
             opt_level_t _opt_level=LEVEL0)
             :
             Win_Farm(_winFunction, _win_len, _slide_len, _winType, _emitter_degree, _pardegree, _name, _ordered, _opt_level, PatternConfig(0, 1, _slide_len, 0, 1, _slide_len), SEQ) {}

    /** 
     *  \brief Constructor II (Incremental Queries)
     *  
     *  \param _winUpdate the incremental window processing function
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _emitter_degree number of replicas of the emitter node
     *  \param _pardegree parallelism degree of the Win_Farm pattern
     *  \param _name string with the unique name of the pattern
     *  \param _ordered true if the results of the same key must be emitted in order, false otherwise
     *  \param _opt_level optimization level used to build the pattern
     */ 
    Win_Farm(f_winupdate_t _winUpdate,
             uint64_t _win_len,
             uint64_t _slide_len,
             win_type_t _winType,
             size_t _emitter_degree,
             size_t _pardegree,
             string _name,
             bool _ordered=true,
             opt_level_t _opt_level=LEVEL0)
             :
             Win_Farm(_winUpdate, _win_len, _slide_len, _winType, _emitter_degree, _pardegree, _name, _ordered, _opt_level, PatternConfig(0, 1, _slide_len, 0, 1, _slide_len), SEQ) {}

    Win_Farm( f_winlift_t _winLift,
              f_wincombine_t _winCombine,
              uint64_t _win_len,
              uint64_t _slide_len,
              size_t _emitter_degree,
              size_t _pardegree,
              string _name,
              bool _ordered=true,
              opt_level_t _opt_level=LEVEL0
    ) : Win_Farm( 
            _winLift, _winCombine, _win_len, _slide_len, _emitter_degree, 
            _pardegree, _name, _ordered, _opt_level, 
            PatternConfig( 0, 1, _slide_len, 0, 1, _slide_len ), SEQ 
        )
    { }
    /** 
     *  \brief Constructor III (Nesting with Pane_Farm)
     *  
     *  \param _pf Pane_Farm instance to be replicated within the Win_Farm pattern
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _emitter_degree number of replicas of the emitter node
     *  \param _pardegree parallelism degree of the Win_Farm pattern
     *  \param _name string with the unique name of the pattern
     *  \param _ordered true if the results of the same key must be emitted in order, false otherwise
     *  \param _opt_level optimization level used to build the pattern
     */ 
    Win_Farm(const pane_farm_t &_pf,
             uint64_t _win_len,
             uint64_t _slide_len,
             win_type_t _winType,
             size_t _emitter_degree,
             size_t _pardegree,
             string _name,
             bool _ordered=true,
             opt_level_t _opt_level=LEVEL0): hasComplexWorkers(true), opt_level(_opt_level), winType(_winType), num_emitters(_emitter_degree)
    {
        // type of the Pane_Farm to be created within the Win_Farm pattern
        using panewrap_farm_t = Pane_Farm<tuple_t, result_t, wrapper_in_t>;
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the emitter degree
        if (_emitter_degree == 0) {
            cerr << RED << "WindFlow Error: at least one emitter is needed" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the parallelism degree
        if (_pardegree == 0) {
            cerr << RED << "WindFlow Error: parallelism degree cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the compatibility of the windowing parameters
        if (_pf.win_len != _win_len || _pf.slide_len != _slide_len || _pf.winType != _winType) {
            cerr << RED << "WindFlow Error: incompatible windowing parameters" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // vector of Pane_Farm instances
        vector<ff_node *> w;
        // standard case: one Emitter node
        if (_emitter_degree == 1) {
            // create the Pane_Farm instances starting from the input one
            for (size_t i = 0; i < _pardegree; i++) {
                // configuration structure of the Pane_Farm instances
                PatternConfig configPF(0, 1, _slide_len, i, _pardegree, _slide_len);
                // create the correct Pane_Farm instance
                panewrap_farm_t *pf_W = nullptr;
                if (_pf.isNICPLQ && _pf.isNICWLQ) // PLQ and WLQ are non-incremental
                    pf_W = new panewrap_farm_t(_pf.plqFunction, _pf.wlqFunction, _pf.win_len, _pf.slide_len * _pardegree, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _name + "_wf_" + to_string(i), false, _pf.opt_level, configPF);
                if (!_pf.isNICPLQ && !_pf.isNICWLQ) // PLQ and WLQ are incremental
                    pf_W = new panewrap_farm_t(_pf.plqUpdate, _pf.wlqUpdate, _pf.win_len, _pf.slide_len * _pardegree, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _name + "_wf_" + to_string(i), false, _pf.opt_level, configPF);
                if (_pf.isNICPLQ && !_pf.isNICWLQ) // PLQ is non-incremental and the WLQ is incremental
                    pf_W = new panewrap_farm_t(_pf.plqFunction, _pf.wlqUpdate, _pf.win_len, _pf.slide_len * _pardegree, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _name + "_wf_" + to_string(i), false, _pf.opt_level, configPF);
                if (!_pf.isNICPLQ && _pf.isNICWLQ) // PLQ is incremental and the WLQ is non-incremental
                    pf_W = new panewrap_farm_t(_pf.plqUpdate, _pf.wlqFunction, _pf.win_len, _pf.slide_len * _pardegree, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _name + "_wf_" + to_string(i), false, _pf.opt_level, configPF);
                w.push_back(pf_W);
            }
        }
        // advanced case: multiple Emitter nodes
        else {
            ff_a2a *a2a = new ff_a2a();
            // create the Emitter nodes
            vector<ff_node *> emitters(_emitter_degree);
            for (size_t i = 0; i < _emitter_degree; i++) {
                auto *emitter = new wf_emitter_t(_winType, _win_len, _slide_len, _pardegree, 0, 1, _slide_len, SEQ);
                emitters[i] = emitter;
            }
            a2a->add_firstset(emitters, 0, true);
            // create the correct Pane_Farm instances
            vector<ff_node *> pfs(_pardegree);
            for (size_t i = 0; i < _pardegree; i++) {
                // an ordering node must be composed before the first node of the Pane_Farm instance
                auto *ord = new OrderingNode<tuple_t, wrapper_in_t>(((_winType == CB) ? ID : TS));
                // configuration structure of the Pane_Farm instances
                PatternConfig configPF(0, 1, _slide_len, i, _pardegree, _slide_len);
                // create the correct Pane_Farm instance
                panewrap_farm_t *pf_W = nullptr;
                if (_pf.isNICPLQ && _pf.isNICWLQ) // PLQ and WLQ are non-incremental
                    pf_W = new panewrap_farm_t(_pf.plqFunction, _pf.wlqFunction, _pf.win_len, _pf.slide_len * _pardegree, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _name + "_wf_" + to_string(i), false, _pf.opt_level, configPF);
                if (!_pf.isNICPLQ && !_pf.isNICWLQ) // PLQ and WLQ are incremental
                    pf_W = new panewrap_farm_t(_pf.plqUpdate, _pf.wlqUpdate, _pf.win_len, _pf.slide_len * _pardegree, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _name + "_wf_" + to_string(i), false, _pf.opt_level, configPF);
                if (_pf.isNICPLQ && !_pf.isNICWLQ) // PLQ is non-incremental and the WLQ is incremental
                    pf_W = new panewrap_farm_t(_pf.plqFunction, _pf.wlqUpdate, _pf.win_len, _pf.slide_len * _pardegree, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _name + "_wf_" + to_string(i), false, _pf.opt_level, configPF);
                if (!_pf.isNICPLQ && _pf.isNICWLQ) // PLQ is incremental and the WLQ is non-incremental
                    pf_W = new panewrap_farm_t(_pf.plqUpdate, _pf.wlqFunction, _pf.win_len, _pf.slide_len * _pardegree, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _name + "_wf_" + to_string(i), false, _pf.opt_level, configPF);
                // combine the first node of the Pane_Farm instance with the buffering node
                combine_with_firststage(*pf_W, ord, true);
                pfs[i] = pf_W;
            }
            a2a->add_secondset(pfs, true);
            w.push_back(a2a);
        }
        ff_farm::add_workers(w);
        // create the Emitter and Collector nodes
        if (_emitter_degree == 1)
            ff_farm::add_emitter(new wf_emitter_t(_winType, _win_len, _slide_len, _pardegree, 0, 1, _slide_len, SEQ));
        if (_ordered)
            ff_farm::add_collector(new wf_collector_t());
        else
            ff_farm::add_collector(nullptr);
        // optimization process according to the provided optimization level
        this->optimize_WinFarm(_opt_level);
        // when the Win_Farm will be destroyed we need aslo to destroy the emitter, workers and collector
        ff_farm::cleanup_all();
    }

    /** 
     *  \brief Constructor IV (Nesting with Win_MapReduce)
     *  
     *  \param _wm Win_MapReduce instance to be replicated within the Win_Farm pattern
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _emitter_degree number of replicas of the emitter node
     *  \param _pardegree parallelism degree of the Win_Farm pattern
     *  \param _name string with the unique name of the pattern
     *  \param _ordered true if the results of the same key must be emitted in order, false otherwise
     *  \param _opt_level optimization level used to build the pattern
     */ 
    Win_Farm(const win_mapreduce_t &_wm,
             uint64_t _win_len,
             uint64_t _slide_len,
             win_type_t _winType,
             size_t _emitter_degree,
             size_t _pardegree,
             string _name,
             bool _ordered=true,
             opt_level_t _opt_level=LEVEL0): hasComplexWorkers(true), opt_level(_opt_level), winType(_winType), num_emitters(_emitter_degree)
    {
        // type of the Win_MapReduce to be created within the Win_Farm pattern
        using winwrap_map_t = Win_MapReduce<tuple_t, result_t, wrapper_in_t>;
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the emitter degree
        if (_emitter_degree == 0) {
            cerr << RED << "WindFlow Error: at least one emitter is needed" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the parallelism degree
        if (_pardegree == 0) {
            cerr << RED << "WindFlow Error: parallelism degree cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the compatibility of the windowing parameters
        if (_wm.win_len != _win_len || _wm.slide_len != _slide_len || _wm.winType != _winType) {
            cerr << RED << "WindFlow Error: incompatible windowing parameters" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // vector of Win_MapReduce instances
        vector<ff_node *> w;
        // standard case: one Emitter node
        if (_emitter_degree == 1) {
            // create the Win_MapReduce instances starting from the input one
            for (size_t i = 0; i < _pardegree; i++) {
                // configuration structure of the Win_mapReduce instances
                PatternConfig configWM(0, 1, _slide_len, i, _pardegree, _slide_len);
                // create the correct Win_MapReduce instance
                winwrap_map_t *wm_W = nullptr;
                if (_wm.isNICMAP && _wm.isNICREDUCE) // MAP and REDUCE are non-incremental
                    wm_W = new winwrap_map_t(_wm.mapFunction, _wm.reduceFunction, _wm.win_len, _wm.slide_len * _pardegree, _wm.winType, _wm.map_degree, _wm.reduce_degree, _name + "_wf_" + to_string(i), false, _wm.opt_level, configWM);
                if (!_wm.isNICMAP && !_wm.isNICREDUCE) // MAP and REDUCE are incremental
                    wm_W = new winwrap_map_t(_wm.mapUpdate, _wm.reduceUpdate, _wm.win_len, _wm.slide_len * _pardegree, _wm.winType, _wm.map_degree, _wm.reduce_degree, _name + "_wf_" + to_string(i), false, _wm.opt_level, configWM);
                if (_wm.isNICMAP && !_wm.isNICREDUCE) // MAP is non-incremental and the REDUCE is incremental
                    wm_W = new winwrap_map_t(_wm.mapFunction, _wm.reduceUpdate, _wm.win_len, _wm.slide_len * _pardegree, _wm.winType, _wm.map_degree, _wm.reduce_degree, _name + "_wf_" + to_string(i), false, _wm.opt_level, configWM);
                if (!_wm.isNICMAP && _wm.isNICREDUCE) // MAP is incremental and the REDUCE is non-incremental
                    wm_W = new winwrap_map_t(_wm.mapUpdate, _wm.reduceFunction, _wm.win_len, _wm.slide_len * _pardegree, _wm.winType, _wm.map_degree, _wm.reduce_degree, _name + "_wf_" + to_string(i), false, _wm.opt_level, configWM);
                w.push_back(wm_W);
            }
        }
        // advanced case: multiple Emitter nodes
        else {
            ff_a2a *a2a = new ff_a2a();
            // create the Emitter nodes
            vector<ff_node *> emitters(_emitter_degree);
            for (size_t i = 0; i < _emitter_degree; i++) {
                auto *emitter = new wf_emitter_t(_winType, _win_len, _slide_len, _pardegree, 0, 1, _slide_len, SEQ);
                emitters[i] = emitter;
            }
            a2a->add_firstset(emitters, 0, true);
            // create the correct Win_MapReduce instances
            vector<ff_node *> wms(_pardegree);
            for (size_t i = 0; i < _pardegree; i++) {
                // an ordering node must be composed before the first node of the Win_MapReduce instance
                auto *ord = new OrderingNode<tuple_t, wrapper_in_t>(((_winType == CB) ? ID : TS));
                // configuration structure of the Win_MapReduce instances
                PatternConfig configWM(0, 1, _slide_len, i, _pardegree, _slide_len);
                // create the correct Win_MapReduce instance
                winwrap_map_t *wm_W = nullptr;
                if (_wm.isNICMAP && _wm.isNICREDUCE) // MAP and REDUCE are non-incremental
                    wm_W = new winwrap_map_t(_wm.mapFunction, _wm.reduceFunction, _wm.win_len, _wm.slide_len * _pardegree, _wm.winType, _wm.map_degree, _wm.reduce_degree, _name + "_wf_" + to_string(i), false, _wm.opt_level, configWM);
                if (!_wm.isNICMAP && !_wm.isNICREDUCE) // MAP and REDUCE are incremental
                    wm_W = new winwrap_map_t(_wm.mapUpdate, _wm.reduceUpdate, _wm.win_len, _wm.slide_len * _pardegree, _wm.winType, _wm.map_degree, _wm.reduce_degree, _name + "_wf_" + to_string(i), false, _wm.opt_level, configWM);
                if (_wm.isNICMAP && !_wm.isNICREDUCE) // MAP is non-incremental and the REDUCE is incremental
                    wm_W = new winwrap_map_t(_wm.mapFunction, _wm.reduceUpdate, _wm.win_len, _wm.slide_len * _pardegree, _wm.winType, _wm.map_degree, _wm.reduce_degree, _name + "_wf_" + to_string(i), false, _wm.opt_level, configWM);
                if (!_wm.isNICMAP && _wm.isNICREDUCE) // MAP is incremental and the REDUCE is non-incremental
                    wm_W = new winwrap_map_t(_wm.mapUpdate, _wm.reduceFunction, _wm.win_len, _wm.slide_len * _pardegree, _wm.winType, _wm.map_degree, _wm.reduce_degree, _name + "_wf_" + to_string(i), false, _wm.opt_level, configWM);                
                // combine the first node of the Win_MapReduce instance with the buffering node
                combine_with_firststage(*wm_W, ord, true);
                wms[i] = wm_W;
            }
            a2a->add_secondset(wms, true);
            w.push_back(a2a);    
        }
        ff_farm::add_workers(w);
        // create the Emitter and Collector nodes
        if (_emitter_degree == 1)
            ff_farm::add_emitter(new wf_emitter_t(_winType, _win_len, _slide_len, _pardegree, 0, 1, _slide_len, SEQ));
        if (_ordered)
            ff_farm::add_collector(new wf_collector_t());
        else
            ff_farm::add_collector(nullptr);
        // optimization process according to the provided optimization level
        this->optimize_WinFarm(_opt_level);
        // when the Win_Farm will be destroyed we need aslo to destroy the emitter, workers and collector
        ff_farm::cleanup_all();
    }

    /** 
     *  \brief Check whether the Win_Farm has been instantiated with complex patterns inside
     *  \return true if the Win_Farm has complex patterns inside
     */
    bool useComplexNesting() { return hasComplexWorkers; }

    /** 
     *  \brief Get the optimization level used to build the pattern
     *  \return adopted utilization level by the pattern
     */
    opt_level_t getOptLevel() { return opt_level; }

    /** 
     *  \brief Get the window type (CB or TB) utilized by the pattern
     *  \return adopted windowing semantics (count- or time-based)
     */
    win_type_t getWinType() { return winType; }

    /** 
     *  \brief Get the number of emitters utilized by the pattern
     *  \return number of emitters
     */
    size_t getNumEmitters() { return num_emitters; }
};

#endif
