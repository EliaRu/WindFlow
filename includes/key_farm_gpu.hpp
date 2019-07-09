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
 *  @file    key_farm_gpu.hpp
 *  @author  Gabriele Mencagli
 *  @date    21/05/2018
 *  
 *  @brief Key_Farm_GPU pattern executing a windowed transformation in parallel
 *         on a CPU+GPU system
 *  
 *  @section Key_Farm_GPU (Description)
 *  
 *  This file implements the Key_Farm_GPU pattern able to executes windowed queries
 *  on a heterogeneous system (CPU+GPU). The pattern prepares batches of input tuples
 *  in parallel on the CPU cores and offloads on the GPU the parallel processing of the
 *  windows within each batch. Batches of different sub-streams can be executed in
 *  parallel while consecutive batches of the same sub-stream are preparel on the CPU and
 *  offloaded on the GPU sequentially.
 *  
 *  The template parameters tuple_t and result_t must be default constructible, with a copy
 *  Constructor and copy assignment operator, and they must provide and implement the
 *  setControlFields() and getControlFields() methods. The third template argument win_F_t
 *  is the type of the callable object to be used for GPU processing.
 */ 

#ifndef KEY_FARM_GPU_H
#define KEY_FARM_GPU_H

/// includes
#include <ff/farm.hpp>
#include <ff/optimize.hpp>
#include <win_seq_gpu.hpp>
#include <kf_nodes.hpp>
#include <pane_farm_gpu.hpp>
#include <win_mapreduce_gpu.hpp>
#include <tree_combiner.hpp>
#include <transformations.hpp>

/** 
 *  \class Key_Farm_GPU
 *  
 *  \brief Key_Farm_GPU pattern executing a windowed transformation in parallel on a CPU+GPU system
 *  
 *  This class implements the Key_Farm_GPU pattern. The pattern prepares in parallel distinct
 *  batches of tuples (on the CPU cores) and offloads the processing of the batches on the GPU
 *  by computing in parallel all the windows within a batch on the CUDA cores of the GPU. Batches
 *  with tuples of same sub-stream are prepared/offloaded sequentially on the CPU.
 */ 
template<typename tuple_t, typename result_t, typename win_F_t>
class Key_Farm_GPU: public ff_farm
{
public:
    /// function type to map the key hashcode onto an identifier starting from zero to pardegree-1
    using routing_func_t = function<size_t(size_t, size_t)>;
    /// type of the Pane_Farm_GPU passed to the proper nesting Constructor
    using pane_farm_gpu_t = Pane_Farm_GPU<tuple_t, result_t, win_F_t>;
    /// type of the Win_MapReduce_GPU passed to the proper nesting Constructor
    using win_mapreduce_gpu_t = Win_MapReduce_GPU<tuple_t, result_t, win_F_t>;

private:
    // type of the wrapper of input tuples
    using wrapper_in_t = wrapper_tuple_t<tuple_t>;
    // type of the Win_Seq_GPU to be created within the regular constructor
    using win_seq_gpu_t = Win_Seq_GPU<tuple_t, result_t, win_F_t, wrapper_in_t>;

    using win_fat_gpu_t = Win_FAT_GPU<tuple_t, result_t, win_F_t, wrapper_in_t>;

    using f_winlift_t =
        function<int( size_t, uint64_t, const tuple_t &, result_t & )>;
    // type of the KF_Emitter node
    using kf_emitter_t = KF_Emitter<tuple_t>;
    // type of the KF_Collector node
    using kf_collector_t = KF_NestedCollector<result_t>;
    // friendships with other classes in the library
    template<typename T>
    friend auto get_KF_GPU_nested_type(T);
    // flag stating whether the Key_Farm_GPU has been instantiated with complex workers (Pane_Farm_GPU or Win_MapReduce_GPU)
    bool hasComplexWorkers;
    // optimization level of the Key_Farm_GPU
    opt_level_t outer_opt_level;
    // optimization level of the inner patterns
    opt_level_t inner_opt_level;
    // type of the inner patterns
    pattern_t inner_type;
    // parallelism of the Key_Farm_GPU
    size_t parallelism;
    // parallelism degrees of the inner patterns
    size_t inner_parallelism_1;
    size_t inner_parallelism_2;
    // window type (CB or TB)
    win_type_t winType;

    // Private Constructor (stub)
    Key_Farm_GPU() {}

    // method to optimize the structure of the Key_Farm_GPU pattern
    template<typename inner_emitter_t>
    void optimize_KeyFarmGPU(opt_level_t opt)
    {
        if (opt == LEVEL0) // no optimization
            return;
        else if (opt == LEVEL1) // optimization level 1
            remove_internal_collectors(*this); // remove all the default collectors in the Key_Farm_GPU
        else { // optimization level 2
            kf_emitter_t *kf_e = static_cast<kf_emitter_t *>(this->getEmitter());
            auto &oldWorkers = this->getWorkers();
            vector<inner_emitter_t *> Es;
            bool tobeTransformmed = true;
            // change the workers by removing their first emitter (if any)
            for (auto *w: oldWorkers) {
                ff_pipeline *pipe = static_cast<ff_pipeline *>(w);
                ff_node *e = remove_emitter_from_pipe(*pipe);
                if (e == nullptr)
                    tobeTransformmed = false;
                else {
                    inner_emitter_t *my_e = static_cast<inner_emitter_t *>(e);
                    Es.push_back(my_e);
                }
            }
            if (tobeTransformmed) {
                // create the tree emitter
                auto *treeEmitter = new TreeComb<kf_emitter_t, inner_emitter_t>(kf_e, Es);
                this->cleanup_emitter(false);
                this->change_emitter(treeEmitter, true);
            }
            remove_internal_collectors(*this);
            return;
        }
    }

public:
    /** 
     *  \brief Constructor I
     *  
     *  \param _win_func the non-incremental window processing function (CPU/GPU function)
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _pardegree parallelism degree of the Key_Farm_GPU pattern
     *  \param _batch_len no. of windows in a batch (i.e. 1 window mapped onto 1 CUDA thread)
     *  \param _n_thread_block number of threads (i.e. windows) per block
     *  \param _name string with the unique name of the pattern
     *  \param _scratchpad_size size in bytes of the scratchpad area per CUDA thread (on the GPU)
     *  \param _routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
     *  \param _opt_level optimization level used to build the pattern
     */ 
    Key_Farm_GPU(win_F_t _win_func,
                 uint64_t _win_len,
                 uint64_t _slide_len,
                 win_type_t _winType,
                 size_t _pardegree,
                 size_t _batch_len,
                 size_t _n_thread_block,
                 string _name,
                 size_t _scratchpad_size,
                 routing_func_t _routing_func,
                 opt_level_t _opt_level)
                 :
                 hasComplexWorkers(false),
                 outer_opt_level(_opt_level),
                 inner_opt_level(LEVEL0),
                 inner_type(SEQ_GPU),
                 parallelism(_pardegree),
                 inner_parallelism_1(1),
                 inner_parallelism_2(0),
                 winType(_winType)
    {
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the parallelism degree
        if (_pardegree == 0) {
            cerr << RED << "WindFlow Error: parallelism degree cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the batch length
        if (_batch_len == 0) {
            cerr << RED << "WindFlow Error: batch length cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the optimization level
        if (_opt_level != LEVEL0) {
            //cerr << YELLOW << "WindFlow Warning: optimization level has no effect" << DEFAULT << endl;
            outer_opt_level = LEVEL0;
        }
        // vector of Win_Seq_GPU
        vector<ff_node *> w(_pardegree);
        // create the Win_Seq_GPU
        for (size_t i = 0; i < _pardegree; i++) {
            auto *seq = new win_seq_gpu_t(_win_func, _win_len, _slide_len, _winType, _batch_len, _n_thread_block, _name + "_kf", _scratchpad_size);
            w[i] = seq;
        }
        ff_farm::add_workers(w);
        ff_farm::add_collector(nullptr);
        // create the Emitter node
        ff_farm::add_emitter(new kf_emitter_t(_routing_func, _pardegree));
        // when the Key_Farm_GPU will be destroyed we need aslo to destroy the emitter, workers and collector
        ff_farm::cleanup_all();
    }
    
    Key_Farm_GPU(f_winlift_t _winLift,
                 win_F_t _winFunction,
                 uint64_t _win_len,
                 uint64_t _slide_len,
                 size_t _batch_len,
                 bool _rebuildFAT,
                 size_t _pardegree,
                 string _name,
                 f_routing_t _routing=[](size_t k, size_t n) { return k%n; },
                 opt_level_t _opt_level=LEVEL0): hasComplexWorkers(false), opt_level(_opt_level), winType(CB)
    {
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the parallelism degree
        if (_pardegree == 0) {
            cerr << RED << "WindFlow Error: parallelism degree cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the batch length
        if (_batch_len == 0) {
            cerr << RED << "WindFlow Error: batch length cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the optimization level
        if (_opt_level != LEVEL0) {
            cerr << YELLOW << "WindFlow Warning: optimization level has no effect" << DEFAULT << endl;
            opt_level = LEVEL0;
        }
        // vector of Win_Seq_GPU instances
        vector<ff_node *> w(_pardegree);
        // create the Win_Seq_GPU instances
        for (size_t i = 0; i < _pardegree; i++) {
            auto *seq = new win_fat_gpu_t(  _winLift, _winFunction, _win_len, _slide_len, _batch_len, _rebuildFAT, _name + "_kf" );
            w[i] = seq;
        }
        ff_farm::add_workers(w);
        ff_farm::add_collector(nullptr);
        // create the Emitter node
        ff_farm::add_emitter(new kf_emitter_t(_routing, _pardegree));
        // when the Key_Farm_GPU will be destroyed we need aslo to destroy the emitter, workers and collector
        ff_farm::cleanup_all();
    }

    Key_Farm_GPU(f_winlift_t _winLift,
                 win_F_t _winFunction,
                 uint64_t _win_len,
                 uint64_t _slide_len,
                 uint64_t _quantum,
                 size_t _batch_len,
                 bool _rebuildFAT,
                 size_t _pardegree,
                 string _name,
                 f_routing_t _routing=[](size_t k, size_t n) { return k%n; },
                 opt_level_t _opt_level=LEVEL0): hasComplexWorkers(false), opt_level(_opt_level), winType(CB)
    {
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the parallelism degree
        if (_pardegree == 0) {
            cerr << RED << "WindFlow Error: parallelism degree cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the batch length
        if (_batch_len == 0) {
            cerr << RED << "WindFlow Error: batch length cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the optimization level
        if (_opt_level != LEVEL0) {
            cerr << YELLOW << "WindFlow Warning: optimization level has no effect" << DEFAULT << endl;
            opt_level = LEVEL0;
        }
        // vector of Win_Seq_GPU instances
        vector<ff_node *> w(_pardegree);
        // create the Win_Seq_GPU instances
        for (size_t i = 0; i < _pardegree; i++) {
            auto *seq = new win_fat_gpu_t(  _winLift, _winFunction, _win_len, _slide_len, _quantum, _batch_len, _rebuildFAT, _name + "_kf" );
            w[i] = seq;
        }
        ff_farm::add_workers(w);
        ff_farm::add_collector(nullptr);
        // create the Emitter node
        ff_farm::add_emitter(new kf_emitter_t(_routing, _pardegree));
        // when the Key_Farm_GPU will be destroyed we need aslo to destroy the emitter, workers and collector
        ff_farm::cleanup_all();
    }

    /** 
     *  \brief Constructor II (Nesting with Pane_Farm_GPU)
     *  
     *  \param _pf Pane_Farm_GPU to be replicated within the Key_Farm_GPU pattern
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _pardegree parallelism degree of the Key_Farm_GPU pattern
     *  \param _batch_len no. of windows in a batch (i.e. 1 window mapped onto 1 CUDA thread)
     *  \param _n_thread_block number of threads (i.e. windows) per block
     *  \param _name string with the unique name of the pattern
     *  \param _scratchpad_size size in bytes of the scratchpad area per CUDA thread (on the GPU)
     *  \param _routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
     *  \param _opt_level optimization level used to build the pattern
     */ 
    Key_Farm_GPU(const pane_farm_gpu_t &_pf,
                 uint64_t _win_len,
                 uint64_t _slide_len,
                 win_type_t _winType,
                 size_t _pardegree,
                 size_t _batch_len,
                 size_t _n_thread_block,
                 string _name,
                 size_t _scratchpad_size,
                 routing_func_t _routing_func,
                 opt_level_t _opt_level)
                 :
                 hasComplexWorkers(true),
                 outer_opt_level(_opt_level),
                 inner_type(PF_GPU),
                 parallelism(_pardegree),
                 winType(_winType)
    {
        // type of the PLQ emitter in the first stage of the Pane_Farm_GPU
        using plq_emitter_t = WF_Emitter<tuple_t>;        
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the parallelism degree
        if (_pardegree == 0) {
            cerr << RED << "WindFlow Error: parallelism degree cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the batch length
        if (_batch_len == 0) {
            cerr << RED << "WindFlow Error: batch length cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the compatibility of the windowing/batching parameters
        if (_pf.win_len != _win_len || _pf.slide_len != _slide_len || _pf.winType != _winType || _pf.batch_len != _batch_len || _pf.n_thread_block != _n_thread_block) {
            cerr << RED << "WindFlow Error: incompatible windowing and batching parameters" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        inner_opt_level = _pf.opt_level;
        inner_parallelism_1 = _pf.plq_degree;
        inner_parallelism_2 = _pf.wlq_degree;
        // vector of Pane_Farm_GPU
        vector<ff_node *> w(_pardegree);
        // create the Pane_Farm_GPU starting from the passed one
        for (size_t i = 0; i < _pardegree; i++) {
            // configuration structure of the Pane_Farm_GPU
            PatternConfig configPF(0, 1, _slide_len, 0, 1, _slide_len);
            // create the correct Pane_Farm_GPU
            pane_farm_gpu_t *pf_W = nullptr;
            if (_pf.isGPUPLQ) {
                if (_pf.isNICWLQ)
                    pf_W = new pane_farm_gpu_t(_pf.gpuFunction, _pf.wlq_func, _pf.win_len, _pf.slide_len, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _pf.batch_len, _pf.n_thread_block, _name + "_kf_" + to_string(i), _pf.scratchpad_size, false, _pf.opt_level, configPF);
                else
                    pf_W = new pane_farm_gpu_t(_pf.gpuFunction, _pf.wlqupdate_func, _pf.win_len, _pf.slide_len, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _pf.batch_len, _pf.n_thread_block, _name + "_kf_" + to_string(i), _pf.scratchpad_size, false, _pf.opt_level, configPF);
            }
            else {
                if (_pf.isNICPLQ)
                    pf_W = new pane_farm_gpu_t(_pf.plq_func, _pf.gpuFunction, _pf.win_len, _pf.slide_len, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _pf.batch_len, _pf.n_thread_block, _name + "_kf_" + to_string(i), _pf.scratchpad_size, false, _pf.opt_level, configPF);
                else
                    pf_W = new pane_farm_gpu_t(_pf.plqupdate_func, _pf.gpuFunction, _pf.win_len, _pf.slide_len, _pf.winType, _pf.plq_degree, _pf.wlq_degree, _pf.batch_len, _pf.n_thread_block, _name + "_kf_" + to_string(i), _pf.scratchpad_size, false, _pf.opt_level, configPF);
            }
            w[i] = pf_W;
        }
        ff_farm::add_workers(w);
        // create the Emitter and Collector nodes
        ff_farm::add_collector(new kf_collector_t());
        ff_farm::add_emitter(new kf_emitter_t(_routing_func, _pardegree));
        // optimization process according to the provided optimization level
        this->optimize_KeyFarmGPU<plq_emitter_t>(_opt_level);
        // when the Key_Farm_GPU will be destroyed we need aslo to destroy the emitter, workers and collector
        ff_farm::cleanup_all();
    }

    /** 
     *  \brief Constructor III (Nesting with Win_MapReduce_GPU)
     *  
     *  \param _wm Win_MapReduce_GPU to be replicated within the Key_Farm_GPU pattern
     *  \param _win_len window length (in no. of tuples or in time units)
     *  \param _slide_len slide length (in no. of tuples or in time units)
     *  \param _winType window type (count-based CB or time-based TB)
     *  \param _pardegree parallelism degree of the Key_Farm_GPU pattern
     *  \param _batch_len no. of windows in a batch (i.e. 1 window mapped onto 1 CUDA thread)
     *  \param _n_thread_block number of threads (i.e. windows) per block
     *  \param _name string with the unique name of the pattern
     *  \param _scratchpad_size size in bytes of the scratchpad area per CUDA thread (on the GPU)
     *  \param _routing_func function to map the key hashcode onto an identifier starting from zero to pardegree-1
     *  \param _opt_level optimization level used to build the pattern
     */ 
    Key_Farm_GPU(const win_mapreduce_gpu_t &_wm,
                 uint64_t _win_len,
                 uint64_t _slide_len,
                 win_type_t _winType,
                 size_t _pardegree,
                 size_t _batch_len,
                 size_t _n_thread_block,
                 string _name,
                 size_t _scratchpad_size,
                 routing_func_t _routing_func,
                 opt_level_t _opt_level)
                 :
                 hasComplexWorkers(true),
                 outer_opt_level(_opt_level),
                 inner_type(WMR_GPU),
                 parallelism(_pardegree),
                 winType(_winType)
    {
        // type of the MAP emitter in the first stage of the Win_MapReduce_GPU
        using map_emitter_t = WinMap_Emitter<tuple_t>;        
        // check the validity of the windowing parameters
        if (_win_len == 0 || _slide_len == 0) {
            cerr << RED << "WindFlow Error: window length or slide cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the parallelism degree
        if (_pardegree == 0) {
            cerr << RED << "WindFlow Error: parallelism degree cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the validity of the batch length
        if (_batch_len == 0) {
            cerr << RED << "WindFlow Error: batch length cannot be zero" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        // check the compatibility of the windowing/batching parameters
        if (_wm.win_len != _win_len || _wm.slide_len != _slide_len || _wm.winType != _winType || _wm.batch_len != _batch_len || _wm.n_thread_block != _n_thread_block) {
            cerr << RED << "WindFlow Error: incompatible windowing and batching parameters" << DEFAULT << endl;
            exit(EXIT_FAILURE);
        }
        inner_opt_level = _wm.opt_level;
        inner_parallelism_1 = _wm.map_degree;
        inner_parallelism_2 = _wm.reduce_degree;
        // vector of Win_MapReduce_GPU
        vector<ff_node *> w(_pardegree);
        // create the Win_MapReduce_GPU starting from the passed one
        for (size_t i = 0; i < _pardegree; i++) {
            // configuration structure of the Win_MapReduce_GPU
            PatternConfig configWM(0, 1, _slide_len, 0, 1, _slide_len);
            // create the correct Win_MapReduce_GPU
            win_mapreduce_gpu_t *wm_W = nullptr;
            if (_wm.isGPUMAP) {
                if (_wm.isNICREDUCE)
                    wm_W = new win_mapreduce_gpu_t(_wm.gpuFunction, _wm.reduce_func, _wm.win_len, _wm.slide_len, _wm.winType, _wm.map_degree, _wm.reduce_degree, _wm.batch_len, _wm.n_thread_block, _name + "_kf_" + to_string(i), _wm.scratchpad_size, false, _wm.opt_level, configWM);
                else
                    wm_W = new win_mapreduce_gpu_t(_wm.gpuFunction, _wm.reduceupdate_func, _wm.win_len, _wm.slide_len, _wm.winType, _wm.map_degree, _wm.reduce_degree, _wm.batch_len, _wm.n_thread_block, _name + "_kf_" + to_string(i), _wm.scratchpad_size, false, _wm.opt_level, configWM);
            }
            else {
                if (_wm.isNICMAP)
                    wm_W = new win_mapreduce_gpu_t(_wm.map_func, _wm.gpuFunction, _wm.win_len, _wm.slide_len , _wm.winType, _wm.map_degree, _wm.reduce_degree, _wm.batch_len, _wm.n_thread_block, _name + "_kf_" + to_string(i), _wm.scratchpad_size, false, _wm.opt_level, configWM);
                else
                    wm_W = new win_mapreduce_gpu_t(_wm.mapupdate_func, _wm.gpuFunction, _wm.win_len, _wm.slide_len, _wm.winType, _wm.map_degree, _wm.reduce_degree, _wm.batch_len, _wm.n_thread_block, _name + "_kf_" + to_string(i), _wm.scratchpad_size, false, _wm.opt_level, configWM);
            }
            w[i] = wm_W;
        }
        ff_farm::add_workers(w);
        // create the Emitter and Collector nodes
        ff_farm::add_collector(new kf_collector_t());
        ff_farm::add_emitter(new kf_emitter_t(_routing_func, _pardegree));
        // optimization process according to the provided optimization level
        this->optimize_KeyFarmGPU<map_emitter_t>(_opt_level);
        // when the Key_Farm_GPU will be destroyed we need aslo to destroy the emitter, workers and collector
        ff_farm::cleanup_all();
    }

    /** 
     *  \brief Check whether the Key_Farm_GPU has been instantiated with complex patterns inside
     *  \return true if the Key_Farm_GPU has complex patterns inside
     */ 
    bool useComplexNesting() const { return hasComplexWorkers; }

    /** 
     *  \brief Get the optimization level used to build the pattern
     *  \return adopted utilization level by the pattern
     */ 
    opt_level_t getOptLevel() const { return outer_opt_level; }

    /** 
     *  \brief Type of the inner patterns used by this Key_Farm_GPU
     *  \return type of the inner patterns
     */ 
    pattern_t getInnerType() const { return inner_type; }

    /** 
     *  \brief Get the optimization level of the inner patterns within this Key_Farm_GPU
     *  \return adopted utilization level by the inner patterns
     */ 
    opt_level_t getInnerOptLevel() const { return inner_opt_level; }

    /** 
     *  \brief Get the parallelism degree of the Key_Farm_GPU
     *  \return parallelism degree of the Key_Farm_GPU
     */ 
    size_t getParallelism() const { return parallelism; }        

    /** 
     *  \brief Get the parallelism degrees of the inner patterns within this Key_Farm_GPU
     *  \return parallelism degrees of the inner patterns
     */ 
    pair<size_t, size_t> getInnerParallelism() const { return make_pair(inner_parallelism_1, inner_parallelism_2); }

    /** 
     *  \brief Get the window type (CB or TB) utilized by the pattern
     *  \return adopted windowing semantics (count- or time-based)
     */ 
    win_type_t getWinType() const { return winType; }
};

#endif
