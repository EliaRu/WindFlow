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
 *  @file    transformations.hpp
 *  @author  Gabriele Mencagli
 *  @date    15/06/2019
 *  
 *  @brief Set of transformations useful for patterns optimization
 *  
 *  @section Transformations (Description)
 *  
 *  This file implements a set of transformations useful to optimize
 *  the structure of the WindFlow patterns.
 */ 

#ifndef TRANSF_H
#define TRANSF_H

// includes
#include <ff/ff.hpp>

using namespace ff;
using namespace std;

// struct of the dummy multi-input node
struct dummy_mi: ff_minode { void *svc(void *in) { return in; } };

// struct of the dummy multi-ouput node
struct dummy_mo: ff_monode { void *svc(void *in) { return in; } };

// method to transform a farm into a all-to-all (with collector)
ff_a2a *farm2A2A_collector(ff_farm *farm)
{
    // create the a2a
    ff_a2a *a2a = new ff_a2a();
    auto &ws = farm->getWorkers();
    vector<ff_node *> first_set;
    for (auto *w: ws) {
        ff_comb *comb = new ff_comb(w, new dummy_mo(), false, true);
        first_set.push_back(comb);
    }
    a2a->add_firstset(first_set, 0, true);
    vector<ff_node *> second_set;
    second_set.push_back(farm->getCollector());
    a2a->add_secondset(second_set, false);
    return a2a;
}

// method to transform a farm into a all-to-all (with emitter)
ff_a2a *farm2A2A_emitter(ff_farm *farm)
{
    // create the a2a
    ff_a2a *a2a = new ff_a2a();
    vector<ff_node *> first_set;
    first_set.push_back(farm->getEmitter());
    a2a->add_firstset(first_set, 0, false);
    auto &ws = farm->getWorkers();
    vector<ff_node *> second_set;
    for (auto *w: ws) {
        ff_comb *comb = new ff_comb(new dummy_mi(), w, true, false);
        second_set.push_back(comb);
    }
    a2a->add_secondset(second_set, true);
    return a2a;
}

// method to combine a set of nodes with the first set of an a2a
void combine_a2a_withFirstNodes(ff_a2a *a2a, const vector<ff_node*> nodes, bool cleanup=false)
{
    auto firstset = a2a->getFirstSet();
    assert(firstset.size() == nodes.size());
    vector<ff_node *> new_firstset;
    size_t i=0;
    for (auto *w: firstset) {
        ff_comb *comb = new ff_comb(nodes[i++], w, cleanup, false);
        new_firstset.push_back(comb);
    }
    a2a->change_firstset(new_firstset, 0, true);
}

// method to remove the first emitter in a pipeline
ff_node *remove_emitter_from_pipe(ff_pipeline &pipe_in)
{
    auto &stages = pipe_in.getStages();
    if (stages.size() == 1 && stages[0]->isFarm()) {
        ff_farm *farm = static_cast<ff_farm *>(stages[0]);
        farm->cleanup_emitter(false);
        farm->cleanup_workers(false);
        farm->cleanup_collector(false);
        ff_node *emitter = farm->getEmitter();
        if (!emitter->isComp() && (farm->getWorkers()).size() > 1) {
            // remove the farm from the pipeline
            pipe_in.remove_stage(0);
            // create the a2a
            ff_a2a *a2a = farm2A2A_collector(farm);
            pipe_in.insert_stage(0, a2a, false); // should be true the cleanup here!
            return emitter;
        }
        else if (!emitter->isComp() && (farm->getWorkers()).size() == 1) {
            // remove the farm from the pipeline
            pipe_in.remove_stage(0);
            const svector<ff_node*> &ws = farm->getWorkers();
            pipe_in.insert_stage(0, ws[0], false);
            return emitter;
        }
        else {
            // remove the farm from the pipeline
            pipe_in.remove_stage(0);
            ff_a2a *a2a = farm2A2A_emitter(farm);
            pipe_in.insert_stage(0, a2a, false); // should be true the cleanup here!
            return nullptr;
        }
    }
    else if (stages.size() == 2 && stages[0]->isFarm()) {
        ff_farm *farm = static_cast<ff_farm *>(stages[0]);
        farm->cleanup_emitter(false);
        farm->cleanup_workers(false);
        farm->cleanup_collector(false);
        ff_node *emitter = farm->getEmitter();
        if (farm->getCollector() != nullptr) {
            // remove the farm from the pipeline
            pipe_in.remove_stage(0);
            // create the a2a
            ff_a2a *a2a = farm2A2A_collector(farm);
            pipe_in.insert_stage(0, a2a, false); // should be true the cleanup here!
            return emitter;
        }
        else {
            // remove the farm from the pipeline
            ff_farm *farm2 = static_cast<ff_farm *>(stages[1]);
            pipe_in.remove_stage(0);
            pipe_in.remove_stage(0);
            // create the a2a
            ff_a2a *a2a = new ff_a2a();
            auto &ws = farm->getWorkers();
            vector<ff_node *> first_set;
            for (auto *w: ws) {
                ff_comb *comb = new ff_comb(w, new dummy_mo(), false, true);
                first_set.push_back(comb);
            }
            a2a->add_firstset(first_set, 0, true);
            vector<ff_node *> second_set;
            second_set.push_back(farm2);
            a2a->add_secondset(second_set, false);
            pipe_in.insert_stage(0, a2a, false); // should be true the cleanup here!
            return emitter;
        }
    }
    else
        return nullptr;
}

#endif
