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
 *  @file    stream_archive.hpp
 *  @author  Gabriele Mencagli
 *  @date    28/06/2017
 *  
 *  @brief Stream archive
 *  
 *  @section Stream_Archive (Description)
 *  
 *  Stream archive of tuples received from the input stream and still useful
 *  for the query processing.
 */ 

#ifndef ARCHIVE_H
#define ARCHIVE_H

// includes
#include <deque>
#include <utility>
#include <algorithm>
#include <functional>
#include <assert.h>

using namespace std;

// class StreamArchive
template<typename tuple_t, typename container_t=deque<tuple_t>>
class StreamArchive
{
private:
    // function to compare two tuples
    using f_compare_t = function<bool(const tuple_t &t1, const tuple_t &t2)>;
    // const iterator type
    using const_iterator_t = typename container_t::const_iterator;
    f_compare_t lessThan; // function to compare two tuples
    container_t archive; // container implementing the archive (elements are stored in increasing order)

public:
    StreamArchive( ) { } 
    // constructor
    StreamArchive(f_compare_t _lessThan): lessThan(_lessThan) {}

    // method to add a tuple to the archive
    void insert(const tuple_t &_t)
    {
        auto it = lower_bound(archive.begin(), archive.end(), _t, lessThan);
        // _t must be added at the end
        if (it == archive.end())
            archive.push_back(_t);
        // otherwise it must be added to the correct position
        else
            archive.insert(it, _t);
    }

    // method to remove all the tuples prior to _t 
    size_t purge(const tuple_t &_t)
    {
        auto it = lower_bound(archive.begin(), archive.end(), _t, lessThan);
        size_t n = distance(archive.begin(), it);
        archive.erase(archive.begin(), it);
        return n;
    }

    // method to get the size of the archive
    size_t size() const
    {
        return archive.size();
    }

    // method to return the iterator to the first tuple in the archive 
    const_iterator_t begin() const
    {
        return archive.begin();
    }

    // method to return the iterator to the end of the archive
    const_iterator_t end() const
    {
        return archive.end();
    }

    /*  
     *  Method to get a pair of constant iterators that represent the window range [first, last) given
     *  two tuples _t1 and _t2. Tuple _t1 must compare less than _t2. The method returns the constant
     *  iterator (first) to the smallest tuple in the archive that compares greater or equal than _t1,
     *  and the constant iterator (last) to the smallest tuple in the archive that compares greater or
     *  equal than _t2.
     */ 
    pair<const_iterator_t, const_iterator_t> getWinRange(const tuple_t &_t1, const tuple_t &_t2) const
    {
        assert(lessThan(_t1, _t2));
        pair<const_iterator_t, const_iterator_t> its;
        its.first = lower_bound(archive.begin(), archive.end(), _t1, lessThan);
        its.second = lower_bound(archive.begin(), archive.end(), _t2, lessThan);
        return its;
    }

    /*  
     *  Method to get a pair of constant iterators that represent the window range [first, end) given
     *  an input tuple _t. The method returns the constant iterator (first) to the smallest tuple in
     *  the archive that compares greater or equal than _t, and the constant iterator (end) to the end
     *  of the archive.
     */ 
    pair<const_iterator_t, const_iterator_t> getWinRange(const tuple_t &_t) const
    {
        pair<const_iterator_t, const_iterator_t> its;
        its.first = lower_bound(archive.begin(), archive.end(), _t, lessThan);
        its.second = archive.end();
        return its;
    }

    /*  
     *  Method which, given a pair of two tuples _t1 and _t2 contained in the archive, returns
     *  the distance from _t1 to _t2.
     */ 
    size_t getDistance(const tuple_t &_t1, const tuple_t &_t2) const
    {
        pair<const_iterator_t, const_iterator_t> its;
        its.first = lower_bound(archive.begin(), archive.end(), _t1, lessThan);
        its.second = lower_bound(archive.begin(), archive.end(), _t2, lessThan);
        return distance(its.first, its.second);
    }

    /*  
     *  Method which, given a tuple _t contained in the archive, returns
     *  the distance from _t to the end of the archive.
     */ 
    size_t getDistance(const tuple_t &_t1) const
    {
        pair<const_iterator_t, const_iterator_t> its;
        its.first = lower_bound(archive.begin(), archive.end(), _t1, lessThan);
        return distance(its.first, archive.end());
    }

    /*  
     *  Method used to get an iterator to a given tuple in the archive. The tuple must be
     *  in the archive.
     */ 
    const_iterator_t getIterator(const tuple_t &_t1) const
    {
        return lower_bound(archive.begin(), archive.end(), _t1, lessThan);
    }
};

#endif
