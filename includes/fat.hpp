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
 *  @file    fat.hpp
 *  @author  Elia Ruggeri
 *  @date    06/05/2019
 *  
 *  @brief Flat fixed-size aggregator tree
 *  
 *  @section Flat_FAT (Description)
 *  
 *  Flat implementation of a tree used to speed up 
 *  incremental queries.
 */ 

#ifndef FAT_H
#define FAT_H

// includes
#include <deque>
#include <utility>
#include <algorithm>
#include <functional>
#include <assert.h>
#include <cmath>
#include <list>
#include <vector>

using namespace std;

//class FlatFAT
template<typename tuple_t, typename result_t>
class FlatFAT {
private:
    //function for inserting a tuple in the tree
    using f_winlift_t = 
        function<int(size_t, uint64_t, const tuple_t&, result_t&)>;
    //function of the incremental window processing
    using f_wincombine_t =
        function<int(size_t, uint64_t, const result_t&, const result_t&, result_t&)>;

    f_winlift_t winLift;
    f_wincombine_t winCombine;

    vector<result_t> tree;
    size_t n;
    size_t front; //most recent inserted element
    size_t back; //next element to remove
    bool is_empty;

    size_t root = 1; //position of the root in the array
    //methods for traversing the tree
    size_t left_child( size_t pos ) const { return pos << 1; }
    size_t right_child( size_t pos ) const { return ( pos << 1 ) + 1; }
    size_t leaf( size_t pos ) const { return n + pos - 1;}
    size_t parent( size_t pos ) const { 
        return static_cast<size_t>( floor( pos / 2.0 ) );
    }

    result_t prefix( size_t key, uint64_t id, size_t pos ) const {
        size_t i = pos;
        result_t acc = tree[pos];
        while( i != root ) {
            size_t p = parent( i );
            //if i is the right child of p then both its left child
            //and right child are in the prefix.
            //Otherwise only the left child is so we pass acc unmodified
            if( i == right_child( p ) ) {
                result_t tmp = acc;
                winCombine( key, id, tree[left_child( p )], tmp, acc );
            }
            i = p;
        }
        return acc;
    }

    result_t suffix( size_t key, uint64_t id, size_t pos ) const {
        size_t i = pos;
        result_t acc = tree[pos];
        while( i != root ) {
            //if i is the left child of p then both its left child
            //and right child are in the suffix.
            //Otherwise only the right child is so we pass acc unmodified
            size_t p = parent( i );
            if( i == left_child( p ) ) {
                result_t tmp = acc;
                winCombine( key, id, tmp, tree[right_child( p )], acc );
            }
            i = p;
        }
        return acc;
    }

    //Update of a single element in the tree. It has already been inserted
    //in the correct leaf
    int update( size_t key, uint64_t id, size_t pos ) {
        size_t nextNode = parent( pos );
        //The method traverses the tree updating each node it
        //encounters until it updates the root
        while( nextNode != 0 ) {
            size_t lc = left_child( nextNode );
            size_t rc= right_child( nextNode );
            int res = winCombine(
                key, 
                id, 
                tree[lc],
                tree[rc], 
                tree[nextNode] 
            );
            if( res < 0 ) {
                return -1;
            }
            nextNode = parent( nextNode );
        }
        return 0;
    }

public:
    FlatFAT( ) : n( 0 ) { }

    FlatFAT( f_winlift_t _winLift, f_wincombine_t _winCombine, size_t _n ):
        root( 1 ), is_empty( true ), 
        winLift( _winLift ), winCombine( _winCombine )
    { 
        //The tree must be a complete binary tree so n must be rounded
        //to the next power of two
        int noBits = ( int ) ceil( log2( _n ) );
        n = 1 << noBits;
        front = n - 1;
        back = n - 1;
        tree.resize( n * 2 );
    }

    int insert( size_t key, uint64_t id, tuple_t const& input ) {
        //Checks if the tree is empty
        if( front == back && front == n - 1 ) {
            front++, back++;
            is_empty = false;
        //Check if front is the last leaf so it must wrap around
        } else if( back == 2 * n - 1 ) {
            //But it needs to check if the first leaf is empty
            if( front != n ) {
                back = n;
            } else {
                return -1;
            }
        //Check if front < back and the tree is full
        } else if( front != back + 1 ) {
            back++;
        } else {
            return -1;
        }
        //Insert the element in the next empty position
        if( winLift( key, id, input, tree[back] ) < 0 ) {
            return -1;
        }
        return update( key, id, back );
    }

    int insert( size_t key, uint64_t id, vector<tuple_t> const& inputs ) 
    {
        list<size_t> nodesToUpdate;
        for( size_t i = 0; i < inputs.size( ); i++ ) {
            //Checks if the tree is empty
            if( front == back && front == n - 1 ) {
                front++, back++;
                is_empty = false;
            //Check if front is the last leaf so it must wrap around
            } else if( back == 2 * n - 1 ) {
                //But it needs to check if the first leaf is empty
                if( front != n ) {
                    back = n;
                } else {
                    return -1;
                }
            //Check if front < back and the tree is full
            } else if( front != back + 1 ) {
                back++;
            } else {
                return -1;
            }
            //Insert the element in the next empty position
            if( winLift( key, id, inputs[i], tree[back] ) < 0 ) {
                return -1;
            }
            size_t p = parent( back );
            if( back != root &&
                ( nodesToUpdate.empty( ) || 
                nodesToUpdate.back( ) != p ) ) 
            {
                nodesToUpdate.push_back( p );
            }
        }
        while( !nodesToUpdate.empty( ) ) {
            size_t nextNode = nodesToUpdate.front( );
            nodesToUpdate.pop_front( );
            size_t lc = left_child( nextNode );
            size_t rc= right_child( nextNode );
            int res = winCombine(
                key, 
                id, 
                tree[lc],
                tree[rc], 
                tree[nextNode] 
            );
            if( res < 0 ) {
                return -1;
            }
            size_t p = parent( nextNode );
            if( nextNode != root &&
                ( nodesToUpdate.empty( ) || 
                nodesToUpdate.back( ) != p ) ) 
            {
                nodesToUpdate.push_back( p );
            }
        }
        return 0;
    }

    bool isEmpty( ) {
        return is_empty;
    }

    int removeOldestTuple( size_t key, uint64_t id ) {
        tuple_t t = tuple_t( );
        //It removes the element by inserting in its place
        //a default constructed element
        if( winLift( key, id, t, tree[front] ) < 0 ) {
            return -1;
        }
        if( update( key, id, front ) < 0 ) {
            return -1;
        }
        //Then the front pointer is updated.
        //First checks if this was the last element of the tree
        if( front == back ) {
            front = back = n - 1;
            is_empty = true;
        //Then if it must wrap around
        } else if( front == 2 * n -1 ) {
            front = n;
        } else {
            front++;
        }
        return 0;
    }

    int remove( size_t key, uint64_t id, size_t count ) {
        tuple_t t = tuple_t( );
        list<size_t> nodesToUpdate;
        for( size_t i = 0; i < count; i++ ) {
            if( winLift( key, id, t, tree[front] ) < 0 ) {
                return -1;
            }
            size_t p = parent( front );
            if( front != root &&
                ( nodesToUpdate.empty( ) || 
                nodesToUpdate.back( ) != p ) ) 
            {
                nodesToUpdate.push_back( p );
            }
            if( front == back ) {
                front = back = n - 1;
                is_empty = true;
                break;
            } else if( front == 2 * n -1 ) {
                front = n;
            } else {
                front++;
            }
        }
        while( !nodesToUpdate.empty( ) ) {
            size_t nextNode = nodesToUpdate.front( );
            nodesToUpdate.pop_front( );
            size_t lc = left_child( nextNode );
            size_t rc= right_child( nextNode );
            int res = winCombine(
                key, 
                id, 
                tree[lc],
                tree[rc], 
                tree[nextNode] 
            );
            if( res < 0 ) {
                return -1;
            }
            size_t p = parent( nextNode );
            if( nextNode != root &&
                ( nodesToUpdate.empty( ) || 
                nodesToUpdate.back( ) != p ) ) 
            {
                nodesToUpdate.push_back( p );
            }
        }
        return 0;
    }

    result_t *getResult( size_t key, uint64_t id ) const {
        result_t* res = new result_t( );
        if( front <= back ) {
            //The elements are in the correct order so the result
            //in the root is valid
            *res = tree[root];
        } else {
            //In case winCombine is not commutative we need to 
            //compute the value of the combination of the elements
            //at positions [n, back], the prefix, and the ones at 
            //positions [front, 2*n-1], the suffix, and combine
            //them accordingly
            result_t prefixRes = prefix( key, id, back );
            result_t suffixRes = suffix( key, id, front );
            winCombine( key, id, suffixRes, prefixRes, *res );
        }
        return res;
    }
       
};

#endif
