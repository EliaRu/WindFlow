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

#ifndef FAT_H
#define FAT_H

// includes
#include <deque>
#include <utility>
#include <algorithm>
#include <functional>
#include <assert.h>
#include <cmath>

using namespace std;

/*Usato:
 */
template<typename tuple_t, typename result_t>
class FlatFAT {
private:
    using f_winlift_t = 
        function<int(size_t, uint64_t, const tuple_t&, result_t&)>;
    using f_wincombine_t =
        function<int(size_t, uint64_t, const result_t&, const result_t&, result_t&)>;

    f_winlift_t winLift;
    f_wincombine_t winCombine;

    //vector<tuple_t> archive;
    vector<result_t> tree;
    size_t n;
    size_t front;
    size_t back;
    bool is_empty;

    size_t root = 1;
    size_t left_child( size_t pos ) { return pos << 1; }
    size_t right_child( size_t pos ) { return ( pos << 1 ) + 1; }
    size_t leaf( size_t pos ) { return n + pos - 1;}
    size_t parent( size_t pos ) { 
        return floor( static_cast<double>( pos ) / 2.0 );
    }

    result_t prefix( size_t key, uint64_t id, size_t pos ) {
        size_t i = pos;
        result_t acc = tree[pos];
        while( i != root ) {
            size_t p = parent( i );
            if( i == right_child( p ) ) {
                result_t tmp = acc;
                winCombine( key, id, tree[left_child( p )], tmp, acc );
            }
            i = p;
        }
        return acc;
    }

    result_t suffix( size_t key, uint64_t id, size_t pos ) {
        size_t i = pos;
        result_t acc = tree[pos];
        while( i != root ) {
            size_t p = parent( i );
            if( i == left_child( p ) ) {
                result_t tmp = acc;
                winCombine( key, id, tmp, tree[right_child( p )], acc );
            }
            i = p;
        }
        return acc;
    }

public:
    FlatFAT( ) : n( 0 ) { }

    FlatFAT( f_winlift_t _winLift, f_wincombine_t _winCombine, size_t _n ):
        root( 1 ), is_empty( true ), 
        winLift( _winLift ), winCombine( _winCombine )
    { 
        int noBits = ( int ) ceil( log2( _n ) );
        n = 1 << noBits;
        front = n - 1;
        back = n - 1;
        tree.resize( n * 2 );
    }

    int insert( size_t key, uint64_t id, tuple_t const& input ) {
        if( front == back && front == n - 1 ) {
            front++, back++;
            is_empty = false;
        } else if( back == 2 * n - 1 ) {
            if( front != n ) {
                back = n;
            } else {
                return -1;
            }
        } else if( front != back + 1 ) {
            back++;
        } else {
            return -1;
        }
        if( winLift( key, id, input, tree[back] ) < 0 ) {
            return -1;
        }
        size_t nextNode = parent( back );
        size_t f = front;
        size_t b = back;
        while( nextNode != 0 ) {
            size_t lc = left_child( nextNode );
            size_t rc= right_child( nextNode );
            if( f <= b ) {
                if( lc < f || lc > b) {
                    tree[nextNode] = tree[rc];
                } else if( rc < f|| rc > b ) {
                    tree[nextNode] = tree[lc];
                } else {
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
                }
            } else {
                if( lc > b && lc < f ) {
                    tree[nextNode] = tree[rc];
            } else if( rc > b && rc < f ) {
                    tree[nextNode] = tree[lc];
                } else {
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
                }
            }
            f = parent( f );
            b = parent( b );
            nextNode = parent( nextNode );
        }
        return 0;
    }

    bool isEmpty( ) {
        return is_empty;
    }

    int removeOldestTuple( size_t key, uint64_t id ) {
       if( front == back ) {
            front = back = n - 1;
            is_empty = true;
        } else if( front == 2 * n -1 ) {
            front = n;
        } else {
            front++;
        }
        size_t nextNode = parent( front );
        size_t f = front;
        size_t b = back;
        while( nextNode != 0 ) {
            size_t lc = left_child( nextNode );
            size_t rc= right_child( nextNode );
            if( f <= b ) {
                if( lc < f || lc > b) {
                    tree[nextNode] = tree[rc];
                } else if( rc < f|| rc > b ) {
                    tree[nextNode] = tree[lc];
                } else {
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
                }
            } else {
                if( lc > b && lc < f ) {
                    tree[nextNode] = tree[rc];
                } else if( rc > b && rc < f ) {
                    tree[nextNode] = tree[lc];
                } else {
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
                }
            }
            f = parent( f );
            b = parent( b );
            nextNode = parent( nextNode );
        }
        return 0;
    }

    result_t *getResult( size_t key, uint64_t id ) {
        if( is_empty ) {
            return nullptr;
        }
        result_t* res = new result_t( );
        if( front <= back ) {
            *res = tree[root];
            return res;
        } else {
            result_t prefixRes = prefix( key, id, back );
            result_t suffixRes = suffix( key, id, front );
            winCombine( key, id, suffixRes, prefixRes, *res );
        }
        return res;
    }
       
    /*... update( ... ) {
        if( archive.size( ) < win_size - 1 ) {
            archive.push_back( t );
        } else if( archive.size( ) == win_size - 1 ) {
            build_tree( ... );
        } else {
            update_tree( ... );
        }
    }
    */
};

#endif
