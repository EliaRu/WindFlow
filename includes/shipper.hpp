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
 *  @file    shipper.hpp
 *  @author  Gabriele Mencagli
 *  @date    10/01/2019
 *  
 *  @brief Shipper class used to send output data items by the Source and FlatMap patterns
 *  
 *  @section Shipper (Description)
 *  
 *  This file implements the Shipper class used to send produced output results
 *  to the next stage of the application.
 *  
 *  The template argument of the data items that can be used with the Shipper must be default
 *  constructible, with a copy constructor and copy assignment operator, and they
 *  must provide and implement the setInfo() and getInfo() methods.
 */ 

#ifndef SHIPPER_H
#define SHIPPER_H

// includes
#include <ff/node.hpp>

using namespace ff;
using namespace std;

/** 
 *  \class Shipper
 *  
 *  \brief Shipper class used to send output data items by the Source and FlatMap patterns
 *  
 *  This class implements the Shipper object to send produced output results to
 *  the next stage of the application. It is used by the FlatMap pattern.
 */ 
template<typename result_t>
class Shipper
{
private:
    // ff_node to be used for the delivery
    ff_node *node;
    // counter of the delivered results
    unsigned long n_delivered;

public:
    /** 
     *  \brief Constructor
     *  
     *  \param _node fastflow node used for the delivery of results
     */ 
    Shipper(ff_node &_node): node(&_node), n_delivered(0) {}

    /** 
     *  \brief Return the number of results delivered
     *  
     *  \return number of results
     */  
    unsigned long delivered() const
    {
        return n_delivered;
    }

    /** 
     *  \brief Deliver a new result
     *  
     *  \param r reference to the result to be delivered
     *  \return delivery status (done -> true, failed -> false)
     */  
    bool push(const result_t &r)
    {
        result_t *out = new result_t();
        *out = r; // copy of the message!
        n_delivered++;
        return node->ff_send_out(out);
    }

    /** 
     *  \brief Deliver a new result
     *  
     *  \param r a pointer to the result to be delivered (it must be allocated in the heap)
     *  \return delivery status (done -> true, failed -> false)
     */  
    bool push(result_t *r)
    {
        n_delivered++;
        return node->ff_send_out(r);
    }
};

#endif
