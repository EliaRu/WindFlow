#ifndef BATCHED_FAT_H
#define BATCHED_FAT_H

#include <vector>
#include <cmath>
#include <cassert>

#include <basic.hpp>

using namespace std;

template<typename T, typename F>
__global__ void Init( 
    F combine, 
    size_t key,
    uint64_t gwid,
    T* levelA, 
    T* levelB, 
    size_t levelBSize )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if ( i >= levelBSize ) return;
    combine( key, gwid, levelA[i * 2], levelA[ i * 2 + 1 ], levelB[i] );
}

__device__ int Parent( int pos, int B )
{
    return ( pos >> 1 ) | B;
}

template<typename T, typename F>
__global__ void ComputeResults( 
    F combine,
    size_t key,
    uint64_t gwid,
    T* fat, 
    T* results, 
    int B,
    int W, 
    int b_id, 
    int Nb,
    int S )
{
    int win_local_id = blockIdx.x * blockDim.x + threadIdx.x;
    if( win_local_id >= Nb ) return;
    int win_global_id = win_local_id  + b_id * Nb;
    int wS = win_local_id * S;

    while( W > 0 ) {
        int range;
        wS = wS % B;
        if( wS == 0 ) {
            range = B;
        } else {
            range = ( wS & -wS );
        }
        int64_t pow = W;
        pow |= pow >> 1;
        pow |= pow >> 2;
        pow |= pow >> 4;
        pow |= pow >> 8;
        pow |= pow >> 16;
        pow |= pow >> 32;
        pow = ( pow >> 1 ) + 1;
        range = range < pow ? range : pow;
        int tr = range;
        int tn = wS;
        while( tr > 1 ) {
            tn = Parent( tn, B );
            tr >>= 1;
        }
        combine( 
            key, 
            gwid, 
            results[win_local_id], 
            fat[tn], 
            results[win_local_id] 
        );
        wS += range;
        W -= range;
    }
}

template<typename T, typename F>
class BatchedFAT {
private:
/*
 * Rappresentazione:
 * 0 radice
 * [0,batchSize-2] nodi interni
 * [batchSize-1, 2*batchSize-2] foglie
 * nodo << 1 + 1 figlio sinistro
 * nodo << 1 * 2 figlio destro
 */
    T* d_tree;
    T* d_results;
    vector<T> results;
    vector<T> initResults;
    size_t treeSize;
    size_t treeMemSize;
    size_t batchSize;
    size_t batchMemSize;
    size_t noLeaves;
    size_t leavesMemSize;
    size_t windowSize;
    size_t slide;
    size_t Nb;
    F combine;

public:
    BatchedFAT( ) { }

    BatchedFAT( 
        F _combine,
        size_t _batchSize,
        size_t _numWindows,
        size_t _windowSize, 
        size_t _slide 
    ) : d_tree( nullptr ),
        d_results( nullptr ),
        batchSize( _batchSize ),
        Nb( _numWindows ),
        windowSize( _windowSize ),
        slide( _slide ),
        combine( _combine )
    { 
        size_t noBits = ( size_t ) ceil( log2( batchSize ) );
        size_t n = 1 << noBits;

        treeSize = n * 2 - 1;
        treeMemSize = treeSize * sizeof( T );
        noLeaves = n;
        leavesMemSize = noLeaves * sizeof( T );
        batchMemSize = batchSize * sizeof( T );

        gpuErrChk( cudaMalloc( ( void ** ) &d_tree, treeMemSize ) );

        results.resize( Nb );
        initResults.resize( Nb );
        gpuErrChk( cudaMalloc( ( void ** )&d_results, Nb * sizeof( T ) ) );
    }

    bool build( 
        const vector<T>& tuples, 
        size_t key, 
        uint64_t gwid, 
        int b_id )
    {
        if( tuples.size( ) != batchSize ) return false;

        vector<T> tree( tuples.begin( ), tuples.end( ) );
        tree.insert( tree.end( ), treeSize - batchSize, T( ) );
        assert( tree.size( ) == treeSize );

        gpuErrChk( cudaMemcpy( 
            ( void * ) d_tree,
            ( void * ) tree.data( ), 
            treeMemSize, 
            cudaMemcpyHostToDevice ) 
        );

        T* d_levelA = d_tree;
        int pow = 1;
        T* d_levelB = d_levelA + noLeaves / pow;
        int i = noLeaves / 2;
        cudaError_t err;
        while( d_levelB < d_tree + treeSize && i > 0 ) {
            int noBlocks = ( int ) ceil( i / 1024.0 );    
            Init<T, F><<<noBlocks, 1024>>>( 
                combine, key, gwid, d_levelA, d_levelB, i 
            );
            if( err = cudaGetLastError( ) ) {
                printf( "Esecuzione Kernel Init.\nErrore: %d\n", err );
            }
            d_levelA = d_levelB;
            pow = pow << 1;
            d_levelB = d_levelA + noLeaves / pow;
            i /= 2;
        }
        /*Controllo valori albero
        vector<T> results( treeSize );
        cudaMemcpy( results.data( ), d_tree, treeMemSize, cudaMemcpyDeviceToHost );
        for( auto j : results ) {
            cout << j << " ";
        }
        cout << endl;
        cout << endl;
        cout << endl;
        */  
        //TODO:it needs to stop when it reaches level i s.t. 2^i > W
        //it may also reduce memory occupation

        

        gpuErrChk( 
            cudaMemcpy( 
                ( void * ) d_results,
                ( void * ) initResults.data( ),
                Nb * sizeof( T ),
                cudaMemcpyHostToDevice )
        );

        if( Nb > 1024 ) {
            int noBlocks = ( int ) ceil( Nb / 1024.0 );
            ComputeResults<T, F><<<noBlocks, 1024>>>( 
                combine,
                key,
                gwid,
                d_tree, 
                d_results, 
                noLeaves, 
                windowSize, 
                b_id, 
                Nb, 
                slide 
            );
            cudaError_t err;
            if( err = cudaGetLastError( ) ) {
                printf( 
                    "Esecuzione Kernel ComputeResults.\nErrore: %d\n", err 
                );
            }
        } else {
            ComputeResults<T, F><<<1, Nb>>>( 
                combine,
                key,
                gwid,
                d_tree, 
                d_results, 
                noLeaves, 
                windowSize, 
                b_id, 
                Nb, 
                slide 
            );
            cudaError_t err;
            if( err = cudaGetLastError( ) ) {
                printf( 
                    "Esecuzione Kernel ComputeResults.\nErrore: %d\n", err 
                );
            }
        }
        return true;
    }

    const vector<T>& getResults( )
    {
        gpuErrChk( 
            cudaMemcpy( 
                ( void * ) results.data( ),
                ( void * ) d_results,
                Nb * sizeof( T ), 
                cudaMemcpyDeviceToHost ) 
        );
        return results;
    }
    
    BatchedFAT( BatchedFAT const & ) = delete;
    BatchedFAT& operator=( BatchedFAT const & ) = delete;

    BatchedFAT( BatchedFAT&& _fat )
    : results( move( _fat.results ) ),
      initResults( move( _fat.initResults ) ),
      treeSize( move( _fat.treeSize ) ),
      treeMemSize( move( _fat.treeMemSize ) ),
      batchSize( move( _fat.batchSize ) ),
      batchMemSize( move( _fat.batchMemSize ) ),
      noLeaves( move( _fat.noLeaves ) ),
      leavesMemSize( move( _fat.leavesMemSize ) ),
      windowSize( move( _fat.windowSize ) ),
      slide( move( _fat.slide ) ),
      Nb( move( _fat.Nb ) ),
      combine( move( _fat.combine ) )
    { 
        gpuErrChk( cudaMalloc( ( void ** ) &d_tree, treeMemSize ) );
        gpuErrChk( cudaMalloc( ( void ** )&d_results, Nb * sizeof( T ) ) );
    }

    ~BatchedFAT( ) {
        gpuErrChk( cudaFree( d_tree ) );
        gpuErrChk( cudaFree( d_results ) );
    }
};

#endif
