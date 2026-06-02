#include "density.h"

namespace kernel {

__global__
/**
 * @brief kernel for injecting a uniform plasma density (mk1)
 * 
 * Particles in the same cell are injected contiguously
 * 
 * @param range     Cell range (global) to inject particles in 
 * @param ppc       Number of particles per cell 
 * @param nx        Number of cells in tile 
 * @param tiles     Particle tile information 
 * @param data      Particle data 
 */
void uniform_mk1( 
    bnd<unsigned int> range,
    uint2 const ppc,
    ParticleData const part )
{
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    
    // Tile ID
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    const int tile_np = part.np[ tile_id ];

    block_sync();

    // Find injection range in tile coordinates
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tile_id ];
        int2   * __restrict__ const ix = &part.ix[ offset ];
        float2 * __restrict__ const x  = &part.x[ offset ];
        float3 * __restrict__ const u  = &part.u[ offset ];

        const int np_cell = ppc.x * ppc.y;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        // Each thread takes 1 cell
        for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads() ) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );

            int part_idx = tile_np + idx * np_cell;

            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    ix[ part_idx ] = cell;
                    x[ part_idx ] = pos;
                    u[ part_idx ] = make_float3( 0, 0, 0 );
                    part_idx++;
                }
            }
        }

        // Update global number of particles in tile
        if ( block_thread_rank() == 0 ) {
            part.np[ tile_id ] = tile_np + vol * np_cell ;
        }
    }
}

__global__
/**
 * @brief kernel for injecting a uniform plasma density (mk2)
 * 
 * Places contiguous particles in different cells. This minimizes memory collisions
 * when depositing current, especially for very low temperatures.
 * 
 * @param range     Cell range (global) to inject particles in 
 * @param ppc       Number of particles per cell 
 * @param nx        Number of cells in tile 
 * @param data      Particle data 
 */
void uniform( 
    bnd<unsigned int> range,
    uint2 const ppc,
    ParticleData const part )
{
    
    // Tile ID
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    const int tile_np = part.np[ tile_id ];

    block_sync();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    
    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tile_id ];

        int2   * __restrict__ const ix = &part.ix[ offset ];
        float2 * __restrict__ const x  = &part.x[ offset ];
        float3 * __restrict__ const u  = &part.u[ offset ];

        const int np_cell = ppc.x * ppc.y;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );

                int ppc_idx = i1 * ppc.x + i0;

                // Each thread takes 1 cell
                for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads() ) {
                    int2 const cell = make_int2( 
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    int part_idx = tile_np + vol * ppc_idx + idx;

                    ix[ part_idx ] = cell;
                    x[ part_idx ] = pos;
                    u[ part_idx ] = make_float3( 0, 0, 0 );
                }
                ppc_idx ++;
            }
        }

        // Update global number of particles in tile
        if ( block_thread_rank() == 0 ) {
            part.np[ tile_id ] = tile_np + vol * np_cell ;
        }
    }
}

}

void Density::Uniform::inject( Particles & particles, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{
    dim3 grid( particles.ntiles.x, particles.ntiles.y );
    dim3 block( 1024 );

#if 0
    std::cout << "(*info*) Injecting uniform density using algorithm mk I\n";
    kernel::uniform_mk1 <<< grid, block >>> (
        range, ppc, particles
    );
#else
    kernel::uniform <<< grid, block >>> (
        range, ppc, particles
    );
#endif

}

namespace kernel {

__global__
void uniform_np( 
    bnd<unsigned int> range,
    uint2 const ppc,
    ParticleData const part,
    int * __restrict__ np
) {
    // Tile ID - 1 thread per tile
    const int tile_id = blockIdx.x * blockDim.x + threadIdx.x;

    if ( tile_id < part.ntiles.x * part.ntiles.y ) {

        const uint2 tile_idx = make_uint2(
            tile_id % part.ntiles.x, 
            tile_id % part.ntiles.y
        );

        const int2 nx = make_int2( part.nx.x, part.nx.y );

        // Find injection range in tile coordinates
        int ri0 = range.x.lower - tile_idx.x * nx.x;
        int ri1 = range.x.upper - tile_idx.x * nx.x;

        int rj0 = range.y.lower - tile_idx.y * nx.y;
        int rj1 = range.y.upper - tile_idx.y * nx.y;

        int local_np;

        // If range overlaps with tile
        if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
            ( rj0 < nx.y ) && ( rj1 >= 0 )) {
            
            // Limit to range inside this tile
            if (ri0 < 0) ri0 = 0;
            if (rj0 < 0) rj0 = 0;
            if (ri1 >= nx.x ) ri1 = nx.x-1;
            if (rj1 >= nx.y ) rj1 = nx.y-1;

            int const row = (ri1-ri0+1);
            int const vol = (rj1-rj0+1) * row;

            local_np = vol * ppc.x * ppc.y;
        } else {
            local_np = 0;
        }

        np[ tile_id ] = local_np;
    }
}

}

void Density::Uniform::np_inject( Particles & particles, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    const int ntiles = particles.ntiles.x * particles.ntiles.y;

    int block = ( ntiles > 1024 ) ? 1024 : ntiles;
    int grid  = ( ntiles - 1 ) / block + 1;

    kernel::uniform_np<<< grid, block >>> ( range, ppc, particles, np );
}

namespace kernel {

/**
 * @brief kernel for injecting step profile
 * 
 * @note This kernel must be launched using a 2D grid with 1 block per tile
 * 
 * @param range     Cell range (global) to inject particles in
 * @param step      Step position normalized to cell size
 * @param ppc       Number of particles per cell
 * @param nx        Tile size
 * @param d_tiles    Tile information
 * @param d_ix      Particle buffer (cells)
 * @param d_x       Particle buffer (positions)
 * @param d_u       Particle buffer (momenta)
 */

template < coord::cart dir >
__global__
void step( 
    bnd<unsigned int> range,
    const float step, const uint2 ppc,
    ParticleData const part )
{
    // Tile ID
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    __shared__ int np_tile;
    np_tile = part.np[ tile_id ];

    block_sync();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if ( ri0 < 0 ) ri0 = 0;
        if ( rj0 < 0 ) rj0 = 0;
        if ( ri1 >= nx.x ) ri1 = nx.x-1;
        if ( rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tile_id ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads()) {
                    int2 const cell = make_int2(
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    float t;
                    if constexpr ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if constexpr ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);

                    int inj = t > step;
                    int off = block::exscan_add( inj );

                    if ( inj ) {
                        const int k = np_tile + off;
                        ix[ k ] = cell;
                        x[ k ]  = pos;
                        u[ k ]  = make_float3( 0, 0, 0 );
                    }

                    inj = warp::reduce_add( inj );
                    if ( warp::thread_rank() == 0 ) {
                        block::atomic_fetch_add( &np_tile, inj );
                    }
                    block_sync();
                }
            }
        }

        if ( block_thread_rank() == 0 ) {
            part.np[ tile_id ] = np_tile;
        }
    }
}

}

/**
 * @brief Inject a step density profile
 * 
 * @param particles     Particle data
 * @param ppc           Number of particles per cell
 * @param dx            Cell size
 * @param ref           Reference for step position
 * @param range         Global cell range for injection
 */
void Density::Step::inject( Particles & particles,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{

    dim3 grid( particles.ntiles.x, particles.ntiles.y );
    dim3 block( 1024 );

    float step_pos;

    switch( dir ) {
    case( coord::x ):
        step_pos = (pos - ref.x) / dx.x;
        kernel::step <coord::x> <<< grid, block >>> ( range, step_pos, ppc, particles );
        break;

    case( coord::y ):
        step_pos = (pos - ref.y) / dx.y;
        kernel::step <coord::y> <<< grid, block >>> ( range, step_pos, ppc, particles );
        break;
    }
}

namespace kernel {

template < coord::cart dir >
__global__
void step_np( 
    bnd<unsigned int> range,
    const float step, const uint2 ppc,
    ParticleData const part, int * np )
{
    // Tile ID
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int  tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    __shared__ int np_tile; np_tile = 0;
    block_sync();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads() ) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float t;
                    if constexpr ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if constexpr ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = t > step;
                    inj_np += inj;
                }
            }
        }
    }

    inj_np = warp::reduce_add( inj_np );
    if ( warp::thread_rank() == 0 ) {
        block::atomic_fetch_add( &np_tile, inj_np );
    } 

    block_sync();

    if ( block_thread_rank() == 0 ) {
        np[ tile_id ] = np_tile;
    }
}

}

void Density::Step::np_inject( Particles & particles, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    dim3 grid( particles.ntiles.x, particles.ntiles.y );
    dim3 block( 1024 );

    float step_pos;
    switch( dir ) {
    case( coord::x ):
        step_pos = (pos - ref.x) / dx.x;
        kernel::step_np <coord::x> <<< grid, block >>> ( range, step_pos, ppc, particles, np );
        break;
    case( coord::y ):
        step_pos = (pos - ref.y) / dx.y;
        kernel::step_np <coord::y> <<< grid, block >>> ( range, step_pos, ppc, particles, np );
        break;
    }
}

namespace kernel {

template < coord::cart dir >
__global__
void slab( 
    bnd<unsigned int> range,
    const float start, const float finish, const uint2 ppc,
    ParticleData const part )
{
    // Tile ID
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    __shared__ int np_tile; np_tile = part.np[ tile_id ];

    block_sync();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tile_id ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads()) {
                    int2 const cell = make_int2(
                        idx % row + ri0,
                        idx / row + rj0
                    );

                    float t;
                    if constexpr ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if constexpr ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = (t >= start) && (t<finish );

                    int off = block::exscan_add( inj );
                    
                    if (inj) {
                        const int k = np_tile + off;
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = make_float3( 0, 0, 0 );
                    }

                    inj = warp::reduce_add( inj );
                    if ( warp::thread_rank() == 0 ) {
                        block::atomic_fetch_add( &np_tile, inj );
                    }
                    block_sync();
                }
            }
        }

        if ( block_thread_rank() == 0 ) {
            part.np[ tile_id ] = np_tile;
        }
    }
}

}

void Density::Slab::inject( Particles & particles,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{
    dim3 grid( particles.ntiles.x, particles.ntiles.y );
    dim3 block( 1024 );

    float slab_begin, slab_end;

    switch( dir ) {
    case( coord::x ):
        slab_begin = (begin - ref.x)/ dx.x;
        slab_end = (end - ref.x)/ dx.x;
        kernel::slab <coord::x> <<< grid, block >>> ( range, slab_begin, slab_end, ppc, particles );
        break;

    case( coord::y ):
        slab_begin = (begin - ref.y)/ dx.y;
        slab_end = (end - ref.y)/ dx.y;
        kernel::slab <coord::y> <<< grid, block >>> ( range, slab_begin, slab_end, ppc, particles );
        break;
    }
}

namespace kernel {

template < coord::cart dir >
__global__
void slab_np( 
    bnd<unsigned int> range,
    const float start, const float finish, uint2 ppc,
    ParticleData const part, int * np )
{
    // Tile ID
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    __shared__ int np_tile; np_tile = 0;
    block_sync();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );
    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads() ) {
            int2 const cell = make_int2(
                idx % row + ri0,
                idx / row + rj0
            );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float t;
                    if constexpr ( dir == coord::x ) t = (shiftx + cell.x) + (pos.x + 0.5);
                    if constexpr ( dir == coord::y ) t = (shifty + cell.y) + (pos.y + 0.5);
                    
                    int inj = (t >= start) && (t<finish );
                    inj_np += inj;
                }
            }
        }
    }

    inj_np = warp::reduce_add( inj_np );
    if ( warp::thread_rank() == 0 ) {
        block::atomic_fetch_add( &np_tile, inj_np );
    } 

    block_sync();

    if ( block_thread_rank() == 0 ) {
        np[ tile_id ] = np_tile;
    }
}

}

void Density::Slab::np_inject( Particles & particles, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    dim3 grid( particles.ntiles.x, particles.ntiles.y );
    dim3 block( 1024 );

    float slab_begin, slab_end;

    switch( dir ) {
    case( coord::x ):
        slab_begin = (begin - ref.x)/ dx.x;
        slab_end = (end - ref.x)/ dx.x;
        kernel::slab_np <coord::x> <<< grid, block >>> ( 
            range, slab_begin, slab_end, ppc, particles, np
        );
        break;

    case( coord::y ):
        slab_begin = (begin - ref.y)/ dx.y;
        slab_end = (end - ref.y)/ dx.y;
        kernel::slab_np <coord::y> <<< grid, block >>> (
            range, slab_begin, slab_end, ppc, particles, np
        );
        break;
    }
}

namespace kernel {

__global__
void sphere( 
    bnd<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc,
    ParticleData const part )
{
    // Tile ID
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Store number of particles before injection
    __shared__ int np_tile; np_tile = part.np[ tile_id ];

    block_sync();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        const int offset =  part.offset[ tile_id ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;
        const float r2 = radius*radius;

        for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
            for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                float2 const pos = make_float2(
                    dpcx * ( i0 + 0.5 ) - 0.5,
                    dpcy * ( i1 + 0.5 ) - 0.5
                );
                for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads() ) {
                    int2 const cell = make_int2( 
                        idx % row + ri0,
                        idx / row + rj0
                    );
                    float gx = ((shiftx + cell.x) + (pos.x+0.5)) * dx.x;
                    float gy = ((shifty + cell.y) + (pos.y+0.5)) * dx.y;
                    
                    int inj = ((gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y)) < r2;
                    int off = block::exscan_add( inj );

                    if ( inj ) {
                        const int k = np_tile + off;
                        ix[ k ] = cell;
                        x[ k ] = pos;
                        u[ k ] = make_float3( 0, 0, 0 );
                    }
                    
                    inj = warp::reduce_add( inj );
                    if ( warp::thread_rank() == 0 ) {
                        block::atomic_fetch_add( &np_tile, inj );
                    }
                    block_sync();
                }
            }
        }

        if ( block_thread_rank() == 0 ) {
            part.np[ tile_id ] = np_tile;
        }
    }
}

}

void Density::Sphere::inject( Particles & particles,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{

    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    dim3 grid( particles.ntiles.x, particles.ntiles.y );
    dim3 block( 1024 );
    kernel::sphere <<< grid, block >>> ( 
        range, sphere_center, radius, dx, ppc, particles
    );
}


namespace kernel {

__global__
void sphere_np(
    bnd<unsigned int> range,
    float2 center, float radius, float2 dx, uint2 ppc,
    ParticleData const part, int * np )
{
    // Tile ID
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    __shared__ int np_tile; np_tile = 0;
    block_sync();

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;
    
    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        int const row = (ri1-ri0+1);
        int const vol = (rj1-rj0+1) * row;

        double dpcx = 1.0 / ppc.x;
        double dpcy = 1.0 / ppc.y;

        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;
        const float r2 = radius*radius;

        for( int idx = block_thread_rank(); idx < vol; idx += block_num_threads() ) {
            const int2 cell = make_int2( 
                idx % row + ri0,
                idx / row + rj0
            );
            for( unsigned i1 = 0; i1 < ppc.y; i1++ ) {
                for( unsigned i0 = 0; i0 < ppc.x; i0++) {
                    float2 const pos = make_float2(
                        dpcx * ( i0 + 0.5 ) - 0.5,
                        dpcy * ( i1 + 0.5 ) - 0.5
                    );
                    float gx = ((shiftx + cell.x) + (pos.x+0.5)) * dx.x;
                    float gy = ((shifty + cell.y) + (pos.y+0.5)) * dx.y;
                    
                    int inj = ((gx - center.x)*(gx - center.x) + (gy - center.y)*(gy - center.y)) < r2;
                    inj_np += inj;
                }
            }
        }
    }

    inj_np = warp::reduce_add( inj_np );
    if ( warp::thread_rank() == 0 ) {
        block::atomic_fetch_add( &np_tile, inj_np );
    } 

    block_sync();

    if ( block_thread_rank() == 0 ) {
        np[ tile_id ] = np_tile;
    }
}

}

void Density::Sphere::np_inject( Particles & particles, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    float2 sphere_center = center;
    sphere_center.x -= ref.x;
    sphere_center.y -= ref.y;

    dim3 grid( particles.ntiles.x, particles.ntiles.y );
    dim3 block( 1024 );
    kernel::sphere_np <<< grid, block >>> ( 
        range, sphere_center, radius, dx, ppc, particles, np
    );
}

namespace kernel {

__global__
void sparse( 
    bnd<unsigned int> range,
    float2 dx_particles, float2 dx,
    ParticleData const part )
{
    // Tile ID
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Number of particles already in the tile (injection appends after these)
    const int np_tile = part.np[ tile_id ];

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        const int offset =  part.offset[ tile_id ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        // Cell index shift for this tile with respect to global (0,0)
        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        // Distance between macroparticles in units of gridcell
        const float dpcx = dx_particles.x / dx.x;
        const float dpcy = dx_particles.y / dx.y;

        // Global index of the first macroparticle at or after each tile left edge.
        // Equivalent to ceil( left-tile-edge / spacing ).
        int npx0 = ( shiftx + ri0 ) / dpcx;
        int npy0 = ( shifty + rj0 ) / dpcy; 
        npx0 += ( npx0 * dpcx < shiftx + ri0 );
        npy0 += ( npy0 * dpcy < shifty + rj0     );
        // Global index of the first macroparticle at or after each tile right edge.
        int npx1 = ( shiftx + ri1 + 1 ) / dpcx;
        int npy1 = ( shifty + rj1 + 1 ) / dpcy; 
        npx1 += ( npx1 * dpcx < shiftx + ri1 + 1 );
        npy1 += ( npy1 * dpcy < shifty + rj1 + 1 );

        // Number of particles to inject in this tile (exact)
        const int npxmax = npx1 - npx0;
        const int npymax = npy1 - npy0;
        const int npmax  = npxmax * npymax;

        // Threads loop over particles to place. The count is exact, so the write
        // index is simply np_tile + idx (no stream compaction needed).
        for( int idx = block_thread_rank(); idx < npmax; idx += block_num_threads() ) {
            // (x,y) global index of selected particle
            const int npx = npx0 + idx % npxmax;
            const int npy = npy0 + idx / npxmax;

            // position in units of grid cell inside the tile
            float i = npx * dpcx - shiftx;
            float j = npy * dpcy - shifty;

            int2 cell = make_int2( (int) i, (int) j );
            if ( cell.x >= nx.x ) cell.x = nx.x - 1;
            if ( cell.y >= nx.y ) cell.y = nx.y - 1;

            const int k = np_tile + idx;
            ix[ k ] = cell;
            x[ k ]  = make_float2( i - cell.x - 0.5f, j - cell.y - 0.5f );
            u[ k ]  = make_float3( 0, 0, 0 );
        }

        if ( block_thread_rank() == 0 ) {
            part.np[ tile_id ] = np_tile + npmax;
        }
    }
}

}

void Density::Sparse::inject( Particles & particles,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{
    dim3 grid( particles.ntiles.x, particles.ntiles.y );
    dim3 block( 1024 );

    if ( ppc.x != 1 || ppc.y != 1 )
        std::cout << "(*info*) ppc is ignored for sparse density" << std::endl;

    kernel::sparse <<< grid, block >>> (
        range, dx_particles, dx, particles
    );

    // Capture the injection count so norm_charge() can attribute the correct charge to 
    // each macroparticle (q = n0 * inj_ncells / inj_ncells).
    inj_ncells = ( unsigned long ) ( range.x.upper - range.x.lower + 1 ) *
                 ( range.y.upper - range.y.lower + 1 );
    inj_np = particles.np_total();
}

namespace kernel {

__global__
void sparse_np(
    bnd<unsigned int> range,
    float2 dx_particles, float2 dx,
    ParticleData const part, int * np )
{
    // Tile ID - 1 thread per tile (returns immediately for excess threads)
    const int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tile_id >= part.ntiles.x * part.ntiles.y ) return;

    const int2 tile_idx = make_int2(
        tile_id % part.ntiles.x,
        tile_id / part.ntiles.x
    );

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        // Cell index shift for this tile with respect to global (0,0)
        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        // Distance between macroparticles in units of gridcell
        const float dpcx = dx_particles.x / dx.x;
        const float dpcy = dx_particles.y / dx.y;

        // Global index of the first macroparticle at or after each tile left edge.
        // Equivalent to ceil( left-tile-edge / spacing ).
        int npx0 = ( shiftx + ri0 ) / dpcx;
        int npy0 = ( shifty + rj0 ) / dpcy; 
        npx0 += ( npx0 * dpcx < shiftx + ri0 );
        npy0 += ( npy0 * dpcy < shifty + rj0     );
        // Global index of the first macroparticle at or after each tile right edge.
        int npx1 = ( shiftx + ri1 + 1 ) / dpcx;
        int npy1 = ( shifty + rj1 + 1 ) / dpcy; 
        npx1 += ( npx1 * dpcx < shiftx + ri1 + 1 );
        npy1 += ( npy1 * dpcy < shifty + rj1 + 1 );
        
        inj_np = ( npx1 - npx0 ) * ( npy1 - npy0 );
    }

    np[ tile_id ] = inj_np;
}

}

void Density::Sparse::np_inject( Particles & particles, 
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    const int ntiles = particles.ntiles.x * particles.ntiles.y;

    // Only need one thread per tile since the count is exact.
    // Extra threads will be ignored by the kernel.
    int block = ( ntiles > 1024 ) ? 1024 : ntiles;
    int grid  = ( ntiles - 1 ) / block + 1;

    kernel::sparse_np <<< grid, block >>> ( range, dx_particles, dx, particles, np );
}

namespace kernel {

__global__
void lattice(
    bnd<unsigned int> range,
    uint2 spacing,
    ParticleData const part )
{
    // Tile ID
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id = tile_idx.y * part.ntiles.x + tile_idx.x;

    // Number of particles already in the tile (injection appends after these)
    const int np_tile = part.np[ tile_id ];

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        const int offset =  part.offset[ tile_id ];
        int2   * __restrict__ ix = &part.ix[ offset ];
        float2 * __restrict__ x  = &part.x[ offset ];
        float3 * __restrict__ u  = &part.u[ offset ];

        // Cell shift of this tile w.r.t. the global grid
        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        // Get index of first and last lattice point in this tile's range.
        // Operation is equivalent px = ceil( edge / spacing ) 
        const int px0 = ( shiftx + ri0     + spacing.x - 1 ) / spacing.x;
        const int px1 = ( shiftx + ri1 + 1 + spacing.x - 1 ) / spacing.x;
        const int py0 = ( shifty + rj0     + spacing.y - 1 ) / spacing.y;
        const int py1 = ( shifty + rj1 + 1 + spacing.y - 1 ) / spacing.y;

        // Number of particles to inject in this tile (exact)
        const int npx = px1 - px0;
        const int npy = py1 - py0;
        const int npmax  = npx * npy;

        // Threads loop over particles to place. Each point lies on a cell corner.
        for( int idx = block_thread_rank(); idx < npmax; idx += block_num_threads() ) {
            const int px = px0 + idx % npx;
            const int py = py0 + idx / npx;

            const int k = np_tile + idx;
            ix[ k ] = make_int2( px * spacing.x - shiftx, py * spacing.y - shifty );
            x[ k ]  = make_float2( -0.5f, -0.5f );
            u[ k ]  = make_float3( 0, 0, 0 );
        }

        if ( block_thread_rank() == 0 ) {
            part.np[ tile_id ] = np_tile + npmax;
        }
    }
}

}

void Density::Lattice::inject( Particles & particles,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range ) const
{
    dim3 grid( particles.ntiles.x, particles.ntiles.y );
    dim3 block( 1024 );

    kernel::lattice <<< grid, block >>> ( range, spacing, particles );

    // Capture the injection count so norm_charge() can attribute the correct
    // charge to each macroparticle (q = n0 * inj_ncells / inj_np).
    inj_ncells = ( unsigned long ) ( range.x.upper - range.x.lower + 1 ) *
                 ( range.y.upper - range.y.lower + 1 );
    inj_np = particles.np_total();
}

namespace kernel {

__global__
void lattice_np(
    bnd<unsigned int> range,
    uint2 spacing,
    ParticleData const part, int * np )
{
    // Tile ID - 1 thread per tile (returns immediately for excess threads)
    const int tile_id = blockIdx.x * blockDim.x + threadIdx.x;
    if ( tile_id >= part.ntiles.x * part.ntiles.y ) return;

    const int2 tile_idx = make_int2(
        tile_id % part.ntiles.x,
        tile_id / part.ntiles.x
    );

    // Find injection range in tile coordinates
    const int2 nx = make_int2( part.nx.x, part.nx.y );

    int ri0 = range.x.lower - tile_idx.x * nx.x;
    int ri1 = range.x.upper - tile_idx.x * nx.x;

    int rj0 = range.y.lower - tile_idx.y * nx.y;
    int rj1 = range.y.upper - tile_idx.y * nx.y;

    int inj_np = 0;

    // If range overlaps with tile
    if (( ri0 < nx.x ) && ( ri1 >= 0 ) &&
        ( rj0 < nx.y ) && ( rj1 >= 0 )) {

        // Limit to range inside this tile
        if (ri0 < 0) ri0 = 0;
        if (rj0 < 0) rj0 = 0;
        if (ri1 >= nx.x ) ri1 = nx.x-1;
        if (rj1 >= nx.y ) rj1 = nx.y-1;

        // Lattice points sit on every spacing-th global cell; their count in this
        // tile's range follows from ceil( edge / spacing ) at each edge.
        const int shiftx = tile_idx.x * nx.x;
        const int shifty = tile_idx.y * nx.y;

        const int px0 = ( shiftx + ri0     + spacing.x - 1 ) / spacing.x;
        const int px1 = ( shiftx + ri1 + 1 + spacing.x - 1 ) / spacing.x;
        const int py0 = ( shifty + rj0     + spacing.y - 1 ) / spacing.y;
        const int py1 = ( shifty + rj1 + 1 + spacing.y - 1 ) / spacing.y;

        inj_np = ( px1 - px0 ) * ( py1 - py0 );
    }

    np[ tile_id ] = inj_np;
}

}

void Density::Lattice::np_inject( Particles & particles,
    uint2 const ppc, float2 const dx, float2 const ref, bnd<unsigned int> range,
    int * np ) const
{
    const int ntiles = particles.ntiles.x * particles.ntiles.y;

    // Only need one thread per tile since the count is exact.
    // Extra threads will be ignored by the kernel.
    int block = ( ntiles > 1024 ) ? 1024 : ntiles;
    int grid  = ( ntiles - 1 ) / block + 1;

    kernel::lattice_np <<< grid, block >>> ( range, spacing, particles, np );
}
