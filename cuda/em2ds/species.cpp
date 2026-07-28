/**
 * @file species.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#include "species.h"
#include <iostream>

/**
 * The following values were determined experimentally using a single
 * NVIDIA A100 80GB PCIe board
 * 
 */

/// @brief Optimal block size for push kernel
int constexpr opt_push_block = 1024;

/// @brief Optimal block size for move kernel
int constexpr opt_move_block = 256;

/// @brief Optimal minimum number of blocks for move/push kernels
int constexpr opt_min_blocks = 2048;

/**
 * @brief Construct a new Species object
 * 
 * @param name  Name for the species object (used for diagnostics)
 * @param m_q   Mass over charge ratio
 * @param ppc   Number of particles per cell
 */
Species::Species( std::string const name, float const m_q, uint2 const ppc,
    bool const add_tag ):
    ppc(ppc), add_tag(add_tag), name(name), m_q(m_q)
{

    // Validate parameters
    if ( m_q == 0 ) {
        std::cerr << "(*error*) Invalid m_q value, must be not 0, aborting...\n";
        exit(1);
    }

    if ( ppc.x < 1 || ppc.y < 1 ) {
        std::cerr << "(*error*) Invalid ppc value, must be >= 1 in all directions\n";
        exit(1);
    }

    // Set default parameters
    density   = new Density::Uniform( 1.0 );
    udist     = new UDistribution::None();
    bc        = species::bc_type (species::bc::periodic);
    push_type = species::boris;

    // Nullify pointers to data structures
    particles = nullptr;
    tmp = nullptr;
    sort = nullptr;

    d_energy = nullptr;
}


/**
 * @brief Destroy the Species object
 * 
 */
Species::~Species() {
    device::free( np_inj );
    device::free( d_energy );

    delete( tmp );
    delete( sort );
    delete( particles );
    delete( density );
    delete( udist );
};


/**
 * @brief Returns reciprocal Lorentz gamma factor: $ \frac{1}{\sqrt{u_x^2 + u_y^2 + u_z^2 + 1 }} $
 * 
 * @note Uses CUDA intrinsic fma() and rsqrt() functions
 * 
 * @param u         Generalized momentum in units of c
 * @return float    Reciprocal Lorentz gamma factor
 */
__host__ __device__ __inline__
float rgamma( const float3 u ) {

    return rsqrt( fma( u.z, u.z, fma( u.y, u.y, fma( u.x, u.x, 1.0f ) ) ) );

}


/**
 * @brief Inject particles in the complete simulation box
 * 
 */
void Species::inject( ) {

    float2 ref = make_float2( 0, 0 );

    density -> inject( *particles, ppc, dx, ref, particles -> g_range() );
}

/**
 * @brief Inject particles in a specific cell range
 * 
 */
void Species::inject( bnd<unsigned int> range ) {

    float2 ref = make_float2( 0, 0 );

    density -> inject( *particles, ppc, dx, ref, range );
}

/**
 * @brief Gets the number of particles that would be injected in a specific cell range
 * 
 * Although the routine only considers injection in a specific range, the
 * number of particles to be injected is calculated on all tiles (returning
 * zero on those, as expected)
 * 
 * @param range 
 * @param np        (device pointer) Number of particles to inject in each tile
 */
void Species::np_inject( bnd<unsigned int> range, int * np ) {

    float2 ref = make_float2( 0, 0 );

    density -> np_inject( *particles, ppc, dx, ref, range, np );
}

namespace kernel {

__global__
/**
 * @brief Physical boundary conditions for the x direction 
 * 
 * @param ntiles    Number of tiles
 * @param tile_idx  Tile index
 * @param tiles     Particle tile information
 * @param data      Particle data
 * @param nx        Tile grid size
 * @param bc        Boundary condition
 */
void species_bcx(
    ParticleData const part,
    species::bc_type const bc ) 
{
    const auto ntiles  = part.ntiles;
    const int nx = part.nx.x;
    const int2 tile_idx = make_int2( blockIdx.x * ( ntiles.x - 1 ), blockIdx.y );
    
    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    if ( tile_idx.x == 0 ) {
        // Lower boundary
        switch( bc.x.lower ) {
        case( species::bc::reflecting ) :
            for( int i = block_thread_rank(); i < np; i+= block_num_threads() ) {
                if( ix[i].x < 0 ) {
                    ix[i].x += 1;
                    x[i].x = -x[i].x;
                    u[i].x = -u[i].x;
                }
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( bc.x.upper ) {
        case( species::bc::reflecting ) :
            for( int i = block_thread_rank(); i < np; i+= block_num_threads() ) {
                if( ix[i].x >=  nx ) {
                    ix[i].x -= 1;
                    x[i].x = -x[i].x;
                    u[i].x = -u[i].x;
                }
            }
            break;
        default:
            break;
        }
    }
}

__global__
/**
 * @brief Physical boundary conditions for the y direction 
 * 
 * @param ntiles    Number of tiles
 * @param tile_idx  Tile index
 * @param tiles     Particle tile information
 * @param data      Particle data
 * @param nx        Tile grid size
 * @param bc        Boundary condition
 */
void species_bcy(
    ParticleData const part,
    species::bc_type const bc ) 
{
    const auto ntiles  = part.ntiles;
    const auto tile_idx = make_int2( blockIdx.x, blockIdx.y * (ntiles.y-1) );
    const int ny = part.nx.y;

    const int tid = tile_idx.y * ntiles.x + tile_idx.x;

    const int part_offset    = part.offset[ tid ];
    const int np             = part.np[ tid ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    if ( tile_idx.y == 0 ) {
        // Lower boundary
        switch( bc.y.lower ) {
        case( species::bc::reflecting ) :
            for( int i = block_thread_rank(); i < np; i+= block_num_threads() ) {
                if( ix[i].y < 0 ) {
                    ix[i].y += 1;
                    x[i].y = -x[i].y;
                    u[i].y = -u[i].y;
                }
            }
            break;
        default:
            break;
        }
    } else {
        // Upper boundary
        switch( bc.y.upper ) {
        case( species::bc::reflecting ) :
            for( int i = block_thread_rank(); i < np; i+= block_num_threads() ) {
                if( ix[i].y >=  ny ) {
                    ix[i].y -= 1;
                    x[i].y = -x[i].y;
                    u[i].y = -u[i].y;
                }
            }
            break;
        default:
            break;
        }
    }
}

}

/**
 * @brief Processes "physical" boundary conditions
 * 
 */
void Species::process_bc() {

    dim3 block( 1024 );

    // x boundaries
    if ( bc.x.lower > species::bc::periodic || bc.x.upper > species::bc::periodic ) {
        dim3 grid( 2, particles->ntiles.y );
        kernel::species_bcx <<< grid, block >>> ( *particles, bc );
    }

    // y boundaries
    if ( bc.y.lower > species::bc::periodic || bc.y.upper > species::bc::periodic ) {
        dim3 grid( particles->ntiles.x, 2 );
        kernel::species_bcx <<< grid, block >>> ( *particles, bc );
    }
}

/**
 * @brief Free stream particles 1 iteration
 * 
 * No acceleration or current deposition is performed. Used for debug purposes.
 * 
 */
void Species::advance( ) {

    // Advance positions
    move();
    
    // Process physical boundary conditions
    process_bc();
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

    // Increase internal iteration number
    iter++;
}

/**
 * @brief Free-stream particles 1 iteration
 * 
 * This routine will:
 * 1. Advance positions and deposit current
 * 2. Process boundary conditions
 * 3. Sort particles according to tiles
 * 
 * @param emf       EM fields
 * @param current   Electric durrent density
 */
void Species::advance( Current &current, Charge &charge ) {

    // Advance positions and deposit current/charge
    move( current.J, charge.rho );

    // Process physical boundary conditions
    process_bc();
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

    // Increase internal iteration number
    iter++;
}

/**
 * @brief Advance particles 1 iteration
 * 
 * This routine will:
 * 1. Advance momenta
 * 2. Advance positions and deposit current
 * 3. Process boundary conditions
 * 4. Sort particles according to tiles
 * 
 * @param emf       EM fields
 * @param current   Electric durrent density
 */
void Species::advance( EMF const &emf, Current &current, Charge & charge ) {

    // Advance momenta
    push( emf.E, emf.B );

    // Advance positions and deposit current
    move( current.J, charge.rho );

    // Process physical boundary conditions
    // process_bc();
    
    // Sort particles according to tile
    particles -> tile_sort( *tmp, *sort );

    // Increase internal iteration number
    iter++;
}

/**
 * @brief Quartic (4th order) spline weights for an in-cell position
 *
 * Even-order splines are centered on the nearest grid point (not the cell center)
 * so the 5 point stencil must be shifted by the returned cell offset.
 *
 * @param x     Particle position inside cell ( -0.5 <= x < 0.5 )
 * @param s     [out] 5 weights, ordered from point-2 to point+2 around the
 *              nearest grid point
 * @return     Cell offset of the nearest grid point (0 or 1)
 */
__device__ inline int spline_s4( const float x, float s[5] )
{
    // offset to be returned:
    // 0 (no shift), closest grid point is the current cell left-corner
    // 1 (shift), closest grid point is the next cell left-corner
    const int ix_offset = ( x >= 0.0f );

    // position with respect to closest grid-point (not grid-cell center)
    const float dx = x + 0.5f - ix_offset;

    const float t0 = 0.5f - dx;
    const float t1 = 0.5f + dx;
    const float t2 = t0 * t1;

    s[0] = ( t0 * t0 ) * ( t0 * t0 ) / 24.f;
    s[1] = ( 0.25f + t0 * ( 1.f + t0 * ( 1.5f + t2 ) ) ) / 6.f;
    s[2] = 0.45833334f + t2 * ( 0.5f + 0.25f * t2 );
    s[3] = ( 0.25f + t1 * ( 1.f + t1 * ( 1.5f + t2 ) ) ) / 6.f;
    s[4] = ( t1 * t1 ) * ( t1 * t1 ) / 24.f;

    return ix_offset;
}

namespace kernel {

__device__
/**
 * @brief Deposit current from single particle
 * 
 * @param J         Electric current buffer (in shared memory)
 * @param ystride   y-stride for J
 * @param ix0       Initial position of particle (cell)
 * @param x0        Initial position of particle (position inside cell)
 * @param u         Particle momenta
 * @param rg        1 / Lorentz γ
 * @param dx        Particle motion normalized to cell size
 * @param q         Particle charge
 */
inline void dep_current( float3 * const __restrict__ J, const int ystride,
    int2 ix0, float2 x0, float3 u, float rg, float2 dx, float q ) {

    // Find position time centered with velocity
    float2 x1 = make_float2( x0.x + 0.5f * dx.x, x0.y + 0.5f * dx.y );
    
    int2 const deltai = make_int2(
        ((x1.x >= 0.5f) - (x1.x < -0.5f)),
        ((x1.y >= 0.5f) - (x1.y < -0.5f))
    );

    int2 ix  = make_int2( ix0.x + deltai.x, ix0.y + deltai.y );
    float2 x = make_float2( x1.x - deltai.x, x1.y - deltai.y );

    const float jx = q * u.x * rg;
    const float jy = q * u.y * rg;
    const float jz = q * u.z * rg;

    // Linear Shape
    // int idx = ix.y * ystride + ix.x;
    // const float S0x = 0.5f - x.x;
    // const float S1x = 0.5f + x.x;

    // const float S0y = 0.5f - x.y;
    // const float S1y = 0.5f + x.y;

    // block::atomic_fetch_add( & J[ idx               ].x, S0y * S0x * jx );
    // block::atomic_fetch_add( & J[ idx               ].y, S0y * S0x * jy );
    // block::atomic_fetch_add( & J[ idx               ].z, S0y * S0x * jz );

    // block::atomic_fetch_add( & J[ idx + 1           ].x, S0y * S1x * jx );
    // block::atomic_fetch_add( & J[ idx + 1           ].y, S0y * S1x * jy );
    // block::atomic_fetch_add( & J[ idx + 1           ].z, S0y * S1x * jz );

    // block::atomic_fetch_add( & J[ idx + ystride     ].x, S1y * S0x * jx );
    // block::atomic_fetch_add( & J[ idx + ystride     ].y, S1y * S0x * jy );
    // block::atomic_fetch_add( & J[ idx + ystride     ].z, S1y * S0x * jz );

    // block::atomic_fetch_add( & J[ idx + ystride + 1 ].x, S1y * S1x * jx );
    // block::atomic_fetch_add( & J[ idx + ystride + 1 ].y, S1y * S1x * jy );
    // block::atomic_fetch_add( & J[ idx + ystride + 1 ].z, S1y * S1x * jz );

    // Quartic Shape
    float sx[5], sy[5];
    // Extra offset needed for even-order shape function since they are centered
    // with respect to nearest cell corner and not grid cell.
    const int ix_offset = spline_s4( x.x, sx );
    const int iy_offset = spline_s4( x.y, sy );

    const int idx = ( ix.y + iy_offset ) * ystride + ( ix.x + ix_offset );

    for ( int jj = -2; jj < 3; jj++ ) {
        for ( int ii = -2; ii < 3; ii++ ) {
            const float s = sx[ii + 2] * sy[jj + 2];
            block::atomic_fetch_add( & J[ idx + ii + jj * ystride].x, s * jx );
            block::atomic_fetch_add( & J[ idx + ii + jj * ystride].y, s * jy );
            block::atomic_fetch_add( & J[ idx + ii + jj * ystride].z, s * jz );
        }
    }
}

__device__
inline void dep_charge( float * const __restrict__ rho, const int ystride, int2 ix, float2 x, float q )
{
    // Linear Shape
    // int idx = ix.y * ystride + ix.x;
    // const float S0x = 0.5f - x.x;
    // const float S1x = 0.5f + x.x;
    // const float S0y = 0.5f - x.y;
    // const float S1y = 0.5f + x.y;
    
    // block::atomic_fetch_add( & rho[ idx               ], S0y * S0x * q );
    // block::atomic_fetch_add( & rho[ idx + 1           ], S0y * S1x * q );
    // block::atomic_fetch_add( & rho[ idx + ystride     ], S1y * S0x * q );
    // block::atomic_fetch_add( & rho[ idx + ystride + 1 ], S1y * S1x * q );

    // Quartic Shape
    float sx[5], sy[5];
    // Extra offset needed for even-order shape function since they are centered
    // with respect to nearest cell corner and not grid cell.
    const int ix_offset = spline_s4( x.x, sx );
    const int iy_offset = spline_s4( x.y, sy );

    const int idx = ( ix.y + iy_offset ) * ystride + ( ix.x + ix_offset );

    for ( int jj = -2; jj < 3; jj++ ) {
        for ( int ii = -2; ii < 3; ii++ ) {
            const float s = sx[ii + 2] * sy[jj + 2];
            block::atomic_fetch_add( & rho[ idx + ii + jj * ystride], s * q );
        }
    }
}

__global__
void __launch_bounds__(opt_move_block) move_deposit(
    ParticleData part,
    float3 * const __restrict__ d_current, unsigned int const current_offset, uint2 const ext_nx,
    float  * const __restrict__ d_charge, unsigned int const charge_offset, uint2 const charge_ext_nx,
    float2 const dt_dx, float const q, float2 const qnx, 
    unsigned long long * const __restrict__ d_nmove
) {
    const int ystride = ext_nx.x; 
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const auto ntiles = part.ntiles;

    extern __shared__ char block_shm[];
    float  * __restrict__ rho_local = reinterpret_cast< float * >  ( & block_shm[0] );
    float3 * __restrict__ J_local   = reinterpret_cast< float3 * > ( & block_shm[ tile_vol * sizeof(float) ] );

    // Zero local current/charge buffers
    for( auto i = block_thread_rank(); i < tile_vol; i+= block_num_threads() ) {
        J_local[i] = make_float3( 0, 0, 0 );
        rho_local[i] = 0;
    }

    block_sync();

    float3 * __restrict__ J   = & J_local[ current_offset ];
    float  * __restrict__ rho = & rho_local[ charge_offset ];

    // Move particles and deposit current
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id  = tile_idx.y * ntiles.x + tile_idx.x;

    const int offset          = part.offset[ tile_id ];
    const int tile_np         = part.np[ tile_id ];
    int2   * __restrict__ ix  = &part.ix[ offset ];
    float2 * __restrict__ x   = &part.x[ offset ];
    float3 * __restrict__ u   = &part.u[ offset ];

    // Select subset of particles to move
    // We are using multiple blocks per tile (blockIdx.z)
    const int set_size = ( tile_np + gridDim.z - 1 ) / gridDim.z;
    const int begin = blockIdx.z * set_size;
    const int end   = ( begin + set_size < tile_np ) ? begin + set_size : tile_np;

    for( auto i = begin + block_thread_rank(); i < end; i+= block_num_threads() ) {
        float3 pu = u[i];
        float2 const x0 = x[i];
        int2   const ix0 =ix[i];

        // Get 1 / Lorentz gamma
        float const rg = rgamma( pu );

        // Get particle motion
        float2 const delta = make_float2(
            dt_dx.x * rg * pu.x,
            dt_dx.y * rg * pu.y
        );

        // Deposit current
        dep_current( J, ystride, ix0, x0, pu, rg, delta, q );

        // Advance position
        float2 x1 = make_float2(
            x0.x + delta.x,
            x0.y + delta.y
        );

        // Check for cell crossings
        int2 const deltai = make_int2(
            ((x1.x >= 0.5f) - (x1.x < -0.5f)),
            ((x1.y >= 0.5f) - (x1.y < -0.5f))
        );

        // Correct position 
        x1.x -= deltai.x;
        x1.y -= deltai.y;

        int2 ix1 = make_int2(
            ix0.x + deltai.x,
            ix0.y + deltai.y
        );

        // Deposit charge
        dep_charge( rho, ystride, ix1, x1, q );

        // Store result
        x[i] = x1;
        ix[i] = ix1;

    }

    block_sync();

    // Add current and charge to global buffer
    const int tile_off = tile_id * tile_vol;

    if ( gridDim.z > 1 ) {
        for( auto i =  block_thread_rank(); i < tile_vol; i+= block_num_threads() ) {
            device::atomic_fetch_add( & d_current[tile_off + i].x, J_local[i].x );
            device::atomic_fetch_add( & d_current[tile_off + i].y, J_local[i].y );
            device::atomic_fetch_add( & d_current[tile_off + i].z, J_local[i].z );

            device::atomic_fetch_add( & d_charge[tile_off + i], rho_local[i] );
        }
    } else {
        for( auto i =  block_thread_rank(); i < tile_vol; i+= block_num_threads() ) {
            d_current[tile_off + i] += J_local[i];
            d_charge[tile_off + i] += rho_local[i];
        }
    }

    if ( block_thread_rank() == 0 && blockIdx.z == 0 ) {
        // Update total particle pushes counter (for performance metrics)
        unsigned long long np64 = tile_np;
        device::atomic_fetch_add( d_nmove, np64 );
    }
}

}

/**
 * @brief Moves particles and deposit current
 * 
 * Current will be accumulated on existing data
 * 
 * @param current   Current grid
 */
void Species::move( vec3grid<float3> * J, grid<float> * rho )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    const float2 qnx = make_float2(
        q * dx.x / dt,
        q * dx.y / dt
    );

    int tile_blocks = opt_min_blocks / (particles -> ntiles.x * particles -> ntiles.y);
    if ( tile_blocks < 1 ) tile_blocks = 1;

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y, tile_blocks );

    size_t shm_size = J -> tile_vol * ( sizeof(float) + sizeof(float3) );

    auto block = opt_move_block;
    block::set_shmem_size( kernel::move_deposit, shm_size );
    kernel::move_deposit <<< grid, block, shm_size >>> ( 
        *particles,
        J -> d_buffer, J -> offset, J -> ext_nx,
        rho -> d_buffer, rho -> offset, rho -> ext_nx,
        dt_dx, q, qnx, d_nmove
    );
}

namespace kernel {

__global__
void move(
    ParticleData part,
    float2 const dt_dx,
    unsigned long long * const __restrict__ d_nmove
) {

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    const int offset         = part.offset[ tile_id ];
    const int np             = part.np[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ offset ];
    float2 * __restrict__ x  = &part.x[ offset ];
    float3 * __restrict__ u  = &part.u[ offset ];

    for( int i = block_thread_rank(); i < np; i+= block_num_threads() ) {
        float3 pu = u[i];
        float2 x0 = x[i];
        int2 ix0 = ix[i];

        // Get 1 / Lorentz gamma
        float rg = rgamma( pu );

        // Get particle motion
        float2 delta = make_float2(
            dt_dx.x * rg * pu.x,
            dt_dx.y * rg * pu.y
        );

        // Advance position
        float2 x1 = make_float2(
            x0.x + delta.x,
            x0.y + delta.y
        );

        // Check for cell crossings
        int2 deltai = make_int2(
            ((x1.x >= 0.5f) - (x1.x < -0.5f)),
            ((x1.y >= 0.5f) - (x1.y < -0.5f))
        );

        // Correct position and store
        x1.x -= deltai.x;
        x1.y -= deltai.y;
                
        x[i] = x1;

        // Modify cell and store
        int2 ix1 = make_int2(
            ix0.x + deltai.x,
            ix0.y + deltai.y
        );
        ix[i] = ix1;
    }

    if ( block_thread_rank() == 0 ) { 
        // Update total particle pushes counter (for performance metrics)
        unsigned long long np64 = np;
        device::atomic_fetch_add( d_nmove, np64 );
    }
}

}

/**
 * @brief Moves particles (no current deposition)
 * 
 * This is usually used for test species: species that do not self-consistently
 * influence the simulation
 * 
 * @param current   Current grid
 */
void Species::move( )
{
    const float2 dt_dx = make_float2(
        dt / dx.x,
        dt / dx.y
    );

    const float2 qnx = make_float2(
        q * dx.x / dt,
        q * dx.y / dt
    );

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = 1024;

    kernel::move <<< grid, block >>> ( 
        *particles, dt_dx, d_nmove
    );
}

namespace kernel {

/**
 * @brief Advance momentum using a relativistic Boris pusher.
 * 
 * @note
 * 
 * The momemtum advance in this method is split into 3 parts:
 * 1. Perform half of E-field acceleration
 * 2. Perform full B-field rotation
 * 3. Perform half of E-field acceleration
 * 
 * Note that this implementation (as it is usual in textbooks) uses a
 * linearization of a tangent calculation in the rotation, which may lead
 * to issues for high magnetic fields.
 * 
 * For the future, other, more accurate, rotation algorithms should be used
 * instead, such as employing the full Euler-Rodrigues formula.
 * 
 * Uses CUDA intrinsic fma() functions
 * 
 * @param tem 
 * @param e 
 * @param b 
 * @param u 
 * @return float3 
 */
__device__ float3 dudt_boris( const float alpha, float3 e, float3 b, float3 u, double & energy )
{

    // First half of acceleration
    e.x *= alpha;
    e.y *= alpha;
    e.z *= alpha;

    float3 ut = make_float3( 
        u.x + e.x,
        u.y + e.y,
        u.z + e.z
    );

    {
        const float utsq = fma( ut.z, ut.z, fma( ut.y, ut.y, ut.x * ut.x ) );
        const float gamma = sqrt( 1.0f + utsq );
        
        // Get time centered energy
        energy += utsq / (gamma + 1.0f);

        // Time centered \alpha / \gamma
        const float alpha_gamma = alpha / gamma;

        // Rotation
        b.x *= alpha_gamma;
        b.y *= alpha_gamma;
        b.z *= alpha_gamma;
    }

    u.x = fma( b.z, ut.y, ut.x );
    u.y = fma( b.x, ut.z, ut.y );
    u.z = fma( b.y, ut.x, ut.z );

    u.x = fma( -b.y, ut.z, u.x );
    u.y = fma( -b.z, ut.x, u.y );
    u.z = fma( -b.x, ut.y, u.z );

    {
        const float otsq = 2.0f / 
            fma( b.z, b.z, fma( b.y, b.y, fma( b.x, b.x, 1.0f ) ) );
        
        b.x *= otsq;
        b.y *= otsq;
        b.z *= otsq;
    }

    ut.x = fma( b.z, u.y, ut.x );
    ut.y = fma( b.x, u.z, ut.y );
    ut.z = fma( b.y, u.x, ut.z );

    ut.x = fma( -b.y, u.z, ut.x );
    ut.y = fma( -b.z, u.x, ut.y );
    ut.z = fma( -b.x, u.y, ut.z );

    // Second half of acceleration
    ut.x += e.x;
    ut.y += e.y;
    ut.z += e.z;

    return ut;
}


/**
 * @brief Advance memntum using a relativistic Boris pusher for high magnetic fields
 * 
 * @note This is similar to the dudt_boris method above, but the rotation is done using
 * using an exact Euler-Rodriguez method.2
 * 
 * @param tem 
 * @param e 
 * @param b 
 * @param u 
 * @return float3 
 */
__device__ float3 dudt_boris_euler( const float alpha, float3 e, float3 b, float3 u, double & energy )
{

    // First half of acceleration
    e.x *= alpha;
    e.y *= alpha;
    e.z *= alpha;

    float3 ut = make_float3( 
        u.x + e.x,
        u.y + e.y,
        u.z + e.z
    );

    {
        const float utsq = fma( ut.z, ut.z, fma( ut.y, ut.y, ut.x * ut.x ) );
        const float gamma = sqrt( 1.0f + utsq );
        
        // Get time centered energy
        energy += utsq / (gamma + 1.0f);
        
        // Time centered 2 * \alpha / \gamma
        float const alpha2_gamma = ( alpha * 2 ) / gamma ;

        b.x *= alpha2_gamma;
        b.y *= alpha2_gamma;
        b.z *= alpha2_gamma;
    }

    {
        float const bnorm = sqrt(fma( b.x, b.x, fma( b.y, b.y, b.z * b.z ) ));
        float const s = -(( bnorm > 0 ) ? sin( bnorm / 2 ) / bnorm : 1 );

        float const ra = cos( bnorm / 2 );
        float const rb = b.x * s;
        float const rc = b.y * s;
        float const rd = b.z * s;

        float const r11 =   fma(ra,ra,rb*rb)-fma(rc,rc,rd*rd);
        float const r12 = 2*fma(rb,rc,ra*rd);
        float const r13 = 2*fma(rb,rd,-ra*rc);

        float const r21 = 2*fma(rb,rc,-ra*rd);
        float const r22 =   fma(ra,ra,rc*rc)-fma(rb,rb,rd*rd);
        float const r23 = 2*fma(rc,rd,ra*rb);

        float const r31 = 2*fma(rb,rd,ra*rc);
        float const r32 = 2*fma(rc,rd,-ra*rb);
        float const r33 =   fma(ra,ra,rd*rd)-fma(rb,rb,-rc*rc);

        u.x = fma( r11, ut.x, fma( r21, ut.y , r31 * ut.z ));
        u.y = fma( r12, ut.x, fma( r22, ut.y , r32 * ut.z ));
        u.z = fma( r13, ut.x, fma( r23, ut.y , r33 * ut.z ));
    }


    // Second half of acceleration
    u.x += e.x;
    u.y += e.y;
    u.z += e.z;

    return u;
}


// /**
//  * @brief Interpolate EM field values at particle position using linear 
//  * (1st order) interpolation.
//  * 
//  * The EM fields are assumed to be organized according to the Yee scheme with
//  * the charge defined at lower left corner of the cell
//  * 
//  * @param E         Pointer to position (0,0) of E field grid
//  * @param B         Pointer to position (0,0) of B field grid
//  * @param ystride   E and B grids y stride (must be signed)
//  * @param ix        Particle cell index
//  * @param x         Particle postion inside cell
//  * @param e[out]    E field at particle position
//  * @param b[out]    B field at particleposition
//  */
// __device__ void interpolate_fld( 
//     float3 const * const __restrict__ E, 
//     float3 const * const __restrict__ B, 
//     const int ystride,
//     const int2 ix, const float2 x, float3 & e, float3 & b)
// {
//     const int i = ix.x;
//     const int j = ix.y;

//     const float s0x = 0.5f - x.x;
//     const float s1x = 0.5f + x.x;

//     const float s0y = 0.5f - x.y;
//     const float s1y = 0.5f + x.y;

//     // Interpolate E field

//     e.x = ( E[i +    j *ystride].x * s0x + E[i+1 +    j *ystride].x * s1x ) * s0y +
//           ( E[i + (j+1)*ystride].x * s0x + E[i+1 + (j+1)*ystride].x * s1x ) * s1y;

//     e.y = ( E[i +    j *ystride].y * s0x + E[i+1 +    j *ystride].y * s1x ) * s0y +
//           ( E[i + (j+1)*ystride].y * s0x + E[i+1 + (j+1)*ystride].y * s1x ) * s1y;

//     e.z = ( E[i +    j *ystride].z * s0x + E[i+1 +    j *ystride].z * s1x ) * s0y +
//           ( E[i + (j+1)*ystride].z * s0x + E[i+1 + (j+1)*ystride].z * s1x ) * s1y;

//     // Interpolate B field
//     b.x = ( B[i +    j *ystride].x * s0x + B[i+1 +    j *ystride].x * s1x ) * s0y +
//           ( B[i + (j+1)*ystride].x * s0x + B[i+1 + (j+1)*ystride].x * s1x ) * s1y;

//     b.y = ( B[i +    j *ystride].y * s0x + B[i+1 +    j *ystride].y * s1x ) * s0y +
//           ( B[i + (j+1)*ystride].y * s0x + B[i+1 + (j+1)*ystride].y * s1x ) * s1y;

//     b.z = ( B[i +    j *ystride].z * s0x + B[i+1 +    j *ystride].z * s1x ) * s0y +
//           ( B[i + (j+1)*ystride].z * s0x + B[i+1 + (j+1)*ystride].z * s1x ) * s1y;
// }


/**
 * @brief Interpolate EM field values at particle position using quartic
 * (4th order) interpolation.
 * 
 * The EM fields are assumed to be organized according to the Yee scheme with
 * the charge defined at lower left corner of the cell
 * 
 * @param E         Pointer to position (0,0) of E field grid
 * @param B         Pointer to position (0,0) of B field grid
 * @param ystride   E and B grids y stride (must be signed)
 * @param ix        Particle cell index
 * @param x         Particle postion inside cell
 * @param e[out]    E field at particle position
 * @param b[out]    B field at particleposition
 */
__device__ void interpolate_fld( 
    float3 const * const __restrict__ E, 
    float3 const * const __restrict__ B, 
    const int ystride,
    const int2 ix, const float2 x, float3 & e, float3 & b)
{
    float sx[5], sy[5];
    // Extra offset needed for even-order shape function since they are centered
    // with respect to nearest cell corner and not grid cell.
    const int ix_offset = spline_s4( x.x, sx );
    const int iy_offset = spline_s4( x.y, sy );

    const int idx = ( ix.y + iy_offset ) * ystride + ( ix.x + ix_offset );

    e = make_float3( 0.f, 0.f, 0.f );
    b = make_float3( 0.f, 0.f, 0.f );

    for ( int jj = -2; jj < 3; jj++ ) {
        for ( int ii = -2; ii < 3; ii++ ) {
            const float s = sx[ii + 2] * sy[jj + 2];

            const float3 fe = E[ idx + ii + jj * ystride ];
            const float3 fb = B[ idx + ii + jj * ystride ];

            e.x += s * fe.x;
            e.y += s * fe.y;
            e.z += s * fe.z;

            b.x += s * fb.x;
            b.y += s * fb.y;
            b.z += s * fb.z;
        }
    }
}


template < species::pusher type >
__global__
void __launch_bounds__(opt_push_block) push ( 
    ParticleData const part,
    float3 const __restrict__ * E_buffer, float3 const __restrict__ *  B_buffer,
    uint2 const ntiles, const uint2 ext_nx, const int field_offset,
    float const alpha, double * const __restrict__ d_energy )
{
    const int field_vol = roundup4( ext_nx.x * ext_nx.y );
    extern __shared__ float3 local[];
    float3 * __restrict__ E_local = & local[0];
    float3 * __restrict__ B_local = & local[ field_vol ];

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id  = tile_idx.y * ntiles.x + tile_idx.x;

    const int tile_off = tile_id * field_vol;

    // Copy field values into shared memory
    block::memcpy2 ( E_local, & E_buffer[ tile_off ], 
                     B_local, & B_buffer[ tile_off ],
                     field_vol );

    float3 const * const __restrict__ E = & E_local[ field_offset ];
    float3 const * const __restrict__ B = & B_local[ field_offset ];

    block_sync();

    // Push particles
    const int part_offset    = part.offset[ tile_id ];
    const int tile_np        = part.np[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    double energy = 0;
    const int ystride = ext_nx.x;

    // Select subset of particles to move
    // We are using multiple blocks per tile (blockIdx.z)
    const int set_size = ( tile_np + gridDim.z - 1 ) / gridDim.z;
    const int begin = blockIdx.z * set_size;
    const int end   = ( begin + set_size < tile_np ) ? begin + set_size : tile_np;

    for( auto i = begin + block_thread_rank(); i < end; i+= block_num_threads() ) {

        // Interpolate field
        float3 e, b;
        interpolate_fld( E, B, ystride, ix[i], x[i], e, b );
        
        // Advance momentum
        float3 pu = u[i];
        
        if ( type == species::boris ) u[i] = dudt_boris( alpha, e, b, pu, energy );
        if ( type == species::euler ) u[i] = dudt_boris_euler( alpha, e, b, pu, energy );
    }

    // Add up energy from all threads
    energy = warp::reduce_add( energy );
    if ( warp::thread_rank() == 0 ) { 
        device::atomic_fetch_add( d_energy, energy );
    }
}

}

/**
 * @brief       Accelerates particles
 * 
 * @param E     Electric field
 * @param B     Magnetic field
 */
void Species::push( vec3grid<float3> * const E, vec3grid<float3> * const B )
{

    const float alpha = 0.5 * dt / m_q;
    
    device::zero( d_energy, 1 );

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y, 1 );
    auto block = opt_push_block;
    size_t shm_size = 2 * ( E -> tile_vol * sizeof(float3) );

    switch( push_type ) {
    case( species :: euler ):
        block::set_shmem_size( kernel::push <species::euler>, shm_size );
        kernel::push <species::euler> <<< grid, block, shm_size >>> ( 
            *particles, E->d_buffer, B->d_buffer,
            E->ntiles, E->ext_nx, E->offset, 
            alpha, d_energy
        );
        break;
    case( species :: boris ):
        block::set_shmem_size( kernel::push <species::boris>, shm_size );
        kernel::push <species::boris> <<< grid, block, shm_size >>> (
            *particles, E->d_buffer, B->d_buffer,
            E->ntiles, E->ext_nx, E->offset,
            alpha, d_energy
        );
        break;
    }
}

namespace kernel {

/**
 * @brief Deposit the four Darwin moment grids for a single particle
 *
 * Computes the quartic weights once and deposits all four grids in a single
 * stencil pass.
 *
 * @param J         Centered current buffer (points to cell (0,0) of the tile)
 * @param A         Acceleration density buffer
 * @param M1        Velocity flux diagonal buffer
 * @param M2        Velocity flux off-diagonal buffer
 * @param ystride   y-stride (shared by the four grids)
 * @param ix        Particle cell index
 * @param x         Particle position inside cell
 * @param j         Centered current to deposit into J
 * @param a         Acceleration density to deposit into A
 * @param m1        Velocity flux diagonal to deposit into M1
 * @param m2        Velocity flux off-diagonal to deposit into M2
 */
__device__ inline void dep_darwin_moments(
    float3 * const __restrict__ J,  float3 * const __restrict__ A,
    float3 * const __restrict__ M1, float3 * const __restrict__ M2,
    const int ystride, int2 ix, float2 x,
    float3 j, float3 a, float3 m1, float3 m2 )
{
    float sx[5], sy[5];
    // Extra offset needed for even-order shape function since they are centered
    // with respect to nearest cell corner and not grid cell.
    const int ix_offset = spline_s4( x.x, sx );
    const int iy_offset = spline_s4( x.y, sy );

    const int idx = ( ix.y + iy_offset ) * ystride + ( ix.x + ix_offset );

    for ( int jj = -2; jj < 3; jj++ ) {
        for ( int ii = -2; ii < 3; ii++ ) {
            const float s = sx[ii + 2] * sy[jj + 2];
            const int k = idx + ii + jj * ystride;

            device::atomic_fetch_add( & J[k].x, s * j.x );
            device::atomic_fetch_add( & J[k].y, s * j.y );
            device::atomic_fetch_add( & J[k].z, s * j.z );

            device::atomic_fetch_add( & A[k].x, s * a.x );
            device::atomic_fetch_add( & A[k].y, s * a.y );
            device::atomic_fetch_add( & A[k].z, s * a.z );

            device::atomic_fetch_add( & M1[k].x, s * m1.x );
            device::atomic_fetch_add( & M1[k].y, s * m1.y );
            device::atomic_fetch_add( & M1[k].z, s * m1.z );

            device::atomic_fetch_add( & M2[k].x, s * m2.x );
            device::atomic_fetch_add( & M2[k].y, s * m2.y );
            device::atomic_fetch_add( & M2[k].z, s * m2.z );
        }
    }
}

__global__
/**
 * @brief Kernel to deposit the time-centered Darwin moments
 *
 * Performs a tentative Boris push v(t-dt/2) -> v(t+dt/2) without storing it, then
 * deposits the centered velocity moments at the frozen position x(t).
 * 
 * Could be optimized to minimize clashes by writing to local buffer copies instead of 
 * global moment buffers (not a priority since it is only used at t=0).
 *
 * @param part          Particle data
 * @param q             Particle charge
 * @param alpha         Boris push coefficient ( 0.5 dt / (m/q) )
 * @param dt            Time step (for the dv/dt centered difference)
 * @param E_buffer      E field buffer
 * @param B_buffer      B field buffer
 * @param fld_offset    Offset to cell (0,0) in the field tile
 * @param fld_ext_nx    Field tile size (includes guard cells)
 * @param J_buffer      [out] centered current density    q v(t)
 * @param A_buffer      [out] acceleration density        q dv/dt(t)
 * @param M1_buffer     [out] velocity flux diagonal      q (vx^2, vy^2, vz^2)
 * @param M2_buffer     [out] velocity flux off-diagonal  q (vx vy, vx vz, vy vz)
 * @param mom_offset    Offset to cell (0,0) in the moment tile
 * @param mom_ext_nx    Moment tile size (includes guard cells)
 */
void deposit_darwin_moments(
    ParticleData const part, const float q, const float alpha, const float dt,
    float3 const * const __restrict__ E_buffer,
    float3 const * const __restrict__ B_buffer,
    int const fld_offset, uint2 const fld_ext_nx,
    float3 * const __restrict__ J_buffer,  float3 * const __restrict__ A_buffer,
    float3 * const __restrict__ M1_buffer, float3 * const __restrict__ M2_buffer,
    int const mom_offset, uint2 const mom_ext_nx )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id   = tile_idx.y * part.ntiles.x + tile_idx.x;

    const int fld_ystride   = fld_ext_nx.x;
    const int mom_ystride   = mom_ext_nx.x;
    const auto fld_tile_vol = roundup4( fld_ext_nx.x * fld_ext_nx.y );
    const auto mom_tile_vol = roundup4( mom_ext_nx.x * mom_ext_nx.y );

    float3 const * const __restrict__ E = & E_buffer[ tile_id * fld_tile_vol + fld_offset ];
    float3 const * const __restrict__ B = & B_buffer[ tile_id * fld_tile_vol + fld_offset ];

    float3 * const __restrict__ J  = & J_buffer [ tile_id * mom_tile_vol + mom_offset ];
    float3 * const __restrict__ A  = & A_buffer [ tile_id * mom_tile_vol + mom_offset ];
    float3 * const __restrict__ M1 = & M1_buffer[ tile_id * mom_tile_vol + mom_offset ];
    float3 * const __restrict__ M2 = & M2_buffer[ tile_id * mom_tile_vol + mom_offset ];

    const int p_offset = part.offset[ tile_id ];
    const int np       = part.np[ tile_id ];
    int2   const * __restrict__ const ix = &part.ix[ p_offset ];
    float2 const * __restrict__ const x  = &part.x[ p_offset ];
    float3 const * __restrict__ const u  = &part.u[ p_offset ];

    const float rdt = 1.0f / dt;

    for( int i = block_thread_rank(); i < np; i += block_num_threads() ) {

        // Interpolate fields at the frozen position
        float3 e, b;
        interpolate_fld( E, B, fld_ystride, ix[i], x[i], e, b );

        // Tentative Boris push (NOT stored): v(t-dt/2) -> v(t+dt/2)
        float3 const u0 = u[i];
        double energy = 0;
        float3 const u1 = dudt_boris( alpha, e, b, u0, energy );

        // Convert momenta to velocities
        float const rg0 = rgamma( u0 );
        float const rg1 = rgamma( u1 );
        float3 const v0 = make_float3( u0.x * rg0, u0.y * rg0, u0.z * rg0 );
        float3 const v1 = make_float3( u1.x * rg1, u1.y * rg1, u1.z * rg1 );

        // Time-centered velocity and acceleration
        float3 const v    = make_float3( 0.5f * ( v1.x + v0.x ), 0.5f * ( v1.y + v0.y ), 0.5f * ( v1.z + v0.z ) );
        float3 const dvdt = make_float3( ( v1.x - v0.x ) * rdt, ( v1.y - v0.y ) * rdt, ( v1.z - v0.z ) * rdt );

        const float3 j  = make_float3( q * v.x, q * v.y, q * v.z );
        const float3 a  = make_float3( q * dvdt.x, q * dvdt.y, q * dvdt.z );
        const float3 m1 = make_float3( q * v.x * v.x, q * v.y * v.y, q * v.z * v.z );
        const float3 m2 = make_float3( q * v.x * v.y, q * v.x * v.z, q * v.y * v.z );

        dep_darwin_moments( J, A, M1, M2, mom_ystride, ix[i], x[i], j, a, m1, m2 );
    }
}

}

/**
 * @brief Deposit the time-centered Darwin moments
 *
 * @param E     Electric field (real space, at current Darwin iteration)
 * @param B     Magnetic field (real space, at current Darwin iteration)
 * @param J     [out] centered current density
 * @param A     [out] acceleration density
 * @param M1    [out] velocity flux tensor (diagonal components)
 * @param M2    [out] velocity flux tensor (off-diagonal components)
 */
void Species::deposit_darwin_moments(
    vec3grid<float3> * const E, vec3grid<float3> * const B,
    vec3grid<float3> * const J, vec3grid<float3> * const A,
    vec3grid<float3> * const M1, vec3grid<float3> * const M2 ) const
{
    const float alpha = 0.5 * dt / m_q;

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = 64;

    kernel::deposit_darwin_moments <<< grid, block >>> (
        *particles, q, alpha, dt,
        E->d_buffer, B->d_buffer, E->offset, E->ext_nx,
        J->d_buffer, A->d_buffer, M1->d_buffer, M2->d_buffer,
        J->offset, J->ext_nx
    );
}


namespace kernel {

/**
 * @brief Deposit a single particle's retarded current using a quartic shape
 *
 * @param V         Current buffer (points to cell (0,0) of the tile)
 * @param ystride   y-stride for V
 * @param ix        Particle cell index
 * @param x         Particle position inside cell
 * @param val       Retarded current q v(t-dt/2) to deposit
 */
__device__ inline void dep_darwin_retarded_current(
    float3 * const __restrict__ V, const int ystride, int2 ix, float2 x, float3 val )
{
    float sx[5], sy[5];
    // Extra offset needed for even-order shape function since they are centered
    // with respect to nearest cell corner and not grid cell.
    const int ix_offset = spline_s4( x.x, sx );
    const int iy_offset = spline_s4( x.y, sy );

    const int idx = ( ix.y + iy_offset ) * ystride + ( ix.x + ix_offset );

    for ( int jj = -2; jj < 3; jj++ ) {
        for ( int ii = -2; ii < 3; ii++ ) {
            const float s = sx[ii + 2] * sy[jj + 2];
            const int k = idx + ii + jj * ystride;

            device::atomic_fetch_add( & V[k].x, s * val.x );
            device::atomic_fetch_add( & V[k].y, s * val.y );
            device::atomic_fetch_add( & V[k].z, s * val.z );
        }
    }
}

__global__
/**
 * @brief Kernel to deposit the retarded current q v(t-dt/2) at x(t)
 * 
 * Could be optimized to minimize clashes by writing to local buffer copy instead of 
 * global current buffer (not a priority since it is only used at t=0).
 *
 * @param part          Particle data
 * @param q             Particle charge
 * @param J_buffer      Current buffer
 * @param J_offset      Offset to cell (0,0) in the current tile
 * @param ext_nx        Current tile size (includes guard cells)
 */
void deposit_darwin_retarded_current(
    ParticleData const part, const float q,
    float3 * const __restrict__ J_buffer, int const J_offset, uint2 const ext_nx )
{
    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id   = tile_idx.y * part.ntiles.x + tile_idx.x;

    const int ystride   = ext_nx.x;
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );
    float3 * const __restrict__ J = & J_buffer[ tile_id * tile_vol + J_offset ];

    const int p_offset = part.offset[ tile_id ];
    const int np       = part.np[ tile_id ];
    int2   const * __restrict__ const ix = &part.ix[ p_offset ];
    float2 const * __restrict__ const x  = &part.x[ p_offset ];
    float3 const * __restrict__ const u  = &part.u[ p_offset ];

    for( int i = block_thread_rank(); i < np; i += block_num_threads() ) {
        const float rg = rgamma( u[i] );
        dep_darwin_retarded_current( J, ystride, ix[i], x[i],
                  make_float3( q * u[i].x * rg, q * u[i].y * rg, q * u[i].z * rg ) );
    }
}

}

/**
 * @brief Deposit the retarded current q v(t-dt/2) at x(t)
 *
 * Used only to seed the Darwin magnetic field at t=0.
 *
 * @param J     Current density grid
 */
void Species::deposit_darwin_retarded_current( vec3grid<float3> * const J ) const
{
    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = 64;

    kernel::deposit_darwin_retarded_current <<< grid, block >>> (
        *particles, q, J->d_buffer, J->offset, J->ext_nx
    );
}

namespace kernel {
__global__
/**
 * @brief Accumulate the boosted-Coulomb field of every particle in k-space
 *
 * One block per k-space line (iy); each thread owns a set of k-columns (ix) and
 * loops over ALL particles (all tiles), summing the exact frozen field of each as
 * a uniformly moving point charge. See Species::deposit_boosted_fields for the
 * formulas. The k=0 mode is skipped (left unchanged).
 *
 * @param part  Particle data (tiled)
 * @param q     Macroparticle charge
 * @param dx    Real-space cell size (to reconstruct global particle position)
 * @param d_fE  [in,out] full E-field accumulator (k-space)
 * @param d_fB  [in,out] B-field accumulator (k-space)
 * @param dims  Global (half-)spectrum dimensions
 * @param dk    k-space cell size
 */
void deposit_boosted_fields(
    ParticleData const part, float const q, float2 const dx,
    fft::complex64 * const __restrict__ d_fE,
    fft::complex64 * const __restrict__ d_fB,
    uint2 const dims, float2 const dk )
{
    fft::complex64 * const __restrict__ fEx = & d_fE[                   0 ];
    fft::complex64 * const __restrict__ fEy = & d_fE[     dims.x * dims.y ];
    fft::complex64 * const __restrict__ fEz = & d_fE[ 2 * dims.x * dims.y ];

    fft::complex64 * const __restrict__ fBx = & d_fB[                   0 ];
    fft::complex64 * const __restrict__ fBy = & d_fB[     dims.x * dims.y ];
    fft::complex64 * const __restrict__ fBz = & d_fB[ 2 * dims.x * dims.y ];

    const int iy   = blockIdx.x;
    const float ky = ((iy < int(dims.y/2)) ? iy : (iy - int(dims.y)) ) * dk.y;

    const int ntiles = part.ntiles.x * part.ntiles.y;
    const int stride = dims.x;

    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;
        if ( k2 == 0.f ) continue;    // k=0 mode left unchanged

        fft::complex64 Ex(0), Ey(0), Ez(0), Bx(0), By(0), Bz(0);

        // Loop over every particle in every tile
        for( int tid = 0; tid < ntiles; tid++ ) {
            const int    tx       = tid % part.ntiles.x;
            const int    ty       = tid / part.ntiles.x;
            const int    tile_off = part.offset[ tid ];
            const int    tile_np  = part.np[ tid ];
            int2   const * const __restrict__ pix = & part.ix[ tile_off ];
            float2 const * const __restrict__ pxf = & part.x[ tile_off ];
            float3 const * const __restrict__ pu  = & part.u[ tile_off ];

            for( int p = 0; p < tile_np; p++ ) {
                // Global physical position
                const float X = ( tx * part.nx.x + pix[p].x + 0.5f + pxf[p].x ) * dx.x;
                const float Y = ( ty * part.nx.y + pix[p].y + 0.5f + pxf[p].y ) * dx.y;

                // Velocity beta = u * rgamma(u)
                const float3 u  = pu[p];
                const float rg = rgamma( u );
                const float bx = u.x * rg;
                const float by = u.y * rg;
                const float bz = u.z * rg;

                const float bk = bx * kx + by * ky;   // beta . k   ( k_z = 0 )
                const float D = k2 - bk * bk;         // > 0 for beta < 1

                // phi = q e^{-i k.x} / D
                const float ph = -( kx * X + ky * Y );
                float s, c;
                sincosf( ph, &s, &c );
                const float invD = q / D;
                const fft::complex64 phi( c * invD, s * invD );

                // E = -i [ k - (beta.k) beta ] phi     ( -i*phi = ( phi.y, -phi.x ) )
                const fft::complex64 miphi( phi.y, -phi.x );
                Ex += miphi * ( kx - bk * bx );
                Ey += miphi * ( ky - bk * by );
                Ez += miphi * (    - bk * bz );

                // B = i ( k x beta ) phi               (  i*phi = ( -phi.y, phi.x ) )
                const fft::complex64 iphi( -phi.y, phi.x );
                Bx += iphi * (  ky * bz );
                By += iphi * ( -kx * bz );
                Bz += iphi * (  kx * by - ky * bx );
            }
        }

        fEx[idx] += Ex;
        fEy[idx] += Ey;
        fEz[idx] += Ez;
        fBx[idx] += Bx;
        fBy[idx] += By;
        fBz[idx] += Bz;
    }
}
}

void Species::deposit_boosted_fields(
    basic_grid3<std::complex<float>> & fE,
    basic_grid3<std::complex<float>> & fB ) const
{
    // One k-line per block
    // Each thread loops over all particles (exact, and thus O(n_particles x n_grid))
    kernel::deposit_boosted_fields <<< fE.dims.y, 256 >>> (
        *particles, q, dx,
        reinterpret_cast<fft::complex64 *>( fE.d_buffer ),
        reinterpret_cast<fft::complex64 *>( fB.d_buffer ),
        fE.dims, fft::dk( box )
    );
}

namespace kernel {
__global__
/**
 * @brief Kernel for charge density deposition
 * 
 * @param part              Particle data
 * @param q                 Particle charge
 * @param charge_buffer     Charge buffer
 * @param charge_offset     Offset to cell 0,0 in charge tile
 * @param ext_nx            External tile size (includes guard cells)
 */
void deposit_charge(
    ParticleData const part, const float q,
    float * const __restrict__ charge_buffer,
    int const charge_offset, uint2 const ext_nx
) {
    const int ystride = ext_nx.x; 
    const auto tile_vol = roundup4( ext_nx.x * ext_nx.y );
    extern __shared__ float charge_local[];

    // Zero shared memory and sync.
    for( auto i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        charge_local[i] = 0;
    }

    float *charge = &charge_local[ charge_offset ];

    block_sync();

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    const int offset   = part.offset[ tile_id ];
    const int np       = part.np[ tile_id ];
    int2   const * __restrict__ const ix = &part.ix[ offset ];
    float2 const * __restrict__ const x  = &part.x[ offset ];

    for( int i = block_thread_rank(); i < np; i += block_num_threads() ) {
        // Linear Shape
        // const int idx = ix[i].y * ystride + ix[i].x;
        // const float s0x = 0.5f - x[i].x;
        // const float s1x = 0.5f + x[i].x;
        // const float s0y = 0.5f - x[i].y;
        // const float s1y = 0.5f + x[i].y;

        // block::atomic_fetch_add( & charge[ idx               ], s0y * s0x * q );
        // block::atomic_fetch_add( & charge[ idx + 1           ], s0y * s1x * q );
        // block::atomic_fetch_add( & charge[ idx     + ystride ], s1y * s0x * q );
        // block::atomic_fetch_add( & charge[ idx + 1 + ystride ], s1y * s1x * q );

        // Quartic Shape
        float sx[5], sy[5];
        // Extra offset needed for even-order shape function since they are centered
        // with respect to nearest cell corner and not grid cell.
        const int ix_offset = spline_s4( x[i].x, sx );
        const int iy_offset = spline_s4( x[i].y, sy );

        const int idx = ( ix[i].x + iy_offset ) + ( ix[i].y + ix_offset ) * ystride;

        for ( int jj = -2; jj < 3; jj++ ) {
            for ( int ii = -2; ii < 3; ii++ ) {
                const float s = sx[ii + 2] * sy[jj + 2];
                block::atomic_fetch_add( & charge[ idx + ii + jj * ystride], s * q );
            }
        }
    }

    block_sync();

    // Copy data to global memory
    const int tile_off = tile_id * tile_vol;
    for( auto i = block_thread_rank(); i < tile_vol; i += block_num_threads() ) {
        charge_buffer[ tile_off + i ] += charge_local[i];
    } 

}

}

/**
 * @brief Deposit charge density
 * 
 * @param charge    Charge density grid
 */
void Species::deposit_charge( grid<float> &charge ) const {

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = 64;
    size_t shm_size = charge.tile_vol * sizeof(float);

    block::set_shmem_size( kernel::deposit_charge, shm_size );
    kernel::deposit_charge <<< grid, block, shm_size >>> (
        *particles, q, 
        charge.d_buffer, charge.offset, charge.ext_nx
    );
}


/**
 * @brief Save particle data to file
 * 
 */
void Species::save() const {

    const char * quants[] = {
        "x","y",
        "ux","uy","uz",
        "tag",
    };

    const char * qlabels[] = {
        "x","y",
        "u_x","u_y","u_z",
        "tag"
    };

    const char * qunits[] = {
        "c/\\omega_n", "c/\\omega_n",
        "c","c","c",
        ""
    };

    zdf::iteration iter_info = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Omit number of particles, this will be filled in later
    zdf::part_info info = {
        .name = (char *) name.c_str(),
        .label = (char *) name.c_str(),
        .nquants = add_tag ? 6u : 5u,
        .quants = (char **) quants,
        .qlabels = (char **) qlabels,
        .qunits = (char **) qunits,
    };

    // The particles object does not know the box dimensions
    // particles -> save( info, iter_info, "PARTICLES" );

    uint32_t np = particles -> np_total();
    info.np = np;

    // Open file
    zdf::file part_file;
    zdf::open_part_file( part_file, info, iter_info, "PARTICLES/" + name );

    // Gather and save each quantity
    float *d_data = nullptr;
    float *h_data = nullptr;
    if( np > 0 ) {
        d_data = device::malloc<float>( np );
        h_data = host::malloc<float>( np );
    }

    if ( np > 0 ) {
        float2 scale = make_float2( dx.x, 0 );
        particles -> gather( part::quant::x, d_data, scale );
        device::memcpy_tohost( h_data, d_data, np );
    }
    zdf::add_quant_part_file( part_file, "x", h_data, np );

    if ( np > 0 ) {
        float2 scale = make_float2( dx.y, 0 );
        particles -> gather( part::quant::y, d_data, scale );
        device::memcpy_tohost( h_data, d_data, np );
    }
    zdf::add_quant_part_file( part_file, "y", h_data, np );

    if ( np > 0 ) {
        particles -> gather( part::quant::ux, d_data );
        device::memcpy_tohost( h_data, d_data, np );
    }
    zdf::add_quant_part_file( part_file, "ux", h_data, np );

    if ( np > 0 ) {
        particles -> gather( part::quant::uy, d_data );
        device::memcpy_tohost( h_data, d_data, np );
    }
    zdf::add_quant_part_file( part_file, "uy", h_data, np );

    if ( np > 0 ) {
        particles -> gather( part::quant::uz, d_data );
        device::memcpy_tohost( h_data, d_data, np );
    }
    zdf::add_quant_part_file( part_file, "uz", h_data, np );

    // Particle tags (integer quantity)
    uint64_t *d_tag = nullptr;
    uint64_t *h_tag = nullptr;
    if ( add_tag ) {
        if ( np > 0 ) {
            d_tag = device::malloc<uint64_t>( np );
            h_tag = host::malloc<uint64_t>( np );
            particles -> gather( part::quant::tag, d_tag );
            device::memcpy_tohost( h_tag, d_tag, np );
        }
        zdf::add_quant_part_file( part_file, "tag", h_tag, np );
    }

    // Close the file
    zdf::close_file( part_file );

    // Cleanup
    if ( np > 0 ) {
        device::free( d_data );
        host::free( h_data );
        if ( add_tag ) {
            device::free( d_tag );
            host::free( h_tag );
        }
    }

}

/**
 * @brief Saves charge density to file
 * 
 * The routine will create a new charge grid, deposit the charge and save the grid
 * 
 */
void Species::save_charge() const {

    bnd<unsigned int> gc;

    // Linear Shape (1 guard cell at the upper boundary)
    // gc.x = {0,1};
    // gc.y = {0,1};

    // Quartic Shape (2 below, 3 above: ±2 stencil around the nearest grid point)
    gc.x = {2,3};
    gc.y = {2,3};

    // Deposit charge on device
    grid<float> charge( particles -> ntiles, particles -> nx, gc );

    charge.zero();

    deposit_charge( charge );

    charge.add_from_gc();

    // Prepare file info
    zdf::grid_axis axis[2];
    axis[0] = (zdf::grid_axis) {
        .name = (char *) "x",
        .min = 0.,
        .max = box.x,
        .label = (char *) "x",
        .units = (char *) "c/\\omega_n"
    };

    axis[1] = (zdf::grid_axis) {
        .name = (char *) "y",
        .min = 0.,
        .max = box.y,
        .label = (char *) "y",
        .units = (char *) "c/\\omega_n"
    };

    std::string grid_name = name + "-charge";
    std::string grid_label = name + " \\rho";

    zdf::grid_info info = {
        .name = (char *) grid_name.c_str(),
        .label = (char *) grid_label.c_str(),
        .units = (char *) "n_e",
        .axis  = axis
    };

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    std::string path = "CHARGE/";
    path += name;
    
    charge.save( info, iter_info, path );
}

namespace kernel {

/**
 * @brief kernel for depositing 1d phasespace
 * 
 * @tparam q        Phasespace quantity
 * @param d_data    Output data
 * @param range     Phasespace value range
 * @param size      Phasespace grid size
 * @param tile_nx   Size of tile grid
 * @param norm      Normalization factor
 * @param d_tiles   Particle tile information
 * @param d_ix      Particle data (cell)
 * @param d_x       Particle data (pos)
 * @param d_u       Particle data (generalized momenta)
 */
template < phasespace::quant quant >
__global__
void dep_pha1(
    float * const __restrict__ d_data, float2 const range, int const size,
    float const norm, 
    ParticleData const part )
{
    uint2 const tile_nx = part.nx;

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    const int part_offset = part.offset[ tile_id ];
    const int np          = part.np[ tile_id ];
    int2   * __restrict__ ix = &part.ix[ part_offset ];
    float2 * __restrict__ x  = &part.x[ part_offset ];
    float3 * __restrict__ u  = &part.u[ part_offset ];

    float const pha_rdx = size / (range.y - range.x);

    for( int i = block_thread_rank(); i < np; i += block_num_threads() ) {
        float d;
        switch( quant ) {
        case( phasespace:: x ): d = ( tile_idx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
        case( phasespace:: y ): d = ( tile_idx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
        case( phasespace:: ux ): d = u[i].x; break;
        case( phasespace:: uy ): d = u[i].y; break;
        case( phasespace:: uz ): d = u[i].z; break;
        }

        float n =  (d - range.x ) * pha_rdx - 0.5f;
        int   k = int( n + 1 ) - 1;
        float w = n - k;

        if ((k   >= 0) && (k   < size-1)) device::atomic_fetch_add( &d_data[k  ], (1-w) * norm );
        if ((k+1 >= 0) && (k+1 < size-1)) device::atomic_fetch_add( &d_data[k+1],    w  * norm );
    }
}

}

/**
 * @brief Deposit 1D phasespace
 * 
 * @note Output data will be zeroed before deposition
 * 
 * @param d_data    Output (device) data
 * @param quant     Phasespace quantity
 * @param range     Phasespace value range
 * @param size      Phasespace grid size
 */
void Species::dep_phasespace( float * const d_data, phasespace::quant quant, 
    float2 range, unsigned const size ) const
{
    // Zero device memory
    device::zero( d_data, size );
    
    // In OSIRIS we don't take the absolute value of q
    float norm = fabs(q) * ( dx.x * dx.y ) *
                 size / (range.y - range.x) ;

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = 64;

    float2 lrange = range;

    switch(quant) {
    case( phasespace::x ):
        lrange.y /= dx.x;
        lrange.x /= dx.x;
        kernel::dep_pha1<phasespace::x> <<< grid, block >>> 
            ( d_data, lrange, size, norm, *particles );
        break;
    case( phasespace:: y ):
        kernel::dep_pha1<phasespace::y> <<< grid, block >>> 
            ( d_data, lrange, size, norm, *particles );
        break;
    case( phasespace:: ux ):
        kernel::dep_pha1<phasespace::ux> <<< grid, block >>> 
            ( d_data, lrange, size, norm, *particles );
        break;
    case( phasespace:: uy ):
        kernel::dep_pha1<phasespace::uy> <<< grid, block >>> 
            ( d_data, lrange, size, norm, *particles );
        break;
    case( phasespace:: uz ):
        kernel::dep_pha1<phasespace::uz> <<< grid, block >>> 
            ( d_data, lrange, size, norm, *particles );
        break;
    };
}

/**
 * @brief Save 1D phasespace
 * 
 * @param q         Phasespace quantity
 * @param range     Phasespace range
 * @param size      Phasespace grid size
 */
void Species::save_phasespace( phasespace::quant quant, float2 const range, 
    int const size ) const
{
    std::string qname, qlabel, qunits;

    phasespace::qinfo( quant, qname, qlabel, qunits );
    
    // Prepare file info
    zdf::grid_axis axis = {
        .name = (char *) qname.c_str(),
        .min = range.x,
        .max = range.y,
        .label = (char *) qlabel.c_str(),
        .units = (char *) qunits.c_str()
    };

    std::string pha_name  = name + "-" + qname;
    std::string pha_label = name + "\\,(" + qlabel+")";

    zdf::grid_info info = {
        .name = (char *) pha_name.c_str(),
        .ndims = 1,
        .label = (char *) pha_label.c_str(),
        .units = (char *) "n_e",
        .axis  = &axis
    };

    info.count[0] = size;

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    // Deposit 1D phasespace
    float * d_data = device::malloc<float>( size );
    float * h_data = host::malloc<float>( size );
    
    dep_phasespace( d_data, quant, range, size );
    device::memcpy_tohost(  h_data, d_data, size );

    // Save file
    zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );

    host::free( h_data );
    device::free( d_data );
}

namespace kernel {

/**
 * @brief CUDA kernel for depositing 2D phasespace
 * 
 * @tparam q0       Quantity 0
 * @tparam q1       Quantity 1
 * @param d_data    Ouput data
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param range1    Range of values of quantity 1
 * @param size1     Range of values of quantity 1
 * @param tile_nx   Size of tile grid
 * @param norm      Normalization factor
 * @param d_tiles   Particle tile information
 * @param d_ix      Particle data (cell)
 * @param d_x       Particle data (pos)
 * @param d_u       Particle data (generalized momenta)
 */
template < phasespace::quant quant0, phasespace::quant quant1 >
__global__
void dep_pha2(
    float * const __restrict__ d_data, 
    float2 const range0, int const size0,
    float2 const range1, int const size1,
    float const norm, 
    ParticleData const part
) {
    static_assert( quant1 > quant0, "quant1 must be > quant0" );
    
    const auto tile_nx  = part.nx;

    const int2 tile_idx = make_int2( blockIdx.x, blockIdx.y );
    const int tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    const int offset = part.offset[ tile_id ];
    const int np     = part.np[ tile_id ];
    int2   * __restrict__ ix  = &part.ix[ offset ];
    float2 * __restrict__ x   = &part.x [ offset ];
    float3 * __restrict__ u   = &part.u [ offset ];

    float const pha_rdx0 = size0 / (range0.y - range0.x);
    float const pha_rdx1 = size1 / (range1.y - range1.x);

    for( int i = block_thread_rank(); i < np; i += block_num_threads() ) {
        float d0;
        switch( quant0 ) {
            case( phasespace:: x ):  d0 = ( tile_idx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
            case( phasespace:: y ):  d0 = ( tile_idx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
            case( phasespace:: ux ): d0 = u[i].x; break;
            case( phasespace:: uy ): d0 = u[i].y; break;
            case( phasespace:: uz ): d0 = u[i].z; break;
        }

        float n0 =  (d0 - range0.x ) * pha_rdx0 - 0.5f;
        int   k0 = int( n0 + 1 ) - 1;
        float w0 = n0 - k0;

        float d1;
        switch( quant1 ) {
            //case( phasespace:: x ):  d1 = ( tile_idx.x * tile_nx.x + ix[i].x) + (x[i].x + 0.5f); break;
            case( phasespace:: y ):  d1 = ( tile_idx.y * tile_nx.y + ix[i].y) + (x[i].y + 0.5f); break;
            case( phasespace:: ux ): d1 = u[i].x; break;
            case( phasespace:: uy ): d1 = u[i].y; break;
            case( phasespace:: uz ): d1 = u[i].z; break;
        }

        float n1 =  (d1 - range1.x ) * pha_rdx1 - 0.5f;
        int   k1 = int( n1 + 1 ) - 1;
        float w1 = n1 - k1;

        if ((k0   >= 0) && (k0   < size0-1) && (k1   >= 0) && (k1   < size1-1))
            device::atomic_fetch_add( &d_data[(k1  )*size0 + k0  ] , (1-w0) * (1-w1) * norm );
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1   >= 0) && (k1   < size1-1))
            device::atomic_fetch_add( &d_data[(k1  )*size0 + k0+1] ,    w0  * (1-w1) * norm );
        if ((k0   >= 0) && (k0   < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0  ] , (1-w0) *    w1  * norm );
        if ((k0+1 >= 0) && (k0+1 < size0-1) && (k1+1 >= 0) && (k1+1 < size1-1))
            device::atomic_fetch_add( &d_data[(k1+1)*size0 + k0+1] ,    w0  *    w1  * norm );
    }
}

}

/**
 * @brief Deposits a 2D phasespace in a device buffer
 * 
 * @note Output data will be zeroed before deposition
 *
 * @param d_data    Pointer to device buffer
 * @param quant0    Quantity 0
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param quant0    Quantity 1
 * @param range1    Range of values of quantity 1
 * @param size1     Phasespace grid size for quantity 1
 */
void Species::dep_phasespace( 
    float * const d_data,
    phasespace::quant quant0, float2 range0, unsigned const size0,
    phasespace::quant quant1, float2 range1, unsigned const size1 ) const
{
    // Zero device memory
    device::zero( d_data, size0 * size1 );

    // In OSIRIS we don't take the absolute value of q
    float norm = fabs(q) * ( dx.x * dx.y ) *
                          ( size0 / (range0.y - range0.x) ) *
                          ( size1 / (range1.y - range1.x) );

    dim3 grid( particles -> ntiles.x, particles -> ntiles.y );
    auto block = 64;

    float2 lrange0 = range0;
    float2 lrange1 = range1;

    switch(quant0) {
    case( phasespace::x ):
        lrange0.y /= dx.x;
        lrange0.x /= dx.x;
        switch(quant1) {
        case( phasespace::y ):
            lrange1.y /= dx.y;
            lrange1.x /= dx.y;
            kernel::dep_pha2 <phasespace::x,phasespace::y> <<< grid, block >>> (
                d_data, lrange0, size0, lrange1, size1, norm, *particles
            );
            break;
        case( phasespace::ux ):
            kernel::dep_pha2 <phasespace::x,phasespace::ux> <<< grid, block >>> (
                d_data, lrange0, size0, lrange1, size1, norm, *particles
            );
            break;
        case( phasespace::uy ):
            kernel::dep_pha2 <phasespace::x,phasespace::uy> <<< grid, block >>> (
                d_data, lrange0, size0, lrange1, size1, norm, *particles
            );
            break;
        case( phasespace::uz ):
            kernel::dep_pha2 <phasespace::x,phasespace::uz> <<< grid, block >>> (
                d_data, lrange0, size0, lrange1, size1, norm, *particles
            );
            break;
        default:
            break;
        }
        break;
    case( phasespace:: y ):
        lrange0.y /= dx.y;
        lrange0.x /= dx.y;
        switch(quant1) {
        case( phasespace::ux ):
            kernel::dep_pha2 <phasespace::y,phasespace::ux> <<< grid, block >>> (
                d_data, lrange0, size0, lrange1, size1, norm, *particles
            );
            break;
        case( phasespace::uy ):
            kernel::dep_pha2 <phasespace::y,phasespace::uy> <<< grid, block >>> (
                d_data, lrange0, size0, lrange1, size1, norm, *particles
            );
            break;
        case( phasespace::uz ):
            kernel::dep_pha2 <phasespace::y,phasespace::uz> <<< grid, block >>> (
                d_data, lrange0, size0, lrange1, size1, norm, *particles
            );
            break;
        default:
            break;
        }
        break;
    case( phasespace:: ux ):
        switch(quant1) {
        case( phasespace::uy ):
            kernel::dep_pha2 <phasespace::ux,phasespace::uy> <<< grid, block >>> (
                d_data, lrange0, size0, lrange1, size1, norm, *particles
            );
            break;
        case( phasespace::uz ):
            kernel::dep_pha2 <phasespace::ux,phasespace::uz> <<< grid, block >>> (
                d_data, lrange0, size0, lrange1, size1, norm, *particles
            );
            break;
        default:
            break;
        }
        break;
    case( phasespace:: uy ):
        kernel::dep_pha2 <phasespace::uy,phasespace::uz> <<< grid, block >>> (
            d_data, lrange0, size0, lrange1, size1, norm, *particles
        );
        break;
    default:
        break;
    };
}


/**
 * @brief Save 2D phasespace
 * 
 * @param quant0    Quantity 0
 * @param range0    Range of values of quantity 0
 * @param size0     Phasespace grid size for quantity 0
 * @param quant1    Quantity 1
 * @param range1    Range of values of quantity 1
 * @param size1     Phasespace grid size for quantity 0
 */
void Species::save_phasespace( 
    phasespace::quant quant0, float2 const range0, int const size0,
    phasespace::quant quant1, float2 const range1, int const size1 )
    const
{

    if ( quant0 >= quant1 ) {
        std::cerr << "(*error*) for 2D phasespaces, the 2nd quantity must be indexed higher than the first one\n";
        return;
    }

    std::string qname0, qlabel0, qunits0;
    std::string qname1, qlabel1, qunits1;

    phasespace::qinfo( quant0, qname0, qlabel0, qunits0 );
    phasespace::qinfo( quant1, qname1, qlabel1, qunits1 );
    
    // Prepare file info
    zdf::grid_axis axis[2] = {
        zdf::grid_axis {
            .name = (char *) qname0.c_str(),
            .min = range0.x,
            .max = range0.y,
            .label = (char *) qlabel0.c_str(),
            .units = (char *) qunits0.c_str()
        },
        zdf::grid_axis {
            .name = (char *) qname1.c_str(),
            .min = range1.x,
            .max = range1.y,
            .label = (char *) qlabel1.c_str(),
            .units = (char *) qunits1.c_str()
        }
    };

    std::string pha_name  = name + "-" + qname0 + qname1;
    std::string pha_label = name + " \\,(" + qlabel0 + "\\rm{-}" + qlabel1+")";

    zdf::grid_info info = {
        .name = (char *) pha_name.c_str(),
        .ndims = 2,
        .count = { static_cast<unsigned>(size0), static_cast<unsigned>(size1), 0 },
        .label = (char *) pha_label.c_str(),
        .units = (char *) "n_e",
        .axis  = axis
    };

    zdf::iteration iter_info = {
        .name = (char *) "ITERATION",
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };

    float * d_data = device::malloc<float>( size0 * size1 );
    float * h_data = host::malloc<float>( size0 * size1 );

    dep_phasespace( d_data, quant0, range0, size0, quant1, range1, size1 );
    device::memcpy_tohost(  h_data, d_data, size0 * size1 );

    zdf::save_grid( h_data, info, iter_info, "PHASESPACE/" + name );

    host::free( h_data );
    device::free( d_data );
}

/**
 * @brief Initialize data structures and inject particles
 * 
 * @param box_      Simulation global box size
 * @param ntiles    Number of tiles
 * @param nx        Grid size per tile
 * @param dt_       Time step
 * @param id_       Species unique id
 */
void Species::initialize( float2 const box_, uint2 const ntiles, uint2 const nx,
    float const dt_, int const id_ ) {

    // Store simulation box size
    box = box_;

    // Store simulation time step
    dt = dt_;

    // Store species id (used by RNG)
    id = id_;

    float2 gnx = make_float2( nx.x * ntiles.x, nx.y * ntiles.y );

    // Set cell size
    dx.x = box.x / (gnx.x);
    dx.y = box.y / (gnx.y);

    // Reference number maximum number of particles
    unsigned int max_part = 1.2 * gnx.x * gnx.y * ppc.x * ppc.y;

    particles = new Particles( ntiles, nx, max_part, add_tag );
    particles->periodic.x = ( bc.x.lower == species::bc::periodic );
    particles->periodic.y = ( bc.y.lower == species::bc::periodic );

    tmp = new Particles( ntiles, nx, max_part, add_tag );
    sort = new ParticleSort( ntiles, max_part );
    np_inj = device::malloc<int>( ntiles.x * ntiles.y );

    // Initialize energy diagnostic
    d_energy = device::malloc<double>( 1 );
    device::zero( d_energy, 1 );

    d_nmove = device::malloc<unsigned long long>( 1 );
    device::zero( d_nmove, 1 );

    // Reset iteration numbers
    iter = 0;

    // Inject initial distribution

    // Count particles to inject and store in particles -> offset
    np_inject( particles -> g_range(), np_inj );

    // Do an exclusive scan to get the required offsets
    device::exscan_add( particles -> offset, np_inj, ntiles.x * ntiles.y );

    // Inject the particles
    inject( particles -> g_range() );
    std::printf( "[%s] injected np_total=%lu particles\n", name.c_str(), particles->np_total() );

    // Assign a unique tag to every injected particle (buffer is still compact)
    if ( add_tag )
        particles -> set_tags();

    // Set charge normalization factor
    // Computed after injection so count-based profiles (e.g. Sparse) can use the
    // number of macroparticles actually placed.
    q = copysign( density->norm_charge( ppc ), m_q );

    // Set inital velocity distribution. The user-supplied `seed` offsets the
    // per-species seed so independent replicas (same physics, different seed)
    // draw distinct velocity realizations.
    udist -> set( *particles, id + seed );

    // particles -> validate( "After initial injection");
}