#include "udist.h"
#include "random.h"

namespace kernel {

__global__
void none( ParticleData const part ) {

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() )
        u[i] = make_float3( 0, 0, 0 );
}

}

/**
 * @brief Sets none(0 temperature, 0 fluid) u distribution
 * 
 * @param part  Particle data
 */
void UDistribution::None::set( Particles & part, unsigned int seed ) const {

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    kernel::none <<< grid, block >>> ( part );
}

namespace kernel {

__global__
void cold( ParticleData const part, const float3 ufl ) {

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() )
        u[i] = ufl;
}

}

/**
 * @brief Sets cold(0 temperatures) u distribution
 * 
 * @param part  Particle data
 */
void UDistribution::Cold::set( Particles & part, unsigned int seed ) const {

    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    kernel::cold <<< grid, block >>> ( part, ufl );
}

namespace kernel {

__global__
void thermal( ParticleData const part, uint2 rnd_seed, const float3 uth, const float3 ufl ) {

    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    // Initialize random state variables
    uint2 state; double norm;
    zrandom::rand_init( 
        tile_id * block_num_threads() +  block_thread_rank(),
        rnd_seed, state, norm
    );

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() ) {
        u[i] = make_float3(
            ufl.x + uth.x * zrandom::rand_norm( state, norm ),
            ufl.y + uth.y * zrandom::rand_norm( state, norm ),
            ufl.z + uth.z * zrandom::rand_norm( state, norm )
        );
    }
}

}

/**
 * @brief Sets momentum of all particles in object using uth / ufl
 * 
 */
void UDistribution::Thermal::set( Particles & part, unsigned int seed ) const
{
    uint2 rnd_seed{ 12345 + seed, 67890 };
    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    kernel::thermal <<< grid, block >>> ( part, rnd_seed, uth, ufl );
}

namespace kernel {

__global__
void thermal_corr( ParticleData const part, uint2 rnd_seed, const float3 uth, const float3 ufl, int const npmin )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;
    
    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    const auto ystride = part.nx.x;

    float3 * __restrict__ const u  = &part.u [ tile_off ];
    int2   * __restrict__ const ix = &part.ix[ tile_off ];

    // Initialize random state variables
    uint2 state; double norm;
    zrandom::rand_init( 
        tile_id * block_num_threads() +  block_thread_rank(),
        rnd_seed, state, norm
    );

    // Get shared memory addresses
    extern __shared__ char block_shm[];

    auto const bsize = part.nx.x * part.nx.y;
    int * const __restrict__ npcell   = reinterpret_cast<int*>    ( & block_shm[0] );
    float3 * const __restrict__ fluid = reinterpret_cast<float3*> ( & npcell[ bsize ] );

    for( auto idx = block_thread_rank(); idx < bsize; idx += block_num_threads() ) {
        fluid[idx] = make_float3( 0, 0, 0 );
        npcell[idx] = 0;
    }

    block_sync();

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() ) {
        float3 upart = make_float3(
            uth.x * zrandom::rand_norm( state, norm ),
            uth.y * zrandom::rand_norm( state, norm ),
            uth.z * zrandom::rand_norm( state, norm )
        );

        u[i] = upart;

        int const idx = ix[i].x + ystride * ix[i].y;

        block::atomic_fetch_add( & npcell[idx], 1 );
        block::atomic_fetch_add( & fluid[idx].x, upart.x );
        block::atomic_fetch_add( & fluid[idx].y, upart.y );
        block::atomic_fetch_add( & fluid[idx].z, upart.z );
    }

    block_sync();

    for( auto idx = block_thread_rank(); idx < bsize; idx += block_num_threads() ) {
        if ( npcell[idx] > npmin ) {
            fluid[idx].x /= npcell[idx];
            fluid[idx].y /= npcell[idx];
            fluid[idx].z /= npcell[idx];
        } else {
            fluid[idx] = make_float3( 0, 0, 0 );
        }
    }

    block_sync();

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() ) {
        int const idx = ix[i].x + ystride * ix[i].y;
        u[i] += ufl - fluid[idx];
    }
}

}

/**
 * @brief Sets particle momentum correcting local ufl fluctuations
 * 
 */
void UDistribution::ThermalCorr::set( Particles & part, unsigned int seed ) const {

    uint2 rnd_seed{ 12345 + seed, 67890 };
    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    const auto bsize = part.nx.x * part.nx.y;
    const size_t shm_size = bsize * ( sizeof( float3 ) + sizeof( int ) );
    block::set_shmem_size( kernel::thermal_corr, shm_size );
    kernel::thermal_corr <<< grid, block, shm_size >>> ( part, rnd_seed, uth, ufl, npmin );
}

namespace kernel {

/**
 * @brief Draws a single Maxwell-Jüttner proper momentum (rest frame)
 *
 * Sobol rejection sampling of the magnitude (Zenitani 2015, Phys. Plasmas 22,
 * 042116) followed by an isotropic direction. 
 * 
 * @note Acceptance becomes inefficient (but stays correct) for theta << 1.
 *
 * @param state     PRNG state, will be modified by this call
 * @param theta     Normalized temperature kT/(mc^2)
 * @return Proper momentum u = gamma * beta
 */
__device__
float3 maxwell_juttner_sample( uint2 & state, const float theta )
{
    // Sobol method
    double mag, eta;
    do {
        double x1 = zrandom::rand_real3( state );
        double x2 = zrandom::rand_real3( state );
        double x3 = zrandom::rand_real3( state );
        double x4 = zrandom::rand_real3( state );
        // candidate magnitude, sampled from Gamma(3,theta)
        mag = -theta * ( log(x1) + log(x2) + log(x3) );
        eta = mag - theta * log(x4);
    } while ( eta*eta - mag*mag <= 1.0 );

    // Isotropic direction on the sphere
    double cos_theta = 2.0 * zrandom::rand_real3( state ) - 1.0;
    double sin_theta = sqrt( 1.0 - cos_theta*cos_theta );
    double phi = 2.0 * M_PI * zrandom::rand_real3( state );

    return make_float3( mag*sin_theta*cos(phi), mag*sin_theta*sin(phi), mag*cos_theta );
}

__global__
void maxwell_juttner( ParticleData const part, uint2 rnd_seed, const float theta )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    float3 * __restrict__ const u  = &part.u[ tile_off ];

    // Initialize random state (no rand_norm here, so the normal cache is unused)
    uint2 state;
    zrandom::rand_init(
        tile_id * block_num_threads() +  block_thread_rank(),
        rnd_seed, state
    );

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() ) {
        u[i] = maxwell_juttner_sample( state, theta );
    }
}

__global__
void maxwell_juttner_corr( ParticleData const part, uint2 rnd_seed, const float theta,
    int const np_cell )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    const auto ystride = part.nx.x;

    float3 * __restrict__ const u  = &part.u [ tile_off ];
    int2   * __restrict__ const ix = &part.ix[ tile_off ];

    // Get shared memory address
    extern __shared__ char block_shm[];

    auto const bsize = part.nx.x * part.nx.y;
    int * const __restrict__ npcell = reinterpret_cast<int*> ( & block_shm[0] );

    for( auto idx = block_thread_rank(); idx < bsize; idx += block_num_threads() ) {
        npcell[idx] = 0;
    }

    block_sync();

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() ) {

        int const idx = ix[i].x + ystride * ix[i].y;

        // Rank of this particle inside its cell. Ranks (2p, 2p+1) form an
        // antithetic pair: both draw the same momentum, the odd one takes -u.
        // The distribution is isotropic so f(u) stays exactly Maxwell-Jüttner,
        // while sum(u) over the cell becomes exactly 0.
        int const rank = block::atomic_fetch_add( & npcell[idx], 1 );

        // The two members of a pair sit at arbitrary indices (injection places
        // consecutive particles in different cells), are handled by different
        // threads, and never communicate: they agree on the momentum solely by
        // seeding the same stream, keyed on (cell, pair). Hence the state is
        // seeded per particle here, rather than once per thread as in
        // kernel::maxwell_juttner above - the stream follows the pair, not the
        // thread. Cost is one redundant Sobol draw per pair.
        uint2 state;
        zrandom::rand_init(
            ( tile_id * bsize + idx ) * ( np_cell / 2 ) + ( rank >> 1 ),
            rnd_seed, state
        );

        float const sign = ( rank & 1 ) ? -1.f : +1.f;
        u[i] = maxwell_juttner_sample( state, theta ) * sign;
    }
}

/**
 * @brief Scrambles a stream key
 *
 * `zrandom::rand_init()` seeds the MWC lanes additively, so consecutive keys - which
 * here means neighbouring sub-areas - start from adjacent states. Since the whole
 * point of this loader is to suppress correlated long wavelength noise, the key is
 * scrambled first. This is the splitmix32 finalizer, a bijection on uint32, so
 * distinct keys keep mapping to distinct streams.
 */
__device__ __inline__
unsigned int scramble( unsigned int key )
{
    key ^= key >> 16; key *= 0x21f0aaad;
    key ^= key >> 15; key *= 0x735a2d97;
    key ^= key >> 15;
    return key;
}

__global__
void maxwell_juttner_corr_group( ParticleData const part, uint2 rnd_seed, const float theta,
    uint2 const group, int * const __restrict__ odd_groups )
{
    const uint2 tile_idx = { blockIdx.x, blockIdx.y };
    const int   tile_id  = tile_idx.y * part.ntiles.x + tile_idx.x;

    const auto  tile_off = part.offset[ tile_id ];
    const auto  tile_np  = part.np[ tile_id ];

    float3 * __restrict__ const u  = &part.u [ tile_off ];
    int2   * __restrict__ const ix = &part.ix[ tile_off ];

    // Sub-area grid inside the tile. `group` divides `nx` (checked on the host) so a
    // sub-area never straddles a tile boundary, which blocks could not handle.
    const int gstride = part.nx.x / group.x;
    const int ngroups = gstride * ( part.nx.y / group.y );

    // Get shared memory addresses
    extern __shared__ char block_shm[];

    // Particle counter per sub-area, reused as the rank counter after the scan
    int * const __restrict__ npgroup   = reinterpret_cast<int*> ( & block_shm[0] );
    // Index of the first antithetic pair of each sub-area, within this tile
    int * const __restrict__ pair_base = & npgroup[ ngroups ];

    for( auto idx = block_thread_rank(); idx < ngroups; idx += block_num_threads() ) {
        npgroup[idx] = 0;
    }

    block_sync();

    // Count the particles of each sub-area
    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() ) {
        int const idx = ( ix[i].x / group.x ) + gstride * ( ix[i].y / group.y );
        block::atomic_fetch_add( & npgroup[idx], 1 );
    }

    block_sync();

    // Serial exclusive scan of the number of pairs per sub-area. `ngroups` is at most
    // the number of cells in a tile and this runs once per simulation, so a single
    // thread is fast enough and much clearer than a chunked block scan.
    if ( block_thread_rank() == 0 ) {
        int base = 0;
        int odd = 0;
        for( int idx = 0; idx < ngroups; idx++ ) {
            int const np = npgroup[idx];
            odd += ( np & 1 );
            pair_base[idx] = base;
            // ceil() so that an odd (invalid) count still gets a private key range
            base += ( np + 1 ) / 2;
            // Reuse the counter for the particle ranks below
            npgroup[idx] = 0;
        }
        if ( odd ) device::atomic_fetch_add( odd_groups, odd );
    }

    block_sync();

    for( auto i = block_thread_rank(); i < tile_np; i += block_num_threads() ) {

        int const idx = ( ix[i].x / group.x ) + gstride * ( ix[i].y / group.y );

        // Rank of this particle inside its sub-area. Ranks (2p, 2p+1) form an
        // antithetic pair: both draw the same momentum, the odd one takes -u. The
        // distribution is isotropic so f(u) stays exactly Maxwell-Jüttner, while
        // sum(u) - and, since gamma(|u|) is even in u, sum(v) - over the sub-area
        // becomes exactly 0.
        int const rank = block::atomic_fetch_add( & npgroup[idx], 1 );

        // As in kernel::maxwell_juttner_corr the two members of a pair never
        // communicate: they agree on the momentum by seeding the same stream, keyed on
        // (sub-area, pair). Unlike that kernel the key needs no particles-per-cell
        // stride: the number of pairs preceding this sub-area in the tile comes from
        // the scan above, and sum(ceil(np/2)) <= tile_np, so the key stays inside
        // [tile_off, tile_off + tile_np) and never collides with another tile.
        uint2 state;
        zrandom::rand_init(
            scramble( tile_off + pair_base[idx] + ( rank >> 1 ) ),
            rnd_seed, state
        );

        float const sign = ( rank & 1 ) ? -1.f : +1.f;
        u[i] = maxwell_juttner_sample( state, theta ) * sign;
    }
}

}

/**
 * @brief Sets a Maxwell-Jüttner (relativistic Maxwellian) momentum distribution
 *
 * Uses the Sobol method to sample the isotropic proper-momentum magnitude
 * (Zenitani 2015, Phys. Plasmas 22, 042116). 
 * Rest-frame only (zero net drift).
 */
void UDistribution::MaxwellJuttner::set( Particles & part, unsigned int seed ) const
{
    uint2 rnd_seed{ 12345 + seed, 67890 };
    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    kernel::maxwell_juttner <<< grid, block >>> ( part, rnd_seed, theta );
}

/**
 * @brief Sets a Maxwell-Jüttner momentum distribution with zero net cell current
 *
 * Same sampling as MaxwellJuttner, but the ppc particles of each cell are loaded
 * as antithetic pairs ( u, -u ), so every cell carries exactly zero net drift.
 */
void UDistribution::MaxwellJuttnerCorr::set( Particles & part, unsigned int seed ) const
{
    uint2 rnd_seed{ 12345 + seed, 67890 };
    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    const auto bsize = part.nx.x * part.nx.y;
    const size_t shm_size = bsize * sizeof( int );
    block::set_shmem_size( kernel::maxwell_juttner_corr, shm_size );
    kernel::maxwell_juttner_corr <<< grid, block, shm_size >>> (
        part, rnd_seed, theta, ppc.x * ppc.y );
}

/**
 * @brief Sets a Maxwell-Jüttner momentum distribution with zero net current per sub-area
 *
 * Same sampling as MaxwellJuttner, but the particles of each sub-area (a block of
 * `group` cells, e.g. a Debye square) are loaded as antithetic pairs ( u, -u ), so every
 * sub-area carries exactly zero net momentum and zero net current, and no magnetic field
 * is seeded below k ~ 2 pi / ( group * dx ).
 *
 * This is the loader for ppc < 1 (sparse / lattice) densities, where a cell holds at
 * most one particle and MaxwellJuttnerCorr cannot pair anything.
 */
void UDistribution::MaxwellJuttnerCorrGroup::set( Particles & part, unsigned int seed ) const
{
    // A sub-area must not straddle a tile boundary
    if ( part.nx.x % group.x || part.nx.y % group.y ) {
        std::cout << "(*error*) MaxwellJuttnerCorrGroup group size (" << group.x << ","
                  << group.y << ") does not divide the tile size (" << part.nx.x << ","
                  << part.nx.y << ")\n";
        exit(1);
    }

    uint2 rnd_seed{ 12345 + seed, 67890 };
    dim3 grid( part.ntiles.x, part.ntiles.y );
    dim3 block( 64 );

    const auto ngroups = ( part.nx.x / group.x ) * ( part.nx.y / group.y );
    // Particle counter + first pair index, per sub-area
    const size_t shm_size = 2 * ngroups * sizeof( int );

    // Number of sub-areas holding an odd number of particles, i.e. with one particle
    // left without a partner
    device::Var<int> odd_groups( 0 );

    block::set_shmem_size( kernel::maxwell_juttner_corr_group, shm_size );
    kernel::maxwell_juttner_corr_group <<< grid, block, shm_size >>> (
        part, rnd_seed, theta, group, odd_groups.ptr() );

    // Also synchronizes with the kernel above
    const int odd = odd_groups.get();
    if ( odd > 0 ) {
        std::cout << "(*error*) MaxwellJuttnerCorrGroup: " << odd
                  << " sub-areas of (" << group.x << "," << group.y << ") cells hold an "
                  << "odd number of particles, antithetic pairing cannot cancel them.\n"
                  << "(*error*) Adjust the group size or the macroparticle spacing so "
                  << "that every sub-area holds an even number of particles.\n";
        exit(1);
    }
}