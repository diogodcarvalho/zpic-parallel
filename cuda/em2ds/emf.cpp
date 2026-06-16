#include "emf.h"

#include <iostream>
#include "zdf-cpp.h"

/**
 * @brief Construct a new EMF::EMF object
 * 
 * @param ntiles    Number of tiles
 * @param nx        Tile grid size
 * @param box       Simulation box size
 * @param dt        Time step
 * @param q         Sycl queue
 */
EMF::EMF( uint2 const ntiles, uint2 const nx, float2 const box,
    double const dt ) : 
    dx( float2{ box.x / ( nx.x * ntiles.x ), box.y / ( nx.y * ntiles.y ) } ),
    dt( dt ), box(box)
{
    // Verify Courant condition
    float cour = std::sqrt( 1.0f/( 1.0f/(dx.x*dx.x) + 1.0f/(dx.y*dx.y) ) );
    if ( dt >= cour ){
        std::cerr << "(*error*) Invalid timestep, courant condition violation.\n";
        std::cerr << "(*error*) For the current resolution [" << dx.x << "," << dx.y << "]\n";
        std::cerr << " the maximum timestep is dt = " << cour <<'\n';
        exit(-1);
    }

    // These are required for the Yee solver AND for field interpolation
    bnd<unsigned int> gc;

    // Linear Shape (1 below, 2 above)
    // gc.x = {1,2};
    // gc.y = {1,2};

    // Quartic Shape (2 below, 3 above: ±2 interpolation stencil around the
    // nearest grid point)
    gc.x = {2,3};
    gc.y = {2,3};

    E = new vec3grid<float3> ( ntiles, nx, gc );
    E -> name = "Electric field";

    // Check that local memory can hold up to 2 times the tile buffer
    auto local_mem_size = block::shared_mem_size();
    if ( local_mem_size < 2 * E->tile_vol * sizeof( float3 ) ) {
        std::cerr << "(*error*) Tile size too large " << nx << " (plus guard cells)\n";
        std::cerr << "(*error*) Insufficient local memory (" << local_mem_size << " B) for EMF object.\n";
        abort();
    }

    B = new vec3grid<float3> ( ntiles, nx, gc );
    B -> name = "Magnetic field";

    // Create FFT plan
    fft_backward = new fft::plan( E -> dims, fft::type::c2r_v3 );

    auto fdims = fft::fdims( E -> dims );
    fE  = new basic_grid3< std::complex<float> >( fdims );
    fEt = new basic_grid3< std::complex<float> >( fdims );
    fB  = new basic_grid3< std::complex<float> >( fdims );

    fE  -> name = "F(E)";
    fEt -> name = "F(Et)";
    fB  -> name = "F(B)";

    // Zero fields
    E -> zero();
    B -> zero();

    fE  -> zero();
    fEt -> zero();
    fB  -> zero();

    // Reserve device memory for energy diagnostic
    d_energy = device::malloc<double>( 6 );

    // Set default boundary conditions to periodic
    bc = emf::bc_type (emf::bc::periodic);

    // Reset iteration number
    iter = 0;

}



namespace kernel {

__global__
/**
 * @brief Advance transverse component of E field and B field using PSATD
 *        algorithm
 * 
 * @param fEt   Fourier transform of transverse component of E-field
 * @param fB    Fourier transform of magnetic field
 * @param dims  Global grid dimensions
 * @param dk    k-space cell size
 * @param dt    Time step
 */
void advance_psatd_nocurr( 
    fft::complex64 * const __restrict__ fEt, 
    fft::complex64 * const __restrict__ fB,
    uint2 const dims, float2 const dk, float const dt )
{

    fft::complex64 * const __restrict__ fEtx = & fEt[                   0 ];
    fft::complex64 * const __restrict__ fEty = & fEt[     dims.x * dims.y ];
    fft::complex64 * const __restrict__ fEtz = & fEt[ 2 * dims.x * dims.y ];

    fft::complex64 * const __restrict__ fBx  = & fB[                   0 ];
    fft::complex64 * const __restrict__ fBy  = & fB[     dims.x * dims.y ];
    fft::complex64 * const __restrict__ fBz  = & fB[ 2 * dims.x * dims.y ];

    const int iy   = blockIdx.x;  // Line
    const float ky = ((iy < int(dims.y/2)) ? iy : (iy - int(dims.y)) ) * dk.y;

    const int stride = dims.x;
    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;
        const float k  = sqrtf( k2 );

        // PSATD Field advance equations
        const float C   = cosf( k * dt );
        const float S_k = ( k > 0 ) ? sinf( k * dt ) / k : dt;

        fft::complex64 Ex = fEtx[idx];
        fft::complex64 Ey = fEty[idx];
        fft::complex64 Ez = fEtz[idx];

        fft::complex64 Bx = fBx[idx];
        fft::complex64 By = fBy[idx];
        fft::complex64 Bz = fBz[idx];

        Ex = C * Ex + S_k * ( fft::I * (  ky *  fBz[idx]                  ) );
        Ey = C * Ey + S_k * ( fft::I * ( -kx *  fBz[idx]                  ) );
        Ez = C * Ez + S_k * ( fft::I * (  kx *  fBy[idx] - ky *  fBx[idx] ) );

        Bx = C * Bx - S_k * ( fft::I * (  ky * fEtz[idx]                  ) );
        By = C * By - S_k * ( fft::I * ( -kx * fEtz[idx]                  ) );
        Bz = C * Bz - S_k * ( fft::I * (  kx * fEty[idx] - ky * fEtx[idx] ) );

        fEtx[idx] = Ex;
        fEty[idx] = Ey;
        fEtz[idx] = Ez;

        fBx[idx]  = Bx;
        fBy[idx]  = By;
        fBz[idx]  = Bz;
    }
}

__global__
/**
 * @brief Advance transverse component of E field and B field using PSATD
 *        algorithm
 * 
 * @param fEt   Fourier transform of transverse component of E-field
 * @param fB    Fourier transform of magnetic field
 * @param fJ    Fourier transform of current density
 * @param dims  Global grid dimensions
 * @param dk    k-space cell size
 * @param dt    Time step
 */
void advance_psatd( fft::complex64 * const __restrict__ fEt, 
                    fft::complex64 * const __restrict__ fB,
                    fft::complex64 * const __restrict__ fJ,
                    uint2 const dims, float2 const dk, float const dt )
{

    fft::complex64 * const __restrict__ fEtx = & fEt[                   0 ];
    fft::complex64 * const __restrict__ fEty = & fEt[     dims.x * dims.y ];
    fft::complex64 * const __restrict__ fEtz = & fEt[ 2 * dims.x * dims.y ];

    fft::complex64 * const __restrict__ fBx = & fB[                   0 ];
    fft::complex64 * const __restrict__ fBy = & fB[     dims.x * dims.y ];
    fft::complex64 * const __restrict__ fBz = & fB[ 2 * dims.x * dims.y ];

    fft::complex64 * const __restrict__ fJx = & fJ[                   0 ];
    fft::complex64 * const __restrict__ fJy = & fJ[     dims.x * dims.y ];
    fft::complex64 * const __restrict__ fJz = & fJ[ 2 * dims.x * dims.y ];

    const int iy   = blockIdx.x;  // Line
    const float ky = ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) * dk.y;

    const int stride = dims.x;
    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;
        const float k  = sqrt( k2 );

        // Calculate transverse current
        const fft::complex64 kdJ  = (kx * fJx[idx] + ky * fJy[idx]);
        const fft::complex64 fJtx = (k2 > 0) ? fJx[idx] - kx * kdJ / k2 : 0;
        const fft::complex64 fJty = (k2 > 0) ? fJy[idx] - ky * kdJ / k2 : 0; 
        const fft::complex64 fJtz = fJz[idx];

        // PSATD Field advance equations
        const float C   = cos( k * dt );
        const float S_k = ( k > 0 ) ? sin( k * dt ) / k : dt;
        const fft::complex64 I1mC_k2 = ( k2 > 0 )? fft::I * (1.0f - C) / k2 : 0;

        fft::complex64 Ex = fEtx[idx];
        fft::complex64 Ey = fEty[idx];
        fft::complex64 Ez = fEtz[idx];

        fft::complex64 Bx = fBx[idx];
        fft::complex64 By = fBy[idx];
        fft::complex64 Bz = fBz[idx];

        Ex = C * Ex + S_k * ( fft::I * (  ky *  fBz[idx]                  ) - fJtx );
        Ey = C * Ey + S_k * ( fft::I * ( -kx *  fBz[idx]                  ) - fJty );
        Ez = C * Ez + S_k * ( fft::I * (  kx *  fBy[idx] - ky *  fBx[idx] ) - fJtz );

        Bx = C * Bx - S_k * ( fft::I * (  ky * fEtz[idx]                  ) ) + I1mC_k2 * (  ky * fJz[idx]                 );
        By = C * By - S_k * ( fft::I * ( -kx * fEtz[idx]                  ) ) + I1mC_k2 * ( -kx * fJz[idx]                 );
        Bz = C * Bz - S_k * ( fft::I * (  kx * fEty[idx] - ky * fEtx[idx] ) ) + I1mC_k2 * (  kx * fJy[idx] - ky * fJx[idx] );

        fEtx[idx] = Ex;
        fEty[idx] = Ey;
        fEtz[idx] = Ez;

        fBx[idx]  = Bx;
        fBy[idx]  = By;
        fBz[idx]  = Bz;
    }
}

__global__
/**
 * @brief Update Electric field from charge density
 * 
 * @param fE    Fourier transform of E-field (full)
 * @param fEt   Fourier transform of E-field (transverse)
 * @param frho  Fourier transform of charge density
 * @param dims  Global grid dimensions
 * @param dk    k-space cell size
 */
void update_fE( fft::complex64 * const __restrict__ d_fE, 
                fft::complex64 * const __restrict__ d_fEt, 
                fft::complex64 * const __restrict__ d_frho,
                uint2 const dims, float2 const dk )
{

    fft::complex64 * const __restrict__ fEx = & d_fE[                   0 ];
    fft::complex64 * const __restrict__ fEy = & d_fE[     dims.x * dims.y ];
    fft::complex64 * const __restrict__ fEz = & d_fE[ 2 * dims.x * dims.y ];

    fft::complex64 const * const __restrict__ fEtx = & d_fEt[                   0 ];
    fft::complex64 const * const __restrict__ fEty = & d_fEt[     dims.x * dims.y ];
    fft::complex64 const * const __restrict__ fEtz = & d_fEt[ 2 * dims.x * dims.y ];

    fft::complex64 const * const __restrict__ frho = & d_frho[                   0 ];

    const int iy   = blockIdx.x;  // Line
    const float ky = ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) * dk.y;

    const int stride = dims.x;
    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;
        const float β = ( k2 > 0 ) ? 1.f / k2 : 0;

        fEx[idx] = -fft::I * kx * frho[idx] * β + fEtx[idx];
		fEy[idx] = -fft::I * ky * frho[idx] * β + fEty[idx];
		fEz[idx] =                                fEtz[idx] ;
    }
}

__global__
/**
 * @brief Update the magnetostatic (Darwin) magnetic field from the current
 *
 * B = i (k x J) / k^2  (with k_z = 0)
 *
 * @param fB    [out] Fourier transform of magnetic field
 * @param fJ    Fourier transform of (centered) current density
 * @param dims  Global grid dimensions
 * @param dk    k-space cell size
 */
void update_fB_darwin(
    fft::complex64 * const __restrict__ d_fB,
    fft::complex64 const * const __restrict__ d_fJ,
    uint2 const dims, float2 const dk )
{
    fft::complex64 * const __restrict__ fBx = & d_fB[                   0 ];
    fft::complex64 * const __restrict__ fBy = & d_fB[     dims.x * dims.y ];
    fft::complex64 * const __restrict__ fBz = & d_fB[ 2 * dims.x * dims.y ];

    fft::complex64 const * const __restrict__ fJx = & d_fJ[                   0 ];
    fft::complex64 const * const __restrict__ fJy = & d_fJ[     dims.x * dims.y ];
    fft::complex64 const * const __restrict__ fJz = & d_fJ[ 2 * dims.x * dims.y ];

    const int iy   = blockIdx.x;
    const float ky = ((iy < int(dims.y/2)) ? iy : (iy - int(dims.y)) ) * dk.y;

    const int stride = dims.x;
    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;
        const float β  = ( k2 > 0 ) ? 1.f / k2 : 0;

        fBx[idx] = fft::I * (  ky * fJz[idx]                  ) * β;
        fBy[idx] = fft::I * ( -kx * fJz[idx]                  ) * β;
        fBz[idx] = fft::I * (  kx * fJy[idx] - ky * fJx[idx]  ) * β;
    }
}

__global__
/**
 * @brief Update the transverse (Darwin) electric field via the shifted Helmholtz
 *        equation
 *
 * Computes:
 *      dJ/dt = A - i k.M
 * Transverse-projects it, and solves:
 *      E_T = -1/(k^2 + wp2) [ dJ_T/dt - wp2 E_T_old ] 
 * 
 * @param fEt    [in,out] Fourier transform of transverse E-field
 * @param fA     Fourier transform of acceleration density
 * @param fM1    Fourier transform of velocity flux (diagonal)
 * @param fM2    Fourier transform of velocity flux (off-diagonal)
 * @param dims   Global grid dimensions
 * @param dk     k-space cell size
 * @param wp2    Reference background plasma frequency squared
 * @param d_res  [out] Accumulated squared change in E_T
 * @param d_norm [out] Accumulated squared magnitude of E_T (new iterate)
 */
void update_fEt_darwin(
    fft::complex64 * const __restrict__ d_fEt,
    fft::complex64 const * const __restrict__ d_fA,
    fft::complex64 const * const __restrict__ d_fM1,
    fft::complex64 const * const __restrict__ d_fM2,
    uint2 const dims, float2 const dk, float const wp2,
    double * const __restrict__ d_res, double * const __restrict__ d_norm )
{
    fft::complex64 * const __restrict__ fEtx = & d_fEt[                   0 ];
    fft::complex64 * const __restrict__ fEty = & d_fEt[     dims.x * dims.y ];
    fft::complex64 * const __restrict__ fEtz = & d_fEt[ 2 * dims.x * dims.y ];

    fft::complex64 const * const __restrict__ fAx = & d_fA[                   0 ];
    fft::complex64 const * const __restrict__ fAy = & d_fA[     dims.x * dims.y ];
    fft::complex64 const * const __restrict__ fAz = & d_fA[ 2 * dims.x * dims.y ];

    // M1 = { Mxx, Myy, Mzz }; Mzz is not needed (k_z = 0 always)
    fft::complex64 const * const __restrict__ fMxx = & d_fM1[                   0 ];
    fft::complex64 const * const __restrict__ fMyy = & d_fM1[     dims.x * dims.y ];
    // M2 = { Mxy, Mxz, Myz }
    fft::complex64 const * const __restrict__ fMxy = & d_fM2[                   0 ];
    fft::complex64 const * const __restrict__ fMxz = & d_fM2[     dims.x * dims.y ];
    fft::complex64 const * const __restrict__ fMyz = & d_fM2[ 2 * dims.x * dims.y ];

    const int iy   = blockIdx.x;
    const float ky = ((iy < int(dims.y/2)) ? iy : (iy - int(dims.y)) ) * dk.y;

    const int stride = dims.x;

    double res = 0;
    double nrm = 0;

    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;

        // dJ/dt = A - i k.M  (with k_z = 0)
        const fft::complex64 kMx = kx * fMxx[idx] + ky * fMxy[idx];
        const fft::complex64 kMy = kx * fMxy[idx] + ky * fMyy[idx];
        const fft::complex64 kMz = kx * fMxz[idx] + ky * fMyz[idx];

        const fft::complex64 djx = fAx[idx] - fft::I * kMx;
        const fft::complex64 djy = fAy[idx] - fft::I * kMy;
        const fft::complex64 djz = fAz[idx] - fft::I * kMz;

        // Transverse projection of dJ/dt
        const fft::complex64 kdj  = kx * djx + ky * djy;
        const fft::complex64 djtx = ( k2 > 0 ) ? djx - kx * kdj / k2 : fft::complex64(0);
        const fft::complex64 djty = ( k2 > 0 ) ? djy - ky * kdj / k2 : fft::complex64(0);
        const fft::complex64 djtz = djz;

        const fft::complex64 Etx_old = fEtx[idx];
        const fft::complex64 Ety_old = fEty[idx];
        const fft::complex64 Etz_old = fEtz[idx];

        // Shifted Helmholtz solve; k=0 mode is left at zero
        const float β = ( k2 > 0 ) ? 1.f / ( k2 + wp2 ) : 0;

        const fft::complex64 Etx_new = -( djtx - wp2 * Etx_old ) * β;
        const fft::complex64 Ety_new = -( djty - wp2 * Ety_old ) * β;
        const fft::complex64 Etz_new = -( djtz - wp2 * Etz_old ) * β;

        fEtx[idx] = Etx_new;
        fEty[idx] = Ety_new;
        fEtz[idx] = Etz_new;

        const fft::complex64 dEx = Etx_new - Etx_old;
        const fft::complex64 dEy = Ety_new - Ety_old;
        const fft::complex64 dEz = Etz_new - Etz_old;

        // Residuals are useful to check convergence
        res += double( dEx.x * dEx.x + dEx.y * dEx.y )
             + double( dEy.x * dEy.x + dEy.y * dEy.y )
             + double( dEz.x * dEz.x + dEz.y * dEz.y );

        nrm += double( Etx_new.x * Etx_new.x + Etx_new.y * Etx_new.y )
             + double( Ety_new.x * Ety_new.x + Ety_new.y * Ety_new.y )
             + double( Etz_new.x * Etz_new.x + Etz_new.y * Etz_new.y );
    }

    res = warp::reduce_add( res );
    nrm = warp::reduce_add( nrm );
    if ( warp::thread_rank() == 0 ) {
        device::atomic_fetch_add( d_res,  res );
        device::atomic_fetch_add( d_norm, nrm );
    }
}

}

/**
 * @brief Advance EM fields 1 time step (no current or charge)
 * 
 */
void EMF::advance() {

    // Advance transverse fields
    kernel::advance_psatd_nocurr <<< fEt -> dims.y, 256 >>> ( 
        reinterpret_cast<fft::complex64 *>( fEt -> d_buffer ), 
        reinterpret_cast<fft::complex64 *>( fB -> d_buffer ), 
        fEt -> dims, fft::dk( box ), dt );

    // Transform to real fields
    fft_backward -> transform( *fEt, *E );
    fft_backward -> transform( *fB, *B );

    // Update guard cell values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Advance internal iteration number
    iter += 1;
}

/**
 * @brief Advance EM fields 1 time step including current
 * 
 * @param current   Electric current
 * @param charge    Electric charge
 */
void EMF::advance( Current & current, Charge & charge ) {

    // Advance transverse fields
    kernel::advance_psatd <<< fEt -> dims.y, 256 >>> ( 
        reinterpret_cast<fft::complex64 *>( fEt -> d_buffer ),
        reinterpret_cast<fft::complex64 *>( fB -> d_buffer ),
        reinterpret_cast<fft::complex64 *>( current.fJ -> d_buffer ),
        fEt -> dims, fft::dk( box ), dt
    );

    // Update total E-field
    kernel::update_fE<<< fEt -> dims.y, 256 >>> ( 
        reinterpret_cast<fft::complex64 *>( fE -> d_buffer ),
        reinterpret_cast<fft::complex64 *>( fEt -> d_buffer ),
        reinterpret_cast<fft::complex64 *>( charge.frho -> d_buffer ),
        fE -> dims, fft::dk( box )
    );

    // Transform to real fields
    fft_backward -> transform( *fE, *E );
    fft_backward -> transform( *fB, *B );

    // Update guard cell values
    E -> copy_to_gc();
    B -> copy_to_gc();

    // Advance internal iteration number
    iter += 1;
}

/**
 * @brief Solve the longitudinal (Poisson) electric field
 *
 * @param charge    Charge object with frho already populated
 */
void EMF::poisson_solver( Charge & charge ) {

    // fE = E_L(rho) + fEt. With fEt = 0 (the t=0 seed) this is a pure Poisson
    // solve; during the Darwin iteration fEt carries the transverse field.
    kernel::update_fE <<< fE -> dims.y, 256 >>> (
        reinterpret_cast<fft::complex64 *>( fE  -> d_buffer ),
        reinterpret_cast<fft::complex64 *>( fEt -> d_buffer ),
        reinterpret_cast<fft::complex64 *>( charge.frho -> d_buffer ),
        fE -> dims, fft::dk( box )
    );

    // Transform to real-space E and update guard cells
    fft_backward -> transform( *fE, *E );
    E -> copy_to_gc();
}

/**
 * @brief Solve the magnetostatic (Darwin) B
 *
 * @param fJ    k-space current
 */
void EMF::darwin_solver_B( basic_grid3<std::complex<float>> & fJ )
{
    kernel::update_fB_darwin <<< fB -> dims.y, 256 >>> (
        reinterpret_cast<fft::complex64 *>( fB -> d_buffer ),
        reinterpret_cast<fft::complex64 *>( fJ.d_buffer ),
        fB -> dims, fft::dk( box )
    );

    // Transform B to real space so the next tentative push sees the new value
    fft_backward -> transform( *fB, *B );
    B -> copy_to_gc();
}

/**
 * @brief Solve the Darwin (magnetostatic / radiation-free) fields
 *
 * @param fJ        k-space centered current
 * @param fA        k-space acceleration density
 * @param fM1       k-space velocity flux (diagonal)
 * @param fM2       k-space velocity flux (off-diagonal)
 * @param charge    Charge object (for the fixed longitudinal field)
 * @param wp2       Reference background plasma frequency squared
 * @return double   Relative L2 change of E_T ( |dEt| / |Et| )
 */
double EMF::darwin_solver(
    basic_grid3<std::complex<float>> & fJ,
    basic_grid3<std::complex<float>> & fA,
    basic_grid3<std::complex<float>> & fM1,
    basic_grid3<std::complex<float>> & fM2,
    Charge & charge,
    float const wp2 )
{
    const float2 dk = fft::dk( box );

    // Magnetostatic B from the centered current
    darwin_solver_B( fJ );

    // Accumulator for the squared change in E_T (acc[0]) and the squared magnitude of 
    // E_T (acc[1]).
    double * d_acc = device::malloc<double>( 2 );
    device::zero( d_acc, 2 );

    // Shifted Helmholtz solve for the transverse E. 
    kernel::update_fEt_darwin <<< fEt -> dims.y, 256 >>> (
        reinterpret_cast<fft::complex64 *>( fEt -> d_buffer ),
        reinterpret_cast<fft::complex64 *>( fA.d_buffer ),
        reinterpret_cast<fft::complex64 *>( fM1.d_buffer ),
        reinterpret_cast<fft::complex64 *>( fM2.d_buffer ),
        fEt -> dims, dk, wp2, &d_acc[0], &d_acc[1]
    );

    // Reconstruct the total E = E_L(rho) + E_T (and transform to real space)
    poisson_solver( charge );

    double acc[2];
    device::memcpy_tohost( acc, d_acc, 2 );
    device::free( d_acc );

    // Relative L2 change of the transverse field (falls back to absolute if
    // E_T is still identically zero, e.g. the first pass with no current)
    return ( acc[1] > 0 ) ? std::sqrt( acc[0] / acc[1] ) : std::sqrt( acc[0] );
}

/**
 * @brief Save EMF data to diagnostic file
 *
 * @param field     Field to save (0:E, 1:B)
 * @param fc        Field component to save (0, 1 or 2)
 */
void EMF::save( const emf::field field, fcomp::cart const fc ) {

    std::string vfname;  // Dataset name
    std::string vflabel; // Dataset label (for plots)

    vec3grid<float3> * f = nullptr;
    basic_grid3<std::complex<float>> * cf = nullptr;

    switch (field ) {
        case emf::e :
            f = E;
            vfname = "E";
            vflabel = "E_";
            break;
        case emf::b :
            f = B;
            vfname = "B";
            vflabel = "B_";
            break;
        case emf::fe :
            cf = fE;
            vfname = "fE";
            vflabel = "\\mathcal{F} E_";
            break;
        case emf::fet :
            cf = fEt;
            vfname = "fEt";
            vflabel = "\\mathcal{F} E^\\perp_";
            break;
        case emf::fb :
            cf = fB;
            vfname = "fB";
            vflabel = "\\mathcal{F} B_";
            break;
        default:
            ABORT("Invalid field type selected, aborting");
    }

    switch ( fc ) {
        case( fcomp::x ) :
            vfname  += 'x';
            vflabel += 'x';
            break;
        case( fcomp::y ) :
            vfname  += 'y';
            vflabel += 'y';
            break;
        case( fcomp::z ) :
            vfname  += 'z';
            vflabel += 'z';
            break;
        default:
            ABORT("Invalid field component (fc) selected, aborting");
    }

    zdf::iteration iteration = {
        .n = iter,
        .t = iter * dt,
        .time_units = (char *) "1/\\omega_n"
    };


    zdf::grid_info info = {
        .name = (char *) vfname.c_str(),
        .ndims = 2,
        .label = (char *) vflabel.c_str(),
        .units = (char *) "m_e c \\omega_n e^{-1}"
    };

    zdf::grid_axis axis[2];

    if ( field == emf::e || field == emf::b ) {
        axis[0] = (zdf::grid_axis) {
            .name = (char *) "x",
            .min = 0.0,
            .max = box.x,
            .label = (char *) "x",
            .units = (char *) "c/\\omega_n"
        };

        axis[1] = (zdf::grid_axis) {
            .name = (char *) "y",
            .min = 0.0,
            .max = box.y,
            .label = (char *) "y",
            .units = (char *) "c/\\omega_n"
        };

        info.axis = axis;
        info.count[0] = E -> dims.x;
        info.count[1] = E -> dims.y;

        f -> save( fc, info, iteration, "EMF" );

    } else {
        float2 dk = fft::dk( box );

        axis[0] = (zdf::grid_axis) {
            .name = (char *) "kx",
            .min = 0.0,
            .max = (fEt -> dims.x - 1) * dk.x,
            .label = (char *) "k_x"
        };

        axis[1] = (zdf::grid_axis) {
            .name = (char *) "ky",
            .min =  - dk.y * ( fEt -> dims.y / 2 ),
            .max =    dk.y * ( fEt -> dims.y / 2 - 1 ),
            .label = (char *) "k_y"
        };

        info.axis = axis;
        info.count[0] = fEt -> dims.x;
        info.count[1] = fEt -> dims.y;

        cf -> save( fc, info, iteration, "EMF" );
    }
}

namespace kernel {

__global__
void get_energy( 
    float3 * const __restrict__ E_buffer,
    float3 * const __restrict__ B_buffer,
    uint2 const ntiles, uint2 const nx, uint2 const ext_nx, unsigned int const offset, 
    double * const __restrict__ d_energy ) {

    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * ntiles.x + tile_idx.x;
    const int    tile_vol = roundup4( ext_nx.x * ext_nx.y );
    const size_t tile_off = tile_id * tile_vol;

    float3 * const __restrict__ E_local = & E_buffer[ tile_off + offset ];
    float3 * const __restrict__ B_local = & B_buffer[ tile_off + offset ];

    const int ystride = ext_nx.x;

    double3 ene_E = double3{0};
    double3 ene_B = double3{0};

    for( int idx = block_thread_rank(); idx < nx.y * nx.x; idx += block_num_threads() ) {
        int const i = idx % nx.x;
        int const j = idx / nx.x;

        float3 const efld = E_local[ j * ystride + i ];
        float3 const bfld = B_local[ j * ystride + i ];

        ene_E.x += efld.x * efld.x;
        ene_E.y += efld.y * efld.y;
        ene_E.z += efld.z * efld.z;

        ene_B.x += bfld.x * bfld.x;
        ene_B.y += bfld.y * bfld.y;
        ene_B.z += bfld.z * bfld.z;
    }

    // Add up energy from all warps
    ene_E.x = warp::reduce_add( ene_E.x );
    ene_E.y = warp::reduce_add( ene_E.y );
    ene_E.z = warp::reduce_add( ene_E.z );

    ene_B.x = warp::reduce_add( ene_B.x );
    ene_B.y = warp::reduce_add( ene_B.y );
    ene_B.z = warp::reduce_add( ene_B.z );

    if ( warp::thread_rank() == 0 ) {
        device::atomic_fetch_add( &(d_energy[0]), ene_E.x );
        device::atomic_fetch_add( &(d_energy[1]), ene_E.y );
        device::atomic_fetch_add( &(d_energy[2]), ene_E.z );

        device::atomic_fetch_add( &(d_energy[3]), ene_B.x );
        device::atomic_fetch_add( &(d_energy[4]), ene_B.y );
        device::atomic_fetch_add( &(d_energy[5]), ene_B.z );
    }
}

}


/**
 * @brief Get total field energy per field component
 * 
 * @warning This function will always recalculate the energy each time it is
 *          called.
 * 
 * @param ene_E     Total E-field energy (per component)
 * @param ene_B     Total B-field energy (per component)
 */
void EMF::get_energy( double3 & ene_E, double3 & ene_B ) {

    // Zero energy values
    device::zero( d_energy, 6 );

    // Add up energy from all cells
    dim3 grid( E->ntiles.x, E->ntiles.y );
    dim3 block( 1024 );
    kernel::get_energy <<< grid, block >>> ( 
        E->d_buffer, B->d_buffer,
        E->ntiles, E->nx, E->ext_nx, E->offset,
        d_energy
    );

    // Copy results to host and normalize
    double h_energy[6];
    device::memcpy_tohost( h_energy, d_energy, 6 );

    ene_E.x = 0.5 * dx.x * dx.y * h_energy[0];
    ene_E.y = 0.5 * dx.x * dx.y * h_energy[1];
    ene_E.z = 0.5 * dx.x * dx.y * h_energy[2];

    ene_B.x = 0.5 * dx.x * dx.y * h_energy[3];
    ene_B.y = 0.5 * dx.x * dx.y * h_energy[4];
    ene_B.z = 0.5 * dx.x * dx.y * h_energy[5];

}