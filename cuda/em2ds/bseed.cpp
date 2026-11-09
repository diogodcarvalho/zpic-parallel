/**
 * @file bseed.cpp
 * @brief Magnetic-silent start (see bseed.h)
 *
 * The long-wavelength transverse magnetic modes of a thermal plasma are overdamped:
 * the Landau-continued transverse dispersion D = w^2 - Pi(k,w) - k^2 (c = w_p = 1 code
 * units, Pi the filtered transverse response) has a root w = -i Gamma_k with
 *
 *      Gamma_k ~ k^2 / a,      a = pi P g_perp(0) / kappa,     kappa = k v_th,
 *
 * where P = F S^2 w_p^2 and g_perp the transverse MJ marginal in w = v_par / v_th
 * units. A quiet start (J = 0, so B = 0) has to grow them at this ~k^3 rate. Expanding
 * Pi to second order in Gamma / kappa (the MJ identity g_perp' = -w g_par makes the
 * quadratic coefficient exactly P), the pole and its share of the stationary B
 * variance (FDT residue) are
 *
 *      b Gamma^2 + a Gamma - k^2 = 0,  b = P / kappa^2 - 1,   A_inf = k^2 / ( k^2 + b Gamma^2 )
 *
 * so the stationary slow-mode variance is F A_inf per polarization, in units of the
 * per-mode thermal level theta n0 |m_q| / ppc (unnormalized FFT, per grid cell).
 *
 * The seed is an independent Gaussian B_0 (both polarizations) carried by a smooth
 * momentum kick. A kick du is an impulsive field, so choosing the deposited, filtered
 * current to balance B_0 ( F J = i k x B_0, E_T = 0 ) cancels the light-wave pole to
 * second order. The kicked state then puts A_s = A_inf ( 1 + ( k^2 + Gamma^2 ) / Pi_inf )
 * of B_0 into the slow mode ( Pi_inf = P m0, m0 = Omega^2 / w_p^2 ), and the start is
 * stationary for sigma^2 A_s^2 = F A_inf, i.e.
 *
 *      sigma^2 = F / ( A_inf ( 1 + ( k^2 + Gamma^2 ) / Pi_inf )^2 )
 *
 * The current of a smooth kick is rho0 lambda S^2 du, with lambda = < (1 - v_i^2)/gamma >
 * measured from the particles (one S from interpolating du, one from the deposit).
 */

#include "bseed.h"
#include "random.h"

#include <algorithm>
#include <cmath>
#include <vector>
#include <iostream>

namespace {

/**
 * @brief Constants of the transverse marginal of an isotropic 3D Maxwell-Juttner
 *        distribution, in w = v_par / v_th units
 *
 * With the rapidity eta along k ( v = tanh eta, gamma = cosh eta ) the longitudinal
 * marginal is
 *
 *      g_par(v) dv ~ ( 1 + 2 theta / gamma + 2 theta^2 / gamma^2 ) exp( -(gamma-1)/theta ) d eta
 *
 * and, from the closed forms of both marginals,
 *
 *      g_perp(0) = sqrt( theta ) ( 1 + theta ) / Z,    Z the full-line integral of the above
 *      m0 = int g_perp dw = < v_par^2 > / theta  ( = Omega^2 / w_p^2 )
 *
 * @param theta     MJ temperature
 * @param g0        [out] g_perp(0)
 * @param m0        [out] int g_perp dw
 */
void mj_perp( double const theta, double & g0, double & m0 ) {
    // exp( -(gamma-1)/theta ) < exp(-40) beyond eta_max
    const int n = 4000;
    const double eta_max = std::acosh( 1 + 40 * theta );
    const double deta = eta_max / n;

    double Z = 0, v2 = 0;
    for( int j = 0; j < n; j++ ) {
        const double eta = ( j + 0.5 ) * deta;
        const double g   = std::cosh( eta );
        const double v   = std::tanh( eta );
        const double w   = ( 1 + 2 * theta / g + 2 * theta * theta / ( g * g ) ) *
                           std::exp( -( g - 1 ) / theta );
        Z  += w;
        v2 += w * v * v;
    }

    // Even integrands, eta >= 0 sampled only
    g0 = std::sqrt( theta ) * ( 1 + theta ) / ( 2 * Z * deta );
    m0 = v2 / Z / theta;
}

/**
 * @brief Relaxation rate Gamma_k and slow share A_inf of one mode (see file header)
 *
 * @param k2        |k|^2
 * @param P         F S^2 w_p^2
 * @param theta     MJ temperature
 * @param g0        g_perp(0)
 * @param Gamma     [out] relaxation rate
 * @param A         [out] A_inf
 */
__host__ __device__ inline
void slow_mode( float const k2, float const P, float const theta, float const g0,
    float & Gamma, float & A ) {
    const float kappa2 = k2 * theta;
    const float a = 3.14159265f * P * g0 / sqrtf( kappa2 );
    const float b = P / kappa2 - 1;
    Gamma = 2 * k2 / ( a + sqrtf( a * a + 4 * b * k2 ) );
    A     = k2 / ( k2 + b * Gamma * Gamma );
}

}

namespace kernel {

__global__
/**
 * @brief k-space seed field B_0 and momentum kick of the magnetic-silent start
 *
 * @param d_fB      [out] seed magnetic field (k-space, all modes written)
 * @param d_fdu     [out] momentum kick field (k-space)
 * @param d_fF      Source filter transfer function (real part used)
 * @param dims      Global (half-)spectrum dimensions
 * @param dk        k-space cell size
 * @param dx        Real-space cell size (for the shape factor)
 * @param kmax      Largest seeded |k|
 * @param wp2       Plasma frequency squared
 * @param theta     MJ temperature
 * @param m0        Omega^2 / w_p^2 = int g_perp dw
 * @param g0        g_perp(0)
 * @param amp0      Per-mode thermal level theta n0 |m_q| / ppc
 * @param ncells    Number of grid cells ( <|xi|^2> of the unnormalized FFT )
 * @param rho0      Species charge density
 * @param lambda    Kick response < (1 - v^2)/gamma >, ( in-plane, z )
 * @param rnd_seed  RNG seed
 */
void bseed_fields(
    fft::complex64 * const __restrict__ d_fB,
    fft::complex64 * const __restrict__ d_fdu,
    fft::complex64 const * const __restrict__ d_fF,
    uint2 const dims, float2 const dk, float2 const dx,
    float const kmax, float const wp2, float const theta, float const m0, float const g0,
    float const amp0, float const ncells, float const rho0, float2 const lambda,
    uint2 const rnd_seed )
{
    const int vol = dims.x * dims.y;

    fft::complex64 * const __restrict__ fBx  = & d_fB [       0 ];
    fft::complex64 * const __restrict__ fBy  = & d_fB [     vol ];
    fft::complex64 * const __restrict__ fBz  = & d_fB [ 2 * vol ];
    fft::complex64 * const __restrict__ fdux = & d_fdu[       0 ];
    fft::complex64 * const __restrict__ fduy = & d_fdu[     vol ];
    fft::complex64 * const __restrict__ fduz = & d_fdu[ 2 * vol ];

    const int iy   = blockIdx.x;
    const float ky = ((iy < int(dims.y/2)) ? iy : (iy - int(dims.y)) ) * dk.y;

    const int stride = dims.x;
    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;

        fft::complex64 Bx = 0, By = 0, Bz = 0;
        fft::complex64 dux = 0, duy = 0, duz = 0;

        // Nyquist row / column are never seeded (kmax is well below them)
        if ( k2 > 0 && k2 < kmax * kmax && iy != int(dims.y/2) && ix != int(dims.x) - 1 ) {

            // The kx = 0 column holds both ky and -ky: build the ky > 0 mode and
            // conjugate it for ky < 0, so the field stays Hermitian
            const bool flip = ( ix == 0 && ky < 0 );
            const float kyr = flip ? -ky : ky;
            const int   iyr = flip ? int(dims.y) - iy : iy;

            // Quartic (order-4 B-spline) shape factor S(k) = [sinc(kx dx/2) sinc(ky dy/2)]^5
            const float ax = 0.5f * kx  * dx.x;
            const float ay = 0.5f * kyr * dx.y;
            const float sx = ( ax != 0.f ) ? sinf(ax) / ax : 1.f;
            const float sy = ( ay != 0.f ) ? sinf(ay) / ay : 1.f;
            const float sk = sx * sy;
            const float S  = sk*sk*sk*sk*sk;
            const float F  = d_fF[idx].x;

            if ( F > 0 ) {
                // Slow pole and stationary seed variance (see file header)
                const float P    = F * S * S * wp2;
                const float Pinf = P * m0;
                float Gamma, A;
                slow_mode( k2, P, theta, g0, Gamma, A );
                const float r     = 1 + ( k2 + Gamma * Gamma ) / Pinf;
                const float sigma = sqrtf( amp0 * F / ( A * r * r ) );

                // Two independent complex Gaussians, < |xi|^2 > = ncells
                uint2 state; double norm;
                zrandom::rand_init( iyr * stride + ix, rnd_seed, state, norm );
                const float s  = sqrtf( 0.5f * ncells ) * sigma;
                const float n1 = zrandom::rand_norm( state, norm );
                const float n2 = zrandom::rand_norm( state, norm );
                const float n3 = zrandom::rand_norm( state, norm );
                const float n4 = zrandom::rand_norm( state, norm );
                const fft::complex64 xi1( s * n1, s * n2 );
                const fft::complex64 xi2( s * n3, s * n4 );

                // B_z (in-plane J_T) and in-plane B = xi2 ( z x khat ) (J_z)
                const float rk = rsqrtf( k2 );
                Bz =  xi1;
                Bx = -( kyr * rk ) * xi2;
                By =  ( kx  * rk ) * xi2;

                // Kick whose filtered current balances B_0: F J = i k x B_0
                const fft::complex64 Jx = fft::I * (  kyr * Bz              ) / F;
                const fft::complex64 Jy = fft::I * ( -kx  * Bz              ) / F;
                const fft::complex64 Jz = fft::I * (  kx  * By - kyr * Bx   ) / F;

                const float cp = 1.f / ( rho0 * lambda.x * S * S );
                const float cz = 1.f / ( rho0 * lambda.y * S * S );
                dux = cp * Jx;
                duy = cp * Jy;
                duz = cz * Jz;

                if ( flip ) {
                    Bx = conj( Bx ); By = conj( By ); Bz = conj( Bz );
                    dux = conj( dux ); duy = conj( duy ); duz = conj( duz );
                }
            }
        }

        fBx[idx]  = Bx;
        fBy[idx]  = By;
        fBz[idx]  = Bz;
        fdux[idx] = dux;
        fduy[idx] = duy;
        fduz[idx] = duz;
    }
}

}

void bseed::kick( Simulation & sim, Species & species, float const theta, float const n0,
    uint2 const ppc, float kmax, unsigned int const seed ) {

    // Kick response of the actual load, and the analytic MJ constants
    double3 lambda; double T;
    species.kick_response( lambda, T );

    double g0, m0;
    mj_perp( theta, g0, m0 );

    const double wp2    = n0 / std::fabs( species.m_q );
    const double Omega2 = wp2 * m0;
    if ( kmax <= 0 ) kmax = std::sqrt( Omega2 / 3 );
    if ( kmax * kmax > Omega2 )
        std::cout << "(*warning*) Magnetic-silent start: kmax = " << kmax << " exceeds Omega = "
                  << std::sqrt( Omega2 ) << ", where the kick no longer keeps the light waves silent\n";

    const double lambda_D = std::sqrt( theta / wp2 );

    std::cout << "(*info*) Magnetic-silent start: Omega^2/wp^2 = " << m0
              << " (measured < (1-v^2)/gamma > = " << lambda.x << ", " << lambda.y << ", " << lambda.z
              << "), g_perp(0) = " << g0
              << ", T = < u.v >/3 = " << T << " (theta = " << theta << ")\n";

    // Transfer function of the source filter (any type), from filtering a field of ones
    const uint2 dims  = sim.emf.B -> dims;
    const uint2 fdims = fft::fdims( dims );

    basic_grid<std::complex<float>> fF( fdims );
    std::vector<std::complex<float>> ones( fdims.x * fdims.y, std::complex<float>( 1, 0 ) );
    fF.copy_todevice( ones.data() );
    sim.current.filter -> apply( fF );

    // Seed B (written straight into the EMF) and the k-space kick
    const float2 dx = make_float2( sim.box.x / dims.x, sim.box.y / dims.y );
    const float amp0 = theta * n0 * std::fabs( species.m_q ) / ( ppc.x * ppc.y );
    const float rho0 = std::copysign( n0, species.m_q );
    const float2 lam = make_float2( 0.5 * ( lambda.x + lambda.y ), lambda.z );
    const uint2 rnd_seed{ 24680 + seed, 13579 };

    basic_grid3<std::complex<float>> fdu( fdims );

    kernel::bseed_fields <<< fdims.y, 256 >>> (
        reinterpret_cast<fft::complex64 *>( sim.emf.fB -> d_buffer ),
        reinterpret_cast<fft::complex64 *>( fdu.d_buffer ),
        reinterpret_cast<fft::complex64 *>( fF.d_buffer ),
        fdims, fft::dk( sim.box ), dx, kmax, wp2, theta, m0, g0,
        amp0, float( dims.x ) * dims.y, rho0, lam, rnd_seed
    );

    sim.emf.fft_backward -> transform( *sim.emf.fB, *sim.emf.B );
    sim.emf.B -> copy_to_gc();

    auto gc = sim.emf.B -> gc;
    vec3grid<float3> du( sim.ntiles, sim.nx, gc );
    fft::plan fft_backward( dims, fft::type::c2r_v3 );
    fft_backward.transform( fdu, du );
    du.copy_to_gc();

    // Kick the particles (mirror pairs get identical kicks)
    const float max = species.kick( du );

    // Report: seeded modes, relaxation times at the box scale and at kmax (S = F = 1),
    // and the seeded magnetic energy
    const float2 dk = fft::dk( sim.box );
    unsigned long nmodes = 0;
    for( int iy = -int(dims.y/2); iy < int(dims.y/2); iy++ )
        for( int ix = -int(dims.x/2); ix < int(dims.x/2); ix++ ) {
            const float k2 = ix * ix * dk.x * dk.x + iy * iy * dk.y * dk.y;
            if ( k2 > 0 && k2 < kmax * kmax ) nmodes++;
        }

    const float kbox = std::min( dk.x, dk.y );
    float G_box, G_max, A;
    slow_mode( kbox * kbox, wp2, theta, g0, G_box, A );
    slow_mode( kmax * kmax, wp2, theta, g0, G_max, A );

    double3 ene_E, ene_B;
    sim.emf.get_energy( ene_E, ene_B );

    std::cout << "(*info*) Magnetic-silent start: kmax = " << kmax << " (k lambda_D = " << kmax * lambda_D
              << "), " << nmodes / 2 << " independent modes seeded, 1/(2 Gamma) = "
              << 0.5 / G_box << " (box mode) to " << 0.5 / G_max << " (kmax)\n";
    std::cout << "(*info*) Magnetic-silent start: max |du| = " << max << " ("
              << max / std::sqrt( 3 * T ) << " of sqrt(3T)), seeded B energy = "
              << ene_B.x + ene_B.y + ene_B.z << '\n';
}
