/**
 * @file silent.cpp
 * @brief Langmuir-silent start (see silent.h)
 *
 * For an undamped Langmuir mode ( v_phi > c ) at +/- w_L(k), the linear (Klimontovich)
 * initial value problem leaves a free oscillation of amplitude proportional to
 *
 *      sum_p exp( -i k.x_p ) / ( +/- w_L - k.v_p )
 *
 * Co-located ( u, -u ) pairs cancel the part odd in v exactly, leaving
 *
 *      sum_p exp( -i k.x_p ) W_p,      W_p = w_L / ( w_L^2 - (k.v_p)^2 )
 *
 * On the quiet lattice only the site-to-site fluctuation of W survives. Displacing the
 * pairs so that the (bare) particle density changes by
 *
 *      dN_k = - sum_p exp( -i k.x_p ) ( W_p - <W> ) / <W>
 *
 * cancels it to first order. With zeta = w_L / k and v = khat.v_p, k W = f(v) =
 * zeta / ( zeta^2 - v^2 ), which is least squares fitted (weighted by the MJ marginal
 * g(v)) as c0 + c1 v^2 + c2 v^4. The c0 term vanishes on the lattice, so only the
 * deposited 2nd and 4th order in-plane velocity moments are needed:
 *
 *      dN_k = - ( r1 M2_par + r2 M4_par ),      r_i = c_i / <f>
 *
 * For a Maxwell-Juttner plasma the dispersion relation is
 *
 *      eps = 1 - a < v^2 / ( zeta^2 - v^2 ) > = 0,     a = F S^2 / ( k lambda_D )^2
 *
 * (F the source filter, S the quartic shape factor, lambda_D^2 = theta / w_p^2), which
 * has an undamped ( zeta > 1 ) root only for a > a_c = 1 / < v^2 / ( 1 - v^2 ) >. So r1
 * and r2 depend on k only through a, and are tabulated in a. Modes with a <= a_c are
 * Landau damped and left untouched.
 */

#include "silent.h"

#include <cmath>
#include <vector>
#include <iostream>

namespace {

/// @brief Number of table entries
constexpr int ntab = 512;

/// @brief Table range, a_c to a_c * tab_range (log spaced, clamped beyond)
constexpr double tab_range = 1.0e8;

/**
 * @brief Quadrature for averages over the marginal distribution g(v) of one velocity
 *        component of an isotropic 3D Maxwell-Juttner distribution
 *
 * With the rapidity eta along that component ( v = tanh eta, gamma = cosh eta ),
 *
 *      g(v) dv ~ ( 1 + 2 theta / gamma + 2 theta^2 / gamma^2 ) exp( -(gamma-1)/theta ) d eta
 *
 * g is even, so only eta >= 0 is sampled. Weights are normalized to 1.
 */
class MJMarginal {
    std::vector<double> v, w;

    public:

    MJMarginal( double const theta, int const n = 4000 ) : v( n ), w( n ) {
        // exp( -(gamma-1)/theta ) < exp(-40) beyond eta_max
        const double eta_max = std::acosh( 1 + 40 * theta );
        double norm = 0;
        for( int j = 0; j < n; j++ ) {
            const double eta = ( j + 0.5 ) * eta_max / n;
            const double g   = std::cosh( eta );
            v[j] = std::tanh( eta );
            w[j] = ( 1 + 2 * theta / g + 2 * theta * theta / ( g * g ) ) *
                   std::exp( -( g - 1 ) / theta );
            norm += w[j];
        }
        for( auto & x : w ) x /= norm;
    }

    /// @brief < h(v) >
    template< class H > double avg( H const & h ) const {
        double s = 0;
        for( size_t j = 0; j < v.size(); j++ ) s += w[j] * h( v[j] );
        return s;
    }
};

/**
 * @brief Coefficients r1, r2 for one undamped mode
 *
 * Works with e = 1 / zeta^2 in ( 0, 1 ] so that the fit stays well conditioned when
 * zeta >> 1:  zeta f = 1 / ( 1 - e v^2 ) = 1 + e v^2 + e^2 q(v),  q = v^4 / ( 1 - e v^2 ).
 * The least squares projection is linear, so only q needs to be fitted.
 *
 * @param g     MJ marginal
 * @param m1    < v^2 >
 * @param e     1 / zeta^2
 * @param r1    [out] c1 / <f>
 * @param r2    [out] c2 / <f>
 */
void coefficients( MJMarginal const & g, double const m1, double const e,
    double & r1, double & r2 ) {

    auto q = [e]( double v ) { const double v2 = v * v; return v2 * v2 / ( 1 - e * v2 ); };

    // Fit q on { 1, t, t^2 }, t = v^2 / m1 (scaled for conditioning)
    double mu[5], b[3];
    for( int n = 0; n < 5; n++ )
        mu[n] = g.avg( [=]( double v ) { return std::pow( v * v / m1, n ); } );
    for( int n = 0; n < 3; n++ )
        b[n] = g.avg( [=]( double v ) { return q( v ) * std::pow( v * v / m1, n ); } );

    auto det3 = []( double const m[3][3] ) {
        return m[0][0] * ( m[1][1] * m[2][2] - m[1][2] * m[2][1] )
             - m[0][1] * ( m[1][0] * m[2][2] - m[1][2] * m[2][0] )
             + m[0][2] * ( m[1][0] * m[2][1] - m[1][1] * m[2][0] );
    };

    double G[3][3];
    for( int i = 0; i < 3; i++ ) for( int j = 0; j < 3; j++ ) G[i][j] = mu[i+j];
    const double detG = det3( G );

    double d[3];
    for( int k = 0; k < 3; k++ ) {
        double Gk[3][3];
        for( int i = 0; i < 3; i++ ) for( int j = 0; j < 3; j++ )
            Gk[i][j] = ( j == k ) ? b[i] : G[i][j];
        d[k] = det3( Gk ) / detG;
    }

    // zeta f ~ ( 1 + e^2 d0 ) + ( e + e^2 d1 / m1 ) v^2 + ( e^2 d2 / m1^2 ) v^4
    const double zf = 1 + e * m1 + e * e * b[0];
    r1 = ( e + e * e * d[1] / m1 ) / zf;
    r2 = ( e * e * d[2] / ( m1 * m1 ) ) / zf;
}

}

namespace kernel {

__global__
/**
 * @brief k-space displacement field of the Langmuir-silent start
 *
 * @param d_fxi     [out] displacement field (x, y; z = 0)
 * @param d_fM2     2nd order moments ( vx^2, vx vy, vy^2 )
 * @param d_fM4a    4th order moments ( vx^4, vx^3 vy, vx^2 vy^2 )
 * @param d_fM4b    4th order moments ( vx vy^3, vy^4, - )
 * @param d_fF      Source filter transfer function (real part used)
 * @param tab       ( r1 a, r2 a^2 ) table, log spaced in a from a_c
 * @param dla       Table spacing in log(a)
 * @param a_c       Smallest a with an undamped Langmuir mode
 * @param dims      Global (half-)spectrum dimensions
 * @param dk        k-space cell size
 * @param dx        Real-space cell size (for the shape factor)
 * @param lambda_D2 Debye length squared
 * @param n0        Uniform (number) density of the moment deposits
 */
void silent_xi(
    fft::complex64 * const __restrict__ d_fxi,
    fft::complex64 const * const __restrict__ d_fM2,
    fft::complex64 const * const __restrict__ d_fM4a,
    fft::complex64 const * const __restrict__ d_fM4b,
    fft::complex64 const * const __restrict__ d_fF,
    float2 const * const __restrict__ tab, float const dla, float const a_c,
    uint2 const dims, float2 const dk, float2 const dx,
    float const lambda_D2, float const n0 )
{
    const int vol = dims.x * dims.y;

    fft::complex64 * const __restrict__ fxix = & d_fxi[       0 ];
    fft::complex64 * const __restrict__ fxiy = & d_fxi[     vol ];
    fft::complex64 * const __restrict__ fxiz = & d_fxi[ 2 * vol ];

    fft::complex64 const * const __restrict__ fMxx   = & d_fM2 [       0 ];
    fft::complex64 const * const __restrict__ fMxy   = & d_fM2 [     vol ];
    fft::complex64 const * const __restrict__ fMyy   = & d_fM2 [ 2 * vol ];
    fft::complex64 const * const __restrict__ fMxxxx = & d_fM4a[       0 ];
    fft::complex64 const * const __restrict__ fMxxxy = & d_fM4a[     vol ];
    fft::complex64 const * const __restrict__ fMxxyy = & d_fM4a[ 2 * vol ];
    fft::complex64 const * const __restrict__ fMxyyy = & d_fM4b[       0 ];
    fft::complex64 const * const __restrict__ fMyyyy = & d_fM4b[     vol ];

    const int iy   = blockIdx.x;
    const float ky = ((iy < int(dims.y/2)) ? iy : (iy - int(dims.y)) ) * dk.y;

    const int stride = dims.x;
    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;

        // Quartic (order-4 B-spline) shape factor S(k) = [sinc(kx dx/2) sinc(ky dy/2)]^5
        const float ax = 0.5f * kx * dx.x;
        const float ay = 0.5f * ky * dx.y;
        const float sx = ( ax != 0.f ) ? sinf(ax) / ax : 1.f;
        const float sy = ( ay != 0.f ) ? sinf(ay) / ay : 1.f;
        const float sk = sx * sy;
        const float S  = sk*sk*sk*sk*sk;

        // Coupling of mode k: a = F S^2 / ( k lambda_D )^2
        const float a = ( k2 > 0 ) ? d_fF[idx].x * S * S / ( k2 * lambda_D2 ) : 0;

        fft::complex64 xix = 0, xiy = 0;

        if ( a > a_c ) {
            // Interpolate table in log(a), clamped beyond its range
            float t = logf( a / a_c ) / dla;
            if ( t > ntab - 1 ) t = ntab - 1;
            int i = (int) t;
            if ( i > ntab - 2 ) i = ntab - 2;
            const float w = t - i;
            const float r1 = ( ( 1 - w ) * tab[i].x + w * tab[i+1].x ) / a;
            const float r2 = ( ( 1 - w ) * tab[i].y + w * tab[i+1].y ) / ( a * a );

            // Velocity moments along khat
            const float rk = rsqrtf( k2 );
            const float hx = kx * rk;
            const float hy = ky * rk;

            const fft::complex64 M2par = ( hx * hx ) * fMxx[idx] + ( 2 * hx * hy ) * fMxy[idx] +
                                         ( hy * hy ) * fMyy[idx];
            const fft::complex64 M4par = ( hx * hx * hx * hx ) * fMxxxx[idx] +
                                         ( 4 * hx * hx * hx * hy ) * fMxxxy[idx] +
                                         ( 6 * hx * hx * hy * hy ) * fMxxyy[idx] +
                                         ( 4 * hx * hy * hy * hy ) * fMxyyy[idx] +
                                         ( hy * hy * hy * hy ) * fMyyyy[idx];

            // Bare density change, with the deposit shape factor removed
            const fft::complex64 dn = -( r1 * M2par + r2 * M4par ) / S;

            // Longitudinal displacement with dn = -i n0 k.xi, divided by S again to
            // compensate the quartic interpolation of xi at the particle positions
            const fft::complex64 c = fft::I * dn / ( n0 * k2 * S );
            xix = c * kx;
            xiy = c * ky;
        }

        fxix[idx] = xix;
        fxiy[idx] = xiy;
        fxiz[idx] = 0;
    }
}

}

void silent::displace( Simulation & sim, Species & species, float const theta,
    float const n0, uint2 const ppc ) {

    // Debye length ( exact for MJ: lambda_D^2 = theta / w_p^2 )
    const float wp2 = n0 / std::fabs( species.m_q );
    const float lambda_D2 = theta / wp2;

    // ( r1 a, r2 a^2 ) table vs a, both tend to constants for a >> a_c
    MJMarginal g( theta );
    const double m1  = g.avg( []( double v ) { return v * v; } );
    const double a_c = 1.0 / g.avg( []( double v ) { return v * v / ( 1 - v * v ); } );
    const double dla = std::log( tab_range ) / ( ntab - 1 );

    std::vector<float2> h_tab( ntab );
    for( int i = 0; i < ntab; i++ ) {
        const double a = a_c * std::exp( i * dla );

        // Undamped root: a e < v^2 / ( 1 - e v^2 ) > = 1, e = 1 / zeta^2 in ( 0, 1 ]
        double lo = 0, hi = 1;
        for( int it = 0; it < 200; it++ ) {
            const double e = 0.5 * ( lo + hi );
            const double H = e * g.avg( [e]( double v ) { return v * v / ( 1 - e * v * v ); } );
            if ( a * H < 1 ) lo = e; else hi = e;
        }

        double r1, r2;
        coefficients( g, m1, 0.5 * ( lo + hi ), r1, r2 );
        h_tab[i] = make_float2( r1 * a, r2 * a * a );
    }

    std::cout << "(*info*) Langmuir-silent start: a_c = " << a_c
              << " (undamped for k lambda_D < sqrt( F S^2 / a_c ), "
              << 1.0 / std::sqrt( a_c ) << " unfiltered)\n";

    float2 * d_tab = device::malloc<float2>( ntab );
    host::memcpy_todevice( d_tab, h_tab.data(), ntab );

    // Deposit the in-plane velocity moments at the lattice positions
    auto gc = sim.emf.E -> gc;
    vec3grid<float3> M2 ( sim.ntiles, sim.nx, gc );
    vec3grid<float3> M4a( sim.ntiles, sim.nx, gc );
    vec3grid<float3> M4b( sim.ntiles, sim.nx, gc );
    M2.zero();
    M4a.zero();
    M4b.zero();

    species.deposit_vmoments( M2, M4a, M4b );

    M2.add_from_gc();
    M4a.add_from_gc();
    M4b.add_from_gc();

    // FFT moments to k-space
    const uint2 dims  = M2.dims;
    const uint2 fdims = fft::fdims( dims );

    fft::plan fft_forward( dims, fft::type::r2c_v3 );
    basic_grid3<std::complex<float>> fM2 ( fdims );
    basic_grid3<std::complex<float>> fM4a( fdims );
    basic_grid3<std::complex<float>> fM4b( fdims );
    fft_forward.transform( M2,  fM2  );
    fft_forward.transform( M4a, fM4a );
    fft_forward.transform( M4b, fM4b );

    // Transfer function of the source filter (any type), from filtering a field of ones
    basic_grid<std::complex<float>> fF( fdims );
    std::vector<std::complex<float>> ones( fdims.x * fdims.y, std::complex<float>( 1, 0 ) );
    fF.copy_todevice( ones.data() );
    sim.charge.filter -> apply( fF );

    // Displacement field
    const float2 dx = make_float2( sim.box.x / dims.x, sim.box.y / dims.y );
    basic_grid3<std::complex<float>> fxi( fdims );

    kernel::silent_xi <<< fdims.y, 256 >>> (
        reinterpret_cast<fft::complex64 *>( fxi.d_buffer ),
        reinterpret_cast<fft::complex64 *>( fM2.d_buffer ),
        reinterpret_cast<fft::complex64 *>( fM4a.d_buffer ),
        reinterpret_cast<fft::complex64 *>( fM4b.d_buffer ),
        reinterpret_cast<fft::complex64 *>( fF.d_buffer ),
        d_tab, float( dla ), float( a_c ),
        fdims, fft::dk( sim.box ), dx, lambda_D2, float( ppc.x * ppc.y )
    );

    vec3grid<float3> xi( sim.ntiles, sim.nx, gc );
    fft::plan fft_backward( dims, fft::type::c2r_v3 );
    fft_backward.transform( fxi, xi );
    xi.copy_to_gc();

    // Move the particles (mirror pairs get identical displacements)
    const float max = species.displace( xi );

    std::cout << "(*info*) Langmuir-silent start: max |xi| = " << max << " cells\n";

    device::free( d_tab );
}
