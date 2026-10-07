#ifndef FILTER_H_
#define FILTER_H_

#include "zpic.h"
#include  "basic_grid.h"
#include  "basic_grid3.h"

namespace {

__global__
void kernel_lowpass( fft::complex64 * const __restrict__ data, 
    uint2 const dims, float2 const cutoff  )
{
    const int iy  = blockIdx.x;  // Line
    const int ky  = abs( ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) );

    const int kcx = cutoff.x * ( dims.x - 1 );
    const int kcy = cutoff.y * ( dims.y / 2 );

    const int stride = dims.x;

    if ( ky > kcy ) {
        for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
            auto idx = iy * stride + ix;
            data[ idx ] = 0;
        }
    } else {
        for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
            auto idx = iy * stride + ix;

            const auto kx = ix;
            if ( kx > kcx ) data[ idx ] = 0;
        }
    }
}

__global__
void kernel_lowpass3( fft::complex64 * const __restrict__ fld, 
    uint2 const dims, float2 const cutoff  )
{
    const int iy  = blockIdx.x;  // Line
    const int ky  = abs( ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) );

    const int kcx = cutoff.x * ( dims.x - 1 );
    const int kcy = cutoff.y * ( dims.y / 2 );

    const int stride = dims.x;

    fft::complex64 * const __restrict__ fldx = & fld [ 0 ];
    fft::complex64 * const __restrict__ fldy = & fld [ dims.x * dims.y ];
    fft::complex64 * const __restrict__ fldz = & fld [ 2 * dims.x * dims.y ];

    if ( ky > kcy ) {
        for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
            auto idx = iy * stride + ix;

            fldx[ idx ] = 0;
            fldy[ idx ] = 0;
            fldz[ idx ] = 0;
        }
    } else {
        for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
            auto idx = iy * stride + ix;

            const auto kx = ix;
            if ( kx > kcx ) {
                fldx[ idx ] = 0;
                fldy[ idx ] = 0;
                fldz[ idx ] = 0;
            }
        }
    }
}

/**
 * @brief 1D transfer of the (compensated) binomial filter, for c = cos(k.dx/2)
 *
 * Binomial part is [cos^2(k.dx/2)]^order = (1-s)^order, with s = sin^2(k.dx/2). The
 * compensator of order `comp` is the series of (1-s)^-order truncated at s^comp, so that
 * the product is 1 - O(s^(comp+1)) at low k while keeping the order-`order` zero at the
 * Nyquist frequency. comp = 1 gives 1 + order.s, i.e. the [-1,(4+2 order)/order,-1] stencil
 * of the real-space Filter::Compensated in cuda/em2d. comp = 0 is the plain binomial.
 */
__device__ inline
float binomial_transfer( float const c, unsigned int const order, unsigned int const comp )
{
    const float c2 = c * c;
    const float s  = 1.0f - c2;

    float term = 1.0f, h = 1.0f;
    for( unsigned int j = 1; j <= comp; j++ ) {
        term *= s * ( order + j - 1 ) / float( j );
        h    += term;
    }

    return powf( c2, order ) * h;
}

__global__
void kernel_binomial( fft::complex64 * const __restrict__ data,
    uint2 const dims, unsigned int const order, unsigned int const comp )
{
    const int iy  = blockIdx.x;  // Line
    const int ky  = abs( ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) );

    const int stride = dims.x;

    // Binomial transfer on the (complex) y-axis: [cos^2( pi*ky / dims.y )]^order.
    // This is the k-space equivalent of an order-fold [1/4,1/2,1/4] real-space stencil.
    const float cy = cosf( M_PI * ky / float( dims.y ) );
    const float hy = binomial_transfer( cy, order, comp );

    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        // x is the real-FFT axis: Nyquist sits at ix = dims.x - 1
        const float cx = cosf( M_PI * ix / float( 2 * ( dims.x - 1 ) ) );
        const float hx = binomial_transfer( cx, order, comp );

        data[ idx ] *= hx * hy;
    }
}

__global__
void kernel_binomial3( fft::complex64 * const __restrict__ fld,
    uint2 const dims, unsigned int const order, unsigned int const comp )
{
    const int iy  = blockIdx.x;  // Line
    const int ky  = abs( ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) );

    const int stride = dims.x;

    fft::complex64 * const __restrict__ fldx = & fld [ 0 ];
    fft::complex64 * const __restrict__ fldy = & fld [ dims.x * dims.y ];
    fft::complex64 * const __restrict__ fldz = & fld [ 2 * dims.x * dims.y ];

    const float cy = cosf( M_PI * ky / float( dims.y ) );
    const float hy = binomial_transfer( cy, order, comp );

    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float cx = cosf( M_PI * ix / float( 2 * ( dims.x - 1 ) ) );
        const float hx = binomial_transfer( cx, order, comp );

        const float h = hx * hy;
        fldx[ idx ] *= h;
        fldy[ idx ] *= h;
        fldz[ idx ] *= h;
    }
}

__global__
void kernel_gaussian( fft::complex64 * const __restrict__ data,
    uint2 const dims, float const sigma )
{
    const int iy  = blockIdx.x;  // Line
    const int ky  = abs( ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) );

    const int stride = dims.x;

    // Gaussian transfer exp( -(sigma * k.dx)^2 / 2 ) on the (complex) y-axis, i.e. a
    // real-space convolution with a Gaussian of rms width `sigma` cells. The Nyquist
    // frequency ( k.dx = pi ) sits at ky = dims.y / 2.
    const float a = -0.5f * sigma * sigma;

    const float kdy = 2 * M_PI * ky / float( dims.y );
    const float hy  = expf( a * kdy * kdy );

    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        // x is the real-FFT axis: Nyquist sits at ix = dims.x - 1
        const float kdx = M_PI * ix / float( dims.x - 1 );
        const float hx  = expf( a * kdx * kdx );

        data[ idx ] *= hx * hy;
    }
}

__global__
void kernel_gaussian3( fft::complex64 * const __restrict__ fld,
    uint2 const dims, float const sigma )
{
    const int iy  = blockIdx.x;  // Line
    const int ky  = abs( ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) );

    const int stride = dims.x;

    fft::complex64 * const __restrict__ fldx = & fld [ 0 ];
    fft::complex64 * const __restrict__ fldy = & fld [ dims.x * dims.y ];
    fft::complex64 * const __restrict__ fldz = & fld [ 2 * dims.x * dims.y ];

    const float a = -0.5f * sigma * sigma;

    const float kdy = 2 * M_PI * ky / float( dims.y );
    const float hy  = expf( a * kdy * kdy );

    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float kdx = M_PI * ix / float( dims.x - 1 );
        const float hx  = expf( a * kdx * kdx );

        const float h = hx * hy;
        fldx[ idx ] *= h;
        fldy[ idx ] *= h;
        fldz[ idx ] *= h;
    }
}

}

namespace Filter {

class Digital {
    public:
    virtual Digital * clone() const = 0;
    virtual void apply( basic_grid<std::complex<float>> & fld )  = 0;
    virtual void apply( basic_grid3<std::complex<float>> & fld ) = 0;
    virtual ~Digital() = default;
};

class None : public Digital {
    public:
    None * clone() const override { return new None(); };
    void apply( basic_grid<std::complex<float>> & fld ) override { /* do nothing */ };
    void apply( basic_grid3<std::complex<float>> & fld ) override { /* do nothing */ };
};

class Lowpass : public Digital {
    protected:

    const float2 cutoff;
    
    public:

    Lowpass( const float2 cutoff ) : cutoff( cutoff ) {};

    Lowpass * clone() const override { return new Lowpass ( cutoff ); };

    void apply( basic_grid<std::complex<float>> & fld ) {
        kernel_lowpass <<< fld.dims.y, 256 >>> ( 
            reinterpret_cast<fft::complex64 *>( fld.d_buffer ),
            fld.dims, cutoff );
    }
    void apply( basic_grid3<std::complex<float>> & fld ) {
        kernel_lowpass3 <<< fld.dims.y, 256 >>> (
            reinterpret_cast<fft::complex64 *>( fld.d_buffer ),
            fld.dims, cutoff );
    }
};

// Binomial low-pass filter. 
// The k-space transfer [cos^2(k.dx/2)]^order is the exact equivalent of applying an 
// [1/4,1/2,1/4] real-space stencil `order` times (cf. the real-space Filter::Binomial in cuda/em2d)
// but evaluated directly on the spectral fields.
class Binomial : public Digital {
    protected:

    unsigned int order;

    public:

    Binomial( unsigned int order = 0 ) : order( (order > 0) ? order : 1 ) {};

    Binomial * clone() const override { return new Binomial ( order ); };

    void apply( basic_grid<std::complex<float>> & fld ) {
        kernel_binomial <<< fld.dims.y, 256 >>> (
            reinterpret_cast<fft::complex64 *>( fld.d_buffer ),
            fld.dims, order, 0 );
    }
    void apply( basic_grid3<std::complex<float>> & fld ) {
        kernel_binomial3 <<< fld.dims.y, 256 >>> (
            reinterpret_cast<fft::complex64 *>( fld.d_buffer ),
            fld.dims, order, 0 );
    }
};

// Compensated binomial low-pass filter.
// Binomial( order ) followed by a compensator of order `comp`, the series of
// [cos^2(k.dx/2)]^-order truncated at sin^(2 comp)(k.dx/2). The combined (maximally flat)
// transfer is monotonic, 1 - O( (k.dx)^(2 comp + 2) ) at low k, and keeps the binomial
// zero of order `order` at the Nyquist frequency. The roll-off moves towards Nyquist as
// `comp` grows (1D transfer = 1/2 at k.dx = pi/2 for comp = order - 1). comp = 1 is the
// compensator of the real-space Filter::Compensated in cuda/em2d.
class Compensated : public Binomial {
    protected:

    unsigned int comp;

    public:

    Compensated( unsigned int order = 0, unsigned int comp = 1 ) : Binomial( order ), comp( comp ) {};

    Compensated * clone() const override { return new Compensated ( order, comp ); };

    void apply( basic_grid<std::complex<float>> & fld ) {
        kernel_binomial <<< fld.dims.y, 256 >>> (
            reinterpret_cast<fft::complex64 *>( fld.d_buffer ),
            fld.dims, order, comp );
    }
    void apply( basic_grid3<std::complex<float>> & fld ) {
        kernel_binomial3 <<< fld.dims.y, 256 >>> (
            reinterpret_cast<fft::complex64 *>( fld.d_buffer ),
            fld.dims, order, comp );
    }
};

// Gaussian low-pass filter. 
// The k-space transfer exp( -(sigma k.dx)^2 / 2 ) is a real-space convolution with a
// Gaussian of rms width `sigma` cells. The width is continuous rather than quantized by the
// stencil order: sigma = sqrt( order / 2 ) matches [cos^2(k.dx/2)]^order to O(k^4), so the
// default sigma = sqrt(1/2) is the Gaussian equivalent of Binomial( 1 ). Note that, unlike
// the binomial, the transfer never reaches exactly 0 at the Nyquist frequency
// ( exp( -sigma^2 pi^2 / 2 ) = 0.085 for the default sigma ).
class Gaussian : public Digital {
    protected:

    float sigma;

    public:

    Gaussian( float sigma = M_SQRT1_2 ) : sigma( (sigma > 0) ? sigma : float(M_SQRT1_2) ) {};

    Gaussian * clone() const override { return new Gaussian ( sigma ); };

    void apply( basic_grid<std::complex<float>> & fld ) {
        kernel_gaussian <<< fld.dims.y, 256 >>> (
            reinterpret_cast<fft::complex64 *>( fld.d_buffer ),
            fld.dims, sigma );
    }
    void apply( basic_grid3<std::complex<float>> & fld ) {
        kernel_gaussian3 <<< fld.dims.y, 256 >>> (
            reinterpret_cast<fft::complex64 *>( fld.d_buffer ),
            fld.dims, sigma );
    }
};


}


#endif
