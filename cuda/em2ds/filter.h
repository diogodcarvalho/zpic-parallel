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

__global__
void kernel_binomial( fft::complex64 * const __restrict__ data,
    uint2 const dims, unsigned int const order  )
{
    const int iy  = blockIdx.x;  // Line
    const int ky  = abs( ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) );

    const int stride = dims.x;

    // Binomial transfer on the (complex) y-axis: [cos^2( pi*ky / dims.y )]^order.
    // This is the k-space equivalent of an order-fold [1/4,1/2,1/4] real-space stencil.
    const float cy = cosf( M_PI * ky / float( dims.y ) );
    const float hy = powf( cy * cy, order );

    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        // x is the real-FFT axis: Nyquist sits at ix = dims.x - 1
        const float cx = cosf( M_PI * ix / float( 2 * ( dims.x - 1 ) ) );
        const float hx = powf( cx * cx, order );

        data[ idx ] *= hx * hy;
    }
}

__global__
void kernel_binomial3( fft::complex64 * const __restrict__ fld,
    uint2 const dims, unsigned int const order  )
{
    const int iy  = blockIdx.x;  // Line
    const int ky  = abs( ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) );

    const int stride = dims.x;

    fft::complex64 * const __restrict__ fldx = & fld [ 0 ];
    fft::complex64 * const __restrict__ fldy = & fld [ dims.x * dims.y ];
    fft::complex64 * const __restrict__ fldz = & fld [ 2 * dims.x * dims.y ];

    const float cy = cosf( M_PI * ky / float( dims.y ) );
    const float hy = powf( cy * cy, order );

    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float cx = cosf( M_PI * ix / float( 2 * ( dims.x - 1 ) ) );
        const float hx = powf( cx * cx, order );

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
            fld.dims, order );
    }
    void apply( basic_grid3<std::complex<float>> & fld ) {
        kernel_binomial3 <<< fld.dims.y, 256 >>> (
            reinterpret_cast<fft::complex64 *>( fld.d_buffer ),
            fld.dims, order );
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
