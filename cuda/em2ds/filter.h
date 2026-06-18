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

// Binomial low-pass filter. The k-space transfer [cos^2(k.dx/2)]^order is the exact
// equivalent of applying an [1/4,1/2,1/4] real-space stencil `order` times (cf. the
// real-space Filter::Binomial in cuda/em2d), but evaluated directly on the spectral
// fields. Unlike the brick-wall Lowpass it rolls off smoothly, so it suppresses high-k
// content without the Gibbs ringing a hard spectral cut introduces in the fields.
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


}


#endif
