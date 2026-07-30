// For getopt
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include <cstdlib>

#include "gpu.h"
#include "utils.h"
#include "grid.h"
#include "basic_grid.h"

#include "fft.h"
#include "simulation.h"

namespace kernel {

__global__
void set_charge( 
    float * const __restrict__ d_buffer, 
    uint2 const nx, uint2 const ext_nx, 
    float2 dx, float2 x0, float r ) 
{
    const uint2  tile_idx = { blockIdx.x, blockIdx.y };
    const int    tile_id  = tile_idx.y * gridDim.x + tile_idx.x;
    const size_t tile_off = tile_id * roundup4( ext_nx.x * ext_nx.y );
    auto * const __restrict__ tile_data = & d_buffer[ tile_off ];

    for( auto idx = block_thread_rank(); idx < nx.y * nx.x; idx += block_num_threads() ) {
        const auto iy =  idx / nx.x; 
        const auto ix =  idx % nx.x;    

        float x = ( tile_idx.x * nx.x + ix ) * dx.x;
        float y = ( tile_idx.y * nx.y + iy ) * dx.y;

        tile_data[ iy * ext_nx.x + ix ] = (x-x0.x)*(x-x0.x) + (y-x0.y) * (y-x0.y) <= r*r; 
    }
}

__global__
/**
 * @brief CUDA kernel for Poisson equation
 * 
 * @note Kernel must be called with grid(dims.y)
 * 
 * @param data 
 * @param dims 
 * @param dk 
 */
void poisson(
        fft::complex64 * const __restrict__ data, uint2 const dims, float2 const dk
    )
{
    const int iy   = blockIdx.x;  // Line
    const float ky = ((iy < dims.y/2) ? iy : (iy - int(dims.y)) ) * dk.y;

    const int stride = dims.x;
    for( auto ix = block_thread_rank(); ix < dims.x; ix += block_num_threads() ) {
        auto idx = iy * stride + ix;

        const float kx = ix * dk.x;
        const float k2 = kx*kx + ky*ky;

        const float scale = (k2 > 0)? 1.f / k2 : 0.;

        data[ idx ] *= scale;
    }
}

}

void test_grid( void ) {
    
    std::cout << ansi::bold;
    std::cout << "Running " << __func__ << "()...";
    std::cout << ansi::reset << std::endl;

    const float2 box{1.0, 1.0};
    const uint2 ntiles{ 16, 16 };
    const uint2 nx    { 16, 16 };

    uint2 in_dims{ ntiles.x * nx.x, ntiles.y * nx.y };
    float2 dx{ box.x / in_dims.x, box.y / in_dims.y };

    bnd<unsigned int> gc;
    gc.x = {0,1};
    gc.y = {0,1};

    std::cout << "Allocating arrays..." << '\n';

    grid<float> charge( ntiles, nx, gc );
    grid<float> potential( ntiles, nx, gc );

    dim3 grid( ntiles.x, ntiles.y );
    dim3 block( 64 );

    std::cout << "Setting charge..." << '\n';

    kernel::set_charge <<< grid, block >>> (
        charge.d_buffer + charge.offset, nx, charge.ext_nx,
        dx, float2{ 0.25, 0.25 }, 0.1
    );

    std::cout << "Saving charge to disk..." << '\n';
    charge.save("charge.zdf");

    fft::plan plan_r2c( in_dims, fft::type::r2c );
    fft::plan plan_c2r( in_dims, fft::type::c2r );
    
    uint2 out_dims = fft::fdims( in_dims );
    basic_grid<std::complex<float>> fpotential( out_dims );

    plan_r2c.transform( charge, fpotential );

    fpotential.name = "F(charge)";
    fpotential.save( "charge_k.zdf" );

    kernel::poisson <<< out_dims.y, 64 >>> (
        reinterpret_cast< fft::complex64 * > ( fpotential.d_buffer ),
        out_dims, fft::dk( box )
    );

    fpotential.name = "F(potential)";
    fpotential.save( "potential_k.zdf" );

    plan_c2r.transform( fpotential, charge );

    std::cout << "Saving potential to disk..." << '\n';
    charge.save("potential.zdf");

    std::cout << ansi::bold;
    std::cout << "Done!\n";
    std::cout << ansi::reset;       
}

#include "emf.h"
#include "laser.h"

#include "timer.h"

void test_laser( ) {

    std::cout << ansi::bold
              << "Running " << __func__ << "()..."
              << ansi::reset << std::endl;

    uint2 ntiles{ 64, 16 };
    uint2 nx{ 16, 16 };

    float2 box{ 20.48, 25.6 };
    double dt{ 0.014 };

    EMF emf( ntiles, nx, box, dt );

    auto save_emf = [ & emf ]( ) {
        emf.save( emf::e, fcomp::x );
        emf.save( emf::e, fcomp::y );
        emf.save( emf::e, fcomp::z );

        emf.save( emf::b, fcomp::x );
        emf.save( emf::b, fcomp::y );
        emf.save( emf::b, fcomp::z );

        emf.save( emf::fet, fcomp::x );
        emf.save( emf::fet, fcomp::y );
        emf.save( emf::fet, fcomp::z );

        emf.save( emf::fb, fcomp::x );
        emf.save( emf::fb, fcomp::y );
        emf.save( emf::fb, fcomp::z );

    };

/*
    Laser::PlaneWave laser;
    laser.start = 10.2;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
*/

    Laser::Gaussian laser;
    laser.start = 10.2;
    laser.fwhm = 4.0;
    laser.a0 = 1.0;
    laser.omega0 = 10.0;
    laser.W0 = 1.5;
    laser.focus = 20.48;
    laser.axis = 12.8;

    laser.sin_pol = 0;
    laser.cos_pol = 1;

    std::cout << "Adding laser...\n";
    laser.add( emf );

    save_emf();

    int niter = 20.48 / dt / 2;
    // int niter{ 10 };

    std::cout << "Starting test - " << niter << " iterations...\n";

    Timer t0("test");

    t0.start();

    for( int i = 0; i < niter; i ++) {
        emf.advance( );
    }

    t0.stop();
    
    save_emf();

    std::ostringstream buffer;
    buffer << niter << " iterations: ";

    t0.report( buffer.str() );

    std::cout << ansi::bold
              << "Done!\n"
              << ansi::reset;   

}

void test_mov( ) {

    std::cout << "Starting " << __func__ << "...\n";

    uint2 ntiles{ 4, 4 };
    uint2 nx{ 32, 32 };

    float2 box{ 12.8, 12.8 };

    auto dt = 0.99 * zpic::courant( ntiles, nx, box );

    uint2 ppc{ 8, 8 };
    Species electrons( "electrons", -1.0f, ppc );

    electrons.set_density( Density::Sphere( 1.0, float2{2.1, 2.1}, 2.0 ) );
    electrons.set_udist( UDistribution::Cold( float3{ -1, -2, -3 } ) );

    electrons.initialize( box, ntiles, nx, dt, 0 );

    electrons.save();
    electrons.save_charge();
    electrons.save_phasespace(
        phasespace::x, float2{0, 12.8}, 128,
        phasespace::y, float2{0, 12.8}, 128
    );

    int niter = 200;
    for( auto i = 0; i < niter; i ++ ) {
        electrons.advance();
    }

    electrons.save_charge();
    electrons.save();

    std::cout << __func__ << " complete.\n";
}


void test_weibel( ) {

    std::cout << ansi::bold
              << "Running " << __func__ << "()..."
              << ansi::reset << std::endl;

    uint2 ntiles{ 16, 16 };
    uint2 nx{ 32, 32 };
    
    float2 box = 0.1f * make_float2( ntiles.x * nx.x, ntiles.y * nx.y );
    float dt = 0.07;

    Simulation sim( ntiles, nx, box, dt );

    uint2 ppc{ 8, 8 };

    Species electrons("electrons", -1.0f, ppc);
    electrons.set_udist(
        UDistribution::ThermalCorr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, 0.6 }
        )
    );

    sim.add_species( electrons );

    Species positrons("positrons", +1.0f, ppc);
    positrons.set_udist(
        UDistribution::ThermalCorr( 
            float3{ 0.1, 0.1, 0.1 },
            float3{ 0, 0, -0.6 }
        )
    );

    sim.add_species( positrons );

    // Lambda function for diagnostic output
    auto diag = [ & ]( ) {
        sim.emf.save(emf::b, fcomp::x);
        sim.emf.save(emf::b, fcomp::y);
        sim.emf.save(emf::b, fcomp::z);

        sim.current.save(fcomp::x);
        sim.current.save(fcomp::y);
        sim.current.save(fcomp::z);

        sim.charge.save();

        electrons.save_charge();
        positrons.save_charge();

        sim.energy_info();
    };

    Timer timer; timer.start();

    while ( sim.get_t() <= 35.0 ) {
        //if ( sim.get_iter() % 10 == 0 )
        //    std::cout << "t = " << sim.get_t() << '\n';
        sim.advance();
    }
    timer.stop();

    diag();

    std::cout << ansi::bold
              << "Done!\n"
              << ansi::reset;

    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
              << ", Performance: " << perf << " GPart/s\n";
}


/**
 * @brief Regression test for the particle count reductions (`-t np_total`)
 *
 * `Particles::np_total()` and `np_max_tile()` launch one thread per tile with
 * the grid rounded up to whole blocks of 1024. When the tile count is above
 * 1024 and not a multiple of it, the excess threads index past the end of
 * `np[]` and fold whatever is in that memory into the reduction.
 *
 * Injection uses the standard uniform density with ppc > 1 of test_weibel, so
 * the count is known exactly: gnx.x * gnx.y * ppc.x * ppc.y particles, all
 * tiles holding the same nx.x * nx.y * ppc.x * ppc.y. Tile counts at or below
 * one block are included as controls -- they must pass both before and after
 * the fix, which is what pins the failure on the reduction rather than on the
 * per-tile counts written by `np_inject`.
 *
 * A uniform species takes its charge normalization from ppc, so there a wrong
 * total only corrupts the size of the particle dumps. Lattice and Sparse
 * instead divide n0 by the injected count in `norm_charge`, so for those the
 * macroparticle charge -- and with it the plasma density actually simulated --
 * is off by the same factor.
 */
void test_np_total( ) {

    std::cout << ansi::bold
              << "Running " << __func__ << "()..."
              << ansi::reset << std::endl;

    // Tiles are deliberately small: the bug depends on the number of tiles, not
    // on the grid size, so there is no reason to pay for a test_weibel-sized
    // grid on top of the 2048-tile case.
    const uint2 nx  { 16, 16 };
    const uint2 ppc {  4,  4 };
    const float dx  = 0.1f;

    /// @brief Tile geometry to test, with the reason it is interesting
    struct testcase {
        uint2 ntiles;
        const char * note;
    };

    const testcase cases[] = {
        { { 16, 16 }, "below one block"                 },
        { { 32, 32 }, "exactly one block"               },
        { { 33, 32 }, "one block + 32"                  },
        { { 40, 40 }, "as runs/20 boosted-dx-001"       },
        { { 64, 32 }, "exactly two blocks"              },
    };
    const int ncases = sizeof( cases ) / sizeof( cases[0] );

    std::cout << "nx (tile) : " << nx  << '\n';
    std::cout << "ppc       : " << ppc << '\n';
    std::cout << "dx        : " << dx  << "\n\n";

    // Uniform density places ppc particles in every cell, so every tile holds
    // the same number and np_max_tile is predictable
    const unsigned long np_tile_expected =
        ( unsigned long ) nx.x * nx.y * ppc.x * ppc.y;

    int nfail = 0;

    for ( auto const & c : cases ) {

        const uint2 ntiles = c.ntiles;
        const uint2 gnx { ntiles.x * nx.x, ntiles.y * nx.y };

        const unsigned long ntiles_tot  = ( unsigned long ) ntiles.x * ntiles.y;
        const unsigned long np_expected = np_tile_expected * ntiles_tot;

        const float2 box = dx * make_float2( gnx.x, gnx.y );

        // Species::initialize() runs the full injection path on its own, so the
        // test does not need a Simulation (nor its FFT plans and field grids).
        Species electrons( "electrons", -1.0f, ppc );
        electrons.set_density( Density::Uniform( 1.0f ) );
        electrons.set_udist(
            UDistribution::ThermalCorr( float3{ 0.1, 0.1, 0.1 }, float3{ 0, 0, 0 } )
        );
        electrons.initialize( box, ntiles, nx, 0.01f, 1 );

        const unsigned long np_reduction  = electrons.np_total();
        const unsigned long npt_reduction = electrons.np_max_tile();

        const bool ok = ( np_reduction  == np_expected      ) &&
                        ( npt_reduction == np_tile_expected );

        std::cout << ansi::bold << "ntiles = " << ntiles << ansi::reset
                  << "  (" << ntiles_tot << " tiles, " << c.note << ")\n";
        std::cout << "  np_total()    : " << np_reduction  << " (expected " << np_expected      << ")\n";
        std::cout << "  np_max_tile() : " << npt_reduction << " (expected " << np_tile_expected << ")\n";
        std::cout << "  " << ansi::bold << ( ok ? "PASS" : "FAIL" ) << ansi::reset << "\n\n";

        if ( ! ok ) nfail++;
    }

    if ( nfail ) {
        std::cout << ansi::bold << nfail << " of " << ncases << " cases FAILED"
                  << ansi::reset << '\n';
        // std::exit, not device::exit: the latter recurses into itself (gpu.h)
        std::exit(1);
    }

    std::cout << ansi::bold << "All " << ncases << " cases passed."
              << ansi::reset << '\n';
}


void test_sparse( std::string param ) {

    std::cout << ansi::bold
              << "Running " << __func__ << "()..."
              << ansi::reset << std::endl;

    // Default parameters.
    // Override any of them via -p "key=value ..." pairs,
    uint2  ntiles { 1, 1 };
    uint2  nx     { 32, 32 };
    float  dx      = 0.001f;             // cell size (same in x and y)
    float  dx_part = 0.032f;             // sparse density spacing (same in x and y)
    uint2  spacing{ 0, 0 };              // lattice spacing in cells (overrides dx_part if set)
    uint2  ppc    { 0, 0 };              // if set, use uniform density with this ppc instead of sparse
    float  density = 1.0f;               // plasma density (n0) for the selected profile
    float  dt      = 0.000692965f;       // time step
    uint  n_dump  = 100;                // diagnostic output interval (in number of timesteps)
    uint  n_skip  = 0;                  // warm-up steps to run before the first dump (first dump at step n_skip)
    float  tmax    = 0.002f;             // total simulation time
    float3 uth    { 0.0f, 0.0f, 0.0f };  // thermal velocity
    float3 ufl    { 0.0f, 0.0f, 0.0f };  // fluid velocity
    std::string udist = "thermal";       // momentum distribution: thermal | maxwell_juttner
    float theta   = 0.0f;                // normalized temperature kT/mc^2 (maxwell_juttner only)
    std::string filter_mode = "lowpass"; // source filter: none | lowpass | binomial | gaussian
    unsigned filter_order = 1;           // binomial filter order (binomial only)
    float filter_sigma = 0.707107f;      // gaussian width in cells (gaussian only; sqrt(1/2) = binomial order 1)
    unsigned seed  = 0;                  // RNG seed offset (vary for independent replicas)
    uint  save_part = 1;                 // dump particle phase space at each diag (0 = energy only, e.g. warm-up probe)
    uint  save_fld  = 1;                 // dump E, B, J and charge grids at each diag (0 = skip; the grids dominate the output of high-resolution, low-particle-count runs)
    float dump_start = 0.0f;             // write field/particle dumps only from t >= this (in 1/w_p); energy is logged from t=0
    std::string init = "poisson";        // field initialization: poisson | darwin | none
    uint  darwin_iter = 2;               // number of Darwin bootstrap passes (init=darwin only)

    // Parse whitespace-separated "key=value" pairs. 
    // Vector values are comma-separated, e.g. "ntiles=4,4" or "uth=0.1,0,0".
    std::istringstream tokens( param );
    std::string tok;
    while ( tokens >> tok ) {
        auto eq = tok.find('=');
        if ( eq == std::string::npos ) {
            std::cerr << "Invalid parameter (expected key=value): '" << tok << "'\n";
            std::exit(1);
        }
        std::string key = tok.substr( 0, eq );
        std::string val = tok.substr( eq + 1 );

        int got, want;
        if      ( key == "ntiles"     ) { got = std::sscanf( val.c_str(), "%u,%u",    &ntiles.x, &ntiles.y );   want = 2; }
        else if ( key == "nx"         ) { got = std::sscanf( val.c_str(), "%u,%u",    &nx.x, &nx.y );           want = 2; }
        else if ( key == "dx"         ) { got = std::sscanf( val.c_str(), "%f",       &dx );                    want = 1; }
        else if ( key == "dt"         ) { got = std::sscanf( val.c_str(), "%f",       &dt );                    want = 1; }
        else if ( key == "uth"        ) { got = std::sscanf( val.c_str(), "%f,%f,%f", &uth.x, &uth.y, &uth.z ); want = 3; }
        else if ( key == "ufl"        ) { got = std::sscanf( val.c_str(), "%f,%f,%f", &ufl.x, &ufl.y, &ufl.z ); want = 3; }
        else if ( key == "udist"      ) { udist = val; got = 1;                                                want = 1; }
        else if ( key == "theta"      ) { got = std::sscanf( val.c_str(), "%f",       &theta );                want = 1; }
        else if ( key == "n_dump"     ) { got = std::sscanf( val.c_str(), "%u",       &n_dump );                want = 1; }
        else if ( key == "n_skip"     ) { got = std::sscanf( val.c_str(), "%u",       &n_skip );                want = 1; }
        else if ( key == "tmax"       ) { got = std::sscanf( val.c_str(), "%f",       &tmax );                  want = 1; }
        else if ( key == "dx_part"    ) { got = std::sscanf( val.c_str(), "%f",       &dx_part );               want = 1; }
        else if ( key == "density"    ) { got = std::sscanf( val.c_str(), "%f",       &density );               want = 1; }
        else if ( key == "spacing"    ) { got = std::sscanf( val.c_str(), "%u,%u",    &spacing.x, &spacing.y ); want = 2; }
        else if ( key == "ppc"        ) { got = std::sscanf( val.c_str(), "%u,%u",    &ppc.x, &ppc.y );
                                          if ( got == 1 ) { ppc.y = ppc.x; got = 2; }                             want = 2; }
        else if ( key == "filter"     ) { // accept the legacy 0/1 boolean as well as a mode string
                                          if      ( val == "0" || val == "off"     ) filter_mode = "none";
                                          else if ( val == "1" || val == "on"      ) filter_mode = "lowpass";
                                          else                                        filter_mode = val;
                                          got = 1; want = 1; }
        else if ( key == "filter_order") { got = std::sscanf( val.c_str(), "%u",       &filter_order );           want = 1; }
        else if ( key == "filter_sigma") { got = std::sscanf( val.c_str(), "%f",       &filter_sigma );           want = 1; }
        else if ( key == "seed"       ) { got = std::sscanf( val.c_str(), "%u",       &seed );                    want = 1; }
        else if ( key == "save_part"  ) { got = std::sscanf( val.c_str(), "%u",       &save_part );               want = 1; }
        else if ( key == "save_fld"   ) { got = std::sscanf( val.c_str(), "%u",       &save_fld );                want = 1; }
        else if ( key == "dump_start" ) { got = std::sscanf( val.c_str(), "%f",       &dump_start );              want = 1; }
        else if ( key == "init"       ) { init = val; got = 1;                                                    want = 1; }
        else if ( key == "darwin_iter") { got = std::sscanf( val.c_str(), "%u",       &darwin_iter );             want = 1; }
        else {
            std::cerr << "Unknown parameter key: '" << key << "'\n";
            std::exit(1);
        }
        if ( got != want ) {
            std::cerr << "Invalid value for '" << key << "': '" << val << "'\n";
            std::exit(1);
        }
    }

    // Binomial filter order must be at least 1 (matches Filter::Binomial clamp)
    if ( filter_order < 1 ) filter_order = 1;
    // Gaussian filter width must be positive (matches Filter::Gaussian clamp)
    if ( filter_sigma <= 0 ) filter_sigma = 0.707107f;

    float2 box = dx * make_float2( ntiles.x * nx.x, ntiles.y * nx.y );

    std::cout << "ntiles    : " << ntiles  << '\n';
    std::cout << "nx (tile) : " << nx      << '\n';
    std::cout << "dx        : " << dx      << '\n';
    std::cout << "box       : " << box     << '\n';
    if ( ppc.x > 0 && ppc.y > 0 )
        std::cout << "ppc       : " << ppc     << " (uniform density)\n";
    else if ( spacing.x > 0 && spacing.y > 0 )
        std::cout << "spacing   : " << spacing << '\n';
    else
        std::cout << "dx_part   : " << dx_part << '\n';
    std::cout << "density   : " << density << '\n';
    std::cout << "dt        : " << dt      << '\n';
    std::cout << "n_dump    : " << n_dump  << '\n';
    std::cout << "n_skip    : " << n_skip  << " (first dump at t = " << n_skip * dt << ")\n";
    std::cout << "tmax      : " << tmax    << '\n';
    std::cout << "udist     : " << udist   << ( udist.rfind( "maxwell_juttner", 0 ) == 0 ? " (theta=" + std::to_string(theta) + ")" : "" ) << '\n';
    std::cout << "uth       : " << uth     << '\n';
    std::cout << "ufl       : " << ufl     << '\n';
    std::cout << "filter    : " << filter_mode << ( filter_mode == "binomial" ? " (order=" + std::to_string(filter_order) + ")" :
                                                    filter_mode == "gaussian" ? " (sigma=" + std::to_string(filter_sigma) + " cells)" : "" ) << '\n';
    std::cout << "seed      : " << seed    << '\n';
    std::cout << "save_part : " << save_part << '\n';
    std::cout << "save_fld  : " << save_fld  << '\n';
    std::cout << "dump_start: " << dump_start << " (field/particle dumps from t >= this)\n";
    std::cout << "init      : " << init    << ( init == "darwin" ? " (iter=" + std::to_string(darwin_iter) + ")" : "" ) << '\n';

    // Map the init string to the field initialization type
    emf::init_type::type init_type;
    if      ( init == "poisson" ) init_type = emf::init_type::poisson;
    else if ( init == "darwin"  ) init_type = emf::init_type::darwin;
    else if ( init == "boosted" ) init_type = emf::init_type::boosted;
    else if ( init == "none"    ) init_type = emf::init_type::none;
    else {
        std::cerr << "Unknown init type: '" << init << "' (expected poisson, darwin, boosted or none)\n";
        std::exit(1);
    }

    if ( udist != "thermal" && udist != "maxwell_juttner" && udist != "maxwell_juttner_corr" ) {
        std::cerr << "Unknown udist: '" << udist
                  << "' (expected thermal, maxwell_juttner or maxwell_juttner_corr)\n";
        std::exit(1);
    }

    if ( filter_mode != "none" && filter_mode != "lowpass" && filter_mode != "binomial" && filter_mode != "gaussian" ) {
        std::cerr << "Unknown filter: '" << filter_mode << "' (expected none, lowpass, binomial or gaussian)\n";
        std::exit(1);
    }

    Simulation sim( ntiles, nx, box, dt );

    // Override the default brick-wall Lowpass on current/charge with the requested
    // source filter (none to measure collisionality, or binomial / gaussian to avoid
    // the Gibbs ringing the hard spectral cut introduces in the fields).
    if ( filter_mode == "none" ) {
        sim.current.set_filter( Filter::None() );
        sim.charge.set_filter ( Filter::None() );
    } else if ( filter_mode == "binomial" ) {
        sim.current.set_filter( Filter::Binomial( filter_order ) );
        sim.charge.set_filter ( Filter::Binomial( filter_order ) );
    } else if ( filter_mode == "gaussian" ) {
        sim.current.set_filter( Filter::Gaussian( filter_sigma ) );
        sim.charge.set_filter ( Filter::Gaussian( filter_sigma ) );
    }

    // When a ppc is given, fall back to uniform density. 
    // Otherwise the sparse/lattice profiles place one particle per density site.
    bool uniform = ( ppc.x > 0 && ppc.y > 0 );
    if ( ! uniform ) ppc = uint2{ 1, 1 };

    Species electrons("electrons", -1.0f, ppc, true);
    if ( uniform )
        electrons.set_density(
            Density::Uniform(density)
        );
    else if ( spacing.x > 0 && spacing.y > 0 )
        electrons.set_density(
            Density::Lattice(density, spacing)
        );
    else
        electrons.set_density(
            Density::Sparse(density, float2{ dx_part, dx_part })
        );
    if ( udist == "maxwell_juttner" )
        electrons.set_udist(
            UDistribution::MaxwellJuttner( theta )
        );
    else if ( udist == "maxwell_juttner_corr" ) {
        // Antithetic pairs are built inside each cell, so the profile must place a
        // fixed (and even) number of particles per cell.
        if ( ! uniform ) {
            std::cerr << "udist=maxwell_juttner_corr requires a uniform density (set ppc)\n";
            std::exit(1);
        }
        electrons.set_udist(
            UDistribution::MaxwellJuttnerCorr( theta, ppc )
        );
    }
    else
        electrons.set_udist(
            UDistribution::ThermalCorr( uth, ufl )
        );
    electrons.seed = seed;

    sim.add_species( electrons );

    // Initialize fields, must be done before first advance()
    sim.init_fields( init_type, darwin_iter );

    // Lambda function for diagnostic output
    auto diag = [ & ]( ) {
        // Heavy field / particle dumps only from t >= dump_start; the energy
        // report is always written so the budget is tracked from t = 0.
        if ( sim.get_t() >= dump_start ) {
            if ( save_fld ) {
                sim.emf.save(emf::e, fcomp::x);
                sim.emf.save(emf::e, fcomp::y);
                sim.emf.save(emf::e, fcomp::z);

                sim.emf.save(emf::b, fcomp::x);
                sim.emf.save(emf::b, fcomp::y);
                sim.emf.save(emf::b, fcomp::z);

                sim.current.save(fcomp::x);
                sim.current.save(fcomp::y);
                sim.current.save(fcomp::z);

                sim.charge.save();
                electrons.save_charge();
            }

            if ( save_part )
                electrons.save();
        }
        sim.energy_info();
    };

    Timer timer; timer.start();

    if ( n_skip == 0 )
        diag();

    while ( sim.get_t() <= tmax ) {
        sim.advance();
        if ( sim.get_iter() % 100 == 0 ) {
            std::cout << "iter = " << sim.get_iter()
                      << ", t = " << sim.get_t() << '\n';
        }
        unsigned const it = sim.get_iter();
        if ( it >= n_skip && ( it - n_skip ) % n_dump == 0 ) {
            diag();
        }
    }
    timer.stop();

    std::cout << ansi::bold
              << "Done!\n"
              << ansi::reset;

    auto perf = sim.get_nmove() / timer.elapsed(timer::s) / 1.e9;

    std::cerr << "Elapsed time: " << timer.elapsed(timer::s) << " s"
              << ", Performance: " << perf << " GPart/s\n";
}

/**
 * @brief Initialize GPU device
 * 
 */
void gpu_init( ) {

    // Reset current device
    deviceReset();
}

/**
 * @brief Print information about the environment
 * 
 */
void info( void ) {

    std::cout << ansi::bold;
    std::cout << "Environment\n";
    std::cout << ansi::reset;

    char name[HOST_NAME_MAX + 1];
    gethostname(name, HOST_NAME_MAX);

    std::cout << "GPU device on " << name << ":\n";
    print_gpu_info();
}

void cli_help( char * argv0 ) {
    std::cerr << "Usage: " << argv0 << " [-h] [-s] [-t name] [-n parameter]\n";

    std::cerr << '\n';
    std::cerr << "Options:\n";
    std::cerr << "  -h                  Display this message and exit\n";
    std::cerr << "  -s                  Silence information about host/CUDA device\n";
    std::cerr << "  -t <name>           Name of the test to run. Defaults to 'weibel'\n";
    std::cerr << "                      'np_total' runs the particle count reduction check\n";
    std::cerr << "                      (takes no -p parameters) and exits nonzero on failure\n";
    std::cerr << "  -p <parameters>     Test parameters (string). Purpose will depend on the \n";
    std::cerr << "                      test chosen. Defaults to '2,2,16,16'\n";
    std::cerr << '\n';
}

/**
 * @brief Two-particle test for the field initializations (boosted / poisson / darwin)
 *
 * Places exactly two macroparticles (two single-Point species) at chosen positions
 * with independent velocities and charge signs, then advances the system. Lets you
 * study how the chosen field init behaves for a co-moving pair (near-force-free in
 * the relativistic limit) or an interacting pair (where the uniform-motion boosted
 * field is only approximate). Selected with `-t pair`.
 *
 * Params (key=value): ntiles, nx, dx, dt, tmax, n_dump, n_skip, save_part,
 *   p1="x,y", p2="x,y" (positions, sim units),
 *   u1="ux,uy,uz", u2="ux,uy,uz" (proper momentum u=gamma*beta),
 *   s1, s2 (charge signs, +1/-1), q (charge magnitude),
 *   init (poisson|darwin|boosted|none), filter (none|lowpass|binomial|gaussian),
 *   filter_order (binomial), filter_sigma (gaussian).
 */
void test_boosted_pair( std::string param ) {

    std::cout << ansi::bold << "Running " << __func__ << "()..." << ansi::reset << std::endl;

    // Grid / time (defaults match the single-particle boosted test: 128 x 128 box)
    uint2  ntiles { 2, 2 };
    uint2  nx     { 64, 64 };
    float  dx     = 1.0f;
    float  dt     = 0.1f;
    float  tmax   = 300.0f;
    uint   n_dump = 100;
    uint   n_skip = 0;
    uint   save_part = 1;

    // Two particles: positions (sim units), velocities (u = gamma*beta), charge signs
    float2 p1 { 64.0f, 56.0f };
    float2 p2 { 64.0f, 72.0f };
    float3 u1 { 0.0f, 0.0f, 0.0f };
    float3 u2 { 0.0f, 0.0f, 0.0f };
    float  s1 = -1.0f, s2 = -1.0f;   // charge signs (mass/charge sign)
    float  q  = 1.0f;                // charge magnitude (macroparticle charge)

    std::string init = "boosted";
    std::string filter_mode = "binomial";
    unsigned filter_order = 1;
    float filter_sigma = 0.707107f;
    uint darwin_iter = 2;

    std::istringstream tokens( param );
    std::string tok;
    while ( tokens >> tok ) {
        auto eq = tok.find('=');
        if ( eq == std::string::npos ) { std::cerr << "Invalid parameter: '" << tok << "'\n"; std::exit(1); }
        std::string key = tok.substr(0,eq), val = tok.substr(eq+1);
        int got, want;
        if      ( key == "ntiles"      ) { got = std::sscanf(val.c_str(),"%u,%u",&ntiles.x,&ntiles.y);   want=2; }
        else if ( key == "nx"          ) { got = std::sscanf(val.c_str(),"%u,%u",&nx.x,&nx.y);           want=2; }
        else if ( key == "dx"          ) { got = std::sscanf(val.c_str(),"%f",&dx);                      want=1; }
        else if ( key == "dt"          ) { got = std::sscanf(val.c_str(),"%f",&dt);                      want=1; }
        else if ( key == "tmax"        ) { got = std::sscanf(val.c_str(),"%f",&tmax);                    want=1; }
        else if ( key == "n_dump"      ) { got = std::sscanf(val.c_str(),"%u",&n_dump);                  want=1; }
        else if ( key == "n_skip"      ) { got = std::sscanf(val.c_str(),"%u",&n_skip);                  want=1; }
        else if ( key == "save_part"   ) { got = std::sscanf(val.c_str(),"%u",&save_part);               want=1; }
        else if ( key == "p1"          ) { got = std::sscanf(val.c_str(),"%f,%f",&p1.x,&p1.y);           want=2; }
        else if ( key == "p2"          ) { got = std::sscanf(val.c_str(),"%f,%f",&p2.x,&p2.y);           want=2; }
        else if ( key == "u1"          ) { got = std::sscanf(val.c_str(),"%f,%f,%f",&u1.x,&u1.y,&u1.z);  want=3; }
        else if ( key == "u2"          ) { got = std::sscanf(val.c_str(),"%f,%f,%f",&u2.x,&u2.y,&u2.z);  want=3; }
        else if ( key == "s1"          ) { got = std::sscanf(val.c_str(),"%f",&s1);                      want=1; }
        else if ( key == "s2"          ) { got = std::sscanf(val.c_str(),"%f",&s2);                      want=1; }
        else if ( key == "q"           ) { got = std::sscanf(val.c_str(),"%f",&q);                       want=1; }
        else if ( key == "init"        ) { init = val; got=1;                                            want=1; }
        else if ( key == "filter"      ) { if      (val=="0"||val=="off") filter_mode="none";
                                           else if (val=="1"||val=="on" ) filter_mode="lowpass";
                                           else                            filter_mode=val;   got=1; want=1; }
        else if ( key == "filter_order") { got = std::sscanf(val.c_str(),"%u",&filter_order);            want=1; }
        else if ( key == "filter_sigma") { got = std::sscanf(val.c_str(),"%f",&filter_sigma);            want=1; }
        else if ( key == "darwin_iter" ) { got = std::sscanf(val.c_str(),"%u",&darwin_iter);             want=1; }
        else { std::cerr << "Unknown parameter key: '" << key << "'\n"; std::exit(1); }
        if ( got != want ) { std::cerr << "Invalid value for '" << key << "': '" << val << "'\n"; std::exit(1); }
    }
    if ( filter_order < 1 ) filter_order = 1;
    if ( filter_sigma <= 0 ) filter_sigma = 0.707107f;

    float2 box = dx * make_float2( ntiles.x * nx.x, ntiles.y * nx.y );

    emf::init_type::type init_type;
    if      ( init == "poisson" ) init_type = emf::init_type::poisson;
    else if ( init == "darwin"  ) init_type = emf::init_type::darwin;
    else if ( init == "boosted" ) init_type = emf::init_type::boosted;
    else if ( init == "none"    ) init_type = emf::init_type::none;
    else { std::cerr << "Unknown init type: '" << init << "'\n"; std::exit(1); }

    std::cout << "box   : " << box << '\n';
    std::cout << "p1    : " << p1 << "  u1 : " << u1 << "  s1 : " << s1 << '\n';
    std::cout << "p2    : " << p2 << "  u2 : " << u2 << "  s2 : " << s2 << '\n';
    std::cout << "q     : " << q  << '\n';
    std::cout << "init  : " << init << ( init == "darwin" ? " (iter=" + std::to_string(darwin_iter) + ")" : "" ) << '\n';
    std::cout << "filter: " << filter_mode << ( filter_mode == "binomial" ? " (order=" + std::to_string(filter_order) + ")" :
                                                filter_mode == "gaussian" ? " (sigma=" + std::to_string(filter_sigma) + " cells)" : "" ) << '\n';

    Simulation sim( ntiles, nx, box, dt );

    if ( filter_mode == "none" ) {
        sim.current.set_filter( Filter::None() );
        sim.charge.set_filter ( Filter::None() );
    } else if ( filter_mode == "binomial" ) {
        sim.current.set_filter( Filter::Binomial( filter_order ) );
        sim.charge.set_filter ( Filter::Binomial( filter_order ) );
    } else if ( filter_mode == "gaussian" ) {
        sim.current.set_filter( Filter::Gaussian( filter_sigma ) );
        sim.charge.set_filter ( Filter::Gaussian( filter_sigma ) );
    }

    // Two single-particle species: independent position, velocity, charge sign.
    // m_q sign sets the charge sign (q = copysign(norm_charge, m_q)); |m_q| = 1.
    Species sp1( "part1", s1, uint2{1,1}, true );
    sp1.set_density( Density::Point( q, p1 ) );
    sp1.set_udist ( UDistribution::Cold( u1 ) );

    Species sp2( "part2", s2, uint2{1,1}, true );
    sp2.set_density( Density::Point( q, p2 ) );
    sp2.set_udist ( UDistribution::Cold( u2 ) );

    sim.add_species( sp1 );
    sim.add_species( sp2 );

    sim.init_fields( init_type, darwin_iter );

    auto diag = [&]() {
        sim.emf.save(emf::e, fcomp::x); sim.emf.save(emf::e, fcomp::y); sim.emf.save(emf::e, fcomp::z);
        sim.emf.save(emf::b, fcomp::x); sim.emf.save(emf::b, fcomp::y); sim.emf.save(emf::b, fcomp::z);
        sim.charge.save();
        if ( save_part ) { sp1.save(); sp2.save(); }
        sim.energy_info();
    };

    Timer timer; timer.start();
    if ( n_skip == 0 ) diag();
    while ( sim.get_t() <= tmax ) {
        sim.advance();
        if ( sim.get_iter() % 100 == 0 )
            std::cout << "iter = " << sim.get_iter() << ", t = " << sim.get_t() << '\n';
        unsigned const it = sim.get_iter();
        if ( it >= n_skip && ( it - n_skip ) % n_dump == 0 ) diag();
    }
    timer.stop();
    std::cout << ansi::bold << "Done!\n" << ansi::reset;
}

int main( int argc, char *argv[] ) {

    // Line-buffer stdout so progress prints appear immediately when output is
    // redirected to a file (e.g. under srun), instead of being block-buffered.
    std::setvbuf( stdout, nullptr, _IOLBF, 0 );
    std::cout.setf( std::ios::unitbuf );

    // Initialize the gpu device
    gpu_init();

    // Process command line arguments
    int opt;
    int silent = 0;
    std::string test = "weibel";
    std::string param = "";
    while ((opt = getopt(argc, argv, "ht:p:s")) != -1) {
        switch (opt) {
            case 't':
            test = optarg;
            break;
        case 'p':
            param = optarg;
            break;
        case 's':
            silent = 1;
            break;
        case 'h':
        case '?':
            cli_help( argv[0] );
            device::exit(0);
        default:
            cli_help( argv[0] );    
            device::exit(1);
        }
    }
    
    // Print information about the environment
    if ( ! silent ) info();    

    // test_grid();
    // test_laser();
    // test_mov();
    // test_weibel();
    if ( test == "pair" )
        test_boosted_pair( param );
    else if ( test == "np_total" )
        test_np_total();
    else
        test_sparse( param );
}