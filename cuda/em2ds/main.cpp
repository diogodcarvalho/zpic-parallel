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
    float  dt      = 0.000692965f;       // time step
    uint  n_dump  = 100;                // diagnostic output interval (in number of timesteps)
    float  tmax    = 0.002f;             // total simulation time
    float3 uth    { 0.0f, 0.0f, 0.0f };  // thermal velocity
    float3 ufl    { 0.0f, 0.0f, 0.0f };  // fluid velocity
    
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
        if      ( key == "ntiles"  ) { got = std::sscanf( val.c_str(), "%u,%u",    &ntiles.x, &ntiles.y );   want = 2; }
        else if ( key == "nx"      ) { got = std::sscanf( val.c_str(), "%u,%u",    &nx.x, &nx.y );           want = 2; }
        else if ( key == "dx"      ) { got = std::sscanf( val.c_str(), "%f",       &dx );                    want = 1; }
        else if ( key == "dt"      ) { got = std::sscanf( val.c_str(), "%f",       &dt );                    want = 1; }
        else if ( key == "uth"     ) { got = std::sscanf( val.c_str(), "%f,%f,%f", &uth.x, &uth.y, &uth.z ); want = 3; }
        else if ( key == "ufl"     ) { got = std::sscanf( val.c_str(), "%f,%f,%f", &ufl.x, &ufl.y, &ufl.z ); want = 3; }
        else if ( key == "n_dump"  ) { got = std::sscanf( val.c_str(), "%u",       &n_dump );                want = 1; }
        else if ( key == "tmax"    ) { got = std::sscanf( val.c_str(), "%f",       &tmax );                  want = 1; }
        else if ( key == "dx_part" ) { got = std::sscanf( val.c_str(), "%f",       &dx_part );               want = 1; }
        else if ( key == "spacing" ) { got = std::sscanf( val.c_str(), "%u,%u",    &spacing.x, &spacing.y ); want = 2; }
        else if ( key == "ppc"     ) { got = std::sscanf( val.c_str(), "%u,%u",    &ppc.x, &ppc.y );
                                       if ( got == 1 ) { ppc.y = ppc.x; got = 2; }                          want = 2; }
        else {
            std::cerr << "Unknown parameter key: '" << key << "'\n";
            std::exit(1);
        }
        if ( got != want ) {
            std::cerr << "Invalid value for '" << key << "': '" << val << "'\n";
            std::exit(1);
        }
    }

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
    std::cout << "dt        : " << dt      << '\n';
    std::cout << "n_dump    : " << n_dump  << '\n';
    std::cout << "tmax      : " << tmax    << '\n';
    std::cout << "uth       : " << uth     << '\n';
    std::cout << "ufl       : " << ufl     << '\n';
    
    Simulation sim( ntiles, nx, box, dt );

    // When a ppc is given, fall back to uniform density; otherwise the
    // sparse/lattice profiles place one particle per density site (ppc 1,1).
    bool uniform = ( ppc.x > 0 && ppc.y > 0 );
    if ( ! uniform ) ppc = uint2{ 1, 1 };

    Species electrons("electrons", -1.0f, ppc, true);
    if ( uniform )
        electrons.set_density(
            Density::Uniform(1.0)
        );
    else if ( spacing.x > 0 && spacing.y > 0 )
        electrons.set_density(
            Density::Lattice(1.0, spacing)
        );
    else
        electrons.set_density(
            Density::Sparse(1.0, float2{ dx_part, dx_part })
        );
    electrons.set_udist(
        UDistribution::ThermalCorr( uth, ufl )
    );

    sim.add_species( electrons );

    // Initialize fields via Poisson solve from initial charge density
    sim.init_fields( emf::init_type::poisson );

    // Lambda function for diagnostic output
    auto diag = [ & ]( ) {
        // sim.emf.save(emf::e, fcomp::x);
        // sim.emf.save(emf::e, fcomp::y);
        // sim.emf.save(emf::e, fcomp::z);

        // sim.emf.save(emf::b, fcomp::x);
        // sim.emf.save(emf::b, fcomp::y);
        // sim.emf.save(emf::b, fcomp::z);

        // sim.current.save(fcomp::x);
        // sim.current.save(fcomp::y);
        // sim.current.save(fcomp::z);

        // sim.charge.save();
        // electrons.save_charge();

        electrons.save();
        sim.energy_info();
    };

    Timer timer; timer.start();

    diag();

    while ( sim.get_t() <= tmax ) {
        sim.advance();
        if ( sim.get_iter() % 100 == 0 ) {
            std::cout << "iter = " << sim.get_iter()
                      << ", t = " << sim.get_t() << '\n';
        }
        if ( sim.get_iter() % n_dump == 0 ) {
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
    std::cerr << "  -p <parameters>     Test parameters (string). Purpose will depend on the \n";
    std::cerr << "                      test chosen. Defaults to '2,2,16,16'\n";
    std::cerr << '\n';
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
    test_sparse( param );
}