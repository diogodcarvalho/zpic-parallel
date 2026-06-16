#ifndef SIMULATION_H_
#define SIMULATION_H_

#include "zpic.h"

#include "emf.h"
#include "current.h"
#include "species.h"

#include "timer.h"

#include <vector>

class Simulation {

    private:

    unsigned int iter;

    public:

    const uint2 ntiles;
    const uint2 nx;
    const float2 box;
    const float dt;

    EMF emf;
    Current current;
    Charge charge;
    std::vector <Species*> species;

    /**
     * @brief Construct a new Simulation object
     * 
     * @param ntiles    Number of tiles
     * @param nx        Tile grid size
     * @param box       Simulation box size
     * @param dt        Time step
     */
    Simulation( uint2 const ntiles, uint2 const nx, float2 const box, float dt ):
        iter(0), ntiles( ntiles ), nx( nx ), box( box ), dt( dt ), 
        emf( ntiles, nx, box, dt ),
        current( ntiles, nx, box, dt ),
        charge( ntiles, nx, box, dt ) {
    }

    /**
     * @brief Destroy the Simulation object
     * 
     */
    ~Simulation() {
    };

    /**
     * @brief Adds particle species to the simulation
     *
     * @param s     Particle species 
     */
    void add_species( Species & s ) {
        species.push_back( &s );
        int species_id = species.size();
        s.initialize( box, ntiles, nx, dt, species_id );
    }

    /**
     * @brief Gets a pointer to a specific species object
     * 
     * @param name                Species name
     * @return Species const* 
     */
    Species * get_species( std::string name ) {
        unsigned id = 0;
        for( id = 0; id < species.size(); id++ )
            if ( (species[id])->name == name ) break;
        return ( id < species.size() ) ? species[id] : nullptr;
    }

    /**
     * @brief Initialize EM fields at t=0
     *
     * @param type          Initialization type
     * @param darwin_iter   Number of Darwin bootstrap passes (only used for the
     *                      darwin initialization type)
     */
    void init_fields( emf::init_type::type type, unsigned int darwin_iter = 0 ) {

        if ( iter != 0 ) {
            ABORT( "Simulation::init_fields() may only be called at iter = 0" );
        }

        // The spectral field solver and the guard-cell reduction assume periodic
        // boundaries; init_fields only supports fully periodic BCs.
        if ( ! ( emf.E->periodic.x   && emf.E->periodic.y   &&
                 current.J->periodic.x && current.J->periodic.y &&
                 charge.rho->periodic.x && charge.rho->periodic.y ) ) {
            ABORT( "Simulation::init_fields() only supports periodic boundary conditions" );
        }

        // Zero global current and charge
        current.zero( );
        charge.zero( );

        switch ( type ) {
            case emf::init_type::poisson:

                // Deposit initial charge density from all species
                for ( auto & sp : species ) sp -> deposit_charge( *charge.rho );

                // Sum partial deposits across tile boundaries
                charge.rho -> add_from_gc();

                // FFT to k-space (no neutral background, Poisson kernel zeros k=0 mode)
                charge.fft_forward -> transform( *charge.rho, *charge.frho );

                // For now no filtering applied
                // charge.filter -> apply( *frho );

                // Solve Poisson for longitudinal E-field
                emf.poisson_solver( charge );

                break;

            case emf::init_type::darwin:
            {
                // Solve Poisson for the (fixed) longitudinal E-field (fEt = 0 here).
                // Field needs set before Darwin iterations start for push calculation.
                for ( auto & sp : species ) sp -> deposit_charge( *charge.rho );
                charge.rho -> add_from_gc();
                charge.fft_forward -> transform( *charge.rho, *charge.frho );
                emf.poisson_solver( charge );

                // Seed B from the retarded current q v(-dt/2) at x(0), so the
                // first tentative push already sees a physical magnetic field.
                for ( auto & sp : species ) sp -> deposit_darwin_retarded_current( current.J );
                current.J -> add_from_gc();
                current.fft_forward -> transform( *current.J, *current.fJ );
                emf.darwin_solver_B( *current.fJ );

                // Reference background plasma frequency squared (implicit term in
                // the transverse-field shifted Helmholtz solve)
                double wp2 = 0;
                for ( auto & sp : species ) wp2 += sp -> darwin_wp2();

                // Temporary Darwin moment grids (used for init-only).
                // They reuse current.J's guard-cell layout and FFT plans
                vec3grid<float3> A ( ntiles, nx, current.J->gc );
                vec3grid<float3> M1( ntiles, nx, current.J->gc );
                vec3grid<float3> M2( ntiles, nx, current.J->gc );
                basic_grid3<std::complex<float>> fA ( current.fJ->dims );
                basic_grid3<std::complex<float>> fM1( current.fJ->dims );
                basic_grid3<std::complex<float>> fM2( current.fJ->dims );

                // Repeat Darwin solver update for n steps
                for ( unsigned int pass = 0; pass < darwin_iter; pass++ ) {

                    // Zero the moment grids
                    current.zero();
                    A.zero();
                    M1.zero();
                    M2.zero();

                    // Deposit time-centered Darwin moments
                    // (this does a tentative push but particle momenta are not modified)
                    for ( auto & sp : species )
                        sp -> deposit_darwin_moments( emf.E, emf.B, current.J, &A, &M1, &M2 );

                    // Sum partial deposits across tile boundaries
                    current.J -> add_from_gc();
                    A.add_from_gc();
                    M1.add_from_gc();
                    M2.add_from_gc();

                    // FFT moments to k-space
                    current.fft_forward -> transform( *current.J, *current.fJ );
                    current.fft_forward -> transform( A,  fA  );
                    current.fft_forward -> transform( M1, fM1 );
                    current.fft_forward -> transform( M2, fM2 );

                    // Solve for B and the shifted-Helmholtz transverse E
                    double res = emf.darwin_solver(
                        *current.fJ, fA, fM1, fM2, charge, wp2 );

                    std::cout << "(*info*) Darwin init pass " << pass
                              << " : |dEt|/|Et| = " << res << '\n';
                }

                break;
            }

            case emf::init_type::none:
            default:
                break;
        }

        // Reset all iteration counters to 0
        charge.reset_iter();
        current.reset_iter();
        emf.reset_iter();

    }

    /**
     * @brief Advance simulation 1 iteration
     *
     */
    void advance( ) {

        // Zero global current and charge
        current.zero( );
        charge.zero( );

        // Advance all species
        for ( auto & sp : species ) {
            sp -> advance( emf, current, charge );
        }

        // Update current edge values and guard cells
        current.advance( );
        charge.advance();

        // Advance EM fields
        emf.advance( current, charge );

        iter++;
    }

    /**
     * @brief Get current iteration value
     * 
     * @return unsigned int     Iteration
     */
    unsigned int get_iter() { return iter; };

    /**
     * @brief Get current simulation time
     * 
     * @return double   Simulation time
     */
    double get_t() { return iter * double(dt); };

    /**
     * @brief Print global energy diagnostic
     * 
     */
    void energy_info( ) {
        std::cout << "(*info*) Energy at n = " << iter << ", t = " << iter * double(dt)  << '\n';
        double part_ene = 0;
        for ( auto & sp : species ) {
            double kin = sp -> get_energy( );
            std::cout << "(*info*) " << sp -> name << " = " << kin << '\n';
            part_ene += kin;
        }

        if ( species.size() > 1 )
            std::cout << "(*info*) Total particle energy = " << part_ene << '\n';

        double3 ene_E, ene_B;
        emf.get_energy( ene_E, ene_B );
        std::cout << "(*info*) Electric field = " << ene_E.x + ene_E.y + ene_E.z << '\n';
        std::cout << "(*info*) Magnetic field = " << ene_B.x + ene_B.y + ene_B.z << '\n';

        double total = part_ene + ene_E.x + ene_E.y + ene_E.z + ene_B.x + ene_B.y + ene_B.z;
        std::cout << "(*info*) total = " << total << '\n';
    }

    /**
     * @brief Returns total number of particles moved
     * 
     * @return unsigned long long 
     */
    uint64_t get_nmove() {
        uint64_t nmove = 0;
        for ( auto & sp : species ) 
            nmove += sp -> get_nmove();
        return nmove;
    }
};


#endif
