/**
 * @file species.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2022-08-06
 * 
 * @copyright Copyright (c) 2022
 * 
 */

#ifndef SPECIES_H_
#define SPECIES_H_

#include <string>

#include "zpic.h"

#include "particles.h"

#include "emf.h"
#include "current.h"

#include "density.h"
#include "udist.h"

namespace phasespace {
    enum quant { x, y, ux, uy, uz };

    static inline void qinfo( quant q, std::string & name, std::string & label, std::string & units ) {
        switch(q) {
        case x :
            name = "x"; label = "x"; units = "c/\\omega_n";
            break;
        case y :
            name = "y"; label = "y"; units = "c/\\omega_n";
            break;
        case ux :
            name = "ux"; label = "u_x"; units = "c";
            break;
        case uy :
            name = "uy"; label = "u_y"; units = "c";
            break;
        case uz :
            name = "uz"; label = "u_z"; units = "c";
            break;
        }
    }
}

namespace species {
    enum pusher { boris, euler };
    namespace bc {
        enum type { open = 0, periodic, reflecting };
    }
    typedef bnd<bc::type> bc_type;

}

/**
 * @brief Charged particles class
 * 
 */
class Species {

protected:

     /// @brief Unique species identifier
    int id;

    /// @brief Nunber of particles per cell
    uint2 ppc;

    /// @brief reference particle charge
    float q;

    /// @brief Cell dize
    float2 dx;

    /// @brief Simulation box size
    float2 box;

    /// @brief Time step
    float dt;

     /// @brief Iteration
    int iter;

     /// @brief Particle data buffer
    Particles *particles;

    /// @brief Secondary data buffer to speed up some calculations
    Particles *tmp;

    /// @brief Particle tile sort aux. data
    ParticleSort *sort;

    /// @brief Initial density profile
    Density::Profile * density;

    /// @brief Number of particles being injected
    int * np_inj;

    /// @brief Whether to assign and save per-particle tags
    bool add_tag;

    /**
     * @brief Process (physical) boundary conditions
     * 
     */
    void process_bc();

private:

    /// @brief Boundary condition
    species::bc_type bc;

    /// @brief Initial velocity distribution
    UDistribution::Type * udist;

    /// @brief Total species energy on device
    double * d_energy;

    /// @brief Total number of particles moved
    unsigned long long * d_nmove;

    /**
     * @brief Deposit 1D phasespace density
     * 
     * @param d_data    Data buffer
     * @param q         Quantity for axis
     * @param range     Value range
     * @param size      Number of grid points
     */
    void dep_phasespace( float * const d_data, 
        phasespace::quant q, float2 const range, unsigned const size ) const;

    /**
     * @brief Deposit 2D phasespace density
     * 
     * @param d_data    Data buffer
     * @param quant0    axis 0 quantity
     * @param range0    axis 0 value range
     * @param size0     axis 0 number of points
     * @param quant1    axis 1 quantity
     * @param range1    axis 1 value range
     * @param size1     axis 1 number of points
     */
    void dep_phasespace( float * const d_data,
        phasespace::quant quant0, float2 range0, unsigned const size0,
        phasespace::quant quant1, float2 range1, unsigned const size1 ) const;

public:

     /// @brief Species name
    const std::string name;

     /// @brief  Mass over charge ratio
    const float m_q;

    /// @brief Type of particle pusher to use
    species::pusher push_type;

    /// @brief Base RNG seed for the initial velocity distribution
    unsigned int seed = 0;

    /// @brief Load the initial ppc as ppc/2 lattice sites, each holding a ( u, -u ) pair at
    ///        the same position, so the initial current is exactly zero (needs even ppc)
    bool mirror = false;

    /**
     * @brief Construct a new Species object
     * 
     * @param name     Name for the species object (used for diagnostics)
     * @param m_q       Mass over charge ratio
     * @param ppc       Number of particles per cell
     * @param add_tag   If true, assign and save a unique tag per particle
     */
    Species( std::string const name, float const m_q, uint2 const ppc,
        bool const add_tag = false );

    /**
     * @brief Initialize data structures
     * 
     * @param box       Simulation global box size
     * @param ntiles    Number of tiles
     * @param nx        Title grid dimension
     * @param dt        
     * @param id 
     */
    virtual void initialize( float2 const box, uint2 const ntiles, uint2 const nx,
        float const dt, int const id_ );

    /**
     * @brief Destroy the Species object
     * 
     */
    ~Species();

    /**
     * @brief Set the density profile object
     * 
     * @param new_density   New density object to be cloned
     */
    virtual void set_density( Density::Profile const & new_density ) {
        delete density;
        density = new_density.clone();
    }

    /**
     * @brief Get the density object
     * 
     * @return Density::Profile& 
     */
    Density::Profile & get_density() {
        return * density;
    }

    /**
     * @brief Set the velocity distribution object
     * 
     * @param new_udist     New udist object to be cloned
     */
    virtual void set_udist( UDistribution::Type const & new_udist ) {
        delete udist;
        udist = new_udist.clone();
    }

    /**
     * @brief Get the udist object
     * 
     * @return UDistribution::Type& 
     */
    UDistribution::Type & get_udist() {
        return *udist;
    } 

    /**
     * @brief Sets the boundary condition type
     * 
     * @param new_bc 
     */
    void set_bc( species::bc_type new_bc ) {

        // Validate parameters
        if ( (new_bc.x.lower == species::bc::periodic) || (new_bc.x.upper == species::bc::periodic) ) {
            if ( new_bc.x.lower != new_bc.x.upper ) {
                std::cerr << "(*error*) Species boundary type mismatch along x.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to species::bc::periodic.\n";
                exit(1);
            }
        }

        if ( (new_bc.y.lower == species::bc::periodic) || (new_bc.y.upper == species::bc::periodic) ) {
            if ( new_bc.y.lower != new_bc.y.upper ) {
                std::cerr << "(*error*) Species boundary type mismatch along y.\n";
                std::cerr << "(*error*) When choosing periodic boundaries both lower and upper types must be set to species::bc::periodic.\n";
                exit(1);
            }
        }

        // Store new values
        bc = new_bc;

/*
        std::string bc_name[] = {"open", "periodic", "reflecting"};
        std::cout << "(*info*) Species " << name << " boundary conditions\n";
        std::cout << "(*info*) x : [ " << bc_name[ bc.x.lower ] << ", " << bc_name[ bc.x.upper ] << " ]\n";
        std::cout << "(*info*) y : [ " << bc_name[ bc.y.lower ] << ", " << bc_name[ bc.y.upper ] << " ]\n";
*/

        // Set periodic flags on tile grids
        if ( particles ) {
            particles->periodic.x = ( bc.x.lower == species::bc::periodic );
            particles->periodic.y = ( bc.y.lower == species::bc::periodic );
        }
    }

    /**
     * @brief Get the current boundary condition types
     * 
     * @return species::bc_type 
     */
    species::bc_type get_bc( ) { return bc; }

    /**
     * @brief Inject particles in the simulation box
     * 
     */
    virtual void inject();

    /**
     * @brief Inject particles in the specified range of the simulation
     * 
     * @param range     Range in which to inject particles
     */
    virtual void inject( bnd<unsigned int> range );


    /**
     * @brief Gets number of particles that will be injected per tile
     * 
     * @param range     Range in which to inject particles
     * @param np        Device pointer to number of particles to be injected per tile
     */
    virtual void np_inject( bnd<unsigned int> range, int * np );

    /**
     * @brief Advance particle velocities
     * 
     * @param E     Electric field
     * @param B     Magnetic field
     */
    void push( vec3grid<float3> * const E, vec3grid<float3> * const B );

    /**
     * @brief Move particles (advance positions) and deposit current
     * 
     * @param current   Electric current density
     */
    void move( vec3grid<float3> * const current, grid<float> * const charge );

    /**
     * @brief Move particles (advance positions) without depositing current
     * 
     */
    void move( );

    /**
     * @brief Free stream particles 1 timestep
     * 
     * Particles are free-streamed (no momentum update), no current is deposited.
     * Used mostly for debug purposes.
     * 
     */
    virtual void advance();

    /**
     * @brief Free stream particles 1 timestep
     * 
     * Particles are free-streamed (no momentum update) and current is deposited
     * 
     * @param current   Electric current density
     * @param charge    Charge density
     */
    virtual void advance( Current &current, Charge & charge );

    /**
     * @brief Advance particles 1 timestep
     * 
     * Momentum is advanced from EMF fields and current is deposited
     * 
     * @param emf       EM fields
     * @param current   Electric current density
     * @param charge    Charge density
     */
    virtual void advance( EMF const &emf, Current &current, Charge & charge );

    /**
     * @brief Deposit species charge
     *
     * @param charge    Charge density grid
     */
    void deposit_charge( grid<float> &charge ) const;

    /**
     * @brief Deposit the time-centered Darwin moments
     *
     * A tentative Boris push v(t-dt/2) -> v(t+dt/2) is performed using the supplied 
     * fields but the particle momentum is NOT written back. The time-centered velocity 
     * v(t) and acceleration dv/dt(t) are then used to deposit, at the (frozen) particle 
     * position x(t):
     * 
     *  - J  : centered current         q v(t)
     *  - A  : acceleration density     q dv/dt(t)
     *  - M1 : velocity flux diagonal   q (vx^2, vy^2, vz^2)
     *  - M2 : velocity flux off-diag.  q (vx vy, vx vz, vy vz)
     *
     * @param E     Electric field (real space, at current Darwin iteration)
     * @param B     Magnetic field (real space, at current Darwin iteration)
     * @param J     [out] centered current density
     * @param A     [out] acceleration density
     * @param M1    [out] velocity flux tensor (diagonal components)
     * @param M2    [out] velocity flux tensor (off-diagonal components)
     */
    void deposit_darwin_moments(
        vec3grid<float3> * const E, vec3grid<float3> * const B,
        vec3grid<float3> * const J, vec3grid<float3> * const A,
        vec3grid<float3> * const M1, vec3grid<float3> * const M2 ) const;

    /**
     * @brief Deposit the retarded current q v(t-dt/2) at x(t) 
     * 
     * Used only to seed the Darwin magnetic field at t=0.
     *
     * @param J     Current density grid
     */
    void deposit_darwin_retarded_current( vec3grid<float3> * const J ) const;

    /**
     * @brief Deposit the in-plane velocity moments used by the Langmuir-silent start
     *
     * Number weighted (1 per particle, no charge factor) deposit at the particle
     * position, with v = u / gamma:
     *
     *  - M2  : ( vx^2, vx vy, vy^2 )
     *  - M4a : ( vx^4, vx^3 vy, vx^2 vy^2 )
     *  - M4b : ( vx vy^3, vy^4, 0 )
     *
     * @param M2    [out] second order moments
     * @param M4a   [out] fourth order moments (first 3)
     * @param M4b   [out] fourth order moments (last 2)
     */
    void deposit_vmoments( vec3grid<float3> & M2, vec3grid<float3> & M4a,
        vec3grid<float3> & M4b ) const;

    /**
     * @brief Displace all particles by a grid displacement field
     *
     * The (x,y) components of the field, in simulation length units, are
     * interpolated at the particle position with the same quartic weights used
     * by the field push, so particles at the same position (e.g. mirror pairs)
     * get bitwise identical displacements. Particles are re-sorted afterwards.
     *
     * @param xi        Displacement field (guard cells must be up to date)
     * @return float    Maximum |displacement| component, in cells
     */
    float displace( vec3grid<float3> const & xi );

    /**
     * @brief Kick the momenta of all particles by a grid momentum field
     *
     * The field (in units of u = p / m c) is interpolated at the particle position
     * with the same quartic weights used by the field push, so particles at the same
     * position (e.g. mirror pairs) get bitwise identical kicks. Positions are not
     * changed, so no re-sort is needed.
     *
     * @param du        Momentum kick field (guard cells must be up to date)
     * @return float    Maximum |du| component
     */
    float kick( vec3grid<float3> const & du );

    /**
     * @brief Particle averages that set the current response to a momentum kick
     *
     * A kick du changes the velocity by (1/gamma)( 1 - v v ) du, so the current of a
     * smooth kick is q n0 lambda du, with lambda_i = < ( 1 - v_i^2 ) / gamma >. Also
     * returns the kinetic temperature T = < u.v > / 3 (= theta for Maxwell-Juttner).
     *
     * @param lambda    [out] ( < (1 - vx^2)/gamma >, < (1 - vy^2)/gamma >, < (1 - vz^2)/gamma > )
     * @param T         [out] < u.v > / 3
     */
    void kick_response( double3 & lambda, double & T ) const;

    /**
     * @brief Accumulate this species' contribution to the boosted-Coulomb field init
     *
     * Adds, in k-space, the exact frozen field of every macroparticle treated as a
     * uniformly moving point charge (rest-frame Coulomb field boosted to the lab):
     *
     *   phi_i = q e^{-i k.x_i} / ( k^2 - (beta_i.k)^2 )
     *   E_i   = -i [ k - (beta_i.k) beta_i ] phi_i
     *   B_i   =  i ( k x beta_i ) phi_i
     *
     * summed over all particles into fE (full E) and fB (B). The k=0 mode is left
     * untouched. The quartic shape factor and digital filter are applied afterwards
     * (EMF::boosted_solver / the init driver).
     *
     * @warning O(N_particles x N_grid): one full k-grid pass per particle. Fine for
     *          single particles / beams, but slow for large thermal plasmas. A future
     *          faster path (velocity-group binning, or a low-beta moment expansion of
     *          1/(k^2 - (beta.k)^2)) would avoid the per-particle loop.
     *
     * @param fE    [in,out] k-space full E-field accumulator
     * @param fB    [in,out] k-space B-field accumulator
     */
    void deposit_boosted_fields(
        basic_grid3<std::complex<float>> & fE,
        basic_grid3<std::complex<float>> & fB ) const;

    /**
     * @brief Per-species contribution to the background plasma frequency squared
     *
     * Returns n0 / |m_q| = n0 (q_phys^2 / m_phys), the physical plasma
     * frequency squared, used as the implicit reference term omega_p0^2 in the
     * Darwin transverse-field (shifted Helmholtz) solve.
     *
     * NOTE: this must use the *physical* charge-to-mass ratio (1/m_q), not the
     * macroparticle charge `q`. The self-force response the shifted-Helmholtz
     * term cancels is the deposit of q_p * dv/dt summed over particles, which
     * equals (sum q_p)/m_q = n0/|m_q|, independent of how the density is split
     * into macroparticles. This makes wp2 correct for the < 1ppc (Sparse /
     * Lattice) case too, where the macroparticle charge `q` is not n0/ppc.
     * Using `q` here scaled wp2 by the macroparticle charge, making the
     * implicit term ~q too small (dense case) and diverging the iteration.
     *
     * With `gamma_corr` the result is multiplied by the measured < 1/gamma >:
     * relativistically the velocity response to the field is dv/dt =
     * (1/gamma)(1/m_q)[ E - v (v.E) ], so the response the implicit term must
     * cancel is reduced by 1/gamma per particle. For a warm/relativistic load
     * the uncorrected wp2 overshoots the response by <gamma>, which leaves the
     * fixed-point iteration with a gain of ~(1 - <1/gamma>) at long wavelengths
     * and slows convergence (it does NOT change the converged solution: at the
     * fixed point the wp2 terms cancel and k^2 E_T + dJ_T/dt = 0 either way).
     *
     * @param gamma_corr    Include the relativistic < 1/gamma > correction
     * @return double
     */
    double darwin_wp2( bool gamma_corr = false ) const;

    /**
     * @brief Particle average of the inverse Lorentz factor, < 1/gamma >
     *
     * @return double   < 1/gamma > (1 for a cold species, < 1 otherwise)
     */
    double mean_rgamma() const;

    /**
     * @brief Returns total time centered kinetic energy
     * 
     * @return double 
     */
    double get_energy() const {
        double energy;
        device::memcpy_tohost( &energy, d_energy, 1 );

        // Normalize enery value and return
        return energy * q * m_q * dx.x * dx.y;
    }

    /**
     * @brief Returns total number of particles moved
     * 
     * @return uint64_t
     */
    unsigned long long get_nmove() const {
        unsigned long long nmove;
        device::memcpy_tohost( &nmove, d_nmove, 1 );

        return nmove;
    }

    /**
     * @brief Returns the maximum number of particles per tile
     * 
     * @return auto 
     */
    uint32_t np_max_tile() const {
        return particles -> np_max_tile();
    }

    /**
     * @brief Returns the total number of particles
     * 
     * @return auto 
     */
    uint64_t np_total() const {
        return particles -> np_total();
    }

    /**
     * @brief Save particle data to file
     * 
     * @brief Saves positions and velocities for all particles in simulation units
     */
    void save() const;

    /**
     * @brief Save charge density for species to file
     * 
     */
    void save_charge() const;

    /**
     * @brief Save 1D phasespace density to file
     * 
     * @param quant     Phasespace quantity
     * @param range     Value range
     * @param size      Number of grid points
     */
    void save_phasespace ( 
        phasespace::quant quant, float2 const range, int const size ) const;

    /**
     * @brief Save 2D phasespace density to file
     * 
     * @param quant0    axis 0 quantity
     * @param range0    axis 0 value range
     * @param size0     axis 0 number of points
     * @param quant1    axis 1 quantity
     * @param range1    axis 1 value range
     * @param size1     axis 1 number of points
     */
    void save_phasespace ( 
        phasespace::quant quant0, float2 const range0, int const size0,
        phasespace::quant quant1, float2 const range1, int const size1 ) const;

};


#endif