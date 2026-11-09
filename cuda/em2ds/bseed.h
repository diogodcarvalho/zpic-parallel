#ifndef BSEED_H_
#define BSEED_H_

#include "simulation.h"

namespace bseed {

/**
 * @brief Magnetic-silent start
 *
 * Seeds the overdamped (Landau damped, zero frequency) long-wavelength magnetic modes
 * of a quiet (mirror / antithetic, J = 0) Maxwell-Juttner start at their stationary
 * level, so they do not have to grow from B = 0 at the slow rate Gamma_k ~ k^3.
 *
 * Every mode 0 < |k| < kmax gets an independent Gaussian B (both polarizations),
 * carried by a smooth momentum kick of the particles chosen so that the filtered
 * current balances it ( F J = curl B, E_T = 0 ), which leaves the undamped light waves
 * unexcited to second order. Mirror pairs get identical kicks. B is written to the
 * EMF; init=darwin rebuilds the same B from the kicked current.
 *
 * Must be called after Simulation::add_species() (and silent::displace(), if used)
 * and before Simulation::init_fields() (init=poisson or init=darwin).
 *
 * @param sim       Simulation (grid, box, current filter and EMF)
 * @param species   Species to kick (uniform density, quiet current load)
 * @param theta     Maxwell-Juttner temperature kT / mc^2
 * @param n0        Species density
 * @param ppc       Number of particles per cell
 * @param kmax      Largest seeded |k| [omega_p / c] (0 = Omega / sqrt(3), light share 1/4)
 * @param seed      RNG seed offset
 */
void kick( Simulation & sim, Species & species, float const theta, float const n0,
    uint2 const ppc, float kmax, unsigned int const seed );

}

#endif
