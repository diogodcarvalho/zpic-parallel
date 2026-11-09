#ifndef SILENT_H_
#define SILENT_H_

#include "simulation.h"

namespace silent {

/**
 * @brief Langmuir-silent start
 *
 * Displaces the (mirrored, quiet) lattice of a Maxwell-Juttner species so that, to
 * linear order, no undamped Langmuir mode (v_phi > c) is excited at t = 0. The
 * velocities are not modified and both particles of a mirror pair are moved together,
 * so the initial current stays exactly 0. Damped modes are left untouched.
 *
 * Must be called after Simulation::add_species() and before Simulation::init_fields()
 * (which must then rebuild E from the displaced charge, i.e. not init=none).
 *
 * @param sim       Simulation (grid, box and charge filter)
 * @param species   Species to displace (uniform density, mirror = true)
 * @param theta     Maxwell-Juttner temperature kT / mc^2
 * @param n0        Species density
 * @param ppc       Number of particles per cell
 */
void displace( Simulation & sim, Species & species, float const theta, float const n0,
    uint2 const ppc );

}

#endif
