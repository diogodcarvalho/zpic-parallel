#ifndef UDIST_H_
#define UDIST_H_

#include "zpic.h"

#include <iostream>
#include "particles.h"

namespace UDistribution {

    class Type {
        public:
        virtual Type * clone() const = 0;
        virtual void set( Particles & part, unsigned int seed ) const = 0;
        virtual ~Type() = default;
    };
    
    class None : public Type {
        public:
        None * clone() const override { return new None(); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    class Cold : public Type {
        public:
        const float3 ufl;
        Cold( float3 const ufl ) : ufl(ufl) {};
        Cold * clone() const override { return new Cold(ufl); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    class Thermal : public Type {
        public:
        const float3 uth;
        const float3 ufl;
        Thermal( float3 const uth, float3 const ufl ) : uth(uth), ufl(ufl) {};

        Thermal * clone() const override { return new Thermal(uth, ufl); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    class ThermalCorr : public Type {
        
        public:
        const float3 uth;
        const float3 ufl;
        const int npmin;
        ThermalCorr( float3 const uth, float3 const ufl, int const npmin = 2 ) : uth(uth), ufl(ufl), npmin(npmin) {
            if ( npmin <= 1 ) {
                std::cout << "(*error*) invalid npmin parameter, must be > 1\n";
                exit(1);
            }
        };

        ThermalCorr * clone() const override { return new ThermalCorr(uth, ufl,npmin); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    class MaxwellJuttner : public Type {

        public:
        const float theta;    // normalized temperature kT/(mc^2)
        // TODO: add drift (ufl) support via the Zenitani flipping method.
        //       Only the rest-frame (zero net drift) case is implemented for now.
        MaxwellJuttner( float const theta ) : theta(theta) {
            if ( theta <= 0 ) {
                std::cout << "(*error*) MaxwellJuttner requires theta > 0\n";
                exit(1);
            }
        };

        MaxwellJuttner * clone() const override { return new MaxwellJuttner(theta); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    class MaxwellJuttnerCorr : public Type {

        public:
        const float theta;    // normalized temperature kT/(mc^2)
        const uint2 ppc;      // particles per cell, must be even (antithetic pairs)
        MaxwellJuttnerCorr( float const theta, uint2 const ppc ) : theta(theta), ppc(ppc) {
            if ( theta <= 0 ) {
                std::cout << "(*error*) MaxwellJuttnerCorr requires theta > 0\n";
                exit(1);
            }
            if ( ( ppc.x * ppc.y ) % 2 ) {
                std::cout << "(*error*) MaxwellJuttnerCorr requires an even number of "
                          << "particles per cell (ppc.x * ppc.y), got "
                          << ppc.x * ppc.y << "\n";
                exit(1);
            }
        };

        MaxwellJuttnerCorr * clone() const override { return new MaxwellJuttnerCorr(theta, ppc); };
        void set( Particles & part, unsigned int seed ) const override ;
    };

    class MaxwellJuttnerCorrGroup : public Type {

        public:
        const float theta;    // normalized temperature kT/(mc^2)
        const uint2 group;    // sub-area size in grid cells, must divide the tile size

        // Antithetic pairs are formed inside a sub-area of `group` cells (e.g. a Debye
        // square) instead of inside a single cell. This is the ppc < 1 loader: with the
        // sparse / lattice density profiles a cell holds 0 or 1 particles, so the pairs
        // must be built over a larger region. Every sub-area must hold an even number of
        // particles; this is checked at run time in set().
        MaxwellJuttnerCorrGroup( float const theta, uint2 const group ) : theta(theta), group(group) {
            if ( theta <= 0 ) {
                std::cout << "(*error*) MaxwellJuttnerCorrGroup requires theta > 0\n";
                exit(1);
            }
            if ( group.x < 1 || group.y < 1 ) {
                std::cout << "(*error*) MaxwellJuttnerCorrGroup requires a group size of at "
                          << "least 1 cell in each direction, got " << group.x << ","
                          << group.y << "\n";
                exit(1);
            }
        };

        MaxwellJuttnerCorrGroup * clone() const override { return new MaxwellJuttnerCorrGroup(theta, group); };
        void set( Particles & part, unsigned int seed ) const override ;
    };
}

#endif