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
}

#endif