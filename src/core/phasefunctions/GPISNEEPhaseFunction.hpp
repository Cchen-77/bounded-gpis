#ifndef GPISNEEPHASEFUNCTION_HPP_
#define GPISNEEPHASEFUNCTION_HPP_

#include "PhaseFunction.hpp"

#include "math/TangentFrame.hpp"



namespace Tungsten {
struct GPISNEEInfos {
   Vec3f rd;
   Vec3d M;
   Vec3d gradientMean;
   double gradZ;
   double kDerivative2;
};
class GPISNEEPhaseFunction : public PhaseFunction
{
private:
    std::shared_ptr<PhaseFunction> _phase;
    GPISNEEInfos _info;
public:
    GPISNEEPhaseFunction(std::shared_ptr<PhaseFunction> phase, GPISNEEInfos info) :_phase(phase), _info(info) {};
    virtual Vec3f eval(const Vec3f& wi, const Vec3f& wo, const MediumSample& mediumSample) const override;
    virtual bool sample(PathSampleGenerator& sampler, const Vec3f& wi, const MediumSample& mediumSample, PhaseSample& sample) const override;
    virtual bool invert(WritablePathSampleGenerator &sampler, const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const;
    virtual float pdf(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const override;
};

}

#endif 
