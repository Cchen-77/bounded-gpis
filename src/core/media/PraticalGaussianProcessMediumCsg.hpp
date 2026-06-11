#ifndef PRATICALGAUSSIANPROCESSMEDIUMCSG_HPP_
#define PRATICALGAUSSIANPROCESSMEDIUMCSG_HPP_

#include "GaussianProcessMedium.hpp"
#include "math/ConvolutionGaussianProcess.hpp"

#include "primitives/VoxelOctree2.hpp"

#include <optional>

namespace Tungsten {

class GaussianProcess;

struct GPContextPratical1DCsg : public GPContext {
    std::optional<Vec3d> intersection;
    std::optional<Vec3d> gradientLeft;

    std::optional<Vec3d> gradientRight;

    double valueLeft = 0.;
    double valueRight = 0.;

    bool intersectLeft = false;


    virtual void reset() override {
        intersection.reset();
        gradientLeft.reset();
        gradientRight.reset();
        intersectLeft = false;
    }
};

class PraticalGaussianProcessMediumCsg : public GaussianProcessMedium
{
    struct SVOElement {
        bool isActiveLeft = false;
        bool isActiveRight = false;

        uint32_t childIdx;

        uint32_t childCount;
        uint32_t level;

        SVOElement() : isActiveLeft(false), isActiveRight(false), childIdx(0), childCount(0),level(0) {};

        operator bool() const {
            return isActiveLeft || isActiveRight;
        }

        void addChild(const SVOElement& child) {
            isActiveLeft = true;
            isActiveRight = true;
            ++childCount;
        }

        void setChildIdx(uint32_t idx) {
            childIdx = idx;
        }

    };

    struct SVOTraceState {
       
    };


    static const int SVOMaxLevel = 8;
    VoxelOctree2<SVOMaxLevel, SVOElement, SVOTraceState> _SVO;

    bool _useFast = false;
    bool _fastUseLbSkip = false;
    bool _fastUseSVO = false;
    bool _fastUseLocalSigma = false;
    Mat4f _transformSVO;

    bool _useBernoulliWeights = true;

    bool _useStratifiedImpulses = true;
    bool _useSingleRealization = false;

    Vec3d _boundingMin;
    int _impulsePerCell;
    float _rayMarchStepSize;


    std::shared_ptr<GaussianProcess> _gpLeft;
    std::shared_ptr<GaussianProcess> _gpRight;

    std::shared_ptr<ConvolutionFunctionLUT> _lutLeft;
    std::shared_ptr<ConvolutionFunctionLUT> _lutRight;

    std::shared_ptr<ConvolutionFunction> _convolutionFunctionLeft;
    std::shared_ptr<ConvolutionFunction> _convolutionFunctionRight;

public:
    PraticalGaussianProcessMediumCsg();
    PraticalGaussianProcessMediumCsg(std::shared_ptr<GaussianProcess> gp, std::vector<std::shared_ptr<PhaseFunction>> phases,
        float materialSigmaA, float materialSigmaS, float density, bool use1DNoise, bool useSingleRealization, Vec3d boundingMin, int impulsePerCell, float rayMarchStepSize) :
        GaussianProcessMedium(gp, phases, materialSigmaA, materialSigmaS, density),
        _useSingleRealization(!use1DNoise&&useSingleRealization), _boundingMin(boundingMin), _impulsePerCell(impulsePerCell), _rayMarchStepSize(rayMarchStepSize)
    {}

    virtual void fromJson(JsonPtr value, const Scene& scene) override;
    virtual void loadResources() override;
    virtual rapidjson::Value toJson(Allocator& allocator) const override;

    virtual bool sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state,
        Vec3d& grad) const override;

    virtual bool intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const override;

    
private:
    bool sampleGradient_1D(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state,
        Vec3d& grad) const;

    bool intersectGP_1D(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const;
    bool intersectGP_Fast(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const;

    bool intersectGP_SVO(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const;
};

}

#endif 
