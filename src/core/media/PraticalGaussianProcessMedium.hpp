#ifndef PRATICALGAUSSIANPROCESSMEDIUM_HPP_
#define PRATICALGAUSSIANPROCESSMEDIUM_HPP_

#include "GaussianProcessMedium.hpp"
#include "math/ConvolutionGaussianProcess.hpp"

#include "primitives/VoxelOctree2.hpp"

#include <optional>


#define ENABLE_PROFILE true
#if ENABLE_PROFILE

#define PROFILE_START_TIMER(timer) 
#define PROFILE_STOP_TIMER(timer, atomic)    

#define PROFILE_COUNT(atomic) atomic++;
#define PROFILE_COUNT2(atomic, k) atomic+=k;
#else

#define PROFILE_START_TIMER(timer)  
#define PROFILE_STOP_TIMER(timer, atomic) 
#define PROFILE_COUNT(atomic)
#define PROFILE_COUNT2(atomic, k)
#endif

namespace Tungsten {

class GaussianProcess;

struct GPContextPratical3D : public GPContext {
    ConvolutionRealization3D real;
    virtual void reset() override {
        
    }
};

struct GPContextPratical1D : public GPContext {
    std::optional<Vec3d> intersection;
    std::optional<Vec3d> gradient;


    //for NEE
    bool enableNEE;
    double gradZ;
    double kDerivative2;
    Vec3d gradientMean;
    Vec3d M;

    Vec3d aniso;

    virtual void reset() override {
        intersection.reset();
        gradient.reset();
    }
};

class PraticalGaussianProcessMedium : public GaussianProcessMedium
{
    struct SVOElement {
        bool isActive;
        float maxSigma;

        float minMean;
        float maxMean;

        uint32_t childIdx;

        uint32_t childCount;
        uint32_t level;

        SVOElement() : isActive(false), minMean(1e9), maxMean(-1e9), maxSigma(0), childIdx(0), childCount(0),level(0) {};

        operator bool() const {
            return isActive;
        }

        void addChild(const SVOElement& child) {
            minMean = std::min(minMean, child.minMean);
            maxMean = std::max(maxMean, child.maxMean);
            maxSigma = std::max(maxSigma, child.maxSigma);
            isActive = true;
            ++childCount;
        }

        void setChildIdx(uint32_t idx) {
            childIdx = idx;
        }

    };

    struct SVOTraceState {
        double lipchitzSegmentEnd;
        double lipchitz;

        double segmentMinUpdate;
        double segmentMinMean;
    };
    static const int SVOMaxLevel = 8;
    VoxelOctree2<SVOMaxLevel, SVOElement, SVOTraceState> _SVO;

    std::shared_ptr<ConvolutionFunction> _convolutionFunction;

    bool _useFast = false;
    bool _fastUseLbSkip = false;
    bool _fastUseSVO = false;
    bool _fastUseLocalSigma = false;
    Mat4f _transformSVO;
    std::shared_ptr<ConvolutionFunctionLUT> _lut;

    bool _use1DNoise = false;
    bool _useSingleRealization = false;

    ConvolutionRealization3D _globalReal;
    Vec3d _boundingMin;
    int _impulsePerCell;
    float _rayMarchStepSize;

    bool _enableNEE;

public:
    PraticalGaussianProcessMedium();
    PraticalGaussianProcessMedium(std::shared_ptr<GaussianProcess> gp, std::vector<std::shared_ptr<PhaseFunction>> phases,
        float materialSigmaA, float materialSigmaS, float density, std::shared_ptr<ConvolutionFunction> convolutionFunction, bool use1DNoise, bool useSingleRealization, Vec3d boundingMin, int impulsePerCell, float rayMarchStepSize) :
        GaussianProcessMedium(gp, phases, materialSigmaA, materialSigmaS, density),_convolutionFunction(convolutionFunction),
        _use1DNoise(use1DNoise),_useSingleRealization(!use1DNoise&&useSingleRealization), _boundingMin(boundingMin), _impulsePerCell(impulsePerCell), _rayMarchStepSize(rayMarchStepSize)
    {}

    virtual void fromJson(JsonPtr value, const Scene& scene) override;
    virtual void loadResources() override;
    virtual rapidjson::Value toJson(Allocator& allocator) const override;

    virtual bool sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state,
        Vec3d& grad) const override;

    virtual bool intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const override;

    
private:
    bool sampleGradient_3D(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state,
        Vec3d& grad) const;

    bool intersectGP_3D(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const;

    bool sampleGradient_1D(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip,
        MediumState& state,
        Vec3d& grad) const;

    bool intersectGP_1D(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const;
    bool intersectGP_Fast(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const;

    bool intersectGP_SVO(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const;

#if(ENABLE_PROFILE)
private:
    mutable std::atomic<uint64_t> lbSkipPoints;
    mutable std::atomic<uint64_t> bundleLBSkipPoints;
    mutable std::atomic<uint64_t> psiEvalPoints;

    mutable std::atomic<double> bundleLBEvalTime;
    mutable std::atomic<double> lbEvalTime;
    mutable std::atomic<double> psiEvalTime;


    mutable std::atomic<double> intersectGPTotalTime;
public:
    //just to have somewhere to log things
    ~PraticalGaussianProcessMedium() {
        std::cout << "\n===============================PROFILE===============================" << '\n';
        std::cout << "Intersect GP Total Time: " << intersectGPTotalTime << '\n';
        std::cout << "Bundle LB Skiped Marching Points: " << bundleLBSkipPoints << '\n';
        std::cout << "LB Skiped Marching Points: " << lbSkipPoints << '\n';
        std::cout << "Evaluate Psi Marching Points: " << psiEvalPoints << '\n';
        std::cout << "Threads' Total LB Evaluation Time: " << lbEvalTime << '\n';
        std::cout << "Threads' Total Bundle LB Evaluation Time: " << bundleLBEvalTime << '\n';
        std::cout << "Threads' Total Psi Evaluation Time: " << psiEvalTime << '\n';
        std::cout << "=====================================================================" << '\n';
    }
#endif
};

}

#endif 
