#include "PraticalGaussianProcessMediumCsg.hpp"

#include "sampling/UniformPathSampler.hpp"
#include "bsdfs/Microfacet.hpp"
#include <bsdfs/NDFs/beckmann.h>
#include <bsdfs/NDFs/GGX.h>

#include "Timer.hpp"
namespace Tungsten {

    PraticalGaussianProcessMediumCsg::PraticalGaussianProcessMediumCsg()
	{
	}

	void PraticalGaussianProcessMediumCsg::fromJson(JsonPtr value, const Scene& scene)
	{

		GaussianProcessMedium::fromJson(value, scene);

        
        auto csg = std::static_pointer_cast<GPSampleNodeCSG>(_gp);
        _gpLeft = std::static_pointer_cast<GaussianProcess>(csg->_left);
        _gpRight = std::static_pointer_cast<GaussianProcess>(csg->_right);

        _convolutionFunctionLeft = _gpLeft->_cov->getConvolutionFunction();
        _convolutionFunctionRight = _gpRight->_cov->getConvolutionFunction();
        
        value.getField("fast", _useFast);
        value.getField("fast_lb_skip", _fastUseLbSkip);
        value.getField("fast_SVO", _fastUseSVO);
        value.getField("fast_SVO_local_sigma", _fastUseLocalSigma);
        value.getField("bernoulli_weights", _useBernoulliWeights);
        value.getField("stratified_impulses", _useStratifiedImpulses);
		value.getField("single_realization", _useSingleRealization);
		value.getField("bounding_min", _boundingMin);
		value.getField("impulse_per_cell", _impulsePerCell);

		value.getField("step_size", _rayMarchStepSize);
	

        if (_useFast && _fastUseLbSkip) {
            _lutLeft = _convolutionFunctionLeft->makeLUT(_impulsePerCell);
            _lutRight = _convolutionFunctionRight->makeLUT(_impulsePerCell);
        }

        if (_useFast && _fastUseSVO) {

            if (auto box = value["processBox"]) {
                box.getField("transform", _transformSVO);
                //additional transform from [0,1]^3 to [-0.5,0.5]^3
                _transformSVO = _transformSVO * Mat4f::translate(Vec3f(-.5f, -.5f, -.5f));
            }
            else {
                _fastUseSVO = false;
                std::cout << "can't not found field \"processBox\", disable SVO!" << '\n';
            }
        }
	}

    void PraticalGaussianProcessMediumCsg::loadResources()
    {
        GaussianProcessMedium::loadResources();
        if (_useFast && _fastUseSVO) {
            std::cout << "Start Building SVO for Fast Practical GPIS Medium..." << '\n';

            std::unique_ptr<SVOElement[]> SVOElements(new SVOElement[1 << (3 * SVOMaxLevel)]);
            uint32_t size = (1 << SVOMaxLevel);

            Vec3f p0 = Vec3f(0.f);
            Vec3f p1 = Vec3f(1.f) / size;
            double maxCellDistance = (_transformSVO.transformPoint(p0) - _transformSVO.transformPoint(p1)).length();

            float maxSigmaLeft = _gpLeft->_cov->getMaxSigma();
            float maxSigmaRight = _gpRight->_cov->getMaxSigma();


            for (int z = 0; z < size; ++z) {
                for (int y = 0; y < size; ++y) {
                    for (int x = 0; x < size; ++x) {
                        Vec3f p = Vec3f((x + .5f) / size, (y + .5f) / size, (z + .5f) / size);
                        p = _transformSVO.transformPoint(p);
                        float meanLeft = _gpLeft->_mean->operator()(Derivative::None, vec_conv<Vec3d>(p), Vec3d());
                        float meanRight = _gpRight->_mean->operator()(Derivative::None, vec_conv<Vec3d>(p), Vec3d());

                        float minMeanLeft = meanLeft - maxCellDistance / 2;
                        float maxMeanLeft = meanLeft + maxCellDistance / 2;

                        float minMeanRight = meanRight - maxCellDistance / 2;
                        float maxMeanRight = meanRight + maxCellDistance / 2;

                        int idx = x + y * size + z * size * size;
                        
                        SVOElements[idx].isActiveLeft = (minMeanLeft < 3 * maxSigmaLeft);
                        SVOElements[idx].isActiveRight = (minMeanRight < 3 * maxSigmaRight);
                    }
                }
            }
            _SVO = VoxelOctree2<SVOMaxLevel, SVOElement, SVOTraceState>(_transformSVO, SVOElements.get());
        }

    }

	rapidjson::Value PraticalGaussianProcessMediumCsg::toJson(Allocator& allocator) const
	{
		auto obj =  JsonObject{ GaussianProcessMedium::toJson(allocator), allocator,
			"type", "pratical_gaussian_process",
            "fast",_useFast,
            "fast_lb_skip",_fastUseLbSkip,
            "fast_svo",_fastUseSVO,
            "bernoulli_weights",_useBernoulliWeights,
            "stratified_impulses",_useStratifiedImpulses,
			"single_realization", _useSingleRealization,
			"bounding_min",_boundingMin,
			"impulse_per_cell",_impulsePerCell,
			"step_size", _rayMarchStepSize
		};
        return obj;
	}

    inline void atomic_add(std::atomic<double>& a, double v) {
        for (double f = a; !a.compare_exchange_weak(f, f + v););
    }

	bool PraticalGaussianProcessMediumCsg::sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip, MediumState& state, Vec3d& grad) const
	{
            return sampleGradient_1D(sampler, ray, ip, state, grad);
     
	}

	bool PraticalGaussianProcessMediumCsg::intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const
	{

        bool res;
        if (_useFast) {
            if (_fastUseSVO) {
                res = intersectGP_SVO(sampler, ray, state, t);
            }
            else {
                res = intersectGP_Fast(sampler, ray, state, t);
            }
        }
        else {
            res =  intersectGP_1D(sampler, ray, state, t);
        }
        return res;
	}

	bool PraticalGaussianProcessMediumCsg::sampleGradient_1D(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip, MediumState& state, Vec3d& grad) const
	{
        GPContextPratical1DCsg& ctxt = *(GPContextPratical1DCsg*)state.gpContext.get();

        auto rd = vec_conv<Vec3d>(ray.dir());

        switch (_normalSamplingMethod) {
        case GPNormalSamplingMethod::FiniteDifferences: // Get rid of this for simplicity
        case GPNormalSamplingMethod::ConditionedGaussian:
        {
            // handled in sampleDistance
            if (ctxt.intersectLeft) {
                grad = ctxt.gradientLeft.value();
            }
            else {
                grad = ctxt.gradientRight.value();
            }
            break;
        }
        }

        return true;
	}
	bool PraticalGaussianProcessMediumCsg::intersectGP_1D(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const
	{
        auto ctxt = std::dynamic_pointer_cast<GPContextPratical1DCsg>(state.gpContext);
        auto newCtxt = std::make_shared<GPContextPratical1DCsg>();


        Vec3d ro = vec_conv<Vec3d>(ray.pos());
        Vec3d rd = vec_conv<Vec3d>(ray.dir());

        auto realLeft = ConvolutionRealization1D::sample(ray, _gpLeft, _convolutionFunctionLeft, _impulsePerCell,sampler);
        auto realRight = ConvolutionRealization1D::sample(ray, _gpRight, _convolutionFunctionRight, _impulsePerCell, sampler);

        bool conditioned = ctxt && (ctxt->intersection || ctxt->gradientLeft || ctxt->gradientRight);
        int nPoints = 1;

        TangentFrame rayFrame(ray.dir());

        if (conditioned) {
            Eigen::VectorXd uLeft;
            Eigen::VectorXd uRight;
            std::array<Vec3d, 4> points = { ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value() };
            std::array<Derivative, 4> derivs = { Derivative::None, Derivative::First, Derivative::First, Derivative::First };
            std::array<Vec3d, 4> ddirs = { Vec3d{},vec_conv<Vec3d>(rayFrame.tangent),vec_conv<Vec3d>(rayFrame.bitangent),vec_conv<Vec3d>(rayFrame.normal) };


            switch (_ctxt)
            {
            case GPCorrelationContext::Goldfish:
            {
                nPoints = 4;

                Eigen::VectorXd vLeft(4), vRight(4);
                Eigen::VectorXd psiLeft(4), psiRight(4);

                /* =======================
                 * Left side
                 * ======================= */
                Vec3f localGradientLeft =
                    rayFrame.toLocal(vec_conv<Vec3f>(ctxt->gradientLeft.value()));
                Vec3f dmeanLocalLeft =
                    rayFrame.toLocal(vec_conv<Vec3f>(_gpLeft->_mean->dmean_da(ro)));

                vLeft(0) = ctxt->valueLeft;
                vLeft(1) = localGradientLeft.x();
                vLeft(2) = localGradientLeft.y();
                vLeft(3) = localGradientLeft.z();

                double covDerivative2Left =
                    -_gpLeft->_cov->evalDerivative2_Isotropic(ctxt->intersection.value(), 0);

                auto sampleLeft = rand_normal_2(sampler);

                psiLeft(0) =
                    _gpLeft->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0)
                    + realLeft.evaluate(0);

                psiLeft(1) =
                    std::sqrt(covDerivative2Left) * sampleLeft[0]
                    + dmeanLocalLeft.x();

                psiLeft(2) =
                    std::sqrt(covDerivative2Left) * sampleLeft[1]
                    + dmeanLocalLeft.y();

                psiLeft(3) =
                    realLeft.evaluateGradient(0)
                    + dmeanLocalLeft.z();


                /* =======================
                 * Right side
                 * ======================= */
                Vec3f localGradientRight =
                    rayFrame.toLocal(vec_conv<Vec3f>(ctxt->gradientRight.value()));
                Vec3f dmeanLocalRight =
                    rayFrame.toLocal(vec_conv<Vec3f>(_gpRight->_mean->dmean_da(ro)));

                vRight(0) = ctxt->valueRight;
                vRight(1) = localGradientRight.x();
                vRight(2) = localGradientRight.y();
                vRight(3) = localGradientRight.z();

                double covDerivative2Right =
                    -_gpRight->_cov->evalDerivative2_Isotropic(ctxt->intersection.value(), 0);

                auto sampleRight = rand_normal_2(sampler);

                psiRight(0) =
                    _gpRight->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0)
                    + realRight.evaluate(0);

                psiRight(1) =
                    std::sqrt(covDerivative2Right) * sampleRight[0]
                    + dmeanLocalRight.x();

                psiRight(2) =
                    std::sqrt(covDerivative2Right) * sampleRight[1]
                    + dmeanLocalRight.y();

                psiRight(3) =
                    realRight.evaluateGradient(0)
                    + dmeanLocalRight.z();


                /* =======================
                 * Solve
                 * ======================= */
                Eigen::MatrixXd kCCLeft =
                    _gpLeft->cov_sym(points.data(), derivs.data(), nullptr, rd, nPoints);

                Eigen::MatrixXd kCCRight =
                    _gpRight->cov_sym(points.data(), derivs.data(), nullptr, rd, nPoints);

                uLeft = kCCLeft.ldlt().solve(vLeft - psiLeft);
                uRight = kCCRight.ldlt().solve(vRight - psiRight);
                break;
            }
            }

            Derivative derivNone = Derivative::None;
            Derivative derivFirst = Derivative::First;

            double pfL = 0., pfR = 0., pf = 0.;
            t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pt = t;

            while (t < ray.farT()) {
                Vec3d p = ro + rd * t;

                double meanL =
                    _gpLeft->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                double updateL =
                    (_gpLeft->cov(&p, points.data(),
                        &derivNone, derivs.data(),
                        nullptr, ddirs.data(), rd, 1, nPoints) * uLeft)(0);

                double fL = meanL + realLeft.evaluate(t) + updateL;

     
                double meanR =
                    _gpRight->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                double updateR =
                    (_gpRight->cov(&p, points.data(),
                        &derivNone, derivs.data(),
                        nullptr, ddirs.data(), rd, 1, nPoints) * uRight)(0);

                double fR = meanR + realRight.evaluate(t) + updateR;

                bool useLeft = fL < fR;
                double f_c = useLeft ? fL : fR;

                if (f_c < 0.0) {
                    
                    t = lerp(pt, t, pf / (pf - f_c));
                    Vec3d intersection = ro + rd * t;
                    newCtxt->intersection = intersection;

                    newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                    newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));

                    if (useLeft)
                    {
                        newCtxt->intersectLeft = true;
                        state.lastGPId = _gpLeft->_id;
                    }
                    else {
                        newCtxt->intersectLeft = false;
                        state.lastGPId = _gpRight->_id;
                    }

                    TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                        vec_conv<Eigen::Vector3d>(rd));

                    double gradZLeft =
                        realLeft.evaluateGradient(t) +
                        (_gpLeft->cov(&intersection, points.data(),
                            &derivFirst, derivs.data(),
                            nullptr, ddirs.data(), rd, 1, nPoints) * uLeft)(0);

                    double gradZRight =
                        realRight.evaluateGradient(t) +
                        (_gpRight->cov(&intersection, points.data(),
                            &derivFirst, derivs.data(),
                            nullptr, ddirs.data(), rd, 1, nPoints) * uRight)(0);

                    double covDerivative2Left =
                        -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                    double covDerivative2Right =
                        -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                    auto xiLeft = rand_normal_2(sampler);
                    auto xiRight = rand_normal_2(sampler);

                    Vec3d localGradLeft{
                        std::sqrt(covDerivative2Left) * xiLeft[0],
                        std::sqrt(covDerivative2Left) * xiLeft[1],
                        gradZLeft
                    };

                    Vec3d localGradRight{
                        std::sqrt(covDerivative2Right) * xiRight[0],
                        std::sqrt(covDerivative2Right) * xiRight[1],
                        gradZRight
                    };

                    newCtxt->gradientLeft =
                        vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                        + _gpLeft->_mean->dmean_da(intersection);

                    newCtxt->gradientRight =
                        vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                        + _gpRight->_mean->dmean_da(intersection);

                    state.gpContext = newCtxt;
                    return true;
                }

                pf = f_c;
                pfL = fL;
                pfR = fR;
                pt = t;
                t += _rayMarchStepSize;
            }

        }
        else {
            Derivative derivNone = Derivative::None;
            Derivative derivFirst = Derivative::First;

            double pfL = 0., pfR = 0., pf = 0.;
            t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pt = t;

            while (t < ray.farT()) {
                Vec3d p = ro + rd * t;

                double meanL =
                    _gpLeft->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);

                double fL = meanL + realLeft.evaluate(t);


                double meanR =
                    _gpRight->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);

                double fR = meanR + realRight.evaluate(t);

                bool useLeft = fL < fR;
                double f_c = useLeft ? fL : fR;

                if (f_c < 0.0) {

                    t = lerp(pt, t, pf / (pf - f_c));
                    Vec3d intersection = ro + rd * t;
                    newCtxt->intersection = intersection;

                    newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                    newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));

                    if (useLeft)
                    {
                        newCtxt->intersectLeft = true;
                        state.lastGPId = _gpLeft->_id;
                    }
                    else {
                        newCtxt->intersectLeft = false;
                        state.lastGPId = _gpRight->_id;
                    }

                    TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                        vec_conv<Eigen::Vector3d>(rd));

                    double gradZLeft =
                        realLeft.evaluateGradient(t);
                    double gradZRight =
                        realRight.evaluateGradient(t);

                    double covDerivative2Left =
                        -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                    double covDerivative2Right =
                        -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                    auto xiLeft = rand_normal_2(sampler);
                    auto xiRight = rand_normal_2(sampler);

                    Vec3d localGradLeft{
                        std::sqrt(covDerivative2Left) * xiLeft[0],
                        std::sqrt(covDerivative2Left) * xiLeft[1],
                        gradZLeft
                    };

                    Vec3d localGradRight{
                        std::sqrt(covDerivative2Right) * xiRight[0],
                        std::sqrt(covDerivative2Right) * xiRight[1],
                        gradZRight
                    };

                    newCtxt->gradientLeft =
                        vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                        + _gpLeft->_mean->dmean_da(intersection);

                    newCtxt->gradientRight =
                        vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                        + _gpRight->_mean->dmean_da(intersection);

                    state.gpContext = newCtxt;
                    return true;
                }

                pf = f_c;
                pfL = fL;
                pfR = fR;
                pt = t;
                t += _rayMarchStepSize;
            }
        }

        state.gpContext = newCtxt;
        t = ray.farT();
        return false;
	}

    bool PraticalGaussianProcessMediumCsg::intersectGP_Fast(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const
    {
        auto ctxt = std::dynamic_pointer_cast<GPContextPratical1DCsg>(state.gpContext);
        auto newCtxt = std::make_shared<GPContextPratical1DCsg>();


        Vec3d ro = vec_conv<Vec3d>(ray.pos());
        Vec3d rd = vec_conv<Vec3d>(ray.dir());

        auto realLeft = FastConvolutionRealization1D::sample(ray, _gpLeft, _convolutionFunctionLeft, _lutLeft, _impulsePerCell, sampler);
        auto realRight = FastConvolutionRealization1D::sample(ray, _gpRight, _convolutionFunctionRight, _lutRight, _impulsePerCell, sampler);

        bool conditioned = ctxt && (ctxt->intersection || ctxt->gradientLeft || ctxt->gradientRight);
        int nPoints = 1;

        TangentFrame rayFrame(ray.dir());

        if (conditioned) {
            Eigen::VectorXd uLeft;
            Eigen::VectorXd uRight;
            std::array<Vec3d, 4> points = { ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value() };
            std::array<Derivative, 4> derivs = { Derivative::None, Derivative::First, Derivative::First, Derivative::First };
            std::array<Vec3d, 4> ddirs = { Vec3d{},vec_conv<Vec3d>(rayFrame.tangent),vec_conv<Vec3d>(rayFrame.bitangent),vec_conv<Vec3d>(rayFrame.normal) };


            switch (_ctxt)
            {
            case GPCorrelationContext::Goldfish:
            {
                nPoints = 4;

                Eigen::VectorXd vLeft(4), vRight(4);
                Eigen::VectorXd psiLeft(4), psiRight(4);

                /* =======================
                 * Left side
                 * ======================= */
                Vec3f localGradientLeft =
                    rayFrame.toLocal(vec_conv<Vec3f>(ctxt->gradientLeft.value()));
                Vec3f dmeanLocalLeft =
                    rayFrame.toLocal(vec_conv<Vec3f>(_gpLeft->_mean->dmean_da(ro)));

                vLeft(0) = ctxt->valueLeft;
                vLeft(1) = localGradientLeft.x();
                vLeft(2) = localGradientLeft.y();
                vLeft(3) = localGradientLeft.z();

                double covDerivative2Left =
                    -_gpLeft->_cov->evalDerivative2_Isotropic(ctxt->intersection.value(), 0);

                auto sampleLeft = rand_normal_2(sampler);

                psiLeft(0) =
                    _gpLeft->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0)
                    + realLeft.evaluate(0);

                psiLeft(1) =
                    std::sqrt(covDerivative2Left) * sampleLeft[0]
                    + dmeanLocalLeft.x();

                psiLeft(2) =
                    std::sqrt(covDerivative2Left) * sampleLeft[1]
                    + dmeanLocalLeft.y();

                psiLeft(3) =
                    realLeft.evaluateGradient(0)
                    + dmeanLocalLeft.z();


                /* =======================
                 * Right side
                 * ======================= */
                Vec3f localGradientRight =
                    rayFrame.toLocal(vec_conv<Vec3f>(ctxt->gradientRight.value()));
                Vec3f dmeanLocalRight =
                    rayFrame.toLocal(vec_conv<Vec3f>(_gpRight->_mean->dmean_da(ro)));

                vRight(0) = ctxt->valueRight;
                vRight(1) = localGradientRight.x();
                vRight(2) = localGradientRight.y();
                vRight(3) = localGradientRight.z();

                double covDerivative2Right =
                    -_gpRight->_cov->evalDerivative2_Isotropic(ctxt->intersection.value(), 0);

                auto sampleRight = rand_normal_2(sampler);

                psiRight(0) =
                    _gpRight->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0)
                    + realRight.evaluate(0);

                psiRight(1) =
                    std::sqrt(covDerivative2Right) * sampleRight[0]
                    + dmeanLocalRight.x();

                psiRight(2) =
                    std::sqrt(covDerivative2Right) * sampleRight[1]
                    + dmeanLocalRight.y();

                psiRight(3) =
                    realRight.evaluateGradient(0)
                    + dmeanLocalRight.z();


                /* =======================
                 * Solve
                 * ======================= */
                Eigen::MatrixXd kCCLeft =
                    _gpLeft->cov_sym(points.data(), derivs.data(), nullptr, rd, nPoints);

                Eigen::MatrixXd kCCRight =
                    _gpRight->cov_sym(points.data(), derivs.data(), nullptr, rd, nPoints);

                uLeft = kCCLeft.ldlt().solve(vLeft - psiLeft);
                uRight = kCCRight.ldlt().solve(vRight - psiRight);
                break;
            }
            }

            Derivative derivNone = Derivative::None;
            Derivative derivFirst = Derivative::First;

            double pfL = 0., pfR = 0., pf = 0.;
            t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pt = t;

            bool lastJumpedLeft = false;
            double lastGPMeanLeft = 0. , lastUpdateLeft = 0.;

            bool lastJumpedRight = false;
            double lastGPMeanRight = 0., lastUpdateRight = 0.;


            while (t < ray.farT()) {
                Vec3d p = ro + rd * t;

                double meanL =
                    _gpLeft->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                double updateL =
                    (_gpLeft->cov(&p, points.data(),
                        &derivNone, derivs.data(),
                        nullptr, ddirs.data(), rd, 1, nPoints) * uLeft)(0);


                double meanR =
                    _gpRight->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                double updateR =
                    (_gpRight->cov(&p, points.data(),
                        &derivNone, derivs.data(),
                        nullptr, ddirs.data(), rd, 1, nPoints) * uRight)(0);


                double lbLeft = -1e10;
                double lbRight = -1e10;

               
                if (_fastUseLbSkip) {
                    lbLeft = meanL + realLeft.evaluateLB(t);
                    lbRight = meanR + realRight.evaluateLB(t);
                }

                if (lbLeft + updateL < 0 && lbRight + updateR >= 0) {
                    double fL = meanL + realLeft.evaluate(t) + updateL;
                    double f_c = fL;
                    if (f_c < 0.) {
                        Vec3d intersection = ro + rd * t;
                        newCtxt->intersection = intersection;

                        if (lastJumpedLeft) {
                            pfL = lastGPMeanLeft + realLeft.evaluate(pt) + lastUpdateLeft;
                        }
                        if (lastJumpedRight) {
                            pfR = lastGPMeanRight + realRight.evaluate(pt) + lastUpdateRight;
                        }

                        pf = std::min(pfL, pfR);
    
                        t = lerp(pt, t, pf / (pf - f_c));

                        double fR = meanR + realRight.evaluate(t) + updateR;
                        newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                        newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));

                        
                        newCtxt->intersectLeft = true;
                        state.lastGPId = _gpLeft->_id;
                       
                        TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                            vec_conv<Eigen::Vector3d>(rd));

                        double gradZLeft =
                            realLeft.evaluateGradient(t) +
                            (_gpLeft->cov(&intersection, points.data(),
                                &derivFirst, derivs.data(),
                                nullptr, ddirs.data(), rd, 1, nPoints) * uLeft)(0);

                        double gradZRight =
                            realRight.evaluateGradient(t) +
                            (_gpRight->cov(&intersection, points.data(),
                                &derivFirst, derivs.data(),
                                nullptr, ddirs.data(), rd, 1, nPoints) * uRight)(0);

                        double covDerivative2Left =
                            -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                        double covDerivative2Right =
                            -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                        auto xiLeft = rand_normal_2(sampler);
                        auto xiRight = rand_normal_2(sampler);

                        Vec3d localGradLeft{
                            std::sqrt(covDerivative2Left) * xiLeft[0],
                            std::sqrt(covDerivative2Left) * xiLeft[1],
                            gradZLeft
                        };

                        Vec3d localGradRight{
                            std::sqrt(covDerivative2Right) * xiRight[0],
                            std::sqrt(covDerivative2Right) * xiRight[1],
                            gradZRight
                        };

                        newCtxt->gradientLeft =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                            + _gpLeft->_mean->dmean_da(intersection);

                        newCtxt->gradientRight =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                            + _gpRight->_mean->dmean_da(intersection);

                        state.gpContext = newCtxt;
                        return true;
                    }
                    pf = pfL = fL;
                    lastJumpedLeft = false;
                    lastJumpedRight = true;
                }
                else if (lbLeft + updateL >=0 && lbRight + updateR < 0) {
                    double fR = meanR + realRight.evaluate(t) + updateR;
                    double f_c = fR;
                    if (f_c < 0.) {
                       
                        Vec3d intersection = ro + rd * t;
                        newCtxt->intersection = intersection;

                        if (lastJumpedLeft) {
                            pfL = lastGPMeanLeft + realLeft.evaluate(pt) + lastUpdateLeft;
                        }
                        if (lastJumpedRight) {
                            pfR = lastGPMeanRight + realRight.evaluate(pt) + lastUpdateRight;
                        }

                        pf = std::min(pfL, pfR);
                        t = lerp(pt, t, pf / (pf - f_c));

                        double fL = meanL + realLeft.evaluate(t) + updateL;
                        newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                        newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));


                        newCtxt->intersectLeft = false;
                        state.lastGPId = _gpRight->_id;

                        TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                            vec_conv<Eigen::Vector3d>(rd));

                        double gradZLeft =
                            realLeft.evaluateGradient(t) +
                            (_gpLeft->cov(&intersection, points.data(),
                                &derivFirst, derivs.data(),
                                nullptr, ddirs.data(), rd, 1, nPoints) * uLeft)(0);

                        double gradZRight =
                            realRight.evaluateGradient(t) +
                            (_gpRight->cov(&intersection, points.data(),
                                &derivFirst, derivs.data(),
                                nullptr, ddirs.data(), rd, 1, nPoints) * uRight)(0);

                        double covDerivative2Left =
                            -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                        double covDerivative2Right =
                            -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                        auto xiLeft = rand_normal_2(sampler);
                        auto xiRight = rand_normal_2(sampler);

                        Vec3d localGradLeft{
                            std::sqrt(covDerivative2Left) * xiLeft[0],
                            std::sqrt(covDerivative2Left) * xiLeft[1],
                            gradZLeft
                        };

                        Vec3d localGradRight{
                            std::sqrt(covDerivative2Right) * xiRight[0],
                            std::sqrt(covDerivative2Right) * xiRight[1],
                            gradZRight
                        };

                        newCtxt->gradientLeft =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                            + _gpLeft->_mean->dmean_da(intersection);

                        newCtxt->gradientRight =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                            + _gpRight->_mean->dmean_da(intersection);

                        state.gpContext = newCtxt;
                        return true;
                    }
                    pf = pfR = fR;
                    lastJumpedLeft = true;
                    lastJumpedRight = false;
                }
                else if(lbLeft + updateL <0 && lbRight + updateR < 0){
                    double fL = meanL + realLeft.evaluate(t) + updateL;
                    double fR = meanR + realRight.evaluate(t) + updateR;
                    bool useLeft = fL < fR;
                    double f_c = useLeft ? fL : fR;
                    if (f_c < 0.) {
                        
                        Vec3d intersection = ro + rd * t;
                        newCtxt->intersection = intersection;

                        if (lastJumpedLeft) {
                            pfL = lastGPMeanLeft + realLeft.evaluate(pt) + lastUpdateLeft;
                        }
                        if (lastJumpedRight) {
                            pfR = lastGPMeanRight + realRight.evaluate(pt) + lastUpdateRight;
                        }

                        pf = std::min(pfL, pfR);
                        t = lerp(pt, t, pf / (pf - f_c));

                        newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                        newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));

                        if (useLeft)
                        {
                            newCtxt->intersectLeft = true;
                            state.lastGPId = _gpLeft->_id;
                        }
                        else {
                            newCtxt->intersectLeft = false;
                            state.lastGPId = _gpRight->_id;
                        }

                        TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                            vec_conv<Eigen::Vector3d>(rd));

                        double gradZLeft =
                            realLeft.evaluateGradient(t) +
                            (_gpLeft->cov(&intersection, points.data(),
                                &derivFirst, derivs.data(),
                                nullptr, ddirs.data(), rd, 1, nPoints) * uLeft)(0);

                        double gradZRight =
                            realRight.evaluateGradient(t) +
                            (_gpRight->cov(&intersection, points.data(),
                                &derivFirst, derivs.data(),
                                nullptr, ddirs.data(), rd, 1, nPoints) * uRight)(0);

                        double covDerivative2Left =
                            -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                        double covDerivative2Right =
                            -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                        auto xiLeft = rand_normal_2(sampler);
                        auto xiRight = rand_normal_2(sampler);

                        Vec3d localGradLeft{
                            std::sqrt(covDerivative2Left) * xiLeft[0],
                            std::sqrt(covDerivative2Left) * xiLeft[1],
                            gradZLeft
                        };

                        Vec3d localGradRight{
                            std::sqrt(covDerivative2Right) * xiRight[0],
                            std::sqrt(covDerivative2Right) * xiRight[1],
                            gradZRight
                        };

                        newCtxt->gradientLeft =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                            + _gpLeft->_mean->dmean_da(intersection);

                        newCtxt->gradientRight =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                            + _gpRight->_mean->dmean_da(intersection);

                        state.gpContext = newCtxt;
                        return true;
                    }

                    pf = f_c;
                    pfL = fL;
                    pfR = fR;
                    lastJumpedLeft = false;
                    lastJumpedRight = false;
                }
                else {
                    lastJumpedLeft = true;
                    lastJumpedRight = true;

                    pf = std::min(lbLeft + updateL,lbRight + updateR);
                }

                pt = t;
                lastGPMeanRight = meanR;
                lastUpdateRight = updateR;
                lastGPMeanLeft = meanL;
                lastUpdateLeft = updateL;
                t += _rayMarchStepSize;
            }
             
        }
        else {
            Derivative derivNone = Derivative::None;
            Derivative derivFirst = Derivative::First;

            double pfL = 0., pfR = 0., pf = 0.;
            t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pt = t;

            bool lastJumpedLeft = false;
            double lastGPMeanLeft = 0., lastUpdateLeft = 0.;

            bool lastJumpedRight = false;
            double lastGPMeanRight = 0., lastUpdateRight = 0.;


            while (t < ray.farT()) {
                Vec3d p = ro + rd * t;

                double meanL =
                    _gpLeft->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
      

                double meanR =
                    _gpRight->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);



                double lbLeft = -1e10;
                double lbRight = -1e10;


                if (_fastUseLbSkip) {
                    lbLeft = meanL + realLeft.evaluateLB(t);
                    lbRight = meanR + realRight.evaluateLB(t);
                }

                if (lbLeft < 0 && lbRight >= 0) {
                    double fL = meanL + realLeft.evaluate(t);
                    double f_c = fL;
                    if (f_c < 0.) {
                        Vec3d intersection = ro + rd * t;
                        newCtxt->intersection = intersection;

                        if (lastJumpedLeft) {
                            pfL = lastGPMeanLeft + realLeft.evaluate(pt);
                        }
                        if (lastJumpedRight) {
                            pfR = lastGPMeanRight + realRight.evaluate(pt);
                        }

                        pf = std::min(pfL, pfR);
                        t = lerp(pt, t, pf / (pf - f_c));

                        double fR = meanR + realRight.evaluate(t);
                        newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                        newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));


                        newCtxt->intersectLeft = true;
                        state.lastGPId = _gpLeft->_id;

                        TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                            vec_conv<Eigen::Vector3d>(rd));

                        double gradZLeft =
                            realLeft.evaluateGradient(t);

                        double gradZRight =
                            realRight.evaluateGradient(t);

                        double covDerivative2Left =
                            -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                        double covDerivative2Right =
                            -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                        auto xiLeft = rand_normal_2(sampler);
                        auto xiRight = rand_normal_2(sampler);

                        Vec3d localGradLeft{
                            std::sqrt(covDerivative2Left) * xiLeft[0],
                            std::sqrt(covDerivative2Left) * xiLeft[1],
                            gradZLeft
                        };

                        Vec3d localGradRight{
                            std::sqrt(covDerivative2Right) * xiRight[0],
                            std::sqrt(covDerivative2Right) * xiRight[1],
                            gradZRight
                        };

                        newCtxt->gradientLeft =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                            + _gpLeft->_mean->dmean_da(intersection);

                        newCtxt->gradientRight =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                            + _gpRight->_mean->dmean_da(intersection);

                        state.gpContext = newCtxt;
                        return true;
                    }
                    pf = pfL = fL;
                    lastJumpedLeft = false;
                    lastJumpedRight = true;
                }
                else if (lbLeft >= 0 && lbRight < 0) {
                    double fR = meanR + realRight.evaluate(t);
                    double f_c = fR;
                    if (f_c < 0.) {
                        Vec3d intersection = ro + rd * t;
                        newCtxt->intersection = intersection;

                        if (lastJumpedLeft) {
                            pfL = lastGPMeanLeft + realLeft.evaluate(pt);
                        }
                        if (lastJumpedRight) {
                            pfR = lastGPMeanRight + realRight.evaluate(pt);
                        }

                        pf = std::min(pfL, pfR);
                        t = lerp(pt, t, pf / (pf - f_c));

                        double fL = meanL + realLeft.evaluate(t) ;
                        newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                        newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));


                        newCtxt->intersectLeft = false;
                        state.lastGPId = _gpRight->_id;

                        TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                            vec_conv<Eigen::Vector3d>(rd));

                        double gradZLeft =
                            realLeft.evaluateGradient(t);

                        double gradZRight =
                            realRight.evaluateGradient(t);

                        double covDerivative2Left =
                            -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                        double covDerivative2Right =
                            -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                        auto xiLeft = rand_normal_2(sampler);
                        auto xiRight = rand_normal_2(sampler);

                        Vec3d localGradLeft{
                            std::sqrt(covDerivative2Left) * xiLeft[0],
                            std::sqrt(covDerivative2Left) * xiLeft[1],
                            gradZLeft
                        };

                        Vec3d localGradRight{
                            std::sqrt(covDerivative2Right) * xiRight[0],
                            std::sqrt(covDerivative2Right) * xiRight[1],
                            gradZRight
                        };

                        newCtxt->gradientLeft =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                            + _gpLeft->_mean->dmean_da(intersection);

                        newCtxt->gradientRight =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                            + _gpRight->_mean->dmean_da(intersection);

                        state.gpContext = newCtxt;
                        return true;
                    }
                    pf = pfR = fR;
                    lastJumpedLeft = true;
                    lastJumpedRight = false;
                }
                else if (lbLeft  < 0 && lbRight < 0) {
                    double fL = meanL + realLeft.evaluate(t);
                    double fR = meanR + realRight.evaluate(t);
                    bool useLeft = fL < fR;
                    double f_c = useLeft ? fL : fR;
                    if (f_c < 0.) {
                        Vec3d intersection = ro + rd * t;
                        newCtxt->intersection = intersection;

                        if (lastJumpedLeft) {
                            pfL = lastGPMeanLeft + realLeft.evaluate(pt) + lastUpdateLeft;
                        }
                        if (lastJumpedRight) {
                            pfR = lastGPMeanRight + realRight.evaluate(pt) + lastUpdateRight;
                        }

                        pf = std::min(pfL, pfR);
                        t = lerp(pt, t, pf / (pf - f_c));

                        newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                        newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));

                        if (useLeft)
                        {
                            newCtxt->intersectLeft = true;
                            state.lastGPId = _gpLeft->_id;
                        }
                        else {
                            newCtxt->intersectLeft = false;
                            state.lastGPId = _gpRight->_id;
                        }

                        TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                            vec_conv<Eigen::Vector3d>(rd));

                        double gradZLeft =
                            realLeft.evaluateGradient(t);

                        double gradZRight =
                            realRight.evaluateGradient(t);

                        double covDerivative2Left =
                            -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                        double covDerivative2Right =
                            -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                        auto xiLeft = rand_normal_2(sampler);
                        auto xiRight = rand_normal_2(sampler);

                        Vec3d localGradLeft{
                            std::sqrt(covDerivative2Left) * xiLeft[0],
                            std::sqrt(covDerivative2Left) * xiLeft[1],
                            gradZLeft
                        };

                        Vec3d localGradRight{
                            std::sqrt(covDerivative2Right) * xiRight[0],
                            std::sqrt(covDerivative2Right) * xiRight[1],
                            gradZRight
                        };

                        newCtxt->gradientLeft =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                            + _gpLeft->_mean->dmean_da(intersection);

                        newCtxt->gradientRight =
                            vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                            + _gpRight->_mean->dmean_da(intersection);

                        state.gpContext = newCtxt;
                        return true;
                    }

                    pf = f_c;
                    pfL = fL;
                    pfR = fR;
                    lastJumpedLeft = false;
                    lastJumpedRight = false;
                }
                else {
                    lastJumpedLeft = true;
                    lastJumpedRight = true;
                    pf = std::min(lbLeft, lbRight);
                }
                lastGPMeanRight = meanR;
                lastGPMeanLeft = meanL;
                pt = t;
                t += _rayMarchStepSize;
            }
        }

        state.gpContext = newCtxt;
        t = ray.farT();
        return false;
    }
    bool PraticalGaussianProcessMediumCsg::intersectGP_SVO(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const
    {
        auto ctxt = std::dynamic_pointer_cast<GPContextPratical1DCsg>(state.gpContext);
        auto newCtxt = std::make_shared<GPContextPratical1DCsg>();


        Vec3d ro = vec_conv<Vec3d>(ray.pos());
        Vec3d rd = vec_conv<Vec3d>(ray.dir());

        auto realLeft = FastConvolutionRealization1D::sample(ray, _gpLeft, _convolutionFunctionLeft, _lutLeft, _impulsePerCell, sampler);
        auto realRight = FastConvolutionRealization1D::sample(ray, _gpRight, _convolutionFunctionRight, _lutRight, _impulsePerCell, sampler);

        bool conditioned = ctxt && (ctxt->intersection || ctxt->gradientLeft || ctxt->gradientRight);
        int nPoints = 1;

        TangentFrame rayFrame(ray.dir());

        if (conditioned) {
            Eigen::VectorXd uLeft;
            Eigen::VectorXd uRight;
            std::array<Vec3d, 4> points = { ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value() };
            std::array<Derivative, 4> derivs = { Derivative::None, Derivative::First, Derivative::First, Derivative::First };
            std::array<Vec3d, 4> ddirs = { Vec3d{},vec_conv<Vec3d>(rayFrame.tangent),vec_conv<Vec3d>(rayFrame.bitangent),vec_conv<Vec3d>(rayFrame.normal) };


            switch (_ctxt)
            {
            case GPCorrelationContext::Goldfish:
            {
                nPoints = 4;

                Eigen::VectorXd vLeft(4), vRight(4);
                Eigen::VectorXd psiLeft(4), psiRight(4);

                /* =======================
                 * Left side
                 * ======================= */
                Vec3f localGradientLeft =
                    rayFrame.toLocal(vec_conv<Vec3f>(ctxt->gradientLeft.value()));
                Vec3f dmeanLocalLeft =
                    rayFrame.toLocal(vec_conv<Vec3f>(_gpLeft->_mean->dmean_da(ro)));

                vLeft(0) = ctxt->valueLeft;
                vLeft(1) = localGradientLeft.x();
                vLeft(2) = localGradientLeft.y();
                vLeft(3) = localGradientLeft.z();

                double covDerivative2Left =
                    -_gpLeft->_cov->evalDerivative2_Isotropic(ctxt->intersection.value(), 0);

                auto sampleLeft = rand_normal_2(sampler);

                psiLeft(0) =
                    _gpLeft->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0)
                    + realLeft.evaluate(0);

                psiLeft(1) =
                    std::sqrt(covDerivative2Left) * sampleLeft[0]
                    + dmeanLocalLeft.x();

                psiLeft(2) =
                    std::sqrt(covDerivative2Left) * sampleLeft[1]
                    + dmeanLocalLeft.y();

                psiLeft(3) =
                    realLeft.evaluateGradient(0)
                    + dmeanLocalLeft.z();


                /* =======================
                 * Right side
                 * ======================= */
                Vec3f localGradientRight =
                    rayFrame.toLocal(vec_conv<Vec3f>(ctxt->gradientRight.value()));
                Vec3f dmeanLocalRight =
                    rayFrame.toLocal(vec_conv<Vec3f>(_gpRight->_mean->dmean_da(ro)));

                vRight(0) = ctxt->valueRight;
                vRight(1) = localGradientRight.x();
                vRight(2) = localGradientRight.y();
                vRight(3) = localGradientRight.z();

                double covDerivative2Right =
                    -_gpRight->_cov->evalDerivative2_Isotropic(ctxt->intersection.value(), 0);

                auto sampleRight = rand_normal_2(sampler);

                psiRight(0) =
                    _gpRight->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0)
                    + realRight.evaluate(0);

                psiRight(1) =
                    std::sqrt(covDerivative2Right) * sampleRight[0]
                    + dmeanLocalRight.x();

                psiRight(2) =
                    std::sqrt(covDerivative2Right) * sampleRight[1]
                    + dmeanLocalRight.y();

                psiRight(3) =
                    realRight.evaluateGradient(0)
                    + dmeanLocalRight.z();


                /* =======================
                 * Solve
                 * ======================= */
                Eigen::MatrixXd kCCLeft =
                    _gpLeft->cov_sym(points.data(), derivs.data(), nullptr, rd, nPoints);

                Eigen::MatrixXd kCCRight =
                    _gpRight->cov_sym(points.data(), derivs.data(), nullptr, rd, nPoints);

                uLeft = kCCLeft.ldlt().solve(vLeft - psiLeft);
                uRight = kCCRight.ldlt().solve(vRight - psiRight);
                break;
            }
            }

            Derivative derivNone = Derivative::None;
            Derivative derivFirst = Derivative::First;

            double pfL = 0., pfR = 0., pf = 0.;
            double startT = t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pt = t;

            bool lastJumpedLeft = false;
            double lastGPMeanLeft = 0., lastUpdateLeft = 0.;

            bool lastJumpedRight = false;
            double lastGPMeanRight = 0., lastUpdateRight = 0.;


            auto handleRaySegment = [&](SVOElement& element, float minT, float maxT, bool& realCheck, SVOTraceState& traceState) -> bool {

                t = std::max(t, std::ceil((minT - startT) / _rayMarchStepSize) * _rayMarchStepSize + startT);

                if (maxT < t) {
                    return false;
                }


                Derivative derivNone = Derivative::None;
                Derivative derivFirst = Derivative::First;
                int gridLevel = 0;


                if (element.level > 0) {
                    return true;
                }
          

                realCheck = true;

                if (pt < t - _rayMarchStepSize) {
                    pt = t - _rayMarchStepSize;
                    lastJumpedLeft = lastJumpedRight = true;

                    Vec3d pp = ro + rd * pt;
                    lastGPMeanLeft = _gpLeft->mean_prior(&pp, &derivNone, nullptr, {}, 1)(0);
                    lastUpdateLeft = (_gpLeft->cov(&pp, points.data(), &derivNone, derivs.data(), nullptr, ddirs.data(), rd, 1, nPoints) * uLeft)(0);

                    lastGPMeanRight = _gpRight->mean_prior(&pp, &derivNone, nullptr, {}, 1)(0);
                    lastUpdateRight = (_gpRight->cov(&pp, points.data(), &derivNone, derivs.data(), nullptr, ddirs.data(), rd, 1, nPoints) * uRight)(0);
                }

                while (t < maxT) {
                    Vec3d p = ro + rd * t;

                    double meanL =
                        _gpLeft->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                    double updateL =
                        (_gpLeft->cov(&p, points.data(),
                            &derivNone, derivs.data(),
                            nullptr, ddirs.data(), rd, 1, nPoints) * uLeft)(0);


                    double meanR =
                        _gpRight->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                    double updateR =
                        (_gpRight->cov(&p, points.data(),
                            &derivNone, derivs.data(),
                            nullptr, ddirs.data(), rd, 1, nPoints) * uRight)(0);


                    double lbLeft = -1e10;
                    double lbRight = -1e10;


                    if (_fastUseLbSkip) {
                        if (element.isActiveLeft) {
                            lbLeft = meanL + realLeft.evaluateLB(t);
                        }
                        if (element.isActiveRight) {
                            lbRight = meanR + realRight.evaluateLB(t);
                        }
                    }

                    if (!element.isActiveLeft) {
                        lbLeft = 1e10;
                    }
                    if (!element.isActiveRight) {
                        lbRight = 1e10;
                    }

                    if (lbLeft + updateL < 0 && lbRight + updateR >= 0) {
                        double fL = meanL + realLeft.evaluate(t) + updateL;
                        double f_c = fL;
                        if (f_c < 0.) {
                            Vec3d intersection = ro + rd * t;
                            newCtxt->intersection = intersection;

                            if (lastJumpedLeft) {
                                pfL = lastGPMeanLeft + realLeft.evaluate(pt) + lastUpdateLeft;
                            }
                            if (lastJumpedRight) {
                                pfR = lastGPMeanRight + realRight.evaluate(pt) + lastUpdateRight;
                            }

                            pf = std::min(pfL, pfR);

                            t = lerp(pt, t, pf / (pf - f_c));

                            double fR = meanR + realRight.evaluate(t) + updateR;
                            newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                            newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));


                            newCtxt->intersectLeft = true;
                            state.lastGPId = _gpLeft->_id;

                            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                                vec_conv<Eigen::Vector3d>(rd));

                            double gradZLeft =
                                realLeft.evaluateGradient(t) +
                                (_gpLeft->cov(&intersection, points.data(),
                                    &derivFirst, derivs.data(),
                                    nullptr, ddirs.data(), rd, 1, nPoints) * uLeft)(0);

                            double gradZRight =
                                realRight.evaluateGradient(t) +
                                (_gpRight->cov(&intersection, points.data(),
                                    &derivFirst, derivs.data(),
                                    nullptr, ddirs.data(), rd, 1, nPoints) * uRight)(0);

                            double covDerivative2Left =
                                -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                            double covDerivative2Right =
                                -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                            auto xiLeft = rand_normal_2(sampler);
                            auto xiRight = rand_normal_2(sampler);

                            Vec3d localGradLeft{
                                std::sqrt(covDerivative2Left) * xiLeft[0],
                                std::sqrt(covDerivative2Left) * xiLeft[1],
                                gradZLeft
                            };

                            Vec3d localGradRight{
                                std::sqrt(covDerivative2Right) * xiRight[0],
                                std::sqrt(covDerivative2Right) * xiRight[1],
                                gradZRight
                            };

                            newCtxt->gradientLeft =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                                + _gpLeft->_mean->dmean_da(intersection);

                            newCtxt->gradientRight =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                                + _gpRight->_mean->dmean_da(intersection);

                            state.gpContext = newCtxt;
                            return true;
                        }
                        pf = pfL = fL;
                        lastJumpedLeft = false;
                        lastJumpedRight = true;
                    }
                    else if (lbLeft + updateL >= 0 && lbRight + updateR < 0) {
                        double fR = meanR + realRight.evaluate(t) + updateR;
                        double f_c = fR;
                        if (f_c < 0.) {

                            Vec3d intersection = ro + rd * t;
                            newCtxt->intersection = intersection;

                            if (lastJumpedLeft) {
                                pfL = lastGPMeanLeft + realLeft.evaluate(pt) + lastUpdateLeft;
                            }
                            if (lastJumpedRight) {
                                pfR = lastGPMeanRight + realRight.evaluate(pt) + lastUpdateRight;
                            }

                            pf = std::min(pfL, pfR);
                            t = lerp(pt, t, pf / (pf - f_c));

                            double fL = meanL + realLeft.evaluate(t) + updateL;
                            newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                            newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));


                            newCtxt->intersectLeft = false;
                            state.lastGPId = _gpRight->_id;

                            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                                vec_conv<Eigen::Vector3d>(rd));

                            double gradZLeft =
                                realLeft.evaluateGradient(t) +
                                (_gpLeft->cov(&intersection, points.data(),
                                    &derivFirst, derivs.data(),
                                    nullptr, ddirs.data(), rd, 1, nPoints) * uLeft)(0);

                            double gradZRight =
                                realRight.evaluateGradient(t) +
                                (_gpRight->cov(&intersection, points.data(),
                                    &derivFirst, derivs.data(),
                                    nullptr, ddirs.data(), rd, 1, nPoints) * uRight)(0);

                            double covDerivative2Left =
                                -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                            double covDerivative2Right =
                                -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                            auto xiLeft = rand_normal_2(sampler);
                            auto xiRight = rand_normal_2(sampler);

                            Vec3d localGradLeft{
                                std::sqrt(covDerivative2Left) * xiLeft[0],
                                std::sqrt(covDerivative2Left) * xiLeft[1],
                                gradZLeft
                            };

                            Vec3d localGradRight{
                                std::sqrt(covDerivative2Right) * xiRight[0],
                                std::sqrt(covDerivative2Right) * xiRight[1],
                                gradZRight
                            };

                            newCtxt->gradientLeft =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                                + _gpLeft->_mean->dmean_da(intersection);

                            newCtxt->gradientRight =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                                + _gpRight->_mean->dmean_da(intersection);

                            state.gpContext = newCtxt;
                            return true;
                        }
                        pf = pfR = fR;
                        lastJumpedLeft = true;
                        lastJumpedRight = false;
                    }
                    else if (lbLeft + updateL < 0 && lbRight + updateR < 0) {
                        double fL = meanL + realLeft.evaluate(t) + updateL;
                        double fR = meanR + realRight.evaluate(t) + updateR;
                        bool useLeft = fL < fR;
                        double f_c = useLeft ? fL : fR;
                        if (f_c < 0.) {

                            Vec3d intersection = ro + rd * t;
                            newCtxt->intersection = intersection;

                            if (lastJumpedLeft) {
                                pfL = lastGPMeanLeft + realLeft.evaluate(pt) + lastUpdateLeft;
                            }
                            if (lastJumpedRight) {
                                pfR = lastGPMeanRight + realRight.evaluate(pt) + lastUpdateRight;
                            }

                            pf = std::min(pfL, pfR);
                            t = lerp(pt, t, pf / (pf - f_c));

                            newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                            newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));

                            if (useLeft)
                            {
                                newCtxt->intersectLeft = true;
                                state.lastGPId = _gpLeft->_id;
                            }
                            else {
                                newCtxt->intersectLeft = false;
                                state.lastGPId = _gpRight->_id;
                            }

                            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                                vec_conv<Eigen::Vector3d>(rd));

                            double gradZLeft =
                                realLeft.evaluateGradient(t) +
                                (_gpLeft->cov(&intersection, points.data(),
                                    &derivFirst, derivs.data(),
                                    nullptr, ddirs.data(), rd, 1, nPoints) * uLeft)(0);

                            double gradZRight =
                                realRight.evaluateGradient(t) +
                                (_gpRight->cov(&intersection, points.data(),
                                    &derivFirst, derivs.data(),
                                    nullptr, ddirs.data(), rd, 1, nPoints) * uRight)(0);

                            double covDerivative2Left =
                                -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                            double covDerivative2Right =
                                -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                            auto xiLeft = rand_normal_2(sampler);
                            auto xiRight = rand_normal_2(sampler);

                            Vec3d localGradLeft{
                                std::sqrt(covDerivative2Left) * xiLeft[0],
                                std::sqrt(covDerivative2Left) * xiLeft[1],
                                gradZLeft
                            };

                            Vec3d localGradRight{
                                std::sqrt(covDerivative2Right) * xiRight[0],
                                std::sqrt(covDerivative2Right) * xiRight[1],
                                gradZRight
                            };

                            newCtxt->gradientLeft =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                                + _gpLeft->_mean->dmean_da(intersection);

                            newCtxt->gradientRight =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                                + _gpRight->_mean->dmean_da(intersection);

                            state.gpContext = newCtxt;
                            return true;
                        }

                        pf = f_c;
                        pfL = fL;
                        pfR = fR;
                        lastJumpedLeft = false;
                        lastJumpedRight = false;
                    }
                    else {
                        lastJumpedLeft = true;
                        lastJumpedRight = true;

                        pf = std::min(lbLeft + updateL, lbRight + updateR);
                    }

                    pt = t;
                    lastGPMeanRight = meanR;
                    lastUpdateRight = updateR;
                    lastGPMeanLeft = meanL;
                    lastUpdateLeft = updateL;
                    t += _rayMarchStepSize;
                }
                return false;
                };

                SVOTraceState traceState{};
                if (_SVO.trace(ray, Vec3f(), ray.nearT(), traceState, handleRaySegment)) {
                    return true;
                }
        }
        else {

            Derivative derivNone = Derivative::None;
            Derivative derivFirst = Derivative::First;

            double pfL = 0., pfR = 0., pf = 0.;
            double startT = t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pt = t;

            bool lastJumpedLeft = false;
            double lastGPMeanLeft = 0.;

            bool lastJumpedRight = false;
            double lastGPMeanRight = 0.;
            
            auto handleRaySegment = [&](SVOElement& element, float minT, float maxT, bool& realCheck, SVOTraceState& traceState) -> bool {

                t = std::max(t, std::ceil((minT - startT) / _rayMarchStepSize) * _rayMarchStepSize + startT);

                if (maxT < t) {
                    return false;
                }


                if (element.level > 0) {
                    return true;
                }

                
                realCheck = true;

                if (pt < t - _rayMarchStepSize) {
                    pt = t - _rayMarchStepSize;
                    lastJumpedLeft = lastJumpedRight = true;

                    Vec3d pp = ro + rd * pt;
                    lastGPMeanLeft = _gpLeft->mean_prior(&pp, &derivNone, nullptr, {}, 1)(0);
                    lastGPMeanRight = _gpRight->mean_prior(&pp, &derivNone, nullptr, {}, 1)(0);
                }

                while (t < maxT) {
                    Vec3d p = ro + rd * t;

                    double meanL =
                        _gpLeft->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);


                    double meanR =
                        _gpRight->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);



                    double lbLeft = -1e10;
                    double lbRight = -1e10;

                    if (_fastUseLbSkip) {
                        if (element.isActiveLeft) {
                            lbLeft = meanL + realLeft.evaluateLB(t);
                        }
                        if(element.isActiveRight) {
                            lbRight = meanR + realRight.evaluateLB(t);
                        }
                    }

                    if (!element.isActiveLeft) {
                        lbLeft = 1e10;
                    }

                    if (!element.isActiveRight) {
                        lbRight = 1e10;
                    }

                    if (lbLeft < 0 && lbRight >= 0) {
                        double fL = meanL + realLeft.evaluate(t);
                        double f_c = fL;
                        if (f_c < 0.) {
                            Vec3d intersection = ro + rd * t;
                            newCtxt->intersection = intersection;

                            if (lastJumpedLeft) {
                                pfL = lastGPMeanLeft + realLeft.evaluate(pt);
                            }
                            if (lastJumpedRight) {
                                pfR = lastGPMeanRight + realRight.evaluate(pt);
                            }

                            pf = std::min(pfL, pfR);
                            t = lerp(pt, t, pf / (pf - f_c));

                            double fR = meanR + realRight.evaluate(t);
                            newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                            newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));


                            newCtxt->intersectLeft = true;
                            state.lastGPId = _gpLeft->_id;

                            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                                vec_conv<Eigen::Vector3d>(rd));

                            double gradZLeft =
                                realLeft.evaluateGradient(t);

                            double gradZRight =
                                realRight.evaluateGradient(t);

                            double covDerivative2Left =
                                -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                            double covDerivative2Right =
                                -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                            auto xiLeft = rand_normal_2(sampler);
                            auto xiRight = rand_normal_2(sampler);

                            Vec3d localGradLeft{
                                std::sqrt(covDerivative2Left) * xiLeft[0],
                                std::sqrt(covDerivative2Left) * xiLeft[1],
                                gradZLeft
                            };

                            Vec3d localGradRight{
                                std::sqrt(covDerivative2Right) * xiRight[0],
                                std::sqrt(covDerivative2Right) * xiRight[1],
                                gradZRight
                            };

                            newCtxt->gradientLeft =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                                + _gpLeft->_mean->dmean_da(intersection);

                            newCtxt->gradientRight =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                                + _gpRight->_mean->dmean_da(intersection);

                            state.gpContext = newCtxt;
                            return true;
                        }
                        pf = pfL = fL;
                        lastJumpedLeft = false;
                        lastJumpedRight = true;
                    }
                    else if (lbLeft >= 0 && lbRight < 0) {
                        double fR = meanR + realRight.evaluate(t);
                        double f_c = fR;
                        if (f_c < 0.) {
                            Vec3d intersection = ro + rd * t;
                            newCtxt->intersection = intersection;

                            if (lastJumpedLeft) {
                                pfL = lastGPMeanLeft + realLeft.evaluate(pt);
                            }
                            if (lastJumpedRight) {
                                pfR = lastGPMeanRight + realRight.evaluate(pt);
                            }

                            pf = std::min(pfL, pfR);
                            t = lerp(pt, t, pf / (pf - f_c));

                            double fL = meanL + realLeft.evaluate(t);
                            newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                            newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));


                            newCtxt->intersectLeft = false;
                            state.lastGPId = _gpRight->_id;

                            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                                vec_conv<Eigen::Vector3d>(rd));

                            double gradZLeft =
                                realLeft.evaluateGradient(t);

                            double gradZRight =
                                realRight.evaluateGradient(t);

                            double covDerivative2Left =
                                -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                            double covDerivative2Right =
                                -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                            auto xiLeft = rand_normal_2(sampler);
                            auto xiRight = rand_normal_2(sampler);

                            Vec3d localGradLeft{
                                std::sqrt(covDerivative2Left) * xiLeft[0],
                                std::sqrt(covDerivative2Left) * xiLeft[1],
                                gradZLeft
                            };

                            Vec3d localGradRight{
                                std::sqrt(covDerivative2Right) * xiRight[0],
                                std::sqrt(covDerivative2Right) * xiRight[1],
                                gradZRight
                            };

                            newCtxt->gradientLeft =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                                + _gpLeft->_mean->dmean_da(intersection);

                            newCtxt->gradientRight =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                                + _gpRight->_mean->dmean_da(intersection);

                            state.gpContext = newCtxt;
                            return true;
                        }
                        pf = pfR = fR;
                        lastJumpedLeft = true;
                        lastJumpedRight = false;
                    }
                    else if (lbLeft < 0 && lbRight < 0) {
                        double fL = meanL + realLeft.evaluate(t);
                        double fR = meanR + realRight.evaluate(t);
                        bool useLeft = fL < fR;
                        double f_c = useLeft ? fL : fR;
                        if (f_c < 0.) {
                            Vec3d intersection = ro + rd * t;
                            newCtxt->intersection = intersection;

                            if (lastJumpedLeft) {
                                pfL = lastGPMeanLeft + realLeft.evaluate(pt);
                            }
                            if (lastJumpedRight) {
                                pfR = lastGPMeanRight + realRight.evaluate(pt);
                            }

                            pf = std::min(pfL, pfR);
                            t = lerp(pt, t, pf / (pf - f_c));

                            newCtxt->valueLeft = lerp(pfL, fL, pf / (pf - f_c));
                            newCtxt->valueRight = lerp(pfR, fR, pf / (pf - f_c));

                            if (useLeft)
                            {
                                newCtxt->intersectLeft = true;
                                state.lastGPId = _gpLeft->_id;
                            }
                            else {
                                newCtxt->intersectLeft = false;
                                state.lastGPId = _gpRight->_id;
                            }

                            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(
                                vec_conv<Eigen::Vector3d>(rd));

                            double gradZLeft =
                                realLeft.evaluateGradient(t);

                            double gradZRight =
                                realRight.evaluateGradient(t);

                            double covDerivative2Left =
                                -_gpLeft->_cov->evalDerivative2_Isotropic(intersection, 0);

                            double covDerivative2Right =
                                -_gpRight->_cov->evalDerivative2_Isotropic(intersection, 0);

                            auto xiLeft = rand_normal_2(sampler);
                            auto xiRight = rand_normal_2(sampler);

                            Vec3d localGradLeft{
                                std::sqrt(covDerivative2Left) * xiLeft[0],
                                std::sqrt(covDerivative2Left) * xiLeft[1],
                                gradZLeft
                            };

                            Vec3d localGradRight{
                                std::sqrt(covDerivative2Right) * xiRight[0],
                                std::sqrt(covDerivative2Right) * xiRight[1],
                                gradZRight
                            };

                            newCtxt->gradientLeft =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradLeft)))
                                + _gpLeft->_mean->dmean_da(intersection);

                            newCtxt->gradientRight =
                                vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(localGradRight)))
                                + _gpRight->_mean->dmean_da(intersection);

                            state.gpContext = newCtxt;
                            return true;
                        }

                        pf = f_c;
                        pfL = fL;
                        pfR = fR;
                        lastJumpedLeft = false;
                        lastJumpedRight = false;
                    }
                    else {
                        lastJumpedLeft = true;
                        lastJumpedRight = true;
                        pf = std::min(lbLeft, lbRight);
                    }
                    lastGPMeanRight = meanR;
                    lastGPMeanLeft = meanL;
                    pt = t;
                    t += _rayMarchStepSize;
                }
                return false;
                };

            SVOTraceState traceState{};
            if (_SVO.trace(ray, Vec3f(), ray.nearT(), traceState, handleRaySegment)) {
                return true;
            }
        }

        state.gpContext = newCtxt;
        t = ray.farT();
        return false;
    }
}
