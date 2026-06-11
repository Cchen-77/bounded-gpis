#include "PraticalGaussianProcessMedium.hpp"

#include "sampling/UniformPathSampler.hpp"
#include "bsdfs/Microfacet.hpp"
#include <bsdfs/NDFs/beckmann.h>
#include <bsdfs/NDFs/GGX.h>

#include "Timer.hpp"
namespace Tungsten {

	PraticalGaussianProcessMedium::PraticalGaussianProcessMedium()
	{
	}

	void PraticalGaussianProcessMedium::fromJson(JsonPtr value, const Scene& scene)
	{

		GaussianProcessMedium::fromJson(value, scene);

        _convolutionFunction = nullptr;
        
        auto gp = std::static_pointer_cast<GaussianProcess>(_gp);
        _convolutionFunction = gp->_cov->getConvolutionFunction();
        
        
        value.getField("fast", _useFast);
        value.getField("fast_lb_skip", _fastUseLbSkip);
        value.getField("fast_SVO", _fastUseSVO);
        value.getField("fast_SVO_local_sigma", _fastUseLocalSigma);
		value.getField("1d_noise", _use1DNoise);
        value.getField("enable_NEE", _enableNEE);
		value.getField("single_realization", _useSingleRealization);
		_useSingleRealization &= (!_use1DNoise);
		value.getField("bounding_min", _boundingMin);
		value.getField("impulse_per_cell", _impulsePerCell);

		value.getField("step_size", _rayMarchStepSize);
		if (_useSingleRealization) {
			auto uniformSampler = UniformPathSampler(0xdeadbeef);
			auto gp = std::static_pointer_cast<GaussianProcess>(_gp);

			_globalReal = ConvolutionRealization3D::sample(gp, _convolutionFunction, _boundingMin, _impulsePerCell, uniformSampler);
		}

        if (!_use1DNoise) {
            _useFast = false;
        }

        if (_useFast && _fastUseLbSkip) {
            _lut = _convolutionFunction->makeLUT(_impulsePerCell);
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

    void PraticalGaussianProcessMedium::loadResources()
    {
        GaussianProcessMedium::loadResources();
        if (_useFast && _fastUseSVO) {
            auto gp = std::static_pointer_cast<GaussianProcess>(_gp);
            std::cout << "Start Building SVO for Fast Practical GPIS Medium..." << '\n';

            std::unique_ptr<SVOElement[]> SVOElements(new SVOElement[1 << (3 * SVOMaxLevel)]);
            uint32_t size = (1 << SVOMaxLevel);

            Vec3f p0 = Vec3f(0.f);
            Vec3f p1 = Vec3f(1.f) / size;
            double maxCellDistance = (_transformSVO.transformPoint(p0) - _transformSVO.transformPoint(p1)).length();
            float maxSigma = gp->_cov->getMaxSigma();


            for (int z = 0; z < size; ++z) {
                for (int y = 0; y < size; ++y) {
                    for (int x = 0; x < size; ++x) {
                        Vec3f p = Vec3f((x + .5f) / size, (y + .5f) / size, (z + .5f) / size);
                        p = _transformSVO.transformPoint(p);
                        float mean = gp->_mean->operator()(Derivative::None, vec_conv<Vec3d>(p), Vec3d());
                        int idx = x + y * size + z * size * size;
                        SVOElements[idx].minMean = mean - maxCellDistance / 2;
                        SVOElements[idx].maxMean = mean + maxCellDistance / 2;
                        if (_fastUseLocalSigma) {
                            SVOElements[idx].maxSigma = std::sqrt(gp->_cov->eval_Isotropic(vec_conv<Vec3d>(p), 0));
                        }
                        else {
                            SVOElements[idx].maxSigma = maxSigma;
                        }

                        SVOElements[idx].isActive = (SVOElements[idx].minMean < 3 * SVOElements[idx].maxSigma && SVOElements[idx].maxMean > -3 * SVOElements[idx].maxSigma);
                    }
                }
            }
            _SVO = VoxelOctree2<SVOMaxLevel, SVOElement, SVOTraceState>(_transformSVO, SVOElements.get());
        }

    }

	rapidjson::Value PraticalGaussianProcessMedium::toJson(Allocator& allocator) const
	{
		auto obj =  JsonObject{ GaussianProcessMedium::toJson(allocator), allocator,
			"type", "pratical_gaussian_process",
            "fast",_useFast,
            "fast_lb_skip",_fastUseLbSkip,
            "fast_svo",_fastUseSVO,
			"1d_noise",_use1DNoise,
			"single_realization", _useSingleRealization,
			"bounding_min",_boundingMin,
			"impulse_per_cell",_impulsePerCell,
			"step_size", _rayMarchStepSize,
            "enable_NEE",_enableNEE
		};
        return obj;
	}

    inline void atomic_add(std::atomic<double>& a, double v) {
        for (double f = a; !a.compare_exchange_weak(f, f + v););
    }

	bool PraticalGaussianProcessMedium::sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip, MediumState& state, Vec3d& grad) const
	{
        if (_use1DNoise) {
            return sampleGradient_1D(sampler, ray, ip, state, grad);
        }
        else {
            return sampleGradient_3D(sampler, ray, ip, state, grad);;
        }
	}

	bool PraticalGaussianProcessMedium::intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const
	{

        bool res;
        PROFILE_START_TIMER(timer);
        if (_useFast) {
            if (_fastUseSVO) {
                res = intersectGP_SVO(sampler, ray, state, t);
            }
            else {
                res = intersectGP_Fast(sampler, ray, state, t);
            }
        }
        else if (_use1DNoise) {
            res =  intersectGP_1D(sampler, ray, state, t);
        }
        else {
            res =  intersectGP_3D(sampler, ray, state, t);
        }
        PROFILE_STOP_TIMER(timer, intersectGPTotalTime);
        return res;
	}
	bool PraticalGaussianProcessMedium::sampleGradient_3D(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip, MediumState& state, Vec3d& grad) const
	{
        GPContextPratical3D& ctxt = *(GPContextPratical3D*)state.gpContext.get();

        switch (_normalSamplingMethod) {
        case GPNormalSamplingMethod::FiniteDifferences:
        {
            float eps = 0.0001f;
            std::array<Vec3d, 6> gradPs{
                ip + Vec3d(eps, 0.f, 0.f),
                ip + Vec3d(0.f, eps, 0.f),
                ip + Vec3d(0.f, 0.f, eps),
                ip - Vec3d(eps, 0.f, 0.f),
                ip - Vec3d(0.f, eps, 0.f),
                ip - Vec3d(0.f, 0.f, eps),
            };

            std::array<double, 6> gradVs;
            for (int i = 0; i < 6; i++) {
                gradVs[i] = ctxt.real.evaluate(gradPs[i]);
            }

            grad = Vec3d{
                gradVs[0] - gradVs[3],
                gradVs[1] - gradVs[4],
                gradVs[2] - gradVs[5],
            } / (2 * eps);

            break;
        }
        case GPNormalSamplingMethod::ConditionedGaussian:
        {
            grad = ctxt.real.evaluateGradient(ip);
            break;
        }
        case GPNormalSamplingMethod::Beckmann:
        {
            auto deriv = Derivative::First;
            Vec3d normal = Vec3d(
                _gp->mean(&ip, &deriv, nullptr, Vec3d(1.f, 0.f, 0.f), 1)(0),
                _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 1.f, 0.f), 1)(0),
                _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 0.f, 1.f), 1)(0)).normalized();

            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

            Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
            float alpha = _gp->compute_beckmann_roughness(ip);
            BeckmannNDF ndf(0, alpha, alpha);

            grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));
            break;
        }
        case GPNormalSamplingMethod::GGX:
        {
            auto deriv = Derivative::First;
            Vec3d normal = Vec3d(
                _gp->mean(&ip, &deriv, nullptr, Vec3d(1.f, 0.f, 0.f), 1)(0),
                _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 1.f, 0.f), 1)(0),
                _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 0.f, 1.f), 1)(0)).normalized();

            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

            Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
            float alpha = _gp->compute_beckmann_roughness(ip);
            GGXNDF ndf(0, alpha, alpha);
            grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));
            break;
        }
        }

        return true;
	}
	bool PraticalGaussianProcessMedium::intersectGP_3D(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const
	{
        if (!state.gpContext) {
            auto gp = std::static_pointer_cast<GaussianProcess>(_gp);
            auto ctxt = std::make_shared<GPContextPratical3D>();

            if (_useSingleRealization) {
                ctxt->real = _globalReal;
            }
            else {
                ctxt->real = ConvolutionRealization3D::sample(gp, _convolutionFunction, _boundingMin, _impulsePerCell, sampler);
            }

            state.gpContext = ctxt;
        }
        state.lastGPId = std::static_pointer_cast<GaussianProcess>(_gp)->_id;

        GPContextPratical3D& ctxt = *(GPContextPratical3D*)state.gpContext.get();
        const ConvolutionRealization3D& real = ctxt.real;

        double farT = ray.farT();
        auto rd = vec_conv<Vec3d>(ray.dir());

        if (_rayMarchStepSize == 0) {
            const double sig_0 = (farT - ray.nearT()) * 0.1f;
            const double delta = 0.01;
            const double np = 1.5;
            const double nm = 0.5;

            t = 0;
            double sig = sig_0;

            auto p = vec_conv<Vec3d>(ray.pos()) + (t + ray.nearT()) * rd;
            double f0 = real.evaluate(p);

            int sign0 = f0 < 0 ? -1 : 1;

            for (int i = 0; i < 2048 * 4; i++) {
                auto p_c = p + (t + ray.nearT() + delta) * rd;
                double f_c = real.evaluate(p_c);
                int signc = f_c < 0 ? -1 : 1;

                if (signc != sign0) {
                    t += ray.nearT();
                    return true;
                }

                t += delta;

                if (t + ray.nearT() >= farT) {
                    t += ray.nearT();
                    return false;
                }
            }

            std::cerr << "Ran out of iterations in mean intersect IA." << std::endl;
            t = ray.farT();
            return false;
        }
        else {
            double eps = 1e-3;
            auto p = vec_conv<Vec3d>(ray.pos());
            double f0 = real.evaluate(p+rd*eps);
            int sign0 = f0 < 0 ? -1 : 1;

            double pt = eps;
            double pf = f0;
            t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            while (t < ray.farT()) {
                auto p_c = p + t * rd;
                double f_c = real.evaluate(p_c);
                int signc = f_c < 0 ? -1 : 1;
                if (signc != sign0) {
                    t = lerp(pt, t, pf / (pf - f_c));
                    return true;
                }

                pf = f_c;
                pt = t;
                t += _rayMarchStepSize;
            }

            t = ray.farT();
            return false;
        }
	}

	bool PraticalGaussianProcessMedium::sampleGradient_1D(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& ip, MediumState& state, Vec3d& grad) const
	{
        GPContextPratical1D& ctxt = *(GPContextPratical1D*)state.gpContext.get();

        auto rd = vec_conv<Vec3d>(ray.dir());

        switch (_normalSamplingMethod) {
        case GPNormalSamplingMethod::FiniteDifferences: // Get rid of this for simplicity
        case GPNormalSamplingMethod::ConditionedGaussian:
        {
            // handled in sampleDistance
            grad = ctxt.gradient.value();
            break;
        }
        case GPNormalSamplingMethod::Beckmann:
        {
            auto deriv = Derivative::First;
            Vec3d normal = Vec3d(
                _gp->mean(&ip, &deriv, nullptr, Vec3d(1.f, 0.f, 0.f), 1)(0),
                _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 1.f, 0.f), 1)(0),
                _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 0.f, 1.f), 1)(0)).normalized();

            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

            Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
            float alpha = std::min(_gp->compute_beckmann_roughness(ip), 10.);
            BeckmannNDF ndf(0, alpha, alpha);

            grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));
            ctxt.gradient = grad;
            break;
        }
        case GPNormalSamplingMethod::GGX:
        {
            auto deriv = Derivative::First;
            Vec3d normal = Vec3d(
                _gp->mean(&ip, &deriv, nullptr, Vec3d(1.f, 0.f, 0.f), 1)(0),
                _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 1.f, 0.f), 1)(0),
                _gp->mean(&ip, &deriv, nullptr, Vec3d(0.f, 0.f, 1.f), 1)(0)).normalized();

            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(normal));

            Eigen::Vector3d wi = frame.toLocal(vec_conv<Eigen::Vector3d>(-ray.dir()));
            float alpha = _gp->compute_beckmann_roughness(ip);
            GGXNDF ndf(0, alpha, alpha);

            grad = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(ndf.sampleD_wi(vec_conv<Vector3>(wi)))));
            ctxt.gradient = grad;
            break;
        }
        }

        return true;
	}
	bool PraticalGaussianProcessMedium::intersectGP_1D(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const
	{
        auto ctxt = std::dynamic_pointer_cast<GPContextPratical1D>(state.gpContext);
        auto newCtxt = std::make_shared<GPContextPratical1D>();
        newCtxt->enableNEE = _enableNEE;

        auto gp = std::static_pointer_cast<GaussianProcess>(_gp);
        state.lastGPId = gp->_id;

        Vec3d ro = vec_conv<Vec3d>(ray.pos());
        Vec3d rd = vec_conv<Vec3d>(ray.dir());

        auto real = ConvolutionRealization1D::sample(ray, gp, _convolutionFunction, _impulsePerCell,sampler);


        bool conditioned = ctxt && (ctxt->gradient || ctxt->intersection);
        int nPoints = 1;

        TangentFrame rayFrame(ray.dir());

        if (conditioned) {
            Eigen::VectorXd u;
            std::array<Vec3d, 4> points = { ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value() };
            std::array<Derivative, 4> derivs = { Derivative::None, Derivative::First, Derivative::First, Derivative::First };
            std::array<Vec3d, 4> ddirs = { Vec3d{},vec_conv<Vec3d>(rayFrame.tangent),vec_conv<Vec3d>(rayFrame.bitangent),vec_conv<Vec3d>(rayFrame.normal) };
            switch (_ctxt)
            {
            case GPCorrelationContext::Goldfish:
            {
                nPoints = 4;
                Eigen::VectorXd v(4);
                Eigen::VectorXd psi(4);
                Vec3f localGraident = rayFrame.toLocal(vec_conv<Vec3f>(ctxt->gradient.value()));
                Vec3f dmeanLocal = rayFrame.toLocal(vec_conv<Vec3f>(gp->_mean->dmean_da(ro)));
                v(0) = 0.; v(1) =localGraident.x(); v(2) = localGraident.y(); v(3) = localGraident.z();

                double covDerivative2 = -gp->_cov->evalDerivative2_Isotropic(ctxt->intersection.value(), 0);
                auto sample = rand_normal_2(sampler);

                psi(0) = gp->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0) + real.evaluate(0); psi(1) = std::sqrt(covDerivative2) * sample[0] + dmeanLocal.x();
                psi(2) = std::sqrt(covDerivative2) * sample[1] + dmeanLocal.y(); psi(3) = real.evaluateGradient(0) + dmeanLocal.z();
                Eigen::MatrixXd kCC = gp->cov_sym(points.data(), derivs.data(), nullptr, rd, nPoints);
                u = kCC.ldlt().solve(v - psi);
                break;
            }
            case GPCorrelationContext::Dori: {
                nPoints = 1;
                Eigen::VectorXd v(1);
                Eigen::VectorXd psi(1);
                v(0) = 0.;
                psi(0) = gp->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0) + real.evaluate(0);
                double invKCC = 1. / gp->cov_sym(points.data(), derivs.data(), nullptr, rd, 1)(0);
                u = invKCC * (v - psi);
                break;
            }
            }

            Derivative derivNone = Derivative::None;
            Derivative derivFirst = Derivative::First;
            double pf = 0.;
            t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pt = t;
            while (t < ray.farT()) {
                Vec3d p = ro + rd * t;
                double gpMean = gp->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                double update = (gp->cov(&p, points.data(), &derivNone, derivs.data(), nullptr, ddirs.data(), rd, 1, nPoints) * u)(0);
                PROFILE_START_TIMER(psiTimer);
                double f_c = gpMean + real.evaluate(t) + update;
                PROFILE_STOP_TIMER(psiTimer, psiEvalTime);
                PROFILE_COUNT(psiEvalPoints);
                int signc = f_c < 0 ? -1 : 1;
                if (f_c < 0) {
                    t = lerp(pt, t, pf / (pf-f_c));
                    Vec3d intersection = ro + rd * t;
                    newCtxt->intersection = intersection;

                    Vec3d M = std::sqrt(gp->_cov->getAniso());
                    Vec3d Mr = rd.cwiseProduct(M);

                    TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(Mr.normalized()));
                    Vec3d tangent = vec_conv<Vec3d>(frame.tangent);
                    Vec3d bitangent = vec_conv<Vec3d>(frame.bitangent);
                    double gradZ = real.evaluateGradient(t) + (gp->cov(&intersection, points.data(), &derivFirst, derivs.data(), nullptr, nullptr, rd, 1, nPoints) * u)(0);

                    if (_enableNEE) {
                        newCtxt->gradZ = gradZ;
                        newCtxt->kDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                        newCtxt->M = M;
                        newCtxt->gradient = -rd;
                        newCtxt->gradientMean = gp->_mean->dmean_da(intersection);
                    }
                    else {
                        double covDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                        auto u = rand_normal_2(sampler);
                        newCtxt->gradient = vec_conv<Vec3d>(frame.toGlobal({ std::sqrt(covDerivative2) * u[0] ,std::sqrt(covDerivative2) * u[1] ,gradZ / Mr.length() })).cwiseProduct(M) + gp->_mean->dmean_da(intersection);
                    }
                    state.gpContext = newCtxt;
                    return true;
                }

                pf = f_c;
                pt = t;
                t += _rayMarchStepSize;
            }
        }
        else {
            Derivative derivNone = Derivative::None;
            Derivative derivFirst = Derivative::First;

            double pf = 0;
            t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pt = t;
            while (t < ray.farT()) {
                auto p = ro + rd * t;
                double gpMean = gp->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                PROFILE_START_TIMER(psiTimer);
                double f_c = gpMean + real.evaluate(t);
                PROFILE_STOP_TIMER(psiTimer, psiEvalTime);
                PROFILE_COUNT(psiEvalPoints);
                if (f_c < 0) {
                    t = lerp(pt, t, pf / (pf - f_c));
                    Vec3d intersection = ro + rd * t;
                    newCtxt->intersection = intersection;

                    Vec3d M = std::sqrt(gp->_cov->getAniso());
                    Vec3d Mr = rd.cwiseProduct(M);
                    TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(Mr.normalized()));

                    Vec3d tangent = vec_conv<Vec3d>(frame.tangent);
                    Vec3d bitangent = vec_conv<Vec3d>(frame.bitangent);
                    double gradZ = real.evaluateGradient(t);

                    if (_enableNEE) {
                        newCtxt->gradZ = gradZ;
                        newCtxt->kDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                        newCtxt->M = M;
                        newCtxt->gradient = -rd;
                        newCtxt->gradientMean = gp->_mean->dmean_da(intersection);
                    }
                    else {
                       
                        double covDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                        auto u = rand_normal_2(sampler);
                        newCtxt->gradient = vec_conv<Vec3d>(frame.toGlobal({ std::sqrt(covDerivative2) * u[0] ,std::sqrt(covDerivative2) * u[1] ,gradZ / Mr.length()})).cwiseProduct(M) + gp->_mean->dmean_da(intersection);
                        
                    }
                    state.gpContext = newCtxt;
                    return true;
                }

                pf = f_c;
                pt = t;
                t += _rayMarchStepSize;
            }
        }

        state.gpContext = newCtxt;
        t = ray.farT();
        return false;
	}

    bool PraticalGaussianProcessMedium::intersectGP_Fast(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const
    {
        auto ctxt = std::dynamic_pointer_cast<GPContextPratical1D>(state.gpContext);
        auto newCtxt = std::make_shared<GPContextPratical1D>();
        newCtxt->enableNEE = _enableNEE;

        auto gp = std::static_pointer_cast<GaussianProcess>(_gp);
        state.lastGPId = gp->_id;

        Vec3d ro = vec_conv<Vec3d>(ray.pos());
        Vec3d rd = vec_conv<Vec3d>(ray.dir());

        auto real = FastConvolutionRealization1D::sample(ray, gp, _convolutionFunction, _lut, _impulsePerCell, sampler);

        bool conditioned = ctxt && (ctxt->gradient || ctxt->intersection);
        int nPoints = 1;

        TangentFrame rayFrame(ray.dir());

        if (conditioned) {
            Eigen::VectorXd u;
            std::array<Vec3d, 4> points = { ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value() };
            std::array<Derivative, 4> derivs = { Derivative::None, Derivative::First, Derivative::First, Derivative::First };
            std::array<Vec3d, 4> ddirs = { Vec3d{},vec_conv<Vec3d>(rayFrame.tangent),vec_conv<Vec3d>(rayFrame.bitangent),vec_conv<Vec3d>(rayFrame.normal) };
            switch (_ctxt)
            {
            case GPCorrelationContext::Goldfish:
            {
                nPoints = 4;
                Eigen::VectorXd v(4);
                Eigen::VectorXd psi(4);
                Vec3f localGraident = rayFrame.toLocal(vec_conv<Vec3f>(ctxt->gradient.value()));
                Vec3f dmeanLocal = rayFrame.toLocal(vec_conv<Vec3f>(gp->_mean->dmean_da(ro)));
                v(0) = 0.; v(1) = localGraident.x(); v(2) = localGraident.y(); v(3) = localGraident.z();

                double covDerivative2 = -gp->_cov->evalDerivative2_Isotropic(ctxt->intersection.value(), 0);
                auto sample = rand_normal_2(sampler);

                psi(0) = gp->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0) + real.evaluate(0); psi(1) = std::sqrt(covDerivative2) * sample[0] + dmeanLocal.x();
                psi(2) = std::sqrt(covDerivative2) * sample[1] + dmeanLocal.y(); psi(3) = real.evaluateGradient(0) + dmeanLocal.z();
                Eigen::MatrixXd kCC = gp->cov_sym(points.data(), derivs.data(), nullptr, rd, nPoints);
                u = kCC.ldlt().solve(v - psi);
                break;
            }
            case GPCorrelationContext::Dori: {
                nPoints = 1;
                Eigen::VectorXd v(1);
                Eigen::VectorXd psi(1);
                v(0) = 0.;
                psi(0) = gp->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0) + real.evaluate(0);
                double invKCC = 1. / gp->cov_sym(points.data(), derivs.data(), nullptr, rd, 1)(0);
                u = invKCC * (v - psi);
                break;
            }
            }

            Derivative derivNone = Derivative::None;
            Derivative derivFirst = Derivative::First;

            double pf = 0;
            t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pt = t;
            bool lastJumped = false;
            double lastGPMean = 0. , lastUpdate = 0.;
            while (t < ray.farT()) {
                Vec3d p = ro + rd * t;
                double gpMean = gp->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                double update = (gp->cov(&p, points.data(), &derivNone, derivs.data(), nullptr, ddirs.data(), rd, 1, nPoints) * u)(0);
                double lb = -1e10;
                if (_fastUseLbSkip) {
                    PROFILE_START_TIMER(lbTimer);
                    lb = gpMean + real.evaluateLB(t);
                    PROFILE_STOP_TIMER(lbTimer, lbEvalTime);
                }
                if (lb + update < 0) {
                    PROFILE_START_TIMER(psiTimer);
                    double f_c = gpMean + real.evaluate(t) + update;
                    PROFILE_STOP_TIMER(psiTimer, psiEvalTime);
                    PROFILE_COUNT(psiEvalPoints);
                    if (f_c < 0) {
                        if (lastJumped) {
                            pf = lastGPMean + real.evaluate(pt) + lastUpdate;
                        }
                        t = lerp(pt, t, pf / (pf - f_c));
                        Vec3d intersection = ro + rd * t;
                        newCtxt->intersection = intersection;

                        Vec3d M = std::sqrt(gp->_cov->getAniso());
                        Vec3d Mr = rd.cwiseProduct(M);

                        TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(Mr.normalized()));
                        Vec3d tangent = vec_conv<Vec3d>(frame.tangent);
                        Vec3d bitangent = vec_conv<Vec3d>(frame.bitangent);
                        double gradZ = real.evaluateGradient(t) + (gp->cov(&intersection, points.data(), &derivFirst, derivs.data(), nullptr, nullptr, rd, 1, nPoints) * u)(0);

                        if (_enableNEE) {
                            newCtxt->gradZ = gradZ;
                            newCtxt->kDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                            newCtxt->M = M;
                            newCtxt->gradient = -rd;
                            newCtxt->gradientMean = gp->_mean->dmean_da(intersection);
                        }
                        else {
                            double covDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                            auto u = rand_normal_2(sampler);
                            newCtxt->gradient = vec_conv<Vec3d>(frame.toGlobal({ std::sqrt(covDerivative2) * u[0] ,std::sqrt(covDerivative2) * u[1] ,gradZ / Mr.length() })).cwiseProduct(M) + gp->_mean->dmean_da(intersection);
                        }
                        state.gpContext = newCtxt;
                        return true;
                    }
                    lastJumped = false;
                    pf = f_c;
                }
                else {
                    PROFILE_COUNT(lbSkipPoints);
                    lastJumped = true;
                    lastGPMean = gpMean;
                    lastUpdate = update;
                    pf = lb + update;
                }
                pt = t;
                t += _rayMarchStepSize;
            }
             
        }
        else {
            Derivative derivNone = Derivative::None;
            Derivative derivFirst = Derivative::First;

            double pf = 0;
            t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pt = t;
            bool lastJumped = false;
            double lastGPMean = 0.;
            while (t < ray.farT()) {
                Vec3d p = ro + rd * t;
                double gpMean = gp->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                double lb = -1e10;
                if (_fastUseLbSkip) {
                    PROFILE_START_TIMER(lbTimer);
                    lb = gpMean + real.evaluateLB(t);
                    PROFILE_STOP_TIMER(lbTimer, lbEvalTime);
                }
                if (lb < 0) {
                    PROFILE_START_TIMER(psiTimer);
                    double f_c = gpMean + real.evaluate(t);
                    PROFILE_STOP_TIMER(psiTimer, psiEvalTime);
                    PROFILE_COUNT(psiEvalPoints);
                    if (f_c < 0) {
                        if (lastJumped) {
                            pf = lastGPMean + real.evaluate(pt);
                        }
                        t = lerp(pt, t, pf / (pf - f_c));
                        Vec3d intersection = ro + rd * t;
                        newCtxt->intersection = intersection;


                        Vec3d M = std::sqrt(gp->_cov->getAniso());
                        Vec3d Mr = rd.cwiseProduct(M);

                        TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(Mr.normalized()));
                        Vec3d tangent = vec_conv<Vec3d>(frame.tangent);
                        Vec3d bitangent = vec_conv<Vec3d>(frame.bitangent);
                        double gradZ = real.evaluateGradient(t);

                        if (_enableNEE) {
                            newCtxt->gradZ = gradZ;
                            newCtxt->kDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                            newCtxt->M = M;
                            newCtxt->gradient = -rd;
                            newCtxt->gradientMean = gp->_mean->dmean_da(intersection);
                        }
                        else {
                            double covDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                            auto u = rand_normal_2(sampler);
                            newCtxt->gradient = vec_conv<Vec3d>(frame.toGlobal({ std::sqrt(covDerivative2) * u[0] ,std::sqrt(covDerivative2) * u[1] ,gradZ / Mr.length() })).cwiseProduct(M) + gp->_mean->dmean_da(intersection);
                        }
                        state.gpContext = newCtxt;
                        return true;
                    }
                    lastJumped = false;
                    pf = f_c;
                }
                else {
                    PROFILE_COUNT(lbSkipPoints);
                    pf = lb;
                    lastGPMean = gpMean;
                    lastJumped = true;
                }
                pt = t;
                t += _rayMarchStepSize;
            }
              
        }

        state.gpContext = newCtxt;
        t = ray.farT();
        return false;
    }
    bool PraticalGaussianProcessMedium::intersectGP_SVO(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const
    {
        auto ctxt = std::dynamic_pointer_cast<GPContextPratical1D>(state.gpContext);
        auto newCtxt = std::make_shared<GPContextPratical1D>();
        newCtxt->enableNEE = _enableNEE;

        auto gp = std::static_pointer_cast<GaussianProcess>(_gp);
        state.lastGPId = gp->_id;

        Vec3d ro = vec_conv<Vec3d>(ray.pos());
        Vec3d rd = vec_conv<Vec3d>(ray.dir());

        auto real = FastConvolutionRealization1D::sample(ray, gp, _convolutionFunction, _lut, _impulsePerCell, sampler);

        bool conditioned = ctxt && (ctxt->gradient || ctxt->intersection);
        int nPoints = 1;

        TangentFrame rayFrame(ray.dir());

        if (conditioned) {
            Eigen::VectorXd u;
            std::array<Vec3d, 4> points = { ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value(),ctxt->intersection.value() };
            std::array<Derivative, 4> derivs = { Derivative::None, Derivative::First, Derivative::First, Derivative::First };
            std::array<Vec3d, 4> ddirs = { Vec3d{},vec_conv<Vec3d>(rayFrame.tangent),vec_conv<Vec3d>(rayFrame.bitangent),vec_conv<Vec3d>(rayFrame.normal) };
            switch (_ctxt)
            {
            case GPCorrelationContext::Goldfish:
            {
                nPoints = 4;
                Eigen::VectorXd v(4);
                Eigen::VectorXd psi(4);
                Vec3f localGraident = rayFrame.toLocal(vec_conv<Vec3f>(ctxt->gradient.value()));
                Vec3f dmeanLocal = rayFrame.toLocal(vec_conv<Vec3f>(gp->_mean->dmean_da(ro)));
                v(0) = 0.; v(1) = localGraident.x(); v(2) = localGraident.y(); v(3) = localGraident.z();

                double covDerivative2 = -gp->_cov->evalDerivative2_Isotropic(ctxt->intersection.value(), 0);
                auto sample = rand_normal_2(sampler);

                psi(0) = gp->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0) + real.evaluate(0); psi(1) = std::sqrt(covDerivative2) * sample[0] + dmeanLocal.x();
                psi(2) = std::sqrt(covDerivative2) * sample[1] + dmeanLocal.y(); psi(3) = real.evaluateGradient(0) + dmeanLocal.z();
                Eigen::MatrixXd kCC = gp->cov_sym(points.data(), derivs.data(), nullptr, rd, nPoints);
                u = kCC.ldlt().solve(v - psi);
                break;
            }
            case GPCorrelationContext::Dori: {
                nPoints = 1;
                Eigen::VectorXd v(1);
                Eigen::VectorXd psi(1);
                v(0) = 0.;
                psi(0) = gp->mean_prior(points.data(), derivs.data(), nullptr, {}, 1)(0) + real.evaluate(0);
                double invKCC = 1. / gp->cov_sym(points.data(), derivs.data(), nullptr, rd, 1)(0);
                u = invKCC * (v - psi);
                break;
            }
            }

            double startT = t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pf = 0;
            double pt = startT;
            bool lastJumped = false;
            double lastGPMean = 0., lastUpdate = 0.;


            auto handleRaySegment = [&](SVOElement& element, float minT, float maxT, bool& realCheck, SVOTraceState& traceState) -> bool {
                if (maxT < t) {
                    return false;
                }

                t = std::max(t, std::ceil((minT - startT) / _rayMarchStepSize) * _rayMarchStepSize + startT);


                Derivative derivNone = Derivative::None;
                Derivative derivFirst = Derivative::First;
                int gridLevel = 0;


                if (element.level > 0) {
                    return true;
                }
          

                realCheck = true;

                if (pt < t - _rayMarchStepSize) {
                    pt = t - _rayMarchStepSize;
                    lastJumped = true;
                    Vec3d pp = ro + rd * pt;
                    lastGPMean = gp->mean_prior(&pp, &derivNone, nullptr, {}, 1)(0);
                    lastUpdate = (gp->cov(&pp, points.data(), &derivNone, derivs.data(), nullptr, ddirs.data(), rd, 1, nPoints) * u)(0);
                }

                while (t < maxT) {
                    Vec3d p = ro + rd * t;
                    double gpMean = gp->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                    double update = (gp->cov(&p, points.data(), &derivNone, derivs.data(), nullptr, ddirs.data(), rd, 1, nPoints) * u)(0);
                    double lb = -1e10;
                    if (_fastUseLbSkip) {
                        PROFILE_START_TIMER(lbTimer);
                        lb = gpMean + real.evaluateLB(t);
                        PROFILE_STOP_TIMER(lbTimer, lbEvalTime);
                    }
                    if (lb + update < 0) {
                        PROFILE_START_TIMER(psiTimer);
                        double f_c = gpMean + real.evaluate(t) + update;
                        PROFILE_STOP_TIMER(psiTimer, psiEvalTime);
                        PROFILE_COUNT(psiEvalPoints);
                        if (f_c < 0) {
                            if (lastJumped) {
                                pf = lastGPMean + real.evaluate(pt) + lastUpdate;
                            }
                            t = lerp(pt, t, std::clamp(pf / (pf - f_c),0.,1.));

                            Vec3d intersection = ro + rd * t;
                            newCtxt->intersection = intersection;

                            Vec3d M = std::sqrt(gp->_cov->getAniso());
                            Vec3d Mr = rd.cwiseProduct(M);

                            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(Mr.normalized()));
                            Vec3d tangent = vec_conv<Vec3d>(frame.tangent);
                            Vec3d bitangent = vec_conv<Vec3d>(frame.bitangent);
                            double gradZ = real.evaluateGradient(t) + (gp->cov(&intersection, points.data(), &derivFirst, derivs.data(), nullptr, nullptr, rd, 1, nPoints) * u)(0);

                            if (_enableNEE) {
                                newCtxt->gradZ = gradZ;
                                newCtxt->kDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                                newCtxt->M = M;
                                newCtxt->gradient = -rd;
                                newCtxt->gradientMean = gp->_mean->dmean_da(intersection);

                            }
                            else {
                                double covDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                                auto u = rand_normal_2(sampler);
                                newCtxt->gradient = vec_conv<Vec3d>(frame.toGlobal({ std::sqrt(covDerivative2) * u[0] ,std::sqrt(covDerivative2) * u[1] ,gradZ / Mr.length() })).cwiseProduct(M) + gp->_mean->dmean_da(intersection);
                            }
                            state.gpContext = newCtxt;
                            return true;
                        }
                        lastJumped = false;
                        pf = f_c;
                    }
                    else {
                        PROFILE_COUNT(lbSkipPoints);
                        lastJumped = true;
                        lastGPMean = gpMean;
                        lastUpdate = update;
                        pf = lb + update;
                    }
                    pt = t;
                    t += _rayMarchStepSize;
                }
                return false;
                };

                SVOTraceState traceState{};
                traceState.lipchitzSegmentEnd = startT;
                if (_SVO.trace(ray, Vec3f(), ray.nearT(), traceState, handleRaySegment)) {
                    return true;
                }
        }
        else {

            Derivative derivNone = Derivative::None;
            Derivative derivFirst = Derivative::First;

            double startT = t = ray.nearT() + _rayMarchStepSize * sampler.next1D();
            double pf = 0;
            double pt = startT;
            bool lastJumped = false;
            double lastGPMean = 0.;
            
            auto handleRaySegment = [&](SVOElement& element, float minT, float maxT, bool& realCheck, SVOTraceState& traceState) -> bool {
                if (maxT < t) {
                    return false;
                }

                t = std::max(t, std::ceil((minT - startT) / _rayMarchStepSize) * _rayMarchStepSize + startT);

                int gridLevel = 0;

                if (element.level > 0) {
                    return true;
                }

                
                realCheck = true;

                if (pt < t - _rayMarchStepSize) {
                    pt = t - _rayMarchStepSize;
                    lastJumped = true;
                    Vec3d pp = ro + rd * pt;
                    lastGPMean = gp->mean_prior(&pp, &derivNone, nullptr, {}, 1)(0);
                }

                while (t < maxT) {
                    Vec3d p = ro + rd * t;
                    double gpMean = gp->mean_prior(&p, &derivNone, nullptr, {}, 1)(0);
                    double lb = -1e10;
                    if (_fastUseLbSkip) {
                        PROFILE_START_TIMER(lbTimer);
                        lb = gpMean + real.evaluateLB(t);
                        PROFILE_STOP_TIMER(lbTimer, lbEvalTime);
                    }
                    if (lb < 0) {
                        PROFILE_START_TIMER(psiTimer);
                        double f_c = gpMean + real.evaluate(t);
                        PROFILE_STOP_TIMER(psiTimer, psiEvalTime);
                        PROFILE_COUNT(psiEvalPoints);
                        if (f_c < 0) {
                            if (lastJumped) {
                                pf = lastGPMean + real.evaluate(pt);
                            }
                            t = lerp(pt, t, std::clamp(pf / (pf - f_c), 0., 1.));
                            Vec3d intersection = ro + rd * t;
                            newCtxt->intersection = intersection;

                            Vec3d M = std::sqrt(gp->_cov->getAniso());
                            Vec3d Mr = rd.cwiseProduct(M);

                            TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d> frame(vec_conv<Eigen::Vector3d>(Mr.normalized()));
                            Vec3d tangent = vec_conv<Vec3d>(frame.tangent);
                            Vec3d bitangent = vec_conv<Vec3d>(frame.bitangent);
                            double gradZ = real.evaluateGradient(t);

                            if (_enableNEE) {
                                newCtxt->gradZ = gradZ;
                                newCtxt->kDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                                newCtxt->M = M;
                                newCtxt->gradient = -rd;
                                newCtxt->gradientMean = gp->_mean->dmean_da(intersection);
                            }
                            else {
                                double covDerivative2 = -gp->_cov->evalDerivative2_Isotropic(intersection, 0);
                                auto u = rand_normal_2(sampler);
                                newCtxt->gradient = vec_conv<Vec3d>(frame.toGlobal({ std::sqrt(covDerivative2) * u[0] ,std::sqrt(covDerivative2) * u[1] ,gradZ / Mr.length() })).cwiseProduct(M) + gp->_mean->dmean_da(intersection);
                            }
                            state.gpContext = newCtxt;
                            return true;
                        }



                        lastJumped = false;
                        pf = f_c;
                    }
                    else {
                        PROFILE_COUNT(lbSkipPoints);
                        pf = lb;
                        lastGPMean = gpMean;
                        lastJumped = true;
                    }
                    pt = t;
                    t += _rayMarchStepSize;
                }
                return false;
                };

            SVOTraceState traceState{};
            traceState.lipchitzSegmentEnd = startT;
            if (_SVO.trace(ray, Vec3f(), ray.nearT(), traceState, handleRaySegment)) {
                return true;
            }
        }

        state.gpContext = newCtxt;
        t = ray.farT();
        return false;
    }
}
