#include "ConvolutionGaussianProcess.hpp"

#include "sampling/UniformSampler.hpp"
#include "sampling/UniformPathSampler.hpp"
#include "math/GaussianProcess.hpp"

#include <algorithm>

namespace Tungsten {
double ConvolutionRealization3D::evaluate(const Vec3d& p) const
{
	Derivative deriv = Derivative::None;
	double gpMean = gp->mean_prior(&p, &deriv, nullptr, Vec3d(0.), 1)(0);

	if (h->isStationary()) {
		return gpMean + evaluate(p, maxKernelRadius, realizationSeedOffset,{});
	}
	else {
		auto kernelParams = h->getParams(p);
		double r = h->getRadius(p,kernelParams);
		int l0 = std::floor(std::log2(r/minKernelRadius));
		int l1 = l0 + 1;
		double r0 = minKernelRadius * (1 << l0);
		double r1 = minKernelRadius * (1 << l1);

		double alpha = (r - r0) / (r1 - r0);
		
		double eps = 1e-3;
		if (alpha < eps) {
			return gpMean + evaluate(p, r0, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams);
		}
		else if (alpha > 1 - eps) {
			return gpMean + evaluate(p, r1, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
		else {
			double wDenom = 1 + alpha;
			double w0 = (1 - alpha) / wDenom, w1 = 2 * alpha / wDenom;
			return gpMean + w0 * evaluate(p, r0, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams) + w1 * evaluate(p, r1, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
	}
}

double ConvolutionRealization3D::evaluate(const Vec3d& p, double radius, uint32 seedOffset, const KernelParams& params) const
{
	Vec3d offset = p - boundingMin;
	int x0 = std::floor(offset.x() / radius) + 1;
	int y0 = std::floor(offset.y() / radius) + 1;
	int z0 = std::floor(offset.z() / radius) + 1;


	auto mortonEncode3D = [](int x, int y, int z) {
		auto part1By2 = [](uint32_t n) {
			n &= 0x000003ff;                  // 10 bits
			n = (n | (n << 16)) & 0x30000ff;
			n = (n | (n << 8)) & 0x300f00f;
			n = (n | (n << 4)) & 0x30c30c3;
			n = (n | (n << 2)) & 0x9249249;
			return n;
			};
		return (part1By2(x) << 2) | (part1By2(y) << 1) | part1By2(z);
	};

	double psi = 0.;
	double lambda = impulsePerCell / cube(radius);
	double sigma = h->getSigma(p);

	for (int x = x0 - 1; x <= x0 + 1; ++x) {
		for (int y = y0 - 1; y <= y0 + 1; ++y) {
			for (int z = z0 - 1; z <= z0 + 1; ++z) {

				uint32 seed = mortonEncode3D(x, y, z) + seedOffset;
				UniformSampler uniformSampler(seed);

				Vec3d cellMin = boundingMin + radius * Vec3d(x - 1., y - 1., z - 1.);
				
				for (int i = 0; i < impulsePerCell; ++i){
					Vec3d impulse = cellMin + radius * Vec3d(uniformSampler.next1D(), uniformSampler.next1D(), uniformSampler.next1D());
					double weight = rand_normal_2(uniformSampler.next1D(),uniformSampler.next1D())[0] * sigma / std::sqrt(lambda);
					if ((impulse - p).lengthSq() < radius * radius) {
						psi += weight * (*h)(p, impulse, params);
					}
				}

			}
		}
	}

	return psi;
}

Eigen::VectorXd ConvolutionRealization3D::evaluate(const Vec3d* ps, size_t num_ps) const
{
	Eigen::VectorXd result(num_ps);
	for (size_t p = 0; p < num_ps; p++) {
		result[p] = evaluate(ps[p]);
	}
	return result;
}

Vec3d ConvolutionRealization3D::evaluateGradient(const Vec3d& p) const
{
	Vec3d gpdMean = gp->_mean->dmean_da(p);

	if (h->isStationary()) {
		return gpdMean + evaluateGradient(p, maxKernelRadius, realizationSeedOffset,{});
	}
	else {
		auto kernelParams = h->getParams(p);
		double r = h->getRadius(p, kernelParams);
		int l0 = std::floor(std::log2(r / minKernelRadius));
		int l1 = l0 + 1;
		double r0 = minKernelRadius * (1 << l0);
		double r1 = minKernelRadius * (1 << l1);

		double alpha = (r - r0) / (r1 - r0);

		double eps = 1e-3;
		if (alpha < eps) {
			return gpdMean + evaluateGradient(p, r0, realizationSeedOffset ^ (l0 * 0x9E3779B9),kernelParams);
		}
		else if (alpha > 1 - eps) {
			return gpdMean + evaluateGradient(p, r1, realizationSeedOffset ^ (l1 * 0x9E3779B9),kernelParams);
		}
		else {
			double wDenom = sqr(1 + alpha);
			double w0 = (1 - alpha) / wDenom, w1 = 4 * alpha / (wDenom);
			return gpdMean + w0 * evaluateGradient(p, r0, realizationSeedOffset ^ (l0 * 0x9E3779B9),kernelParams) + w1 * evaluateGradient(p, r1, realizationSeedOffset ^ (l1 * 0x9E3779B9),kernelParams);
		}
	}
}

Vec3d ConvolutionRealization3D::evaluateGradient(const Vec3d& p, double radius, uint32 seedOffset, const KernelParams& params) const
{
	Vec3d offset = p - boundingMin;
	int x0 = std::floor(offset.x() / radius) + 1;
	int y0 = std::floor(offset.y() / radius) + 1;
	int z0 = std::floor(offset.z() / radius) + 1;


	auto mortonEncode3D = [](int x, int y, int z) {
		auto part1By2 = [](uint32_t n) {
			n &= 0x000003ff;                  // 10 bits
			n = (n | (n << 16)) & 0x30000ff;
			n = (n | (n << 8)) & 0x300f00f;
			n = (n | (n << 4)) & 0x30c30c3;
			n = (n | (n << 2)) & 0x9249249;
			return n;
			};
		return (part1By2(x) << 2) | (part1By2(y) << 1) | part1By2(z);
	};

	Vec3d dpsi_dp = Vec3d(0.,0.,0.);
	double lambda = impulsePerCell / cube(radius);
	double sigma = h->getSigma(p);

	for (int x = x0 - 1; x <= x0 + 1; ++x) {
		for (int y = y0 - 1; y <= y0 + 1; ++y) {
			for (int z = z0 - 1; z <= z0 + 1; ++z) {
				uint32 seed = mortonEncode3D(x, y, z) + seedOffset;
				UniformSampler uniformSampler(seed);

				Vec3d cellMin = boundingMin + radius * Vec3d(x - 1., y - 1., z - 1.);

				for (int i = 0; i < impulsePerCell; ++i) {
					Vec3d impulse = cellMin + radius * Vec3d(uniformSampler.next1D(), uniformSampler.next1D(), uniformSampler.next1D());
					double weight = rand_normal_2(uniformSampler.next1D(), uniformSampler.next1D())[0] * sigma / std::sqrt(lambda);
					if ((impulse - p).lengthSq() < radius * radius) {
						dpsi_dp += weight * h->evalGradient(p, impulse, params);
					}
				}

			}
		}
	}
	
	return dpsi_dp;
}

ConvolutionRealization3D ConvolutionRealization3D::sample(std::shared_ptr<GaussianProcess> gp, std::shared_ptr<ConvolutionFunction> convolutionKernel, Vec3d boundingMin, int impulsePerCell, PathSampleGenerator& sampler)
{
	if (!convolutionKernel) {
		convolutionKernel = gp->_cov->getConvolutionFunction();
	}
	ConvolutionRealization3D realization;
	realization.gp = gp;
	realization.boundingMin = boundingMin;
	realization.h = convolutionKernel;
	realization.minKernelRadius = convolutionKernel->getMinRadius();
	realization.maxKernelRadius = convolutionKernel->getMaxRadius();
	realization.impulsePerCell = impulsePerCell;
	//we need two offset in nonstationary case
	realization.realizationSeedOffset = BitManip::floatBitsToUint(sampler.next1D());

	return realization;
}


double ConvolutionRealization1D::evaluate(double t) const
{
	if (h->isStationary() && !h->isAnisotropic()){
		return evaluate(t, maxKernelRadius, realizationSeedOffset, {});
	}
	else {
		auto kernelParams = h->getParams(ray,t);
		double r = h->getRadius(ray,t, kernelParams);
		int l0 = std::floor(std::log2(r / minKernelRadius));
		int l1 = l0 + 1;
		double r0 = minKernelRadius * (1 << l0);
		double r1 = minKernelRadius * (1 << l1);

		double alpha = (r - r0) / (r1 - r0);

		double eps = 1e-3;
		if (alpha < eps) {
			return evaluate(t, r0, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams);
		}
		else if (alpha > 1 - eps) {
			return evaluate(t, r1, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
		else {
			double wDenom = 1 + alpha;
			double w0 = (1 - alpha) / wDenom, w1 = 2 * alpha / wDenom;
			return w0 * evaluate(t, r0, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams) + w1 * evaluate(t, r1, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
	}
}

double ConvolutionRealization1D::evaluate(double t, double radius, uint32 seedOffset, const KernelParams& params) const
{
	Vec3d ro = vec_conv<Vec3d>(ray.pos());
	Vec3d rd = vec_conv<Vec3d>(ray.dir());

	int index0 = t / radius + 1;

	double psi = 0.;
	double lambda = impulsePerCell / radius;
	double sigma = h->getSigma(ro + rd*t);

	for(int index = index0-1;index <= index0 + 1;++index){
		uint32 seed = index + seedOffset;
		UniformSampler uniformSampler(seed);

		double tMin = radius * (index - 1);
		for (int i = 0; i < impulsePerCell; ++i) {
			double tImpulse = tMin + uniformSampler.next1D() * radius;
			double weight = rand_normal_2(uniformSampler.next1D(), uniformSampler.next1D())[0] * sigma / std::sqrt(lambda);
				rand_normal_2(uniformSampler.next1D(), uniformSampler.next1D())[0] * sigma / std::sqrt(lambda);
			if (std::abs(tImpulse - t) < radius) {
				psi += weight * (*h)(ray, t, tImpulse, params);
			}
		}
	}

	return psi;
}
double ConvolutionRealization1D::evaluateGradient(double t) const
{
	if (h->isStationary() && !h->isAnisotropic()) {
		return evaluateGradient(t, maxKernelRadius, realizationSeedOffset, {});
	}
	else {
		auto kernelParams = h->getParams(ray, t);
		double r = h->getRadius(ray, t, kernelParams);
		int l0 = std::floor(std::log2(r / minKernelRadius));
		int l1 = l0 + 1;
		double r0 = minKernelRadius * (1 << l0);
		double r1 = minKernelRadius * (1 << l1);

		double alpha = (r - r0) / (r1 - r0);

		double eps = 1e-3;
		if (alpha < eps) {
			return evaluateGradient(t, r0, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams);
		}
		else if (alpha > 1 - eps) {
			return evaluateGradient(t, r1, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
		else {
			double wDenom = 1 + alpha;
			double w0 = (1 - alpha) / wDenom, w1 = 2 * alpha / wDenom;
			return w0 * evaluateGradient(t, r0, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams) + w1 * evaluateGradient(t, r1, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
	}
}
double ConvolutionRealization1D::evaluateGradient(double t, double radius, uint32 seedOffset, const KernelParams& params) const
{
	Vec3d ro = vec_conv<Vec3d>(ray.pos());
	Vec3d rd = vec_conv<Vec3d>(ray.dir());
	Vec3d p = ro + t * rd;

	int index0 = t / radius + 1;

	double dpsi_dt = 0.;
	double lambda = impulsePerCell / radius;
	double sigma = h->getSigma(ro + rd * t);

	for (int index = index0 - 1; index <= index0 + 1; ++index) {
		uint32 seed = index + seedOffset;
		UniformSampler uniformSampler(seed);

		double tMin = radius * (index - 1);
		for (int i = 0; i < impulsePerCell; ++i) {
			double tImpulse = tMin + uniformSampler.next1D() * radius;
			double weight = rand_normal_2(uniformSampler.next1D(), uniformSampler.next1D())[0] * sigma / std::sqrt(lambda);
			if (std::abs(tImpulse - t) < radius) {
				dpsi_dt += weight * h->evalGradient(ray, t, tImpulse, params);
			}
		}
	}

	return dpsi_dt;
}
ConvolutionRealization1D ConvolutionRealization1D::sample(Ray ray, std::shared_ptr<GaussianProcess> gp, std::shared_ptr<ConvolutionFunction> convolutionKernel, int impulsePerCell, PathSampleGenerator& sampler)
{
	if (!convolutionKernel) {
		convolutionKernel = gp->_cov->getConvolutionFunction();
	}
	ConvolutionRealization1D realization;
	realization.ray = ray;
	realization.gp = gp;
	realization.h = convolutionKernel;
	realization.minKernelRadius = convolutionKernel->getMinRadius();
	realization.maxKernelRadius = convolutionKernel->getMaxRadius();
	realization.impulsePerCell = impulsePerCell;
	
	realization.realizationSeedOffset = BitManip::floatBitsToUint(sampler.next1D());

	return realization;
}

double FastConvolutionRealization1D::evaluateLB(double t)
{
	if (h->isStationary() && !h->isAnisotropic()) {
		return evaluateLB(t, maxKernelRadius, 0, 1, realizationSeedOffset, {});
	}
	else {
		auto kernelParams = h->getParams(ray, t);
		double r = h->getRadius(ray, t, kernelParams);
		int l0 = std::floor(std::log2(r / minKernelRadius));
		int l1 = l0 + 1;
		double r0 = minKernelRadius * (1 << l0);
		double r1 = minKernelRadius * (1 << l1);

		double alpha = (r - r0) / (r1 - r0);

		double eps = 1e-3;
		if (alpha < eps) {
			return evaluateLB(t, r0, l0, 1 + alpha, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams);
		}
		else if (alpha > 1 - eps) {
			return evaluateLB(t, r1, l1, (1 + alpha) /2, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
		else {
			double wDenom = 1 + alpha;
			double w0 = (1 - alpha) / wDenom, w1 = 2 * alpha / wDenom;
			return w0 * evaluateLB(t, r0, l0, 1 + alpha, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams) + w1 * evaluateLB(t, r1, l1, (1 + alpha) / 2, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
	}
}
double FastConvolutionRealization1D::evaluateLB(double t, double radius, int level, double Nscale, uint32 seedOffset, const KernelParams& kernelParams)
{

	Vec3d ro = vec_conv<Vec3d>(ray.pos());
	Vec3d rd = vec_conv<Vec3d>(ray.dir());


	double lambda = impulsePerCell / radius;
	double sigma = h->getSigma(ro + rd * t);
	double w = sigma / std::sqrt(lambda);

	double subcellSize = radius / impulsePerCell;

	int subcellIndex0 = std::max(2 * impulsePerCell, int(std::floor((t + 2 * radius) / subcellSize)));

	UniformSampler positionSampler0(uint32_t(subcellIndex0) + seedOffset);
	double tImpulse0 = subcellIndex0 * subcellSize - 2*radius + subcellSize * positionSampler0.next1D();

	double lowerBound = (weightSequences[level][size_t(subcellIndex0)] ? 1 : -1) * (*h)(ray, t, tImpulse0);

	int N = std::ceil(impulsePerCell * Nscale);

	weightSequences[level].sliceReversed(subcellIndex0 - N, subcellIndex0 - 1, leftLUTBlocks.data(), lutBlockCount);
	weightSequences[level].slice(subcellIndex0 + 1, subcellIndex0 + N, rightLUTBlocks.data(), lutBlockCount);
	

	lowerBound += lut->queryPerPoint(ray, t, level, leftLUTBlocks.data(), lutBlockCount, kernelParams) +
		lut->queryPerPoint(ray, t, level, rightLUTBlocks.data(), lutBlockCount, kernelParams);

	lowerBound *= w;
	
	double evaluate1 = lowerBound;
	return lowerBound;
}
double FastConvolutionRealization1D::evaluateUB(double t)
{
	if (h->isStationary() && !h->isAnisotropic()) {
		return evaluateUB(t, maxKernelRadius, 0, 1, realizationSeedOffset, {});
	}
	else {
		auto kernelParams = h->getParams(ray, t);
		double r = h->getRadius(ray, t, kernelParams);
		int l0 = std::floor(std::log2(r / minKernelRadius));
		int l1 = l0 + 1;
		double r0 = minKernelRadius * (1 << l0);
		double r1 = minKernelRadius * (1 << l1);

		double alpha = (r - r0) / (r1 - r0);

		double eps = 1e-3;
		if (alpha < eps) {
			return evaluateUB(t, r0, l0, 1 + alpha, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams);
		}
		else if (alpha > 1 - eps) {
			return evaluateUB(t, r1, l1, (1 + alpha) / 2, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
		else {
			double wDenom = 1 + alpha;
			double w0 = (1 - alpha) / wDenom, w1 = 2 * alpha / wDenom;
			return w0 * evaluateUB(t, r0, l0, 1 + alpha, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams) + w1 * evaluateUB(t, r1, l1, (1 + alpha) / 2, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
	}
}
double FastConvolutionRealization1D::evaluateUB(double t, double radius, int level, double Nscale, uint32 seedOffset, const KernelParams& kernelParams)
{

	Vec3d ro = vec_conv<Vec3d>(ray.pos());
	Vec3d rd = vec_conv<Vec3d>(ray.dir());


	double lambda = impulsePerCell / radius;
	double sigma = h->getSigma(ro + rd * t);
	double w = sigma / std::sqrt(lambda);

	double subcellSize = radius / impulsePerCell;

	int subcellIndex0 = std::max(2 * impulsePerCell, int(std::floor((t + 2 * radius) / subcellSize)));

	UniformSampler positionSampler0(uint32_t(subcellIndex0) + seedOffset);
	double tImpulse0 = subcellIndex0 * subcellSize - 2 * radius + subcellSize * positionSampler0.next1D();

	double upperbound = (weightSequences[level][size_t(subcellIndex0)] ? 1 : -1) * (*h)(ray, t, tImpulse0);

	int N = std::ceil(impulsePerCell * Nscale);

	weightSequences[level].sliceReversed(subcellIndex0 - N, subcellIndex0 - 1, leftLUTBlocks.data(), lutBlockCount);
	weightSequences[level].slice(subcellIndex0 + 1, subcellIndex0 + N, rightLUTBlocks.data(), lutBlockCount);


	upperbound += lut->queryPerPoint2(ray, t, level, leftLUTBlocks.data(), lutBlockCount, kernelParams) +
		lut->queryPerPoint2(ray, t, level, rightLUTBlocks.data(), lutBlockCount, kernelParams);

	upperbound *= w;

	return upperbound;
}
double FastConvolutionRealization1D::evaluate(double t)
{
	if (h->isStationary() && !h->isAnisotropic()) {
		return evaluate(t, minKernelRadius, 0, 1,  realizationSeedOffset,{});
	}
	else {
		auto kernelParams = h->getParams(ray, t);
		double r = h->getRadius(ray, t, kernelParams);
		int l0 = std::floor(std::log2(r / minKernelRadius));
		int l1 = l0 + 1;
		double r0 = minKernelRadius * (1 << l0);
		double r1 = minKernelRadius * (1 << l1);

		double alpha = (r - r0) / (r1 - r0);

		double eps = 1e-3;
		if (alpha < eps) {
			return evaluate(t, r0, l0, 1 + alpha, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams);
		}
		else if (alpha > 1 - eps) {
			return evaluate(t, r1, l1, (1 + alpha) / 2, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
		else {
			double wDenom = 1 + alpha;
			double w0 = (1 - alpha) / wDenom, w1 = 2 * alpha / wDenom;
			return w0 * evaluate(t, r0, l0, 1 + alpha, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams) + w1 * evaluate(t, r1, l1, (1 + alpha) / 2, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
	}
}
double FastConvolutionRealization1D::evaluate(double t, double radius, int level, double Nscale, uint32 seedOffset, const KernelParams& params)
{
	Vec3d ro = vec_conv<Vec3d>(ray.pos());
	Vec3d rd = vec_conv<Vec3d>(ray.dir());


	double subcellSize = radius / impulsePerCell;

	int subcellIndex0 = std::max(2 * impulsePerCell, int(std::floor((t + 2 * radius) / subcellSize)));
	

	double psi = 0.;
	double lambda = impulsePerCell / radius;
	double sigma = h->getSigma(ro + rd * t);
	double w = sigma / std::sqrt(lambda);
	int N = std::ceil(impulsePerCell * Nscale);

	for (int subcellIndex = subcellIndex0 - N; subcellIndex <= subcellIndex0 + N; ++subcellIndex) {
		uint32_t seed = subcellIndex + seedOffset;
		UniformSampler positionSampler(seed);
		double tImpulse = subcellIndex * subcellSize - 2*radius + subcellSize*positionSampler.next1D();
		
 		double weight = weightSequences[level][size_t(subcellIndex)] ? w : -w;
		psi += weight * (*h)(ray, t, tImpulse, params);
		
	}

	return psi;
}
double FastConvolutionRealization1D::evaluateGradient(double t)
{
	if (h->isStationary() && !h->isAnisotropic()) {
		return evaluateGradient(t, maxKernelRadius, 0, 1, realizationSeedOffset, {});
	}
	else {
		auto kernelParams = h->getParams(ray, t);
		double r = h->getRadius(ray, t, kernelParams);
		int l0 = std::floor(std::log2(r / minKernelRadius));
		int l1 = l0 + 1;
		double r0 = minKernelRadius * (1 << l0);
		double r1 = minKernelRadius * (1 << l1);

		double alpha = (r - r0) / (r1 - r0);

		double eps = 1e-3;
		if (alpha < eps) {
			return evaluateGradient(t, r0, l0, 1 + alpha, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams);
		}
		else if (alpha > 1 - eps) {
			return evaluateGradient(t, r1, l1, (1 + alpha) / 2, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
		else {
			double wDenom = 1 + alpha;
			double w0 = (1 - alpha) / wDenom, w1 = 2 * alpha / wDenom;
			return w0 * evaluateGradient(t, r0, l0, 1 + alpha, realizationSeedOffset ^ (l0 * 0x9E3779B9), kernelParams) + w1 * evaluateGradient(t, r1, l1, (1 + alpha) / 2, realizationSeedOffset ^ (l1 * 0x9E3779B9), kernelParams);
		}
	}
}
double FastConvolutionRealization1D::evaluateGradient(double t, double radius, int level, double Nscale, uint32 seedOffset, const KernelParams& params)
{
	Vec3d ro = vec_conv<Vec3d>(ray.pos());
	Vec3d rd = vec_conv<Vec3d>(ray.dir());
	Vec3d p = ro + t * rd;

	double subcellSize = radius / impulsePerCell;

	int subcellIndex0 = std::max(2 * impulsePerCell, int(std::floor((t + 2 * radius) / subcellSize)));

	double dpsi_dt = 0.;
	double lambda = impulsePerCell / radius;
	double sigma = h->getSigma(ro + rd*t);
	double w = sigma / std::sqrt(lambda);
	int N = std::ceil(impulsePerCell * Nscale);

	for (int subcellIndex = subcellIndex0 - N; subcellIndex <= subcellIndex0 + N; ++subcellIndex) {
		uint32_t seed = subcellIndex + seedOffset;
		UniformSampler positionSampler(seed);
		double tImpulse = subcellIndex * subcellSize - 2*radius + subcellSize * positionSampler.next1D();
		
		double weight = weightSequences[level][size_t(subcellIndex)] ? w : -w;
		dpsi_dt += weight * h->evalGradient(ray, t, tImpulse, params);
		
	}

	return dpsi_dt;
}
FastConvolutionRealization1D FastConvolutionRealization1D::sample(Ray ray, std::shared_ptr<GaussianProcess> gp, std::shared_ptr<ConvolutionFunction> convolutionKernel, std::shared_ptr<ConvolutionFunctionLUT> lut,
	int impulsePerCell, PathSampleGenerator& sampler)
{
	if (!convolutionKernel) {
		convolutionKernel = gp->_cov->getConvolutionFunction();
	}
	FastConvolutionRealization1D realization;
	realization.ray = ray;
	realization.gp = gp;
	realization.h = convolutionKernel;
	realization.lut = lut;
	realization.minKernelRadius = convolutionKernel->getMinRadius();
	realization.maxKernelRadius = convolutionKernel->getMaxRadius();
	realization.impulsePerCell = impulsePerCell;


	realization.realizationSeedOffset = BitManip::floatBitsToUint(sampler.next1D());
	
	int maxLevel = std::ceil(std::log2(realization.maxKernelRadius / realization.minKernelRadius));
	
	realization.weightSequences.reserve(maxLevel + 1);
	for (int level = 0; level <= maxLevel; ++level) {
		realization.weightSequences.emplace_back(realization.realizationSeedOffset, level);
	}

	realization.lutBlockCount = (impulsePerCell - 1) / ConvolutionFunctionLUT::BLOCK_BITS + 1;

	return realization;
}
}
