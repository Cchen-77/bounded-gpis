#ifndef CONVOLUTIONGAUSSIANPROCESS_HPP_
#define CONVOLUTIONGAUSSIANPROCESS_HPP_

#include <math/GaussianProcess.hpp>
#include <math/AffineArithmetic.hpp>

#include "Box.hpp"
#include "math/GPFunctions.hpp"

#include "pcg-cpp/pcg_random.hpp"
#include "boost/core/bit.hpp"

#include <optional>

namespace Tungsten {
	struct ConvolutionRealization3D {
		std::shared_ptr<GaussianProcess> gp;
		Vec3d boundingMin;
		std::shared_ptr<ConvolutionFunction> h;
		double minKernelRadius;
		double maxKernelRadius;
		int impulsePerCell;
		uint32 realizationSeedOffset;

		double evaluate(const Vec3d& p) const;
		double evaluate(const Vec3d& p, double radius, uint32 seedOffset, const KernelParams& params) const;
		Eigen::VectorXd evaluate(const Vec3d* ps, size_t num_ps) const;
		Vec3d evaluateGradient(const Vec3d& p) const;
		Vec3d evaluateGradient(const Vec3d& p, double radius, uint32 seedOffset, const KernelParams& params) const;

		static ConvolutionRealization3D sample(std::shared_ptr<GaussianProcess> gp, std::shared_ptr<ConvolutionFunction> convolutionKernel, Vec3d boundingMin, int impulsePerCell, PathSampleGenerator& sampler);

	};

	// for ConvolutionRealizatipn1D series, we exclude we exclude the GP mean from the evaluation
	// for a better profiling and avoiding caculate it twice (both in 'evaluateLB' and 'evaluate')

	struct ConvolutionRealization1D {
		Ray ray;

		std::shared_ptr<GaussianProcess> gp;
		std::shared_ptr<ConvolutionFunction> h;
		double minKernelRadius;
		double maxKernelRadius;
		int impulsePerCell;
		uint32 realizationSeedOffset;

		double evaluate(double t) const;
		double evaluate(double t, double radius, uint32 seedOffset, const KernelParams& params) const;
		double evaluateGradient(double t) const;
		double evaluateGradient(double t, double radius, uint32 seedOffset, const KernelParams& params) const;

		static ConvolutionRealization1D sample(Ray ray, std::shared_ptr<GaussianProcess> gp, std::shared_ptr<ConvolutionFunction> convolutionKernel, int impulsePerCell, PathSampleGenerator& sampler);

	};

	struct FastConvolutionRealization1D {
		Ray ray;

		std::shared_ptr<GaussianProcess> gp;
		std::shared_ptr<ConvolutionFunction> h;
		std::shared_ptr<ConvolutionFunctionLUT> lut;
		double minKernelRadius;
		double maxKernelRadius;
		int impulsePerCell;


		int lutBlockCount;
		std::array<ConvolutionFunctionLUT::BLOCK_TYPE,10> leftLUTBlocks;
		std::array<ConvolutionFunctionLUT::BLOCK_TYPE,10> rightLUTBlocks;

		uint32 realizationSeedOffset;
		
		double evaluateLB(double t);
		double evaluateLB(double t, double radius, int level, double Nscale, uint32 seedOffset, const KernelParams& kernelParams);
		double evaluateUB(double t);
		double evaluateUB(double t, double radius, int level, double Nscale, uint32 seedOffset, const KernelParams& kernelParams);

		double evaluate(double t);
		double evaluate(double t, double radius, int level, double Nscale, uint32 seedOffset, const KernelParams& params);
		double evaluateGradient(double t);
		double evaluateGradient(double t, double radius, int level, double Nscale, uint32 seedOffset, const KernelParams& params);

		static FastConvolutionRealization1D sample(Ray ray, std::shared_ptr<GaussianProcess> gp, std::shared_ptr<ConvolutionFunction> convolutionKernel, std::shared_ptr<ConvolutionFunctionLUT> lut,int impulsePerCell, PathSampleGenerator& sampler);

	public:

		class WeightSequence {
		public:
			WeightSequence(uint32_t seed, uint32_t stream) : rng(seed, stream) {};

			std::unordered_map<size_t, uint64_t> blocks;

			inline bool operator[](size_t index) {
				size_t blockIndex = index >> 6;
				size_t offset = index & 63;
				uint64_t block = blockFor(blockIndex);
				return block & (1ull << offset);
			}

		
			inline uint64_t sliceBlock(size_t i, size_t j) {
				size_t ki = i >> 6;
				size_t kj = j >> 6;
				size_t oi = i & 63;
				size_t oj = j & 63;

				if (ki == kj) {
					return (blockFor(ki) >> oi) & ((1ull << (j - i + 1)) - 1);
				}
				else {
					uint64_t left = blockFor(ki) >> oi;
					uint64_t right = blockFor(kj) & ((1ull << (oj + 1)) - 1);
					return left | (right << (64 - oi));
				}
			}

			template<typename BLOCK_TYPE>
			inline void slice(int begin, int end, BLOCK_TYPE* lutBlocks, int blockCount) {
				auto BLOCK_BITS = sizeof(BLOCK_TYPE) * 8;
				for (size_t i = 0; i < blockCount; ++i) {
					int b = begin + i * BLOCK_BITS;
					int e = std::min(end, b + int(BLOCK_BITS) - 1);
					lutBlocks[i] = static_cast<BLOCK_TYPE>(sliceBlock(b, e));
				}
			}

			template<typename BLOCK_TYPE>
			inline void sliceReversed(int begin, int end, BLOCK_TYPE* lutBlocks, int blockCount) {
				auto BLOCK_BITS = sizeof(BLOCK_TYPE) * 8;
				for (int i = 0; i < blockCount; ++i) {
					int e = end - i * BLOCK_BITS;
					int b = std::max(begin, e - int(BLOCK_BITS) + 1);
					lutBlocks[i] = static_cast<BLOCK_TYPE>(sliceBlock(b, e));
					if (i == blockCount - 1) {
						lutBlocks[i] >>= (BLOCK_BITS - (end - begin + 1));
					}
					lutBlocks[i] = reverse<BLOCK_TYPE>(lutBlocks[i]);
				}
			}

		private:
			pcg64 rng;

			inline uint64_t blockFor(size_t blockIndex)  {
				if (blocks.find(blockIndex) == blocks.end()) {
					blocks[blockIndex] = rng();
				}
				return blocks[blockIndex];
			}
		};

		std::vector<WeightSequence> weightSequences;
	};
}

#endif /* WEIGHTSPACEGAUSSIANPROCESS_HPP_ */