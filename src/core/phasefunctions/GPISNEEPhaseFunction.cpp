#include "GPISNEEPhaseFunction.hpp"

#include "math/GaussianProcess.hpp"
#include "media/PraticalGaussianProcessMedium.hpp"


//just for perfectly specular
namespace Tungsten {
Vec3f GPISNEEPhaseFunction::eval(const Vec3f& wi, const Vec3f& wo, const MediumSample& mediumSample) const
{
    auto wh = (vec_conv<Vec3d>(wo - wi)).normalized();
    
    auto frame = TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d>(vec_conv<Eigen::Vector3d>(_info.rd));
    Vec3d whTransformed = vec_conv<Vec3d>(frame.toLocal(vec_conv<Eigen::Vector3d>(wh))).normalized();
    Vec3d desired = whTransformed;
    Vec3d meanGraidentTransformed = vec_conv<Vec3d>(frame.toLocal(vec_conv<Eigen::Vector3d>(_info.gradientMean)));
    
    desired *= (_info.gradZ  + meanGraidentTransformed.z()) / whTransformed.z();
    
    double k = std::sqrt(_info.kDerivative2);


    auto M = _info.M;
    auto Mr = vec_conv<Vec3d>(_info.rd).cwiseProduct(M);
    auto MTransposeInv = 1. / M;
    auto frameM = TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d>(vec_conv<Eigen::Vector3d>(Mr.normalized()));
    

    Vec3d desiredGlobal = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(desired - meanGraidentTransformed)));
    Vec3d desiredGlobalM = desiredGlobal.cwiseProduct(MTransposeInv);
    Vec3d desiredLocalM = vec_conv<Vec3d>(frameM.toLocal(vec_conv<Eigen::Vector3d>(desiredGlobalM)));
        
    std::array<double, 2> u = { desiredLocalM.x() / k, desiredLocalM.y() / k};

    float pdf = std::exp(-0.5 * (u[0] * u[0] + u[1] * u[1])) / (2 * PI * _info.kDerivative2);
    pdf *= MTransposeInv.x() * MTransposeInv.y() * MTransposeInv.z();
    pdf *= desired.lengthSq() / std::abs(whTransformed.z());
    pdf /= 4 * std::abs(wi.dot(vec_conv<Vec3f>(wh)));

    MediumSample mSample = mediumSample;
    mSample.aniso = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(desired)));
    
    return pdf * _phase->eval(wi, wo, mSample);
}

bool GPISNEEPhaseFunction::sample(PathSampleGenerator& sampler, const Vec3f& wi, const MediumSample& mediumSample, PhaseSample& sample) const
{
    MediumSample& mSample = const_cast<MediumSample&>(mediumSample);
    auto u = rand_normal_2(sampler);
    double gx = std::sqrt(_info.kDerivative2) * u[0];
    double gy = std::sqrt(_info.kDerivative2) * u[1];

    auto M = _info.M;
    auto Mr = vec_conv<Vec3d>(_info.rd).cwiseProduct(M);
    auto MTransposeInv = 1. / M;
    
    auto frame = TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d>(vec_conv<Eigen::Vector3d>(Mr.normalized()));

    mSample.aniso = vec_conv<Vec3d>(frame.toGlobal({ gx,gy, _info.gradZ / Mr.length() })).cwiseProduct(M) + _info.gradientMean;
    if (mSample.ctxt) {
        static_cast<GPContextPratical1D*>(mSample.ctxt)->gradient = mSample.aniso;
    }

    auto frameR = TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d>(vec_conv<Eigen::Vector3d>(_info.rd));

    Vec3d gLocal = vec_conv<Vec3d>(frameR.toLocal(vec_conv<Eigen::Vector3d>(mSample.aniso)));

    bool result = _phase->sample(sampler, wi, mSample, sample);

    Vec3d wh = mSample.aniso.normalized();
    Vec3d whLocal = vec_conv<Vec3d>(frameR.toLocal(vec_conv<Eigen::Vector3d>(wh)));

    float pdf = std::exp(-0.5 * (u[0] * u[0] + u[1] * u[1])) / (2 * PI * _info.kDerivative2);
    pdf *= MTransposeInv.x() * MTransposeInv.y() * MTransposeInv.z();
    pdf *= gLocal.lengthSq() / std::abs(whLocal.z());
    pdf /= 4 * std::abs(wi.dot(vec_conv<Vec3f>(wh)));

    sample.pdf *= pdf;

    return result;
}

bool GPISNEEPhaseFunction::invert(WritablePathSampleGenerator& sampler, const Vec3f&/*wi*/, const Vec3f& wo, const MediumSample& mediumSample) const
{
    return false;
}

float GPISNEEPhaseFunction::pdf(const Vec3f& wi, const Vec3f& wo, const MediumSample& mediumSample) const
{
    auto wh = (vec_conv<Vec3d>(wo - wi)).normalized();
    auto frame = TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d>(vec_conv<Eigen::Vector3d>(_info.rd));
    Vec3d whTransformed = vec_conv<Vec3d>(frame.toLocal(vec_conv<Eigen::Vector3d>(wh)));
    Vec3d desired = whTransformed;
    Vec3d meanGraidentTransformed = vec_conv<Vec3d>(frame.toLocal(vec_conv<Eigen::Vector3d>(_info.gradientMean)));
    desired *= (_info.gradZ + meanGraidentTransformed.z()) / whTransformed.z();

    double k = std::sqrt(_info.kDerivative2);

    auto M = _info.M;
    auto Mr = vec_conv<Vec3d>(_info.rd).cwiseProduct(M);
    auto MTransposeInv = 1. / M;
    auto frameM = TangentFrameD<Eigen::Matrix3d, Eigen::Vector3d>(vec_conv<Eigen::Vector3d>(Mr.normalized()));

    Vec3d desiredGlobal = vec_conv<Vec3d>(frame.toGlobal(vec_conv<Eigen::Vector3d>(desired - meanGraidentTransformed)));
    Vec3d desiredGlobalM = desiredGlobal.cwiseProduct(MTransposeInv);
    Vec3d desiredLocalM = vec_conv<Vec3d>(frameM.toLocal(vec_conv<Eigen::Vector3d>(desiredGlobalM)));


    std::array<double, 2> u = { desiredLocalM.x() / k, desiredLocalM.y() / k };

    float pdf = std::exp(-0.5 * (u[0] * u[0] + u[1] * u[1])) / (2 * PI * _info.kDerivative2);
    pdf *= MTransposeInv.x() * MTransposeInv.y() * MTransposeInv.z();
    pdf *= desired.lengthSq() / std::abs(whTransformed.z());
    pdf /= 4 * std::abs(wi.dot(vec_conv<Vec3f>(wh)));

    return pdf;
}
}
