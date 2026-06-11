#include "Spot.hpp"
#include "TriangleMesh.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

#include "Debug.hpp"

namespace Tungsten {

    Spot::Spot(Mat4f& transform)
    {
        _transform = transform;
    }

    void Spot::buildProxy()
    {
        _proxy = std::make_shared<TriangleMesh>();
        _proxy->makeSphere(0.05f);
    }

    float Spot::powerToRadianceFactor() const
    {
        return INV_FOUR_PI;
    }

    void Spot::fromJson(JsonPtr value, const Scene& scene)
    {
        Primitive::fromJson(value, scene);
        Primitive::fromJson(value, scene);
        value.getField("inner_angle", _spotInnerAngleDeg);
        value.getField("outer_angle", _spotOuterAngleDeg);
    }

    rapidjson::Value Spot::toJson(Allocator& allocator) const
    {
        return JsonObject{ Primitive::toJson(allocator), allocator,
            "type", "point"
        };
    }

    bool Spot::intersect(Ray&/*ray*/, IntersectionTemporary&/*data*/) const
    {
        return false;
    }

    bool Spot::occluded(const Ray&/*ray*/) const
    {
        return false;
    }

    bool Spot::hitBackside(const IntersectionTemporary&/*data*/) const
    {
        return false;
    }

    void Spot::intersectionInfo(const IntersectionTemporary&/*data*/, IntersectionInfo& info) const
    {
        info.Ng = info.Ns = -info.w;
        info.uv = Vec2f(0.0f);
    }

    bool Spot::tangentSpace(const IntersectionTemporary&/*data*/, const IntersectionInfo&/*info*/,
        Vec3f&/*T*/, Vec3f&/*B*/) const
    {
        return false;
    }

    bool Spot::isSamplable() const
    {
        return true;
    }

    void Spot::makeSamplable(const TraceableScene&/*scene*/, uint32 /*threadIndex*/)
    {
    }

    bool Spot::samplePosition(PathSampleGenerator&/*sampler*/, PositionSample& sample) const
    {
     /*   std::cout << "??";
        sample.p = _pos;
        sample.pdf = 1.0f;
        sample.uv = Vec2f(0.0f);
        sample.weight = FOUR_PI * (*_emission)[Vec2f(0.0f)];
        sample.Ng = Vec3f(0.0f);*/

        return true;
    }

    bool Spot::sampleDirection(PathSampleGenerator& sampler, const PositionSample&/*point*/, DirectionSample& sample) const
    {
   /*     std::cout << "??";
        sample.d = SampleWarp::uniformSphere(sampler.next2D());
        sample.weight = Vec3f(1.0f);
        sample.pdf = SampleWarp::uniformSpherePdf();*/

        return true;
    }

    bool Spot::sampleDirect(uint32 /*threadIndex*/, const Vec3f& p, PathSampleGenerator&/*sampler*/, LightSample& sample) const
    {
        sample.d = _pos - p;
        float rSq = sample.d.lengthSq();
        sample.dist = std::sqrt(rSq);
        sample.d /= sample.dist;
        sample.pdf = rSq;
        return true;
    }

    bool Spot::invertPosition(WritablePathSampleGenerator&/*sampler*/, const PositionSample&/*point*/) const
    {
        return true;
    }

    bool Spot::invertDirection(WritablePathSampleGenerator& sampler, const PositionSample&/*point*/,
        const DirectionSample& direction) const
    {
        sampler.put2D(SampleWarp::invertUniformSphere(direction.d, sampler.untracked1D()));
        return true;
    }

    float Spot::positionalPdf(const PositionSample&/*point*/) const
    {
        return 1.0f;
    }

    float Spot::directionalPdf(const PositionSample&/*point*/, const DirectionSample&/*sample*/) const
    {
        return SampleWarp::uniformSpherePdf();
    }

    float Spot::directPdf(uint32 /*threadIndex*/, const IntersectionTemporary&/*data*/,
        const IntersectionInfo&/*info*/, const Vec3f& p) const
    {
        return (p - _pos).lengthSq();
    }

    Vec3f Spot::evalPositionalEmission(const PositionSample&/*sample*/) const
    {
        return FOUR_PI * (*_emission)[Vec2f(0.0f)];
    }

    Vec3f Spot::evalDirectionalEmission(const PositionSample&/*point*/, const DirectionSample&/*sample*/) const
    {
        return Vec3f(INV_FOUR_PI);
    }

    float Spot::fallOff(const Vec3f& w) const {
        float cosTheta = _spotDirection.dot(w.normalized());
        if (cosTheta < _costOuterAngle) return 0;
        if (cosTheta > _cosInnerAngle) return 1;

        float delta = (cosTheta - _costOuterAngle) /
            (_cosInnerAngle - _costOuterAngle);

        return delta * delta * delta * delta;
    }

    Vec3f Spot::evalDirect(const IntersectionTemporary&/*data*/, const IntersectionInfo& info) const
    {
        return fallOff(-info.w) * (*_emission)[Vec2f(0.0f)];
    }

    bool Spot::invertParametrization(Vec2f /*uv*/, Vec3f&/*pos*/) const
    {
        return false;
    }

    bool Spot::isDirac() const
    {
        return true;
    }

    bool Spot::isInfinite() const
    {
        return false;
    }

    float Spot::approximateRadiance(uint32 /*threadIndex*/, const Vec3f& p) const
    {
        return fallOff((p-_pos).normalized()) * _emission->average().max() / (_pos - p).lengthSq();;
    }

    Box3f Spot::bounds() const
    {
        return Box3f(_pos);
    }

    const TriangleMesh& Spot::asTriangleMesh()
    {
        if (!_proxy)
            buildProxy();
        return *_proxy;
    }

    void Spot::prepareForRender()
    {
        _pos = _transform.extractTranslationVec();
        _power = _emission ? FOUR_PI * _emission->average() : Vec3f(0.0f);


        _spotDirection = _transform.transformVector(Vec3f(0.0f, 1.0f, 0.0f)).normalized();
        _spotInnerAngleRad = Angle::degToRad(_spotInnerAngleDeg);
        _spotOuterAngleRad = Angle::degToRad(_spotOuterAngleDeg);
        
        _cosInnerAngle = std::cos(_spotInnerAngleRad);
        _costOuterAngle = std::cos(_spotOuterAngleRad);

        _spotFrame = TangentFrame(_spotDirection);

        Primitive::prepareForRender();
    }

    int Spot::numBsdfs() const
    {
        return 0;
    }

    std::shared_ptr<Bsdf>& Spot::bsdf(int /*index*/)
    {
        FAIL("Point::bsdf should never be called");
    }

    void Spot::setBsdf(int /*index*/, std::shared_ptr<Bsdf>&/*bsdf*/)
    {
        FAIL("Point::setBsdf should never be called");
    }

    Primitive* Spot::clone()
    {
        return new Spot(_transform);
    }

}
