#ifndef VOXELOCTREE2_HPP_
#define VOXELOCTREE2_HPP_

#include "math/Vec.hpp"
#include "math/Mat4f.hpp"
#include "math/Ray.hpp"

#include <memory>
#include <limits>
#include <array>

namespace Tungsten {

template<int NumLevels, typename ElementType, typename TraceState>
class VoxelOctree2
{
    Vec3f _offset;
    Mat4f _transform;
    Mat4f _invTransform;

    std::array<std::unique_ptr<ElementType[]>, NumLevels> _grids;

    void buildHierarchy(int level, const ElementType *data, ElementType *parent)
    {
        const int size = 1 << (NumLevels - level);
        const int parentSize = size/2;

        auto brickContainsVoxels = [&](int bx, int by, int bz) {
            for (int z = bz*2; z < (bz + 1)*2; ++z)
                for (int y = by*2; y < (by + 1)*2; ++y)
                    for (int x = bx*2; x < (bx + 1)*2; ++x)
                        if (data[x + size*y + size*size*z])
                            return true;
            return false;
        };
        auto copyAndCheckContainsVoxels = [&](int bx, int by, int bz, ElementType *dst) {
            int base = (bx + size*by + size*size*bz)*2;

            ElementType parent = ElementType();

            for (int z = 0; z < 2; ++z)
                for (int y = 0; y < 2; ++y)
                    for (int x = 0; x < 2; ++x) {
                        if ((dst[x + y * 2 + z * 4] = data[base + x + size * y + size * size * z])) {
                            parent.addChild(dst[x + y * 2 + z * 4]);
                        }
                    }
            return parent;
        };

        int nonZeroCount = 0;
        for (int z = 0; z < parentSize; ++z)
            for (int y = 0; y < parentSize; ++y)
                for (int x = 0; x < parentSize; ++x)
                    if (brickContainsVoxels(x, y, z))
                        nonZeroCount++;

        _grids[level].reset(new ElementType[nonZeroCount*8]);


        std::memset(parent, 0, sizeof(ElementType)*parentSize*parentSize*parentSize);

        int count = 0;
        for (int z = 0; z < parentSize && count < nonZeroCount; ++z) {
            for (int y = 0; y < parentSize && count < nonZeroCount; ++y) {
                for (int x = 0; x < parentSize && count < nonZeroCount; ++x) {
                    if (parent[x + parentSize * y + parentSize * parentSize * z] = copyAndCheckContainsVoxels(x, y, z, &_grids[level][count*8])) {
                        parent[x + parentSize * y + parentSize * parentSize * z].setChildIdx(8 * count);
                        parent[x + parentSize * y + parentSize * parentSize * z].level = level + 1;
                        count++;
                    }
                }
            }
        }
    }

public:
    VoxelOctree2() = default;
    VoxelOctree2(Mat4f transform, const ElementType *data)
    {
        _transform = transform;
        _invTransform = _transform.invert();

        int temporarySize = 1 << 3*(NumLevels - 1);
        std::unique_ptr<ElementType[]> temporaryA(new ElementType[temporarySize]);
        std::unique_ptr<ElementType[]> temporaryB(new ElementType[temporarySize]);

        ElementType *bufferA = temporaryA.get(), *bufferB = temporaryB.get();

        _grids[NumLevels - 1].reset(new ElementType[8]);

        for (int i = 0; i < NumLevels - 1; ++i) {
            const ElementType *src = (i == 0 ? data : bufferA);
            ElementType *dst = (i == NumLevels - 2 ? _grids[NumLevels - 1].get() : bufferB);
            buildHierarchy(i, src, dst);
            std::swap(bufferA, bufferB);
        }
    }

    template<typename raySegmentLambda>
    bool trace(const Ray &ray, const Vec3f &/*dT*/, float tMin, TraceState traceState, raySegmentLambda handleRaySegment) const{
        CONSTEXPR int MaxScale = 23;
        CONSTEXPR int ScaleOffset = MaxScale - NumLevels;

        struct StackEntry
        {
            uint32 parentIdx;
            float maxT;
        };
        StackEntry rayStack[MaxScale + 1];

        // Transform to [1, 2]^3 cube
        Vec3f o = _invTransform.transformPoint(ray.pos()) + 1.;
        Vec3f d = _invTransform.transformVector(ray.dir());

        if (std::abs(d.x()) < 1e-8f) d.x() = 1e-8f;
        if (std::abs(d.y()) < 1e-8f) d.y() = 1e-8f;
        if (std::abs(d.z()) < 1e-8f) d.z() = 1e-8f;

        Vec3f dT = 1.0f/-std::abs(d);
        Vec3f bT = dT*o;

        uint32 octantMask = 0;
        if (d.x() > 0.0f) octantMask ^= 1, bT.x() = 3.0f*dT.x() - bT.x();
        if (d.y() > 0.0f) octantMask ^= 2, bT.y() = 3.0f*dT.y() - bT.y();
        if (d.z() > 0.0f) octantMask ^= 4, bT.z() = 3.0f*dT.z() - bT.z();

        float minT = max((2.0f*dT - bT).max(), tMin);
        float maxT = min((dT - bT).min(), ray.farT());

        uint32 parentIdx  = 0;
        int idx     = 0;
        Vec3f pos(1.0f);
        int scale   = MaxScale - 1;
        float scaleExp2 = 0.5f;

        if (1.5f*dT.x() - bT.x() > minT) idx ^= 1, pos.x() = 1.5f;
        if (1.5f*dT.y() - bT.y() > minT) idx ^= 2, pos.y() = 1.5f;
        if (1.5f*dT.z() - bT.z() > minT) idx ^= 4, pos.z() = 1.5f;


        while (scale < MaxScale) {
            Vec3f cornerT = pos*dT - bT;
            float maxTC = cornerT.min();

            if (minT <= maxT) {
                float maxTV = min(maxT, maxTC);
                float half = scaleExp2*0.5f;
                Vec3f centerT = half*dT + cornerT;

                if (minT <= maxTV) {
                    ElementType& element = _grids[scale - ScaleOffset][parentIdx + (idx ^ octantMask)];

                    if (element) {
                        bool realCheck = false;
                        if (handleRaySegment(element, minT, maxTV, realCheck, traceState)) {
                            if (realCheck) {
                                return true;
                            }
                            else if(scale - ScaleOffset != 0){
                                uint32_t childIdx = element.childIdx;
                                rayStack[scale] = StackEntry{ parentIdx, maxT };

                                parentIdx = childIdx;

                                idx = 0;
                                scale--;
                                scaleExp2 = half;

                                if (centerT.x() > minT) idx ^= 1, pos.x() += scaleExp2;
                                if (centerT.y() > minT) idx ^= 2, pos.y() += scaleExp2;
                                if (centerT.z() > minT) idx ^= 4, pos.z() += scaleExp2;

                                maxT = maxTV;

                                continue;
                            }
                        }
                    }
                }
            }

            int stepMask = 0;
            if (cornerT.x() <= maxTC) stepMask ^= 1, pos.x() -= scaleExp2;
            if (cornerT.y() <= maxTC) stepMask ^= 2, pos.y() -= scaleExp2;
            if (cornerT.z() <= maxTC) stepMask ^= 4, pos.z() -= scaleExp2;

            minT = maxTC;
            idx ^= stepMask;

            if ((idx & stepMask) != 0) {
                int differingBits = 0;
                if (stepMask & 1) differingBits |= BitManip::floatBitsToUint(pos.x()) ^ BitManip::floatBitsToUint(pos.x() + scaleExp2);
                if (stepMask & 2) differingBits |= BitManip::floatBitsToUint(pos.y()) ^ BitManip::floatBitsToUint(pos.y() + scaleExp2);
                if (stepMask & 4) differingBits |= BitManip::floatBitsToUint(pos.z()) ^ BitManip::floatBitsToUint(pos.z() + scaleExp2);
                scale = (BitManip::floatBitsToUint((float)differingBits) >> 23) - 127;
                scaleExp2 = BitManip::uintBitsToFloat((scale - MaxScale + 127) << 23);

                parentIdx = rayStack[scale].parentIdx;
                maxT   = rayStack[scale].maxT;

                int shX = BitManip::floatBitsToUint(pos.x()) >> scale;
                int shY = BitManip::floatBitsToUint(pos.y()) >> scale;
                int shZ = BitManip::floatBitsToUint(pos.z()) >> scale;
                pos.x() = BitManip::uintBitsToFloat(shX << scale);
                pos.y() = BitManip::uintBitsToFloat(shY << scale);
                pos.z() = BitManip::uintBitsToFloat(shZ << scale);
                idx = (shX & 1) | ((shY & 1) << 1) | ((shZ & 1) << 2);
            }
        }
        return false;
    }
};

}

#endif /* VOXELOCTREE_HPP_ */
