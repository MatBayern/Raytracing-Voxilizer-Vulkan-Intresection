#pragma once
#include "obj_loader.h"
#include "shaders/host_device.h"
#include <glm/glm.hpp>
namespace {
struct AabbInternal
{
    vec3 minimum = {0.f, 0.f, 0.f};
    vec3 maximum = {0.f, 0.f, 0.f};
    bool isUsed = false;
};
}

class VoxelGridAABBstruct final : public VoxelGrid<AabbInternal>
{
public:
    using VoxelType = AabbInternal;

    VoxelGridAABBstruct(size_t x, size_t y, size_t z, float voxelSize, vec3 org = {0.f, 0.f, 0.f});

    virtual std::vector<Aabb> getAabbs() const noexcept;

    virtual void setVoxel(size_t x, size_t y, size_t z, const MaterialObj& material = MaterialObj{});

private:
};