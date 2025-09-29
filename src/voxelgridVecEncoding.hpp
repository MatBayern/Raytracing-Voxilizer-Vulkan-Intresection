#pragma once
#include "shaders/host_device.h"
#include "voxelgrid.hpp"
#include <glm/glm.hpp>

class VoxelGridVec final : public VoxelGrid<Aabb>
{
public:
    using VoxelType = Aabb;

    VoxelGridVec(size_t x, size_t y, size_t z, float voxelSize, vec3 org);

    virtual std::vector<Aabb> getAabbs() const noexcept;

    virtual void setVoxel(size_t x, size_t y, size_t z, const MaterialObj& material = MaterialObj{});

private:
    size_t m_voxelCount = 0;
};