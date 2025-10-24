#pragma once
#include "shaders/host_device.h"
#include "voxelgrid.hpp"
#include <glm/glm.hpp>

class VoxelGridVec final : public VoxelGrid<Aabb>
{
public:
    using VoxelType = Aabb;

    VoxelGridVec(size_t x, size_t y, size_t z, float voxelSize, vec3 org);

    std::vector<Aabb> getAabbs() const noexcept override;

    void setVoxel(size_t x, size_t y, size_t z, const MaterialObj& material = MaterialObj{}) override;

private:
    size_t m_voxelCount = 0;
};