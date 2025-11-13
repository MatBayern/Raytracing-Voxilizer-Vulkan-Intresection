#include "voxelgridVecEncoding.hpp"

#include "shaders/host_device.h"
#include "voxelgrid.hpp"
#include <glm/glm.hpp>

// STD
#include <vector>

VoxelGridVec::VoxelGridVec(size_t x, size_t y, size_t z, float voxelSize, vec3 org) : VoxelGrid(x, y, z, voxelSize, org)
{
}

std::vector<Aabb> VoxelGridVec::getAabbs() const noexcept
{
    return m_voxel;
}
void VoxelGridVec::setVoxel(size_t x, size_t y, size_t z, const MaterialObj& material)
{

    if (x >= m_x || y >= m_y || z >= m_z) [[unlikely]] {
        throw std::runtime_error("Index out of bounds");
    }

    addMatrialIfNeeded(m_voxelSet, material);

    // Treat voxelSize as cube edge length we assume this are the center corrdinates
    const float half = 0.5f * m_voxelSize;

    const glm::vec3 pos{x, y, z};

    const glm::vec3 aabbVector = m_org + ((pos + 0.5f) * m_voxelSize);

    m_voxel.emplace_back(aabbVector - half, aabbVector + half);

    m_voxelSet++;
}
