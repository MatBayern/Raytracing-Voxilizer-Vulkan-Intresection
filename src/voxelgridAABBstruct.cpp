#include "voxelgrid.hpp"
#include "voxelgridAABBstruct.hpp"


VoxelGridAABBstruct::VoxelGridAABBstruct(size_t x, size_t y, size_t z, float voxelSize, vec3 org) : VoxelGrid(x, y, z, voxelSize, org)
{
    m_voxel.resize(x * y * z);
}

std::vector<Aabb> VoxelGridAABBstruct::getAabbs() const noexcept
{
    std::vector<Aabb> aabbVector;
    aabbVector.reserve(m_voxelSet);
    // Remove all unset Voxels
    for (const auto& voxel : m_voxel) {
        if (voxel.isUsed) {

            aabbVector.emplace_back(voxel.minimum, voxel.maximum); // min max
        }
    }
    aabbVector.shrink_to_fit();
    return aabbVector;
}
void VoxelGridAABBstruct::setVoxel(size_t x, size_t y, size_t z, const MaterialObj& material)
{
    if (x >= m_x || y >= m_y || z >= m_z) [[unlikely]] {
        throw std::runtime_error("Index out of bounds");
    }

    const size_t idx = map3dto1d(x, y, z);
    addMatrialIfNeeded(idx, material);

    // Treat voxelSize as cube edge length we assume this are the center corrdinates

    const glm::vec3 pos{x, y, z};

    const glm::vec3 aabbVector = m_org + ((pos + 0.5f) * m_voxelSize);
    const float half = 0.5f * m_voxelSize;

    AabbInternal aabbTmp;

    aabbTmp.minimum = aabbVector - half;
    aabbTmp.maximum = aabbVector + half;
    aabbTmp.isUsed = true;

    m_voxel[idx] = std::move(aabbTmp);
    m_voxelSet++;
}