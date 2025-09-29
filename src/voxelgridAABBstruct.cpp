#include "voxelgrid.hpp"
#include "voxelgridAABBstruct.hpp"


VoxelGridAABBstruct::VoxelGridAABBstruct(size_t x, size_t y, size_t z, float voxelSize, vec3 org) : VoxelGrid(x, y, z, voxelSize, org)
{
}

std::vector<Aabb> VoxelGridAABBstruct::getAabbs() const noexcept
{
    std::vector<Aabb> aabbVector;
    aabbVector.reserve(m_voxel.size() / 4);
    // Remove all unset Voxels
    for (const auto& voxel : m_voxel) {
        if (voxel.isUsed) {
            Aabb tmp{};
            tmp.maximum = voxel.maximum;
            tmp.minimum = voxel.minimum;

            aabbVector.push_back(tmp);
        }
    }

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
    const float half = 0.5f * m_voxelSize;
    const float xF = m_org.x + (x + 0.5f) * m_voxelSize;
    const float yF = m_org.y + (y + 0.5f) * m_voxelSize;
    const float zF = m_org.z + (z + 0.5f) * m_voxelSize;

    AabbInternal aabbTmp;
    aabbTmp.minimum = {xF - half, yF - half, zF - half};
    aabbTmp.maximum = {xF + half, yF + half, zF + half};
    aabbTmp.isUsed = true;

    m_voxel[idx] = std::move(aabbTmp);
    m_voxelSet++;
}