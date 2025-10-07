#include "voxelgridBool.hpp"

#include "shaders/host_device.h"
#include "voxelgrid.hpp"
#include <glm/glm.hpp>

// STD
#include <bit>
#include <vector>

VoxelGridBool::VoxelGridBool(size_t x, size_t y, size_t z, float voxelSize, vec3 org) : VoxelGrid(x, y, z, voxelSize, org)
{
}

std::vector<Aabb> VoxelGridBool::getAabbs() const noexcept
{
    std::vector<Aabb> ret;
    ret.reserve(m_voxelSet);
    const float half = 0.5f * m_voxelSize;

    const size_t totalVoxels = m_x * m_y * m_z;
    const size_t totalInts = (totalVoxels + 31) / 32; // ceil

    for (size_t intIdx = 0; intIdx < totalInts; intIdx++) {
        unsigned int intVal = m_voxel[intIdx];

        if (intVal == 0) continue; // Skip empty ints

        // Process each set bit in this int
        while (intVal != 0) {
            const int trailingZeros = std::countr_zero(intVal);
            const size_t i = intIdx * 32 + trailingZeros;

            if (i < totalVoxels) {
                const glm::vec3 gridCords = map1dto3d(i);

                const glm::vec3 aabbVector = m_org + ((gridCords + 0.5f) * m_voxelSize);
    
                // const float xF = m_org.x + (gridCords.x + 0.5f) * m_voxelSize;
                // const float yF = m_org.y + (gridCords.y + 0.5f) * m_voxelSize;
                // const float zF = m_org.z + (gridCords.z + 0.5f) * m_voxelSize;

                ret.emplace_back(aabbVector - half, aabbVector + half);
            } else {
                break;
            }
            // Clear the processed bit
            intVal &= ~(1u << trailingZeros);
        }
    }
    return ret;
}

void VoxelGridBool::setVoxel(size_t x, size_t y, size_t z, const MaterialObj& material)
{

    if (x >= m_x || y >= m_y || z >= m_z) [[unlikely]] {
        throw std::runtime_error("Index out of bounds");
    }

    const size_t idx = map3dto1d(x, y, z);
    const size_t intIdx = idx / 32; // Which int
    const size_t bitIdx = idx % 32; // Which bit in that int
    addMatrialIfNeeded(idx, material);

    m_voxel[intIdx] |= (1u << bitIdx);
    m_voxelSet++;
}
