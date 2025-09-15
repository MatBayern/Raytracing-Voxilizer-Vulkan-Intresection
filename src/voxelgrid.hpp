#pragma once
#include "obj_loader.h"
#include "shaders/host_device.h"
#include <glm/glm.hpp>

// STD
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <exception>
#include <execution>
#include <vector>
struct AabbInternal
{
    vec3 minimum = {0.f, 0.f, 0.f};
    vec3 maximum = {0.f, 0.f, 0.f};
    bool isUsed = false;
};
class VoxelGrid
{
    const size_t m_x;
    const size_t m_y;
    const size_t m_z;

    const vec3 m_org;

    const float m_voxelSize;
    const float m_voxelDiameter;

    //
    std::vector<MaterialObj> m_materials;
    std::vector<int> m_matIdx;
    std::vector<AabbInternal> m_voxel;

    constexpr size_t map3dto1d(size_t x, size_t y, size_t z) const noexcept
    {
        return x + m_x * (y + m_y * z);
    }

public:
    VoxelGrid(size_t x, size_t y, size_t z, float voxelSize, vec3 org = {0.f, 0.f, 0.f})
        : m_x(x),
          m_y(y),
          m_z(z),
          m_org(org),
          m_voxelSize(voxelSize),
          m_voxelDiameter(std::sqrt(3.f * m_voxelSize * m_voxelSize)),
          m_voxel(x * y * z, AabbInternal{}),
          m_matIdx(x * y * z, -1)
    {
        m_materials.reserve((x * y * z) / 4);
    }

    ~VoxelGrid() = default;

    std::vector<Aabb> getAabbs() const noexcept
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

    const std::vector<AabbInternal>& data() const noexcept
    {
        return m_voxel;
    }

    AabbInternal getVoxel(size_t x, size_t y, size_t z) const
    {
        if (x >= m_x || y >= m_y || z >= m_z) {
            throw std::runtime_error("Index out of bounds");
        }
        return m_voxel[map3dto1d(x, y, z)];
    }

    std::vector<MaterialObj> getMatrials() const
    {
        return m_materials;
    }

    std::vector<int> getMatIdx() const noexcept
    {
        std::vector<int> ret;
        ret.reserve(m_materials.size());

        for (size_t i = 0; i < m_matIdx.size(); i++) {
            if (m_matIdx[i] >= 0) {
                ret.push_back(m_matIdx[i]);
            }
        }
        return ret;
    }

    void setVoxel(size_t x, size_t y, size_t z, const MaterialObj material = MaterialObj{})
    {
        if (x >= m_x || y >= m_y || z >= m_z) {
            throw std::runtime_error("Index out of bounds");
        }

        const size_t idx = map3dto1d(x, y, z);
        const auto it = std::find(m_materials.begin(), m_materials.end(), material); // maybe use std::execution::par?

        // Set correct material
        if (it != m_materials.end()) {
            m_matIdx[idx] = static_cast<int>(std::distance(m_materials.begin(), it));
        } else {
            m_materials.push_back(material);
            m_matIdx[idx] = static_cast<int>(m_materials.size());
        }

        // Treat voxelSize as cube edge length
        const float half = 0.5f * m_voxelSize;
        const float xF = x * m_voxelDiameter;
        const float yF = m_y - y * m_voxelDiameter;
        const float zF = m_z - z * m_voxelDiameter;

        // Set Voxel size
        AabbInternal aabbTmp;
        aabbTmp.maximum = m_org + glm::vec3(xF + 0.5f * m_voxelDiameter, yF + 0.5f * m_voxelDiameter, zF + 0.5f * m_voxelDiameter);
        aabbTmp.minimum = m_org + glm::vec3(xF - 0.5f * m_voxelDiameter, yF - 0.5f * m_voxelDiameter, zF - 0.5f * m_voxelDiameter);
        aabbTmp.isUsed = true;

        m_voxel[idx] = std::move(aabbTmp);
    }
};
