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
#include <unordered_map>
#include <vector>

template <typename T>
class VoxelGrid
{
protected:
    const size_t m_x;
    const size_t m_y;
    const size_t m_z;

    size_t m_voxelSet = 0;

    const vec3 m_org;

    const float m_voxelSize;
    const float m_voxelDiameter;

    //
    std::vector<MaterialObj> m_materials;
    std::vector<int> m_matIdx;
    std::vector<T> m_voxel;
    std::unordered_map<MaterialObj, int> m_materialMap;

    //
    glm::mat3 m_Tranfomermatrix;

    // Helpers
    constexpr size_t map3dto1d(size_t x, size_t y, size_t z) const noexcept
    {
        return x + m_x * (y + m_y * z);
    }

    constexpr glm::vec3 map1dto3d(size_t i) const noexcept
    {
        const size_t x = i % m_x;
        const size_t y = (i / m_x) % m_y;
        const size_t z = i / (m_x * m_y);

        return {x, y, z};
    }

    constexpr glm::mat3 mat3FromLinear()
    {
        glm::mat3 A(0.f); // column-major
        for (int j = 0; j < 3; ++j) {
            glm::vec3 e(0.0f);
            e[j] = 1.0f; // standard basis e_j
            A[j] = m_org + (e + 0.5f) * m_voxelSize; // column j is f(e_j)
        }
        return A;
    }

public:
    VoxelGrid(size_t x, size_t y, size_t z, float voxelSize, vec3 org = {0.f, 0.f, 0.f})
        : m_x(x),
          m_y(y),
          m_z(z),
          m_org(org),
          m_voxelSize(voxelSize),
          m_voxelDiameter(std::sqrt(3.f * m_voxelSize * m_voxelSize)),
          m_voxel(x * y * z, T{}),
          m_matIdx(x * y * z, -1)
    {
        m_materials.reserve((x * y * z) / 4);
        m_Tranfomermatrix = mat3FromLinear();
    }

    virtual ~VoxelGrid() = default;

    T getVoxel(size_t x, size_t y, size_t z) const
    {
        if (x >= m_x || y >= m_y || z >= m_z) [[unlikely]] {
            throw std::runtime_error("Index out of bounds");
        }
        return m_voxel[map3dto1d(x, y, z)];
    }

    std::vector<MaterialObj> getMatrials() const noexcept
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
    vec3 getCorrds(size_t x, size_t y, size_t z) const
    {
        if (x >= m_x || y >= m_y || z >= m_z) [[unlikely]] {
            throw std::runtime_error("Index out of bounds");
        }
        const float worldX = m_org.x + (static_cast<float>(x) + 0.5f) * m_voxelSize;
        const float worldY = m_org.y + (static_cast<float>(y) + 0.5f) * m_voxelSize;
        const float worldZ = m_org.z + (static_cast<float>(z) + 0.5f) * m_voxelSize;

        vec3 ret = {worldX, worldY, worldZ}; // RVO
        return ret;
    }

    void addMatrialIfNeeded(size_t idx, const MaterialObj& material)
    {
        const auto it = m_materialMap.find(material);

        if (it != m_materialMap.end()) [[likely]] {
            m_matIdx[idx] = it->second;
        } else {
            const int newIndex = static_cast<int>(m_materials.size());
            m_materials.push_back(material);
            m_materialMap[material] = newIndex;
            m_matIdx[idx] = newIndex;
        }
    }

    // Abstract Methods
    virtual std::vector<Aabb> getAabbs() const noexcept = 0;

    virtual void setVoxel(size_t x, size_t y, size_t z, const MaterialObj& material = MaterialObj{}) = 0;
};
