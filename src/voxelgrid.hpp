#include "obj_loader.h"
#include "shaders/host_device.h"
#include <glm/glm.hpp>

// STD
#include <cmath>
#include <cstddef>
#include <exception>
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

    const float m_xOrg;
    const float m_yOrg;
    const float m_zOrg;

    const float m_voxelSize;

    //
    std::vector<MaterialObj> m_materials;
    std::vector<int> m_matIdx;
    std::vector<AabbInternal> m_voxel;

    constexpr size_t map3dto1d(size_t x, size_t y, size_t z) const noexcept
    {
        return x + m_x * (y + m_y * z);
    }

public:
    VoxelGrid(size_t x, size_t y, size_t z, float voxelSize, float xO = 0.f, float yO = 0.f, float zO = 0.f)
        : m_x(x),
          m_y(y),
          m_z(z),
          m_xOrg(xO),
          m_yOrg(yO),
          m_zOrg(zO),
          m_voxelSize(voxelSize),
          m_voxel(x * y * z, AabbInternal{}),
          m_materials(x * y * z, MaterialObj{}),
          m_matIdx(x * y * z, -1)
    {
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
            if (m_matIdx[i] > 0) {
                ret.push_back(m_matIdx[i]);
            }
        }
        return ret;
    }

    void SetVoxel(size_t x, size_t y, size_t z, const MaterialObj material = MaterialObj{})
    {
        if (x >= m_x || y >= m_y || z >= m_z) {
            throw std::runtime_error("Index out of bounds");
        }

        const size_t idx = map3dto1d(x, y, z);

        // Set correct material
        m_materials.push_back(material);
        m_matIdx[idx] = static_cast<int>(m_materials.size());

        // Treat voxelSize as cube edge length
        const float half = 0.5f * m_voxelSize;

        const float voxelDiameter = std::sqrt(2.f * m_voxelSize * m_voxelSize);
        const float xF = m_xOrg + x * voxelDiameter;
        const float yF = m_yOrg + m_y - y * voxelDiameter;
        const float zF = m_zOrg + m_z - z * voxelDiameter;

        // Set Voxel size
        AabbInternal aabbTmp;
        aabbTmp.maximum = glm::vec3(xF + 0.5f * voxelDiameter,
            yF + 0.5f * voxelDiameter,
            zF + 0.5f * voxelDiameter);
        aabbTmp.minimum = glm::vec3(xF - 0.5f * voxelDiameter,
            yF - 0.5f * voxelDiameter,
            zF - 0.5f * voxelDiameter);
        aabbTmp.isUsed = true;

        m_voxel[idx] = aabbTmp;
    }
};
