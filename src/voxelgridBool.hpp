#pragma once
#include "shaders/host_device.h"
#include "voxelgrid.hpp"
#include <glm/glm.hpp>
#include "octTree.hpp"

class VoxelGridBool final : public VoxelGrid<unsigned int>
{
public:
    using VoxelType = unsigned int;

    VoxelGridBool(size_t x, size_t y, size_t z, float voxelSize, vec3 org);

    virtual std::vector<Aabb> getAabbs() const noexcept;

    virtual void setVoxel(size_t x, size_t y, size_t z, const MaterialObj& material = MaterialObj{});

private:
    // OctTree<VoxelType> m_tree;
};