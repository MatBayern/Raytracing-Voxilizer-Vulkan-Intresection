#pragma once
#include "shaders/host_device.h"
#include "tiny_obj_loader.h"
#include "voxelgrid.hpp"
#include <glm/glm.hpp>

// STD
#include <filesystem>

struct BBox
{
    glm::vec3 min, max, center;
};

class VoxelBuilder
{
public:
    VoxelBuilder(const std::filesystem::path& path);
    VoxelBuilder() = default;
    ~VoxelBuilder() = default;
    VoxelGrid buildVoxelGrid(float voxelSize);

private:
    // Attributes
    tinyobj::attrib_t m_attribs;
    std::vector<tinyobj::shape_t> m_shapes;
    std::vector<tinyobj::material_t> m_materials;

    void readObjFile(const std::filesystem::path& path);

    // This test was taken from https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox_tam.pdf
    bool axisSeparates(const glm::vec3& axis, float R, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) const;
    bool aabbAxisSeparates(const glm::vec3& h, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) const;
    bool planeSeparates(const glm::vec3& normal, const glm::vec3& h, const glm::vec3& p0) const;
    bool triBoxOverlap(const glm::vec3& c, const glm::vec3& h, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) const;
    ///

    void recenterMesh(tinyobj::attrib_t* attrib, const glm::vec3& center);
    void computeIntersection(size_t depth, size_t height, size_t width, const glm::vec3& halfVoxelSize, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, VoxelGrid& voxelGrid, const glm::vec3& gridMin);
    BBox computeBboxFromAttrib(const tinyobj::attrib_t& attrib);

    // Helpers
    size_t countSetVoxels(const VoxelGrid& grid) const;
};
