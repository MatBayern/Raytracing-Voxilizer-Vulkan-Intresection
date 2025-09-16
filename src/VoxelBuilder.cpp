#include "VoxelBuilder.hpp"
#include <glm/glm.hpp>

// STD
#include <exception>
#include <filesystem>
#include <limits>
#include <print>
#include <vector>
VoxelBuilder::VoxelBuilder(const std::filesystem::path& path)
{
    readObjFile(path);
}

void VoxelBuilder::readObjFile(const std::filesystem::path& path)
{

    if (!std::filesystem::exists(path)) {
        throw std::invalid_argument("Path does not exist!");
    }
    tinyobj::ObjReader reader;

    reader.ParseFromFile(path.string());

    if (!reader.Valid()) {
        throw std::runtime_error(std::format("Colud not get valid reader! Error message {}", reader.Error()));
    }

    m_attribs = reader.GetAttrib();
    m_shapes = reader.GetShapes();
    m_materials = reader.GetMaterials();
}

bool VoxelBuilder::axisSeparates(const glm::vec3& axis, float R, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) const
{
    // If axis is (near) zero, skip — it can't be a separating axis.
    const float eps = 1e-8f;
    const float ax = std::fabs(axis.x) + std::fabs(axis.y) + std::fabs(axis.z);
    if (ax < eps) return false;

    const float p0d = glm::dot(p0, axis);
    const float p1d = glm::dot(p1, axis);
    const float p2d = glm::dot(p2, axis);
    const float triMin = std::min(p0d, std::min(p1d, p2d));
    const float triMax = std::max(p0d, std::max(p1d, p2d));
    return (triMin > R) || (triMax < -R);
}

// AABB axis tests: clamp triangle's component ranges against box half sizes.
bool VoxelBuilder::aabbAxisSeparates(const glm::vec3& h, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) const
{
    const float minx = std::min(p0.x, std::min(p1.x, p2.x));
    const float maxx = std::max(p0.x, std::max(p1.x, p2.x));
    if (minx > h.x || maxx < -h.x) return true;

    const float miny = std::min(p0.y, std::min(p1.y, p2.y));
    const float maxy = std::max(p0.y, std::max(p1.y, p2.y));
    if (miny > h.y || maxy < -h.y) return true;

    const float minz = std::min(p0.z, std::min(p1.z, p2.z));
    const float maxz = std::max(p0.z, std::max(p1.z, p2.z));
    if (minz > h.z || maxz < -h.z) return true;

    return false;
}

// Plane (triangle normal) vs box test.
bool VoxelBuilder::planeSeparates(const glm::vec3& normal, const glm::vec3& h, const glm::vec3& p0) const
{
    // If triangle is degenerate (very small area), skip plane test.
    const float eps = 1e-8f;
    glm::vec3 an = glm::abs(normal);
    const float len = an.x + an.y + an.z;
    if (len < eps) return false;

    const float r = h.x * an.x + h.y * an.y + h.z * an.z; // projection radius of box onto normal
    const float s = glm::dot(normal, p0); // signed distance of triangle to origin (box-centered)
    return std::fabs(s) > r;
}

// Main entry: box center c, half sizes h; triangle v0,v1,v2 (all in same space).
bool VoxelBuilder::triBoxOverlap(const glm::vec3& c, const glm::vec3& h, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) const
{
    // Translate triangle so the box is centered at the origin.
    glm::vec3 p0 = v0 - c;
    glm::vec3 p1 = v1 - c;
    glm::vec3 p2 = v2 - c;

    // Edges
    glm::vec3 e0 = p1 - p0;
    glm::vec3 e1 = p2 - p1;
    glm::vec3 e2 = p0 - p2;

    // 1) Test the 3 AABB axes (x, y, z).
    if (aabbAxisSeparates(h, p0, p1, p2)) return false;

    // 2) Test 9 cross-product axes = cross(edge, axisX/Y/Z).
    // For robustness and simplicity, project all three triangle verts each time.
    // Axis from e × (1,0,0) = (0, -ez, ey) etc.
    auto testEdgeAxes = [&](const glm::vec3& e) -> bool {
        glm::vec3 Lx = {0.0f, -e.z, e.y};
        float Rx = h.y * std::fabs(Lx.y) + h.z * std::fabs(Lx.z);
        if (axisSeparates(Lx, Rx, p0, p1, p2)) return true;

        glm::vec3 Ly = {e.z, 0.0f, -e.x};
        float Ry = h.x * std::fabs(Ly.x) + h.z * std::fabs(Ly.z);
        if (axisSeparates(Ly, Ry, p0, p1, p2)) return true;

        glm::vec3 Lz = {-e.y, e.x, 0.0f};
        float Rz = h.x * std::fabs(Lz.x) + h.y * std::fabs(Lz.y);
        if (axisSeparates(Lz, Rz, p0, p1, p2)) return true;

        return false;
    };

    if (testEdgeAxes(e0)) return false;
    if (testEdgeAxes(e1)) return false;
    if (testEdgeAxes(e2)) return false;

    // 3) Test triangle plane against the box.
    glm::vec3 n = glm::cross(e0, e1);
    if (planeSeparates(n, h, p0)) return false;

    // If none of the axes separate, there is overlap.
    return true;
}

BBox VoxelBuilder::computeBboxFromAttrib(const tinyobj::attrib_t& attrib)
{
    BBox bb{};
    bb.min = {std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity()};
    bb.max = {-std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity()};

    const auto& v = attrib.vertices;
    for (size_t i = 0; i + 2 < v.size(); i += 3) {
        const float x = v[i];
        const float y = v[i + 1];
        const float z = v[i + 2];
        bb.min.x = std::min(bb.min.x, x);
        bb.max.x = std::max(bb.max.x, x);
        bb.min.y = std::min(bb.min.y, y);
        bb.max.y = std::max(bb.max.y, y);
        bb.min.z = std::min(bb.min.z, z);
        bb.max.z = std::max(bb.max.z, z);
    }

    bb.center = {
        0.5 * (bb.min.x + bb.max.x),
        0.5 * (bb.min.y + bb.max.y),
        0.5 * (bb.min.z + bb.max.z)};
    return bb;
}

void VoxelBuilder::recenterMesh(tinyobj::attrib_t* attrib, const glm::vec3& center)
{
    auto& v = attrib->vertices;
    for (size_t i = 0; i + 2 < v.size(); i += 3) {
        v[i] = static_cast<float>(static_cast<double>(v[i]) - center.x);
        v[i + 1] = static_cast<float>(static_cast<double>(v[i + 1]) - center.y);
        v[i + 2] = static_cast<float>(static_cast<double>(v[i + 2]) - center.z);
    }
}

VoxelGrid VoxelBuilder::buildVoxelGrid(float voxelSize)
{
    const auto bb = computeBboxFromAttrib(m_attribs);

    // Debug: Print bounding box
    std::cout << "Bounding box: min(" << bb.min.x << ", " << bb.min.y << ", " << bb.min.z << ")" << std::endl;
    std::cout << "              max(" << bb.max.x << ", " << bb.max.y << ", " << bb.max.z << ")" << std::endl;
    std::cout << "              center(" << bb.center.x << ", " << bb.center.y << ", " << bb.center.z << ")" << std::endl;

    const size_t width = static_cast<size_t>(std::ceil((bb.max.x - bb.min.x) / voxelSize));
    const size_t height = static_cast<size_t>(std::ceil((bb.max.y - bb.min.y) / voxelSize));
    const size_t depth = static_cast<size_t>(std::ceil((bb.max.z - bb.min.z) / voxelSize));

    std::cout << "Grid dimensions: " << width << "x" << height << "x" << depth << std::endl;
    std::cout << "Voxel size: " << voxelSize << std::endl;

    VoxelGrid voxelGrid{width, height, depth, voxelSize, bb.min};

    auto loadPos = [&](const tinyobj::index_t& idx) {
        const size_t vi = static_cast<size_t>(idx.vertex_index);
        const tinyobj::real_t vx = m_attribs.vertices[3 * vi];
        const tinyobj::real_t vy = m_attribs.vertices[3 * vi + 1];
        const tinyobj::real_t vz = m_attribs.vertices[3 * vi + 2];
        return glm::vec3{vx, vy, vz};
    };

    size_t triangleCount = 0;
    size_t voxelsSet = 0;

    for (size_t s = 0; s < m_shapes.size(); s++) {
        const auto& mesh = m_shapes[s].mesh;
        std::cout << "Shape " << s << " has " << mesh.indices.size() / 3 << " triangles" << std::endl;

        for (size_t i = 0; i < mesh.indices.size(); i += 3) { // Changed condition
            if (i + 2 >= mesh.indices.size()) break; // Safety check

            const tinyobj::index_t i0 = mesh.indices[i];
            const tinyobj::index_t i1 = mesh.indices[i + 1];
            const tinyobj::index_t i2 = mesh.indices[i + 2];

            const auto p0 = loadPos(i0);
            const auto p1 = loadPos(i1);
            const auto p2 = loadPos(i2);

            // Debug: Print first few triangles
            if (triangleCount < 3) {
                std::println("Triangle {}:", triangleCount);
                std::println("p0({},{},{})", p0.x, p0.y, p0.z);
                std::println("p1({},{},{})", p1.x, p1.y, p1.z);
                std::println("p2({},{},{})", p2.x, p2.y, p2.z);
            }

            size_t voxelsBeforeIntersection = countSetVoxels(voxelGrid);

            computeIntersection(depth, height, width,
                {voxelSize * 0.5f, voxelSize * 0.5f, voxelSize * 0.5f},
                p0, p1, p2, voxelGrid, bb.min);

            const size_t voxelsAfterIntersection = countSetVoxels(voxelGrid);

            if (voxelsAfterIntersection > voxelsBeforeIntersection) {
                voxelsSet += (voxelsAfterIntersection - voxelsBeforeIntersection);
                if (triangleCount < 3) {
                    std::println("-> Set {} voxels", (voxelsAfterIntersection - voxelsBeforeIntersection));
                }
            }

            triangleCount++;
        }
    }
    std::println("Total triangles processed: {}", triangleCount);
    std::println("Total voxels set: {}", voxelsSet);

    return voxelGrid;
}

void VoxelBuilder::computeIntersection(size_t depth, size_t height, size_t width, const glm::vec3& halfVoxelSize, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, VoxelGrid& voxelGrid, const glm::vec3& gridMin)
{
    // Compute triangle bounding box for optimization
    glm::vec3 triMin = glm::min(v0, glm::min(v1, v2));
    glm::vec3 triMax = glm::max(v0, glm::max(v1, v2));

    // Convert to voxel indices with bounds checking
    const float voxelSize = halfVoxelSize.x * 2.0f;
    int xStart = std::max(0, static_cast<int>((triMin.x - gridMin.x) / voxelSize));
    int yStart = std::max(0, static_cast<int>((triMin.y - gridMin.y) / voxelSize));
    int zStart = std::max(0, static_cast<int>((triMin.z - gridMin.z) / voxelSize));

    int xEnd = std::min(static_cast<int>(width), static_cast<int>((triMax.x - gridMin.x) / voxelSize) + 2);
    int yEnd = std::min(static_cast<int>(height), static_cast<int>((triMax.y - gridMin.y) / voxelSize) + 2);
    int zEnd = std::min(static_cast<int>(depth), static_cast<int>((triMax.z - gridMin.z) / voxelSize) + 2);

    // Only check voxels that could potentially intersect the triangle
    for (int z = zStart; z < zEnd; z++) {
        for (int y = yStart; y < yEnd; y++) {
            for (int x = xStart; x < xEnd; x++) {
                if (triBoxOverlap(voxelGrid.getCorrds(x, y, z), halfVoxelSize, v0, v1, v2)) {
                    voxelGrid.setVoxel(x, y, z);
                }
            }
        }
    }
}

// Helper function to count set voxels
size_t VoxelBuilder::countSetVoxels(const VoxelGrid& grid) const
{
    size_t count = 0;
    const auto& data = grid.data();
    for (const auto& voxel : data) {
        if (voxel.isUsed) count++;
    }
    return count;
}