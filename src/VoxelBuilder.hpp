#pragma once
#include "shaders/host_device.h"
#include "tiny_obj_loader.h"
#include "voxelgrid.hpp"
#include "voxelgridAABBstruct.hpp"
#include <glm/glm.hpp>

// STD
#include <concepts>
#include <exception>
#include <filesystem>
#include <limits>
#include <print>
#include <thread>
#include <type_traits>
#include <vector>
namespace {
struct BBox
{
    glm::vec3 min, max, center;
};
} // namespace

template <typename Derived>
concept DerivedFromVoxelGrid = requires {
    // Check if there exists some T such that Derived inherits from Base<T>
    // This uses SFINAE to detect inheritance
    typename std::enable_if_t<
        std::is_base_of_v<VoxelGrid<typename Derived::VoxelType>, Derived>>;
    // Require that the derived class defines base_type
    typename Derived::VoxelType;
};

template <DerivedFromVoxelGrid T, bool inParaell = false>
class VoxelBuilder final
{
public:
    explicit VoxelBuilder(const std::filesystem::path& path)
    {
        readObjFile(path);
    }
    VoxelBuilder() = default;

private:
    // Attributes
    tinyobj::attrib_t m_attribs;
    std::vector<tinyobj::shape_t> m_shapes;
    std::vector<tinyobj::material_t> m_materials;

    // Functions
    void readObjFile(const std::filesystem::path& path)
    {

        if (!std::filesystem::exists(path)) {
            throw std::invalid_argument("Path does not exist!");
        }
        // tinyobj::ObjReaderConfig readerConfig;
        // readerConfig.mtl_search_path = "./"; // Path to material files
        tinyobj::ObjReader reader;

        reader.ParseFromFile(path.string());

        if (!reader.Valid()) {
            throw std::runtime_error(std::format("Colud not get valid reader! Error message {}", reader.Error()));
        }

        m_attribs = reader.GetAttrib();
        m_shapes = reader.GetShapes();
        m_materials = reader.GetMaterials();
    }

    // This test was taken from https://fileadmin.cs.lth.se/cs/Personal/Tomas_Akenine-Moller/code/tribox_tam.pdf
    bool axisSeparates(const glm::vec3& axis, float R, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) const
    {
        // If axis is (near) zero, skip it can't be a separating axis.
        constexpr float eps = 1e-8f;
        const float ax = std::fabs(axis.x) + std::fabs(axis.y) + std::fabs(axis.z);
        if (ax < eps) return false;

        const float p0d = glm::dot(p0, axis);
        const float p1d = glm::dot(p1, axis);
        const float p2d = glm::dot(p2, axis);
        const float triMin = std::min(p0d, std::min(p1d, p2d));
        const float triMax = std::max(p0d, std::max(p1d, p2d));
        return (triMin > R) || (triMax < -R);
    }

    bool aabbAxisSeparates(const glm::vec3& h, const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2) const
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
    bool planeSeparates(const glm::vec3& normal, const glm::vec3& h, const glm::vec3& p0) const
    {
        // If triangle is degenerate (very small area), skip plane test.
        constexpr float eps = 1e-8f;
        glm::vec3 an = glm::abs(normal);
        const float len = an.x + an.y + an.z;
        if (len < eps) return false;

        const float r = h.x * an.x + h.y * an.y + h.z * an.z; // projection radius of box onto normal
        const float s = glm::dot(normal, p0); // signed distance of triangle to origin (box-centered)
        return std::fabs(s) > r;
    }

    // Main entry: box center c, half sizes h; triangle v0,v1,v2 (all in same space).
    bool triBoxOverlap(const glm::vec3& c, const glm::vec3& h, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) const
    {
        // Translate triangle so the box is centered at the origin.
        const glm::vec3 p0 = v0 - c;
        const glm::vec3 p1 = v1 - c;
        const glm::vec3 p2 = v2 - c;

        // Edges
        const glm::vec3 e0 = p1 - p0;
        const glm::vec3 e1 = p2 - p1;
        const glm::vec3 e2 = p0 - p2;

        // 1) Test the 3 AABB axes (x, y, z).
        if (aabbAxisSeparates(h, p0, p1, p2)) return false;

        // 2) Test 9 cross-product axes = cross(edge, axisX/Y/Z).
        // For robustness and simplicity, project all three triangle verts each time.
        // Axis from e * (1,0,0) = (0, -ez, ey) etc.
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
        const glm::vec3 n = glm::cross(e0, e1);
        if (planeSeparates(n, h, p0)) return false;

        // If none of the axes separate, there is overlap.
        return true;
    }
    ///
    void computeIntersection(size_t depth, size_t height, size_t width, const glm::vec3& halfVoxelSize, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2, T& voxelGrid,
        const glm::vec3& grid, // unused but kept for API compatibility
        const glm::vec3& gridMin,
        MaterialObj material)
    {
        // Compute triangle bounding box for optimization
        const glm::vec3 triMin = glm::min(v0, glm::min(v1, v2));
        const glm::vec3 triMax = glm::max(v0, glm::max(v1, v2));

        const float voxelSize = halfVoxelSize.x * 2.0f;

        const int xStart = std::max(0, static_cast<int>((triMin.x - gridMin.x) / voxelSize));
        const int yStart = std::max(0, static_cast<int>((triMin.y - gridMin.y) / voxelSize));
        const int zStart = std::max(0, static_cast<int>((triMin.z - gridMin.z) / voxelSize));

        const int xEnd = std::min(static_cast<int>(width),
            static_cast<int>((triMax.x - gridMin.x) / voxelSize) + 2);
        const int yEnd = std::min(static_cast<int>(height),
            static_cast<int>((triMax.y - gridMin.y) / voxelSize) + 2);
        const int zEnd = std::min(static_cast<int>(depth),
            static_cast<int>((triMax.z - gridMin.z) / voxelSize) + 2);

        for (int z = zStart; z < zEnd; ++z) {
            for (int y = yStart; y < yEnd; ++y) {
                for (int x = xStart; x < xEnd; ++x) {
                    if (triBoxOverlap(voxelGrid.getCorrds(x, y, z),
                            halfVoxelSize, v0, v1, v2)) {
                        voxelGrid.setVoxel(x, y, z, material);
                    }
                }
            }
        }
    }

    BBox computeBboxFromAttrib(const tinyobj::attrib_t& attrib) const noexcept
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

        bb.center = (bb.min + bb.max) * 0.5f;

        return bb;
    }

    bool triBoxOverlapSchwarzSeidel(const glm::vec3& c, const glm::vec3& h, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
    {
        // Translate triangle to box-centered coordinates
        glm::vec3 p0 = v0 - c;
        glm::vec3 p1 = v1 - c;
        glm::vec3 p2 = v2 - c;

        // Triangle edges
        const glm::vec3 e0 = p1 - p0;
        const glm::vec3 e1 = p2 - p1;
        const glm::vec3 e2 = p0 - p2;

        // 1) AABB axis tests (x,y,z)

        float minx = fminf(p0.x, fminf(p1.x, p2.x));
        float maxx = fmaxf(p0.x, fmaxf(p1.x, p2.x));
        if (minx > h.x || maxx < -h.x) return false;

        float miny = fminf(p0.y, fminf(p1.y, p2.y));
        float maxy = fmaxf(p0.y, fmaxf(p1.y, p2.y));
        if (miny > h.y || maxy < -h.y) return false;

        float minz = fminf(p0.z, fminf(p1.z, p2.z));
        float maxz = fmaxf(p0.z, fmaxf(p1.z, p2.z));
        if (minz > h.z || maxz < -h.z) return false;

        // Precompute |edges| used by cross/axis projections (reduces flops & branches)
        const glm::vec3 ae0 = glm::abs(e0);
        const glm::vec3 ae1 = glm::abs(e1);
        const glm::vec3 ae2 = glm::abs(e2);

        // 2) 9 edge*axis SAT tests (optimized projections)
        const auto sepAxis = [&](float px0, float px1, float px2, float ra) -> bool {
            const float mn = fminf(px0, fminf(px1, px2));
            const float mx = fmaxf(px0, fmaxf(px1, px2));
            return (mn > ra) || (mx < -ra);
        };

        // For each edge e, the relevant axes are e * X, e * Y, e * Z.
        // We project p0,p1,p2 onto each axis; R is the box projection radius on that axis.
        {
            // e0  X = (0, -e0.z, e0.y)
            float p0d = -p0.z * e0.y + p0.y * e0.z;
            float p1d = -p1.z * e0.y + p1.y * e0.z;
            float p2d = -p2.z * e0.y + p2.y * e0.z;
            float R = h.y * ae0.z + h.z * ae0.y;
            if (sepAxis(p0d, p1d, p2d, R)) return false;

            // e0 * Y = (e0.z, 0, -e0.x)
            p0d = p0.x * e0.z - p0.z * e0.x;
            p1d = p1.x * e0.z - p1.z * e0.x;
            p2d = p2.x * e0.z - p2.z * e0.x;
            R = h.x * ae0.z + h.z * ae0.x;
            if (sepAxis(p0d, p1d, p2d, R)) return false;

            // e0 * Z = (-e0.y, e0.x, 0)
            p0d = -p0.y * e0.x + p0.x * e0.y;
            p1d = -p1.y * e0.x + p1.x * e0.y;
            p2d = -p2.y * e0.x + p2.x * e0.y;
            R = h.x * ae0.y + h.y * ae0.x;
            if (sepAxis(p0d, p1d, p2d, R)) return false;
        }
        {
            float p0d = -p0.z * e1.y + p0.y * e1.z;
            float p1d = -p1.z * e1.y + p1.y * e1.z;
            float p2d = -p2.z * e1.y + p2.y * e1.z;
            float R = h.y * ae1.z + h.z * ae1.y;
            if (sepAxis(p0d, p1d, p2d, R)) return false;

            p0d = p0.x * e1.z - p0.z * e1.x;
            p1d = p1.x * e1.z - p1.z * e1.x;
            p2d = p2.x * e1.z - p2.z * e1.x;
            R = h.x * ae1.z + h.z * ae1.x;
            if (sepAxis(p0d, p1d, p2d, R)) return false;

            p0d = -p0.y * e1.x + p0.x * e1.y;
            p1d = -p1.y * e1.x + p1.x * e1.y;
            p2d = -p2.y * e1.x + p2.x * e1.y;
            R = h.x * ae1.y + h.y * ae1.x;
            if (sepAxis(p0d, p1d, p2d, R)) return false;
        }
        {
            float p0d = -p0.z * e2.y + p0.y * e2.z;
            float p1d = -p1.z * e2.y + p1.y * e2.z;
            float p2d = -p2.z * e2.y + p2.y * e2.z;
            float R = h.y * ae2.z + h.z * ae2.y;
            if (sepAxis(p0d, p1d, p2d, R)) return false;

            p0d = p0.x * e2.z - p0.z * e2.x;
            p1d = p1.x * e2.z - p1.z * e2.x;
            p2d = p2.x * e2.z - p2.z * e2.x;
            R = h.x * ae2.z + h.z * ae2.x;
            if (sepAxis(p0d, p1d, p2d, R)) return false;

            p0d = -p0.y * e2.x + p0.x * e2.y;
            p1d = -p1.y * e2.x + p1.x * e2.y;
            p2d = -p2.y * e2.x + p2.x * e2.y;
            R = h.x * ae2.y + h.y * ae2.x;
            if (sepAxis(p0d, p1d, p2d, R)) return false;
        }

        // 3) Triangle plane vs box (using projected radius)
        const glm::vec3 n = glm::cross(e0, e1);
        const glm::vec3 an = glm::abs(n);
        const float r = h.x * an.x + h.y * an.y + h.z * an.z;
        const float s = n.x * p0.x + n.y * p0.y + n.z * p0.z; // signed distance
        if (fabsf(s) > r) return false;

        return true; // Overlap
    }

public:
    T buildVoxelGrid(float voxelSize)
    {
        const auto bb = computeBboxFromAttrib(m_attribs);

        // Debug: Print bounding box
        std::println("Bounding box: min({},{},{}):", bb.min.x, bb.min.y, bb.min.z);
        std::println("Bounding box: max({},{},{}):", bb.max.x, bb.max.y, bb.max.z);
        std::println("Bounding box: center({},{},{}):", bb.center.x, bb.center.y, bb.center.z);

        const size_t width = static_cast<size_t>(std::ceil((bb.max.x - bb.min.x) / voxelSize));
        const size_t height = static_cast<size_t>(std::ceil((bb.max.y - bb.min.y) / voxelSize));
        const size_t depth = static_cast<size_t>(std::ceil((bb.max.z - bb.min.z) / voxelSize));

        std::println("Grid dimensions: {}x{}x{}", width, height, depth);
        std::println("Voxel size: {}", voxelSize);

        T voxelGrid{width, height, depth, voxelSize, bb.min};

        const auto loadPos = [&](const tinyobj::index_t& idx) {
            const size_t vi = static_cast<size_t>(idx.vertex_index);
            const tinyobj::real_t vx = m_attribs.vertices[3 * vi];
            const tinyobj::real_t vy = m_attribs.vertices[3 * vi + 1];
            const tinyobj::real_t vz = m_attribs.vertices[3 * vi + 2];
            return glm::vec3{vx, vy, vz};
        };

        size_t triangleCount = 0;

        // ----------------- SERIAL PATH -----------------
        if constexpr (!inParaell) {
            for (size_t s = 0; s < m_shapes.size(); ++s) {
                const auto& mesh = m_shapes[s].mesh;

                for (size_t i = 0; i < mesh.indices.size(); i += 3) {

                    if (i + 2 >= mesh.indices.size()) break; // Safety check

                    int materialId = -1;
                    /*if (!mesh.material_ids.empty()) {
                        const size_t faceIndex = i / 3;
                        if (faceIndex < mesh.material_ids.size()) {
                            materialId = mesh.material_ids[faceIndex];
                        }
                    }*/

                    MaterialObj material{};
                    /*if (materialId > -1 && static_cast<size_t>(materialId) < m_materials.size()) {
                        const auto& materialToCopy = m_materials[materialId];
                        material.ior = materialToCopy.ior;
                        material.dissolve = materialToCopy.dissolve;
                        material.shininess = materialToCopy.shininess;
                        material.illum = materialToCopy.illum;
                        material.ambient = {materialToCopy.ambient[0], materialToCopy.ambient[1], materialToCopy.ambient[2]};
                        material.diffuse = {materialToCopy.diffuse[0], materialToCopy.diffuse[1], materialToCopy.diffuse[2]};
                        material.specular = {materialToCopy.specular[0], materialToCopy.specular[1], materialToCopy.specular[2]};
                        material.transmittance = {materialToCopy.transmittance[0], materialToCopy.transmittance[1], materialToCopy.transmittance[2]};
                        material.emission = {materialToCopy.emission[0], materialToCopy.emission[1], materialToCopy.emission[2]};
                    }*/

                    const tinyobj::index_t i0 = mesh.indices[i];
                    const tinyobj::index_t i1 = mesh.indices[i + 1];
                    const tinyobj::index_t i2 = mesh.indices[i + 2];

                    const auto p0 = loadPos(i0);
                    const auto p1 = loadPos(i1);
                    const auto p2 = loadPos(i2);

                    computeIntersection(
                        depth, height, width,
                        {voxelSize * 0.5f, voxelSize * 0.5f, voxelSize * 0.5f},
                        p0, p1, p2,
                        voxelGrid,
                        bb.center, bb.min,
                        material);

                    ++triangleCount;
                }
            }

            std::println("Total triangles processed: {}", triangleCount);
            m_materials.shrink_to_fit();
            return voxelGrid;
        }

        // ----------------- PARALLEL PATH -----------------
        // inParaell == true
        struct TriRef
        {
            size_t shapeIndex;
            size_t indexOffset; // index into mesh.indices (multiple of 3)
        };

        std::vector<TriRef> triList;
        triList.reserve(1024);

        for (size_t s = 0; s < m_shapes.size(); ++s) {
            const auto& mesh = m_shapes[s].mesh;
            for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
                triList.push_back(TriRef{s, i});
            }
        }

        const size_t numTris = triList.size();
        if (numTris == 0) {
            std::println("No triangles in OBJ, nothing to voxelize.");
            m_materials.shrink_to_fit();
            return voxelGrid;
        }

        triangleCount = numTris;

        const glm::vec3 halfVoxelSize{
            voxelSize * 0.5f,
            voxelSize * 0.5f,
            voxelSize * 0.5f};

        // We'll compute voxel centers ourselves to avoid touching voxelGrid from threads.
        auto voxelCenter = [&](int x, int y, int z) {
            return glm::vec3{
                bb.min.x + (static_cast<float>(x) + 0.5f) * voxelSize,
                bb.min.y + (static_cast<float>(y) + 0.5f) * voxelSize,
                bb.min.z + (static_cast<float>(z) + 0.5f) * voxelSize};
        };

        unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());

        std::println("Using {} threads for voxelization over {} triangles.", numThreads, numTris);

        const size_t chunkSize = (numTris + numThreads - 1) / numThreads;

        std::vector<std::vector<glm::uvec3>> threadHits(numThreads);
        std::vector<std::thread> workers;
        workers.reserve(numThreads);

        for (unsigned int t = 0; t < numThreads; ++t) {
            const size_t startTri = t * chunkSize;
            if (startTri >= numTris)
                break;

            const size_t endTri = std::min(numTris, startTri + chunkSize);

            workers.emplace_back([&, t, startTri, endTri]() {
                auto& localHits = threadHits[t];
                localHits.reserve(2048);

                for (size_t triIdx = startTri; triIdx < endTri; ++triIdx) {
                    const TriRef& ref = triList[triIdx];
                    const auto& mesh = m_shapes[ref.shapeIndex].mesh;
                    const size_t i = ref.indexOffset;

                    const tinyobj::index_t i0 = mesh.indices[i];
                    const tinyobj::index_t i1 = mesh.indices[i + 1];
                    const tinyobj::index_t i2 = mesh.indices[i + 2];

                    const glm::vec3 p0 = loadPos(i0);
                    const glm::vec3 p1 = loadPos(i1);
                    const glm::vec3 p2 = loadPos(i2);

                    // Same voxel range logic as computeIntersection
                    const glm::vec3 triMin = glm::min(p0, glm::min(p1, p2));
                    const glm::vec3 triMax = glm::max(p0, glm::max(p1, p2));

                    const float vSize = voxelSize;

                    const int xStart = std::max(0, static_cast<int>((triMin.x - bb.min.x) / vSize));
                    const int yStart = std::max(0, static_cast<int>((triMin.y - bb.min.y) / vSize));
                    const int zStart = std::max(0, static_cast<int>((triMin.z - bb.min.z) / vSize));

                    const int xEnd = std::min(static_cast<int>(width),
                        static_cast<int>((triMax.x - bb.min.x) / vSize) + 2);
                    const int yEnd = std::min(static_cast<int>(height),
                        static_cast<int>((triMax.y - bb.min.y) / vSize) + 2);
                    const int zEnd = std::min(static_cast<int>(depth),
                        static_cast<int>((triMax.z - bb.min.z) / vSize) + 2);

                    for (int z = zStart; z < zEnd; ++z) {
                        for (int y = yStart; y < yEnd; ++y) {
                            for (int x = xStart; x < xEnd; ++x) {
                                glm::vec3 center = voxelCenter(x, y, z);
                                if (triBoxOverlapSchwarzSeidel(center, halfVoxelSize, p0, p1, p2)) {

                                    localHits.emplace_back(x, y, z);
                                }
                            }
                        }
                    }
                }
            });
        }

        for (auto& th : workers) {
            if (th.joinable())
                th.join();
        }

        for (const auto& bucket : threadHits) {
            for (const auto& hit : bucket) {
                voxelGrid.setVoxel(hit.x, hit.y, hit.z);
            }
        }

        std::println("Total triangles processed: {}", triangleCount);
        m_materials.shrink_to_fit();
        return voxelGrid;
    }
};
