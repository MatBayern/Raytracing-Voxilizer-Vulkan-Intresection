#pragma once

#include "shaders/host_device.h"
#include "tiny_obj_loader.h"
#include <glm/glm.hpp>

// STD
#include <array>
#include <cmath>
#include <cstdint>
#include <execution>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <print>
#include <thread>
#include <vector>
/**
 *
 *  Stack size needs to be increaed for this to work !!!
 *
 */
//=========================================================
// Basic AABB utilities for your Aabb struct
//=========================================================
inline glm::vec3 aabbCenter(const Aabb& b)
{
    return (b.minimum + b.maximum) * 0.5f;
}

inline glm::vec3 aabbHalfSize(const Aabb& b)
{
    return (b.maximum - b.minimum) * 0.5f;
}

inline bool aabbContains(const Aabb& b, const glm::vec3& p)
{
    return glm::all(glm::lessThanEqual(b.minimum, p)) && glm::all(glm::lessThanEqual(p, b.maximum));
}

inline bool aabbIntersects(const Aabb& a, const Aabb& b)
{
    return (a.minimum.x <= b.maximum.x && a.maximum.x >= b.minimum.x) && (a.minimum.y <= b.maximum.y && a.maximum.y >= b.minimum.y) && (a.minimum.z <= b.maximum.z && a.maximum.z >= b.minimum.z);
}

// Create a sub AABB from parent + octant index (0..7)
inline Aabb makeChildAabb(const Aabb& parent, int octant)
{
    glm::vec3 center = aabbCenter(parent);
    glm::vec3 min = parent.minimum;
    glm::vec3 max = parent.maximum;

    glm::vec3 cmin = min;
    glm::vec3 cmax = max;

    // X
    if (octant & 1)
        cmin.x = center.x;
    else
        cmax.x = center.x;

    // Y
    if (octant & 2)
        cmin.y = center.y;
    else
        cmax.y = center.y;

    // Z
    if (octant & 4)
        cmin.z = center.z;
    else
        cmax.z = center.z;

    return {cmin, cmax};
}

//=========================================================
// OCTREE (Morton-code, flat node array)
//=========================================================
class Octree final
{
public:
    using MortonCode = std::uint64_t;
    struct Item
    {
        glm::vec3 position; // world-space position (voxel center)
        MortonCode morton; // morton code in [0, 2^(3*maxDepth) )
    };

private:
    //=========================================================
    // Morton code utilities
    //=========================================================

    // Expands a 21-bit integer into 63 bits by inserting 2 zeros between each bit.
    constexpr MortonCode expandBits(std::uint32_t v) const noexcept
    {
        MortonCode x = v & 0x1fffffu; // 21 bits
        x = (x | (x << 32)) & 0x1f00000000ffffULL;
        x = (x | (x << 16)) & 0x1f0000ff0000ffULL;
        x = (x | (x << 8)) & 0x100f00f00f00f00FULL;
        x = (x | (x << 4)) & 0x10c30c30c30c30C3ULL;
        x = (x | (x << 2)) & 0x1249249249249249ULL;
        return x;
    }

    constexpr MortonCode morton3D(std::uint32_t x, std::uint32_t y, std::uint32_t z) const noexcept
    {
        MortonCode xx = expandBits(x);
        MortonCode yy = expandBits(y) << 1;
        MortonCode zz = expandBits(z) << 2;
        return xx | yy | zz;
    }

    struct Node
    {
        // AABB is no longer stored per node to save memory.
        std::array<std::uint32_t, 8> children{}; // indices into m_nodes, or INVALID

        std::uint32_t start = 0; // start index into m_items
        std::uint32_t count = 0; // number of items in this subtree

        constexpr bool isLeaf() const noexcept
        {
            for (auto c : children) {
                if (c != INVALID_INDEX) return false;
            }
            return true;
        }
    };

    size_t estimateMaxItemsPerLeaf(size_t numItems) const noexcept
    {
        // Aim for roughly ~1000 leaves, clamp to sane ranges.
        // Larger scenes → larger leaves.
        if (numItems == 0) return 64; // fallback

        size_t k = numItems / 1000; // target items per leaf
        if (k < 32) k = 32; // lower bound
        if (k > 512) k = 512; // upper bound
        return k;
    }

    static constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

    // Tree data
    std::vector<Item> m_items; // sorted by morton after build
    std::vector<Node> m_nodes; // flat array, node 0 = root

    Aabb m_rootBounds{};
    size_t m_maxItems;
    size_t m_maxDepth;

    glm::vec3 m_halfVoxelSize{0.0f};
    // OBJ data
    struct ObjMesh
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
    };

    //=====================================================
    // Helpers
    //=====================================================
    ObjMesh readObjFile(const std::filesystem::path& path) const
    {
        if (!std::filesystem::exists(path)) {
            throw std::invalid_argument("Path does not exist!");
        }

        tinyobj::ObjReader reader;
        reader.ParseFromFile(path.string());

        if (!reader.Valid()) {
            throw std::runtime_error(
                std::format("Could not get valid reader! Error message {}", reader.Error()));
        }

        ObjMesh mesh;
        mesh.attrib = reader.GetAttrib();
        mesh.shapes = reader.GetShapes();
        return mesh; // NRVO
    }

    // Map a world-space position into [0, 2^maxDepth-1]^3 and then to a Morton code
    MortonCode encodePositionToMorton(const glm::vec3& pos) const noexcept
    {
        glm::vec3 size = m_rootBounds.maximum - m_rootBounds.minimum;
        // Avoid division by zero; clamp degenerate axes
        if (size.x == 0.0f) size.x = 1.0f;
        if (size.y == 0.0f) size.y = 1.0f;
        if (size.z == 0.0f) size.z = 1.0f;

        glm::vec3 rel = (pos - m_rootBounds.minimum) / size;

        const float gridMaxF = static_cast<float>((1u << m_maxDepth) - 1u);

        auto clamp01 = [](float v) {
            return std::max(0.0f, std::min(1.0f, v));
        };

        const std::uint32_t ix = static_cast<std::uint32_t>(clamp01(rel.x) * gridMaxF + 0.5f);
        const std::uint32_t iy = static_cast<std::uint32_t>(clamp01(rel.y) * gridMaxF + 0.5f);
        const std::uint32_t iz = static_cast<std::uint32_t>(clamp01(rel.z) * gridMaxF + 0.5f);

        return morton3D(ix, iy, iz);
    }

    // Recursive build of nodes over sorted m_items
    std::uint32_t buildNodeRecursive(std::uint32_t begin,
        std::uint32_t end,
        const Aabb& bounds,
        std::uint32_t depth)
    {
        const std::uint32_t nodeIndex = static_cast<std::uint32_t>(m_nodes.size());
        m_nodes.emplace_back();

        m_nodes[nodeIndex].start = begin;
        m_nodes[nodeIndex].count = end - begin;
        m_nodes[nodeIndex].children.fill(INVALID_INDEX);

        if (depth >= m_maxDepth || m_nodes[nodeIndex].count <= m_maxItems) {
            return nodeIndex;
        }

        const std::uint32_t totalDepth = static_cast<std::uint32_t>(m_maxDepth);
        const std::uint32_t levelShift = 3u * (totalDepth - 1u - depth);

        std::uint32_t cur = begin;
        for (int child = 0; child < 8; ++child) {
            if (cur >= end)
                break;

            const std::uint32_t childBegin = cur;

            while (cur < end) {
                MortonCode code = m_items[cur].morton;
                int octant = static_cast<int>((code >> levelShift) & 0x7u);
                if (octant != child) break;
                ++cur;
            }

            if (childBegin == cur)
                continue;

            const Aabb childBounds = makeChildAabb(bounds, child);
            std::uint32_t childIndex = buildNodeRecursive(childBegin, cur, childBounds, depth + 1);

            m_nodes[nodeIndex].children[child] = childIndex;
        }

        return nodeIndex;
    }

    void buildTree()
    {
        // Sort by morton code
        std::sort(std::execution::par_unseq, m_items.begin(), m_items.end(),
            [](const Item& a, const Item& b) { return a.morton < b.morton; });

        m_nodes.clear();
        // A more conservative reserve: roughly one node per several items
        if (!m_items.empty()) {
            m_nodes.reserve(std::max<size_t>(1, m_items.size() / 4));
        }

        buildNodeRecursive(0, static_cast<std::uint32_t>(m_items.size()), m_rootBounds, 0);
    }

    void queryRecursive(std::uint32_t nodeIndex,
        const Aabb& nodeBounds,
        const Aabb& range,
        std::vector<Item>* out) const
    {
        if (!aabbIntersects(nodeBounds, range))
            return;

        const Node& node = m_nodes[nodeIndex];

        if (node.isLeaf()) {
            const std::uint32_t end = node.start + node.count;
            for (std::uint32_t i = node.start; i < end; ++i) {
                const Item& it = m_items[i];
                if (aabbContains(range, it.position))
                    out->push_back(it);
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                std::uint32_t ci = node.children[c];
                if (ci != INVALID_INDEX) {
                    Aabb childBounds = makeChildAabb(nodeBounds, c);
                    queryRecursive(ci, childBounds, range, out);
                }
            }
        }
    }

    void traverseNodesRawRecursive(std::uint32_t nodeIndex, std::vector<Aabb>* out) const
    {
        const Node& node = m_nodes[nodeIndex];

        if (node.isLeaf()) {
            const std::uint32_t end = node.start + node.count;
            for (std::uint32_t i = node.start; i < end; ++i) {
                const Item& it = m_items[i];
                out->emplace_back(it.position - m_halfVoxelSize, it.position + m_halfVoxelSize); // min max
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                std::uint32_t ci = node.children[c];
                if (ci != INVALID_INDEX)
                    traverseNodesRawRecursive(ci, out);
            }
        }
    }

    bool triBoxOverlapSchwarzSeidel(const glm::vec3& c, const glm::vec3& h, const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2) const noexcept
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

        // Precompute |edges|
        const glm::vec3 ae0 = glm::abs(e0);
        const glm::vec3 ae1 = glm::abs(e1);
        const glm::vec3 ae2 = glm::abs(e2);

        const auto sepAxis = [&](float px0, float px1, float px2, float ra) -> bool {
            const float mn = fminf(px0, fminf(px1, px2));
            const float mx = fmaxf(px0, fmaxf(px1, px2));
            return (mn > ra) || (mx < -ra);
        };

        {
            float p0d = -p0.z * e0.y + p0.y * e0.z;
            float p1d = -p1.z * e0.y + p1.y * e0.z;
            float p2d = -p2.z * e0.y + p2.y * e0.z;
            float R = h.y * ae0.z + h.z * ae0.y;
            if (sepAxis(p0d, p1d, p2d, R)) return false;

            p0d = p0.x * e0.z - p0.z * e0.x;
            p1d = p1.x * e0.z - p1.z * e0.x;
            p2d = p2.x * e0.z - p2.z * e0.x;
            R = h.x * ae0.z + h.z * ae0.x;
            if (sepAxis(p0d, p1d, p2d, R)) return false;

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
            float R2 = h.x * ae2.z + h.z * ae2.x;
            if (sepAxis(p0d, p1d, p2d, R2)) return false;

            p0d = -p0.y * e2.x + p0.x * e2.y;
            p1d = -p1.y * e2.x + p1.x * e2.y;
            p2d = -p2.y * e2.x + p2.x * e2.y;
            float R3 = h.x * ae2.y + h.y * ae2.x;
            if (sepAxis(p0d, p1d, p2d, R3)) return false;
        }

        // Triangle plane vs box
        const glm::vec3 n = glm::cross(e0, e1);
        const glm::vec3 an = glm::abs(n);
        const float r = h.x * an.x + h.y * an.y + h.z * an.z;
        const float s = n.x * p0.x + n.y * p0.y + n.z * p0.z; // signed distance
        if (fabsf(s) > r) return false;

        return true;
    }

public:
    explicit Octree(const std::filesystem::path& path, float voxSize)
    {
        // We support up to 21 levels (Morton 21 bits per axis)

        ObjMesh mesh = readObjFile(path);

        m_rootBounds = computeBboxFromAttrib(mesh.attrib);
        m_maxDepth = static_cast<size_t>(std::log2(std::ceil((m_rootBounds.maximum.x - m_rootBounds.minimum.x) / voxSize)) + 2);

        if (m_maxDepth > 21) {
            throw std::runtime_error("We support up to 21 levels (Morton 21 bits per axis)!");
        }

        buildVoxelGrid(voxSize, mesh);
    }

    std::vector<Aabb> getAabbs() const
    {
        std::vector<Aabb> ret;
        traverseNodesRawRecursive(0, &ret);
        return ret;
    }

    size_t getMemoryUsageBytes() const noexcept
    {
        size_t bytes = 0;

        // 1) Speicher der Items
        bytes += m_items.capacity() * sizeof(Item);

        // 2) Speicher der Nodes
        bytes += m_nodes.capacity() * sizeof(Node);

        return bytes;
    }

    Octree(const Octree&) = delete;
    Octree& operator=(const Octree&) = delete;
    Octree(Octree&&) noexcept = default;
    Octree& operator=(Octree&&) noexcept = default;

private:
    // Insert a point (voxel center) into the octree
    bool insert(const glm::vec3& pos)
    {
        if (!aabbContains(m_rootBounds, pos)) {
            return false;
        }

        m_items.emplace_back(pos, encodePositionToMorton(pos)); // pos morton

        return true;
    }

    // Query items inside range
    void query(const Aabb& range, std::vector<Item>* out)
    {
        buildTree();
        if (m_nodes.empty())
            return;

        queryRecursive(0, m_rootBounds, range, out);
    }

    std::vector<Item> query(const Aabb& range)
    {
        std::vector<Item> result;
        query(range, &result);
        return result;
    }

    Aabb computeBboxFromAttrib(const tinyobj::attrib_t& attrib) const noexcept
    {
        Aabb bb{};
        bb.minimum = {
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity()};
        bb.maximum = {
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity(),
            -std::numeric_limits<float>::infinity()};

        const auto& v = attrib.vertices;
        for (size_t i = 0; i + 2 < v.size(); i += 3) {
            const float x = v[i];
            const float y = v[i + 1];
            const float z = v[i + 2];
            bb.minimum.x = std::min(bb.minimum.x, x);
            bb.maximum.x = std::max(bb.maximum.x, x);
            bb.minimum.y = std::min(bb.minimum.y, y);
            bb.maximum.y = std::max(bb.maximum.y, y);
            bb.minimum.z = std::min(bb.minimum.z, z);
            bb.maximum.z = std::max(bb.maximum.z, z);
        }

        return bb;
    }

    // Build voxel grid and fill octree with voxel centers (Morton-coded)
    // Build voxel grid and fill octree with voxel centers (Morton-coded)
    void buildVoxelGrid(float voxelSize, const ObjMesh& ObjData)
    {
        const auto bb = computeBboxFromAttrib(ObjData.attrib);
        m_rootBounds = bb;

        const size_t width = static_cast<size_t>(std::ceil((bb.maximum.x - bb.minimum.x) / voxelSize));
        const size_t height = static_cast<size_t>(std::ceil((bb.maximum.y - bb.minimum.y) / voxelSize));
        const size_t depth = static_cast<size_t>(std::ceil((bb.maximum.z - bb.minimum.z) / voxelSize));

        std::println("Grid dimensions: {}x{}x{}", width, height, depth);
        std::println("Voxel size: {}", voxelSize);

        const auto getCoords = [voxelSize, &bb](size_t x, size_t y, size_t z) -> glm::vec3 {
            glm::vec3 posvec{
                static_cast<float>(x),
                static_cast<float>(y),
                static_cast<float>(z)};
            glm::vec3 ret = bb.minimum + (posvec + 0.5f) * voxelSize;
            return ret;
        };

        const auto loadPos = [&](const tinyobj::index_t& idx) -> glm::vec3 {
            const size_t vi = static_cast<size_t>(idx.vertex_index);
            const tinyobj::real_t vx = ObjData.attrib.vertices[3 * vi];
            const tinyobj::real_t vy = ObjData.attrib.vertices[3 * vi + 1];
            const tinyobj::real_t vz = ObjData.attrib.vertices[3 * vi + 2];
            return glm::vec3{vx, vy, vz};
        };

        const glm::vec3 halfVoxelSize{
            voxelSize * 0.5f,
            voxelSize * 0.5f,
            voxelSize * 0.5f};
        m_halfVoxelSize = halfVoxelSize;

        // Flatten all triangles from all shapes into a single list
        struct TriRef
        {
            size_t shapeIndex;
            size_t indexOffset; // index into mesh.indices (multiple of 3)
        };

        std::vector<TriRef> triList;
        triList.reserve(1024);

        size_t totalTriangles = 0;
        for (size_t s = 0; s < ObjData.shapes.size(); ++s) {
            const auto& mesh = ObjData.shapes[s].mesh;
            totalTriangles += mesh.indices.size() / 3;
        }
        triList.reserve(totalTriangles);

        for (size_t s = 0; s < ObjData.shapes.size(); ++s) {
            const auto& mesh = ObjData.shapes[s].mesh;
            for (size_t i = 0; i + 2 < mesh.indices.size(); i += 3) {
                triList.push_back(TriRef{s, i});
            }
        }

        const size_t numTris = triList.size();

        if (numTris == 0) {
            std::println("No triangles in OBJ, nothing to voxelize.");
            return;
        }

        // Decide number of worker threads
        unsigned int numThreads = std::max(1u, std::thread::hardware_concurrency());

        std::println("Using {} threads for voxelization over {} triangles.", numThreads, numTris);

        const size_t chunkSize = (numTris + numThreads - 1) / numThreads;

        // One bucket per (potential) thread
        std::vector<std::vector<Item>> threadBuckets(numThreads);
        std::vector<size_t> threadTriangleCounts(numThreads, 0);

        std::vector<std::thread> workers;
        workers.reserve(numThreads);

        for (unsigned int t = 0; t < numThreads; ++t) {
            const size_t startTri = t * chunkSize;
            if (startTri >= numTris)
                break; // no more work chunks

            const size_t endTri = std::min(numTris, startTri + chunkSize);

            workers.emplace_back([&, t, startTri, endTri]() {
                auto& localItems = threadBuckets[t];
                size_t localTriangleCount = 0;
                localItems.reserve(256); // heuristic; will grow if needed

                for (size_t triIdx = startTri; triIdx < endTri; ++triIdx) {
                    const TriRef& ref = triList[triIdx];
                    const auto& mesh = ObjData.shapes[ref.shapeIndex].mesh;
                    const size_t i = ref.indexOffset;

                    const tinyobj::index_t i0 = mesh.indices[i];
                    const tinyobj::index_t i1 = mesh.indices[i + 1];
                    const tinyobj::index_t i2 = mesh.indices[i + 2];

                    const glm::vec3 p0 = loadPos(i0);
                    const glm::vec3 p1 = loadPos(i1);
                    const glm::vec3 p2 = loadPos(i2);

                    const glm::vec3 triMin = glm::min(p0, glm::min(p1, p2));
                    const glm::vec3 triMax = glm::max(p0, glm::max(p1, p2));

                    const int xStart = std::max(0, static_cast<int>((triMin.x - bb.minimum.x) / voxelSize));
                    const int yStart = std::max(0, static_cast<int>((triMin.y - bb.minimum.y) / voxelSize));
                    const int zStart = std::max(0, static_cast<int>((triMin.z - bb.minimum.z) / voxelSize));

                    const int xEnd = std::min(static_cast<int>(width),
                        static_cast<int>((triMax.x - bb.minimum.x) / voxelSize) + 2);
                    const int yEnd = std::min(static_cast<int>(height),
                        static_cast<int>((triMax.y - bb.minimum.y) / voxelSize) + 2);
                    const int zEnd = std::min(static_cast<int>(depth),
                        static_cast<int>((triMax.z - bb.minimum.z) / voxelSize) + 2);

                    for (int z = zStart; z < zEnd; ++z) {
                        for (int y = yStart; y < yEnd; ++y) {
                            for (int x = xStart; x < xEnd; ++x) {
                                glm::vec3 center = getCoords(
                                    static_cast<size_t>(x),
                                    static_cast<size_t>(y),
                                    static_cast<size_t>(z));

                                if (triBoxOverlapSchwarzSeidel(center, halfVoxelSize, p0, p1, p2)) {
                                    Item it;
                                    it.position = center;
                                    it.morton = encodePositionToMorton(center);
                                    localItems.push_back(it);
                                }
                            }
                        }
                    }

                    ++localTriangleCount;
                }

                threadTriangleCounts[t] = localTriangleCount;
            });
        }

        // Join all threads
        for (auto& th : workers) {
            if (th.joinable())
                th.join();
        }

        // Merge thread-local buckets into m_items (single-threaded, safe)
        size_t totalItems = 0;
        for (const auto& bucket : threadBuckets)
            totalItems += bucket.size();

        m_items.reserve(totalItems);

        for (auto& bucket : threadBuckets) {
            m_items.insert(m_items.end(),
                std::make_move_iterator(bucket.begin()),
                std::make_move_iterator(bucket.end()));
        }

        const size_t triangleCount = std::accumulate(threadTriangleCounts.begin(), threadTriangleCounts.end(), size_t{0u});

        std::println("Total triangles processed: {}", triangleCount);
        std::println("Total voxels inserted (before tree build): {}", m_items.size());

        m_maxItems = estimateMaxItemsPerLeaf(m_items.size());
        // Now actually build the Morton octree (this already uses parallel sort)
        buildTree();
        m_nodes.shrink_to_fit();
        m_items.shrink_to_fit();

        std::println("Total octree nodes: {}", m_nodes.size());
    }
};
