#pragma once
#include <glm/glm.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>
#include <vector>

// Triangle structure
struct Triangle
{
    glm::vec3 v0, v1, v2;
    glm::vec3 normal;

    Triangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
        : v0(v0), v1(v1), v2(v2)
    {
        glm::vec3 e1 = v1 - v0;
        glm::vec3 e2 = v2 - v0;
        normal = glm::cross(e1, e2);
    }
};

// Octree node structure
struct OctreeNode
{
    static const int SUBGRID_SIZE = 4; // 4x4x4 voxel sub-grid

    int level;
    int mortonCode;

    // Pointers to neighbors and children
    int parent;
    int firstChild; // -1 if leaf
    int xPosNeighbor, xNegNeighbor;

    // Voxel sub-grid data (64 bits for 4x4x4 grid)
    uint64_t voxelData;

    // For propagation
    bool flipFlag;
    bool insideFlag;

    OctreeNode() : level(0), mortonCode(0), parent(-1), firstChild(-1),
                   xPosNeighbor(-1), xNegNeighbor(-1), voxelData(0),
                   flipFlag(false), insideFlag(false) {}
};

// Main voxelization class
class SparseOctreeVoxelizer
{
private:
    std::vector<Triangle> m_triangles;
    std::vector<std::vector<OctreeNode>> m_levelNodes; // Nodes per level
    int m_maxLevel;
    int m_gridSize;
    float m_voxelSize;
    glm::vec3 m_gridMin;

public:
    SparseOctreeVoxelizer(int gridSize) : m_gridSize(gridSize)
    {
        m_maxLevel = static_cast<int>(std::log2(gridSize)) - 2; // -2 for sub-grid
        m_voxelSize = 1.0f / gridSize;
        m_gridMin = glm::vec3(0, 0, 0);
        m_levelNodes.resize(m_maxLevel + 1);
    }

    // Add triangle to the mesh
    void addTriangle(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2)
    {
        m_triangles.push_back(Triangle(v0, v1, v2));
    }

    // Morton code generation (interleave bits)
    int encodeMorton(int x, int y, int z) const
    {
        auto splitBy3 = [](int a) -> int {
            int x = a & 0x3FF; // Only 10 bits
            x = (x | x << 16) & 0x30000FF;
            x = (x | x << 8) & 0x300F00F;
            x = (x | x << 4) & 0x30C30C3;
            x = (x | x << 2) & 0x9249249;
            return x;
        };

        return splitBy3(x) | (splitBy3(y) << 1) | (splitBy3(z) << 2);
    }

    // Triangle/box overlap test (conservative)
    bool triangleBoxOverlap(const Triangle& tri, const glm::vec3& boxMin, float boxSize)
    {
        glm::vec3 boxMax = boxMin + glm::vec3(boxSize, boxSize, boxSize);
        glm::vec3 boxCenter = (boxMin + boxMax) * 0.5f;
        glm::vec3 boxHalfSize(boxSize * 0.5f, boxSize * 0.5f, boxSize * 0.5f);

        // Translate triangle as if box center is at origin
        glm::vec3 v0 = tri.v0 - boxCenter;
        glm::vec3 v1 = tri.v1 - boxCenter;
        glm::vec3 v2 = tri.v2 - boxCenter;

        // Test triangle normal
        float d = glm::dot(v0, tri.normal);
        float r = boxHalfSize.x * std::abs(tri.normal.x) + boxHalfSize.y * std::abs(tri.normal.y) + boxHalfSize.z * std::abs(tri.normal.z);
        if (std::abs(d) > r) return false;

        // Test edge normals (9 tests)
        glm::vec3 edges[3] = {v1 - v0, v2 - v1, v0 - v2};

        for (int i = 0; i < 3; i++) {
            // Test X-axis cross edge
            float a = edges[i].z * v0.y - edges[i].y * v0.z;
            float b = edges[i].z * v2.y - edges[i].y * v2.z;
            r = boxHalfSize.y * std::abs(edges[i].z) + boxHalfSize.z * std::abs(edges[i].y);
            if (std::min(a, b) > r || std::max(a, b) < -r) return false;

            // Test Y-axis cross edge
            a = -edges[i].z * v0.x + edges[i].x * v0.z;
            b = -edges[i].z * v2.x + edges[i].x * v2.z;
            r = boxHalfSize.x * std::abs(edges[i].z) + boxHalfSize.z * std::abs(edges[i].x);
            if (std::min(a, b) > r || std::max(a, b) < -r) return false;

            // Test Z-axis cross edge
            a = edges[i].y * v0.x - edges[i].x * v0.y;
            b = edges[i].y * v2.x - edges[i].x * v2.y;
            r = boxHalfSize.x * std::abs(edges[i].y) + boxHalfSize.y * std::abs(edges[i].x);
            if (std::min(a, b) > r || std::max(a, b) < -r) return false;
        }

        return true;
    }

    // Step 1: Determine active level-1 nodes
    std::vector<int> determineActiveNodes()
    {
        int level1Size = m_gridSize / OctreeNode::SUBGRID_SIZE;
        std::unordered_set<int> activeSet;

        // Conservative voxelization at level-1 resolution
        for (const auto& tri : m_triangles) {
            // Compute bounding box
            glm::vec3 minBB(std::min({tri.v0.x, tri.v1.x, tri.v2.x}),
                std::min({tri.v0.y, tri.v1.y, tri.v2.y}),
                std::min({tri.v0.z, tri.v1.z, tri.v2.z}));
            glm::vec3 maxBB(std::max({tri.v0.x, tri.v1.x, tri.v2.x}),
                std::max({tri.v0.y, tri.v1.y, tri.v2.y}),
                std::max({tri.v0.z, tri.v1.z, tri.v2.z}));

            float level1VoxelSize = m_voxelSize * OctreeNode::SUBGRID_SIZE;

            int minX = static_cast<int>(minBB.x / level1VoxelSize);
            int minY = static_cast<int>(minBB.y / level1VoxelSize);
            int minZ = static_cast<int>(minBB.z / level1VoxelSize);
            int maxX = static_cast<int>(maxBB.x / level1VoxelSize);
            int maxY = static_cast<int>(maxBB.y / level1VoxelSize);
            int maxZ = static_cast<int>(maxBB.z / level1VoxelSize);

            // Test all voxels in bounding box
            for (int z = minZ; z <= maxZ && z < level1Size; z++) {
                for (int y = minY; y <= maxY && y < level1Size; y++) {
                    for (int x = minX; x <= maxX && x < level1Size; x++) {
                        glm::vec3 voxelMin(x * level1VoxelSize,
                            y * level1VoxelSize,
                            z * level1VoxelSize);

                        if (triangleBoxOverlap(tri, voxelMin, level1VoxelSize)) {
                            activeSet.insert(encodeMorton(x, y, z));
                        }
                    }
                }
            }
        }

        std::vector<int> activeNodes(activeSet.begin(), activeSet.end());
        std::sort(activeNodes.begin(), activeNodes.end());
        return activeNodes;
    }

    // Step 2: Construct octree bottom-up
    void constructOctree(const std::vector<int>& activeLevel1)
    {
        if (activeLevel1.empty()) return;

        // Create level-0 nodes (8 children per active level-1 node)
        m_levelNodes[0].reserve(activeLevel1.size() * 8);
        for (int morton : activeLevel1) {
            for (int child = 0; child < 8; child++) {
                OctreeNode node;
                node.level = 0;
                node.mortonCode = (morton << 3) | child;
                m_levelNodes[0].push_back(node);
            }
        }

        // Setup x-neighbors for level-0
        setupXNeighbors(0);

        // Build higher levels
        std::vector<int> currentActive = activeLevel1;

        for (int level = 1; level <= m_maxLevel; level++) {
            std::unordered_set<int> parentSet;

            for (int morton : currentActive) {
                int parentMorton = morton >> 3;
                parentSet.insert(parentMorton);
            }

            std::vector<int> parents(parentSet.begin(), parentSet.end());
            std::sort(parents.begin(), parents.end());

            // Create nodes for this level
            int childBase = level > 1 ? static_cast<int>(m_levelNodes[static_cast<size_t>(level - 1)].size()) / 8 : 0;

            for (int i = 0; i < parents.size(); i++) {
                for (int child = 0; child < 8; child++) {
                    OctreeNode node;
                    node.level = level;
                    node.mortonCode = (parents[i] << 3) | child;
                    node.firstChild = (childBase + i * 8) * 8;
                    m_levelNodes[level].push_back(node);
                }
            }

            setupXNeighbors(level);
            currentActive = parents;
        }
    }

    // Setup x-direction neighbors
    void setupXNeighbors(int level)
    {
        auto& nodes = m_levelNodes[level];

        for (int i = 0; i < nodes.size(); i++) {
            // Check if next node is x-neighbor
            if (i + 1 < nodes.size()) {
                int dx = (nodes[i + 1].mortonCode & 1) - (nodes[i].mortonCode & 1);
                if (dx == 1) {
                    nodes[i].xPosNeighbor = i + 1;
                    nodes[i + 1].xNegNeighbor = i;
                }
            }
        }
    }

    // 2D edge function test for solid voxelization
    struct EdgeFunction
    {
        float nx, ny; // Edge normal in 2D
        float d; // Offset

        bool test(float px, float py) const
        {
            return (nx * px + ny * py + d) > 0;
        }
    };

    // Setup edge functions for YZ plane
    std::array<EdgeFunction, 3> setupEdgeFunctions(const Triangle& tri)
    {
        std::array<EdgeFunction, 3> edges;

        glm::vec3 v[3] = {tri.v0, tri.v1, tri.v2};

        for (int i = 0; i < 3; i++) {
            int next = (i + 1) % 3;

            // Edge vector in YZ plane
            float ey = v[next].y - v[i].y;
            float ez = v[next].z - v[i].z;

            // Normal (perpendicular to edge)
            edges[i].nx = -ez;
            edges[i].ny = ey;

            // Flip based on triangle normal
            if (tri.normal.x < 0) {
                edges[i].nx = -edges[i].nx;
                edges[i].ny = -edges[i].ny;
            }

            // Offset
            edges[i].d = -(edges[i].nx * v[i].y + edges[i].ny * v[i].z);
        }

        return edges;
    }

    // Get sub-grid voxel bit index
    int getSubGridBit(int lx, int ly, int lz) const noexcept
    {
        return lx + ly * 4 + lz * 16;
    }

    // Set bit in sub-grid
    void setSubGridBit(uint64_t& data, int lx, int ly, int lz) const noexcept
    {
        int bit = getSubGridBit(lx, ly, lz);
        data |= (1ULL << bit);
    }

    // Flip bit in sub-grid
    void flipSubGridBit(uint64_t& data, int lx, int ly, int lz) const noexcept
    {
        int bit = getSubGridBit(lx, ly, lz);
        data ^= (1ULL << bit);
    }

    // Get bit from sub-grid
    bool getSubGridBit(uint64_t data, int lx, int ly, int lz) const noexcept
    {
        int bit = getSubGridBit(lx, ly, lz);
        return (data >> bit) & 1;
    }

    // Find node by Morton code
    OctreeNode* findNode(int level, int morton)
    {
        for (auto& node : m_levelNodes[level]) {
            if (node.mortonCode == morton) {
                return &node;
            }
        }
        return nullptr;
    }

    // Step 3: Voxelize into octree
    void voxelizeIntoOctree()
    {
        float sgVoxelSize = m_voxelSize; // Sub-grid voxel size

        for (const auto& tri : m_triangles) {
            // Setup edge functions for YZ plane
            auto edges = setupEdgeFunctions(tri);

            // Compute bounding box in sub-grid coordinates
            glm::vec3 minBB(std::min({tri.v0.x, tri.v1.x, tri.v2.x}),
                std::min({tri.v0.y, tri.v1.y, tri.v2.y}),
                std::min({tri.v0.z, tri.v1.z, tri.v2.z}));
            glm::vec3 maxBB(std::max({tri.v0.x, tri.v1.x, tri.v2.x}),
                std::max({tri.v0.y, tri.v1.y, tri.v2.y}),
                std::max({tri.v0.z, tri.v1.z, tri.v2.z}));

            int minY = static_cast<int>(minBB.y / sgVoxelSize);
            int minZ = static_cast<int>(minBB.z / sgVoxelSize);
            int maxY = static_cast<int>(maxBB.y / sgVoxelSize);
            int maxZ = static_cast<int>(maxBB.z / sgVoxelSize);

            // Loop over YZ columns
            for (int gz = minZ; gz <= maxZ && gz < m_gridSize; gz++) {
                for (int gy = minY; gy <= maxY && gy < m_gridSize; gy++) {
                    // Center of voxel column
                    float cy = (gy + 0.5f) * sgVoxelSize;
                    float cz = (gz + 0.5f) * sgVoxelSize;

                    // Test if column center overlaps triangle in YZ
                    bool overlaps = true;
                    for (int i = 0; i < 3; i++) {
                        if (!edges[i].test(cy, cz)) {
                            overlaps = false;
                            break;
                        }
                    }

                    if (!overlaps) continue;

                    // Project center onto triangle plane along X axis
                    float t = (glm::dot(tri.v0, tri.normal) - tri.normal.y * cy - tri.normal.z * cz) / tri.normal.x;
                    float projX = t;

                    // Find first voxel to flip
                    int qBar = static_cast<int>(std::floor(projX / sgVoxelSize + 0.5f));

                    if (qBar >= m_gridSize) continue;
                    if (qBar < 0) qBar = 0;

                    // Convert to level-0 and local coordinates
                    int level0Y = gy / 4;
                    int level0Z = gz / 4;
                    int level0X = qBar / 4;

                    int localY = gy % 4;
                    int localZ = gz % 4;
                    int localX = qBar % 4;

                    // Find the level-0 node
                    int morton = encodeMorton(level0X, level0Y, level0Z);
                    OctreeNode* node = findNode(0, morton);

                    if (node) {
                        // Flip bits from localX to 3
                        for (int lx = localX; lx < 4; lx++) {
                            flipSubGridBit(node->voxelData, lx, localY, localZ);
                        }
                    }
                }
            }
        }
    }

    // Step 4: Propagate inside/outside hierarchically
    void propagateInsideOutside()
    {
        // Phase 1: Propagate along X within level-0
        for (auto& node : m_levelNodes[0]) {
            // Check if last X column has any set bits
            bool hasFlip = false;
            for (int ly = 0; ly < 4; ly++) {
                for (int lz = 0; lz < 4; lz++) {
                    if (getSubGridBit(node.voxelData, 3, ly, lz)) {
                        hasFlip = true;
                        break;
                    }
                }
                if (hasFlip) break;
            }

            // Propagate to X neighbor
            if (hasFlip && node.xPosNeighbor >= 0) {
                auto& neighbor = m_levelNodes[0][node.xPosNeighbor];
                // Flip all bits in neighbor
                neighbor.voxelData ^= 0xFFFFFFFFFFFFFFFFULL;
            }
        }

        // Phase 2: Propagate to coarser levels
        for (int level = 0; level < m_maxLevel; level++) {
            // Determine flip flags for next level
            if (level + 1 <= m_maxLevel) {
                for (auto& node : m_levelNodes[level]) {
                    // Check if this node is at end of X chain
                    if (node.xPosNeighbor < 0 || (node.mortonCode >> 3) != (m_levelNodes[level][node.xPosNeighbor].mortonCode >> 3)) {

                        // Check if boundary bits are set
                        bool boundarySet = false;
                        for (int ly = 0; ly < 4; ly++) {
                            for (int lz = 0; lz < 4; lz++) {
                                if (getSubGridBit(node.voxelData, 3, ly, lz)) {
                                    boundarySet = true;
                                    break;
                                }
                            }
                            if (boundarySet) break;
                        }

                        // Set parent's flip flag
                        if (boundarySet && node.parent >= 0) {
                            int parentIdx = node.parent;
                            if (parentIdx < static_cast<int>(m_levelNodes[level + 1].size())) {
                                m_levelNodes[level + 1][parentIdx].flipFlag = true;
                            }
                        }
                    }
                }

                // Propagate flip flags along X in next level
                for (auto& node : m_levelNodes[level + 1]) {
                    if (node.flipFlag && node.xPosNeighbor >= 0) {
                        auto& neighbor = m_levelNodes[level + 1][node.xPosNeighbor];
                        neighbor.flipFlag = !neighbor.flipFlag;
                        neighbor.insideFlag = !neighbor.insideFlag;
                    }
                }
            }
        }

        // Phase 3: Propagate down from top level
        for (int level = m_maxLevel; level >= 0; level--) {
            for (auto& node : m_levelNodes[level]) {
                if (node.insideFlag) {
                    if (level == 0) {
                        // Flip all sub-grid voxels
                        node.voxelData ^= 0xFFFFFFFFFFFFFFFFULL;
                    } else if (node.firstChild >= 0) {
                        // Flip children's inside flags
                        for (int i = 0; i < 8; i++) {
                            int childIdx = node.firstChild + i;
                            if (childIdx < static_cast<int>(m_levelNodes[level - 1].size())) {
                                m_levelNodes[level - 1][childIdx].insideFlag = !m_levelNodes[level - 1][childIdx].insideFlag;
                            }
                        }
                    }
                }
            }
        }
    }

    // Main voxelization function
    void voxelize()
    {
        std::vector<int> activeNodes = determineActiveNodes();
        constructOctree(activeNodes);
        voxelizeIntoOctree();
        propagateInsideOutside();
    }

    // Query if a point is inside
    bool isInside(const glm::vec3& point) const
    {
        // Convert to grid coordinates
        int gx = static_cast<int>(point.x / m_voxelSize);
        int gy = static_cast<int>(point.y / m_voxelSize);
        int gz = static_cast<int>(point.z / m_voxelSize);

        if (gx < 0 || gx >= m_gridSize || gy < 0 || gy >= m_gridSize || gz < 0 || gz >= m_gridSize) {
            return false;
        }

        // Convert to level-0 coordinates
        int level0X = gx / 4;
        int level0Y = gy / 4;
        int level0Z = gz / 4;
        int morton = encodeMorton(level0X, level0Y, level0Z);

        // Find the node
        for (const auto& node : m_levelNodes[0]) {
            if (node.mortonCode == morton) {
                int lx = gx % 4;
                int ly = gy % 4;
                int lz = gz % 4;
                return getSubGridBit(node.voxelData, lx, ly, lz);
            }
        }

        // If node doesn't exist, check parent's inside flag
        for (int level = 1; level <= m_maxLevel; level++) {
            int levelSize = m_gridSize / (4 << (level - 1));
            int levelX = gx / (4 << (level - 1));
            int levelY = gy / (4 << (level - 1));
            int levelZ = gz / (4 << (level - 1));

            if (levelX >= levelSize || levelY >= levelSize || levelZ >= levelSize) break;

            morton = encodeMorton(levelX, levelY, levelZ);

            for (const auto& node : m_levelNodes[level]) {
                if (node.mortonCode == morton) {
                    return node.insideFlag;
                }
            }
        }

        return false;
    }

    // Get memory usage
    size_t getMemoryUsage() const
    {
        size_t total = 0;
        for (const auto& level : m_levelNodes) {
            total += level.size() * sizeof(OctreeNode);
        }
        return total;
    }

    // Export voxelization statistics
    void printStatistics() const
    {
        std::cout << "Octree Statistics:\n";
        std::cout << "Grid size: " << m_gridSize << "^3\n";
        std::cout << "Number of levels: " << (m_maxLevel + 1) << "\n";

        for (int i = 0; i <= m_maxLevel; i++) {
            std::cout << "Level " << i << ": " << m_levelNodes[i].size() << " nodes\n";
        }

        size_t totalVoxels = m_gridSize * m_gridSize * m_gridSize;
        size_t storedVoxels = m_levelNodes[0].size() * 64; // Each level-0 node has 64 voxels
        float sparsity = 100.0f * storedVoxels / totalVoxels;

        std::cout << "Stored level-0 nodes: " << m_levelNodes[0].size() << "\n";
        std::cout << "Sparsity: " << sparsity << "%\n";
        std::cout << "Memory usage: " << (getMemoryUsage() / 1024.0f / 1024.0f) << " MB\n";
    }

    // Export to binary file
    void exportToBinary(const std::string& filename) const
    {
        std::ofstream file(filename, std::ios::binary);

        // Write header
        file.write(reinterpret_cast<const char*>(&m_gridSize), sizeof(m_gridSize));
        file.write(reinterpret_cast<const char*>(&m_maxLevel), sizeof(m_maxLevel));

        // Write each level
        for (int level = 0; level <= m_maxLevel; level++) {
            size_t count = m_levelNodes[level].size();
            file.write(reinterpret_cast<const char*>(&count), sizeof(count));

            for (const auto& node : m_levelNodes[level]) {
                file.write(reinterpret_cast<const char*>(&node.mortonCode), sizeof(node.mortonCode));
                file.write(reinterpret_cast<const char*>(&node.voxelData), sizeof(node.voxelData));
                file.write(reinterpret_cast<const char*>(&node.insideFlag), sizeof(node.insideFlag));
            }
        }

        file.close();
    }

    // Get coverage factor for visualization (0.0 = outside, 1.0 = fully inside)
    float getCoverageFactor(const glm::vec3& point) const
    {
        // For level-0 nodes, count set bits in sub-grid
        int gx = static_cast<int>(point.x / m_voxelSize);
        int gy = static_cast<int>(point.y / m_voxelSize);
        int gz = static_cast<int>(point.z / m_voxelSize);

        int level0X = gx / 4;
        int level0Y = gy / 4;
        int level0Z = gz / 4;
        int morton = encodeMorton(level0X, level0Y, level0Z);

        for (const auto& node : m_levelNodes[0]) {
            if (node.mortonCode == morton) {
                int count = 0;
                for (int i = 0; i < 64; i++) {
                    if ((node.voxelData >> i) & 1) count++;
                }
                return count / 64.0f;
            }
        }

        // Check higher levels
        for (int level = 1; level <= m_maxLevel; level++) {
            int levelScale = 4 << (level - 1);
            int levelX = gx / levelScale;
            int levelY = gy / levelScale;
            int levelZ = gz / levelScale;
            morton = encodeMorton(levelX, levelY, levelZ);

            for (const auto& node : m_levelNodes[level]) {
                if (node.mortonCode == morton) {
                    return node.insideFlag ? 1.0f : 0.0f;
                }
            }
        }

        return 0.0f;
    }
};