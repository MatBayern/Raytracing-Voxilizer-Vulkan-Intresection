#pragma once

#include "obj_loader.h"
#include "shaders/host_device.h"
#include <glm/glm.hpp>

// STD
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

struct Triangle
{
    glm::vec3 v0, v1, v2;
    glm::vec3 normal;
    Triangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2) : v0(v0), v1(v1), v2(v2), normal(glm::normalize(glm::cross(v1 - v0, v2 - v0)))
    {
    }
};

struct OctreeNode
{
    uint32_t level = 0;
    uint64_t mortonPath = 0; // packed (z,y,x) grid key (21 bits each)
    int parent = -1;
    int firstChild = -1;
    int xPosNeighbor = -1;
    int xNegNeighbor = -1;
    uint64_t voxelSubgrid = 0; // 4x4x4 bits
    bool insideFlag = false;
    bool boundaryFlag = false;
};

class OctTree
{
public:
    uint32_t maxLevel = 3;
    float worldSize = 1.0f;
    uint32_t level0Grid = 64;

    std::vector<std::vector<OctreeNode>> nodesPerLevel;
    std::vector<std::unordered_map<uint64_t, int>> idxOf;

    OctTree(uint32_t maxLevel_, uint32_t level0Grid_, float worldSize_)
        : maxLevel(maxLevel_), worldSize(worldSize_), level0Grid(level0Grid_)
    {
        if (level0Grid % 4 != 0) level0Grid += (4 - (level0Grid % 4));
        nodesPerLevel.resize(maxLevel + 1);
        idxOf.resize(maxLevel + 1);
    }

    void build(const std::vector<Triangle>& tris)
    {
        generateActiveLeaves(tris);
        completeBandsLevel0();
        buildUpperLevels();
        linkXNeighborsAllLevels();
        solidVoxelize(tris);
        propagateInsideOutside();
    }

    inline float levelVoxelSize(uint32_t level) const
    {
        uint32_t cells = level0Grid >> level;
        return worldSize / float(cells);
    }

private:
    static inline bool triBoxOverlapSAT(const Triangle& tri, const glm::vec3& boxMin, float boxSize)
    {
        const glm::vec3 h(boxSize * 0.5f);
        const glm::vec3 c = boxMin + h;
        glm::vec3 p0 = tri.v0 - c, p1 = tri.v1 - c, p2 = tri.v2 - c;
        glm::vec3 e0 = p1 - p0, e1 = p2 - p1, e2 = p0 - p2;

        auto aabbSep = [&](const glm::vec3& H) -> bool {
            auto mn = [](float a, float b, float c) { return std::min(a, std::min(b, c)); };
            auto mx = [](float a, float b, float c) { return std::max(a, std::max(b, c)); };
            if (mn(p0.x, p1.x, p2.x) > H.x || mx(p0.x, p1.x, p2.x) < -H.x) return true;
            if (mn(p0.y, p1.y, p2.y) > H.y || mx(p0.y, p1.y, p2.y) < -H.y) return true;
            if (mn(p0.z, p1.z, p2.z) > H.z || mx(p0.z, p1.z, p2.z) < -H.z) return true;
            return false;
        };
        if (aabbSep(h)) return false;

        auto sep = [&](const glm::vec3& L, float R) -> bool {
            float a = glm::dot(p0, L), b = glm::dot(p1, L), c = glm::dot(p2, L);
            float mn = std::min(a, std::min(b, c)), mx = std::max(a, std::max(b, c));
            return (mn > R) || (mx < -R);
        };
        auto edgeAxes = [&](const glm::vec3& e) -> bool {
            glm::vec3 Lx(0, -e.z, e.y);
            if (sep(Lx, h.y * std::fabs(Lx.y) + h.z * std::fabs(Lx.z))) return true;
            glm::vec3 Ly(e.z, 0, -e.x);
            if (sep(Ly, h.x * std::fabs(Ly.x) + h.z * std::fabs(Ly.z))) return true;
            glm::vec3 Lz(-e.y, e.x, 0);
            if (sep(Lz, h.x * std::fabs(Lz.x) + h.y * std::fabs(Lz.y))) return true;
            return false;
        };
        if (edgeAxes(e0) || edgeAxes(e1) || edgeAxes(e2)) return false;

        glm::vec3 n = glm::cross(e0, e1);
        glm::vec3 an = glm::abs(n);
        float r = h.x * an.x + h.y * an.y + h.z * an.z;
        float s = glm::dot(n, p0);
        return std::fabs(s) <= r;
    }

    void generateActiveLeaves(const std::vector<Triangle>& tris)
    {
        nodesPerLevel.assign(maxLevel + 1, {});
        idxOf.assign(maxLevel + 1, {});

        const float h0 = levelVoxelSize(0);
        const uint32_t N0 = level0Grid;

        std::unordered_map<uint64_t, int> created;

        auto clampi = [](int v, int lo, int hi) -> int { return v < lo ? lo : (v > hi ? hi : v); };

        for (const Triangle& t : tris) {
            glm::vec3 tmin(std::min(t.v0.x, std::min(t.v1.x, t.v2.x)),
                std::min(t.v0.y, std::min(t.v1.y, t.v2.y)),
                std::min(t.v0.z, std::min(t.v1.z, t.v2.z)));
            glm::vec3 tmax(std::max(t.v0.x, std::max(t.v1.x, t.v2.x)),
                std::max(t.v0.y, std::max(t.v1.y, t.v2.y)),
                std::max(t.v0.z, std::max(t.v1.z, t.v2.z)));

            int minX = clampi((int)std::floor(tmin.x / h0), 0, (int)N0 - 1);
            int minY = clampi((int)std::floor(tmin.y / h0), 0, (int)N0 - 1);
            int minZ = clampi((int)std::floor(tmin.z / h0), 0, (int)N0 - 1);
            int maxX = clampi((int)std::floor(tmax.x / h0), 0, (int)N0 - 1);
            int maxY = clampi((int)std::floor(tmax.y / h0), 0, (int)N0 - 1);
            int maxZ = clampi((int)std::floor(tmax.z / h0), 0, (int)N0 - 1);

            for (int z = minZ; z <= maxZ; ++z)
                for (int y = minY; y <= maxY; ++y)
                    for (int x = minX; x <= maxX; ++x) {
                        glm::vec3 bmin(x * h0, y * h0, z * h0);
                        if (!triBoxOverlapSAT(t, bmin, h0)) continue;
                        uint64_t key = ((uint64_t)z << 42) | ((uint64_t)y << 21) | (uint64_t)x;
                        if (!created.count(key)) {
                            OctreeNode node;
                            node.level = 0;
                            node.mortonPath = key;
                            int idx = (int)nodesPerLevel[0].size();
                            nodesPerLevel[0].push_back(node);
                            created[key] = idx;
                            idxOf[0][key] = idx;
                        }
                    }
        }
    }

    void completeBandsLevel0()
    {
        if (nodesPerLevel.empty() || nodesPerLevel[0].empty()) return;
        struct Range
        {
            uint32_t minx, maxx;
            bool inited = false;
        };
        std::unordered_map<uint64_t, Range> bands;
        bands.reserve(nodesPerLevel[0].size() * 2);

        for (auto& leaf : nodesPerLevel[0]) {
            uint32_t x = (uint32_t)(leaf.mortonPath & ((1u << 21) - 1));
            uint32_t y = (uint32_t)((leaf.mortonPath >> 21) & ((1u << 21) - 1));
            uint32_t z = (uint32_t)((leaf.mortonPath >> 42) & ((1u << 21) - 1));
            uint64_t yz = ((uint64_t)z << 21) | (uint64_t)y;
            auto& r = bands[yz];
            if (!r.inited) {
                r.minx = r.maxx = x;
                r.inited = true;
            } else {
                r.minx = std::min(r.minx, x);
                r.maxx = std::max(r.maxx, x);
            }
        }
        auto& L0 = nodesPerLevel[0];
        auto& map = idxOf[0];
        for (auto& kv : bands) {
            uint64_t yz = kv.first;
            auto r = kv.second;
            if (!r.inited) continue;
            uint32_t y = (uint32_t)(yz & ((1u << 21) - 1));
            uint32_t z = (uint32_t)(yz >> 21);
            for (uint32_t x = r.minx; x <= r.maxx; ++x) {
                uint64_t key = ((uint64_t)z << 42) | ((uint64_t)y << 21) | x;
                if (map.find(key) != map.end()) continue;
                OctreeNode e;
                e.level = 0;
                e.mortonPath = key;
                int idx = (int)L0.size();
                L0.push_back(e);
                map[key] = idx;
            }
        }
    }

    void buildUpperLevels()
    {
        for (uint32_t L = 1; L <= maxLevel; ++L) {
            auto& cur = nodesPerLevel[L - 1];
            auto& nxt = nodesPerLevel[L];
            auto& mapNxt = idxOf[L];
            std::unordered_map<uint64_t, int> parentIndex;
            parentIndex.reserve(cur.size());
            for (int ci = 0; ci < (int)cur.size(); ++ci) {
                auto key = cur[ci].mortonPath;
                uint32_t x = (uint32_t)(key & ((1u << 21) - 1));
                uint32_t y = (uint32_t)((key >> 21) & ((1u << 21) - 1));
                uint32_t z = (uint32_t)((key >> 42) & ((1u << 21) - 1));
                uint64_t pkey = ((uint64_t)(z >> 1) << 42) | ((uint64_t)(y >> 1) << 21) | (uint64_t)(x >> 1);
                int pidx;
                auto pit = parentIndex.find(pkey);
                if (pit == parentIndex.end()) {
                    OctreeNode p;
                    p.level = L;
                    p.mortonPath = pkey;
                    pidx = (int)nxt.size();
                    nxt.push_back(p);
                    parentIndex[pkey] = pidx;
                    mapNxt[pkey] = pidx;
                } else
                    pidx = pit->second;
                cur[ci].parent = pidx;
            }
            auto& mapCur = idxOf[L - 1];
            mapCur.clear();
            for (int i = 0; i < (int)cur.size(); ++i)
                mapCur[cur[i].mortonPath] = i;
        }
    }

    void linkXNeighborsAllLevels()
    {
        for (uint32_t L = 0; L <= maxLevel; ++L) {
            auto& vec = nodesPerLevel[L];
            auto& map = idxOf[L];
            for (int i = 0; i < (int)vec.size(); ++i) {
                auto key = vec[i].mortonPath;
                uint32_t x = (uint32_t)(key & ((1u << 21) - 1));
                uint32_t y = (uint32_t)((key >> 21) & ((1u << 21) - 1));
                uint32_t z = (uint32_t)((key >> 42) & ((1u << 21) - 1));
                uint32_t step = (1u << L);
                uint32_t xn = x ^ step;
                if ((x & ~(step * 2u - 1u)) == (xn & ~(step * 2u - 1u))) {
                    uint64_t nkey = ((uint64_t)z << 42) | ((uint64_t)y << 21) | xn;
                    auto it = map.find(nkey);
                    if (it != map.end()) {
                        vec[i].xPosNeighbor = it->second;
                        nodesPerLevel[L][it->second].xNegNeighbor = i;
                    }
                }
            }
        }
    }

    void solidVoxelize(const std::vector<Triangle>& tris)
    {
        if (nodesPerLevel[0].empty()) return;
        const float h = levelVoxelSize(0);
        for (const Triangle& t : tris) {
            glm::vec3 n = t.normal, an = glm::abs(n);
            int axis = 0;
            if (an.y > an.x && an.y >= an.z)
                axis = 1;
            else if (an.z > an.x && an.z >= an.y)
                axis = 2;
            float d = -glm::dot(n, t.v0);
            for (auto& leaf : nodesPerLevel[0]) {
                uint32_t x = (uint32_t)(leaf.mortonPath & ((1u << 21) - 1));
                uint32_t y = (uint32_t)((leaf.mortonPath >> 21) & ((1u << 21) - 1));
                uint32_t z = (uint32_t)((leaf.mortonPath >> 42) & ((1u << 21) - 1));
                glm::vec3 bbmin(x * h, y * h, z * h), bbmax = bbmin + glm::vec3(h);
                if (!triBoxOverlapSAT(t, bbmin, h)) continue;
                float ofs = h * 0.25f;
                for (int lu = 0; lu < 4; ++lu)
                    for (int lv = 0; lv < 4; ++lv) {
                        if (axis == 0) {
                            if (std::abs(n.x) < 1e-12f) continue;
                            float y0 = bbmin.y + (lv + 0.5f) * ofs, z0 = bbmin.z + (lu + 0.5f) * ofs;
                            float x0 = -(n.y * y0 + n.z * z0 + d) / n.x;
                            if (x0 >= bbmin.x && x0 < bbmax.x) {
                                int lx = (int)std::floor((x0 - bbmin.x) / ofs);
                                lx = std::clamp(lx, 0, 3);
                                setSubgridBit(leaf.voxelSubgrid, lx, lv, lu, true);
                                if (lx == 3) leaf.boundaryFlag = true;
                            }
                        } else if (axis == 1) {
                            if (std::abs(n.y) < 1e-12f) continue;
                            float x0 = bbmin.x + (lv + 0.5f) * ofs, z0 = bbmin.z + (lu + 0.5f) * ofs;
                            float y0 = -(n.x * x0 + n.z * z0 + d) / n.y;
                            if (y0 >= bbmin.y && y0 < bbmax.y) {
                                int ly = (int)std::floor((y0 - bbmin.y) / ofs);
                                ly = std::clamp(ly, 0, 3);
                                setSubgridBit(leaf.voxelSubgrid, lv, ly, lu, true);
                                if (lv == 3) leaf.boundaryFlag = true;
                            }
                        } else {
                            if (std::abs(n.z) < 1e-12f) continue;
                            float x0 = bbmin.x + (lv + 0.5f) * ofs, y0 = bbmin.y + (lu + 0.5f) * ofs;
                            float z0 = -(n.x * x0 + n.y * y0 + d) / n.z;
                            if (z0 >= bbmin.z && z0 < bbmax.z) {
                                int lz = (int)std::floor((z0 - bbmin.z) / ofs);
                                lz = std::clamp(lz, 0, 3);
                                setSubgridBit(leaf.voxelSubgrid, lv, lu, lz, true);
                                if (lv == 3) leaf.boundaryFlag = true;
                            }
                        }
                    }
            }
        }
    }

    void propagateInsideOutside()
    {
        if (nodesPerLevel.empty() || nodesPerLevel[0].empty()) return;
        {
            auto& L0 = nodesPerLevel[0];
            std::vector<int> heads;
            for (int i = 0; i < (int)L0.size(); ++i)
                if (L0[i].xNegNeighbor < 0) heads.push_back(i);
            for (int h : heads) {
                int cur = h;
                bool flip = false;
                while (cur >= 0) {
                    if (L0[cur].boundaryFlag) flip = !flip;
                    if (flip) L0[cur].voxelSubgrid = ~L0[cur].voxelSubgrid;
                    cur = L0[cur].xPosNeighbor;
                }
            }
        }
        for (uint32_t L = 1; L <= maxLevel; ++L) {
            auto& V = nodesPerLevel[L];
            for (auto& p : V)
                p.boundaryFlag = false;
            for (auto& c : nodesPerLevel[L - 1])
                if (c.parent >= 0) nodesPerLevel[L][c.parent].boundaryFlag |= c.boundaryFlag;
            std::vector<int> heads;
            for (int i = 0; i < (int)V.size(); ++i)
                if (V[i].xNegNeighbor < 0) heads.push_back(i);
            for (int h : heads) {
                int cur = h;
                bool inside = false;
                while (cur >= 0) {
                    if (V[cur].boundaryFlag) inside = !inside;
                    V[cur].insideFlag = inside;
                    cur = V[cur].xPosNeighbor;
                }
            }
            for (auto& c : nodesPerLevel[L - 1])
                if (c.parent >= 0)
                    if (nodesPerLevel[L][c.parent].insideFlag) c.voxelSubgrid = ~c.voxelSubgrid;
        }
    }

    static inline void setSubgridBit(uint64_t& grid, int lx, int ly, int lz, bool on)
    {
        int bit = ((lz << 4) | (ly << 2) | lx);
        if (on)
            grid |= (1ull << bit);
        else
            grid &= ~(1ull << bit);
    }
};

// Helpers to iterate/collect finest voxels
struct VoxelInfo
{
    int x, y, z;
    glm::vec3 worldPos;
    int level = 0;
    bool isSet = true;
    int nodeIndex = -1;
};
using VoxelCallback = std::function<void(const VoxelInfo&)>;

inline void iterateSetVoxels(const OctTree& tree, VoxelCallback cb)
{
    if (tree.nodesPerLevel.empty()) return;
    const float finest = tree.worldSize / float(tree.level0Grid * 4);
    auto getBit = [](uint64_t g, int lx, int ly, int lz) { int bit=(lz<<4)|(ly<<2)|lx; return (g>>bit)&1ull; };
    const auto& L0 = tree.nodesPerLevel[0];
    for (size_t i = 0; i < L0.size(); ++i) {
        const auto& n = L0[i];
        uint32_t x = (uint32_t)(n.mortonPath & ((1u << 21) - 1));
        uint32_t y = (uint32_t)((n.mortonPath >> 21) & ((1u << 21) - 1));
        uint32_t z = (uint32_t)((n.mortonPath >> 42) & ((1u << 21) - 1));
        for (int lz = 0; lz < 4; ++lz)
            for (int ly = 0; ly < 4; ++ly)
                for (int lx = 0; lx < 4; ++lx) {
                    if (!getBit(n.voxelSubgrid, lx, ly, lz)) continue;
                    VoxelInfo v;
                    v.x = int(x) * 4 + lx;
                    v.y = int(y) * 4 + ly;
                    v.z = int(z) * 4 + lz;
                    v.worldPos = glm::vec3((v.x + 0.5f) * finest, (v.y + 0.5f) * finest, (v.z + 0.5f) * finest);
                    v.nodeIndex = (int)i;
                    cb(v);
                }
    }
}

inline std::vector<Aabb> getAllSetVoxels(const OctTree& tree)
{
    std::vector<Aabb> out;
    const float s = tree.worldSize / float(tree.level0Grid * 4);
    const glm::vec3 h(s * 0.5f);
    iterateSetVoxels(tree, [&](const VoxelInfo& v) { out.push_back(Aabb{v.worldPos - h, v.worldPos + h}); });
    return out;
}

// =================== New: build using ONLY voxel size ===================
namespace octree_voxelsize_detail {
inline uint32_t nextPow2(uint32_t v)
{
    if (v <= 1) return 1;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}
inline uint32_t roundUpMultiple(uint32_t v, uint32_t m)
{
    return ((v + m - 1) / m) * m;
}
}

inline OctTree BuildOctreeWithVoxelSize(std::vector<Triangle> tris, float voxelSize, uint32_t maxLevel = 4)
{
    // Compute bbox
    glm::vec3 mn(std::numeric_limits<float>::infinity());
    glm::vec3 mx(-std::numeric_limits<float>::infinity());
    auto upd = [&](const glm::vec3& p) { mn=glm::min(mn,p); mx=glm::max(mx,p); };
    for (auto& t : tris) {
        upd(t.v0);
        upd(t.v1);
        upd(t.v2);
    }

    // Translate so min -> 0 (no scaling)
    glm::vec3 off = -mn;
    for (auto& t : tris) {
        t.v0 += off;
        t.v1 += off;
        t.v2 += off;
        t.normal = glm::normalize(glm::cross(t.v1 - t.v0, t.v2 - t.v0));
    }

    glm::vec3 ext = mx - mn;
    float maxDim = std::max(ext.x, std::max(ext.y, ext.z));
    if (maxDim <= 0.f) return OctTree(maxLevel, 4, 4 * voxelSize);

    uint32_t grid = (uint32_t)std::ceil(maxDim / voxelSize);
    grid = octree_voxelsize_detail::roundUpMultiple(grid, 4);
    grid = std::max(4u, octree_voxelsize_detail::nextPow2(grid));
    float worldSize = float(grid) * voxelSize;

    OctTree tree(maxLevel, grid, worldSize);
    tree.build(tris);
    return tree;
}
