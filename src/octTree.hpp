#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <vector>

// Requires GLM (header-only)
#include <glm/glm.hpp>

/*
    octTree_fixed_auto.hpp
    ----------------------
    Header-only octree voxelizer with:
      1) Correct triangle–AABB SAT overlap.
      2) Clamped active-voxel bounds.
      3) Parent/child wiring + per-level morton→index maps.
      4) X-neighbor linking via level-aware math.
      5) Dominant-axis solid voxelization (no division-by-zero).
      6) Inside/outside parity propagation along whole +X chains.
      7) 64-bit packed key (z,y,x) to avoid aliasing.
      8) O(1) lookups.
      9) **Automatic band completion** so the interior gets filled without manual steps.
*/

// ----------------------------- Geometry types --------------------------------
struct Triangle
{
    glm::vec3 v0, v1, v2;
    glm::vec3 normal; // should be unit or any scale; only ratios used
    Triangle(glm::vec3 v0, glm::vec3 v1, glm::vec3 v2) : v0(v0), v1(v1), v2(v2), normal(glm::normalize(glm::cross(v1 - v0, v2 - v0)))
    {
    }
};

// --------------------------------- Octree ------------------------------------
struct OctreeNode
{
    uint32_t level = 0; // 0 == leaf (finest)
    uint64_t mortonPath = 0; // here: packed (z,y,x) key, 21 bits per component
    int parent = -1; // index in level+1 vector
    int firstChild = -1; // optional
    int xPosNeighbor = -1; // +X neighbor in same level
    int xNegNeighbor = -1; // -X neighbor in same level
    uint64_t voxelSubgrid = 0; // 4x4x4 = 64 bits
    bool insideFlag = false;
    bool boundaryFlag = false; // boundary crossing info for parity
};

class OctTree
{
public:
    uint32_t maxLevel = 3; // 0..maxLevel (0 is finest)
    float worldSize = 1.0f; // [0,worldSize]^3
    uint32_t level0Grid = 64; // voxels/axis at level 0 (multiple of 4)

    std::vector<std::vector<OctreeNode>> nodesPerLevel; // level 0 is finest
    std::vector<std::unordered_map<uint64_t, int>> idxOf; // per-level key -> index

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
        completeBandsLevel0(); // NEW: continuous +X chains automatically
        buildUpperLevels();
        linkXNeighborsAllLevels();
        solidVoxelize(tris);
        propagateInsideOutside();
    }

    inline float levelVoxelSize(uint32_t level) const
    {
        uint32_t cells = level0Grid >> level; // halves per level
        return worldSize / float(cells);
    }

    Aabb nodeBounds(uint32_t level, uint64_t mortonPath) const
    {
        // This overload isn't used for level 0 in this header;
        // provided for completeness if you switch back to path-by-level encoding.
        const float h = levelVoxelSize(level);
        uint32_t x = (uint32_t)(mortonPath & ((1u << 21) - 1));
        uint32_t y = (uint32_t)((mortonPath >> 21) & ((1u << 21) - 1));
        uint32_t z = (uint32_t)((mortonPath >> 42) & ((1u << 21) - 1));
        glm::vec3 mn(x * h, y * h, z * h);
        return {mn, mn + glm::vec3(h)};
    }

private:
    // ---------------- Triangle / AABB overlap (fixed) ----------------
    static inline bool triBoxOverlapSAT(const Triangle& tri, const glm::vec3& boxMin, float boxSize)
    {
        const glm::vec3 h(boxSize * 0.5f);
        const glm::vec3 c = boxMin + h;

        glm::vec3 p0 = tri.v0 - c;
        glm::vec3 p1 = tri.v1 - c;
        glm::vec3 p2 = tri.v2 - c;

        glm::vec3 e0 = p1 - p0;
        glm::vec3 e1 = p2 - p1;
        glm::vec3 e2 = p0 - p2;

        auto aabbAxisSeparates = [&](const glm::vec3& half) -> bool {
            float minx = std::min(p0.x, std::min(p1.x, p2.x));
            float maxx = std::max(p0.x, std::max(p1.x, p2.x));
            if (minx > half.x || maxx < -half.x) return true;
            float miny = std::min(p0.y, std::min(p1.y, p2.y));
            float maxy = std::max(p0.y, std::max(p1.y, p2.y));
            if (miny > half.y || maxy < -half.y) return true;
            float minz = std::min(p0.z, std::min(p1.z, p2.z));
            float maxz = std::max(p0.z, std::max(p1.z, p2.z));
            if (minz > half.z || maxz < -half.z) return true;
            return false;
        };
        if (aabbAxisSeparates(h)) return false;

        auto axisSeparates = [&](const glm::vec3& L, float R) -> bool {
            float p0d = glm::dot(p0, L);
            float p1d = glm::dot(p1, L);
            float p2d = glm::dot(p2, L);
            float mn = std::min(p0d, std::min(p1d, p2d));
            float mx = std::max(p0d, std::max(p1d, p2d));
            return (mn > R) || (mx < -R);
        };

        auto testEdgeAxes = [&](const glm::vec3& e) -> bool {
            glm::vec3 Lx = {0.0f, -e.z, e.y};
            float Rx = h.y * std::fabs(Lx.y) + h.z * std::fabs(Lx.z);
            if (axisSeparates(Lx, Rx)) return true;
            glm::vec3 Ly = {e.z, 0.0f, -e.x};
            float Ry = h.x * std::fabs(Ly.x) + h.z * std::fabs(Ly.z);
            if (axisSeparates(Ly, Ry)) return true;
            glm::vec3 Lz = {-e.y, e.x, 0.0f};
            float Rz = h.x * std::fabs(Lz.x) + h.y * std::fabs(Lz.y);
            if (axisSeparates(Lz, Rz)) return true;
            return false;
        };
        if (testEdgeAxes(e0) || testEdgeAxes(e1) || testEdgeAxes(e2)) return false;

        glm::vec3 n = glm::cross(e0, e1);
        glm::vec3 an = glm::abs(n);
        float r = h.x * an.x + h.y * an.y + h.z * an.z;
        float s = glm::dot(n, p0);
        if (std::fabs(s) > r) return false;
        return true;
    }

    // ----------------- Leaves: active voxels from triangles ------------------
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

    // ----------------- NEW: Complete (y,z) bands automatically ----------------
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
            const uint64_t yz = kv.first;
            const auto& r = kv.second;
            if (!r.inited) continue;
            uint32_t y = (uint32_t)(yz & ((1u << 21) - 1));
            uint32_t z = (uint32_t)(yz >> 21);
            for (uint32_t x = r.minx; x <= r.maxx; ++x) {
                uint64_t key = ((uint64_t)z << 42) | ((uint64_t)y << 21) | (uint64_t)x;
                if (map.find(key) != map.end()) continue;
                OctreeNode empty;
                empty.level = 0;
                empty.mortonPath = key;
                int idx = (int)L0.size();
                L0.push_back(empty);
                map[key] = idx;
            }
        }
    }

    // ----------------- Build parents (wire parent/children) ------------------
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

                auto pit = parentIndex.find(pkey);
                int pidx;
                if (pit == parentIndex.end()) {
                    OctreeNode p;
                    p.level = L;
                    p.mortonPath = pkey;
                    p.firstChild = -1;
                    p.parent = -1;
                    pidx = (int)nxt.size();
                    nxt.push_back(p);
                    parentIndex[pkey] = pidx;
                    mapNxt[p.mortonPath] = pidx;
                } else {
                    pidx = pit->second;
                }
                cur[ci].parent = pidx;
            }

            auto& mapCur = idxOf[L - 1];
            mapCur.clear();
            for (int i = 0; i < (int)cur.size(); ++i)
                mapCur[cur[i].mortonPath] = i;
        }
    }

    // -------------------- Neighbor linking (+X only) -------------------------
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
                    uint64_t nkey = ((uint64_t)z << 42) | ((uint64_t)y << 21) | (uint64_t)xn;
                    auto it = map.find(nkey);
                    if (it != map.end()) {
                        vec[i].xPosNeighbor = it->second;
                        nodesPerLevel[L][it->second].xNegNeighbor = i;
                    }
                }
            }
        }
    }

    // -------------------- Solid voxelization (dominant axis) -----------------
    void solidVoxelize(const std::vector<Triangle>& tris)
    {
        if (nodesPerLevel[0].empty()) return;
        const float h = levelVoxelSize(0);

        for (const Triangle& t : tris) {
            glm::vec3 n = t.normal;
            glm::vec3 an = glm::abs(n);
            int axis = 0;
            if (an.y > an.x && an.y >= an.z)
                axis = 1;
            else if (an.z > an.x && an.z >= an.y)
                axis = 2;

            float d = -glm::dot(n, t.v0);

            for (auto& leaf : nodesPerLevel[0]) {
                Aabb bb = nodeBoundsFromPackedKey(leaf.mortonPath, h);
                if (!triBoxOverlapSAT(t, bb.minimum, h)) continue;

                float ofs = h * 0.25f;
                for (int lu = 0; lu < 4; ++lu) {
                    for (int lv = 0; lv < 4; ++lv) {
                        if (axis == 0) {
                            float y = bb.minimum.y + (lv + 0.5f) * ofs;
                            float z = bb.minimum.z + (lu + 0.5f) * ofs;
                            if (std::abs(n.x) < 1e-12f) continue;
                            float x = -(n.y * y + n.z * z + d) / n.x;
                            if (x >= bb.minimum.x && x < bb.maximum.x) {
                                int lx = (int)std::floor((x - bb.minimum.x) / ofs);
                                lx = std::clamp(lx, 0, 3);
                                setSubgridBit(leaf.voxelSubgrid, lx, lv, lu, true);
                                if (lx == 3) leaf.boundaryFlag = true;
                            }
                        } else if (axis == 1) {
                            float x = bb.minimum.x + (lv + 0.5f) * ofs;
                            float z = bb.minimum.z + (lu + 0.5f) * ofs;
                            if (std::abs(n.y) < 1e-12f) continue;
                            float y = -(n.x * x + n.z * z + d) / n.y;
                            if (y >= bb.minimum.y && y < bb.maximum.y) {
                                int ly = (int)std::floor((y - bb.minimum.y) / ofs);
                                ly = std::clamp(ly, 0, 3);
                                setSubgridBit(leaf.voxelSubgrid, lv, ly, lu, true);
                                if (lv == 3) leaf.boundaryFlag = true;
                            }
                        } else {
                            float x = bb.minimum.x + (lv + 0.5f) * ofs;
                            float y = bb.minimum.y + (lu + 0.5f) * ofs;
                            if (std::abs(n.z) < 1e-12f) continue;
                            float z = -(n.x * x + n.y * y + d) / n.z;
                            if (z >= bb.minimum.z && z < bb.maximum.z) {
                                int lz = (int)std::floor((z - bb.minimum.z) / ofs);
                                lz = std::clamp(lz, 0, 3);
                                setSubgridBit(leaf.voxelSubgrid, lv, lu, lz, true);
                                if (lv == 3) leaf.boundaryFlag = true;
                            }
                        }
                    }
                }
            }
        }
    }

    // -------------- Inside/outside parity propagation along +X chains --------
    void propagateInsideOutside()
    {
        if (nodesPerLevel.empty() || nodesPerLevel[0].empty()) return;

        { // Level 0
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
                if (c.parent >= 0) {
                    nodesPerLevel[L][c.parent].boundaryFlag = nodesPerLevel[L][c.parent].boundaryFlag || c.boundaryFlag;
                }

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
                if (c.parent >= 0) {
                    if (nodesPerLevel[L][c.parent].insideFlag) c.voxelSubgrid = ~c.voxelSubgrid;
                }
        }
    }

    // ------------------------ Helpers ----------------------------------------
    static inline void setSubgridBit(uint64_t& grid, int lx, int ly, int lz, bool on)
    {
        int bit = ((lz << 4) | (ly << 2) | lx);
        if (on)
            grid |= (1ull << bit);
        else
            grid &= ~(1ull << bit);
    }

    Aabb nodeBoundsFromPackedKey(uint64_t key, float h) const
    {
        uint32_t x = (uint32_t)(key & ((1u << 21) - 1));
        uint32_t y = (uint32_t)((key >> 21) & ((1u << 21) - 1));
        uint32_t z = (uint32_t)((key >> 42) & ((1u << 21) - 1));
        glm::vec3 mn(x * h, y * h, z * h);
        return {mn, mn + glm::vec3(h)};
    }

    
};
// Hilfsfunktion: verschiebt + skaliert Triangles in [0, worldSize]^3
inline std::vector<Triangle> fitToCube(const std::vector<Triangle>& in, float worldSize, glm::vec3* outMin = nullptr, glm::vec3* outScale = nullptr)
{
    // 1) BBox
    glm::vec3 mn(std::numeric_limits<float>::infinity());
    glm::vec3 mx(-std::numeric_limits<float>::infinity());
    auto upd = [&](const glm::vec3& p) { mn = glm::min(mn,p); mx = glm::max(mx,p); };
    for (auto& t : in) {
        upd(t.v0);
        upd(t.v1);
        upd(t.v2);
    }

    glm::vec3 ext = mx - mn;
    float maxDim = std::max(ext.x, std::max(ext.y, ext.z));
    if (maxDim <= 0.0f) return in;

    // 2) uniform scale + kleiner Puffer, damit nichts genau auf die Kante fällt
    float s = (worldSize * 0.98f) / maxDim;
    glm::vec3 off = -mn; // schiebt min -> 0

    // 3) anwenden
    std::vector<Triangle> out;
    out.reserve(in.size());
    for (auto& t : in) {
        Triangle o{(t.v0 + off) * s, (t.v1 + off) * s, (t.v2 + off) * s};
        out.push_back(o);
    }
    if (outMin) *outMin = mn;
    if (outScale) *outScale = glm::vec3(s);
    return out;
}
// -------- Convenience: iterate & collect set finest voxels (like old API) ----
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

    auto getBit = [](uint64_t grid, int lx, int ly, int lz) -> bool {
        int bit = (lz << 4) | (ly << 2) | lx;
        return (grid >> bit) & 1ull;
    };

    const auto& L0 = tree.nodesPerLevel[0];
    for (size_t nodeIdx = 0; nodeIdx < L0.size(); ++nodeIdx) {
        const auto& node = L0[nodeIdx];
        uint32_t x = (uint32_t)(node.mortonPath & ((1u << 21) - 1));
        uint32_t y = (uint32_t)((node.mortonPath >> 21) & ((1u << 21) - 1));
        uint32_t z = (uint32_t)((node.mortonPath >> 42) & ((1u << 21) - 1));
        for (int lz = 0; lz < 4; ++lz)
            for (int ly = 0; ly < 4; ++ly)
                for (int lx = 0; lx < 4; ++lx) {
                    if (!getBit(node.voxelSubgrid, lx, ly, lz)) continue;
                    VoxelInfo v;
                    v.x = int(x) * 4 + lx;
                    v.y = int(y) * 4 + ly;
                    v.z = int(z) * 4 + lz;
                    v.worldPos = glm::vec3((v.x + 0.5f) * finest, (v.y + 0.5f) * finest, (v.z + 0.5f) * finest);
                    v.nodeIndex = (int)nodeIdx;
                    cb(v);
                }
    }
}

inline std::vector<Aabb> getAllSetVoxels(const OctTree& tree)
{
    std::vector<Aabb> out;
    const float finest = tree.worldSize / float(tree.level0Grid * 4);
    const glm::vec3 half(finest * 0.5f);
    iterateSetVoxels(tree, [&](const VoxelInfo& v) {
        out.push_back(Aabb{v.worldPos - half, v.worldPos + half});
    });
    return out;
}
