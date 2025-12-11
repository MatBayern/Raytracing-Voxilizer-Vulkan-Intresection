#pragma once

#include "shaders/host_device.h"
#include "tiny_obj_loader.h"
#include <glm/glm.hpp>

// STD
#include <algorithm>
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
namespace {
// LUT taken from https://www.forceflow.be/2013/10/07/morton-encodingdecoding-through-bit-interleaving-implementations
static constexpr uint32_t morton256_x[256] = {
    0x00000000,
    0x00000001, 0x00000008, 0x00000009, 0x00000040, 0x00000041, 0x00000048, 0x00000049, 0x00000200,
    0x00000201, 0x00000208, 0x00000209, 0x00000240, 0x00000241, 0x00000248, 0x00000249, 0x00001000,
    0x00001001, 0x00001008, 0x00001009, 0x00001040, 0x00001041, 0x00001048, 0x00001049, 0x00001200,
    0x00001201, 0x00001208, 0x00001209, 0x00001240, 0x00001241, 0x00001248, 0x00001249, 0x00008000,
    0x00008001, 0x00008008, 0x00008009, 0x00008040, 0x00008041, 0x00008048, 0x00008049, 0x00008200,
    0x00008201, 0x00008208, 0x00008209, 0x00008240, 0x00008241, 0x00008248, 0x00008249, 0x00009000,
    0x00009001, 0x00009008, 0x00009009, 0x00009040, 0x00009041, 0x00009048, 0x00009049, 0x00009200,
    0x00009201, 0x00009208, 0x00009209, 0x00009240, 0x00009241, 0x00009248, 0x00009249, 0x00040000,
    0x00040001, 0x00040008, 0x00040009, 0x00040040, 0x00040041, 0x00040048, 0x00040049, 0x00040200,
    0x00040201, 0x00040208, 0x00040209, 0x00040240, 0x00040241, 0x00040248, 0x00040249, 0x00041000,
    0x00041001, 0x00041008, 0x00041009, 0x00041040, 0x00041041, 0x00041048, 0x00041049, 0x00041200,
    0x00041201, 0x00041208, 0x00041209, 0x00041240, 0x00041241, 0x00041248, 0x00041249, 0x00048000,
    0x00048001, 0x00048008, 0x00048009, 0x00048040, 0x00048041, 0x00048048, 0x00048049, 0x00048200,
    0x00048201, 0x00048208, 0x00048209, 0x00048240, 0x00048241, 0x00048248, 0x00048249, 0x00049000,
    0x00049001, 0x00049008, 0x00049009, 0x00049040, 0x00049041, 0x00049048, 0x00049049, 0x00049200,
    0x00049201, 0x00049208, 0x00049209, 0x00049240, 0x00049241, 0x00049248, 0x00049249, 0x00200000,
    0x00200001, 0x00200008, 0x00200009, 0x00200040, 0x00200041, 0x00200048, 0x00200049, 0x00200200,
    0x00200201, 0x00200208, 0x00200209, 0x00200240, 0x00200241, 0x00200248, 0x00200249, 0x00201000,
    0x00201001, 0x00201008, 0x00201009, 0x00201040, 0x00201041, 0x00201048, 0x00201049, 0x00201200,
    0x00201201, 0x00201208, 0x00201209, 0x00201240, 0x00201241, 0x00201248, 0x00201249, 0x00208000,
    0x00208001, 0x00208008, 0x00208009, 0x00208040, 0x00208041, 0x00208048, 0x00208049, 0x00208200,
    0x00208201, 0x00208208, 0x00208209, 0x00208240, 0x00208241, 0x00208248, 0x00208249, 0x00209000,
    0x00209001, 0x00209008, 0x00209009, 0x00209040, 0x00209041, 0x00209048, 0x00209049, 0x00209200,
    0x00209201, 0x00209208, 0x00209209, 0x00209240, 0x00209241, 0x00209248, 0x00209249, 0x00240000,
    0x00240001, 0x00240008, 0x00240009, 0x00240040, 0x00240041, 0x00240048, 0x00240049, 0x00240200,
    0x00240201, 0x00240208, 0x00240209, 0x00240240, 0x00240241, 0x00240248, 0x00240249, 0x00241000,
    0x00241001, 0x00241008, 0x00241009, 0x00241040, 0x00241041, 0x00241048, 0x00241049, 0x00241200,
    0x00241201, 0x00241208, 0x00241209, 0x00241240, 0x00241241, 0x00241248, 0x00241249, 0x00248000,
    0x00248001, 0x00248008, 0x00248009, 0x00248040, 0x00248041, 0x00248048, 0x00248049, 0x00248200,
    0x00248201, 0x00248208, 0x00248209, 0x00248240, 0x00248241, 0x00248248, 0x00248249, 0x00249000,
    0x00249001, 0x00249008, 0x00249009, 0x00249040, 0x00249041, 0x00249048, 0x00249049, 0x00249200,
    0x00249201, 0x00249208, 0x00249209, 0x00249240, 0x00249241, 0x00249248, 0x00249249};

// pre-shifted table for Y coordinates (1 bit to the left)
static constexpr uint32_t morton256_y[256] = {
    0x00000000,
    0x00000002, 0x00000010, 0x00000012, 0x00000080, 0x00000082, 0x00000090, 0x00000092, 0x00000400,
    0x00000402, 0x00000410, 0x00000412, 0x00000480, 0x00000482, 0x00000490, 0x00000492, 0x00002000,
    0x00002002, 0x00002010, 0x00002012, 0x00002080, 0x00002082, 0x00002090, 0x00002092, 0x00002400,
    0x00002402, 0x00002410, 0x00002412, 0x00002480, 0x00002482, 0x00002490, 0x00002492, 0x00010000,
    0x00010002, 0x00010010, 0x00010012, 0x00010080, 0x00010082, 0x00010090, 0x00010092, 0x00010400,
    0x00010402, 0x00010410, 0x00010412, 0x00010480, 0x00010482, 0x00010490, 0x00010492, 0x00012000,
    0x00012002, 0x00012010, 0x00012012, 0x00012080, 0x00012082, 0x00012090, 0x00012092, 0x00012400,
    0x00012402, 0x00012410, 0x00012412, 0x00012480, 0x00012482, 0x00012490, 0x00012492, 0x00080000,
    0x00080002, 0x00080010, 0x00080012, 0x00080080, 0x00080082, 0x00080090, 0x00080092, 0x00080400,
    0x00080402, 0x00080410, 0x00080412, 0x00080480, 0x00080482, 0x00080490, 0x00080492, 0x00082000,
    0x00082002, 0x00082010, 0x00082012, 0x00082080, 0x00082082, 0x00082090, 0x00082092, 0x00082400,
    0x00082402, 0x00082410, 0x00082412, 0x00082480, 0x00082482, 0x00082490, 0x00082492, 0x00090000,
    0x00090002, 0x00090010, 0x00090012, 0x00090080, 0x00090082, 0x00090090, 0x00090092, 0x00090400,
    0x00090402, 0x00090410, 0x00090412, 0x00090480, 0x00090482, 0x00090490, 0x00090492, 0x00092000,
    0x00092002, 0x00092010, 0x00092012, 0x00092080, 0x00092082, 0x00092090, 0x00092092, 0x00092400,
    0x00092402, 0x00092410, 0x00092412, 0x00092480, 0x00092482, 0x00092490, 0x00092492, 0x00400000,
    0x00400002, 0x00400010, 0x00400012, 0x00400080, 0x00400082, 0x00400090, 0x00400092, 0x00400400,
    0x00400402, 0x00400410, 0x00400412, 0x00400480, 0x00400482, 0x00400490, 0x00400492, 0x00402000,
    0x00402002, 0x00402010, 0x00402012, 0x00402080, 0x00402082, 0x00402090, 0x00402092, 0x00402400,
    0x00402402, 0x00402410, 0x00402412, 0x00402480, 0x00402482, 0x00402490, 0x00402492, 0x00410000,
    0x00410002, 0x00410010, 0x00410012, 0x00410080, 0x00410082, 0x00410090, 0x00410092, 0x00410400,
    0x00410402, 0x00410410, 0x00410412, 0x00410480, 0x00410482, 0x00410490, 0x00410492, 0x00412000,
    0x00412002, 0x00412010, 0x00412012, 0x00412080, 0x00412082, 0x00412090, 0x00412092, 0x00412400,
    0x00412402, 0x00412410, 0x00412412, 0x00412480, 0x00412482, 0x00412490, 0x00412492, 0x00480000,
    0x00480002, 0x00480010, 0x00480012, 0x00480080, 0x00480082, 0x00480090, 0x00480092, 0x00480400,
    0x00480402, 0x00480410, 0x00480412, 0x00480480, 0x00480482, 0x00480490, 0x00480492, 0x00482000,
    0x00482002, 0x00482010, 0x00482012, 0x00482080, 0x00482082, 0x00482090, 0x00482092, 0x00482400,
    0x00482402, 0x00482410, 0x00482412, 0x00482480, 0x00482482, 0x00482490, 0x00482492, 0x00490000,
    0x00490002, 0x00490010, 0x00490012, 0x00490080, 0x00490082, 0x00490090, 0x00490092, 0x00490400,
    0x00490402, 0x00490410, 0x00490412, 0x00490480, 0x00490482, 0x00490490, 0x00490492, 0x00492000,
    0x00492002, 0x00492010, 0x00492012, 0x00492080, 0x00492082, 0x00492090, 0x00492092, 0x00492400,
    0x00492402, 0x00492410, 0x00492412, 0x00492480, 0x00492482, 0x00492490, 0x00492492};

// Pre-shifted table for z (2 bits to the left)
static constexpr uint32_t morton256_z[256] = {
    0x00000000,
    0x00000004, 0x00000020, 0x00000024, 0x00000100, 0x00000104, 0x00000120, 0x00000124, 0x00000800,
    0x00000804, 0x00000820, 0x00000824, 0x00000900, 0x00000904, 0x00000920, 0x00000924, 0x00004000,
    0x00004004, 0x00004020, 0x00004024, 0x00004100, 0x00004104, 0x00004120, 0x00004124, 0x00004800,
    0x00004804, 0x00004820, 0x00004824, 0x00004900, 0x00004904, 0x00004920, 0x00004924, 0x00020000,
    0x00020004, 0x00020020, 0x00020024, 0x00020100, 0x00020104, 0x00020120, 0x00020124, 0x00020800,
    0x00020804, 0x00020820, 0x00020824, 0x00020900, 0x00020904, 0x00020920, 0x00020924, 0x00024000,
    0x00024004, 0x00024020, 0x00024024, 0x00024100, 0x00024104, 0x00024120, 0x00024124, 0x00024800,
    0x00024804, 0x00024820, 0x00024824, 0x00024900, 0x00024904, 0x00024920, 0x00024924, 0x00100000,
    0x00100004, 0x00100020, 0x00100024, 0x00100100, 0x00100104, 0x00100120, 0x00100124, 0x00100800,
    0x00100804, 0x00100820, 0x00100824, 0x00100900, 0x00100904, 0x00100920, 0x00100924, 0x00104000,
    0x00104004, 0x00104020, 0x00104024, 0x00104100, 0x00104104, 0x00104120, 0x00104124, 0x00104800,
    0x00104804, 0x00104820, 0x00104824, 0x00104900, 0x00104904, 0x00104920, 0x00104924, 0x00120000,
    0x00120004, 0x00120020, 0x00120024, 0x00120100, 0x00120104, 0x00120120, 0x00120124, 0x00120800,
    0x00120804, 0x00120820, 0x00120824, 0x00120900, 0x00120904, 0x00120920, 0x00120924, 0x00124000,
    0x00124004, 0x00124020, 0x00124024, 0x00124100, 0x00124104, 0x00124120, 0x00124124, 0x00124800,
    0x00124804, 0x00124820, 0x00124824, 0x00124900, 0x00124904, 0x00124920, 0x00124924, 0x00800000,
    0x00800004, 0x00800020, 0x00800024, 0x00800100, 0x00800104, 0x00800120, 0x00800124, 0x00800800,
    0x00800804, 0x00800820, 0x00800824, 0x00800900, 0x00800904, 0x00800920, 0x00800924, 0x00804000,
    0x00804004, 0x00804020, 0x00804024, 0x00804100, 0x00804104, 0x00804120, 0x00804124, 0x00804800,
    0x00804804, 0x00804820, 0x00804824, 0x00804900, 0x00804904, 0x00804920, 0x00804924, 0x00820000,
    0x00820004, 0x00820020, 0x00820024, 0x00820100, 0x00820104, 0x00820120, 0x00820124, 0x00820800,
    0x00820804, 0x00820820, 0x00820824, 0x00820900, 0x00820904, 0x00820920, 0x00820924, 0x00824000,
    0x00824004, 0x00824020, 0x00824024, 0x00824100, 0x00824104, 0x00824120, 0x00824124, 0x00824800,
    0x00824804, 0x00824820, 0x00824824, 0x00824900, 0x00824904, 0x00824920, 0x00824924, 0x00900000,
    0x00900004, 0x00900020, 0x00900024, 0x00900100, 0x00900104, 0x00900120, 0x00900124, 0x00900800,
    0x00900804, 0x00900820, 0x00900824, 0x00900900, 0x00900904, 0x00900920, 0x00900924, 0x00904000,
    0x00904004, 0x00904020, 0x00904024, 0x00904100, 0x00904104, 0x00904120, 0x00904124, 0x00904800,
    0x00904804, 0x00904820, 0x00904824, 0x00904900, 0x00904904, 0x00904920, 0x00904924, 0x00920000,
    0x00920004, 0x00920020, 0x00920024, 0x00920100, 0x00920104, 0x00920120, 0x00920124, 0x00920800,
    0x00920804, 0x00920820, 0x00920824, 0x00920900, 0x00920904, 0x00920920, 0x00920924, 0x00924000,
    0x00924004, 0x00924020, 0x00924024, 0x00924100, 0x00924104, 0x00924120, 0x00924124, 0x00924800,
    0x00924804, 0x00924820, 0x00924824, 0x00924900, 0x00924904, 0x00924920, 0x00924924};
}

/**
 *
 *  Stack size needs to be increaed for this to work !!!
 *
 */

//=========================================================
// OCTREE (Morton-code, flat node array)
//=========================================================
class Octree final
{
public:
    using MortonCode = std::uint64_t;
    struct Item
    {
        // glm::vec3 position; // world-space position (voxel center)
        MortonCode morton; // morton code in [0, 2^(3*maxDepth) )
    };

private:
    //=========================================================
    // Basic AABB utilities for your Aabb struct
    //=========================================================
    constexpr glm::vec3 aabbCenter(const Aabb& b) const noexcept
    {
        return (b.minimum + b.maximum) * 0.5f;
    }

    constexpr glm::vec3 aabbHalfSize(const Aabb& b) const noexcept
    {
        return (b.maximum - b.minimum) * 0.5f;
    }

    constexpr bool aabbContains(const Aabb& b, const glm::vec3& p) const noexcept
    {
        return glm::all(glm::lessThanEqual(b.minimum, p)) && glm::all(glm::lessThanEqual(p, b.maximum));
    }

    constexpr bool aabbIntersects(const Aabb& a, const Aabb& b) const noexcept
    {
        return (a.minimum.x <= b.maximum.x && a.maximum.x >= b.minimum.x) && (a.minimum.y <= b.maximum.y && a.maximum.y >= b.minimum.y) && (a.minimum.z <= b.maximum.z && a.maximum.z >= b.minimum.z);
    }

    // Create a sub AABB from parent + octant index (0..7)
    constexpr Aabb makeChildAabb(const Aabb& parent, int octant) const noexcept
    {
        glm::vec3 center = aabbCenter(parent);
        glm::vec3 min = parent.minimum;
        glm::vec3 max = parent.maximum;

        glm::vec3 cmin = min;
        glm::vec3 cmax = max;

        // X
        if (octant & 1) {
            cmin.x = center.x;
        } else {
            cmax.x = center.x;
        }

        // Y
        if (octant & 2) {
            cmin.y = center.y;
        } else {
            cmax.y = center.y;
        }

        // Z
        if (octant & 4) {
            cmin.z = center.z;
        } else {
            cmax.z = center.z;
        }

        return {cmin, cmax};
    }
    //=========================================================
    // Morton code utilities
    //=========================================================

    // Expands a 21-bit integer into 63 bits by inserting 2 zeros between each bit.
    constexpr MortonCode morton3D(std::uint32_t x, std::uint32_t y, std::uint32_t z) const noexcept
    {
        MortonCode mortton = 0;
        mortton = morton256_z[(z >> 16) & 0xFF] | morton256_y[(y >> 16) & 0xFF] | morton256_x[(x >> 16) & 0xFF];
        mortton = mortton << 48 | morton256_z[(z >> 8) & 0xFF] | morton256_y[(y >> 8) & 0xFF] | morton256_x[(x >> 8) & 0xFF];
        mortton = mortton << 24 | morton256_z[(z) & 0xFF] | morton256_y[(y) & 0xFF] | morton256_x[(x) & 0xFF];
        return mortton;
    }

    constexpr std::uint32_t compactBits(MortonCode v) const noexcept
    {
        v &= 0x1249249249249249ULL;
        v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3ULL;
        v = (v ^ (v >> 4)) & 0x100f00f00f00f00fULL;
        v = (v ^ (v >> 8)) & 0x1f0000ff0000ffULL;
        v = (v ^ (v >> 16)) & 0x1f00000000ffffULL;
        v = (v ^ (v >> 32)) & 0x1fffffULL; // 21 bits back
        return static_cast<std::uint32_t>(v);
    }

    // Decode a Morton code back to integer voxel indices
    glm::uvec3 decodeMortonToVoxel(MortonCode morton) const noexcept
    {
        const std::uint32_t ix = compactBits(morton);
        const std::uint32_t iy = compactBits(morton >> 1);
        const std::uint32_t iz = compactBits(morton >> 2);
        return glm::uvec3(ix, iy, iz);
    }

    glm::vec3 voxelIndexToCenter(const glm::uvec3& idx) const noexcept
    {
        return m_rootBounds.minimum + (glm::vec3(idx) + 0.5f) * m_VoxelSize;
    }

    glm::vec3 decodeMortonToPosition(MortonCode morton) const noexcept
    {
        glm::uvec3 idx = decodeMortonToVoxel(morton);
        return voxelIndexToCenter(idx);
    }

    struct Node
    {
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

    static constexpr std::uint32_t INVALID_INDEX = std::numeric_limits<std::uint32_t>::max();

    // Tree data
    std::vector<Item> m_items; // sorted by morton after build
    std::vector<Node> m_nodes; // flat array, node 0 = root

    Aabb m_rootBounds{};
    size_t m_maxItems;
    size_t m_maxDepth = 0; // number of octree levels actually used
    std::uint32_t m_bitsPerAxis = 0; // number of Morton bits per axis in use

    const float m_VoxelSize;
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

    // Recursive build of nodes over sorted m_items
    std::uint32_t buildNodeRecursive(std::uint32_t begin,
        std::uint32_t end,
        const Aabb& bounds,
        std::uint32_t depth) noexcept
    {
        const std::uint32_t nodeIndex = static_cast<std::uint32_t>(m_nodes.size());
        m_nodes.emplace_back();

        Node& node = m_nodes[nodeIndex];
        node.start = begin;
        node.count = end - begin;
        node.children.fill(INVALID_INDEX);

        if (depth >= m_maxDepth || node.count <= m_maxItems) {
            return nodeIndex;
        }

        const std::uint32_t totalDepth = static_cast<std::uint32_t>(m_maxDepth);
        const std::uint32_t levelShift = 3 * (totalDepth - 1 - depth);

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
                if (aabbContains(range, decodeMortonToPosition(it.morton)))
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
                const auto pos = decodeMortonToPosition(it.morton);
                out->emplace_back(pos - (m_VoxelSize * 0.5f), pos + (m_VoxelSize * 0.5f)); // min max
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
    explicit Octree(const std::filesystem::path& path,
        float voxSize,
        size_t maxItemsPerLeaf = 16)
        : m_maxItems(maxItemsPerLeaf),
          m_VoxelSize(voxSize)
    {
        ObjMesh mesh = readObjFile(path);

        // Initial bounds from geometry; will be expanded to match the Morton grid
        m_rootBounds = computeBboxFromAttrib(mesh.attrib);

        // m_bitsPerAxis and m_maxDepth are computed inside buildVoxelGrid()
        buildVoxelGrid(voxSize, mesh);
    }

    std::vector<Aabb> getAabbs() const noexcept
    {
        std::vector<Aabb> ret;
        if (m_nodes.empty()) {
            return ret;
        }
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

        const size_t width = static_cast<size_t>(std::ceil((bb.maximum.x - bb.minimum.x) / voxelSize));
        const size_t height = static_cast<size_t>(std::ceil((bb.maximum.y - bb.minimum.y) / voxelSize));
        const size_t depth = static_cast<size_t>(std::ceil((bb.maximum.z - bb.minimum.z) / voxelSize));

        std::println("Grid dimensions: {}x{}x{}", width, height, depth);
        std::println("Voxel size: {}", voxelSize);

        // If there is literally no extent, there is nothing to voxelize
        const size_t maxDim = std::max(width, std::max(height, depth));
        if (maxDim == 0) {
            std::println("Empty voxel grid (zero extent).");
            return;
        }

        // Number of bits per axis needed to index [0 .. maxDim-1]
        m_bitsPerAxis = static_cast<std::uint32_t>(
            std::ceil(std::log2(static_cast<double>(maxDim))));

        // Limit: 21 bits per axis (Morton layout uses 21 bits per axis)
        if (m_bitsPerAxis > 21) {
            throw std::runtime_error("We support up to 21 bits per axis (max 2^21 voxels per dimension)!");
        }

        // One octree level per Morton bit
        m_maxDepth = static_cast<size_t>(m_bitsPerAxis);

        // The Morton hierarchy assumes a conceptual grid of 2^m_bitsPerAxis cells per axis.
        // We only actually use [0..width/height/depth), but the AABB must match the full Morton grid
        const float gridExtent = voxelSize * static_cast<float>(1u << m_bitsPerAxis);
        m_rootBounds.minimum = bb.minimum;
        m_rootBounds.maximum = bb.minimum + glm::vec3(gridExtent, gridExtent, gridExtent);

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

        if (true) {
            for (size_t s = 0; s < ObjData.shapes.size(); ++s) {
                const auto& mesh = ObjData.shapes[s].mesh;

                for (size_t i = 0; i < mesh.indices.size(); i += 3) {

                    if (i + 2 >= mesh.indices.size()) break; // Safety check

                    int materialId = -1;
                    if (!mesh.material_ids.empty()) {
                        const size_t faceIndex = i / 3;
                        if (faceIndex < mesh.material_ids.size()) {
                            materialId = mesh.material_ids[faceIndex];
                        }
                    }

                    const tinyobj::index_t i0 = mesh.indices[i];
                    const tinyobj::index_t i1 = mesh.indices[i + 1];
                    const tinyobj::index_t i2 = mesh.indices[i + 2];

                    const auto p0 = loadPos(i0);
                    const auto p1 = loadPos(i1);
                    const auto p2 = loadPos(i2);

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
                                    m_items.emplace_back(morton3D(x, y, z));
                                }
                            }
                        }
                    }
                }
            }
        } else {

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
                                        localItems.emplace_back(morton3D(x, y, z));
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
        }

        std::println("Total voxels inserted (before tree build): {}", m_items.size());

        // Now actually build the Morton octree (this already uses parallel sort)
        buildTree();
        m_nodes.shrink_to_fit();
        m_items.shrink_to_fit();

        std::println("Total octree nodes: {}", m_nodes.size());
    }
};
