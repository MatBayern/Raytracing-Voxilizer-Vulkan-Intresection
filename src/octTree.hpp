#pragma once
#include <array>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>

template <typename T>
class OctTree
{
public:
    struct Node
    {
        explicit Node(T data) : m_data(std::move(data)) {}
        T m_data;
        std::array<std::unique_ptr<Node>, 8> m_child{}; // 8-way fanout
    };

    OctTree() = default;
    ~OctTree() = default;

    // Create a node from value and insert it
    Node& insert(T value)
    {
        auto node = std::make_unique<Node>(std::move(value));
        return insert(node.release());
    }

    // Insert an externally created (childless) node
    Node& insert(Node* node)
    {
        if (!node) {
            throw std::invalid_argument("insert(Node*): node is null");
        }

        // Ensure the provided node has no children
        for (const auto& c : node->m_child) {
            if (c) {
                throw std::invalid_argument("insert(Node*): node must not have children");
            }
        }

        // Empty tree -> become root
        if (!m_root) {
            m_root.reset(node);
            return *m_root;
        }

        // BFS: attach to first free child slot
        std::queue<Node*> q;
        q.push(m_root.get());

        while (!q.empty()) {
            Node* cur = q.front();
            q.pop();

            for (auto& childPtr : cur->m_child) {
                if (!childPtr) {
                    childPtr.reset(node);
                    return *childPtr;
                }
                q.push(childPtr.get());
            }
        }

        // Should never be reached
        throw std::runtime_error("insert(Node*): unexpected failure");
    }

    // BFS search by value; returns reference or throws if not found
    const Node& find(const T& value) const
    {
        if (!m_root)
            throw std::out_of_range("find: value not found (empty tree)");

        std::queue<const Node*> q;
        q.push(m_root.get());

        while (!q.empty()) {
            const Node* cur = q.front();
            q.pop();

            if (cur->m_data == value)
                return *cur;

            for (const auto& childPtr : cur->m_child)
                if (childPtr)
                    q.push(childPtr.get());
        }

        throw std::out_of_range("find: value not found");
    }

private:
    std::unique_ptr<Node> m_root;
};
