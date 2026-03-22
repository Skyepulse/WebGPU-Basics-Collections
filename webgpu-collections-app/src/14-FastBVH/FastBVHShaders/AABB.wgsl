// Initialize each leaf AABB and prepare for the 
// next batch of N dispatches that will 
// go up the tree level by level
// and construct the internal node AABBs without racing conditions.

//================================//
override THREADS_PER_WORKGROUP: u32;
override INTERNAL_NODE_COUNT: u32;
override LEAF_NODE_COUNT: u32;
const LEAF_BIT: u32 = 0x80000000u; // The flag bit

//================================//
struct BVHNode 
{
    aabbMin: vec3f,
    parent:  u32,
    aabbMax: vec3f,
    triangleCount: u32,

    left:    u32,
    right:   u32,
    sahCost: f32,
    subTreeNodeCount: u32,
}; // Size 48

struct LeafAABB 
{
    aabbMin: vec3f,
    _pad0: u32,
    aabbMax: vec3f,
    _pad1: u32,
}; // Size 32

//================================//
@group(0) @binding(0) var<storage, read> vertices: array<f32>;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> sortedIndices: array<u32>;
@group(0) @binding(3) var<storage, read> internalNodes: array<BVHNode>;
@group(0) @binding(4) var<storage, read_write> leafAABBs: array<LeafAABB>;
@group(0) @binding(5) var<storage, read_write> pendingChildCounts: array<atomic<u32>>;

// Frontier buffers are used to know which internal nodes are ready to be dispatches for AABB computation.
// We are only ready once we know both children AABBs have been computed.
@group(0) @binding(6) var<storage, read_write> frontierNodes: array<u32>;
@group(0) @binding(7) var<storage, read_write> frontierCount: array<atomic<u32>>;

//================================//
@compute
@workgroup_size(THREADS_PER_WORKGROUP)
fn cs(@builtin(global_invocation_id) gid: vec3u)
{
    let threadIndex = gid.x;

    if (threadIndex < LEAF_NODE_COUNT)
    {
        let leafIndex = threadIndex;
        let originalTriangleIndex = sortedIndices[leafIndex];

        let i0 = indices[originalTriangleIndex * 3u + 0u];
        let i1 = indices[originalTriangleIndex * 3u + 1u];
        let i2 = indices[originalTriangleIndex * 3u + 2u];
        let v0 = vec3f(vertices[i0 * 3u + 0u], vertices[i0 * 3u + 1u], vertices[i0 * 3u + 2u]);
        let v1 = vec3f(vertices[i1 * 3u + 0u], vertices[i1 * 3u + 1u], vertices[i1 * 3u + 2u]);
        let v2 = vec3f(vertices[i2 * 3u + 0u], vertices[i2 * 3u + 1u], vertices[i2 * 3u + 2u]);

        leafAABBs[leafIndex].aabbMin = min(v0, min(v1, v2));
        leafAABBs[leafIndex].aabbMax = max(v0, max(v1, v2));
    }

    if (threadIndex >= INTERNAL_NODE_COUNT)
    {
        return;
    }

    let leftChildIndex = internalNodes[threadIndex].left;
    let rightChildIndex = internalNodes[threadIndex].right;
    let pendingChildren = select(0u, 1u, (leftChildIndex & LEAF_BIT) == 0u) +
        select(0u, 1u, (rightChildIndex & LEAF_BIT) == 0u);

    atomicStore(&pendingChildCounts[threadIndex], pendingChildren);

    if (pendingChildren == 0u)
    {
        let slot = atomicAdd(&frontierCount[0], 1u);
        frontierNodes[slot] = threadIndex;
    }
}
