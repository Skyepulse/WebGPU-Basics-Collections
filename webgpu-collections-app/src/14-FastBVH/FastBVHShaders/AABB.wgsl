// After having the Patricia Tree, we do a bottom-up AABB parallel construction of the BVH.
// We assign one thread per leaf, and we climb through parent pointers, with an atomic
// counter per internal node. The first thread to reach the node finishes,
// the second computes the AABB of the union of the children boxes.
// From Karras 2012: https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf

// We also store the initial values for the number of triangles in each subtree,
// and the sahCost as described in Karras 2013: "In practice, we initialize N(n) and C(n) 
// during the AABB fitting step of the initial BVH construction and keep them up to date throughout 
// the optimization."

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
@group(0) @binding(0) var<storage, read>        vertices:      array<f32>;
@group(0) @binding(1) var<storage, read>        indices:       array<u32>;
@group(0) @binding(2) var<storage, read>        sortedIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write>  leafAABBs:     array<LeafAABB>;

//================================//
@compute
@workgroup_size(THREADS_PER_WORKGROUP)
fn cs(@builtin(global_invocation_id) gid: vec3u)
{
    if (gid.x >= LEAF_NODE_COUNT) 
    {
        return;
    }

    let leafIndex = gid.x;
    let originalTriangleIndex = sortedIndices[leafIndex];

    let i0 = indices[originalTriangleIndex * 3u + 0u];
    let i1 = indices[originalTriangleIndex * 3u + 1u];
    let i2 = indices[originalTriangleIndex * 3u + 2u];
    let v0 = vec3f(vertices[i0 * 3u + 0u], vertices[i0 * 3u + 1u], vertices[i0 * 3u + 2u]);
    let v1 = vec3f(vertices[i1 * 3u + 0u], vertices[i1 * 3u + 1u], vertices[i1 * 3u + 2u]);
    let v2 = vec3f(vertices[i2 * 3u + 0u], vertices[i2 * 3u + 1u], vertices[i2 * 3u + 2u]);

    // Each thread writes only its own leaf AABB.
    // Opposed to the CUDA version of the paper, the atomic counter
    // does not guarantee that top threads can read properly the children aggregations
    // such as the AABB of those children or the nodeCount in the subtree it forms.

    // This was very painful to debug, to create and in the end I had no choice but to make
    // another set of passes to complete the aggregation...
    leafAABBs[leafIndex].aabbMin = min(v0, min(v1, v2));
    leafAABBs[leafIndex].aabbMax = max(v0, max(v1, v2));
}