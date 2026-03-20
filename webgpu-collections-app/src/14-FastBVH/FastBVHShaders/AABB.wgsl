// After having the Patricia Tree, we do a bottom-up AABB parallel construction of the BVH.
// We assign one thread per leaf, and we climb through parent pointers, with an atomic
// counter per internal node. The first thread to reach the node finishes,
// the second computes the AABB of the union of the children boxes.
// From Karras 2012: https://research.nvidia.com/sites/default/files/publications/karras2012hpg_paper.pdf

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
    _pad0: u32,
    _pad1: u32,
}; // Size 48

struct LeafAABB {
    aabbMin: vec3f,
    _pad0: u32,
    aabbMax: vec3f,
    _pad1: u32,
}; // Size 32

//================================//
@group(0) @binding(0) var<storage, read_write>  internalNodes:  array<BVHNode>;
@group(0) @binding(1) var<storage, read>        leafParents:    array<u32>;
@group(0) @binding(2) var<storage, read_write>  atomicCounters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read>        vertices:       array<f32>;
@group(0) @binding(4) var<storage, read>        indices:        array<u32>;
@group(0) @binding(5) var<storage, read>        sortedIndices:  array<u32>;
@group(0) @binding(6) var<storage, read_write>  leafAABBs:      array<LeafAABB>;

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

    let i0 = indices[originalTriangleIndex * 3u];
    let i1 = indices[originalTriangleIndex * 3u + 1u];
    let i2 = indices[originalTriangleIndex * 3u + 2u];
    let v0 = vec3f(vertices[i0 * 3u], vertices[i0 * 3u + 1u], vertices[i0 * 3u + 2u]);
    let v1 = vec3f(vertices[i1 * 3u], vertices[i1 * 3u + 1u], vertices[i1 * 3u + 2u]);
    let v2 = vec3f(vertices[i2 * 3u], vertices[i2 * 3u + 1u], vertices[i2 * 3u + 2u]);

    // Each threads AT LEATS writes their start AABB into the leafAABBs buffer.
    leafAABBs[leafIndex].aabbMin = min(v0, min(v1, v2));
    leafAABBs[leafIndex].aabbMax = max(v0, max(v1, v2));

    // climb up the tree
    var parentNode = leafParents[leafIndex];

    loop {
        if (parentNode == 0xFFFFFFFFu) // Reached the root
        {
            break;
        }
        
        let isFirstToReach = atomicAdd(&atomicCounters[parentNode], 1u);
        if (isFirstToReach == 0u)
        {
            return;
        }

        // Compute the union of the AABBs of the children.
        let leftChildIndex = internalNodes[parentNode].left;
        let rightChildIndex = internalNodes[parentNode].right;

        let leftChildAABB: LeafAABB = getChildAABB(leftChildIndex);
        let rightChildAABB: LeafAABB = getChildAABB(rightChildIndex);

        let mergedMin: vec3f = min(leftChildAABB.aabbMin, rightChildAABB.aabbMin);
        let mergedMax: vec3f = max(leftChildAABB.aabbMax, rightChildAABB.aabbMax);

        internalNodes[parentNode].aabbMin = mergedMin;
        internalNodes[parentNode].aabbMax = mergedMax;
        internalNodes[parentNode].triangleCount = getChildTriangleCount(leftChildIndex) + getChildTriangleCount(rightChildIndex);

        parentNode = internalNodes[parentNode].parent;
    }
}

//================================//
fn getChildAABB(childIndex: u32) -> LeafAABB
{
    if ((childIndex & LEAF_BIT) != 0u)
    {
        let index = childIndex & 0x7FFFFFFFu; // Do not take the leaf bit
        return leafAABBs[index];
    }
    else
    {
        return LeafAABB(internalNodes[childIndex].aabbMin, 0u, internalNodes[childIndex].aabbMax, 0u);
    }
}

//================================//
fn getChildTriangleCount(childIndex: u32) -> u32
{
    if ((childIndex & LEAF_BIT) != 0u)
    {
        return 1u;
    }
    else
    {
        return internalNodes[childIndex].triangleCount;
    }
}