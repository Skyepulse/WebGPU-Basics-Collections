// Deterministic construction of AABB per internal node,
// which then gives us the dispatch indirect arguments for the next pass
// until we end up processing the entire tree.
// This is the fix of not being able to one shot a bottom-up pass on the tree
// which would construct the final AABB in one single dispatch...

//================================//
override THREADS_PER_WORKGROUP: u32;
const LEAF_BIT: u32 = 0x80000000u;
const INVALID_NODE: u32 = 0xFFFFFFFFu;

//================================//
struct BVHNode
{
    aabbMin: vec3f,
    parent: u32,
    aabbMax: vec3f,
    triangleCount: u32,

    left: u32,
    right: u32,
    sahCost: f32,
    subTreeNodeCount: u32,
};

//================================//
struct LeafAABB
{
    aabbMin: vec3f,
    _pad0: u32,
    aabbMax: vec3f,
    _pad1: u32,
};

//================================//
@group(0) @binding(0) var<storage, read_write> internalNodes: array<BVHNode>;
@group(0) @binding(1) var<storage, read> leafAABBs: array<LeafAABB>;
@group(0) @binding(2) var<storage, read_write> pendingChildCounts: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> frontierIn: array<u32>;
@group(0) @binding(4) var<storage, read_write> frontierInCount: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> frontierOut: array<u32>;
@group(0) @binding(6) var<storage, read_write> frontierOutCount: array<atomic<u32>>;

//================================//
@compute
@workgroup_size(THREADS_PER_WORKGROUP)
fn cs_reduce(@builtin(global_invocation_id) gid: vec3u)
{
    let frontierCount = atomicLoad(&frontierInCount[0]);
    if (gid.x >= frontierCount)
    {
        return;
    }

    let nodeIndex = frontierIn[gid.x];

    let leftChildIndex = internalNodes[nodeIndex].left;
    let rightChildIndex = internalNodes[nodeIndex].right;

    let leftChildAABB = getChildAABB(leftChildIndex);
    let rightChildAABB = getChildAABB(rightChildIndex);

    let mergedMin = min(leftChildAABB.aabbMin, rightChildAABB.aabbMin);
    let mergedMax = max(leftChildAABB.aabbMax, rightChildAABB.aabbMax);

    internalNodes[nodeIndex].aabbMin = mergedMin;
    internalNodes[nodeIndex].aabbMax = mergedMax;
    internalNodes[nodeIndex].sahCost = 0.0;

    let parentIndex = internalNodes[nodeIndex].parent;
    if (parentIndex == INVALID_NODE)
    {
        return;
    }

    let previousPending = atomicSub(&pendingChildCounts[parentIndex], 1u);
    if (previousPending == 1u)
    {
        let slot = atomicAdd(&frontierOutCount[0], 1u);
        frontierOut[slot] = parentIndex;
    }
}

//================================//
@compute
@workgroup_size(1)
fn cs_prepare()
{
    atomicStore(&frontierOutCount[0], 0u);
}

//================================//
fn getChildAABB(childIndex: u32) -> LeafAABB
{
    if ((childIndex & LEAF_BIT) != 0u)
    {
        let index = childIndex & 0x7FFFFFFFu;
        return leafAABBs[index];
    }

    return LeafAABB(internalNodes[childIndex].aabbMin, 0u, internalNodes[childIndex].aabbMax, 0u);
}
