// Deterministic internal-node finalization for the BVH.
// The original leaf-parallel AABB pass computes leaf bounds, but parent aggregation
// through cross-invocation state can be unstable. This pass rebuilds all internal-node
// fields level by level across dispatches.

override THREADS_PER_WORKGROUP: u32;
override INTERNAL_NODE_COUNT: u32;

const LEAF_BIT: u32 = 0x80000000u;

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

struct LeafAABB
{
    aabbMin: vec3f,
    _pad0: u32,
    aabbMax: vec3f,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read_write> internalNodes: array<BVHNode>;
@group(0) @binding(1) var<storage, read> leafAABBs: array<LeafAABB>;
@group(0) @binding(2) var<storage, read> readyIn: array<u32>;
@group(0) @binding(3) var<storage, read_write> readyOut: array<u32>;

@compute
@workgroup_size(THREADS_PER_WORKGROUP)
fn cs(@builtin(global_invocation_id) gid: vec3u)
{
    let nodeIndex = gid.x;
    if (nodeIndex >= INTERNAL_NODE_COUNT)
    {
        return;
    }

    let leftChildIndex = internalNodes[nodeIndex].left;
    let rightChildIndex = internalNodes[nodeIndex].right;

    let leftReady = isChildReady(leftChildIndex);
    let rightReady = isChildReady(rightChildIndex);

    if (!leftReady || !rightReady)
    {
        readyOut[nodeIndex] = readyIn[nodeIndex];
        return;
    }

    let leftChildAABB = getChildAABB(leftChildIndex);
    let rightChildAABB = getChildAABB(rightChildIndex);

    let mergedMin = min(leftChildAABB.aabbMin, rightChildAABB.aabbMin);
    let mergedMax = max(leftChildAABB.aabbMax, rightChildAABB.aabbMax);

    let triangleCount = getChildTriangleCount(leftChildIndex) + getChildTriangleCount(rightChildIndex);
    let leftSAHCost = getChildSAHCost(leftChildIndex);
    let rightSAHCost = getChildSAHCost(rightChildIndex);
    let area = surfaceArea(mergedMin, mergedMax);
    let internalCost = 1.2 * area + leftSAHCost + rightSAHCost;
    let collapseCost = 1.0 * area * f32(triangleCount);

    internalNodes[nodeIndex].aabbMin = mergedMin;
    internalNodes[nodeIndex].aabbMax = mergedMax;
    internalNodes[nodeIndex].triangleCount = triangleCount;
    internalNodes[nodeIndex].sahCost = min(internalCost, collapseCost);
    internalNodes[nodeIndex].subTreeNodeCount = 1u + getChildSubTreeNodeCount(leftChildIndex) + getChildSubTreeNodeCount(rightChildIndex);

    readyOut[nodeIndex] = 1u;
}

fn isChildReady(childIndex: u32) -> bool
{
    if ((childIndex & LEAF_BIT) != 0u)
    {
        return true;
    }

    return readyIn[childIndex] != 0u;
}

fn getChildAABB(childIndex: u32) -> LeafAABB
{
    if ((childIndex & LEAF_BIT) != 0u)
    {
        let index = childIndex & 0x7FFFFFFFu;
        return leafAABBs[index];
    }

    return LeafAABB(internalNodes[childIndex].aabbMin, 0u, internalNodes[childIndex].aabbMax, 0u);
}

fn getChildTriangleCount(childIndex: u32) -> u32
{
    if ((childIndex & LEAF_BIT) != 0u)
    {
        return 1u;
    }

    return internalNodes[childIndex].triangleCount;
}

fn surfaceArea(aabbMin: vec3f, aabbMax: vec3f) -> f32
{
    let d = aabbMax - aabbMin;
    return 2.0 * (d.x * d.y + d.y * d.z + d.z * d.x);
}

fn getChildSAHCost(childIndex: u32) -> f32
{
    if ((childIndex & LEAF_BIT) != 0u)
    {
        let index = childIndex & 0x7FFFFFFFu;
        return surfaceArea(leafAABBs[index].aabbMin, leafAABBs[index].aabbMax);
    }

    return internalNodes[childIndex].sahCost;
}

fn getChildSubTreeNodeCount(childIndex: u32) -> u32
{
    if ((childIndex & LEAF_BIT) != 0u)
    {
        return 1u;
    }

    return internalNodes[childIndex].subTreeNodeCount;
}
