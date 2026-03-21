// This shader is the final step of the BVH construction process.
// It flattens out the BVH into a DFS ordered array for efficient traversal on the GPU.

//================================//
override THREADS_PER_WORKGROUP: u32;
override INTERNAL_NODE_COUNT: u32;
override LEAF_NODE_COUNT: u32;

//================================//
const LEAF_BIT: u32 = 0x80000000u;
const INVALID_NODE: u32 = 0xFFFFFFFFu;

//================================//
struct InputBVHNode
{
    aabbMin: vec3f,
    parent:  u32,
    aabbMax: vec3f,
    triangleCount: u32,

    left:    u32,
    right:   u32,
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
struct FlatBVHNode
{
    minB: vec3f,
    leftOrFirst: u32, // internal: missLink, leaf: first global triangle index

    maxB: vec3f,
    count: u32, // 0 = internal, >0 = leaf
};

//================================//*
@group(0) @binding(0) var<storage, read> internalNodes: array<InputBVHNode>;
@group(0) @binding(1) var<storage, read> leafParents: array<u32>;
@group(0) @binding(2) var<storage, read> leafAABBs: array<LeafAABB>;
@group(0) @binding(3) var<storage, read> sortedTriangleIndices: array<u32>;
@group(0) @binding(4) var<storage, read_write> flatBVHNodes: array<FlatBVHNode>;

//================================//
@compute
@workgroup_size(THREADS_PER_WORKGROUP)
fn cs(@builtin(global_invocation_id) gid: vec3u)
{
    let totalNodes = INTERNAL_NODE_COUNT + LEAF_NODE_COUNT;
    let threadIndex = gid.x;

    if (threadIndex >= totalNodes)
    {
        return;
    }

    if (threadIndex < INTERNAL_NODE_COUNT)
    {
        let nodeIndex = threadIndex;
        let dfsIndex = computeInternalDFSIndex(nodeIndex);

        flatBVHNodes[dfsIndex].minB = internalNodes[nodeIndex].aabbMin;
        flatBVHNodes[dfsIndex].maxB = internalNodes[nodeIndex].aabbMax;

        // correct miss link in the traversal is the first node after this whole subtree
        flatBVHNodes[dfsIndex].leftOrFirst = dfsIndex + internalNodes[nodeIndex].subTreeNodeCount;
        flatBVHNodes[dfsIndex].count = 0u;
        return;
    }

    let leafIndex = threadIndex - INTERNAL_NODE_COUNT;
    let dfsIndex = computeLeafDFSIndex(leafIndex);

    flatBVHNodes[dfsIndex].minB = leafAABBs[leafIndex].aabbMin;
    flatBVHNodes[dfsIndex].maxB = leafAABBs[leafIndex].aabbMax;

    flatBVHNodes[dfsIndex].leftOrFirst = sortedTriangleIndices[leafIndex];
    flatBVHNodes[dfsIndex].count = 1u;
}

//================================//
fn computeInternalDFSIndex(nodeIndex: u32) -> u32
{
    var dfsIndex: u32 = 0u;
    var currentToken: u32 = nodeIndex;
    var parentIndex: u32 = internalNodes[nodeIndex].parent;

    loop {
        if (parentIndex == INVALID_NODE)
        {
            break;
        }

        let leftChild = internalNodes[parentIndex].left;
        let rightChild = internalNodes[parentIndex].right;

        if (currentToken == rightChild)
        {
            dfsIndex += 1u + getChildSubTreeNodeCount(leftChild);
        }
        else
        {
            dfsIndex += 1u;
        }

        currentToken = parentIndex;
        parentIndex = internalNodes[parentIndex].parent;
    }

    return dfsIndex;
}

//================================//
fn computeLeafDFSIndex(leafIndex: u32) -> u32
{
    var dfsIndex: u32 = 0u;
    var currentToken: u32 = LEAF_BIT | leafIndex;
    var parentIndex: u32 = leafParents[leafIndex];

    loop {
        if (parentIndex == INVALID_NODE)
        {
            break;
        } 

        let leftChild = internalNodes[parentIndex].left;
        let rightChild = internalNodes[parentIndex].right;

        if (currentToken == rightChild)
        {
            dfsIndex += 1u + getChildSubTreeNodeCount(leftChild);
        }
        else
        {
            dfsIndex += 1u;
        }

        currentToken = parentIndex;
        parentIndex = internalNodes[parentIndex].parent;
    }

    return dfsIndex;
}

//================================//
fn getChildSubTreeNodeCount(childIndex: u32) -> u32
{
    if ((childIndex & LEAF_BIT) != 0u)
    {
        return 1u;
    }
    else
    {
        return internalNodes[childIndex].subTreeNodeCount;
    }
}