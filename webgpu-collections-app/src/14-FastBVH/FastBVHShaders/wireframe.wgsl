// One thread per BVH node (internal OR leaf).
// Each thread climbs parent pointers to compute its depth, then:
//   - Internal node: draw if depth == maxDepth - 1  (frontier / boundary box)
//   - Leaf node:     draw if depth <  maxDepth       (all leaves within the shown region)
// Nodes that should not be drawn write a degenerate box at the origin (invisible zero-length lines).

override THREADS_PER_WORKGROUP: u32;
override INTERNAL_NODE_COUNT: u32;
override LEAF_NODE_COUNT: u32;

//================================//
struct BVHNode
{
    aabbMin: vec3f,
    parent:  u32,
    aabbMax: vec3f,
    padding: u32,
    left:    u32,
    right:   u32,
    _pad0: u32,
    _pad1: u32,
}; // 48 bytes

struct LeafAABB {
    aabbMin: vec3f,
    _pad0: u32,
    aabbMax: vec3f,
    _pad1: u32,
}; // 32 bytes

struct WireframeDepth {
    maxDepth: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}; // 16 bytes

//================================//
@group(0) @binding(0) var<storage, read>       internalNodes:  array<BVHNode>;
@group(0) @binding(1) var<storage, read>       leafAABBs:      array<LeafAABB>;
@group(0) @binding(2) var<storage, read_write> wireframeVerts: array<f32>;
@group(0) @binding(3) var<storage, read>       leafParents:    array<u32>;
@group(0) @binding(4) var<uniform>             depthUniforms:  WireframeDepth;

//================================//
fn internalDepth(startIdx: u32) -> u32 
{
    var d: u32 = 0u;
    var cur: u32 = startIdx;
    for (var i: u32 = 0u; i < 64u; i++) 
    {
        let p = internalNodes[cur].parent;
        if (p == 0xFFFFFFFFu) 
        { 
            break; 
        }
        d += 1u;
        cur = p;
    }
    return d;
}

//================================//
fn writeVert(i: u32, v: vec3f) 
{
    wireframeVerts[i]     = v.x;
    wireframeVerts[i + 1] = v.y;
    wireframeVerts[i + 2] = v.z;
}

//================================//
fn writeBox(base: u32, bMin: vec3f, bMax: vec3f) 
{
    let c000 = vec3f(bMin.x, bMin.y, bMin.z);
    let c100 = vec3f(bMax.x, bMin.y, bMin.z);
    let c010 = vec3f(bMin.x, bMax.y, bMin.z);
    let c110 = vec3f(bMax.x, bMax.y, bMin.z);
    let c001 = vec3f(bMin.x, bMin.y, bMax.z);
    let c101 = vec3f(bMax.x, bMin.y, bMax.z);
    let c011 = vec3f(bMin.x, bMax.y, bMax.z);
    let c111 = vec3f(bMax.x, bMax.y, bMax.z);

    var i: u32 = base;

    writeVert(i, c000); i += 3u; writeVert(i, c100); i += 3u;
    writeVert(i, c100); i += 3u; writeVert(i, c110); i += 3u;
    writeVert(i, c110); i += 3u; writeVert(i, c010); i += 3u;
    writeVert(i, c010); i += 3u; writeVert(i, c000); i += 3u;

    writeVert(i, c001); i += 3u; writeVert(i, c101); i += 3u;
    writeVert(i, c101); i += 3u; writeVert(i, c111); i += 3u;
    writeVert(i, c111); i += 3u; writeVert(i, c011); i += 3u;
    writeVert(i, c011); i += 3u; writeVert(i, c001); i += 3u;

    writeVert(i, c000); i += 3u; writeVert(i, c001); i += 3u;
    writeVert(i, c100); i += 3u; writeVert(i, c101); i += 3u;
    writeVert(i, c010); i += 3u; writeVert(i, c011); i += 3u;
    writeVert(i, c110); i += 3u; writeVert(i, c111); i += 3u;
}

//================================//
@compute
@workgroup_size(THREADS_PER_WORKGROUP)
fn cs(@builtin(global_invocation_id) gid: vec3u)
{
    let totalNodes = INTERNAL_NODE_COUNT + LEAF_NODE_COUNT;
    if (gid.x >= totalNodes) 
    { 
        return; 
    }

    let maxDepth = depthUniforms.maxDepth;
    let floatsPerNode: u32 = 24u * 3u;
    let outBase: u32 = gid.x * floatsPerNode;

    if (gid.x < INTERNAL_NODE_COUNT)
    {
        let d = internalDepth(gid.x);
        let show = (maxDepth > 0u) && (d == maxDepth - 1u);
        if (!show) {
            writeBox(outBase, vec3f(0.0), vec3f(0.0));
            return;
        }
        let node = internalNodes[gid.x];
        writeBox(outBase, node.aabbMin, node.aabbMax);
    }
    else
    {
        let leafIdx = gid.x - INTERNAL_NODE_COUNT;
        let parentInternalDepth = internalDepth(leafParents[leafIdx]);
        let d = parentInternalDepth + 1u;
        let show = (maxDepth > 0u) && (d < maxDepth);
        if (!show) {
            writeBox(outBase, vec3f(0.0), vec3f(0.0));
            return;
        }
        let leaf = leafAABBs[leafIdx];
        writeBox(outBase, leaf.aabbMin, leaf.aabbMax);
    }
}
