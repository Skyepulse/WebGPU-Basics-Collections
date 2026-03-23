// A shader to draw the BVH nodes as a wireframe for the rasterization pipeline.
// Since the number of drawn wireframes is not known until we compute the exact nodes
// at the desired depth, we keep a buffer for indirect dispatch later in the pipeline.

//================================//
override THREADS_PER_WORKGROUP: u32 = 256;
const DEPTH_INFINITY: u32 = 0xFFFFFFFFu;

//================================//
struct FlatBVHNode
{
    minB:        vec3f,
    leftOrFirst: u32,
    maxB:        vec3f,
    count:       u32,
};

struct Uniforms
{
    totalNodes:  u32,
    targetDepth: u32,
    _pad0:       u32,
    _pad1:       u32,
};

//================================//
@group(0) @binding(0) var<storage, read>           flatBVH:  array<FlatBVHNode>;
@group(0) @binding(1) var<storage, read_write>     diffBuf:  array<atomic<i32>>;
@group(0) @binding(2) var<storage, read_write>     depthBuf: array<u32>;
@group(0) @binding(3) var<storage, read_write>     vertices: array<f32>;
@group(0) @binding(4) var<storage, read_write>     drawArgs: array<atomic<u32>>;
@group(0) @binding(5) var<uniform>                 uniforms: Uniforms;

//================================//
@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_mark(@builtin(global_invocation_id) gid: vec3u)
{
    let i = gid.x;
    if (i >= uniforms.totalNodes) { return; }

    let node = flatBVH[i];
    if (node.count != 0u) { return; } 

    atomicAdd(&diffBuf[i + 1u], 1);

    let miss = node.leftOrFirst;
    if (miss < uniforms.totalNodes) {
        atomicAdd(&diffBuf[miss], -1);
    }
}

//================================//
// Sequential prefix sum
@compute @workgroup_size(1)
fn cs_scan(@builtin(global_invocation_id) gid: vec3u)
{
    var running: i32 = 0;
    for (var i = 0u; i < uniforms.totalNodes; i++) 
    {
        running   += atomicLoad(&diffBuf[i]);
        depthBuf[i] = u32(max(running, 0));
    }
}

//================================//
// Constructs the wireframe vertex buffer for the rasterization pass. 
// Each BVH node is an AABB with 12 edges.
@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs_emit(@builtin(global_invocation_id) gid: vec3u)
{
    let i = gid.x;
    if (i >= uniforms.totalNodes) { return; }

    let node   = flatBVH[i];
    let depth  = depthBuf[i];
    let isLeaf = node.count > 0u;

    var shouldDraw: bool;
    if (uniforms.targetDepth == DEPTH_INFINITY) 
    {
        shouldDraw = isLeaf;
    } 
    else 
    {
        shouldDraw = (depth == uniforms.targetDepth) || (isLeaf && depth < uniforms.targetDepth);
    }
    if (!shouldDraw) { return; }

    let base = atomicAdd(&drawArgs[0], 24u);

    let x0 = node.minB.x;  let y0 = node.minB.y;  let z0 = node.minB.z;
    let x1 = node.maxB.x;  let y1 = node.maxB.y;  let z1 = node.maxB.z;

    let c0 = vec3f(x0, y0, z0);
    let c1 = vec3f(x1, y0, z0);
    let c2 = vec3f(x1, y1, z0);
    let c3 = vec3f(x0, y1, z0);
    let c4 = vec3f(x0, y0, z1);
    let c5 = vec3f(x1, y0, z1);
    let c6 = vec3f(x1, y1, z1);
    let c7 = vec3f(x0, y1, z1);

    writeEdge(base,  0u, c0, c1);
    writeEdge(base,  2u, c1, c2);
    writeEdge(base,  4u, c2, c3);
    writeEdge(base,  6u, c3, c0);
    writeEdge(base,  8u, c4, c5);
    writeEdge(base, 10u, c5, c6);
    writeEdge(base, 12u, c6, c7);
    writeEdge(base, 14u, c7, c4);
    writeEdge(base, 16u, c0, c4);
    writeEdge(base, 18u, c1, c5);
    writeEdge(base, 20u, c2, c6);
    writeEdge(base, 22u, c3, c7);
}

//================================//
fn writeEdge(base: u32, edgeOffset: u32, a: vec3f, b: vec3f)
{
    let vBase = (base + edgeOffset) * 3u;
    vertices[vBase + 0u] = a.x;
    vertices[vBase + 1u] = a.y;
    vertices[vBase + 2u] = a.z;
    vertices[vBase + 3u] = b.x;
    vertices[vBase + 4u] = b.y;
    vertices[vBase + 5u] = b.z;
}