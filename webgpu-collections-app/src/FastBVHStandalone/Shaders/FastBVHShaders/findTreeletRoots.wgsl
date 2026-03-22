// This is shader 1 / 2 of the treelet optimization step described in 
// https://www.highperformancegraphics.org/wp-content/uploads/2013/Karras-BVH.pdf

// This shader is purely bottom-up identification of treelet roots, that will be written 
// into a compact atomic shader.
// We also count the number of roots we find, as it will be the indirect dispatch count
// for the second shader.
// This same counter is used as a grabbable atomic index every time we find a root to
// write into a ID array buffer, with max size the total number of internal nodes.

// The bottom up algorithm is the same as the AABB pass, first thread to reach a node stops.
// The second one records the total number of triangles in the subtree.
// if >= gamma, record it as treelet root.

//================================//
override THREADS_PER_WORKGROUP: u32;
override LEAF_NODE_COUNT: u32;

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
    _pad1: u32,
};

struct Uniform
{
    gamma: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

//================================//
@group(0) @binding(0) var<storage, read> internalNodes: array<BVHNode>;
@group(0) @binding(1) var<storage, read> leafParents: array<u32>;
@group(0) @binding(2) var<storage, read_write>  atomicCounters: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write>  treeletRoots:   array<u32>;
@group(0) @binding(4) var<storage, read_write>  indirectDispatchArgs:   array<atomic<u32>>; // 4 u32: [NUMTREELETS, 1, 1, 0]

@group(1) @binding(0) var<uniform> uniforms: Uniform;

//================================//
@compute
@workgroup_size(THREADS_PER_WORKGROUP)
fn cs(@builtin(global_invocation_id) gid: vec3u)
{
    if (gid.x >= LEAF_NODE_COUNT) 
    {
        return;
    }

    if (gid.x == 0u)
    {
        atomicStore(&indirectDispatchArgs[1], 1u);
        atomicStore(&indirectDispatchArgs[2], 1u);
    }

    var parentNode = leafParents[gid.x];

    loop {
        if (parentNode == 0xFFFFFFFFu) // Reached the root.
        {
            break;
        }

        let firstToArrive = atomicAdd(&atomicCounters[parentNode], 1u);
        if (firstToArrive == 0u)
        {
            return;
        }

        // one mor trelet found, record it and add 1 to the counter.
        if (internalNodes[parentNode].triangleCount >= uniforms.gamma)
        {
            let currentNumTreelets = atomicAdd(&indirectDispatchArgs[0], 1u);
            treeletRoots[currentNumTreelets] = parentNode;
        }

        parentNode = internalNodes[parentNode].parent;
    }
}

