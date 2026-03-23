// This is a single pass bottom-up AABB aggregation shader,
// in order to properly give each internal node its final AABB before the DFS flattening pass.
//
// HUGE PROPS TO AddisonPrairie (at https://addisonprairie.github.io/WebGPU-LVBH-demo/) for a VERY niche use of the atomics that now allows us
// to MAKE SURE the children AABBs written are visible to the thread aggregating,
// which was my main problem on the old shader (AABBold) that had race conditions.
//
// The poitns of the algo here is:
//   - First to arrive  (counter == 0): stores AABB in accumBuffer with atomics and STOPS.
//   - Second to arrive (counter != 0): reads these atomic fields, if not ready CONTINUE
// IN theory this is a very fragile pattern, which is why I was surprised it works in the first place.
// CUDA (used in Karras 2012/2013) has dispatch barrier call primitives that send data to memory
// as barriers so the race conditions are not an issue.
// WebGPU does not have those, so we need to do these shady tricks to make it work...
//
// For example, the perturbation here ensures no real AABB component equals exactly 0.0, so i32(0) 
//(the zero-cleared accumBuffer value) unambiguously means "not yet written". This is what we base ourselves
// on to make sure it is safe to do the aggregation and continue up the tree.

//================================//
override THREADS_PER_WORKGROUP: u32;
override LEAF_NODE_COUNT:       u32;
override INTERNAL_NODE_COUNT:   u32;

const INVALID_NODE: u32 = 0xFFFFFFFFu;

//================================//
struct BVHNode
{
    aabbMin:         vec3f,
    parent:          u32,
    aabbMax:         vec3f,
    triangleCount:   u32,
    left:            u32,
    right:           u32,
    sahCost:         f32,
    subTreeNodeCount: u32,
};

struct LeafAABB
{
    aabbMin: vec3f,
    _pad0:   u32,
    aabbMax: vec3f,
    _pad1:   u32,
};

struct AtomicAABBNode
{
    min_x:   atomic<i32>,
    min_y:   atomic<i32>,
    min_z:   atomic<i32>,
    counter: atomic<i32>,
    max_x:   atomic<i32>,
    max_y:   atomic<i32>,
    max_z:   atomic<i32>,
    _pad:    atomic<i32>,
};

//================================//
@group(0) @binding(0) var<storage, read>       vertices:      array<f32>;
@group(0) @binding(1) var<storage, read>       indices:       array<u32>;
@group(0) @binding(2) var<storage, read>       sortedIndices: array<u32>;
@group(0) @binding(3) var<storage, read_write> internalNodes: array<BVHNode>;
@group(0) @binding(4) var<storage, read>       leafParents:   array<u32>;
@group(0) @binding(5) var<storage, read_write> leafAABBs:     array<LeafAABB>;
@group(0) @binding(6) var<storage, read_write> accumBuffer:   array<AtomicAABBNode>;

//================================//
@compute @workgroup_size(THREADS_PER_WORKGROUP)
fn cs(@builtin(global_invocation_id) gid: vec3u)
{
    let leafIndex = gid.x;
    if (leafIndex >= LEAF_NODE_COUNT) { return; }

    let triIndex = sortedIndices[leafIndex];
    let i0 = indices[triIndex * 3u + 0u];
    let i1 = indices[triIndex * 3u + 1u];
    let i2 = indices[triIndex * 3u + 2u];
    let v0 = vec3f(vertices[i0*3u], vertices[i0*3u+1u], vertices[i0*3u+2u]);
    let v1 = vec3f(vertices[i1*3u], vertices[i1*3u+1u], vertices[i1*3u+2u]);
    let v2 = vec3f(vertices[i2*3u], vertices[i2*3u+1u], vertices[i2*3u+2u]);

    var bboxMin = min(v0, min(v1, v2));
    var bboxMax = max(v0, max(v1, v2));

    leafAABBs[leafIndex].aabbMin = bboxMin;
    leafAABBs[leafIndex].aabbMax = bboxMax;

    if (INTERNAL_NODE_COUNT == 0u) { return; }

    // This is the "shady" perturbation part thatfs make it work
    // Since it never really equals to 0, it means we did not write to it yet...
    bboxMin -= select(vec3f(0.0), vec3f(1e-8), bboxMin == vec3f(0.0));
    bboxMax += select(vec3f(0.0), vec3f(1e-8), bboxMax == vec3f(0.0));

    var nodeIdx: u32 = leafParents[leafIndex];
    var bDone:   bool = false;

    while (nodeIdx < INTERNAL_NODE_COUNT && !bDone) 
    {
        let sibling = atomicAdd(&accumBuffer[nodeIdx].counter, 1);

        if (sibling == 0) // First thread writes and bails out
        {
            // First sibling: park AABB and stop.
            atomicStore(&accumBuffer[nodeIdx].min_x, bitcast<i32>(bboxMin.x));
            atomicStore(&accumBuffer[nodeIdx].min_y, bitcast<i32>(bboxMin.y));
            atomicStore(&accumBuffer[nodeIdx].min_z, bitcast<i32>(bboxMin.z));
            atomicStore(&accumBuffer[nodeIdx].max_x, bitcast<i32>(bboxMax.x));
            atomicStore(&accumBuffer[nodeIdx].max_y, bitcast<i32>(bboxMax.y));
            atomicStore(&accumBuffer[nodeIdx].max_z, bitcast<i32>(bboxMax.z));
            bDone = true;
        } 
        else // meaning it was already visited
        {
            let sibMin = vec3f(
                bitcast<f32>(atomicLoad(&accumBuffer[nodeIdx].min_x)),
                bitcast<f32>(atomicLoad(&accumBuffer[nodeIdx].min_y)),
                bitcast<f32>(atomicLoad(&accumBuffer[nodeIdx].min_z))
            );
            let sibMax = vec3f(
                bitcast<f32>(atomicLoad(&accumBuffer[nodeIdx].max_x)),
                bitcast<f32>(atomicLoad(&accumBuffer[nodeIdx].max_y)),
                bitcast<f32>(atomicLoad(&accumBuffer[nodeIdx].max_z))
            );

            // the continue here is super important (dangerous pattern but we have no choice)
            // Continue until sibling finishes writing.
            if (any(sibMin == vec3f(0.0)) || any(sibMax == vec3f(0.0))) {
                continue;
            }

            bboxMin = min(bboxMin, sibMin);
            bboxMax = max(bboxMax, sibMax);
            internalNodes[nodeIdx].aabbMin = bboxMin;
            internalNodes[nodeIdx].aabbMax = bboxMax;

            nodeIdx = internalNodes[nodeIdx].parent;
        }
    }
}