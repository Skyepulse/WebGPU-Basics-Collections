// Single-pass bottom-up AABB construction.
//
// Replicates the pattern from AddisonPrairie/WebGPU-LVBH-demo bvh-up-pass.js,
// adapted to our data structures.
//
// Each leaf thread computes its AABB and walks up toward the root.
// At every internal node:
//   - First to arrive  (counter == 0): stores AABB in accumBuffer l_* fields, STOPS.
//   - Second to arrive (counter != 0): reads l_* fields; if not yet written, `continue`
//     restarts the outer while loop which re-increments the counter and retries.
//     Each retry passes through a full SC atomicAdd (memory fence), ensuring
//     eventual visibility of the first sibling's stores. Once visible, merges
//     AABBs, writes to internalNodes, and advances to parent.
//
// Perturb ensures no real AABB component equals exactly 0.0, so i32(0) (the
// zero-cleared accumBuffer value) unambiguously means "not yet written".

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

// Per-internal-node scratch: 6 atomic floats (as i32) + counter + pad = 32 bytes.
// min_x/y/z and max_x/y/z hold the FIRST sibling's parked AABB.
// counter is the arrival counter (0 = nobody yet, !=0 = first sibling arrived).
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

    // ── Compute leaf AABB ──────────────────────────────────────────────── //
    let triIndex = sortedIndices[leafIndex];
    let i0 = indices[triIndex * 3u + 0u];
    let i1 = indices[triIndex * 3u + 1u];
    let i2 = indices[triIndex * 3u + 2u];
    let v0 = vec3f(vertices[i0*3u], vertices[i0*3u+1u], vertices[i0*3u+2u]);
    let v1 = vec3f(vertices[i1*3u], vertices[i1*3u+1u], vertices[i1*3u+2u]);
    let v2 = vec3f(vertices[i2*3u], vertices[i2*3u+1u], vertices[i2*3u+2u]);

    var bboxMin = min(v0, min(v1, v2));
    var bboxMax = max(v0, max(v1, v2));

    // Write unperturbed AABB for DFSFlattening (reads leafAABBs directly).
    leafAABBs[leafIndex].aabbMin = bboxMin;
    leafAABBs[leafIndex].aabbMax = bboxMax;

    if (INTERNAL_NODE_COUNT == 0u) { return; }

    // Perturb so no component equals exactly 0.0 after bitcast to i32.
    // accumBuffer is zeroed with clearBuffer before each build, so
    // any-zero component unambiguously means "not yet stored by first sibling".
    bboxMin -= select(vec3f(0.0), vec3f(1e-8), bboxMin == vec3f(0.0));
    bboxMax += select(vec3f(0.0), vec3f(1e-8), bboxMax == vec3f(0.0));

    // ── Walk up the tree (replicates AddisonPrairie bvh-up-pass pattern) ── //
    var nodeIdx: u32 = leafParents[leafIndex];
    var bDone:   bool = false;

    while (nodeIdx < INTERNAL_NODE_COUNT && !bDone) {

        // SC atomicAdd: also acts as a memory fence so first sibling's earlier
        // stores become progressively visible across retries.
        let sibling = atomicAdd(&accumBuffer[nodeIdx].counter, 1);

        if (sibling == 0) {
            // First sibling: park AABB and stop.
            atomicStore(&accumBuffer[nodeIdx].min_x, bitcast<i32>(bboxMin.x));
            atomicStore(&accumBuffer[nodeIdx].min_y, bitcast<i32>(bboxMin.y));
            atomicStore(&accumBuffer[nodeIdx].min_z, bitcast<i32>(bboxMin.z));
            atomicStore(&accumBuffer[nodeIdx].max_x, bitcast<i32>(bboxMax.x));
            atomicStore(&accumBuffer[nodeIdx].max_y, bitcast<i32>(bboxMax.y));
            atomicStore(&accumBuffer[nodeIdx].max_z, bitcast<i32>(bboxMax.z));
            bDone = true;
        } else {
            // Second (or later, due to retries) sibling: try to read first sibling's AABB.
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

            // Not ready yet: continue restarts the outer while loop,
            // re-incrementing the counter (another SC fence) before the next read.
            // This matches the AddisonPrairie pattern exactly.
            if (any(sibMin == vec3f(0.0)) || any(sibMax == vec3f(0.0))) {
                continue;
            }

            // Merge and write final AABB to internalNodes (race-free: exactly
            // one thread reaches this point per node).
            bboxMin = min(bboxMin, sibMin);
            bboxMax = max(bboxMax, sibMax);
            internalNodes[nodeIdx].aabbMin = bboxMin;
            internalNodes[nodeIdx].aabbMax = bboxMax;

            // Advance to parent.
            nodeIdx = internalNodes[nodeIdx].parent;
        }
    }
}
