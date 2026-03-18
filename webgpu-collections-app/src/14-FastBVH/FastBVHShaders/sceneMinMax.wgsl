// Level 0 of hierarchical scene min max reduction to find through
// compute shaders the min max of the whole scene.

//================================//
override THREADS_PER_WORKGROUP: u32;
override SIZE_X: u32;
override SIZE_Y: u32;
override TOTAL_VERTICES: u32;

//================================//
@group(0) @binding(0) var<storage, read> vertices: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>; // [minX, minY, minZ, maxX, maxY, maxZ] per workgroup

var<workgroup> shared_min : array<vec3<f32>, THREADS_PER_WORKGROUP>;
var<workgroup> shared_max : array<vec3<f32>, THREADS_PER_WORKGROUP>;

//================================//
@compute
@workgroup_size(SIZE_X, SIZE_Y, 1)
fn cs(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) num_work: vec3<u32>,
    @builtin(local_invocation_index) l_id: u32)
{
    let workgroupIndex = w_id.x + w_id.y * num_work.x;
    let totalWorkgroups = num_work.x * num_work.y;
    let WID = workgroupIndex * THREADS_PER_WORKGROUP;
    let GID = WID + l_id;

    var localMin = vec3<f32>(f32(1e30));
    var localMax = vec3<f32>(f32(-1e30));

    for (var i: u32 = GID; i < TOTAL_VERTICES; i += totalWorkgroups * THREADS_PER_WORKGROUP)
    {
        let baseIndex = i * 3u;
        let vertex = vec3<f32>(vertices[baseIndex], vertices[baseIndex + 1u], vertices[baseIndex + 2u]);
        localMin = min(localMin, vertex);
        localMax = max(localMax, vertex);
    }

    shared_min[l_id] = localMin;
    shared_max[l_id] = localMax;

    // Memo for me: >>= 1u <==> /= 2u BUT faster because of bit shift (?)
    for (var stride: u32 = THREADS_PER_WORKGROUP >> 1u; stride > 0u; stride >>= 1u)
    {
        workgroupBarrier();

        if (l_id < stride) // Active thread
        {
            shared_min[l_id] = min(shared_min[l_id], shared_min[l_id + stride]);
            shared_max[l_id] = max(shared_max[l_id], shared_max[l_id + stride]);
        }
    }

    // Only need one thread of the workgroup to write the result
    let isFirstThread = l_id == 0u;
    if (isFirstThread)
    {
        let outputIndex = workgroupIndex * 6u;
        output[outputIndex]         = shared_min[0].x;
        output[outputIndex + 1u]    = shared_min[0].y;
        output[outputIndex + 2u]    = shared_min[0].z;
        output[outputIndex + 3u]    = shared_max[0].x;
        output[outputIndex + 4u]    = shared_max[0].y;
        output[outputIndex + 5u]    = shared_max[0].z;
    }
}