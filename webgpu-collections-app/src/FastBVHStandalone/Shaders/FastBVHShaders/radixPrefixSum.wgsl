// Help from: https://github.com/kishimisu/WebGPU-Radix-Sort

//================================//
override THREADS_PER_WORKGROUP: u32;
override X_SIZE: u32;
override Y_SIZE: u32;
override ITEMS_PER_WORKGROUP: u32;
override ELEMENT_COUNT: u32;

//================================//
@group(0) @binding(0) var<storage, read_write> items: array<u32>;
@group(0) @binding(1) var<storage, read_write> blockSums: array<u32>;

var<workgroup> tempBuffer: array<u32, ITEMS_PER_WORKGROUP>;

//================================//
// SCAN SWEEP, Up and then Down
// Blelloch 2 elements per thread
@compute
@workgroup_size(X_SIZE, Y_SIZE, 1)
fn cs_reduce(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) num_work: vec3<u32>, 
    @builtin(local_invocation_index) l_id: u32)
{
    let WID = (w_id.x + w_id.y * num_work.x) * THREADS_PER_WORKGROUP;
    let GID = WID + l_id;

    // elem 1
    tempBuffer[l_id * 2u] = select(items[GID * 2u], 0u, GID * 2u >= ELEMENT_COUNT);
    // elem 2
    tempBuffer[l_id * 2u + 1u] = select(items[GID * 2u + 1u], 0u, GID * 2u + 1u >= ELEMENT_COUNT);

    var offset: u32 = 1u;

    // Up sweep, one half of threads each iter
    // Will output the total sum of the items
    for(var d: u32 = ITEMS_PER_WORKGROUP >> 1u; d > 0u; d >>= 1u) // Divide by 2 each iter
    {
        workgroupBarrier();

        if (l_id < d) // If active thread
        {
            let ai = offset * (2u * l_id + 1u) - 1u;
            let bi = offset * (2u * l_id + 2u) - 1u;

            tempBuffer[bi] += tempBuffer[ai];
        }
        offset *= 2u;
    }

    // only first thread saves result
    if (l_id == 0u)
    {
        let lastOffset = ITEMS_PER_WORKGROUP - 1u;

        blockSums[(w_id.x + w_id.y * num_work.x)] = tempBuffer[lastOffset];
        tempBuffer[lastOffset] = 0u; // need to clear it so the down sweep produces the EXCLUSIVE correct prefix sum
    }

    // Down Sweep, twice as many threads each iter
    for(var d: u32 = 1u; d < ITEMS_PER_WORKGROUP; d *= 2u)
    {   
        offset >>= 1u;
        workgroupBarrier();

        if (l_id < d) // active thread
        {
            var ai: u32 = offset * (l_id * 2u + 1u) - 1u;
            var bi: u32 = offset * (l_id * 2u + 2u) - 1u;

            let temp: u32 = tempBuffer[ai];
            tempBuffer[ai] = tempBuffer[bi];
            tempBuffer[bi] = temp + tempBuffer[bi];
        }
    }
    workgroupBarrier();

    // Results to global memory
    if (GID * 2u >= ELEMENT_COUNT)
    {
        return;
    }
    items[GID * 2u] = tempBuffer[l_id * 2u];

    if (GID * 2u + 1u >= ELEMENT_COUNT)
    {
        return;
    }
    items[GID * 2u + 1u] = tempBuffer[l_id * 2u + 1u];
}

//================================//
@compute
@workgroup_size(X_SIZE, Y_SIZE, 1)
fn cs_add(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) num_work: vec3<u32>, 
    @builtin(local_invocation_index) l_id: u32)
{
    let WID = (w_id.x + w_id.y * num_work.x) * THREADS_PER_WORKGROUP;
    let GID = WID + l_id;

    let addValue = blockSums[w_id.x + w_id.y * num_work.x];

    if (GID * 2u >= ELEMENT_COUNT)
    {
         return;
    }
    items[GID * 2u] += addValue;

    if (GID * 2u + 1u >= ELEMENT_COUNT)
    {
         return;
    }
    items[GID * 2u + 1u] += addValue;
}