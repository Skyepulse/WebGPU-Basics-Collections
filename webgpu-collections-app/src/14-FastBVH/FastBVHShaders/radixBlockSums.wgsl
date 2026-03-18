//================================//
override THREADS_PER_WORKGROUP: u32;
override X_SIZE: u32;
override Y_SIZE: u32;
override ELEMENT_COUNT: u32;
override WORKGROUP_COUNT: u32;

//================================//
struct Uniform
{
    currentBit: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};
//================================//
@group(0) @binding(0) var<storage, read> data: array<u32>;
@group(0) @binding(1) var<storage, read_write> localPrefixSums: array<u32>;
@group(0) @binding(2) var<storage, read_write> blockSums: array<u32>;

@group(1) @binding(0) var<uniform> uniforms: Uniform;

var<workgroup> prefixSumDoubleBuffer: array<u32, 2 * (THREADS_PER_WORKGROUP + 1)>;

//================================//
@compute
@workgroup_size(X_SIZE, Y_SIZE, 1)
fn cs(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) num_work: vec3<u32>, 
    @builtin(local_invocation_index) l_id: u32)
{
    let WID = (w_id.x + w_id.y * num_work.x) * THREADS_PER_WORKGROUP;
    let GID = WID + l_id;

    let element = select(data[GID], 0u, GID >= ELEMENT_COUNT);
    let bits: u32 = (element >> uniforms.currentBit) & 0x3; // 2 bits per pass

    var bitPrefixSums = array<u32, 4>(0u, 0u, 0u, 0u);

    var lastThreadID: u32 = 0xffffffff;
    if ( w_id.x + w_id.y * num_work.x < WORKGROUP_COUNT )
    {
        // In case our workgroup is not fully occupied
        lastThreadID = min(THREADS_PER_WORKGROUP, ELEMENT_COUNT - WID) - 1u;
    }
    let isLastThread = l_id == lastThreadID;

    // Double buffering:we write to A or B
    let TPW = THREADS_PER_WORKGROUP + 1u;
    var swapOffset: u32 = 0u;
    var inOffset: u32 = l_id;
    var outOffset: u32 = l_id + TPW;

    // prefx sum
    for (var bucket: u32 = 0u; bucket < 4u; bucket++)
    {
        let bitMask = select(0u, 1u, bits == bucket); // so 0 everywhere, except in our bucket
        prefixSumDoubleBuffer[inOffset + 1u] = bitMask;
        workgroupBarrier(); // Let all threads write bitmask to A before reading and writing to B

        var prefixSum: u32 = 0u;

        for (var offset: u32 = 1u; offset < THREADS_PER_WORKGROUP; offset *= 2u)
        {
            if (l_id >= offset)
            {
                prefixSum = prefixSumDoubleBuffer[inOffset] + prefixSumDoubleBuffer[inOffset - offset];
            }
            else
            {
                prefixSum = prefixSumDoubleBuffer[inOffset];
            }

            prefixSumDoubleBuffer[outOffset] = prefixSum;

            // Swap, we know read from B and write to A
            outOffset = inOffset;
            swapOffset = TPW - swapOffset;
            inOffset = l_id + swapOffset;

            workgroupBarrier(); // Let all threads write their prefix sum before next iteration
        }

        // in this workgroup, this many elements are in this current bucket
        bitPrefixSums[bucket] = prefixSum;

        // Write it if last thread
        if (isLastThread)
        {
            let totalSum: u32 = prefixSum + bitMask;
            blockSums[bucket * WORKGROUP_COUNT + (w_id.x + w_id.y * num_work.x)] = totalSum;
        }

        outOffset = inOffset;
        swapOffset = TPW - swapOffset;
        inOffset = l_id + swapOffset;
    }

    if (GID < ELEMENT_COUNT)
    {
        localPrefixSums[GID] = bitPrefixSums[bits]; // bits is either 00, 01, 10, 11
    }
}