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
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read> localPrefixSums: array<u32>;
@group(0) @binding(3) var<storage, read> blockSums: array<u32>;

// In case of key-value sort.
@group(0) @binding(4) var<storage, read> valueData: array<u32>;
@group(0) @binding(5) var<storage, read_write> valueOutput: array<u32>;

@group(1) @binding(0) var<uniform> uniforms: Uniform;

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

    if (GID >= ELEMENT_COUNT)
    {
        return;
    }

    // The final position of the element is this formula:
    // finalPosition = blockSum[bucket * WORKGROUP_COUNT + (w_id.x + w_id.y * num_work.x)] + localPrefixSum[GID]

    let key = data[GID];
    let value = valueData[GID];

    let bits: u32 = (key >> uniforms.currentBit) & 0x3;
    let finalPosition = blockSums[bits * WORKGROUP_COUNT + (w_id.x + w_id.y * num_work.x)] + localPrefixSums[GID];

    output[finalPosition] = key;
    valueOutput[finalPosition] = value;
}