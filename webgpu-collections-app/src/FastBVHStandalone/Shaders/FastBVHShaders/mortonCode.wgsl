// Shader that takes all triangles and scene max and min bounds
// Computes the morton code of 30 bit (10 bit per axis) of each
// centroid and writes it to an output buffer, and another with the triangle index for each code
// They will become key/value for the radix sort next step

//================================//
override SIZE_X: u32;
override SIZE_Y: u32;
override THREADS_PER_WORKGROUP: u32;
override TRIANGLE_COUNT: u32;

//================================//
@group(0) @binding(0) var<storage, read> vertexBuffer: array<f32>;
@group(0) @binding(1) var<storage, read> indexBuffer: array<u32>;
@group(0) @binding(2) var<storage, read> sceneBounds: array<f32>; // [minX, minY, minZ, maxX, maxY, maxZ]
@group(0) @binding(3) var<storage, read_write> mortonCodes: array<u32>;
@group(0) @binding(4) var<storage, read_write> triangleIndices: array<u32>;

//================================//
@compute
@workgroup_size(SIZE_X, SIZE_Y, 1)
fn cs(
    @builtin(workgroup_id) w_id: vec3<u32>,
    @builtin(num_workgroups) num_work: vec3<u32>,
    @builtin(local_invocation_index) l_id: u32)
{
    // One triangle per thread
    let workgroupIndex = w_id.x + w_id.y * num_work.x;
    let totalWorkgroups = num_work.x * num_work.y;
    let WID = workgroupIndex * THREADS_PER_WORKGROUP;
    let GID = WID + l_id;

    if (GID >= TRIANGLE_COUNT)
    {
        return;
    }

    let sceneMin = vec3<f32>(sceneBounds[0], sceneBounds[1], sceneBounds[2]);
    let sceneMax = vec3<f32>(sceneBounds[3], sceneBounds[4], sceneBounds[5]);

    let indexBase = GID * 3u;
    let v0Index = indexBuffer[indexBase] * 3u;
    let v1Index = indexBuffer[indexBase + 1u] * 3u;
    let v2Index = indexBuffer[indexBase + 2u] * 3u;

    let v0 = vec3<f32>(vertexBuffer[v0Index], vertexBuffer[v0Index + 1u], vertexBuffer[v0Index + 2u]);
    let v1 = vec3<f32>(vertexBuffer[v1Index], vertexBuffer[v1Index + 1u], vertexBuffer[v1Index + 2u]);
    let v2 = vec3<f32>(vertexBuffer[v2Index], vertexBuffer[v2Index + 1u], vertexBuffer[v2Index + 2u]);

    let centroid = (v0 + v1 + v2) / 3.0;
    let normalizedCentroid = (centroid - sceneMin) / (sceneMax - sceneMin);

    // 10 BITS PER AXIS
    let mortonCode: u32 = morton3D(normalizedCentroid);
    mortonCodes[GID] = mortonCode;
    triangleIndices[GID] = GID;
}

//================================//
fn morton3D(p: vec3<f32>) -> u32
{
    let x = u32(clamp(p.x * 1024.0, 0.0, 1023.0));
    let y = u32(clamp(p.y * 1024.0, 0.0, 1023.0));
    let z = u32(clamp(p.z * 1024.0, 0.0, 1023.0));

    return (expandBits(x) << 2u) | (expandBits(y) << 1u) | expandBits(z);
}

//================================//
fn expandBits(v: u32) -> u32
{
    var x = v & 0x000003ffu;
    x = (x | (x << 16u)) & 0x30000fffu;
    x = (x | (x << 8u)) & 0x300f00f0u;
    x = (x | (x << 4u)) & 0x30c30c30u;
    x = (x | (x << 2u)) & 0x9249249u;
    return x;
}