// We declare a storage variable to read from and write to
@group(0) @binding(0) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(1) fn computeSomething(@builtin(global_invocation_id) id: vec3u) {
    let i = id.x;
    data[i] = data[i] * 2.0;
}

// Over simplification explanation of work group size and invocation ID:
// function dispatchWorkgroups(width, height, depth) {
//   for (z = 0; z < depth; ++z) {
//     for (y = 0; y < height; ++y) {
//       for (x = 0; x < width; ++x) {
//         const workgroup_id = {x, y, z};
//         dispatchWorkgroup(workgroup_id)
//       }
//     }
//   }
// }
//  
// function dispatchWorkgroup(workgroup_id) {
//   // from @workgroup_size in WGSL
//   const workgroup_size = shaderCode.workgroup_size;
//   const {x: width, y: height, z: depth} = workgroup_size;
//   for (z = 0; z < depth; ++z) {
//     for (y = 0; y < height; ++y) {
//       for (x = 0; x < width; ++x) {
//         const local_invocation_id = {x, y, z};
//         const global_invocation_id =
//             workgroup_id * workgroup_size + local_invocation_id;
//         computeShader(global_invocation_id)
//       }
//     }
//   }
// }