struct VertexOutput {
    @builtin(position) pos : vec4f,
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
};

// ============================== //
@fragment
fn fsBVH(input: VertexOutput) -> @location(0) vec4f {
    return vec4f(0.0, 1.0, 0.0, 1.0);
}