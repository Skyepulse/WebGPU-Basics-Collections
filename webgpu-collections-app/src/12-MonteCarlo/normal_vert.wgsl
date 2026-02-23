struct AreaLight
{
    center: vec3f,
    intensity: f32,

    normalDirection: vec3f,
    width: f32,

    color: vec3f,
    height: f32,

    enabled: f32,
    _pad: f32,
    _pad2: f32,
    _pad3: f32,
}; // 4 * 4 = 64 bytes

struct Uniforms {
    viewMat : mat4x4<f32>, // 64 bytes
    projMat : mat4x4<f32>, // 64 bytes

    cameraPosition: vec3f, 
    _pad0: f32,

    a_c: f32,
    a_l: f32,
    a_q: f32,
    _pad2: f32,

    light : AreaLight,
}; // 64 + 64 + 16 + 16 + 64 = 224 bytes

@group(0) @binding(0)
var<uniform> uniforms : Uniforms;

@group(1) @binding(6)
var<storage, read> modelMatrix : mat4x4<f32>;
@group(1) @binding(7)
var<storage, read> normalMatrix : mat3x3<f32>;

struct VertexInput {
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,  
    @builtin(instance_index) instance: u32
};

struct VertexOutput {
    @builtin(position) pos : vec4f,
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
};

@vertex
fn vs(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;

    let worldPos = modelMatrix * vec4f(input.pos, 1.0);
    output.pos = uniforms.projMat * uniforms.viewMat * worldPos;
    output.position = worldPos.xyz;
    output.normal = normalize(normalMatrix * input.normal);
    output.uv = input.uv;
    return output;
}
