struct SpotLight
{
    position: vec3f,
    intensity: f32,

    direction: vec3f,
    coneAngle: f32,

    color: vec3f,
    _pad: f32,
}; // 48 bytes

struct Uniforms {
    viewMat : mat4x4<f32>,
    projMat : mat4x4<f32>,

    cameraPosition: vec3f,
    _pad0: f32,

    a_c: f32,
    a_l: f32,
    a_q: f32,
    _pad2: f32,

    lights : array<SpotLight, 3>,
};

@group(0) @binding(0)
var<uniform> uniforms : Uniforms;

@group(1) @binding(6)
var<storage, read> modelMatrix : mat4x4<f32>;

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
fn vsBVH(@location(0) position: vec3f) -> VertexOutput {
    var output: VertexOutput;
    let worldPos = modelMatrix * vec4f(position, 1.0);
    output.pos = uniforms.projMat * uniforms.viewMat * worldPos;
    output.position = worldPos.xyz;
    output.normal = vec3f(0.0);
    output.uv = vec2f(0.0);
    return output;
}