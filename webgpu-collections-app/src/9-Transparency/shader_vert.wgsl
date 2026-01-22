struct Uniforms {
    modelMat : mat4x4<f32>,
    viewMat : mat4x4<f32>,
    projMat : mat4x4<f32>,
    lightPosition : vec3<f32>,
    _pad0 : f32,
    lightColor : vec3<f32>,
    lightIntensity: f32,
};

@group(0) @binding(0)
var<uniform> uniforms : Uniforms;

struct VertexInput {
    @location(0) pos: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,  
    @location(3) color: vec3f,
    @builtin(instance_index) instance: u32
};

struct VertexOutput {
    @builtin(position) pos : vec4f,
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) color: vec3f,
};

@vertex
fn vs(input: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.pos = uniforms.projMat * uniforms.viewMat * uniforms.modelMat * vec4f (input.pos, 1);
    output.position = input.pos;
    output.normal = input.normal;
    output.uv = input.uv;
    output.color = input.color;
    return output;
}