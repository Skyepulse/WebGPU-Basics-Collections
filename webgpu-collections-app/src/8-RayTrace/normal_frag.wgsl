struct Uniforms {
    modelMat : mat4x4<f32>,
    viewMat : mat4x4<f32>,
    projMat : mat4x4<f32>,
    lightPosition : vec3<f32>,
    _pad0 : f32,
    lightColor : vec3<f32>,
    _pad1 : f32,
};

struct VertexOutput {
    @builtin(position) pos : vec4f,
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
    @location(3) color: vec3f,
};

@group(0) @binding(0)
var<uniform> uniforms : Uniforms;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4f {
    var albedo = input.color;
    const kd = 1.0;
    const ka = 0.1;
    
    var n = normalize(input.normal);
    var wi = normalize(uniforms.lightPosition - input.position);
    
    // Diffuse
    var fd = uniforms.lightColor * max(0.0, dot(wi, n));
    
    // Ambient should also be multiplied by albedo!
    var ambient = ka * albedo;
    var diffuse = kd * fd * albedo;
    
    return vec4f(ambient + diffuse, 1.0);
}