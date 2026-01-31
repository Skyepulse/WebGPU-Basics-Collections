struct Material {
    albedo : vec3<f32>,
    _pad0 : f32,
};

struct SpotLight
{
    position: vec3f,
    intensity: f32,

    direction: vec3f,
    coneAngle: f32,

    color: vec3f,
    enabled: f32,
};
struct Uniforms {
    modelMat : mat4x4<f32>,
    viewMat : mat4x4<f32>,
    projMat : mat4x4<f32>,

    a_c: f32,
    a_l: f32,
    a_q: f32,
    _pad0: f32,

    lights : array<SpotLight, 3>,
};

struct VertexOutput {
    @builtin(position) pos : vec4f,
    @location(0) position: vec3f,
    @location(1) normal: vec3f,
    @location(2) uv: vec2f,
};

@group(0) @binding(0)
var<uniform> uniforms : Uniforms;
@group(1) @binding(0)
var<uniform> material : Material;

@fragment
fn fs(input: VertexOutput) -> @location(0) vec4f {
    var albedo = material.albedo;
    const kd = 1.0;
    const ka = 0.1;
    
    var n = normalize(input.normal);
    var wi = normalize(uniforms.lights[0].position - input.position);
    
    // Diffuse
    var fd = uniforms.lights[0].color * max(0.0, dot(wi, n));
    
    // Ambient should also be multiplied by albedo!
    var toLight = uniforms.lights[0].position - input.position;
    var lightDistance = length(toLight);

    // Check if we are in the cone, we know light direction
    var lightDir = normalize(-uniforms.lights[0].direction);
    var cosAngle = dot(wi, lightDir);
    if (cosAngle < cos(uniforms.lights[0].coneAngle)) 
    {
        fd = vec3f(0.0, 0.0, 0.0);
    }
    else
    {
        let NdotL = max(0.0, dot(n, wi));
        let lightAttenuation = 1.0 / (uniforms.a_c + uniforms.a_l * lightDistance + uniforms.a_q * lightDistance * lightDistance);
        fd = fd * NdotL * lightAttenuation;
    }
    var ambient = ka * albedo;
    var diffuse = kd * fd * albedo * uniforms.lights[0].intensity * uniforms.lights[0].enabled;
    
    return vec4f(ambient + diffuse, 1.0);
}