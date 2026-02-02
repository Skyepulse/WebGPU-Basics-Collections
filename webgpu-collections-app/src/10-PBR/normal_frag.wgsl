struct Material {
    albedo : vec3<f32>,
    metalness : f32,
    roughness : f32,
    _pad0 : f32,
    _pad1 : f32,
    _pad2 : f32,
}; // Total: 32 bytes

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

    cameraPosition: vec3f,
    _pad0: f32,

    a_c: f32,
    a_l: f32,
    a_q: f32,
    _pad2: f32,

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

// ============================== //
@fragment
fn fs(input: VertexOutput) -> @location(0) vec4f {
    
    //var totalColor = lambertShading(input);
    var totalColor = microfacetBRDF(input);
    return vec4f(totalColor, 1.0);
}

// ============================== //
fn lambertShading(input: VertexOutput) -> vec3f
{
    var albedo = material.albedo;

    let ka = 0.1;
    var n = normalize(input.normal);

    var totalColor = ka * albedo;

    for (var i = 0; i < 3; i = i + 1)
    {
        if (uniforms.lights[i].enabled < 0.5)
        {   
            continue;
        }

        var toLight = uniforms.lights[i].position - input.position;
        var lightDistance = length(toLight);
        var wi = normalize(toLight);
        
        // Check if we are in the cone 
        var lightDir = normalize(uniforms.lights[i].direction);
        var cosAngle = dot(-wi, lightDir);
        if (cosAngle < cos(uniforms.lights[i].coneAngle)) 
        {
            continue;
        }

        let NdotL = max(dot(n, wi), 0.0);
        let lightAttenuation = 1.0 / (uniforms.a_c + uniforms.a_l * lightDistance + uniforms.a_q * lightDistance * lightDistance);
        let diffuse = NdotL * lightAttenuation * uniforms.lights[i].intensity * uniforms.lights[i].color * albedo;

        totalColor = totalColor + diffuse;
    }

    return totalColor;
}

// ============================== //
fn microfacetBRDF(input: VertexOutput) -> vec3f
{
    // Trowbridge-Reitz (GGX) normal distribution function
    let albedo = material.albedo;
    let alphap = max(material.roughness, 0.04);
    let metalness = material.metalness;

    let ka = 0.1;
    var n = normalize(input.normal);
    let pi = 3.14159265359;

    var totalColor = ka * albedo * (1.0 - metalness); // Prepare a small ambient term

    for (var i = 0; i < 3; i = i + 1)
    {
        if (uniforms.lights[i].enabled < 0.5)
        {
            continue;
        }

        let toLight = uniforms.lights[i].position - input.position;
        let lightDistance = length(toLight);
        let wi = normalize(toLight);

        let toCamera = uniforms.cameraPosition - input.position;
        let wo = normalize(toCamera);

        let wh = normalize(wi + wo);

        // Dot products
        let NdotV = max(dot(n, wo), 0.0001);
        let NdotL = max(dot(n, wi), 0.0001);
        let NdotH = max(dot(n, wh), 0.0);
        let LdotH = max(dot(wi, wh), 0.0);

        // Check if we are in the cone 
        var lightDir = normalize(uniforms.lights[i].direction);
        var cosAngle = dot(-wi, lightDir);
        if (cosAngle < cos(uniforms.lights[i].coneAngle)) 
        {
            continue;
        }

        // Fresnel term (schlick's approximation)
        let F0 = mix(vec3(0.04), albedo, metalness);
        let F = F0 + (1.0 - F0) * pow(1.0 - LdotH, 5.0);

        // f = fd + fs
        // fd = lambert BRDF
        // fs = microfacet BRDF = DFG term

        // DIFFUSE
        let lambert = albedo / pi;
        let kd = (1.0 - F) * (1.0 - metalness);
        let fd = kd * lambert;

        // SPECULAR

        // Trowbridge-Reitz Distribution
        let D = (alphap * alphap) / (pi * pow((NdotH * NdotH) * (alphap * alphap - 1.0) + 1.0, 2.0));

        // Geometry term (GGX)
        let K = (alphap) * sqrt(2.0 / pi);
        let G_schlick_wo = NdotV / (NdotV * (1.0 - K) + K);
        let G_schlick_wi = NdotL / (NdotL * (1.0 - K) + K);
        let G = G_schlick_wo * G_schlick_wi;

        let EPSILON = 0.0001;
        let fs = (D * F * G) / (4.0 * NdotL * NdotV + EPSILON);

        let f = fd + fs;

        let lightAttenuation = 1.0 / (uniforms.a_c + uniforms.a_l * lightDistance + uniforms.a_q * lightDistance * lightDistance);
        let radiance = uniforms.lights[i].intensity * uniforms.lights[i].color * lightAttenuation;

        totalColor = totalColor + f * radiance * NdotL;
    }

    return totalColor;
}