struct Material {
    albedo : vec3<f32>,
    metalness : f32,

    usePerlinMetalness : f32,
    roughness : f32,
    usePerlinRoughness : f32,
    perlinFreq : f32,

    useAlbedoTexture : f32,
    useMetalnessTexture : f32,
    useRoughnessTexture : f32,
    useNormalTexture : f32,

    textureIndex: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}; // Total: 64  bytes

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
@group(1) @binding(1)
var materialSampler: sampler;
@group(1) @binding(2)
var albedoTexture: texture_2d<f32>;
@group(1) @binding(3)
var metalnessTexture: texture_2d<f32>;
@group(1) @binding(4)
var roughnessTexture: texture_2d<f32>;
@group(1) @binding(5)
var normalTexture: texture_2d<f32>;

// ============================== //
@fragment
fn fs(input: VertexOutput) -> @location(0) vec4f {
    
    var totalColor = microfacetBRDF(input);

    return vec4f(totalColor, 1.0);
}

// ============================== //
fn microfacetBRDF(input: VertexOutput) -> vec3f
{
    // Trowbridge-Reitz (GGX) normal distribution function
    var albedo = material.albedo;
    if (material.useAlbedoTexture > 0.5)
    {
        albedo = textureSample(albedoTexture, materialSampler, input.uv).rgb;
    }

    var alphap = material.roughness;
    if (material.usePerlinRoughness > 0.5)
    {
        let perlinRoughness = fbmPerlin2D(input.uv * 5.0, material.perlinFreq, 0.5, 4, 2.0, 0.5);
        alphap = clamp(perlinRoughness * 0.5 + 0.5, 0.0, 1.0);
    }
    if (material.useRoughnessTexture > 0.5)
    {
        alphap = textureSample(roughnessTexture, materialSampler, input.uv).g; // green channel for roughness
    }
    alphap = max(alphap, 0.001);

    var metalness = material.metalness;
    if (material.usePerlinMetalness > 0.5)
    {
        // Slight offset from UVS of perlin roughness to avoid correlation
        let perlinMetalness = fbmPerlin2D(input.uv * 5.0 + vec2f(5.2, 1.3), material.perlinFreq, 0.5, 4, 2.0, 0.5);
        metalness = clamp(perlinMetalness * 0.5 + 0.5, 0.0, 1.0);
    }
    if (material.useMetalnessTexture > 0.5)
    {
        metalness = textureSample(metalnessTexture, materialSampler, input.uv).r;
    }

    let ka = 0.1;
    var n = normalize(input.normal);

    let pi = 3.14159265359;

    var totalColor = ka * albedo * (1.0 - metalness); // Prepare a small ambient term

    if (uniforms.light.enabled < 0.5)
    {
        return totalColor;
    }

    // ---- Area light: sample the center as a point ----
    let toLight = uniforms.light.center - input.position;
    let lightDistance = length(toLight);
    let wi = normalize(toLight);

    let lightNormal = normalize(uniforms.light.normalDirection);
    let cosAtLight = dot(-wi, lightNormal);
    if (cosAtLight <= 0.0)
    {
        return totalColor;
    }

    let toCamera = uniforms.cameraPosition - input.position;
    let wo = normalize(toCamera);
    let wh = normalize(wi + wo);

    // Dot products
    let NdotV = max(dot(n, wo), 0.0001);
    let NdotL = max(dot(n, wi), 0.0001);
    let NdotH = max(dot(n, wh), 0.0);
    let LdotH = max(dot(wi, wh), 0.0);

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

    // Area light radiance:
    // Power is spread over the light's area, and only the projected area contributes
    let lightArea = uniforms.light.width * uniforms.light.height;
    let geometricTerm = cosAtLight / (lightDistance * lightDistance);
    let radiance = uniforms.light.intensity * uniforms.light.color * geometricTerm * lightArea;

    totalColor = totalColor + f * radiance * NdotL;

    return totalColor;
}

// ============================== //
//      NOISE IMPLEMENTATION      //
// ============================== //
const PERM = array<i32, 256>( 
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
    129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
    49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
);

// ============================== //
fn fade(t: f32) -> f32
{
    return t * t * t *( t * (t * 6.0 - 15.0) + 10.0);
}

// ============================== //
fn lerp(t: f32, a: f32, b: f32) -> f32
{
    return a + t * (b - a);
}

// ============================== //
fn grad(hash: i32, x: f32, y: f32, z: f32) -> f32
{
    let h: i32 = hash & 15;
    var u: f32 = y;
    if (h < 8) {
        u = x;
    }
    
    var v: f32 = y;
    if (h >= 4)
    {
        if (h == 12 || h == 14) {
            v = x;
        }
        else {
            v = z;
        }
    }

    var t1 = -u;
    if ((h & 1) == 0) {
        t1 = u;
    }

    var t2 = -v;
    if ((h & 2) == 0) {
        t2 = v;
    }

    return (t1 + t2);
}

// ============================== //
fn perlinNoise2D(P: vec2f, freq: f32, amp: f32) -> f32
{
    let X: i32 = i32(floor(P.x * freq)) & 255;
    let Y: i32 = i32(floor(P.y * freq)) & 255;

    let relx: f32 = P.x * freq - floor(P.x * freq);
    let rely: f32 = P.y * freq - floor(P.y * freq);

    let u: f32 = fade(relx);
    let v: f32 = fade(rely);

    let A: i32 = PERM[X] + Y;
    let AA: i32 = PERM[A];
    let AB: i32 = PERM[A + 1];
    let B: i32 = PERM[X + 1] + Y;
    let BA: i32 = PERM[B];
    let BB: i32 = PERM[B + 1];

    return lerp(
        v, 
        lerp(u, grad(PERM[AA], relx, rely, 0.0), grad(PERM[BA], relx - 1.0, rely, 0.0)), 
        lerp(u, grad(PERM[AB], relx, rely - 1.0, 0.0), grad(PERM[BB], relx - 1.0, rely - 1.0, 0.0))
    ) * amp;
}

// ============================== //
fn perlinNoise3D(P: vec3f, freq: f32, amp: f32) -> f32
{
    let X: i32 = i32(floor(P.x * freq)) & 255;
    let Y: i32 = i32(floor(P.y * freq)) & 255;
    let Z: i32 = i32(floor(P.z * freq)) & 255;

    let relx: f32 = P.x * freq - floor(P.x * freq);
    let rely: f32 = P.y * freq - floor(P.y * freq);
    let relz: f32 = P.z * freq - floor(P.z * freq);

    let u : f32 = fade(relx);
    let v : f32 = fade(rely);
    let w : f32 = fade(relz);

    let A: i32 = PERM[X] + Y;
    let AA: i32 = PERM[A] + Z;
    let AB: i32 = PERM[A + 1] + Z;

    let B: i32 = PERM[X + 1] + Y;
    let BA: i32 = PERM[B] + Z;
    let BB: i32 = PERM[B + 1] + Z;

    return lerp(
        w, 
        lerp(
            v,
            lerp(u, grad(PERM[AA], relx, rely, relz), grad(PERM[BA], relx - 1.0, rely, relz)),
            lerp(u, grad(PERM[AB], relx, rely - 1.0, relz), grad(PERM[BB], relx - 1.0, rely - 1.0, relz))
        ),
        lerp(
            v,
            lerp(u, grad(PERM[AA + 1], relx, rely, relz - 1.0), grad(PERM[BA + 1], relx - 1.0, rely, relz - 1.0)),
            lerp(u, grad(PERM[AB + 1], relx, rely - 1.0, relz - 1.0), grad(PERM[BB + 1], relx - 1.0, rely - 1.0, relz - 1.0))
        )
    ) * amp;
}

// ============================== //
fn fbmPerlin2D(P: vec2f, base_freq: f32, base_amp: f32, octaves: i32, freq_mult: f32, amp_mult: f32) -> f32
{
    var total: f32 = 0.0;
    var f: f32 = base_freq;
    var a: f32 = base_amp;

    for (var i = 0; i < octaves; i++)
    {
        total = total + perlinNoise2D(P, f, a);
        f = f * freq_mult;
        a = a * amp_mult;
    }

    return total;
}