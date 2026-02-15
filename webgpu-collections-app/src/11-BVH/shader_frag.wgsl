// ============================== //
struct SpotLight
{
    position: vec3f,
    intensity: f32,

    direction: vec3f,
    coneAngle: f32,

    color: vec3f,
    enabled: f32,
}; // Total : 48 bytes

// ============================== //
struct Uniform
{
    pixelToRayMatrix: mat4x4<f32>, // 4 * 4 * 4 = 64 bytes

    cameraPosition: vec3f,
    mode: u32,              // 16 bytes

    a_c: f32,
    a_l: f32,
    a_q: f32,
    bvhVisualizationDepth: f32,

    lights: array<SpotLight, 3>, // 48 * 3 = 144 bytes
}; // Total: 224 bytes

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
};

// MODE FOLLOWS:
// 0 - normal shading
// 1 - normals
// 2 - distance

// ============================== //
struct Ray 
{
    origin: vec3f,
    direction: vec3f,
};

// ============================== //
struct Hit
{
    triIndex: u32,
    barycentricCoords: vec3f,
    distance: f32,
    normalAtHit: vec3f,
    uvAtHit: vec2f,
    accumulatedColor: vec3f,
    instanceIndex: u32,
    numBoxQueries: u32,
    numTriangleQueries: u32,
};

// ============================== //
struct BVHNode
{
    minB: vec3f,
    leftOrFirst: u32,
    maxB: vec3f,
    count: u32,
};

// ============================== //
struct MeshInstance
{
    inverseWorldMatrix: mat4x4<f32>,

    bvhRootIndex: u32,
    triOffset: u32,
    vertOffset: u32,
    matIndex: u32,

}; // Total: 16 * 4 + 4 * 4 = 80 bytes

// ============================== //
struct VertexOutput 
{
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniform;
@group(0) @binding(1) var<storage, read> vertices: array<f32>; // vec3f does not work here cause would expect a 16 byte alignment
@group(0) @binding(2) var<storage, read> normals: array<f32>;
@group(0) @binding(3) var<storage, read> uvs: array<f32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> bvhNodes: array<BVHNode>;
@group(0) @binding(6) var<storage, read> meshInstances: array<MeshInstance>;

@group(1) @binding(0) var<storage, read> materials: array<f32>;
@group(1) @binding(1) var materialSampler: sampler;
@group(1) @binding(2) var textures: texture_2d_array<f32>; // (albedo -> metalness -> roughness -> normal) for each material

// Helper function to read a vec3 from the flat array
// ============================== //
fn getVertex(index: u32) -> vec3f 
{
    let i = index * 3u;
    return vec3f(vertices[i], vertices[i + 1u], vertices[i + 2u]);
}

// ============================== //
fn getNormal(index: u32) -> vec3f 
{
    let i = index * 3u;
    return vec3f(normals[i], normals[i + 1u], normals[i + 2u]);
}

// ============================== //
fn getUV(index: u32) -> vec2f 
{
    let i = index * 2u;
    return vec2f(uvs[i], uvs[i + 1u]);
}

// ============================== //
fn getMaterial(instance: u32) -> Material
{
    let meshInstance = meshInstances[instance];
    let materialIndex = meshInstance.matIndex; // Which material are we talking about ?
    let baseIndex = materialIndex * 16u;

    var mat: Material;
    mat.albedo = vec3f(
        materials[baseIndex],
        materials[baseIndex + 1u],
        materials[baseIndex + 2u]
    );

    mat.metalness = materials[baseIndex + 3u];
    mat.usePerlinMetalness = materials[baseIndex + 4u];
    mat.roughness = materials[baseIndex + 5u];
    mat.usePerlinRoughness = materials[baseIndex + 6u];
    mat.perlinFreq = materials[baseIndex + 7u];

    mat.useAlbedoTexture = materials[baseIndex + 8u];
    mat.useMetalnessTexture = materials[baseIndex + 9u];
    mat.useRoughnessTexture = materials[baseIndex + 10u];
    mat.useNormalTexture = materials[baseIndex + 11u];

    mat.textureIndex = materials[baseIndex + 12u];

    return mat;
}

// ============================== //
fn rayTriangleIntersect(ray: Ray, triIndex: u32, hitCoord: ptr<function, vec3f>) -> bool
{
    // https://scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/moller-trumbore-ray-triangle-intersection.html
    let v0 = getVertex(indices[triIndex * 3u + 0u]);
    let v1 = getVertex(indices[triIndex * 3u + 1u]);
    let v2 = getVertex(indices[triIndex * 3u + 2u]);

    let e0 = v1 - v0;
    let e1 = v2 - v0;

    let pvec: vec3f = cross(ray.direction, e1);
    let det: f32 = dot(e0, pvec);

    let kEpsilon: f32 = 0.000001;

    // culling or not, we do or don't compare absolute value of det
    if (det < kEpsilon) 
    {
        return false; // No intersection
    }

    // compute u. reject if u not in [0,1]
    // then v, same check and reject if u+v > 1
    // if met, compute t to get intersection point we know there is intersection
    let invDet: f32 = 1.0 / det;
    let tvec: vec3f = ray.origin - v0;

    let u = dot(tvec, pvec) * invDet;
    if (u < 0.0 || u > 1.0) 
    {
        return false;
    }

    let qvec: vec3f = cross(tvec, e0);
    let v = dot(ray.direction, qvec) * invDet;
    if (v < 0.0 || (u + v) > 1.0) 
    {
        return false;
    }

    let t = dot(e1, qvec) * invDet;

    if (t < kEpsilon)  // behind camera
    {
        return false;
    }

    let barycentricCoords = vec3f(t, u, v);
    (*hitCoord) = barycentricCoords;

    return true;
}

// ============================== //
fn rayAABBIntersect(ray: Ray, invDir: vec3f, bMin: vec3f, bMax: vec3f, maxDist: f32) -> bool
{
    let t1 = (bMin - ray.origin) * invDir;
    let t2 = (bMax - ray.origin) * invDir;

    let tmin = max(max(min(t1.x, t2.x), min(t1.y, t2.y)), min(t1.z, t2.z));
    let tmax = min(min(max(t1.x, t2.x), max(t1.y, t2.y)), max(t1.z, t2.z));

    return tmax >= max(tmin, 0.0) && tmin < maxDist;
}

// ============================== //
fn traverseBVH(ray: Ray, inst: MeshInstance, closestT: ptr<function, f32>, hit: ptr<function, Hit>, shadow: bool, instanceIndex: u32) -> bool
{
    let invDir = vec3f(1.0 / ray.direction.x, 1.0 / ray.direction.y, 1.0 / ray.direction.z);

    var stack: array<u32, 32>;
    var stackPtr: i32 = 0;

    // Push the root
    stack[0] = inst.bvhRootIndex;
    stackPtr = 1;

    var hitAnything: bool = false;
    var numBoxQueries: u32 = 0u;
    var numTriangleQueries: u32 = 0u;

    while (stackPtr > 0)
    {
        stackPtr = stackPtr - 1;
        let nodeIndex = stack[stackPtr];
        let node: BVHNode = bvhNodes[nodeIndex];

        if (!rayAABBIntersect(ray, invDir, node.minB, node.maxB, (*closestT)))
        {
            continue;
        }

        numBoxQueries = numBoxQueries + 1u;

        if (node.count > 0u) // leaf
        {
            for (var i = 0u; i < node.count; i++) // triangles per se...
            {
                let localTriIdx = node.leftOrFirst + i;
                let globalTriIdx = localTriIdx + inst.triOffset;

                var bary: vec3f;
                if (rayTriangleIntersect(ray, globalTriIdx, &bary))
                {
                    numTriangleQueries = numTriangleQueries + 1u;
                    if (bary.x < *closestT)
                    {
                        if (shadow) { return true; }

                        *closestT = bary.x;
                        hitAnything = true;

                        let idx0 = indices[globalTriIdx * 3u + 0u];
                        let idx1 = indices[globalTriIdx * 3u + 1u];
                        let idx2 = indices[globalTriIdx * 3u + 2u];

                        let w = 1.0 - bary.y - bary.z;
                        let interpNormal = normalize(getNormal(idx0) * w + getNormal(idx1) * bary.y + getNormal(idx2) * bary.z);
                        let interpUV = getUV(idx0) * w + getUV(idx1) * bary.y + getUV(idx2) * bary.z;

                        (*hit).triIndex = globalTriIdx;
                        (*hit).barycentricCoords = bary;
                        (*hit).distance = bary.x;
                        (*hit).normalAtHit = interpNormal;
                        (*hit).uvAtHit = interpUV;
                        (*hit).instanceIndex = instanceIndex;
                        (*hit).numBoxQueries = numBoxQueries;
                        (*hit).numTriangleQueries = numTriangleQueries;
                    }
                }
            }
        }
        else
        {
            // Small optimization: push farthest node first, so we traverse closest one first
            let leftIdx = node.leftOrFirst;
            let rightIdx = node.leftOrFirst + 1u;

            let leftNode = bvhNodes[leftIdx];
            let rightNode = bvhNodes[rightIdx];

            let leftCenter = (leftNode.minB + leftNode.maxB) * 0.5;
            let rightCenter = (rightNode.minB + rightNode.maxB) * 0.5;

            let leftDist = dot(leftCenter - ray.origin, ray.direction);
            let rightDist = dot(rightCenter - ray.origin, ray.direction);

            if (leftDist < rightDist)
            {
                stack[stackPtr] = rightIdx; stackPtr += 1;
                stack[stackPtr] = leftIdx;  stackPtr += 1;
            }
            else
            {
                stack[stackPtr] = leftIdx;  stackPtr += 1;
                stack[stackPtr] = rightIdx; stackPtr += 1;
            }
        }
    }

    return hitAnything;
}

// ============================== //
fn debugBVHTraversal(ray: Ray, targetDepth: u32) -> vec3f
{
    let numInstances = arrayLength(&meshInstances);
    var hitCount: u32 = 0u;
    var leafTriCount: u32 = 0u;
    var deepestHit: u32 = 0u;

    for (var j: u32 = 0u; j < numInstances; j++)
    {
        let inst = meshInstances[j];

        var localRay: Ray;
        localRay.origin = (inst.inverseWorldMatrix * vec4f(ray.origin, 1.0)).xyz;
        localRay.direction = (inst.inverseWorldMatrix * vec4f(ray.direction, 0.0)).xyz;

        let invDir = vec3f(1.0 / localRay.direction.x, 1.0 / localRay.direction.y, 1.0 / localRay.direction.z);

        var stackNode: array<u32, 64>;
        var stackDepth: array<u32, 64>;
        var stackPtr: i32 = 0;
        stackNode[0] = inst.bvhRootIndex;
        stackDepth[0] = 0u;
        stackPtr = 1;

        while (stackPtr > 0)
        {
            stackPtr -= 1;
            let nodeIndex = stackNode[stackPtr];
            let depth = stackDepth[stackPtr];
            let node = bvhNodes[nodeIndex];

            if (!rayAABBIntersect(localRay, invDir, node.minB, node.maxB, 1e30))
            {
                continue;
            }

            if (depth > deepestHit) { deepestHit = depth; }

            if (depth == targetDepth)
            {
                hitCount += 1u;
                continue;
            }

            if (node.count > 0u)
            {
                hitCount += 1u;
                leafTriCount += node.count;
                continue;
            }

            let leftIdx = node.leftOrFirst;
            let rightIdx = node.leftOrFirst + 1u;
            stackNode[stackPtr] = rightIdx;
            stackDepth[stackPtr] = depth + 1u;
            stackPtr += 1;
            stackNode[stackPtr] = leftIdx;
            stackDepth[stackPtr] = depth + 1u;
            stackPtr += 1;
        }
    }

    if (hitCount == 0u) { return vec3f(0.0); }

    // Heatmap: blue (1 hit) → green (few) → yellow → red (many)
    let t = clamp(f32(hitCount) / 8.0, 0.0, 1.0);
    if (t < 0.5)
    {
        return mix(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), t * 2.0);
    }
    return mix(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), (t - 0.5) * 2.0);
}

// ============================== //
fn rayTraceOnce(ray: Ray, hit: ptr<function, Hit>, maxDist: f32, shadow: bool) -> bool
{
    let numInstances: u32 = u32(arrayLength(&meshInstances));

    var closestT: f32 = 1e30;
    var hitSomething: bool = false;

    for (var j: u32 = 0u; j < numInstances; j = j + 1u)
    {
        let meshInstance = meshInstances[j];

        var localRay: Ray;
        localRay.origin = (meshInstance.inverseWorldMatrix * vec4f(ray.origin, 1.0)).xyz;
        localRay.direction = (meshInstance.inverseWorldMatrix * vec4f(ray.direction, 0.0)).xyz;

        if (traverseBVH(localRay, meshInstance, &closestT, hit, shadow, j))
        {
            hitSomething = true;
            if (shadow)
            {
                return true;
            }
        }
    }

    // Get the normal back into world space
    if (hitSomething)
    {
        let inst = meshInstances[(*hit).instanceIndex];
        let normalMatrix = transpose(mat3x3f(
            inst.inverseWorldMatrix[0].xyz,
            inst.inverseWorldMatrix[1].xyz,
            inst.inverseWorldMatrix[2].xyz
        ));
        (*hit).normalAtHit = normalize(normalMatrix * (*hit).normalAtHit);
    }

    return hitSomething;
}

// ============================== //
fn getHitPosition(ray: Ray, distance: f32) -> vec3f
{
    return ray.origin + ray.direction * distance;
}

// ============================== //
fn computeMicrofacetBRDF(hitPos: vec3f, normal: vec3f, material: Material, uv: vec2f) -> vec3f
{
    var albedo = material.albedo;
    if (material.useAlbedoTexture > 0.5 && material.textureIndex >= 0.0)
    {
        albedo = textureSampleLevel(textures, materialSampler, uv, i32(material.textureIndex) * 4, 2.0).rgb;
    }

    var alphap = material.roughness;
    if (material.usePerlinRoughness > 0.5)
    {
        let perlinRoughness = fbmPerlin2D(uv * 5.0, material.perlinFreq, 0.5, 4, 2.0, 0.5);
        alphap = clamp(perlinRoughness * 0.5 + 0.5, 0.0, 1.0);
    }
    if (material.useRoughnessTexture > 0.5 && material.textureIndex >= 0.0)
    {
        alphap = textureSampleLevel(textures, materialSampler, uv, i32(material.textureIndex) * 4 + 2, 2.0).g;
    }
    alphap = max(alphap, 0.001);
    
    var metalness = material.metalness;
    if (material.usePerlinMetalness > 0.5)
    {
        // Slight offset from UVS of perlin roughness to avoid correlation
        let perlinMetalness = fbmPerlin2D(uv * 5.0 + vec2f(5.2, 1.3), material.perlinFreq, 0.5, 4, 2.0, 0.5);
        metalness = clamp(perlinMetalness * 0.5 + 0.5, 0.0, 1.0);
    }
    if (material.useMetalnessTexture > 0.5 && material.textureIndex >= 0.0)
    {
        metalness = textureSampleLevel(textures, materialSampler, uv, i32(material.textureIndex) * 4 + 1, 2.0).r;
    }
    
    let ka = 0.1;
    var n = normalize(normal);
    let pi = 3.14159265359;

    var totalColor = ka * albedo * (1.0 - metalness); // Prepare a small ambient term

    for (var i = 0; i < 3; i = i + 1)
    {
        if (uniforms.lights[i].enabled < 0.5)
        {
            continue;
        }

        let toLight = uniforms.lights[i].position - hitPos;
        let lightDistance = length(toLight);
        let wi = normalize(toLight);

        let toCamera = uniforms.cameraPosition - hitPos;
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

        // Shadow ray tracing
        const shadowBias = 0.0001;
        var shadowRay: Ray;
        shadowRay.origin = hitPos + shadowBias * normal;
        shadowRay.direction = wi;

        var shadowHit: Hit;
        let inShadow = rayTraceOnce(shadowRay, &shadowHit, lightDistance, true);
    
        // If in shadow (and we find blocker)
        if (inShadow && shadowHit.distance < lightDistance)
        {
            continue;
        }

        let fade = smoothstep(cos(uniforms.lights[i].coneAngle), cos(uniforms.lights[i].coneAngle) + 0.05, cosAngle);


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

        totalColor = totalColor + f * radiance * NdotL * fade;
    }

    return totalColor;
} 

// ============================== //
fn computeLambertShading(hitPos: vec3f, normal: vec3f, baseColor: vec3f) -> vec3f
{
    let ambientStrength = 0.1;
    let ambientColor = baseColor * ambientStrength;

    var totalColor = vec3f(0.0, 0.0, 0.0);
    for (var i = 0; i < 3; i++)
    {
        if (uniforms.lights[i].enabled < 0.5)
        {
            continue;
        }

        let pos2light = uniforms.lights[i].position - hitPos;
        let lightDistance = length(pos2light);
        let wi = normalize(pos2light);
        let spotDir = normalize(uniforms.lights[i].direction);

        // Are we in the cone
        let cosAngle = dot(-wi, spotDir);
        if (cosAngle < cos(uniforms.lights[i].coneAngle)) 
        {
            continue;
        }

        // Shadow ray tracing
        const shadowBias = 0.0001;
        var shadowRay: Ray;
        shadowRay.origin = hitPos + shadowBias * normal;
        shadowRay.direction = wi;

        var shadowHit: Hit;
        let inShadow = rayTraceOnce(shadowRay, &shadowHit, lightDistance, true);
    
        // If in shadow (and we find blocker)
        if (inShadow && shadowHit.distance < lightDistance)
        {
            continue;
        }

        // Not in shadow, compute full Lambert shading
        let NdotL = max(0.0, dot(normal, wi));
        let lightAttenuation = uniforms.lights[i].intensity / (uniforms.a_c + uniforms.a_l * lightDistance + uniforms.a_q * lightDistance * lightDistance);
        let diffuse = baseColor * uniforms.lights[i].color * NdotL * lightAttenuation;
        
        totalColor = totalColor + diffuse;
    }
   
    return ambientColor + totalColor;
}

// ============================== //
// Need to pass Screen -> NDC -> Clip -> View -> World
fn ray_at(screen_coord: vec2f) -> Ray 
{
    let ndc = vec2f(screen_coord.x * 2.0 - 1.0, screen_coord.y * 2.0 - 1.0);
    let rayDirection = normalize((uniforms.pixelToRayMatrix * vec4f(ndc, 1.0, 0.0)).xyz);
    
    var ray: Ray;
    ray.origin = uniforms.cameraPosition.xyz;
    ray.direction = rayDirection;

    return ray;
}

// ============================== //
@fragment
fn fs(input: VertexOutput) -> @location(0) vec4f
{
    let ray = ray_at(input.uv);
    let maxDistance: f32 = 2000.0;

    if (uniforms.mode == 1u)
    {
        let depth = u32(uniforms.bvhVisualizationDepth - 1.0);
        let color = debugBVHTraversal(ray, depth);
        return vec4f(color, 1.0);
    }

    var hit: Hit;
    var color: vec3f = vec3f(0.0, 0.0, 0.0);

    if (uniforms.mode == 0u)
    {
        if (rayTraceOnce(ray, &hit, maxDistance, false))
        {
            let hitPos = getHitPosition(ray, hit.distance);
            let material = getMaterial(hit.instanceIndex);
            //color = computeLambertShading(hitPos, hit.normalAtHit, material.albedo);
            color = computeMicrofacetBRDF(hitPos, hit.normalAtHit, material, hit.uvAtHit);
        }
        else
        {
            color = vec3f(0.0, 0.0, 0.0);
        }
    }
    else if (uniforms.mode == 2u)
    {
        if (rayTraceOnce(ray, &hit, maxDistance, false))
        {
            let normal = hit.normalAtHit;
            color = normal * 0.5 + vec3f(0.5, 0.5, 0.5); // map from [-1,1] to [0,1]
        }
        else
        {
            color = vec3f(0.0, 0.0, 0.0);
        }
    }
    else if (uniforms.mode == 3u)
    {
        if (rayTraceOnce(ray, &hit, maxDistance, false))
        {
            let distance = hit.distance;
            let intensity = 1.0 - min(distance / maxDistance, 1.0);
            color = vec3f(intensity, intensity, intensity);
        }
        else
        {
            color = vec3f(0.0, 0.0, 0.0);
        }
    }
    else if (uniforms.mode == 4u)
    {
        // visualize ray directions
        let dir = normalize(ray.direction);
        color = dir;
    }
    
    return vec4f(color, 1.0);
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