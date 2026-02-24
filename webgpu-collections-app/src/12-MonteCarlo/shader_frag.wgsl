// ============================== //
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

// ============================== //
struct Uniform
{
    pixelToRayMatrix: mat4x4<f32>, // 64 bytes

    cameraPosition: vec3f, // 16
    mode: u32, 

    a_c: f32,   // 16
    a_l: f32,
    a_q: f32,
    bvhVisualizationDepth: f32,

    ptDepth: u32, // 16
    frameSeed: u32,
    numSamples: u32,
    roulette: f32,

    canvasDimensions: vec2f, // 16
    frameAccumulation: f32,  
    frameCount: u32,

    light: AreaLight, // 64 bytes
}; // Total: 64 + 16 + 16 + 16 + 16 + 64 = 192 bytes

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

};

// ============================== //
struct VertexOutput 
{
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

@group(0) @binding(0) var<uniform> uniforms: Uniform;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> normals: array<f32>;
@group(0) @binding(3) var<storage, read> uvs: array<f32>;
@group(0) @binding(4) var<storage, read> indices: array<u32>;
@group(0) @binding(5) var<storage, read> bvhNodes: array<BVHNode>;
@group(0) @binding(6) var<storage, read> meshInstances: array<MeshInstance>;
@group(0) @binding(7) var accumTexture: texture_2d<f32>;

@group(1) @binding(0) var<storage, read> materials: array<f32>;
@group(1) @binding(1) var materialSampler: sampler;
@group(1) @binding(2) var textures: texture_2d_array<f32>;

var<private> rngState: u32;

// ============================== //
fn initRNG(pixel: vec2f, sampleIndex: u32) 
{
    rngState = u32(pixel.x) * 1973u 
             + u32(pixel.y) * 9277u 
             + sampleIndex * 26699u 
             + uniforms.frameSeed;
}

// ============================== //
fn rand() -> f32 
{
    rngState = rngState * 747796405u + 2891336453u;
    let word = ((rngState >> ((rngState >> 28u) + 4u)) ^ rngState) * 277803737u;
    rngState = (word >> 22u) ^ word;
    return f32(rngState) / 4294967295.0;
}

// ============================== //
// TODO: better sampling strategy, and pass seed in Uniform
fn sampleHemisphereUniform(normal: vec3f) -> vec3f 
{
    let r1 = rand();
    let r2 = rand();
    let pi = 3.14159265359;

    let phi = 2.0 * pi * r1;
    let cosTheta = r2;
    let sinTheta = sqrt(1.0 - cosTheta * cosTheta);

    let localDir = vec3f(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);

    var up = vec3f(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.999) {
        up = vec3f(1.0, 0.0, 0.0);
    }
    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    return normalize(tangent * localDir.x + bitangent * localDir.y + normal * localDir.z);
}

// ============================== //
fn sampleUniformCosine(normal: vec3f) -> vec3f
{
    let a = rand();
    let b = rand();
    let pi = 3.14159265359;

    let dist1 = cos(2.0 * pi * a) * sqrt(b);
    let dist2 = sin(2.0 * pi * a) * sqrt(b);
    let dist3 = sqrt(1.0 - b);

    var up = vec3f(0.0, 1.0, 0.0);
    if (abs(normal.y) > 0.999) 
    {
        up = vec3f(1.0, 0.0, 0.0);
    }

    let tangent = normalize(cross(up, normal));
    let bitangent = cross(normal, tangent);

    return normalize(tangent * dist1 + bitangent * dist2 + normal * dist3);
}

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
    let materialIndex = meshInstance.matIndex;
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

    if (abs(det) < kEpsilon) 
    {
        return false;
    }

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

    if (t < kEpsilon)
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

        if (node.count > 0u)
        {
            for (var i = 0u; i < node.count; i++)
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
        let rawDir = (inst.inverseWorldMatrix * vec4f(ray.direction, 0.0)).xyz;
        localRay.direction = normalize(rawDir);

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

    var closestT: f32 = maxDist;
    var hitSomething: bool = false;

    for (var j: u32 = 0u; j < numInstances; j = j + 1u)
    {
        let meshInstance = meshInstances[j];

        var localRay: Ray;
        localRay.origin = (meshInstance.inverseWorldMatrix * vec4f(ray.origin, 1.0)).xyz;
        let rawDir = (meshInstance.inverseWorldMatrix * vec4f(ray.direction, 0.0)).xyz;
        let dirScale = length(rawDir);
        localRay.direction = rawDir / dirScale;

        var closestTLocal = closestT * dirScale;

        if (traverseBVH(localRay, meshInstance, &closestTLocal, hit, shadow, j))
        {
            hitSomething = true;
            closestT = closestTLocal / dirScale;
            if (shadow)
            {
                return true;
            }
        }
    }

    if (hitSomething)
    {
        (*hit).distance = closestT;
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
fn reflectRay(direction: vec3f, normal: vec3f) -> vec3f
{
    return direction - 2.0 * dot(direction, normal) * normal;
}

// ============================== //
fn getMaterialProperties(material: Material, uv: vec2f) -> vec3f
{
    var roughness = material.roughness;
    if (material.usePerlinRoughness > 0.5)
    {
        let perlinRoughness = fbmPerlin2D(uv * 5.0, material.perlinFreq, 0.5, 4, 2.0, 0.5);
        roughness = clamp(perlinRoughness * 0.5 + 0.5, 0.0, 1.0);
    }
    if (material.useRoughnessTexture > 0.5 && material.textureIndex >= 0.0)
    {
        roughness = textureSampleLevel(textures, materialSampler, uv, i32(material.textureIndex) * 4 + 2, 2.0).g;
    }
    roughness = max(roughness, 0.001);

    var metalness = material.metalness;
    if (material.usePerlinMetalness > 0.5)
    {
        let perlinMetalness = fbmPerlin2D(uv * 5.0 + vec2f(5.2, 1.3), material.perlinFreq, 0.5, 4, 2.0, 0.5);
        metalness = clamp(perlinMetalness * 0.5 + 0.5, 0.0, 1.0);
    }
    if (material.useMetalnessTexture > 0.5 && material.textureIndex >= 0.0)
    {
        metalness = textureSampleLevel(textures, materialSampler, uv, i32(material.textureIndex) * 4 + 1, 2.0).r;
    }

    return vec3f(metalness, roughness, 0.0);
}

// ============================== //
fn pathTrace(initialRay: Ray, maxDepth: u32) -> vec3f
{
    var currentRay = initialRay;
    var accumulation = vec3f(0.0, 0.0, 0.0);
    var throughput = vec3f(1.0, 1.0, 1.0);
    let maxDistance = 1e30;
    let pi = 3.14159265359;

    for (var depth = 0u; depth < maxDepth; depth++)
    {
        var hit: Hit;
        if (!rayTraceOnce(currentRay, &hit, maxDistance, false))
        {
            break;
        }

        let hitPos = getHitPosition(currentRay, hit.distance);
        let normal = hit.normalAtHit;
        let mat = getMaterial(hit.instanceIndex);

        // ---- DIRECT LIGHTING ----
        let direct = computeDirectLighting(hitPos, normal, mat, hit.uvAtHit);
        accumulation += throughput * direct;

        // ---- INDIRECT BOUNCE ----
        var albedo = mat.albedo;
        if (mat.useAlbedoTexture > 0.5 && mat.textureIndex >= 0.0) 
        {
            albedo = textureSampleLevel(textures, materialSampler, hit.uvAtHit, i32(mat.textureIndex) * 4, 2.0).rgb;
        }

        var newDir = sampleUniformCosine(normal);
        throughput *= albedo;

        // RUSSIAN ROULETTE
        if (depth >= 2u && uniforms.roulette > 0.5)
        {
            let pSurvive = clamp(max(throughput.r, max(throughput.g, throughput.b)), 0.05, 0.95);
            if (rand() > pSurvive)
            {
                break;
            }
            throughput /= pSurvive;
        }

        // Launch indirect ray
        currentRay.origin = hitPos + normal * 0.001;
        currentRay.direction = newDir;
    }

    return accumulation;
}

// ============================== //
fn computeDirectLighting(hitPos: vec3f, normal: vec3f, material: Material, uv: vec2f) -> vec3f
{
    var albedo = material.albedo;
    if (material.useAlbedoTexture > 0.5 && material.textureIndex >= 0.0)
    {
        albedo = textureSampleLevel(textures, materialSampler, uv, i32(material.textureIndex) * 4, 2.0).rgb;
    }

    let matProps = getMaterialProperties(material, uv);
    let roughness = matProps.y;
    let metalness = matProps.x;
    let alphap = roughness;
    
    var n = normalize(normal);
    let pi = 3.14159265359;

    var totalColor = vec3f(0.0, 0.0, 0.0);

    if (uniforms.light.enabled < 0.5)
    {
        return totalColor;
    }

    let lightNormal = normalize(uniforms.light.normalDirection);
    var lightUp = vec3f(0.0, 0.0, 1.0);
    if (abs(lightNormal.z) > 0.999) 
    {
        lightUp = vec3f(1.0, 0.0, 0.0);
    }
    let lightTangent = normalize(cross(lightUp, lightNormal));
    let lightBitangent = cross(lightNormal, lightTangent);

    let u = rand() - 0.5;
    let v = rand() - 0.5;
    let samplePoint = uniforms.light.center 
                    + lightTangent * u * uniforms.light.width 
                    + lightBitangent * v * uniforms.light.height;

    let toLight = samplePoint - hitPos;
    let lightDistance = length(toLight);
    let wi = normalize(toLight);

    // Check: is the shading point on the emitting side?
    let cosAtLight = dot(-wi, lightNormal);
    if (cosAtLight <= 0.0)
    {
        return totalColor;
    }

    // Check: does the surface face the light?
    let NdotL = dot(n, wi);
    if (NdotL <= 0.0)
    {
        return totalColor;
    }

    // Check: is the point in shadow?
    const shadowBias = 0.001;
    var shadowRay: Ray;
    shadowRay.origin = hitPos + shadowBias * n;
    shadowRay.direction = wi;

    var shadowHit: Hit;
    let inShadow = rayTraceOnce(shadowRay, &shadowHit, lightDistance - shadowBias, true);
    if (inShadow)
    {
        return totalColor;
    }

    let toCamera = uniforms.cameraPosition - hitPos;
    let wo = normalize(toCamera);
    let wh = normalize(wi + wo);

    let NdotV = max(dot(n, wo), 0.0001);
    let NdotH = max(dot(n, wh), 0.0);
    let LdotH = max(dot(wi, wh), 0.0);

    let F0 = mix(vec3(0.04), albedo, metalness);
    let F = F0 + (1.0 - F0) * pow(1.0 - LdotH, 5.0);

    let lambert = albedo / pi;
    let kd = (1.0 - F) * (1.0 - metalness);
    let fd = kd * lambert;

    let D = (alphap * alphap) / (pi * pow((NdotH * NdotH) * (alphap * alphap - 1.0) + 1.0, 2.0));

    let K = (alphap) * sqrt(2.0 / pi);
    let G_schlick_wo = NdotV / (NdotV * (1.0 - K) + K);
    let G_schlick_wi = NdotL / (NdotL * (1.0 - K) + K);
    let G = G_schlick_wo * G_schlick_wi;

    let EPSILON = 0.0001;
    let fs = (D * F * G) / (4.0 * NdotL * NdotV + EPSILON);

    let f = fd + fs;

    // Monte carlo area light PDF estimate
    let lightArea = uniforms.light.width * uniforms.light.height;
    let geometricTerm = cosAtLight / (lightDistance * lightDistance);
    let radiance = uniforms.light.intensity * uniforms.light.color;

    totalColor = f * radiance * NdotL * geometricTerm * lightArea;

    return totalColor;
}

// ============================== //
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
        let bvhdepth = u32(uniforms.bvhVisualizationDepth - 1.0);
        let color = debugBVHTraversal(ray, bvhdepth);
        return vec4f(color, 1.0);
    }

    var hit: Hit;
    var color: vec3f = vec3f(0.0, 0.0, 0.0);

    let n = uniforms.numSamples;
    let stratSize = u32(ceil(sqrt(f32(n))));

    if (uniforms.frameAccumulation < 0.5 )
    {
        var totalColor = vec3f(0.0, 0.0, 0.0);
        var sampleIndex = 0u;
        for (var sy = 0u; sy < stratSize; sy++)
        {
            for (var sx = 0u; sx < stratSize; sx++)
            {
                if (sampleIndex >= n) { break; }

                initRNG(input.position.xy, sampleIndex);

                let stratumOffset = vec2f(
                    (f32(sx) + rand()) / f32(stratSize),
                    (f32(sy) + rand()) / f32(stratSize)
                );

                let jitteredUV = vec2f(
                    (input.position.x - 1.0 + stratumOffset.x) / uniforms.canvasDimensions.x,
                    1.0 - (input.position.y - 1.0 + stratumOffset.y) / uniforms.canvasDimensions.y
                );

                let sampleRay = ray_at(jitteredUV);
                totalColor += pathTrace(sampleRay, uniforms.ptDepth);
                sampleIndex++;
            }
        }

        color = totalColor / f32(n);
    }
    else // One sample per frame, accumulate in texture
    {
        initRNG(input.position.xy, uniforms.frameCount - 1u);
        let stratIndex = (uniforms.frameCount - 1u) % (stratSize * stratSize);
        let stratumOffset = vec2f(
            (f32(stratIndex % stratSize) + rand()) / f32(stratSize),
            (f32(stratIndex / stratSize) + rand()) / f32(stratSize)
        );
    
        let jitteredUV = vec2f(
            (input.position.x - 1.0 + stratumOffset.x) / uniforms.canvasDimensions.x,
            1.0 - (input.position.y - 1.0 + stratumOffset.y) / uniforms.canvasDimensions.y
        );
        let sampleRay = ray_at(jitteredUV);
        let newSample = pathTrace(sampleRay, uniforms.ptDepth);
        
        let oldColor = textureLoad(accumTexture, vec2i(input.position.xy), 0).rgb;
        
        let divider = f32(uniforms.frameCount);
        color = oldColor + (newSample - oldColor) / divider;
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