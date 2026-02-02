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
    _pad0: f32,

    lights: array<SpotLight, 3>, // 48 * 3 = 144 bytes
}; // Total: 224 bytes

struct Material
{
    albedo: vec3f,
    metalness: f32,
    roughness: f32,
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
    accumulatedColor: vec3f,
};

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
@group(0) @binding(5) var<storage, read> materialIndices: array<u32>; // which material for each triangle

@group(1) @binding(0) var<storage, read> materials: array<f32>;

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
fn getMaterial(TriIndex: u32) -> Material
{
    let materialIndex = materialIndices[TriIndex]; // Which material are we talking about ?
    let baseIndex = materialIndex * 5u; // Each material uses 5 floats

    var mat: Material;
    mat.albedo = vec3f(
        materials[baseIndex],
        materials[baseIndex + 1u],
        materials[baseIndex + 2u]
    );
    mat.metalness = materials[baseIndex + 3u];
    mat.roughness = materials[baseIndex + 4u];
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
fn rayTraceOnce(ray: Ray, hit: ptr<function, Hit>) -> bool
{
    let numTriangles: u32 = u32(arrayLength(&indices)) / 3u;

    var closestT: f32 = 1e30;
    var hitSomething: bool = false;

    for (var i: u32 = 0u; i < numTriangles; i = i + 1u)
    {
        var barycentricCoords: vec3f;
        if (rayTriangleIntersect(ray, i, &barycentricCoords))
        {
            let t = barycentricCoords.x;
            if (t < closestT)
            {
                closestT = t;
                hitSomething = true;

                let normal0 = getNormal(indices[i * 3u + 0u]);
                let normal1 = getNormal(indices[i * 3u + 1u]);
                let normal2 = getNormal(indices[i * 3u + 2u]);
                let interpolatedNormal = normalize(normal0 * (1.0 - barycentricCoords.y - barycentricCoords.z) + normal1 * barycentricCoords.y + normal2 * barycentricCoords.z);
            
                (*hit).triIndex = i;
                (*hit).barycentricCoords = barycentricCoords;
                (*hit).distance = t;
                (*hit).normalAtHit = interpolatedNormal;
            }
        }
    }

    return hitSomething;
}

// ============================== //
fn getHitColor(hit: Hit) -> vec3f
{
    let i = hit.triIndex;
    let bary = hit.barycentricCoords;
    let w = 1.0 - bary.y - bary.z;
    
    let material = getMaterial(i);
    let color = material.albedo;
    
    return color;
}

// ============================== //
fn getHitPosition(ray: Ray, distance: f32) -> vec3f
{
    return ray.origin + ray.direction * distance;
}

// ============================== //
fn computeMicrofacetBRDF(hitPos: vec3f, normal: vec3f, material: Material) -> vec3f
{
    let albedo = material.albedo;
    let alphap = max(material.roughness, 0.04);
    let metalness = material.metalness;

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
        let inShadow = rayTraceOnce(shadowRay, &shadowHit);
    
        // If in shadow (and we find blocker)
        if (inShadow && shadowHit.distance < lightDistance)
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
        let inShadow = rayTraceOnce(shadowRay, &shadowHit);
    
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

    var hit: Hit;
    var color: vec3f = vec3f(0.0, 0.0, 0.0);

    if (uniforms.mode == 0u)
    {
        if (rayTraceOnce(ray, &hit))
        {
            let hitPos = getHitPosition(ray, hit.distance);
            let material = getMaterial(hit.triIndex);
            //color = computeLambertShading(hitPos, hit.normalAtHit, material.albedo);
            color = computeMicrofacetBRDF(hitPos, hit.normalAtHit, material);
        }
        else
        {
            color = vec3f(0.0, 0.0, 0.0);
        }
    }
    else if (uniforms.mode == 1u)
    {
        if (rayTraceOnce(ray, &hit))
        {
            let normal = hit.normalAtHit;
            color = normal * 0.5 + vec3f(0.5, 0.5, 0.5); // map from [-1,1] to [0,1]
        }
        else
        {
            color = vec3f(0.0, 0.0, 0.0);
        }
    }
    else if (uniforms.mode == 2u)
    {
        if (rayTraceOnce(ray, &hit))
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
    else if (uniforms.mode == 3u)
    {
        // visualize ray directions
        let dir = normalize(ray.direction);
        color = dir;
    }
    
    return vec4f(color, 1.0);
}

