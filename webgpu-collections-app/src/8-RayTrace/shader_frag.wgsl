// ============================== //
struct Uniform
{
    pixelToRayMatrix: mat4x4<f32>,
    cameraPosition: vec4f, // w unused
    lightPosition: vec4f,  // w unused
    lightColor: vec4f,     // w unused
    mode: u32,
    lightIntensity: f32,
    _pad1: u32,
    _pad2: u32,
};

// MODE FOLLOWS:
// 0 - normal shading
// 1 - normals
// 2 - distance
// 3 - reflectance debug

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
@group(0) @binding(3) var<storage, read> colors: array<f32>;
@group(0) @binding(4) var<storage, read> reflectances: array<f32>;
@group(0) @binding(5) var<storage, read> indices: array<u32>;

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
fn getColor(index: u32) -> vec3f 
{
    let i = index * 3u;
    return vec3f(colors[i], colors[i + 1u], colors[i + 2u]);
}

// ============================== //
fn getReflectance(index: u32) -> f32 
{
    return reflectances[index]; // one f32 per vertex
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
    if (abs(det) < kEpsilon) 
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
    
    let color0 = getColor(indices[i * 3u + 0u]);
    let color1 = getColor(indices[i * 3u + 1u]);
    let color2 = getColor(indices[i * 3u + 2u]);
    
    return color0 * w + color1 * bary.y + color2 * bary.z;
}

// ============================== //
fn getHitReflectance(hit: Hit) -> f32
{
    // suffice to take the first vertex's reflectance
    let i = hit.triIndex;
    return getReflectance(indices[i * 3u + 0u]);
}

// ============================== //
fn getHitPosition(ray: Ray, distance: f32) -> vec3f
{
    return ray.origin + ray.direction * distance;
}

// ============================== //
// Reflect around the normal (snell's law)
fn reflectRay(direction: vec3f, normal: vec3f) -> vec3f
{
    return direction - 2.0 * dot(direction, normal) * normal;
}

// ============================== //
fn computeLambertShading(hitPos: vec3f, normal: vec3f, baseColor: vec3f) -> vec3f
{
    let lightDir = normalize(uniforms.lightPosition.xyz - hitPos);
    let lightDistance = length(uniforms.lightPosition.xyz - hitPos);
    
    let NdotL = max(0.0, dot(normal, lightDir));
    let lightAttenuation = uniforms.lightIntensity / (1.0 + lightDistance * 0.01);
    
    return baseColor * uniforms.lightColor.xyz * NdotL * lightAttenuation;
}

// ============================== //
// Multi-bounce ray tracing with color bleeding
fn rayTraceWithBounces(initialRay: Ray, maxBounces: u32) -> vec3f
{
    var currentRay = initialRay;
    let reflectanceEpsilon: f32 = 0.01; // Stop is accumulated reflectance is below this

    // Trace primary ray
    var hit: Hit;
    if (!rayTraceOnce(currentRay, &hit))
    {
        return vec3f(0.0, 0.0, 0.0); // No hit, return black
    }
    
    // we touch something
    var primaryHit: Hit = hit;
    var objectColor = getHitColor(hit); // primary color of the first hit object
    var objectReflectance = getHitReflectance(primaryHit);
    let primaryHitPos = getHitPosition(currentRay, primaryHit.distance);
    
    // if reflectance is already below epsilon, just do lambert shading and return
    if (objectReflectance < reflectanceEpsilon)
    {
        return computeLambertShading(primaryHitPos, primaryHit.normalAtHit, objectColor);
    }
    
    // Compute the non-reflected portion (Lambert shaded) for primary hit
    // Only applies if reflectance < 1.0
    var primaryShadedColor = vec3f(0.0, 0.0, 0.0);
    if (objectReflectance < 1.0)
    {
        primaryShadedColor = computeLambertShading(primaryHitPos, primaryHit.normalAtHit, objectColor);
    }
    
    // Start bouncing rays to gather reflected color
    var accumulatedReflectance: f32 = objectReflectance;
    var reflectedColor = vec3f(0.0, 0.0, 0.0);
    
    // Setup first bounce ray
    let reflectedDir = reflectRay(currentRay.direction, primaryHit.normalAtHit);
    currentRay.origin = primaryHitPos + primaryHit.normalAtHit * 0.001; // self intersection avoidance offset
    currentRay.direction = reflectedDir;
    
    for (var bounce: u32 = 0u; bounce < maxBounces; bounce = bounce + 1u)
    {
        var bounceHit: Hit;
        if (!rayTraceOnce(currentRay, &bounceHit))
        {
            // Ray escaped to background - no more color to gather
            break;
        }
        
        let bounceHitPos = getHitPosition(currentRay, bounceHit.distance);
        let bounceHitColor = getHitColor(bounceHit);
        let bounceHitNormal = bounceHit.normalAtHit;
        let bounceReflectance = getHitReflectance(bounceHit);
        
        // Still compute some sort of distance attenuation
        let distanceAttenuation = 1.0;
        
        // Compute shaded color at this bounce point (if not a perfect mirror)
        var bounceShadedColor = vec3f(0.0, 0.0, 0.0);
        if (bounceReflectance < 1.0)
        {
            bounceShadedColor = computeLambertShading(bounceHitPos, bounceHitNormal, bounceHitColor);
        }
        
        // The contribution from this surface:
        // (1 - bounceReflectance) portion comes from its own shaded color
        let colorContribution = (1.0 - bounceReflectance) * bounceShadedColor * distanceAttenuation;
        reflectedColor = reflectedColor + colorContribution * accumulatedReflectance;
        accumulatedReflectance = accumulatedReflectance * bounceReflectance;
        
        if (accumulatedReflectance < reflectanceEpsilon) // too low of a possible next contribution
        {
            break;
        }
        
        let newReflectedDir = reflectRay(currentRay.direction, bounceHitNormal);
        currentRay.origin = bounceHitPos + bounceHitNormal * 0.001;
        currentRay.direction = newReflectedDir;
    }
    
    let finalColor = (1.0 - objectReflectance) * primaryShadedColor + reflectedColor;
    
    return finalColor;
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
        let maxBounces: u32 = 1u;
        color = rayTraceWithBounces(ray, maxBounces);
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
        if (rayTraceOnce(ray, &hit))
        {
            let distance = hit.distance;
            let intensity = 1.0 - min(distance / maxDistance, 1.0);
            let reflectance = getHitReflectance(hit);

            // A reflectance of 1.0 shows up as full red, 0.0 as totally white
            let totallyWhite = vec3f(1.0, 1.0, 1.0);
            let totallyRed = vec3f(1.0, 0.0, 0.0);

            let reflectanceColor = mix(totallyWhite, totallyRed, reflectance);
            color = reflectanceColor * intensity;
        }
        else
        {
            color = vec3f(0.0, 0.0, 0.0);
        }
    }
    
    return vec4f(color, 1.0);
}