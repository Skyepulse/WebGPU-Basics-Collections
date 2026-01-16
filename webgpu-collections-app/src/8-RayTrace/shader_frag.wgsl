// ============================== //
struct Uniform
{
    pad: vec4f,
};

// ============================== //
struct Ray 
{
    origin: vec3f,
    direction: vec3f,
};

// ============================== //
struct VertexOutput 
{
    @builtin(position) position: vec4f,
    @location(0) uv: vec2f,
};

// Scene geometry buffers (not needed in vertex shader)
// @group(0) @binding(0) var<uniform> uniforms: Uniform;
// @group(0) @binding(1) var<storage, read> vertices: array<vec3f>;
// @group(0) @binding(2) var<storage, read> normals: array<vec3f>;
// @group(0) @binding(3) var<storage, read> indices: array<u32>;

// ============================== //
fn ray_at(screen_coord: vec2f) -> Ray 
{
    var ray: Ray;
    
    ray.origin = vec3f(screen_coord.x, screen_coord.y, 1.0);
    ray.direction = vec3f(0.0, 0.0, -1.0);
    
    return ray;
}

// ============================== //
@fragment
fn fs(input: VertexOutput) -> @location(0) vec4f
{
    let ray = ray_at(input.uv);
    return vec4f(input.uv, 0.0, 1.0); // Red X, Green Y
}