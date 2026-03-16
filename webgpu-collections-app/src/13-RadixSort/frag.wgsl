//================================//
struct VertexOut
{
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec3<f32>,
    @location(1) uv : vec2<f32>,
};

//================================//
@fragment
fn fs(input : VertexOut) -> @location(0) vec4<f32>
{
    // TRANSFORM QUAD INTO A CIRCLE
    let dist = length(input.uv);
    let edge = smoothstep(1.0, 0.85, dist);
    let finalColor = input.color * edge;
    return vec4<f32>(finalColor, 1.0);
}