//================================//
@group(0) @binding(0) var<storage, read> keysBuffer : array<u32>;

override ELEMENT_COUNT: u32 = 1024;
override GRID_SIZE: u32 = 32;

//================================//
struct VertexOut
{
    @builtin(position) position : vec4<f32>,
    @location(0) color : vec3<f32>,
    @location(1) uv : vec2<f32>,
};

//================================//
fn morton2D_decode(code: u32) -> vec2<u32>
{
    var x = code & 0x55555555u;
    var y = (code >> 1u) & 0x55555555u;

    x = (x | (x >> 1u)) & 0x33333333u;
    x = (x | (x >> 2u)) & 0x0F0F0F0Fu;
    x = (x | (x >> 4u)) & 0x00FF00FFu;
    x = (x | (x >> 8u)) & 0x0000FFFFu;

    y = (y | (y >> 1u)) & 0x33333333u;
    y = (y | (y >> 2u)) & 0x0F0F0F0Fu;
    y = (y | (y >> 4u)) & 0x00FF00FFu;
    y = (y | (y >> 8u)) & 0x0000FFFFu;

    return vec2<u32>(x, y);
}

//================================//
@vertex
fn vs(@builtin(vertex_index) vertexIndex : u32, @builtin(instance_index) instanceIndex : u32) -> VertexOut
{
    var output : VertexOut;

    if (instanceIndex >= ELEMENT_COUNT)
    {
        output.position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        output.color = vec3<f32>(0.0);
        output.uv = vec2<f32>(0.0);
        return output;
    }

    let cellSize = 2.0 / f32(GRID_SIZE);
    let mortonPosition = morton2D_decode(instanceIndex);

    let cellX = -1.0 + cellSize * f32(mortonPosition.x);
    let cellY = 1.0 - cellSize * f32(mortonPosition.y);

    let pad = 0.08 * cellSize;
    let drawSize = cellSize - 2.0 * pad;

    var localQuadUV = array<vec2<f32>, 6>(
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 0.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(0.0, 1.0)
    );

    let lp = localQuadUV[vertexIndex];

    let position = vec2<f32>(cellX + pad + lp.x * drawSize, cellY - pad - lp.y * drawSize);
    output.position = vec4<f32>(position, 0.0, 1.0);
    output.uv = lp * 2.0 - 1.0;

    let key = keysBuffer[instanceIndex];
    let t = f32(key) / f32(0x3FFFFFFFu);

    let colorA = vec3<f32>(0.33, 0.21, 0.73);
    let colorB = vec3<f32>(0.94, 0.62, 0.15);
    output.color = mix(colorA, colorB, t);

    return output;
}

