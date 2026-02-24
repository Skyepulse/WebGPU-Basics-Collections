struct VSOut 
{
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@group(0) @binding(0) var displayTex: texture_2d<f32>;


@fragment
fn fs(in: VSOut) -> @location(0) vec4f 
{
    let coord = vec2i(in.pos.xy);
    let color = textureLoad(displayTex, coord, 0).rgb;
    return vec4f(color, 1.0);
}