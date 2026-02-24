struct VSOut 
{
    @builtin(position) pos: vec4f,
    @location(0) uv: vec2f,
};

@vertex
fn vs(@builtin(vertex_index) i: u32) -> VSOut 
{
    var pos = array<vec2f, 6>(
        vec2f(-1, -1), vec2f(1, -1), vec2f(-1, 1),
        vec2f(-1, 1), vec2f(1, -1), vec2f(1, 1)
    );
    var out: VSOut;
    out.pos = vec4f(pos[i], 0, 1);
    out.uv = pos[i] * 0.5 + 0.5;
    return out;
}