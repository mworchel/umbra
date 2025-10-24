#version 450

uniform float screen_width;
uniform float screen_height;

uniform float point_size;

in VertexData
{
    vec3 position;
    vec3 normal;
    vec3 position_mv;
    vec3 normal_mv;
    vec3 color;
} fs_in;  

out vec4 color;

void main() {
    vec2 uv     = fs_in.position.xy;
    vec2 origin = fs_in.normal.xy;

    vec2 diff = uv - origin;
    diff.x *= screen_width;
    diff.y *= screen_height;
    if (diff.x*diff.x + diff.y*diff.y > point_size*point_size)
        discard;

    color = vec4(fs_in.color, 1.0);
}