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
    color = vec4(fs_in.color, 1.0);
}