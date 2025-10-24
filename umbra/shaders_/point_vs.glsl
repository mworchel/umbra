#version 450

uniform mat4 model_view_matrix;
uniform mat4 projection_matrix;

uniform float point_size;

in vec3 position;
in vec3 normal;
in vec3 color;

out VertexData
{
    vec3 position;
    vec3 normal;
    vec3 position_mv;
    vec3 normal_mv;
    vec3 color;
} vs_out;

void main() {
    gl_PointSize = point_size;

    gl_Position = projection_matrix * model_view_matrix * vec4(position, 1);

    vs_out.color = color;
}