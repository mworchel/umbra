#version 450

uniform mat4 model_view_matrix;
uniform mat4 projection_matrix;

uniform float screen_width;
uniform float screen_height;

uniform float point_size;

in vec3 position;
in vec3 color;

out VertexData
{
    vec3 position;
    vec3 normal;
    vec3 position_mv;
    vec3 normal_mv;
    vec3 color;
} vs_out;

void main()
{
    // Generate a quad from vertex ids (avoids binding vertex data)
    // The mapping from vertex id to quad position is
    // 2 ___ 3     (0,1) ___ (1,1) 
    //  |   |           |   |
    //  |___|           |___|
    // 0     1     (0,0)     (1,0)
    vec2 uv = vec2(gl_VertexID & 1, (gl_VertexID & 2) >> 1);
    uv.x -= 0.5f;
    uv.y -= 0.5f;

    // Transform the point to ndc
    vec4 point_ndc = projection_matrix * model_view_matrix * vec4(position, 1);
    
    uv.x *= 2. / screen_width  * point_size;
    uv.y *= 2. / screen_height * point_size;

    // Shift the quad to the point position
    vec2 origin = point_ndc.xy / point_ndc.w;
    uv += origin;

    gl_Position = vec4(uv.x * point_ndc.w, uv.y * point_ndc.w, point_ndc.z, point_ndc.w);
    
    vs_out.color = color;

    // Store NDC position of the vertex and the center point (misusing the normal attribute for the latter)
    vs_out.position = vec3(uv.x, uv.y, point_ndc.w);
    vs_out.normal   = vec3(origin, point_ndc.z / point_ndc.w);
}