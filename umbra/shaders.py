mesh_vertex_shader = '''
                #version 450
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

                uniform mat4 model_view_matrix;
                uniform mat4 projection_matrix;

                void main() {
                    // Handle position
                    vs_out.position = position;
                    vs_out.position_mv = (model_view_matrix * vec4(position, 1)).xyz;

                    // Handle normal
                    mat4 normal_model_view_matrix = transpose(inverse(model_view_matrix));
                    vs_out.normal = normalize(normal);
                    vs_out.normal_mv = normalize((normal_model_view_matrix * vec4(normal, 1)).xyz);

                    // Handle color
                    vs_out.color = color;

                    gl_Position = projection_matrix * vec4(vs_out.position_mv, 1);
                }
'''

mesh_wireframe_geometry_shader = '''
#version 450
layout (triangles) in;
layout (line_strip, max_vertices = 4) out;

in VertexData
{
    vec3 position;
    vec3 normal;
    vec3 position_mv;
    vec3 normal_mv;
    vec3 color;
} gs_in [];

out VertexData
{
    vec3 position;
    vec3 normal;
    vec3 position_mv;
    vec3 normal_mv;
    vec3 color;
} gs_out;

void main() {
    vec4 z_scale = vec4(1.0, 1.0, 0.9999, 1.0);

    for (int i = 0; i < 4; ++i)
    {
        int j = i % 3;

        gs_out.position    = gs_in[j].position;
        gs_out.normal      = gs_in[j].normal;
        gs_out.position_mv = gs_in[j].position_mv;
        gs_out.normal_mv   = gs_in[j].normal_mv;
        gs_out.color       = 0.5*gs_in[j].color;

        gl_Position = gl_in[j].gl_Position * z_scale; 

        EmitVertex();
    }

    EndPrimitive();
}  
'''

fragment_shader_color_smooth = '''
                #version 450

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
                    color = vec4(fs_in.color * fs_in.normal_mv.z, 1.0);
                }
'''

fragment_shader_color_face = '''
                #version 450

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
                    vec3 tangent_x = dFdx( fs_in.position_mv );
                    vec3 tangent_y = dFdy( fs_in.position_mv );
                    vec3 normal_mv = normalize( cross( tangent_x, tangent_y ) );

                    color = vec4(fs_in.color * normal_mv.z, 1.0);
                }
'''

fragment_shader_normal = '''
                #version 450

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
                    color = vec4(0.5*(fs_in.normal_mv + 1), 1.0);
                }
'''

fragment_shader_flat = '''
                #version 450

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
'''

fragment_shader_position_normal = '''
                #version 450

                in VertexData
                {
                    vec3 position;
                    vec3 normal;
                    vec3 position_mv;
                    vec3 normal_mv;
                    vec3 color;
                } fs_in;  

                out vec3 position;
                out vec3 normal;

                void main() {
                    position = fs_in.position;
                    normal = fs_in.normal;
                }
'''