import imgui
from imgui.integrations.glfw import GlfwRenderer
import glfw
from queue import Queue
import OpenGL.GL
import moderngl
import numpy as np
import threading

from .camera import PerspectiveCamera
from .controller import OrbitControl
from .primitives import Quad, CoordinateSystem
from .shaders import *
from .shaders_ import *
from .utils import to_opengl_matrix

class MeshViewer:
    def __init__(self, width=600, height=600, name="OpenGL Window"):
        self.width = width
        self.height = height
        self.name = name
        self.command_queue = Queue()
        self.clear_color = [1, 1, 1]

        self.drag_point_left = None
        self.drag_point_right = None

        self.user_mouse_scroll_callback = None
        self.user_mouse_drag_callback = None
        self.user_mouse_button_callback = None
        self.user_key_callback = None
        self.user_gui_callback = None

        # Buffers and vertex array objects by name
        self.objects = {}

        self.is_open = True

        self.render_thread = threading.Thread(target=self.run)
        self.render_thread.start()

    def run(self):
        self.create_window()

        # Initialize imgui
        imgui.create_context()
        self.imgui_renderer = GlfwRenderer(self.window, attach_callbacks=False)

        # Create the camera and its controller
        self.viewport = (0, 0, self.width, self.height)
        self.context.viewport = self.viewport
        self.camera = PerspectiveCamera(self.viewport)
        self.camera_controller = OrbitControl(self.camera)

        # The model matrix exists for legacy reasons.
        # (it transforms *all* objects in the scene)
        self.model_matrix = np.eye(4)
        self.inverse_model_matrix = np.eye(4)

        self.coordinate_system = CoordinateSystem(self.context)

        # Create the default program for triangle mesh rendering
        self.program_name_default = 'face'
        self.programs_default = {
            'face': self.context.program(vertex_shader=mesh_vertex_shader, fragment_shader=fragment_shader_color_face),
            'smooth': self.context.program(vertex_shader=mesh_vertex_shader, fragment_shader=fragment_shader_color_smooth),
            'normal': self.context.program(vertex_shader=mesh_vertex_shader, fragment_shader=fragment_shader_normal),
            'flat': self.context.program(vertex_shader=mesh_vertex_shader, fragment_shader=fragment_shader_flat),
            'wireframe': self.context.program(vertex_shader=mesh_vertex_shader, geometry_shader=mesh_wireframe_geometry_shader, fragment_shader=fragment_shader_flat),
        }

        # # Legacy program for square points
        # self.program_points           = self.context.program(vertex_shader=point_vs, fragment_shader=point_fs)
        self.program_points_instanced = self.context.program(vertex_shader=point_instanced_vs, fragment_shader=point_fs)

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.imgui_renderer.process_inputs()
            glfw.make_context_current(self.window)

            # Execute all queued commands
            while not self.command_queue.empty():
                try:
                    command = self.command_queue.get_nowait()
                    command()
                except RuntimeError as e:
                    print(e)

            self.context.enable(moderngl.DEPTH_TEST | moderngl.CULL_FACE)
            self.context.clear(*self.clear_color)

            # Update shader data
            model_view_matrix = self.camera.view_matrix @ self.model_matrix

            general_uniforms = {
                'model_view_matrix': to_opengl_matrix(model_view_matrix),
                'projection_matrix': to_opengl_matrix(self.camera.projection_matrix),
                'screen_width': self.viewport[2] - self.viewport[0],
                'screen_height': self.viewport[3] - self.viewport[1]
            }

            for _, obj in self.objects.items():
                for vao in obj['vaos']:
                    for name, value in general_uniforms.items():
                        if name in vao.program:
                            if isinstance(value, np.ndarray):
                                vao.program[name].write(value)
                            else:
                                vao.program[name] = value

                    # TODO: Check implications for shared programs (between meshes)
                    for name, value in obj.get('uniforms', {}).items():
                        vao.program[name] = value

                    vao.render(**obj['render_args'])

            # Render the coordinate system 
            self.coordinate_system.render(self.context, self.camera)

            # Render the GUI
            imgui.new_frame()

            if self.user_gui_callback:
                # TODO: Implement error handling!
                try:
                    self.user_gui_callback(self)
                except RuntimeError as e:
                    print(f"Exception in GUI callback: {e}")

            imgui.render()
            self.imgui_renderer.render(imgui.get_draw_data())

            glfw.swap_buffers(self.window)
    
        glfw.make_context_current(self.window)

        glfw.destroy_window(self.window)
        #glfw.terminate()

        self.is_open = False

    def create_window(self):
        if not glfw.init():
            return
            
        glfw.window_hint(glfw.SRGB_CAPABLE, 1)
        glfw.window_hint(glfw.FLOATING, 1)
    
        self.window = glfw.create_window(self.width, self.height, self.name, None, None)
    
        if not self.window:
            raise RuntimeError("Unable to create window.")
    
        glfw.make_context_current(self.window)
        self.context = moderngl.create_context()

        OpenGL.GL.glEnable(OpenGL.GL.GL_FRAMEBUFFER_SRGB)
        OpenGL.GL.glEnable(OpenGL.GL.GL_PROGRAM_POINT_SIZE)

        glfw.set_cursor_pos_callback(self.window, self.mouse_event_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.window, self.mouse_scroll_callback)
        glfw.set_framebuffer_size_callback(self.window, self.framebuffer_resize_callback)
        glfw.set_window_size_callback(self.window, self.window_resize_callback)
        glfw.set_key_callback(self.window, self.key_callback)
        glfw.set_char_callback(self.window, self.char_callback)

    def mouse_event_callback(self, window, xpos, ypos):
        self.imgui_renderer.mouse_callback(window, xpos, ypos)

        # See: https://github.com/ocornut/imgui/blob/master/docs/FAQ.md#q-how-can-i-tell-whether-to-dispatch-mousekeyboard-to-dear-imgui-or-my-application
        if imgui.get_io().want_capture_mouse:
            return

        if self.drag_point_left:
            if self.user_mouse_drag_callback:
                self.user_mouse_drag_callback(self.drag_point_left[0], xpos, self.drag_point_left[1], ypos, 0)
            self.camera_controller.handle_drag(self.drag_point_left[0], xpos, self.drag_point_left[1], ypos, 0)
            self.drag_point_left = (xpos, ypos)
        elif self.drag_point_right:
            if self.user_mouse_drag_callback:
                self.user_mouse_drag_callback(self.drag_point_right[0], xpos, self.drag_point_right[1], ypos, 1)
            self.camera_controller.handle_drag(self.drag_point_right[0], xpos, self.drag_point_right[1], ypos, 1)
            self.drag_point_right = (xpos, ypos)

    def mouse_button_callback(self, window, button, action, mods):
        if imgui.get_io().want_capture_mouse:
            return

        if self.user_mouse_button_callback:
            self.user_mouse_button_callback(button, action, mods)

        # Detect drag start/end event
        if action == glfw.PRESS:
            xpos, ypos = glfw.get_cursor_pos(window)

            if button == glfw.MOUSE_BUTTON_LEFT:
                self.drag_point_left = (xpos, ypos)
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.drag_point_right = (xpos, ypos)
        elif action == glfw.RELEASE:
            if button == glfw.MOUSE_BUTTON_LEFT:
                self.drag_point_left = None
            elif button == glfw.MOUSE_BUTTON_RIGHT:
                self.drag_point_right = None

    def mouse_scroll_callback(self, window, x_offset: float, y_offset: float):
        self.imgui_renderer.scroll_callback(window, x_offset, y_offset)

        if imgui.get_io().want_capture_mouse:
            return

        self.camera_controller.handle_scroll(x_offset, y_offset)

        if self.user_mouse_scroll_callback:
            self.user_mouse_scroll_callback(x_offset, y_offset)

    def framebuffer_resize_callback(self, window, width, height):
        if width > 0 and height > 0:
            self.viewport = (0, 0, width, height)
            self.context.viewport = self.viewport
            self.camera.viewport = self.viewport

    def window_resize_callback(self, window, width, height):
        self.imgui_renderer.resize_callback(window, width, height)

    def key_callback(self, window, key, scancode, action, mods):
        self.imgui_renderer.keyboard_callback(window, key, scancode, action, mods)

        if imgui.get_io().want_capture_keyboard:
            return

        if self.user_key_callback:
            self.user_key_callback(key, scancode, action, mods)

    def char_callback(self, window, char):
        self.imgui_renderer.char_callback(window, char)

    def __expand_colors(self, vertices, colors):
        if colors is None:
            colors = 0.85*np.ones((vertices.shape[0], 3), dtype=np.float32)
            
        colors = np.asarray(colors)

        if len(colors.shape) == 1 and colors.shape[0] == 3:
            colors = np.tile(colors[None, :], (len(vertices), 1))

        return colors

    def clear(self):
        self.__enqueue_command(lambda: self.__clear())

    def __clear(self):
        self.objects = {}

    def set_mesh(self, v, f, n=None, c=None, object_name='default'):
        self.__enqueue_command(lambda: self.__set_mesh(v, f, n, c, object_name))

    def __get_or_create_object(self, name: str, expected_type: str):
        # Obtain the object information (if not existing, create it)
        if not name in self.objects:
            self.objects[name] = {'type': expected_type, 'buffers': {}, 'vaos': [], 'render_args': {}, 'uniforms': {}}

        obj = self.objects[name]

        # The type of the object must be preserved
        if obj['type'] != expected_type:
            raise RuntimeError(f"Entity '{name}' has type '{obj['type']}' and not of type '{expected_type}'.")
        
        return obj

    def __set_mesh(self, v, f, n, c, object_name):
        v_flat = v.ravel().astype('f4')
        c_flat = self.__expand_colors(v, c).ravel().astype('f4')
        f_flat = f.ravel().astype('i4')

        obj = self.__get_or_create_object(object_name, expected_type='mesh')

        # Fill buffers for this object
        if n is not None:
            n_flat = n.ravel().astype('f4')
            obj['buffers']['vnbo'] = self.context.buffer(n_flat)
        elif 'vnbo' in obj['buffers']:
            del obj['buffers']['vnbo']

        obj['buffers']['vbo'] = self.context.buffer(v_flat)
        obj['buffers']['vcbo'] = self.context.buffer(c_flat)
        obj['buffers']['ibo'] = self.context.buffer(f_flat)
        
        obj['render_args']['mode'] = moderngl.TRIANGLES

        if len(obj['vaos']) > 0:
            # Discard the existing VAOs and recreate them (buffers may have changed)
            for i in range(len(obj['vaos'])):
                obj['vaos'][i] = self.__create_mesh_vao(obj['buffers'], obj['vaos'][i].program)
        else:
            # Create a default VAO with default material
            obj['vaos'] = [self.__create_mesh_vao(obj['buffers'], self.programs_default[self.program_name_default])]

    def set_points(self, v, n=None, c=None, point_size=5, object_name='default'):
        self.__enqueue_command(lambda: self.__set_points(v, n, c, point_size, object_name))
    
    def __set_points(self, v, n=None, c=None, point_size=5, object_name='default'):
        v_flat = v.ravel().astype(np.float32)
        c_flat = self.__expand_colors(v, c).ravel().astype(np.float32)
        
        # Obtain the object information (if not existing, create them)
        obj = self.__get_or_create_object(object_name, expected_type='points')

        # Fill buffers for this object
        if n is not None:
            n_flat = n.ravel().astype('f4')
            obj['buffers']['vnbo'] = self.context.buffer(n_flat)
        elif 'vnbo' in obj['buffers']:
            del obj['buffers']['vnbo']

        obj['buffers']['vbo']  = self.context.buffer(v_flat)
        obj['buffers']['vcbo'] = self.context.buffer(c_flat)

        obj['uniforms'] = { 'point_size': point_size }

        # # Legacy:
        # obj['render_args']['mode'] = moderngl.POINTS
        # obj['vaos'] = [self.__create_point_vao(obj['buffers'], self.program_points, per_instance=False)]

        obj['render_args']['mode'] = moderngl.TRIANGLE_STRIP
        obj['render_args']['vertices'] = 4
        obj['render_args']['instances'] = v.shape[0]
        obj['vaos'] = [self.__create_point_vao(obj['buffers'], self.program_points_instanced, per_instance=True)]

    def set_lines(self, start: np.ndarray, end: np.ndarray, c=None, object_name='default'):
        self.__enqueue_command(lambda: self.__set_lines(start, end, c, object_name))
    
    def __set_lines(self, start: np.ndarray, end: np.ndarray, c=None, object_name='default'):
        # Interleave the start and end tensors
        v = np.empty((start.shape[0]+end.shape[0], start.shape[1]), dtype=start.dtype)
        v[0::2, :] = start
        v[1::2, :] = end
        v_flat     = v.ravel().astype(np.float32)
        c_flat     = self.__expand_colors(start, c).repeat(2, axis=0).ravel().astype(np.float32)
        
        # Obtain the object information (if not existing, create them)
        obj = self.__get_or_create_object(object_name, expected_type='lines')

        # Fill buffers for this object
        obj['buffers']['vbo']  = self.context.buffer(v_flat)
        obj['buffers']['vcbo'] = self.context.buffer(c_flat)

        obj['render_args']['mode'] = moderngl.LINES
        obj['vaos'] = [self.__create_line_vao(obj['buffers'], self.programs_default['flat'])]

    def remove_object(self, object_name):
        self.__enqueue_command(lambda: self.__remove_object(object_name))
        
    def __remove_object(self, object_name):
        assert object_name in self.objects
        self.objects.pop(object_name, None)

    def set_model_matrix(self, model_matrix):
        self.__enqueue_command(lambda: self.__set_model_matrix(model_matrix))

    def __set_model_matrix(self, model_matrix):
        self.model_matrix = model_matrix
        self.inverse_model_matrix = np.linalg.inv(self.model_matrix)

    def set_material(self, material, index=0, object_name='default'):
        self.__enqueue_command(lambda: self.__set_material(material, index, object_name))

    def __set_material(self, material, index, object_name):
        if not material in self.programs_default:
            raise RuntimeError(f"Material '{material}' is not a valid material name.")
        material = self.programs_default[material]
        
        
        obj = self.objects[object_name]

        if obj['type'] != 'mesh':
            raise RuntimeError(f"Materials can only be set for mesh objects (object '{object_name}' is of type '{obj['type']}').")

        vao = self.__create_mesh_vao(obj['buffers'], material)
        if index >= len(obj['vaos']):
            obj['vaos'].append(vao)
        else:
            obj['vaos'][index] = vao

    def remove_material(self, index=0, object_name='default'):
        self.__enqueue_command(lambda: self.__remove_material(index, object_name))

    def __remove_material(self, index: int, object_name: str):
        obj = self.objects[object_name]

        if obj['type'] != 'mesh':
            raise RuntimeError(f"Materials can only be removed from mesh objects (object '{object_name}' is of type '{obj['type']}').")

        vaos = obj['vaos']

        if len(vaos) == 0:
            return

        if index >= len(vaos):
            vaos.pop()
        else:
            vaos.pop(index)

    def __enqueue_command(self, command, wait=False):
        if not wait:
            self.command_queue.put(command)
        else:
            event = threading.Event()
            def execute_and_set():
                command()
                event.set()
            self.command_queue.put(execute_and_set)
            event.wait()

    def __create_content_for_program(self, buffers, program, per_instance: bool = False):
        # [
        #     # Map in_vert to the first 2 floats
        #     # Map in_color to the next 3 floats
        #     #(self.vbo, '2f 3f', 'in_vert', 'in_color'),
        #     (self.vbo, '3f', 'position'),
        #     #(self.vnbo, '3f', 'normal'),
        #     (self.vcbo, '3f', 'color'),
        # ],

        usage = '/v' if not per_instance else '/i' 

        content = [(buffers['vbo'], f'3f {usage}', 'position')]

        if 'vnbo' in buffers and program.get('normal', None):
            content += [(buffers['vnbo'], f'3f {usage}', 'normal')]

        if 'vcbo' in buffers and program.get('color', None):
            content += [(buffers['vcbo'], f'3f {usage}', 'color')]
        
        return content

    def __create_mesh_vao(self, buffers, program):
        return self.context.vertex_array(
            program,
            self.__create_content_for_program(buffers, program),
            index_buffer=buffers['ibo'],
            index_element_size=4
        )
    
    def __create_point_vao(self, buffers, program, per_instance: bool):
        return self.context.vertex_array(
            program,
            self.__create_content_for_program(buffers, program, per_instance=per_instance)
        )

    def __create_line_vao(self, buffers, program):
        return self.context.vertex_array(
            program,
            self.__create_content_for_program(buffers, program)
        )
