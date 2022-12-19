import glfw
from queue import Queue
import moderngl
import numpy as np
from OpenGL.GL import GL_TEXTURE_2D
from pyrr import Matrix44
import threading

from .camera import PerspectiveCamera
from .controller import OrbitControl
from .primitives import Quad, CoordinateSystem
from .shaders import *

class MeshViewer:
    def __init__(self, width=600, height=600, name="OpenGL Window", device='cuda:0'):
        self.width = width
        self.height = height
        self.name = name
        self.command_queue = Queue()
        self.clear_color = [1, 1, 1]

        self.drag_point_left = None
        self.drag_point_right = None

        self.user_mouse_scroll_callback = None
        self.user_key_callback = None

        self.device = device

        # Buffers and vertex array objects by name
        self.buffers_all = {}
        self.vaos_all = {}

        self.render_thread = threading.Thread(target=self.run)
        self.render_thread.start()

    def run(self):
        self.create_window()
        self.is_open = True

        # Create the camera and its controller
        self.viewport = (0, 0, self.width, self.height)
        self.context.viewport = self.viewport
        self.camera = PerspectiveCamera(self.viewport)
        self.camera_controller = OrbitControl(self.camera)
        self.model_matrix = np.eye(4)
        self.inverse_model_matrix = np.eye(4)

        self.coordinate_system = CoordinateSystem(self.context)

        # Create the default program for triangle mesh rendering
        self.mesh_program_name = 'face'
        self.mesh_programs = {
            'face': self.context.program(vertex_shader=mesh_vertex_shader, fragment_shader=fragment_shader_color_face),
            'smooth': self.context.program(vertex_shader=mesh_vertex_shader, fragment_shader=fragment_shader_color_smooth),
            'normal': self.context.program(vertex_shader=mesh_vertex_shader, fragment_shader=fragment_shader_normal),
        }
        self.mesh_program = self.mesh_programs['face']

        while not glfw.window_should_close(self.window):
            glfw.poll_events()
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
            model_view_matrix = self.model_matrix @ self.camera.view_matrix
            self.mesh_program['model_view_matrix'].write(model_view_matrix.astype('f4'))
            self.mesh_program['projection_matrix'].write(self.camera.projection_matrix.astype('f4'))

            for _, v in self.vaos_all.items():
                v.render()

            # Render the coordinate system 
            self.coordinate_system.render(self.context, self.camera)

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

        glfw.set_cursor_pos_callback(self.window, self.mouse_event_callback)
        glfw.set_mouse_button_callback(self.window, self.mouse_button_callback)
        glfw.set_scroll_callback(self.window, self.mouse_scroll_callback)
        glfw.set_framebuffer_size_callback(self.window, self.window_resize_callback)
        glfw.set_key_callback(self.window, self.key_callback)

    def mouse_event_callback(self, window, xpos, ypos):
        if self.drag_point_left:
            self.camera_controller.handle_drag(self.drag_point_left[0], xpos, self.drag_point_left[1], ypos, 0)
            self.drag_point_left = (xpos, ypos)
        elif self.drag_point_right:
            self.camera_controller.handle_drag(self.drag_point_right[0], xpos, self.drag_point_right[1], ypos, 1)
            self.drag_point_right = (xpos, ypos)

    def mouse_button_callback(self, window, button, action, mods):
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
        self.camera_controller.handle_scroll(x_offset, y_offset)

        if self.user_mouse_scroll_callback:
            self.user_mouse_scroll_callback(x_offset, y_offset)

    def window_resize_callback(self, window, width, height):
        if width > 0 and height > 0:
            self.viewport = (0, 0, width, height)
            self.context.viewport = self.viewport
            self.camera.viewport = self.viewport

    def key_callback(self, window, key, scancode, action, mods):
        if self.user_key_callback:
            self.user_key_callback(key, scancode, action, mods)

    def set_mesh(self, v, f, n=None, c=None, object_name='default'):
        self.__enqueue_command(lambda: self.__set_mesh(v, f, n, c, object_name))

    def __set_mesh(self, v, f, n, c, object_name):
        v_flat = v.ravel().astype('f4')
        if c is None:
            c = 0.85*np.ones((v.shape[0], 3), dtype='f4')
        c = np.asarray(c)
        if len(c.shape) == 1 and c.shape[0] == 3:
            c = np.tile(c[None, :], (len(v), 1))
        c_flat = c.ravel().astype('f4')
        f_flat = f.ravel().astype('i4')

        if not object_name in self.buffers_all:
            self.buffers_all[object_name] = {}
        buffers = self.buffers_all[object_name]

        if n is not None:
            n_flat = n.ravel().astype('f4')
            buffers['vnbo'] = self.context.buffer(n_flat)
        else:
            buffers['vnbo'] = None

        buffers['vbo'] = self.context.buffer(v_flat)
        buffers['vcbo'] = self.context.buffer(c_flat)
        buffers['ibo'] = self.context.buffer(f_flat)
        self.__update_vao(object_name)


    def set_model_matrix(self, model_matrix):
        self.__enqueue_command(lambda: self.__set_model_matrix(model_matrix))

    def __set_model_matrix(self, model_matrix):
        self.model_matrix = Matrix44(model_matrix.T)
        self.inverse_model_matrix = self.model_matrix.inverse

    def set_shading_mode(self, mode, object_name=None):
        self.__enqueue_command(lambda: self.__set_shading_mode(mode))

    def __set_shading_mode(self, mode, object_name):
        self.mesh_program_name = mode
        self.mesh_program = self.mesh_programs[self.mesh_program_name]

        if object_name is None:
            # Update all objects
            for k in list(self.vaos_all.keys()):
                self.__update_vao(k)
        else:
            self.__update_vao(object_name)

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

    def __update_vao(self, object_name):
        assert object_name in self.buffers_all

        buffers = self.buffers_all[object_name]

        content = [(buffers['vbo'], '3f', 'position')]

        if 'vnbo' in buffers and self.mesh_program.get('normal', None):
            content += [(buffers['vnbo'], '3f', 'normal')]

        if 'vcbo' in buffers and self.mesh_program.get('color', None):  
            content += [(buffers['vcbo'], '3f', 'color')]

        # We control the 'in_vert' and `in_color' variables
        self.vaos_all[object_name] = self.context.vertex_array(
            self.mesh_program,
            # [
            #     # Map in_vert to the first 2 floats
            #     # Map in_color to the next 3 floats
            #     #(self.vbo, '2f 3f', 'in_vert', 'in_color'),
            #     (self.vbo, '3f', 'position'),
            #     #(self.vnbo, '3f', 'normal'),
            #     (self.vcbo, '3f', 'color'),
            # ],
            content,
            index_buffer=buffers['ibo'],
            index_element_size=4
        )
