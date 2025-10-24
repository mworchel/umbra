
def read_shader(file_name: str) -> str:
    from pathlib import Path
    current_dir = Path(__file__).parent
    with open(current_dir / file_name, 'r') as file:
        return file.read()

point_vs      = read_shader("point_vs.glsl")
point_fs      = read_shader("point_fs.glsl")