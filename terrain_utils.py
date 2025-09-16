import numpy as np
from numpy.random import choice
from scipy import interpolate
# Atualização de imports para IsaacSim 4.5.0
from math import sqrt
from isaacsim.core.utils.prims import define_prim, get_prim_at_path
from isaacsim.core.prims import XFormPrim
from pxr import UsdPhysics, Sdf, Gf, PhysxSchema, Usd
import omni.kit.commands

def random_uniform_terrain(terrain, min_height, max_height, step=1, downsampled_scale=None,):
    """
    Gera um terreno com ruído uniforme

    Parâmetros:
        terrain (SubTerrain): o terreno
        min_height (float): a altura mínima do terreno [metros]
        max_height (float): a altura máxima do terreno [metros]
        step (float): mudança mínima de altura entre dois pontos [metros]
        downsampled_scale (float): distância entre dois pontos amostrados aleatoriamente (deve ser maior ou igual a terrain.horizontal_scale)
    """
    if downsampled_scale is None:
        downsampled_scale = terrain.horizontal_scale

    # converte parâmetros para unidades discretas
    min_height = int(min_height / terrain.vertical_scale)
    max_height = int(max_height / terrain.vertical_scale)
    step = int(step / terrain.vertical_scale)

    heights_range = np.arange(min_height, max_height + step, step)
    height_field_downsampled = np.random.choice(heights_range, (int(terrain.width * terrain.horizontal_scale / downsampled_scale), int(
        terrain.length * terrain.horizontal_scale / downsampled_scale)))

    x = np.linspace(0, terrain.width * terrain.horizontal_scale, height_field_downsampled.shape[0])
    y = np.linspace(0, terrain.length * terrain.horizontal_scale, height_field_downsampled.shape[1])

    f = interpolate.interp2d(y, x, height_field_downsampled, kind='linear')

    x_upsampled = np.linspace(0, terrain.width * terrain.horizontal_scale, terrain.width)
    y_upsampled = np.linspace(0, terrain.length * terrain.horizontal_scale, terrain.length)
    z_upsampled = np.rint(f(y_upsampled, x_upsampled))

    terrain.height_field_raw += z_upsampled.astype(np.int16)
    return terrain

def sloped_terrain(terrain, slope=1):
    """
    Gera um terreno inclinado

    Parâmetros:
        terrain (SubTerrain): o terreno
        slope (int): inclinação positiva ou negativa
    Retorna:
        terrain (SubTerrain): terreno atualizado
    """

    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(terrain.width, 1)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * terrain.width)
    terrain.height_field_raw[:, np.arange(terrain.length)] += (max_height * xx / terrain.width).astype(terrain.height_field_raw.dtype)
    return terrain

def pyramid_sloped_terrain(terrain, slope=1, platform_size=1.):
    """
    Gera um terreno inclinado em pirâmide

    Parâmetros:
        terrain (terrain): o terreno
        slope (int): inclinação positiva ou negativa
        platform_size (float): tamanho da plataforma plana no centro do terreno [metros]
    Retorna:
        terrain (SubTerrain): terreno atualizado
    """
    x = np.arange(0, terrain.width)
    y = np.arange(0, terrain.length)
    center_x = int(terrain.width / 2)
    center_y = int(terrain.length / 2)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = (center_x - np.abs(center_x-xx)) / center_x
    yy = (center_y - np.abs(center_y-yy)) / center_y
    xx = xx.reshape(terrain.width, 1)
    yy = yy.reshape(1, terrain.length)
    max_height = int(slope * (terrain.horizontal_scale / terrain.vertical_scale) * (terrain.width / 2))
    terrain.height_field_raw += (max_height * xx * yy).astype(terrain.height_field_raw.dtype)

    platform_size = int(platform_size / terrain.horizontal_scale / 2)
    x1 = terrain.width // 2 - platform_size
    x2 = terrain.width // 2 + platform_size
    y1 = terrain.length // 2 - platform_size
    y2 = terrain.length // 2 + platform_size

    min_h = min(terrain.height_field_raw[x1, y1], 0)
    max_h = max(terrain.height_field_raw[x1, y1], 0)
    terrain.height_field_raw = np.clip(terrain.height_field_raw, min_h, max_h)
    return terrain

def discrete_obstacles_terrain(terrain, max_height, min_size, max_size, num_rects, platform_size=1.):
    """
    Gera um terreno com obstáculos

    Parâmetros:
        terrain (terrain): o terreno
        max_height (float): altura máxima dos obstáculos (intervalo=[-max, -max/2, max/2, max]) [metros]
        min_size (float): tamanho mínimo de um obstáculo retangular [metros]
        max_size (float): tamanho máximo de um obstáculo retangular [metros]
        num_rects (int): número de obstáculos gerados aleatoriamente
        platform_size (float): tamanho da plataforma plana no centro do terreno [metros]
    Retorna:
        terrain (SubTerrain): terreno atualizado
    """
    # converte parâmetros para unidades discretas
    max_height = int(max_height / terrain.vertical_scale)
    min_size = int(min_size / terrain.horizontal_scale)
    max_size = int(max_size / terrain.horizontal_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    (i, j) = terrain.height_field_raw.shape
    height_range = [-max_height, -max_height // 2, max_height // 2, max_height]
    width_range = range(min_size, max_size, 4)
    length_range = range(min_size, max_size, 4)

    for _ in range(num_rects):
        width = np.random.choice(width_range)
        length = np.random.choice(length_range)
        start_i = np.random.choice(range(0, i-width, 4))
        start_j = np.random.choice(range(0, j-length, 4))
        terrain.height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def wave_terrain(terrain, num_waves=1, amplitude=1.):
    """
    Gera um terreno ondulado

    Parâmetros:
        terrain (terrain): o terreno
        num_waves (int): número de ondas senoidais ao longo do comprimento do terreno
        amplitude (float): amplitude das ondas
    Retorna:
        terrain (SubTerrain): terreno atualizado
    """
    amplitude = int(0.5*amplitude / terrain.vertical_scale)
    if num_waves > 0:
        div = terrain.length / (num_waves * np.pi * 2)
        x = np.arange(0, terrain.width)
        y = np.arange(0, terrain.length)
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = xx.reshape(terrain.width, 1)
        yy = yy.reshape(1, terrain.length)
        terrain.height_field_raw += (amplitude*np.cos(yy / div) + amplitude*np.sin(xx / div)).astype(
            terrain.height_field_raw.dtype)
    return terrain

def stairs_terrain(terrain, step_width, step_height):
    """
    Gera um terreno com escadas

    Parâmetros:
        terrain (terrain): o terreno
        step_width (float): a largura do degrau [metros]
        step_height (float): a altura do degrau [metros]
    Retorna:
        terrain (SubTerrain): terreno atualizado
    """
    # converte parâmetros para unidades discretas
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = terrain.width // step_width
    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width: (i + 1) * step_width, :] += height
        height += step_height
    return terrain

def pyramid_stairs_terrain(terrain, step_width, step_height, platform_size=1.):
    """
    Gera um terreno com escadas em pirâmide

    Parâmetros:
        terrain (terrain): o terreno
        step_width (float): a largura do degrau [metros]
        step_height (float): a altura do degrau [metros]
        platform_size (float): tamanho da plataforma plana no centro do terreno [metros]
    Retorna:
        terrain (SubTerrain): terreno atualizado
    """
    # converte parâmetros para unidades discretas
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)

    height = 0
    start_x = 0
    stop_x = terrain.width
    start_y = 0
    stop_y = terrain.length
    while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
        start_x += step_width
        stop_x -= step_width
        start_y += step_width
        stop_y -= step_width
        height += step_height
        terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = height
    return terrain

def stepping_stones_terrain(terrain, stone_size, stone_distance, max_height, platform_size=1., depth=-10):
    """
    Gera um terreno com pedras para pisar

    Parâmetros:
        terrain (terrain): o terreno
        stone_size (float): tamanho horizontal das pedras [metros]
        stone_distance (float): distância entre as pedras (ou seja, tamanho dos buracos) [metros]
        max_height (float): altura máxima das pedras (positiva e negativa) [metros]
        platform_size (float): tamanho da plataforma plana no centro do terreno [metros]
        depth (float): profundidade dos buracos (padrão=-10.) [metros]
    Retorna:
        terrain (SubTerrain): terreno atualizado
    """
    # converte parâmetros para unidades discretas
    stone_size = int(stone_size / terrain.horizontal_scale)
    stone_distance = int(stone_distance / terrain.horizontal_scale)
    max_height = int(max_height / terrain.vertical_scale)
    platform_size = int(platform_size / terrain.horizontal_scale)
    height_range = np.arange(-max_height-1, max_height, step=1)

    start_x = 0
    start_y = 0
    terrain.height_field_raw[:, :] = int(depth / terrain.vertical_scale)
    if terrain.length >= terrain.width:
        while start_y < terrain.length:
            stop_y = min(terrain.length, start_y + stone_size)
            start_x = np.random.randint(0, stone_size)
            # preenche o primeiro buraco
            stop_x = max(0, start_x - stone_distance)
            terrain.height_field_raw[0: stop_x, start_y: stop_y] = np.random.choice(height_range)
            # preenche a linha
            while start_x < terrain.width:
                stop_x = min(terrain.width, start_x + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_x += stone_size + stone_distance
            start_y += stone_size + stone_distance
    elif terrain.width > terrain.length:
        while start_x < terrain.width:
            stop_x = min(terrain.width, start_x + stone_size)
            start_y = np.random.randint(0, stone_size)
            # preenche o primeiro buraco
            stop_y = max(0, start_y - stone_distance)
            terrain.height_field_raw[start_x: stop_x, 0: stop_y] = np.random.choice(height_range)
            # preenche a coluna
            while start_y < terrain.length:
                stop_y = min(terrain.length, start_y + stone_size)
                terrain.height_field_raw[start_x: stop_x, start_y: stop_y] = np.random.choice(height_range)
                start_y += stone_size + stone_distance
            start_x += stone_size + stone_distance

    x1 = (terrain.width - platform_size) // 2
    x2 = (terrain.width + platform_size) // 2
    y1 = (terrain.length - platform_size) // 2
    y2 = (terrain.length + platform_size) // 2
    terrain.height_field_raw[x1:x2, y1:y2] = 0
    return terrain

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Converte um array de campo de altura para uma malha triangular representada por vértices e triângulos.
    Opcionalmente, corrige superfícies verticais acima do limite de inclinação fornecido:

        Se (y2-y1)/(x2-x1) > slope_threshold -> Move A para A' (define x1 = x2). Faz isso para todas as direções.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parâmetros:
        height_field_raw (np.array): campo de altura de entrada
        horizontal_scale (float): escala horizontal do campo de altura [metros]
        vertical_scale (float): escala vertical do campo de altura [metros]
        slope_threshold (float): o limite de inclinação acima do qual as superfícies são tornadas verticais. Se None, nenhuma correção é aplicada (padrão: None)
    Retorna:
        vertices (np.array(float)): array de formato (num_vertices, 3). Cada linha representa a localização de cada vértice [metros]
        triangles (np.array(int)): array de formato (num_triangles, 3). Cada linha representa os índices dos 3 vértices conectados por este triângulo.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # cria vértices e triângulos da malha a partir da grade do campo de altura
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles
# Não é possível carregar DefinePrim
def add_terrain_to_stage(stage, vertices, triangles, position=None, orientation=None):
    num_faces = triangles.shape[0]
    terrain_mesh = stage.DefinePrim("/World/terrain", "Mesh")
    terrain_mesh.GetAttribute("points").Set(vertices)
    terrain_mesh.GetAttribute("faceVertexIndices").Set(triangles.flatten())
    terrain_mesh.GetAttribute("faceVertexCounts").Set(np.asarray([3]*num_faces))

    terrain = XFormPrim(prim_paths_expr="/World/terrain",
                        name="terrain",
                        positions=position,
                        orientations=orientation
                        )

    # Argumentos indexados de terrain.prims
    print(f"Número de prims adicionados: {len(terrain.prims)}")
    UsdPhysics.CollisionAPI.Apply(terrain.prims[0])
    collision_api = UsdPhysics.MeshCollisionAPI.Apply(terrain.prims[0])
    # collision_api.CreateApproximationAttr().Set("meshSimplification")
    collision_api.CreateApproximationAttr().Set("sdf")
    physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(terrain.prims[0])
    physx_collision_api.GetContactOffsetAttr().Set(0.02)
    physx_collision_api.GetRestOffsetAttr().Set(0.00)


class SubTerrain:
    def __init__(self, terrain_name="terrain", width=256, length=256, vertical_scale=1.0, horizontal_scale=1.0):
        self.terrain_name = terrain_name
        self.vertical_scale = vertical_scale
        self.horizontal_scale = horizontal_scale
        self.width = width
        self.length = length
        self.height_field_raw = np.zeros((self.width, self.length), dtype=np.int16)


