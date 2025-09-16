import numpy as np
import torch
from pxr import UsdGeom, Gf, Vt, UsdPhysics # Adicionado UsdPhysics
import omni.kit.commands

# A classe não depende mais do GridCloner
# from isaacsim.core.cloner import GridCloner

from terrain_utils import SubTerrain, sloped_terrain, stairs_terrain, discrete_obstacles_terrain, wave_terrain, convert_heightfield_to_trimesh

class GeradorTerreno:
    def __init__(self, stage, num_envs: int, env_spacing: float, base_path: str = "/World/Terrains"):
        """
        Classe base para gerar grids de terrenos.

        Argumentos:
            stage: O stage do USD da simulação.
            num_envs (int): O número total de terrenos a serem criados.
            env_spacing (float): O espaçamento entre os centros dos terrenos.
            base_path (str): O caminho base para criar os terrenos.
        """
        self._stage = stage
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._base_path = base_path
        self._terrain_types_pool = ["sloped", "stairs", "discrete_obstacles", "wave"]

    def _gerar_terreno_unico(self, prim_path, terrain_type, difficulty, position, orientation, **kwargs):
        """
        Função interna para criar um único terreno.
        """
        terrain_width = kwargs.get("terrain_width", 8.0)
        terrain_length = kwargs.get("terrain_length", 8.0)
        horizontal_scale = kwargs.get("horizontal_scale", 0.25)
        vertical_scale = kwargs.get("vertical_scale", 0.005)

        num_rows = int(terrain_width / horizontal_scale)
        num_cols = int(terrain_length / horizontal_scale)
        
        sub_terrain = SubTerrain(width=num_rows, length=num_cols, vertical_scale=vertical_scale, horizontal_scale=horizontal_scale)

        if terrain_type == "sloped":
            min_slope, max_slope = -0.1, 0.2
            slope = min_slope + difficulty * (max_slope - min_slope)
            heightfield = sloped_terrain(sub_terrain, slope=slope).height_field_raw
        elif terrain_type == "stairs":
            min_h, max_h = 0.1, 0.3; min_w, max_w = 1.0, 0.4
            step_height = min_h + difficulty * (max_h - min_h)
            step_width = max_w + difficulty * (min_w - max_w)
            heightfield = stairs_terrain(sub_terrain, step_width=step_width, step_height=step_height).height_field_raw
        else:
            min_amp, max_amp = 0.05, 0.4; min_waves, max_waves = 1.0, 8.0
            amplitude = min_amp + difficulty * (max_amp - min_amp)
            num_waves = min_waves + difficulty * (max_waves - min_waves)
            heightfield = wave_terrain(sub_terrain, num_waves=num_waves, amplitude=amplitude).height_field_raw

        vertices, triangles = convert_heightfield_to_trimesh(heightfield, horizontal_scale=horizontal_scale, vertical_scale=vertical_scale, slope_threshold=1.5)

        mesh = UsdGeom.Mesh.Define(self._stage, prim_path)
        mesh.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(np.array(vertices, dtype=np.float32)))
        mesh.GetFaceVertexCountsAttr().Set(Vt.IntArray([3] * len(triangles)))
        mesh.GetFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(triangles.flatten().astype(np.int32)))
        
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim())
        mesh_collision_api.CreateApproximationAttr().Set("none")


        prim = self._stage.GetPrimAtPath(prim_path)
        xform = UsdGeom.Xformable(prim)
        xform.ClearXformOpOrder()
        pos_list = position.tolist()
        xform.AddTranslateOp().Set(Gf.Vec3d(pos_list[0], pos_list[1], pos_list[2]))
        
        xform.AddOrientOp().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    def gerar_grid_com_tensor(self, difficulties: torch.Tensor, **kwargs):
        """
        Gera um grid de terrenos a partir de um tensor de dificuldades.

        Args:
            difficulties (torch.Tensor): Tensor de shape (num_envs,) com a dificuldade de cada terreno.
            **kwargs: Argumentos opcionais para a geometria do terreno (ex: terrain_width).
        """
        if len(difficulties) != self._num_envs:
            raise ValueError(f"O tensor de dificuldades deve ter tamanho {self._num_envs}.")

        if self._stage.GetPrimAtPath(self._base_path):
            omni.kit.commands.execute("DeletePrims", paths=[self._base_path])

        # Calcula as posições da grade manualmente
        num_per_row = int(np.ceil(np.sqrt(self._num_envs)))
        env_positions = []
        for i in range(self._num_envs):
            row = i // num_per_row
            col = i % num_per_row
            offset_x = (num_per_row - 1) * self._env_spacing / 2.0
            num_rows_total = np.ceil(self._num_envs / num_per_row)
            offset_y = (num_rows_total - 1) * self._env_spacing / 2.0
            position = torch.tensor([col * self._env_spacing - offset_x, row * self._env_spacing - offset_y, 0.0])
            env_positions.append(position)
        
        terrain_types = np.random.choice(self._terrain_types_pool, self._num_envs).tolist()

        for i in range(self._num_envs):
            prim_path = f"{self._base_path}/terrain_{i}"
            self._gerar_terreno_unico(
                prim_path=prim_path,
                terrain_type=terrain_types[i],
                difficulty=difficulties[i].item(),
                position=env_positions[i],
                orientation=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                **kwargs
            )
