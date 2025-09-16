import omni
from omni.isaac.kit import SimulationApp
import torch
import numpy as np

simulation_app = SimulationApp({"headless": False})

# Imports do Isaac Sim
from isaacsim.core.api.world import World
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdLux

# >>> Importa a nova classe base <<<
from base_terrain_gen import GeradorTerreno

# --- Configurações da Simulação ---
NUM_ENVS = 16
ENV_SPACING = 7.50
# ----------------------------------

if __name__ == "__main__":
    
    world = World(
        stage_units_in_meters=1.0, 
        rendering_dt=1.0/60.0,
        backend="torch", 
        device="cpu",
    )
    
    stage = get_current_stage()
    distantLight = UsdLux.DistantLight.Define(stage, "/World/DistantLight")
    distantLight.CreateIntensityAttr(3000)

    # 1. Instancia o gerador de terrenos
    gerador = GeradorTerreno(stage=stage, num_envs=NUM_ENVS, env_spacing=ENV_SPACING)

    # 2. Cria um tensor de dificuldades (ex: dificuldades crescentes)
    dificuldades_iniciais = torch.linspace(0.1, 1.0, NUM_ENVS)

    # 3. Chama o método para gerar o grid de terrenos
    gerador.gerar_grid_com_tensor(dificuldades_iniciais)

    # Loop da simulação
    last_reset_time = 0.0
    while simulation_app.is_running():
        world.step(render=True)

        if world.is_playing():
            current_time = world.current_time
            if current_time - last_reset_time > 5.0: # Regenera a cada 5 segundos
                print("--- Regenerando terrenos com novas dificuldades ---")
                
                # Gera um novo tensor de dificuldades aleatórias
                novas_dificuldades = torch.rand(NUM_ENVS)
                
                # Chama o mesmo método para apagar os terrenos antigos e criar novos
                gerador.gerar_grid_com_tensor(novas_dificuldades)
                
                last_reset_time = current_time

    simulation_app.close()
