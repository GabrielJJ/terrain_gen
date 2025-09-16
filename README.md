# Geração de Terrenos no Isaac Sim

## Visão Geral

Este documento descreve o sistema de geração de terrenos para simulações no Isaac Sim, baseado nas classes `GeradorTerreno` e funções utilitárias em `terrain_utils.py`. O sistema permite criar grids de terrenos variados com diferentes tipos e dificuldades para treinamento de robôs.

## Arquitetura do Sistema

### Componentes Principais

1. **`GeradorTerreno`** (`base_terrain_gen.py`): Classe principal para geração de grids de terrenos
2. **`terrain_utils.py`**: Funções utilitárias para criação de diferentes tipos de terreno
3. **`gen_terrain_demo.py`**: Script de demonstração do sistema

## Classe GeradorTerreno

### Inicialização

```python
gerador = GeradorTerreno(
    stage=stage,           # Stage USD da simulação
    num_envs=16,          # Número de terrenos a criar
    env_spacing=7.5,      # Espaçamento entre terrenos (metros)
    base_path="/World/Terrains"  # Caminho base no stage
)
```

### Parâmetros

- **`stage`**: Objeto stage do USD onde os terrenos serão criados
- **`num_envs`**: Número total de terrenos no grid
- **`env_spacing`**: Distância entre centros dos terrenos
- **`base_path`**: Caminho hierárquico no stage USD

### Método Principal: `gerar_grid_com_tensor()`

```python
dificuldades = torch.linspace(0.1, 1.0, num_envs)
gerador.gerar_grid_com_tensor(dificuldades, 
    terrain_width=8.0,    # Largura do terreno (metros)
    terrain_length=8.0,   # Comprimento do terreno (metros)
    horizontal_scale=0.25, # Resolução horizontal (metros/pixel)
    vertical_scale=0.005   # Escala vertical (metros/unidade)
)
```

## Tipos de Terreno Disponíveis

### 1. Terreno Inclinado (`sloped`)
- **Descrição**: Terreno com inclinação constante
- **Parâmetros de Dificuldade**:
  - `min_slope = -0.1`, `max_slope = 0.2`
  - Dificuldade 0.0 → inclinação mínima
  - Dificuldade 1.0 → inclinação máxima

### 2. Escadas (`stairs`)
- **Descrição**: Terreno com degraus uniformes
- **Parâmetros de Dificuldade**:
  - Altura do degrau: `0.1m` a `0.3m`
  - Largura do degrau: `1.0m` a `0.4m`
  - Dificuldade maior = degraus mais altos e estreitos

### 3. Obstáculos Discretos (`discrete_obstacles`)
- **Descrição**: Terreno com obstáculos retangulares aleatórios
- **Características**:
  - Obstáculos de alturas variadas
  - Plataforma central plana
  - Distribuição aleatória

### 4. Ondas (`wave`)
- **Descrição**: Terreno com padrão ondulado
- **Parâmetros de Dificuldade**:
  - Amplitude: `0.05m` a `0.4m`
  - Número de ondas: `1.0` a `8.0`
  - Dificuldade maior = ondas mais pronunciadas

## Funções Utilitárias (terrain_utils.py)

### SubTerrain
Classe base para representar um sub-terreno:
```python
sub_terrain = SubTerrain(
    width=256,           # Largura em pixels
    length=256,          # Comprimento em pixels
    vertical_scale=0.005, # Escala vertical
    horizontal_scale=0.25 # Escala horizontal
)
```

### Funções de Geração

1. **`sloped_terrain()`**: Cria terreno inclinado
2. **`stairs_terrain()`**: Cria terreno com escadas
3. **`discrete_obstacles_terrain()`**: Cria terreno com obstáculos
4. **`wave_terrain()`**: Cria terreno ondulado
5. **`convert_heightfield_to_trimesh()`**: Converte heightfield para mesh triangular

### Funções Adicionais Disponíveis

- `random_uniform_terrain()`: Terreno com ruído uniforme
- `pyramid_sloped_terrain()`: Terreno inclinado em pirâmide
- `pyramid_stairs_terrain()`: Escadas em pirâmide
- `stepping_stones_terrain()`: Terreno com pedras para pular

## Exemplo de Uso Básico

```python
import torch
from isaacsim.core.utils.stage import get_current_stage
from base_terrain_gen import GeradorTerreno

# Configuração
stage = get_current_stage()
num_envs = 9
env_spacing = 10.0

# Criar gerador
gerador = GeradorTerreno(
    stage=stage, 
    num_envs=num_envs, 
    env_spacing=env_spacing
)

# Gerar terrenos com dificuldades crescentes
dificuldades = torch.linspace(0.0, 1.0, num_envs)
gerador.gerar_grid_com_tensor(dificuldades)

# Regenerar com dificuldades aleatórias
novas_dificuldades = torch.rand(num_envs)
gerador.gerar_grid_com_tensor(novas_dificuldades)
```

## Considerações Técnicas

### Performance
- Terrenos são gerados como meshes triangulares
- Colisão configurada com `UsdPhysics.MeshCollisionAPI`
- Aproximação de colisão configurada como "none" para precisão

### Limitações
- Terrenos são estáticos (não mudam durante a simulação)
- Regeneração requer recriação completa dos meshes
- Memória cresce linearmente com número de ambientes

## Troubleshooting

### Problemas Comuns
1. **Terrenos não aparecem**: Verificar se o stage está correto
2. **Colisões não funcionam**: Verificar aplicação de `UsdPhysics.CollisionAPI`
3. **Performance baixa**: Reduzir resolução ou número de ambientes

