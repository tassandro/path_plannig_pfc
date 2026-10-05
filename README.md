# Planejamento de Trajetória para Robô Terrestre Usando Grafos

Projeto de Final de Curso (PFC 2026) — Engenharia da Computação, Instituto Militar de Engenharia.
Autores: Gabriel Dias Vilela e Tassandro Cavalcante Leitão. Orientação: Claudia Marcela Justel, D.Sc., e Paulo Fernando Ferreira Rosa, Ph.D.

Continuação do PFC 2025 (Almeida e Barbosa; código e documentos em [`2025_version/`](2025_version/)). O sistema parte de uma imagem aérea RGB capturada por UAV, identifica as áreas transitáveis por segmentação semântica, constrói um grafo sobre o mapa de navegabilidade, calcula o caminho com BFS, DFS, Dijkstra, A* ou RRT e envia a trajetória, via ROS 2, a um UGV (Clearpath Jackal) que a percorre no simulador Gazebo.

```
Imagem RGB (UAV) → data_ingestion → preprocessing → graph_parsing → algorithm → post_processing → ros_bridge → ugv_bridge → Jackal (Gazebo)
                                   (segmentação U-Net)               (BFS/DFS/Dijkstra/A*/RRT)        (ROS 2)     (seguidor de caminho)
```

## Estrutura do repositório

| Caminho | Conteúdo |
|---|---|
| `data_ingestion/` | Entidades de entrada: imagens (`ImageData`) e nuvens de pontos (`PointCloudData`) |
| `preprocessing/` | Segmentação semântica (U-Net pré-treinadas, prefixo `drone_`), limiarização HSV e modelos alternativos |
| `graph_parsing/` | Conversão do mapa em grafo e função de peso por transformada de distância |
| `algorithm/` | BFS, DFS, Dijkstra, A* e RRT, sob a interface `PathAlgorithm` |
| `post_processing/` | Geração de *waypoints* e desenho dos caminhos sobre a imagem |
| `ros_bridge/` | Publicação da imagem anotada e do caminho em tópicos ROS 2 (opcional) |
| `ros2_ws/src/ugv_bridge/` | Pacote ROS 2 do lado do robô: `path_follower_node`, `tf_odom_bridge_node`, `image_monitor_node` e *launch files* |
| `ros2_ws/JACKAL_SIMULATION_GUIDE.md` | Guia detalhado da simulação do Jackal no Gazebo |
| `scripts/` | `download_drone_models.py` (modelos de segmentação) e `gazebo_world_to_occupancy.py` (mapa de ocupação de um mundo do Gazebo) |
| `main.py` / `gui.py` | Linha de comando e interface gráfica (Gradio) do *pipeline* |

Cada módulo segue a mesma organização: `interfaces.py` (classes abstratas), `entities.py` (*dataclasses*), `index.py` (fábrica de registro) e `__init__.py`. As pastas `models/`, `data/` e `output/` não são versionadas.

## Instalação (pipeline de planejamento)

Requer Python 3.10 ou superior (Windows, Linux ou macOS). A segmentação usa CUDA ou MPS quando disponíveis e, caso contrário, a CPU.

```bash
git clone https://github.com/tassandro/path_plannig_pfc.git
cd path_plannig_pfc
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/download_drone_models.py                # os 5 modelos U-Net
python scripts/download_drone_models.py mobilenet_v2   # ou apenas um
```

Os modelos vêm de [fabioluiz1987/semantic_segmentation_models](https://github.com/fabioluiz1987/semantic_segmentation_models) e ficam em `models/<Backbone>/`. As imagens aéreas dos ensaios (Outeiro da Glória) não estão no repositório; qualquer imagem RGB nadiral pode ser colocada em `data/rgb/`.

## Uso

### Linha de comando

```bash
# Imagem aérea real: segmentação U-Net (MobileNet V2) e comparação A* x RRT
python main.py data/rgb/imagem_aerea.jpg \
    --preprocessor drone_mobilenet_v2 \
    --algorithm a_star rrt --start 850,1200 --end 2900,1600 --save-mask

# Mapa de navegabilidade já pronto (cor transitável fixa, em BGR)
python main.py data/rgb/mapa.png --preprocessor none --bgr 84,1,68 \
    --algorithm bfs dfs dijkstra a_star rrt --start 60,60 --end 340,340
```

A imagem anotada é salva em `output/<data_hora>.png` e as métricas (vértices, custo, vértices explorados, tempo e distância) são impressas no terminal. Principais parâmetros:

| Parâmetro | Significado |
|---|---|
| `--preprocessor` | `drone_resnet50`, `drone_resnet101`, `drone_efficientnet-b4`, `drone_efficientnet-b5`, `drone_mobilenet_v2`, `hsv_threshold` (padrão) ou `none` |
| `--algorithm` | Um ou mais entre `bfs`, `dfs`, `dijkstra`, `a_star`, `rrt` |
| `--start`, `--end` | Pixels `x,y` de origem e destino (precisam estar em área transitável) |
| `--robot-size` | Folga mínima aos obstáculos, em pixels (padrão: 3) |
| `--weight-limit` | Raio do campo de penalização por proximidade, em pixels (padrão: 100) |
| `--no-weighted` | Desativa a penalização de proximidade |
| `--rrt-step-size`, `--rrt-max-iter`, `--rrt-seed` | Parâmetros do RRT |
| `--publish-ros` | Publica imagem e caminho no ROS 2 |
| `--resolution-m-per-px`, `--origin-x-m`, `--origin-y-m` | GSD e origem para converter pixels em metros |

`python main.py --help` lista todas as opções.

### Interface gráfica

```bash
python gui.py      # abrir http://localhost:7860
```

A origem e o destino são escolhidos com dois cliques sobre a imagem.

## Simulação (ROS 2 + Gazebo)

Requer Ubuntu 22.04 (nativo ou WSL2), ROS 2 Humble, `ros-humble-ros-gz`, `ros-humble-cv-bridge` e o [`clearpath_simulator`](https://github.com/clearpathrobotics/clearpath_simulator) (ramo `humble`). O passo a passo completo — instalação dos *workspaces*, `robot.yaml`, compilação do `ugv_bridge` e os terminais do ciclo completo — está em [`ros2_ws/JACKAL_SIMULATION_GUIDE.md`](ros2_ws/JACKAL_SIMULATION_GUIDE.md) e no Apêndice A do relatório final. Resumo:

1. Servidor do Gazebo *headless* com o mundo `warehouse`;
2. Ponte do `/clock` (`ros_gz_bridge`);
3. Criação do Jackal (`clearpath_gz robot_spawn.launch.py`) e ativação dos controladores;
4. `ros2 launch ugv_bridge jackal_bridge.launch.py launch_jackal_sim:=false cmd_vel_topic:=j100_0000/cmd_vel odom_topic:=j100_0000/platform/odom/filtered use_tf_odom_bridge:=false`;
5. `python3 main.py ... --publish-ros --resolution-m-per-px <GSD> --origin-x-m <x0> --origin-y-m <y0>`.

Tópicos ROS 2 envolvidos:

| Publicador | Tópico (tipo) | Assinante |
|---|---|---|
| `ros_bridge` (planejamento) | `/path_planning/path` (`nav_msgs/Path`) | `path_follower_node` |
| `ros_bridge` (planejamento) | `/path_planning/annotated_image` (`sensor_msgs/Image`) | `image_monitor_node` |
| `path_follower_node` | `j100_0000/cmd_vel` (`geometry_msgs/Twist`) | controlador das rodas do Jackal |
| Jackal (EKF rodas + IMU) | `j100_0000/platform/odom/filtered` (`nav_msgs/Odometry`) | `path_follower_node` |

O caminho é publicado com QoS *transient local*: a ponte aguarda os assinantes e permanece ativa por alguns segundos depois de publicar, já que o processo de planejamento termina logo em seguida.

Para planejar sobre a geometria exata do mundo simulado, gere antes o mapa de ocupação; o *script* imprime a resolução e a origem a passar ao `main.py`:

```bash
python scripts/gazebo_world_to_occupancy.py \
    "$(ros2 pkg prefix clearpath_gz)/share/clearpath_gz/worlds/warehouse.sdf" \
    --resolution 0.05 --output data/rgb/warehouse_occupancy.png
```

No ambiente do venv usado com `--publish-ros`, crie-o com `--system-site-packages` e fixe `pip install "numpy<2"` (o `cv_bridge` do Humble é compilado contra NumPy 1).

## Problemas conhecidos

| Sintoma | Solução |
|---|---|
| `Pixel (x, y) is not a walkable node` | Escolher origem/destino em área transitável (inspecionar com `--save-mask`) ou reduzir `--robot-size` |
| `AttributeError: _ARRAY_API not found` ao publicar | `pip install "numpy<2"` |
| GUI do Gazebo/RViz preta ou travando no WSL2 | Rodar o servidor *headless* e acompanhar pela odometria ou pelo vídeo 2D |
| Caminho recebido, mas o robô não anda | Ativar os controladores (`controller_manager spawner`) |
| Segunda publicação do caminho não chega ao seguidor | Reiniciar o *launch* do `ugv_bridge` antes de publicar de novo |
| Trajetória com forma certa, mas deslocada ou fora de escala | Conferir GSD e origem (`--resolution-m-per-px`, `--origin-x-m`, `--origin-y-m`) |
| Compilação lenta ou falhando no WSL2 | Manter os *workspaces* fora de `/mnt/c` |

## Perspectivas futuras

- Validação em ambiente real, com UGV físico e imagem aérea da área de ensaio;
- Erro lateral entre trajetória planejada e executada para todos os algoritmos;
- Ambientes dinâmicos, com replanejamento (RRT e variantes, ou busca incremental em grafos);
- **Seleção automática e alternância entre algoritmos**: escolher o planejador a partir de características da imagem de entrada (área transitável, largura de corredores, tamanho do grafo, distância origem–destino) e alternar entre algoritmos ao longo da trajetória ou no replanejamento — implementável como um novo `PathAlgorithm` que delega aos existentes;
- **Planejamento sobre nuvens de pontos** (LiDAR ou fotogrametria), incorporando elevação e declividade à transitabilidade e aos pesos — a interface `PointCloudIngestedData` já existe em `data_ingestion/`, faltando o *parser* de grafo correspondente;
- Caracterização estatística do RRT (várias sementes), suavização de trajetória e geração implícita das arestas para reduzir memória.
