# Rodando o Jackal (J100) real no Gazebo — guia passo a passo

Este guia foi **testado de ponta a ponta** em 2026-08-10 (WSL2 + Ubuntu
22.04 + ROS 2 Humble), incluindo a parte mais importante: publicar um
caminho real gerado por `main.py` e ver o `path_follower_node` dirigir o
Jackal de verdade no Gazebo até o objetivo (`"Path complete — goal
reached"`).

> **Sobre visualização 3D (GUI do Gazebo/RViz)**: neste ambiente
> específico (WSL2 sem passthrough de GPU), a GUI do Gazebo trava com um
> crash do Ogre2 (`GL3PlusTextureGpu::copyTo`). Um contorno
> (`LIBGL_ALWAYS_SOFTWARE=1`) evita o crash, mas quando a janela é aberta
> por automação (terminal do agente) ela aparece **preta/travada** — o
> mesmo aconteceu com o RViz2. Isso não é um bug do `ugv_bridge`: é uma
> limitação de apresentação de janela do WSLg quando o processo GUI não é
> aberto por você mesmo, interativamente. **Este guia usa o modo headless
> (sem GUI)**, que é 100% confiável e já comprovado — a seção final lista
> como tentar a GUI por sua conta, se quiser arriscar.

## Pré-requisitos (já feitos uma vez, não repetir)

- WSL2 `Ubuntu-22.04` com ROS 2 Humble (`ros-humble-desktop`,
  `ros-humble-ros-gz`, `python3-colcon-common-extensions`) instalado via
  apt.
- `~/clearpath_ws`: workspace separado (native fs, fora de `/mnt/c`) com
  `clearpath_simulator` (branch `humble`) + `clearpath_common` +
  `clearpath_config` + `clearpath_msgs` clonados e buildados
  (`colcon build`, `--symlink-install` ok aqui).
- `~/clearpath/robot.yaml`: cópia do `clearpath_config`'s
  `sample/j100/j100_default.yaml` (namespace `j100_0000`), com
  hostname/ip ajustados para `localhost`/`127.0.0.1`.
- `~/ros2_ws/src/ugv_bridge`: symlink para
  `ros2_ws/src/ugv_bridge` deste repo (que vive em `/mnt/c/...`), buildado
  com `colcon build --packages-select ugv_bridge` (sem
  `--symlink-install`) dentro de `~/ros2_ws`.
- `~/venvs/pfc`: venv Python (`--system-site-packages`) com
  `pip install -r requirements.txt` + `pip install "numpy<2"`.

Se algum desses não existir ainda, siga as notas detalhadas no final deste
arquivo (seção "Setup inicial, se ainda não foi feito").

## Passo a passo (cada bloco = 1 terminal WSL novo)

Abra **4 terminais WSL** (`wsl -d Ubuntu-22.04`) e rode um bloco em cada
um, nesta ordem. Deixe todos abertos/rodando ao mesmo tempo.

### Terminal 1 — servidor do Gazebo (headless, mundo `warehouse`)

```bash
source /opt/ros/humble/setup.bash
source ~/clearpath_ws/install/setup.bash
CP_GZ_SHARE="$(ros2 pkg prefix clearpath_gz)/share/clearpath_gz"
EXTRA_SHARE="$(echo "$AMENT_PREFIX_PATH" | tr ':' '\n' | sed 's#$#/share#' | tr '\n' ':')"
export GZ_SIM_RESOURCE_PATH="${CP_GZ_SHARE}/worlds:${CP_GZ_SHARE}/meshes:${EXTRA_SHARE}"
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
ros2 launch ros_gz_sim gz_sim.launch.py gz_args:="warehouse.sdf -s -r -v 4"
```

Espere até ver `Loaded level [3]` e `World [warehouse] initialized`.

*(Também disponível como script: `ros2_ws/scripts_tmp/gz_server.sh`.)*

### Terminal 2 — ponte de clock (`/clock`)

```bash
source /opt/ros/humble/setup.bash
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
ros2 run ros_gz_bridge parameter_bridge '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'
```

*(Script: `ros2_ws/scripts_tmp/clock_bridge.sh`.)*

### Terminal 3 — spawn do robô Jackal + ativação dos controllers

```bash
source /opt/ros/humble/setup.bash
source ~/clearpath_ws/install/setup.bash
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
ros2 launch clearpath_gz robot_spawn.launch.py \
    rviz:=false use_sim_time:=true world:=warehouse setup_path:=$HOME/clearpath/
```

Espere até ver `OK creation of entity` (o robô foi criado no mundo).

*(Script: `ros2_ws/scripts_tmp/robot_spawn.sh`.)*

Depois que aparecer `OK creation of entity`, **em outro terminal rápido**
(pode ser um 5º terminal só para isso, ou reaproveitar depois), ative os
controllers — sem isso o robô não se move de verdade:

```bash
source /opt/ros/humble/setup.bash
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
ros2 run controller_manager spawner \
    joint_state_broadcaster platform_velocity_controller \
    --controller-manager /j100_0000/controller_manager \
    --ros-args -r __ns:=/j100_0000
```

Deve logar `Loaded ...` e `Configured and activated ...` para os dois
controllers.

**Checagem opcional** (confirma que o robô realmente anda):

```bash
ros2 topic pub --rate 10 /j100_0000/cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.3}}"
# em outro terminal, enquanto isso roda:
ros2 topic echo /j100_0000/platform/odom/filtered --field pose.pose.position --once
```

A posição `x` deve crescer com o tempo (0.3 m/s).

### Terminal 4 — nós do `ugv_bridge` (ponte com o `path_plannig_pfc`)

```bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
ros2 launch ugv_bridge jackal_bridge.launch.py \
    launch_jackal_sim:=false \
    cmd_vel_topic:=j100_0000/cmd_vel \
    odom_topic:=j100_0000/platform/odom/filtered \
    use_tf_odom_bridge:=false
```

- `launch_jackal_sim:=false` porque a simulação já está rodando (Terminais
  1–3).
- `cmd_vel_topic`/`odom_topic` apontam para os tópicos reais e namespaced
  do Jackal (`j100_0000/...`).
- `use_tf_odom_bridge:=false` porque o Jackal já publica uma
  `nav_msgs/Odometry` real (EKF), então o nó de conversão TF→Odometry não
  é necessário (e, sem esse flag, ele viraria um segundo publisher no
  mesmo tópico — bug corrigido nesta sessão, ver nota abaixo).

Deve logar `PathFollowerNode ready` e `ImageMonitorNode ready` (sem
`tf_odom_bridge_node`, que fica desabilitado).

### Terminal 5 (ou reaproveite outro já livre) — gerar e publicar o caminho

Ative o venv do projeto e rode o pipeline de planejamento normalmente,
com `--publish-ros`:

```bash
source /opt/ros/humble/setup.bash
source ~/venvs/pfc/bin/activate
export GZ_IP=127.0.0.1
export GZ_PARTITION=localhost
cd /mnt/c/Users/Admin/Documents/pfc_novo/path_plannig_pfc
python3 main.py data/rgb/test_track.png \
    --preprocessor none --bgr 84,1,68 \
    --algorithm a_star --start 60,60 --end 340,340 \
    --publish-ros --resolution-m-per-px 0.05
```

(Troque `data/rgb/test_track.png`, `--start`/`--end`/`--bgr` pela sua
imagem real e ajuste `--resolution-m-per-px` para o GSD real da sua
imagem, em metros/pixel.)

## Resultado esperado (confirmado)

No **Terminal 4**, você deve ver, em sequência:

```
[path_follower_node]: New path received: 281 waypoints
[path_follower_node]: Path complete — goal reached
```

E, enquanto isso roda, a posição do robô muda de verdade:

```bash
ros2 topic echo /j100_0000/platform/odom/filtered --field pose.pose.position --once
```

`x`/`y` devem se mover continuamente até o robô parar perto do destino
(convertendo pixels → metros pela `--resolution-m-per-px` usada).

A imagem anotada com o caminho planejado fica salva em:
- `output/<timestamp>.png` (gerada pelo `main.py`)
- `~/ugv_bridge/received_images/latest.png` (recebida pelo
  `image_monitor_node` via ROS 2 — prova que a imagem também chegou no
  "lado do robô")

## "Mas eu queria VER o robô na janela do Gazebo..."

**Testado à exaustão nesta e em sessões anteriores — conclusão: não dá
para ver a janela 3D do Gazebo neste ambiente WSL2, e não é algo que dê
para corrigir só daqui (mais detalhes e a alternativa que funciona logo
abaixo).**

1. `ign gazebo -g` (cliente de GUI, conectando no servidor headless do
   Terminal 1) trava com `Ogre::UnimplementedException` em
   `GL3PlusTextureGpu::copyTo` — falta de GPU real / driver GL parcial do
   WSL2.
2. Contornando com renderização por software (evita o crash, mas a
   janela aparece preta/travada, mesmo abrindo o app do Gazebo
   diretamente no Windows, sem nenhuma automação):

   ```bash
   source /opt/ros/humble/setup.bash
   source ~/clearpath_ws/install/setup.bash
   export GZ_IP=127.0.0.1
   export GZ_PARTITION=localhost
   export LIBGL_ALWAYS_SOFTWARE=1
   export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
   ign gazebo -g
   ```

   *(Script: `ros2_ws/scripts_tmp/gz_gui_sw.sh`.)*

3. **Causa raiz identificada**: `/mnt/wslg/stderr.log` mostra
   `Xwayland glamor: GBM Wayland interfaces not available` /
   `Failed to initialize glamor, falling back to sw` toda vez que o WSLg
   inicia — o compositor inteiro (não só o Gazebo) roda 100% em
   software, sem nenhuma aceleração de GPU. É por isso que a barra de
   título mostra "WARN: COPY MODE" (modo de apresentação degradado do
   Weston/Xwayland).
4. **`wsl --update` foi tentado e NÃO resolveu** (WSL 2.7.10.0 → 2.7.11.0,
   MSRDC também atualizado) — o problema persiste idêntico depois de
   `wsl --shutdown` + reabrir. Investigação mais profunda mostrou que não
   é falta de GPU/driver do Windows (`/dev/dxg` existe e está ativo,
   driver Intel Iris Xe recente) nem falta de atualização do Mesa (já
   está na versão mais nova disponível via apt, 23.2.1) — é uma
   limitação do suporte a GBM no driver Mesa `d3d12` usado pelo WSLg.
   Não é algo resolvível via `wsl --update`/`apt`; só restaria tentar um
   PPA de Mesa mais recente (ex. kisak-mesa), o que é mais arriscado e
   não foi feito.

Alternativas para "ver" o robô sem a GUI 3D completa:
- **Gerar um vídeo (MP4) do robô percorrendo o caminho** — ver seção
  abaixo, é a alternativa recomendada e já testada com sucesso.
- Rodar `rviz2` (mesmas variáveis de ambiente acima) você mesmo, também
  interativamente — é bem mais leve que o Gazebo e mostraria pelo menos o
  TF/robot model/path (sofre do mesmo problema de janela preta, então
  não é garantido).
- Acompanhar `ros2 topic echo .../odom/filtered` (ou plotar com
  `rqt_plot`) para ver a posição mudando em tempo real.
- Olhar a imagem anotada com o caminho (`output/*.png`), que já mostra
  visualmente o caminho planejado sobre a imagem de entrada.

## Gerando um vídeo do robô percorrendo o caminho (sem GUI)

Como a janela 3D do Gazebo não é confiável neste ambiente, use
`ros2_ws/scripts_tmp/record_path_video.py`: ele grava a odometria real
do robô enquanto ele anda e renderiza um vídeo 2D (MP4) com o robô se
movendo (seta + rastro) sobre a imagem anotada com o caminho —
totalmente independente de GPU/Xwayland, roda em CPU.

**Ordem importa**: inicie o gravador **antes** de publicar o caminho,
senão ele não recebe a mensagem do caminho (é `transient_local`, mas só
funciona para quem já está inscrito enquanto o publisher — de vida
curta — ainda está de pé) nem descobre onde é o objetivo:

```bash
# Terminal 6 — grava a trajetória e renderiza o vídeo ao final
source /opt/ros/humble/setup.bash
source ~/venvs/pfc/bin/activate
cd /mnt/c/Users/Admin/Documents/pfc_novo/path_plannig_pfc
python3 ros2_ws/scripts_tmp/record_path_video.py \
    --map-image output/<timestamp>.png \
    --output output/robot_run.mp4 \
    --resolution-m-per-px 0.05
```

Aí, no Terminal 5 (ou um novo), rode o `main.py --publish-ros` como de
costume. O gravador para sozinho ~2s depois de detectar que o robô
chegou ao objetivo (ou pressione `Ctrl+C` a qualquer momento para
renderizar o que já foi gravado até ali) e escreve o `.mp4` no caminho
de `--output`.

**Testado e funcionando (2026-08-10)**: gravou 4601 amostras de
odometria ao longo de 93.9s de um percurso real de 281 waypoints e
renderizou corretamente `output/robot_run_final.mp4` (MP4 válido).

**Cuidado com republicar o caminho num `path_follower_node` que já está
rodando há um tempo**: nos testes desta sessão, só a *primeira*
mensagem de caminho recebida por um nó recém-iniciado disparou
"New path received" de forma confiável — publicações seguintes (`main.py
--publish-ros` rodado de novo, mesmo com o `ugv_bridge` já de pé) às
vezes não chegam (bug de entrega ainda não corrigido, ver notas em
`/memories/repo/architecture.md`). Se o robô não se mover depois de um
segundo/terceiro `--publish-ros`, reinicie o `ros2 launch ugv_bridge
jackal_bridge.launch.py ...` (Terminal 4) e o gravador antes de
publicar de novo — a entrega da primeira mensagem sempre funcionou nos
testes.

## Encerrando

`Ctrl+C` em cada terminal (4 e 5, depois 3, 2, 1) ou, se preferir matar
tudo de uma vez:

```bash
pkill -9 -f 'ign gazebo'
pkill -9 -f 'ros_gz_bridge'
pkill -9 -f 'ros2 launch'
```

## Bug corrigido nesta sessão

`ros2_ws/src/ugv_bridge/launch/jackal_bridge.launch.py`: o próprio
docstring do arquivo recomendava apontar `odom_topic` direto para
`j100_0000/platform/odom/filtered` (a Odometry real do EKF) num Jackal
real, mas não havia como desligar o `tf_odom_bridge_node` — ele virava um
segundo publisher no mesmo tópico. Adicionado o argumento
`use_tf_odom_bridge` (default `true`) que desliga esse nó quando `false`.
Sempre use `use_tf_odom_bridge:=false` junto com o override de
`odom_topic` recomendado no docstring, como mostrado no Terminal 4 acima.

## Setup inicial, se ainda não foi feito

Se `~/clearpath_ws`, `~/ros2_ws` ou `~/clearpath/robot.yaml` não
existirem ainda no seu WSL, seguem os passos resumidos (veja também os
comentários no início de `jackal_bridge.launch.py`):

```bash
# 1. Workspace do clearpath_gz (native fs, NUNCA em /mnt/c)
mkdir -p ~/clearpath_ws/src && cd ~/clearpath_ws
git clone -b humble https://github.com/clearpathrobotics/clearpath_simulator src/clearpath_simulator
vcs import src < src/clearpath_simulator/dependencies.repos
sudo apt-get update
rosdep install --from-paths src --ignore-src -r -y
colcon build --symlink-install

# 2. robot.yaml (copie o sample oficial do Jackal, não escreva à mão)
mkdir -p ~/clearpath
cp ~/clearpath_ws/src/clearpath_simulator/clearpath_config/clearpath_config/sample/j100/j100_default.yaml ~/clearpath/robot.yaml
# edite hostname/ip para localhost/127.0.0.1 dentro do robot.yaml

# 3. ugv_bridge (symlink pro repo em /mnt/c, build fora do /mnt/c)
mkdir -p ~/ros2_ws/src
ln -s /mnt/c/Users/Admin/Documents/pfc_novo/path_plannig_pfc/ros2_ws/src/ugv_bridge ~/ros2_ws/src/ugv_bridge
cd ~/ros2_ws && colcon build --packages-select ugv_bridge   # sem --symlink-install

# 4. venv Python do path_plannig_pfc
python3 -m venv --system-site-packages ~/venvs/pfc
source ~/venvs/pfc/bin/activate
cd /mnt/c/Users/Admin/Documents/pfc_novo/path_plannig_pfc
pip install -r requirements.txt
pip install "numpy<2"   # evita incompatibilidade de ABI com o cv_bridge do apt
```
