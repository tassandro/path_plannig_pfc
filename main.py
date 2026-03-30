import heapq
import os
from collections import deque
from datetime import datetime
from math import sqrt
from typing import Literal
import copy
import random

import cv2
import numpy as np

MOVEMENTS_8D = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
MOVEMENTS_4D = MOVEMENTS_8D[:4]
INF = int(1e9)


def calculate_h_value(x, y, end):
    return sqrt((x - end[0]) ** 2 + (y - end[1]) ** 2)


class Node:
    """Class to store the A* graph"""

    def __init__(self):
        self.parent_x = -1  # Coodernada x do nó pai
        self.parent_y = -1  # Coodernada y do nó pai

        # Propriedades A*
        self.f = float("inf")  # Custo total (g+h)
        self.g = float("inf")  # Custo do ínicio até este nó
        self.h = 0  # Custo Heuristico deste nó até o destino
        self.w: float = 0  # Weight


class NodeProcessor:
    def __init__(
        self,
        img_path: str,
        img: cv2.typing.MatLike,
        *,
        has_weight: bool = True,
        robot_size: float = 0,
        movements: int = 8,
        bgr_code: tuple[int, int, int] = (84, 1, 68)
    ):
        self.img_path = img_path
        self.img = img
        self.has_weight = has_weight
        self.movements = MOVEMENTS_8D if movements == 8 else MOVEMENTS_4D
        self.max_weight = -1 * INF
        self.transitable_color = np.array(
            bgr_code, dtype=np.uint8
        )  # Formato BGR ao invés de RGB (68, 1, 84)
        self.node_list = self._create_tree(size=robot_size, has_weight=has_weight)

    def _create_tree(self, *, size: float, has_weight: bool) -> list[list[Node | None]]:
        h, l, _ = self.img.shape  # Dimensões da imagem
        node_list = [
            [Node() if self._is_transitable(x, y) else None for x in range(l)]
            for y in range(h)
        ]

        if has_weight:
            for y in range(h):
                for x in range(l):
                    if node_list[y][x] is not None:
                        node_list[y][x].w = self._check_node(
                            node_list, h, l, x, y, size
                        )

        bw_path = self.draw_black_and_white_image(node_list)
        self.draw_random_color_image(node_list)
        if bw_path:
            self.draw_weighted_image(node_list, bw_path)
        return node_list

    def draw_black_and_white_image(self, node_list):
        h, l, _ = self.img.shape  # Dimensões da imagem
        img = cv2.imread(self.img_path)
        if img is None:
            return ""
        for y in range(h):
            for x in range(l):
                node = node_list[y][x]
                color = (0, 0, 0) if node is None else (255, 255, 255)
                cv2.circle(img, (x, y), 1, color, thickness=2, lineType=8)

        if not os.path.exists("output/graphs"):
            os.makedirs("output/graphs")
        current_timestamp = datetime.now()
        time = current_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
        folder_name = "output/graphs/" + time + "_bw.png"
        cv2.imwrite(folder_name, img)
        self.img = img
        return folder_name
    
    def draw_random_color_image(self, node_list):
        h, l, _ = self.img.shape  # Dimensões da imagem
        img = cv2.imread(self.img_path)
        if img is not None:
            for y in range(h):
                for x in range(l):
                    node = node_list[y][x]
                    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) if node is not None else (0,0,0)
                    cv2.circle(img, (x, y), 1, color, thickness=2, lineType=8)

            if not os.path.exists("output/graphs"):
                os.makedirs("output/graphs")
            current_timestamp = datetime.now()
            time = current_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
            folder_name = "output/graphs/" + time + "_random.png"
            cv2.imwrite(folder_name, img)
        
    def draw_weighted_image(self, node_list, path_bw):
        h, l, _ = self.img.shape  # Dimensões da imagem
        img = cv2.imread(path_bw)
        if img is not None:
            for y in range(h):
                for x in range(l):
                    node = node_list[y][x]
                    if node is None: 
                        continue
                    elif node.w == INF:
                        color = (0, 0, 0)
                    else:
                        color = 255 - ((node.w / self.max_weight) * (255 - 40))
                        color = int(color)
                        color = (color, color, color)
                    cv2.circle(img, (x, y), 1, color, thickness=2, lineType=8)

            if not os.path.exists("output/graphs"):
                os.makedirs("output/graphs")
            current_timestamp = datetime.now()
            time = current_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
            folder_name = "output/graphs/" + time + "_weight.png"
            cv2.imwrite(folder_name, img)
        
    # Função para verificar se um pixel é transitável
    def _is_transitable(self, x, y):
        color = self.img[y][x]
        if np.array_equal(color, self.transitable_color):
            return True
        return False

    def _check_node(self, node_list, h, l, x, y, size) -> int:
        # setting distances until twice the size of the robot
        limit = 100
        for i in range(1, limit):
            for mov in self.movements:
                mov_x = i * mov[0]
                mov_y = i * mov[1]
                nx = x + mov_x
                ny = y + mov_y
                d = i * (1.0 if mov[0] == 0 or mov[1] == 0 else sqrt(2))
                # se o nó é invalido
                if (
                    not (ny > -1 and ny < h and nx > -1 and nx < l)
                    or node_list[ny][nx] is None
                ):
                    # se a distancia for menor ou igual ao minimo recomendado, o nó tem que ser desfeito, ele não é transitavel
                    if d <= size:
                        return INF
                    else:
                        # the higher the i, the lower the weight
                        weight = limit - i
                        self.max_weight = max(self.max_weight, weight)
                        return weight
        return 0

    def draw_point(self, x, y) -> None:
        cv2.circle(self.img, (x, y), 5, (0, 0, 255), thickness=6, lineType=8)

    def a_star(self, *, start, end):
        print("## UNWEIGHTED")
        self._a_star_dijkstra_scope(
            start=start, end=end, dijkstra=False, weighted=False
        )
        print("## WEIGHTED")
        self._a_star_dijkstra_scope(start=start, end=end, dijkstra=False, weighted=True)

    def djikstra(self, *, start, end):
        print("## UNWEIGHTED")
        self._a_star_dijkstra_scope(start=start, end=end, dijkstra=True, weighted=False)
        print("## WEIGHTED")
        self._a_star_dijkstra_scope(start=start, end=end, dijkstra=True, weighted=True)

    def _a_star_dijkstra_scope(self, *, start, end, dijkstra=False, weighted: bool):
        """Executa o algoritmo A* para encontrar um caminho mínimo entre start e end"""

        time_begin = datetime.now()
        print(f"Começando {'Dijsktra' if dijkstra else 'A*'}")

        aux_node_list = copy.deepcopy(self.node_list)

        # Modificar propriedades do nó de ínicio
        start_x, start_y = start
        aux_node_list[start_y][start_x].f = 0
        aux_node_list[start_y][start_x].g = 0
        aux_node_list[start_y][start_x].parent_x = start_x
        aux_node_list[start_y][start_x].parent_y = start_y

        # Inicializar a lista dos nós a serem visitados com o nó de início
        open_list = []
        heapq.heappush(open_list, (0.0, start_x, start_y))

        # Inicializar a lista com os nós já visitados
        h, l, _ = self.img.shape  # Dimensões da imagem
        closed_list = [[False for _ in range(l)] for _ in range(h)]

        # Loop principal do A*
        while len(open_list) > 0:
            # Pop no nó da lista aberta com o menor valor de f
            f, x, y = heapq.heappop(open_list)

            if closed_list[y][x] == True:
                continue

            closed_list[y][x] = True

            for mov in self.movements:
                nx = x + mov[0]
                ny = y + mov[1]

                # Se o nó é transitável e ainda não foi visitado
                if (
                    ny > -1
                    and ny < h
                    and nx > -1
                    and nx < l
                    and aux_node_list[ny][nx]
                    and not closed_list[ny][nx]
                ):
                    if (nx, ny) == end:  # Se o destino foi alcançado
                        aux_node_list[ny][nx].parent_x = x
                        aux_node_list[ny][nx].parent_y = y
                        print(
                            f"Custo {'Dijsktra' if dijkstra else 'A*'}: {f}"
                        )  # Custo total
                        self._output_path(
                            aux_node_list,
                            end,
                            "dijkstra" if dijkstra else "a_star",
                            weighted=weighted,
                        )  # Desenhando o path encontrado na imagem
                        time_finish = datetime.now()
                        print(f"{'Dijsktra' if dijkstra else 'A*'} finalizado")
                        print(
                            f"Tempo de duração: {(time_finish - time_begin).total_seconds()}"
                        )
                        return
                    else:
                        # Calculando os custos para o novo ponto
                        ng = aux_node_list[y][x].g + (
                            1.0 if mov[0] == 0 or mov[1] == 0 else sqrt(2)
                        )
                        nh = calculate_h_value(nx, ny, end) if not dijkstra else 0
                        nw = aux_node_list[y][x].w if weighted else 0
                        nf = ng + nh + nw

                        # Se o custo total for infinito ou maior do que o anterior, alterar na lista de nó e adicionar na priority_queue
                        if (
                            aux_node_list[ny][nx].f == float("inf")
                            or aux_node_list[ny][nx].f > nf
                        ):
                            heapq.heappush(open_list, (nf, nx, ny))
                            aux_node_list[ny][nx].parent_x = x
                            aux_node_list[ny][nx].parent_y = y

                            aux_node_list[ny][nx].g = ng
                            aux_node_list[ny][nx].h = nh
                            aux_node_list[ny][nx].f = nf

    def dfs(self, *, start, end):
        self.dfs_scope(start=start, end=end, weighted=False)
        self.dfs_scope(start=start, end=end, weighted=True)

    def dfs_scope(self, *, start, end, weighted: bool):
        time_begin = datetime.now()
        print(f"Começando DFS")

        start_x, start_y = start
        end_x, end_y = end

        h, l, _ = self.img.shape  # Dimensões da imagem

        vis = set()  # Definindo a lista de nós visitados
        vis.add(start)  # Adicionando nó inicial aos visitados

        aux_node_list = copy.deepcopy(self.node_list)

        aux_node_list[start_y][start_x].parent_x = start_x
        aux_node_list[start_y][start_x].parent_y = start_y
        aux_node_list[start_y][start_x].g = 0

        stack = [start]
        while stack:
            x, y = stack.pop()
            for mov in self.movements:
                nx = x + mov[0]
                ny = y + mov[1]
                ng = aux_node_list[y][x].g + (
                    1.0 if mov[0] == 0 or mov[1] == 0 else sqrt(2)
                )
                ng += aux_node_list[y][x].w if weighted else 0
                if ny > -1 and ny < h and nx > -1 and nx < l and aux_node_list[ny][nx]:
                    if ng < aux_node_list[ny][nx].g:
                        aux_node_list[ny][nx].g = ng
                        aux_node_list[ny][nx].parent_x = x
                        aux_node_list[ny][nx].parent_y = y
                    if (nx, ny) not in vis:
                        vis.add((nx, ny))
                        stack.append((nx, ny))

        print(f"Custo DFS: {aux_node_list[end_y][end_x].g}")

        self._output_path(aux_node_list, end, "dfs")

        time_finish = datetime.now()
        print(f"DFS finalizado")
        print(f"Tempo de duração: {(time_finish - time_begin).total_seconds()}")

    def bfs(self, *, start, end):
        time_begin = datetime.now()
        print(f"Começando BFS")

        start_x, start_y = start
        end_x, end_y = end

        h, l, _ = self.img.shape

        vis = set()
        vis.add(start)

        self.node_list[start_y][start_x].parent_x = start_x
        self.node_list[start_y][start_x].parent_y = start_y
        self.node_list[start_y][start_x].g = 0

        q = deque([start])
        while q:
            x, y = q.popleft()
            for mov in self.movements:
                nx = x + mov[0]
                ny = y + mov[1]
                ng = self.node_list[y][x].g + (
                    1.0 if mov[0] == 0 or mov[1] == 0 else sqrt(2)
                )
                if ny > -1 and ny < h and nx > -1 and nx < l and self.node_list[ny][nx]:
                    if ng < self.node_list[ny][nx].g:
                        self.node_list[ny][nx].g = ng
                        self.node_list[ny][nx].parent_x = x
                        self.node_list[ny][nx].parent_y = y
                    if (nx, ny) not in vis:
                        vis.add((nx, ny))
                        q.append((nx, ny))

        print(f"Custo BFS: {self.node_list[end_y][end_x].g}")

        self._output_path(end, "bfs")

        time_finish = datetime.now()
        print(f"DFS finalizado")
        print(f"Tempo de duração: {(time_finish - time_begin).total_seconds()}")

    def _output_path(
        self,
        node_list,
        end,
        algorithm: Literal["a_star", "dijkstra", "dfs", "bfs"] = "a_star",
        weighted: bool = False,
    ):
        x = end[0]
        y = end[1]

        if algorithm == "a_star":
            color = (255, 0, 0) if weighted else (255, 169, 64)
        elif algorithm == "dijkstra":
            color = (24, 173, 46) if weighted else (96, 247, 62)
        elif algorithm == "dfs":
            color = (255, 42, 245)
        else:
            color = (255, 255, 255)

        cost = 0
        real_distance = 0
        
        path = []
        path.append((x,y))
        while not (node_list[y][x].parent_x == x and node_list[y][x].parent_y == y):
            aux_x = node_list[y][x].parent_x
            aux_y = node_list[y][x].parent_y

            cv2.circle(self.img, (x, y), 2, color, thickness=3, lineType=8)
            cv2.line(self.img, (x, y), (aux_x, aux_y), color, thickness=1, lineType=8)

            real_distance += sqrt((x - aux_x) ** 2 + (y - aux_y) ** 2)
            cost += node_list[y][x].w

            x = aux_x
            y = aux_y
            path.append((x,y))

        print(f"Comprimento do caminho: {cost}")
        print(f"Distância real: {real_distance}")
        path.reverse()
        return path


def str_to_bool(s):
    return s.strip().lower() in ["true", "t", "yes", "y", "1"]


if __name__ == "__main__":

    # Caminho padrão da imagem
    default_img_path = r"imagens_readme/mapa_ctex.png"
    img_path = (
        input(
            f"Escreva o caminho da imagem que contém o mapa desejado [{default_img_path}]: "
        )
        or default_img_path
    )
    img = cv2.imread(img_path)  # Carrega imagem

    if img is None:
        print("Imagem inválida")
    else:
        robot_size = 3 # Valor fixo, pode ser parametrizado se quiser

        default_weight = "True"
        weight_input = (
            input(f"Deseja o grafo com pesos? (True/False) [{default_weight}]: ")
            or default_weight
        )
        weight_input = str_to_bool(weight_input)

        default_movements = "8"
        movements_input = (
            input(f"Movimentos diagonais ou não? (4/8) [{default_movements}]: ")
            or default_movements
        )
        movements_input = int(movements_input)
        
        default_bgr = "84, 1, 68"
        bgr = input(f"Código BGR dos pixels transitáveis [{default_bgr}]: ") or default_bgr
    
        bgr_code = tuple(int(item) for item in bgr.replace(" ", "").split(","))
        
        # validar se tem exatamente 3 elementos
        if len(bgr_code) != 3:
            raise ValueError("Digite exatamente 3 valores separados por vírgula.")
        
        # validar se estão no intervalo 0–255
        if not all(0 <= v <= 255 for v in bgr_code):
            raise ValueError("Cada valor deve estar entre 0 e 255.")
            

        print("Mapeando nós do mapa...")
        processor = NodeProcessor(
            img_path,
            img,
            has_weight=weight_input,
            robot_size=robot_size,
            movements=movements_input,
            bgr_code=bgr_code
        )
        print("Mapeamento finalizado.")

        # Valores padrão de coordenadas (x, y)
        default_start = "634, 914"
        default_end = "1030, 1168"

        start_input = (
            input(f"O ponto de inicio (x, y) [{default_start}]: ") or default_start
        )
        start = tuple(int(item) for item in start_input.replace(" ", "").split(","))

        end_input = input(f"O ponto final (x, y) [{default_end}]: ") or default_end
        end = tuple(int(item) for item in end_input.replace(" ", "").split(","))

        print(f"Start: {start} | End: {end} | Peso: {weight_input}")

        # Desenha os pontos inicial e final na imagem colorida (vermelho)
        processor.draw_point(*start)
        processor.draw_point(*end)

        for _ in range(4):
            print("\n Algoritmos disponíveis para criação de caminhos: ")
            print("1 — A*")
            print("2 — Dijkstra")
            print("3 — DFS")
            print("4 — BFS")

            algorithm = input("Digite sua escolha (1-4) (deixe em branco para pular): ")

            if algorithm == "":
                break
            elif algorithm == "1":
                processor.a_star(start=start, end=end)
            elif algorithm == "2":
                processor.djikstra(start=start, end=end)
            elif algorithm == "3":
                processor.dfs(start=start, end=end)
            elif algorithm == "4":
                processor.bfs(start=start, end=end)
            else:
                print("Opção inválida. Escolha entre 1 e 4.")

        # Gera nome de arquivo com timestamp
        # Cria pasta de saída, se não existir
        if not os.path.exists("output"):
            os.makedirs("output")
        current_timestamp = datetime.now()
        time = current_timestamp.strftime("%Y_%m_%d_%H_%M_%S")
        folder_name = "output/" + time
        cv2.imwrite(folder_name + ".png", processor.img)