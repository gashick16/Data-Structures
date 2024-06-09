from collections import defaultdict
import time


def calculate_performance(time_taken, n):
    if time_taken == 0:
        return float('inf')
    c = 2 * n ** 3
    performance = c / time_taken * 1e-6  # в MFlops
    return performance

def can_pass_labyrinth_array(matrix, starts, ends):
    visited = [[False]*len(matrix[0]) for _ in range(len(matrix))]
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Right, Left, Down, Up

    def dfs(x, y):
        if x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0]):
            return False
        if matrix[x][y] == 1:
            return True
        if visited[x][y]:
            return False
        visited[x][y] = True
        for dx, dy in directions:
            if dfs(x+dx, y+dy):
                return True
        return False

    for start, end in zip(starts, ends):
        if not dfs(*start) or not dfs(*end):
            return False
    return True

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.next = None

def can_pass_labyrinth_list(matrix, starts, ends):
    visited = [[False]*len(matrix[0]) for _ in range(len(matrix))]

    def dfs(node):
        x, y = node.x, node.y
        if x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0]):
            return False
        if matrix[x][y] == 1:
            return True
        if visited[x][y]:
            return False
        visited[x][y] = True
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            next_node = Node(x+dx, y+dy)
            if dfs(next_node):
                return True
        return False

    for start, end in zip(starts, ends):
        path = Node(*start)
        if not dfs(path) or not dfs(Node(*end)):
            return False
    return True

def can_pass_labyrinth_graph(matrix, starts, ends):
    graph = defaultdict(lambda: {'walls': []})
    matrix = matrix
    rows, cols = len(matrix), len(matrix[0])

    for x in range(rows):
        for y in range(cols):
            if matrix[x][y] == 1:
                graph[(x, y)]['walls'].append((x+1, y))
                graph[(x, y)]['walls'].append((x, y+1))

    visited = set()
    queue = [starts]
        
    while queue:
        current = queue.pop(0)
        if current == ends:
            return True
        
        if current not in visited:
            visited.add(current)
            for neighbor in graph[current].get('walls', []):
                if neighbor not in visited:
                    queue.append(neighbor)
                    
    return False

# Определение лабиринта
matrix = [
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0],
    [0, 0, 0, 0, 0]
]

# Координаты входов и выходов для групп людей
starts = [(0, 0), (1, 1)]  # Входы
ends = [(4, 0), (3, 1)]  # Выходы



# Вызов функции для проверки возможности прохождения
start_time = time.perf_counter()
result = can_pass_labyrinth_array(matrix, starts, ends)
end_time = time.perf_counter()

print()

# Время выполнения
time_array = end_time - start_time
print(f"Время выполнения с помощью массива: {time_array} seconds")

if result:
    print("Все группы могут безопасно пройти через лабиринт.")
else:
    print("Некоторые группы не могут безопасно пройти через лабиринт.")
print()




# Вызов функции для проверки возможности прохождения
start_time = time.perf_counter()
result = can_pass_labyrinth_list(matrix, starts, ends)
end_time = time.perf_counter()

# Время выполнения
time_list = end_time - start_time
print(f"Время выполнения с помощью списка: {time_list} seconds")

if result:
    print("Все группы могут безопасно пройти через лабиринт.")
else:
    print("Некоторые группы не могут безопасно пройти через лабиринт.")
print()


starts = (0, 0)
ends = (4, 0)

# Вызов функции для проверки возможности прохождения
start_time = time.perf_counter()
result = can_pass_labyrinth_graph(matrix, starts, ends)
end_time = time.perf_counter()

# Время выполнения
time_graph = end_time - start_time
print(f"Время выполнения с помощью графа: {time_graph} seconds")

if result:
    print("Все группы могут безопасно пройти через лабиринт.")
else:
    print("Некоторые группы не могут безопасно пройти через лабиринт.")
print()

