import heapq

def dijkstra(graph, start):
    # Initialize the distance to all nodes as infinity and the start node distance as 0
    distances = {node: float('infinity') for node in graph}
    distances[start] = 0
    # Priority queue to store the nodes to be evaluated
    priority_queue = [(0, start)]
    # While there are nodes left to evaluate
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        # Skip if we find a longer path to the current node
        if current_distance > distances[current_node]:
            continue
        # Check the neighbors of the current node
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            # Only consider this new path if it's better
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances

# Example usage
graph = {
    'A': {'B': 1, 'C': 4},
    'B': {'A': 1, 'C': 2, 'D': 5},
    'C': {'A': 4, 'B': 2, 'D': 1},
    'D': {'B': 5, 'C': 1}
}

start_node = 'A'
distances = dijkstra(graph, start_node)
print(distances)
