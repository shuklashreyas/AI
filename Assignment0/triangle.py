import time

def fetch_triangle(filepath):
    with open(filepath, 'r') as file:
        return [list(map(int, line.split())) for line in file.readlines()]

def max_path_sum(triangle):
    for row in range(len(triangle) - 2, -1, -1):
        for col in range(len(triangle[row])):
            triangle[row][col] += max(triangle[row + 1][col], triangle[row + 1][col + 1])
    return triangle[0][0]

start = time.time()
triangle_data = fetch_triangle("triangle.txt")
if triangle_data:
    result = max_path_sum(triangle_data)
    print(f"Max path sum: {result}")
    print(f"Runtime: {time.time() - start:.6f} seconds")

