import time

start_time = time.time()

for a in range(1, 1000):
    for b in range(a + 1, 1000 - a):
        c = 1000 - a - b
        if a * a + b * b == c * c:
            print(f"Triplet: a={a}, b={b}, c={c}")
            print(f"Product abc = {a * b * c}")
            end_time = time.time()
            print(f"Runtime: {end_time - start_time:.6f} seconds")
            exit()

