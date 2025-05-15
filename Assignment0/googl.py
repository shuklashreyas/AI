import time

start_time = time.time()

max_sum = 0
max_a = max_b = 0

for a in range(1, 100):
    for b in range(1, 100):
        digit_sum = sum(map(int, str(a ** b)))
        if digit_sum > max_sum:
            max_sum = digit_sum
            max_a, max_b = a, b

end_time = time.time()

print(f"Maximum digital sum: {max_sum} (a={max_a}, b={max_b})")
print(f"Runtime: {end_time - start_time:.6f} seconds")

