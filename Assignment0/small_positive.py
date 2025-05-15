import time

start_time = time.time()

def has_same_digits(x):
    digits = sorted(str(x))
    return all(sorted(str(x * i)) == digits for i in range(2, 7))

x = 1
while True:
    if has_same_digits(x):
        print("Smallest x:", x)
        break
    x += 1

end_time = time.time()
print(f"Runtime: {end_time - start_time:.6f} seconds")

