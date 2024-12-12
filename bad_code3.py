def func(x, y):
    a = 0
    for i in range(x):
        a += y
        if a > 100:
            break
    if x > y:
        return x - y
    elif y > x:
        return y - x
    else:
        return a

result = func(50, 20)
print("Resultado:", result)
