def p(n):
    if n == 1 or n == 2:
        return 1
    if n > 2:
        return p(n-1)+p(n-2)

n=int(input("n:"))
for i in range(1,n):
    print (p(i))