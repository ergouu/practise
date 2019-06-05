def __is_prime (n):
    for i in range(2,n):
        if n%i==0:
            return False
    return True



n=int(input("n:"))


c=0

for i in range(2,n):
    if __is_prime(i):
        c+=i
print (c)
