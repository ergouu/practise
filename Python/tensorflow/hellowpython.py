n=int(input("n(n>0):"))
he=0

for i in range(1,n):
    if i%3==0 or i%5==0:
        he+=i
print (he)