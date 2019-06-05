import math
a=float(input("a(>0):"))
b=float(input("b(>0):"))
c=float(input("c(>0):"))
d=max(a,b,c)

if a+b+c>2*d:
    C=(math.acos((a*a+b*b-c*c)/(2*a*b)))*180/math.pi
    print ("%.1f"%(C))
else:
    print ("error")
