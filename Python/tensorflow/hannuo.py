def hanno(n,A,B,C):
    if n==1:
        return print("Move %d from %s to %s" %(n,A,C))
    else :
        hanno(n-1,A,C,B)
        print("Move %d from %s to %s" %(n,A,C))
        hanno(n-1,B,A,C)

hanno(2,"左","中","右")        