a=int(input("seconds(>=0):"))
b=0
c=0
d=0

if a>=0:
    b=(a-a%3600)/3600
    c=((a%3600)-a%60)/60
    d=a%60
    # print(format(b,'.0f'),format(c,'.0f'),d)
    print("%s %.0f %d" %("hhh",c,d))
else:
    print ("error")