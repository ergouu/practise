
day = 365
m = 0

for year in range(1901,2001):
    for month in range(1,13):
        if day%7==6:
            m+=1
        if month in (1,3,5,7,8,10,12):
            day +=31
        elif month in (4,6,9,11):
            day +=30
        elif month == 2 and ((year%4==0 and year%100!=0) or (year % 400 == 0 )):
            day += 29
        else:
            day += 28
print (m)

"""
def year(y):
    if (y % 4 == 0 and y % 100 !=0) or (y % 400 == 0):
        return True
    else:
        return False

def month(m,y):
    if m in (1,3,5,7,8,10,12):
        return 31
    elif year(y) and m ==2:
        return 29
    elif m in (4,6,9,11):
        return 30
    else:
        return 28

days = 365
QQQQ = 0

for yy in range(1901,2001):
    for mm in range(1,13):
        if days % 7 ==6:
            print(yy,mm)
            QQQQ += 1
        days += month(mm,yy)
print (QQQQ)


import calendar
count=0
for year in range(1901,2001):
    for month in range(1,13):
            if calendar.monthcalendar(year,month)[0].index(1) == 6:
                count+=1
print (count)



"""

