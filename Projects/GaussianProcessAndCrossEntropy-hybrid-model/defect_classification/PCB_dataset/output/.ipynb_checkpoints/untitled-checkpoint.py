n = int(input())
height = map(int,input().split(' '))
wide = map(int,input().split(' '))
q = int(input())
for i in range(q):
    d,e,m,l = map(int,input().split())
    for j in range(1,n+1):
        if d == 1 :
            e = e + (m**2)*(height[j]-height[j-1])
        else :
            e = e + m*(height[j]-height[j-1])
        if e < 0 or l > wide[j] :
            print('No\n',j,'\n')
        else :
            print('yes\n')
