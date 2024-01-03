from gekko import GEKKO
m=GEKKO(remote=False)

x1=m.Var(1,lb=1,ub=5)
x2=m.Var(5,lb=1,ub=5)
x3=m.Var(5,lb=1,ub=5)
x4=m.Var(1,lb=1,ub=5)

m.Equation(x1*x2*x3*x4>=25)
m.Equation(x1**2+x2**2+x3**2+x4**2==40)

m.Obj(x1*x4*(x1+x2+x3)+x3) #Objective function

m.solve()
print(x1,x2,x3,x4)
print('Hello')
q=4
p=5
print(p+q)