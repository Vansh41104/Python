import math
print("enter the numbers")
a=int(input("Enter the number 1: "))
b=int(input("Enter the number 2: "))
choose=int(input("enter the choice:\n1.addition\n2.subtraction\n3.multiplication\n4.Division"))
if(choose==1):
    add=a+b
    print(add)
if(choose==2):
    sub=a-b
    print(sub)
if(choose==3):
    multi=a*b
    print(multi)
if(choose==4):
    div=a/b
    print(div)
