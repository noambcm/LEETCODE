import array
import numpy as np

def sum(n):
    S=0
    for i in range(n+1):
        S=S+i
    return S

def fact(n):
    P=1
    for i in range (1,n+1):
        P=P*i
    print(P)


def max(list):
    m=list[0]
    for i in range(1,len(list)):
        if list[i]>m:
            m=list[i]
    print(m)


def Fibonacci(n):
    if n<0:
        print("Error")
    elif n==0:
        return 0
    elif n==1:
        return 1
    else:
        return Fibonacci(n-1) + Fibonacci(n-2)
    
def Fibonacci1(n):
    if n<0:
        print("Error")
    elif n==0:
        return 0
    elif n==1:
        return 1
    else:
        S=0
        for i in range(n-1):
            S=S+Fibonacci1(i)
        return S+1

def rotation(arr):
    arr = np.array(arr)
    n=len(arr)
    rendu=np.empty(n, dtype=arr.dtype)
    rendu[n-1]=arr[0]
    for i in range(0,n-1):
        rendu[i]=arr[i+1]
    print(rendu)

if __name__ == "__main__":
    rotation([1,4,7,9,65,43,0,98])
    

   

