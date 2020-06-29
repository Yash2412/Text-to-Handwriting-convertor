def isPrime(n) : 
    if (n <= 1) : 
        return False
    if (n <= 3) : 
        return True
  
    if (n % 2 == 0 or n % 3 == 0) : 
        return False
  
    i = 5
    while(i * i <= n) : 
        if (n % i == 0 or n % (i + 2) == 0) : 
            return False
        i = i + 6
  
    return True

def isCorrect(i, n):
    a = list(map(int,str(i)))
    b = list(map(int,str(n)))
    
    a = [0]*(len(b)-len(a)) + a
    c = 0
    
    for j in range(len(a)):
        if a[j] ^ b[j] != 0:
            c += 1
            
    if c != 1:
        return False
    else:
        print(i,n)
        return True


for _ in range(int(input())):
    a, b = map(int,input().strip().split(' '))
    
    i = a+2
    test = a
    count = 0
    while i < b:

        if isPrime(i):
            if isCorrect(test,i):
                test = i
                count += 1
        i +=2

    print(count)