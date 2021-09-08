






import collections, functools, operator
decimalToBinary = lambda n: decimalToBinary(n//2) + str(n%2) if n > 1 else str(n%2)
binaryToDecimal = lambda s: sum(int(c) * (2 ** i) for i, c in enumerate(str(s)[::-1]))

def generateHammingCodes(data):
    count = 0
    for i in range(len(str(data))):
        if (2**i >= len(str(data)) + i + 1): redundancy = i; break
    queue = collections.deque(str(data))
    lenResult = redundancy+len(str(data))
    result = [queue.pop() if (i & -i) != i else None for i in range(1,lenResult+1)]
    for index, bit in enumerate(result):
        if bit is None:
            check = [int(result[i]) if ("0000"+str(decimalToBinary(i+1)))[::-1][count] == "1" and result[i] is not None else None for i in range(len(result))]
            result[index]=str(functools.reduce(operator.xor,list(filter(lambda v: v is not None, check))))
            count = count + 1
    return ''.join(result)[::-1]
    
def findError(string):
    string = str(string)[::-1]; count = 0; binary_result = ""
    for index, bit in enumerate(string):
        if ('1' not in bin(abs(index+1))[3:] and index+1 != 0): # if power of 2
            check = [int(string[i]) if ("0000"+str(decimalToBinary(i+1)))[::-1][count] == "1" else None for i in range(len(string))]
            binary_result+=(str(functools.reduce(operator.xor,list(filter(lambda v: v is not None, check)))))
            count = count + 1  
    return binary_result[::-1]          
    
if __name__=="__main__":    
    code = generateHammingCodes(1011001)
    print(code)
    code = [int(bit) for bit in list(code)]
    code[1] = int(1-int(code[1]))
    code = int(''.join(str(i) for i in code))
    print(code)
    print(findError(code))
            
            
 
    
    
        
        
