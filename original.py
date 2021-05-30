def Bucket_Sort(array, n, max, min):
    """
    INPUT:
    - array, an array of at most 10^8 integers.
    - n, an integer representing the number of elements in the array.
    - max, an integer representing the maximum number in the array.
    - min, an integer representing the minimum number in the array.

    OUTPUT: 
    - array, the sorted array.
    """

    j = 0
    cpos = []
    cneg = []

    # Begin Part 1
    for i in range(max+1):
        cpos.append(0)

    for i in range(-(min-1)+1):
        cneg.append(0)
    # End Part 1

    # Begin Part 2
    for i in range(n):
        if (array[i] >= 0):
            cpos[array[i]] += 1
        else:
            cneg[-array[i]] += 1
    # End Part 2
    
    # Begin Part 3
    for i in range(-min, 0, -1):
        while cneg[i]:
            array[j] = -i
            j += 1
            cneg[i] -= 1
    
    for i in range(max+1):
        while cpos[i]:
            array[j] = i
            j += 1
            cpos[i] -= 1
    # End Part 3
            
    return array
