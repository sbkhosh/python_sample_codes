#!/usr/bin/python3

def countSubstrings(s, c): 
  
    # Length of the string 
    n = len(s) 
  
    cnt = 0
    Sum = 0
  
    # Traverse in the string 
    for i in range(n): 
  
        # If current character is different 
        # from the given character 
        if (s[i] != c): 
            cnt += 1
        else: 
  
            # Update the number of sub-strings 
            Sum += (cnt * (cnt + 1)) // 2
  
            # Reset count to 0 
            cnt = 0
          
    # For the characters appearing 
    # after the last occurrence of c 
    Sum += (cnt * (cnt + 1)) // 2
    return Sum


s = "baa"
c = 'b'
print(countSubstrings(s, c)) 
