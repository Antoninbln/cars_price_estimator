def romanToInt(expression): 
  """
  Return number from a roman expression.

  Param : <type> Description
  - expression <str> Roman expression of a number 

  Return : <type> Description
  - 
  """
  value = { 
    'M': 1000, 
    'D': 500, 
    'C': 100, 
    'L': 50, 
    'X': 10, 
    'V': 5, 
    'I': 1
  }
  tmp_cursor = 0
  result = 0

  for char in expression: 
    if value[char] <= tmp_cursor:
      result -= value[char] 
    else: 
      result += value[char]
    tmp_cursor = value[char]

  return result
