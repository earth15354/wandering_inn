import re

read_file = "./files/original_wandering_inn.txt"
write_file = "./files/preprocessed_wandering_inn.txt"

with open(read_file, 'r') as file:
  
    # Reading the content of the file
    # using the read() function and storing
    # them in a new variable
    data = file.read()
    length = 1200000
    start = 0
    data = data[start:(start+length)]
  
    data = re.sub(r'(\s)1.\w+', r'\1', data)
    data = re.sub(r'(\s)2.\w+', r'\1', data)
    data = re.sub(r'(\s)3.\w+', r'\1', data)
    data = re.sub(r'(\s)4.\w+', r'\1', data)
    data = re.sub(r'(\s)5.\w+', r'\1', data)
    data = re.sub(r'(\s)6.\w+', r'\1', data)
    data = re.sub(r'(\s)7.\w+', r'\1', data)
    data = re.sub(r'(\s)8.\w+', r'\1', data)

    data = re.sub(r'\s\*', r'', data)
    data = re.sub(r'\sPrevious Chapter', r'', data)
    data = re.sub(r'\sNext Chapter', r'', data)

    data = re.sub(r'(\s)\n+', r'\n\n', data)
    # print(len(data))

with open(write_file, 'w') as file:
    file.write(data)