f = open("file.txt", "a+")

for i in range(10):
    f.write(str(i) + '\n')
    # f.write('\n')

f.close()