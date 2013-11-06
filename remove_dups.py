fin = open('test.txt', 'r')
fout = open('clean.txt', 'w')

s = set()
for line in fin:
    if line.rstrip().isalpha():
        if not line in s:
            s.add(line)
            fout.write(line)

fin.close()
fout.close()