import matplotlib.pyplot as plt


file_lists = ["./class1.txt"] #, "./class2.txt", "./class3.txt"]
w = open("./bend_arm_all_posi.txt", 'w')
g = open("./bend_arm_all_nega.txt", 'w')

l = 0
m = 0
n = 0
for file_list in file_lists:
    f = open(file_list, 'r')
    for lines in f.readlines():
        l += 1
        split = lines.split(',')[-1]

        if int(split) == 1:
            m += 1
            w.write(lines)
        elif int(split) == 0:
            n += 1
            g.write(lines)

        else:
            print("error!")
    f.close()

print(l)
print(m)
print(n)
w.close()
g.close()



"""
f = open("C:\\Users\\JM\\Desktop\\Data\\ETRIrelated\\preprocess_data\\000006.txt", 'r')

lin = []
for lines in f.readlines():
    split = lines.split(",")[3:-1]
    lin = map(float, split)
    break

print len(lin)
x = []
y = []
for i in range(18):
    if lin[2*i] == 0 or lin[2*i +1] == 0:
        lin[2*i] = None
        lin[2*i +1] = None
    x.append(lin[2*i])
    y.append(lin[2*i + 1])

coco_pair = [[1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11],
             [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17], [2, 16], [5, 17]]

plt.scatter(x, y)
for i in range(len(x)):
    if i == 1:
        c = 'red'
    elif i == 0:
        c = 'green'
    else:
        c = 'blue'

    plt.scatter(x[i], y[i], c=c)

for pair in coco_pair:
    if x[pair[0]] == 0 or x[pair[1]] == 0 or \
        y[pair[0]] == 0 or y[pair[1]] == 0:
        continue

    X = [x[pair[0]], x[pair[1]]]
    Y = [y[pair[0]], y[pair[1]]]
    plt.plot(X,Y, c='k')

ax = plt.gca()
ax.set_ylim(ax.get_ylim()[::-1])

plt.show()
print x
print y
"""