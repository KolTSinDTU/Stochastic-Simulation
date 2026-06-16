import math
import random
import matplotlib.pyplot as plt

def d(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def cost(p, o):
    return sum(d(p[o[i]], p[o[(i+1)%len(o)]]) for i in range(len(o)))

def swap(o):
    r = list(o)
    i, j = random.sample(range(len(o)), 2)
    r[i], r[j] = r[j], r[i]
    return r

def sa(p, k_max=50000):
    o = list(range(len(p)))
    best = list(o)
    c_min = cost(p, o)
    c_cur = c_min
    
    for k in range(k_max):
        t = 1 / math.sqrt(1 + k)
        o_new = swap(o)
        c_new = cost(p, o_new)
        
        if c_new < c_cur or random.random() < math.exp(-(c_new - c_cur) / t):
            o = o_new
            c_cur = c_new
            if c_cur < c_min:
                c_min = c_cur
                best = list(o)
                
    return best

n = 20
pts = [(math.cos(2*math.pi*i/n), math.sin(2*math.pi*i/n)) for i in range(n)]
random.shuffle(pts)

res = sa(pts)
x = [pts[i][0] for i in res] + [pts[res[0]][0]]
y = [pts[i][1] for i in res] + [pts[res[0]][1]]

plt.plot(x, y, '-o')
plt.axis('equal')
plt.show()