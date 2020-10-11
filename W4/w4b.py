import numpy as np
import matplotlib.pyplot as plt

# 0 white, 1 black
def draw_card():
    return np.random.randint(0,2, size=(2))

def draw():
    x1 = np.random.choice([0, 1], p=[0.5, 0.5])  # equal chance of white or black as 6 sides, 3 black 3 white
    if x1 == 1:  # draw black face so 1/3 to be white other face and 2/3 black other face
        x2 = np.random.choice([0, 1], p=[1.0/3.0, 2.0/3.0])
    elif x1 == 0:   # vice versa
        x2 = np.random.choice([0, 1], p=[2.0/3.0, 1.0/3.0])
    return np.array([x1, x2])

N = 100000
probs = np.zeros((3))
num_faces = np.zeros((2))

for i in range(N):
    cc = draw()
    # if cc[0] == 0 and cc[1] == 0 :
    #     probs[0] += 1
    # elif cc[0] == 0 and cc[1] == 1:
    #     probs[1] += 1
    # elif cc[0] == 1 and cc[1] == 0:
    #     probs[1] += 1
    # elif cc[0] == 1 and cc[1] == 1:
    #     probs[2] += 1

    if cc[0] == 0 or cc[1] == 0:
        num_faces[0] += 1
    if cc[0] == 1 or cc[1] == 1:
        num_faces[1] += 1
 
# probs = probs/float(N)
# print(probs)

print(num_faces)