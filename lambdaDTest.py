import math

My = 126.0
Mcrd = 0.85 * My

for i in range(30):
    beta = 1+0.01*(i+1)
    Mcrd_i = beta * Mcrd
    ld = math.sqrt(My/Mcrd_i)
    Mnd = (1-0.22*math.pow(Mcrd_i/My,0.5))*math.pow(Mcrd_i/My,0.5)*My
    print(f'beta : {beta:.3f}, ld : {ld:.3f}, Mnd : {Mnd:.3f}')
