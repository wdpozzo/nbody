from __future__ import division
from sympy import *
from sympy import simplify, separatevars, radsimp
from sympy.simplify.radsimp import rad_rationalize
x1, y1, z1, px1, py1, pz1, x2, y2, z2, px2, py2, pz2, m1, m2, G = symbols('x1 y1 z1 px1 py1 pz1 x2 y2 z2 px2 py2 pz2 m1 m2 G')

r = sqrt((x1-x2)**2+(y1-y2)**2+(z1-z2)**2)
p1_2 = px1**2+py1**2+pz1**2
p2_2 = px2**2+py2**2+pz2**2
H_N = p1_2/(2*m1)-(1/2)*G*m1*m2/r

print('d0PN/dx := \n', H_N.diff(x1))
print('d0PN/dpx := \n', H_N.diff(px1))

p1dp2  = px1*px2+py1*py2+pz1*pz2
n12dp1 = ((x1-x2)*px1+(y1-y2)*py1+(z1-z2)*pz1)/r
n12dp2 = ((x1-x2)*px2+(y1-y2)*py2+(z1-z2)*pz2)/r

H_1PN = -(1/8)*(p1_2**2)/(m1**3) + (1/8)*(G*m1*m2/r)*(-12*p1_2/(m1**2) + 14*p1dp2/(m1*m2) + 2*(n12dp1*n12dp2)/(m1*m2)) + (1/4)*(G*m1*m2/r)*(G*(m1+m2)/r)

dHdx1 = H_1PN.diff(x1)
dHdpx1 = H_1PN.diff(px1)

#dHdx1 = separatevars(dHdx1)
#dHdpx1 = separatevars(dHdx1)

#dHdx1 = simplify(dHdx1)
#dHdpx1 = simplify(dHdx1)

print('d1PN/dx := \n', dHdx1)
print('d1PN/dpx := \n', dHdpx1)

H_2PN = (1/16)*(p1_2**3)/(m1**5) + (1/8)*(G*m1*m2/r)*(5*p1_2**2/(m1**4) - (11/2)*p1_2*p2_2/((m1**2)*(m2**2))-(p1dp2**2)/((m1**2)*(m2**2)) + 5*p1_2*(n12dp2**2)/((m1**2)*(m2**2)) - 6*(p1dp2*n12dp1*n12dp2)/((m1**2)*(m2**2)) - (3/2)*(n12dp1**2)*(n12dp2**2)/((m1**2)*(m2**2))) + (1/4)*(G**2*m1*m2/(r**2))*(m2*(10*p1_2/(m1**2) + 19*p2_2/m2**2)-(1/2)*(m1+m2)*(27*p1dp2+6*n12dp1*n12dp2)/(m1*m2)) - (1/8)*(G*m1*m2/r)*(G**2*(m1**2+5*m1*m2+m2**2)/r**2)

dHdx1 = H_2PN.diff(x1)
dHdpx1 = H_2PN.diff(px1)

#dHdx1 = sp.separatevars(dHdx1)
#dHdpx1 = sp.separatevars(dHdx1)

#dHdx1 = rs.rad_rationalize(dHdx1)
#dHdpx1 = rs.rad_rationalize(dHdx1)

print('d2PN/dx := \n', H_2PN.diff(x1))
print('d2PN/dpx := \n', H_2PN.diff(px1))
