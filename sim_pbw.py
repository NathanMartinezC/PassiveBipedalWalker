import bipedwalker.pbw as pbw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import numpy as np
import pylab
import collections

# Vector de diseño prueba
"""Inicialización paramétrica
        p = [Rf_0, Rs_1, Fl_2, Fw_3, Al_4, Ah_5, Aw_6, Ll_7, Lh_8,
            Lw_9, Hl_10, Hr_11, Hcp_12, MCro_13, tini_14, gamma_15,
            roF_16, roA_17, roL_18, roH_19, roMC_20]
"""
Rf = 0.4266
Rs = 0.4731
Fl = 0.1784
Fw = 0.1195
Al = 0.0439
Ah = 0.0915
Aw = 0.0786
Ll = 0.0222
Lh = 0.3207
Lw = 0.0535
Hl = 0.0208
Hr = 0.0076
Hcp = 0.0690
MCro = 0.0221
tini = 0.3577
gamma = 0.0262
roF = 1
roA = 3
roL = 3
roH = 3
roMC = 1

p = np.array([Rf, Rs, Fl, Fw, Al, Ah, Aw, Ll, Lh,
              Lw, Hl, Hr, Hcp, MCro, tini, gamma,
              roF, roA, roL, roH, roMC])

# Se crea una instancia del bípedo
robot = pbw.PBW(p)

# Se calcula la dinámica del plano frontal/sagital (obtención de estados)
robot.dinamicaF()
robot.dinamicaS()

# Opcional - Despliegue en consola de vector de diseño y parámetros dinámicos
#robot.print_parametros()
#robot.print_vector_p()

#Opcional - Gráfica de ciclo límite del plano frontal
#pylab.plot(robot.Qfro[0, :], robot.Qfro[1, :])
#pylab.show()

#Opcional - CGráfica de ciclo límite del plano sagital (pierna izquierda)
#pylab.plot(robot.ql, robot.qldot)
#pylab.show()

# Se calcula geometría y trayectoria de componentes estructurales para simulación
robot.sagital_components()
robot.pbw_kinematics()

# Se calcula el tiempo de simulación
t = np.linspace(robot.t_inicial_S, robot.t_final_S, robot.n_S)

# Se crean los objetos de la ventana y ejes
fig = plt.figure(figsize=(12, 2))
ax = fig.add_subplot(111)
ax.set_xlim((-0.5, 2.5))
ax.set_ylim((-0.15, 0.5))

A_p = [[robot.A_points[0, 0], robot.A_points[0, 1]], [robot.A_points[1, 0], robot.A_points[1, 1]], [robot.A_points[2, 0], robot.A_points[2, 1]], [robot.A_points[3, 0], robot.A_points[3, 1]]]
L_p = [[robot.L_points[0, 0], robot.L_points[0, 1]], [robot.L_points[1, 0], robot.L_points[1, 1]], [robot.L_points[2, 0], robot.L_points[2, 1]], [robot.L_points[3, 0], robot.L_points[3, 1]]]

F_p= np.array([[0, 0]])
for i in range(robot.F_points.shape[0]):
    F_p = np.concatenate((F_p, np.array([[ robot.F_points[i,0], robot.F_points[i,1]] ]) ) )
F_p = np.delete(F_p,0,0)

# Inicialización de componentes
alp=0.75
A_SW = plt.Polygon(A_p, color='darkgray', alpha=alp)
L_SW = plt.Polygon(L_p, color='darkgray', alpha=alp)
F_SW = plt.Polygon(F_p, color='darkgray', alpha=alp)

A_ST = plt.Polygon(A_p, color='dimgray')
L_ST = plt.Polygon(L_p, color='dimgray')
F_ST = plt.Polygon(F_p, color='dimgray')

ax.add_patch(A_SW)
ax.add_patch(L_SW)
ax.add_patch(F_SW)

ax.add_patch(A_ST)
ax.add_patch(L_ST)
ax.add_patch(F_ST)
ax.plot([-1*np.cos(robot.gamma),5*np.cos(robot.gamma)],[1*np.sin(robot.gamma),-5*np.sin(robot.gamma)])
# L_SW = plt.Polygon(L_p)

def init():
    return A_SW, A_ST, L_SW, L_ST, F_SW, F_ST

def pbw_draw(i):
    qrc = robot.Qsag[4, i]
    qst = robot.Qsag[0, i]
    qsw = robot.Qsag[1, i]

    # Transformaciones geométricas
    T1 = robot.dh((np.pi/2) + robot.gamma, 0, robot.Rs, 0)
    T2 = robot.dh(np.pi/2, 0, robot.Rs*(qst - robot.gamma), 0)
    T3 = robot.dh(-robot.gamma + qst + (np.pi/2), 0, robot.dsag, 0)
    T4 = robot.dh(0, 0, robot.bsag - robot.dsag, 0)
    T5 = robot.dh(-qst + qsw, 0, robot.bsag - robot.dsag, 0)

    if qrc <= 0.5:

        P_ST = T1 @ T2 @ T3 @ T4
        P_SW = T1 @ T2 @ T3 @ T5

        P_ST[0,3] = robot.ST_x_sag[0,i]
        P_ST[1,3] = robot.ST_y_sag[0,i]

        P_SW[0,3] = robot.SW_x_sag[0,i]
        P_SW[1,3] = robot.SW_y_sag[0,i]

        P_A_ST = P_ST.dot(robot.A_points.T)
        P_L_ST = P_ST.dot(robot.L_points.T)
        P_F_ST = P_ST.dot(robot.F_points.T)

        P_A_SW = P_SW.dot(robot.A_points.T)
        P_L_SW = P_SW.dot(robot.L_points.T)
        P_F_SW = P_SW.dot(robot.F_points.T)

    else:

        P_SW = T1 @ T2 @ T3 @ T4
        P_ST = T1 @ T2 @ T3 @ T5

        P_ST[0,3] = robot.ST_x_sag[0,i]
        P_ST[1,3] = robot.ST_y_sag[0,i]

        P_SW[0,3] = robot.SW_x_sag[0,i]
        P_SW[1,3] = robot.SW_y_sag[0,i]

        P_A_SW = P_SW.dot(robot.A_points.T)
        P_L_SW = P_SW.dot(robot.L_points.T)
        P_F_SW = P_SW.dot(robot.F_points.T)

        P_A_ST = P_ST.dot(robot.A_points.T)
        P_L_ST = P_ST.dot(robot.L_points.T)
        P_F_ST = P_ST.dot(robot.F_points.T)

    A_SW.set_xy([[P_A_SW[0, 0], P_A_SW[1, 0]], [P_A_SW[0, 1], P_A_SW[1, 1]], [P_A_SW[0, 2], P_A_SW[1, 2]], [P_A_SW[0, 3], P_A_SW[1, 3]]])
    A_ST.set_xy([[P_A_ST[0, 0], P_A_ST[1, 0]], [P_A_ST[0, 1], P_A_ST[1, 1]], [P_A_ST[0, 2], P_A_ST[1, 2]], [P_A_ST[0, 3], P_A_ST[1, 3]]])
    L_SW.set_xy([[P_L_SW[0, 0], P_L_SW[1, 0]], [P_L_SW[0, 1], P_L_SW[1, 1]], [P_L_SW[0, 2], P_L_SW[1, 2]], [P_L_SW[0, 3], P_L_SW[1, 3]]])
    L_ST.set_xy([[P_L_ST[0, 0], P_L_ST[1, 0]], [P_L_ST[0, 1], P_L_ST[1, 1]], [P_L_ST[0, 2], P_L_ST[1, 2]], [P_L_ST[0, 3], P_L_ST[1, 3]]])

    F_SW_points = np.array([[0, 0]]) #shape(4,84)
    F_ST_points = np.array([[0, 0]])

    for i in range(P_F_SW.shape[1]):
        F_SW_points = np.concatenate((F_SW_points, np.array([[ P_F_SW[0,i], P_F_SW[1,i]] ]) ) )
    F_SW_points = np.delete(F_SW_points,0,0)
    # F_ST_points = np.vstack((P_F_ST[:, 0], P_F_ST[:, 1]))
    F_SW.set_xy(F_SW_points)

    for i in range(P_F_ST.shape[1]):
        F_ST_points = np.concatenate((F_ST_points, np.array([[ P_F_ST[0,i], P_F_ST[1,i]] ]) ) )
    F_ST_points = np.delete(F_ST_points,0,0)
    # F_ST_points = np.vstack((P_F_ST[:, 0], P_F_ST[:, 1]))
    F_ST.set_xy(F_ST_points)

    # H_points = np.vstack((P_H[:, 0], P_H[:, 1]))
    # MC_points = np.vstack((P_MC[:, 0], P_MC[:, 1]))
    # print(P_A_SW)
    return A_SW, A_ST, L_SW, L_ST, F_SW, F_ST

anim = FuncAnimation(fig, pbw_draw, init_func=init, frames=range(robot.n_S), interval=1, blit=True)
#plt.axis('scaled')
plt.show()
