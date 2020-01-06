# Módulo que incluye la modelación dinámica de distintos caminadores bípedos
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pylab
import collections


class PBW:
    def __init__(self, p):
        """Inicialización paramétrica
        p = [Rf_0, Rs_1, Fl_2, Fw_3, Al_4, Ah_5, Aw_6, Ll_7, Lh_8,
        Lw_9, Hl_10, Hr_11, Hcp_12, MCro_13, tini_14, gamma_15,
        roF_16, roA_17, roL_18, roH_19, roMC_20]
        """

        # Vector de variables de diseño p

        self.Rf = p[0]
        self.Rs = p[1]
        self.Fl = p[2]
        self.Fw = p[3]
        self.Al = p[4]
        self.Ah = p[5]
        self.Aw = p[6]
        self.Ll = p[7]
        self.Lh = p[8]
        self.Lw = p[9]
        self.Hl = p[10]
        self.Hr = p[11]
        self.Hcp = p[12]
        self.MCro = p[13]
        self.tini = p[14]
        self.gamma = p[15]
        self.roF_i = p[16]
        self.roA_i = p[17]
        self.roL_i = p[18]
        self.roH_i = p[19]
        self.roMC_i = p[20]
        self.roF = 0
        self.roA = 0
        self.roL = 0
        self.roH = 0
        self.roMC = 0

        # Parámetros generales
        self.F_phi = 0.0  # Separación entre pies
        self.Fh = 0.0  # Altura de Foot-F
        self.gamma_ = 0.0

        # Parámetros dinámicos plano sagital
        self.msag = 0.0
        self.COMsag = np.array([0, 0, 0], dtype=float)
        self.Isag = 0.0
        self.gammaS = 0.0
        self.bsag = 0.0
        self.dsag = 0.0
        self.Tsag_p = 0.0
        self.Tsag_sp = 0.0

        # Parámetros dinámicos plano frontal
        self.mfro = 0.0
        self.COMfro = np.array([0, 0, 0], dtype=float)
        self.Ifro = 0.0
        self.betaF = 0.0
        self.afro = 0.0
        self.Tfro = 0.0
        self.phi = 0.0

        # Parámetros integración dinámica
        self.t_inicial_F = 0
        self.t_final_F = 5
        self.t_inicial_S = 0
        self.t_final_S = 20
        self.dt = 0.001
        self.n_F = int((self.t_final_F - self.t_inicial_F)/self.dt)
        self.n_S = int((self.t_final_S - self.t_inicial_S)/self.dt)

        self.Qfro = np.zeros((2, self.n_F))  # Vector de estados plano frontal
        self.Qsag = np.zeros((5, self.n_S))  # Vector de estados plano sagital

        # Proceso de incialización
        self.parametros()

    ##################### DINAMICA DE PLANO FRONTAL ##################################

    def dinamicaF(self):
        """ Integración del modelo dinámico del plano frontal
        """
        self.Qfro[:, 0] = np.array([self.tini, 0])  # Condicion inicial de la simulación

        for i in range(self.n_F-1):
            self.Qfro[:, i+1] = self.Qfro[:, i] + self.dt * np.array([self.Qfro[1, i], self.modeloF(self.Qfro[0, i], self.Qfro[1, i])])
            if (np.sign(self.Qfro[0, i]) != np.sign(self.Qfro[0, i+1])):
                self.Qfro[1, i+1] = self.Qfro[1, i+1] * np.cos(2*np.arctan2(self.Rf*np.sin(self.phi), (self.Rf*np.cos(self.phi)-self.afro)))

        self.Tfro = self.T_fro(self.Qfro, self.t_final_F, self.dt)

        t = np.linspace(self.t_inicial_F, self.t_final_F, self.n_F)
        # Cálculo de periodo de oscilación
        # Tfro = T(Q,t_fin,dt)
        # print("periodo: ", Tfro)

        np.save("frontal_states.npy", self.Qfro)
        # Gráficas de comportamiento dinámico
        #pylab.plot(self.Qfro[0, :], self.Qfro[1, :])
        # pylab.plot(t[0:2500],P1[:-1])
        #pylab.show()

    def modeloF(self, q, qp):
        """Modelo dinámico - Plano Frontal
        """

        g = 9.81
        if(np.abs(q) > self.phi):

            H = self.Ifro + self.mfro*(self.afro**2) + self.mfro * \
                (self.Rf**2) - 2*self.mfro*self.afro*self.Rf*np.cos(q)
            C = self.mfro*self.Rf*self.afro*qp*np.sin(q)
            G = self.mfro*g*self.afro*np.sin(q)

        else:
            if (q > 0):
                alpha = q - self.phi
            else:
                alpha = q + self.phi

            H = self.Ifro + self.mfro*(self.afro**2) + self.mfro * \
                (self.Rf**2) - 2*self.mfro*self.Rf*self.afro*np.cos(q-alpha)
            C = 0
            G = self.mfro*g*(self.afro*np.sin(q) - self.Rf*np.sin(alpha))

        qpp = (-C-G)/H

        return qpp

    def T_fro(self, q, t_fin, dt):
        """Cálculo de periodo de oscilación - Plano Frontal
        """
        Y = np.fft.fft(q[0, :])
        P2 = np.abs(Y/(t_fin/dt))
        P1 = P2[0:int((t_fin/dt)/2+1)]
        P1[1:-2] = 2*P1[1:-2]
        f = (1/dt)*np.arange((t_fin/dt)/2+1)/(t_fin/dt)
        index = np.argmax(P1)
        T = 1/f[index]
        return T

    ###################### DINÁMICA DE PLANO SAGITAL ###############################
    def dinamicaS(self):
        """ Integracion del modelo dinámico del plano sagital
        """
        self.gamma_ = -self.gamma
        g = 9.81
        ti = self.Tfro/2

        # Valores de simulación de modelo dinámico
        count = 0
        tcolision = 0
        cflag = 0

        # Estados de colisión
        Q0 = []
        Q1 = []
        Qp0 = []
        Qp1 = []

        for i in range(self.n_S-1):

            qpp0, qpp1 = self.modeloS(self.Qsag[0:2, i], self.Qsag[2:4, i])
            self.Qsag[:, i+1] = self.Qsag[:, i] + self.dt * np.array([self.Qsag[2, i], self.Qsag[3, i], qpp0, qpp1, 0])

            if (i - tcolision >= ti/self.dt):
                self.Qsag[0:2, i+1], self.Qsag[2:4, i +1] = self.colision(self.Qsag[0:2, i+1], self.Qsag[2:4, i+1])
                cflag = cflag + 1
                modulo = np.remainder(cflag, 2)
                self.Qsag[4, i+1] = 1 - self.Qsag[4, i+1]
                tcolision = i

                if (modulo == 0):
                    Q0.append(self.Qsag[0, i+1])
                    Q1.append(self.Qsag[1, i+1])
                    Qp0.append(self.Qsag[2, i+1])
                    Qp1.append(self.Qsag[3, i+1])

        ql = (1 - self.Qsag[4, :]) * self.Qsag[0, :] + self.Qsag[4, :] * self.Qsag[1, :]
        qldot = (1 - self.Qsag[4, :]) * self.Qsag[2, :] + self.Qsag[4, :] * self.Qsag[3, :]
        qr = self.Qsag[4, :] * self.Qsag[0, :] + (1 - self.Qsag[4, :]) * self.Qsag[1, :]
        qrdot = self.Qsag[4, :] * self.Qsag[2, :] + (1 - self.Qsag[4, :]) * self.Qsag[3, :]

        self.ql = ql
        self.qr = qr
        self.qldot = qldot
        self.qrdot = qrdot

        t = np.linspace(self.t_inicial_S, self.t_final_S, self.n_S)

        self.Tsag_p = self.T_sag(ql, self.t_final_S, self.dt)

        # print("periodo: ", Tsag)
        # Gráficas de comportamiento dinámico
        #pylab.plot(ql, qldot)
        #pylab.plot(qr, qrdot)
        #pylab.plot(t, self.Qsag[0, :])

        # pylab.plot(t[0:10000],P1[:-1])
        #pylab.show()

    def modeloS(self, q, qp):
        """Modelo dinámico - Plano Sagital
        """
        g = 9.81
        tst = q[0]
        tsw = q[1]
        tstp = qp[0]
        tswp = qp[1]

        # Matriz de Inercia
        M11 = self.Isag + self.msag*self.bsag**2 + self.msag*self.dsag**2 + 2*self.msag * \
            self.Rs**2 - 2*self.msag*self.Rs*(self.bsag+self.dsag)*np.cos(tst-self.gamma_)
        M12 = self.msag*(self.bsag-self.dsag)*(self.dsag *
                                               np.cos(tst-tsw)-self.Rs*np.cos(tsw-self.gamma_))
        M21 = M12
        M22 = self.Isag + self.msag*(self.bsag-self.dsag)**2

        # Matriz de Coriolis
        C11 = self.msag*self.Rs*(self.bsag+self.dsag)*np.sin(tst-self.gamma_) * \
            tstp + 0.5*self.msag*self.dsag*(self.bsag-self.dsag)*np.sin(tst-tsw)*tswp
        C12 = self.msag*(self.bsag-self.dsag)*(self.dsag*np.sin(tst-tsw) *
                                               (tswp-0.5*tstp)+self.Rs*np.sin(tsw-self.gamma_)*tswp)
        C21 = self.msag*(self.bsag-self.dsag)*(self.dsag*np.sin(tst-tsw) *
                                               (tstp-0.5*tswp)-0.5*self.Rs*np.sin(tsw-self.gamma_)*tswp)
        C22 = 0.5*self.msag*(self.bsag-self.dsag)*(self.dsag*np.sin(tst -
                                                                    self.gamma_)+self.Rs*np.sin(tsw-self.gamma_))*tstp

        # Vector de fuerzas gravitatorias
        G1 = -self.msag*g*2*self.Rs*np.sin(self.gamma_)+self.msag * \
            g*(self.bsag+self.dsag)*np.sin(tst)
        G2 = self.msag*g*(self.bsag-self.dsag)*np.sin(tsw)

        M = np.array([[M11, M12], [M21, M22]])
        C = np.array([[C11, C12], [C21, C22]])
        G = np.array([G1, G2])

        qpp_1 = np.linalg.inv(M)
        qpp_2 = - C.dot(qp) - G
        qpp_ = qpp_1.dot(qpp_2)

        qpp0 = qpp_[0]
        qpp1 = qpp_[1]

        return qpp0, qpp1

    def colision(self, q, qp):
        """ Modelo de colisión - Plano Sagital
        """
        tst = q[0]
        tsw = q[1]
        tstp = qp[0]
        tswp = qp[1]

        # Matriz de momentos - instante anterior
        Om11 = 2*self.bsag*self.dsag*np.cos(tsw-tst)-(self.bsag+self.dsag)*self.Rs*np.cos(
            tsw-self.gamma_) - 2*self.bsag*self.Rs*np.cos(tst-self.gamma_) + 2*self.Rs**2 + self.bsag**2 - self.bsag*self.dsag
        Om12 = (self.bsag-self.dsag)*(self.bsag-self.Rs*np.cos(tsw-self.gamma_))
        Om21 = (self.bsag-self.dsag)*(self.bsag-self.Rs*np.cos(tsw-self.gamma_))
        Om22 = 0
        Om = np.array([[Om11, Om12], [Om21, Om22]])

        # Matriz de momentos - instante posterior
        Op11 = (self.bsag-self.dsag)*(self.dsag*np.cos(tst-tsw) -
                                      self.Rs*np.cos(tst-self.gamma_)+(self.bsag-self.dsag))
        Op12 = - self.Rs*(self.bsag-self.dsag)*np.cos(tst-self.gamma_) - self.Rs*(self.bsag+2*self.dsag)*np.cos(tsw-self.gamma_) + self.dsag**2 + \
            2*self.Rs**2 + self.Rs*self.bsag * \
            np.cos(tsw+self.gamma_) - self.bsag**2*np.cos(2*tsw) + \
            self.dsag*(self.bsag-self.dsag)*np.cos(tst-tsw)
        Op21 = (self.bsag-self.dsag)**2
        Op22 = (self.bsag-self.dsag)*(self.dsag*np.cos(tst-tsw) - self.Rs*np.cos(tst-self.gamma_))
        Op = np.array([[Op11, Op12], [Op21, Op22]])

        O = np.linalg.inv(Op).dot(Om)
        O = O.dot(qp)

        q[0] = tsw
        q[1] = tst
        qp[0] = O[1]
        qp[1] = O[0]

        return q, qp

    def T_sag(self, q, t_fin, dt):
        """Cálculo periodo de oscilación - Plano Sagital
        """
        Y = np.fft.fft(q)
        P2 = np.abs(Y/(t_fin/dt))
        P1 = P2[0:int((t_fin/dt)/2+1)]
        P1[1:-2] = 2*P1[1:-2]
        f = (1/dt)*np.arange((t_fin/dt)/2+1)/(t_fin/dt)

        index = np.argmax(P1)
        T = 1/f[index]

        return T

    def parametros(self):
        """Cálculo de parámetros estructurales y dinámicos de ambos planos del bípedo
        """

        # Densidades
        roSt = 7850
        roAl = 2700

        # Dimensiones Rodamiento-B
        Bri = 0.005
        Bro = 0.007175
        Bw = 0.01262

        # Dimensiones Motor-M
        Mbl = 0.0747
        Mbr = 0.0125
        Msl = 0.0125
        Msr = 0.0020
        M_m = 0.1060

        # Dimensiones Motor Case - MC
        MCcl = 0.0030
        MCl = Mbl
        MCsr = 0.0055
        MCsl = 0.0200
        MCri = Mbr
        MCcr = MCsr

        # Dimensiones Acoplador - C
        Cro = 0.0095
        Cri = 0.0020
        Cw = 0.0040
        roC = roAl

        # Asignación de valores de densidad
        # ros = [450, 1050, 1250, 2700]
        ros = np.array([0.3*1250, 0.5*1250, 0.8*1250], dtype=float)
        # Densidad Foot-F
        if self.roF_i == 1:
            self.roF = ros[0]
        elif self.roF_i == 2:
            self.roF = ros[1]
        elif self.roF_i == 3:
            self.roF = ros[2]
        elif self.roF_i == 4:
            self.roF = ros[3]
        else:
            print("Falla asignación material Foot-F")

        # Densidad Ankle-A
        if self.roA_i == 1:
            self.roA = ros[0]
        elif self.roA_i == 2:
            self.roA = ros[1]
        elif self.roA_i == 3:
            self.roA = ros[2]
        elif self.roA_i == 4:
            self.roA = ros[3]
        else:
            print("Falla asignación material Ankle-A")

        # Densidad Leg-L
        if self.roL_i == 1:
            self.roL = ros[0]
        elif self.roL_i == 2:
            self.roL = ros[1]
        elif self.roL_i == 3:
            self.roL = ros[2]
        elif self.roL_i == 4:
            self.roL = ros[3]
        else:
            print("Falla asignación material Leg-L")

        # Densidad Hip-H
        if self.roH_i == 1:
            self.roH = ros[0]
        elif self.roH_i == 2:
            self.roH = ros[1]
        elif self.roH_i == 3:
            self.roH = ros[2]
        elif self.roH_i == 4:
            self.roH = ros[3]
        else:
            print("Falla asignación material Hip-H")

        # Densidad Motor Case - MC
        if self.roMC_i == 1:
            self.roMC = ros[0]
        elif self.roMC_i == 2:
            self.roMC = ros[1]
        elif self.roMC_i == 3:
            self.roMC = ros[2]
        elif self.roMC_i == 4:
            self.roMC = ros[3]
        else:
            print("Falla asignación material Motor Case - MC")

        # Cálculo de parámetros cinemáticos complementarios
        self.F_phi = self.Hl + 2*MCcl + MCl + Cw + 0.5 * (self.Ll - self.Fl)
        self.phi = np.arcsin(self.F_phi/self.Rf)

        # Parámetros para evaluación de sobregiro en plano sagital
        lam_ = np.arcsin(self.Fw/self.Rs)
        theta_ = (np.pi - lam_)/2
        d_ = (np.sin(lam_)/np.sin(theta_)) * self.Rs
        e_ = self.Lh - self.Hcp + self.Ah + self.F_phi
        L_ = np.sqrt(e_**2 + d_**2 - 2*d_*e_*np.cos(theta_))
        h_ = L_ - e_

        # Parámetros de sobregiro en el plano frontal
        fact_ = 1
        qf_ = self.tini/2
        hp_ = self.Rf - self.Rf*np.cos(qf_)
        dif_h = hp_ - h_

        self.betaF = np.arcsin((self.Fl + self.F_phi)/self.Rf) - self.phi
        self.gammaS = np.arcsin(d_*np.sin(theta_)/L_)

        # Factibilidad estructural
        if dif_h < 0:
            fact_ = 0

        # Cálculo de volúmenes
        A_vol = self.Al * self.Ah * self.Aw

        L_vol = self.Ll * self.Lh * self.Lw

        C_vol_1 = np.pi * (Cro**2) * Cw
        C_vol_2 = -np.pi * (Cri**2) * Cw
        C_vol = C_vol_1 + C_vol_2

        M_vol_1 = np.pi * (Mbr**2) * Mbl
        M_vol_2 = np.pi * (Msr**2) * Msl
        M_vol = M_vol_1 + M_vol_2

        MC_vol_1 = np.pi * (self.MCro**2) * MCl
        MC_vol_2 = -np.pi * (MCri**2) * MCl
        MC_vol_3 = np.pi * (self.MCro**2) * MCcl
        MC_vol_4 = -np.pi * (MCcr**2) * MCcl
        MC_vol = MC_vol_1 + MC_vol_2 + 2*(MC_vol_3 + MC_vol_4)

        B_vol = np.pi * ((Bro**2)-(Bri**2)) * Bw

        H_vol_1 = np.pi * (self.Hr**2) * self.Hl
        H_vol_2 = -np.pi * (Bro**2) * Bw
        H_vol = H_vol_1 + H_vol_2

        S_vol = np.pi * (MCsr**2) * MCsl

        # Cálculo de masas
        A_m = self.roA * A_vol
        L_m = self.roL * L_vol
        C_m = roC * C_vol
        MC_m = self.roMC * MC_vol
        B_m = roAl * B_vol
        H_m = self.roH * H_vol
        S_m = self.roH * S_vol

        # Obtención paránetros de Foot-F

        F_COM, F_I, F_m = self.foot()

        # Asignación de masas Frontal y Sagital
        self.msag = F_m + A_m + L_m + B_m + H_m + C_m + MC_m + M_m + S_m
        self.mfro = 2*self.msag

        # Coordendas de COM por componente - Plano sagital
        # FOOT-F
        F_x = F_COM[0]
        F_y = F_COM[1]
        F_z = F_COM[2]

        # ANKLE- A
        A_x = self.Fl/2
        A_y = self.Ah/2
        A_z = 0

        # LEG-L
        L_x = self.Fl/2
        L_y = self.Ah + (self.Lh/2)
        L_z = 0

        # BEARING-B
        B_x = (self.Fl/2) - (self.Ll/2) - Cw - 2*MCcl - MCl - (Bw/2)
        B_y = self.Ah + self.Lh - self.Hcp
        B_z = 0

        # HIP-H
        H_x1_aux = (-self.Hl/2)
        H_x2_aux = (-Bw/2)
        H_x_aux = (1/H_vol)*(H_x1_aux*H_vol_1 + H_x2_aux*H_vol_2)
        H_x = (self.Fl/2) - (self.Ll/2) - Cw - 2*MCcl - MCl + H_x_aux
        H_y = self.Ah + self.Lh - self.Hcp
        H_z = 0

        # COUPLER-P
        C_x = (self.Fl/2) - (self.Fl/2) - (Cw/2)
        C_y = self.Ah + self.Lh - self.Hcp
        C_z = 0

        # MOTOR CASE - MC
        MC_x = (self.Fl/2) - (self.Fl/2) - Cw - MCcl - (MCl/2)
        MC_y = self.Ah + self.Lh - self.Hcp
        MC_z = 0

        # SHAFT-S
        S_x = (self.Fl/2) - (self.Ll/2) - Cw - MCcl - MCl - (MCsl/2)
        S_y = self.Ah + self.Lh - self.Hcp
        S_z = 0

        # MOTOR-M
        M_x1_aux = (-Msl/2)
        M_x2_aux = -Msl - (Mbl/2)
        M_x_aux = (1/M_vol)*(M_x1_aux*M_vol_2 + M_x2_aux*M_vol_1)
        M_x = (self.Fl/2) - (self.Ll/2) - Cw - MCcl + Msl + M_x_aux
        M_y = self.Ah + self.Lh - self.Hcp
        M_z = 0

        # Determinación COM - PLANO SAGITAL
        self.COMsag[0] = (1/self.msag)*(F_m*F_x + A_m*A_x + L_m*L_x + C_m*C_x +
                                        MC_m*MC_x + M_m*M_x + B_m*B_x + H_m*H_x + S_m*S_x)
        self.COMsag[1] = (1/self.msag)*(F_m*F_y + A_m*A_y + L_m*L_y + C_m*C_y +
                                        MC_m*MC_y + M_m*M_y + B_m*B_y + H_m*H_y + S_m*S_y)
        self.COMsag[2] = (1/self.msag)*(F_m*F_z + A_m*A_z + L_m*L_z + C_m*C_z +
                                        MC_m*MC_z + M_m*M_z + B_m*B_z + H_m*H_z + S_m*S_z)

        # Determinación de momentos de inercia - PLANO SAGITAL

        # Magnitudes para Teorema de Ejes Paralelos
        F_par_sag = F_m*np.power(np.sqrt((F_y - self.COMsag[1])**2 + (F_z - self.COMsag[2])**2), 2)
        A_par_sag = A_m*np.power(np.sqrt((A_y - self.COMsag[1])**2 + (A_z - self.COMsag[2])**2), 2)
        L_par_sag = L_m*np.power(np.sqrt((L_y - self.COMsag[1])**2 + (L_z - self.COMsag[2])**2), 2)
        C_par_sag = C_m*np.power(np.sqrt((C_y - self.COMsag[1])**2 + (C_z - self.COMsag[2])**2), 2)
        MC_par_sag = MC_m * \
            np.power(np.sqrt((MC_y - self.COMsag[1])**2 + (MC_z - self.COMsag[2])**2), 2)
        M_par_sag = M_m*np.power(np.sqrt((M_y - self.COMsag[1])**2 + (M_z - self.COMsag[2])**2), 2)
        B_par_sag = B_m*np.power(np.sqrt((B_y - self.COMsag[1])**2 + (B_z - self.COMsag[2])**2), 2)
        H_par_sag = H_m*np.power(np.sqrt((H_y - self.COMsag[1])**2 + (H_z - self.COMsag[2])**2), 2)
        S_par_sag = S_m*np.power(np.sqrt((S_y - self.COMsag[1])**2 + (S_z - self.COMsag[2])**2), 2)

        # Momentos de inercia con respecto a eje paralelo
        F_Isag = F_I[0] + F_par_sag
        A_Isag = (1/12)*A_m*(self.Ah**2 + self.Aw**2) + A_par_sag
        L_Isag = (1/12)*L_m*(self.Lh**2 + self.Lw**2) + L_par_sag
        C_Isag = (1/2)*roAl*np.pi*(Cw*(Cro**4) - Cw*(Cri**4)) + C_par_sag
        MC_Isag_1 = 2*(1/2)*np.pi*self.roMC*(MCcl*(self.MCro**4) - MCcl*(MCsr**4))
        MC_Isag_2 = (1/2)*np.pi*self.roMC*(MCl*(self.MCro**4) - MCl*(MCri**4))
        MC_Isag = MC_Isag_1 + MC_Isag_2 + MC_par_sag
        M_Isag = (1/2)*(0.2)*M_m*(Msr**2) + (1/2)*(0.8)*M_m*(Mbr**2) + M_par_sag
        B_Isag = (1/2)*B_m*(Bro**2 + Bri**2) + B_par_sag
        H_Isag = (1/2)*np.pi*self.roH*(self.Hl*(self.Hr**4) - Bw*(Bro**4)) + H_par_sag
        S_Isag = (1/2)*S_m*(Bri**2) + S_par_sag

        self.Isag = F_Isag + A_Isag + L_Isag + C_Isag + MC_Isag + M_Isag + B_Isag + H_Isag + S_Isag

        # Coordendas de COM por componente - Plano Frontal
        # FOOT-F / Left
        F_x_le = -self.F_phi - F_COM[0]
        F_y_le = F_COM[1]
        F_z_le = 0

        # ANKLE-A / Left
        A_x_le = -self.F_phi - (self.Fl/2)
        A_y_le = self.Ah/2
        A_z_le = 0

        # LEG-L /Left
        L_x_le = -self.F_phi - (self.Fl/2)
        L_y_le = self.Ah + (self.Lh/2)
        L_z_le = 0

        # COUPLER-C /Left
        C_x_le = -self.F_phi - (self.Fl/2) + (self.Ll/2) + (Cw/2)
        C_y_le = self.Ah + self.Lh - self.Hcp
        C_z_le = 0

        # MOTOR-M / Left
        M_x_le1 = Msl + (Mbl/2)
        M_x_le2 = Msl/2
        M_x_le_aux = (1/M_vol)*(M_x_le1*M_vol_1 + M_x_le2*M_vol_2)
        M_x_le = -self.F_phi - (self.Fl/2) + (self.Ll/2) + Cw + MCcl - Msl + M_x_le_aux
        M_y_le = self.Ah + self.Lh - self.Hcp
        M_z_le = 0

        # MOTOR CASE-MC / Left
        MC_x_le = -self.F_phi - (self.Fl/2) + (self.Ll/2) + Cw + MCcl + (MCl/2)
        MC_y_le = self.Ah + self.Lh - self.Hcp
        MC_z_le = 0

        # HIP-H / Left
        H_x_le1 = self.Hl/2
        H_x_le2 = Bw/2
        H_x_le_aux = (1/H_vol)*(H_x_le1*H_vol_1 + H_x_le2*H_vol_2)
        H_x_le = -self.F_phi - (self.Fl/2) + (self.Ll/2) + Cw + 2*MCcl + MCl + H_x_le_aux
        H_y_le = self.Ah + self.Lh - self.Hcp
        H_z_le = 0

        # BEARING-B / Left
        B_x_le = -self.F_phi - (self.Fl/2) + (self.Ll/2) + Cw + 2*MCcl + MCl + (Bw/2)
        B_y_le = self.Ah + self.Lh - self.Hcp
        B_z_le = 0

        # SHAFT-S / Left
        S_x_le = -self.F_phi - (self.Fl/2) + (self.Ll/2) + Cw + MCcl + MCl + (MCsl/2)
        S_y_le = self.Ah + self.Lh - self.Hcp
        S_z_le = 0

        ############## PIERNA DERECHA ##############

        # FOOT-F / Right
        F_x_ri = self.F_phi + F_COM[0]
        F_y_ri = F_COM[1]
        F_z_ri = 0

        # ANKLE-A / Right
        A_x_ri = self.F_phi + (self.Fl/2)
        A_y_ri = self.Ah/2
        A_z_ri = 0

        # LEG-L / Right
        L_x_ri = self.F_phi + (self.Fl/2)
        L_y_ri = self.Ah + (self.Lh/2)
        L_z_ri = 0

        # COUPLER-C / Right
        C_x_ri = self.F_phi + (self.Fl/2) - (self.Ll/2) + (Cw/2)
        C_y_ri = self.Ah + self.Lh - self.Hcp
        C_z_ri = 0

        # MOTOR-M / Right
        M_x_ri1 = -Msl - (Mbl/2)
        M_x_ri2 = -Msl/2
        M_x_ri_aux = (1/M_vol)*(M_x_ri1*M_vol_1 + M_x_ri2*M_vol_2)
        M_x_ri = self.F_phi + (self.Fl/2) - (self.Ll/2) - Cw - MCcl + Msl + M_x_ri_aux
        M_y_ri = self.Ah + self.Lh - self.Hcp
        M_z_ri = 0

        # MOTOR CASE-MC / Right
        MC_x_ri = self.F_phi + (self.Fl/2) - (self.Ll/2) - Cw - MCcl - (MCl/2)
        MC_y_ri = self.Ah + self.Lh - self.Hcp
        MC_z_ri = 0

        # HIP-H / Right
        H_x_ri1 = -self.Hl/2
        H_x_ri2 = -Bw/2
        H_x_ri_aux = (1/H_vol)*(H_x_ri1*H_vol_1 + H_x_ri2*H_vol_2)
        H_x_ri = self.F_phi + (self.Fl/2) - (self.Ll/2) - Cw - 2*MCcl - MCl + H_x_ri_aux
        H_y_ri = self.Ah + self.Lh - self.Hcp
        H_z_ri = 0

        # BEARING-B / Right
        B_x_ri = self.F_phi + (self.Fl/2) - (self.Ll/2) - Cw - 2*MCcl - MCl - (Bw/2)
        B_y_ri = self.Ah + self.Lh - self.Hcp
        B_z_ri = 0

        # SHAFT-S / Right
        S_x_ri = self.F_phi + (self.Fl/2) - (self.Ll/2) - Cw - MCcl - MCl - (MCsl/2)
        S_y_ri = self.Ah + self.Lh - self.Hcp
        S_z_ri = 0

        # Coordendas del COM - Plano frontal
        x_le = (F_m*F_x_le) + (A_m*A_x_le) + (L_m*L_x_le) + (C_m*C_x_le) + \
            (M_m*M_x_le) + (MC_m*MC_x_le) + (H_m*H_x_le) + (B_m*B_x_le) + (S_m*S_x_le)
        y_le = (F_m*F_y_le) + (A_m*A_y_le) + (L_m*L_y_le) + (C_m*C_y_le) + \
            (M_m*M_y_le) + (MC_m*MC_y_le) + (H_m*H_y_le) + (B_m*B_y_le) + (S_m*S_y_le)
        z_le = (F_m*F_z_le) + (A_m*A_z_le) + (L_m*L_z_le) + (C_m*C_z_le) + \
            (M_m*M_z_le) + (MC_m*MC_z_le) + (H_m*H_z_le) + (B_m*B_z_le) + (S_m*S_z_le)

        x_ri = (F_m*F_x_ri) + (A_m*A_x_ri) + (L_m*L_x_ri) + (C_m*C_x_ri) + \
            (M_m*M_x_ri) + (MC_m*MC_x_ri) + (H_m*H_x_ri) + (B_m*B_x_ri) + (S_m*S_x_ri)
        y_ri = (F_m*F_y_ri) + (A_m*A_y_ri) + (L_m*L_y_ri) + (C_m*C_y_ri) + \
            (M_m*M_y_ri) + (MC_m*MC_y_ri) + (H_m*H_y_ri) + (B_m*B_y_ri) + (S_m*S_y_ri)
        z_ri = (F_m*F_z_ri) + (A_m*A_z_ri) + (L_m*L_z_ri) + (C_m*C_z_ri) + \
            (M_m*M_z_ri) + (MC_m*MC_z_ri) + (H_m*H_z_ri) + (B_m*B_z_ri) + (S_m*S_z_ri)

        self.COMfro[0] = (1/self.mfro)*(x_le + x_ri)
        self.COMfro[1] = (1/self.mfro)*(y_le + y_ri)
        self.COMfro[2] = (1/self.mfro)*(z_le + z_ri)

        # Determinación de momentos de inercia - Plano frontal

        # Magnitudes para Teorema de Ejes Paralelos - Pierna Izquierda
        F_par_le = F_m * \
            np.power(np.sqrt((F_x_le - self.COMfro[0])**2 + (F_y_le - self.COMfro[1])**2), 2)
        A_par_le = A_m * \
            np.power(np.sqrt((A_x_le - self.COMfro[0])**2 + (A_y_le - self.COMfro[1])**2), 2)
        L_par_le = L_m * \
            np.power(np.sqrt((L_x_le - self.COMfro[0])**2 + (L_y_le - self.COMfro[1])**2), 2)
        C_par_le = C_m * \
            np.power(np.sqrt((C_x_le - self.COMfro[0])**2 + (C_y_le - self.COMfro[1])**2), 2)
        M_par_le = M_m * \
            np.power(np.sqrt((M_x_le - self.COMfro[0])**2 + (M_y_le - self.COMfro[1])**2), 2)
        MC_par_le = MC_m * \
            np.power(np.sqrt((MC_x_le - self.COMfro[0])**2 + (MC_y_le - self.COMfro[1])**2), 2)
        H_par_le = H_m * \
            np.power(np.sqrt((H_x_le - self.COMfro[0])**2 + (H_y_le - self.COMfro[1])**2), 2)
        B_par_le = B_m * \
            np.power(np.sqrt((B_x_le - self.COMfro[0])**2 + (B_y_le - self.COMfro[1])**2), 2)
        S_par_le = S_m * \
            np.power(np.sqrt((S_x_le - self.COMfro[0])**2 + (S_y_le - self.COMfro[1])**2), 2)

        # Magnitudes para Teorema de Ejes Paralelos - Pierna Derecha
        F_par_ri = F_m * \
            np.power(np.sqrt((F_x_ri - self.COMfro[0])**2 + (F_y_ri - self.COMfro[1])**2), 2)
        A_par_ri = A_m * \
            np.power(np.sqrt((A_x_ri - self.COMfro[0])**2 + (A_y_ri - self.COMfro[1])**2), 2)
        L_par_ri = L_m * \
            np.power(np.sqrt((L_x_ri - self.COMfro[0])**2 + (L_y_ri - self.COMfro[1])**2), 2)
        C_par_ri = C_m * \
            np.power(np.sqrt((C_x_ri - self.COMfro[0])**2 + (C_y_ri - self.COMfro[1])**2), 2)
        M_par_ri = M_m * \
            np.power(np.sqrt((M_x_ri - self.COMfro[0])**2 + (M_y_ri - self.COMfro[1])**2), 2)
        MC_par_ri = MC_m * \
            np.power(np.sqrt((MC_x_ri - self.COMfro[0])**2 + (MC_y_ri - self.COMfro[1])**2), 2)
        H_par_ri = H_m * \
            np.power(np.sqrt((H_x_ri - self.COMfro[0])**2 + (H_y_ri - self.COMfro[1])**2), 2)
        B_par_ri = B_m * \
            np.power(np.sqrt((B_x_ri - self.COMfro[0])**2 + (B_y_ri - self.COMfro[1])**2), 2)
        S_par_ri = S_m * \
            np.power(np.sqrt((S_x_ri - self.COMfro[0])**2 + (S_y_ri - self.COMfro[1])**2), 2)

        # Momentos de inercia respecto al eje paralelo - Pierna Izquierda
        F_Ifro_le = F_I[2] + F_par_le
        A_Ifro_le = (1/12)*A_m*(self.Al**2 + self.Ah**2) + A_par_le
        L_Ifro_le = (1/12)*L_m*(self.Ll**2 + self.Lh**2) + L_par_le

        C_Ifro_le1 = (1/4)*roC*C_vol_1*(Cro**2) + (1/12)*roC*C_vol_1*(Cw**2)
        C_Ifro_le2 = (1/4)*roC*C_vol_2*(Cri**2) + (1/12)*roC*C_vol_2*(Cw**2)
        C_Ifro_le = C_Ifro_le1 + C_Ifro_le2 + C_par_le

        M_x1_aux = Msl + (Mbl/2)
        M_y1_aux = 0
        M_x2_aux = Msl/2
        M_y2_aux = 0
        M_Ifro_le1 = (1/4)*(0.8*M_m)*(Mbr**2) + (1/12)*(0.8*M_m)*(Mbl**2) + (0.8*M_m) * \
            np.power(np.sqrt((M_x1_aux - M_x_le_aux)**2 + (M_y1_aux - 0)**2), 2)
        M_Ifro_le2 = (1/4)*(0.2*M_m)*(Msr**2) + (1/12)*(0.2*M_m)*(Msl**2) + (0.2*M_m) * \
            np.power(np.sqrt((M_x2_aux - M_x_le_aux)**2 + (M_y2_aux - 0)**2), 2)
        M_Ifro_le = M_Ifro_le1 + M_Ifro_le2 + M_par_le

        MC_x1_aux = MCcl/2
        MC_x2_aux = MCcl/2
        MC_x3_aux = MCcl + (MCl/2)
        MC_x4_aux = MCcl + (MCl/2)
        MC_x5_aux = MCcl + MCl + (MCcl/2)
        MC_x6_aux = MCcl + MCl + (MCcl/2)
        MC_y1_aux = 0
        MC_y2_aux = 0
        MC_y3_aux = 0
        MC_y4_aux = 0
        MC_y5_aux = 0
        MC_y6_aux = 0
        MC_x_le_aux = MCcl + (MCl/2)

        MC_Ifro_le1 = (1/4)*self.roMC*MC_vol_3*(self.MCro**2) + (1/12)*self.roMC*MC_vol_3*(MCcl**2) + \
            self.roMC*MC_vol_3*np.power(np.sqrt((MC_x1_aux - MC_x_le_aux)
                                                ** 2 + (MC_y1_aux - 0)**2), 2)
        MC_Ifro_le2 = (1/4)*self.roMC*MC_vol_4*(MCcr**2) + (1/12)*self.roMC*MC_vol_4*(MCcl**2) + \
            self.roMC*MC_vol_4*np.power(np.sqrt((MC_x2_aux - MC_x_le_aux)
                                                ** 2 + (MC_y2_aux - 0)**2), 2)
        MC_Ifro_le3 = (1/4)*self.roMC*MC_vol_1*(self.MCro**2) + (1/12)*self.roMC*MC_vol_1*(MCl**2) + \
            self.roMC*MC_vol_1*np.power(np.sqrt((MC_x3_aux - MC_x_le_aux)
                                                ** 2 + (MC_y3_aux - 0)**2), 2)
        MC_Ifro_le4 = (1/4)*self.roMC*MC_vol_2*(MCri**2) + (1/12)*self.roMC*MC_vol_2*(MCl**2) + \
            self.roMC*MC_vol_4*np.power(np.sqrt((MC_x4_aux - MC_x_le_aux)
                                                ** 2 + (MC_y4_aux - 0)**2), 2)
        MC_Ifro_le5 = (1/4)*self.roMC*MC_vol_3*(self.MCro**2) + (1/12)*self.roMC*MC_vol_3*(MCcl**2) + \
            self.roMC*MC_vol_3*np.power(np.sqrt((MC_x5_aux - MC_x_le_aux)
                                                ** 2 + (MC_y5_aux - 0)**2), 2)
        MC_Ifro_le6 = (1/4)*self.roMC*MC_vol_4*(MCcr**2) + (1/12)*self.roMC*MC_vol_4*(MCcl**2) + \
            self.roMC*MC_vol_4*np.power(np.sqrt((MC_x6_aux - MC_x_le_aux)
                                                ** 2 + (MC_y6_aux - 0)**2), 2)
        MC_Ifro_le = MC_Ifro_le1 + MC_Ifro_le2 + MC_Ifro_le3 + \
            MC_Ifro_le4 + MC_Ifro_le5 + MC_Ifro_le6 + MC_par_le

        H_x1_aux = self.Hl/2
        H_x2_aux = Bw/2
        H_y1_aux = 0
        H_y2_aux = 0

        H_Ifro_le1 = (1/4)*self.roH*H_vol_1*(self.Hr**2) + (1/12)*self.roH*H_vol_1*(self.Hl**2) + \
            self.roH*H_vol_1*np.power(np.sqrt((H_x1_aux - H_x_le_aux)**2 + (H_y1_aux - 0)**2), 2)
        H_Ifro_le2 = (1/4)*self.roH*H_vol_2*(Bro**2) + (1/12)*self.roH*H_vol_2*(Bw**2) + \
            self.roH*H_vol_2*np.power(np.sqrt((H_x2_aux - H_x_le_aux)**2 + (H_y2_aux - 0)**2), 2)
        H_Ifro_le = H_Ifro_le1 + H_Ifro_le2 + H_par_le

        B_Ifro_le = (B_m/12)*(3*(Bro**2 + Bri**2) + Bw**2) + B_par_le
        S_Ifro_le = (1/4)*S_m*(Bri**2) + (1/12)*S_m*(MCsl**2) + S_par_le

        # Momentos de inercia respecto al eje paralelo - Pierna Derecha
        F_Ifro_ri = F_I[2] + F_par_ri
        A_Ifro_ri = (1/12)*A_m*(self.Al**2 + self.Ah**2) + A_par_ri
        L_Ifro_ri = (1/12)*L_m*(self.Ll**2 + self.Lh**2) + L_par_ri

        C_Ifro_ri1 = (1/4)*roC*C_vol_1*(Cro**2) + (1/12)*roC*C_vol_1*(Cw**2)
        C_Ifro_ri2 = (1/4)*roC*C_vol_2*(Cri**2) + (1/12)*roC*C_vol_2*(Cw**2)
        C_Ifro_ri = C_Ifro_ri1 + C_Ifro_ri2 + C_par_ri

        M_x1_aux = -Msl - (Mbl/2)
        M_y1_aux = 0
        M_x2_aux = -Msl/2
        M_y2_aux = 0
        M_Ifro_ri1 = (1/4)*(0.8*M_m)*(Mbr**2) + (1/12)*(0.8*M_m)*(Mbl**2) + (0.8*M_m) * \
            np.power(np.sqrt((M_x1_aux - M_x_ri_aux)**2 + (M_y1_aux - 0)**2), 2)
        M_Ifro_ri2 = (1/4)*(0.2*M_m)*(Msr**2) + (1/12)*(0.2*M_m)*(Msl**2) + (0.2*M_m) * \
            np.power(np.sqrt((M_x2_aux - M_x_ri_aux)**2 + (M_y2_aux - 0)**2), 2)
        M_Ifro_ri = M_Ifro_ri1 + M_Ifro_ri2 + M_par_ri

        MC_x1_aux = -MCcl/2
        MC_x2_aux = -MCcl/2
        MC_x3_aux = -MCcl - (MCl/2)
        MC_x4_aux = -MCcl - (MCl/2)
        MC_x5_aux = -MCcl - MCl - (MCcl/2)
        MC_x6_aux = -MCcl - MCl - (MCcl/2)
        MC_y1_aux = 0
        MC_y2_aux = 0
        MC_y3_aux = 0
        MC_y4_aux = 0
        MC_y5_aux = 0
        MC_y6_aux = 0
        MC_x_ri_aux = -MCcl - (MCl/2)

        MC_Ifro_ri1 = (1/4)*self.roMC*MC_vol_3*(self.MCro**2) + (1/12)*self.roMC*MC_vol_3*(MCcl**2) + \
            self.roMC*MC_vol_3*np.power(np.sqrt((MC_x1_aux - MC_x_ri_aux)
                                                ** 2 + (MC_y1_aux - 0)**2), 2)
        MC_Ifro_ri2 = (1/4)*self.roMC*MC_vol_4*(MCcr**2) + (1/12)*self.roMC*MC_vol_4*(MCcl**2) + \
            self.roMC*MC_vol_4*np.power(np.sqrt((MC_x2_aux - MC_x_ri_aux)
                                                ** 2 + (MC_y2_aux - 0)**2), 2)
        MC_Ifro_ri3 = (1/4)*self.roMC*MC_vol_1*(self.MCro**2) + (1/12)*self.roMC*MC_vol_1*(MCl**2) + \
            self.roMC*MC_vol_1*np.power(np.sqrt((MC_x3_aux - MC_x_ri_aux)
                                                ** 2 + (MC_y3_aux - 0)**2), 2)
        MC_Ifro_ri4 = (1/4)*self.roMC*MC_vol_2*(MCri**2) + (1/12)*self.roMC*MC_vol_2*(MCl**2) + \
            self.roMC*MC_vol_4*np.power(np.sqrt((MC_x4_aux - MC_x_ri_aux)
                                                ** 2 + (MC_y4_aux - 0)**2), 2)
        MC_Ifro_ri5 = (1/4)*self.roMC*MC_vol_3*(self.MCro**2) + (1/12)*self.roMC*MC_vol_3*(MCcl**2) + \
            self.roMC*MC_vol_3*np.power(np.sqrt((MC_x5_aux - MC_x_ri_aux)
                                                ** 2 + (MC_y5_aux - 0)**2), 2)
        MC_Ifro_ri6 = (1/4)*self.roMC*MC_vol_4*(MCcr**2) + (1/12)*self.roMC*MC_vol_4*(MCcl**2) + \
            self.roMC*MC_vol_4*np.power(np.sqrt((MC_x6_aux - MC_x_ri_aux)
                                                ** 2 + (MC_y6_aux - 0)**2), 2)
        MC_Ifro_ri = MC_Ifro_ri1 + MC_Ifro_ri2 + MC_Ifro_ri3 + \
            MC_Ifro_ri4 + MC_Ifro_ri5 + MC_Ifro_ri6 + MC_par_ri

        H_x1_aux = -self.Hl/2
        H_x2_aux = -Bw/2
        H_y1_aux = 0
        H_y2_aux = 0

        H_Ifro_ri1 = (1/4)*self.roH*H_vol_1*(self.Hr**2) + (1/12)*self.roH*H_vol_1*(self.Hl**2) + \
            self.roH*H_vol_1*np.power(np.sqrt((H_x1_aux - H_x_ri_aux)**2 + (H_y1_aux - 0)**2), 2)
        H_Ifro_ri2 = (1/4)*self.roH*H_vol_2*(Bro**2) + (1/12)*self.roH*H_vol_2*(Bw**2) + \
            self.roH*H_vol_2*np.power(np.sqrt((H_x2_aux - H_x_ri_aux)**2 + (H_y2_aux - 0)**2), 2)
        H_Ifro_ri = H_Ifro_ri1 + H_Ifro_ri2 + H_par_ri

        B_Ifro_ri = (B_m/12)*(3*(Bro**2 + Bri**2) + Bw**2) + B_par_ri
        S_Ifro_ri = (1/4)*S_m*(Bri**2) + (1/12)*S_m*(MCsl**2) + S_par_ri

        # Cálculo de momentos de inercia totales - plano frontal
        I_le = F_Ifro_le + A_Ifro_le + L_Ifro_le + C_Ifro_le + \
            M_Ifro_le + MC_Ifro_le + H_Ifro_le + B_Ifro_le + S_Ifro_le
        I_ri = F_Ifro_ri + A_Ifro_ri + L_Ifro_ri + C_Ifro_ri + \
            M_Ifro_ri + MC_Ifro_ri + H_Ifro_ri + B_Ifro_ri + S_Ifro_ri
        self.Ifro = I_le + I_ri

        self.afro = self.Rf - (self.Fh + self.COMfro[1])
        self.bsag = self.Rs - (self.Fh + self.COMsag[1])
        self.dsag = self.Rs - (self.Fh + self.Ah + self.Lh - self.Hcp)

    def foot(self):
        """ Cálculo de parámetros dinámicos del componente Foot-F
        """

        F_vol_p = 0.0
        F_COM_p = np.array([0, 0, 0], dtype=float)
        F_I_p = np.array([0, 0, 0], dtype=float)

        self.Fh = np.abs(np.sqrt((self.Rf**2) - (self.Fl**2)) - self.Rf +
                         np.sqrt((self.Rs**2) - (self.Fw**2)) - self.Rs)

        # Se obtienen datos correspondientes a la mitad del pie (parciales)
        F_vol_p, F_COM_p, F_I_p = self.integrales_foot()

        F_Ix_p = F_I_p[0]
        F_Iy_p = F_I_p[1]
        F_Iz_p = F_I_p[2]

        F_m_p = self.roF * F_vol_p

        # Cálculos de la parte complementaria del pie (complementario)
        F_x_p = F_COM_p[0]
        F_y_p = F_COM_p[1]
        F_z_p = -F_COM_p[2]

        # Centro de masa total del pie
        F_x = (1/(2*F_m_p))*(F_x_p*F_m_p + F_COM_p[0]*F_m_p)
        F_y = (1/(2*F_m_p))*(F_y_p*F_m_p + F_COM_p[1]*F_m_p)
        F_z = (1/(2*F_m_p))*(F_z_p*F_m_p + F_COM_p[2]*F_m_p)

        # Momento de inercia total del pie
        F_Ix = 2*(F_Ix_p - F_m_p*(np.sqrt(F_y**2 + F_z**2))**2)
        F_Iy = 2*(F_Iy_p - F_m_p*(np.sqrt(F_x**2 + F_z**2))**2)
        F_Iz = 2*(F_Iz_p - F_m_p*(np.sqrt(F_x**2 + F_y**2))**2)

        F_m = 2*F_m_p
        F_COM = np.array([F_x, F_y, F_z])
        F_I = np.array([F_Ix, F_Iy, F_Iz])

        return F_COM, F_I, F_m

    def integrales_foot(self):
        """Desarrollo de expresiones matemáticas de Foot-F
        Regresa el volumen, coordenadas del centro de masa y momento de inercia de Foot-F
        """

        # Términos notables
        R_Rf = np.sqrt((self.Rf**2) - (self.Fl**2))
        R_Rs = np.sqrt((self.Rs**2) - (self.Fw**2))
        ASINZRs = np.arcsin(self.Fw/self.Rs)
        ASINXRf = np.arcsin(self.Fl/self.Rf)
        POWRf = ((self.Rf**2) - (self.Fl**2))**(3/2)
        POWRs = ((self.Rs**2) - (self.Fw**2))**(3/2)

        # Cálculo de volumen
        F_vol = (1/2)*self.Fl*(self.Rs**2)*np.arctan(self.Fw/R_Rs) - self.Fw*((-1/2)*(self.Rf**2)
                                                                              * np.arctan(self.Fl/R_Rf) + self.Fl*(self.Rf+self.Rs-self.Fh-(1/2)*R_Rf-(1/2)*R_Rs))

        # Coordenadas del centro de masa - X
        F_x = (-(1/3)*self.Fw*POWRf + (1/4)*ASINZRs*(self.Rs**2)*(self.Fl**2) + self.Fw*(1/4)*(-2 *
                                                                                               (self.Fl**2)*self.Rs + (self.Fl**2)*(R_Rs+2*self.Fh-2*self.Rf) + (4/3)*(self.Rf**3)))*(1/F_vol)

        # Coordenadas del centro de masa - Y
        F_y = R_Rf*self.Fl*((1/2)*self.Fw*R_Rs + (1/2)*(self.Rs**2)*ASINZRs + self.Fw*(-self.Rf - self.Rs + self.Fh)) + self.Fw*R_Rs*((1/2)*(self.Rf**2)*ASINXRf + self.Fl*(-self.Rf - self.Rs + self.Fh)) \
            + (self.Rf**2)*ASINXRf*((1/2)*(self.Rs**2)*ASINZRs + self.Fw*(-self.Rf - self.Rs + self.Fh)) - (1/3)*self.Fl*(-3*(self.Rs**2)*ASINZRs*(-self.Rf - self.Rs +
                                                                                                                                                   self.Fh) + self.Fw*((self.Fl**2)+(self.Fw**2) - 6*(self.Rf**2) + self.Rf*(-6*self.Rs + 6*self.Fh) - 6*(self.Rs**2) + 6*self.Fh*self.Rs - 3*(self.Fh**2)))

        F_y = -(1/(2*F_vol))*F_y

        # Coordenadas del centro de masa - Z
        F_z = self.Fl*(self.Fw**2)*(1/4)*R_Rf + (1/4)*(self.Fw**2)*(self.Rf**2)*ASINXRf - self.Fl * \
            (1/3)*((self.Fw**2)*(-self.Fh*(3/2) + self.Rf*(3/2) + self.Rs*(3/2)) + POWRs - (self.Rs**3))

        F_z = (1/F_vol)*F_z

        F_COM = np.array([F_x, F_y, F_z])

        # Cálculo de momento de inercia - Ix
        F_Ix = (1/12)*self.Fl*self.Fw*POWRf - (1/6)*self.Fl*self.Fw*POWRs \
            - (1/6)*self.Fw*R_Rs*(-3*self.Fl*R_Rf*(-self.Rf - self.Rs + self.Fh) - 3*(self.Rf**2)*ASINXRf*(-self.Rf - self.Rs + self.Fh) + self.Fl*((self.Fl**2) - (9/2)*(self.Rs**2) + self.Rs*(6*self.Fh - 6*self.Rf) - 3*(self.Fh**2) + 6*self.Rf*self.Fh - 6*(self.Rf**2))) \
            + (1/2)*self.Fl*R_Rf*((self.Rs**2)*ASINZRs*(-self.Rf - self.Rs + self.Fh) + self.Fw*(2*(self.Rs**2) + self.Rs*(-2*self.Fh + 2*self.Rf) + (self.Fh**2) - 2*self.Rf*self.Fh + (5/4)*(self.Rf**2))) \
            - (1/6)*(self.Rs**2)*ASINZRs*(-3*(self.Rf**2)*ASINXRf*(-self.Rf - self.Rs + self.Fh) + self.Fl*((self.Fl**2) - (9/2)*(self.Rs**2) + self.Rs*(6*self.Fh - 6*self.Rf) - 3*(self.Fh**2) + 6*self.Rf*self.Fh - 6*(self.Rf**2))) \
            - (1/3)*self.Fw*(-(3/2)*(self.Rf**2)*ASINXRf*(2*(self.Rs**2) + self.Rs*(-2*self.Fh + 2*self.Rf) + (self.Fh**2) - 2*self.Rf*self.Fh + (5/4)*(self.Rf**2)) +
                             (-self.Rf - self.Rs + self.Fh)*self.Fl*((self.Fl**2) - 4*(self.Rs**2) + self.Rs*(2*self.Fh - 2*self.Rf) - (self.Fh**2) + 2*self.Rf*self.Fh - 4*(self.Rf**2)))

        F_Ix = self.roF*F_Ix

        # Cálculo de momento de inercia - Iy
        F_Iy = -(1/4)*self.Fl*self.Fw*POWRf - (1/4)*self.Fl*self.Fw*POWRs + (1/24)*self.Fl*R_Rf*(3*self.Fw*(self.Rf**2) + 4*(self.Fw**3)) + (1/24)*R_Rs*(3*self.Fl*self.Fw*(self.Rs**2) + 4*self.Fw*(self.Fl**3)) \
            + (1/24)*ASINXRf*(3*self.Fw*(self.Rf**4) + 4*(self.Rf**2)*(self.Fw**3)) + (1/3)*self.Fl*(ASINZRs*((1/2)*(self.Fl**2)
                                                                                                              * (self.Rs**2) + (3/8)*(self.Rs**4)) + self.Fw*((self.Fl**2) + (self.Fw**2))*(-self.Rf - self.Rs + self.Fh))

        F_Iy = self.roF*F_Iy

        # Cálculo de momento de inercia - Iz
        F_Iz = -(1/6)*self.Fl*self.Fw*POWRf + (1/12)*self.Fl*self.Fw*POWRs \
            + (1/2)*self.Fl*R_Rf*(self.Fw*R_Rs*(-self.Rf - self.Rs + self.Fh) + (self.Rs**2)*ASINZRs*(-self.Rf - self.Rs + self.Fh) - (1/3)*self.Fw*((self.Fw**2) - (9/2)*(self.Rf**2) + self.Rf*(-6*self.Rs + 6*self.Fh) - 3*(self.Fh**2) + 6*self.Fh*self.Rs - 6*(self.Rs**2))) \
            + (1/2)*self.Fw*R_Rs*((self.Rf**2)*ASINXRf*(-self.Rf - self.Rs + self.Fh) + self.Fl*(2*(self.Rf**2) + self.Rf*(-2*self.Fh + 2*self.Rs) + (self.Fh**2) - 2*self.Fh*self.Rs + (5/4)*(self.Rs**2))) \
            - (1/6)*(self.Rf**2)*ASINXRf*(-3*(self.Rs**2)*ASINZRs*(-self.Rf - self.Rs + self.Fh) + self.Fw*((self.Fw**2) - (9/2)*(self.Rf**2) + self.Rf*(-6*self.Rs + 6*self.Fh) - 3*(self.Fh**2) + 6*self.Fh*self.Rs - 6*(self.Rs**2))) \
            - (1/3)*self.Fl*(-(3/2)*(self.Rs**2)*ASINZRs*(2*(self.Rf**2) + self.Rf*(-2*self.Fh + 2*self.Rs) + (self.Fh**2) - 2*self.Fh*self.Rs + (5/4)*(self.Rs**2)) +
                             self.Fw*(-self.Rf - self.Rs + self.Fh)*((self.Fw**2) - 4*(self.Rf**2) + self.Rf*(2*self.Fh - 2*self.Rs) - (self.Fh**2) + 2*self.Fh*self.Rs - 4*(self.Rs**2)))

        F_Iz = self.roF*F_Iz

        F_I = np.array([F_Ix, F_Iy, F_Iz])

        return F_vol, F_COM, F_I

    ######################################  SIMULACIÓN #################################
    def sagital_components(self):
        """Cálculo de dimensiones de elementos estructurales para simulación
        """

        # Origen Frame
        lf = 0.02
        Or = np.array([0, 0, 0, 1])
        Px = np.array([lf, 0, 0, 1])
        Py = np.array([0, lf, 0, 1])
        Pz = np.array([0, 0, 0, 1])

        # LEG Vertices
        L_p1 = np.array([-self.bsag + self.dsag - self.Hcp, self.Lw/2, 0, 1])
        L_p2 = np.array([-self.bsag + self.dsag - self.Hcp + self.Lh, self.Lw/2, 0, 1])
        L_p3 = np.array([-self.bsag + self.dsag - self.Hcp + self.Lh, -self.Lw/2, 0, 1])
        L_p4 = np.array([-self.bsag + self.dsag - self.Hcp, -self.Lw/2, 0, 1])

        # LEG - Vector a trasnformar
        self.L_points = np.array([L_p1, L_p2, L_p3, L_p4])

        # ANKLE Vertices
        A_p1 = np.array([-self.bsag + self.dsag - self.Hcp + self.Lh, self.Aw/2, 0, 1])
        A_p2 = np.array([-self.bsag + self.dsag - self.Hcp + self.Lh + self.Ah, self.Aw/2, 0, 1])
        A_p3 = np.array([-self.bsag + self.dsag - self.Hcp + self.Lh + self.Ah, -self.Aw/2, 0, 1])
        A_p4 = np.array([-self.bsag + self.dsag - self.Hcp + self.Lh, -self.Aw/2, 0, 1])
        # ANKLE - Vector a transformar
        self.A_points = np.array([A_p1, A_p2, A_p3, A_p4])

        # FOOT Vertices
        Fx_center = - self.bsag
        Fy_center = 0.0
        th_max = np.arcsin(self.Fw/self.Rs)
        th_F = np.arange(-th_max, th_max, 2*np.pi/1000)
        ux_F = self.Rs*np.cos(th_F) + Fx_center
        uy_F = self.Rs*np.sin(th_F) + Fy_center
        # FOOT - Vertices extremos
        F_p1 = np.array([-self.bsag + self.dsag - self.Hcp + self.Lh + self.Ah, -self.Fw, 0, 1])
        F_p2 = np.array([-self.bsag + self.dsag - self.Hcp + self.Lh + self.Ah, self.Fw, 0, 1])
        # FOOT - unión de vectores (x,y)
        ux_F = np.insert(ux_F, 0, F_p1[0], axis=0)
        ux_F = np.append(ux_F, F_p2[0])
        uy_F = np.insert(uy_F, 0, F_p1[1], axis=0)
        uy_F = np.append(uy_F, F_p2[1])
        # FOOT - Vector a transformar
        F = np.array([[0, 0, 0, 0]])
        for i in range(ux_F.shape[0]):
            F = np.concatenate((F, np.array([[ux_F[i], uy_F[i], 0, 1]])))
        self.F_points = np.delete(F, 0, 0)

        # HIP Vertices
        Hx_center = -(self.bsag - self.dsag)
        Hy_center = 0.0
        th_H = np.arange(0, 2*np.pi, 2*np.pi/1000)
        ux_H = self.Hr*np.cos(th_H) + Hx_center
        uy_H = self.Hr*np.sin(th_H) + Hy_center
        # HIP - Vector a transformar
        H = np.array([[0, 0, 0, 0]])
        for i in range(ux_H.shape[0]):
            H = np.concatenate((H, np.array([[ux_H[i], uy_H[i], 0, 1]])))
        self.H_points = np.delete(H, 0, 0)

        # MOTOR CASE Vertices
        MCx_center = -(self.bsag - self.dsag)
        MCy_center = 0.0
        th_MC = np.arange(0, 2*np.pi, 2*np.pi/1000)
        ux_MC = self.MCro*np.cos(th_MC) + MCx_center
        uy_MC = self.MCro*np.sin(th_MC) + MCy_center
        # MC - Vector a transformar
        MC = np.array([[0, 0, 0, 0]])
        for i in np.arange(ux_MC.shape[0]):
            MC = np.concatenate((MC, np.array([[ux_MC[i], uy_MC[i], 0, 1]])))
        self.MC_points = np.delete(MC, 0, 0)


    def pbw_kinematics(self):
        """Cálculo de vectores de desplazamiento para simulación
        """
        #Vectores de desplazamiento
        self.ST_x_sag = np.zeros((1,self.n_S))  # Vector de desplazamiento
        self.ST_y_sag = np.zeros((1,self.n_S))  # Vector de desplazamiento
        self.SW_x_sag = np.zeros((1,self.n_S))  # Vector de desplazamiento
        self.SW_y_sag = np.zeros((1,self.n_S))  # Vector de desplazamiento

        ST_x_sum = 0
        SW_x_sum = 0

        ST_y_sum = 0
        SW_y_sum = 0

        flag1 = 0
        falg2 = 0
        k=1

        for i in range(self.Qsag.shape[1]):
            qst = self.Qsag[0, i]
            qsw = self.Qsag[1, i]
            qrc = self.Qsag[4, i]

            if qrc <= 0.5:

                if (i>0) & (i<500) & (k==1): #ciclo inicial
                    self.ST_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + self.bsag*np.sin(qst)
                    self.ST_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - self.bsag*np.cos(qst)

                    self.SW_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + (self.bsag - self.dsag)*np.sin(qsw) + self.dsag*np.sin(qst)
                    self.SW_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - (self.bsag - self.dsag)*np.cos(qsw) - self.dsag*np.cos(qst)

                elif ((i%(k*500))==0) & (i>0) & ((k%2)!=0): #indica transición
                    self.ST_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + self.bsag*np.sin(qst) + ST_x_sum
                    self.ST_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - self.bsag*np.cos(qst) + ST_y_sum

                    self.SW_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + (self.bsag - self.dsag)*np.sin(qsw) + self.dsag*np.sin(qst) + SW_x_sum
                    self.SW_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - (self.bsag - self.dsag)*np.cos(qsw) - self.dsag*np.cos(qst) + SW_y_sum
                    flag1=1
                    k=k+1

                elif (flag1==0) & (i>500): #Ocurre transición

                    self.ST_x_sag[0,i] = self.SW_x_sag[0,i-1] #self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + self.bsag*np.sin(qst)
                    self.ST_y_sag[0,i] = self.SW_y_sag[0,i-1] #self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - self.bsag*np.cos(qst)

                    self.SW_x_sag[0,i] = self.ST_x_sag[0,i-1] #self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + (self.bsag - self.dsag)*np.sin(qsw) + self.dsag*np.sin(qst)
                    self.SW_y_sag[0,i] = self.ST_y_sag[0,i-1] #self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - (self.bsag - self.dsag)*np.cos(qsw) - self.dsag*np.cos(qst)

                    ST_x_sum = self.ST_x_sag[0,i-1]
                    SW_x_sum = self.ST_x_sag[0,i-1]


                    ST_y_sum = -ST_x_sum*np.sin(self.gamma) #-np.abs(self.ST_y_sag[0,i-1] - self.SW_y_sag[0,i-496])
                    SW_y_sum = -ST_x_sum*np.sin(self.gamma)#-np.abs(self.ST_y_sag[0,i-1] - self.SW_y_sag[0,i-496])

                    flag1 = 1

                elif (flag1==1):

                    self.ST_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + self.bsag*np.sin(qst) + ST_x_sum
                    self.ST_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - self.bsag*np.cos(qst) + ST_y_sum

                    self.SW_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + (self.bsag - self.dsag)*np.sin(qsw) + self.dsag*np.sin(qst) + SW_x_sum
                    self.SW_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - (self.bsag - self.dsag)*np.cos(qsw) - self.dsag*np.cos(qst) + SW_y_sum

            else:

                if (i>0) & (i<500) & (k==1): #ciclo inicial
                    self.SW_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + self.bsag*np.sin(qst)
                    self.SW_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - self.bsag*np.cos(qst)

                    self.ST_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + (self.bsag - self.dsag)*np.sin(qsw) + self.dsag*np.sin(qst)
                    self.ST_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - (self.bsag - self.dsag)*np.cos(qsw) - self.dsag*np.cos(qst)

                elif ((i%(k*500))==0) & (i>0) & ((k%2)==0): #indica transición
                    self.SW_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + self.bsag*np.sin(qst) + SW_x_sum
                    self.SW_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - self.bsag*np.cos(qst) + SW_y_sum

                    self.ST_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + (self.bsag - self.dsag)*np.sin(qsw) + self.dsag*np.sin(qst) + ST_x_sum
                    self.ST_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - (self.bsag - self.dsag)*np.cos(qsw) - self.dsag*np.cos(qst) + ST_y_sum
                    flag1=0
                    k=k+1

                elif (flag1==1): #Ocurre transición

                    self.ST_x_sag[0,i] = self.SW_x_sag[0,i-1] #self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + self.bsag*np.sin(qst)
                    self.ST_y_sag[0,i] = self.SW_y_sag[0,i-1] #self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - self.bsag*np.cos(qst)

                    self.SW_x_sag[0,i] = self.ST_x_sag[0,i-1] #self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + (self.bsag - self.dsag)*np.sin(qsw) + self.dsag*np.sin(qst)
                    self.SW_y_sag[0,i] = self.ST_y_sag[0,i-1] #self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - (self.bsag - self.dsag)*np.cos(qsw) - self.dsag*np.cos(qst)

                    SW_x_sum = self.SW_x_sag[0,i-1]
                    ST_x_sum = self.SW_x_sag[0,i-1]

                    ST_y_sum = -ST_x_sum*np.sin(self.gamma) #-np.abs(self.SW_y_sag[0,i-1] - self.ST_y_sag[0,i-496])
                    SW_y_sum = -ST_x_sum*np.sin(self.gamma)#-np.abs(self.SW_y_sag[0,i-1] - self.ST_y_sag[0,i-496])

                    flag1=0

                elif (flag1==0): #ciclo inicial

                    self.SW_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + self.bsag*np.sin(qst) + SW_x_sum
                    self.SW_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - self.bsag*np.cos(qst) + SW_y_sum

                    self.ST_x_sag[0,i] = self.Rs*(np.cos(self.gamma)*(self.gamma-qst) - np.sin(self.gamma)) + (self.bsag - self.dsag)*np.sin(qsw) + self.dsag*np.sin(qst) + ST_x_sum
                    self.ST_y_sag[0,i] = self.Rs*(np.sin(self.gamma)*(self.gamma-qst) + np.cos(self.gamma)) - (self.bsag - self.dsag)*np.cos(qsw) - self.dsag*np.cos(qst) + ST_y_sum


    def dh(self, th, d, a, alp):

        T_11 = np.cos(th)
        T_12 = -np.sin(th)*np.cos(alp)
        T_13 = np.sin(th)*np.sin(alp)
        T_14 = a*np.cos(th)

        T_21 = np.sin(th)
        T_22 = np.cos(th)*np.cos(alp)
        T_23 = -np.cos(th)*np.sin(alp)
        T_24 = a*np.sin(th)

        T_31 = 0
        T_32 = np.sin(alp)
        T_33 = np.cos(alp)
        T_34 = d

        T = np.array([[T_11, T_12, T_13, T_14], [T_21, T_22, T_23, T_24],
                      [T_31, T_32, T_33, T_34], [0, 0, 0, 1]])

        return T

    def print_vector_p(self):
        print("Vector de diseño p\n")
        print("Rf: ", self.Rf)
        print("Rs: ", self.Rs)
        print("Fl: ", self.Fl)
        print("Fw: ", self.Fw)
        print("Al: ", self.Al)
        print("Ah: ", self.Ah)
        print("Aw: ", self.Aw)
        print("Ll: ", self.Ll)
        print("Lh: ", self.Lh)
        print("Lw: ", self.Lw)
        print("Hl: ", self.Hl)
        print("Hr: ", self.Hr)
        print("Hcp: ", self.Hcp)
        print("MCro: ", self.MCro)
        print("tini: ", self.tini)
        print("gamma: ", self.gamma)
        print("roF: ", self.roF)
        print("roA: ", self.roA)
        print("roL: ", self.roL)
        print("roH: ", self.roH)
        print("roMC: ", self.roMC)
        print("\nFh: ", self.F_phi)

    def print_parametros(self):
        print("Parámetros de Plano Frontal \n")
        print("mfro: ", self.mfro)
        print("Ifro: ", self.Ifro)
        print("a: ", self.afro)
        print("phi: ", self.phi)
        print("COMfro: ", self.COMfro)
        print("Tfro: ", self.Tfro)

        print("Parámetros de Plano Sagital \n")
        print("msag: ", self.msag)
        print("Isag: ", self.Isag)
        print("b: ", self.bsag)
        print("d: ", self.dsag)
        print("COMsag: ", self.COMsag)
        print("Tsag_p: ", self.Tsag_p)
        print("Tsag_sp: ", self.Tsag_sp)

    def save_parametros(self):
        # Recopilación de datos

        design_vector = {"Rf": self.Rf, "Rs": self.Rs, "Fl": self.Fl, "Fw": self.Fw, "Al": self.Al,
                         "Ah": self.Ah, "Aw": self.Aw, "Ll": self.Ll, "Lh": self.Lh, "Lw": self.Lw, "Hl": self.Hl,
                         "Hr": self.Hr, "Hcp": self.Hcp, "MCro": self.MCro, "tini": self.tini, "gamma": self.gamma,
                         "roF": self.roF, "roA": self.roA, "roL": self.roL, "roMC": self.roMC}

        frontal_parameters = {"mfro": self.mfro, "Ifro": self.Ifro, "a": self.afro, "phi": self.phi,
                              "COMfro": self.COMfro, "Tfro": self.Tfro}

        sagittal_parameters = {"msag": self.msag, "Isag": self.Isag, "b": self.bsag, "d": self.dsag,
                               "COMsag": self.COMsag, "Tsag_p": self.Tsag_p, "Tsag_sp": self.Tsag_sp}

        # print(design_vector)
        np.savez("bipedo.npz", p_vector=design_vector, Fro_par=frontal_parameters,
                 Sag_par=sagittal_parameters, Fro_states=self.Qfro, Sag_states=self.Qsag)

        # Evaluación de parámetros de Foot-F
