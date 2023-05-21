"""
PyHSI - Human Structure Interaction - Solver Module
"""
import math

import numpy
import numpy as np


import scipy
from scipy.linalg import eig
from scipy.linalg import eigh

from crowd import *
from beam import *
from results import *
''' 
import the beams from the beam
'''


class Solver:
    """
    The base class for Human-Structure Interaction.

    Solves the system model combinations
    """
    numSteps = 5000
    g = 9.81
    PedestrianModel = None
    ModelType = None

    def __init__(self, crowd, beam):
        """
        Initializes the solver object and conduct the HSI analysis given the crowd and beam information

        Parameters
        ----------
        crowd : class
            The modelled walking pedestrian
        beam : class
            Modelled arbitrary beam structure

        Returns
        -------
        None.

        """
        self.crowd = crowd
        self.beam = beam

        self.nBDOF = self.beam.nBDOF                            # Beam-only DOFs
        self.nDOF = self.calcnDOF()                             # Overall DOFs
        self.RDOF = self.beam.RDOF                              # Restrained DOFs

        self.lamda = np.zeros((self.numSteps, 2 * self.nDOF))     # ???
        self.eigflag = 0                                        # ???

        self.t, self.dT = self.genTimeVector()                  # Setup time vector
        self.Mb, self.Cb, self.Kb = self.assembleMCK()          # Assemble mass, stiffness and damping matrices
        self.Mb, self.Cb, self.Kb = self.constraints(self.Mb, self.Cb, self.Kb)

        self.q = np.zeros((self.numSteps, self.nDOF))             # Initialize displacement matrix
        self.dq = np.zeros((self.numSteps, self.nDOF))            # Initialize velocity matrix
        self.ddq = np.zeros((self.numSteps, self.nDOF))           # Initialize acceleration matrix
        self.F0 = np.zeros(self.nDOF)                           # Initialize ??? matrix

    # region Prepare Solver
    def calcnDOF(self):
        """
        Calculates the total number of degrees of freedom (DOFs) in the system.

        Returns
        -------
        nBDOF: int
           The total number of DOFs
        """
        if self.PedestrianModel == "Spring Mass Damper":
            return self.nBDOF + self.crowd.numPedestrians
        return self.nBDOF

    def genTimeVector(self):
        """
        Returns the time vector by simulating the time frame for a given space and stride vector

        Returns
        -------
        t : array
            Current time
        dt : float
            Time step size
        """
        f = 1/(2*math.pi) * (math.pi/self.beam.length)**2*math.sqrt(self.beam.EI/self.beam.linearMass)
        period = 1/f
        dTMax = 0.02*period     # Stability of newmark

        maxTimeOff = 0
        for ped in self.crowd.pedestrians:
            timeOff = ped.calcTimeOff(self.beam.length)
            if timeOff > maxTimeOff:
                maxTimeOff = timeOff

        timeEnd = 1.1*maxTimeOff    # Run simulation for a bit after the last ped has left
        print(f"self.numSteps: {self.numSteps}")
        dT = timeEnd / (self.numSteps - 1)
        dT = min(dT, dTMax)
        t = np.arange(0, timeEnd, dT)  # Rounding error created by differing precision in Python vs MATLAB

        self.numSteps = len(t)    # Adjust numSteps

        return t, dT

    def assembleMCK(self):
        """
        This function assembles the mass and stiffness matrices, applies the boundary conditions, and calculates a
        damping matrix based on Rayleigh damping

        Return
        ------
        M : np.ndarray
            System mass
        C : np.ndarray
            System damping
        K : np.ndarray
            System stiffness
        """

        # M - System mass
        # C - System damping
        # K - System stiffness

        nElements = self.beam.numElements
        nDOF = 2*(nElements+1)

        elementalMassMatrix, elementalStiffnessMatrix = self.beam.beamElement()

        M = np.zeros((nDOF, nDOF))
        C = np.zeros((nDOF, nDOF))
        K = np.zeros((nDOF, nDOF))

        # Assemble elements, noting beam is prismatic
        for i in range(nElements):
            ni = 2 * i
            nj = 2 * i + 4
            M[ni:nj, ni:nj] += elementalMassMatrix
            K[ni:nj, ni:nj] += elementalStiffnessMatrix

        # Apply constraints before estimating modal properties
        # Save to temp variables to keep M, K
        Mt = M.copy()
        Ct = C.copy()
        Kt = K.copy()
        Mt, Ct, Kt = self.constraints(Mt, Ct, Kt)

        phi, w = self.modal(Mt, Kt)

        alpha, beta = self.rayleighCoeffs(w, self.beam.modalDampingRatio, self.beam.nHigh)

        C = alpha * M + beta * K

        # self.Mb = M
        # self.Cb = C
        # self.Kb = K
        return M, C, K

    def constraints(self, M, C, K):
        """
        apply constraints before estimating modal properties

        Parameters
        ----------
        M : np.ndarray
            System Mass
        C : np.ndarray
            System Damping
        K : np.ndarray
            System stiffness

        Return
        ------
        M : np.ndarray
        C : np.ndarray
        K : np.ndarray
        """
        def imposeRestraint(A, dof):
            """
            Impose restraints on the matrix A by setting specific entries to zero and the diagonal entry to one.

            Parameters
            ----------
            A : np.ndarray
                The matrix to impose restraints on.
            dof : Any
                The restrained degrees of freedom.

            Return
            ------
            A : np.ndarray
                The modified matrix A
            """
            A[dof] = 0  # column
            A[:, dof] = 0  # row
            A[dof, dof] = 1  # diagonal
            return A

        for i in self.RDOF:
            dof = i
            M = imposeRestraint(M, dof)
            C = imposeRestraint(C, dof)
            K = imposeRestraint(K, dof)

        return M, C, K

    @staticmethod
    def modal(M, K):
        """
        Calculate the mode shape matrix and eigenvalues of a system.

        Parameters
        ----------
        M : np.ndarray
            The system mass matrix.
        K : np.ndarray
            The system stiffness matrix.

        Return
        ------
        phi : Any
            The mode shape matrix.
        omega : Any
            The eigenvalues of the system.
        """
        lam, phi = eigh(K, M)
        n, m = K.shape
        omega = np.sqrt(lam)

        return phi, omega

    @staticmethod
    def rayleighCoeffs(w, modalDampingRatio, nHigh):
        """
        Calculate the Rayleigh coefficients given the modal damping ratio and the higher mode of the damping matrix.

        Parameters
        ----------
        w : Any
            Vector of modal frequencies
        modalDampingRatio : Any
            Modal damping ratio of the beam
        nHigh : Any
            Higher mode of the damping matrix

        Returns
        -------
        alpha : float
            Rayleigh coefficient
        beta  : float
            Rayleigh coefficient
        """
        wr = [i for i in w if i > 1.01]
        wi = wr[0]
        wj = wr[nHigh - 1]

        alpha = modalDampingRatio * (2 * wi * wj) / (wi + wj)
        beta = modalDampingRatio * 2 / (wi + wj)

        return alpha, beta
    # endregion

    # region Solve
    def solve(self):
        """
        Solve the system using a specific model.

        Returns:
            Results: The results of the simulation.
        """
        print(f"Solving system with a '{self.ModelType} - {self.PedestrianModel}' model")
        for i in range(1, self.numSteps):
            u, du, ddu, Ft = self.nonLinearNewmarkBeta(self.t[i], self.q[i-1], self.dq[i-1], self.ddq[i-1])
            self.F0 = Ft
            self.q[i] = u
            self.dq[i] = du
            self.ddq[i] = ddu

            # Progress
            if i % 100 == 0:
                percentCompleted = i/self.numSteps * 100
                print(f"{percentCompleted:.2f}% completed", end='\r')

        return Results(self.t, self.q, self.dq, self.ddq)

    def nonLinearNewmarkBeta(self, t, u0, du0, ddu0):
        """
        This function obtains the matrices M,C,K and F at time t from the function func. It integrates the equations
        over time step dt and returns accelerations, velocities and displacements. it passes the current values of these
        parameters to func.

        Parameters
        ----------
        t : Any
            Current time
        u0 : Any
            Current displacement
        du0 : Any
            Current velocity
        ddu0 : Any
            Current acceleration

        Returns
        -------
        u : int
            Displacement
        du : int
            Velocity
        ddu : int
            Acceleration
        F : np.ndarray
            Force

        """
        # TODO: Complete
        # u = displacement
        # du = velocity
        # ddu = acceleration
        # Ft = force
        # lamda = ??

        gamma = 1/2
        beta = 1/4
        forceTol = 1e-6
        maxInc = 20

        # Get current system matrices
        M, C, K, F = self.getCurrentSystemMatrices(t)
        dF = F-self.F0

        # if eigenvalue output is requested, produce the complex eigenvalues
        if self.eigflag != 0:
            A1 = np.zeros(self.nDOF)
            A2 = np.array(self.nDOF)
            A3 = np.linalg.lstsq(-M, K)[0]
            A4 = np.linalg.lstsq(-M, C)[0]
            A = [[A1, A2], [A3, A4]]
            lamda = eig(A)
            lamda = np.transpose(lamda)

        # Effective stiffness and other parameters
        a0 = 1/(beta*self.dT**2)
        a1 = gamma/(beta*self.dT)
        a2 = 1/(beta*self.dT)
        a3 = gamma/beta
        a4 = 1/(2*beta)
        a5 = self.dT/2*(gamma/beta-2)
        a6 = self.dT*(1-gamma/(2*beta))

        Keff = a0*M + a1*C + K
        # iKeff = numpy.linalg.inv(Keff)

        A = a2*M + a3*C
        B = a4*M + a5*C

        # Effective force and current displacement step
        dFeff = transpose(dF) + np.dot(A, transpose(du0)) + np.dot(B, transpose(ddu0))
        delta_u = reshape(np.linalg.lstsq(Keff, dFeff, rcond=None)[0])

        del_u_i = 0     # incremental displacement
        done = False
        i = 0

        u = 0
        du = 0
        ddu = 0
        while not done and i < maxInc:
            delta_u += del_u_i
            delta_du = a1*delta_u - a3*transpose(du0) + a6*transpose(ddu0)
            delta_ddu = a0*delta_u - a2*transpose(du0) - a4*transpose(ddu0)

            u = u0 + transpose(delta_u)
            du = du0 + transpose(delta_du)
            ddu = ddu0 + transpose(delta_ddu)

            M_ddu = np.dot(M, transpose(ddu))
            C_du = np.dot(C, transpose(du))
            K_u = np.dot(K, transpose(u))

            residualForce = F - transpose(M_ddu + C_du + K_u)   # Residual force
            # del_u_i = Keff/transpose(residualForce)
            del_u_i = reshape(np.linalg.lstsq(Keff, transpose(residualForce), rcond=None)[0])
            done = numpy.linalg.norm(residualForce) < forceTol
            i += 1

        return u, du, ddu, F

    def getCurrentSystemMatrices(self, t):
        """
        Calculates the current system matrices.

        Parameters
        ----------
        t : float
            The current time

        Returns
        -------
        M : np.ndarray
            The mass matrix
        C : np.ndarray
            The damping matrix.
        K : ndarray
            The stiffness matrix.
        F : ndarray
            The force vector.
        """

        # Initialize global matrices
        M = self.Mb.copy()
        C = self.Cb.copy()
        K = self.Kb.copy()
        F = np.zeros(self.nDOF)

        # For each pedestrian
        for ped in self.crowd.pedestrians:
            x, Ft = ped.calcPedForce(t)     # Pedestrian position and force
            N, dN, ddN = self.globalShapeFunction(x)
            Nt = np.array([N]).T            # Transpose of N

            # Calculate adjustments to MCKF
            MStar = ped.mass * Nt * N
            CStar = ped.mass * ped.velocity * 2 * Nt * dN
            KStar = ped.mass * ped.velocity ** 2 * Nt * ddN
            Fp = N * Ft

            # Assemble into global matrices
            M += MStar
            C += CStar
            K += KStar
            F += Fp

        return M, C, K, F

    def applyConstraints(self):
        """
        Applies constraints to the matrices.

        Returns
        -------
        M : np.ndarray
            The modified mass matrix.
        C : np.ndarray
            The modified damping matrix.
        K : np.ndarray
            The modified stiffness matrix.
        """
        def imposeRestraint(A, dof):
            A[dof] = 0          # column
            A[:, dof] = 0       # row
            A[dof, dof] = 1     # diagonal

            return A

        M = self.Mb
        C = self.Cb
        K = self.Kb

        for i in self.RDOF:
            dof = i
            M = imposeRestraint(M, dof)
            C = imposeRestraint(C, dof)
            K = imposeRestraint(K, dof)

        return M, C, K

    def globalShapeFunction(self, x):
        """
        Calculates the global shape function.

        Parameter
        ---------
           x : float
            The position.

        Returns
        -------
          Ng : np.ndarray
            The shape function matrix.
          dNg : np.ndarray
            The derivative of the shape function matrix.
          ddNg np.ndarray
            The second derivative of the shape function matrix.
        """

        # Shape function zero matrices
        Ng = np.zeros(self.nBDOF)
        dNg = np.zeros(self.nBDOF)
        ddNg = np.zeros(self.nBDOF)

        # Check if the force is on the bridge
        if self.beam.onBeam(x):
            L = self.beam.elemLength
            s, zeta = self.beam.locationOnBeam(x)
            dof = np.array([i for i in range(2 * s - 2, 2 * s + 2)])

            N1 = 1 - 3 * zeta ** 2 + 2 * zeta ** 3
            N2 = (zeta - 2 * zeta ** 2 + zeta ** 3) * L
            N3 = 3 * zeta ** 2 - 2 * zeta ** 3
            N4 = (-zeta ** 2 + zeta ** 3) * L
            Ng[dof] = [N1, N2, N3, N4]

            dN1 = -6 * x / L ** 2 + 6 * x ** 2 / L ** 3
            dN2 = 1 - 4 * x / L + 3 * x ** 2 / L ** 2
            dN3 = 6 * x / L ** 2 - 6 * x ** 2 / L ** 3
            dN4 = -2 * x / L + 3 * x ** 2 / L ** 2
            dNg[dof] = [dN1, dN2, dN3, dN4]

            ddN1 = -6 / L ** 2 + 12 * x / L ** 3
            ddN2 = -4 / L + 6 * x / L ** 2
            ddN3 = 6 / L ** 2 - 12 * x / L ** 3
            ddN4 = -2 / L + 6 * x / L ** 2
            ddNg[dof] = [ddN1, ddN2, ddN3, ddN4]

        for i in self.RDOF:
            Ng[i] = 0
            dNg[i] = 0
            ddNg[i] = 0

        return Ng, dNg, ddNg
    # endregion

    # region Return Information
    def getResults(self):
        """
        Returns the simulation results.

        Returns
        -------
            t : float
                The time vector.
            q : np.ndarray
                The displacement vector.
            dq : np.ndarray
                The velocity vector.
            ddq : np.ndarray
                The acceleration vector.
        """

        return self.t, self.q, self.dq, self.ddq

    def getModelType(self):
        """
        Returns the model type.

        Returns
        -------
            PedestrianModel : str
                The pedestrian model.
            ModelType : str
                The model type.
        """
        return self.PedestrianModel, self.ModelType
    # endregion

    @classmethod
    def setNumSteps(cls, numSteps):
        """
        Sets the number of simulation steps.

        Parameters
        ----------
            numSteps : int
                The number of steps.
        """
        cls.numSteps = numSteps


class FeMmSolver(Solver):
    """
    Finite Element Moving Mass Solver.
    """
    PedestrianModel = "Moving Mass"
    ModelType = "Finite Element"

    # Solve class is based on a FE MM System


class FeMfSolver(Solver):
    """
    Finite Element Moving Force Solver.
    """

    PedestrianModel = "Moving Force"
    ModelType = "Finite Element"

    # region Solve
    def getCurrentSystemMatrices(self, t):
        pass
    # endregion


class FeSMDSolver(Solver):
    """
    Finite Element Spring Mass Damper Solver.
    """

    PedestrianModel = "Spring Mass Damper"
    ModelType = "Finite Element"

    # region Solve
    def getCurrentSystemMatrices(self, t):
        # Initialize global matrices
        M = np.zeros((self.nDOF, self.nDOF))
        C = np.zeros((self.nDOF, self.nDOF))
        K = np.zeros((self.nDOF, self.nDOF))
        F = np.zeros(self.nDOF)

        # Put in beam matrices
        M[:self.nBDOF, :self.nBDOF] = self.Mb.copy()
        C[:self.nBDOF, :self.nBDOF] = self.Cb.copy()
        K[:self.nBDOF, :self.nBDOF] = self.Kb.copy()

        # For each pedestrian
        for i in range(len(self.crowd.pedestrians)):
            ped = self.crowd.pedestrians[i]
            x, Ft = ped.calcPedForce(t)  # Pedestrian position and force
            N, dN, ddN = self.globalShapeFunction(x)
            Nt = np.array([N]).T  # Transpose of N

            # Calculate adjustments to MCKF
            # MStar = ped.mass * Nt * N
            CStar = ped.damping * Nt * N
            KStar = (ped.damping * ped.velocity * Nt * dN) + (ped.stiff * Nt * N)
            Fp = Nt * Ft

            # Assemble into global matrices
            # M += MStar
            C[:self.nBDOF, :self.nBDOF] += CStar
            K[:self.nBDOF, :self.nBDOF] += KStar

            pDOF = self.nBDOF + i   # The DOF of this pedestrian

            # Mass matrix - only diagonal terms
            M[pDOF, pDOF] = ped.mass

            # Damping = Diagonal and coupled terms
            C[pDOF, pDOF] = ped.damping
            C[:self.nBDOF, pDOF] = -ped.damping * Nt
            C[pDOF, :self.nBDOF] = -ped.damping * N

            # Stiffness - Diagonal and coupled terms
            K = K

            # Force vector
            F += Fp

        return M, C, K, F
    # endregion


class MoMmSolver(Solver):
    """
    Modal Analysis Moving Mass Solver.
    """

    PedestrianModel = "Moving Mass"
    ModelType = "Modal Analysis"

    # region Solve
    def getCurrentSystemMatrices(self, t):
        # TODO: Rewrite for MO SMD System
        pass
    # endregion


class MoMfSolver(Solver):
    """
    Modal Analysis Moving Force Solver.
    """
    PedestrianModel = "Moving Force"
    ModelType = "Modal Analysis"

    # region Solve
    def getCurrentSystemMatrices(self, t):
        # TODO: Rewrite for MO SMD System
        pass
    # endregion


class MoSMDSolver(Solver):
    """
    Modal Analysis Spring Mass Damper Solver.
    """
    PedestrianModel = "Spring Mass Damper"
    ModelType = "Modal Analysis"

    # region Solve
    def getCurrentSystemMatrices(self, t):
        # TODO: Rewrite for MO SMD System
        pass
    # endregion


def transpose(A):
    """
    Transposes the matrix A.

    Parameters
    ----------
        A : np.ndarray
            The input matrix.

    Returns
    -------
        ndarray
            The transposed matrix.
    """
    return reshape(np.array([A]).T)


def reshape(A):
    """
    Reshapes the matrix A.

    Parameters
    ----------
        A : np.ndarray
            The input matrix.

    Returns
    -------
        ndarray
            The reshaped matrix.
    """
    return A.reshape(len(A))


if __name__ == '__main__':
    print("Go to run_old.py to run a simulation, or results to process results.")
