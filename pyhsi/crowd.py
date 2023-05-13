"""
PyHSI - Human Structure Interaction -  Crowd Class definition
"""

import numpy as np
import math
import csv


class Pedestrian:
    """
    Base Class for creating a Pedestrian
    """

    populationProperties = {}
    meanLognormalModel = 4.28  # mM
    sdLognormalModel = 0.21  # sM

    detK = 14110
    detVelocity = 1.25

    synchedPace = 0
    synchedPhase = 0

    def __init__(self, mass, damping, stiff, pace, phase, location, velocity, iSync):
        """
        this function introduce the properties when creating one pedestrian

        Parameters
        ----------
        mass: human mass
        damping : damping effect of pedesteian
        stiff : stiffness of humans
        pace : pacing frequency
        phase : phase angle
        location : location of mass
        velocity : velocity of travelling mass
        iSync : synchronization

        Returns
        -------
        None.

        """
        self.mass = mass
        self.damping = damping
        self.stiff = stiff
        self.pace = pace
        self.phase = phase
        self.location = location
        self.velocity = velocity
        self.iSync = iSync

    @classmethod
    def setPopulationProperties(cls, populationProperties):
        """
        class method function is used for future user to be able to access and modify the class state

        Parameters
        ---------


        Returns
        ------
        """
        cls.populationProperties = populationProperties

    @classmethod
    def setPaceAndPhase(cls, pace, phase):
        """
        class method function is used for future user to be able to access and modify the class state

        Parameters
        ---------


        Returns
        ------
        """
        cls.synchedPace = pace
        cls.synchedPhase = phase

    @classmethod
    def deterministicPedestrian(cls, location, synched=0):
        """
        class method function is used for future user to be able to access and modify the class state

        Parameters
        ---------


        Returns
        ------
        pMass
            Stores mass into dictionary hp
        pDamp
            Stores damping into
        pStiff
        pPace
        pPhase
        pLocation
        pStiffness
        pVelocity
        iSync

        """
        hp = cls.populationProperties
        pMass = hp['meanMass']
        pDamp = hp['meanDamping']*2*math.sqrt(cls.detK*hp['meanMass'])
        pStiff = cls.detK
        pLocation = location

        if synched == 1:
            iSync = 1
            pPace = cls.synchedPace
            pPhase = cls.synchedPhase
        else:
            iSync = 0
            pPace = np.random.normal(hp['meanPace'], hp['sdPace'])
            pPhase = (2 * math.pi) * np.random.rand()

        pVelocity = cls.detVelocity

        return cls(pMass, pDamp, pStiff, pPace, pPhase, pLocation, pVelocity, iSync)

    @classmethod
    def randomPedestrian(cls, location, synched=0):
        """
        class method function is used for future user to be able to access and modify the class state

        Parameters
        ---------


        Returns
        ------
        """
        hp = cls.populationProperties
        pMass = np.random.lognormal(mean=cls.meanLognormalModel, sigma=cls.sdLognormalModel)
        pDamp = np.random.normal(loc=hp['meanDamping'], scale=hp['sdDamping'])
        pStiff = np.random.normal(loc=hp['meanStiffness'], scale=hp['sdStiffness'])
        pLocation = location

        if synched == 1:
            iSync = 1
            pPace = cls.synchedPace
            pPhase = cls.synchedPhase
        else:
            iSync = 0
            pPace = np.random.normal(hp['meanPace'], hp['sdPace'])
            pPhase = (2 * math.pi) * np.random.rand()

        pStride = np.random.normal(hp['meanStride'], hp['sdStride'])
        pVelocity = np.multiply(pPace, pStride)

        return cls(pMass, pDamp, pStiff, pPace, pPhase, pLocation, pVelocity, iSync)

    @classmethod
    def exactPedestrian(cls, location, synched=0):
        """
        Temporary, used for testing crowds
        """
        hp = cls.populationProperties
        pMass = hp['meanMass']
        pDamp = hp['meanDamping'] * 2 * math.sqrt(cls.detK * hp['meanMass'])
        pStiff = cls.detK
        pPace = 2
        pPhase = 0
        pLocation = location
        pVelocity = cls.detVelocity
        iSync = synched

        return cls(pMass, pDamp, pStiff, pPace, pPhase, pLocation, pVelocity, iSync)

    # region Solver Methods
    def calcTimeOff(self, length):
        """
        returns the departure time of the pedestrian on the bridge

        Returns
        -------
        timeOff
            Departure time of pedestrian on the bridge

        """
        timeOff = (-self.location+length) / self.velocity
        return timeOff

    def calcPedForce(self, t):
        # Question: What are all the commented out parts in matlab ped_force
        g = 9.81

        W = self.mass * g
        x = self.location + self.velocity * t  # Position of Pedestrian at each time t

        # Young
        eta = np.array([0.41 * (self.pace - 0.95),
                        0.069 + 0.0056 * self.pace,
                        0.033 + 0.0064 * self.pace,
                        0.013 + 0.0065 * self.pace])
        phi = np.zeros(4)

        # Now assemble final force, and include weight
        N = len(eta)  # No. of additional terms in harmonic series
        F0 = W * np.insert(eta, 0, 1)  # Force amplitudes (constant amplitude for 1)
        beta = 2 * math.pi * self.pace * np.array([i for i in range(N + 1)])  # Frequencies
        phi = np.insert(phi, 0, 0) + self.phase  # Phases - enforce first phase as zero phase

        omega = beta * t + phi
        Ft = sum(F0 * np.cos(omega))

        return x, Ft
    # endregion


class Crowd:

    populationProperties = {
        'meanMass': 73.85,
        'sdMass': 15.68,
        'meanPace': 1.96,
        'sdPace': 0.209,
        'meanStride': 0.66,
        'sdStride': 0.066,
        'meanStiffness': 28000,
        'sdStiffness': 2800,
        'meanDamping': 0.3,
        'sdDamping': 0.03,
    }
    """
    an empty dictionary is initialized to store population properties which will be introduced in the following lines of code.  
    """

    def __init__(
        self,
        numPedestrians=100,
        length=50,
        width=1,
        sync=0
    ):
        """
        initialization takes arguments numPedestrians, length, width and sync. Then set the corresponding attributes
        """

        Pedestrian.setPopulationProperties(self.populationProperties)

        self.numPedestrians = numPedestrians
        self.length = length
        self.width = width
        self.sync = sync

        self.area = self.length * self.width
        # self.numPedestrians = int(self.density * self.area)
        self.lamda = self.numPedestrians / self.length

        self.locations = []
        self.iSync = []
        self.pedestrians = []

        # Crowd synchronization
        self.determineCrowdSynchronisation()

    def determineCrowdSynchronisation(self):
        sync = self.sync/100
        self.iSync = np.random.choice([0, 1], size=self.numPedestrians, p=[1 - sync, sync])
        pace = np.random.normal(loc=self.populationProperties['meanPace'], scale=self.populationProperties['sdPace'])
        phase = (2 * math.pi) * (np.random.rand())
        Pedestrian.setPaceAndPhase(pace, phase)

    def addRandomPedestrian(self, location, synched):
        self.pedestrians.append(Pedestrian.randomPedestrian(location, synched))

    def addDeterministicPedestrian(self, location, synched):
        self.pedestrians.append(Pedestrian.deterministicPedestrian(location, synched))

    def addExactPedestrian(self, location, synched):
        """
        Temporary, for testing
        """
        self.pedestrians.append(Pedestrian.exactPedestrian(location, synched))

    @classmethod
    def setPopulationProperties(cls, populationProperties):
        """
        classmethod function is used so users can set their own population properties and store it in the dictionary
        """
        cls.populationProperties = populationProperties

    @classmethod
    def fromDict(cls, crowdOptions):
        numPedestrians = crowdOptions['numPedestrians']
        length = crowdOptions['crowdLength']
        width = crowdOptions['crowdWidth']
        sync = crowdOptions['percentSynchronised']
        return cls(numPedestrians, length, width, sync)


class SinglePedestrian(Pedestrian):
    """
    Sub Class of Pedestrian
    """
    def __init__(self):
        """
        Initialize SinglePedestrian object with default parameters

        k: float
        numPedestrian : int
            Number of pedestrian
        """
        # TODO: Where should k come from
        k = 14.11e3

        pMass = self.populationProperties['meanMass']
        pDamp = self.populationProperties['meanDamping'] * 2 * math.sqrt(k * pMass)
        pStiff = k
        pPace = 2
        pPhase = 0
        pLocation = 0
        pVelocity = 1.25
        iSync = 0
        super().__init__(pMass, pDamp, pStiff, pPace, pPhase, pLocation, pVelocity, iSync)
        self.numPedestrians = 1
        self.pedestrians = [self] #???

    @classmethod
    def fromDict(cls, crowdOptions):
        """
        Construct SinglePedestrian object from a dictionary.

        Parameters
        ----------
        crowdOptions: dict

        Returns
        -------
        SinglePedestrian
        """
        return cls()


class DeterministicCrowd(Crowd):
    """
    Deterministic crowd object
    """

    arrivalGap = 1      # HSI Paper Section 5.4

    def __init__(self, numPedestrians, length, width, sync):
        """
        Initialize DeterministicCrowd object.

        Parameters
        ----------
        numPedestrians: int
        length: float
        width: float
        sync: bool
        """
        super().__init__(numPedestrians, length, width, sync)
        self.generateLocations()
        self.populateCrowd()

    def generateLocations(self):
        """
        Generate pedestrian arrival times.
        """
        self.locations = -self.arrivalGap*np.array(range(self.numPedestrians))

    def populateCrowd(self):
        """
        Add pedestrians to the crowd object.
        """
        for i in range(self.numPedestrians):
            self.addDeterministicPedestrian(self.locations[i], self.iSync[i])

    @classmethod
    def setArrivalGap(cls, arrivalGap):
        """
        Set arrivalGap class variable.
        """
        cls.arrivalGap = arrivalGap


class RandomCrowd(Crowd):
    """
    Random crowd object
    """
    def __init__(self, numPedestrians, length, width, sync):
        """
        Initialize RandomCrowd object.

        Parameters
        ----------
        numPedestrians: int
        length: float
        width: float
        sync: bool
        """
        super().__init__(numPedestrians, length, width, sync)
        self.generateLocations()
        self.populateCrowd()

    def generateLocations(self):
        """
        Generate pedestrian arrival times.
        """
        gaps = np.random.exponential(1 / self.lamda, size=self.numPedestrians)
        self.locations = np.cumsum(gaps, axis=None, dtype=None, out=None)

    def populateCrowd(self):
        """
        Add pedestrians to the crowd object.
        """
        for i in range(self.numPedestrians):
            self.addRandomPedestrian(self.locations[i], self.iSync[i])


def getPopulationProperties():
    """
    Return population properties as a dictionary.
    """
    populationProperties = {}

    with open('../simulations/defaults/DefaultPopulationProperties.csv', newline='') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')
        for row in csvReader:
            populationProperties[row[0]] = float(row[1])

    return populationProperties


def updatePopulationProperties(populationProperties):
    """
    Update population properties.
    """
    Pedestrian.setPopulationProperties(populationProperties)
    Crowd.setPopulationProperties(populationProperties)


class ExactCrowd(Crowd):
    """
    A subclass of Crowd that generates pedestrian locations and populates the crowd using exact pedestrian models.
    """
    arrivalGap = 1      # HSI Paper Section 5.4

    def __init__(self, numPedestrians, length, width, sync):
        """
        Initializes an instance of the ExactCrowd class.

        Parameters
        ----------
        numPedestrians :int
            The number of pedestrians in the crowd.
        length : float
            The length of the crowd.
        width : float
            The width of the crowd.
        sync : bool
            Whether the pedestrians are synchronized.

        Returns
        -------
            None
        """
        super().__init__(numPedestrians, length, width, sync)
        self.generateLocations()
        self.populateCrowd()

    def generateLocations(self):
        """
        Generates the pedestrian locations.

        Parameter
        --------
            None

        Returns
        -------
            None
        """
        self.locations = -self.arrivalGap*np.array(range(self.numPedestrians))

    def populateCrowd(self):
        """
        Populates the crowd with exact pedestrian models.

        Parameters
        ----------
            None

        Returns
        -------
            None
        """
        for i in range(self.numPedestrians):
            self.addExactPedestrian(self.locations[i], self.iSync[i])

    @classmethod
    def setArrivalGap(cls, arrivalGap):
        """
        Sets the arrival gap for the crowd.

        Parameters
        ----------
            arrivalGap : float
                The arrival gap for the crowd.

        Returns
        -------
            None
        """
        cls.arrivalGap = arrivalGap

