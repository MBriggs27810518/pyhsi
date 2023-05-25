"""
PyHSI - Human Structure Interaction -  Crowd Class definition
"""

import numpy as np
import math
import csv


class Pedestrian:
    """
    Base class for creating a pedestrian.

    :ivar populationProperties: Dictionary containing population properties
    :ivar meanLognormalModel: Mean of the lognormal distribution model
    :ivar sdLognormalModel: Standard deviation of the lognormal distribution model
    :ivar detK: Deterministic stiffness value
    :ivar detVelocity: Deterministic velocity value
    :ivar synchedPace: Synchronized pacing frequency
    :ivar synchedPhase: Synchronized phase angle

    :param mass: Human mass
    :type mass: float
    :param damping: Damping effect of pedestrian
    :type damping: float
    :param stiff: Stiffness of humans
    :type stiff: float
    :param pace: Pacing frequency
    :type pace: float
    :param phase: Phase angle
    :type phase: float
    :param location: Location of mass
    :type location: float
    :param velocity: Velocity of travelling mass
    :type velocity: float
    :param iSync: Synchronization flag
    :type iSync: int
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
        Initializes a Pedestrian object with the specified properties.

        :param mass: Human mass
        :type mass: float
        :param damping: Damping effect of pedestrian
        :type damping: float
        :param stiff: Stiffness of humans
        :type stiff: float
        :param pace: Pacing frequency
        :type pace: float
        :param phase: Phase angle
        :type phase: float
        :param location: Location of mass
        :type location: float
        :param velocity: Velocity of travelling mass
        :type velocity: float
        :param iSync: Synchronization flag
        :type iSync: int
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
        Sets the population properties for all pedestrians.

        :param populationProperties: Dictionary containing population properties
        :type populationProperties: dict
        """
        cls.populationProperties = populationProperties

    @classmethod
    def setPaceAndPhase(cls, pace, phase):
        """
        Sets the synchronized pacing frequency and phase angle.

        :param pace: Pacing frequency
        :type pace: float
        :param phase: Phase angle
        :type phase: float
        """
        cls.synchedPace = pace
        cls.synchedPhase = phase

    @classmethod
    def deterministicPedestrian(cls, location, synched=0):
        """
        Creates a deterministic pedestrian.

        :param location: Location of the pedestrian
        :type location: float
        :param synched: Synchronization flag, defaults to 0
        :type synched: int, optional
        :return: Deterministic pedestrian object
        :rtype: Pedestrian
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
        Creates a random pedestrian.

        :param location: Location of the pedestrian
        :type location: float
        :param synched: Synchronization flag, defaults to 0
        :type synched: int, optional
        :return: Random pedestrian object
        :rtype: Pedestrian
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
        Temporary method used for testing crowds.

        :param location: Location of the pedestrian
        :type location: float
        :param synched: Synchronization flag, defaults to 0
        :type synched: int, optional
        :return: Exact pedestrian object
        :rtype: Pedestrian
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
        Calculates the departure time of the pedestrian on the bridge.

        :param length: Length of the bridge
        :type length: float
        :return: Departure time of pedestrian on the bridge
        :rtype: float
        """
        timeOff = (-self.location+length) / self.velocity
        return timeOff

    def calcPedForce(self, t):
        """
        Calculates the force exerted by the pedestrian at a given time.

        :param t: Time
        :type t: float
        :return: Position of the pedestrian and force exerted
        :rtype: tuple(float, float)
        """
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
    """
    The Crowd class represents a crowd of pedestrians and provides methods to add and manage pedestrians.

    populationProperties:
    An empty dictionary is initialized to store population properties, which will be introduced in the following lines of code.
    """
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

    def __init__(
        self,
        numPedestrians=100,
        length=50,
        width=1,
        sync=0
    ):
        """
        Initialize the Crowd object with the specified number of pedestrians, length, width, and synchronization.

        :param numPedestrians: The number of pedestrians in the crowd (default: 100).
        :param length: The length of the crowd area (default: 50).
        :param width: The width of the crowd area (default: 1).
        :param sync: The synchronization level of the crowd as a percentage (default: 0).
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
        """
        Determine the synchronization of the crowd based on the specified sync level.
        """
        sync = self.sync/100
        self.iSync = np.random.choice([0, 1], size=self.numPedestrians, p=[1 - sync, sync])
        pace = np.random.normal(loc=self.populationProperties['meanPace'], scale=self.populationProperties['sdPace'])
        phase = (2 * math.pi) * (np.random.rand())
        Pedestrian.setPaceAndPhase(pace, phase)

    def addRandomPedestrian(self, location, synched):
        """
        Add a random pedestrian to the crowd at the specified location and synchronization.

        :param location: The location of the pedestrian.
        :param synched: The synchronization of the pedestrian (0 for unsynchronized, 1 for synchronized).
        """
        self.pedestrians.append(Pedestrian.randomPedestrian(location, synched))

    def addDeterministicPedestrian(self, location, synched):
        """
        Add a deterministic pedestrian to the crowd at the specified location and synchronization.

        :param location: The location of the pedestrian.
        :param synched: The synchronization of the pedestrian (0 for unsynchronized, 1 for synchronized).
        """
        self.pedestrians.append(Pedestrian.deterministicPedestrian(location, synched))

    def addExactPedestrian(self, location, synched):
        """
        Add an exact pedestrian to the crowd at the specified location and synchronization.

        Temporary, for testing.

        :param location: The location of the pedestrian.
        :param synched: The synchronization of the pedestrian (0 for unsynchronized, 1 for synchronized).
        """
        self.pedestrians.append(Pedestrian.exactPedestrian(location, synched))

    @classmethod
    def setPopulationProperties(cls, populationProperties):
        """
        Set the population properties of the crowd.

        :param populationProperties: A dictionary containing the population properties.
        """
        cls.populationProperties = populationProperties

    @classmethod
    def fromDict(cls, crowdOptions):
        """
        Create a Crowd object from a dictionary of crowd options.

        :param crowdOptions: A dictionary containing the crowd options.
        :return: A Crowd object.
        """
        numPedestrians = crowdOptions['numPedestrians']
        length = crowdOptions['crowdLength']
        width = crowdOptions['crowdWidth']
        sync = crowdOptions['percentSynchronised']
        return cls(numPedestrians, length, width, sync)


class SinglePedestrian(Pedestrian):
    """
    The SinglePedestrian class represents a single pedestrian and is a subclass of Pedestrian.
    """
    def __init__(self):
        """
        Initialize a SinglePedestrian object with default parameters.

        :param k: The stiffness parameter for the pedestrian (default: 14.11e3).
        :param numPedestrian: The number of pedestrians (default: 1).
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
        Construct a SinglePedestrian object from a dictionary.

        :param crowdOptions: A dictionary containing the crowd options.
        :return: A SinglePedestrian object.
        """
        return cls()


class DeterministicCrowd(Crowd):
    """
    The DeterministicCrowd class represents a deterministic crowd and is a subclass of Crowd.
    """

    arrivalGap = 1      # HSI Paper Section 5.4

    def __init__(self, numPedestrians, length, width, sync):
        """
        Initialize a DeterministicCrowd object.

        :param numPedestrians: The number of pedestrians in the crowd.
        :param length: The length of the crowd area.
        :param width: The width of the crowd area.
        :param sync: The synchronization level of the crowd (0 for unsynchronized, 1 for synchronized).
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
        Set the arrivalGap class variable.

        :param arrivalGap: The arrival gap between pedestrians.
        """
        cls.arrivalGap = arrivalGap


class RandomCrowd(Crowd):
    """
    The RandomCrowd class represents a random crowd and is a subclass of Crowd.
    """
    def __init__(self, numPedestrians, length, width, sync):
        """
        Initialize a RandomCrowd object.

        :param numPedestrians: The number of pedestrians in the crowd.
        :param length: The length of the crowd area.
        :param width: The width of the crowd area.
        :param sync: The synchronization level of the crowd (0 for unsynchronized, 1 for synchronized).
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

    :return: A dictionary containing population properties.
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

    :param populationProperties: A dictionary containing the updated population properties.
    """
    Pedestrian.setPopulationProperties(populationProperties)
    Crowd.setPopulationProperties(populationProperties)


class ExactCrowd(Crowd):
    """
    The ExactCrowd class is a subclass of Crowd that generates pedestrian locations and populates the crowd using
    exact pedestrian models.
    """

    arrivalGap = 1      # HSI Paper Section 5.4

    def __init__(self, numPedestrians, length, width, sync):
        """
        Initializes an instance of the ExactCrowd class.

        :param numPedestrians: The number of pedestrians in the crowd.
        :param length: The length of the crowd.
        :param width: The width of the crowd.
        :param sync: Whether the pedestrians are synchronized (0 for unsynchronized, 1 for synchronized).
        """
        super().__init__(numPedestrians, length, width, sync)
        self.generateLocations()
        self.populateCrowd()

    def generateLocations(self):
        """
        Generates the pedestrian locations.
        """
        self.locations = -self.arrivalGap*np.array(range(self.numPedestrians))

    def populateCrowd(self):
        """
        Populates the crowd with exact pedestrian models.
        """
        for i in range(self.numPedestrians):
            self.addExactPedestrian(self.locations[i], self.iSync[i])

    @classmethod
    def setArrivalGap(cls, arrivalGap):
        """
        Sets the arrival gap for the crowd.

        :param arrivalGap: The arrival gap for the crowd.
        """
        cls.arrivalGap = arrivalGap

