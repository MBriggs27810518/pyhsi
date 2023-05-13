"""
PyHSI - Human Structure Interaction -  Results Module
"""
import xlsxwriter
import inquirer
import openpyxl
import pandas as pd
import math
import numpy as np
import sys
from matplotlib import pyplot as plt

from tkinter import *
from tkinter import filedialog


def loadResults():
    """
    Load the results from a file and open the options menu.
    """
    results = Results.open()
    results.options()


class Results:
    """
    Class to represent and manipulate results from pedestrian-structure interaction simulations.
    """

    abbr = {
        'Finite Element': 'fe',
        'Modal Analysis': 'mo',
        'Moving Mass': 'mm',
        'Moving Force': 'mf',
        'Spring Mass Damper': 'smd',
    }

    def __init__(self, t, displacement, velocity, acceleration, pedestrianModel=None, modelType=None, filename=None):
        """
        Initialize a Results instance with simulation results of pedestrian-structure interaction.

        Parameters
        ----------
        t : numpy.ndarray
            Array of time steps
        displacement : numpy.ndarray
            Array of displacements at each time step.
        velocity :numpy.ndarray
            Array of velocities at each time step.
        acceleration : numpy.ndarray
            Array of accelerations at each time step.
        pedestrianModel : str, optional
            Type of pedestrian model used in simulation. Defaults to None.
        modelType : str, optional
            Type of model used in simulation. Defaults to None.
        filename : str, optional
            Name of the file where simulation results are saved. Defaults to None.
        """
        self.t = t
        self.displacement = displacement
        self.velocity = velocity
        self.acceleration = acceleration

        self.pedestrianModel = pedestrianModel
        self.modelType = modelType
        self.filename = filename

        self._midspanAcceleration = None
        self._midspanRMS = None
        self._maxMidspanRMS = None

        # TODO: POI Acceleration

    # region Properties
    @property
    def midspanAcceleration(self):
        """
        Get the acceleration at the midspan of the structure.

        Returns
        -------
        midspanAcceleration: numpy.ndarray
            Array of acceleration at the midspan of the structure.
        """
        if self._midspanAcceleration is None:
            midspanX = self.acceleration.shape[1] // 2 - 1
            self._midspanAcceleration = self.acceleration[:, midspanX]
        return self._midspanAcceleration

    @property
    def midspanRMS(self):
        """
        Calculate the Root Mean Square (RMS) of the acceleration at the midspan of the structure.

        Returns
        -------
        midspanRMS: numpy.float64
            RMS of the acceleration at the midspan of the structure.
        """
        if self._midspanRMS is None:
            midspanX = self.acceleration.shape[1] // 2 - 1
            self._midspanRMS = self.calculateRMS(midspanX)
        return self._midspanRMS

    @property
    def maxMidspanRMS(self):
        """
        Get the maximum RMS of the acceleration at the midspan of the structure.

        Returns:
        maxMidspanRMS: numpy.float64
            Maximum RMS of the acceleration at the midspan of the structure.
        """
        if self._maxMidspanRMS is None:
            self._maxMidspanRMS = max(self.midspanRMS)
        return self._maxMidspanRMS
    # endregion

    # region Open and Save results
    def askSave(self):
        """
        Method to ask user if they want to save the results.
        """
        # Check if user wants to save the results
        saveMessage = "Do you want to save the results?"
        saveChoices = ['Yes', 'No']
        saveQuestion = [inquirer.List('save', message=saveMessage, choices=saveChoices)]
        answer = inquirer.prompt(saveQuestion)

        if answer['save'] == 'Yes':
            self.save()

    def save(self):
        """
        Method to save the results as an Excel file.
        """

        # Get name to save workbook
        filenameMessage = "Enter a filename to save the results under"
        filenameDefault = f"{self.filename[15:-4]}_{self.abbr[self.modelType]}_{self.abbr[self.pedestrianModel]}"
        filenameQuestion = [inquirer.Text('filename', message=filenameMessage, default=filenameDefault)]
        filenameAnswer = inquirer.prompt(filenameQuestion)
        path = f"../simulations/results/{filenameAnswer['filename']}"
        if not path[-5:] == '.xlsx':
            path += '.xlsx'
        self.filename = path

        # Create a Pandas dataframe from the data
        tDF = pd.DataFrame(self.t)
        displacementDF = pd.DataFrame(self.displacement)
        velocityDF = pd.DataFrame(self.velocity)
        accelerationDF = pd.DataFrame(self.acceleration)

        # Create a Panas Excel writer using XlsxWriter as the engine
        writer = pd.ExcelWriter(path, engine='xlsxwriter')

        # Convert the dataframe to an XlsxWriter Excel object
        tDF.to_excel(writer, sheet_name='time', header=False, index=False)
        displacementDF.to_excel(writer, sheet_name='displacement', header=False, index=False)
        velocityDF.to_excel(writer, sheet_name='velocity', header=False, index=False)
        accelerationDF.to_excel(writer, sheet_name='acceleration', header=False, index=False)

        # Close the Pandas Excel writer and output the Excel file
        writer.save()

        print("Saved results as: ", path)

    @classmethod
    def open(cls, filename=None):
        """
        Class method to open an Excel file and load results.

        Parameters
        ----------
        filename: str
            Name of file to open

        Return
        ------
            an instance of Results class with the results loaded from the file
        """

        # Select file from file explorer
        if not filename:
            root = Tk()
            root.withdraw()
            root.call('wm', 'attributes', '.', '-topmost', True)
            filename = filedialog.askopenfilename(
                parent=root,
                title='Select results file',
                filetypes=[("Excel Files", ".xlsx")],
                initialdir="../simulations/results")

            if filename == '':
                print('No file chosen, stopping program.')
                sys.exit()

        t = pd.read_excel(filename, sheet_name='time', header=None).to_numpy().transpose()[0]
        displacement = pd.read_excel(filename, sheet_name='displacement', header=None).to_numpy()
        velocity = pd.read_excel(filename, sheet_name='velocity', header=None).to_numpy()
        acceleration = pd.read_excel(filename, sheet_name='acceleration', header=None).to_numpy()

        print("Loading results from: ", filename)

        return cls(t, displacement, velocity, acceleration, filename)
    # endregion

    def options(self):
        """
        Display options for processing the results.
        """
        # Options for processing the results
        optionsMessage = "How would you like to proceed?"
        optionsChoices = [
            'Print maxRMS at midspan',
            'Plot maxRMS at midspan',
            'Save results',
            'Finish viewing results']
        question = [inquirer.List('next', message=optionsMessage, choices=optionsChoices)]
        answer = inquirer.prompt(question)

        while answer['next'] != 'Finish viewing results':
            if answer['next'] == 'Print maxRMS at midspan':
                self.printMaxMidspanRMS()
            elif answer['next'] == 'Plot maxRMS at midspan':
                self.plotMidspanAcceleration()
            elif answer['next'] == 'Save results':
                self.save()
            answer = inquirer.prompt(question)

    # region Process Results
    def printMaxMidspanRMS(self):
        """
        Print the maximum root mean square acceleration at midspan.
        """
        print(f"Max RMS: {self.maxMidspanRMS:.6f} m/s^2\n")

    def plotMidspanAcceleration(self, title='Acceleration'):
        """
            Plot the acceleration and root mean square at midspan versus time.

            Parameters
            ----------
            title : str, optional
                Title of the plot. Defaults
        """
        plt.figure(figsize=(9, 4))

        # creating the bar plot
        plt.plot(self.t, self.midspanAcceleration, 'r', self.t, self.midspanRMS, 'b')

        plt.xlabel("Time (s)")
        plt.ylabel("Acceleration (m/s^2)")
        plt.title(title)

        # plt.xlim([0, 40])
        plt.show()

    def calculateRMS(self, x):
        """
        Calculate the Root Mean Square of the acceleration at a specified location x.

        Parameters
        ----------
        x : int
            Location of the acceleration.

        Returns
        -------
        float
            The Root Mean Square of the acceleration at the specified location x.
        """
        accelerationAtX = self.acceleration[:, x]
        rms = self.timeRMS(accelerationAtX)
        return rms

    def timeRMS(self, x, RMS_Window=1):
        """
            Calculate the time RMS of the signal.

            Parameters
            ----------
            x : array_like
                The input acceleration.
            RMS_Window : int, optional
                Window of time to use for RMS calculation. Defaults to 1.

            Returns
            -------
            array_like
                The time RMS of the input acceleration.
            """
        # This function returns the tspan-rms of the signal

        n = len(x)
        i = 0
        while self.t[i] < RMS_Window:
            i += 1
        Npts = i
        rNpts = math.sqrt(Npts)

        rms = np.zeros(n)

        i = 1
        while i < Npts:
            vec = x[0:i]
            rms[i - 1] = np.linalg.norm(vec) / math.sqrt(i)
            i += 1

        while i < n:
            vec = x[i - Npts:i]
            rms[i - 1] = np.linalg.norm(vec) / rNpts
            i += 1

        return rms
    # endregion


if __name__ == '__main__':
    loadResults()
