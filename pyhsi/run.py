import pyhsi


def main():
    """
        Runs the simulation for analyzing pedestrian-induced vibrations on a beam.

        Returns:
            None
        """
    # Define Crowd
    numPedestrians = 100
    length = 50
    width = 1
    sync = 0
    # crowd = pyhsi.DeterministicCrowd(numPedestrians, length, width, sync)   # Define Crowd
    crowd = pyhsi.RandomCrowd(numPedestrians, length, width, sync)   # Define Crowd
    # crowd = pyhsi.SinglePedestrian()

    # Define Beam
    beam = pyhsi.Beam()

    # Initialize Solver
    FeMmSolver = pyhsi.FeMmSolver(crowd, beam)
    FeMmResults = FeMmSolver.solve()
    FeMmResults.printMaxMidspanRMS()
    FeMmResults.plotMidspanAcceleration()
    # FeMmResults.save()

    # FeSmdSolver = pyhsi.FeSMDSolver(crowd, beam)
    # FeSmdResults = FeSmdSolver.solve()
    # FeSmdResults.save()

    # results.printMaxRMS(poi=4)          # Process results
    # results.plotAcceleration(poi=4)

    pass


if __name__ == '__main__':
    main()