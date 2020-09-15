# --- © Joe Tupper, 2020 ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import scipy.constants as const

from MainWalk import *

saveNewGraphs = False

def ExploreVeloDifference(step, initCoinState):
    tOutputV = np.zeros(65)
    globID = 0

    for t in np.arange(0, 2*np.pi + (np.pi / 32), (np.pi / 32)):
        newCoin = np.array([[.5 * np.cos(t), -.5 * np.sin(t)],
                          [.5 * np.sin(t), .5 * np.cos(t)]])
        if not ignoreImaginaryCoin:
            newCoin -= np.array([[.5j * np.sin(t), -.5j * np.cos(t)],
                                       [-.5j * np.cos(t), -.5j * np.sin(t)]], dtype='complex')

        localXExpect = np.zeros(2)

        id = 0
        for i in range(step -1, step + 1):
            tXdata = np.linspace(-step, step, ((step * 2 + 1)))
            tCurrentRange = (i * 2) + 1
            tZeroP = np.zeros(tCurrentRange)
            tZeroP[i] = 1
            tInitState = np.kron(tZeroP, initCoinState)

            tWaveFunc = GetFullWalk(newCoin, tCurrentRange, i, tInitState)
            tProbabilities = np.pad(PerformMeasurement(tCurrentRange, tWaveFunc, i), (step - i, step - i))

            localXExpect[id] = np.around(np.sum(tProbabilities * tXdata), 10)

            if id > 0:
                tOutputV[globID] = localXExpect[id] - localXExpect[id - 1]

            id +=1

        globID += 1

    return tOutputV


def CalculateFullAverageWalkRange(maxRange):
    averageVeloPerTheta = np.zeros(65)
    thetaRange = np.zeros(65)
    exponentTest = np.zeros(65)
    spacings = list()

    globID = 0

    for t in np.arange(0, 1*np.pi + (np.pi / 32), (np.pi / 32)):
        newCoin = np.array([[.5 * np.cos(t), -.5 * np.sin(t)],
                          [.5 * np.sin(t), .5 * np.cos(t)]])

        thetaRange[globID] = t
        _, _, _, velo, veloSq, _ = RunCompleteWalk(True, True, customCoin=newCoin)
        averageVeloPerTheta[globID] = np.average(velo[1:])
        exponentTest[globID] = averageVeloPerTheta[globID] / (np.e**(1j * t))

        spacings.append(extract_position_lines(velo))
        globID += 1

    return averageVeloPerTheta, thetaRange, np.asarray(spacings), exponentTest



# ----
analysisStep = 40
# ----

augXAxis = np.arange(0, 2*np.pi + (np.pi / 32), (np.pi / 32))
#vData = CalculateFullAverageWalkRange(analysisStep, upK)
vData, tData, sData, eData = CalculateFullAverageWalkRange(analysisStep)

plt.clf()
fig, ax = plt.subplots()
ax.plot(augXAxis, vData, 'b')
ax.set_title("Changes in <v> for differing theta values at step {0}".format(analysisStep))
plt.xticks(plt.xticks()[0],[r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in plt.xticks()[0]])

ax.set_xlim(0, 1*np.pi)
ax.set_xlabel("Theta")
ax.set_ylabel("<V>")

#ax.legend()
if saveNewGraphs:
    plt.savefig('GeneralAnalysis/Evolution_at_step_{0}.png'.format(analysisStep))
plt.show()


moddedC = 1

#  ---------------- Reletavistic gamma ----------------

wavelength = 1 / (np.sin(augXAxis) * vData)
relEnergy = np.sqrt((wavelength**2) + (np.sin(augXAxis)**2))
relEnergy = (1 / wavelength) + np.sin(augXAxis)

photWave = (wavelength**2) / (((wavelength**2)*(1+(np.sin(augXAxis)**2))) - 1)
photWave = np.nan_to_num(photWave, nan=1)

photEnergy = 1 - (relEnergy)

total_wavelength = photWave + wavelength
total_wavelength = 1 / total_wavelength


gamma = moddedC / np.sqrt(((moddedC**2) - (vData))) # c is 1 in this simulation
gamma = 1 / (np.sqrt(1 - (((vData)**2)/(moddedC**2))))

sinDim = 1 / np.sin(augXAxis)




plt.clf()
fig, ax = plt.subplots()
ax.plot(augXAxis, sinDim, 'C1', label='1/sin(theta)')
ax.plot(augXAxis, gamma**2, 'C2', label='Gamma calculated from <v>')
ax.set_title("Gamma for increasing theta at step {0}".format(analysisStep))
plt.xticks(plt.xticks()[0],[r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in plt.xticks()[0]])

ax.set_xlim(0, 1*np.pi)
ax.set_ylim(1, 4)
ax.set_xlabel("Theta")
ax.set_ylabel("gamma")

ax.legend()
if saveNewGraphs:
    plt.savefig('GeneralAnalysis/Gamma_at_step_{0}.png'.format(analysisStep))
plt.show()



plt.clf()
fig, ax = plt.subplots()
ax.plot(augXAxis, vData / (np.cos(augXAxis)), label='v/e^(i*theta)')
ax.set_title("Exponent tests")
plt.xticks(plt.xticks()[0],[r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in plt.xticks()[0]])

ax.set_xlim(0, 1*np.pi)
#ax.set_ylim(1, 4)
ax.set_xlabel("Theta")
ax.set_ylabel("v/e^(i*theta)")

if saveNewGraphs:
    plt.savefig('GeneralAnalysis/Exponent_at_step_{0}.png'.format(analysisStep))
plt.show()



plt.clf()
fig, ax = plt.subplots()
for i in range(0, len(sData)):
    ax.scatter(sData[i], np.full_like(sData[i], tData[i]), marker='|')


ax.set_title("Full Spacing Analysis")
plt.yticks(plt.yticks()[0],[r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in plt.yticks()[0]])

ax.set_ylim(-(1 * np.pi / 32), 1*np.pi + (1 * np.pi / 32))
ax.set_xlabel("Movement per unit time")
ax.set_ylabel("Theta")

if saveNewGraphs:
    plt.savefig('GeneralAnalysis/Gamma_at_step_{0}.png'.format(analysisStep))
plt.show()



plt.clf()
fig, ax = plt.subplots()
ax.plot(augXAxis, vData, label="V")
ax.plot(augXAxis, np.sqrt(vData), label="Sqrt(V)")


ax.set_title("Velocity Analysis")
#plt.xticks(plt.xticks()[0],[r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in plt.xticks()[0]])

ax.set_xlim(0, np.pi / 2)
#ax.set_ylim(0, 1)
ax.set_xlabel("theta")
ax.set_ylabel("Velocity")
ax.legend()

if saveNewGraphs:
    plt.savefig('GeneralAnalysis/Velocities_at_step_{0}.png'.format(analysisStep))
plt.show()



plt.clf()
fig, ax = plt.subplots()
wavelength = np.nan_to_num(wavelength, nan=1, neginf=0)
ax.plot(augXAxis, vData / wavelength, label='Mass Frequency')
ax.plot(augXAxis, vData / photWave, label='Photonic Frequency')


ax.set_title("RFT Frequency Analysis")
plt.xticks(plt.xticks()[0],[r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in plt.xticks()[0]])

ax.set_xlim(0, np.pi / 2 - (1 * np.pi / 32))
#ax.set_ylim(-190, 190)
ax.set_xlabel("Theta")
ax.set_ylabel("Frequency")
ax.legend()

if saveNewGraphs:
    plt.savefig('GeneralAnalysis/Energies_at_step_{0}.png'.format(analysisStep))
plt.show()



for i in range(0, len(vData)):
    planck = 6.62607015E-34
    wavelength = (np.sin(augXAxis[i]) * vData[i])
    relEnergy = np.sqrt((wavelength**2) + (np.sin(augXAxis)**2))
    #relEnergy = gamma * np.sin(augXAxis)


    print("{0}: {1}: {2}: {3}".format(vData[i], np.sin(augXAxis[i]), wavelength, relEnergy[i]))
relEnergy[0] = 0

plt.clf()
fig, ax = plt.subplots()
wavelength = np.nan_to_num(wavelength, nan=1, neginf=0)
ax.plot(augXAxis, relEnergy)

ax.set_title("")
plt.xticks(plt.xticks()[0],[r"$" + format(r/np.pi, ".2g")+ r"\pi$" for r in plt.xticks()[0]])

ax.set_xlim(0, np.pi / 2 - (1 * np.pi / 32))
ax.set_ylim(0, 1)
ax.set_xlabel("Theta")
ax.set_ylabel("Energy")

if saveNewGraphs:
    plt.savefig('GeneralAnalysis/Energies_at_step_{0}.png'.format(analysisStep))
plt.show()
