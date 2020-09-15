import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# ------ Shortcuts to allow quick switching of theta ------
theta_variants = [
    0, #0
    np.pi / 32, #1
    np.pi / 16, #2
    3 * np.pi / 32, #3
    3 * np.pi / 16, #4
    6 * np.pi / 32, #5
    6 * np.pi / 16, #6
    np.pi / 2 #7
]


# --------------- USER OPTIONS ---------------
saveOutput = False
displayOutput = True
steps = 20
theta = theta_variants[7]

usingRawWave = False

verboseOutput = False
dropGraphZeros = True
thresholdDropValue = 0
# --------------- EXPERIMENTAL ---------------
debugNormalizeState = 0
ignoreImaginaryCoin = True

drawReal = False
compareRealToImaginary = False
combineRealAndImaginary = True

#---------------------------------------------

# Used for the titles of graphs
strTheta = r"pi÷$" + format(theta / np.pi, ".2g") + r"$"
if usingReal:
    strTheta = "Real " + strTheta
else:
    strTheta = "Imag " + strTheta



# Construct |up> and |down>, here represented by |0> and |1>
upK = np.array([1, 0], dtype='complex')  # 0
downK = np.array([0, 1], dtype='complex')  # 1

# Construct a number of different 'quantum coins'
coin_imaginary = np.array([[1, 1j], [1j, 1]], dtype='complex') / np.sqrt(2)
coin_hadamard = np.array([[1, 1], [1, -1]], dtype='complex') / np.sqrt(2)


coin_fauxReal = np.array([[.5 * np.cos(theta), -.5 * np.sin(theta)],
                          [.5 * np.sin(theta), .5 * np.cos(theta)]], dtype='complex')
if not ignoreImaginaryCoin:
    coin_fauxReal -= np.array([[.5j * np.sin(theta), -.5j * np.cos(theta)],
                              [-.5j * np.cos(theta), -.5j * np.sin(theta)]], dtype='complex')


# For the main run of the program, this is where the coin to use is set
myCoin = coin_fauxReal

initialCoin = upK # What is the initial state of the quantum coin?

# ---- The following are frequent alternate coin states that can be used --------
#initialCoin = np.array([(1/np.sqrt(2)), (1j/np.sqrt(2))], dtype='complex')
#initialCoin = initialCoin - (initialCoin * 1j)

#initialCoin = (((upK * 1j) + (downK)) / np.sqrt(2)) - (((upK) + (downK * 1j)) / np.sqrt(2))
#initialCoin = downK

# ----------------------- QUBIT FUNCTIONS --------------------------
# Helper function
def GetPosShift(range, shift):
    return np.roll(np.eye(range, dtype='complex'), shift, axis=0)

# Move left or right along the number line
def GetStepOperator(coin_matrix, range):
    tl, br = np.zeros((2, 2), dtype=complex), np.zeros((2, 2), dtype=complex)
    # -- REAL i guess
    tl[0][0] = coin_matrix[0][0]
    br[1][1] = coin_matrix[1][1]

    ret = np.kron(GetPosShift(range, 1), tl) + np.kron(GetPosShift(range, -1), br)
    return ret

# Makes the coin operator act on the coin state, and the step operator on both coin and step states
def GetWalkOperator(coin_matrix, range):
    kron = np.kron(np.eye(range), coin_matrix)
    return GetStepOperator(coin_matrix, range).dot(np.kron(np.eye(range), coin_matrix))

def GetFullWalk(coin_matrix, range, steps, initialState):
    m_power = np.linalg.matrix_power(GetWalkOperator(coin_matrix, range), steps)
    wave = m_power.dot(initialState)
    wave = NormalizeWave(wave)
    return wave

def NormalizeWave(wave):
    probs = np.abs(wave)**2
    return wave * np.sqrt(1 / np.sum(probs))

def PerformMeasurement(max_range, waveFunction, steps):

    probabilities = np.empty(max_range, dtype='complex')
    for i in range(max_range):
        position = np.zeros(max_range, dtype='complex')
        position[i] = 1
        diagonal = np.outer(position, position)

        fullDiagonal = np.kron(diagonal, np.eye(2, dtype='complex'))
        projection = fullDiagonal.dot(waveFunction)

        conj = projection.conjugate()
        probabilities[i] = np.vdot(projection, conj)



    return probabilities

# ------------------------------------------------------------------


max_range = (steps * 2) + 1 # force odd

zeroPosition = np.zeros(max_range) # Array to store the initial position
zeroPosition[steps] = 1 # The first position will be in the middle of the number line



# ---------------- Implementation ---------------
xdata = np.linspace(-steps, steps, ((steps * 2 + 1)), dtype='complex')



# ------------------- LOOP -------------------------
# Collect the probabilities in an array
def RunCompleteWalk(b_isUsingRaw, b_isUsingReal, customCoin=myCoin):
    rawData = np.zeros((steps + 1, (steps * 2) + 1), dtype='complex')

    xExpect = np.zeros(steps + 1, dtype='complex')
    xSquaredExpect = np.zeros(steps + 1, dtype='complex')
    vExpect = np.zeros(steps + 1, dtype='complex')
    vSquaredExpect = np.zeros(steps + 1, dtype='complex')
    xdata = np.linspace(-steps, steps, ((steps * 2 + 1)), dtype='complex')

    xUncert = np.zeros(steps + 1, dtype='complex')
    vUncert = np.zeros(steps + 1, dtype='complex')

    for i in range(steps + 1):
        currentRange = (i * 2) + 1
        zeroP = np.zeros(currentRange)
        zeroP[i] = 1
        initState = np.kron(zeroP, initialCoin)
        if b_isUsingRaw:
            waveFunc = GetFullWalk(customCoin, currentRange, i, initState)
        else:
            if b_isUsingReal:
                waveFunc = GetFullWalk(customCoin, currentRange, i, initState).real
            else:
                waveFunc = GetFullWalk(customCoin, currentRange, i, initState).imag

        # technically this should be in the measurement step, but it's easier to
        # put it here
        waveFunc = np.abs(waveFunc)

        # measure and pad so every output is the same size (for graphing)
        padding = np.zeros((steps * 2) + 1, dtype='complex')
        measured = PerformMeasurement(currentRange, waveFunc, i)
        padding[steps - i :steps + 1 + i] = measured
        rawData[i] = padding

        # Now we calculate the expectation values
        xExpect[i] = np.around(np.sum(rawData[i] * xdata), 10)
        xSquaredExpect[i] = np.around(np.sum(rawData[i] * (xdata**2)), 10)
        if i > 0:
            vExpect[i] = xExpect[i] - xExpect[i - 1]  # This doesn't feel right...
            vSquaredExpect[i] = xSquaredExpect[i] - xSquaredExpect[i - 1]


        if verboseOutput:
            print("{0}% Done (Total prob {1})".format((i / (steps + 1))*100, np.sum(rawData[i])))

    xUncert = np.sqrt(xSquaredExpect - np.power(xExpect, 2))
    vUncert = np.sqrt(vSquaredExpect - np.power(vExpect, 2))
    totalUncert = xUncert * vUncert

    return rawData, xExpect, xSquaredExpect, vExpect, vSquaredExpect, totalUncert

def extract_position_lines(velocity):
    newXOut = np.zeros_like(velocity)
    newXOut[0] = 0
    for i in range(1, len(velocity)):
        newXOut[i] = newXOut[i - 1] + velocity[i]

    return newXOut


def PerformComplexSpaceEvolution():
    _realOut, _imOut = list(), list()
    _realDiff, _imDiff = list(), list()
    for t in np.arange(0, 1*np.pi, (np.pi / 32)):
        newCoin = np.array([[.5 * np.cos(t), -.5 * np.sin(t)],
                          [.5 * np.sin(t), .5 * np.cos(t)]], dtype='complex')
        newCoin -= np.array([[.5j * np.sin(t), -.5j * np.cos(t)],
                             [-.5j * np.cos(t), -.5j * np.sin(t)]], dtype='complex')

        _real, _, _, _, _, _ = RunCompleteWalk(usingRawWave, True, customCoin=newCoin)
        _imag, _, _, _, _, _ = RunCompleteWalk(usingRawWave, False, customCoin=newCoin)

        peakToObserve = 10

        for i in range(steps):
            _realOut.append((_real[i][peakToObserve]))
            _imOut.append((_imag[i][peakToObserve]))

            if i > 0 and _realOut[i] + _imOut[i] is not (0 + 0j):
                _realDiff.append(_realOut[i] - _realOut[i - 1])
                _imDiff.append(_imOut[i] - _imOut[i-1])

    return _realOut, _imOut, _realDiff, _imDiff


_data, _xExpect, _xSqExpect, _vExpect, _VSqExpect, _uncert = RunCompleteWalk(usingRawWave, True)
_data2, _xExpect2, _xSqExpect2, _vExpect2, _VSqExpect2, _uncert2 = RunCompleteWalk(usingRawWave, False)


# ---------------- Animation ---------------
fig, axis = plt.subplots()
axis.set_xlim(-steps, steps)
axis.set_ylim(0, 1)
axis.set_title("Position evolution for {0} theta, {1} steps".format(strTheta, steps))
axis.set_xlabel("X")
axis.set_ylabel("Probability")
pltLine, = axis.plot(0, 0)

def get_anim_frame(i):
    if dropGraphZeros:
        if drawReal:
            non_zero_raw = np.ma.masked_equal(_data[i], thresholdDropValue)
        else:
            non_zero_raw = np.ma.masked_equal(_data2[i], thresholdDropValue)

        if combineRealAndImaginary:
            non_zero_raw = np.ma.masked_equal(_data[i] + _data2[i], thresholdDropValue)


        non_zero_raw = non_zero_raw.compressed()
        non_zero_raw = np.pad(non_zero_raw, (1, 1), 'constant')

        xdata = np.linspace(-i, i, ((i * 2 + 1)), dtype='complex')
        newXData = xdata[::2]
        newXData = np.pad(newXData, (1, 1), 'constant', constant_values=(-i-2, i+2))
        pltLine.set_ydata(non_zero_raw)
        pltLine.set_xdata(newXData)

    else:
        xdata = np.linspace(-steps, steps, ((steps * 2 + 1)), dtype='complex')
        if drawReal:
            pltLine.set_ydata(_data[i])
        else:
            pltLine.set_ydata(_data2[i])

        pltLine.set_xdata(xdata)

    return pltLine

animation = FuncAnimation(fig, func=get_anim_frame, frames= steps + 1, interval = 400)
if saveOutput:
    animation.save('FullFauxFinal/{0}_theta_{1}_steps.mp4'.format(strTheta, steps), writer='ffmpeg')

oneDAxis = np.linspace(0, steps, steps + 1)

# ----------------- Final Value Graph ---------------------
plt.clf()
fig, ax = plt.subplots()

if dropGraphZeros:
    if drawReal:
        mask = np.ma.masked_where(_data[steps] <= thresholdDropValue, _data[steps])

        non_zero_raw = np.ma.masked_where(np.ma.getmask(mask), _data[steps])
        non_zero_raw = non_zero_raw.compressed()

        newXData = np.ma.masked_where(np.ma.getmask(mask), xdata)
        newXData = newXData.compressed()

        ax.plot(newXData, non_zero_raw, 'b', label='Re')

    if compareRealToImaginary:
        mask2 = np.ma.masked_where(_data2[steps] <= thresholdDropValue, _data2[steps])

        non_zero_raw2 = np.ma.masked_where(np.ma.getmask(mask2), _data2[steps])
        non_zero_raw2 = non_zero_raw2.compressed()

        newXData = np.ma.masked_where(np.ma.getmask(mask2), xdata)
        newXData = newXData.compressed()
        ax.plot(newXData, non_zero_raw2, 'r', label='Im')

    if combineRealAndImaginary:
        _data_combined = _data[steps] + _data2[steps]
        mask = np.ma.masked_where(_data_combined <= thresholdDropValue, _data_combined)

        non_zero_raw = np.ma.masked_where(np.ma.getmask(mask), _data_combined)
        non_zero_raw = non_zero_raw.compressed()

        newXData = np.ma.masked_where(np.ma.getmask(mask), xdata)
        newXData = newXData.compressed()

        ax.plot(newXData, non_zero_raw, 'g', label='Total Probability Distribution')
        ax.plot(newXData, non_zero_raw / 2, 'b', label='Real Decomposition')

else:
    if drawReal:
        ax.plot(xdata, _data[steps], 'b', label='Re')
    if compareRealToImaginary:
        ax.plot(xdata, _data2[steps], 'r', label='Im')
    if combineRealAndImaginary:
        ax.plot(xdata, _data[steps] + _data2[steps], 'g', label='Combined')

ax.set_title("Final step for {0} theta, {1} steps".format(strTheta, steps))
ax.set_xlabel("X")
ax.set_ylabel("Probability")
ax.set_ylim(0, 1)


ax.legend()
if saveOutput:
    plt.savefig('FullFauxFinal/FINAL_{0}_theta_{1}_step.png'.format(strTheta, steps))
if displayOutput:
    plt.show()


drawAdditional = True
if drawAdditional:
    # ----------------- Expected values graph ---------------------
    plt.clf()

    fig, ax = plt.subplots()

    if drawReal:
        ax.plot(oneDAxis, _vExpect, 'C1', label='<v> (Real)')
    if compareRealToImaginary:
        ax.plot(oneDAxis, _vExpect2, 'C2', label='<v> (Imag)')
    if combineRealAndImaginary:
        ax.plot(oneDAxis, _vExpect2 + _vExpect, 'C3', label='<v> (Combined)')

    #ax.plot(oneDAxis, vSquaredExpect, 'C2', label='<v^2>')
    #ax.plot(oneDAxis, xExpect, 'C3', label='<x>')
    ax.set_title("Velocity at {0} theta, {1} steps".format(strTheta, steps))
    ax.set_xlabel("X")

    #if theta > 0:
    #    ax.set_ylim(0, 2)

    ax.legend()
    if saveOutput:
        plt.savefig('ThetaVariants/EXPECTS_{0}_theta_{1}_step.png'.format(strTheta, steps))
    if displayOutput:
        plt.show()


    # ----------------- Combined uncertainty graph ---------------------
    plt.clf()
    fig, ax = plt.subplots()
    if drawReal:
        ax.plot(oneDAxis, _uncert, 'C1', label='Δx * Δv (Real)')
    if compareRealToImaginary:
        ax.plot(oneDAxis, _uncert2, 'C2', label='Δx * Δv (Imag)')
    if combineRealAndImaginary:
        ax.plot(oneDAxis, _uncert2 + _uncert, 'C3', label='Δx * Δv (Combined)')

    ax.set_title("Uncertainty evolution for {0} theta, {1} steps".format(strTheta, steps))
    ax.set_xlabel("T")
    ax.set_ylabel("Uncertainty")
    ax.set_xlim(0, steps)

    ax.legend()
    if saveOutput:
        plt.savefig('ThetaVariants/UNCERT_{0}_theta_{1}_step.png'.format(strTheta, steps))
    if displayOutput:
        plt.show()



    _testReal, _testImag, _trDiff, _tiDiff = PerformComplexSpaceEvolution()
    plt.clf()
    fig, ax = plt.subplots()

    ax.scatter(np.sin(_trDiff), np.sin(_tiDiff))

    ax.set_title("Complex space distribution at x=28 for each delta theta and step".format(strTheta, steps))
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.set_aspect('equal')
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlim(-0.3, 0.3)

    if saveOutput:
        plt.savefig('ThetaVariants/RVI_{0}_theta_{1}_step.png'.format(strTheta, steps))
    if displayOutput:
        plt.show()


    plt.clf()
    fig, ax = plt.subplots()

    ax.scatter(_vExpect, _vExpect2)

    ax.set_ylim(-1, 1)
    ax.set_xlim(-1, 1)

    ax.set_title("Real vs imaginary velocities over {1} steps at {0} theta".format(strTheta, steps))
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_aspect('equal')

    if saveOutput:
        plt.savefig('ThetaVariants/RVIVELO_{0}_theta_{1}_step.png'.format(strTheta, steps))
    if displayOutput:
        plt.show()
