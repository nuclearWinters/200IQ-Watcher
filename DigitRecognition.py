import numpy as np

samples = np.loadtxt('./data/generalsamples.data', np.float32)
responses = np.loadtxt('./data/generalresponses.data', np.float32)
responses = responses.reshape((responses.size, 1))

samplesPercent = np.loadtxt('./data/generalsamplesPercent.data', np.float32)
responsesPercent = np.loadtxt('./data/generalresponsesPercent.data', np.float32)
responsesPercent = responsesPercent.reshape((responsesPercent.size, 1))