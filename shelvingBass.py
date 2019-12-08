# Name : FNU Rahasya Chandan
# UTA Id: 1000954962
# Shelving Filter to Change Gain of Low Frequencies

import numpy as np 
import matplotlib.pyplot as plt # matlab plotting in python
import soundfile as sf

file = open('shelvingConfig.txt', 'r') 
arr = file.read().splitlines()

# the text file above is for configuring the filter as below:
P_9_1 = arr[0] #name of an audio file
g = int(arr[1])    #gain g of the filter
fc = int(arr[2])   #e cut-off frequency fc of the filter.

data, sample_rate = sf.read(P_9_1) #PySoundFile, data is the orginal data given

#using DSP_Filters_Electronics_Cookbook_Series_ch11.pdf

theta = 2*np.pi*fc/sample_rate  #normalized cut-off frequency 
mu = (10**(g/20))   #bost/cut gain in db
gamma = (1 - (4/(1+mu))*np.tan(theta/2))/(1 + (4/(1+mu))*np.tan(theta/2))  #(11-2)
alpha = (1-gamma)/2    #(11-2)

# CLP_ShelvingFilterStage()
u = np.zeros(len(data)) # u =0
y = np.zeros(len(data)) # y = 0
u[0] = alpha*(data[0])  # alpha*(x+x1), x +x1= data
y[0] = data[0] + (mu - 1)*u[0] # y = u*(mu-1)+x
n = 1
for n in range(len(data)):
    u[n] = alpha*(data[n] + data[n-1]) + gamma * u[n-1] #(11-5a)
    y[n] = data[n] + (mu - 1) * u[n] #(11-5b)

sf.write('shelvingOutput.wav', y, sample_rate)

# plot the side-by-side fourier transformation using sub_plots:
Data_fft = abs(np.fft.fft(data))    #fourier transformation of original data
Filter_fft = abs(np.fft.fft(y)) #fourier transformation of filtered data

#ii. we have to  plot only the first N/4 values of original and filtered signals
lens1 = round(len(data)/4) 
N_value = np.arange(0, lens1)*(sample_rate/len(data))

# iii. Max amplitude + 100
max_amplitude = max(Data_fft) + 100

df = Data_fft[:lens1]
ff = Filter_fft[:lens1]

# Plot the Original Data Graph
org = plt.subplot(1,2,1)
org.set_title('Original Data')
org.plot(N_value, df)
org.set_ylim(0, max_amplitude)

# Plot the Filtered Data Graph
fil = plt.subplot(1,2,2)
fil.set_title('Filtered Data')
fil.plot(N_value, ff)
fil.set_ylim(0, max_amplitude)

plt.tight_layout()
plt.show()

