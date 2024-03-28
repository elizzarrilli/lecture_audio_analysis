import librosa
import matplotlib.pyplot as pyplot
import os

if not os.path.exists("temp_plots"):
	os.makedirs('temp_plots')

genre = "classical"
number = "00000"

audio_file = "genres/" + genre + "/" + genre + "." + num + ".wav"
print(audio_file)
x, sr = librosa.load(audio_file, sr=44100) #sr is sampling rate # of sample points per sec of audio
print(x) #returns an array of the sample points
print(x.shape) # returns one number that is num of points in the full audio file

wave_plot = pyplot.figure(figsize=(13,5)) # makes empty figure
librosa.display.waveshow(x, sr=sr, color="blue") #needs to specify color bc of error btw librosa and matplot
pyplot.title("waveplot of " + genre + "." + number)
pyplot.xlabel('Time')
pyplot.ylabel("Y")
pyplot.savefig("temp_plots/" + genre + "." + number +"_waveplot.png")
pyplot.close()

spectral_centroids = librosa.feature.spectral_centroid(y=x,sr=sr)[0]
# spectral centroid is like the median fequency extractrated from wav file
print(spectral_centroids)
print(spectral_centroids.shape)

spectral_rolloff = librosa.feature.spectral_rolloff(y=x+0.01,sr=sr) # measures the highest frequency within a cetrain period of time 
print(spectral_rolloff) # where 90% of audio energy is below this frequency

spectral_bandwidth = librosa.feature.spectral_bandwidth(y=x,sr=sr) # similar to the spread of the data, length of bandwidth that captures 85% of frequencies
print(spectral_bandwidth)

zero_crossing_rate = librosa.feature.zero_crossing_rate(y=x) # how many times the wave crosses 0 in some time interval
# used to det if tone is from a wood/string instrument (constant rate) vs percussion (irreg rate)
# also can capture human voice (more irregular crossing rate)

mfcc = librosa.feature.mfcc(y=x,sr=sr)
print(mfcc) #specialized to extract human voice features, has 19 diff metrics

## plot spectrogram
stft_data = librosa.stft(x)
db_data = librosa.amplitude_to_db(abs(stft_data))
spectrogram_plot = pyplot.figure(figsize=(13,5))
librosa.display.specshow(db_data,sr=sr, x_axis='time', y_axis='hz')
pyplot.colorbar()
pyplot.title("Spectrogram of" +genre+"."+number)
pyplot.savefig("temp_plots/" + genre + "." + number +"_spectrogram.png")

## plot chroma - uses letters to represent pitch, color to represent concentration; focuses on important pitch classes for audio analysis
chroma_stft_data = librosa.feature.chroma_stft(y=x,sr=sr,hop_length=sr)
chromagram = pyplot.figure(figsize=(13,5))
librosa.display.specshow(chroma_stft_data,x_axis='time',y_axis='chroma', )















