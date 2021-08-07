import librosa   #for audio processing
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile #for audio processing
import warnings
warnings.filterwarnings("ignore")
import math, random
import torch
import torchaudio
from torchaudio import transforms
from IPython.display import Audio
import librosa.display
import logging
from pydub import AudioSegment
import os
import wave,array
from numpy.lib.stride_tricks import as_strided
from mpl_toolkits.axes_grid1 import make_axes_locatable


logging.basicConfig(filename='../logs/audio.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',level=logging.INFO)
audio_path='../data/alldata/'


def caclulate_duration(audio_file):
    print(" ============ Calculating duration of audio file =================")


    logging.info(" ============ Calculating duration of audio file ================= ")
    #pick audio file and let librosa calculate the sample_rate and samples which we shall use to calculate the duration
    
    samples, sample_rate = librosa.load(audio_file)
    duration=float(len(samples)/sample_rate)

    
    
    return duration

def plot_wav(audio_file,sample_rate):
    logging.info(" ============ Plotting audio wav file ================= ")
    audio, rate= librosa.load(audio_file)
    plt.figure(figsize=(20, 5))
    librosa.display.waveplot(audio, sr=sample_rate)


def sample_audio_play(audio_file,sample_rate):
    logging.info(" ============ Accessnig audio sample ================= ")
    audio_sample, rate= librosa.load(audio_file)
    plt.figure(figsize=(20, 5))
    librosa.display.waveplot(audio_sample, sr=sample_rate)
    audios,rate= librosa.load(audio_file,sr=sample_rate)
    ipd.Audio(audio_sample,rate=sample_rate)


def open(audio_file):

    #loading audio file and return signal 


    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)

def mono_conversion(audio_file,new_channel):



    logging.info(" ============ Conerting audio sample from mono to stereo ================= ")
    print("======= Mono to stereo audio conversion")
    
    sig,sir=librosa.load(audio_file)

    #if signal shape is equal to two  (stereo) no need for conversion 
    if (sig.shape[0]==new_channel):

        return audio_file
    #if channel shape is equal to 1 we need to convert it to stereo
    if (new_channel == 1):

        resig =sig[:1,:] 
    # converting mono to stereo 
    else:

        resig=torch.cat([sig,sig])

def standard_sample(audio_file,resample):

    sig,sr=librosa.load(audio_file)

    if (sr == resample):

        return audio_file

    num_channels=sig.shape[0] 
    resig=torchaudio.transforms.Resample(sig,resample)(sig[:1,:])

    if (num_channels > 1):

        sampletwo=torchaudio.transforms.Resample(sr,resample)(sig[1:,:])
        resig=torch.cat([resig,sampletwo])

    
    return ((resig,resample)) 

def resize_audio(audio_file,max_length):

    sig,sr=librosa.load(audio_file) 
    rows,audio_length=sig.shape
    max_len=sr//10000 * max_length

    if (audio_length> max_length):

        sig=sig[:,:max_len]

    elif (audio_length < max_length):

        pad_begin_length=random.randint(0, max_length - audio_length)
        pad_end_length=max_length - audio_length - pad_begin_length

        pad_begin = torch.zeros((rows,pad_begin_length))
        pad_end=torch.zeros((rows,pad_end_length))

       
        sig = torch.cat((pad_begin, sig, pad_end), 1)

    return (sig,sr)  

def time_shift(audio,shift_limit):
    #this function shifts the signal to the left or right based on time 
    sig,sr =audio 
    _,audio_length =sig.shape

    shift_amount=int(random.random()* shift_limit * audio_length)

    return (sig.roll(shift_amount),sr) 


def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    
    sig,sr = aud
    top_db = 80

    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)

def convertor(audio_path):
    sound=AudioSegment.from_wav(audio_path)
    sound=sound.set_channels(2)
    newpath=audio_path.split(".")[0]
    newpath=newpath+'stereo.wav'
    subfolder = os.path.join('stereo',audio_path)
    sound.export(newpath,format='wav')

def make_stereo(audio_path):
    #this function converts mono audio channels into stereo channels 
    logging.info(" ============ Conerting audio sample from mono to stereo ================= ")
    print("======= Mono to stereo audio conversion")
    ifile = wave.open(audio_path)
    #log the info on adio files
    logging.info(ifile.getparams())
    print (ifile.getparams())
    # (1, 2, 44100, 2013900, 'NONE', 'not compressed')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = ifile.getparams()
    assert (comptype == 'NONE')  # Compressed not supported yet
    array_type = {1:'B', 2: 'h', 4: 'l'}[sampwidth]
    print(" ======= Calculting left channel type =====")
    left_channel = array.array(array_type, ifile.readframes(nframes))[::nchannels]
    ifile.close()

    #convert the number of channels to 2
    print("====== converting channels ======= ")
    stereo = 2 * left_channel
    stereo[0::2] = stereo[1::2] = left_channel
    #overwrite the wav file making it a stereo file
    print("====== overwriting wav file ======= ")
    ofile = wave.open(audio_path, 'w')
    ofile.setparams((2, sampwidth, framerate, nframes, comptype, compname))
    ofile.writeframes(stereo.tostring())
    ofile.close()

def create_spectogram(audio_file,fft_length=256,sample_rate=2,hop_length=1):
    samples,sample_rate = librosa.load(audio_file)
    assert not np.iscomplexobj(samples), "Must not pass in complex numbers"

    window = np.hanning(fft_length)[:, None]
    window_norm = np.sum(window**2)

    # The scaling below follows the convention of
    # matplotlib.mlab.specgram which is the same as
    # matlabs specgram.
    scale = window_norm * sample_rate

    trunc = (len(samples) - fft_length) % hop_length
    x = samples[:len(samples) - trunc]

    # "stride trick" reshape to include overlap
    nshape = (fft_length, (len(x) - fft_length) // hop_length + 1)
    nstrides = (x.strides[0], x.strides[0] * hop_length)
    x = as_strided(x, shape=nshape, strides=nstrides)

    # window stride sanity check
    assert np.all(x[:, 1] == samples[hop_length:(hop_length + fft_length)])

    # broadcast window, compute fft over columns and square mod
    x = np.fft.rfft(x * window, axis=0)
    x = np.absolute(x)**2

    # scale, 2.0 for everything except dc and fft_length/2
    x[1:-1, :] *= (2.0 / scale)
    x[(0, -1), :] /= scale

    freqs = float(sample_rate) / fft_length * np.arange(x.shape[0])

    return x, freqs

def plot_spectogram(audio_file):
    x,freqs = create_spectogram(audio_file,fft_length=256,sample_rate=2,hop_length=1)
    fig = plt.figure(figsize=(12,5))
    ax = fig.add_subplot(111)
    im = ax.imshow(x, cmap=plt.cm.jet, aspect='auto')
    plt.title('Spectrogram')
    plt.ylabel('Time')
    plt.xlabel('Frequency')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()

def plot_frequency(audio_file):
    x,freqs = create_spectogram(audio_file,fft_length=256,sample_rate=2,hop_length=1)
    X=librosa.stft(x)
    Xdb=librosa.amplitude_to_db(abs(X))
    plt.figure(figsize=(10,9))
    librosa.display.specshow(Xdb,sr=freqs,x_axis='time',y_axis='hz')
    plt.colorbar

def resmaple_data(audio_file,resample_rate):
    #this function takes in the audio file and resample it a defined sampling rate   
    samples,sample_rate=librosa.load(audio_file)
    samples=librosa.resample(samples,sample_rate,resample_rate)

    return samples  


     


if (__name__== '__main__'):
    # open(audio_path+'SWH-05-20101106_16k-emission_swahili_05h30_-_06h00_tu_20101106_part2.wav')
    # plot_wav()
    make_stereo('..\\data\\alldata\\SWH-05-20101106_16k-emission_swahili_05h30_-_06h00_tu_20101106_part100.wav','newtrack.wav')
 
