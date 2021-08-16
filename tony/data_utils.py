"""
    file containing functions related to data processing
"""
import os
import numpy as np

'''
    Load .wav extension audio files using 'wave' library
'''
import wave

# Function to convert frame level audio into atomic time
def frames_to_time(total_length, sr=44100):
    in_time = total_length / sr
    hour = int(in_time / 3600)
    minute = int((in_time - hour*3600) / 60)
    second = int(in_time - hour*3600 - minute*60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"

# Function to convert atomic labeled time into frames or seconds
def time_to_frames(input_time, to_frames=True, sr=44100):
    hour, minute, second = input_time.split(':')
    total_seconds = int(hour)*3600 + int(minute)*60 + int(second)
    return total_seconds*sr if to_frames else total_seconds

# Function to load total trainable raw audio lengths
def get_total_audio_length(audio_paths):
    total_length = 0
    for cur_audio_path in audio_paths:
        cur_wav = wave.open(cur_audio_path, 'r')
        total_length += cur_wav.getnframes()    # here, length = # of frames
    return total_length


# Function to load length of an input wav audio
def load_wav_length(audio_path):
    pt_wav = wave.open(audio_path, 'r')
    length = pt_wav.getnframes()
    return length


# Function to load only selected 16 bit, stereo wav audio segment from an input wav audio
def load_wav_segment(audio_path, start_point=None, duration=None, axis=1):
    start_point = 0 if start_point==None else start_point
    duration = load_wav_length(audio_path) if duration==None else duration
    pt_wav = wave.open(audio_path, 'r')
    if pt_wav.getsampwidth()!=2:
        raise ValueError("ValueError: input audio's bit depth should be 16-bit")
    pt_wav.setpos(start_point)
    x = pt_wav.readframes(duration)
    x = np.frombuffer(x, dtype=np.int16)
    X = x / float(2**15)    # needs to be 16 bit format 

    # exception for stereo channels 
    if pt_wav.getnchannels()==2:
        X_l = np.expand_dims(X[::2], axis=axis)
        X_r = np.expand_dims(X[1::2], axis=axis)
        X = np.concatenate((X_l, X_r), axis=axis)
    return X


'''
    Visualizing
'''
import matplotlib.pyplot as plt

# function to visualize 2-dimensional data regarding their time dimension
def visualize_2d_data(inputs, output_dir, time_axis=1, input_labels=None, mode=[None]):
    # only works when time dimensions from the input list are equivalent
    # assert all(x.shape[time_axis] == inputs[0].shape[time_axis] for x in inputs), \
    #         f"Received inequivalent time dimension along inputs.\n   Received time dimension of {[x.shape[time_axis] for x in inputs]}"

    os.makedirs(output_dir, exist_ok=True)
    for cur_mode in mode:
        # plot all inputs separately
        if cur_mode==None:
            for i, cur_input in enumerate(inputs):
                cur_name = input_labels[i] if input_labels and len(input_labels)==len(inputs) else f"input#{i+1}"
                plt.imsave(f"{output_dir}{cur_name}.png", cur_input, cmap='jet', format='png')

        # concatenate inputs to y axis
        # elif mode=="concat":
            

        # projection method -> rescale small 
        # elif mode=="proj":



'''
    piano roll 
'''

# convert 2-d piano roll data into sequences
# def piano_roll_to_sequence(input_piano_roll):
    

