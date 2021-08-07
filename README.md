# African-language-Speech-Recognition---Speech-to-Text


The design of this intelligent form requires selected people to install an app on their mobile phone, and whenever they buy food, 
they use their voice to activate the app to register the list of items they just bought in their own language.
The intelligent systems in the app are expected to live to transcribe the speech-to-text and organize the information in an easy-to-process way in a database. 

the project consist of 3 main parts:
  1 data preprocessing:
      -loading data both audio files and text files and merge both to one csv file to be used for data preprocessing
      -convert mono to stereo audio channels
      -Standardize sampling rate: standardize and convert all audio to the same sampling rate so that all arrays have the same dimensions
      -resize the audios to have the same lenght
      -Data argumentation: Time Shift to shift the audio to the left or the right by a random amount. 
      -Feature extraction: Spectrogram or Mel Frequency Cepstrum (MFCC).
      -Acoustic modeling
      
 2 Modelling and Deployment using MLOps :
      
      
      
      
