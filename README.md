# African-language-Speech-Recognition---Speech-to-Text


The design of this intelligent form requires selected people to install an app on their mobile phone, and whenever they buy food, 
they use their voice to activate the app to register the list of items they just bought in their own language.
The intelligent systems in the app are expected to live to transcribe the speech-to-text and organize the information in an easy-to-process way in a database. 

The project will use speech data and their transcriptions to train a speech to text model. The goal is to develop a model that can collect data through speech. Five deep learning models will be compared and the best model will be used in the prediction of text from speech input.

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
      
      
      ●	Modeling: Build a Deep learning model that converts speech to text.
      ●	Choose one of deep learning architecture for speech recognition
      ○	Use Connectionist Temporal Classification Algorithm for training and inference 
      ○	CTC takes the character probabilities output of the last hidden layer and derives the correct sequence of characters
      ●	Evaluate your model. 
      ●	Effect of data augmentation: apply different data augmentation techniques and version all of them in DVC. Train model for using these data and study the effect of data     augmentation on the generalization of the model. 
      ●	Model space exploration: using hyperparameter optimization and by slightly modifying the architecture e.g. increasing and decreasing the number of layers to find the best model. 
      ●	Write test units that can be run with CML that will help code reviewers accept Pull Requests (PRs) based on performance gain and other crucial elements. 
      ●	Version different models and track performance through MLFlow
      ●	Evaluate the model using evaluation metrics for speech recognition Word error rate (WER)

      ○	WER compares the predicted output and the target transcript, word by word (or character by character) to figure out the number of differences between them.
      
      
3 Serving predictions on a web interface:


      Use one of the platforms of your choice (Flask, Streamlit, pure javascript, etc.) to design, and build a backend to make inference using your trained model and input parameters collected through a frontend interface. 

      Your dashboard should provide an easy way for a user (in this case managers of the stores) to enter required input parameters, and output the predicted sales amount and customer numbers. 

      ●	Package your model as a docker container for easy deployment in the cloud



<a href="https://github.com/Rukundo725/African-language-Speech-Recognition---Speech-to-Text/tree/speech_recognition/models">models</a>
<a href="https://github.com/Rukundo725/African-language-Speech-Recognition---Speech-to-Text/tree/speech_recognition/notebooks">notebooks</a>
<a href="https://github.com/Rukundo725/African-language-Speech-Recognition---Speech-to-Text/tree/speech_recognition/scripts">scripts</a>
<a href="https://github.com/Rukundo725/African-language-Speech-Recognition---Speech-to-Text/tree/speech_recognition/data1">data1</a>

     
      
      
