#importing libraries
import pickle
import numpy as np
import cv2
import streamlit as st
from tensorflow import keras
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

#websocket allows interaction with Assembly AI API - taken from Assembly AI instructions
#asyncio - audio inputs and outputs in concurrent matter
#base 64 - encoding and decoding audio signal sent to API
#json - reading audio output
#pyaudio - port library, accept all audio input

import os
from pathlib import Path
import pyaudio
import asyncio
import websockets
import base64
import json


############ EMOTIONS RECOGNITION MODEL

# load emotions model
outcome_dict={0:"angry",
                1:"disgusted",
                2:"afraid",
                3:"happy",
                4:"neutral",
                5:"upset",
                6:"surprised"}

model=pickle.load(open("model.p","rb"))



#detect face
try:
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
except Exception:
    st.write("Face not detected")


RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        #image gray
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(
            image=img_gray, scaleFactor=1.4, minNeighbors=5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(255, 0, 0), thickness=2)
            roi_gray = img_gray[y:y + h, x:x + w]
            #resizing to outcome frame size
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = model.predict(roi)[0]
                #only getting the highest prediction from model until accuracy improves
                maxindex = int(np.argmax(prediction))
                output = str(outcome_dict[maxindex])
            label_position = (x, y)
            cv2.putText(img, output, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img

############ AUDIO TRANSCRIPTION - ASSEMBLY API
# Session state
if 'text' not in st.session_state:
	st.session_state['text'] = 'Listening...'
	st.session_state['run'] = False

#audio stream, parameters can be adjusted, those are the default / usual ones
#build in background noise recognition maybe?
p = pyaudio.PyAudio()
stream = p.open(
   format=pyaudio.paInt16,
   channels=1,
   rate=16000,
   input=True,
   frames_per_buffer=3200
)

#start/stop audio functions
def start_listening():
	st.session_state['run'] = True

def download_transcription():
	read_txt = open('transcription.txt', 'r')
	st.download_button(
		label="Download transcription",
		data=read_txt,
		file_name='transcription_output.txt',
		mime='text/plain')

def stop_listening():
	st.session_state['run'] = False


st.set_page_config(page_title="Classroom Assistant",
                   page_icon="👩‍🏫",
                   layout="centered",
                   initial_sidebar_state="expanded"
                           ) #setting page title and icon
def main():
    # Set the background image
    background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://www.pixground.com/wp-content/uploads/2023/08/Purple-Abstract-Gradient-Layers-AI-Generated-4K-Wallpaper-jpg.webp");
        background-size: 100vw 100vh;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """
    st.markdown(background_image, unsafe_allow_html=True)


    image_path = "sidebar.png"
    st.sidebar.image(image_path, use_column_width=True)

    html_header = """
    <div style="background-color:rgba(68, 27, 125, 0.5);padding:10px">
        <h4 style="color:white;text-align:center;">
            Welcome to your Personal Classroom Assistant
        <br>
        <span style="font-size:16px;">Designed to boost ASD inclusivity in all classrooms</span>
        </h4>
    </div>
    """
    st.markdown(html_header, unsafe_allow_html=True)
    with st.expander("Click here for more info"):
         st.write("This app consists of two main components: live camera feed with emotion recognition features and live transcription app. Camera feed is not being recorded or stored. Transcription data is being saved in text files for further translation.")
         st.checkbox("By proceeding, I agree to processing my personal data.")
    with st.container() as container1:
        # Add a title or header to the box
        st.subheader("Emotion Recognition Feed")

        webrtc_streamer(key="example",
                        mode=WebRtcMode.SENDRECV,
                        rtc_configuration=RTC_CONFIGURATION,
                        video_processor_factory=Faceemotion)
        st.markdown(
        """
        <style>
        [data-testid="stContainer"] {
            background-color: #f7e1ff;
            padding: 10px;
        }
        </style>
        """,
        unsafe_allow_html=True)

    
            
    st.subheader("Translation Feed")
    col1, col2 = st.columns(2)
    col1.button('Start', on_click=start_listening)
    col2.button('Stop', on_click=stop_listening)

	# audio - input, output transcription
    async def send_receive():
        URL = f"wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"
        print(f'Connecting websocket to url {URL}')

        async with websockets.connect(
            URL,
            extra_headers=(("Authorization", st.secrets)), #[YOUR API KEY FOR ASSEMBLY API]#),), 
            ping_interval=5,
            ping_timeout=20
        ) as _ws:
            r = await asyncio.sleep(0.1)
            print("Receiving messages ...")

            session_begins = await _ws.recv()
            print(session_begins)
            print("Sending messages ...")


            async def send():
                while st.session_state['run']:
                    try:
                        data = stream.read(3200)
                        data = base64.b64encode(data).decode("utf-8")
                        json_data = json.dumps({"audio_data":str(data)})
                        r = await _ws.send(json_data)
                    except websockets.exceptions.ConnectionClosedError as e:
                        print(e)
                        assert e.code == 4008
                        break
                    except Exception as e:
                        print(e)
                        assert False, "Not a websocket 4008 error"
                    r = await asyncio.sleep(0.01)


            async def receive():
                while st.session_state['run']:
                    try:
                        result_str = await _ws.recv()
                        result = json.loads(result_str)['text']

                        if json.loads(result_str)['message_type']=='FinalTranscript':
                            print(result)
                            st.session_state['text'] = result
                            st.write(st.session_state["text"])

                            transcription_txt = open('transcription.txt', 'a')
                            transcription_txt.write(st.session_state['text'])
                            transcription_txt.write(' ')
                            transcription_txt.close()

                    except websockets.exceptions.ConnectionClosedError as e:
                        print(e)
                        assert e.code == 4008
                        break
                    except Exception as e:
                        print(e)
                        assert False, "Not a websocket 4008 error"
                
            send_result, receive_result = await asyncio.gather(send(), receive())

    asyncio.run(send_receive())

    if Path('transcription.txt').is_file():
        st.markdown('### Download')
        download_transcription()
        os.remove('transcription.txt')

if __name__ == "__main__":
	main()
