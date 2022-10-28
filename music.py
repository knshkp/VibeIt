import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model
import webbrowser
import pandas as pd

model  = load_model("model.h5")
label = np.load("labels.npy")
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

st.header("Vibe It")
st.image("https://www.speechbuddy.com/blog/wp-content/uploads/2013/04/380593827_fd07c1d722.jpg")
st.write("IF you also want to vibe according to your mood then us this app basically if you also want to listen song according to your mood then use this app")
# Object notation
with st.sidebar:
	with st.echo():
		st.write("Kanishk Pareek")
# Add a selectbox to the sidebar:
# st.sidebar.[st.header("Kanishk Pareek"),
# st.write("How can you connect me")]
# .st.write("How would you like to be contacted?")

# Add a slider to the sidebar:
add_slider = st.sidebar.slider(
    'How much would you like',
    0,100,
)

if "run" not in st.session_state:
	st.session_state["run"] = "true"

try:
	emotion = np.load("emotion.npy")[0]
except:
	emotion=""

if not(emotion):
	st.session_state["run"] = "true"
else:
	st.session_state["run"] = "false"
@st.cache
class EmotionProcessor:
	def recv(self, frame):
		frm = frame.to_ndarray(format="bgr24")

		##############################
		frm = cv2.flip(frm, 1)

		res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

		lst = []

		if res.face_landmarks:
			for i in res.face_landmarks.landmark:
				lst.append(i.x - res.face_landmarks.landmark[1].x)
				lst.append(i.y - res.face_landmarks.landmark[1].y)

			if res.left_hand_landmarks:
				for i in res.left_hand_landmarks.landmark:
					lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			if res.right_hand_landmarks:
				for i in res.right_hand_landmarks.landmark:
					lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
					lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
			else:
				for i in range(42):
					lst.append(0.0)

			lst = np.array(lst).reshape(1,-1)

			pred = label[np.argmax(model.predict(lst))]

			print(pred)
			cv2.putText(frm, pred, (50,50),cv2.FONT_ITALIC, 1, (255,0,0),2)

			np.save("emotion.npy", np.array([pred]))

			
		drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_TESSELATION,
								landmark_drawing_spec=drawing.DrawingSpec(color=(0,0,255), thickness=-1, circle_radius=1),
								connection_drawing_spec=drawing.DrawingSpec(thickness=1))
		drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
		drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)


		##############################

		return av.VideoFrame.from_ndarray(frm, format="bgr24")

lang = st.text_input("Language")
singer = st.text_input("singer")

if lang and singer and st.session_state["run"] != "false":
	webrtc_streamer(key="key", desired_playing_state=True,
				video_processor_factory=EmotionProcessor)


df = pd.DataFrame({
    'first column': ["Spotify","Youtube"]
    })
option = st.selectbox(
    'You will choose?',
     df['first column'])
btn = st.button("Recommend me songs")
if btn:
	if not(emotion):
		st.warning("Please let me capture your emotion first")
		st.session_state["run"] = "true"
	else:
		if option=="Youtube":
			webbrowser.open(f"https://www.youtube.com/results?search_query={lang}+{emotion}+song+{singer}")
		elif option=="Spotify":
			webbrowser.open(f"https://open.spotify.com/search/{singer} {emotion} {lang} song ")
		np.save("emotion.npy", np.array([""]))
		st.session_state["run"] = "false"
