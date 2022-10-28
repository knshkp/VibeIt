FROM python3.8
COPY . /VIBE_IT
WORKDIR /VIBE_IT
EXPOSE  8501
RUN pip install -r requirements.txt
CMD streamlit run music.py

