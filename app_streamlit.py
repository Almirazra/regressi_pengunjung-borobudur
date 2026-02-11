import pandas as pd
import joblib
import streamlit as st

st.set_page_config(
	page_title = ("regressi"),
	page_icon =":alien:"
)

st.title("Pengunjung Borobudur")
st.markdown("Aplikasi machine learning yang memprediksi pengunjung di borobudur")

model_linear=joblib.load("model_linear.joblib")

hari_type = st.selectbox("Hari", ["weekend","weekday"])
musim = st.selectbox("Musim", ["kemarau","hujan"])
suhu_rata_rata = st.slider("Suhu", 19.0, 35.9, 25.6)
ada_event_budaya = st.selectbox("Event Budaya", ["ya","tidak"])
harga_tiket_ribu = st.slider("Harga", 50.0, 100.0, 76.6)
jumlah_pengunjung = st.slider("Jumlah pengunjung", 10000, 50000, 21396)

if st.button("prediksi"):
	data_baru=pd.DataFrame([[hari_type, musim, suhu_rata_rata,ada_event_budaya,harga_tiket_ribu,jumlah_pengunjung]],columns=["hari_type","musim","suhu_rata_rata","ada_event_budaya","harga_tiket_ribu","jumlah_pengunjung"])
	prediksi=model_linear.predict(data_baru)[0]
	st.success(f"model memprediksi {prediksi:.0f} pengunjung")
	st.balloons()

