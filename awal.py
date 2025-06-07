import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# --- Judul Aplikasi ---
st.title("🎓 Prediksi Status Kelulusan Mahasiswa")
st.write("Gunakan model KNN untuk memprediksi apakah mahasiswa lulus tepat waktu atau terlambat.")

# --- Load Dataset ---
@st.cache_data
def load_data():
    df = pd.read_csv("kelulusan_mhs.csv")
    df = df.drop(columns=["NAMA", "STATUS MAHASISWA"])  # Kolom tidak relevan
    df = df.fillna(0)  # Ganti NaN jadi 0 (misalnya IPS 8 kosong)
    return df

df = load_data()

# --- Preprocessing ---
label_cols = ['JENIS KELAMIN', 'STATUS NIKAH', 'STATUS KELULUSAN']
le_dict = {}

for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le  # Simpan encoder agar bisa dipakai kembali

# --- Split Data ---
X = df.drop(columns=['STATUS KELULUSAN'])
y = df['STATUS KELULUSAN']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Model ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# --- Evaluasi Model ---
y_pred = knn.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
st.subheader("📊 Evaluasi Model")
st.write(f"Akurasi: *{akurasi:.2f}*")
st.text(classification_report(y_test, y_pred, target_names=le_dict['STATUS KELULUSAN'].classes_))

# --- Form Input User ---
st.subheader("📝 Input Data Mahasiswa Baru")

jenis_kelamin = st.selectbox("Jenis Kelamin", le_dict['JENIS KELAMIN'].classes_)
status_nikah = st.selectbox("Status Nikah", le_dict['STATUS NIKAH'].classes_)
umur = st.slider("Umur", 18, 35, 24)

ips_values = []
for i in range(1, 9):
    ips = st.slider(f"IPS Semester {i}", 0.00, 4.00, 3.00, step=0.01)
    ips_values.append(ips)

ipk = st.slider("IPK", 0.00, 4.00, 3.00, step=0.01)

# --- Prediksi ---
if st.button("🔮 Prediksi Kelulusan"):
    input_data = pd.DataFrame([[
        le_dict['JENIS KELAMIN'].transform([jenis_kelamin])[0],
        umur,
        le_dict['STATUS NIKAH'].transform([status_nikah])[0],
        *ips_values,
        ipk
    ]], columns=X.columns)

    hasil_prediksi = knn.predict(input_data)[0]
    hasil_label = le_dict['STATUS KELULUSAN'].inverse_transform([hasil_prediksi])[0]

    st.success(f"📌 Prediksi: Mahasiswa akan *{hasil_label}* waktu.")
