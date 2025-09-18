# Yolov5_Pytorch_Deteksi_Kesalamatan_Bermotor
Deskripsi

Project ini merupakan implementasi Computer Vision untuk mendeteksi aspek keselamatan pengendara bermotor menggunakan YOLOv5 (PyTorch).
Model dilatih untuk mengklasifikasikan kondisi pengendara berdasarkan atribut keselamatan, seperti:

•	Pakai Helm

•	Tidak Pakai Helm

•	Pakaian Tertutup

•	Pakaian Terbuka

•	Pakai Sepatu

•	Tidak Pakai Sepatu

Sistem ini berjalan hanya pada console (tanpa UI) dan ditujukan sebagai proyek edukasi/kampus untuk memahami implementasi deteksi objek berbasis deep learning.
________________________________________
Dataset

Dataset digunakan dari Roboflow:

🔗 Keamanan Pengendara – Roboflow Universe
Dataset ini telah dianotasi untuk mendukung training YOLOv5 dengan berbagai kelas keselamatan pengendara.
________________________________________
Training

Proses training dilakukan menggunakan Google Colab agar lebih mudah diakses tanpa memerlukan GPU lokal.
Langkah umum:

1.	Clone YOLOv5 repository di Google Colab.
2.	
3.	Download dataset dari Roboflow (API key atau export langsung).
4.	
5.	Jalankan training dengan perintah seperti:
!python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt --name deteksi_keselamatan

6.	Simpan hasil training (weights .pt) ke Google Drive untuk inferensi selanjutnya.
________________________________________
Kenapa Menarik?

•	Relevan dengan keselamatan berkendara — mendukung edukasi pentingnya atribut keselamatan di jalan.

•	Project kampus yang aplikatif — tidak hanya teori, tetapi implementasi nyata deep learning.

•	Mudah direplikasi — dapat dijalankan di Google Colab tanpa setup rumit.

•	Dasar untuk penelitian lanjutan — bisa diperluas dengan UI, integrasi IoT, atau sistem real-time di CCTV.
________________________________________

Description

This project is an implementation of Computer Vision to detect motorcycle rider safety aspects using YOLOv5 (PyTorch).
The model is trained to classify rider conditions based on safety attributes, such as:

•	Wearing Helmet

•	Not Wearing Helmet

•	Wearing Covered Clothing

•	Wearing Open Clothing

•	Wearing Shoes

•	Not Wearing Shoes

The system runs only in the console (no UI) and is intended as an educational/campus project to understand object detection implementation using deep learning.
________________________________________
Dataset

The dataset is provided by Roboflow:

🔗 Rider Safety – Roboflow Universe
This dataset has been annotated to support YOLOv5 training with various rider safety classes.
________________________________________
Training

The training process is performed using Google Colab, making it accessible without requiring a local GPU.
General steps:

1.	Clone the YOLOv5 repository in Google Colab.
2.	
3.	Download the dataset from Roboflow (via API key or direct export).
4.	
5.	Run the training command:
!python train.py --img 640 --batch 16 --epochs 100 --data data.yaml --weights yolov5s.pt --name rider_safety_detection

6.	Save the trained weights (.pt file) to Google Drive for later inference.
________________________________________
Why It’s Interesting

•	Relevant to road safety — promotes awareness of essential safety attributes for riders.

•	Applicable academic project — not just theory, but a real-world deep learning implementation.

•	Easy to replicate — can be executed on Google Colab without complex setup.

•	Foundation for further research — can be extended with UI, IoT integration, or real-time CCTV systems.
________________________________________


![image](https://github.com/vandot5647/Yolov5_Pytorch_Deteksi_Kesalamatan_Bermotor/assets/95358566/0c35575c-7d38-4195-a972-f160ff1fdcfd)
