import numpy as np
from scipy.fftpack import dct
from PIL import Image
import cv2
from pydub import AudioSegment

# Kompresi gambar menggunakan DCT
def compress_image_dct(input_path, output_path):
    image = Image.open(input_path).convert('L')
    image_data = np.array(image, dtype=np.float32)
    block_size = 8
    h, w = image_data.shape
    dct_image = np.zeros_like(image_data)
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if i + block_size <= h and j + block_size <= w:
                block = image_data[i:i+block_size, j:j+block_size]
                dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
                dct_image[i:i+block_size, j:j+block_size] = dct_block
    compressed_image = np.clip(dct_image, 0, 255).astype(np.uint8)
    compressed_image_pil = Image.fromarray(compressed_image)
    compressed_image_pil.save(output_path)

# Kompresi gambar menggunakan DFT
def compress_image_dft(input_path, output_path):
    image = Image.open(input_path).convert('L')
    image_data = np.array(image, dtype=np.float32)
    dft_image = np.fft.fft2(image_data)
    dft_image_real = np.real(dft_image)
    compressed_image = np.clip(dft_image_real, 0, 255).astype(np.uint8)
    compressed_image_pil = Image.fromarray(compressed_image)
    compressed_image_pil.save(output_path)

# Kompresi audio menggunakan DCT
def compress_audio_dct(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    block_size = 1024
    num_blocks = len(audio_data) // block_size
    for i in range(num_blocks):
        block = audio_data[i * block_size:(i + 1) * block_size]
        dct_block = dct(block, type=2)
        audio_data[i * block_size:(i + 1) * block_size] = dct_block
    compressed_audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    compressed_audio.export(output_path, format="mp3", bitrate="64k")

# Kompresi audio menggunakan DFT
def compress_audio_dft(input_path, output_path):
    audio = AudioSegment.from_file(input_path)
    audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32)
    block_size = 1024
    num_blocks = len(audio_data) // block_size
    for i in range(num_blocks):
        block = audio_data[i * block_size:(i + 1) * block_size]
        dft_block = np.fft.fft(block)
        audio_data[i * block_size:(i + 1) * block_size] = dft_block.real
    compressed_audio = AudioSegment(
        audio_data.tobytes(),
        frame_rate=audio.frame_rate,
        sample_width=audio.sample_width,
        channels=audio.channels
    )
    compressed_audio.export(output_path, format="mp3", bitrate="64k")

# Fungsi untuk melakukan kompresi video menggunakan metode DCT
def compress_video_dct(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    codec = cv2.VideoWriter_fourcc(*'avc1')  # Codec H.264
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out = cv2.VideoWriter(output_path, codec, fps, (width, height))
    if not out.isOpened():
        print("Error: Could not open output video writer.")
        return
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Menerapkan DCT pada setiap blok 8x8 dari frame
        for i in range(0, height, 8):
            for j in range(0, width, 8):
                block = gray_frame[i:i+8, j:j+8].astype(np.float32)
                dct_block = cv2.dct(block)
                gray_frame[i:i+8, j:j+8] = dct_block
        
        # Menulis frame yang sudah dimodifikasi ke dalam file video keluaran
        out.write(gray_frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Kompresi video menggunakan DFT
def compress_video_dft(input_path, output_path):
    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))), isColor=False)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dft_frame = np.fft.fft2(gray_frame)
        dft_frame_real = np.real(dft_frame)
        compressed_frame = np.clip(dft_frame_real, 0, 255).astype(np.uint8)
        out.write(compressed_frame)
    cap.release()
    out.release()