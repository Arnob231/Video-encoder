import cv2
import numpy as np
import argparse
from tqdm import tqdm
import subprocess
import os
from pydub import AudioSegment

def extract_audio(input_video, output_audio):
    command = f"ffmpeg -i {input_video} -q:a 0 -map a {output_audio} -y"
    subprocess.call(command, shell=True)

def merge_audio_video(input_video, input_audio, output_video):
    command = f"ffmpeg -i {input_video} -i {input_audio} -c:v copy -c:a aac -strict experimental {output_video} -y"
    subprocess.call(command, shell=True)

def compress_video(input_video, output_video):
    command = f"ffmpeg -i {input_video} -vcodec libx264 -crf 23 {output_video} -y"
    subprocess.call(command, shell=True)

def should_compress(input_video, output_video):
    input_size = os.path.getsize(input_video)
    output_size = os.path.getsize(output_video)
    return output_size > input_size * 10

def swap_blocks(frame, block_size):
    h, w, c = frame.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size

    blocks = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y = i * block_size
            x = j * block_size
            block = frame[y:y + block_size, x:x + block_size]
            blocks.append(block)

    num_blocks = len(blocks)
    swapped_blocks = [None] * num_blocks
    for i in range(num_blocks):
        swapped_blocks[i] = blocks[num_blocks - i - 1]

    scrambled_frame = np.zeros((h, w, c), dtype=np.uint8)
    block_idx = 0
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y = i * block_size
            x = j * block_size
            scrambled_frame[y:y + block_size, x:x + block_size] = swapped_blocks[block_idx]
            block_idx += 1

    leftover_h = h % block_size
    leftover_w = w % block_size
    if leftover_h > 0:
        scrambled_frame[h-leftover_h:h, :, :] = frame[h-leftover_h:h, :, :]
    if leftover_w > 0:
        scrambled_frame[:, w-leftover_w:w, :] = frame[:, w-leftover_w:w, :]

    return scrambled_frame

def reverse_swap_blocks(frame, block_size):
    h, w, c = frame.shape
    num_blocks_h = h // block_size
    num_blocks_w = w // block_size

    blocks = []
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y = i * block_size
            x = j * block_size
            block = frame[y:y + block_size, x:x + block_size]
            blocks.append(block)

    blocks = blocks[::-1]

    unscrambled_frame = np.zeros((h, w, c), dtype=np.uint8)
    block_idx = 0
    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            y = i * block_size
            x = j * block_size
            unscrambled_frame[y:y + block_size, x:x + block_size] = blocks[block_idx]
            block_idx += 1

    leftover_h = h % block_size
    leftover_w = w % block_size
    if leftover_h > 0:
        unscrambled_frame[h-leftover_h:h, :, :] = frame[h-leftover_h:h, :, :]
    if leftover_w > 0:
        unscrambled_frame[:, w-leftover_w:w, :] = frame[:, w-leftover_w:w, :]

    return unscrambled_frame

def scramble_audio(input_file, output_file, scramble_factor):
    audio = AudioSegment.from_file(input_file)
    samples = np.array(audio.get_array_of_samples())
    block_size = max(1, len(samples) // scramble_factor)

    blocks = [samples[i:i + block_size] for i in range(0, len(samples), block_size)]
    for i in range(len(blocks)):
        blocks[i] = blocks[i][::-1]

    permuted_blocks = [None] * len(blocks)
    for i in range(len(blocks)):
        new_position = (i * scramble_factor) % len(blocks)
        permuted_blocks[new_position] = blocks[i]

    scrambled_data = np.concatenate(permuted_blocks)
    scrambled_audio = audio._spawn(scrambled_data.tobytes())
    scrambled_audio.export(output_file, format=input_file.split('.')[-1])
    print("Audio file scrambled successfully.")

def unscramble_audio(input_file, output_file, scramble_factor):
    audio = AudioSegment.from_file(input_file)
    samples = np.array(audio.get_array_of_samples())
    block_size = max(1, len(samples) // scramble_factor)

    blocks = [samples[i:i + block_size] for i in range(0, len(samples), block_size)]
    permuted_blocks = [None] * len(blocks)
    for i in range(len(blocks)):
        original_position = (i * scramble_factor) % len(blocks)
        permuted_blocks[original_position] = blocks[i]

    for i in range(len(permuted_blocks)):
        permuted_blocks[i] = permuted_blocks[i][::-1]

    unscrambled_data = np.concatenate(permuted_blocks)
    unscrambled_audio = audio._spawn(unscrambled_data.tobytes())
    unscrambled_audio.export(output_file, format=input_file.split('.')[-1])
    print("Audio file unscrambled successfully.")

def process_video(input_video, output_video, mode, block_size, temp_audio_file):
    extract_audio(input_video, temp_audio_file)

    temp_scrambled_audio_file = 'temp_scrambled_audio.wav'
    if mode == 'encode':
        scramble_audio(temp_audio_file, temp_scrambled_audio_file, block_size)
    elif mode == 'decode':
        unscramble_audio(temp_audio_file, temp_scrambled_audio_file, block_size)
    else:
        print("Invalid mode.")
        return

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Error: Could not open input video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter('temp_video.avi', fourcc, fps, (width, height))
    progress_bar = tqdm(total=total_frames, desc='Processing Frames', unit='frame')

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if mode == 'encode':
            frame = swap_blocks(frame, block_size)
        elif mode == 'decode':
            frame = reverse_swap_blocks(frame, block_size)

        out.write(frame)
        progress_bar.update(1)

    progress_bar.close()
    cap.release()
    out.release()

    merge_audio_video('temp_video.avi', temp_scrambled_audio_file, 'temp_output_video.avi')
    
    if mode == 'encode':
        if should_compress(input_video, 'temp_output_video.avi'):
            compress_video('temp_output_video.avi', output_video)
        else:
            os.rename('temp_output_video.avi', output_video)
    else:
        os.rename('temp_output_video.avi', output_video)

    os.remove('temp_video.avi')
    os.remove(temp_audio_file)
    os.remove(temp_scrambled_audio_file)
    print(f"Processing completed. Output video saved as {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Video Encoder/Decoder with Audio Scrambling and Smart Compression")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("output_video", help="Path to the output video file")
    parser.add_argument("mode", choices=['encode', 'decode'], help="Mode: 'encode' to scramble, 'decode' to unscramble")
    parser.add_argument("block_size", type=int, help="Block size for video and audio scrambling")
    args = parser.parse_args()

    temp_audio_file = 'temp_audio.wav'
    process_video(args.input_video, args.output_video, args.mode, args.block_size, temp_audio_file)
