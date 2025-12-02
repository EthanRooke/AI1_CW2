import cv2
import mediapipe as mp
import os
import pandas as pd

# Initialise mediapipe.hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)

# Path to your image folder
image_folder = "~/Documents/Computer Science/Year 3/AI/AI_CW2/CW2_Temp_Dataset"  # folder path
output_file = "Temp_Sorted_Data.csv"  # Output file name

data_rows = []

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg"):
        image = cv2.imread(os.path.join(image_folder, filename))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)


