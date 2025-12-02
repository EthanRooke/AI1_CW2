import cv2
import mediapipe as mp
import os
import pandas as pd

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

# Paths
image_folder = os.path.expanduser("/Users/ethanrooke/Documents/Computer Science/Year 3/AI/AI1_CW2/CW2_Temp_Dataset")
output_file = "Temp_Sorted_Data.csv"

data_rows = []

# Process each letter folder
for letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
    letter_folder = os.path.join(image_folder, letter)

    if not os.path.exists(letter_folder):
        continue

    print(f"\nProcessing letter: {letter}")

    for filename in os.listdir(letter_folder):
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # Read and process image
        image = cv2.imread(os.path.join(letter_folder, filename))
        if image is None:
            continue

        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Extract landmarks
        if results.multi_hand_landmarks:
            landmarks = []
            for landmark in results.multi_hand_landmarks[0].landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])

            data_rows.append([filename] + landmarks + [letter])
            print(f"  ✓ {filename}")
        else:
            print(f"  ✗ No hand: {filename}")

# Save to CSV
columns = ['instance_id'] + [f'{coord}{i}' for i in range(21) for coord in ['x', 'y', 'z']] + ['label']
df = pd.DataFrame(data_rows, columns=columns)
df.to_csv(output_file, index=False)

print(f"\n✓ Complete! {len(data_rows)} images processed → {output_file}")
print(f"\nClass distribution:\n{df['label'].value_counts().sort_index()}")

hands.close()