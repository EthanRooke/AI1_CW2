import cv2
import pandas as pd
import numpy as np
import mediapipe as mp

# Initialize MediaPipe drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Load CSV
csv_file = "Temp_Sorted_Data.csv"
df = pd.read_csv(csv_file)

# Specify the image name you want to visualize
target_image = "A_sample_34.jpg"

# Find the row
row = df[df['instance_id'] == target_image]

if row.empty:
    print(f"Image {target_image} not found in CSV!")
else:
    # Load the original image
    print(f"Image {target_image} found in CSV!")
    letter = row['label'].values[0]
    image_path = f"CW2_Temp_Dataset/{letter}/{target_image}"
    print(image_path)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Could not load image: {image_path}")
    else:
        # Extract landmarks from CSV
        landmarks_data = []
        for i in range(21):
            x = row[f'x{i}'].values[0]
            y = row[f'y{i}'].values[0]
            z = row[f'z{i}'].values[0]
            landmarks_data.append([x, y, z])

        # Get image dimensions
        h, w, _ = image.shape

        # Convert normalized coordinates to pixel coordinates
        landmarks_px = []
        for x, y, z in landmarks_data:
            px = int(x * w)
            py = int(y * h)
            landmarks_px.append((px, py))

        # Draw connections (lines between landmarks)
        connections = mp_hands.HAND_CONNECTIONS
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            start_point = landmarks_px[start_idx]
            end_point = landmarks_px[end_idx]
            cv2.line(image, start_point, end_point, (0, 255, 0), 2)

        # Draw landmarks (circles at each point)
        for idx, (px, py) in enumerate(landmarks_px):
            # Draw outer circle
            cv2.circle(image, (px, py), 5, (0, 0, 255), -1)
            # Draw inner circle
            cv2.circle(image, (px, py), 3, (255, 255, 255), -1)

        # Display the result
        # cv2.imshow(f'Hand Landmarks - {target_image}', image)
        # print(f"✓ Displaying {target_image} (Label: {letter})")
        # print("Press any key to close...")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # Method 1: Try matplotlib (works in Jupyter, VS Code, PyCharm)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            # Convert BGR to RGB for matplotlib
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.imshow(image_rgb)
            plt.title(f'Hand Landmarks - {target_image} (Label: {letter})')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            print(f"✓ Displayed using matplotlib")
        except ImportError:
            # Method 2: Fall back to cv2.imshow
            cv2.imshow(f'Hand Landmarks - {target_image}', image)
            print(f"✓ Displaying {target_image} (Label: {letter})")
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Optionally save the output
        # output_path = f"visualized_{target_image}"
        # cv2.imwrite(output_path, image)
        # print(f"✓ Saved to {output_path}")