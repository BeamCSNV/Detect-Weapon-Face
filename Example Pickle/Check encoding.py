import pickle

# กำหนดชื่อไฟล์ encoding.pickle
file_name = 'C:/Users/HP/Desktop/Yolov8 Detect Face&Weapon/Example Pickle/encodings.pickle'

try:
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(data)
except Exception as e:
    print(f"Error loading {file_name}: {e}")

print(data)
