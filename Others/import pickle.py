import pickle

# Open the file in binary read mode ('rb')
with open('stgnf_scores.pkl', 'rb') as file:
    data = pickle.load(file)

# Check what kind of object it is
print(f"Type of data: {type(data)}")

# View the contents
print(data)