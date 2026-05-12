import pickle
# Load data from data.pkl
with open('data.pkl', 'rb') as f:
    data = pickle.load(f)

# Split data
train_data = data[:1000]
valid_data = data[1000:1100]
test_data = data[1100:1200]

# Save to novel_train.pkl
with open('novel_train.pkl', 'wb') as f:
    pickle.dump(train_data, f)

# Save to novel_valid.pkl
with open('novel_valid.pkl', 'wb') as f:
    pickle.dump(valid_data, f)

# Save to novel_test.pkl
with open('novel_test.pkl', 'wb') as f:
    pickle.dump(test_data, f)