
def extract_window(data, window = 50):
    sample = []

    for i in range(0, len(data) - window + 1, window):
        window_data = data.iloc[i : i + window ]
        features = extract_features_window(window_data)
        sample.append(features)

    return pd.DataFrame(sample) 



def extract_peak(data, window=25):
    data = data.copy()
    data['svm'] = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
  
    peak_index = np.argmax(data['svm'].values)
    sample = []

    for shift in range(-5, 6, 10): # Đối với tập train => Sử dụng 2 peak defense with noise
        center = peak_index + shift
        start = center - window
        end   = center + window

        if start < 0 or end > len(data):
            continue

        window_event = data.iloc[start:end].reset_index(drop=True)

        if len(window_event) != 2 * window:
            continue


        data_features = extract_features_window(window_event)
        
        sample.append(data_features)

    if len(sample) > 0:
        return pd.DataFrame(sample)
    else:
        return pd.DataFrame()
        
        features = extract_features_window(window_event)

        sample.append(features)
    return pd.DataFrame(sample)



