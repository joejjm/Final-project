import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Labels for training videos
dante_labels = {
    'dante-pitch-1.mp4': 'fastball',
    'dante-pitch-2.mp4': 'fastball',
    'dante-pitch-3.mp4': 'fastball',
    'dante-pitch-4.mp4': 'fastball',
    'dante-pitch-5.mp4': 'fastball',
    'dante-pitch-6.mp4': 'curveball',
    'dante-pitch-7.mp4': 'fastball',
    'dante-pitch-8.mp4': 'fastball',
    'dante-pitch-9.mp4': 'curveball',
    'dante-pitch-10.mp4': 'fastball',
}

# Feature extraction: use ball trajectory (x, y) as features
# Assumes you have already run YOLO tracking and saved ball_positions for each video as CSV

def compute_curvature(x, y):
    # Compute discrete curvature for a 2D trajectory
    x = np.array(x)
    y = np.array(y)
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx * dx + dy * dy + 1e-8) ** 1.5
    return curvature

def load_features(video_name, feature_dir='data/ball_features'):
    csv_path = os.path.join(feature_dir, video_name.replace('.mp4', '.csv'))
    if not os.path.exists(csv_path):
        return None
    df = pd.read_csv(csv_path)
    # Only keep rows where both x and y are present (not NaN)
    df_xy = df.dropna(subset=['x', 'y'])
    # If no valid x/y, return None
    if df_xy.empty:
        features = np.zeros(300)  # fallback, will be excluded later
    else:
        # Compute curvature on valid x/y
        curvature = compute_curvature(df_xy['x'], df_xy['y'])
        max_len = 100
        x_vals = df_xy['x'].values
        y_vals = df_xy['y'].values
        curvature_vals = curvature
        # Pad or truncate to max_len
        if len(x_vals) < max_len:
            x_vals = np.pad(x_vals, (0, max_len - len(x_vals)), 'constant')
            y_vals = np.pad(y_vals, (0, max_len - len(y_vals)), 'constant')
            curvature_vals = np.pad(curvature_vals, (0, max_len - len(curvature_vals)), 'constant')
        else:
            x_vals = x_vals[:max_len]
            y_vals = y_vals[:max_len]
            curvature_vals = curvature_vals[:max_len]
        features = np.concatenate([x_vals, y_vals, curvature_vals])
    # Also load glove and vertical height features
    # Compute mean, max, min for glove_height_peak_leg_lift and glove_to_person_top
    glove_max = None
    glove2person_max = None
    cap_height_max = None
    if 'glove_height_peak_leg_lift' in df.columns:
        vals = df['glove_height_peak_leg_lift'].dropna()
        if not vals.empty:
            glove_max = vals.max()
    if 'glove_to_person_top' in df.columns:
        vals2 = df['glove_to_person_top'].dropna()
        if not vals2.empty:
            glove2person_max = vals2.max()
    if 'cap_height' in df.columns:
        vals3 = df['cap_height'].dropna()
        if not vals3.empty:
            cap_height_max = vals3.max()
    # Use only max values for glove features, allow NaN if no values at all
    glove_features = []
    if glove_max is not None and not np.isnan(glove_max):
        glove_features.append(glove_max)
    if glove2person_max is not None and not np.isnan(glove2person_max):
        glove_features.append(glove2person_max)
    vertical_feature = glove2person_max if glove2person_max is not None and not np.isnan(glove2person_max) else None
    if glove_features:
        features = np.concatenate([features, glove_features])
    # Only skip if all features are NaN or all zero
    if np.all(np.isnan(features)) or np.all(features == 0):
        return None, None, None, None, None
    return features, vertical_feature, glove_max, glove2person_max, cap_height_max

def prepare_dataset(video_list, labels=None):
    X, y, valid_videos = [], [], []
    X_vert = []
    for v in video_list:
        # Only use the first two return values for training set
        result = load_features(v)
        if result is None or len(result) < 2:
            print(f"Excluding {v}: missing or NaN in required features (glove or trajectory)")
            continue
        features, vertical_feature = result[0], result[1]
        if features is None:
            print(f"Excluding {v}: missing or NaN in required features (glove or trajectory)")
            continue
        if len(features) == 0:
            print(f"Excluding {v}: empty feature vector")
            continue
        # Exclude if all zeros or NaNs
        if np.all(np.isnan(features)):
            print(f"Excluding {v}: all NaN features")
            continue
        if np.all(features == 0):
            print(f"Excluding {v}: all zero features")
            continue
        X.append(features)
        X_vert.append(vertical_feature)
        valid_videos.append(v)
        if labels:
            y.append(0 if labels[v] == 'fastball' else 1)
    X = np.array(X)
    X_vert = np.array(X_vert).reshape(-1, 1)
    # Outlier exclusion removed: all training videos are included
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if labels:
        return X, X_vert, np.array(y)
    else:
        return X, X_vert

def main():
    # Training
    train_videos = [f'dante-pitch-{i}.mp4' for i in range(1, 11)]


    # # Visualize curvature for training videos
    # plt.figure(figsize=(10, 6))
    # for v in train_videos:
    #     label = dante_labels[v]
    #     csv_path = os.path.join('data/ball_features', v.replace('.mp4', '.csv'))
    #     if os.path.exists(csv_path):
    #         df = pd.read_csv(csv_path)
    #         curvature = compute_curvature(df['x'], df['y'])
    #         plt.plot(curvature, label=f"{v} ({label})")
    # plt.title('Ball Trajectory Curvature for Training Videos')
    # plt.xlabel('Frame Index')
    # plt.ylabel('Curvature')
    # plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.savefig('data/ball_curvature_train.png')
    # plt.show()


    # # Visualize trajectories for training videos
    # plt.figure(figsize=(10, 6))
    # for v in train_videos:
    #     label = dante_labels[v]
    #     csv_path = os.path.join('data/ball_features', v.replace('.mp4', '.csv'))
    #     if os.path.exists(csv_path):
    #         df = pd.read_csv(csv_path)
    #         plt.plot(df['x'], df['y'], label=f"{v} ({label})")
    # plt.title('Ball Trajectories for Training Videos')
    # plt.xlabel('X Position (pixels)')
    # plt.ylabel('Y Position (pixels)')
    # plt.legend(fontsize='small', bbox_to_anchor=(1.05, 1), loc='upper left')
    # plt.tight_layout()
    # plt.savefig('data/ball_trajectories_train.png')
    # plt.show()
    # Update prepare_dataset to handle new return signature, but only for test set below
    X_train, X_train_vert, y_train = prepare_dataset(train_videos, dante_labels)
    if X_train.shape[0] == 0 or X_train.shape[1] == 0:
        print('No valid training data after filtering. Exiting.')
        return
    # Model 1: Trajectory features
    clf_traj = LogisticRegression(max_iter=1000)
    clf_traj.fit(X_train, y_train)
    joblib.dump(clf_traj, 'data/pitch_classifier_traj.joblib')
    print('Trajectory model trained and saved.')
    print('\nTrajectory model coefficients (first 10):', clf_traj.coef_[0][:10])
    print('Trajectory model intercept:', clf_traj.intercept_)
    # Model 2: Vertical height difference feature
    clf_vert = LogisticRegression(max_iter=1000)
    clf_vert.fit(X_train_vert, y_train)
    joblib.dump(clf_vert, 'data/pitch_classifier_vert.joblib')
    print('Vertical difference model trained and saved.')
    print('\nVertical model coefficient:', clf_vert.coef_[0])
    print('Vertical model intercept:', clf_vert.intercept_)
    # Model 3: Combined features (trajectory + vertical)
    X_train_combined = np.hstack([X_train, X_train_vert])
    clf_combined = LogisticRegression(max_iter=1000)
    clf_combined.fit(X_train_combined, y_train)
    joblib.dump(clf_combined, 'data/pitch_classifier_combined.joblib')
    print('Combined model trained and saved.')
    print('\nCombined model coefficients (first 10):', clf_combined.coef_[0][:10])
    print('Combined model intercept:', clf_combined.intercept_)

    # Testing on all videos (1-15)
    test_videos = [f'dante-pitch-{i}.mp4' for i in range(1, 16)]
    # Actual labels for all videos
    test_labels = {
        'dante-pitch-1.mp4': 'fastball',
        'dante-pitch-2.mp4': 'fastball',
        'dante-pitch-3.mp4': 'fastball',
        'dante-pitch-4.mp4': 'fastball',
        'dante-pitch-5.mp4': 'fastball',
        'dante-pitch-6.mp4': 'curveball',
        'dante-pitch-7.mp4': 'fastball',
        'dante-pitch-8.mp4': 'fastball',
        'dante-pitch-9.mp4': 'curveball',
        'dante-pitch-10.mp4': 'fastball',
        'dante-pitch-11.mp4': 'fastball',
        'dante-pitch-12.mp4': 'curveball',
        'dante-pitch-13.mp4': 'fastball',
        'dante-pitch-14.mp4': 'fastball',
        'dante-pitch-15.mp4': 'curveball',
    }
    X_test, X_test_vert = [], []
    valid_test_videos = []
    valid_test_labels = []
    glove_height_peaks = []
    glove_to_person_tops = []
    cap_height_maxes = []
    for v in test_videos:
        features, vertical_feature, glove_max, glove2person_max, cap_height_max = load_features(v)
        if features is not None and not (np.all(np.isnan(features)) or np.all(features == 0)):
            X_test.append(features)
            X_test_vert.append(vertical_feature)
            valid_test_videos.append(v)
            valid_test_labels.append(test_labels.get(v, 'unknown'))
            glove_height_peaks.append(glove_max)
            glove_to_person_tops.append(glove2person_max)
            cap_height_maxes.append(cap_height_max)
        else:
            print(f'Warning: {v} missing, empty, or excluded (all NaN/zero)')
    if X_test:
        X_test = np.array(X_test)
        X_test_vert = np.array(X_test_vert).reshape(-1, 1)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        # Trajectory model predictions
        y_pred_traj = clf_traj.predict(X_test)
        # Vertical difference model predictions
        y_pred_vert = clf_vert.predict(X_test_vert)
        # Combined model predictions
        X_test_combined = np.hstack([X_test, X_test_vert])
        y_pred_combined = clf_combined.predict(X_test_combined)
        print('Predictions for test videos:')
        summary_rows = []
        # Prepare to save all features and predictions
        save_rows = []
        for i, v in enumerate(valid_test_videos):
            actual = valid_test_labels[i]
            pred_traj = 'fastball' if y_pred_traj[i] == 0 else 'curveball'
            pred_vert = 'fastball' if y_pred_vert[i] == 0 else 'curveball'
            pred_combined = 'fastball' if y_pred_combined[i] == 0 else 'curveball'
            # Get raw prediction and probability for vertical model
            vert_raw = int(y_pred_vert[i])
            vert_proba = float(clf_vert.predict_proba(X_test_vert[i].reshape(1, -1))[0][1])
            print(f'--- {v} ---')
            print('Actual label:', actual)
            print('Trajectory model prediction:', pred_traj)
            print('Vertical difference model prediction:', pred_vert)
            print('Combined model prediction:', pred_combined)
            print('Vertical feature value:', X_test_vert[i][0])
            print('Vertical model raw prediction (0=fastball,1=curveball):', vert_raw)
            print('Vertical model curveball probability:', vert_proba)
            print('Feature vector (first 10 x, y, curvature):')
            print('x:', X_test[i][:10])
            print('y:', X_test[i][30:40])
            print('curvature:', X_test[i][60:70])
            summary_rows.append([v, actual, pred_traj, pred_vert, pred_combined])
            glove_max = glove_height_peaks[i]
            cap_height = cap_height_maxes[i]
            glove2person = glove_to_person_tops[i]
            vf = X_test_vert[i][0]
            save_row = {
                'video': v,
                'actual_label': actual,
                'pred_traj': pred_traj,
                'pred_vert': pred_vert,
                'pred_combined': pred_combined,
                'vertical_feature': vf,
                'vertical_raw': vert_raw,
                'vertical_curveball_proba': vert_proba,
                'glove_height_peak_leg_lift': glove_max,
                'glove_to_person_top': glove_to_person_tops[i],
                'cap_height': cap_height
            }
            # Save first 10 x, y, curvature values for inspection
            for j in range(10):
                save_row[f'x_{j}'] = X_test[i][j]
                save_row[f'y_{j}'] = X_test[i][30 + j]
                save_row[f'curvature_{j}'] = X_test[i][60 + j]
            save_rows.append(save_row)
        print('\nSummary Table:')
        print(f"{'Video':<20} {'Actual':<10} {'Trajectory':<12} {'Vertical':<10} {'Combined':<10}")
        print('-' * 68)
        for row in summary_rows:
            print(f"{row[0]:<20} {row[1]:<10} {row[2]:<12} {row[3]:<10} {row[4]:<10}")
        print('')
        # Save to CSV for RAG pipeline
        df_save = pd.DataFrame(save_rows)
        df_save.to_csv('data/pitch_classifier_predictions.csv', index=False)
        print('Predictions and features saved to data/pitch_classifier_predictions.csv')
    else:
        print('No valid test videos to predict.')

if __name__ == '__main__':
    main()
