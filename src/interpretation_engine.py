
import pandas as pd
import numpy as np
import joblib

def summarize(val):
    if pd.isna(val):
        return np.nan
    if isinstance(val, str) and ("," in val or val.startswith("[") or val.endswith("]")):
        try:
            arr = [float(x) for x in val.replace("[","").replace("]","").split(",") if x.strip()]
            arr = [x for x in arr if not pd.isna(x)]
            if arr:
                return float(np.nanmedian(arr))
            else:
                return np.nan
        except Exception:
            return np.nan
    try:
        return float(val)
    except Exception:
        return np.nan

def add_vertical_model_predictions(csv_path, model_path, output_path):
    df = pd.read_csv(csv_path)
    clf_vert = joblib.load(model_path)
    clf_traj = joblib.load("data/pitch_classifier_traj.joblib")
    # Summarize glove_to_person_top for each row
    df["glove_to_person_top_summary"] = df["glove_to_person_top"].apply(summarize)
    # Prepare input for vertical model (reshape to (-1, 1))
    X_vert = df["glove_to_person_top_summary"].values.reshape(-1, 1)
    # Prepare input for trajectory model (first 10 x, y, curvature)
    def get_traj_features(row):
        xs = [row.get(f"x_{i}", np.nan) for i in range(100)]
        ys = [row.get(f"y_{i}", np.nan) for i in range(100)]
        cs = [row.get(f"curvature_{i}", np.nan) for i in range(100)]
        # Append glove_height_peak_leg_lift as the 301st feature
        glove = row.get("glove_height_peak_leg_lift", np.nan)
        arr = np.array(xs + ys + cs + [glove], dtype=float)
        # If all nan, return nan vector
        if np.all(np.isnan(arr)):
            return np.full(301, np.nan)
        # Fill nans with 0 (as in training)
        arr = np.nan_to_num(arr, nan=0.0)
        return arr
    X_traj = np.vstack(df.apply(get_traj_features, axis=1).values)
    # Vertical model predictions
    y_pred_vert = clf_vert.predict(X_vert)
    y_pred_label_vert = np.where(y_pred_vert == 0, "fastball", "curveball")
    decision_vert = clf_vert.decision_function(X_vert)
    proba_vert = clf_vert.predict_proba(X_vert)[:, 1]
    coef_vert = clf_vert.coef_[0][0]
    intercept_vert = clf_vert.intercept_[0]
    if coef_vert != 0:
        boundary_vert = -intercept_vert / coef_vert
    else:
        boundary_vert = np.nan
    # Trajectory model predictions
    y_pred_traj = clf_traj.predict(X_traj)
    y_pred_label_traj = np.where(y_pred_traj == 0, "fastball", "curveball")
    decision_traj = clf_traj.decision_function(X_traj)
    proba_traj = clf_traj.predict_proba(X_traj)[:, 1]
    coef_traj = clf_traj.coef_[0]
    intercept_traj = clf_traj.intercept_[0]
    # For 1D boundary, use first nonzero coef
    nonzero_idx = np.flatnonzero(coef_traj)
    if nonzero_idx.size > 0:
        boundary_traj = -intercept_traj / coef_traj[nonzero_idx[0]]
    else:
        boundary_traj = np.nan
    # Value minus boundary: use first feature (x_0) minus boundary
    value_minus_boundary_traj = X_traj[:, 0] - boundary_traj if not np.isnan(boundary_traj) else np.full(X_traj.shape[0], np.nan)
    # Add columns
    df["vertical_model_pred"] = y_pred_label_vert
    df["vertical_model_decision"] = decision_vert
    df["vertical_model_curveball_proba"] = proba_vert
    df["vertical_model_boundary"] = boundary_vert
    df["vertical_model_value_minus_boundary"] = df["glove_to_person_top_summary"] - boundary_vert
    df["trajectory_model_pred"] = y_pred_label_traj
    df["trajectory_model_decision"] = decision_traj
    df["trajectory_model_curveball_proba"] = proba_traj
    df["trajectory_model_boundary"] = boundary_traj
    df["trajectory_model_value_minus_boundary"] = value_minus_boundary_traj

    # Data-driven bands for NL explanation (from histogram)
    traj_bins = [2, 149, 299, 448, 597, 747, 896]
    traj_labels = [
        "very low trajectory model evidence",
        "low trajectory model evidence",
        "slightly below average trajectory model evidence",
        "slightly above average trajectory model evidence",
        "high trajectory model evidence",
        "very high trajectory model evidence"
    ]
    def explain_traj(val):
        if np.isnan(val):
            return "No trajectory model evidence available"
        for i in range(len(traj_bins)-1):
            if traj_bins[i] <= val < traj_bins[i+1]:
                return traj_labels[i]
        if val >= traj_bins[-1]:
            return traj_labels[-1]
        return traj_labels[0]
    df["trajectory_model_nl_explanation"] = df["trajectory_model_value_minus_boundary"].apply(explain_traj)

    df.to_csv(output_path, index=False)
    print(f"Saved interpreted CSV with vertical and trajectory model predictions, boundary info, and NL explanations to {output_path}")

if __name__ == "__main__":
    add_vertical_model_predictions(
        "data/pitch_classifier_predictions.csv",
        "data/pitch_classifier_vert.joblib",
        "data/pitch_classifier_predictions_interpreted.csv"
    )
