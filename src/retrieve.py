def get_by_vertical_raw(value):
    """Retrieve all pitches with a given vertical_raw prediction (0=fastball, 1=curveball)."""
    return df[df['vertical_raw'] == value]

def get_by_vertical_curveball_proba(threshold, above=True):
    """Retrieve all pitches where the vertical_curveball_proba is above or below a threshold."""
    if above:
        return df[df['vertical_curveball_proba'] > threshold]
    else:
        return df[df['vertical_curveball_proba'] < threshold]
import pandas as pd

# Load the predictions and features CSV
df = pd.read_csv('data/pitch_classifier_predictions.csv')

# Example retrieval functions

def get_by_label(label):
    """Retrieve all pitches with a given actual label (e.g., 'fastball', 'curveball')."""
    return df[df['actual_label'] == label]

def get_by_pred(pred_type, value):
    """Retrieve all pitches with a given prediction type and value.
    pred_type: 'pred_traj', 'pred_vert', or 'pred_combined'
    value: 'fastball' or 'curveball'
    """
    return df[df[pred_type] == value]

def get_by_vertical_feature(threshold, above=True):
    """Retrieve all pitches where the vertical feature is above or below a threshold."""
    if above:
        return df[df['vertical_feature'] > threshold]
    else:
        return df[df['vertical_feature'] < threshold]

def get_top_n_by_feature(feature, n=5, ascending=True):
    """Retrieve top n pitches by a given feature (e.g., 'x_0', 'curvature_0')."""
    return df.sort_values(by=feature, ascending=ascending).head(n)

if __name__ == "__main__":
    # Example usage
    print("All curveballs:")
    print(get_by_label('curveball'))
    print("\nAll pitches predicted as fastball by combined model:")
    print(get_by_pred('pred_combined', 'fastball'))
    print("\nPitches with vertical feature > 10:")
    print(get_by_vertical_feature(10, above=True))
    print("\nTop 3 pitches with highest x_0:")
    print(get_top_n_by_feature('x_0', n=3, ascending=False))
