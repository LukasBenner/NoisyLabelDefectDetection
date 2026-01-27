"""
Streamlit app for labeling image pair similarities.
Run with: streamlit run label_clusters_app.py
"""

import streamlit as st
import pandas as pd
import os
from PIL import Image
from pathlib import Path

# Configuration
ARTIFACTS_DIR = Path("artifacts/cluster_samples")
SAMPLED_PAIRS_CSV = ARTIFACTS_DIR / "sampled_pairs.csv"
LABELS_CSV = ARTIFACTS_DIR / "labels.csv"

# Label options
LABEL_OPTIONS = {
    "EXACT": "Exact duplicates (same image)",
    "SAME_INSTANCE": "Confidently the same instance of an instrument",
    "DIFF_INSTANCE": "Clearly different instances of the instrument",
    "AMBIGUOUS": "Not sure if it's the same instance or not"
}

def load_data():
    """Load sampled pairs and existing labels"""
    if not SAMPLED_PAIRS_CSV.exists():
        st.error(f"Sampled pairs CSV not found at {SAMPLED_PAIRS_CSV}")
        st.stop()
    
    pairs_df = pd.read_csv(SAMPLED_PAIRS_CSV)
    
    # Load existing labels if available
    if LABELS_CSV.exists():
        labels_df = pd.read_csv(LABELS_CSV)
    else:
        labels_df = pd.DataFrame(columns=["pair_id", "image_id_a", "image_id_b", "similarity", "label"])
    
    return pairs_df, labels_df

def save_label(pair_id, image_id_a, image_id_b, similarity, label):
    """Save a label to CSV"""
    if LABELS_CSV.exists():
        labels_df = pd.read_csv(LABELS_CSV)
    else:
        labels_df = pd.DataFrame(columns=["pair_id", "image_id_a", "image_id_b", "similarity", "label"])
    
    # Remove existing label for this pair if any
    labels_df = labels_df[labels_df["pair_id"] != pair_id]
    
    # Add new label
    new_row = pd.DataFrame([{
        "pair_id": pair_id,
        "image_id_a": image_id_a,
        "image_id_b": image_id_b,
        "similarity": similarity,
        "label": label
    }])
    labels_df = pd.concat([labels_df, new_row], ignore_index=True)
    
    # Save
    labels_df.to_csv(LABELS_CSV, index=False)

def get_image_path(pair_id, category, img_num):
    """Get the path to an image file"""
    # Extract the numeric index from pair_id (e.g., "high_pair_001" -> 1)
    pair_idx = int(pair_id.split("_")[-1])
    
    # Find the image file in the category folder
    folder = ARTIFACTS_DIR / category
    pattern = f"pair_{pair_idx:03d}_img{img_num}.*"
    
    for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"]:
        img_path = folder / f"pair_{pair_idx:03d}_img{img_num}{ext}"
        if img_path.exists():
            return img_path
    
    return None

def main():
    st.set_page_config(page_title="Image Pair Labeling", layout="wide")
    
    st.title("🏷️ Image Pair Similarity Labeling")
    
    # Load data
    pairs_df, labels_df = load_data()
    
    # Sidebar - Progress and Navigation
    st.sidebar.header("📊 Progress")
    labeled_pairs = set(labels_df["pair_id"].values)
    total_pairs = len(pairs_df)
    num_labeled = len(labeled_pairs)
    
    st.sidebar.metric("Labeled", f"{num_labeled}/{total_pairs}")
    st.sidebar.progress(num_labeled / total_pairs if total_pairs > 0 else 0)
    
    # Show distribution of labels
    if not labels_df.empty:
        st.sidebar.subheader("Label Distribution")
        label_counts = labels_df["label"].value_counts()
        for label in LABEL_OPTIONS.keys():
            count = label_counts.get(label, 0)
            st.sidebar.text(f"{label}: {count}")
    
    # Filter options
    st.sidebar.header("🔍 Filter")
    filter_category = st.sidebar.selectbox(
        "Category",
        ["All"] + list(pairs_df["category"].unique())
    )
    
    show_labeled = st.sidebar.checkbox("Show already labeled", value=True)
    
    # Filter pairs
    filtered_pairs = pairs_df.copy()
    if filter_category != "All":
        filtered_pairs = filtered_pairs[filtered_pairs["category"] == filter_category]
    
    if not show_labeled:
        filtered_pairs = filtered_pairs[~filtered_pairs["pair_id"].isin(labeled_pairs)]
    
    # Pair selection
    if len(filtered_pairs) == 0:
        st.warning("No pairs to label with current filters!")
        st.stop()
    
    # Initialize session state for navigation
    if "pair_idx" not in st.session_state:
        st.session_state.pair_idx = 0
    
    # Ensure pair_idx is within bounds
    if st.session_state.pair_idx >= len(filtered_pairs):
        st.session_state.pair_idx = len(filtered_pairs) - 1
    if st.session_state.pair_idx < 0:
        st.session_state.pair_idx = 0
    
    pair_idx = st.session_state.pair_idx
    
    # Get current pair
    current_pair = filtered_pairs.iloc[pair_idx]
    pair_id = current_pair["pair_id"]
    category = current_pair["category"]
    image_id_a = current_pair["image_id_a"]
    image_id_b = current_pair["image_id_b"]
    similarity = current_pair["similarity"]
    
    # Display current pair info - compact
    existing_label = labels_df[labels_df["pair_id"] == pair_id]["label"].values
    existing_label = existing_label[0] if len(existing_label) > 0 else "❓"
    
    st.subheader(f"Pair {pair_idx + 1}/{len(filtered_pairs)} | {category.upper()} | Sim: {similarity:.3f} | Label: {existing_label}")
    
    # Display images side by side - smaller
    img1_path = get_image_path(pair_id, category, 1)
    img2_path = get_image_path(pair_id, category, 2)
    
    col_img1, col_img2 = st.columns(2)
    
    with col_img1:
        if img1_path and img1_path.exists():
            img1 = Image.open(img1_path)
            st.image(img1, width=400)
        else:
            st.error(f"Image not found: {img1_path}")
    
    with col_img2:
        if img2_path and img2_path.exists():
            img2 = Image.open(img2_path)
            st.image(img2, width=400)
        else:
            st.error(f"Image not found: {img2_path}")
    
    # Labeling section - compact
    st.markdown("**Label:**")
    
    # Display label options with descriptions - more compact
    cols = st.columns(4)
    for idx, (label, description) in enumerate(LABEL_OPTIONS.items()):
        with cols[idx]:
            button_type = "primary" if existing_label == label else "secondary"
            if st.button(label, key=f"btn_{label}", use_container_width=True, type=button_type):
                save_label(pair_id, image_id_a, image_id_b, similarity, label)
                st.success(f"✓ {label}")
                st.rerun()
    
    # Navigation - compact
    col_nav1, col_nav2, col_nav3, col_nav4 = st.columns([1, 1, 1, 1])
    
    with col_nav1:
        if st.button("⬅️ Previous", use_container_width=True, disabled=(pair_idx == 0)):
            st.session_state.pair_idx = max(0, pair_idx - 1)
            st.rerun()
    
    with col_nav2:
        if st.button("Next ➡️", use_container_width=True, disabled=(pair_idx >= len(filtered_pairs) - 1)):
            st.session_state.pair_idx = min(len(filtered_pairs) - 1, pair_idx + 1)
            st.rerun()
    
    with col_nav3:
        if st.button("⏭️ Skip to Unlabeled", use_container_width=True):
            # Find next unlabeled pair
            for i in range(pair_idx + 1, len(filtered_pairs)):
                next_pair_id = filtered_pairs.iloc[i]["pair_id"]
                if next_pair_id not in labeled_pairs:
                    st.session_state.pair_idx = i
                    st.rerun()
                    break
    
    with col_nav4:
        if st.button("🔄 Reset Index", use_container_width=True):
            st.session_state.pair_idx = 0
            st.rerun()
    
    # Export section
    st.sidebar.markdown("---")
    st.sidebar.header("💾 Export")
    if st.sidebar.button("Download Labels CSV"):
        if LABELS_CSV.exists():
            with open(LABELS_CSV, "r") as f:
                st.sidebar.download_button(
                    label="Download labels.csv",
                    data=f.read(),
                    file_name="labels.csv",
                    mime="text/csv"
                )
        else:
            st.sidebar.warning("No labels to download yet!")

if __name__ == "__main__":
    main()
