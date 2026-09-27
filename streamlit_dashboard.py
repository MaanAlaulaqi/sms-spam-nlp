from io import StringIO

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from analysis import (
    build_class_summary,
    build_correlations,
    build_feature_rates,
    build_preprocessing_summary,
    build_quality_summary,
    build_source_summary,
    build_top_word_associations,
    build_word_associations,
)
from preprocessing import prepare_dataset

# -------------------------------------------------------------------
# Page setup
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Spam Preprocessing Dashboard",
    page_icon="📨",
    layout="wide",
)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #111318;
            color: #f2f4f8;
        }

        html, body, [class*="css"] {
            font-family: Inter, "Segoe UI", Arial, sans-serif;
        }

        code, pre {
            font-family: "JetBrains Mono", "Cascadia Code", "Fira Code",
                         "DejaVu Sans Mono", monospace;
        }

        [data-testid="stMetric"] {
            background: #181b22;
            border: 1px solid #343a46;
            border-radius: 12px;
            padding: 14px 16px;
        }

        [data-testid="stMetricLabel"] {
            color: #b9c0cc;
        }

        [data-testid="stMetricValue"] {
            color: #f2f4f8;
            font-size: 1.8rem;
        }

        [data-testid="stSidebar"] {
            background-color: #181b22;
        }
        [data-testid="stHeader"] {
    height: 0rem;
    min-height: 0rem;
}       
        [data-testid="stSidebarHeader"] {
    height: 0rem;
    min-height: 0rem;
}       

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0rem;
    }

    .block-container {
        padding-top: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -------------------------------------------------------------------
# Cached data + analysis
# -------------------------------------------------------------------


@st.cache_data(show_spinner="Preparing dataset...")
def load_application_data(path):
    raw_df, df, df_clean = prepare_dataset(path)

    source_summary = build_source_summary(raw_df, df)
    quality_summary = build_quality_summary(df)
    class_summary = build_class_summary(df)
    preprocessing_summary = build_preprocessing_summary(df, df_clean)
    feature_rates = build_feature_rates(df)
    correlations = build_correlations(df)

    word_associations = build_word_associations(df_clean)
    top_word_associations = build_top_word_associations(
        word_associations,
        count=10,
    )

    return {
        "raw_df": raw_df,
        "df": df,
        "df_clean": df_clean,
        "source_summary": source_summary,
        "quality_summary": quality_summary,
        "class_summary": class_summary,
        "preprocessing_summary": preprocessing_summary,
        "feature_rates": feature_rates,
        "correlations": correlations,
        "word_associations": word_associations,
        "top_word_associations": top_word_associations,
    }


data = load_application_data("spam.csv")

raw_df = data["raw_df"]
df = data["df"]
df_clean = data["df_clean"]
source_summary = data["source_summary"]
quality_summary = data["quality_summary"]
class_summary = data["class_summary"]
preprocessing_summary = data["preprocessing_summary"]
feature_rates = data["feature_rates"]
correlations = data["correlations"]
top_word_associations = data["top_word_associations"]


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def dataframe_to_csv_bytes(dataframe):
    return dataframe.to_csv(index=False).encode("utf-8")


def create_feature_heatmap():
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(8, 4))

        heatmap = ax.imshow(
            feature_rates.values,
            aspect="auto",
            cmap="viridis",
        )

        ax.set_xticks(range(len(feature_rates.columns)))
        ax.set_xticklabels(
            ["URL", "Money", "Phone-like", "Shortcode-like"],
            rotation=20,
            ha="right",
        )

        ax.set_yticks(range(len(feature_rates.index)))
        ax.set_yticklabels(feature_rates.index.str.upper())

        for row in range(len(feature_rates.index)):
            for column in range(len(feature_rates.columns)):
                value = feature_rates.iloc[row, column]
                rgba = heatmap.cmap(heatmap.norm(value))

                red, green, blue, _ = rgba
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                text_color = "black" if luminance > 0.55 else "white"

                ax.text(
                    column,
                    row,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontweight="bold",
                )

        ax.set_title("Feature Presence in HAM vs SPAM")
        fig.colorbar(
            heatmap,
            ax=ax,
            label="Messages containing feature (%)",
        )

        fig.tight_layout()

    return fig


def create_correlation_heatmap():
    with plt.style.context("dark_background"):
        fig, ax = plt.subplots(figsize=(9, 7))

        heatmap = ax.imshow(
            correlations.values,
            vmin=-1,
            vmax=1,
            aspect="auto",
            cmap="coolwarm",
        )

        ax.set_xticks(range(len(correlations.columns)))
        ax.set_xticklabels(
            correlations.columns,
            rotation=45,
            ha="right",
        )

        ax.set_yticks(range(len(correlations.index)))
        ax.set_yticklabels(correlations.index)

        for row in range(len(correlations.index)):
            for column in range(len(correlations.columns)):
                value = correlations.iloc[row, column]
                rgba = heatmap.cmap(heatmap.norm(value))

                red, green, blue, _ = rgba
                luminance = 0.299 * red + 0.587 * green + 0.114 * blue
                text_color = "black" if luminance > 0.55 else "white"

                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                    fontweight="bold",
                )

        ax.set_title("Correlation Between Message Features")
        fig.colorbar(
            heatmap,
            ax=ax,
            label="Correlation",
        )

        fig.tight_layout()

    return fig


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------

st.sidebar.title("NLP Preprocessing")

st.sidebar.write("Explore each stage of the preprocessing pipeline.")

pipeline_steps = [
    (
        "1. Dataset Reconstruction",
        "Rebuild messages that were split across CSV columns.\n\n"
        '`v2: "WIN a prize"`\n\n'
        '`Unnamed: 2: " call now"`\n\n'
        '→ `"WIN a prize, call now"`',
    ),
    (
        "2. Text Normalization",
        "Standardize selected patterns without throwing away useful information.\n\n"
        '`"Win £900 now!"` → `"Win MONEYTOKEN now!"`',
    ),
    (
        "3. Tokenization",
        "Split a message into smaller units called tokens.\n\n"
        '`"Free entry now!"` → `["Free", "entry", "now", "!"]`',
    ),
    (
        "4. Lowercasing",
        "Convert lexical tokens to lowercase.\n\n"
        '`["FREE", "Free", "free"]` → `["free", "free", "free"]`',
    ),
    (
        "5. Punctuation Filtering",
        "Remove tokens that contain only punctuation.\n\n"
        '`["free", "!", "...", "entry"]` → `["free", "entry"]`',
    ),
    (
        "6. Stopword Removal",
        "Optionally remove very common English words.\n\n"
        '`["win", "a", "free", "prize"]` → `["win", "free", "prize"]`',
    ),
    (
        "7. Stemming",
        "Apply mechanical suffix-removal rules.\n\n"
        '`"available" → "avail"`\n\n'
        '`"entry" → "entri"`',
    ),
    (
        "8. POS Tagging",
        'Estimate grammatical roles before lemmatization.\n\n`"goes" → VBZ (verb)`',
    ),
    (
        "9. Lemmatization",
        "Reduce words to meaningful dictionary forms using POS context.\n\n"
        '`"goes" → "go"`\n\n'
        '`"lives" → "live"`',
    ),
    (
        "10. Feature Analysis",
        "Compare structural signals such as digits, uppercase words, money, "
        "URLs, phone-like values and shortcodes.",
    ),
]

for title, body in pipeline_steps:
    with st.sidebar.expander(title):
        st.markdown(body)


# -------------------------------------------------------------------
# Main navigation
# -------------------------------------------------------------------

st.title("Spam Preprocessing Dashboard")

tabs = st.tabs(
    [
        "Overview",
        "Feature Analysis",
        "Word Analysis",
        "Message Explorer",
        "Downloads",
    ]
)


# -------------------------------------------------------------------
# Overview
# -------------------------------------------------------------------

with tabs[0]:
    st.header("Dataset Overview")

    metric_columns = st.columns(5)

    metric_columns[0].metric("Total Messages", len(df))
    metric_columns[1].metric("HAM", int((df["label"] == "ham").sum()))
    metric_columns[2].metric("SPAM", int((df["label"] == "spam").sum()))
    metric_columns[3].metric("Duplicates", int(df.duplicated().sum()))
    metric_columns[4].metric("Unique Messages", len(df_clean))

    st.subheader("HAM vs SPAM Distribution")

    st.write(
        "The chart shows how many messages belong to each class. "
        "The dataset is imbalanced, with substantially more HAM than SPAM."
    )

    label_counts = (
        df["label"]
        .value_counts()
        .rename_axis("label")
        .reset_index(name="count")
        .set_index("label")
    )

    st.bar_chart(label_counts)

    st.subheader("Dataset Quality")

    st.write(
        "These checks identify missing values, duplicates, conflicting labels "
        "and known encoding problems that could affect preprocessing."
    )

    quality_left, quality_right = st.columns(2)

    with quality_left:
        st.markdown("#### Source Structure")
        st.dataframe(source_summary, use_container_width=True, hide_index=True)

    with quality_right:
        st.markdown("#### Quality Checks")
        st.dataframe(quality_summary, use_container_width=True, hide_index=True)

    st.subheader("HAM vs SPAM Characteristics")

    st.write(
        "This table compares average structural characteristics of messages "
        "in each class. Avg Words / Message is the average number of "
        "whitespace-separated words per message."
    )

    st.dataframe(
        class_summary.reset_index(),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Preprocessing Impact")

    st.write(
        "The table shows how many duplicate representations exist after "
        "different preprocessing stages. More aggressive transformations can "
        "cause originally different messages to collapse into the same text."
    )

    st.dataframe(
        preprocessing_summary,
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Dataset Preview")

    st.write(
        "The original CSV contains message fragments across multiple columns. "
        "They are reconstructed into one message field while preserving the "
        "HAM or SPAM label."
    )

    st.dataframe(
        df[["label", "message"]].head(10),
        use_container_width=True,
        hide_index=True,
    )


# -------------------------------------------------------------------
# Feature analysis
# -------------------------------------------------------------------

with tabs[1]:
    st.header("Message Feature Analysis")

    st.subheader("Feature Presence")

    st.write(
        "This heatmap shows the percentage of HAM and SPAM messages containing "
        "URLs, money values, phone-like sequences and shortcode-like sequences."
    )

    feature_fig = create_feature_heatmap()
    st.pyplot(feature_fig, use_container_width=True)
    plt.close(feature_fig)

    st.subheader("Correlation Matrix")

    st.write(
        "The matrix shows linear relationships between numeric features. "
        "`is_spam` is encoded as 0 for HAM and 1 for SPAM. Values near +1 "
        "indicate a stronger positive association with SPAM; values near 0 "
        "indicate little linear relationship."
    )

    correlation_fig = create_correlation_heatmap()
    st.pyplot(correlation_fig, use_container_width=True)
    plt.close(correlation_fig)


# -------------------------------------------------------------------
# Word analysis
# -------------------------------------------------------------------

with tabs[2]:
    st.header("Word Association Analysis")

    st.write(
        "The chart compares the percentage of messages in each class that "
        "contain each selected word. This avoids comparing raw counts across "
        "classes of very different sizes."
    )

    association_chart = top_word_associations[
        ["word", "ham_rate", "spam_rate"]
    ].set_index("word")

    st.bar_chart(association_chart)


# -------------------------------------------------------------------
# Message explorer
# -------------------------------------------------------------------

with tabs[3]:
    st.header("Message Explorer")

    st.write(
        "Filter the dataset and inspect how an individual message changes "
        "through each preprocessing stage."
    )

    filter_col, search_col = st.columns([1, 3])

    with filter_col:
        selected_class = st.selectbox(
            "Class",
            ["All", "HAM", "SPAM"],
        )

    with search_col:
        search_text = st.text_input(
            "Search Messages",
            placeholder="Search message text...",
        )

    filtered = df_clean

    if selected_class != "All":
        filtered = filtered[filtered["label"] == selected_class.lower()]

    if search_text.strip():
        filtered = filtered[
            filtered["message"].str.contains(
                search_text.strip(),
                case=False,
                regex=False,
                na=False,
            )
        ]

    if filtered.empty:
        st.warning("No messages match the current filters.")

    else:
        max_message = len(filtered)

        message_number = st.number_input(
            "Message",
            min_value=1,
            max_value=max_message,
            value=1,
            step=1,
        )

        row = filtered.iloc[int(message_number) - 1]

        st.markdown(f"### Message {int(message_number)} of {max_message}")

        st.markdown(f"**Label:** `{row['label'].upper()}`")

        with st.expander("Original", expanded=True):
            st.code(row["message"], language=None, wrap_lines=True)

        with st.expander("Normalized", expanded=True):
            st.code(row["normalized_text"], language=None, wrap_lines=True)

        with st.expander("Clean", expanded=True):
            st.code(row["clean_text"], language=None, wrap_lines=True)

        with st.expander("Without Stopwords"):
            st.code(row["no_stopword_text"], language=None, wrap_lines=True)

        with st.expander("Stemmed"):
            st.code(row["stemmed_text"], language=None, wrap_lines=True)

        with st.expander("Lemmatized", expanded=True):
            st.code(row["lemmatized_text"], language=None, wrap_lines=True)

        with st.expander("Lemmatized + No Stopwords"):
            st.code(
                row["lemmatized_no_stopwords_text"],
                language=None,
                wrap_lines=True,
            )


# -------------------------------------------------------------------
# Downloads
# -------------------------------------------------------------------

with tabs[4]:
    st.header("Download Processed Data")

    st.write(
        "Download individual preprocessing stages or a complete audit file "
        "with all text representations side-by-side."
    )

    original_export = df[["label", "message"]].copy()

    normalized_export = df_clean[["label", "normalized_text"]].copy()

    clean_export = df_clean[["label", "clean_text"]].copy()

    no_stopword_export = df_clean[["label", "no_stopword_text"]].copy()

    stemmed_export = df_clean[["label", "stemmed_text"]].copy()

    lemmatized_export = df_clean[["label", "lemmatized_text"]].copy()

    lemmatized_no_stopwords_export = df_clean[
        ["label", "lemmatized_no_stopwords_text"]
    ].copy()

    audit_export = df_clean[
        [
            "label",
            "message",
            "normalized_text",
            "clean_text",
            "no_stopword_text",
            "stemmed_text",
            "lemmatized_text",
            "lemmatized_no_stopwords_text",
        ]
    ].copy()

    st.subheader("Complete Audit File")

    st.download_button(
        "Download Full Comparison",
        dataframe_to_csv_bytes(audit_export),
        file_name="spam_preprocessing_audit.csv",
        mime="text/csv",
        type="primary",
    )

    st.subheader("Individual Stages")

    download_columns = st.columns(2)

    with download_columns[0]:
        st.download_button(
            "Download Original",
            dataframe_to_csv_bytes(original_export),
            file_name="01_original_messages.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download Clean",
            dataframe_to_csv_bytes(clean_export),
            file_name="03_clean_messages.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download Stemmed",
            dataframe_to_csv_bytes(stemmed_export),
            file_name="05_stemmed_messages.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download Lemmatized + No Stopwords",
            dataframe_to_csv_bytes(lemmatized_no_stopwords_export),
            file_name="07_lemmatized_no_stopwords.csv",
            mime="text/csv",
        )

    with download_columns[1]:
        st.download_button(
            "Download Normalized",
            dataframe_to_csv_bytes(normalized_export),
            file_name="02_normalized_messages.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download No Stopwords",
            dataframe_to_csv_bytes(no_stopword_export),
            file_name="04_no_stopwords.csv",
            mime="text/csv",
        )

        st.download_button(
            "Download Lemmatized",
            dataframe_to_csv_bytes(lemmatized_export),
            file_name="06_lemmatized_messages.csv",
            mime="text/csv",
        )
