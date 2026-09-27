from io import StringIO

import hvplot.pandas
import matplotlib.pyplot as plt
import panel as pn

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
from preprocessing import (
    prepare_dataset,
)

# -------------------------------------------------------------------
# Setup
# -------------------------------------------------------------------

pn.extension("tabulator", sizing_mode="stretch_width")

APP_CSS = """
:root {
    --body-font: Inter, "Segoe UI", Arial, sans-serif;
    --design-background-color: #111318;
    --design-background-text-color: #f2f4f8;
    --design-surface-color: #181b22;
    --design-surface-text-color: #f2f4f8;
}

body {
    font-family: Inter, "Segoe UI", Arial, sans-serif;
    font-color: #1111318;
}
h1 {
    font-size: 2rem;
    font-color: #1111318;
}

h2 {
    font-size: 1.55rem;
    font-color: #1111318;
}

h3 {
    font-size: 1.2rem;
    font-color: #1111318 !important;
}

p {
    font-size: 1rem;
    line-height: 1.6;
    font-color: #1111318;
}

code,
pre {
    font-size: 0.9rem;
    font-family: "JetBrains Mono", "Cascadia Code", "Fira Code",
                 "DejaVu Sans Mono", monospace;
}
"""


# -------------------------------------------------------------------
# Load + process data
# -------------------------------------------------------------------

@pn.cache(max_items=4, policy="LRU")
def load_application_data(path):
    """
    Run the expensive preprocessing and analysis once per input path.

    The cached result is reused by new Panel sessions while the server
    process remains alive.
    """
    raw_df, df, df_clean = prepare_dataset(path)

    results = {
        "source_summary": build_source_summary(raw_df, df),
        "quality_summary": build_quality_summary(df),
        "class_summary": build_class_summary(df),
        "preprocessing_summary": build_preprocessing_summary(df, df_clean),
        "feature_rates": build_feature_rates(df),
        "correlations": build_correlations(df),
    }

    word_associations = build_word_associations(df_clean)
    results["word_associations"] = word_associations
    results["top_word_associations"] = build_top_word_associations(
        word_associations,
        count=10,
    )

    return raw_df, df, df_clean, results


raw_df, df, df_clean, analysis_results = load_application_data("spam.csv")

# -------------------------------------------------------------------
# Download helpers
# -------------------------------------------------------------------

def dataframe_to_csv(dataframe):
    buffer = StringIO()

    dataframe.to_csv(
        buffer,
        index=False
    )

    buffer.seek(0)

    return buffer


original_export = df[
    [
        "label",
        "message",
    ]
].copy()


normalized_export = df_clean[
    [
        "label",
        "normalized_text",
    ]
].copy()


clean_export = df_clean[
    [
        "label",
        "clean_text",
    ]
].copy()


no_stopword_export = df_clean[
    [
        "label",
        "no_stopword_text",
    ]
].copy()


stemmed_export = df_clean[
    [
        "label",
        "stemmed_text",
    ]
].copy()


lemmatized_export = df_clean[
    [
        "label",
        "lemmatized_text",
    ]
].copy()


lemmatized_no_stopwords_export = df_clean[
    [
        "label",
        "lemmatized_no_stopwords_text",
    ]
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

original_download = pn.widgets.FileDownload(
    callback=lambda: dataframe_to_csv(
        original_export
    ),
    filename="01_original_messages.csv",
    label="Download Original",
)


normalized_download = pn.widgets.FileDownload(
    callback=lambda: dataframe_to_csv(
        normalized_export
    ),
    filename="02_normalized_messages.csv",
    label="Download Normalized",
)


clean_download = pn.widgets.FileDownload(
    callback=lambda: dataframe_to_csv(
        clean_export
    ),
    filename="03_clean_messages.csv",
    label="Download Clean",
)


no_stopword_download = pn.widgets.FileDownload(
    callback=lambda: dataframe_to_csv(
        no_stopword_export
    ),
    filename="04_no_stopwords.csv",
    label="Download No Stopwords",
)


stemmed_download = pn.widgets.FileDownload(
    callback=lambda: dataframe_to_csv(
        stemmed_export
    ),
    filename="05_stemmed_messages.csv",
    label="Download Stemmed",
)


lemmatized_download = pn.widgets.FileDownload(
    callback=lambda: dataframe_to_csv(
        lemmatized_export
    ),
    filename="06_lemmatized_messages.csv",
    label="Download Lemmatized",
)


lemma_no_stop_download = pn.widgets.FileDownload(
    callback=lambda: dataframe_to_csv(
        lemmatized_no_stopwords_export
    ),
    filename="07_lemmatized_no_stopwords.csv",
    label="Download Lemmatized + No Stopwords",
)


audit_download = pn.widgets.FileDownload(
    callback=lambda: dataframe_to_csv(
        audit_export
    ),
    filename="spam_preprocessing_audit.csv",
    label="Download Full Comparison",
    button_type="primary",
)

# -------------------------------------------------------------------
# Analysis
# -------------------------------------------------------------------

source_summary = analysis_results["source_summary"]
quality_summary = analysis_results["quality_summary"]
class_summary = analysis_results["class_summary"]
preprocessing_summary = analysis_results["preprocessing_summary"]
feature_rates = analysis_results["feature_rates"]
correlations = analysis_results["correlations"]
word_associations = analysis_results["word_associations"]
top_word_associations = analysis_results["top_word_associations"]


# -------------------------------------------------------------------
# Dashboard statistics
# -------------------------------------------------------------------

total_messages = len(df)

ham_count = (df["label"] == "ham").sum()

spam_count = (df["label"] == "spam").sum()

duplicate_count = df.duplicated().sum()

unique_messages = len(df_clean)


# -------------------------------------------------------------------
# Class distribution
# -------------------------------------------------------------------

label_counts = df["label"].value_counts().rename_axis("label").reset_index(name="count")


class_chart = label_counts.hvplot.bar(
    x="label",
    y="count",
    title=("HAM vs SPAM Distribution"),
    ylabel=("Number of Messages"),
    xlabel="Label",
    height=350,
    responsive=True,
    color="#7E57C2",
)


# -------------------------------------------------------------------
# Heatmap utilities
# -------------------------------------------------------------------


def readable_text_color(rgba):
    red, green, blue, _ = rgba

    luminance = 0.299 * red + 0.587 * green + 0.114 * blue

    if luminance > 0.55:
        return "black"

    return "white"


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
            [
                "URL",
                "Money",
                "Phone-like",
                "Shortcode-like",
            ],
            rotation=20,
            ha="right",
        )

        ax.set_yticks(range(len(feature_rates.index)))
        ax.set_yticklabels(feature_rates.index.str.upper())

        for row in range(len(feature_rates.index)):
            for column in range(len(feature_rates.columns)):
                value = feature_rates.iloc[row, column]
                rgba = heatmap.cmap(heatmap.norm(value))

                ax.text(
                    column,
                    row,
                    f"{value:.1f}%",
                    ha="center",
                    va="center",
                    color=readable_text_color(rgba),
                    fontweight="bold",
                )

        ax.set_title("Feature Presence in HAM vs SPAM")

        colorbar = fig.colorbar(
            heatmap,
            ax=ax,
            label="Messages containing feature (%)",
        )
        colorbar.ax.tick_params(colors="white")
        colorbar.set_label("Messages containing feature (%)", color="white")

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

                ax.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=readable_text_color(rgba),
                    fontweight="bold",
                )

        ax.set_title("Correlation Between Message Features")

        colorbar = fig.colorbar(
            heatmap,
            ax=ax,
            label="Correlation",
        )
        colorbar.ax.tick_params(colors="white")
        colorbar.set_label("Correlation", color="white")

        fig.tight_layout()

    return fig


# -------------------------------------------------------------------
# Word association chart
# -------------------------------------------------------------------

word_association_chart = top_word_associations.hvplot.bar(
    x="word",
    y=[
        "ham_rate",
        "spam_rate",
    ],
    title=("HAM vs SPAM Word Association"),
    ylabel=("Messages containing word (%)"),
    xlabel="Word",
    height=480,
    responsive=True,
    rot=55,
    color=[
        "#42A5F5",
        "#EF5350",
    ],
    legend="top_right",
)


# -------------------------------------------------------------------
# Indicator cards
# -------------------------------------------------------------------

total_card = pn.indicators.Number(
    name="Total Messages",
    value=total_messages,
    format="{value:,.0f}",
)

ham_card = pn.indicators.Number(
    name="HAM",
    value=ham_count,
    format="{value:,.0f}",
)

spam_card = pn.indicators.Number(
    name="SPAM",
    value=spam_count,
    format="{value:,.0f}",
)

duplicate_card = pn.indicators.Number(
    name="Duplicates",
    value=duplicate_count,
    format="{value:,.0f}",
)

unique_card = pn.indicators.Number(
    name="Unique Messages",
    value=unique_messages,
    format="{value:,.0f}",
)


# -------------------------------------------------------------------
# Table helper
# -------------------------------------------------------------------

def data_table(dataframe, *, show_index=False, height=220):
    """
    Render tables with Panel's Fast-aware Tabulator theme so the table
    follows the fixed dark application theme.
    """
    return pn.widgets.Tabulator(
        dataframe,
        show_index=show_index,
        disabled=True,
        theme="fast",
        layout="fit_data_stretch",
        height=height,
        sizing_mode="stretch_width",
    )


# -------------------------------------------------------------------
# Overview
# -------------------------------------------------------------------

overview = pn.Column(
    "## Dataset Overview",
    pn.Row(
        total_card,
        ham_card,
        spam_card,
        duplicate_card,
        unique_card,
    ),
    pn.Spacer(height=15),
    pn.pane.Markdown(
        """
The chart below shows the distribution of HAM and SPAM messages.

The dataset is imbalanced, with substantially more HAM than SPAM.
This will matter later when evaluating a classifier because accuracy
alone may give a misleading impression of model performance.
"""
    ),
    class_chart,
    pn.Spacer(height=20),
    "### Dataset Quality",
    pn.pane.Markdown(
        """
These checks identify issues that may affect preprocessing or model
training.

Missing or empty records may be unusable, while duplicate messages
can cause repeated examples to influence a model disproportionately.

Conflicting labels would be particularly problematic because the
same message would be classified as both HAM and SPAM.

The encoding checks show two known corruption patterns discovered
during inspection of the original CSV.
"""
    ),
    pn.Row(
        pn.Card(
            data_table(
                source_summary,
                height=250,
            ),
            title="Source Structure",
            collapsed=False,
            sizing_mode=("stretch_width"),
        ),
        pn.Card(
            data_table(
                quality_summary,
                height=330,
            ),
            title="Quality Checks",
            collapsed=False,
            sizing_mode=("stretch_width"),
        ),
        sizing_mode=("stretch_width"),
    ),
    pn.Spacer(height=20),
    "### HAM vs SPAM Characteristics",
    pn.pane.Markdown(
        """
This table compares average structural characteristics of messages
in each class.

- **Avg Characters / Message** — average message length in characters.
- **Avg Words / Message** — average number of whitespace-separated
  words per message.
- **Avg ! / Message** — average number of exclamation marks.
- **Avg ? / Message** — average number of question marks.
- **Avg Digits / Message** — average number of numeric characters.
- **Avg Uppercase Words / Message** — average number of fully-uppercase
  words.

These measurements help identify structural differences between
HAM and SPAM before the message text is used by a machine-learning
model.
"""
    ),
    data_table(
        class_summary.reset_index(),
        height=180,
    ),
    pn.Spacer(height=20),
    "### Preprocessing Impact",
    pn.pane.Markdown(
        """
This table shows how preprocessing changes the dataset.

The original duplicate messages are removed before text processing.
Additional duplicates may appear afterward because different
original messages can become identical once punctuation, word forms,
or other differences are normalized.

A larger increase in duplicate representations can therefore indicate
a more aggressive preprocessing method.
"""
    ),
    data_table(
        preprocessing_summary,
        height=260,
    ),
    pn.Spacer(height=20),
    "### Dataset Preview",
    pn.pane.Markdown(
        """
The original CSV stores some message fragments across several columns.
Those fragments are reconstructed into one `message` field while
preserving the original HAM or SPAM label.

The table below shows a small preview of the reconstructed dataset.
"""
    ),
    data_table(
        df[
            [
                "label",
                "message",
            ]
        ].head(10),
        height=340,
    ),
    sizing_mode="stretch_width",
)


# -------------------------------------------------------------------
# Feature analysis
# -------------------------------------------------------------------

feature_analysis = pn.Column(
    "## Message Feature Analysis",
    "### Feature Presence",
    pn.pane.Markdown(
        """
This heatmap shows the percentage of HAM and SPAM messages containing
several structural features.

Rather than treating numbers, links and monetary values as meaningless
noise, this analysis checks whether their presence carries information
about the class.

**Phone-like** values are sequences of seven or more digits.

**Shortcode-like** values are sequences of five or six digits.

These are heuristic patterns rather than verified phone numbers.
"""
    ),
    pn.pane.Matplotlib(
        create_feature_heatmap(),
        tight=True,
        sizing_mode=("stretch_width"),
    ),
    "### Correlation Matrix",
    pn.pane.Markdown(
        """
This matrix measures linear relationships between numeric message
features.

`is_spam` is encoded as:

- **0 = HAM**
- **1 = SPAM**

Values closer to **+1** indicate that higher values of a feature tend
to be associated with SPAM.

Values closer to **-1** indicate an association with HAM.

Values near **0** indicate little linear relationship.

Correlation describes association within this dataset; it does not
mean that a feature causes a message to be SPAM.
"""
    ),
    pn.pane.Matplotlib(
        create_correlation_heatmap(),
        tight=True,
        sizing_mode=("stretch_width"),
    ),
    sizing_mode="stretch_width",
)


# -------------------------------------------------------------------
# Word analysis
# -------------------------------------------------------------------

word_analysis = pn.Column(
    "## Word Association Analysis",
    pn.pane.Markdown(
        """
This chart compares how often selected words appear in HAM and SPAM
messages.

The values represent the **percentage of messages in each class
containing the word**, rather than the total number of times that word
occurs.

This makes the comparison fairer because the dataset contains far more
HAM messages than SPAM messages.

Words with a large difference between the two bars may provide useful
signals for classification.

**HAM:** blue

**SPAM:** red
"""
    ),
    word_association_chart,
    sizing_mode="stretch_width",
)


# -------------------------------------------------------------------
# Message explorer widgets
# -------------------------------------------------------------------

class_filter = pn.widgets.Select(
    name="Class",
    options=[
        "All",
        "HAM",
        "SPAM",
    ],
    value="All",
)

search_input = pn.widgets.TextInput(
    name="Search Messages",
    placeholder=("Search message text..."),
)

message_number = pn.widgets.IntInput(
    name="Message",
    value=1,
    start=1,
    end=len(df_clean),
)

previous_button = pn.widgets.Button(
    name="← Previous",
    button_type="default",
)

next_button = pn.widgets.Button(
    name="Next →",
    button_type="primary",
)


# -------------------------------------------------------------------
# Message explorer logic
# -------------------------------------------------------------------


def filter_messages(selected_class, search_text):
    filtered = df_clean

    if selected_class != "All":
        filtered = filtered[filtered["label"] == selected_class.lower()]

    search_text = search_text.strip()

    if search_text:
        filtered = filtered[
            filtered["message"].str.contains(
                search_text,
                case=False,
                regex=False,
                na=False,
            )
        ]

    return filtered


def update_navigation(event=None):
    filtered = filter_messages(
        class_filter.value,
        search_input.value,
    )

    total = len(filtered)

    message_number.end = max(total, 1)

    message_number.value = 1


def previous_message(event):
    if message_number.value > 1:
        message_number.value -= 1


def next_message(event):
    if message_number.value < message_number.end:
        message_number.value += 1


class_filter.param.watch(update_navigation, "value")

search_input.param.watch(update_navigation, "value")

previous_button.on_click(previous_message)

next_button.on_click(next_message)


# -------------------------------------------------------------------
# Message explorer view
# -------------------------------------------------------------------


def message_view(position, selected_class, search_text):
    filtered = filter_messages(
        selected_class,
        search_text,
    )

    total = len(filtered)

    if total == 0:
        return pn.pane.Alert(
            ("No messages match the current filters."),
            alert_type="warning",
        )

    position = min(
        max(position, 1),
        total,
    )

    row = filtered.iloc[position - 1]

    return pn.Column(
        pn.pane.Markdown(
            f"""
### Message {position} of {total}

**Label:** `{row["label"].upper()}`
"""
        ),
        pn.Card(
            pn.pane.Str(
                row["message"],
                styles={
                    "color": "#f2f4f8",
                    "font-size": "0.95rem",
                    "line-height": "1.5",
                },
            ),
            title="Original",
            collapsed=False,
        ),
        pn.Card(
            pn.pane.Str(
                row["normalized_text"],
                styles={
                    "color": "#f2f4f8",
                    "font-size": "0.95rem",
                    "line-height": "1.5",
                },
            ),
            title="Normalized",
            collapsed=False,
        ),
        pn.Card(
            pn.pane.Str(
                row["clean_text"],
                styles={
                    "color": "#f2f4f8",
                    "font-size": "0.95rem",
                    "line-height": "1.5",
                },
            ),
            title="Clean",
            collapsed=False,
        ),
        pn.Card(
            pn.pane.Str(
                row["no_stopword_text"],
                styles={
                    "color": "#f2f4f8",
                    "font-size": "0.95rem",
                    "line-height": "1.5",
                },
            ),
            title="Without Stopwords",
            collapsed=True,
        ),
        pn.Card(
            pn.pane.Str(
                row["stemmed_text"],
                styles={
                    "color": "#f2f4f8",
                    "font-size": "0.95rem",
                    "line-height": "1.5",
                },
            ),
            title="Stemmed",
            collapsed=True,
        ),
        pn.Card(
            pn.pane.Str(
                row["lemmatized_text"],
                styles={
                    "color": "#f2f4f8",
                    "font-size": "0.95rem",
                    "line-height": "1.5",
                },
            ),
            title="Lemmatized",
            collapsed=False,
        ),
        pn.Card(
            pn.pane.Str(
                row["lemmatized_no_stopwords_text"],
                styles={
                    "color": "#f2f4f8",
                    "font-size": "0.95rem",
                    "line-height": "1.5",
                },
            ),
            title=("Lemmatized + No Stopwords"),
            collapsed=True,
        ),
        sizing_mode="stretch_width",
    )


message_explorer = pn.Column(
    "## Message Explorer",
    pn.pane.Markdown(
        """
Use the controls below to inspect individual messages and compare
how each preprocessing stage transforms the original text.

Filtering by class or searching for text updates the available message
set. The Previous and Next buttons move only through the currently
filtered results.
"""
    ),
    pn.Row(
        class_filter,
        search_input,
        sizing_mode=("stretch_width"),
    ),
    pn.Row(
        previous_button,
        message_number,
        next_button,
    ),
    pn.bind(
        message_view,
        position=message_number,
        selected_class=class_filter,
        search_text=search_input,
    ),
    sizing_mode="stretch_width",
)

downloads = pn.Column(

    "## Download Processed Data",

    pn.pane.Markdown(
        """
Each preprocessing stage can be downloaded as a CSV for independent
inspection.

The **Full Comparison** file places the original message and every
processed representation side-by-side. This makes it easier to verify
how individual messages changed throughout the pipeline.

The original export contains all reconstructed records. Processed
exports use the duplicate-removed dataset used by the preprocessing
pipeline.
"""
    ),

    "### Complete Audit File",

    audit_download,

    pn.Spacer(height=20),

    "### Individual Stages",

    pn.Row(
        original_download,
        normalized_download,
    ),

    pn.Row(
        clean_download,
        no_stopword_download,
    ),

    pn.Row(
        stemmed_download,
        lemmatized_download,
    ),

    lemma_no_stop_download,

    sizing_mode="stretch_width",
)

# -------------------------------------------------------------------
# Tabs
# -------------------------------------------------------------------

tabs = pn.Tabs(

    (
        "Overview",
        overview,
    ),

    (
        "Feature Analysis",
        feature_analysis,
    ),

    (
        "Word Analysis",
        word_analysis,
    ),

    (
        "Message Explorer",
        message_explorer,
    ),

    (
        "Downloads",
        downloads,
    ),

    dynamic=True,
)
pipeline_accordion = pn.Accordion(
    (
        "1. Dataset Reconstruction",
        pn.pane.Markdown(
            """
Rebuild messages that were split across several CSV columns.

```text
v2:          "WIN a prize"
Unnamed: 2:  " call now"

→ "WIN a prize, call now"
```
"""
        ),
    ),
    (
        "2. Text Normalization",
        pn.pane.Markdown(
            """
Standardize selected patterns without throwing away useful information.

```text
"Win £900 now!"

→

"Win MONEYTOKEN now!"
```

URLs and email addresses are handled similarly.
"""
        ),
    ),
    (
        "3. Tokenization",
        pn.pane.Markdown(
            """
Split a message into smaller units called **tokens**.

```text
"Free entry now!"

→

["Free", "entry", "now", "!"]
```
"""
        ),
    ),
    (
        "4. Lowercasing",
        pn.pane.Markdown(
            """
Convert lexical tokens to lowercase so equivalent words share one form.

```text
["FREE", "Free", "free"]

→

["free", "free", "free"]
```

Capitalization information is preserved separately as a feature.
"""
        ),
    ),
    (
        "5. Punctuation Filtering",
        pn.pane.Markdown(
            """
Remove tokens that contain only punctuation.

```text
["free", "!", "...", "entry"]

→

["free", "entry"]
```

Useful punctuation counts are measured before this step.
"""
        ),
    ),
    (
        "6. Stopword Removal",
        pn.pane.Markdown(
            """
Optionally remove very common English words.

```text
["win", "a", "free", "prize"]

→

["win", "free", "prize"]
```

Both versions are retained for comparison.
"""
        ),
    ),
    (
        "7. Stemming",
        pn.pane.Markdown(
            """
Reduce words using mechanical suffix-removal rules.

```text
"available" → "avail"
"entry"     → "entri"
"crazy"     → "crazi"
```

The result does not need to be a valid English word.
"""
        ),
    ),
    (
        "8. POS Tagging",
        pn.pane.Markdown(
            """
Estimate the grammatical role of each word.

```text
"goes"  → VBZ (verb)
"lives" → VBZ (verb)
```

The tags give lemmatization more grammatical context.
"""
        ),
    ),
    (
        "9. Lemmatization",
        pn.pane.Markdown(
            """
Reduce words to meaningful dictionary forms using their POS tags.

```text
"goes"  → "go"
"lives" → "live"
"got"   → "get"
```
"""
        ),
    ),
    (
        "10. Feature Analysis",
        pn.pane.Markdown(
            """
Measure patterns that may help distinguish HAM from SPAM.

```text
digits
uppercase words
money
URLs
phone-like numbers
shortcodes
```

These features are compared across both classes.
"""
        ),
    ),
    active=[],
    toggle=True,
    sizing_mode="stretch_width",
)


# -------------------------------------------------------------------
# Template
# -------------------------------------------------------------------
template = pn.template.FastListTemplate(
    title="Spam Preprocessing Dashboard",
    theme="dark",
    theme_toggle=False,
    raw_css=[APP_CSS],
    background_color="#111318",
    neutral_color="#181b22",
    header_background="#181b22",
    header_color="#f2f4f8",
    sidebar=[
        pn.pane.Markdown(
            """
## NLP Preprocessing

Explore each stage of the text preprocessing pipeline.

Select a step below for a short explanation and example.
"""
        ),
        pipeline_accordion,
    ],
)

template.main.append(tabs)


template.servable()