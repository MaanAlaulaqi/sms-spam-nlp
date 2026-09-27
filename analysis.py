from collections import Counter

import pandas as pd


PRESENCE_FEATURES = [
    "contains_url",
    "contains_money",
    "contains_phone",
    "contains_shortcode",
]


CORRELATION_COLUMNS = [
    "is_spam",
    "message_length",
    "word_count",
    "exclamation_count",
    "question_count",
    "digit_count",
    "uppercase_word_count",
]


def build_source_summary(
    raw_df,
    df
):
    return pd.DataFrame(
        {
            "Property": [
                "Original rows",
                "Original columns",
                "Reconstructed rows",
            ],
            "Value": [
                raw_df.shape[0],
                raw_df.shape[1],
                len(df),
            ],
        }
    )


def build_quality_summary(df):
    conflicting_labels = (
        df.groupby("message")["label"]
        .nunique()
    )

    conflicting_label_count = (
        conflicting_labels > 1
    ).sum()

    corrupted_pound_count = (
        df["message"]
        .str.contains(
            "å£",
            regex=False
        )
        .sum()
    )

    corrupted_apostrophe_count = (
        df["message"]
        .str.contains(
            "\x89\xdb\xf7",
            regex=False
        )
        .sum()
    )

    return pd.DataFrame(
        {
            "Metric": [
                "Missing labels",
                "Missing messages",
                "Empty messages",
                "Duplicate rows",
                "Duplicate messages",
                "Conflicting labels",
                "Corrupted pound sequences",
                "Corrupted apostrophe sequences",
            ],
            "Value": [
                df["label"].isna().sum(),
                df["message"].isna().sum(),
                (
                    df["message"]
                    .str.strip()
                    .eq("")
                    .sum()
                ),
                df.duplicated().sum(),
                (
                    df["message"]
                    .duplicated()
                    .sum()
                ),
                conflicting_label_count,
                corrupted_pound_count,
                corrupted_apostrophe_count,
            ],
        }
    )


def build_class_summary(df):
    summary = (
        df.groupby("label")[
            [
                "message_length",
                "word_count",
                "exclamation_count",
                "question_count",
                "digit_count",
                "uppercase_word_count",
            ]
        ]
        .mean()
        .round(2)
    )

    summary.columns = [
        "Avg Characters / Message",
        "Avg Words / Message",
        "Avg ! / Message",
        "Avg ? / Message",
        "Avg Digits / Message",
        "Avg Uppercase Words / Message",
    ]

    return summary


def build_preprocessing_summary(
    df,
    df_clean
):
    return pd.DataFrame(
        {
            "Representation": [
                "Original reconstructed dataset",
                "After duplicate removal",
                "Clean text duplicates",
                "Stemmed text duplicates",
                "Lemmatized text duplicates",
            ],
            "Count": [
                len(df),
                len(df_clean),
                (
                    df_clean["clean_text"]
                    .duplicated()
                    .sum()
                ),
                (
                    df_clean["stemmed_text"]
                    .duplicated()
                    .sum()
                ),
                (
                    df_clean["lemmatized_text"]
                    .duplicated()
                    .sum()
                ),
            ],
        }
    )


def build_feature_rates(df):
    return (
        df
        .groupby("label")[
            PRESENCE_FEATURES
        ]
        .mean()
        * 100
    )


def build_correlations(df):
    return (
        df[
            CORRELATION_COLUMNS
        ]
        .corr()
    )


def word_message_frequency(
    dataframe,
    label,
    token_column
):
    messages = dataframe[
        dataframe["label"] == label
    ]

    word_counts = Counter()

    for tokens in messages[
        token_column
    ]:
        unique_tokens = set(
            tokens
        )

        for token in unique_tokens:
            word_counts[token] += 1

    total_messages = len(
        messages
    )

    return {
        word: count / total_messages
        for word, count
        in word_counts.items()
    }


def build_word_associations(
    df_clean,
    token_column=(
        "lemmatized_no_stopwords"
    )
):
    ham_frequency = (
        word_message_frequency(
            df_clean,
            "ham",
            token_column
        )
    )

    spam_frequency = (
        word_message_frequency(
            df_clean,
            "spam",
            token_column
        )
    )

    rows = []

    all_words = (
        set(ham_frequency)
        | set(spam_frequency)
    )

    for word in all_words:
        ham_rate = (
            ham_frequency
            .get(
                word,
                0
            )
        )

        spam_rate = (
            spam_frequency
            .get(
                word,
                0
            )
        )

        rows.append(
            {
                "word": word,
                "ham_rate": (
                    ham_rate * 100
                ),
                "spam_rate": (
                    spam_rate * 100
                ),
                "difference": (
                    spam_rate
                    - ham_rate
                ) * 100,
            }
        )

    return pd.DataFrame(
        rows
    )


def build_top_word_associations(
    word_associations,
    count=10
):
    spam_words = (
        word_associations
        .sort_values(
            "difference",
            ascending=False
        )
        .head(count)
    )

    ham_words = (
        word_associations
        .sort_values(
            "difference",
            ascending=True
        )
        .head(count)
    )

    combined = pd.concat(
        [
            spam_words,
            ham_words,
        ]
    )

    combined = (
        combined
        .drop_duplicates(
            subset="word"
        )
        .assign(
            strength=lambda data:
            data["difference"].abs()
        )
        .sort_values(
            "strength",
            ascending=False
        )
    )

    return combined