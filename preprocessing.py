import re

import nltk
import pandas as pd

from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize


MESSAGE_COLUMNS = [
    "v2",
    "Unnamed: 2",
    "Unnamed: 3",
    "Unnamed: 4",
]


def setup_nltk():
    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4",
        "averaged_perceptron_tagger_eng",
    ]

    for resource in resources:
        nltk.download(
            resource,
            quiet=True
        )


setup_nltk()


STOP_WORDS = set(
    stopwords.words("english")
)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def rebuild_message(row):
    parts = []

    for column in MESSAGE_COLUMNS:
        value = row[column]

        if pd.notna(value):
            parts.append(
                str(value)
            )

    return ",".join(parts)


def uppercase_word_count(text):
    words = text.split()

    return sum(
        1
        for word in words
        if word.isupper()
        and len(word) > 1
    )


def contains_url(text):
    return bool(
        re.search(
            r"https?://\S+|www\.\S+",
            text,
            flags=re.IGNORECASE
        )
    )


def contains_money(text):
    return bool(
        re.search(
            r"(?:å£|£|\$|€)\s?\d+(?:[.,]\d+)?",
            text
        )
    )


def contains_phone(text):
    return bool(
        re.search(
            r"(?<!\d)\d{7,}(?!\d)",
            text
        )
    )


def contains_shortcode(text):
    return bool(
        re.search(
            r"(?<!\d)\d{5,6}(?!\d)",
            text
        )
    )


def normalize_text(text):
    # Repair known corrupted apostrophe sequence
    text = re.sub(
        r"(?<=\w)\x89\xdb\xf7(?=\w)",
        "'",
        text
    )

    # Normalize URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " URLTOKEN ",
        text,
        flags=re.IGNORECASE
    )

    # Normalize emails
    text = re.sub(
        r"\b[\w.+-]+@[\w.-]+\.\w+\b",
        " EMAILTOKEN ",
        text
    )

    # Normalize monetary amounts
    text = re.sub(
        r"(?:å£|£|\$|€)\s?\d+(?:[.,]\d+)?",
        " MONEYTOKEN ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def clean_tokens(tokens):
    lowercase_tokens = [
        token.lower()
        for token in tokens
    ]

    return [
        token
        for token in lowercase_tokens
        if re.search(
            r"[a-z0-9]",
            token
        )
    ]


def remove_stopwords(tokens):
    return [
        token
        for token in tokens
        if token not in STOP_WORDS
    ]


def get_wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ

    if tag.startswith("V"):
        return wordnet.VERB

    if tag.startswith("N"):
        return wordnet.NOUN

    if tag.startswith("R"):
        return wordnet.ADV

    return wordnet.NOUN


def lemmatize_with_pos(tokens):
    tagged_tokens = pos_tag(
        tokens
    )

    return [
        lemmatizer.lemmatize(
            word,
            get_wordnet_pos(tag)
        )
        for word, tag
        in tagged_tokens
    ]


def load_dataset(path):
    raw_df = pd.read_csv(
        path,
        encoding="latin-1"
    )

    df = raw_df.copy()

    df["message"] = df.apply(
        rebuild_message,
        axis=1
    )

    df = df[
        [
            "v1",
            "message",
        ]
    ]

    df.columns = [
        "label",
        "message",
    ]

    return raw_df, df


def add_structural_features(df):
    df = df.copy()

    df["message_length"] = (
        df["message"]
        .str.len()
    )

    df["word_count"] = (
        df["message"]
        .str.split()
        .str.len()
    )

    df["exclamation_count"] = (
        df["message"]
        .str.count("!")
    )

    df["question_count"] = (
        df["message"]
        .str.count(r"\?")
    )

    df["digit_count"] = (
        df["message"]
        .apply(
            lambda text: sum(
                char.isdigit()
                for char in text
            )
        )
    )

    df["uppercase_word_count"] = (
        df["message"]
        .apply(
            uppercase_word_count
        )
    )

    df["contains_url"] = (
        df["message"]
        .apply(
            contains_url
        )
    )

    df["contains_money"] = (
        df["message"]
        .apply(
            contains_money
        )
    )

    df["contains_phone"] = (
        df["message"]
        .apply(
            contains_phone
        )
    )

    df["contains_shortcode"] = (
        df["message"]
        .apply(
            contains_shortcode
        )
    )

    df["is_spam"] = (
        df["label"] == "spam"
    ).astype(int)

    return df


def preprocess_text(df):
    df_clean = (
        df
        .drop_duplicates(
            subset=[
                "label",
                "message",
            ]
        )
        .copy()
    )

    df_clean["normalized_text"] = (
        df_clean["message"]
        .apply(
            normalize_text
        )
    )

    df_clean["tokens"] = (
        df_clean["normalized_text"]
        .apply(
            word_tokenize
        )
    )

    df_clean["clean_tokens"] = (
        df_clean["tokens"]
        .apply(
            clean_tokens
        )
    )

    df_clean["no_stopword_tokens"] = (
        df_clean["clean_tokens"]
        .apply(
            remove_stopwords
        )
    )

    df_clean["stemmed_tokens"] = (
        df_clean["no_stopword_tokens"]
        .apply(
            lambda tokens: [
                stemmer.stem(token)
                for token in tokens
            ]
        )
    )

    df_clean["lemmatized_tokens"] = (
        df_clean["clean_tokens"]
        .apply(
            lemmatize_with_pos
        )
    )

    df_clean[
        "lemmatized_no_stopwords"
    ] = (
        df_clean["lemmatized_tokens"]
        .apply(
            remove_stopwords
        )
    )

    df_clean["clean_text"] = (
        df_clean["clean_tokens"]
        .apply(
            " ".join
        )
    )

    df_clean["no_stopword_text"] = (
        df_clean["no_stopword_tokens"]
        .apply(
            " ".join
        )
    )

    df_clean["stemmed_text"] = (
        df_clean["stemmed_tokens"]
        .apply(
            " ".join
        )
    )

    df_clean["lemmatized_text"] = (
        df_clean["lemmatized_tokens"]
        .apply(
            " ".join
        )
    )

    df_clean[
        "lemmatized_no_stopwords_text"
    ] = (
        df_clean[
            "lemmatized_no_stopwords"
        ]
        .apply(
            " ".join
        )
    )

    return df_clean


def prepare_dataset(path):
    raw_df, df = load_dataset(
        path
    )

    df = add_structural_features(
        df
    )

    df_clean = preprocess_text(
        df
    )

    return raw_df, df, df_clean