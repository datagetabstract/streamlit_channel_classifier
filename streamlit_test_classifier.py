from sentence_transformers import SentenceTransformer
import streamlit as st
import tensorflow as tf
from keras.models import load_model
import json
import boto3
import os

access_key = st.secrets["access_key"]
secret_key = st.secrets["secret_key"]

st.title('channel classificator')
topk = 20
classifier_filename = 'tmp_classifier.h5'
ix_ch_filename = 'ix_ch.json'


def get_classifier(classifier_filename):
    client = boto3.client('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    client.download_file('model-input-streamlit', classifier_filename, classifier_filename)
    return load_model('tmp_classifier.h5')


def get_ix_ch():
    s3 = boto3.resource('s3', aws_access_key_id=access_key, aws_secret_access_key=secret_key)
    content_object = s3.Object('model-input-streamlit', ix_ch_filename)
    file_content = content_object.get()['Body'].read().decode('utf-8')
    return {int(i): c for i, c in json.loads(file_content).items()}


def get_channel_preds(in_text, transformer, classifier, ix_ch):
    preds = classifier.predict(transformer.encode(in_text).reshape(1, -1))
    sort_pred = tf.argsort(preds, axis=-1, direction='DESCENDING', stable=False, name=None)
    proba_sort = tf.sort(preds, axis=-1, direction='DESCENDING')
    channel_proba = [(ix_ch[sp], ps) for sp, ps in zip(sort_pred.numpy()[0], proba_sort.numpy()[0])]
    return channel_proba


if os.path.isfile(classifier_filename) and os.path.isfile(ix_ch_filename) and 'classifier' in locals() and 'ix_ch' in locals():
    pass
else:
    classifier = get_classifier(classifier_filename)
    ix_ch = get_ix_ch()

transformer = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

col1, col2 = st.columns(2)
with col1:
    text_input = st.text_input("Enter some text 👇")
    st.write(f"You entered: {text_input}")

with col2:
    if text_input:
        st.write(f"Top {topk} channels with a score [range from 0 to 1]. The higher the score the higher the likelihood for the text to belong to that channel.")
        channel_proba = get_channel_preds(text_input, transformer, classifier, ix_ch)
        st.text("\n".join([f"{c}\t({p:.2})" for c, p in channel_proba[0: topk]]))
