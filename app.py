import streamlit as st

if __name__ == "__main__":

    file = st.file_uploader("Upload an image")
    print(file)