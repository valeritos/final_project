import streamlit as st
from deepl import Translator
from pathlib import Path

#deepL needs API key for translation, free
DEEPL_AUTH_KEY="xxxxxxxxxx"
translator = Translator(DEEPL_AUTH_KEY)
lang_dict={}
for language in translator.get_target_languages():
    lang_dict[language.code]=language.name

lang_names=list(lang_dict.values())

def main():
    # Set the background image
    background_image = """
    <style>
    [data-testid="stAppViewContainer"] > .main {
        background-image: url("https://www.pixground.com/wp-content/uploads/2023/08/Purple-Abstract-Gradient-Layers-AI-Generated-4K-Wallpaper-jpg.webp");
        background-size: 100vw 100vh;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """
    st.markdown(background_image, unsafe_allow_html=True)
    html_header = """
    <div style="background-color:rgba(68, 27, 125, 0.5);padding:10px">
        <h4 style="color:white;text-align:center;">
            Welcome to your Personal Classroom Assistant
        <br>
        <span style="font-size:16px;">BETA version of translation suite</span>
        </h4>
    </div>
    """
    st.markdown(html_header, unsafe_allow_html=True)        

    with st.expander("Click here for more info"):
         st.write("This is the BETA version of the translation suite: transcripted text is translated to a language of your choice and displayed in the text box below.")    
    col1,col2=st.columns(2)
    selected_lang=str(col1.radio("Please select a language",lang_names))
    def lang_code_loc(dict):
        for code,lang in dict.items():
            if lang==selected_lang:
                return code
    
    selected_code=lang_code_loc(lang_dict)
    trans_button = col2.button("Translate")

    if trans_button:
        with open(f"{Path.home()}/Downloads/transcription_output.txt", 'r', encoding='utf-8') as file:
            text = file.read()
        col2.text_area("Initial transcription",text)
        result = translator.translate_text(text, target_lang=selected_code)
        col2.text_area("Translation", result)

if __name__ == "__main__":
    main()