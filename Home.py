# This is the Home Page Design.

#Import Necessary Libraries.

import streamlit as st
import cv2
import base64
import time
from streamlit_extras.switch_page_button import switch_page

# Function to set the background of the app.

def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "jpg"
        
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Styling for hiding the Navbar and Sidebar.

hide_menu = """
<style>
    #MainMenu, header{
        visibility: hidden;
    }

    footer{
        display: flex;
        justify-content: center;
    }

    footer:after{
        content:"| Copyright © 2023 | Image Credits: Shutterstock";
        visibility: visible;
        display: flex;
        justify-content: center;
        position: relative;
        color: white;
        padding-left: 5px;
        align-items: right;
    }

    [data-testid="collapsedControl"] {
        display: none;
    }
</style>
"""

# Additinal Styling.

additional_styles = """
    <style>
        span, p{
            display: flex;
            justify-content: center;
        }
        .block-container{
            padding-top: 2rem;
        }
    </style>
"""

# Main Function.

if __name__ == "__main__":

    st.set_page_config(
        page_title="Chess Prognosticator",
        page_icon="♟",
        initial_sidebar_state = "collapsed",
    )


    st.markdown(hide_menu, unsafe_allow_html=True)
    st.markdown(additional_styles, unsafe_allow_html=True)
    set_bg_hack("images/bg wallpaper.jpg")

    st.title('Chess Progonosticator')
    st.header('♜ ♞ ♝ ♛ ♔ ♟')
    st.markdown("***")
    st.write("This AI-driven chess predictor analyzes opponent moves, detects checkmate possibilities, and provides strategic guidance, empowering players to make informed decisions and elevate their gameplay.")

    uploaded_img = st.file_uploader("Upload the board :chess_pawn:")

    # Check if the image is uploaded successfully.
    
    if uploaded_img is not None:

            img_path = 'images/' + uploaded_img.name
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.success("Image Upload Successfully :white_check_mark:")
            st.image(img, caption='Uploaded Chess Board')

            # Set the session state variables to be used in the other pages.
            if 'org_f_path' not in st.session_state:
                st.session_state['org_f_path'] = './' + img_path
            
            move = st.selectbox("Whose Move it is ?", ("White","Black"))
            st.session_state['move'] = 'w' if move=="White" else 'b'

            # Once everything is set, go for the Analysis page.
            col1, col2, col3, col4, col5 = st.columns(5)
            btn = col3.button("Analyze :sparkles:")
            if btn:
                with st.spinner('Processing the board ...'):
                    # time.sleep(1)
                    switch_page("Analyze")
