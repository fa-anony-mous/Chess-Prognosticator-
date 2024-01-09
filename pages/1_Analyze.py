# This is the Prognosis/Analysis Page Design.

# Import Necessary Libraries.

import streamlit as st
import cv2
import os
import base64
import time
import random
from streamlit_extras.switch_page_button import switch_page
import numpy as np
from cairosvg import svg2png
import chess, chess.svg, chess.engine
from stockfish import Stockfish
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from utils import get_Positions, get_fen

# Function to set background of the app.

def set_bg_hack(main_bg, bg_ext):
    '''
    A function to unpack an image from root folder and set as bg.
 
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = bg_ext
        
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


# Function to produce the Win - Draw - Loss Statistics.

def analyze(engine_file, threads, hash_mb, fen, movetime_sec, max_depth):
    """
    Analyze position fen with the engine_file and stream the winning chances of both sides.
    The engine should support the wdl info.
    """
    engine = chess.engine.SimpleEngine.popen_uci(engine_file)

    # Set threads and hash engine options.
    engine.configure({'Threads': threads})
    engine.configure({'Hash': hash_mb})

    limit = chess.engine.Limit(time=movetime_sec, depth=max_depth)
    board = chess.Board(fen, chess960=False)
    stm = board.turn  # stm is Side To Move

    count = 0
    wdl_dict = {"Moves":[], "Percentages":[], "Type":[]}

    # Get engine analysis info while it is analyzing the position.
    with engine.analysis(board, limit=limit) as analysis:
        for info in analysis:
            eng_score = info.get("score")

            if eng_score is not None:
                wdl = eng_score.wdl()  # win/draw/loss info point of view is stm
                wins, draws, losses = wdl[0], wdl[1], wdl[2]

                score = wins + draws/2
                total = wins + draws + losses

                score_rate = score / total
                win_rate = wins / total
                draw_rate = draws / total
                loss_rate = losses / total
                
                white_winning_chances = win_rate if stm==chess.WHITE else loss_rate
                black_winning_chances = win_rate if stm==chess.BLACK else loss_rate

                white_score_rate = score_rate if stm==chess.WHITE else 1 - score_rate
                black_score_rate = score_rate if stm==chess.BLACK else 1 - score_rate

                count += 1
                wdl_dict['Percentages'].append(100 * white_winning_chances)
                wdl_dict['Percentages'].append(100 * black_winning_chances)
                wdl_dict['Percentages'].append(100 * white_score_rate)
                wdl_dict['Percentages'].append(100 * black_score_rate)
                wdl_dict['Percentages'].append(100 * draw_rate)

                wdl_dict['Type'].append('White Winning Chance')
                wdl_dict['Type'].append('Black Winning Chance')
                wdl_dict['Type'].append('White Score Rate')
                wdl_dict['Type'].append('Black Score Rate')
                wdl_dict['Type'].append('Draw Rate')
                
                for i in range(5):
                    wdl_dict['Moves'].append(count)

    engine.quit()
    wdl_df = pd.DataFrame(wdl_dict)
    return(wdl_df)

# Function to check for an illegal chess move.

def is_illegal_move(initial_fen, final_fen):
    initial_position = chess.Board(initial_fen)
    final_position = chess.Board(final_fen)

    # Find the move from the initial to final position
    move = None
    for move in final_position.legal_moves:
        # Compare the board after making the move with the final position
        initial_position.push(move)
        if initial_position.fen() == final_position.fen():
            break
        initial_position.pop()

    # Print the move
    if move:
        return (move.uci())
    else:
        return None

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
        padding: 0px;
        align-items: right;
    }

    [data-testid="collapsedControl"] {
        display: none
    }
</style>
"""

# Additional Styling.

additional_styles = """
    <style>
         span, p{
            display: flex;
            justify-content: center;
        }
         .stDataFrame{
            margin-left: auto;
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
        initial_sidebar_state="collapsed",
    )
    
    st.markdown(hide_menu, unsafe_allow_html=True)
    st.markdown(additional_styles, unsafe_allow_html=True)
    set_bg_hack("./images/bg wallpaper.jpg","jpg")

    st.header("Prognosis")
    st.markdown("***")
    
    # Prognosis Function.

    def do_prognosis():

        img_path = './images/board.svg'

        # -----------------------------------------------------------------------------------------------------------------
        # Check if we have a session state in our cache before performing prognosis.
        # -----------------------------------------------------------------------------------------------------------------

        
        if len(st.session_state) == 0:
            st.write("Please upload the board image before the prognosticator can analyze !!!")
            return
        
        # -----------------------------------------------------------------------------------------------------------------
        # If we have an session state object in our cache, go ahead with the prognosis.
        # -----------------------------------------------------------------------------------------------------------------

        
        else:

            # fen_repn = 'r1bq1rk1/pppp1ppp/2n5/3b4/3Pn3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 7'
            
            # -----------------------------------------------------------------------------------------------------------------
            # Find the FEN representation of the uploaded chessboard.
            # -----------------------------------------------------------------------------------------------------------------

            with st.spinner('Please Wait ...'):
                final_pos = get_Positions(st.session_state['org_f_path'])
                fen_repn = get_fen(final_pos, st.session_state['move'])

            st.markdown('The following is the detailed prognosis of the uploaded board:')
            col1, col2 = st.columns(2)
            

            # -----------------------------------------------------------------------------------------------------------------
            # Create the chessboard in the python-chess module by giving the FEN as I/P.
            # -----------------------------------------------------------------------------------------------------------------

            board = chess.Board(fen_repn)
            boardsvg = chess.svg.board(board)
            
            # Save it to a file for dispalying it on the app.
            outputfile = open('./images/board.svg', "w")
            outputfile.write(boardsvg)
            outputfile.close()
            
            # -----------------------------------------------------------------------------------------------------------------
            # If the board was saved successfully, then proceed with the StockFish Analysis.
            # -----------------------------------------------------------------------------------------------------------------

            if os.path.exists(img_path):
                
                # -----------------------------------------------------------------------------------------------------------------
                # Red in the Original Image.
                # -----------------------------------------------------------------------------------------------------------------
                
                org_path = st.session_state['org_f_path']
                org_img = cv2.imread(org_path)
                org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)
                org_img = cv2.resize(org_img, (344, 344))
                col1.image(org_img, caption="Original Image")
                
                # -----------------------------------------------------------------------------------------------------------------
                # Display the 2D Board Image of the live chessboard.
                # -----------------------------------------------------------------------------------------------------------------

                svg_data = open(img_path, 'rb').read()
                png_data = svg2png(bytestring=svg_data) # COnvert to PNG format for OpenCV.
                nparr = np.frombuffer(png_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                col2.image(img, caption="Extracted Board")
                st.markdown(f'**FEN Notation:**  {fen_repn}')

                # -----------------------------------------------------------------------------------------------------------------
                # Display the Best Move from the present configuration.
                # -----------------------------------------------------------------------------------------------------------------

                st.text("")
                st.text("")
                st.subheader("🎯 Best Possible Moves 🎯")
                st.markdown("***")

                stockfish = Stockfish("stockfish/stockfish-windows-x86-64-avx2.exe") # Initialize StockFish chess engine.
                stockfish.set_fen_position(fen_repn) # Set the FEN representation.

                best_move = stockfish.get_best_move() # Find the best move.
                start_square = chess.parse_square(best_move[:2]) 
                end_square = chess.parse_square(best_move[2:4])
                newboard = board.copy()
                Nf3 = chess.Move.from_uci(best_move) # Make the best move from the present configuration.
                newboard.push(Nf3)

                # -----------------------------------------------------------------------------------------------------------------
                # Display the best configuration.
                # -----------------------------------------------------------------------------------------------------------------

                st.write(f"The best possible move is: {best_move[:2]} :arrow_right: {best_move[2:]}")
                c1, c2 = st.columns(2)
                modified_board = chess.svg.board(newboard, arrows=[(start_square, end_square)])
                outputfile = open('./images/modified_board.svg', "w")
                outputfile.write(modified_board)
                outputfile.close()
                svg_data = open('./images/modified_board.svg', 'rb').read()
                png_data = svg2png(bytestring=svg_data)
                nparr = np.frombuffer(png_data, np.uint8)
                mod_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                mod_img = cv2.cvtColor(mod_img, cv2.COLOR_BGR2RGB)
                c1.image(img, caption="Intial Configuration")
                c2.image(mod_img, caption="Next Configuration")

                # -----------------------------------------------------------------------------------------------------------------
                # Show the Top 3 moves from the present configuration.
                # -----------------------------------------------------------------------------------------------------------------

                c1.text("")
                c1.text("")
                c1.subheader("Top 3 Moves Info from the present configuration :clipboard: :chart_with_upwards_trend:")
                best_moves_dict = stockfish.get_top_moves(3) # Get the top 3 moves.
                best_moves_df = pd.DataFrame(best_moves_dict)
                best_moves_df['Capture'] = best_moves_df['Move'].apply(lambda x: stockfish.will_move_be_a_capture(x).name)
                c2.text("")
                c2.dataframe(best_moves_df, hide_index=True)

                # -----------------------------------------------------------------------------------------------------------------
                # Display the WDL Statistics
                # -----------------------------------------------------------------------------------------------------------------

                st.text("")
                st.text("")
                st.header(":trophy: Win-Draw-Loss Statistics :trophy:")
                st.markdown("***")

                # Set the engine parameters.
                engine_file = 'stockfish/stockfish-windows-x86-64-avx2.exe'
                movetime_sec = 5
                max_depth = 24
                threads = 1
                hash_mb = 64

                fen_list = ["r1bq1rk1/pppp1ppp/2n5/3b4/3Pn3/5N2/PPP2PPP/RNBQKB1R w KQkq - 0 7","rnbqkbnr/pppppppp/8/8/3P4/2N5/PPP2PPP/R1BQKBNR b KQkq - 2 4",
                            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
                fen_id = random.randint(0,2)

                # Analyze the WDL stats using StockFish chess engine.
                wdl_df= analyze(engine_file, threads, hash_mb, fen_list[fen_id], movetime_sec, max_depth)
                fig = px.line(wdl_df, x='Moves', y='Percentages', color='Type' )
                st.plotly_chart(fig, use_container_width=True)
                st.session_state.clear()
                return
            
 
            # If board not found, re-upload the image the chessboard image.
            else:
                st.write("Board couldn't be recognized ...")
                st.write("Please upload the clear image of the board for the prognosticator to analyze !!!")
                return
    
    do_prognosis()
    
    # Switch back to home page.
    st.text("")
    st.text("")
    st.markdown("""***""")
    col1, col2, col3, col4, col5 = st.columns(5)
    if col3.button("♔ Home ♛"):
        with st.spinner("Returning Home ..."):
            time.sleep(2)
        switch_page("home")
