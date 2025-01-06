import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai

import time
from pathlib import Path

import tempfile

from dotenv import load_dotenv
load_dotenv()

import os

API_KEY=os.getenv("GOOGLE_API_KEY")
if API_KEY:
    genai.configure(api_key=API_KEY)


#Page config
st.set_page_config(
    page_title="Multi Modal AI- Video Summarizer",
    page_icon="",
    layout="wide"
)

st.title("PhiData Multimodal Video Summarizer AI Agent")
st.header("Powered by Gemini 2.0 Flash exp")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        markdown=True,
        tools=[DuckDuckGo()]
    )

##Initializing Agent
multimodal_agent=initialize_agent()

# Video File uploader
video_file=st.file_uploader(
    "Upload a video file", type=['mp4', 'mov', 'avi'], help="Upload video for AI Analysis"
)

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='mp4') as temp_video:
        temp_video.write(video_file.read())
        video_path=temp_video.name

    st.video(video_path, format="video/mp4", start_time=0)

    user_query=st.text_area(
        "What Insights do you seek from this video?",
        placeholder="Try asking for a video summary",
        help="Provide specific questions or insights on the video",

    )

    if st.button(" Analyze video", key="analyze_video_button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyse the video")
        else:
            try:
                with st.spinner("Processing Video and gathering insights..."):
                    # Upload the video file
                    processed_video=upload_file(video_path, mime_type="video/mp4")
                    while processed_video.state.name=="PROCESSING":
                        time.sleep(1)
                        processed_video=get_file(processed_video.name)

                    #Prompt generation for analysis
                    analysis_prompt=(
                        f"""
                        Analyse the uploaded video for content and context. 
                        Respond to the following query using the video insights and supplementary web research. 
                        {user_query}

                        Provide a detailed, user-friendly and actionable response. 
                        """
                    )

                    #AI Agent processing
                    response=multimodal_agent.run(analysis_prompt, videos=[processed_video])

                #Display the result
                st.subheader("Analysis Result:")
                st.markdown(response.content)
            except Exception as e:
                st.error(f"Error has occured while analysing: {e}")
            finally:
                #Clean up the temporary file
                Path(video_path).unlink(missing_ok=True)

else:
    st.info("Upload a video file to begin analysis")

# Customizing the text area height
st.markdown(
    """
    <style>
    .stTextArea textarea{
        height: 100px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


