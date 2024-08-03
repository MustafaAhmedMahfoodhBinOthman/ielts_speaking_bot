import os
import re
import requests
import json
import random
import ast
# from exa_py import Exa
from youtube_search import YoutubeSearch
import json
from supabase import create_client, Client
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton,ReplyKeyboardRemove
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.error import BadRequest, NetworkError, TimedOut
from telegram.error import Forbidden, TelegramError
from datetime import datetime, timedelta
from telegram.request import HTTPXRequest
from groq import Groq
from deepgram import DeepgramClient, PrerecordedOptions, SpeakOptions
from PIL import Image, ImageDraw, ImageFont
import logging
import asyncio
from speech_assessment import assess_speech,assess_speech2,assess_speech3

from topic_vocabularies import topic_vocabularies
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import google.generativeai as genai
import re
import random
import time
from part2_questions import ielts_questions  # Import the list of Part 2 questions
from telegram import Voice
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
from collections import defaultdict
from datetime import datetime, timedelta
import pytz
from datetime import datetime, timedelta
from collections import defaultdict
from dateutil.parser import parse
from dateutil.tz import tzutc
import pytz
from concurrent.futures import ThreadPoolExecutor
from unify import Unify
import aiohttp
from openai import AsyncOpenAI
# from pydub import AudioSegment
# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# Supabase database connection

#--------------------------APIs----------------------------------------

url: str = "https://wqlryzngdnfrarolbmma.supabase.co"
key: str=  os.getenv("supabase")
supabase: Client = create_client(url, key)

# Telegram bot token
BOT_TOKEN = os.getenv('BOT_TOKEN')
perplexity_API = os.getenv("perplexity_API")
# Groq API client
groq_API1 = os.getenv("groq_API1")
groq_client = Groq(api_key=groq_API1)

# Deepgram API client
deepgram_API = os.getenv("deepgram_API")
deepgram_API2 = os.getenv("deepgram_API2")
deepgram_api_keys = [deepgram_API, deepgram_API2]
#UnrealSpeech TTS API
unreal_speech_API1 = os.getenv("unreal_speech_API1")
unreal_speech_API2 = os.getenv("unreal_speech_API2")
unreal_speech_API3 = os.getenv("unreal_speech_API3")
unreal_speech_API4 = os.getenv("unreal_speech_API4")
# unreal_speech_API_keys=  [unreal_speech_API1, unreal_speech_API2,unreal_speech_API3,unreal_speech_API4]
unreal_speech_API_keys = [os.getenv(f"unreal_speech_API{i}") for i in range(1, 11)]
#Gemini_API_Key

Gemini_API_Key = os.getenv("Gemini_API_Key")
Gemini_API_Key2 = os.getenv("Gemini_API_Key2")
Gemini_API_Key3 = os.getenv("Gemini_API_Key3")
Gemini_API_Key4 = os.getenv("Gemini_API_Key4")
Gemini_API_Key5 = os.getenv("Gemini_API_Key5")
keys = [Gemini_API_Key,Gemini_API_Key2,Gemini_API_Key3,Gemini_API_Key4,Gemini_API_Key5]

unify_API = os.getenv("unify_API")
# deepgram_client = DeepgramClient(deepgram_API)

# List of common languages
common_languages = [
    "English", "Spanish", "French", "German", "Italian", "Portuguese", "Russian", "Japanese", "Korean", "Chinese", 
    "Arabic", "Hindi", "Bengali", "Punjabi", "Telugu", "Marathi", "Tamil", "Urdu", "Vietnamese", "Turkish",
    "Polish", "Ukrainian", "Dutch", "Greek", "Swedish", "Norwegian", "Danish", "Finnish", "Czech", "Romanian",
    "Hungarian", "Serbian", "Croatian", "Bulgarian", "Lithuanian", "Latvian", "Estonian", "Slovenian", "Slovak",
    "Afrikaans", "Swahili", "Zulu", "Amharic", "Oromo", "Hausa", "Yoruba", "Igbo", "Malay", "Indonesian",
    "Tagalog", "Thai", "Lao", "Khmer", "Burmese", "Nepali", "Sinhalese", "Mongolian", "Kazakh", "Uzbek",
    "Azerbaijani", "Georgian", "Armenian", "Hebrew", "Persian", "Pashto", "Dari", "Kurdish", "Turkmen",
    "Tajik", "Kyrgyz", "Maori", "Samoan", "Tongan", "Fijian", "Marshallese", "Chamorro", "Hawaiian"
]

translated_languages = [
    "English", "Español", "Français", "Deutsch", "Italiano", "Português", "Русский", "日本語", "한국어", "中文",
    "العربية", "हिन्दी", "বাংলা", "ਪੰਜਾਬੀ", "తెలుగు", "मराठी", "தமிழ்", "اردو", "Tiếng Việt", "Türkçe",
    "Polski", "Українська", "Nederlands", "Ελληνικά", "Svenska", "Norsk", "Dansk", "Suomi", "Čeština", "Română",
    "Magyar", "Српски", "Hrvatski", "Български", "Lietuvių", "Latviešu", "Eesti", "Slovenščina", "Slovenčina",
    "Afrikaans", "Kiswahili", "IsiZulu", "አማርኛ", "Oromoo", "Hausa", "Yorùbá", "Igbo", "Bahasa Melayu", "Bahasa Indonesia",
    "Tagalog", "ภาษาไทย", "ພາສາລາວ", "ភាសាខ្មែរ", "မြန်မာဘာသာ", "नेपाली", "සිංහල", "Монгол хэл", "Қазақ тілі", "Oʻzbek tili",
    "Azərbaycan dili", "ქართული", "Հայերեն", "עברית", "فارسی", "پښتو", "دری", "Kurdî", "Türkmen dili",
    "Тоҷикӣ", "Кыргызча", "Te Reo Māori", "Gagana Samoa", "Lea faka-Tonga", "Vosa Vakaviti", "Kajin Majol", "Fino' Chamorro", "ʻŌlelo Hawaiʻi"
]

groq_model = "llama3-70b-8192"
ADMIN_IDS = [1115038445]
voice_samples = {
    "Dan": "Dan.mp3",  # Assuming these are the filenames in your examiners_voice folder
    "William": "William.mp3",
    "Scarlett": "Scarlett.mp3",
    "Liv": "Vectoria.mp3",
    "Amy": "Amy.mp3"
}
# List to store Part 2 answers
# Handler for the /start command
#----------------------------- General Code -------------------------------------


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"{update.effective_user.id} -------------------start----------------------")
    user_id = update.effective_user.id
    username = update.effective_user.username
    print("UserID: ", user_id)
    print("username: ", username)
   
    await user_data_update(update,context)
    try:
        user = supabase.table('ielts_speaking_users').select('*').eq('user_id', user_id).execute()
        if len(user.data) == 0:
            current_time = datetime.now()
            # New user, add to database (same as before)
            data = {
                'user_id': user_id,
                'username': username,
                'email': '',
                'native_language': '',
                'english_level': '',
                'start_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'last_attempt_time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'attempts_remaining': 5,
                'practice_count': 0,
                'in_channel': False
            }
            supabase.table('ielts_speaking_users').insert(data).execute()
            user_data = context.user_data['user_data']
            user_id = update.effective_user.id
            user_data['user_id'] = str(user_id)
            userID = user_data['user_id']
            # Send welcome message and request email
            keyboard = [
                [InlineKeyboardButton("Okay, no problem", callback_data=f'{userID}provide_email'),
                InlineKeyboardButton("No", callback_data=f'{userID}skip_email')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("Welcome to the IELTS Speaking Practice Bot! Would you like to provide your email address?", reply_markup=reply_markup)

        else:
            # Existing user, check for missing information
            user_data1 = user.data[0]
            if not user_data1['native_language']:
                await ask_language(update, context)
            elif not user_data1['english_level']:
                await ask_english_level(update, context)
            elif not user_data1['target_ielts_score']:
                await ask_target_ielts_score(update, context)
            elif not user_data1['examiner_voice']:
                await ask_preferred_voice(update, context)
            else:
                user = supabase.table('ielts_speaking_users').select('examiner_voice', 'target_ielts_score').eq('user_id', user_id).execute()
                if user.data:
                    context.user_data['user_data']['examiner_voice'] = user.data[0]['examiner_voice']
                    context.user_data['user_data']['targeted_score'] = user.data[0]['target_ielts_score']
                    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")

                    # Update the last_practice_date in the Supabase database
                    supabase.table('ielts_speaking_users').update({'last_practice_date': current_date}).eq('user_id', user_id).execute()
                    text = "Welcome back! What would you like to do?"
                    await show_main_menu(update, context, text)

    except Exception as e:
        text = (f"🚨 Start function", e)
        await error_handling(update, context,text)

# Create a ThreadPoolExecutor
try:
    num_cores = os.cpu_count()

    # Create the thread pool
    executor = ThreadPoolExecutor(max_workers=num_cores)  # Adjust the number of workers as needed
except Exception as e:
    print("🚨 executor = ThreadPoolExecutor(max_workers=8)",e)
    executor = ThreadPoolExecutor(max_workers=8)
# Helper function to ask for language
async def ask_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"{update.effective_user.id} ask language")
    try:
        # Ensure user_data exists for this user
        if 'user_data' not in context.user_data:
            context.user_data['user_data'] = {}
        user_data = context.user_data['user_data']

        # Store the current state in user_data
        user_data['current_state'] = 'asking_language'
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        keyboard = [
            [InlineKeyboardButton("Arabic", callback_data=f'{userID}language_Arabic'),
            InlineKeyboardButton("Urdu", callback_data=f'{userID}language_Urdu')],
            [InlineKeyboardButton("Uzbek", callback_data=f'{userID}language_Uzbek'),
            InlineKeyboardButton("Persian", callback_data=f'{userID}language_Persian')],
            [InlineKeyboardButton("Hindi", callback_data=f'{userID}language_Hindi'),
            InlineKeyboardButton("Other", callback_data=f'{userID}language_other')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.effective_message.reply_text("Please select your native language:", reply_markup=reply_markup)
    except Exception as e:
        text = ("🚨 ask language function", e)
        await error_handling(update, context,text)

async def check_channel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Ensure user_data exists for this user
        if 'user_data' not in context.user_data:
            context.user_data['user_data'] = {}
        user_data = context.user_data['user_data']

        channel_id = "@IELTS_SpeakingBOT"
        user_id = update.effective_user.id
        chat_member = await context.bot.get_chat_member(chat_id=channel_id, user_id=user_id)
        
        if chat_member.status in ["member", "creator", "administrator"]:
            print(f"Yes {user_id} is a member")
            supabase.table('ielts_speaking_users').update({
                'in_channel': True
            }).eq('user_id', user_id).execute()
            user_data['in_channel'] = True
            return True
        else:
            supabase.table('ielts_speaking_users').update({
                'in_channel': False
            }).eq('user_id', user_id).execute()
            user_data['in_channel'] = False
            return False
    except Exception as e:
        text = ("🚨 check channel ", e)
        user_data['in_channel'] = False
        return False


async def ask_channel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Ensure user_data exists for this user
        if 'user_data' not in context.user_data:
            context.user_data['user_data'] = {}
        user_data = context.user_data['user_data']

        channel_id = "@IELTS_SpeakingBOT"
        user_id = update.effective_user.id
        chat_member = await context.bot.get_chat_member(chat_id=channel_id, user_id=user_id)
        
        if chat_member.status in ["member", "creator", "administrator"]:
            print(f"Yes {user_id} is a member")
            supabase.table('ielts_speaking_users').update({
                'in_channel': True
            }).eq('user_id', user_id).execute()
            user_data['in_channel'] = True
            return True
        else:
            # supabase.table('ielts_speaking_users').update({
            #     'in_channel': False
            # }).eq('user_id', user_id).execute()
            # user_data['in_channel'] = False
            
            print(f"No {user_id} isn't a member")
            # pass
            text = "I have created a channel to share updates about the bot and provide the best resources for IELTS. \nPlease join us at @IELTS_SpeakingBOT."
            await context.bot.send_message(chat_id=update.effective_chat.id, text=text)
            
            return False
    except Exception as e:
        text = ("🚨 check channel ", e)
        user_data['in_channel'] = False
        return False

async def score_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if 'user_data' not in context.user_data:
            context.user_data['user_data'] = {}
    user_data = context.user_data['user_data']
    user = supabase.table('ielts_speaking_users').select('examiner_voice', 'target_ielts_score').eq('user_id', user_id).execute()
    if user.data:
        user_data['examiner_voice'] = user.data[0]['examiner_voice']
        user_data['targeted_score'] = user.data[0]['target_ielts_score']
    # print(user_data['examiner_voice'] , user_data['targeted_score'])

async def user_data_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    username = update.effective_user.username
    # print("UserID: ", user_id)
    # print("username: ", username)

    # Initialize user-specific data
    if 'user_data' not in context.user_data:
        context.user_data['user_data'] = {}
    user_data = context.user_data['user_data']

    # Clear user-specific data
    user_data.clear()

    # Initialize user-specific lists and variables
    user_data['questions_list'] = []
    user_data['answers_list'] = []
    user_data['detailed_feedback_list'] = []
    user_data['voice_urls'] = []
    user_data['questions'] = []
    user_data['analysis_list'] = []
    user_data['list_previous_quetions'] = []
    user_data['list_previous_answers'] = []
    user_data['translated_feedback1'] = []
    user_data['current_state'] =[]

    user_data['part_1_minute_part_1'] =[]
    user_data['part_1_minute_part_2'] =[]
    user_data['part_1_minute_part_3'] =[]
    user_data['part_1_minute_part_1_mock'] =[]
    user_data['part_1_minute_part_1_mock'] =[]
    
    user_data['part2_voice_urls'] = []
    user_data['part2_questions'] = []
    user_data['part2_answers'] = []
    user_data['analysis2_list'] = []
    user_data['detailed_feedback2_list'] = []
    user_data['translated_feedback2'] = []
    user_data['part3_voice_urls'] = []
    user_data['part3_questions'] = []
    user_data['part3_answers'] = []
    user_data['analysis3_list'] = []
    user_data['detailed_feedback3_list'] = []
    user_data['translated_feedback3'] = []
    user_data['mock_part1_questions'] = []
    user_data['mock_part1_answers'] = []
    user_data['mock_part1_voice_urls'] = []
    user_data['mock_part2_questions'] = []
    user_data['mock_part2_answers'] = []
    user_data['mock_part2_voice_urls'] = []
    user_data['mock_part3_questions'] = []
    user_data['mock_part3_answers'] = []
    user_data['mock_part3_voice_urls'] = []
    user_data['mock_part1_analysis_list'] = []
    user_data['mock_part2_analysis_list'] = []
    user_data['mock_part3_analysis_list'] = []
    user_data['mock_part1_detailed_feedback_list'] = []
    user_data['mock_part2_detailed_feedback_list'] = []
    user_data['mock_part3_detailed_feedback_list'] = []
    user_data['in_channel']= False
    user_data['part_1_minute'] = False
    user_data['part_3_minute'] = False
    user_data['test_stop'] = False
    user_data['continue_countdown'] =True
    user_data['current_question_index'] = 0          
    print(f"{update.effective_user.id} user_data has been updated") 
# async def error_handling(update: Update, context: ContextTypes.DEFAULT_TYPE,error_message):
#     print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} | {error_message}")
#     # await update.message.reply_text(issue_message)
#     issue_message = "🚨 Sorry for the inconvenience, it seems there is an issue with the bot. If that happens, please contact me @ielts_pathway."
#     text = issue_message
    
#     await show_main_menu(update, context, text)
async def error_handling(update: Update, context: ContextTypes.DEFAULT_TYPE, error_message):
    print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} | {error_message}")
    issue_message = "🚨 Apologies for the inconvenience; there's an issue with the bot. Please try again, and if the problem persists, contact me at @ielts_pathway."
    text = issue_message
    userID = update.effective_user.id
    keyboard = [[InlineKeyboardButton("Try Again 🔁", callback_data=f"{userID}try_again")]]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(text, reply_markup=reply_markup)  
async def ask_english_level(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"{update.effective_user.id} ask english level")
    try:
        # Ensure user_data exists for this user
        if 'user_data' not in context.user_data:
            context.user_data['user_data'] = {}
        user_data = context.user_data['user_data']

        # Store the current state in user_data
        user_data['current_state'] = 'asking_english_level'
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        keyboard = [
            [InlineKeyboardButton("Beginner", callback_data=f'{userID}level_Beginner')],
            [InlineKeyboardButton("Elementary", callback_data=f'{userID}level_Elementary')],
            [InlineKeyboardButton("Intermediate", callback_data=f'{userID}level_Intermediate')],
            [InlineKeyboardButton("Upper Intermediate", callback_data=f'{userID}level_UpperIntermediate')],
            [InlineKeyboardButton("Advanced", callback_data=f'{userID}level_Advanced')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.effective_message.reply_text("What is your current level of English?", reply_markup=reply_markup)
    except Exception as e:
        text = ("🚨 ask language level function ", e)
        await error_handling(update, context,text)
async def ask_target_ielts_score(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"{update.effective_user.id} ask target ielts score")
    try:
        # Ensure user_data exists for this user
        if 'user_data' not in context.user_data:
            context.user_data['user_data'] = {}
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        # Store the current state in user_data
        user_data['current_state'] = 'asking_target_ielts_score'

        # Create a list of scores including half scores
        scores = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0] 
        keyboard = [[InlineKeyboardButton(str(score), callback_data=f'{userID}score_{score}')] for score in scores]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.effective_message.reply_text("What is your targeted IELTS speaking score?", reply_markup=reply_markup)
    except Exception as e:
        text = ("🚨 ask targeted score function", e)
        await error_handling(update, context,text)

async def format_lesson_titles(lessons_text):
  prompt = f"""
  You are tasked with formatting the following lesson recommendations into a Python list. 
  The input text contains recommendations for YouTube lessons based on IELTS speaking feedback.
  
  Please format the input into a Python list with exactly three string items.
  Each item should contain only the YouTube video title, nothing else.
  
  Input text:
  {lessons_text}
  
  Formatted output example: i want it exactly like this example format
  ["title 1", "title 2", "title 3"]
  
  for the title 3 the lesson whould be from IELTS advantage or english speaking success youtube channels so please include the name of the channel in the title
  Please provide the formatted output as a valid Python list:
  make sure that you include only the titles to the list and make it exactly like the python list and do not write anything else only write the list do not include any other text i hope you follow this instruction exactly your output should be only a python list and it will be processed to extract the titles for other operations in python functions so be 100% accurate
  """
  
  formatted_lessons = await feedback_lessons(prompt)
  
  # Use ast.literal_eval to safely evaluate the string as a Python expression
  try:
      lessons_list = ast.literal_eval(formatted_lessons.strip())
      if isinstance(lessons_list, list) and len(lessons_list) == 3:
          return lessons_list
      else:
          raise ValueError("Output is not a list with 3 items")
  except:
      # If parsing fails, fall back to simple string splitting
      return [title.strip() for title in formatted_lessons.split(',') if title.strip()][:3]  
async def feedback_lessons(prompt):
    
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model="llama3-groq-70b-8192-tool-use-preview",
        )
        feedback = chat_completion.choices[0].message.content
        # feedback = re.sub(r'\*', '', feedback)  # Remove asterisks (*)
        # feedback = re.sub(r'#', '', feedback)  # Remove hash symbols (#)
        
        return feedback
        # client = AsyncOpenAI(
        #         base_url="https://api.unify.ai/v0/",
        #         api_key=unify_API
        #     )

        
                
        # response = await client.chat.completions.create(
        #                 model="gpt-4o@openai",
        #                 messages=[
        #                     {"role": "user", "content": prompt}
        #                 ],
        #                 max_tokens=4096,
        #                 temperature=0.0,
        #                 stream=False
        #             )
                    
        # feedback = response.choices[0].message.content
        # return feedback
        
    except Exception as e:
        print("🚨 Groq error switching to perplexity generate_feedback_with_llm ",e)
        messages = [
                {
                    "role": "system",
                    "content": (
                        ""
                       
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                    ),
                },
            ]
        client = OpenAI(api_key=perplexity_API, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model="llama-3.1-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        feedback = (response.choices[0].message.content)
        # feedback = re.sub(r'\*', '', feedback)  # Remove asterisks (*)
        # feedback = re.sub(r'#', '', feedback)  # Remove hash symbols (#)
        
        return feedback

async def recommendation_lesson(feedback):
  initial_prompt = f"""As an IELTS speaking expert, analyze the following detailed feedback for an IELTS Speaking test and recommend three highly specific YouTube lessons to address the main weaknesses identified. Your recommendations should be tailored to significantly improve the test-taker's performance.

  Guidelines:
  1. Carefully examine the feedback across all areas: Pronunciation, Fluency, Grammatical Range and Accuracy, Lexical Resource, and Relevance and Coherence.
  2. Identify the three most critical areas for improvement based on the lowest scores and most impactful weaknesses.
  3. For each area, recommend one highly specific YouTube lesson that directly addresses the identified weakness.
  4. Ensure that one of the three lessons is from a popular IELTS YouTube channel such as IELTS Advantage or English Speaking Success (please include the name of the channel in the lesson title ).
  5. Provide only the exact titles of the YouTube videos, as these will be used for searching.
  6. Choose lessons that are practical, engaging, and likely to yield significant improvements.

  The detailed feedback is as follows:

  {feedback}

  Based on this feedback, please provide three YouTube lesson titles that address the main weaknesses identified in this feedback.
  """
  
  initial_lessons = await feedback_lessons(initial_prompt)
#   print("Initial lesson recommendations:", initial_lessons)
  
  formatted_lessons = await format_lesson_titles(initial_lessons)
#   print("Formatted lesson titles:", formatted_lessons)
  
  recommendations = []
  for lesson_title in formatted_lessons:
      print("lesson_title: ",lesson_title)
      results = YoutubeSearch(lesson_title, max_results=1).to_json()
      parsed_results = json.loads(results)
      
      if parsed_results['videos']:
          url_suffix = parsed_results['videos'][0]['url_suffix']
          title = parsed_results['videos'][0]['title']
          full_url = f"https://www.youtube.com{url_suffix}"
          recommendations.append({
              'title': title,
              'url': full_url
          })
  
#   print("Final recommendations:", recommendations)
  return recommendations

async def send_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE, feedback):
  recommendations = await recommendation_lesson(feedback)
  chat_id = update.effective_chat.id
  
  if recommendations:
      await context.bot.send_message(chat_id=chat_id, text="Based on your feedback, here are recommended lessons to enhance your IELTS speaking skills.\n\n🔶This new feature is still in the experimental stage; please contact me at @ielts_pathway for any issues. ")
      
      for i, rec in enumerate(recommendations, 1):
          message = f"{i}. {rec['title']}\n{rec['url']}"
          print(message)
          await context.bot.send_message(chat_id=chat_id, text=message)
          await asyncio.sleep(0.5)  # Add a 0.5 second delay between messages
  else:
      await context.bot.send_message(chat_id=chat_id, text="Sorry, I couldn't find any relevant lessons based on your feedback.")

async def ask_preferred_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"{update.effective_user.id} ask preferred examiner voice")
    try:
        # Ensure user_data exists for this user
        if 'user_data' not in context.user_data:
            context.user_data['user_data'] = {}
        user_data = context.user_data['user_data']

        # Store the current state in user_data
        user_data['current_state'] = 'asking_preferred_voice'

        user_id = update.effective_user.id

        # Send each voice sample as an audio message
        for voice_name, filename in voice_samples.items():
            # Construct the full path to the audio file
            file_path = os.path.join("examiners_voice", filename) 

            # Send the audio file
            with open(file_path, 'rb') as audio_file:
                await context.bot.send_audio(chat_id=update.effective_chat.id, audio=audio_file)
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        # Create inline keyboard for voice selection
        keyboard = [
            [InlineKeyboardButton("Dan", callback_data=f'{userID}voice_Dan')],
            [InlineKeyboardButton("William", callback_data=f'{userID}voice_Will')],
            [InlineKeyboardButton("Scarlett", callback_data=f'{userID}voice_Scarlett')],
            [InlineKeyboardButton("Vectoria", callback_data=f'{userID}voice_Liv')],
            [InlineKeyboardButton("Amy", callback_data=f'{userID}voice_Amy')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text("Please listen to the voice samples above and choose your preferred voice:", reply_markup=reply_markup)
    except Exception as e:
        text = ("🚨 ask preferred voice function ", e)
        await error_handling(update, context,text)
# Handler for user messages
# Handler for user messages
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # global  test_stop, part_1_minute, part_3_minute
    if 'user_data' not in context.user_data:
            context.user_data['user_data'] = {}
    user_data = context.user_data['user_data']
    user_id = update.effective_user.id
    user_data['user_id'] = str(user_id)
    userID = user_data['user_id']
    query = update.callback_query
    # user = supabase.table('ielts_speaking_users').select('examiner_voice', 'target_ielts_score').eq('user_id', user_id).execute()
    # if user.data:
    #     context.user_data['user_data']['examiner_voice'] = user.data[0]['examiner_voice']
    #     context.user_data['user_data']['targeted_score'] = user.data[0]['target_ielts_score']
    # print(userID)
    print(f"{update.effective_user.id} message handler")
    try:
        # Ensure user_data exists for this user
        if 'user_data' not in context.user_data:
            context.user_data['user_data'] = {}
        user_data = context.user_data['user_data']

        user_id = update.effective_user.id
        username = update.effective_user.username
        
        text = update.message.text
        print(f"{update.effective_user.id} Received message: {text}")
        query = update.callback_query
        user = supabase.table('ielts_speaking_users').select('*').eq('user_id', user_id).execute()

        if 'email_prompt' in user_data:
            print(f"{update.effective_user.id} email_prompt")
            if is_valid_gmail(text) and is_real_gmail(text):
                supabase.table('ielts_speaking_users').update({'email': text}).eq('user_id', user_id).execute()
                del user_data['email_prompt']
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
                await ask_language(update, context)
            else:
                await update.message.reply_text("Please enter a valid Gmail address.")
        elif 'other_language_prompt' in user_data:
            print(f"{update.effective_user.id} other_language_prompt")
            if text in translated_languages:
                index = translated_languages.index(text)
                native_language = common_languages[index]
                supabase.table('ielts_speaking_users').update({'native_language': native_language}).eq('user_id', user_id).execute()
                del user_data['other_language_prompt']
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
                await ask_english_level(update, context)
            elif text.capitalize() in common_languages:
                native_language = text.capitalize()
                supabase.table('ielts_speaking_users').update({'native_language': native_language}).eq('user_id', user_id).execute()
                del user_data['other_language_prompt']
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
                await ask_english_level(update, context)
            else:
                await update.message.reply_text("Please enter a valid language name in either your native language or English.")
        elif 'other_language_prompt2' in user_data:
            print(f"{update.effective_user.id} other_language_prompt2")
            if text in translated_languages:
                index = translated_languages.index(text)
                native_language = common_languages[index]
                supabase.table('ielts_speaking_users').update({'native_language': native_language}).eq('user_id', user_id).execute()
                del user_data['other_language_prompt2']
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
                text = "The language has been successfully changed."
                await show_main_menu(update, context, text)
            elif text.capitalize() in common_languages:
                native_language = text.capitalize()
                supabase.table('ielts_speaking_users').update({'native_language': native_language}).eq('user_id', user_id).execute()
                del user_data['other_language_prompt2']
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
                text = "The language has been successfully changed."
                await show_main_menu(update, context, text)
            else:
                await update.message.reply_text("Please enter a valid language name in either your native language or English.")
        elif 'english_level_prompt' in user_data:
            print(f"{update.effective_user.id} english_level_prompt")
            if text in ['Beginner', 'Elementary', 'Intermediate', 'Upper Intermediate', 'Advanced']:
                supabase.table('ielts_speaking_users').update({'english_level': text}).eq('user_id', user_id).execute()
                del user_data['english_level_prompt']
                await ask_target_ielts_score(update, context)
            else:
                await update.message.reply_text("Please select a valid English level from the options provided.")
        elif 'target_ielts_score_prompt' in user_data:
            print(f"{update.effective_user.id} target_ielts_score_prompt")
            try:
                score = float(text)
                if 5.0 <= score <= 9.0:
                    supabase.table('ielts_speaking_users').update({'target_ielts_score': score}).eq('user_id', user_id).execute()
                    del user_data['target_ielts_score_prompt']
                    await ask_test_part(update, context)
                else:
                    await update.message.reply_text("Please enter a valid IELTS score between 5.0 and 9.0.")
            except ValueError:
                await update.message.reply_text("Please enter a valid IELTS score between 5.0 and 9.0.")
        elif 'part_1_topic_selection' in user_data:
            print(f"{update.effective_user.id} part_1_topic_selection")
            topic = text
            if topic.isdigit() and 1 <= int(topic) <= len(user_data['part_1_topics']):
                selected_topic = user_data['part_1_topics'][int(topic) - 1]
                user_data['selected_topic'] = selected_topic
                message_id = user_data.get('topic_message_id')
                if message_id:
                    await context.bot.edit_message_text(
                        chat_id=update.effective_chat.id,
                        message_id=message_id,
                        text=f"You have selected the topic: {selected_topic}"
                    )
                del user_data['part_1_topic_selection']
                await generate_and_ask_questions(update, context, selected_topic)
            else:
                if text == "Stop the Test" :
                    print(f"{update.effective_user.id} stop the test")
                    try:
                        del user_data['part_1_topic_selection']
                    except Exception as e:
                        pass
                    try:
                        #print("Find and remove the inline keyboard from the previous message")
                        chat_id = update.effective_chat.id
                        message_id = update.message.message_id - 1  # Assuming the previous message has the keyboard
                    except Exception as e:
                        print(e)
                    try:
                        await context.bot.edit_message_reply_markup(
                            chat_id=chat_id,
                            message_id=message_id,
                            reply_markup=None
                        )
                    except Exception as e:
                        print(f"Error removing inline keyboard: {e}")
                    await update.message.reply_text("The test will stop now....")
                    await start(update, context)
                else:
                    await update.message.reply_text("Please enter a valid topic number.")
        elif text == "Part 1":
            # user_id = update.effective_user.id
            # user_data['user_id'] = str(user_id)
            # userID = user_data['user_id']
            # print(userID)
            await check_user_attempts(update, context)
            if user_data['remaining_attempts'] > 0:
                print(f"{update.effective_user.id} Part 1 selected")
                user_data['questions_list'] = []
                user_data['answers_list'] = []
                user_data['detailed_feedback_list'] = []
                user_data['voice_urls'] = []
                user_data['questions'] = []
                user_data['analysis_list'] = []
                user_data['list_previous_quetions'] = []
                user_data['list_previous_answers'] = []
                user_data['part_1_minute'] = False
                user_data['part_3_minute'] = False
                await user_data_update(update,context)
                await stop_test(update, context, "IELTS Speaking test Part 1 ")
                # await score_voice(update, context)
                asyncio.create_task(ask_part_1_topic(update, context))
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context, text)
        elif text == "Part 2":
            print(f"{update.effective_user.id} Part 2 selected")
            await check_user_attempts(update, context)
            if user_data['remaining_attempts'] > 0:
                user_data['part2_voice_urls'] = []
                user_data['part2_questions'] = []
                user_data['part2_answers'] = []
                user_data['analysis2_list'] = []
                user_data['detailed_feedback2_list'] = []
                user_data['part_1_minute'] = False
                user_data['part_3_minute'] = False
                await user_data_update(update,context)
                await score_voice(update, context)
                await stop_test(update, context, "IELTS Speaking test Part 2 ")
                await update.effective_message.reply_text(
                    "Now, we will begin Part 2 of the IELTS Speaking test. In this part, you will be given a topic and you will have one minute to prepare. \nAfter that, you should speak on the topic for 1 to 2 minutes. \nYou can make notes if you wish. \n\nLet's start")
                asyncio.create_task(start_part2_test(update, context))
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context, text)
        elif text == "Part 3":
            print(f"{update.effective_user.id} Part 3 selected")
            await check_user_attempts(update, context)
            if user_data['remaining_attempts'] > 0:
                user_data['part3_voice_urls'] = []
                user_data['part3_questions'] = []
                user_data['part3_answers'] = []
                user_data['analysis3_list'] = []
                user_data['detailed_feedback3_list'] = []
                user_data['part_1_minute'] = False
                user_data['part_3_minute'] = False
                await user_data_update(update,context)
                # await score_voice(update, context)
                await stop_test(update, context, "IELTS Speaking test Part 3")
                user_data = context.user_data['user_data']
                user_id = update.effective_user.id
                user_data['user_id'] = str(user_id)
                userID = user_data['user_id']
                keyboard = [
                    [InlineKeyboardButton("Take Part 2 first", callback_data=f'{userID}take_part2_first')],
                    [InlineKeyboardButton("Skip to Part 3", callback_data=f'{userID}skip_part2')]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text("Do you want to take Part 2 first or skip to Part 3? (because IELTS Part 3 is related to Part 2)", reply_markup=reply_markup)
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context, text)
        elif text == "Mock Test (Full test)":
            print(f"{update.effective_user.id} mock test selected")
            await check_user_attempts(update, context)
            if user_data['remaining_attempts'] > 0:
                user_data['mock_part1_questions'] = []
                user_data['mock_part1_answers'] = []
                user_data['mock_part1_voice_urls'] = []
                user_data['mock_part2_questions'] = []
                user_data['mock_part2_answers'] = []
                user_data['mock_part2_voice_urls'] = []
                user_data['mock_part3_questions'] = []
                user_data['mock_part3_answers'] = []
                user_data['mock_part3_voice_urls'] = []
                user_data['mock_part1_analysis_list'] = []
                user_data['mock_part2_analysis_list'] = []
                user_data['mock_part3_analysis_list'] = []
                user_data['mock_part1_detailed_feedback_list'] = []
                user_data['mock_part2_detailed_feedback_list'] = []
                user_data['mock_part3_detailed_feedback_list'] = []
                user_data['part_1_minute'] = False
                user_data['part_3_minute'] = False
                await user_data_update(update,context)
                # await score_voice(update, context)
                await stop_test(update, context, "IELTS Speaking Mock test")
                asyncio.create_task(start_mock_test(update, context))
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context, text)
        elif text == "Start Test":
            print(f"{update.effective_user.id} Start Test selected")
            print(f"{update.effective_user.id} ------------------------------ Start Test ---------------------------")
            print("UserID: ", user_id)
            print("username: ", username)
            await user_data_update(update,context)
            # user_id = update.effective_user.id
            # user_data['user_id'] = str(user_id)
            # userID = user_data['user_id']
            # print(userID)
            user_data['part_1_minute'] = False
            user_data['part_3_minute'] = False
            # try:
            #     await query.edit_message_reply_markup(reply_markup=None)
            # except Exception as e:
            #     # print("error: await query.edit_message_reply_markup(reply_markup=None)")
            #     pass
            try:
                # Find and remove the inline keyboard from the previous message
                chat_id = update.effective_chat.id
                message_id = update.message.message_id - 1  # Assuming the previous message has the keyboard
            except Exception as e:
                print(e)
            try:
                await context.bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=None
                )
            except Exception as e:
                print(f"Error removing inline keyboard: {e}")
            await check_channel(update, context)
            try: 
                user = supabase.table('ielts_speaking_users').select('*').eq('user_id', user_id).execute()
                user_data1 = user.data[0]
                if not user_data1['native_language']:
                    await ask_language(update, context)
                elif not user_data1['english_level']:
                    await ask_english_level(update, context)
                elif not user_data1['target_ielts_score']:
                    await ask_target_ielts_score(update, context)
                elif not user_data1['examiner_voice']:
                    await ask_preferred_voice(update, context)
                # elif not user_data1['email']:
                #     print("no email")
                else:
                    await ask_test_part(update, context)
            except Exception as e:
                print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} |🚨 elif text == Start Test: ", e)
                await ask_test_part(update, context)
        elif text == "Main menu":
            print(f"{update.effective_user.id} Main menu selected")
            user_data['part_1_minute'] = False
            user_data['part_3_minute'] = False
            # try:
            #     await query.edit_message_reply_markup(reply_markup=None)
            # except Exception as e:
            #     # print("error: await query.edit_message_reply_markup(reply_markup=None)")
            #     pass
            try:
                #print("Find and remove the inline keyboard from the previous message")
                chat_id = update.effective_chat.id
                message_id = update.message.message_id - 1  # Assuming the previous message has the keyboard
            except Exception as e:
                print(e)
            try:
                await context.bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=None
                )
            except Exception as e:
                print(f"Error removing inline keyboard: {e}")
            await show_main_menu(update, context, "Main menu")
        elif text == "Bot Channel":
            print(f"{update.effective_user.id} Bot channel selected")
            # try:
            #     await query.edit_message_reply_markup(reply_markup=None)
            # except Exception as e:
            #     # print("error: await query.edit_message_reply_markup(reply_markup=None)")
            #     pass
            try:
                #print("Find and remove the inline keyboard from the previous message")
                chat_id = update.effective_chat.id
                message_id = update.message.message_id - 1  # Assuming the previous message has the keyboard
            except Exception as e:
                print(e)
            try:
                await context.bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=None
                )
            except Exception as e:
                print(f"Error removing inline keyboard: {e}")
            text = "I have created a channel to share updates about the bot and provide the best resources for IELTS. Please join us at @IELTS_SpeakingBOT."
            await show_main_menu(update, context, text)
        elif text == "Show Progress":
            print(f"{update.effective_user.id} Show Progress selected")
            # try:
            #     await query.edit_message_reply_markup(reply_markup=None)
            # except Exception as e:
            #     # print("error: await query.edit_message_reply_markup(reply_markup=None)")
            #     pass
            try:
                #print("Find and remove the inline keyboard from the previous message")
                chat_id = update.effective_chat.id
                message_id = update.message.message_id - 1  # Assuming the previous message has the keyboard
            except Exception as e:
                print(e)
            try:
                await context.bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=None
                )
            except Exception as e:
                print(f"Error removing inline keyboard: {e}")
            await update.message.reply_text("Your progress in IELTS Speaking ")
            await show_progress(update, context)
        elif text == "Stop the Test":
            print(f"{update.effective_user.id} stop the test")
            try:
                del user_data['part_1_topic_selection']
            except Exception as e:
                pass
            # try:
            #     await query.edit_message_reply_markup(reply_markup=None)
            # except Exception as e:
            #     # print("error: await query.edit_message_reply_markup(reply_markup=None)")
            #     pass
            try:
                #print("Find and remove the inline keyboard from the previous message")
                chat_id = update.effective_chat.id
                message_id = update.message.message_id - 1  # Assuming the previous message has the keyboard
            except Exception as e:
                print(e)
            try:
                await context.bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=None
                )
            except Exception as e:
                print(f"Error removing inline keyboard: {e}")
            await update.message.reply_text("The test will stop now....")
            user_data['test_stop'] = True
            try:
                user_data['continue_countdown'] = False  # Stop the countdown

                messages_to_delete = [
                    'part2_question_message_id',
                    'part2_audio_message_id',
                    'part2_preparation_message_id',
                    'part2_waiting_message_id',
                    'part2_countdown_message_id',
                    'part2_recording_message_id'  # Add this if you want to delete the recording message too
                ]

                for message_type in messages_to_delete:
                    message_id = user_data.get(message_type)
                    if message_id:
                        try:
                            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=message_id)
                        except BadRequest as e:
                            if "Message to delete not found" not in str(e):
                                print(f"{update.effective_user.id} 🚨 Error deleting message: {str(e)}")
                        finally:
                            # Clear the message ID from user_data regardless of whether deletion was successful
                            user_data[message_type] = None
            except Exception as e:
                print(e)
            await user_data_update(update,context)
            await start(update, context)
        elif text == "Contact Me":
            print(f"{update.effective_user.id} Contact Me selected")
            # try:
            #     await query.edit_message_reply_markup(reply_markup=None)
            # except Exception as e:
            #     # print("error: await query.edit_message_reply_markup(reply_markup=None)")
            #     pass
            try:
                #print("Find and remove the inline keyboard from the previous message")
                chat_id = update.effective_chat.id
                message_id = update.message.message_id - 1  # Assuming the previous message has the keyboard
            except Exception as e:
                print(e)
            try:
                await context.bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=None
                )
            except Exception as e:
                print(f"Error removing inline keyboard: {e}")
            await update.message.reply_text("You can contact me at @ielts_pathway.")
        elif text == "Settings":
            keyboard = [
            # [KeyboardButton("Start Test")],
            # [KeyboardButton("IELTS Writing 📝"),KeyboardButton("Show Progress")],
            # [KeyboardButton("Contact Me"), KeyboardButton("Bot Channel")],
            [KeyboardButton("Change language"), KeyboardButton("Change voice")],
            # [KeyboardButton("Change Email")],
            [KeyboardButton("Main menu")],
           
        ]
            reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
            await update.effective_message.reply_text(text, reply_markup=reply_markup)
        elif text == "Change language":
            print(f"{update.effective_user.id} change language selected")
            # try:
            #     await query.edit_message_reply_markup(reply_markup=None)
            # except Exception as e:
            #     # print("error: await query.edit_message_reply_markup(reply_markup=None)")
            #     pass
            try:
                #print("Find and remove the inline keyboard from the previous message")
                chat_id = update.effective_chat.id
                message_id = update.message.message_id - 1  # Assuming the previous message has the keyboard
            except Exception as e:
                print(e)
            try:
                await context.bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=None
                )
            except Exception as e:
                print(f"Error removing inline keyboard: {e}")
            await change_language(update, context)
        elif text == "Change voice":
            print(f"{update.effective_user.id} change voice selected")
            # try:
            #     await query.edit_message_reply_markup(reply_markup=None)
            # except Exception as e:
            #     # print("error: await query.edit_message_reply_markup(reply_markup=None)")
            #     pass
            try:
                #print("Find and remove the inline keyboard from the previous message")
                chat_id = update.effective_chat.id
                message_id = update.message.message_id - 1  # Assuming the previous message has the keyboard
            except Exception as e:
                print(e)
            try:
                await context.bot.edit_message_reply_markup(
                    chat_id=chat_id,
                    message_id=message_id,
                    reply_markup=None
                )
            except Exception as e:
                print(f"Error removing inline keyboard: {e}")
            await change_voice(update, context)
        elif text == "Admin":
            if update.effective_user.id in ADMIN_IDS:
                keyboard = [
                    [InlineKeyboardButton("Broadcast", callback_data="admin_broadcast"),
                     InlineKeyboardButton("Bot Statistics", callback_data="admin_stats")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text("Admin options:", reply_markup=reply_markup)
                # return ADMIN_OPTIONS
            else:
                await update.message.reply_text("You are not authorized to access admin options.")
                text2 = "You are not authorized to access admin options."
                return await show_main_menu(update, context, text2)
        elif text == "IELTS Writing 📝":
            await update.message.reply_text("Now you can evalaute your IELTS Writing essay using our IELTS Writing Evaluator Bot:\n\n@ielts_writing2_bot")
        elif text == "All Users":
            context.user_data['broadcast_target'] = 'all'
            # print(f"Broadcast target all users: {context.user_data.get('broadcast_target')}")
            print("Target set to all users")
            await handle_broadcast_target(update ,context)
        elif text == "Never Practiced Users":
            context.user_data['broadcast_target'] = 'never_practiced'
            # print(f"Broadcast target never_practiced: {context.user_data.get('broadcast_target')}")
            print("Target set to never_practiced")
            await handle_broadcast_target(update ,context)
        elif context.user_data.get('in_broadcast_mode', False):
          print(f"Handling broadcast message")
          if update.message.text == "Done":
              print("Received 'Done' in broadcast mode")
              if not context.user_data.get('broadcast_messages'):
                  await update.message.reply_text("You haven't sent any messages to broadcast. Please send at least one message.")
                  return

              keyboard = [
                  [InlineKeyboardButton("Confirm", callback_data="confirm_broadcast"),
                   InlineKeyboardButton("Cancel", callback_data="cancel_broadcast")]
              ]
              reply_markup = InlineKeyboardMarkup(keyboard)
              
              await update.message.reply_text("Here's a preview of your broadcast messages:", reply_markup=reply_markup)
              for message in context.user_data['broadcast_messages']:
                  await send_message_copy(update.effective_chat.id, message, context)
              
              # Remove the "Done" and "Cancel" buttons
              await update.message.reply_text("Please confirm or cancel the broadcast.", 
                                              reply_markup=ReplyKeyboardRemove())
              
          elif update.message.text == "Cancel":
              context.user_data['in_broadcast_mode'] = False
              context.user_data.pop('broadcast_messages', None)
              await show_main_menu(update, context, "Broadcast cancelled. Returning to main menu.")
          else:
              if 'broadcast_messages' not in context.user_data:
                  context.user_data['broadcast_messages'] = []
              context.user_data['broadcast_messages'].append(update.message)
              keyboard = [
      [KeyboardButton("Done"), KeyboardButton("Cancel")]
                ]
              reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
              await update.message.reply_text("Message added to broadcast. Send more messages or click 'Done' when finished.",reply_markup=reply_markup)
          return
        else:
            print("I'm sorry, but while practicing speaking, it's best to only record your voice instead of sending text messages. If you need further assistance, please reach out to me at @ielts_pathway.")
            await update.message.reply_text("I'm sorry, but while practicing speaking, it's best to only record your voice instead of sending text messages. If you need further assistance, please reach out to me at @ielts_pathway.")

    
    except Exception as e:
        text = ("🚨 message handler function ", e)
        await error_handling(update, context,text)
# Handler for voice messages
async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE): 
    print(f"{update.effective_user.id} voice handler")
    try:
        # Ensure user_data exists for this user
        if 'user_data' not in context.user_data:
            context.user_data['user_data'] = {}
        user_data = context.user_data['user_data']

        user_id = update.effective_user.id
        
        if 'part2_answering' in user_data:
            voice_duration = update.message.voice.duration
            print(f"{update.effective_user.id} answering_question part 2")
            if voice_duration < 60:
                await update.message.reply_text("Your answer is less than 1 minute. Please re-record your answer and make it longer.")
            elif voice_duration > 121:
                await update.message.reply_text("Your answer is more than 2 minutes . Please try to shorten your answer and re-record it.")
            else:
                transcribed_text = await convert_audio_to_text(update.message.voice.file_id, update, context)
                
                if transcribed_text:
                    user_data['part2_answer'] = transcribed_text
                    user_data['part2_voice_file_id'] = update.message.voice.file_id
                    user_data = context.user_data['user_data']
                    user_id = update.effective_user.id
                    user_data['user_id'] = str(user_id)
                    userID = user_data['user_id']
                    await update.message.reply_text(f"Your answer: \n{transcribed_text}\nAre you sure about your answer?", reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("Yes, I am sure", callback_data=f'{userID}part2_confirm_answer')],
                        [InlineKeyboardButton("Try again", callback_data=f'{userID}part2_retry_answer')]
                    ]))
                else:
                    await update.message.reply_text("Sorry, I couldn't get your answer. Please try again.")
        elif 'answering_question' in user_data:
            print(f"{update.effective_user.id} answering_question part 1")
            if 'current_question' in user_data:
                question = user_data['current_question']
                if update.message.voice:
                    print(f"{update.effective_user.id} Voice message received")
                    file_id = update.message.voice.file_id
                    file = await context.bot.get_file(file_id)
                    file_path = file.file_path
                    part_1_minute_part1 = user_data['part_1_minute_part_1']

                    voice_duration = update.message.voice.duration
                    if voice_duration > 58:
                        user_data['part_1_minute'] = True
                        part_1_minute_part1.append(True)
                        print(f"{update.effective_user.id} user exceeds 1 minute: ", user_data['part_1_minute'])
                    else:
                        part_1_minute_part1.append(False)
                        user_data['part_1_minute'] = False
                    if voice_duration > 80:  # 1 minute and 30 seconds
                        await update.message.reply_text("Your answer is too long. Please record another answer shorter than 1 minute.")
                    else:
                        transcribed_text = await convert_audio_to_text(file_id, update, context)
                        
                        if transcribed_text:
                            user_data['current_answer'] = transcribed_text
                            user_data['current_voice_file_id'] = file_id
                            user_data = context.user_data['user_data']
                            user_id = update.effective_user.id
                            user_data['user_id'] = str(user_id)
                            userID = user_data['user_id']
                            await update.message.reply_text(f"Your answer: \n{transcribed_text}\nAre you sure about your answer?", reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("Yes, I am sure", callback_data=f'{userID}confirm_answer')],
                                [InlineKeyboardButton("Try again", callback_data=f'{userID}retry_answer')]
                            ]))
                        else:
                            await update.message.reply_text("Sorry, I couldn't get your answer. Please try again.")
                else:
                    print(f"{update.effective_user.id} No voice message received")
                    await update.message.reply_text("Please send a voice message as your answer.")
            else:
                text = ("🚨 No current question in context")
                await error_handling(update, context,text)
        elif 'answering_part3_question' in user_data:
            print(f"{update.effective_user.id} answering part 3")
            question = user_data['current_part3_question']
            voice_file_id = update.message.voice.file_id
            voice_file = await context.bot.get_file(voice_file_id)
            voice_url = voice_file.file_path
            voice_duration = update.message.voice.duration
            if voice_duration > 58:
                user_data['part_3_minute'] = True
                print(f"{update.effective_user.id} user exceeds 1 minute  : ", user_data['part_3_minute'])
            if voice_duration > 90:  # 1 minute and 30 seconds
                        await update.message.reply_text("Your answer is too long. Please record another answer shorter than 90 seconds.")
            else: 
                transcribed_text = await convert_audio_to_text(voice_file_id, update, context)

                if transcribed_text:
                    user_data['current_part3_answer'] = transcribed_text
                    user_data['current_part3_voice_url'] = update.message.voice.file_id
                    user_data = context.user_data['user_data']
                    user_id = update.effective_user.id
                    user_data['user_id'] = str(user_id)
                    userID = user_data['user_id']
                    await update.message.reply_text(f"Your answer: \n{transcribed_text}\nAre you sure about your answer?", reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("Yes, I am sure", callback_data=f'{userID}part3_confirm_answer')],
                        [InlineKeyboardButton("Try again", callback_data=f'{userID}part3_retry_answer')],
                    ]))
                else:
                    await update.message.reply_text("Sorry, I couldn't get your answer. Please try again.")
        
        elif 'mock_part1_answering' in user_data:
            print(f"{update.effective_user.id} mock_part1_answering")
            current_question_index = user_data.get('mock_part1_current_question_index', 0)

            user_answer_voice = update.message.voice
            voice_duration = update.message.voice.duration
            if voice_duration > 58:
                user_data['part_1_minute'] = True
                print(f"{update.effective_user.id} user exceeds 1 minute (part 1 mock)", user_data['part_1_minute'])
            if voice_duration > 80:  # 1 minute and 30 seconds
                        await update.message.reply_text("Your answer is too long. Please record another answer shorter than 1 minute.")
            else:
                if user_answer_voice:
                    voice_file_path_url = await get_voice_file_path_url(user_answer_voice)
                    user_answer_text = await convert_audio_to_text(user_answer_voice.file_id, update, context)

                    if user_answer_text:
                        user_data.setdefault('mock_part1_answers', []).append(user_answer_text)
                        user_data.setdefault('mock_part1_voice_urls', []).append(voice_file_path_url)

                        user_data['mock_part1_current_question_index'] = current_question_index + 1
                        del user_data['mock_part1_answering']

                        await mock_part1_process(update, context)
                    else:
                        await update.message.reply_text("Sorry, I couldn't get your answer seems you have not answered the question. Please try again.")
                else:
                    await update.message.reply_text("Please provide a voice message with your answer.")
        elif 'mock_part2_answering' in user_data:
            user_answer_voice = update.message.voice
            voice_duration = update.message.voice.duration
            if user_answer_voice:
                if voice_duration < 60:
                    await update.message.reply_text("Your answer is less than 1 minute. Please re-record your answer and make it longer.")
                elif voice_duration > 121:
                    await update.message.reply_text("Your answer is too long more than 2 minutes . Please try to shorten your answer and re-record it.")
                else:
                    voice_file_path_url = await get_voice_file_path_url(user_answer_voice)
                    user_answer_text = await convert_audio_to_text(user_answer_voice.file_id, update, context)
                    
                    if user_answer_text:
                        user_data.setdefault('mock_part2_answers', []).append(user_answer_text)
                        user_data.setdefault('mock_part2_voice_urls', []).append(voice_file_path_url)

                        del user_data['mock_part2_answering']

                        await update.effective_message.reply_text("Mock Test - Part 2 completed. Moving to Part 3.")
                        await asyncio.sleep(3)
                        await mock_part3_process(update, context)
                    else:
                        await update.message.reply_text("Sorry, I couldn't get your answer. Please try again.")
            else:
                await update.message.reply_text("Please provide a voice message with your answer.")

        elif 'mock_part3_answering' in user_data:
            user_answer_voice = update.message.voice
            voice_duration = update.message.voice.duration
            if voice_duration > 58:
                user_data['part_3_minute'] = True
                print(f"{update.effective_user.id} user exceeds 1 minute (part 3 mock)", user_data['part_3_minute'])
            if voice_duration > 90:  # 1 minute and 30 seconds
                        await update.message.reply_text("Your answer is too long. Please record another answer shorter than 90 seconds.")
            else:     
                if user_answer_voice:
                    voice_file_path_url = await get_voice_file_path_url(user_answer_voice)
                    user_answer_text = await convert_audio_to_text(user_answer_voice.file_id, update, context)

                    if user_answer_text:
                        user_data.setdefault('mock_part3_answers', []).append(user_answer_text)
                        user_data.setdefault('mock_part3_voice_urls', []).append(voice_file_path_url)

                        user_data['mock_part3_current_question_index'] += 1
                        del user_data['mock_part3_answering']

                        await mock_part3_process(update, context)
                    else:
                        await update.message.reply_text("Sorry, I couldn't get your answer. Please try again.")
                else:
                    await update.message.reply_text("Please provide a voice message with your answer.")
        else:
            print(f"{update.effective_user.id} Not answering a question")
            await update.message.reply_text("Please select a topic and start answering the questions. \nto start from the beginning /start")
    except Exception as e:
        text = ("🚨 voice handler function ", e)
        await error_handling(update, context,text)
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"{update.effective_user.id} button handler")
    try:
        

        
        # Ensure user_data exists for this user
        if 'user_data' not in context.user_data:
            context.user_data['user_data'] = {}
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = user_id
        userID = user_data['user_id']
        # print("r",userID)
        query = update.callback_query
        try:
            # await query.answer()
            await query.answer()
        except BadRequest as e:
            print(f"{update.effective_user.id} 🚨 Error answering callback query: {str(e)}")

        if query.data == f'{userID}provide_email':
            user_data['email_prompt'] = True
            print(f"{update.effective_user.id} user will add his email")
            await query.edit_message_text("Please enter your email address:")

        elif query.data == f'{userID}skip_email':
            print(f"{update.effective_user.id} user will not add his email")
            await query.edit_message_reply_markup(reply_markup=None)
            await query.edit_message_text("Skipping email registration.")
            await ask_language(update, context)

        elif query.data.startswith(f'{userID}language_'):
            language = query.data.split('_')[1]
            if language == 'other':
                user_data['other_language_prompt'] = True
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text("Please enter the name of your native language:")
            else:
                user_id = query.from_user.id
                supabase.table('ielts_speaking_users').update({'native_language': language}).eq('user_id', user_id).execute()
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text(f"You selected {language} as your native language.")
                await ask_english_level(update, context)

        elif query.data.startswith(f'{userID}language2_'):
            user_data = context.user_data.setdefault('user_data', {})
            language = query.data.split('_')[1]
            user_id = query.from_user.id

            if language == 'other':
                user_data['other_language_prompt2'] = True
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text("Please enter the name of your native language:")
            else:
                try:
                    # Update the language in Supabase
                    supabase.table('ielts_speaking_users').update({'native_language': language}).eq('user_id', user_id).execute()
                    
                    # Store the language in user_data as well
                    user_data['language'] = language
                    
                    await query.edit_message_reply_markup(reply_markup=None)
                    await query.edit_message_text(f"You selected {language} as your native language.")
                    text = "The language has been successfully changed."
                    await show_main_menu(update, context, text)
                except Exception as e:
                    text = (f"🚨 Error updating language in Supabase: {e}")
                    await query.edit_message_text("There was an error updating your language. Please try again later.")
                    await error_handling(update, context,text)

        elif query.data.startswith(f'{userID}level_'):
            english_level = query.data.split('_')[1]
            user_id = query.from_user.id
            supabase.table('ielts_speaking_users').update({'english_level': english_level}).eq('user_id', user_id).execute()
            await query.edit_message_reply_markup(reply_markup=None)
            await query.edit_message_text(f"You selected {english_level} as your English level.")
            await ask_target_ielts_score(update, context)

        elif query.data.startswith(f'{userID}score_'):
            target_score = query.data.split('_')[1]
            user_id = query.from_user.id
            supabase.table('ielts_speaking_users').update({'target_ielts_score': target_score}).eq('user_id', user_id).execute()
            await query.edit_message_reply_markup(reply_markup=None)
            await query.edit_message_text(f"You have set your target IELTS score to {target_score}.")
            await ask_preferred_voice(update, context)

        elif query.data.startswith(f'{userID}voice_'):
            selected_voice = query.data.split('_')[1]
            user_id = query.from_user.id
            supabase.table('ielts_speaking_users').update({'examiner_voice': selected_voice}).eq('user_id', user_id).execute()
            await query.edit_message_reply_markup(reply_markup=None)
            await show_main_menu(update, context, f"You have selected {selected_voice} as your examiner voice.")
            message_text = "I have created a channel to share updates about the bot and provide the best resources for IELTS. Please join us at @IELTS_SpeakingBOT."
            keyboard_buttons = [[InlineKeyboardButton("Join Channel ✨", url="https://t.me/IELTS_SpeakingBOT")]]
            reply_markup = InlineKeyboardMarkup(keyboard_buttons)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=message_text, reply_markup=reply_markup)
            # await context.bot.send_message(chat_id=update.effective_chat.id, text="I have created a channel to share updates about the bot and provide the best resources for IELTS. Please join us at @IELTS_SpeakingBOT.")
        
        elif query.data == f"{userID}try_again":
            try:
                # Remove the inline keyboard from the message with the "Try Again" button
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception as e:
                print(f"Error removing inline keyboard: {e}")
            
            await query.message.reply_text("Try again.")
            await user_data_update(update, context)
            await ask_test_part(update, context)
            # await show_main_menu(update,context,"Try Again")
        elif query.data == f"{userID}topic_random":
            selected_topic = random.choice(user_data['part_1_topics'])
            user_data['selected_topic'] = selected_topic
            await query.edit_message_text(f"the topic is: {selected_topic}")
            
            if 'part_1_topic_selection' in user_data:
                del user_data['part_1_topic_selection']
            await generate_and_ask_questions(update, context, selected_topic)

        elif query.data == f'{userID}confirm_answer':
            current_question = user_data.get('current_question')
            current_answer = user_data.get('current_answer')
            
            voice_file_id = user_data.get('current_voice_file_id')
            if voice_file_id:
                file = await context.bot.get_file(voice_file_id)
                file_path = file.file_path
                
                user_data.setdefault('voice_urls', []).append(file_path)
            
            user_data.setdefault('answers_list', []).append(current_answer)
            # print("answers_listconfirmanswer   ",user_data['answers_list'])
            await query.edit_message_reply_markup(reply_markup=None)
            user_data['current_question_index'] = user_data.get('current_question_index', 0) + 1
            await ask_current_question(update, context)

        elif query.data == f'{userID}retry_answer':
            await query.edit_message_reply_markup(reply_markup=None)
            await ask_current_question(update, context, retry=True)

        elif query.data == f'{userID}suggest_answer':
            question = user_data.get('current_question')
            previous_answer = user_data.get('current_answer')
            suggested_answer = await generate_suggested_answer(question, previous_answer, "part 1 ",context)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Suggested Answer:\n\n {suggested_answer}")
            await query.edit_message_reply_markup(reply_markup=None)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please record your answer again.")

        elif query.data == f'{userID}retake_part_1':
            await query.edit_message_reply_markup(reply_markup=None)
            await check_user_attempts(update, context)
            if user_data['remaining_attempts'] > 0:
                print(f"{update.effective_user.id} Part 1 selected")
                user_data['questions_list'] = []
                user_data['answers_list'] = []
                user_data['detailed_feedback_list'] = []
                user_data['voice_urls'] = []
                user_data['questions'] = []
                user_data['analysis_list'] = []
                user_data['list_previous_quetions'] = []
                user_data['list_previous_answers'] = []
                user_data['part_1_minute'] = False
                user_data['part_3_minute'] = False
                await user_data_update(update,context)
                await stop_test(update, context, "IELTS Speaking test Part 1 ")
                await ask_part_1_topic(update, context)
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n For more information please contact me @ielts_pathway"
                await show_main_menu(update, context, text)

        elif query.data == f'{userID}show_results':
            await query.edit_message_reply_markup(reply_markup=None)
            asyncio.create_task(show_results_part1(update, context))

        elif query.data == f'{userID}continue_part_2':
            print(f"{update.effective_user.id} continue to part 2")
            await check_user_attempts(update, context)
            # if context.user_data.get('remaining_attempts', 0) > 0:
            if user_data['remaining_attempts'] > 0:   
                # user_data.clear()
                user_data['part_1_minute'] = False
                user_data['part_3_minute'] = False
                await user_data_update(update,context)
                try:
                    del user_data['answering_question']
                except Exception as e:
                    pass
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text("Continuing to Part 2...")
                asyncio.create_task(start_part2_test(update, context))
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n For more information please contact me @ielts_pathway"
                await show_main_menu(update, context, text)

        elif query.data == f'{userID}detailed_results':
            animated_emoji = "⏳"
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
            await query.edit_message_reply_markup(reply_markup=None)
            
            # Create a background task for generating detailed feedback
            context.application.create_task(generate_detailed_feedback(update, context, waiting_message))

            # Don't wait for the task to complete here
            return

        
        elif query.data == f'{userID}translate_overall_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            user_id = query.from_user.id
            overall_feedback = user_data.get('overall_feedback')
            animated_emoji = "⏳"
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
            
            keyboard = [
                [InlineKeyboardButton("Continue to Part 2", callback_data=f'{userID}continue_part_2')],
                [InlineKeyboardButton("See Detailed Results", callback_data=f'{userID}detailed_results')],
                [InlineKeyboardButton("Retake Part 1", callback_data=f'{userID}retake_part_1')],
                [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)

            if overall_feedback:
                # Create a background task for translation
                context.application.create_task(
                    _translate_and_send_feedback(update, context, user_id, overall_feedback, waiting_message, reply_markup)
                )
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry there is an issue please contact me if this happened again")
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
        elif query.data == f'{userID}translate_detailed_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            user_id = query.from_user.id
            animated_emoji = "⏳"
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait while we translate the detailed feedback. This may take a few minutes.")
            
            keyboard = [
                [InlineKeyboardButton("Continue to Part 2", callback_data=f'{userID}continue_part_2')],
                [InlineKeyboardButton("Retake Part 1", callback_data=f'{userID}retake_part_1')],
                [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            # Create a background task for translating detailed feedback
            context.application.create_task(
                _translate_and_send_detailed_feedback(update, context, user_id, waiting_message, reply_markup)
            )
            
        
         # part2 button handler 
        elif query.data == f'{userID}part2_confirm_answer':
            user_data = context.user_data.setdefault('user_data', {})
            current_question = user_data.get('part2_question')
            current_answer = user_data.get('part2_answer')
            voice_file_id = user_data.get('part2_voice_file_id')

            user_data.setdefault('part2_questions', []).append(current_question)
            user_data.setdefault('part2_answers', []).append(current_answer)

            if voice_file_id:
                file = await context.bot.get_file(voice_file_id)
                file_path = file.file_path
                # print(file_path)
                user_data.setdefault('part2_voice_urls', []).append(file_path)

            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Continue to Part 3", callback_data=f'{userID}continue_part3')],
                [InlineKeyboardButton("Retake Part 2", callback_data=f'{userID}retake_part2')],
                [InlineKeyboardButton("Check Your Score ✅", callback_data=f'{userID}show_part2_results')],
                [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
            ]))

        elif query.data == f'{userID}part2_retry_answer':
            await query.edit_message_reply_markup(reply_markup=None)
            await update.effective_message.reply_text("Please record your answer again.")

        elif query.data == f'{userID}continue_part3':
            print(f"{update.effective_user.id} continue to part 3")
            await check_user_attempts(update, context)
            if user_data['remaining_attempts'] > 0:
                user_data = context.user_data.setdefault('user_data', {})
                user_data['part3_voice_urls'] = []
                user_data['part3_answers'] = []
                user_data['analysis3_list'] = []
                user_data['detailed_feedback3_list'] = []
                user_data['translated_feedback3'] = []
                user_data['part_1_minute'] = False
                user_data['part_3_minute'] = False
                try:
                    del user_data['part2_answering']
                except Exception as e:
                    pass
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text("Continuing to Part 3...")
                asyncio.create_task(start_part3_test(update, context))
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n For more information please contact me @ielts_pathway"
                await show_main_menu(update, context, text)

        elif query.data == f'{userID}retake_part2':
            print(f"{update.effective_user.id} retake part 2")
            await check_user_attempts(update, context)
            if user_data['remaining_attempts'] > 0:
                user_data = context.user_data.setdefault('user_data', {})
                user_data['part2_questions'] = []
                user_data['translated_feedback2'] = []
                user_data['part2_answers'] = []
                user_data['part2_voice_urls'] = []
                user_data['analysis2_list'] = []
                user_data['detailed_feedback2_list'] = []
                user_data['overall_feedback'] = []
                user_data['part_1_minute'] = False
                user_data['part_3_minute'] = False
                await user_data_update(update,context)
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text("Retaking Part 2...")
                asyncio.create_task(start_part2_test(update, context))
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n For more information please contact me @ielts_pathway"
                await show_main_menu(update, context, text)

        elif query.data == f'{userID}show_part2_results':
            await query.edit_message_reply_markup(reply_markup=None)
            asyncio.create_task(show_result2(update, context))

        # elif query.data == f'{userID}change_question':
        #     try:
        #         print("change_question")
                
        #         user_data = context.user_data.get('user_data', {})
        #         for message_type in ['part2_question_message_id', 'part2_audio_message_id', 'part2_preparation_message_id', 'part2_waiting_message_id']:
        #             message_id = user_data.get(message_type)
        #             if message_id:
        #                 await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=message_id)
        #     except BadRequest as e:
        #         print(f"{update.effective_user.id} 🚨 Error deleting message: {str(e)}")
            
        #     asyncio.create_task(start_part2_test(update, context))
        elif query.data == f'{userID}change_question':
            try:
                print("change_question")
                
                user_data = context.user_data.get('user_data', {})
                user_data['continue_countdown'] = False  # Stop the countdown

                messages_to_delete = [
                    'part2_question_message_id',
                    'part2_audio_message_id',
                    'part2_preparation_message_id',
                    'part2_waiting_message_id',
                    'part2_countdown_message_id',
                    'part2_recording_message_id'  # Add this if you want to delete the recording message too
                ]

                for message_type in messages_to_delete:
                    message_id = user_data.get(message_type)
                    if message_id:
                        try:
                            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=message_id)
                        except BadRequest as e:
                            if "Message to delete not found" not in str(e):
                                print(f"{update.effective_user.id} 🚨 Error deleting message: {str(e)}")
                        finally:
                            # Clear the message ID from user_data regardless of whether deletion was successful
                            user_data[message_type] = None

                # Wait a short time to ensure the countdown loop has stopped
                await asyncio.sleep(0.5)
                
                # Start a new Part 2 test
                asyncio.create_task(start_part2_test(update, context))
            except Exception as e:
                print(f"{update.effective_user.id} 🚨 Error in change_question: {str(e)}")
        elif query.data == f'{userID}detailed2_results':
            animated_emoji = "⏳"
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
            await query.edit_message_reply_markup(reply_markup=None)
            
            # Create a background task for generating detailed feedback
            context.application.create_task(generate_detailed2_feedback(update, context, waiting_message))

            # Don't wait for the task to complete here
            return

        elif query.data == f'{userID}end_test':
            await query.edit_message_reply_markup(reply_markup=None)
            await query.edit_message_text("Ending the test...")
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Thank you for taking the test. Goodbye!")
            rating_keyboard = [
                [InlineKeyboardButton("👍", callback_data=f'{userID}rate_up')],
                [InlineKeyboardButton("👎", callback_data=f'{userID}rate_down')]
            ]
            rating_reply_markup = InlineKeyboardMarkup(rating_keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="How was your experience:", reply_markup=rating_reply_markup)
            await user_data_update(update,context)
        elif query.data == f'{userID}rate_up':
            await query.edit_message_text("❤️")
            print(f"{userID} 👍")
            share_message = (
                f"Try the IELTS Speaking Bot! It simulates the IELTS speaking test and offers detailed feedback on your speaking skills and estimated band score. Try it for free: https://t.me/ielts_speakingAI_bot"
            )
            keyboard = [
                [InlineKeyboardButton("Share the Bot", switch_inline_query=share_message)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="If you find it helpful, don't forget to share the bot 😊\n\nIf you have any suggestions to improve the bot, please contact me @ielts_pathway",
                reply_markup=reply_markup,
                parse_mode='HTML'
            )
            text = "Thank you so much for using the bot!"
            await show_main_menu(update, context, text)

        elif query.data == f'{userID}rate_down':
            print(f"{userID} 👎")
            text = "I really appreciate your feedback. \nPlease contact me and tell me what was the problem and try to enhance your experience next time: \n@ielts_pathway"
            await show_main_menu(update, context, text)

        elif query.data == f'{userID}translate_overall2_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            user_id = query.from_user.id
            user_data = context.user_data.get('user_data', {})
            overall_feedback = user_data.get('overall_part2_feedback')
            keyboard = [
                [InlineKeyboardButton("Continue to Part 3", callback_data=f'{userID}continue_part3')],
                [InlineKeyboardButton("See Detailed Results", callback_data=f'{userID}detailed2_results')],
                [InlineKeyboardButton("Retake Part 2", callback_data=f'{userID}retake_part2')],
                [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            animated_emoji = "⏳"
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")

            if overall_feedback:
                # Create a background task for translation
                context.application.create_task(
                    _translate_and_send_feedback2(update, context, user_id, overall_feedback, waiting_message, reply_markup)
                )
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, there is an issue. Please contact me if this happens again.")
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)

        elif query.data == f'{userID}translate_detailed2_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            keyboard = [
                [InlineKeyboardButton("Continue to Part 3", callback_data=f'{userID}continue_part3')],
                [InlineKeyboardButton("Retake Part 2", callback_data=f'{userID}retake_part2')],
                [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            user_id = query.from_user.id
            animated_emoji = "⏳"
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait while we translate the detailed feedback. This may take a few minutes.")
            
            user_data = context.user_data.get('user_data', {})
            detailed_feedback2_list = user_data.get('detailed_feedback2_list', [])
            
            # Create a background task for translating detailed feedback
            context.application.create_task(
                _translate_and_send_detailed2_feedback(update, context, user_id, detailed_feedback2_list, waiting_message, reply_markup)
            )

        elif query.data == f'{userID}take_part2_first':
            await query.edit_message_reply_markup(reply_markup=None)
            user_data = context.user_data.setdefault('user_data', {})
            user_data['part2_questions'] = []
            user_data['translated_feedback2'] = []
            user_data['part2_answers'] = []
            user_data['part2_voice_urls'] = []
            user_data['analysis2_list'] = []
            user_data['detailed_feedback2_list'] = []
            user_data['overall_feedback'] = []
            user_data['part_1_minute'] = False
            user_data['part_3_minute'] = False
            await user_data_update(update,context)
            print(f"{update.effective_user.id} take part 2 first")
            asyncio.create_task(start_part2_test(update, context))

        elif query.data == f'{userID}skip_part2':
            await query.edit_message_reply_markup(reply_markup=None)
            selected_topic = random.choice(list(ielts_questions.keys()))
            part2_question = ielts_questions[selected_topic]["part_2_question"]
            part_3_questions = ielts_questions[selected_topic]["part_3_questions"]

            user_data = context.user_data.setdefault('user_data', {})
            user_data['part2_question'] = part2_question
            main_part2_question = part2_question.split('.')[0] + '.'
            # print(part_3_questions)
            user_data['part3_questions'] = part_3_questions
            user_data['part_1_minute'] = False
            user_data['part_3_minute'] = False

            await query.edit_message_text(f"Assuming you have answered the following Part 2 question:\n\n{main_part2_question}\n\nNow, let's proceed to Part 3.")
            asyncio.create_task(start_part3_test(update, context))
        
                # part3 button handler 
        elif query.data == f'{userID}part3_confirm_answer':
            user_data = context.user_data.setdefault('user_data', {})
            current_question = user_data.get('current_part3_question')
            current_answer = user_data.get('current_part3_answer')
            
            voice_file_id = user_data.get('current_part3_voice_url')
            if voice_file_id:
                file = await context.bot.get_file(voice_file_id)
                file_path = file.file_path
                user_data.setdefault('part3_voice_urls', []).append(file_path)
            else:       
                print(f"{update.effective_user.id} sorry there is no path for the voice")

            user_data.setdefault('part3_answers', []).append(current_answer)
            
            await query.edit_message_reply_markup(reply_markup=None)
            
            user_data['part3_current_question_index'] = user_data.get('part3_current_question_index', 0) + 1
            await ask_part3_question(update, context)

        elif query.data == f'{userID}part3_retry_answer':
            print(f"{update.effective_user.id} retry part3")   
            await query.edit_message_reply_markup(reply_markup=None)
            await ask_part3_question(update, context, retry=True)

        elif query.data == f'{userID}part3_suggest_answer':
            await part3_suggest_answer(update, context)

        elif query.data == f'{userID}part3_show_summary':
            user_data = context.user_data.get('user_data', {})
            part3_questions = user_data.get('part3_questions', [])
            part3_answers = user_data.get('part3_answers', [])
            
            summary_message = "Part 3 Questions and Answers:\n\n"
            for i in range(len(part3_questions)):
                question = part3_questions[i]
                answer = part3_answers[i]
                summary_message += f"Question {i+1}: {question}\nAnswer: {answer}\n\n"
            
            await context.bot.send_message(chat_id=update.effective_chat.id, text=summary_message)
            
            keyboard = [
                [InlineKeyboardButton("Check Your Score ✅", callback_data=f'{userID}part3_show_results')],
                [InlineKeyboardButton("Retake Part 3", callback_data=f'{userID}part3_retake')],
                [InlineKeyboardButton("End Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)

        elif query.data == f'{userID}part3_show_results':
            await query.edit_message_reply_markup(reply_markup=None)
            asyncio.create_task(part3_show_results(update, context)) 

        elif query.data == f'{userID}part3_retake':
            await query.edit_message_reply_markup(reply_markup=None)
            keyboard = [
                [InlineKeyboardButton("Take Part 2 first", callback_data=f'{userID}take_part2_first')],
                [InlineKeyboardButton("Skip to Part 3", callback_data=f'{userID}skip_part2')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await check_user_attempts(update, context)
            if user_data['remaining_attempts'] > 0:
                user_data = context.user_data.setdefault('user_data', {})
                user_data['part3_voice_urls'] = []
                user_data['part3_questions'] = []
                user_data['part3_answers'] = []
                user_data['analysis3_list'] = []
                user_data['detailed_feedback3_list'] = []
                user_data['translated_feedback3'] = []
                user_data['part_1_minute'] = False
                user_data['part_3_minute'] = False
                await stop_test(update, context, "IELTS Speaking test Part 3")
                
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Do you want to take Part 2 first or skip to Part 3? (because IELTS Part 3 is related to Part 2)", reply_markup=reply_markup)
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n For more information please contact me @ielts_pathway"
                await show_main_menu(update, context, text)

        elif query.data == f'{userID}part3_detailed_results':
            animated_emoji = "⏳"
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
            await query.edit_message_reply_markup(reply_markup=None)
            
            # Create a background task for generating detailed feedback
            context.application.create_task(part3_detailed_results(update, context, waiting_message))

            # Don't wait for the task to complete here
            return

        elif query.data == f'{userID}part3_translate_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            user_id = query.from_user.id
            user_data = context.user_data.get('user_data', {})
            overall_feedback = user_data.get('overall_part3_feedback')
            keyboard = [
                [InlineKeyboardButton("See Detailed Results", callback_data=f'{userID}part3_detailed_results')],
                [InlineKeyboardButton("Retake Part 3", callback_data=f'{userID}part3_retake')],
                [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            animated_emoji = "⏳"
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")

            if overall_feedback:
                # Create a background task for translation
                context.application.create_task(
                    _translate_and_send_part3_feedback(update, context, user_id, overall_feedback, waiting_message, reply_markup)
                )
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, there is an issue. Please contact me if this happens again.")
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
        elif query.data == f'{userID}part3_translate_detailed_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            keyboard = [
                [InlineKeyboardButton("Retake Part 3", callback_data=f'{userID}part3_retake')],
                [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            user_id = query.from_user.id
            animated_emoji = "⏳"
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait while we translate the detailed feedback. This may take a few minutes.")
            
            user_data = context.user_data.get('user_data', {})
            detailed_feedback3_list = user_data.get('detailed_feedback3_list', [])
            
            # Create a background task for translating detailed feedback
            context.application.create_task(
                _translate_and_send_part3_detailed_feedback(update, context, user_id, detailed_feedback3_list, waiting_message, reply_markup)
            )
        # mock test button handler 
        elif query.data == f'{userID}mock_test_retake':
            print(f"{update.effective_user.id} retake mock test")
            await query.edit_message_reply_markup(reply_markup=None)
            
            await check_user_attempts(update, context)
            if user_data['remaining_attempts'] > 0:
                user_data = context.user_data.setdefault('user_data', {})
                
                # Clear all data
                # for key in ['questions_list', 'answers_list', 'detailed_feedback_list', 'voice_urls', 'questions', 
                #             'analysis_list', 'list_previous_questions', 'list_previous_answers', 
                #             'part2_voice_urls', 'part2_questions', 'part2_answers', 'analysis2_list', 
                #             'detailed_feedback2_list', 'part3_voice_urls', 'part3_questions', 'part3_answers', 
                #             'analysis3_list', 'detailed_feedback3_list', 'mock_part1_questions', 
                #             'mock_part1_answers', 'mock_part1_voice_urls', 'mock_part2_questions', 
                #             'mock_part2_answers', 'mock_part2_voice_urls', 'mock_part3_questions', 
                #             'mock_part3_answers', 'mock_part3_voice_urls', 'mock_part1_analysis_list', 
                #             'mock_part2_analysis_list', 'mock_part3_analysis_list', 
                #             'mock_part1_detailed_feedback_list', 'mock_part2_detailed_feedback_list', 
                #             'mock_part3_detailed_feedback_list']:
                #     user_data[key] = []
                await user_data_update(update,context)
                user_data['part_1_minute'] = False
                user_data['part_3_minute'] = False
                
                await stop_test(update, context, "IELTS Speaking Mock test")
                asyncio.create_task(start_mock_test(update, context))
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n For more information please contact me @ielts_pathway"
                await show_main_menu(update, context, text)

        elif query.data == f'{userID}mock_test_show_results':
            await query.edit_message_reply_markup(reply_markup=None)
            asyncio.create_task(show_mock_test_results(update, context))

        elif query.data == f'{userID}mock_test_detailed_results':
            await query.edit_message_reply_markup(reply_markup=None)
            # await context.bot.send_message(
            #     chat_id=update.effective_chat.id,
            #     text="⏳ Generating detailed feedback for each part of the mock test. Please wait, this may take a few minutes."
            # )
            # Create a background task for generating detailed feedback
            context.application.create_task(generate_mock_test_detailed_feedback(update, context))
            return  # Don't wait for the task to complete here

        elif query.data == f'{userID}mock_test_translate_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            context.application.create_task(translate_mock_test_overall_feedback(update, context))

        elif query.data == f'{userID}mock_test_translate_detailed_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            context.application.create_task(translate_mock_test_detailed_feedback(update, context))
        elif query.data in ["confirm_broadcast", "cancel_broadcast"]:
            await confirm_broadcast(update, context)
        elif query.data == "admin_broadcast":
        # Simulate the /broadcast command
            await query.edit_message_text("/broadcast")
            # return await broadcast(update, context)
                # Update your handler
        elif query.data == "admin_stats":
            await admin_stats(update, context)
        else:
            await query.message.reply_text("Invalid admin option.")
            text = "Invalid admin option."
            await show_main_menu(update, context, text)
    except Exception as e:
        text = ("🚨 button handler function ", e)
        await error_handling(update, context,text)
async def change_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allows the user to change their preferred language."""
    try:
        user_id = update.effective_user.id
        user_data = context.user_data.setdefault('user_data', {})
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        keyboard = [
            [InlineKeyboardButton("Arabic", callback_data=f'{userID}language2_Arabic'),
            InlineKeyboardButton("Urdu", callback_data=f'{userID}language2_Urdu')],
            [InlineKeyboardButton("Uzbek", callback_data=f'{userID}language2_Uzbek'),
            InlineKeyboardButton("Persian", callback_data=f'{userID}language2_Persian')],
            [InlineKeyboardButton("Hindi", callback_data=f'{userID}language2_Hindi'),
            InlineKeyboardButton("Other", callback_data=f'{userID}language2_other')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.effective_message.reply_text("Change your language", reply_markup=reply_markup)
        
        # Store the fact that the user is in the process of changing language
        user_data['changing_language'] = True
        
        text = "."
        await show_main_menu(update, context, text)
    except Exception as e:
        text = ("🚨 change language", e)
        await error_handling(update, context,text)

async def change_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allows the user to change their preferred voice."""
    try:
        user_data = context.user_data.setdefault('user_data', {})
        user_data['changing_voice'] = True

        # Send each voice sample as an audio message
        for voice_name, filename in voice_samples.items():
            file_path = os.path.join("examiners_voice", filename)
            with open(file_path, 'rb') as audio_file:
                await context.bot.send_audio(chat_id=update.effective_chat.id, audio=audio_file)
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        # Create inline keyboard for voice selection
        keyboard = [
            [InlineKeyboardButton("Dan", callback_data=f'{userID}voice_Dan')],
            [InlineKeyboardButton("William", callback_data=f'{userID}voice_William')],
            [InlineKeyboardButton("Scarlett", callback_data=f'{userID}voice_Scarlett')],
            [InlineKeyboardButton("Liv", callback_data=f'{userID}voice_Liv')],
            [InlineKeyboardButton("Amy", callback_data=f'{userID}voice_Amy')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text("Select your examiner voice:", reply_markup=reply_markup)
    except Exception as e:
        text = ("🚨 changing voice function ", e)
        await error_handling(update, context,text)
# Helper function to ask for test part
blocked_users = set()
async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, text):
    """Displays the main menu with options: Start Test, Show Progress, Contact Me."""
    print(f"{update.effective_user.id} Showing main menu")
    try:
        keyboard = [
            [KeyboardButton("Start Test")],
            [KeyboardButton("IELTS Writing 📝"),KeyboardButton("Show Progress")],
            [KeyboardButton("Contact Me"), KeyboardButton("Bot Channel")],
            [KeyboardButton("Settings")],
            # [KeyboardButton("Change language"), KeyboardButton("Change voice")],
        ]
        user_id = update.effective_user.id
        if user_id in ADMIN_IDS:
            keyboard.append([KeyboardButton("Admin")])
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.effective_message.reply_text(text, reply_markup=reply_markup)
        if user_id in blocked_users:
          blocked_users.remove(user_id)
    except Forbidden as e:
      if user_id not in blocked_users:
          username = update.effective_user.username
          print(f"ID: {user_id} Username: {username} | Bot was blocked by the user")
          blocked_users.add(user_id)
    except Exception as e:
        text = ("🚨 show main menu function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context,text)
async def stop_test(update: Update, context: ContextTypes.DEFAULT_TYPE, part):
    """Displays the stop test button."""
    print(f"{update.effective_user.id} Showing stop test button")
    try:
        keyboard = [
            [KeyboardButton("Stop the Test")],
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.effective_message.reply_text(part, reply_markup=reply_markup)
    except Exception as e:
        text = ("🚨 Stop test ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context,text)
async def append_speaking_score(update: Update, context: ContextTypes.DEFAULT_TYPE ,part_type, band_score):
    try:
        user_id = update.effective_user.id
        username = update.effective_user.username
        current_date = datetime.now().strftime('%d/%m/%Y %H:%M')
        # Convert the date to the correct format for timestamptz
        date = datetime.strptime(current_date, '%d/%m/%Y %H:%M').strftime('%Y-%m-%d %H:%M:%S')
        
        if part_type == "part1":
            score_col = "part1_score"
            date_col = "part1_date"
        elif part_type == "part2":
            score_col = "part2_score"
            date_col = "part2_date"
        elif part_type == "part3":
            score_col = "part3_score"
            date_col = "part3_date"
        elif part_type == "mock_test":
            score_col = "mock_test_score"
            date_col = "mock_test_date"
        else:
            return False  # Invalid part type

        # Insert a new record with the provided score and date
        supabase.table('ielts_speaking_scores').insert({
            "user_id": user_id,
            "username": username,
            score_col: band_score,
            date_col: date,
        }).execute()

        return True
    except Exception as e:
        print(f"{update.effective_user.id} 🚨 An error occurred in appending speaking score function : {e}")
        return False

async def show_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = update.effective_user.id
        scores_df = get_user_scores(user_id)
        if not scores_df.empty:
            charts = await generate_charts(scores_df)
            await display_charts(update, context, charts)
        else:
            # await update.message.reply_text("No scores found. Please practice the test to see your progress.")
            text = "No scores found. Please practice the test to see your progress."
            await show_main_menu(update, context, text)
    except Exception as e:
        text = ("🚨 show progress",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context,text)
def get_user_scores(user_id):
    try:
        # Query the ielts_speaking_scores table to get all records for the specified user_id
        response = supabase.table('ielts_speaking_scores').select('*').eq('user_id', user_id).execute()
        data = response.data
        
        # Convert the data to a DataFrame
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"🚨 Failed to retrieve data get user scores function: {e}")
        return pd.DataFrame()
async def generate_charts(scores_df):
    
    charts = []
    parts = ['part1', 'part2', 'part3', 'mock_test']
    now = datetime.now()

    # Define a default date range if no scores exist at all
    date_range = pd.date_range(start=now, periods=4, freq='D')

    for part in parts:
        part_score_column = f'{part}_score'
        part_date_column = f'{part}_date'
        
        if part_score_column in scores_df.columns and part_date_column in scores_df.columns:
            scores_df[part_date_column] = pd.to_datetime(scores_df[part_date_column])
            scores = scores_df[[part_date_column, part_score_column]].dropna()
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if not scores.empty:
                ax.scatter(scores[part_date_column], scores[part_score_column], color='blue')
                ax.set_xticks(scores[part_date_column])
                ax.set_xticklabels(scores[part_date_column].dt.strftime('%Y-%m-%d %H:%M'), rotation=45)
            else:
                # Use the date range similar to one of the parts or generate dates starting from the current date
                ax.scatter([], [])  # Empty scatter plot
                ax.set_xticks(date_range)
                ax.set_xticklabels(date_range.strftime('%Y-%m-%d %H:%M'), rotation=45)

            ax.set_title(f"{part.capitalize()} Scores", fontsize=16)
            ax.set_xlabel("Date of test", fontsize=14)
            ax.set_ylabel("Score", fontsize=14)
            ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.5)
            ax.set_ylim(4, 9)
            ax.set_yticks(np.arange(4, 9.5, 0.5))  # Include half scores
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            charts.append(buf)
            
            plt.close(fig)  # Close the figure to avoid memory leaks

    return charts
async def increment_practice_count(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = update.effective_user.id
        
        # Retrieve the current practice_count, attempts_remaining, and last_attempt_time for the user
        result = supabase.table('ielts_speaking_users').select('practice_count', 'attempts_remaining', 'last_attempt_time').eq('user_id', user_id).execute()
        
        if result.data:
            current_count = result.data[0]['practice_count']
            attempts_remaining = result.data[0]['attempts_remaining']
            last_attempt_time = result.data[0]['last_attempt_time']
            
            # Increment the practice_count by 1
            new_count = current_count + 1
            
            # Decrement the attempts_remaining by 1 if it is greater than 0
            new_attempts_remaining = attempts_remaining - 1 if attempts_remaining > 0 else 0
            
            # Update the last_attempt_time to the current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M")
            
            # Update the practice_count, attempts_remaining, and last_attempt_time in the Supabase database
            supabase.table('ielts_speaking_users').update({
                'practice_count': new_count,
                'attempts_remaining': new_attempts_remaining,
                'last_attempt_time': current_time
            }).eq('user_id', user_id).execute()
            
            # print(f"Practice count and attempts updated for user {user_id}: practice_count={new_count}, attempts_remaining={new_attempts_remaining}, last_attempt_time={current_time}")
        else:
            # User doesn't exist in the database, handle accordingly
            print(f"User {user_id} not found in the database")
    except Exception as e:
        print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} |🚨 increment practice count function ",e)
        # await update.message.reply_text(issue_message)
async def display_charts(update: Update, context: ContextTypes.DEFAULT_TYPE, charts: list):
    for chart in charts:
        chart.seek(0)  # Ensure the buffer is at the start
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=chart)
async def check_user_attempts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = update.effective_user.id
        user_data = context.user_data.setdefault('user_data', {})

        # Retrieve the user's last_attempt_time and attempts_remaining from the database
        result = supabase.table('ielts_speaking_users').select('last_attempt_time', 'attempts_remaining').eq('user_id', user_id).execute()
        
        if result.data:
            last_attempt_time_str = result.data[0]['last_attempt_time']
            attempts_remaining = result.data[0]['attempts_remaining']
            current_time = datetime.now()
            
            # print(f"Number of attempts left: {attempts_remaining}")
            user_data['remaining_attempts'] = attempts_remaining
            print(f"{update.effective_user.id} Remaining attempts: {user_data['remaining_attempts']}")
        else:
            # User doesn't exist in the database, allow the attempt
            user_data['remaining_attempts'] = 5  # Set a default value
            return True

    except Exception as e:
        print(f"{update.effective_user.id} 🚨 check user attempts function error: {e}")
        # In case of an error, we'll allow the attempt but won't update the database
        user_data['remaining_attempts'] = 5  # Set a default value in case of error
        return True

async def convert_text_to_audio(text, examiner_voice):

  def make_request(text, examiner_voice,api):
      if not text.strip():
          raise ValueError("🚨 Input text contains no characters.")
      
      if examiner_voice == "":
          examiner_voice = "Liv"
      
    #   unreal_speech_api = random.choice(unreal_speech_API_keys)
      response = requests.post(
          'https://api.v7.unrealspeech.com/stream',
          headers={
              'Authorization': f"Bearer {api}"
            #   'Authorization' : 'Bearer O1NloOK2SySJS72tnQ9ZaeRDYQPWecclkKgK6v1UmOsvfpsuSLDBTb'
          },
          json={
              'Text': text,
              'VoiceId': examiner_voice,
              'Bitrate': '64k',
              'Speed': '0',
              'Pitch': '1',
              'Codec': 'libmp3lame',
          }
      )
      
      if response.status_code == 200:
          # Save the audio content to a file
          with open('audio.oga', 'wb') as f:
              f.write(response.content)
          return 'audio.oga'
      else:
          raise Exception(f"🚨 Failed to convert text to audio. Status code: {response.status_code} API key: {unreal_speech_api}")

  try:
      print("convert text to audio")
      attempts = 0
      while attempts < 6:
          try:
              unreal_speech_api = random.choice(unreal_speech_API_keys)
              return make_request(text, examiner_voice, unreal_speech_api)
          except Exception as e:
              print(f"Attempt {attempts + 1} failed: {e}")
              attempts += 1
              continue
      try:
          return make_request(text, examiner_voice,"KtOIJsoP6mzYmu4WUJx3aGrEWMhEUdl4kUYAQPM7VOR08bQVcIXA7x")
      except Exception as e:
          print(f"Second attempt failed: {e}")
          raise Exception("🚨 Both attempts to convert text to audio failed.")
  except Exception as e:
      print(f"First attempt failed: {e}")
      print("convert text to audio second option")
      try:
          return make_request(text, examiner_voice,"KtOIJsoP6mzYmu4WUJx3aGrEWMhEUdl4kUYAQPM7VOR08bQVcIXA7x")
      except Exception as e:
          print(f"Second attempt failed: {e}")
          raise Exception("🚨 Both attempts to convert text to audio failed.")


# Function to convert audio to text using Deepgram STT API
async def convert_audio_to_text(file_id, update, context):
    # print("DeepGram version: ",deepgram.__version__)
    try:
        try:
            deppgarm_api = random.choice(deepgram_api_keys)
            deepgram_client = DeepgramClient(deppgarm_api)
            print(f"{update.effective_user.id} convert audio to text")
            file = await context.bot.get_file(file_id)
            
            file_path = file.file_path
            AUDIO_URL = {
                "url": file_path
            }
            
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                filler_words=True,
                numerals=False,
            )
            # print(f"File path: {file_path}")
            response = deepgram_client.listen.rest.v("1").transcribe_url(AUDIO_URL, options)
            transcript = response.to_json(indent=4)
            response_data = json.loads(transcript)
            transcript_text = response_data['results']['channels'][0]['alternatives'][0]['transcript']
            return transcript_text
        except Exception as e:
            deppgarm_api = random.choice(deepgram_api_keys)
            deepgram_client = DeepgramClient(deppgarm_api)
            print(f"{update.effective_user.id} convert audio to text second option")
            file = await context.bot.get_file(file_id)
            
            file_path = file.file_path
            AUDIO_URL = {
                "url": file_path
            }
            
            options = PrerecordedOptions(
                model="nova-2",
                smart_format=True,
                filler_words=True,
                numerals=False,
            )
            # print(f"File path: {file_path}")
            response = deepgram_client.listen.rest.v("1").transcribe_url(AUDIO_URL, options)
            transcript = response.to_json(indent=4)
            response_data = json.loads(transcript)
            transcript_text = response_data['results']['channels'][0]['alternatives'][0]['transcript']
            return transcript_text
    except Exception as e:
        text = ("🚨 convert audio to text function",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context,text)
def round_to_ielts_score(score):
    # print(score)
    if score % 1 < 0.25:
        return round(score // 1)
    elif score % 1 < 0.75:
        return round(score // 1) + 0.5
    else:
        return round(score // 1) + 1
def get_cefr_level(ielts_score):
    if ielts_score >= 9.0:
        return "C2"
    elif ielts_score >= 8.5:
        return "C2"
    elif ielts_score >= 8.0:
        return "C1+"
    elif ielts_score >= 7.5:
        return "C1"
    elif ielts_score >= 7.0:
        return "C1"
    elif ielts_score >= 6.5:
        return "B2"
    elif ielts_score >= 6.0:
        return "B2"
    elif ielts_score >= 5.5:
        return "B1"
    elif ielts_score >= 5.0:
        return "B1"
    elif ielts_score >= 4.5:
        return "A2"
    elif ielts_score >= 4.0:
        return "A1"
    else:
        return "A0"
async def generate_suggested_answer(question, previous_answer, part_type,context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})
        # print(previous_answer)
        prompt = f"Question: {question}\nuser Answer: {previous_answer}\nIELTS Speaking {part_type}\n\n"
        prompt += f"Provide a suggested response for the given IELTS speaking question type, ensuring that the answer is appropriate in length and complexity based on the part type [{part_type}] specified. If the part is part 1, the suggested answer should be simple and not too long. If it is part 3, the suggested answer should not be too short or too long. Please only provide the suggested answer without any additional content."
        
        
        # Use the LLM to generate the suggested answer
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model=groq_model,
        )
        result = chat_completion.choices[0].message.content
        print("suggetion generated")
        user_data['generate_suggested_answer'] = []
        user_data['generate_suggested_answer'] = result
        return user_data['generate_suggested_answer']
    except Exception as e:
        print("🚨 Groq error switching to perplexity generate_suggested_answer ",e)
        messages = [
                {
                    "role": "system",
                    "content": (
                        ""
                       
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                    ),
                },
            ]
        client = OpenAI(api_key=perplexity_API, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model="llama-3.1-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("feadback report generated")
        return result
async def send_long_message(update, context, message):
    try:
        num_mesages = len(message)
    except Exception as e:
        num_mesages = 2000
    try:
        max_length = 4096  # Maximum message length allowed by Telegram
        message_chunks = [message[i:i+max_length] for i in range(0, num_mesages, max_length)]
        
        for chunk in message_chunks:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk)
    except Exception as e:
        text = ("🚨 send long messages function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context,text)
async def generate_typical_answers(questions, answers,context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})
        prompt = "Provide typical answers for the following IELTS speaking questions and user answers:\n\n"
        for i in range(len(questions)):
            prompt += f"Question {i+1}: {questions[i]}\nUser Answer: {answers[i]}\n\n"
        
        prompt += "Provide a typical answer for each question based on the user's response. Each typical answer should be presented below the original answer. These responses are from a user who was answering the IELTS Speaking test, and the typical answers should be of high quality to help the user improve for the next time. Please only include the structured format of: 1- Question, 2- Answer, 3- Typical Answer. Do not include any other text except what has been specified. and  you  should make a new line between the question and the answers  and typical answers      "

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model=groq_model,
        )
        result = chat_completion.choices[0].message.content
        # Remove Markdown formatting characters using regular expressions
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        user_data['generate_typical_answers'] = []
        user_data['generate_typical_answers'] = result
        return user_data['generate_typical_answers']
    except Exception as e:
        print("🚨 Groq error switching to perplexity generate_typical_answers",e)
        messages = [
                {
                    "role": "system",
                    "content": (
                        ""
                       
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                    ),
                },
            ]
        client = OpenAI(api_key=perplexity_API, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model="llama-3.1-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        # print("feadback report generated")
        return result
async def display_feedback(update, context, overall_avg, pronunciation_avg, fluency_avg, grammar_avg, vocabulary_avg):
    try:
        # Create a radar chart for visualization
        labels = ['IELTS Score', 'Pronunciation', 'Fluency', 'Grammar', 'Vocabulary']
        scores = [overall_avg, pronunciation_avg, fluency_avg, grammar_avg, vocabulary_avg]
        
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        scores.append(scores[0])
        angles = np.append(angles, angles[0])
        labels.append(labels[0])  # Append the first label to close the radar chart
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # Define colors and shapes for each skill
        colors = ['#005C9E', '#ff7f0e', '#009F00', '#C50404', '#41007D']
        shapes = ['o', 'o', 'o', 'o', 'o']
        
        # Plot each skill with a different color and shape
        for i in range(len(labels) - 1):
            ax.plot(angles[i:i + 2], scores[i:i + 2], marker=shapes[i], linestyle='', markersize=6, color=colors[i], label=labels[i])
        
        ax.fill(angles, scores, alpha=0.25)  # Adjust the alpha value to set the desired transparency
        ax.set_thetagrids(angles * 180 / np.pi, labels)
        ax.set_ylim(0, 9)
        ax.grid(True)
        
        # Create a legend with the scores and shapes
        legend_elements = [Line2D([0], [0], color=colors[i], marker=shapes[i], linestyle='None', markersize=6, label=f"{labels[i]}: {scores[i]:.1f}") for i in range(len(labels) - 1)]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        # Save the plot as an image
        plt.savefig('feedback.png', bbox_inches='tight')
        
        # Send the image to the user
        with open('feedback.png', 'rb') as photo:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=photo)
    except Exception as e:
        text = ("🚨 dispaly feedback function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context,text)
def generate_pronunciation_visualization(answer_data):
    try:
        word_pronunciation_details = answer_data['word_pronunciation_details']
        
        colors = {
            'Correct Pronunciation +80/100': (22, 219, 101),
            'Slightly Incorrect Pronunciation +50/100': (255, 203, 5),
            'Incorrect Pronunciation -50/100': (216, 0, 50)
        }
        
        padding = 20
        max_line_width = 1200 - 2 * padding
        
        title_font = ImageFont.truetype("Roboto-Bold.ttf", 36)
        text_font = ImageFont.truetype("Roboto-Regular.ttf", 24)
        guide_font = ImageFont.truetype("Roboto-Regular.ttf", 18)
        
        dummy_image = Image.new('RGB', (1, 1), color='white')
        draw = ImageDraw.Draw(dummy_image)
        
        title = "Pronunciation Score"
        title_width, title_height = draw.textbbox((0, 0), title, font=title_font)[2:]
        
        # Calculate the height needed for the image
        y = padding + title_height + 20
        x = padding
        max_y = y
        
        for word_info in word_pronunciation_details:
            word = word_info['word']
            word_width, word_height = draw.textbbox((0, 0), word, font=text_font)[2:]
            
            if x + word_width > max_line_width:
                x = padding
                y += word_height + 5
            
            x += word_width + 10
            max_y = max(max_y, y + word_height)
        
        # Add space for the guide
        total_height = max_y + word_height + 50
        
        image_width = 1200
        image_height = total_height
        image = Image.new('RGB', (image_width, image_height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw title
        title_x = (image_width - title_width) // 2
        title_y = padding
        draw.text((title_x, title_y), title, font=title_font, fill=(0, 0, 0))
        
        # Draw words with color coding
        y = title_y + title_height + 20
        x = padding
        for word_info in word_pronunciation_details:
            word = word_info['word']
            score = float(word_info['pronunciation'])
            
            if score >= 80:
                color = colors['Correct Pronunciation +80/100']
            elif score >= 50:
                color = colors['Slightly Incorrect Pronunciation +50/100']
            else:
                color = colors['Incorrect Pronunciation -50/100']
            
            word_width, word_height = draw.textbbox((0, 0), word, font=text_font)[2:]
            
            if x + word_width > max_line_width:
                x = padding
                y += word_height + 5
            
            draw.text((x, y), word, font=text_font, fill=color)
            x += word_width + 10
        
        # Draw color guide
        guide_y = image_height - padding - 30
        guide_x = padding
        for color_name, color_code in colors.items():
            draw.rectangle((guide_x, guide_y, guide_x + 20, guide_y + 20), fill=color_code)
            draw.text((guide_x + 30, guide_y), color_name.capitalize(), font=guide_font, fill=(0, 0, 0))
            guide_x += draw.textbbox((0, 0), color_name.capitalize(), font=guide_font)[2] + 60
        
        image.save('pronunciation_visualization_with_padding.png')
    except Exception as e:
        print("🚨 generate pronunciation visualization function ", e)

    
async def generate_feedback_with_llm(prompt,context: ContextTypes.DEFAULT_TYPE):
    # Use the LLM to generate the detailed feedback based on the prompt
    # You can integrate with your chosen LLM service here
    # Example:
    try:
        user_data = context.user_data.setdefault('user_data', {})
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model=groq_model,
        )
        feedback = chat_completion.choices[0].message.content
        feedback = re.sub(r'\*', '', feedback)  # Remove asterisks (*)
        feedback = re.sub(r'#', '', feedback)  # Remove hash symbols (#)
        user_data['generate_feedback_with_llm'] = []
        user_data['generate_feedback_with_llm'] = feedback
        return user_data['generate_feedback_with_llm']
    except Exception as e:
        print("🚨 Groq error switching to perplexity generate_feedback_with_llm ",e)
        messages = [
                {
                    "role": "system",
                    "content": (
                        ""
                       
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                    ),
                },
            ]
        client = OpenAI(api_key=perplexity_API, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model="llama-3.1-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        feedback = (response.choices[0].message.content)
        feedback = re.sub(r'\*', '', feedback)  # Remove asterisks (*)
        feedback = re.sub(r'#', '', feedback)  # Remove hash symbols (#)
        user_data['generate_feedback_with_llm'] = []
        user_data['generate_feedback_with_llm'] = feedback
        return user_data['generate_feedback_with_llm']

# unreal_speech_api = random.choice(unreal_speech_API_keys)

async def convert_answer_to_audio(user_answer, speed,examiner_voice):
    # global  examiner_voice
    # user_data = context.user_data.setdefault('user_data', {})
    # print(user_data['examiner_voice'])
     
    def make_request(user_answer, examiner_voice,api):
        
        if not user_answer.strip():
          raise ValueError("🚨 Input text contains no characters.")
        if examiner_voice== "":
            examiner_voice = "Liv" 
        
        
        response = requests.post(
            'https://api.v7.unrealspeech.com/stream',
            headers={
                'Authorization': f"Bearer {api}"
            },
            json={
                'Text': user_answer,
                'VoiceId': examiner_voice,
                'Bitrate': '64k',
                'Speed': speed,
                'Pitch': '1',
                'Codec': 'libmp3lame',
            }
        )
        if response.status_code == 200:
            # Generate a unique filename for the audio file
            audio_filename = f"user_audio_{int(time.time())}.mp3"
            # Save the audio content to a file
            with open(audio_filename, 'wb') as f:
                f.write(response.content)
            return audio_filename
        else:
            print(f"🚨 Error converting answer to audio. Status code: {response.status_code} API key: {unreal_speech_api}")
            raise Exception(f"Error converting answer to audio. Status code: {response.status_code} API key: {unreal_speech_api}")
            # return ""  # Return an empty string instead of None
    try:
      print("convert answer to audio")
      attempts = 0
      while attempts < 6:
          try:
              unreal_speech_api = random.choice(unreal_speech_API_keys)
              return make_request(user_answer, examiner_voice, unreal_speech_api)
          except Exception as e:
              print(f"Attempt {attempts + 1} failed: {e}")
              attempts += 1
              continue
      print(f"First attempt failed: {e}")
      print("convert answer to audio second option")
      try:
          return make_request(user_answer, examiner_voice,"KtOIJsoP6mzYmu4WUJx3aGrEWMhEUdl4kUYAQPM7VOR08bQVcIXA7x")
      except Exception as e:
          print(f"Second attempt failed: {e}")
          raise Exception("🚨 Both attempts to convert answer to audio failed.")
    except Exception as e:
        print(f"First attempt failed: {e}")
        print("convert answer to audio second option")
        try:
            return make_request(user_answer, examiner_voice,"KtOIJsoP6mzYmu4WUJx3aGrEWMhEUdl4kUYAQPM7VOR08bQVcIXA7x")
        except Exception as e:
            print(f"Second attempt failed: {e}")
            raise Exception("🚨 Both attempts to convert answer to audio failed.")
        # except Exception as e:
        #     # global  examiner_voice
        #     if examiner_voice== "":
        #             examiner_voice = "Liv" 

            
        #     response = requests.post(
        #             'https://api.v7.unrealspeech.com/stream',
        #             headers={
        #                 'Authorization': f"Bearer {unreal_speech_api}" 
        #             },
        #             json={
        #                 'Text': user_answer,
        #                 'VoiceId': examiner_voice,
        #                 'Bitrate': '64k',
        #                 'Speed': speed,
        #                 'Pitch': '1',
        #                 'Codec': 'libmp3lame',
        #             }
        #         )

        #     if response.status_code == 200:
        #             # Generate a unique filename for the audio file
        #         audio_filename = f"user_audio_{int(time.time())}.mp3"

        #             # Save the audio content to a file
        #         with open(audio_filename, 'wb') as f:
        #             f.write(response.content)

        #         return audio_filename
        #     else:
        #         print(f"🚨 Error converting answer to audio. Status code: {response.status_code} API key: {unreal_speech_api}")
        #         return ""  # Return an empty string instead of None


async def translate_feedback(user_id, feedback, update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Retrieve the user's native language from the database
        user = supabase.table('ielts_speaking_users').select('native_language').eq('user_id', user_id).execute()
        user_data = context.user_data.setdefault('user_data', {})
        user_data['native_language'] = user.data[0]['native_language']
        native_language = user_data['native_language']
        print(f"{update.effective_user.id} language ", native_language)
        
        prompt = f"""
            Translate the provided IELTS evaluation of speaking test text from english into {native_language}. Ensure that the translation is accurate, contextually appropriate to the translated language, and adheres to the linguistic standards of {native_language}.
            Instructions:
            1- Content Focus: Only include the evaluation text. Exclude any non-evaluative content to maintain the focus on the assessment aspects of the text.
            2- Rearrange the text to align with the typical format and flow of {native_language}, while preserving the original order and organization of content.
            3- Language Specifics:
            - You should translate based on the required context. If the context requires any word or sentence to remain in English, leave it in English for grammar or pronunciation or any place in the text it always be between two ("") and make them inside the qoutes (""). Be cautious when you encounter this.
            - Adjust the sentence structure and phrasing to fit the grammatical and stylistic norms of {native_language}, ensuring that the translation reads naturally to native speakers and also cosider using the proper meaning.
            4- Accuracy and Contextual Integrity:
            -Carefully maintain the original context and meaning of the evaluation text during translation.
            - Ensure that all translated terms and phrases are appropriate for the context and do not alter the evaluative tone or content. 
            
            the evaluation text that needed to translates is:\n\n
            {feedback}
        """
        
        max_retries = 3
        retry_count = 0
        
        if native_language:
            client = AsyncOpenAI(
                base_url="https://api.unify.ai/v0/",
                api_key=unify_API
            )

            while retry_count < max_retries:
                try:
                    response = await client.chat.completions.create(
                        model="gpt-4o@openai",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=4096,
                        temperature=0.0,
                        stream=False
                    )
                    
                    translated_feedback = response.choices[0].message.content
                    translated_feedback = re.sub(r'\*', '', translated_feedback)
                    translated_feedback = re.sub(r'#', '', translated_feedback)
                    user_data['translate_feedback'] = translated_feedback
                    return user_data['translate_feedback']
                except Exception as e:  
                    retry_count += 1
                    print(f"{update.effective_user.id} 🚨 Translation error occurred: {str(e)}. Retrying ({retry_count}/{max_retries})...")
            
            # If all retries fail
            keyboard = [
                [InlineKeyboardButton("End the Test", callback_data=f'{user_id}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, the translation service is currently unavailable. Please try again later.", reply_markup=reply_markup)
        else:
            return None
    except Exception as e:
        text = ("🚨 translate feedback", e)
        await error_handling(update, context, text)


def is_valid_gmail(email):
    try:
        # print("is valid gmail")
        gmail_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@gmail\.com$')
        print("is valid gmail")
        return gmail_regex.match(email) is not None
    except Exception as e:
        print("is valid gmail function",e)
        return True
        # await update.message.reply_text(issue_message)

# def is_real_gmail(email):
#     print("start is real gmail")
#     try:
        
#         api_keys = [
#            "5d64616c34d64b4e98f2647a29648a53",
#             "ff1552bbc5dd4dbc87e5c85645db1cb7",
#             "e9f36cc57580421184a1bc62fd297fb0",
#             # "043fbf429a1a43d09be00d0f74b67f80",
#         ]
#         retries = 3
#         while retries > 0:
#             api_key = random.choice(api_keys)
#             print(api_key)
#             try:
#                 print("check is real email")
#                 url = "https://emailvalidation.abstractapi.com/v1"
#                 querystring = {"api_key": api_key, "email": email}
#                 print("querystring")
#                 print(querystring)
#                 response = requests.get(url, params=querystring)
#                 print("response")
#                 print(response)
#                 data = response.json()
#                 print("data")
#                 print(data)
#                 deliverability = data.get("deliverability")
#                 print("deliverability")
#                 print(deliverability)
#                 if deliverability == "DELIVERABLE":
#                     print("DELIVERABLE")
#                     return True
#                 elif deliverability == "UNDELIVERABLE":
#                     print("UNDELIVERABLE")
#                     return False
#             except Exception as e:
#                 print(f"Error verifying email with API key {api_key}: {e}")
#                 retries -= 1
#         print("is real gmail")
#         return True
#     except Exception as e:
#         print("🚨is_real_gmail(email): ",e)
#         return True
#         # await update.message.reply_text(issue_message)

def is_real_gmail(email):
  print("start is real gmail")
  api_keys = [
      "si7Zcxl8VhGBXBiJg2Q8PBxhxX35OBT49U0yPuu",
    "5OH007HvEZ5JmWNVrPXsfmzguN4H3KA0sn5j2KS",
    "HRTHKnq3c8NCDOHDd6sfLtTbjbo6uHHskdDcPhP",
    "ojDRXqsv10BKA6dAAYaJDD5FjXesaGF9SZUN5IP",
  ]
  retries = 3
  
  while retries > 0:
      api_key = random.choice(api_keys)
      print(f"Trying API key: {api_key}")
      try:
          print("Checking if email is real")
          url = "https://api.usebouncer.com/v1.1/email/verify"
          querystring = {"email":email}
          headers = {"x-api-key": api_key}
          print("querystring:", querystring)
          
        #   response = requests.get(url, params=querystring)
          response = requests.request("GET", url, headers=headers, params=querystring)
          print("Response status code:", response.status_code)
          
          data = response.json()
          print("Response data:", data)
          
          if response.status_code != 200:
              raise requests.RequestException(f"API returned status code {response.status_code}")
          
          deliverability = data.get("status")
          print("Deliverability:", deliverability)
          
          if deliverability == "deliverable":
              print("Email is DELIVERABLE")
              return True
          elif deliverability == "undeliverable":
              print("Email is UNDELIVERABLE")
              return False
          else:
              print(f"Unexpected deliverability status: {deliverability}")
              raise ValueError(f"Unexpected deliverability status: {deliverability}")
      
      except (requests.RequestException, ValueError) as e:
          print(f"Error verifying email with API key {api_key}: {e}")
          retries -= 1
      
      except Exception as e:
          print(f"Unexpected error: {e}")
          retries -= 1
  
  print("All retries exhausted")
  return True  # Default to True if all retries fail
#------------------------ Part 1---------------------------- 
# Helper function to ask for Part 1 topic
async def ask_part_1_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await score_voice(update, context)
        print(f"{update.effective_user.id} ask part 1 topic")
        user_data = context.user_data.setdefault('user_data', {})
        
        user_data['part_1_topics'] = [
            "Study 📚",
            "Work 💼",
            "Hometown 🏡",
            "Home/ Accommodation 🏘️",
            "Family 👨‍👩‍👧‍👦",
            "Friends 👥",
            "Clothes 👕",
            "Fashion 👗",
            "Gifts 🎁",
            "Daily routine 📅",
            "Daily activities 🏃‍♂️",
            "Food/ Cooking 🍳",
            "Drinks 🥤",
            "Going out 🎉",
            "Hobbies 🎨",
            "Language 🌐",
            "Leisure time activity ⏰",
            "Sports ⚽",
            "Future plan 🔮",
            "Music 🎵",
            "Newspapers 📰",
            "Pets 🐾",
            "Flowers & Plants 🌸",
            "Reading 📖",
            "Dancing 💃",
            "Exercise 💪",
            "Shopping 🛍️",
            "Magazines & TV 📺",
            "Travelling ✈️",
            "Interesting places 🏰",
            "Bicycle 🚲",
            "Seasons 🍂",
            "Maps 🗺️",
            "Internet & Technology 💻",
            "Weather ☀️",
            "Festivals 🎆",
            "Culture/ Tradition 🎭"
        ]
        
        topics_message = "Please select the topic you want to practice \n(write the number of the topic):\n\n"
        topics_message += "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(user_data['part_1_topics'])])
        
        user_data['part_1_topic_selection'] = True
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        # await update.effective_message.reply_text(topics_message, reply_markup=InlineKeyboardMarkup([
        #     [InlineKeyboardButton("Random", callback_data=f"{userID}topic_random")]
        # ]))
        message = await update.effective_message.reply_text(
            topics_message,
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("Random", callback_data=f"{userID}topic_random")]])
        )
        user_data['topic_message_id'] = message.message_id
    except Exception as e:
        text = ("🚨 ask part 1 topic function ", e)
        await error_handling(update, context,text)
async def ask_test_part(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"{update.effective_user.id} ask test part")
    try:
        keyboard = [
            [KeyboardButton("Part 1"), KeyboardButton("Part 2")],
            [KeyboardButton("Part 3"), KeyboardButton("Mock Test (Full test)")],
            [KeyboardButton("Main menu")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.effective_message.reply_text("Which part of the IELTS Speaking test would you like to practice today?", reply_markup=reply_markup)
    except Exception as e:
        text = ("🚨 ask task part ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context,text)
async def generate_and_ask_questions(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    try:
        print(f"{update.effective_user.id} generate and ask questions")
        user_data = context.user_data.setdefault('user_data', {})
        
        # Use Groq API to generate questions
        questions = await generate_questions(topic,context)
        if not questions:
            text = "🚨 there is no questions have been generated | questions = await generate_questions(topic,context)"
            await error_handling(update, context,text)
            return
        
        v_topic = topic 
        # Get the vocabulary list for the selected topic
        vocabularies = topic_vocabularies.get(v_topic, [])
        
        # Send the vocabulary list to the user
        if vocabularies:
            vocabulary_message = "Here are some vocabularies you can use in your speaking:\n" + ", ".join(vocabularies)
            await update.effective_message.reply_text(vocabulary_message)
        
        user_data['questions'] = questions
        # user_data['answers_list'] = []
        user_data['answers'] = {}
        user_data['current_question_index'] = 0
        
        await update.effective_message.reply_text(
            f"IELTS Speaking Part 1:\n\nNow, we will begin Part 1 of the IELTS Speaking test. "
            f"In this part, I will ask you ({len(questions)}) questions including general questions about yourself, "
            f"your home, or your job, etc., as well as questions on the topic of {topic[:-2]}.\n\n"
            f"Please answer each question in 15 to 40 seconds.\n\nLet's start"
        )
        
        await ask_current_question(update, context)
    except Exception as e:
        text = ("🚨 generate and ask questions function", e)
        await error_handling(update, context,text)
# Helper function to ask the current question
async def ask_current_question(update: Update, context: ContextTypes.DEFAULT_TYPE, retry=False):
    try:
        print(f"{update.effective_user.id} ask current question")
        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        current_question_index = user_data.get('current_question_index', 0)
        
        if retry:
            # Provide only the "Suggest Answer" option
            user_data = context.user_data['user_data']
            user_id = update.effective_user.id
            user_data['user_id'] = str(user_id)
            userID = user_data['user_id']
            keyboard = [
                [InlineKeyboardButton("Suggest Answer", callback_data=f'{userID}suggest_answer')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.effective_message.reply_text("Please re-answer the question.", reply_markup=reply_markup)
        else:
            questions_list = user_data.get('questions', [])
            # answers_list = user_data.get('answers', {})
            answers_list = user_data['answers_list']
            # print(answers_list)
            # answers_list = user_data.get('answers', {})
            list_previous_questions = user_data.setdefault('list_previous_questions', [])
            list_previous_answers = user_data.setdefault('list_previous_answers', [])
            
            if current_question_index < len(questions_list):
                previous_question = questions_list[current_question_index - 1] if current_question_index > 0 else ""
                current_question = questions_list[current_question_index].strip()
                # print("1")
                # user_answer = answers_list.get(current_question_index - 1, "")
                user_answer = answers_list[current_question_index - 1]  if current_question_index>0 else ""
                selected_topic = user_data.get('selected_topic', '')
                # print("user_answer", user_answer,"\n\nselected_topic:",selected_topic)
                list_previous_questions.append(previous_question)
                list_previous_answers.append(user_answer)
                # print("list_of_pre_answers",list_previous_answers )
                # Generate an interactive question based on the previous question, user's answer, and the current question
                interactive_question = await generate_interactive_question(previous_question, user_answer, current_question, selected_topic, list_previous_questions, list_previous_answers, context)
                
                if interactive_question:
                    # Replace the original question in the questions_list with the generated interactive question
                    questions_list[current_question_index] = interactive_question
                    
                    user_data['current_question'] = interactive_question
                    
                    # Convert question to audio using Deepgram TTS API
                    try:
                        await update.effective_message.reply_text(interactive_question)
                        
                        audio_file_path = await convert_text_to_audio(interactive_question,examiner_voice)
                        
                        with open(audio_file_path, 'rb') as audio:
                            await update.effective_message.reply_voice(voice=audio)
                        await update.effective_message.reply_text("Please record your answer.")
                        user_data['answering_question'] = True
                        print(f"{update.effective_user.id} Set answering_question to True")
                    except Exception as e:
                        print(f"{update.effective_user.id} 🚨 Error converting text to audio: {e}")
                        # Retry the conversion without sending an error message
                        await update.effective_message.reply_text("Please record your answer.")
                        user_data['answering_question'] = True
                        print(f"{update.effective_user.id} Set answering_question to True")
                else:
                    current_question_index += 1
                    user_data['current_question_index'] = current_question_index
                    await ask_current_question(update, context)
            else:
                await show_results(update, context)
    except Exception as e:
        text = ("🚨 ask current question function ", e)
        await error_handling(update, context,text)
# Helper function to move to the next question
async def move_to_next_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print(f"{update.effective_user.id} move to next question")
        user_data = context.user_data.setdefault('user_data', {})
        
        current_question_index = user_data.get('current_question_index', 0)
        questions_list = user_data.get('questions', [])
        
        if current_question_index < len(questions_list) - 1:
            user_data['current_question_index'] = current_question_index + 1
            await ask_current_question(update, context)
        else:
            await show_results(update, context)
    except Exception as e:
        text = ("🚨 move to next question function ", e)
        await error_handling(update, context,text)
# Helper function to show results
async def show_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print(f"{update.effective_user.id} show results")
        user_data = context.user_data.setdefault('user_data', {})
        
        questions_list = user_data.get('questions', [])
        answers_list = user_data.get('answers_list', {})
        voice_urls = user_data.get('voice_urls', [])
        
        # # Format the questions and answers into a single message
        # result_message = "Here are your questions and answers:\n\n"
        # for i in range(len(questions_list)):
        #     question = questions_list[i]
        #     # answer = answers_list.get(i, "No answer provided")
        #     answer = answers_list[i]
        #     result_message += f"Question: {question}\n\nAnswer: {answer}\n\n"
        
        # await send_long_message(update, context, result_message)
        typical_answers = await generate_typical_answers(questions_list, answers_list,context)
        await send_long_message(update, context, f"Typical Answers:\n\n{typical_answers}")
        # Prompt the user to continue to Part 2, retake Part 1, or end the test
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        await update.effective_message.reply_text("Please select one of these options.", reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("Continue to Part 2", callback_data=f'{userID}continue_part_2')],
            [InlineKeyboardButton("Retake Part 1", callback_data=f'{userID}retake_part_1')],
            [InlineKeyboardButton("Check Your Score ✅", callback_data=f'{userID}show_results')],
            [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
        ]))
    except Exception as e:
        text = ("🚨 show results function ", e)
        await error_handling(update, context,text)
# Function to generate questions using Groq API
async def generate_questions(topic,context: ContextTypes.DEFAULT_TYPE):
    try:
        print("generate questions")
        user_data = context.user_data.setdefault('user_data', {})
        
        prompt = f"You are an IELTS Speaking examiner and now you are testing an IELTS candidate. This is Part 1 of the test, and you need to ask between 5 to 6 questions about this topic: {topic}. You should only ask questions and nothing else. First, ask 2 or 3 general questions about the candidate, followed by 4 or 5 questions on the given topic. Ensure your questions are exactly like those a real examiner would ask. Number each question."
        
        # Try generating questions using Groq API
        try:
            chat_completion = groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": ""},
                    {"role": "user", "content": prompt}
                ],
                model=groq_model,
            )
            result = chat_completion.choices[0].message.content
        except Exception as e:
            print("🚨 Groq error switching to Perplexity generate_questions", e)
            
            # Fallback to Perplexity API
            messages = [
                {
                    "role": "system",
                    "content": "",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            client = OpenAI(api_key=perplexity_API, base_url="https://api.perplexity.ai")
            response = client.chat.completions.create(
                model="llama-3.1-70b-instruct",
                messages=messages
            )
            result = response.choices[0].message.content
        
        questions = result.split('\n')
        valid_questions = [q.strip() for q in questions if re.match(r'^\d+\.\s', q.strip())]
        
        user_data['questions'] = valid_questions
        
        print("number of questions: ", len(valid_questions))
        return user_data['questions']
    except Exception as e:
        print("🚨 generate questions function ", e)
        # await error_handling(update, context,text)
        return []
        # await update.message.reply_text(issue_message)
async def generate_interactive_question(previous_question, user_answer, next_question, selected_topic, pre_questions, pre_answers,context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})
        # prompt = f"Previous Question: {previous_question}\nUser's Answer: {user_answer}\nNext Question: {next_question}\n\n"
        # prompt += f"Selected Topic for part1: {selected_topic}\n\n"
        # prompt += f"Based on the user's answer to the previous question, generate a more relevant and context-aware question that is related to the topic and the user's response (these quetions is part of IELTS Speaking part 1 and your quetios shold be simple and not complex to help the ielts candidate asnwer properly). If the next question is on a different topic, include a transitional phrase to smoothly move to the new topic. Provide only the modified question and the number of the quetion without any additional text (in the first quetion you might will not recieve any asnwer or previvous question or any text so just ask the same quetion again also your quetions should be simple and relevent to the topic {selected_topic} when the quetions refers to move to the topic note that first questios are about the IELTS Candidate )."
        # print(pre_questions)
        # print(pre_answers)
        # print(user_answer)
        # print(previous_question)
        prompt= f"""
                Previous Question: {previous_question}
                User's Answer: {user_answer}
                Next Question: {next_question}

                Selected Topic for part1: {selected_topic}

                First, ask general questions about the IELTS candidate to start the interaction. Then, based on the user's answer to the previous question, generate a more relevant and context-aware question related to the topic and the user's response. These questions are part of IELTS Speaking part 1, and your questions should be simple and not complex to help the IELTS candidate answer properly. If the next question is on a different topic, include a transitional phrase to smoothly move to the new topic for example (now let us talk  about....  Note: you can use different phrases ). Provide only the modified question and the number of the question without any additional text. In the first question, you might not receive any answer or previous question or any text, so just ask the same question again. Also, your questions should be simple and relevant to the topic {selected_topic} when the questions refer to moving to the topic. Note that the first questions are general quetions about the IELTS Candidate. another note only ask the questions without any introduction
                and to help you not repeating questions here are the list of previous {pre_questions} also thses are the previous answers that the user gave {pre_answers} this will help you not ask quetions that thae user has already answered from the previous questions.
                """
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model=groq_model,
        )
        result = chat_completion.choices[0].message.content
        user_data['generate_interactive_question'] = []
        user_data['generate_interactive_question'] = result.strip()
        # print("user_data['generate_interactive_question'] ", user_data['generate_interactive_question'])
        return user_data['generate_interactive_question']
    except Exception as e:
        print("🚨 Groq error switching to perplexity generate_interactive_question",e)
        messages = [
                {
                    "role": "system",
                    "content": (
                        ""
                       
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                    ),
                },
            ]
        client = OpenAI(api_key=perplexity_API, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model="llama-3.1-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        # result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        # result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        # print("feadback report generated")
        user_data['generate_interactive_question'] = []
        user_data['generate_interactive_question'] = result.strip()
        return user_data['generate_interactive_question']
# Function to convert text to audio using Deepgram TTS API
async def generate_feedback(scores_list, questions, answers, overall_avg,targeted_score,context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})
        # print("questions ", questions)
        # print("answers ",answers)
        prompt = "Provide detailed feedback for the following IELTS speaking assessment for Part 1:\n\n"
        for i in range(len(questions)):
            prompt += f"Question {i+1}: {questions[i]}\n"
            prompt += f"Transcription of user answer: {answers[i]}\n\n"
            prompt += f"Pronunciation score: {scores_list[i]['pronunciation']}\n"
            prompt += f"Fluency score: {scores_list[i]['fluency']}\n"
            prompt += f"Grammar score: {scores_list[i]['grammar']}\n"
            prompt += f"Vocabulary score: {scores_list[i]['vocabulary']}\n"
            # prompt += f"Fillers: {scores_list[i]['pause_filler']}\n"
            # prompt += "Sentences of the answer (here is a detailed summary of the sentences in the answer):\n"
            # for sentence in scores_list[i]['sentences']:
            #     prompt += f"  - Sentence: {sentence['sentence']}\n"
            #     prompt += f"    Pronunciation of the sentence: {sentence['pronunciation']}\n"
            #     prompt += f"    Grammar Correction if mistakes exist: {sentence['grammar']}\n"
            # prompt += "\n"
        # prompt += f"Provide an overall feedback report on the user's performance, including pronunciation, fluency, grammar, and vocabulary etc. and here are the ielts speaking band score of part 1 {overall_avg} you should provide a clear and good feedback to the IELTS Candidate he expects to kmow his mistakes and improve next time and  best tips and suggetions to  get hisa imed score {targeted_score}"
        prompt += f"""
            Please provide a detailed feedback report on the user's performance in Part 1 of the IELTS speaking test, covering aspects such as pronunciation, fluency and coherence, grammatical range and accuracy, lexical resource (vocabulary), interactive communication, and overall performance. Offer specific examples, constructive feedback, and practical suggestions for each aspect to help the candidate identify their strengths and weaknesses. Emphasize the importance of continuous practice, active listening, and critical thinking to enhance their performance and provide language learning strategies. Encourage the candidate to reflect on their performance and set achievable goals for their IELTS speaking development. Your feedback should be supportive and encouraging, aiming to motivate the candidate in their IELTS speaking journey. Organize the feedback in a clear and structured manner, using headings and bullet points for easy readability and do not  write any non-needed text.

            1. Pronunciation:
            - Clarity and intelligibility
            - Stress, rhythm, and intonation
            - Individual sounds and phonemes
            - Areas for improvement and specific examples

            2. Fluency and Coherence:
            - Smooth flow of speech
            - Hesitation, repetition, and self-correction
            - Logical sequencing of ideas
            - Cohesive devices and discourse markers
            - Suggestions for enhancing fluency

            3. Lexical Resource (Vocabulary):
            - Range and variety of vocabulary
            - Accuracy and appropriacy of word choice
            - Idiomatic language and collocation
            - Recommendations for expanding vocabulary

            4. Grammatical Range and Accuracy:
            - Variety and complexity of grammatical structures
            - Accuracy and control of grammar
            - Errors and their impact on clarity
            - Tips for improving grammatical accuracy

            5. Overall Performance:
            - Here is the list of all overall scores of speaking skills with the overall IELTS band score of Part 1 {overall_avg} 
            - Strengths and weaknesses
            - Comparison to the target score of {targeted_score}
            - Actionable steps to bridge the gap and achieve the desired score

            Please provide specific examples, constructive feedback, and practical suggestions throughout the report to help the candidate identify areas for improvement and work towards their target score. Use a friendly and encouraging tone to motivate the candidate in their IELTS speaking journey.
            """
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model=groq_model,
        )
        result = chat_completion.choices[0].message.content
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("feadback report generated")
        user_data['generate_feedback'] = []
        user_data['generate_feedback'] = result
        return user_data['generate_feedback']
    except Exception as e:
        print("🚨 Groq error switching to perplexity generate_feedback ",e)
        messages = [
                {
                    "role": "system",
                    "content": (
                        ""
                       
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                    ),
                },
            ]
        client = OpenAI(api_key=perplexity_API, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model="llama-3.1-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("feadback report generated")
        user_data['generate_feedback'] = []
        user_data['generate_feedback'] = result
        return user_data['generate_feedback']
async def show_results_part1(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print(f"{update.effective_user.id} -----------------------FEEDBACK PART 1------------------------")
        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        voice_urls = user_data.get('voice_urls', [])
        targeted_score = user_data.get('targeted_score', 7)
        questions_list = user_data.get('questions', [])
        answers_list = user_data.get('answers_list', {})
        one_minute = user_data['part_1_minute_part_1']

        print(f"{update.effective_user.id} user_targetes_score: ", targeted_score)

        # Send the sticker and waiting message
        animated_emoji = "⏳"  # Hourglass emoji
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)

        # Send the initial waiting message with 0% progress and an empty progress bar
        progress_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Wait a few minutes until results are ready...\n\n[                             ] 0%"
        )

        # Send a message asking the user to share the bot
        share_message = (
            f"Discover this IELTS Speaking Bot! It simulates the IELTS speaking test and provides detailed feedback "
            f"about your speaking skills and estimated IELTS band score to help you improve. Try it for free now: "
            f"https://t.me/ielts_speakingAI_bot"
        )
        keyboard = [
            [InlineKeyboardButton("Share the Bot", switch_inline_query=share_message)]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await ask_channel(update, context)
        share_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="While waiting for the results, would you like to share this bot? 😊",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        total_steps = len(questions_list)   # +6 for additional processing steps
        current_step = 0

        def update_progress():
            nonlocal current_step
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_bar = "[" + "█" * (progress // 5) + " " * (20 - (progress // 5)) + "]"
            return progress, progress_bar

        user_data['scores_list1'] = []
        scores_list = user_data['scores_list1']

        for i in range(len(questions_list)):
            audio_url = voice_urls[i]
            question_prompt = questions_list[i]
            print("one_minute: ",one_minute[i])
            user_data['part_1_minute'] = one_minute[i]
            one_minute1 = one_minute[i]
            task_type = "ielts_part1"

            processed_scores = await assess_speech_async(audio_url, question_prompt, task_type, context,one_minute1)
            # print(processed_scores)
            if processed_scores:
                scores_list.append(processed_scores)
                print(f"{update.effective_user.id} Assessment successful for answer {i + 1}")
            else:
                print(f"🚨ID: {update.effective_user.id} Username: {update.effective_user.username} Assessment failed for answer {i + 1}")
                print(f"Voice URL: {audio_url}")
                print(f"Question: {question_prompt}")

            # Update progress
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Wait a few minutes until results are ready...\n{progress_bar} {progress}%"
            )

        if scores_list:
            # Delete the waiting message (sticker)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)

            # Calculate average scores
            overall_avg = sum(score.get("overall", 0) for score in scores_list) / len(scores_list)
            pronunciation_avg = sum(score.get("pronunciation", 0) for score in scores_list) / len(scores_list)
            fluency_avg = sum(score.get("fluency", 0) for score in scores_list) / len(scores_list)
            grammar_avg = sum(score.get("grammar", 0) for score in scores_list) / len(scores_list)
            vocabulary_avg = sum(score.get("vocabulary", 0) for score in scores_list) / len(scores_list)

            # Round the scores to the nearest 0.5
            overall_avg = round_to_ielts_score(overall_avg)
            pronunciation_avg = round_to_ielts_score(pronunciation_avg)
            fluency_avg = round_to_ielts_score(fluency_avg)
            grammar_avg = round_to_ielts_score(grammar_avg)
            vocabulary_avg = round_to_ielts_score(vocabulary_avg)

            overall_scores = {
                "pronunciation": pronunciation_avg,
                "fluency": fluency_avg,
                "grammar": grammar_avg,
                "vocabulary": vocabulary_avg,
                "IELTS band score": overall_avg,
            }

            # Generate feedback report
            feedback_report = await generate_feedback(scores_list, questions_list, answers_list, overall_scores, targeted_score,context)
            user_data['overall_feedback'] = feedback_report

            # Update progress
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Wait a few minutes until results are ready...\n{progress_bar} 100%"
            )

            # Delete the waiting message, share message, and progress message
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=progress_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=share_message.message_id)

            # Send feedback report
            await send_long_message(update, context, feedback_report)
            try:
                await send_recommendations(update, context, feedback_report)
            except Exception as e:
                print(e)
            # Display feedback visualization
            await display_feedback(update, context, overall_avg, pronunciation_avg, fluency_avg, grammar_avg, vocabulary_avg)

            # Send the band score as text
            band_score = f"Your estimated IELTS band score is: {overall_avg:.1f}"
            await context.bot.send_message(chat_id=update.effective_chat.id, text=band_score)

            # Get the CEFR level based on the IELTS score
            cefr_level = get_cefr_level(overall_avg)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Your language level is: {cefr_level}")

            await append_speaking_score(update, context, "part1", overall_avg)
            await increment_practice_count(update, context)
            userID = user_data.get('user_id', '')
            
            # Provide user options
            keyboard = [
                [InlineKeyboardButton("Continue to Part 2", callback_data=f'{userID}continue_part_2')],
                [InlineKeyboardButton("See Detailed Results", callback_data=f'{userID}detailed_results')],
                [InlineKeyboardButton("Translate", callback_data=f'{userID}translate_overall_feedback')],
                [InlineKeyboardButton("Retake Part 1", callback_data=f'{userID}retake_part_1')],
                [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?",
                                        reply_markup=reply_markup)

        else:
            text = ("🚨 Failed to assess the answers. Please try again. show results part 1")
            await error_handling(update, context,text)
    except Exception as e:
        text = ("🚨 show results part 1 function ", e)
        await error_handling(update, context,text)

async def assess_speech_async(audio_url, question_prompt, task_type, context: ContextTypes.DEFAULT_TYPE,one_minute1):
    user_data = context.user_data.setdefault('user_data', {})
    try:
        # Download the voice file from the URL
        response = requests.get(audio_url)
        
        filename = f"voice_{int(time.time())}.oga"
        with open(filename, "wb") as file:
            file.write(response.content)

        # loop = asyncio.get_running_loop()
        # print("user_data['part_1_minute']: ",one_minute1)
        # print("user_data['part_1_minute_part_1']: ",user_data['part_1_minute_part_1'])
        # if one_minute1:  
        #     print("use speech super API (part 1)")
        #     scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompt, task_type)
        # else:
        #     try:
        #         scores, analysis_data = await loop.run_in_executor(executor, assess_speech, filename, question_prompt)
        #     except Exception as e:
        #         print("🚨Error on Speech Ace API and switching to Speech Super API ", e)
        #         scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompt, task_type)
        
        loop = asyncio.get_running_loop()
        # print("user_data['part_1_minute']: ",one_minute1)
        # print("user_data['part_1_minute_part_1']: ",user_data['part_1_minute_part_1'])
        try:
            print("use speech super assess_speech3 API (part 1)")
            scores, analysis_data = await loop.run_in_executor(executor, assess_speech3, filename, question_prompt, task_type)
        except Exception as e:
            print("🚨 Error on assess_speech3 speech super API and switching to Speech ace API ", e)
            try:
                scores, analysis_data = await loop.run_in_executor(executor, assess_speech, filename, question_prompt)
            except Exception as e:
                print("🚨Error on Speech Ace API and switching to Speech Super API assess_speech2 ", e)
                scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompt, task_type)
        # if one_minute1:  
        #     print("use speech super assess_speech2 API (part 1)")
        #     scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompt, task_type)
        # else:
        #     try:
        #         scores, analysis_data = await loop.run_in_executor(executor, assess_speech, filename, question_prompt)
        #     except Exception as e:
        #         print("🚨Error on Speech Ace API and switching to Speech Super API ", e)
        #         scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompt, task_type)
        # loop = asyncio.get_running_loop()
        # if user_data['part_1_minute']:
        #     print("use speech super API (part 1)")
        #     # print(filename)
        #     # print(question_prompt)
        #     # print(task_type)
            
        #     scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompt, task_type)
        #     # print(scores)
        #     # print("----------------")
        #     # print(analysis_data)
        #     response_json = scores
        # else:
        #     scores, analysis_data = await loop.run_in_executor(executor, assess_speech, filename, question_prompt)
        if scores is None or analysis_data is None:
            raise Exception("🚨 Assessment failed")

        # Immediately append analysis_data to the analysis_list
        analysis_list = user_data.setdefault('analysis_list', [])
        analysis_list.append(analysis_data)
        # if user_data['part_1_minute']:
        #     if 'result' in response_json:
        #         result = response_json['result']
        #         scores = {
        #             "overall": result.get("overall", 0),
        #             "pronunciation": result.get("pronunciation", 0),
        #             "fluency": result.get("fluency_coherence", 0),
        #             "grammar": result.get("grammar", 0),
        #             "vocabulary": result.get("lexical_resource", 0),
        #             "relevance": result.get("relevance", 0),
        #             "transcription": result.get("transcription", 0),
        #             # "pause_filler": result.get("pause_filler", {}),
        #         #     "sentences": [
        #         #         {
        #         #             "sentence": sentence.get("sentence", ""),
        #         #             "pronunciation": sentence.get("pronunciation", "N/A"),
        #         #             "grammar": sentence.get("grammar", {}).get("corrected", "")
        #         #         }
        #         #         for sentence in result.get("sentences", [])
        #         #     ]
        #         }
            
        #     os.remove(filename)
        #     print(len(user_data['analysis_list']))
        #     return scores
        # else:
        #     processed_scores = {
        #         "overall": scores['ielts_score']['overall'],
        #         "pronunciation": scores['ielts_score']['pronunciation'],
        #         "fluency": scores['ielts_score']['fluency'],
        #         "grammar": scores['ielts_score']['grammar'],
        #         "vocabulary": scores['ielts_score']['vocabulary'],
        #         "relevance": scores['relevance']['class'],
        #         "transcription": scores['transcription'],
        #     }

        #     os.remove(filename)
        #     print(len(user_data['analysis_list']))
        #     return processed_scores
        # if one_minute1:
        #     processed_scores = process_speech_super_scores(scores)
        #     print("process_speech_super_scores: ",processed_scores)
        # else:
        #     processed_scores = process_speechace_scores(scores)
        #     print("process_speechace_scores: ",processed_scores)
        try:
            processed_scores = process_speech_super_scores(scores)
        except Exception as e:
            print("🚨 process_speech_super_scores(scores): ", e)
            processed_scores = process_speechace_scores(scores)
        os.remove(filename)
        print(len(user_data['analysis_list']))
        return processed_scores
    except Exception as e:
        print(f"🚨 Error assessing speech: {str(e)}")
        return None

def process_speech_super_scores(response_json):
    # if 'result' in response_json:
        result = response_json
    #     result = response_json['result']
        return {
            "overall": result.get("overall", 0),
            "pronunciation": result.get("pronunciation", 0),
            "fluency": result.get("fluency", 0),
            "grammar": result.get("grammar", 0),
            "vocabulary": result.get("vocabulary", 0),
            "relevance": result.get("relevance", "N/A"),
            "transcription": result.get("transcription", 0),
        }
    # return None

def process_speechace_scores(scores):
    return {
        "overall": scores['ielts_score']['overall'],
        "pronunciation": scores['ielts_score']['pronunciation'],
        "fluency": scores['ielts_score']['fluency'],
        "grammar": scores['ielts_score']['grammar'],
        "vocabulary": scores['ielts_score']['vocabulary'],
        "relevance": scores['relevance']['class'],
        "transcription": scores['transcription'],
    }

async def generate_detailed_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE, waiting_message):
    try:
        print(f"{update.effective_user.id} detailed part 1 feedback")
        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        questions_list = user_data.get('questions', [])
        answers_list = user_data.get('answers_list', [])
        analysis_list = user_data.get('analysis_list', [])
        voice_urls = user_data.get('voice_urls', [])
        one_minute = user_data['part_1_minute_part_1']
        user_data['detailed_feedback'] = []
        detailed_feedback = user_data['detailed_feedback']

        for i in range(len(questions_list)):
            question = questions_list[i]
            user_answer = answers_list[i]
            analysis_data = analysis_list[i]
            user_voice_url = voice_urls[i]
            user_data['part_1_minute'] = one_minute[i]
            print("detailed user_data['part_1_minute']: ",user_data['part_1_minute'])
            prompt = f"Question: {question}\nUser Answer: {user_answer}\nAnalysis Data: {analysis_data}\n\n"
            prompt += """Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance. Generate useful feedback that helps the user understand their performance in depth.
            You should organize it in a clear and good feedback to the IELTS Candidate. They expect to know their mistakes and improve next time. Also, it is a good idea to write the question in the feedback and the answer and then start your analysis.
            """

            feedback = await generate_feedback_with_llm(prompt, context)
            detailed_feedback.append(feedback)
            user_data.setdefault('detailed_feedback_list', []).append(feedback)
            await send_long_message(update, context, feedback)

            # Use asyncio.to_thread for CPU-bound operations
            # if user_data['part_1_minute']:
            #     await asyncio.to_thread(generate_pronunciation_visualization2, analysis_data)
            # else:
            #     await asyncio.to_thread(generate_pronunciation_visualization, analysis_data)
            try:
                await asyncio.to_thread(generate_pronunciation_visualization2, analysis_data)
            except Exception as e:
                print("asyncio.to_thread(generate_pronunciation_visualization2, analysis_data): ", e)
                await asyncio.to_thread(generate_pronunciation_visualization, analysis_data)
            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)

            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text="Now, let's compare your pronunciation with a native speaker's pronunciation."
            )

            if user_voice_url:
                response = await asyncio.to_thread(requests.get, user_voice_url)
                voice_filename = f"user_voice_{i}.oga"
                with open(voice_filename, "wb") as file:
                    file.write(response.content)

                await context.bot.send_message(chat_id=update.effective_chat.id, text="Your Voice:")
                with open(voice_filename, "rb") as user_voice_file:
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=user_voice_file)

                os.remove(voice_filename)
                await asyncio.sleep(2)

                slow_audio_path = await convert_answer_to_audio(user_answer, '-0.56', examiner_voice)

                with open(slow_audio_path, 'rb') as slow_audio_file:
                    await context.bot.send_message(chat_id=update.effective_chat.id,
                                                   text="Native Speaker's Voice (Slow Speed):")
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
                os.remove(slow_audio_path)
                await asyncio.sleep(2)

                normal_audio_path = await convert_answer_to_audio(user_answer, '0', examiner_voice)

                with open(normal_audio_path, 'rb') as normal_audio_file:
                    await context.bot.send_message(chat_id=update.effective_chat.id,
                                                   text="Native Speaker's Voice (Normal Speed):")
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
                os.remove(normal_audio_path)

                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text="Listen to your voice and compare it with the native speaker's pronunciation at both speeds."
                )

        # Delete the waiting message
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        
        # Send the "What would you like to do next?" message with options
        userID = user_data['user_id']
        keyboard = [
            [InlineKeyboardButton("Continue to Part 2", callback_data=f'{userID}continue_part_2')],
            [InlineKeyboardButton("Translate", callback_data=f'{userID}translate_detailed_feedback')],
            [InlineKeyboardButton("Retake Part 1", callback_data=f'{userID}retake_part_1')],
            [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)

        return detailed_feedback
    except Exception as e:
        text = ("🚨 generate detailed feedback part 1 function ", e)
        await error_handling(update, context,text)

async def _translate_and_send_feedback(update, context, user_id, feedback, waiting_message, reply_markup):
    try:
        translated_feedback = await translate_feedback(user_id, feedback, update, context)
        if translated_feedback:
            # Send feedback report
            await send_long_message(update, context, translated_feedback)
            # await context.bot.send_message(chat_id=update.effective_chat.id, text=translated_feedback)
    except Exception as e:
        print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} | 🚨 Error in translation feedback 1: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, there was an error during translation. Please try again later.")
    finally:
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)

async def _translate_and_send_detailed_feedback(update, context, user_id, waiting_message, reply_markup):
    try:
        user_data = context.user_data.get('user_data', {})

        translated_feedback = user_data['translated_feedback1']
        
        for feedback in user_data.get('detailed_feedback_list', []):
            translated_msg = await translate_feedback(user_id, feedback, update, context)
            if translated_msg:
                translated_feedback.append(translated_msg)
        
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        
        if translated_feedback:
            for feedback in translated_feedback:
                await asyncio.sleep(2)
                # Send feedback report
                await send_long_message(update, context, feedback)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, no translated feedback is available.")
    
    except Exception as e:
        print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} | 🚨 Error in detailed feedback1 translation: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, there was an error during translation. Please try again later.")
    
    finally:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
# ---------------------------------- Part 2 ------------------------------------
async def assess_part2_speech_async(audio_url, question_prompt, task_type, context: ContextTypes.DEFAULT_TYPE):
    user_data = context.user_data.setdefault('user_data', {})
    try:
        # Download the voice file from the URL
        response = requests.get(audio_url)
        
        filename = f"voice_{int(time.time())}.oga"
        with open(filename, "wb") as file:
            file.write(response.content)

        loop = asyncio.get_running_loop()
        scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompt, task_type)
        response_json = scores 
        if scores is None or analysis_data is None:
            raise Exception("🚨 Assessment failed")

        # Immediately append analysis_data to the analysis_list
        analysis_list2 = user_data['analysis2_list']
        analysis_list2.append(analysis_data)

        if 'result' in response_json:
            result = response_json['result']
            scores = {
                "overall": result.get("overall", 0),
                "pronunciation": result.get("pronunciation", 0),
                "fluency": result.get("fluency_coherence", 0),
                "grammar": result.get("grammar", 0),
                "vocabulary": result.get("lexical_resource", 0),
                "relevance": result.get("relevance", 0),
                "transcription": result.get("transcription", 0),
                # "pause_filler": result.get("pause_filler", {}),
                "sentences": [
                    {
                        "sentence": sentence.get("sentence", ""),
                        "pronunciation": sentence.get("pronunciation", "N/A"),
                        "grammar": sentence.get("grammar", {}).get("corrected", "")
                    }
                    for sentence in result.get("sentences", [])
                ]
            }
        
        os.remove(filename)
        
        print("part2: ", len(user_data['analysis2_list']))
        return scores
        
    except Exception as e:
        print(f"🚨 Error assessing speech: {str(e)}")
        return None
async def generate_feedback2(scores_list, questions, answers,overall_score,targeted_score,context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})
        # prompt = "Provide detailed feedback for the following IELTS speaking assessment:\n\n"
        for i in range(len(questions)):
        #     prompt += f"Question {i+1}: {questions[i]}\n"
        #     prompt += f"Transcription of user answer: {answers[i]}\n\n"
        #     prompt += f"Pronunciation score: {scores_list[i]['pronunciation']}\n"
        #     prompt += f"Fluency score: {scores_list[i]['fluency']}\n"
        #     prompt += f"Grammar score: {scores_list[i]['grammar']}\n"
        #     prompt += f"Vocabulary score: {scores_list[i]['vocabulary']}\n"
        #     prompt += f"Fillers: {scores_list[i]['pause_filler']}\n"
        #     prompt += "Sentences of the answer (here is a detailed summary of the sentences in the answer):\n"
        #     for sentence in scores_list[i]['sentences']:
        #         prompt += f"  - Sentence: {sentence['sentence']}\n"
        #         prompt += f"    Pronunciation of the sentence: {sentence['pronunciation']}\n"
        #         prompt += f"    Grammar Correction if mistakes exist: {sentence['grammar']}\n"
        #     prompt += "\n"
        # prompt += "Provide an overall feedback report on the user's performance, including pronunciation, fluency, grammar, and vocabulary etc. you should provide a clear and good feedback to the IELTS Candidate he expects to kmow his mistakes and improve next time"
            prompt = f"Question: {questions}\nUser Answer: {answers}\nAnalysis Data of speaking skills and IELTS  Speaking Part 2: {scores_list}\n\n"
            # prompt += f"""Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance.and here are the ielts speaking band score of part 2 {overall_score}, Generate useful feedback that helps the user understand their performance in depth.
            # you should orginze it in a clear and good feedback to the IELTS Candidate he expects to kmow his mistakes and improve next time and  best tips and suggetions to  get hisa imed score {targeted_score} 
            
            # """
            prompt += f"""
                Please provide a detailed feedback report on the user's performance in Part 2 of the IELTS speaking test, covering aspects such as pronunciation, fluency and coherence, grammatical range and accuracy, lexical resource (vocabulary), interactive communication, and overall performance. Offer specific examples, constructive feedback, and practical suggestions for each aspect to help the candidate identify their strengths and weaknesses. Emphasize the importance of continuous practice, active listening, and critical thinking to enhance their performance and provide language learning strategies. Encourage the candidate to reflect on their performance and set achievable goals for their IELTS speaking development. Your feedback should be supportive and encouraging, aiming to motivate the candidate in their IELTS speaking journey. Organize the feedback in a clear and structured manner, using headings and bullet points for easy readability and do not  write any non-needed text.

                1. Pronunciation:
                - Clarity and intelligibility of speech
                - Stress, rhythm, and intonation patterns
                - Pronunciation of individual sounds and phonemes
                - Areas for improvement and specific examples

                2. Fluency:
                - Smooth flow of speech without excessive hesitation or pauses
                - Ability to maintain a consistent speed and rhythm
                - Use of fillers, repetition, and self-correction
                - Suggestions for enhancing fluency

                3. Grammatical Range and Accuracy:
                - Variety and complexity of grammatical structures used
                - Accuracy and control of grammar
                - Errors and their impact on clarity and meaning
                - Tips for improving grammatical accuracy

                4. Lexical Resource (Vocabulary):
                - Range and variety of vocabulary used
                - Accuracy and appropriacy of word choice
                - Use of idiomatic language and collocation
                - Recommendations for expanding vocabulary

                5. Relevance and Coherence:
                - Adherence to the given topic and task
                - Logical development and organization of ideas
                - Use of cohesive devices and discourse markers
                - Suggestions for improving relevance and coherence

                6. Overall Performance:
                - IELTS speaking band score for Part 2: {overall_score}
                - Strengths and areas for improvement
                - Comparison to the target score of {targeted_score}
                - Actionable steps and strategies to bridge the gap and achieve the desired score

                Please provide specific examples, constructive feedback, and practical suggestions for each aspect to help the candidate identify their strengths and weaknesses. Use a supportive and encouraging tone to motivate the candidate in their IELTS speaking journey.

                Organize the feedback in a clear and structured manner, using headings and bullet points for easy readability. Emphasize the importance of continuous practice and targeted skill development to improve their performance and reach their target score.
                """
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model=groq_model,
        )
        result = chat_completion.choices[0].message.content
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("feadback report generated")
        user_data['generate_feedback2'] = []
        user_data['generate_feedback2'] = result
        return user_data['generate_feedback2']
    except Exception as e:
        print("🚨 Groq error switching to perplexity generate_feedback2",e)
        messages = [
                {
                    "role": "system",
                    "content": (
                        ""
                       
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                    ),
                },
            ]
        client = OpenAI(api_key=perplexity_API, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model="llama-3.1-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("feadback report generated")
        user_data['generate_feedback2'] = []
        user_data['generate_feedback2'] = result
        return user_data['generate_feedback2']
async def show_result2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print(f"{update.effective_user.id} -----------------------FEEDBACK PART 2------------------------")
        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        part2_voice_urls = user_data.get('part2_voice_urls', [])
        targeted_score = user_data.get('targeted_score', 7)
        part2_questions = user_data.get('part2_questions', [])
        part2_answers = user_data.get('part2_answers', [])

        print(f"{update.effective_user.id} user_targetes_score: ", targeted_score)

        animated_emoji = "⏳"
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        
        progress_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Wait a few minutes until results are ready...\n\n[                             ] 0%"
        )

        share_message = f"Try the IELTS Speaking Bot! It simulates the IELTS speaking test and offers detailed feedback on your speaking skills and estimated band score. Try it for free: https://t.me/ielts_speakingAI_bot"
        keyboard = [[InlineKeyboardButton("Share the Bot", switch_inline_query=share_message)]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await ask_channel(update, context)
        share_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="While waiting for the results, would you like to share this bot? 😊",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        total_steps = len(part2_questions) + 4  # +4 for additional processing steps
        current_step = 5

        def update_progress():
            nonlocal current_step
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_bar = "[" + "█" * (progress // 5) + " " * (20 - (progress // 5)) + "]"
            return progress, progress_bar

        user_data['scores2_list'] = []
        scores2_list = user_data['scores2_list']

        for i, question_prompt in enumerate(part2_questions):
            audio_url = part2_voice_urls[i] if i < len(part2_voice_urls) else None
            task_type = "ielts_part2"

            processed_scores = await assess_part2_speech_async(audio_url, question_prompt, task_type, context)
            if processed_scores:
                scores2_list.append(processed_scores)
                print(f"{update.effective_user.id} Assessment successful for answer part 2 {i+1}")
            else:
                print(f"🚨ID: {update.effective_user.id} Username: {update.effective_user.username} Assessment failed for answer part 2 {i+1}")
                print(f"Voice URL: {audio_url}")
                print(f"Question: {question_prompt}")

            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Wait a few minutes until results are ready...\n{progress_bar} {progress}%"
            )

        if scores2_list:
            overall_score = scores2_list[0].get("overall", 0)
            pronunciation_score = scores2_list[0].get("pronunciation", 0)
            fluency_score = scores2_list[0].get("fluency", 0)
            grammar_score = scores2_list[0].get("grammar", 0)
            vocabulary_score = scores2_list[0].get("vocabulary", 0)

            overall_score = round_to_ielts_score(overall_score)
            pronunciation_score = round_to_ielts_score(pronunciation_score)
            fluency_score = round_to_ielts_score(fluency_score)
            grammar_score = round_to_ielts_score(grammar_score)
            vocabulary_score = round_to_ielts_score(vocabulary_score)

            feedback2_report = await generate_feedback2(scores2_list, part2_questions, part2_answers, overall_score, targeted_score,context)
            user_data['overall_part2_feedback'] = feedback2_report

            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Wait a few minutes until results are ready...\n{progress_bar} 100%"
            )

            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=progress_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=share_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)

            await send_long_message(update, context, feedback2_report)
            try:
                await send_recommendations(update, context, feedback2_report)
            except Exception as e:
                print(e)
            await display_feedback(update, context, overall_score, pronunciation_score, fluency_score, grammar_score, vocabulary_score)

            band_score = f"Your estimated IELTS band score for Part 2 is: {overall_score:.1f}"
            await context.bot.send_message(chat_id=update.effective_chat.id, text=band_score)

            cefr_level = get_cefr_level(overall_score)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Your language level is: {cefr_level}")
            
            await append_speaking_score(update, context, "part2", overall_score)
            await increment_practice_count(update, context)
            
            userID = str(update.effective_user.id)
            user_data['user_id'] = userID
            
            keyboard = [
                [InlineKeyboardButton("Continue to Part 3", callback_data=f'{userID}continue_part3')],
                [InlineKeyboardButton("See Detailed Results", callback_data=f'{userID}detailed2_results')],
                [InlineKeyboardButton("Translate", callback_data=f'{userID}translate_overall2_feedback')],
                [InlineKeyboardButton("Retake Part 2", callback_data=f'{userID}retake_part2')],
                [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)

        else:
            text = ("🚨 Failed to assess the answers. Please try again. show results part 2")
            await error_handling(update, context,text)
    except Exception as e:
        text = ("🚨 show results part 2", e)
        await error_handling(update, context,text)
async def generate_detailed2_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE, waiting_message):
    try:
        print(f"{update.effective_user.id} detailed part 2 feedback")
        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        part2_questions = user_data.get('part2_questions', [])
        part2_answers = user_data.get('part2_answers', [])
        analysis2_list = user_data.get('analysis2_list', [])
        part2_voice_urls = user_data.get('part2_voice_urls', [])

        user_data['detailed2_feedback'] = []
        detailed_feedback = user_data['detailed2_feedback']

        for i in range(len(part2_questions)):
            question = part2_questions[i]
            user_answer = part2_answers[i]
            analysis_data = analysis2_list[i]
            user_voice_url = part2_voice_urls[i]

            prompt = f"Question: {question}\nUser Answer: {user_answer}\nAnalysis Data: {analysis_data}\n\n"
            prompt += """Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance. Generate useful feedback that helps the user understand their performance in depth and should be in details.
            You should organize it in a clear and good feedback to the IELTS Candidate. They expect to know their mistakes and improve next time. Also, it is a good idea to write the question in the feedback and the answer and then start your analysis.
            """

            feedback = await generate_feedback_with_llm(prompt,context)
            detailed_feedback.append(feedback)
            user_data.setdefault('detailed_feedback2_list', []).append(feedback)
            await send_long_message(update, context, feedback)

            # Use asyncio.to_thread for CPU-bound operations
            await asyncio.to_thread(generate_pronunciation_visualization2, analysis_data)
            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)

            await context.bot.send_message(chat_id=update.effective_chat.id, text="Now, let's compare your pronunciation with a native speaker's pronunciation.")

            if user_voice_url:
                response = await asyncio.to_thread(requests.get, user_voice_url)
                voice_filename = f"user_voice_{i+1}.oga"
                with open(voice_filename, "wb") as file:
                    file.write(response.content)

                await context.bot.send_message(chat_id=update.effective_chat.id, text="Your Voice:")
                with open(voice_filename, "rb") as user_voice_file:
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=user_voice_file)

                os.remove(voice_filename)

                slow_audio_path = await convert_answer_to_audio(user_answer, '-0.56', examiner_voice)

                if slow_audio_path:
                    with open(slow_audio_path, 'rb') as slow_audio_file:
                        await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Slow Speed):")
                        await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
                    os.remove(slow_audio_path)
                else:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to generate native speaker's voice (slow speed).")

                normal_audio_path = await convert_answer_to_audio(user_answer, '0', examiner_voice)

                if normal_audio_path:
                    with open(normal_audio_path, 'rb') as normal_audio_file:
                        await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Normal Speed):")
                        await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
                    os.remove(normal_audio_path)
                else:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to generate native speaker's voice (normal speed).")

        # Delete the waiting message
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        
        # Send the "What would you like to do next?" message with options
        userID = str(update.effective_user.id)
        keyboard = [
            [InlineKeyboardButton("Continue to Part 3", callback_data=f'{userID}continue_part3')],
            [InlineKeyboardButton("Translate", callback_data=f'{userID}translate_detailed2_feedback')],
            [InlineKeyboardButton("Retake Part 2", callback_data=f'{userID}retake_part2')],
            [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)

        return detailed_feedback
    except Exception as e:
        text = ("🚨 generate detailed feedback part 2 function ", e)
        await error_handling(update, context,text)
def generate_pronunciation_visualization2(answer_data):
    try:
        # Extract the sentences and word pronunciation details from the answer data
        sentences = answer_data['scores']['sentences']
        word_pronunciation_details = answer_data['word_pronunciation_details']
        
        # Define colors for pronunciation scores
        colors = {
            'Correct Pronunciation +80/100': (22, 219, 101),    # Correct pronunciation
            'Slightly Incorrect Pronunciation +50/100': (255, 203, 5), # Slightly incorrect pronunciation
            'Incorrect Pronunciation -50/100': (216, 0, 50)  # Incorrect pronunciation
        }
        
        # Define padding
        padding = 20
        max_line_width = 1200 - 2 * padding  # Max line width with padding
        
        # Load fonts
        title_font = ImageFont.truetype("Roboto-Bold.ttf", 36)
        text_font = ImageFont.truetype("Roboto-Regular.ttf", 24)
        guide_font = ImageFont.truetype("Roboto-Regular.ttf", 18)
        
        # Dummy image for calculation
        dummy_image = Image.new('RGB', (1, 1), color='white')
        draw = ImageDraw.Draw(dummy_image)
        
        # Calculate title size
        title = "Pronunciation Score"
        title_width, title_height = draw.textbbox((0, 0), title, font=title_font)[2:]
        
        # Calculate the height needed for the image
        y = padding + title_height + 40  # Initial y position after the title
        
        total_height = y + padding  # Start with the initial padding
        
        for sentence_details in word_pronunciation_details:
            x = padding  # Starting x position for each sentence
            for word_info in sentence_details:
                word = word_info['word']
                word_width, word_height = draw.textbbox((0, 0), word, font=text_font)[2:]
                
                # Check if word fits in the current line, if not move to the next line
                if x + word_width > max_line_width:
                    x = padding
                    y += word_height + 10
                
                x += word_width + 10
            
            y += word_height + 20  # Increase y position for next sentence
        
        # Add space for the guide
        total_height = y + word_height + 50  # Extra space for the guide
        
        # Create a new image with a white background
        image_width = 1200
        image_height = total_height
        image = Image.new('RGB', (image_width, image_height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw title
        title_x = (image_width - title_width) // 2
        title_y = padding
        draw.text((title_x, title_y), title, font=title_font, fill=(0, 0, 0))
        
        # Draw sentences and colored words with word wrapping
        y = title_y + title_height + 40
        for sentence, sentence_details in zip(sentences, word_pronunciation_details):
            x = padding  # Starting x position for each sentence
            for word_info in sentence_details:
                word = word_info['word']
                score = word_info['pronunciation']
                
                if score >= 80:
                    color = colors['Correct Pronunciation +80/100']
                elif score >= 50:
                    color = colors['Slightly Incorrect Pronunciation +50/100']
                else:
                    color = colors['Incorrect Pronunciation -50/100']
                
                word_width, word_height = draw.textbbox((0, 0), word, font=text_font)[2:]
                
                # Check if word fits in the current line, if not move to the next line
                if x + word_width > max_line_width:
                    x = padding
                    y += word_height + 10
                
                draw.text((x, y), word, font=text_font, fill=color)
                x += word_width + 10
            
            y += word_height + 20  # Increase y position for next sentence
        
        # Draw color guide
        guide_x = padding
        guide_y = image_height - padding - 30
        guide_text = ""
        draw.text((guide_x, guide_y), guide_text, font=guide_font, fill=(0, 0, 0))
        
        guide_x += draw.textbbox((0, 0), guide_text, font=guide_font)[2] + 20
        for color_name, color_code in colors.items():
            draw.rectangle((guide_x, guide_y, guide_x + 20, guide_y + 20), fill=color_code)
            draw.text((guide_x + 30, guide_y), color_name.capitalize(), font=guide_font, fill=(0, 0, 0))
            guide_x += draw.textbbox((0, 0), color_name.capitalize(), font=guide_font)[2] + 60
        
        # Save the image to a file or send it directly to the user
        image.save('pronunciation_visualization_with_padding.png')
    except Exception as e:
        print("🚨 generate pronunciation visulization function ",e)
        # await update.message.reply_text(issue_message)
async def start_part2_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        # Randomly select a topic from the dictionary
        selected_topic = random.choice(list(ielts_questions.keys()))
        # print("user_data['examiner_voice'] ",user_data['examiner_voice'])
        # Extract the Part 2 question and Part 3 questions for the selected topic
        part2_question = ielts_questions[selected_topic]["part_2_question"]
        part_3_questions = ielts_questions[selected_topic]["part_3_questions"]

        # Store the Part 2 question and Part 3 questions in the user_data
        user_data['part2_question'] = part2_question
        user_data['part3_questions'] = part_3_questions
        
        # Clear and update the part3_questions list
        # user_data.setdefault('part3_questions', []).clear()
        # user_data['part3_questions'].extend(part_3_questions)
        # print("user_data['part3_questions'] from part 2 func: ",user_data['part3_questions'])
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        keyboard = [
            [InlineKeyboardButton("Change Question", callback_data=f'{userID}change_question')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # print(examiner_voice)
        # Convert the question to audio using the Unrealspeech API
        audio_path = await convert_text_to_audio(part2_question,examiner_voice)

        # Send the question as both text and audio to the user
        question_message = await update.effective_message.reply_text(part2_question, reply_markup=reply_markup)
        with open(audio_path, 'rb') as audio:
            audio_message = await update.effective_message.reply_voice(voice=audio)

        # Inform the user that they have 1 minute to prepare their answer
        preparation_message = await update.effective_message.reply_text("You have 1 minute to prepare your answer.")
        animated_emoji = "⏳"  # Hourglass emoji
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)

        # Start a countdown timer for 1 minute
        countdown_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="60 seconds remaining...")

        # for remaining in range(59, 0, -1):
        #     if user_data['test_stop']:
        #         print(f"{update.effective_user.id} Test stopped")
        #         break
        #     else:
        #         await asyncio.sleep(1)
        #         try:
        #             await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=countdown_message.message_id, text=f"{remaining} seconds remaining...")
        #         except Exception as e:
        #             print(f"{update.effective_user.id} 🚨 Failed to update countdown message: {e}")
              # Store all message IDs
        user_data['part2_question_message_id'] = question_message.message_id
        user_data['part2_audio_message_id'] = audio_message.message_id
        user_data['part2_preparation_message_id'] = preparation_message.message_id
        user_data['part2_waiting_message_id'] = waiting_message.message_id
        user_data['part2_countdown_message_id'] = countdown_message.message_id
        
        user_data['continue_countdown'] = True
        for remaining in range(59, 0, -1):
            if not user_data['continue_countdown'] or user_data['test_stop']:
                print(f"{update.effective_user.id} Countdown stopped")
                break
            await asyncio.sleep(1)
            try:
                await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=countdown_message.message_id, text=f"{remaining} seconds remaining...")
            except Exception as e:
                print(f"{update.effective_user.id} 🚨 Failed to update countdown message: {e}")
        # Delete the countdown message and hourglass emoji

        try:
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=countdown_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=preparation_message.message_id)
        except Exception as e:
            print(e)
        # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=question_message.message_id)
        # # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=recording_message.message_id)
        # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=audio_message.message_id)
        # Request the user to record their answer
        user_data['part2_answering'] = True
        if user_data['continue_countdown']:
             recording_message = await update.effective_message.reply_text("Please record your answer. It should be between 1 to 2 minutes long.")
             user_data['part2_recording_message_id'] = recording_message.message_id
        # # Store the message IDs in the user_data
        # user_data['part2_question_message_id'] = question_message.message_id
        # user_data['part2_audio_message_id'] = audio_message.message_id
        # user_data['part2_preparation_message_id'] = preparation_message.message_id
        # user_data['part2_waiting_message_id'] = waiting_message.message_id
        
        # user_data['part2_countdown_message_id'] = countdown_message.message_id
    except Exception as e:
        text = ("🚨 start part 2 test", e)
        await error_handling(update, context,text)
async def _translate_and_send_feedback2(update, context, user_id, feedback, waiting_message, reply_markup):
    try:
        translated_feedback = await translate_feedback(user_id, feedback, update, context)
        if translated_feedback:
            await send_long_message(update, context, translated_feedback)
    except Exception as e:
        print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} | 🚨 Error in translation feedback 2: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, there was an error during translation. Please try again later.")
    finally:
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)

async def _translate_and_send_detailed2_feedback(update, context, user_id, detailed_feedback_list, waiting_message, reply_markup):
    try:
        user_data = context.user_data.get('user_data', {})
        translated_feedback = user_data['translated_feedback2']
        for feedback in detailed_feedback_list:
            translated_msg = await translate_feedback(user_id, feedback, update, context)
            if translated_msg:
                translated_feedback.append(translated_msg)
        
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        
        if translated_feedback:
            for feedback in translated_feedback:
                await asyncio.sleep(2)
                await send_long_message(update, context, feedback)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, no translated feedback is available.")
    
    except Exception as e:
        print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} | 🚨 Error in detailed feedback2 translation: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, there was an error during translation. Please try again later.")
    
    finally:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
#----------------------------- Part 3 --------------------------
async def start_part3_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print(f"{update.effective_user.id} start part 3 test")
    try:
        user_data = context.user_data.setdefault('user_data', {})

        # Initialize part3 specific data
        user_data['part3_answers'] = []
        user_data['part3_current_question_index'] = 0

        # Get the part3_questions from user_data
        # part3_questions = user_data.get('part3_questions', [])
        part3_questions = user_data['part3_questions']
        # print("part3_questions: ",part3_questions)
        await update.effective_message.reply_text(
            "IELTS Speaking Part 3:\n\n"
            f"Now, we will begin Part 3 of the IELTS Speaking test. In this part, I will ask you ({len(part3_questions)}) "
            "abstract and complex questions related to the topic from Part 2. Please answer each question in detail, "
            "with as much depth as possible. \n\nLet's start"
        )
        
        await asyncio.sleep(3)  # Using asyncio.sleep instead of time.sleep for asynchronous operation
        await ask_part3_question(update, context)
    except Exception as e:
        text = ("🚨 start part 3 test function ", e)
        await error_handling(update, context,text)
async def ask_part3_question(update: Update, context: ContextTypes.DEFAULT_TYPE, retry=False):
    print(f"{update.effective_user.id} ask part 3 question")
    try:
        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        # part3_questions = user_data.get('part3_questions', [])
        # part3_answers = user_data.get('part3_answers', [])
        part3_questions = user_data['part3_questions']
        part3_answers = user_data['part3_answers']
        current_question_index = user_data.get('part3_current_question_index', 0)
        # print("ask part3_questions : ",part3_questions)
        if retry:
            user_data = context.user_data['user_data']
            user_id = update.effective_user.id
            user_data['user_id'] = str(user_id)
            userID = user_data['user_id']
            keyboard = [
                [InlineKeyboardButton("Suggest Answer", callback_data=f'{userID}part3_suggest_answer')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.effective_message.reply_text("Please re-answer the question.", reply_markup=reply_markup)
        else:
            if current_question_index < len(part3_questions):
                current_question = part3_questions[current_question_index].strip()
                # print("1")
                # user_answer = part3_answers.get(current_question_index, "")
                user_answer = part3_answers[current_question_index] if current_question_index < len(part3_answers) else ""
                user_data['current_part3_question'] = current_question
                question_number = current_question_index + 1
                formatted_message = f"{question_number}. {current_question}"
                # print("current question: ", formatted_message)
                try:
                    await update.effective_message.reply_text(formatted_message)
                    
                    audio_file_path = await convert_text_to_audio(formatted_message,examiner_voice)
                    
                    with open(audio_file_path, 'rb') as audio:
                        await update.effective_message.reply_voice(voice=audio)
                    await update.effective_message.reply_text("Please record your answer.")
                    user_data['answering_part3_question'] = True
                    print(f"{update.effective_user.id} Set answering_part3_question to True")
                except Exception as e:
                    print(f"🚨 Error converting text to audio: {e}")
                    # Retry the conversion without sending an error message
                    await update.effective_message.reply_text("Please record your answer.")
                    user_data['answering_part3_question'] = True
                    print(f"{update.effective_user.id} Set answering_part3_question to True")
            else:
                await show_part3_summary(update, context)
    except Exception as e:
        text = ("🚨 ask part 3 questions function ", e)
        await error_handling(update, context,text)
async def part3_suggest_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})

        query = update.callback_query
        await query.answer()

        question = user_data.get('current_part3_question', '')
        previous_answer = user_data.get('current_part3_answer', '')

        suggested_answer = await generate_suggested_answer(question, previous_answer, "part 3",context)
        await query.edit_message_reply_markup(reply_markup=None)
        await update.effective_message.reply_text(f"Suggested Answer:\n\n{suggested_answer}")
        await update.effective_message.reply_text("Please record your answer again.")
        
        # Set the flag to indicate that the user is answering a Part 3 question
        user_data['answering_part3_question'] = True

    except Exception as e:
        text = ("🚨 part 3 suggested answer function ", e)
        await error_handling(update, context,text)
async def show_part3_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})

        part3_questions = user_data.get('part3_questions', [])
        part3_answers = user_data.get('part3_answers', [])

        # summary_message = "Part 3 Questions and Answers:\n\n"
        # for i, question in enumerate(part3_questions):
        #     answer = part3_answers.get(i, "No answer provided")
        #     summary_message += f"Question {i+1}: {question}\nAnswer: {answer}\n\n"

        # await update.effective_message.reply_text(summary_message)
        typical_answers = await generate_typical_answers(part3_questions, part3_answers, context)
        await send_long_message(update, context, f"Typical Answers:\n\n{typical_answers}")
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        keyboard = [
            [InlineKeyboardButton("Check Your Score ✅", callback_data=f'{userID}part3_show_results')],
            [InlineKeyboardButton("Retake Part 3", callback_data=f'{userID}part3_retake')],
            [InlineKeyboardButton("End Test", callback_data=f'{userID}end_test')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.effective_message.reply_text("What would you like to do next?", reply_markup=reply_markup)

    except Exception as e:
        text = ("🚨 show part 3 summary function  ", e)
        await error_handling(update, context,text)
async def assess_part3_speech_async(audio_url, question_prompt, task_type, context: ContextTypes.DEFAULT_TYPE):
    user_data = context.user_data.setdefault('user_data', {})
    try:
        # Download the voice file from the URL
        response = requests.get(audio_url)
        
        filename = f"voice_{int(time.time())}.oga"
        with open(filename, "wb") as file:
            file.write(response.content)

        print("user_data['part_3_minute']: ", user_data['part_3_minute'])

        # loop = asyncio.get_running_loop()
        # if user_data['part_3_minute']:
        #     print("use speech super API (part 3)")
        #     scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompt, task_type)
        #     response_json = scores
        # else:
        #     scores, analysis_data = await loop.run_in_executor(executor, assess_speech, filename, question_prompt)
        loop = asyncio.get_running_loop()
        # print("user_data['part_1_minute']: ",one_minute1)
        # print("user_data['part_1_minute_part_1']: ",user_data['part_1_minute_part_1'])
        try:
            print("use speech super assess_speech3 API (part 3)")
            scores, analysis_data = await loop.run_in_executor(executor, assess_speech3, filename, question_prompt, task_type)
            response_json = scores
        except Exception as e:
            print("🚨 Error on assess_speech3 speech super API and switching to Speech ace API ", e)
            try:
                scores, analysis_data = await loop.run_in_executor(executor, assess_speech, filename, question_prompt)
            except Exception as e:
                print("🚨Error on Speech Ace API and switching to Speech Super API assess_speech2 ", e)
                scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompt, task_type)
                response_json = scores
        if scores is None or analysis_data is None:
            raise Exception("🚨 Assessment failed")

        # Immediately append analysis_data to the analysis_list
        analysis_list = user_data['analysis3_list']
        analysis_list.append(analysis_data)

        # if user_data['part_3_minute']:
        try:
            if 'result' in response_json:
                result = response_json['result']
                scores = {
                    "overall": result.get("overall", 0),
                    "pronunciation": result.get("pronunciation", 0),
                    "fluency": result.get("fluency_coherence", 0),
                    "grammar": result.get("grammar", 0),
                    "vocabulary": result.get("lexical_resource", 0),
                    "relevance": result.get("relevance", 0),
                    "transcription": result.get("transcription", 0),
                    # "pause_filler": result.get("pause_filler", {}),
                #     "sentences": [
                #         {
                #             "sentence": sentence.get("sentence", ""),
                #             "pronunciation": sentence.get("pronunciation", "N/A"),
                #             "grammar": sentence.get("grammar", {}).get("corrected", "")
                #         }
                #         for sentence in result.get("sentences", [])
                #     ]
                }
            
            os.remove(filename)
            print(len(user_data['analysis3_list']))
            return scores
        # else:
        except Exception as e:
            print('🚨 Error assess_part 3 speech assessing', e)
            processed_scores = {
                "overall": scores['ielts_score']['overall'],
                "pronunciation": scores['ielts_score']['pronunciation'],
                "fluency": scores['ielts_score']['fluency'],
                "grammar": scores['ielts_score']['grammar'],
                "vocabulary": scores['ielts_score']['vocabulary'],
                "relevance": scores['relevance']['class'],
                "transcription": scores['transcription'],
            }

            os.remove(filename)
            print(len(user_data['analysis3_list']))
            return processed_scores
    except Exception as e:
        print(f"🚨 Error assessing speech: {str(e)}")
        return None
async def generate_feedback3(scores_list, questions, answers, overall_avg,targeted_score,context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})
        # global targeted_score
        prompt = "Provide detailed feedback for the following IELTS Speaking Part 3 assessment:\n\n"
        for i in range(len(questions)):
            prompt += f"Question {i+1}: {questions[i]}\n"
            prompt += f"Transcription of user answer: {answers[i]}\n\n"
            prompt += f"Pronunciation score: {scores_list[i]['pronunciation']}\n"
            prompt += f"Fluency score: {scores_list[i]['fluency']}\n"
            prompt += f"Grammar score: {scores_list[i]['grammar']}\n"
            prompt += f"Vocabulary score: {scores_list[i]['vocabulary']}\n"
            # prompt += f"Fillers: {scores_list[i]['pause_filler']}\n"
            # prompt += "Sentences of the answer (here is a detailed summary of the sentences in the answer):\n"
            # for sentence in scores_list[i]['sentences']:
            #     prompt += f"  - Sentence: {sentence['sentence']}\n"
            #     prompt += f"    Pronunciation of the sentence: {sentence['pronunciation']}\n"
            #     prompt += f"    Grammar Correction if mistakes exist: {sentence['grammar']}\n"
            # prompt += "\n"
        # prompt +=  f"Provide an overall feedback report on the user's performance, including pronunciation, fluency, grammar, and vocabulary etc. and here are the ielts speaking band score of part 3 {overall_avg} you should provide a clear and good feedback to the IELTS Candidate he expects to kmow his mistakes and improve next time and  best tips and suggetions to  get hisa imed score {targeted_score}"
        prompt += f"""
           Please provide a detailed feedback report on the user's performance in Part 3 of the IELTS speaking test, covering aspects such as pronunciation, fluency and coherence, grammatical range and accuracy, lexical resource (vocabulary), interactive communication, and overall performance. Offer specific examples, constructive feedback, and practical suggestions for each aspect to help the candidate identify their strengths and weaknesses. Emphasize the importance of continuous practice, active listening, and critical thinking to enhance their performance and provide language learning strategies. Encourage the candidate to reflect on their performance and set achievable goals for their IELTS speaking development. Your feedback should be supportive and encouraging, aiming to motivate the candidate in their IELTS speaking journey. Organize the feedback in a clear and structured manner, using headings and bullet points for easy readability and do not  write any non-needed text.

            1. Pronunciation:
            - Clarity and intelligibility of speech
            - Stress, rhythm, and intonation patterns
            - Pronunciation of individual sounds and phonemes
            - Areas for improvement and specific examples

            2. Fluency and Coherence:
            - Smooth flow of speech without excessive hesitation or pauses
            - Ability to maintain a consistent speed and rhythm
            - Use of fillers, repetition, and self-correction
            - Logical development and organization of ideas
            - Use of cohesive devices and discourse markers
            - Suggestions for enhancing fluency and coherence

            3. Grammatical Range and Accuracy:
            - Variety and complexity of grammatical structures used
            - Accuracy and control of grammar
            - Errors and their impact on clarity and meaning
            - Tips for improving grammatical accuracy

            4. Lexical Resource (Vocabulary):
            - Range and variety of vocabulary used
            - Accuracy and appropriacy of word choice
            - Use of idiomatic language and collocation
            - Ability to paraphrase and explain ideas
            - Recommendations for expanding vocabulary

            5. Interactive Communication:
            - Ability to engage in a discussion and express opinions
            - Responsiveness to the examiner's questions and prompts
            - Ability to ask for clarification or elaboration when needed
            - Suggestions for improving interactive communication skills

            6. Overall Performance:
            - Here is the list of all overall scores of speaking skills with the overall IELTS band score of Part 3 {overall_avg} 
            - Strengths and areas for improvement
            - Comparison to the target score of {targeted_score}
            - Actionable steps and strategies to bridge the gap and achieve the desired score

            Please provide specific examples, constructive feedback, and practical suggestions for each aspect to help the candidate identify their strengths and weaknesses. Use a supportive and encouraging tone to motivate the candidate in their IELTS speaking journey.

            Organize the feedback in a clear and structured manner, using headings and bullet points for easy readability. Emphasize the importance of continuous practice, active listening, and critical thinking to enhance their performance in Part 3 of the IELTS speaking test.

            Offer targeted recommendations and resources for further improvement, such as practice materials, language learning strategies, and opportunities for authentic language use. Encourage the candidate to reflect on their performance and set achievable goals for their IELTS speaking development.
            """

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model=groq_model,
        )
        result = chat_completion.choices[0].message.content
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("Feedback report generated")
        user_data['generate_feedback3'] = []
        user_data['generate_feedback3'] = result
        return user_data['generate_feedback3']
    except Exception as e:
        print("🚨 Groq error switching to perplexity generate_feedback3",e)
        
        messages = [
                {
                    "role": "system",
                    "content": (
                        ""
                       
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                    ),
                },
            ]
        client = OpenAI(api_key=perplexity_API, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model="llama-3.1-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("Feedback report generated")
        user_data['generate_feedback3'] = []
        user_data['generate_feedback3'] = result
        return user_data['generate_feedback3']
async def generate_detailed3_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print(f"{update.effective_user.id} detailed part 3 feedback")
        user_data = context.user_data.setdefault('user_data', {})

        part3_questions = user_data.get('part3_questions', [])
        part3_answers = user_data.get('part3_answers', [])
        analysis3_list = user_data.get('analysis3_list', [])
        part3_voice_urls = user_data.get('part3_voice_urls', [])
        part_3_minute = user_data.get('part_3_minute', False)
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        # print("part3_voice_urls ",len(part3_voice_urls))
        # print("part3_questions ",len(part3_questions))
        # print("part3_answers ",len(part3_answers))
        user_data['detailed3_feedback'] = []
        detailed_feedback = user_data['detailed3_feedback']
        
        for i, question in enumerate(part3_questions):
            user_answer = part3_answers[i]
            analysis_data = analysis3_list[i] #if i < len(analysis3_list) else {}
            user_voice_url = part3_voice_urls[i] #if i < len(part3_voice_urls) else None

            prompt = f"Question: {question}\nUser Answer: {user_answer}\nAnalysis Data: {analysis_data}\n\n"
            prompt += """Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance. Generate useful feedback that helps the user understand their performance in depth.
            You should organize it in a clear and good feedback to the IELTS Candidate. They expect to know their mistakes and improve next time. Also, it is a good idea to write the question in the feedback and the answer and then start your analysis.
            """
            
            feedback = await generate_feedback_with_llm(prompt,context)
            detailed_feedback.append(feedback)
            user_data.setdefault('detailed_feedback3_list', []).append(feedback)
            await send_long_message(update, context, feedback)

            # if user_data['part_3_minute']:
            #     generate_pronunciation_visualization2(analysis_data)
            # else:
            #     generate_pronunciation_visualization(analysis_data)
            try:
                await asyncio.to_thread(generate_pronunciation_visualization2, analysis_data)
            except Exception as e:
                print("asyncio.to_thread(generate_pronunciation_visualization2, analysis_data): ", e)
                await asyncio.to_thread(generate_pronunciation_visualization, analysis_data)
            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)
            
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Now, let's compare your pronunciation with a native speaker's pronunciation.")
            
            if user_voice_url:
                response = requests.get(user_voice_url)
                voice_filename = f"user_voice_{i+1}.oga"
                with open(voice_filename, "wb") as file:
                    file.write(response.content)
                await asyncio.sleep(2)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Your Voice:")
                with open(voice_filename, "rb") as user_voice_file:
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=user_voice_file)
                
                os.remove(voice_filename)
                await asyncio.sleep(2)

                slow_audio_path = await convert_answer_to_audio(user_answer, '-0.56',examiner_voice)
                
                if slow_audio_path:
                    with open(slow_audio_path, 'rb') as slow_audio_file:
                        await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Slow Speed):")
                        await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
                    os.remove(slow_audio_path)
                await asyncio.sleep(2)

                normal_audio_path = await convert_answer_to_audio(user_answer, '0',examiner_voice)
                
                if normal_audio_path:
                    with open(normal_audio_path, 'rb') as normal_audio_file:
                        await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Normal Speed):")
                        await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
                    os.remove(normal_audio_path)

                await context.bot.send_message(chat_id=update.effective_chat.id, text="Listen to your voice and compare it with the native speaker's pronunciation at both speeds.")
        
        return detailed_feedback 
    except Exception as e:
        text = ("🚨 generate detailed feedback part 3 function  ", e)
        await error_handling(update, context,text)
async def part3_show_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print(f"{update.effective_user.id} -----------------------FEEDBACK PART 3------------------------")
        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        part3_questions = user_data.get('part3_questions', [])
        part3_answers = user_data.get('part3_answers', [])
        part3_voice_urls = user_data.get('part3_voice_urls', [])
        targeted_score = user_data.get('targeted_score', 7)

        query = update.callback_query
        await query.answer()
        print(f"{update.effective_user.id} user_targetes_score: ", targeted_score)

        animated_emoji = "⏳"
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        
        progress_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Wait a few minutes until results are ready...\n\n[                             ] 0%"
        )

        share_message = f"Try the IELTS Speaking Bot! It simulates the IELTS speaking test and offers detailed feedback on your speaking skills and estimated band score. Try it for free: https://t.me/ielts_speakingAI_bot"
        keyboard = [[InlineKeyboardButton("Share the Bot", switch_inline_query=share_message)]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await ask_channel(update, context)
        share_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="While waiting for the results, would you like to share this bot? 😊",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        total_steps = len(part3_questions) + 4  # +4 for additional processing steps
        current_step = 0

        def update_progress():
            nonlocal current_step
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_bar = "[" + "█" * (progress // 5) + " " * (20 - (progress // 5)) + "]"
            return progress, progress_bar

        user_data['scores_list3'] = []
        scores_list = user_data['scores_list3']

        for i, question in enumerate(part3_questions):
            audio_url = part3_voice_urls[i] if i < len(part3_voice_urls) else None
            question_prompt = question
            task_type = "ielts_part3"
            
            processed_scores = await assess_part3_speech_async(audio_url, question_prompt, task_type, context)
            if processed_scores:
                scores_list.append(processed_scores)
                print(f"{update.effective_user.id} Assessment successful for question {i+1}")
            else:
                print(f"🚨ID: {update.effective_user.id} Username: {update.effective_user.username} Assessment failed for question {i+1}")
                print(f"Voice URL: {audio_url}")
                print(f"Question: {question_prompt}")

            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Wait a few minutes until results are ready...\n{progress_bar} {progress}%"
            )

        print(f"{update.effective_user.id} All assessments completed.")

        if scores_list:
            overall_avg = sum(score["overall"] for score in scores_list) / len(scores_list)
            pronunciation_avg = sum(score["pronunciation"] for score in scores_list) / len(scores_list)
            fluency_avg = sum(score["fluency"] for score in scores_list) / len(scores_list)
            grammar_avg = sum(score["grammar"] for score in scores_list) / len(scores_list)
            vocabulary_avg = sum(score["vocabulary"] for score in scores_list) / len(scores_list)

            overall_avg = round_to_ielts_score(overall_avg)
            pronunciation_avg = round_to_ielts_score(pronunciation_avg)
            fluency_avg = round_to_ielts_score(fluency_avg)
            grammar_avg = round_to_ielts_score(grammar_avg)
            vocabulary_avg = round_to_ielts_score(vocabulary_avg)

            overall_scores = {
                "pronunciation": pronunciation_avg,
                "fluency": fluency_avg,
                "grammar": grammar_avg,
                "vocabulary": vocabulary_avg,
                "IELTS band score": overall_avg,
            }

            overall_feedback3 = await generate_feedback3(scores_list, part3_questions, part3_answers, overall_scores, targeted_score,context)
            user_data['overall_part3_feedback'] = overall_feedback3

            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Wait a few minutes until results are ready...\n{progress_bar} 100%"
            )

            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=progress_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=share_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)

            await send_long_message(update, context, overall_feedback3)
            try:
                await send_recommendations(update, context, overall_feedback3)
            except Exception as e:
                print(e)
            await display_feedback(update, context, overall_avg, pronunciation_avg, fluency_avg, grammar_avg, vocabulary_avg)

            band_score = f"Your estimated IELTS band score is: {overall_avg:.1f}"
            await context.bot.send_message(chat_id=update.effective_chat.id, text=band_score)

            cefr_level = get_cefr_level(overall_avg)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Your language level is: {cefr_level}")

            await append_speaking_score(update, context, "part3", overall_avg)
            await increment_practice_count(update, context)
            
            userID = str(update.effective_user.id)
            user_data['user_id'] = userID
            
            keyboard = [
                [InlineKeyboardButton("Show Detailed Results", callback_data=f'{userID}part3_detailed_results')],
                [InlineKeyboardButton("Translate", callback_data=f'{userID}part3_translate_feedback')],
                [InlineKeyboardButton("Retake Part 3", callback_data=f'{userID}part3_retake')],
                [InlineKeyboardButton("End Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
        else:
            text = ("🚨 Failed to assess the answers. Please try again. show results part 3")
            await error_handling(update, context,text)
    except Exception as e:
        text = ("🚨 part 3 show results function ", e)
        await error_handling(update, context,text)
async def part3_detailed_results(update: Update, context: ContextTypes.DEFAULT_TYPE, waiting_message):
    try:
        query = update.callback_query
        await query.answer()

        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        part3_questions = user_data.get('mock_part3_questions', [])
        part3_answers = user_data.get('mock_part3_answers', [])
        analysis3_list = user_data.get('mock_part3_analysis_list', [])
        part3_voice_urls = user_data.get('mock_part3_voice_urls', [])

        user_data['detailed3_feedback'] = []
        detailed_feedback = user_data['detailed3_feedback']

        for i in range(len(part3_questions)):
            question = part3_questions[i]
            user_answer = part3_answers[i]
            analysis_data = analysis3_list[i]
            user_voice_url = part3_voice_urls[i]

            prompt = f"Question: {question}\nUser Answer: {user_answer}\nAnalysis Data: {analysis_data}\n\n"
            prompt += """Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance. Generate useful feedback that helps the user understand their performance in depth and should be in details.
            You should organize it in a clear and good feedback to the IELTS Candidate. They expect to know their mistakes and improve next time. Also, it is a good idea to write the question in the feedback and the answer and then start your analysis.
            """

            feedback = await generate_feedback_with_llm(prompt, context)
            detailed_feedback.append(feedback)
            user_data.setdefault('detailed_feedback3_list', []).append(feedback)
            await send_long_message(update, context, feedback)

            # Use asyncio.to_thread for CPU-bound operations
            await asyncio.to_thread(generate_pronunciation_visualization2, analysis_data)

            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)

            await context.bot.send_message(chat_id=update.effective_chat.id, text="Now, let's compare your pronunciation with a native speaker's pronunciation.")

            if user_voice_url:
                response = await asyncio.to_thread(requests.get, user_voice_url)
                voice_filename = f"user_voice_{i+1}.oga"
                with open(voice_filename, "wb") as file:
                    file.write(response.content)

                await context.bot.send_message(chat_id=update.effective_chat.id, text="Your Voice:")
                with open(voice_filename, "rb") as user_voice_file:
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=user_voice_file)

                os.remove(voice_filename)

                slow_audio_path = await convert_answer_to_audio(user_answer, '-0.56', examiner_voice)

                if slow_audio_path:
                    with open(slow_audio_path, 'rb') as slow_audio_file:
                        await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Slow Speed):")
                        await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
                    os.remove(slow_audio_path)
                else:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to generate native speaker's voice (slow speed).")

                normal_audio_path = await convert_answer_to_audio(user_answer, '0', examiner_voice)

                if normal_audio_path:
                    with open(normal_audio_path, 'rb') as normal_audio_file:
                        await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Normal Speed):")
                        await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
                    os.remove(normal_audio_path)
                else:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to generate native speaker's voice (normal speed).")

        # Delete the waiting message
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        
        # Send the "What would you like to do next?" message with options
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        keyboard = [
            [InlineKeyboardButton("Retake Part 3", callback_data=f'{userID}part3_retake')],
            [InlineKeyboardButton("Translate", callback_data=f'{userID}part3_translate_detailed_feedback')],
            [InlineKeyboardButton("End the Test", callback_data=f'{userID}end_test')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)

        return detailed_feedback
    except Exception as e:
        text = ("🚨 part 3 detailed results function ", e)
        await error_handling(update, context,text)
async def _translate_and_send_part3_feedback(update, context, user_id, overall_feedback, waiting_message, reply_markup):
    try:
        translated_feedback = await translate_feedback(user_id, overall_feedback, update, context)
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        if translated_feedback:
            await send_long_message(update, context, translated_feedback)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, translation is not available for your language at the moment. We are working on adding support for more languages.")
    except Exception as e:
        print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} | 🚨 Error in translation feedback 3: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, there was an error during translation. Please try again later.")
    finally:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
async def _translate_and_send_part3_detailed_feedback(update, context, user_id, detailed_feedback_list, waiting_message, reply_markup):
    try:
        user_data = context.user_data.get('user_data', {})
        translated_feedback = user_data['translated_feedback3']
        for feedback in detailed_feedback_list:
            translated_msg = await translate_feedback(user_id, feedback, update, context)
            if translated_msg:
                translated_feedback.append(translated_msg)
        
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        
        if translated_feedback:
            for feedback in translated_feedback:
                await asyncio.sleep(2)
                await send_long_message(update, context, feedback)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, translation is not available for your language at the moment. We are working on adding support for more languages.")
    
    except Exception as e:
        print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} | 🚨 Error in detailed feedback3 translation: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, there was an error during translation. Please try again later.")
    
    finally:
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
#-----------------MOCK TEST FUNCTIONS-----------------
async def start_mock_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})

        # # Clear all previous data
        # user_data.clear()

        # # Initialize all necessary lists and dictionaries
        # user_data['questions_list'] = []
        # user_data['answers_list'] = []
        # user_data['detailed_feedback_list'] = []
        # user_data['voice_urls'] = []
        # user_data['questions'] = []
        # user_data['analysis_list'] = []
        # user_data['list_previous_questions'] = []
        # user_data['list_previous_answers'] = []

        # user_data['part2_voice_urls'] = []
        # user_data['part2_questions'] = []
        # user_data['part2_answers'] = []
        # user_data['analysis2_list'] = []
        # user_data['detailed_feedback2_list'] = []

        # user_data['part3_voice_urls'] = []
        # user_data['part3_questions'] = []
        # user_data['part3_answers'] = []
        # user_data['analysis3_list'] = []
        # user_data['detailed_feedback3_list'] = []

        # user_data['mock_part1_questions'] = []
        # user_data['mock_part1_answers'] = []
        # user_data['mock_part1_voice_urls'] = []

        # user_data['mock_part2_questions'] = []
        # user_data['mock_part2_answers'] = []
        # user_data['mock_part2_voice_urls'] = []

        # user_data['mock_part3_questions'] = []
        # user_data['mock_part3_answers'] = []
        # user_data['mock_part3_voice_urls'] = []
        await user_data_update(update,context)
        # Define part_1_topics
        user_data['part_1_topics'] = [
            "Study 📚", "Work 💼", "Hometown 🏡", "Home/ Accommodation 🏘️", "Family 👨‍👩‍👧‍👦",
            "Friends 👥", "Clothes 👕", "Fashion 👗", "Gifts 🎁", "Daily routine 📅",
            "Daily activities 🏃‍♂️", "Food/ Cooking 🍳", "Drinks 🥤", "Going out 🎉", "Hobbies 🎨",
            "Language 🌐", "Leisure time activity ⏰", "Sports ⚽", "Future plan 🔮", "Music 🎵",
            "Newspapers 📰", "Pets 🐾", "Flowers & Plants 🌸", "Reading 📖", "Dancing 💃",
            "Exercise 💪", "Shopping 🛍️", "Magazines & TV 📺", "Travelling ✈️", "Interesting places 🏰",
            "Bicycle 🚲", "Seasons 🍂", "Maps 🗺️", "Internet & Technology 💻", "Weather ☀️",
            "Festivals 🎆", "Culture/ Tradition 🎭"
        ]

        # Randomly select a topic for Part 1 from the topics list
        selected_topic = random.choice(user_data['part_1_topics'])
        user_data['selected_topic'] = selected_topic

        # Generate initial questions for the selected topic using the Groq API
        initial_questions = await generate_questions(selected_topic,context)
        # print(initial_questions)
        if not initial_questions:
            text = "🚨 there is no questions have been intialized start mock test | initial_questions = await generate_questions(selected_topic,context)"
            await error_handling(update, context,text)
            return
        # print("user_data['questions'] b: ", user_data['questions'])
        del user_data['questions']
        try:
            print("user_data['questions'] atart mock test: ", user_data['questions'])
        except Exception as e:
            print(" ")
            pass
            
        # Store the initial questions in the mock_part1_questions list
        user_data['mock_part1_questions'] = initial_questions
        # print("user_data['mock_part1_questions'] ", user_data['mock_part1_questions'])
        # Send a message to the user indicating the start of the mock test
        await update.effective_message.reply_text(
            "The IELTS Speaking mock test will now begin.\n\n"
            "To help you achieve higher scores, please consider the following guidelines for your answers:\n\n"
            "🔹 In Part 1: Your answers should be between 10 - 30 seconds.\n"
            "🔹 In Part 2: Your answers should be between 1 - 2 minutes.\n"
            "🔹 In Part 3: Your answers should be between 30 seconds - 1 minute.\n\n"
            "Let's start!"
        )
        await asyncio.sleep(3)
        await update.effective_message.reply_text("Mock Test - Part 1")
        await asyncio.sleep(3)
        await mock_part1_process(update, context)
    except Exception as e:
        text = ("🚨 start mock test function ", e)
        await error_handling(update, context,text)
async def mock_part1_process(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        # Get the current question index from user_data
        current_question_index = user_data.get('mock_part1_current_question_index', 0)

        mock_part1_questions = user_data.get('mock_part1_questions', [])
        mock_part1_answers = user_data.get('mock_part1_answers', [])
        list_previous_questions = user_data.get('list_previous_questions', [])
        list_previous_answers = user_data.get('list_previous_answers', [])

        # Check if there are more questions to ask
        if current_question_index < len(mock_part1_questions):
            previous_question = mock_part1_questions[current_question_index - 1] if current_question_index > 0 else ""
            current_question = mock_part1_questions[current_question_index]
            user_answer = mock_part1_answers[current_question_index - 1] if current_question_index > 0 else ""
            selected_topic = user_data.get('selected_topic', '')

            list_previous_questions.append(previous_question)
            list_previous_answers.append(user_answer)

            # Generate an interactive question based on the previous question, user's answer, and the current question
            interactive_question = await generate_interactive_question(previous_question, user_answer, current_question, selected_topic, list_previous_questions, list_previous_answers, context)

            if interactive_question:
                # Replace the current question with the generated interactive question
                mock_part1_questions[current_question_index] = interactive_question
                current_question = interactive_question

            # Convert the current question to audio using text-to-speech
            
            question_audio_path = await convert_text_to_audio(current_question,examiner_voice)

            # Send the current question as voice message
            with open(question_audio_path, 'rb') as audio:
                await update.effective_message.reply_voice(voice=audio)

            # Send a message to the user to record their answer
            await update.effective_message.reply_text("Please record your answer.")

            # Set the flag to indicate that the user is answering a question in the mock test
            user_data['mock_part1_answering'] = True

            # Store the current question index in user_data for the voice answer handler
            user_data['mock_part1_current_question_index'] = current_question_index

            # Update the user_data with modified lists
            user_data['mock_part1_questions'] = mock_part1_questions
            user_data['list_previous_questions'] = list_previous_questions
            user_data['list_previous_answers'] = list_previous_answers

        else:
            # All questions have been asked, move to the next part
            await update.effective_message.reply_text("Mock Test - Part 1 completed. Moving to Part 2.")
            await asyncio.sleep(5)
            await mock_part2_process(update, context)

    except Exception as e:
        text = ("🚨 mock test part 1 process function ", e)
        await error_handling(update, context,text)
async def mock_part2_process(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        # Randomly select a topic from the ielts_questions dictionary
        selected_topic = random.choice(list(ielts_questions.keys()))

        # Extract the Part 2 question and Part 3 questions for the selected topic
        part2_question = ielts_questions[selected_topic]["part_2_question"]
        part3_questions = ielts_questions[selected_topic]["part_3_questions"]

        # Add the Part 2 question to the mock_part2_questions list
        user_data.setdefault('mock_part2_questions', []).append(part2_question)

        # # Clear the mock_part3_questions list before adding new questions
        user_data['mock_part3_questions'] = []

        # # Add the Part 3 questions to the mock_part3_questions list
        user_data['mock_part3_questions'] = part3_questions
        # print("user_data['mock_part3_questions']: ", user_data['mock_part3_questions'])
        # Convert the Part 2 question to audio using text-to-speech
        
        question_audio_path = await convert_text_to_audio(part2_question,examiner_voice)

        # Send the Part 2 question as both text and audio to the user
        await update.effective_message.reply_text(part2_question)
        with open(question_audio_path, 'rb') as audio:
            await update.effective_message.reply_voice(voice=audio)

        # Send a message to the user indicating the preparation time
        preparation_message = await update.effective_message.reply_text("You have 1 minute to prepare your answer.")
        animated_emoji = "⏳"  # Hourglass emoji
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)

        # Start a countdown timer for 1 minute
        countdown_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="60 seconds remaining...")

        for remaining in range(59, 0, -1):
            await asyncio.sleep(1)
            try:
                await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=countdown_message.message_id, text=f"{remaining} seconds remaining...")
            except Exception as e:
                print(f"Failed to update countdown message: {e}")
        # for remaining in range(59, 0, -1):
        #     if user_data['test_stop']:
        #         print(f"{update.effective_user.id} Countdown stopped")
        #         break
        #     await asyncio.sleep(1)
        #     try:
        #         await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=countdown_message.message_id, text=f"{remaining} seconds remaining...")
        #     except Exception as e:
        #         print(f"{update.effective_user.id} 🚨 Failed to update countdown message: {e}")
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=countdown_message.message_id)
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=preparation_message.message_id)
        # del user_data['mock_part1_answering'] 
        # Set the flag to indicate that the user is answering a question in the mock test Part 2
        user_data['mock_part2_answering'] = True

        # Send a message to the user to record their answer
        await update.effective_message.reply_text("Please record your answer.")

    except Exception as e:
        text = ("🚨 mock test part 2 process function ", e)
        await error_handling(update, context,text)
async def mock_part3_process(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        # Get the current question index from user_data
        current_question_index = user_data.get('mock_part3_current_question_index', 0)

        # Get the mock_part3_questions from user_data
        mock_part3_questions = user_data.get('mock_part3_questions', [])

        # Check if there are more questions to ask
        if current_question_index < len(mock_part3_questions):
            current_question = mock_part3_questions[current_question_index]
            
            # Convert the current question to audio using text-to-speech
            question_audio_path = await convert_text_to_audio(current_question,examiner_voice)

            # Send the current question as voice message
            with open(question_audio_path, 'rb') as audio:
                await update.effective_message.reply_voice(voice=audio)

            # Send a message to the user to record their answer
            await update.effective_message.reply_text("Please record your answer.")

            # Set the flag to indicate that the user is answering a question in the mock test Part 3
            user_data['mock_part3_answering'] = True

            # Store the current question index in user_data for the voice answer handler
            user_data['mock_part3_current_question_index'] = current_question_index

        else:
            await mock_test_completed(update, context)
            # All questions have been asked, show the inline keyboard markup for next steps
            user_data = context.user_data['user_data']
            user_id = update.effective_user.id
            user_data['user_id'] = str(user_id)
            userID = user_data['user_id']
            keyboard = [
                [InlineKeyboardButton("Check Your Score ✅", callback_data=f'{userID}mock_test_show_results')],
                [InlineKeyboardButton("Retake Mock Test", callback_data=f'{userID}mock_test_retake')],
                [InlineKeyboardButton("End Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.effective_message.reply_text("Mock Test - Part 3 completed. What would you like to do next?", reply_markup=reply_markup)

    except Exception as e:
        text = ("🚨 mock test part 3 process function ", e)
        await error_handling(update, context,text)
async def get_voice_file_path_url(voice_message: Voice):
    # try:
        # Get the file path URL of the voice message
        voice_file = await voice_message.get_file()
        voice_file_path_url = voice_file.file_path
        # print("voice file path url: ", voice_file_path_url)
        
        return voice_file_path_url
    # except Exception as e:
    #     print("get voice file path function ",e)
    #     # await update.message.reply_text(issue_message)
    #     await error_handling(update, context,text)
async def mock_test_retake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})

        query = update.callback_query
        await query.answer()

        # Clear the mock test lists
        user_data['mock_part1_questions'] = []
        user_data['mock_part1_answers'] = []
        user_data['mock_part1_voice_urls'] = []
        
        user_data['mock_part2_questions'] = []
        user_data['mock_part2_answers'] = []
        user_data['mock_part2_voice_urls'] = []
        
        user_data['mock_part3_questions'] = []
        user_data['mock_part3_answers'] = []
        user_data['mock_part3_voice_urls'] = []

        # Reset the current question indices
        user_data['mock_part1_current_question_index'] = 0
        user_data['mock_part3_current_question_index'] = 0

        # Randomly select a new topic for Part 1 from the topics list
        user_data['part_1_topics'] = [
            "Study 📚", "Work 💼", "Hometown 🏡", "Home/ Accommodation 🏘️", "Family 👨‍👩‍👧‍👦",
            "Friends 👥", "Clothes 👕", "Fashion 👗", "Gifts 🎁", "Daily routine 📅",
            "Daily activities 🏃‍♂️", "Food/ Cooking 🍳", "Drinks 🥤", "Going out 🎉", "Hobbies 🎨",
            "Language 🌐", "Leisure time activity ⏰", "Sports ⚽", "Future plan 🔮", "Music 🎵",
            "Newspapers 📰", "Pets 🐾", "Flowers & Plants 🌸", "Reading 📖", "Dancing 💃",
            "Exercise 💪", "Shopping 🛍️", "Magazines & TV 📺", "Travelling ✈️", "Interesting places 🏰",
            "Bicycle 🚲", "Seasons 🍂", "Maps 🗺️", "Internet & Technology 💻", "Weather ☀️",
            "Festivals 🎆", "Culture/ Tradition 🎭"
        ]
        selected_topic = random.choice(user_data['part_1_topics'])
        user_data['selected_topic'] = selected_topic

        # Generate new initial questions for the selected topic using the Groq API
        initial_questions = await generate_questions(selected_topic, context)
        if not initial_questions:
            text = "🚨 there is no questions have been intialized mock test retake | initial_questions = await generate_questions(selected_topic,context)"
            await error_handling(update, context,text)
            return

        # Store the new initial questions in the mock_part1_questions list
        user_data['mock_part1_questions'] = initial_questions

        # Edit the message to indicate the start of the new mock test
        await query.edit_message_text("Retaking the Mock Test - Part 1")
        # await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Topic: {selected_topic}")
        # await context.bot.send_message(chat_id=update.effective_chat.id, text="The mock test will now begin again. Please answer the following questions.")

        # Call the mock_part1_process function to start Part 1
        await mock_part1_process(update, context)

    except Exception as e:
        text = ("🚨 mock test retake function ", e)
        await error_handling(update, context,text)
async def send_user_answers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})

        # Send Part 1 answers
        mock_part1_questions = user_data.get('mock_part1_questions', [])
        mock_part1_answers = user_data.get('mock_part1_answers', [])
        part1_answers_text = "Mock Test - Part 1 Answers:\n\n"
        # for i, (question, answer) in enumerate(zip(mock_part1_questions, mock_part1_answers)):
        #     part1_answers_text += f"Question {i+1}: {question}\n\nAnswer: {answer}\n\n"
        # await context.bot.send_message(chat_id=update.effective_chat.id, text=part1_answers_text)
        typical_answers = await generate_typical_answers(mock_part1_questions, mock_part1_answers,context)
        await send_long_message(update, context, f"Typical Answers:\n\n{typical_answers}")
        # Send Part 2 answer
        mock_part2_questions = user_data.get('mock_part2_questions', [])
        mock_part2_answers = user_data.get('mock_part2_answers', [])
        if mock_part2_questions and mock_part2_answers:
            part2_answer_text = "Mock Test - Part 2 Answer:\n\n"
            question = mock_part2_questions[0]
            answer = mock_part2_answers[0]
            part2_answer_text += f"Question: {question}\n\nAnswer: {answer}\n\n"
            await context.bot.send_message(chat_id=update.effective_chat.id, text=part2_answer_text)

        # Send Part 3 answers
        mock_part3_questions = user_data.get('mock_part3_questions', [])
        mock_part3_answers = user_data.get('mock_part3_answers', [])
        part3_answers_text = "Mock Test - Part 3 Answers:\n\n"
        # for i, (question, answer) in enumerate(zip(mock_part3_questions, mock_part3_answers)):
        #     part3_answers_text += f"Question {i+1}: {question}\n\nAnswer: {answer}\n\n"
        # await context.bot.send_message(chat_id=update.effective_chat.id, text=part3_answers_text)
        typical_answers = await generate_typical_answers(mock_part3_questions, mock_part3_answers,context)
        await send_long_message(update, context, f"Typical Answers:\n\n{typical_answers}")
    except Exception as e:
        text = ("🚨 send user answers function ", e)
        await error_handling(update, context,text)
async def mock_test_completed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})

        # Send a waiting message to the user
        waiting_message = await update.effective_message.reply_text("Please wait while your answers and conversation are being prepared...")
        animated_emoji = "⏳"  # Hourglass emoji
        waiting_message2 = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")

        # Prepare the user's answers in text
        await send_user_answers(update, context)
        
        # Delete the waiting messages
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message2.message_id)
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)

        # Optionally, you can store the completion status in user_data
        user_data['mock_test_completed'] = True

        # Send a message to the user indicating that the answers are ready
        # await update.effective_message.reply_text("Your answers are ready!")

    except Exception as e:
        text = ("🚨 mock test completed function ", e)
        await error_handling(update, context,text)
async def show_mock_test_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print(f"{update.effective_user.id} ------------------------------mock test feedback ---------------------------------------")
        user_data = context.user_data.setdefault('user_data', {})

        mock_part1_questions = user_data.get('mock_part1_questions', [])
        mock_part1_answers = user_data.get('mock_part1_answers', [])
        mock_part1_voice_urls = user_data.get('mock_part1_voice_urls', [])

        mock_part2_questions = user_data.get('mock_part2_questions', [])
        mock_part2_answers = user_data.get('mock_part2_answers', [])
        mock_part2_voice_urls = user_data.get('mock_part2_voice_urls', [])

        mock_part3_questions = user_data.get('mock_part3_questions', [])
        mock_part3_answers = user_data.get('mock_part3_answers', [])
        mock_part3_voice_urls = user_data.get('mock_part3_voice_urls', [])

        await score_voice(update, context)
        targeted_score = user_data.get('targeted_score', 7)

        print(f"{update.effective_user.id} user_targeted_score: ", targeted_score)

        # Send waiting messages and progress bar
        animated_emoji = "⏳"
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        progress_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Wait a few minutes until results are ready...\n\n[                             ] 0%")

        # Share message (unchanged)
        share_message = f"Try the IELTS Speaking Bot! It simulates the IELTS speaking test and offers detailed feedback on your speaking skills and estimated band score. Try it for free: https://t.me/ielts_speakingAI_bot"
        keyboard = [[InlineKeyboardButton("Share the Bot", switch_inline_query=share_message)]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await ask_channel(update, context)
        share_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="While waiting for the results, would you like to share this bot? 😊",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        # Progress tracking
        steps = ["Transcribing answers...", "Analyzing responses...", "Generating feedback...", "Compiling results..."]
        total_steps = len(steps) + len(mock_part1_questions) + len(mock_part3_questions) + 1
        current_step = 0

        def update_progress():
            nonlocal current_step
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_bar = "[" + "█" * (progress // 5) + " " * (20 - (progress // 5)) + "]"
            return progress, progress_bar

        # Initialize lists in user_data
        user_data['mock_scores_list1'] = []
        user_data['mock_scores_list3'] = []

        # Assess Part 1
        for i, (audio_url, question_prompt) in enumerate(zip(mock_part1_voice_urls, mock_part1_questions)):
            processed_scores = await assess_part1_mock_async(audio_url, question_prompt, "ielts_part1", context)
            if processed_scores:
                user_data['mock_scores_list1'].append(processed_scores)
                print(f"{update.effective_user.id} Assessment successful for mock Part 1 answer {i+1}")
            else:
                print(f"🚨ID: {update.effective_user.id} Username: {update.effective_user.username} Assessment failed for mock Part 1 answer {i+1}")
                print(f"Voice URL: {audio_url}")
                print(f"Question: {question_prompt}")
            
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Wait a few minutes until results are ready...\n{progress_bar} {progress}%"
            )

        # Assess Part 2
        if mock_part2_voice_urls and mock_part2_questions:
            part2_scores = await assess_part2_mock_async(mock_part2_voice_urls[0], mock_part2_questions[0], "ielts_part2", context)
        else:
            part2_scores = None
        if part2_scores:
                # scores2_list.append(processed_scores)
            print(f"{update.effective_user.id} Assessment successful for answer part 2 ")
        else:
            print(f"🚨ID: {update.effective_user.id} Username: {update.effective_user.username} Assessment failed for answer part 2 ")
            print(f"Voice URL: {mock_part2_voice_urls[0]}")
            print(f"Question: {mock_part2_questions[0]}")
        progress, progress_bar = update_progress()
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=progress_message.message_id,
            text=f"Processing your mock test results...\n{progress_bar} {progress}%"
        )

        # Assess Part 3
        for i, (audio_url, question_prompt) in enumerate(zip(mock_part3_voice_urls, mock_part3_questions)):
            processed_scores = await assess_part3_mock_async(audio_url, question_prompt, "ielts_part3", context)
            if processed_scores:
                user_data['mock_scores_list3'].append(processed_scores)
                print(f"{update.effective_user.id} Assessment successful for mock Part 3 answer {i+1}")
            else:
                # print(audio_url, question_prompt)
                print(f"🚨ID: {update.effective_user.id} Username: {update.effective_user.username} Assessment failed for mock Part 3 answer {i+1}")
                print(f"Voice URL: {audio_url}")
                print(f"Question: {question_prompt}")
            
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Processing your mock test results...\n{progress_bar} {progress}%"
            )

        if user_data['mock_scores_list1'] and part2_scores and user_data['mock_scores_list3']:
            # Calculate average scores
            part1_avg_scores = calculate_average_scores(user_data['mock_scores_list1'], len(user_data['mock_scores_list1']))
            part2_avg_scores = part2_scores
            part3_avg_scores = calculate_average_scores(user_data['mock_scores_list3'], len(user_data['mock_scores_list3']))

            overall_avg_scores = calculate_overall_average_scores(part1_avg_scores, part2_avg_scores, part3_avg_scores)

            # Round scores
            for key in overall_avg_scores:
                overall_avg_scores[key] = round_to_ielts_score(overall_avg_scores[key])

            mock_score = overall_avg_scores['overall']

            # Generate overall feedback
            overall_mock_feedback = await generate_overall_feedback(
                user_data['mock_scores_list1'], part2_scores, user_data['mock_scores_list3'],
                mock_part1_questions, mock_part1_answers,
                mock_part2_questions[0] if mock_part2_questions else "", mock_part2_answers[0] if mock_part2_answers else "",
                mock_part3_questions, mock_part3_answers, mock_score, targeted_score,context
            )

            user_data['overall_mock_test_feedback'] = overall_mock_feedback

            # Update progress and delete messages
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Wait a few minutes until results are ready...\n{progress_bar} 100%"
            )
            for msg in [progress_message, share_message, waiting_message]:
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=msg.message_id)

            # Send feedback and scores
            await send_long_message(update, context, overall_mock_feedback)
            try:
                await send_recommendations(update, context, overall_mock_feedback)
            except Exception as e:
                print(e)
            await display_feedback(update, context, overall_avg_scores['overall'], overall_avg_scores['pronunciation'],
                                   overall_avg_scores['fluency'], overall_avg_scores['grammar'], overall_avg_scores['vocabulary'])

            band_score = f"Your estimated IELTS band score for the mock test is: {overall_avg_scores['overall']:.1f}"
            await context.bot.send_message(chat_id=update.effective_chat.id, text=band_score)

            cefr_level = get_cefr_level(overall_avg_scores['overall'])
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Your language level is: {cefr_level}")

            await append_speaking_score(update, context, "mock_test", mock_score)
            await increment_practice_count(update, context)
            
            userID = str(update.effective_user.id)
            user_data['user_id'] = userID
            
            # Provide user options
            keyboard = [
                [InlineKeyboardButton("Show Detailed Results", callback_data=f'{userID}mock_test_detailed_results')],
                [InlineKeyboardButton("Translate", callback_data=f'{userID}mock_test_translate_feedback')],
                [InlineKeyboardButton("Retake Mock Test", callback_data=f'{userID}mock_test_retake')],
                [InlineKeyboardButton("End Test", callback_data=f'{userID}end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
        else:
            raise Exception("🚨 Failed to assess one or more parts of the mock test")
    except Exception as e:
        text = ("🚨 show mock test result function ", e)
        await error_handling(update, context,text)
async def assess_part1_mock_async(audio_urls, question_prompts, task_type, context: ContextTypes.DEFAULT_TYPE):
    user_data = context.user_data.setdefault('user_data', {})
    try:
        # Download the voice file from the URL
        response = requests.get(audio_urls)
        
        filename = f"voice_{int(time.time())}.oga"
        with open(filename, "wb") as file:
            file.write(response.content)

        # print("user_data['part_1_minute']: ", user_data['part_1_minute'])

        # loop = asyncio.get_running_loop()
        # if user_data['part_1_minute']:
        #     print("use speech super API Part1 mock test")
        #     scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompts, task_type)
        #     response_json = scores
        # else:
        #     scores, analysis_data = await loop.run_in_executor(executor, assess_speech, filename, question_prompts)
        loop = asyncio.get_running_loop()
        # print("user_data['part_1_minute']: ",one_minute1)
        # print("user_data['part_1_minute_part_1']: ",user_data['part_1_minute_part_1'])
        try:
            print("use speech super assess_speech3 API (mock test part 1)")
            scores, analysis_data = await loop.run_in_executor(executor, assess_speech3, filename, question_prompts, task_type)
            response_json = scores
        except Exception as e:
            print("🚨 Error on assess_speech3 speech super API and switching to Speech ace API ", e)
            try:
                scores, analysis_data = await loop.run_in_executor(executor, assess_speech, filename, question_prompts)
            except Exception as e:
                print("🚨Error on Speech Ace API and switching to Speech Super API assess_speech2 ", e)
                scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompts, task_type)
                response_json = scores
        if scores is None or analysis_data is None:
            raise Exception("🚨 Assessment failed")

        # Immediately append analysis_data to the analysis_list
        mock_part1_analysis_list = user_data['mock_part1_analysis_list']
        mock_part1_analysis_list.append(analysis_data)

        # if user_data['part_1_minute']:
        try:
            if 'result' in response_json:
                result = response_json['result']
                scores = {
                    "overall": result.get("overall", 0),
                    "pronunciation": result.get("pronunciation", 0),
                    "fluency": result.get("fluency_coherence", 0),
                    "grammar": result.get("grammar", 0),
                    "vocabulary": result.get("lexical_resource", 0),
                    "relevance": result.get("relevance", 0),
                    "transcription": result.get("transcription", 0),
                    # "pause_filler": result.get("pause_filler", {}),
                #     "sentences": [
                #         {
                #             "sentence": sentence.get("sentence", ""),
                #             "pronunciation": sentence.get("pronunciation", "N/A"),
                #             "grammar": sentence.get("grammar", {}).get("corrected", "")
                #         }
                #         for sentence in result.get("sentences", [])
                #     ]
                }
            os.remove(filename)
            print(len(user_data['mock_part1_analysis_list']))
            return scores
        # else:
        except Exception as e:
            print("🚨 assess_part1 mock test", e)
            processed_scores = {
                "overall": scores['ielts_score']['overall'],
                "pronunciation": scores['ielts_score']['pronunciation'],
                "fluency": scores['ielts_score']['fluency'],
                "grammar": scores['ielts_score']['grammar'],
                "vocabulary": scores['ielts_score']['vocabulary'],
                "relevance": scores['relevance']['class'],
                "transcription": scores['transcription'],
            }

            os.remove(filename)
            print(len(user_data['mock_part1_analysis_list']))
            return processed_scores
    except Exception as e:
        print(f"🚨 Error assessing speech: {str(e)}")
        return None
async def assess_part2_mock_async(audio_url, question_prompt, task_type,context: ContextTypes.DEFAULT_TYPE):
   
    try:
        # Download the voice file from the URL
        user_data = context.user_data.setdefault('user_data', {})
        response = requests.get(audio_url)
        
        # Generate a unique filename for the downloaded file
        filename = f"voice_{int(time.time())}.oga"
        
        # Save the downloaded file locally
        with open(filename, "wb") as file:
            file.write(response.content)
        
        # loop = asyncio.get_running_loop()
        # scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompt, task_type)
        scores,analysis_data =(assess_speech2(filename, question_prompt,task_type))
        if scores is None or analysis_data is None:
            raise Exception("🚨 Assessment failed")
        response_json = scores  # Assuming assess_speech returns the JSON response directly
        mock_part2_analysis_list = user_data['mock_part2_analysis_list']
        mock_part2_analysis_list.append(analysis_data)
        # print (analysis_data)
        if 'result' in response_json:
            result = response_json['result']
            scores = {
                "overall": result.get("overall", 0),
                "pronunciation": result.get("pronunciation", 0),
                "fluency": result.get("fluency_coherence", 0),
                "grammar": result.get("grammar", 0),
                "vocabulary": result.get("lexical_resource", 0),
                "relevance": result.get("relevance", 0),
                "transcription": result.get("transcription", 0),
                "pause_filler": result.get("pause_filler", {}),
                "sentences": [
                    {
                        "sentence": sentence.get("sentence", ""),
                        "pronunciation": sentence.get("pronunciation", "N/A"),
                        "grammar": sentence.get("grammar", {}).get("corrected", "")
                    }
                    for sentence in result.get("sentences", [])
                ]
            }
            # analysis_list.append(scores)
        
        # end_time = time.time()
        # execution_time = end_time - start_time
        # print(f"Execution time: {execution_time} seconds")
        os.remove(filename)
        print("analysis_data part 2 mock test: ",len(user_data['mock_part2_analysis_list']))
        return scores
        # processed_scores = {
        #     "overall": scores['ielts_score']['overall'],
        #     "pronunciation": scores['ielts_score']['pronunciation'],
        #     "fluency": scores['ielts_score']['fluency'],
        #     "grammar": scores['ielts_score']['grammar'],
        #     "vocabulary": scores['ielts_score']['vocabulary'],
        #     "relevance": scores['relevance']['class'],
        #     "transcription": scores['transcription'],
        #     # "sentences": []  # SpeechAce doesn't provide sentence-level data in the same format
        # }

        # # Add word-level details to the processed_scores
        # # processed_scores["word_details"] = analysis_data["word_pronunciation_details"]

        # # Clean up the temporary file
        # os.remove(filename)
        # # print(processed_scores)
        # return processed_scores
    except Exception as e:
        print(f"🚨 Error assessing speech: {str(e)}")
        return None
async def assess_part3_mock_async(audio_urls, question_prompts, task_type, context: ContextTypes.DEFAULT_TYPE):
    user_data = context.user_data.setdefault('user_data', {})
    try:
        # Download the voice file from the URL
        response = requests.get(audio_urls)
        
        filename = f"voice_{int(time.time())}.oga"
        with open(filename, "wb") as file:
            file.write(response.content)

        # print("user_data['part_3_minute']: ", user_data['part_3_minute'])

        # loop = asyncio.get_running_loop()
        # if user_data['part_3_minute']:
        #     print("use speech super API Part3 mock test")
        #     scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompts, task_type)
        #     response_json = scores
        # else:
        #     scores, analysis_data = await loop.run_in_executor(executor, assess_speech, filename, question_prompts)
        loop = asyncio.get_running_loop()
        # print("user_data['part_1_minute']: ",one_minute1)
        # print("user_data['part_1_minute_part_1']: ",user_data['part_1_minute_part_1'])
        try:
            print("use speech super assess_speech3 API (mock test part 3)")
            scores, analysis_data = await loop.run_in_executor(executor, assess_speech3, filename, question_prompts, task_type)
            response_json = scores
        except Exception as e:
            print("🚨 Error on assess_speech3 speech super API and switching to Speech ace API ", e)
            try:
                scores, analysis_data = await loop.run_in_executor(executor, assess_speech, filename, question_prompts)
            except Exception as e:
                print("🚨Error on Speech Ace API and switching to Speech Super API assess_speech2 ", e)
                scores, analysis_data = await loop.run_in_executor(executor, assess_speech2, filename, question_prompts, task_type)
                response_json = scores
        if scores is None or analysis_data is None:
            raise Exception("🚨 Assessment failed")

        # Immediately append analysis_data to the analysis_list
        mock_part3_analysis_list = user_data['mock_part3_analysis_list']
        mock_part3_analysis_list.append(analysis_data)

        # if user_data['part_3_minute']:
        try:
            if 'result' in response_json:
                result = response_json['result']
                scores = {
                    "overall": result.get("overall", 0),
                    "pronunciation": result.get("pronunciation", 0),
                    "fluency": result.get("fluency_coherence", 0),
                    "grammar": result.get("grammar", 0),
                    "vocabulary": result.get("lexical_resource", 0),
                    "relevance": result.get("relevance", 0),
                    "transcription": result.get("transcription", 0),
                    # "pause_filler": result.get("pause_filler", {}),
                #     "sentences": [
                #         {
                #             "sentence": sentence.get("sentence", ""),
                #             "pronunciation": sentence.get("pronunciation", "N/A"),
                #             "grammar": sentence.get("grammar", {}).get("corrected", "")
                #         }
                #         for sentence in result.get("sentences", [])
                #     ]
                }
            
            os.remove(filename)
            print(len(user_data['mock_part3_analysis_list']))
            return scores
        # else:
        except Exception as e:
            processed_scores = {
                "overall": scores['ielts_score']['overall'],
                "pronunciation": scores['ielts_score']['pronunciation'],
                "fluency": scores['ielts_score']['fluency'],
                "grammar": scores['ielts_score']['grammar'],
                "vocabulary": scores['ielts_score']['vocabulary'],
                "relevance": scores['relevance']['class'],
                "transcription": scores['transcription'],
            }

            os.remove(filename)
            print(len(user_data['mock_part3_analysis_list']))
            return processed_scores
    except Exception as e:
        print(f"🚨 Error assessing speech: {str(e)}")
        return None
async def generate_overall_feedback(part1_scores, part2_scores, part3_scores, part1_questions, part1_answers, part2_question, part2_answer, part3_questions, part3_answers, mock_score,targeted_score,context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})
        # Prepare the prompt for generating overall feedback
        # global targeted_score
        prompt = "Generate an overall feedback message for an IELTS Speaking mock test based on the following scores, questions, and answers:\n\n"
        
        # Add Part 1 scores, questions, and answers to the prompt
        prompt += "Part 1:\n"
        for i, (score, question, answer) in enumerate(zip(part1_scores, part1_questions, part1_answers)):
            prompt += f"Question {i+1}: {question}\n"
            prompt += f"Answer: {answer}\n"
            prompt += f"Scores:\n"
            prompt += f"  Overall: {score['overall']}\n"
            prompt += f"  Pronunciation: {score['pronunciation']}\n"
            prompt += f"  Fluency: {score['fluency']}\n"
            prompt += f"  Grammar: {score['grammar']}\n"
            prompt += f"  Vocabulary: {score['vocabulary']}\n\n"
        
        # Add Part 2 scores, question, and answer to the prompt
        prompt += "Part 2:\n"
        prompt += f"Question: {part2_question}\n"
        prompt += f"Answer: {part2_answer}\n"
        prompt += f"Scores:\n"
        prompt += f"  Overall: {part2_scores['overall']}\n"
        prompt += f"  Pronunciation: {part2_scores['pronunciation']}\n"
        prompt += f"  Fluency: {part2_scores['fluency']}\n"
        prompt += f"  Grammar: {part2_scores['grammar']}\n"
        prompt += f"  Vocabulary: {part2_scores['vocabulary']}\n\n"
        
        # Add Part 3 scores, questions, and answers to the prompt
        prompt += "Part 3:\n"
        for i, (score, question, answer) in enumerate(zip(part3_scores, part3_questions, part3_answers)):
            prompt += f"Question {i+1}: {question}\n"
            prompt += f"Answer: {answer}\n"
            prompt += f"Scores:\n"
            prompt += f"  Overall: {score['overall']}\n"
            prompt += f"  Pronunciation: {score['pronunciation']}\n"
            prompt += f"  Fluency: {score['fluency']}\n"
            prompt += f"  Grammar: {score['grammar']}\n"
            prompt += f"  Vocabulary: {score['vocabulary']}\n\n"
        
        # prompt += "Based on the scores, questions, and answers above, provide an overall feedback message for the IELTS Speaking mock test. The feedback should include:\n"
        # prompt += "- An assessment of the candidate's overall performance across all three parts\n"
        # prompt += "- Strengths and areas for improvement in pronunciation, fluency, grammar, and vocabulary\n"
        # prompt += "- Specific examples or observations from the candidate's responses to support the feedback\n"
        # prompt += f"- Encouragement and suggestions for further practice and improvement and you should provide a clear and good feedback to the IELTS Candidate he expects to kmow his mistakes and improve next time and reach his aimed ielts speaking {targeted_score}\n\n"
        # prompt += f"The feedback should be comprehensive, constructive, and tailored to the candidate's performance based on the provided scores and the IELTS Speaking  Band score for the mock test is {mock_score}, questions, and answers."
        prompt += f"""
            Please provide a detailed and constructive feedback report for the IELTS Speaking mock test, considering the candidate's performance across all three parts. The feedback should include the following aspects:

            1. Overall Performance:
            - Assessment of the candidate's performance in Parts 1, 2, and 3
            - IELTS Speaking Band score for the mock test: {mock_score}
            - Strengths and areas for improvement across all parts
            - Comparison to the target score of {targeted_score}

            2. Pronunciation:
            - Evaluation of clarity, intelligibility, stress, rhythm, and intonation
            - Specific examples or observations from the candidate's responses
            - Areas for improvement and targeted practice suggestions

            3. Fluency and Coherence:
            - Assessment of the smooth flow of speech, hesitation, repetition, and self-correction
            - Logical development and organization of ideas
            - Use of cohesive devices and discourse markers
            - Recommendations for enhancing fluency and coherence

            4. Grammatical Range and Accuracy:
            - Evaluation of the variety, complexity, and accuracy of grammatical structures
            - Specific examples of strengths and errors from the candidate's responses
            - Tips for improving grammatical accuracy and expanding the range of structures

            5. Lexical Resource (Vocabulary):
            - Assessment of the range, variety, accuracy, and appropriacy of vocabulary
            - Specific examples of effective vocabulary usage and areas for improvement
            - Suggestions for expanding vocabulary and using idiomatic language

            6. Interactive Communication (Part 3):
            - Evaluation of the candidate's ability to engage in a discussion and express opinions
            - Responsiveness to the examiner's questions and prompts
            - Ability to ask for clarification or elaboration when needed
            - Recommendations for improving interactive communication skills

            7. Encouragement and Suggestions:
            - Acknowledge the candidate's efforts and progress
            - Provide specific and actionable suggestions for further practice and improvement
            - Recommend resources, strategies, and activities to target the identified areas of weakness
            - Motivate the candidate to continue their IELTS Speaking preparation journey

            Please ensure that the feedback is comprehensive, constructive, and tailored to the candidate's performance based on the provided scores, questions, and answers. Use a supportive and encouraging tone throughout the feedback to boost the candidate's confidence and motivation.

            Organize the feedback in a clear and structured manner, using headings and bullet points for easy readability. Provide specific examples and observations from the candidate's responses to support the feedback and make it more meaningful and actionable.

            Emphasize the importance of regular practice, self-reflection, and seeking feedback to continuously improve their IELTS Speaking skills and reach their target score of {targeted_score}. Encourage the candidate to stay focused, persistent, and positive in their IELTS preparation journey.
            """
        # Use Groq to generate the overall feedback
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides feedback on IELTS Speaking mock tests."},
                {"role": "user", "content": prompt}
            ],
            model=groq_model,
        )
        
        # Extract the generated feedback from the response
        feedback = chat_completion.choices[0].message.content
        print("overall_feedback_generated")
        feedback = re.sub(r'\*', '', feedback)  # Remove asterisks (*)
        feedback = re.sub(r'#', '', feedback)  # Remove hash symbols (#)
        user_data['generate_feedback'] = []
        user_data['generate_feedback'] = feedback
        return user_data['generate_feedback']
    except Exception as e:
        text = ("🚨 Groq error switching to perplexity generate_overall_feedback ",e)
        
        messages = [
                {
                    "role": "system",
                    "content": (
                        ""
                       
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        prompt
                    ),
                },
            ]
        client = OpenAI(api_key=perplexity_API, base_url="https://api.perplexity.ai")
        response = client.chat.completions.create(
            model="llama-3.1-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        feedback = (response.choices[0].message.content)
        return feedback
def calculate_average_scores(scores,len_scores):
    # num_scores=  len(scores)
    # try:
        avg_scores = {
            'overall': sum(score['overall'] for score in scores) / len_scores,
            'pronunciation': sum(score['pronunciation'] for score in scores) / len_scores,
            'fluency': sum(score['fluency'] for score in scores) / len_scores,
            'grammar': sum(score['grammar'] for score in scores) / len_scores,
            'vocabulary': sum(score['vocabulary'] for score in scores) / len_scores
        }
        return avg_scores
    # except Exception as e:
    #     print("caculate average scores function ",e)
    #     # await update.message.reply_text(issue_message)
    #     await error_handling(update, context,text)
def calculate_overall_average_scores(part1_avg_scores, part2_avg_scores, part3_avg_scores):
    # try:
        overall_avg_scores = {
            'overall': (part1_avg_scores['overall'] + part2_avg_scores['overall'] + part3_avg_scores['overall']) / 3,
            'pronunciation': (part1_avg_scores['pronunciation'] + part2_avg_scores['pronunciation'] + part3_avg_scores['pronunciation']) / 3,
            'fluency': (part1_avg_scores['fluency'] + part2_avg_scores['fluency'] + part3_avg_scores['fluency']) / 3,
            'grammar': (part1_avg_scores['grammar'] + part2_avg_scores['grammar'] + part3_avg_scores['grammar']) / 3,
            'vocabulary': (part1_avg_scores['vocabulary'] + part2_avg_scores['vocabulary'] + part3_avg_scores['vocabulary']) / 3
        }
        return overall_avg_scores
    # except Exception as e:
    #     print("calculate overall average scores function ",e)
    #     await error_handling(update, context,text)
        # await update.message.reply_text(issue_message)

async def generate_detailed_feedback_part1(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})

        user_data['detailed_feedback_part1'] = []
        mock_part1_questions = user_data.get('mock_part1_questions', [])
        mock_part1_answers = user_data.get('mock_part1_answers', [])
        mock_part1_analysis_list = user_data.get('mock_part1_analysis_list', [])
        mock_part1_voice_urls = user_data.get('mock_part1_voice_urls', [])
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Detailed Feedback for Part 1:")

        for i in range(len(mock_part1_questions)):
            question = mock_part1_questions[i]
            user_answer = mock_part1_answers[i]
            analysis_data = mock_part1_analysis_list[i]
            user_voice_url = mock_part1_voice_urls[i]

            prompt = f"Question: {question}\nUser Answer: {user_answer}\nAnalysis Data: {analysis_data}\n\n"
            prompt += """Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance. Generate useful feedback that helps the user understand their performance in depth.
            You should organize it in a clear and good feedback to the IELTS Candidate. They expect to know their mistakes and improve next time. Also, it is a good idea to write the question in the feedback and the answer and then start your analysis.
            """

            feedback = await generate_feedback_with_llm(prompt,context)
            user_data['detailed_feedback_part1'].append(feedback)
            user_data.setdefault('mock_part1_detailed_feedback_list', []).append(feedback)
            await send_long_message(update, context, feedback)

            # Generate and send the pronunciation visualization image
            # if user_data['part_1_minute']:
            #     await asyncio.to_thread(generate_pronunciation_visualization2, analysis_data)
            # else:
            #     await asyncio.to_thread(generate_pronunciation_visualization, analysis_data)
            try:
                await asyncio.to_thread(generate_pronunciation_visualization2, analysis_data)
            except Exception as e:
                print("asyncio.to_thread(generate_pronunciation_visualization2, analysis_data): ", e)
                await asyncio.to_thread(generate_pronunciation_visualization, analysis_data)
            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)

            await context.bot.send_message(chat_id=update.effective_chat.id,
                                            text="Now, let's compare your pronunciation with a native speaker's pronunciation.")

            # Download the voice file from the URL
            response = await asyncio.to_thread(requests.get, user_voice_url)
            voice_filename = f"user_voice_part1_{i}.oga"
            with open(voice_filename, "wb") as file:
                file.write(response.content)

            # Send user's original voice
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Your Voice:")
            with open(voice_filename, "rb") as user_voice_file:
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=user_voice_file)

            # Delete the user's voice file from disk
            os.remove(voice_filename)

            # Generate native speaker's audio with slow speed
            slow_audio_path = await convert_answer_to_audio(user_answer, '-0.56', examiner_voice)

            # Send native speaker's audio with slow speed
            if slow_audio_path:
                with open(slow_audio_path, 'rb') as slow_audio_file:
                    await context.bot.send_message(chat_id=update.effective_chat.id,
                                                    text="Native Speaker's Voice (Slow Speed):")
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
                os.remove(slow_audio_path)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id,
                                                text="Failed to generate native speaker's voice (slow speed).")

            # Generate native speaker's audio with normal speed
            normal_audio_path = await convert_answer_to_audio(user_answer, '0', examiner_voice)

            # Send native speaker's audio with normal speed
            if normal_audio_path:
                with open(normal_audio_path, 'rb') as normal_audio_file:
                    await context.bot.send_message(chat_id=update.effective_chat.id,
                                                    text="Native Speaker's Voice (Normal Speed):")
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
                os.remove(normal_audio_path)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id,
                                                text="Failed to generate native speaker's voice (normal speed).")

            await context.bot.send_message(chat_id=update.effective_chat.id,
                                            text="Listen to your voice and compare it with the native speaker's pronunciation at both speeds.")

        return user_data['detailed_feedback_part1']
    except Exception as e:
        text = ("🚨 generate detailed feedback part 1 function ", e)
        await error_handling(update, context,text)
async def generate_detailed_feedback_part2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})

        user_data['detailed_feedback_part2'] = []
        mock_part2_questions = user_data.get('mock_part2_questions', [])
        mock_part2_answers = user_data.get('mock_part2_answers', [])
        mock_part2_analysis_list = user_data.get('mock_part2_analysis_list', [])
        mock_part2_voice_urls = user_data.get('mock_part2_voice_urls', [])
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Detailed Feedback for Part 2:")

        if mock_part2_questions and mock_part2_answers and mock_part2_analysis_list and mock_part2_voice_urls:
            question = mock_part2_questions[0]
            user_answer = mock_part2_answers[0]
            analysis_data = mock_part2_analysis_list[0]
            user_voice_url = mock_part2_voice_urls[0]

            prompt = f"Question: {question}\nUser Answer: {user_answer}\nAnalysis Data: {analysis_data}\n\n"
            prompt += """Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance. Generate useful feedback that helps the user understand their performance in depth.
            You should organize it in a clear and good feedback to the IELTS Candidate. They expect to know their mistakes and improve next time. Also, it is a good idea to write the question in the feedback and the answer and then start your analysis.
            """

            feedback = await generate_feedback_with_llm(prompt, context)
            user_data['detailed_feedback_part2'].append(feedback)
            user_data.setdefault('mock_part2_detailed_feedback_list', []).append(feedback)
            await send_long_message(update, context, feedback)

            # Generate and send the pronunciation visualization image
            await asyncio.to_thread(generate_pronunciation_visualization2, analysis_data)
            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)

            await context.bot.send_message(chat_id=update.effective_chat.id,
                                            text="Now, let's compare your pronunciation with a native speaker's pronunciation.")

            # Download the voice file from the URL
            response = await asyncio.to_thread(requests.get, user_voice_url)
            voice_filename = "user_voice_part2.oga"
            with open(voice_filename, "wb") as file:
                file.write(response.content)

            # Send user's original voice
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Your Voice:")
            with open(voice_filename, "rb") as user_voice_file:
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=user_voice_file)

            # Delete the user's voice file from disk
            os.remove(voice_filename)

            # Generate native speaker's audio with slow speed
            slow_audio_path = await convert_answer_to_audio(user_answer, '-0.56', examiner_voice)

            # Send native speaker's audio with slow speed
            if slow_audio_path:
                with open(slow_audio_path, 'rb') as slow_audio_file:
                    await context.bot.send_message(chat_id=update.effective_chat.id,
                                                    text="Native Speaker's Voice (Slow Speed):")
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
                os.remove(slow_audio_path)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id,
                                                text="Failed to generate native speaker's voice (slow speed).")

            # Generate native speaker's audio with normal speed
            normal_audio_path = await convert_answer_to_audio(user_answer, '0', examiner_voice)

            # Send native speaker's audio with normal speed
            if normal_audio_path:
                with open(normal_audio_path, 'rb') as normal_audio_file:
                    await context.bot.send_message(chat_id=update.effective_chat.id,
                                                    text="Native Speaker's Voice (Normal Speed):")
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
                os.remove(normal_audio_path)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id,
                                                text="Failed to generate native speaker's voice (normal speed).")

            await context.bot.send_message(chat_id=update.effective_chat.id,
                                            text="Listen to your voice and compare it with the native speaker's pronunciation at both speeds.")

        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="No data available for Part 2.")

        return user_data['detailed_feedback_part2']
    except Exception as e:
        text = ("🚨generate detailed feedback part 2 function ", e)
        await error_handling(update, context,text)
async def generate_detailed_feedback_part3(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_data = context.user_data.setdefault('user_data', {})

        user_data['detailed_feedback_part3'] = []
        mock_part3_questions = user_data.get('mock_part3_questions', [])
        mock_part3_answers = user_data.get('mock_part3_answers', [])
        mock_part3_analysis_list = user_data.get('mock_part3_analysis_list', [])
        mock_part3_voice_urls = user_data.get('mock_part3_voice_urls', [])
        await score_voice(update, context)
        examiner_voice = user_data['examiner_voice']
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Detailed Feedback for Part 3:")

        for i in range(len(mock_part3_questions)):
            question = mock_part3_questions[i]
            user_answer = mock_part3_answers[i]
            analysis_data = mock_part3_analysis_list[i]
            user_voice_url = mock_part3_voice_urls[i]

            prompt = f"Question: {question}\nUser Answer: {user_answer}\nAnalysis Data: {analysis_data}\n\n"
            prompt += """Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance. Generate useful feedback that helps the user understand their performance in depth.
            You should organize it in a clear and good feedback to the IELTS Candidate. They expect to know their mistakes and improve next time. Also, it is a good idea to write the question in the feedback and the answer and then start your analysis.
            """

            feedback = await generate_feedback_with_llm(prompt,context)
            user_data['detailed_feedback_part3'].append(feedback)
            user_data.setdefault('mock_part3_detailed_feedback_list', []).append(feedback)
            await send_long_message(update, context, feedback)

            # Generate and send the pronunciation visualization image
            # if user_data['part_3_minute']:
            #     await asyncio.to_thread(generate_pronunciation_visualization2, analysis_data)
            # else:
            #     await asyncio.to_thread(generate_pronunciation_visualization, analysis_data)
            try:
                await asyncio.to_thread(generate_pronunciation_visualization2, analysis_data)
            except Exception as e:
                print("asyncio.to_thread(generate_pronunciation_visualization2, analysis_data): ", e)
                await asyncio.to_thread(generate_pronunciation_visualization, analysis_data)
            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)

            await context.bot.send_message(chat_id=update.effective_chat.id,
                                            text="Now, let's compare your pronunciation with a native speaker's pronunciation.")

            # Download the voice file from the URL
            response = await asyncio.to_thread(requests.get, user_voice_url)
            voice_filename = f"user_voice_part3_{i}.oga"
            with open(voice_filename, "wb") as file:
                file.write(response.content)

            # Send user's original voice
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Your Voice:")
            with open(voice_filename, "rb") as user_voice_file:
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=user_voice_file)

            # Delete the user's voice file from disk
            os.remove(voice_filename)

            # Generate native speaker's audio with slow speed
            slow_audio_path = await convert_answer_to_audio(user_answer, '-0.56', examiner_voice)

            # Send native speaker's audio with slow speed
            if slow_audio_path:
                with open(slow_audio_path, 'rb') as slow_audio_file:
                    await context.bot.send_message(chat_id=update.effective_chat.id,
                                                    text="Native Speaker's Voice (Slow Speed):")
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
                os.remove(slow_audio_path)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id,
                                                text="Failed to generate native speaker's voice (slow speed).")

            # Generate native speaker's audio with normal speed
            normal_audio_path = await convert_answer_to_audio(user_answer, '0', examiner_voice)

            # Send native speaker's audio with normal speed
            if normal_audio_path:
                with open(normal_audio_path, 'rb') as normal_audio_file:
                    await context.bot.send_message(chat_id=update.effective_chat.id,
                                                    text="Native Speaker's Voice (Normal Speed):")
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
                os.remove(normal_audio_path)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id,
                                                text="Failed to generate native speaker's voice (normal speed).")

            await context.bot.send_message(chat_id=update.effective_chat.id,
                                            text="Listen to your voice and compare it with the native speaker's pronunciation at both speeds.")

        return user_data['detailed_feedback_part3']
    except Exception as e:
        text = ("🚨 generate detailed feedback part 3 mock test function ", e)
        await error_handling(update, context,text)
async def generate_mock_test_detailed_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print(f"{update.effective_user.id} detailed mock test feedback")
        user_data = context.user_data.setdefault('user_data', {})

        # Notify the user that detailed feedback generation is starting
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Starting detailed feedback generation for each part of the mock test.")

        # Generate detailed feedback for Part 1
        part1_detailed_feedback = await generate_detailed_feedback_part1(update, context)
        
        # Add a delay to avoid hitting the rate limit
        await asyncio.sleep(4)
        
        # Generate detailed feedback for Part 2
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Detailed Feedback for Part 2:")
        part2_detailed_feedback = await generate_detailed_feedback_part2(update, context)
        
        # Add a delay to avoid hitting the rate limit
        await asyncio.sleep(4)
        
        # Generate detailed feedback for Part 3
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Detailed Feedback for Part 3:")
        part3_detailed_feedback = await generate_detailed_feedback_part3(update, context)
        
        # Combine the detailed feedback from all parts
        user_data['detailed_feedback_mock_test'] = part1_detailed_feedback + part2_detailed_feedback + part3_detailed_feedback
        user_data = context.user_data['user_data']
        user_id = update.effective_user.id
        user_data['user_id'] = str(user_id)
        userID = user_data['user_id']
        # Provide user options after detailed results
        keyboard = [
            [InlineKeyboardButton("Translate Detailed Feedback", callback_data=f'{userID}mock_test_translate_detailed_feedback')],
            [InlineKeyboardButton("Retake Mock Test", callback_data=f'{userID}mock_test_retake')],
            [InlineKeyboardButton("End Test", callback_data=f'{userID}end_test')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)

        return user_data['detailed_feedback_mock_test']
    except Exception as e:
        text = ("🚨 generate mock test detailed feedback function ", e)
        await error_handling(update, context,text)

async def translate_mock_test_overall_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        user_id = query.from_user.id
        user_data = context.user_data.setdefault('user_data', {})
        overall_feedback = user_data.get('overall_mock_test_feedback')
        userID = str(user_id)

        keyboard = [
            [InlineKeyboardButton("See Detailed Results", callback_data=f'{userID}mock_test_detailed_results')],
            [InlineKeyboardButton("Retake Mock Test", callback_data=f'{userID}mock_test_retake')],
            [InlineKeyboardButton("End Test", callback_data=f'{userID}end_test')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)   

        if overall_feedback:
            animated_emoji = "⏳"
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until the translation is ready.")

            # Create a background task for translation
            context.application.create_task(
                _translate_and_send_mock_test_feedback(update, context, user_id, overall_feedback, waiting_message, reply_markup)
            )
        else:
            text = "🚨 translate_mock_test_overall_feedback: overall_feedback = False"
            await error_handling(update, context, text)
            # await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, there was an issue retrieving the feedback.")
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
    
    except Exception as e:
        text = ("🚨 translate mock test overall feedback function ", e)
        await error_handling(update, context, text)
async def _translate_and_send_mock_test_feedback(update, context, user_id, overall_feedback, waiting_message, reply_markup):
    try:
        translated_feedback = await translate_feedback(user_id, overall_feedback, update, context)
        
        if translated_feedback:
            await send_long_message(update, context, translated_feedback)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, translation is not available for your language at the moment.")
    
    except Exception as e:
        print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} | 🚨 Error in translation mock test feedback : {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, there was an error during translation. Please try again later.")
    
    finally:
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)

async def translate_mock_test_detailed_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        user_id = query.from_user.id
        user_data = context.user_data.setdefault('user_data', {})

        animated_emoji = "⏳"
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes while the detailed feedback is being translated.")

        keyboard = [
            [InlineKeyboardButton("Retake Mock Test", callback_data=f'{user_id}mock_test_retake')],
            [InlineKeyboardButton("End Test", callback_data=f'{user_id}end_test')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        # Create a background task for translation
        context.application.create_task(
            _translate_and_send_mock_test_detailed_feedback(update, context, user_id, user_data, waiting_message, reply_markup)
        )

    except Exception as e:
        text = ("🚨 translate mock test detailed feedback function ", e)
        await error_handling(update, context, text)
async def _translate_and_send_mock_test_detailed_feedback(update, context, user_id, user_data, waiting_message, reply_markup):
    try:
        chat_id = update.effective_chat.id

        async def translate_and_send_part(part_name, feedback_list):
            await context.bot.send_message(chat_id=chat_id, text=f"Translated Detailed Feedback for {part_name}:")
            for feedback in feedback_list:
                translated_msg = await translate_feedback(user_id, feedback, update, context)
                if translated_msg:
                    await send_long_message(update, context, translated_msg)
                    await asyncio.sleep(1)  # Add a delay to avoid hitting the rate limit
            await asyncio.sleep(2)

        # Translate detailed feedback for all parts
        await translate_and_send_part("Part 1", user_data.get('mock_part1_detailed_feedback_list', []))
        await translate_and_send_part("Part 2", user_data.get('mock_part2_detailed_feedback_list', []))
        await translate_and_send_part("Part 3", user_data.get('mock_part3_detailed_feedback_list', []))

    except Exception as e:
        print(f"ID: {update.effective_user.id} Username: {update.effective_user.username} | 🚨 Error in detailed mock test feedback translation: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Sorry, there was an error during translation. Please try again later.")

    finally:
        await context.bot.delete_message(chat_id=chat_id, message_id=waiting_message.message_id)
        await context.bot.send_message(chat_id=chat_id, text="What would you like to do next?", reply_markup=reply_markup)   
#------------------------- Broadcasting ----------------------------------

async def confirm_broadcast(update: Update, context: ContextTypes.DEFAULT_TYPE):
  query = update.callback_query
  await query.answer()
  
  if query.data == "confirm_broadcast":
      await query.edit_message_text("Broadcasting messages...")
      print(f"Broadcast target: {context.user_data.get('broadcast_target')}")
      if context.user_data.get('broadcast_target') == 'all':
          print("message to all users")
          user_ids = await get_all_user_ids()
      else:
          print("message for users_never_practiced")
          user_ids = await get_users_never_practiced(context)
        #   user_ids = await get_all_user_ids()
      # Ensure we're not sending "All Users" or "Never Practiced Users" messages
      broadcast_messages = [
          msg for msg in context.user_data.get('broadcast_messages', [])
          if msg.text not in ["All Users", "Never Practiced Users"]
      ]
      
      success_count = 0
      fail_count = 0
      
      for user_id in user_ids:
          try:
              for message in broadcast_messages:
                  if message.text:
                      await context.bot.send_message(chat_id=user_id, text=message.text)
                  elif message.photo:
                      await context.bot.send_photo(chat_id=user_id, photo=message.photo[-1].file_id, caption=message.caption)
                  elif message.video:
                      await context.bot.send_video(chat_id=user_id, video=message.video.file_id, caption=message.caption)
                  elif message.document:
                      await context.bot.send_document(chat_id=user_id, document=message.document.file_id, caption=message.caption)
              success_count += 1
          except Exception as e:
              print(f"Failed to send broadcast to user {user_id}: {e}")
              fail_count += 1
          
          await asyncio.sleep(0.05)  # To avoid hitting rate limits
      
      target_text = "all users" if context.user_data.get('broadcast_target') == 'all' else "users who never practiced"
      await query.edit_message_text(f"Broadcast to {target_text} complete.\nSuccessful: {success_count}\nFailed: {fail_count}")
  else:
      await query.edit_message_text("Broadcast cancelled.")

  # Clear broadcast data
  context.user_data.pop('broadcast_messages', None)
  context.user_data.pop('broadcast_target', None)
  context.user_data['in_broadcast_mode'] = False


async def handle_broadcast_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
  if not context.user_data.get('in_broadcast_mode'):
      return

  if update.message.text == "Done":
      if not context.user_data.get('broadcast_messages'):
          await update.message.reply_text("You haven't sent any messages to broadcast. Please send at least one message.")
          return

      # Remove "All Users" and "Never Practiced Users" messages from the broadcast list
      context.user_data['broadcast_messages'] = [
          msg for msg in context.user_data.get('broadcast_messages', [])
          if msg.text not in ["All Users", "Never Practiced Users"]
      ]

      keyboard = [
          [InlineKeyboardButton("Confirm", callback_data="confirm_broadcast"),
           InlineKeyboardButton("Cancel", callback_data="cancel_broadcast")]
      ]
      reply_markup = InlineKeyboardMarkup(keyboard)
      
      await update.message.reply_text("Here's a preview of your broadcast messages:", reply_markup=reply_markup)
      for message in context.user_data['broadcast_messages']:
          await message.copy(chat_id=update.effective_chat.id)
      
      context.user_data['in_broadcast_mode'] = False
  elif update.message.text == "Cancel":
      context.user_data['in_broadcast_mode'] = False
      await update.message.reply_text("Broadcast cancelled.")
  else:
      context.user_data.setdefault('broadcast_messages', []).append(update.message)
      await update.message.reply_text("Message added to broadcast. Send more messages or click 'Done' when finished.")

async def send_message_copy(chat_id: int, message, context: ContextTypes.DEFAULT_TYPE):
  if message.text:
      await context.bot.send_message(chat_id, message.text)
  elif message.photo:
      await context.bot.send_photo(chat_id, message.photo[-1].file_id, caption=message.caption)
  elif message.video:
      await context.bot.send_video(chat_id, message.video.file_id, caption=message.caption)
  elif message.document:
      await context.bot.send_document(chat_id, message.document.file_id, caption=message.caption)

# async def send_broadcast_messages(messages, user_id, context: ContextTypes.DEFAULT_TYPE,update: Update,):
#   try:
#       for message in messages:
#           await send_message_copy(user_id, message, context)
#       return True
#   except Exception as e:
#       print(f"Failed to send broadcast to user {user_id} username: {update.effective_user.username}: {e}")
#       raise e
async def send_broadcast_messages(messages, user_id, context: ContextTypes.DEFAULT_TYPE, update: Update):
  try:
      for message in messages:
          await send_message_copy(user_id, message, context)
      return True
  except Exception as e:
      print(f"Failed to send broadcast to user {user_id}: {e}")
      return False

async def get_users_never_practiced(context: ContextTypes.DEFAULT_TYPE):
  if 'never_practiced_users' in context.user_data:
      return context.user_data['never_practiced_users']

  users = fetch_all_data("ielts_speaking_users")
  
  never_practiced = [user['user_id'] for user in users if user.get('practice_count', 0) == 0]
  
  context.user_data['never_practiced_users'] = never_practiced
  
  return never_practiced

async def broadcast_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
  user_id = update.effective_user.id
  if user_id not in ADMIN_IDS:
      await update.message.reply_text("You are not authorized to use this command.")
      return

  context.user_data['broadcast_messages'] = []
  context.user_data['in_broadcast_mode'] = True
  
  keyboard = [
      [KeyboardButton("All Users"), KeyboardButton("Never Practiced Users")],
      [KeyboardButton("Cancel")]
  ]
  reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
  
  await update.message.reply_text(
      "Please select the target audience for your broadcast:",
      reply_markup=reply_markup
  )
async def handle_broadcast_target(update: Update, context: ContextTypes.DEFAULT_TYPE):
  target = update.message.text
#   if target == "All Users":
#       context.user_data['broadcast_target'] = 'all'
#       print(f"Broadcast target all users: {context.user_data.get('broadcast_target')}")
#       print("Target set to all users")
#   elif target == "Never Practiced Users":
#       context.user_data['broadcast_target'] = 'never_practiced'
#   else:
#       await update.message.reply_text("Invalid selection. Broadcast cancelled.")
#       context.user_data['in_broadcast_mode'] = False
#       return

  # Clear any previously stored broadcast messages
  context.user_data['broadcast_messages'] = []
  context.user_data['in_broadcast_mode'] = True

  keyboard = [
      [KeyboardButton("Done"), KeyboardButton("Cancel")]
  ]
  reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
  
  await update.message.reply_text(
      "Please send the messages you want to broadcast. You can send multiple messages (text, image, or video). "
      "When you're finished, click 'Done'. To cancel the broadcast, click 'Cancel'.",
      reply_markup=reply_markup
  )

# async def send_weekly_encouragement(context: ContextTypes.DEFAULT_TYPE):
#   never_practiced_users = await get_users_never_practiced(context)
#   message = "Hey there! We noticed you haven't tried our IELTS Speaking practice yet. Why not give it a go today? It's a great way to improve your skills!"
  
#   success_count = 0
#   fail_count = 0
  
#   for user_id in never_practiced_users:
#       try:
#           await context.bot.send_message(chat_id=user_id, text=message)
#           success_count += 1
#       except Exception as e:
#           print(f"Failed to send encouragement to user {user_id}: {e}")
#           fail_count += 1
      
#       await asyncio.sleep(0.05)  # To avoid hitting rate limits
  
#   print(f"Weekly encouragement sent. Successful: {success_count}, Failed: {fail_count}")
#   context.user_data.pop('never_practiced_users', None)

# def setup_weekly_encouragement(application: Application):
#   job_queue = application.job_queue
#   job_queue.run_repeating(send_weekly_encouragement, interval=timedelta(days=7), first=timedelta(minutes=1))
async def get_all_user_ids():
    # Implement this function to fetch user IDs from your Supabase database
    response = supabase.table('ielts_speaking_users').select('user_id').execute()
    return [record['user_id'] for record in response.data]
# async def get_all_user_ids():
#     # For testing, we'll return only the specified user ID
#     print("Fetching test user ID")
#     return [5357232217,7345217368]
def parse_date(date_string):
  if not date_string:
      return None
  try:
      dt = parse(date_string)
      return dt.replace(tzinfo=tzutc()) if dt.tzinfo is None else dt
  except:
      return None

def fetch_all_data(table_name, select_query="*"):
  all_data = []
  page = 1
  while True:
      result = supabase.table(table_name).select(select_query).range((page-1)*1000, page*1000-1).execute()
      all_data.extend(result.data)
      if len(result.data) < 1000:
          break
      page += 1
      time.sleep(1)  # Add a small delay between requests
  return all_data

def generate_bot_statistics():
  try:
      now = datetime.now(tzutc())
      today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
      yesterday_start = today_start - timedelta(days=1)
      week_ago = today_start - timedelta(days=7)
      month_ago = today_start - timedelta(days=30)

      # Fetch all data
      users = fetch_all_data("ielts_speaking_users")
      scores = fetch_all_data("ielts_speaking_scores")
      subscriptions = fetch_all_data("bot_subscriptions")

      total_users = len(users)

      # Users registered in different time periods
      users_today = sum(1 for user in users if parse_date(user['start_date']) and (now - parse_date(user['start_date'])).days == 0)
      users_yesterday = sum(1 for user in users if parse_date(user['start_date']) and (now - parse_date(user['start_date'])).days == 1)
      users_last_week = sum(1 for user in users if parse_date(user['start_date']) and (now - parse_date(user['start_date'])).days <= 7)
      users_last_month = sum(1 for user in users if parse_date(user['start_date']) and (now - parse_date(user['start_date'])).days <= 30)

      # Practice statistics (completed tests)
      completed_practices_today = 0
      completed_practices_yesterday = 0
      users_completed_today = set()
      users_completed_yesterday = set()
      practice_types = defaultdict(int)
      practice_hours = defaultdict(int)

      for score in scores:
          user_id = score.get('user_id')
          for date_field, type_name in [('part1_date', 'Part 1'), ('part2_date', 'Part 2'), ('part3_date', 'Part 3'), ('mock_test_date', 'Mock Test')]:
              practice_date = parse_date(score.get(date_field))
              if practice_date:
                  practice_types[type_name] += 1
                  practice_hours[practice_date.hour] += 1
                  if practice_date >= today_start:
                      completed_practices_today += 1
                      users_completed_today.add(user_id)
                  elif yesterday_start <= practice_date < today_start:
                      completed_practices_yesterday += 1
                      users_completed_yesterday.add(user_id)

      # Practice attempts (from ielts_speaking_users table)
      practice_attempts_today = 0
      practice_attempts_yesterday = 0
      users_attempted_today = set()
      users_attempted_yesterday = set()

      for user in users:
          last_practice_date = parse_date(user.get('last_practice_date'))
          if last_practice_date:
              if last_practice_date >= today_start:
                  practice_attempts_today += 1
                  users_attempted_today.add(user['user_id'])
              elif yesterday_start <= last_practice_date < today_start:
                  practice_attempts_yesterday += 1
                  users_attempted_yesterday.add(user['user_id'])

      # Total practice statistics
      total_practices_today = completed_practices_today + practice_attempts_today
      total_practices_yesterday = completed_practices_yesterday + practice_attempts_yesterday
      total_users_practiced_today = len(users_completed_today.union(users_attempted_today))
      total_users_practiced_yesterday = len(users_completed_yesterday.union(users_attempted_yesterday))

      # Total practice count
      total_completed_practices = sum(1 for score in scores for date_field in ['part1_date', 'part2_date', 'part3_date', 'mock_test_date'] if score.get(date_field))
      total_practice_attempts = sum(1 for user in users if user.get('last_practice_date'))
      total_practice_count = total_completed_practices + total_practice_attempts

      # Calculate average practices per active user
      active_users = total_users_practiced_today + total_users_practiced_yesterday
      total_recent_practices = total_practices_today + total_practices_yesterday
      avg_practices_per_active_user = total_recent_practices / active_users if active_users > 0 else 0

      # Find most active practice type
      most_active_practice_type = max(practice_types, key=practice_types.get)

      # Find time of day with most practice activity
      most_active_hour = max(practice_hours, key=practice_hours.get)

      # Users who practiced at least once, never practiced, and practiced more than 10 times
      user_practice_counts = defaultdict(int)
      for score in scores:
          user_id = score.get('user_id')
          for date_field in ['part1_date', 'part2_date', 'part3_date', 'mock_test_date']:
              if score.get(date_field):
                  user_practice_counts[user_id] += 1
      for user in users:
          if user.get('last_practice_date'):
              user_practice_counts[user['user_id']] += 1

      users_practiced_at_least_once = sum(1 for count in user_practice_counts.values() if count > 0)
      users_never_practiced = total_users - users_practiced_at_least_once
      users_practiced_more_than_10 = sum(1 for count in user_practice_counts.values() if count > 10)

      # Users in channel
      users_in_channel = sum(1 for user in users if user['in_channel'] is True)

      # Users by native language and English level
      users_by_language = defaultdict(int)
      users_by_level = defaultdict(int)
      for user in users:
          lang = user.get('native_language', '').strip()
          users_by_language[lang if lang else 'Unknown'] += 1
          level = user.get('english_level', '').strip()
          users_by_level[level if level else 'Unknown'] += 1

      # Average scores
      part1_scores = [score['part1_score'] for score in scores if score.get('part1_score')]
      part2_scores = [score['part2_score'] for score in scores if score.get('part2_score')]
      part3_scores = [score['part3_score'] for score in scores if score.get('part3_score')]
      mock_test_scores = [score['mock_test_score'] for score in scores if score.get('mock_test_score')]

      avg_scores = {
          'part1': sum(part1_scores) / len(part1_scores) if part1_scores else 0,
          'part2': sum(part2_scores) / len(part2_scores) if part2_scores else 0,
          'part3': sum(part3_scores) / len(part3_scores) if part3_scores else 0,
          'mock_test': sum(mock_test_scores) / len(mock_test_scores) if mock_test_scores else 0,
      }

      # Subscription statistics
      total_subscriptions = len(subscriptions)
      subscriptions_today = sum(1 for sub in subscriptions if parse_date(sub['start_date']) >= today_start)
      subscriptions_yesterday = sum(1 for sub in subscriptions if yesterday_start <= parse_date(sub['start_date']) < today_start)
      subscriptions_last_week = sum(1 for sub in subscriptions if week_ago <= parse_date(sub['start_date']) < today_start)
      subscriptions_last_month = sum(1 for sub in subscriptions if month_ago <= parse_date(sub['start_date']) < today_start)

      active_subscriptions = sum(1 for sub in subscriptions if sub['status'] == 'active')
      expired_subscriptions = sum(1 for sub in subscriptions if sub['status'] == 'expired')
      canceled_subscriptions = sum(1 for sub in subscriptions if sub['status'] == 'canceled')

      # Generate the report
      report = f"""
📊 Bot Statistics Report

👥 Total Users: {total_users}
🆕 New Users:
• Today: {users_today}
• Yesterday: {users_yesterday}
• Last Week: {users_last_week}
• Last Month: {users_last_month}

💳 Subscription Statistics:
• Total Subscriptions: {total_subscriptions}
• New Subscriptions:
◦ Today: {subscriptions_today}
◦ Yesterday: {subscriptions_yesterday}
◦ Last Week: {subscriptions_last_week}
◦ Last Month: {subscriptions_last_month}
• Subscription Status:
◦ Active: {active_subscriptions} ({active_subscriptions/total_subscriptions*100:.1f}%)
◦ Expired: {expired_subscriptions} ({expired_subscriptions/total_subscriptions*100:.1f}%)
◦ Canceled: {canceled_subscriptions} ({canceled_subscriptions/total_subscriptions*100:.1f}%)

🏋️ Practice Statistics:
• Total Practices: {total_practice_count}
◦ Practices with result: {total_completed_practices}
◦ Practices without result: {total_practice_attempts}
• Practices Today: {total_practices_today}
◦ Practices with result: {completed_practices_today}
◦ Practices without result: {practice_attempts_today}
• Users Practiced Today: {total_users_practiced_today}
• Practices Yesterday: {total_practices_yesterday}
◦ Practices with result: {completed_practices_yesterday}
◦ Practices without result: {practice_attempts_yesterday}
• Users Practiced Yesterday: {total_users_practiced_yesterday}
• Average Practices per Active User: {avg_practices_per_active_user:.2f}
• Most Active Practice Type: {most_active_practice_type}
• Most Active Hour of the Day: {most_active_hour:02d}:00 - {(most_active_hour + 1) % 24:02d}:00 UTC

👨‍🎓 User Practice Breakdown:
• Practiced at least once: {users_practiced_at_least_once} ({users_practiced_at_least_once/total_users*100:.1f}%)
• Never practiced: {users_never_practiced} ({users_never_practiced/total_users*100:.1f}%)
• Practiced more than 10 times: {users_practiced_more_than_10} ({users_practiced_more_than_10/total_users*100:.1f}%)

📺 Channel Membership:
• Users in Channel: {users_in_channel} ({users_in_channel/total_users*100:.1f}%)

🌎 Users by Native Language:
{chr(10).join(f"  • {lang}: {count} ({count/total_users*100:.1f}%)" for lang, count in sorted(users_by_language.items(), key=lambda x: x[1], reverse=True))}

🎓 Users by English Level:
{chr(10).join(f"  • {level}: {count} ({count/total_users*100:.1f}%)" for level, count in sorted(users_by_level.items(), key=lambda x: x[1], reverse=True))}

📈 Average Scores:
• Part 1: {avg_scores['part1']:.2f}
• Part 2: {avg_scores['part2']:.2f}
• Part 3: {avg_scores['part3']:.2f}
• Mock Test: {avg_scores['mock_test']:.2f}


"""

      return report

  except Exception as e:
      print(f"Error generating statistics: {e}")
      return "Error generating statistics. Please try again later."
# Usage in your bot
# @bot.on_callback_query(filters.regex("^admin_stats$"))
async def admin_stats(update, context):
    # supabase = context.bot_data['supabase']  # Assuming you've stored the Supabase client in bot_data
    try:
        query = update.callback_query
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception as e:
        print(e)
    stats_report = generate_bot_statistics()
    await update.callback_query.message.reply_text(stats_report)
    await show_main_menu(update, context, "Here are the bot statistics.")
# requirment.txt python-telegram-bot[job-queue]
def main():
  print("main")
  request = HTTPXRequest(
      connection_pool_size=50,
      connect_timeout=5.0,
      read_timeout=200,
      pool_timeout=10.0
  )
  
  application = (
      Application.builder()
      .token(BOT_TOKEN)
      .request(request)
      .build()
  )

  # Existing handlers
  application.add_handler(CommandHandler('start', start))
  application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler))
  application.add_handler(MessageHandler(filters.VOICE, voice_handler))
  application.add_handler(CommandHandler("language", change_language))
  application.add_handler(CommandHandler("voice", change_voice))
  application.add_handler(CallbackQueryHandler(button_handler,block=False))

  # New broadcast handlers
  application.add_handler(CommandHandler("broadcast", broadcast_command))
  application.add_handler(MessageHandler(
  filters.TEXT & ~filters.COMMAND & filters.User(ADMIN_IDS) & 
  filters.Regex("^(All Users|Never Practiced Users)$"),
  handle_broadcast_target
))
  application.add_handler(MessageHandler(
  filters.TEXT & ~filters.COMMAND & filters.User(ADMIN_IDS),
  handle_broadcast_message
))
  application.add_handler(MessageHandler(
      (filters.TEXT | filters.PHOTO | filters.VIDEO | filters.Document.ALL) & 
      ~filters.COMMAND & filters.User(ADMIN_IDS),
      handle_broadcast_message
  ))

  # Setup weekly encouragement
#   setup_weekly_encouragement(application)

  # General message handler (should be last to avoid conflicts)
  application.add_handler(MessageHandler(
      filters.TEXT | filters.PHOTO | filters.VIDEO | filters.Document.ALL,
      message_handler
  ))

  application.run_polling()

if __name__ == '__main__':
    main()
