import os
import re
import requests
import json
import random
from supabase import create_client, Client
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, CallbackQueryHandler
from telegram.error import BadRequest
from datetime import datetime, timedelta
from groq import Groq
from deepgram import DeepgramClient, PrerecordedOptions, SpeakOptions
from PIL import Image, ImageDraw, ImageFont
import logging
import asyncio
from speech_assessment import assess_speech

from topic_vocabularies import topic_vocabularies
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import google.generativeai as genai
import re
import random
import time
from part2_questions import questions_part2,ielts_questions  # Import the list of Part 2 questions
from telegram import Voice
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
# from pydub import AudioSegment
# Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# Supabase database connection

#--------------------------APIs----------------------------------------

url: str = "https://wqlryzngdnfrarolbmma.supabase.co"
# key: str = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6IndxbHJ5em5nZG5mcmFyb2xibW1hIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MTc1MDk4NTksImV4cCI6MjAzMzA4NTg1OX0.zHkAeB9XxyC30WtQJSQnEyvNKCDneZ05EIQ6lfIHqQw"
key: str=  os.getenv("supabase")
supabase: Client = create_client(url, key)

# Telegram bot token
# BOT_TOKEN = "7228259994:AAGRgoOn9a-FPWHlFZf-oFhuGvU72nXc5n4"
BOT_TOKEN = os.getenv('BOT_TOKEN')
# perplexity_API = "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"
perplexity_API = os.getenv("perplexity_API")
# Groq API client
# groq_API1 = "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"
groq_API1 = os.getenv("groq_API1")
groq_client = Groq(api_key=groq_API1)

# Deepgram API client
# deepgram_API = "e2626090e76f953c1e01f3fd069a630d4d5daf5b"
deepgram_API = os.getenv("deepgram_API")
# deepgram_API2 = "0502778c404dbb949dc96d4bd878514d2baea720"
deepgram_API2 = os.getenv("deepgram_API2")
deepgram_api_keys = [deepgram_API, deepgram_API2]
#UnrealSpeech TTS API
# unreal_speech_API1 = 'Bearer Ocz2ouV94whfqIwKDCm2KK4buWVVsywymT5IhcLeoMjQXEFPVSYx4e'
unreal_speech_API1 = os.getenv("unreal_speech_API1")
# unreal_speech_API2 = 'Bearer Qgan9Osp8iTlyBBHrcoU2X1NMdHSoUwdzZ1bdqGYGobSc4rswPig45'
unreal_speech_API2 = os.getenv("unreal_speech_API2")
unreal_speech_API_keys=  [unreal_speech_API1, unreal_speech_API2]

#Gemini_API_Key
# Gemini_API_Key = 'AIzaSyAtnlV6rfm_OsSt9M_w9ZaiFn3NjdjSVuw' #mustafabinothman22
# Gemini_API_Key2 = 'AIzaSyDbU_8cAQCAhr59bqtGf40FV-92KCKkLWs' #mustafanotion
# Gemini_API_Key3 = 'AIzaSyBOb6xrGvLxRBvgMEUyWvTSGKZVDGT4j3w' #mustafabinothman2003
# Gemini_API_Key4 = 'AIzaSyB5Cy4KIg4xKwz2poq3sywJEvqI0BL10iQ' #mustafabinothman2023
# Gemini_API_Key5 = 'AIzaSyBUpws7IJIKo9rZI1YKSBPQj_RpPWwTqFo' #www.binothman24
Gemini_API_Key = os.getenv("Gemini_API_Key")
Gemini_API_Key2 = os.getenv("Gemini_API_Key2")
Gemini_API_Key3 = os.getenv("Gemini_API_Key3")
Gemini_API_Key4 = os.getenv("Gemini_API_Key4")
Gemini_API_Key5 = os.getenv("Gemini_API_Key5")
keys = [Gemini_API_Key,Gemini_API_Key2,Gemini_API_Key3,Gemini_API_Key4,Gemini_API_Key5]


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

#--------------PART1-----------------
questions_list = []
answers_list = []
detailed_feedback_list = []
voice_urls = []  # Replace with actual URLs
questions = []
analysis_list = []
list_previous_quetions= []
list_previous_answers = []
#--------------PART2-----------------
part2_voice_urls = []  # List to store Part 2 voice URLs
part2_questions = []  # List to store Part 2 questions
part2_answers = []
analysis2_list = []
detailed_feedback2_list = []
#--------------PART3-----------------
part3_voice_urls = []  # List to store Part 3 voice URLs
part3_questions = []  # List to store Part 3 questions
part3_answers = []
analysis3_list = []
detailed_feedback3_list = []

# Mock test lists
mock_part1_questions = []
mock_part1_answers = []
mock_part1_voice_urls = []

mock_part2_questions = []
mock_part2_answers = []
mock_part2_voice_urls = []

mock_part3_questions = []
mock_part3_answers = []
mock_part3_voice_urls = []

mock_part1_analysis_list = []
mock_part2_analysis_list = []
mock_part3_analysis_list = []

mock_part1_detailed_feedback_list = []
mock_part2_detailed_feedback_list = []
mock_part3_detailed_feedback_list = []

examiner_voice = ""
targeted_score = 0.0
userID = ''

# Define a dictionary to map voice names to their URLs
# voice_samples = {
#     "Dan": "https://unreal-expire-in-90-days.s3-us-west-2.amazonaws.com/149950b4-5ebc-4c60-a448-35582cc402be-0.mp3",
#     "William": "https://unreal-expire-in-90-days.s3-us-west-2.amazonaws.com/723efcfc-c08f-49cf-819c-95af7450bd11-0.mp3",
#     "Scarlett": "https://unreal-expire-in-90-days.s3-us-west-2.amazonaws.com/8e823f3c-a594-4ed5-a910-330698609af3-0.mp3",
#     "Liv": "https://unreal-expire-in-90-days.s3-us-west-2.amazonaws.com/629cd171-6f8a-4fa6-9304-291b69b48304-0.mp3",
#     "Amy": "https://unreal-expire-in-90-days.s3-us-west-2.amazonaws.com/5184dea3-f7f5-4f0c-b7c5-6454330eaca5-0.mp3"
# }

# Define a dictionary to map voice names to their filenames
voice_samples = {
    "Dan": "Dan.mp3",  # Assuming these are the filenames in your examiners_voice folder
    "William": "William.mp3",
    "Scarlett": "Scarlett.mp3",
    "Liv": "Vectoria.mp3",
    "Amy": "Amy.mp3"
}
# List to store Part 2 answers
# Handler for the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("----------------start----------------------")
    user_id = update.effective_user.id
    username = update.effective_user.username
    userID = user_id
    print("UserID: ", userID)
    #--------------PART1-----------------
    questions_list.clear()
    answers_list.clear()
    detailed_feedback_list.clear()
    voice_urls.clear()  # Replace with actual URLs
    questions.clear()
    analysis_list.clear()
    list_previous_quetions.clear()
    list_previous_answers.clear()
    #--------------PART2-----------------
    part2_voice_urls.clear()  # List to store Part 2 voice URLs
    part2_questions.clear()  # List to store Part 2 questions
    part2_answers.clear()
    analysis2_list.clear()
    detailed_feedback2_list.clear()
    #--------------PART3-----------------
    part3_voice_urls.clear()  # List to store Part 3 voice URLs
    part3_questions.clear()  # List to store Part 3 questions
    part3_answers.clear()
    analysis3_list.clear()
    detailed_feedback3_list.clear()

    # Mock test lists
    mock_part1_questions.clear()
    mock_part1_answers.clear()
    mock_part1_voice_urls.clear()

    mock_part2_questions.clear()
    mock_part2_answers.clear()
    mock_part2_voice_urls.clear()

    mock_part3_questions.clear()
    mock_part3_answers.clear()
    mock_part3_voice_urls.clear()

    mock_part1_analysis_list.clear()
    mock_part2_analysis_list.clear()
    mock_part3_analysis_list.clear()

    mock_part1_detailed_feedback_list.clear()
    mock_part2_detailed_feedback_list.clear()
    mock_part3_detailed_feedback_list.clear()
    context.user_data.clear()
    # Reset the current question index
    context.user_data[f'{userID}current_question_index'] = 0
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
                # 'target_ielts_score': '',
                'start_date': datetime.now().strftime("%Y-%m-%d %H:%M"),
                # 'last_practice_date': datetime.now().strftime("%Y-%m-%d"),
                'last_attempt_time': datetime.now().strftime("%Y-%m-%d %H:%M"),
                'attempts_remaining': 5,
                'practice_count': 0,
                # 'in_channel': False
            }
            supabase.table('ielts_speaking_users').insert(data).execute()

            # Send welcome message and request email
            keyboard = [
                [InlineKeyboardButton("Okay, no problem", callback_data='provide_email'),
                InlineKeyboardButton("No", callback_data='skip_email')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.message.reply_text("Welcome to the IELTS Speaking Practice Bot! Would you like to provide your email address?", reply_markup=reply_markup)

        else:
            # Existing user, check for missing information
            user_data = user.data[0]
            if not user_data['native_language']:
                await ask_language(update, context)
            elif not user_data['english_level']:
                await ask_english_level(update, context)
            elif not user_data['target_ielts_score']:
                await ask_target_ielts_score(update, context)
            elif not user_data['examiner_voice']:
                await ask_preferred_voice(update, context)
            else:
                # All information is present, ask which part of the test they want to take
                # await ask_test_part(update, context)
                user = supabase.table('ielts_speaking_users').select('examiner_voice').eq('user_id', user_id).execute()
                if user.data:
                    global examiner_voice
                    preferred_voice = user.data[0]['examiner_voice']
                    examiner_voice = preferred_voice
                else:
                    examiner_voice = "Dan"
                user_id = update.effective_user.id
            user = supabase.table('ielts_speaking_users').select('target_ielts_score').eq('user_id', user_id).execute()
            channel_id = "@IELTS_SpeakingBOT"  # Replace with your channel's username or ID
            if user.data:
                # chat_member = await context.bot.get_chat_member(chat_id=channel_id,user_id= user_id)
                # if chat_member.status in ["member", "creator", "administrator"]:
                #     print(f"Yes {user_id} is a member")
                    # Proceed with bot functionality
               
                    global targeted_score
                    targeted_score = user.data[0]['target_ielts_score']
                    current_date = datetime.now().strftime("%Y-%m-%d %H:%M")

                    # Update the last_practice_date in the Supabase database
                    supabase.table('ielts_speaking_users').update({'last_practice_date': current_date}).eq('user_id', user_id).execute()
                    text = "Welcome back! What would you like to do?"
                    await show_main_menu(update, context, text)
                # else:
                    # print(f"No, user {user_id} is not a member of the channel.")
                    # print(f"User's status: {chat_member.status}")
                    # await context.bot.send_message(chat_id=update.effective_chat.id, text="Please join our channel (https://t.me/IELTS_SpeakingBOT) to use this bot.")
    except Exception as e:
        print("Start function",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
    
# Helper function to ask for language
async def ask_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("ask language")
    try:
        keyboard = [
            [InlineKeyboardButton("Arabic", callback_data='language_Arabic'),
            InlineKeyboardButton("Urdu", callback_data='language_Urdu')],
            [InlineKeyboardButton("Chinese", callback_data='language_Chinese'),
            InlineKeyboardButton("Persian", callback_data='language_Persian')],
            [InlineKeyboardButton("Hindi", callback_data='language_Hindi'),
            InlineKeyboardButton("Other", callback_data='language_other')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.effective_message.reply_text("Please select your native language:", reply_markup=reply_markup)
    except Exception as e:
        print("ask language function",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def check_channel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    channel_id = "@IELTS_SpeakingBOT"
    user_id = update.effective_user.id
    chat_member = await context.bot.get_chat_member(chat_id=channel_id,user_id= user_id)
    if chat_member.status in ["member", "creator", "administrator"]:
        print(f"Yes {user_id} is a member")
        return True
    else:
        print(f"No, user {user_id} is not a member of the channel.")
        print(f"User's status: {chat_member.status}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please join our channel (https://t.me/IELTS_SpeakingBOT) to use this bot.")
        return False
        
async def error_handling(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    # await update.message.reply_text(issue_message)
    issue_message = "Sorry for the inconvenience, it seems there is an issue with the bot. If that happens, please contact me @ielts_pathway."
    text = issue_message
    await show_main_menu(update, context, text)
    
async def ask_english_level(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("ask english level")
    try:
        keyboard = [
            [InlineKeyboardButton("Beginner", callback_data='level_Beginner')],
            [InlineKeyboardButton("Elementary", callback_data='level_Elementary')],
            [InlineKeyboardButton("Intermediate", callback_data='level_Intermediate')],
            [InlineKeyboardButton("Upper Intermediate", callback_data='level_UpperIntermediate')],
            [InlineKeyboardButton("Advanced", callback_data='level_Advanced')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.effective_message.reply_text("What is your current level of English?", reply_markup=reply_markup)
    except Exception as e:
        print("ask language level function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def ask_target_ielts_score(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("ask target ielts score")
    try:
        # Create a list of scores including half scores
        scores = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0] 
        keyboard = [[InlineKeyboardButton(str(score), callback_data=f'score_{score}')] for score in scores]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.effective_message.reply_text("What is your targeted IELTS speaking score?", reply_markup=reply_markup)
    except Exception as e:
        print("ask targeted score function",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def ask_preferred_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("ask prefeered examiner voice")
    """Sends voice samples and asks the user to choose their preferred voice."""
    
    # user_id = update.effective_user.id
    
    # # Send each voice sample as an audio message
    # for voice_name, voice_url in voice_samples.items():
    #     await context.bot.send_audio(chat_id=update.effective_chat.id, audio=voice_url)
    try:
        user_id = update.effective_user.id

        # Send each voice sample as an audio message
        for voice_name, filename in voice_samples.items():
            # Construct the full path to the audio file
            file_path = os.path.join("examiners_voice", filename) 

            # Send the audio file
            with open(file_path, 'rb') as audio_file:
                await context.bot.send_audio(chat_id=update.effective_chat.id, audio=audio_file)
        # Create inline keyboard for voice selection
        keyboard = [
        [InlineKeyboardButton("Dan", callback_data='voice_Dan')],
        [InlineKeyboardButton("William", callback_data='voice_Will')],
        [InlineKeyboardButton("Scarlett", callback_data='voice_Scarlett')],
        [InlineKeyboardButton("Vectoria", callback_data='voice_Liv')],
        [InlineKeyboardButton("Amy", callback_data='voice_Amy')]
    ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text("Please listen to the voice samples above and choose your preferred voice:", reply_markup=reply_markup)
    except Exception as e:
        print("ask preferred voice function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
# Handler for user messages
# Handler for user messages
async def message_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("message handler")
    try:
        user_id = update.effective_user.id
        text = update.message.text
        print(f"Received message: {text}")

        user = supabase.table('ielts_speaking_users').select('*').eq('user_id', user_id).execute()

        if 'email_prompt' in context.user_data:
            print("email_prompt")
            # User is providing their email
            if is_valid_gmail(text) and is_real_gmail(text):
                supabase.table('ielts_speaking_users').update({'email': text}).eq('user_id', user_id).execute()
                del context.user_data['email_prompt']
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
                await ask_language(update, context)
            else:
                await update.message.reply_text("Please enter a valid Gmail address.")
        elif 'other_language_prompt' in context.user_data:
            print("other_language_prompt")
            # User is providing another language
            if text in translated_languages:
                # User entered the language in the native language format
                native_language = text
                supabase.table('ielts_speaking_users').update({'native_language': native_language}).eq('user_id', user_id).execute()
                del context.user_data['other_language_prompt']
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
                await ask_english_level(update, context)  # Proceed to ask for English level
            elif text.capitalize() in common_languages:
                # User entered the language in the English format
                native_language = text
                supabase.table('ielts_speaking_users').update({'native_language': native_language}).eq('user_id', user_id).execute()
                del context.user_data['other_language_prompt']
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
                await ask_english_level(update, context)  # Proceed to ask for English level
            else:
                await update.message.reply_text("Please enter a valid language name in either your native language or English.")
        elif 'other_language_prompt2' in context.user_data:
            print("other_language_prompt2")
            # User is providing another language
            if text in translated_languages:
                # User entered the language in the native language format
                native_language = text
                supabase.table('ielts_speaking_users').update({'native_language': native_language}).eq('user_id', user_id).execute()
                del context.user_data['other_language_prompt2']
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
                text= "The language has been successfully changed."
                await show_main_menu(update, context,text)
                # await ask_english_level(update, context)  # Proceed to ask for English level
            elif text.capitalize() in common_languages:
                # User entered the language in the English format
                native_language = text
                supabase.table('ielts_speaking_users').update({'native_language': native_language}).eq('user_id', user_id).execute()
                del context.user_data['other_language_prompt2']
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=update.message.message_id)
                text= "The language has been successfully changed."
                await show_main_menu(update, context,text)
                # await ask_english_level(update, context)  # Proceed to ask for English level
            else:
                await update.message.reply_text("Please enter a valid language name in either your native language or English.")
        elif 'english_level_prompt' in context.user_data:
            print("english_level_prompt")
            # User is providing their English level
            if text in ['Beginner', 'Elementary', 'Intermediate', 'Upper Intermediate', 'Advanced']:
                supabase.table('ielts_speaking_users').update({'english_level': text}).eq('user_id', user_id).execute()
                del context.user_data['english_level_prompt']
                await ask_target_ielts_score(update, context)  # Proceed to ask for targeted IELTS score
            else:
                await update.message.reply_text("Please select a valid English level from the options provided.")
        elif 'target_ielts_score_prompt' in context.user_data:
            print("target_ielts_score_prompt")
            # User is providing their targeted IELTS score
            try:
                score = float(text)
                if 5.0 <= score <= 9.0:
                    supabase.table('ielts_speaking_users').update({'target_ielts_score': score}).eq('user_id', user_id).execute()
                    del context.user_data['target_ielts_score_prompt']
                    await ask_test_part(update, context)  # Proceed to ask which part of the test to take
                else:
                    await update.message.reply_text("Please enter a valid IELTS score between 5.0 and 9.0.")
            except ValueError:
                await update.message.reply_text("Please enter a valid IELTS score between 5.0 and 9.0.")
        elif 'part_1_topic_selection' in context.user_data:
            print("part_1_topic_selection")
            # User is selecting a topic for Part 1
            topic = text
            if topic.isdigit() and 1 <= int(topic) <= len(part_1_topics):
                selected_topic = part_1_topics[int(topic) - 1]
                context.user_data[f'{userID}selected_topic'] = selected_topic
                del context.user_data['part_1_topic_selection']
                await generate_and_ask_questions(update, context, selected_topic)
            else:
                if text == "Stop the Test":
                    print("stop the test")
                    try:
                        del context.user_data['part_1_topic_selection']
                    except Exception as e:
                        print(e)  
                    await update.message.reply_text("The test will stop now....")
                    await start(update, context)
                else:
                    await update.message.reply_text("Please enter a valid topic number.")
            
                # await update.message.reply_text("Please enter a valid topic number.")
        elif text == "Part 1":
            
            has_attempts = await check_user_attempts(update, context)
            if has_attempts:
                print("Part 1 selected")
                #--------------PART1-----------------
                questions_list.clear()
                answers_list.clear()
                detailed_feedback_list.clear()
                voice_urls.clear()  # Replace with actual URLs
                questions.clear()
                analysis_list.clear()
                list_previous_quetions.clear()
                list_previous_answers.clear()
                context.user_data.clear()
                print(examiner_voice)
                await stop_test(update,context, "IELTS Speaking test Part 1 ")
                await ask_part_1_topic(update, context)
            else:
                # User has reached the maximum attempts
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context,text)
        elif text == "Part 2":
            
            print("Part 2 selected")
            has_attempts = await check_user_attempts(update, context)
            if has_attempts:
                #--------------PART2-----------------
                part2_voice_urls.clear()  # List to store Part 2 voice URLs
                part2_questions.clear()  # List to store Part 2 questions
                part2_answers.clear()
                analysis2_list.clear()
                detailed_feedback2_list.clear()
                context.user_data.clear()
                await stop_test(update,context, "IELTS Speaking test Part 2 ")
                await update.effective_message.reply_text( 
                                                        #   "IELTS Speaking Part 2:\n\n"
                    "Now, we will begin Part 2 of the IELTS Speaking test. In this part, you will be given a topic and you will have one minute to prepare. \nAfter that, you should speak on the topic for 1 to 2 minutes. \nYou can make notes if you wish. \n\nLet's start") 
                # time.sleep(2)
                await start_part2_test(update, context)
            else:
                # User has reached the maximum attempts
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context,text)
        elif text == "Part 3":
            print("Part 3 selected")
            has_attempts = await check_user_attempts(update, context)
            if has_attempts:
                #--------------PART3-----------------
                part3_voice_urls.clear()  # List to store Part 3 voice URLs
                part3_questions.clear()  # List to store Part 3 questions
                part3_answers.clear()
                analysis3_list.clear()
                detailed_feedback3_list.clear()
                context.user_data.clear()
                await stop_test(update,context, "IELTS Speaking test Part 3")
                keyboard = [
                    [InlineKeyboardButton("Take Part 2 first", callback_data='take_part2_first')],
                    [InlineKeyboardButton("Skip to Part 3", callback_data='skip_part2')]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                await update.message.reply_text("Do you want to take Part 2 first or skip to Part 3? (because IELTS Part 3 is related to Part 2)", reply_markup=reply_markup)
            else:
                # User has reached the maximum attempts
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context,text)
        elif text == "Mock Test":
            # Mock test lists
            print("mock test selected")
            has_attempts = await check_user_attempts(update, context)
            if has_attempts:
                 #--------------PART1-----------------
                questions_list.clear()
                answers_list.clear()
                detailed_feedback_list.clear()
                voice_urls.clear()  # Replace with actual URLs
                questions.clear()
                analysis_list.clear()
                list_previous_quetions.clear()
                list_previous_answers.clear()
                #--------------PART2-----------------
                part2_voice_urls.clear()  # List to store Part 2 voice URLs
                part2_questions.clear()  # List to store Part 2 questions
                part2_answers.clear()
                analysis2_list.clear()
                detailed_feedback2_list.clear()
                #--------------PART3-----------------
                part3_voice_urls.clear()  # List to store Part 3 voice URLs
                part3_questions.clear()  # List to store Part 3 questions
                part3_answers.clear()
                analysis3_list.clear()
                detailed_feedback3_list.clear()
                
                #------------------Mock test---------------------------
                mock_part1_questions.clear()
                mock_part1_answers.clear()
                mock_part1_voice_urls.clear()

                mock_part2_questions.clear()
                mock_part2_answers.clear()
                mock_part2_voice_urls.clear()

                mock_part3_questions.clear()
                mock_part3_answers.clear()
                mock_part3_voice_urls.clear()

                mock_part1_analysis_list.clear()
                mock_part2_analysis_list.clear()
                mock_part3_analysis_list.clear()

                mock_part1_detailed_feedback_list.clear()
                mock_part2_detailed_feedback_list.clear()
                mock_part3_detailed_feedback_list.clear()
                context.user_data.clear()
                await stop_test(update,context, "IELTS Speaking Mock test")
                await start_mock_test(update, context)
            else:
                # User has reached the maximum attempts
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context,text)
        elif text == "Start Test":
            print("Start Test selected")
            await ask_test_part(update, context)  # Proceed to test part selection
        elif text == "Main menu":
            print("Main menu selected")
            
            await show_main_menu(update, context,"Main menu")  # Proceed to test part selection
        elif text == "Bot Channel":
            print("Main menu selected")
            text="I have created a channel to share updates about the bot and provide the best resources for IELTS. Please join us at @IELTS_SpeakingBOT."
            await show_main_menu(update, context,text)  # Proceed to test part selection
        elif text == "Show Progress":
            print("Show Progress selected")
            # TODO: Implement logic to show user progress
            await update.message.reply_text("Youur progress in IELTS Speaking ")
            await show_progress(update, context)
        elif text == "Stop the Test":
            print("stop the test")
            try:
                del context.user_data['part_1_topic_selection']
            except Exception as e:
                print(e)  
            await update.message.reply_text("The test will stop now....")
            await start(update, context)
        elif text == "Contact Me":
            print("Contact Me selected")
            # TODO: Implement logic for contacting you (e.g., provide contact info)
            await update.message.reply_text("You can contact me at @ielts_pathway.")
        elif text=="Change language":
            await change_language(update, context)
        elif text=="Change voice":
            await change_voice(update, context)
    except Exception as e:
        print("message handler function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
# Handler for voice messages
async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("voice handler")
    try:
        user_id = update.effective_user.id
        
        if f'{userID}part2_answering' in context.user_data:
            voice_duration = update.message.voice.duration
            print("answering_question part 2")
            if voice_duration < 60:
                await update.message.reply_text("Your answer is less than 1 minute. Please re-record your answer and make it longer.")
            elif voice_duration > 121:
                await update.message.reply_text("Your answer is too long more than 2 minutes . Please try to shorten your answer and re-record it.")
            else:
                # Convert the user's voice recording to text using Deepgram STT API
                transcribed_text = await convert_audio_to_text(update.message.voice.file_id, update, context)
                
                if transcribed_text:
                    context.user_data[f'{userID}part2_answer'] = transcribed_text
                    context.user_data[f'{userID}part2_voice_file_id'] = update.message.voice.file_id
                    await update.message.reply_text(f"Your answer: \n{transcribed_text}\nAre you sure about your answer?", reply_markup=InlineKeyboardMarkup([
                        [InlineKeyboardButton("Yes, I am sure", callback_data='part2_confirm_answer')],
                        [InlineKeyboardButton("Try again", callback_data='part2_retry_answer')]
                    ]))
                else:
                    await update.message.reply_text("Sorry, I couldn't get your answer. Please try again.")
        elif f'{userID}answering_question' in context.user_data:
            print("answering_question part 1")
            if f'{userID}current_question' in context.user_data:
                question = context.user_data[f'{userID}current_question']
                if update.message.voice:
                    print("Voice message received")
                    file_id = update.message.voice.file_id
                    file = await context.bot.get_file(file_id)
                    file_path = file.file_path
                    
                    # Get the duration of the voice message
                    voice_duration = update.message.voice.duration
                    
                    if voice_duration > 80:  # 1 minute and 30 seconds
                        await update.message.reply_text("Your answer is too long. Please record another answer shorter than 1 minute.")
                    else:
                        transcribed_text = await convert_audio_to_text(file_id, update, context)
                        # print(f"Transcribed text: {transcribed_text}")
                        
                        if transcribed_text:
                            context.user_data[f'{userID}current_answer'] = transcribed_text
                            context.user_data[f'{userID}current_voice_file_id'] = file_id
                            await update.message.reply_text(f"Your answer: \n{transcribed_text}\nAre you sure about your answer?", reply_markup=InlineKeyboardMarkup([
                                [InlineKeyboardButton("Yes, I am sure", callback_data='confirm_answer')],
                                [InlineKeyboardButton("Try again", callback_data='retry_answer')]
                            ]))
                        else:
                            await update.message.reply_text("Sorry, I couldn't get your answer. Please try again.")
                else:
                    print("No voice message received")
                    await update.message.reply_text("Please send a voice message as your answer.")
            else:
                print("No current question in context")
                await update.message.reply_text("Unexpected error. Please try again.")
        elif f'{userID}current_part3_question' in context.user_data:
            question = context.user_data[f'{userID}current_part3_question']
            voice_file_id = update.message.voice.file_id
            voice_file = await context.bot.get_file(voice_file_id)
            voice_url = voice_file.file_path

            transcribed_text = await convert_audio_to_text(voice_file_id, update, context)

            if transcribed_text:
                context.user_data[f'{userID}current_part3_answer'] = transcribed_text
                context.user_data[f'{userID}current_part3_voice_url'] = update.message.voice.file_id
                await update.message.reply_text(f"Your answer: \n{transcribed_text}\nAre you sure about your answer?", reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("Yes, I am sure", callback_data='part3_confirm_answer')],
                    [InlineKeyboardButton("Try again", callback_data='part3_retry_answer')],
                    # [InlineKeyboardButton("Suggest an answer", callback_data='part3_suggest_answer')]
                ]))
            else:
                await update.message.reply_text("Sorry, I couldn't get your answer. Please try again.")
        
        elif f'{userID}mock_part1_answering' in context.user_data:
            print("mock_part1_answering")
            # Get the current question index from the context
            current_question_index = context.user_data.get(f'{userID}mock_part1_current_question_index', 0)

            # Get the user's voice answer
            user_answer_voice = update.message.voice

            if user_answer_voice:
                # Get the file path URL of the user's voice answer
                voice_file_path_url = await get_voice_file_path_url(user_answer_voice)

                # Transcribe the user's voice answer
                user_answer_text = await convert_audio_to_text(user_answer_voice.file_id, update, context)
                # print(user_answer_text)

                if user_answer_text:
                    # Store the user's answer text and voice file path URL in the respective lists
                    mock_part1_answers.append(user_answer_text)
                    # print("user_answer_text added to the list", user_answer_text)
                    mock_part1_voice_urls.append(voice_file_path_url)
                    # print("voice_file_path_url added to the list: ", voice_file_path_url)

                    # Increment the current question index
                    context.user_data[f'{userID}mock_part1_current_question_index'] = current_question_index + 1

                    # Remove the 'mock_part1_answering' flag from context.user_data
                    del context.user_data[f'{userID}mock_part1_answering']

                    # Call the mock_part1_process function to ask the next question
                    await mock_part1_process(update, context)
                else:
                    await update.message.reply_text("Sorry, I couldn't get your answer seems you have not answer the question. Please try again.")
            else:
                # Send a message to the user asking them to try again
                await update.message.reply_text("Please provide a voice message with your answer.")
        elif f'{userID}mock_part2_answering' in context.user_data:
        # Get the user's voice answer
            user_answer_voice = update.message.voice
            voice_duration = update.message.voice.duration
            if user_answer_voice:
                if voice_duration < 60:
                    await update.message.reply_text("Your answer is less than 1 minute. Please re-record your answer and make it longer.")
                elif voice_duration > 121:
                    await update.message.reply_text("Your answer is too long more than 2 minutes . Please try to shorten your answer and re-record it.")
                else:
                # Get the file path URL of the user's voice answer
                    voice_file_path_url = await get_voice_file_path_url(user_answer_voice)

                    # Transcribe the user's voice answer
                    user_answer_text = await convert_audio_to_text(user_answer_voice.file_id, update, context)

                    if user_answer_text:
                        # Store the user's answer text and voice file path URL in the respective lists
                        mock_part2_answers.append(user_answer_text)
                        # print("mock_part2_answers added to the list: ", user_answer_text)
                        mock_part2_voice_urls.append(voice_file_path_url)
                        # print("voice_file_path_url added to the list: ", voice_file_path_url)

                        # Remove the 'mock_part2_answering' flag from context.user_data
                        del context.user_data[f'{userID}mock_part2_answering']

                        await update.effective_message.reply_text("Mock Test - Part 2 completed. Moving to Part 3.")
                        time.sleep(5)
                        await mock_part3_process(update, context)
                    else:
                        await update.message.reply_text("Sorry, I couldn't get your answer. Please try again.")
            else:
                # Send a message to the user asking them to try again
                await update.message.reply_text("Please provide a voice message with your answer.")

        elif f'{userID}mock_part3_answering' in context.user_data:
            # Get the user's voice answer
            user_answer_voice = update.message.voice

            if user_answer_voice:
                # Get the file path URL of the user's voice answer
                voice_file_path_url = await get_voice_file_path_url(user_answer_voice)

                # Transcribe the user's voice answer
                user_answer_text = await convert_audio_to_text(user_answer_voice.file_id, update, context)

                if user_answer_text:
                    # Store the user's answer text and voice file path URL in the respective lists
                    mock_part3_answers.append(user_answer_text)
                    # print("user_answer_text added to the list", user_answer_text)
                    mock_part3_voice_urls.append(voice_file_path_url)
                    # print("voice_file_path_url added to the list: ", voice_file_path_url)

                    # Increment the current question index
                    context.user_data[f'{userID}mock_part3_current_question_index'] += 1

                    # Remove the 'mock_part3_answering' flag from context.user_data
                    del context.user_data[f'{userID}mock_part3_answering']

                    # Call the mock_part3_process function to ask the next question or show next steps
                    await mock_part3_process(update, context)
                else:
                    await update.message.reply_text("Sorry, I couldn't get your answer. Please try again.")
            else:
                # Send a message to the user asking them to try again
                await update.message.reply_text("Please provide a voice message with your answer.")
        else:
            print("Not answering a question")
            await update.message.reply_text("Please select a topic and start answering the questions.")
    except Exception as e:
        print("voice handler function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
# Handler for button clicks
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    
    print("button handler")
    try:
        global answers_list, questions_list,voice_urls
        query = update.callback_query
        try:
            await query.answer()
        except BadRequest as e:
            # Handle the case when the query is too old or has an invalid query ID
            print(f"Error answering callback query: {str(e)}")
        if query.data == 'provide_email':
            context.user_data['email_prompt'] = True
            print("user will add his email")
            await query.edit_message_text("Please enter your email address:")

        elif query.data == 'skip_email':
            print("user will not add his email")
            await query.edit_message_reply_markup(reply_markup=None)
            await query.edit_message_text("Skipping email registration.")
            await ask_language(update, context)

        elif query.data.startswith('language_'):
            language = query.data.split('_')[1]
            if language == 'other':
                context.user_data['other_language_prompt'] = True
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text("Please enter the name of your native language:")
            # elif language == 'other2':
            #     context.user_data['other_language_prompt2'] = True
            #     await query.edit_message_reply_markup(reply_markup=None)
            #     await query.edit_message_text("Please enter the name of your native language:")
            else:
                user_id = query.from_user.id
                supabase.table('ielts_speaking_users').update({'native_language': language}).eq('user_id', user_id).execute()
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text(f"You selected {language} as your native language.")
                await ask_english_level(update, context)  # Proceed to ask for English level
        elif query.data.startswith('language2_'):
            language = query.data.split('_')[1]
            if language == 'other':
                context.user_data['other_language_prompt2'] = True
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text("Please enter the name of your native language:")
            # elif language == 'other2':
            #     context.user_data['other_language_prompt2'] = True
            #     await query.edit_message_reply_markup(reply_markup=None)
            #     await query.edit_message_text("Please enter the name of your native language:")
            else:
                user_id = query.from_user.id
                supabase.table('ielts_speaking_users').update({'native_language': language}).eq('user_id', user_id).execute()
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text(f"You selected {language} as your native language.")
                text= "The language has been successfully changed."
                await show_main_menu(update, context,text)
        elif query.data.startswith('level_'):
            english_level = query.data.split('_')[1]
            user_id = query.from_user.id
            supabase.table('ielts_speaking_users').update({'english_level': english_level}).eq('user_id', user_id).execute()
            await query.edit_message_reply_markup(reply_markup=None)
            await query.edit_message_text(f"You selected {english_level} as your English level.")
            await ask_target_ielts_score(update, context)  # Proceed to ask for targeted IELTS score

        elif query.data.startswith('score_'):
            target_score = query.data.split('_')[1]
            user_id = query.from_user.id
            supabase.table('ielts_speaking_users').update({'target_ielts_score': target_score}).eq('user_id', user_id).execute()
            await query.edit_message_reply_markup(reply_markup=None)
            await query.edit_message_text(f"You have set your target IELTS score to {target_score}.")
            await ask_preferred_voice(update, context)  # Proceed to ask which part of the test to take
        elif query.data.startswith('voice_'):
            selected_voice = query.data.split('_')[1]
            user_id = query.from_user.id
            supabase.table('ielts_speaking_users').update({'examiner_voice': selected_voice}).eq('user_id', user_id).execute()
            await query.edit_message_reply_markup(reply_markup=None)  # Remove the keyboard
            # await query.edit_message_text(f"You have selected {selected_voice} as your examiner voice.")

            # Proceed to the next step (e.g., ask_test_part)
            await show_main_menu(update, context, f"You have selected {selected_voice} as your examiner voice.")
            await context.bot.send_message(chat_id=update.effective_chat.id, text="I have created a channel to share updates about the bot and provide the best resources for IELTS. Please join us at @IELTS_SpeakingBOT.")
        # part1 button handler 
        elif query.data == "topic_random":
            selected_topic = random.choice(part_1_topics)
            context.user_data[f'{userID}selected_topic'] = selected_topic
            await query.edit_message_text(f"the topic is: {selected_topic}")
            
            # # Remove the inline keyboard markup if it exists
            # if query.message.reply_markup:
            #     await query.edit_message_reply_markup(reply_markup=None)
            del context.user_data['part_1_topic_selection']
            await generate_and_ask_questions(update, context, selected_topic)
        elif query.data == 'confirm_answer':
            current_question = context.user_data[f'{userID}current_question']
            current_answer = context.user_data[f'{userID}current_answer']
            
            # Check if the message is a reply to a voice message
            voice_file_id = context.user_data.get(f'{userID}current_voice_file_id')
            if voice_file_id:
                # Get the voice message file path
                file = await context.bot.get_file(voice_file_id)
                file_path = file.file_path
                
                # Store the voice message URL in the voice_urls list
                voice_urls.append(file_path)
                # print(f"Voice URL added to the list: {file_path}")
            
            # Store the answer in the answers_list
            answers_list.append(current_answer)
            # print(f"Answer added to the list: {current_answer}")
            await query.edit_message_reply_markup(reply_markup=None)
            # Move to the next question
            context.user_data[f'{userID}current_question_index'] += 1
            await ask_current_question(update, context)
        elif query.data == 'retry_answer':
            await query.edit_message_reply_markup(reply_markup=None)
            await ask_current_question(update, context, retry=True)
        elif query.data == 'suggest_answer':
            question = context.user_data[f'{userID}current_question']
            previous_answer = context.user_data[f'{userID}current_answer']
            # print(previous_answer)
            # Generate the suggested answer using LLM
            suggested_answer = await generate_suggested_answer(question, previous_answer, "part 1 ")
            # print("preious Answer: \n",previous_answer, "\nsuggested Answer: \n",suggested_answer)
            # Send the suggested answer to the user
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Suggested Answer:\n\n {suggested_answer}")
            
            # Remove the "Suggest Answer" button
            await query.edit_message_reply_markup(reply_markup=None)
            
            # Prompt the user to record their answer again
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please record your answer again.")
        elif query.data == 'retake_part_1':
            await query.edit_message_reply_markup(reply_markup=None)
            has_attempts = await check_user_attempts(update, context)
            if has_attempts:
                print("Part 1 selected")
                #--------------PART1-----------------
                questions_list.clear()
                answers_list.clear()
                detailed_feedback_list.clear()
                voice_urls.clear()  # Replace with actual URLs
                questions.clear()
                analysis_list.clear()
                list_previous_quetions.clear()
                list_previous_answers.clear()
                context.user_data.clear()
                print(examiner_voice)
                await stop_test(update,context, "IELTS Speaking test Part 1 ")
                await ask_part_1_topic(update, context)
            else:
                # User has reached the maximum attempts
                # await update.message.reply_text("You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway")
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context,text)
        elif query.data == 'show_results':
            # Reset the questions and answers lists for the next user
            # questions_list = []
            # answers_list = []
            # voice_urls = []
            # # Reset the user's state
            # context.user_data.clear()
            await query.edit_message_reply_markup(reply_markup=None)
            # await query.edit_message_text("Test ended. Thank you for taking the IELTS Speaking test!")
            await show_results_part1(update, context)
        elif query.data == 'continue_part_2':
            #--------------PART2-----------------
            print("continue to part 2")
            has_attempts = await check_user_attempts(update, context)
            if has_attempts:
                part2_voice_urls.clear()  # List to store Part 2 voice URLs
                part2_questions.clear()  # List to store Part 2 questions
                part2_answers.clear()
                analysis2_list.clear()
                context.user_data.clear()
                # TODO: Implement logic to continue to Part 2
                # await query.edit_message_reply_markup(reply_markup=None)
                # await query.edit_message_text("Continuing to Part 2...")
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text("Continuing to Part 2...")
                await start_part2_test(update, context)
            text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
            await show_main_menu(update, context,text)
        elif query.data == 'detailed_results':
            # await query.edit_message_text("Your detailed feedback will be ready in a few minutes")
            animated_emoji = "⏳"  # Hourglass emoji
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
            await query.edit_message_reply_markup(reply_markup=None)
            # Generate detailed feedback for each answer
            await generate_detailed_feedback(update, context)
            # detailed_feedback = await generate_detailed_feedback(update, context)
            
            # # Send detailed feedback messages
            # for feedback in detailed_feedback:
            #     await asyncio.sleep(4)
            #     await send_long_message(update, context, feedback)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
            # Provide user options after detailed results
            keyboard = [
                [InlineKeyboardButton("Continue to Part 2", callback_data='continue_part_2')],
                [InlineKeyboardButton("Translate", callback_data='translate_detailed_feedback')],
                [InlineKeyboardButton("Retake Part 1", callback_data='retake_part_1')],
                [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)          
        elif query.data == 'translate_overall_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            user_id = query.from_user.id
            overall_feedback = context.user_data.get(f'{userID}overall_feedback')
            # print(len(overall_feedback))
            animated_emoji = "⏳"  # Hourglass emoji
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
            keyboard = [
                [InlineKeyboardButton("Continue to Part 2", callback_data='continue_part_2')],
                [InlineKeyboardButton("See Detailed Results", callback_data='detailed_results')],
                [InlineKeyboardButton("Retake Part 1", callback_data='retake_part_1')],
                [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            if overall_feedback:
                # Translate the overall feedback
                translated_feedback = await translate_feedback(user_id, overall_feedback, update,context)
                
                if translated_feedback:
                    # Send the translated feedback to the user
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=translated_feedback)
                    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
                else:
                    # Send a message indicating that translation is not available for the user's language
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, translation is not available for your language at the moment. We are working on adding support for more languages.")
                    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry there is an issue please contact me if this happened again")
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            
            # Provide the user with options (except for the "Translate" option)
        elif query.data == 'translate_detailed_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            user_id = query.from_user.id
            animated_emoji = "⏳"  # Hourglass emoji
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            keyboard = [
                [InlineKeyboardButton("Continue to Part 2", callback_data='continue_part_2')],
                [InlineKeyboardButton("Retake Part 1", callback_data='retake_part_1')],
                [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            # Translate each detailed feedback message
            translated_feedback = []
            for feedback in detailed_feedback_list:
                translated_msg = await translate_feedback(user_id, feedback, update,context)
                
                if translated_msg:
                    translated_feedback.append(translated_msg)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
            if translated_feedback:
                # Send the translated detailed feedback messages to the user
                for feedback in translated_feedback:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=feedback)
                    # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            else:
                # Send a message indicating that translation is not available for the user's language
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, translation is not available for your language at the moment. We are working on adding support for more languages.")
                # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            # Provide the user with options (except for the "Translate" option)
            
            
        
        # part2 button handler 
        elif query.data == 'part2_confirm_answer':
            # Get the current question, answer, and voice file ID from the context
            current_question = context.user_data[f'{userID}part2_question']
            current_answer = context.user_data[f'{userID}part2_answer']
            voice_file_id = context.user_data.get(f'{userID}part2_voice_file_id')

            # Store the question in the part2_questions list
            part2_questions.append(current_question)
            # print(f"Question added to the list: {current_question}")

            # Store the answer in the part2_answers list
            part2_answers.append(current_answer)
            # print(f"Answer added to the list: {current_answer}")

            if voice_file_id:
                # Get the voice message file path
                file = await context.bot.get_file(voice_file_id)
                file_path = file.file_path

                # Store the voice message URL in the part2_voice_urls list
                part2_voice_urls.append(file_path)
                # print(f"Voice URL part 2 added to the list: {file_path}")

            # Provide options to continue to Part 3, retake Part 2, or show results
            await query.edit_message_reply_markup(reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("Continue to Part 3", callback_data='continue_part3')],
                [InlineKeyboardButton("Retake Part 2", callback_data='retake_part2')],
                [InlineKeyboardButton("Show Results", callback_data='show_part2_results')]
            ]))
        elif query.data == 'part2_retry_answer':
            await query.edit_message_reply_markup(reply_markup=None)
            await update.effective_message.reply_text("Please record your answer again.")
        elif query.data == 'continue_part3':
            print("continue to part 3")
            has_attempts = await check_user_attempts(update, context)
            if has_attempts:
                part3_voice_urls.clear()  # List to store Part 3 voice URLs
                # part3_questions.clear()  # List to store Part 3 questions
                part3_answers.clear()
                analysis3_list.clear()
                detailed_feedback3_list.clear()
                context.user_data.clear()
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text("Continuing to Part 3...")
                print(part3_questions)
                await start_part3_test(update, context)
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context,text)
        elif query.data == 'retake_part2':
            print("retake part 2")
            has_attempts = await check_user_attempts(update, context)
            if has_attempts:
                part2_questions.clear()
                part2_answers.clear()
                part2_voice_urls.clear()
                context.user_data.clear()
                await query.edit_message_reply_markup(reply_markup=None)
                await query.edit_message_text("retaking Part 2...")
                await start_part2_test(update, context)
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context,text)
        elif query.data == 'show_part2_results':
            # TODO: Implement showing Part 2 results
            await query.edit_message_reply_markup(reply_markup=None)
            # print(part2_voice_urls)
            # print(part2_questions)
            # print(part2_answers)
            # print(part2_voice_urls)
            # await query.edit_message_text("Showing Part 2 results...")   
            await show_result2(update, context)
        elif query.data == 'change_question':
            try:
                # Delete the previous question, audio, preparation, and waiting messages
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data[f'{userID}part2_question_message_id'])
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data[f'{userID}part2_audio_message_id'])
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data[f'{userID}part2_preparation_message_id'])
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=context.user_data[f'{userID}part2_waiting_message_id'])
            except BadRequest as e:
                # Handle the case when the message is not found or has already been deleted
                print(f"Error deleting message: {str(e)}")
            
            # Start a new Part 2 test
            await start_part2_test(update, context)

        elif query.data == 'detailed2_results':
            # await query.edit_message_text("Your detailed feedback will be ready in a few minutes")
            animated_emoji = "⏳"  # Hourglass emoji
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
            await query.edit_message_reply_markup(reply_markup=None)
            # Generate detailed feedback for each answer
            await generate_detailed2_feedback(update, context)
            
            # # Send detailed feedback messages
            # for feedback in detailed_feedback:
            #     await asyncio.sleep(4)
            #     await send_long_message(update, context, feedback)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
            # Provide user options after detailed results
            keyboard = [
                [InlineKeyboardButton("Continue to Part 3", callback_data='continue_part3')],
                [InlineKeyboardButton("Translate", callback_data='translate_detailed2_feedback')],
                [InlineKeyboardButton("Retake Part 2", callback_data='retake_part2')],
                [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
        elif query.data == 'end_test':
            # Handle ending the test
            await query.edit_message_reply_markup(reply_markup=None)
            await query.edit_message_text("Ending the test...")
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Thank you for taking the test. Goodbye!")
            # await context.bot.send_message(chat_id=update.effective_chat.id, text="👋")

            # Ask the user to rate the bot
            rating_keyboard = [
                [InlineKeyboardButton("👍", callback_data='rate_up')],
                [InlineKeyboardButton("👎", callback_data='rate_down')]
            ]
            rating_reply_markup = InlineKeyboardMarkup(rating_keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="How was your experience :", reply_markup=rating_reply_markup)
            # await show_main_menu(update, context)
        elif query.data == 'rate_up':
            await query.edit_message_text("❤️")
            # await context.bot.send_message(chat_id=update.effective_chat.id, text="Thank you so much for using the bot!")
            # Ask the user to share the bot
            share_message = (
                f"Discover this IELTS Speaking  Bot! It simulates the IELTS speaking test and provides detailed feedback about your speaking skills and estimated IELTS band score. to help you improve. Try it for free now: https://t.me/ielts_speakingAI_bot"
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
            text="Thank you so much for using the bot!"
            await show_main_menu(update, context, text)
        elif query.data == 'rate_down':
            # await query.edit_message_text("I really appreciate your feedback. \nPlease contact me and tell me what was the problem and try to enhance your experience next time: \n@ielts_pathway")
            text = "I really appreciate your feedback. \nPlease contact me and tell me what was the problem and try to enhance your experience next time: \n@ielts_pathway"
            await show_main_menu(update, context,text)
        elif query.data == 'translate_overall2_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            user_id = query.from_user.id
            overall_feedback = context.user_data.get(f'{userID}overall_part2_feedback')
            keyboard = [
                [InlineKeyboardButton("Continue to Part 3", callback_data='continue_part3')],
                [InlineKeyboardButton("See Detailed Results", callback_data='detailed2_results')],
                [InlineKeyboardButton("Retake Part 2", callback_data='retake_part2')],
                [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            # print(len(overall_feedback))
            animated_emoji = "⏳"  # Hourglass emoji
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")

            if overall_feedback:
                # Translate the overall feedback
                translated_feedback = await translate_feedback(user_id, overall_feedback, update,context)
                
                if translated_feedback:
                    # Send the translated feedback to the user
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=translated_feedback)
                    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
                else:
                    # Send a message indicating that translation is not available for the user's language
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, translation is not available for your language at the moment. We are working on adding support for more languages.")
                    await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry there is an issue please contact me if this happened again")
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            
            # Provide the user with options (except for the "Translate" option)
            
            
        elif query.data == 'translate_detailed2_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            keyboard = [
                [InlineKeyboardButton("Continue to Part 3", callback_data='continue_part3')],
                [InlineKeyboardButton("Retake Part 2", callback_data='retake_part2')],
                [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            user_id = query.from_user.id
            animated_emoji = "⏳"  # Hourglass emoji
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            
            # Translate each detailed feedback message
            translated_feedback = []
            for feedback in detailed_feedback2_list:
                translated_msg = await translate_feedback(user_id, feedback, update,context)
                
                if translated_msg:
                    translated_feedback.append(translated_msg)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
            if translated_feedback:
                # Send the translated detailed feedback messages to the user
                for feedback in translated_feedback:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=feedback)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            else:
                # Send a message indicating that translation is not available for the user's language
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, translation is not available for your language at the moment. We are working on adding support for more languages.")
                # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            # Provide the user with options (except for the "Translate" option)
            
            
        elif query.data == 'take_part2_first':
            await query.edit_message_reply_markup(reply_markup=None)
            
            print("take part 2 first")
            await start_part2_test(update, context)
        elif query.data == 'skip_part2':
            await query.edit_message_reply_markup(reply_markup=None)
        # Randomly select a topic from the dictionary
            selected_topic = random.choice(list(ielts_questions.keys()))

            # Extract the Part 2 question and Part 3 questions for the selected topic
            part2_question = ielts_questions[selected_topic]["part_2_question"]
            part_3_questions = ielts_questions[selected_topic]["part_3_questions"]

            # Store the Part 2 question in the context
            context.user_data[f'{userID}part2_question'] = part2_question
            main_part2_question = part2_question.split('.')[0] + '.'
            # Clear the part3_questions list before adding new questions
            part3_questions.clear()

            # Add the Part 3 questions to the part3_questions list
            part3_questions.extend(part_3_questions)

            # Send a message to the user indicating the assumed Part 2 question and proceed to Part 3
            await query.edit_message_text(f"Assuming you have answered the following Part 2 question:\n\n{main_part2_question}\n\nNow, let's proceed to Part 3.")
            await start_part3_test(update, context)
        # part3 button handler 
        elif query.data == 'part3_confirm_answer':
            current_question = context.user_data[f'{userID}current_part3_question']
            current_answer = context.user_data[f'{userID}current_part3_answer']
            
            # Check if the message is a reply to a voice message
            voice_file_id = context.user_data.get(f'{userID}current_part3_voice_url')
            if voice_file_id:
                # Get the voice message file path
                file = await context.bot.get_file(voice_file_id)
                file_path = file.file_path
                
                # Store the voice message URL in the part3_voice_urls list
                part3_voice_urls.append(file_path)
                # print(f"Voice URL added to the list: {file_path}")
            else:       
                print("sorry there is no path for the voice")
            # Store the question and answer in the respective lists
            # part3_questions.append(current_question)
            part3_answers.append(current_answer)
            # print(f"Question added to the list: {current_question}")
            # print(f"Answer added to the list: {current_answer}")
            
            await query.edit_message_reply_markup(reply_markup=None)
            
            # Move to the next question
            context.user_data[f'{userID}part3_current_question_index'] += 1
            await ask_part3_question(update, context)
        elif query.data == 'part3_retry_answer':
            print("retry part3")   
            await query.edit_message_reply_markup(reply_markup=None)
            await ask_part3_question(update, context, retry=True)
        elif query.data == 'part3_suggest_answer':
            await part3_suggest_answer(update, context)
        elif query.data == 'part3_show_summary':
            
            summary_message = "Part 3 Questions and Answers:\n\n"
            for i in range(len(part3_questions)):
                question = part3_questions[i]
                answer = part3_answers[i]
                summary_message += f"Question {i+1}: {question}\nAnswer: {answer}\n\n"
            
            await context.bot.send_message(chat_id=update.effective_chat.id, text=summary_message)
            
            keyboard = [
                [InlineKeyboardButton("Show Results", callback_data='part3_show_results')],
                [InlineKeyboardButton("Retake Part 3", callback_data='part3_retake')],
                [InlineKeyboardButton("End Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
        elif query.data == 'part3_show_results':
            # ... (code for showing Part 3 results) ...
            await query.edit_message_reply_markup(reply_markup=None)
            await part3_show_results(update,context)
        elif query.data == 'part3_retake':
            # part3_questions.clear()
            await query.edit_message_reply_markup(reply_markup=None)
            has_attempts = await check_user_attempts(update, context)
            keyboard = [
                    [InlineKeyboardButton("Take Part 2 first", callback_data='take_part2_first')],
                    [InlineKeyboardButton("Skip to Part 3", callback_data='skip_part2')]
                ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            if has_attempts:
                #--------------PART3-----------------
                part3_voice_urls.clear()  # List to store Part 3 voice URLs
                part3_questions.clear()  # List to store Part 3 questions
                part3_answers.clear()
                analysis3_list.clear()
                detailed_feedback3_list.clear()
                context.user_data.clear()
                await stop_test(update,context, "IELTS Speaking test Part 3")
                
                await update.message.reply_text("Do you want to take Part 2 first or skip to Part 3? (because IELTS Part 3 is related to Part 2)", reply_markup=reply_markup)
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context,text)
        elif query.data == 'part3_detailed_results':
            # await query.edit_message_text("Your detailed feedback will be ready in a few minutes")
            animated_emoji = "⏳"  # Hourglass emoji
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
            await query.edit_message_reply_markup(reply_markup=None)
            # Generate detailed feedback for each answer
            await part3_detailed_results(update, context)
            # detailed_feedback = await part3_detailed_results(update, context)
            
            # # Send detailed feedback messages
            # for feedback in detailed_feedback:
            #     await send_long_message(update, context, feedback)
            # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
            # Provide user options after detailed results
            keyboard = [
                [InlineKeyboardButton("Retake Part 3", callback_data='part3_retake')],
                [InlineKeyboardButton("Translate", callback_data='part3_translate_detailed_feedback')],
                # [InlineKeyboardButton("Retake Part 3", callback_data='part3_retake')],
                [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
        elif query.data == 'part3_translate_feedback':
            # ... (code for translating Part 3 feedback) ...
            user_id = query.from_user.id
            overall_feedback = context.user_data.get(f'{userID}overall_part3_feedback')
            # Provide the user with options (except for the "Translate" option)
            keyboard = [
                # [InlineKeyboardButton("Continue to Part 2", callback_data='continue_part_2')],
                [InlineKeyboardButton("See Detailed Results", callback_data='part3_detailed_results')],
                [InlineKeyboardButton("Retake Part 3", callback_data='part3_retake')],
                [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            # print(len(overall_feedback))
            await query.edit_message_reply_markup(reply_markup=None)
            animated_emoji = "⏳"  # Hourglass emoji
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")

            if overall_feedback:
                # Translate the overall feedback
                translated_feedback = await translate_feedback(user_id, overall_feedback, update,context)
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                if translated_feedback:
                    # Send the translated feedback to the user
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=translated_feedback)
                    # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
                else:
                    # Send a message indicating that translation is not available for the user's language
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, translation is not available for your language at the moment. We are working on adding support for more languages.")
                    # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry there is an issue please contact me if this happened again")
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            
            
            
        elif query.data == 'part3_translate_detailed_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            keyboard = [
                # [InlineKeyboardButton("Continue to Part 3", callback_data='continue_part3')],
                [InlineKeyboardButton("Retake Part 3", callback_data='part3_retake')],
                [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            user_id = query.from_user.id
            animated_emoji = "⏳"  # Hourglass emoji
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            
            # Translate each detailed feedback message
            translated_feedback = []
            for feedback in detailed_feedback3_list:
                translated_msg = await translate_feedback(user_id, feedback, update,context)
                # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                if translated_msg:
                    translated_feedback.append(translated_msg)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
            if translated_feedback:
                # Send the translated detailed feedback messages to the user
                for feedback in translated_feedback:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=feedback)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            else:
                # Send a message indicating that translation is not available for the user's language
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, translation is not available for your language at the moment. We are working on adding support for more languages.")
                # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            # Provide the user with options (except for the "Translate" option)
            
            
        # mock test  button handler 
        elif query.data == 'mock_test_retake':
            print("retake mock test")
            # print("mock test selected")
            await query.edit_message_reply_markup(reply_markup=None)
            has_attempts = await check_user_attempts(update, context)
            if has_attempts:
                 #--------------PART1-----------------
                questions_list.clear()
                answers_list.clear()
                detailed_feedback_list.clear()
                voice_urls.clear()  # Replace with actual URLs
                questions.clear()
                analysis_list.clear()
                list_previous_quetions.clear()
                list_previous_answers.clear()
                #--------------PART2-----------------
                part2_voice_urls.clear()  # List to store Part 2 voice URLs
                part2_questions.clear()  # List to store Part 2 questions
                part2_answers.clear()
                analysis2_list.clear()
                detailed_feedback2_list.clear()
                #--------------PART3-----------------
                part3_voice_urls.clear()  # List to store Part 3 voice URLs
                part3_questions.clear()  # List to store Part 3 questions
                part3_answers.clear()
                analysis3_list.clear()
                detailed_feedback3_list.clear()
                
                #-----------mock test----------------------------
                mock_part1_questions.clear()
                mock_part1_answers.clear()
                mock_part1_voice_urls.clear()

                mock_part2_questions.clear()
                mock_part2_answers.clear()
                mock_part2_voice_urls.clear()

                mock_part3_questions.clear()
                mock_part3_answers.clear()
                mock_part3_voice_urls.clear()

                mock_part1_analysis_list.clear()
                mock_part2_analysis_list.clear()
                mock_part3_analysis_list.clear()

                mock_part1_detailed_feedback_list.clear()
                mock_part2_detailed_feedback_list.clear()
                mock_part3_detailed_feedback_list.clear()
                context.user_data.clear()
                await stop_test(update,context, "IELTS Speaking Mock test")
                await start_mock_test(update, context)
            else:
                text = "You have reached the maximum number of attempts (5 tests) for today. Please try again after 24 hours.\n for more inforamtion please contact me @ielts_pathway"
                await show_main_menu(update, context,text)
        elif query.data == 'mock_test_show_results':
            await query.edit_message_reply_markup(reply_markup=None)
            await show_mock_test_results(update, context)
        elif query.data == 'mock_test_detailed_results':
            await query.edit_message_reply_markup(reply_markup=None)
            await generate_mock_test_detailed_feedback(update, context)
            
            # # Provide user options after detailed results
            # keyboard = [
            #     [InlineKeyboardButton("Translate Detailed Feedback", callback_data='mock_test_translate_detailed_feedback')],
            #     [InlineKeyboardButton("Retake Mock Test", callback_data='mock_test_retake')],
            #     [InlineKeyboardButton("End Test", callback_data='end_test')]
            # ]
            # reply_markup = InlineKeyboardMarkup(keyboard)
            # await context.bot.send_message
        elif query.data == 'mock_test_translate_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            await translate_mock_test_overall_feedback(update, context)
        elif query.data == 'mock_test_translate_detailed_feedback':
            await query.edit_message_reply_markup(reply_markup=None)
            await translate_mock_test_detailed_feedback(update, context)
    except Exception as e:
        print("button handler function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def change_language(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allows the user to change their preferred language."""
    try:
        user_id = update.effective_user.id

        keyboard = [
            [InlineKeyboardButton("Arabic", callback_data='language2_Arabic'),
            InlineKeyboardButton("Urdu", callback_data='language2_Urdu')],
            [InlineKeyboardButton("Chinese", callback_data='language2_Chinese'),
            InlineKeyboardButton("Persian", callback_data='language2_Persian')],
            [InlineKeyboardButton("Hindi", callback_data='language2_Hindi'),
            InlineKeyboardButton("Other", callback_data='language2_other')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.effective_message.reply_text("Change your language", reply_markup=reply_markup)
        text = "."
        await show_main_menu(update, context, text)
    except Exception as e:
        print("change language",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def change_voice(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Allows the user to change their preferred voice."""
    try:
        user_id = update.effective_user.id

        # Send each voice sample as an audio message
        for voice_name, filename in voice_samples.items():
            # Construct the full path to the audio file
            file_path = os.path.join("examiners_voice", filename)

            # Send the audio file
            with open(file_path, 'rb') as audio_file:
                await context.bot.send_audio(chat_id=update.effective_chat.id, audio=audio_file)

        # Create inline keyboard for voice selection (same as before)
        keyboard = [
            [InlineKeyboardButton("Dan", callback_data='voice_Dan')],
            [InlineKeyboardButton("William", callback_data='voice_William')],
            [InlineKeyboardButton("Scarlett", callback_data='voice_Scarlett')],
            [InlineKeyboardButton("Liv", callback_data='voice_Liv')],
            [InlineKeyboardButton("Amy", callback_data='voice_Amy')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)

        await update.effective_message.reply_text("Select your examiner voice:",
                                                reply_markup=reply_markup)
        text = "."
        await show_main_menu(update, context,text)
    except Exception as e:
        print("changing voice function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
# Helper function to ask for test part
async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE, text):
    """Displays the main menu with options: Start Test, Show Progress, Contact Me."""
    print("Showing main menu")
    try:
        keyboard = [
            [KeyboardButton("Start Test")],
            [KeyboardButton("Show Progress")],
            [KeyboardButton("Contact Me"), KeyboardButton("Bot Channel")],
            [KeyboardButton("Change language"), KeyboardButton("Change voice")],
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.effective_message.reply_text(text, reply_markup=reply_markup)
    except Exception as e:
        print("show main menu function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def stop_test(update: Update, context: ContextTypes.DEFAULT_TYPE, part):
    """Displays the stop test button."""
    print("Showing stop test button")
    try:
        keyboard = [
            [KeyboardButton("Stop the Test")],
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.effective_message.reply_text(part, reply_markup=reply_markup)
    except Exception as e:
        print("Stop test ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def ask_test_part(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("ask test part")
    try:
        keyboard = [
            [KeyboardButton("Part 1"), KeyboardButton("Part 2")],
            [KeyboardButton("Part 3"), KeyboardButton("Mock Test")],
            [KeyboardButton("Main menu")]
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, one_time_keyboard=True, resize_keyboard=True)
        await update.effective_message.reply_text("Which part of the IELTS Speaking test would you like to practice today?", reply_markup=reply_markup)
    except Exception as e:
        print("ask task part ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
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
        print(f"An error occurred in appending speaking score function : {e}")
        return False


async def show_progress(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = update.effective_user.id
        scores_df = get_user_scores(user_id)
        if not scores_df.empty:
            charts = await generate_charts(scores_df)
            await display_charts(update, context, charts)
        else:
            await update.message.reply_text("No scores found. Please practice the test to see your progress.")
    except Exception as e:
        print("show progress",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
def get_user_scores(user_id):
    try:
        # Query the ielts_speaking_scores table to get all records for the specified user_id
        response = supabase.table('ielts_speaking_scores').select('*').eq('user_id', user_id).execute()
        data = response.data
        
        # Convert the data to a DataFrame
        df = pd.DataFrame(data)
        return df
    except Exception as e:
        print(f"Failed to retrieve data get user scores function: {e}")
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
        print("increment practice count function ",e)
        # await update.message.reply_text(issue_message)

async def display_charts(update: Update, context: ContextTypes.DEFAULT_TYPE, charts: list):
    for chart in charts:
        chart.seek(0)  # Ensure the buffer is at the start
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=chart)

async def check_user_attempts(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        user_id = update.effective_user.id
        # Retrieve the user's last_attempt_time and attempts_remaining from the database
        result = supabase.table('ielts_speaking_users').select('last_attempt_time', 'attempts_remaining').eq('user_id', user_id).execute()
        
        if result.data:
            last_attempt_time_str = result.data[0]['last_attempt_time']
            attempts_remaining = result.data[0]['attempts_remaining']
            current_time = datetime.now()
            print("Number of attempts left: ",attempts_remaining)
            if attempts_remaining > 0:
                return True
            else:
                 return False
            # if last_attempt_time_str:
            #     last_attempt_time = datetime.strptime(last_attempt_time_str, "%Y-%m-%dT%H:%M:%S+00:00")
            #     if current_time - last_attempt_time < timedelta(hours=24):
            #         if attempts_remaining > 0:
            #             return True
            #         else:
            #             return False
            #     else:
            #         # Reset attempts to the maximum allowed
            #         supabase.table('ielts_speaking_users').update({
            #             'last_attempt_time': current_time.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            #             'attempts_remaining': 5
            #         }).eq('user_id', user_id).execute()
            #         return True
            # else:
            #     # User doesn't have a last_attempt_time, allow the attempt
            #     return True
        else:
            # User doesn't exist in the database, allow the attempt
            return True
    except Exception as e:
        print("check user attempts function ",e)
        # await update.message.reply_text(issue_message)
# Helper function to ask for Part 1 topic
# Helper function to ask for Part 1 topic
async def ask_part_1_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print("ask part 1 topic")
        global part_1_topics
        part_1_topics = [
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
        topics_message += "\n".join([f"{i+1}. {topic}" for i, topic in enumerate(part_1_topics)])
        
        
        context.user_data['part_1_topic_selection'] = True
        await update.effective_message.reply_text(topics_message, reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("Random", callback_data="topic_random")]
        ]))
    except Exception as e:
        print("ask part 1 topic function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def generate_and_ask_questions(update: Update, context: ContextTypes.DEFAULT_TYPE, topic):
    try:
        print("generate and ask questions")
        # Use Groq API to generate questions
        questions = await generate_questions(topic)
        if not questions:
            await update.effective_message.reply_text("Failed to generate questions. Please try again.")
            return
        v_topic = topic 
        # Get the vocabulary list for the selected topic
        vocabularies = topic_vocabularies.get(v_topic, [])
        # print(topic[:-2])
        # print("vocabularys:\n",vocabularies)
        # Send the vocabulary list to the user
        if vocabularies:
            vocabulary_message = "Here are some vocabularies you can use in your speaking:\n" + ", ".join(vocabularies)
            await update.effective_message.reply_text(vocabulary_message)
        
        context.user_data[f'{userID}questions'] = questions
        context.user_data[f'{userID}answers'] = {}
        context.user_data[f'{userID}current_question_index'] = 0
        await update.effective_message.reply_text(f"IELTS Speaking Part 1:\n\nNow, we will begin Part 1 of the IELTS Speaking test. In this part, I will ask you ({len(questions_list)}) questions including general questions about yourselt your home or your job, etc.. as well as questions on the topic of {topic[:-2]} \n\nPlease answer each question in 15 to 40 seconds. \n\nLet's start") 
        # time.sleep(1)
        
        await ask_current_question(update, context)
    except Exception as e:
        print("generate and ask questions function",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
# Helper function to ask the current question
async def ask_current_question(update: Update, context: ContextTypes.DEFAULT_TYPE, retry=False):
    try:
        print("ask current question")
        global questions_list, list_previous_quetions, list_previous_answers
        
        current_question_index = context.user_data.get(f'{userID}current_question_index', 0)
        
        if retry:
            # await update.effective_message.reply_text("Please re-answer the question.")
            
            # Provide only the "Suggest Answer" option
            keyboard = [
                [InlineKeyboardButton("Suggest Answer", callback_data='suggest_answer')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.effective_message.reply_text("Please re-answer the question.", reply_markup=reply_markup)
        else:
            if current_question_index < len(questions_list):
                previous_question = questions_list[current_question_index - 1] if current_question_index > 0 else ""
                current_question = questions_list[current_question_index].strip()
                user_answer = answers_list[current_question_index - 1] if current_question_index > 0 else ""
                selected_topic = context.user_data[f'{userID}selected_topic']
                list_previous_quetions.append(previous_question) 
                list_previous_answers.append(user_answer)
                # Generate an interactive question based on the previous question, user's answer, and the current question
                interactive_question = await generate_interactive_question(previous_question, user_answer, current_question, selected_topic,list_previous_quetions,list_previous_answers )
                # print("current question: ",current_question, "\ninteractive question: ",interactive_question)
                
                if interactive_question:
                    # Replace the original question in the questions_list with the generated interactive question
                    questions_list[current_question_index] = interactive_question
                    
                    context.user_data[f'{userID}current_question'] = interactive_question
                    
                    # Convert question to audio using Deepgram TTS API
                    try:
                        # await update.effective_message.reply_text("  ")
                        # print(len(list_previous_quetions), len(list_previous_answers))
                        await update.effective_message.reply_text(interactive_question)
                        audio_file_path = await convert_text_to_audio(interactive_question)
                        
                        with open(audio_file_path, 'rb') as audio:
                            await update.effective_message.reply_voice(voice=audio)
                        await update.effective_message.reply_text("Please record your answer.")
                        context.user_data[f'{userID}answering_question'] = True
                        print("Set answering_question to True")
                    except Exception as e:
                        print(f"Error converting text to audio: {e}")
                        # Retry the conversion without sending an error message
                        await update.effective_message.reply_text("Please record your answer.")
                        context.user_data[f'{userID}answering_question'] = True
                        print("Set answering_question to True")
                else:
                    current_question_index += 1
                    context.user_data[f'{userID}current_question_index'] = current_question_index
                    await ask_current_question(update, context)
            else:
                await show_results(update, context)
    except Exception as e:
        print("ask current question function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
# Helper function to move to the next question
async def move_to_next_question(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print("move to next question")
        current_question_index = context.user_data.get(f'{userID}current_question_index', 0)
        
        if current_question_index < len(questions_list) - 1:
            context.user_data[f'{userID}current_question_index'] = current_question_index + 1
            await ask_current_question(update, context)
        else:
            await show_results(update, context)
    except Exception as e:
        print("move to next question function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
# Helper function to show results
async def show_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print("show results")
        global questions_list, answers_list, voice_urls
        # print(questions_list)
        # print("------------------------------")
        # print(answers_list)
        # print("------------------------------")
        # print(voice_urls)
        # print("------------------------------")
        # Format the questions and answers into a single message
        result_message = "Here are your questions and answers:\n\n"
        for i in range(len(questions_list)):
            question = questions_list[i]
            answer = answers_list[i] if i < len(answers_list) else "No answer provided"
            result_message += f"Question: {question}\n\nAnswer: {answer}\n\n"
        
        # # Split the message into smaller chunks
        # max_length = 4096  # Maximum message length allowed by Telegram
        # message_chunks = [result_message[i:i+max_length] for i in range(0, len(result_message), max_length)]
        
        # # Send each chunk as a separate message
        # for chunk in message_chunks:
        #     await update.effective_message.reply_text(chunk)
        await send_long_message(update, context, result_message)
        # Prompt the user to continue to Part 2, retake Part 1, or end the test
        await update.effective_message.reply_text("Please select one of these options.", reply_markup=InlineKeyboardMarkup([
            [InlineKeyboardButton("Continue to Part 2", callback_data='continue_part_2')],
            [InlineKeyboardButton("Retake Part 1", callback_data='retake_part_1')],
            [InlineKeyboardButton("Show Results", callback_data='show_results')],
            [InlineKeyboardButton("End the Test", callback_data='end_test')]
        ]))
    except Exception as e:
        print("show results function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
# Function to generate questions using Groq API
async def generate_questions(topic):
    global questions_list
    try:
        print("generate questions")
        prompt = f"You are an IELTS Speaking examiner and now you are testing an IELTS candidate. This is Part 1 of the test, and you need to ask between 5 to 6 questions about this topic: {topic}. You should only ask questions and nothing else. First, ask 2 or 3 general questions about the candidate, followed by 4 or 5 questions on the given topic. Ensure your questions are exactly like those a real examiner would ask. Number each question."
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
        )
        result = chat_completion.choices[0].message.content
        questions = result.split('\n')
        
        valid_questions = [q.strip() for q in questions if re.match(r'^\d+\.\s', q.strip())]
        
        
        questions_list.extend(valid_questions)
        
        print("number of questions: ",len(questions_list))
        return valid_questions
    except Exception as e:
        print("Groq error switching to perplexity",e)
        
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
            model="llama-3-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        questions = result.split('\n')
        valid_questions = [q.strip() for q in questions if re.match(r'^\d+\.\s', q.strip())]
        
        # global questions_list
        questions_list.extend(valid_questions)
        
        print("number of questions: ",len(questions_list))
        return valid_questions
        # await update.message.reply_text(issue_message)
async def generate_interactive_question(previous_question, user_answer, next_question, selected_topic, pre_questions, pre_answers):
    try:
        # prompt = f"Previous Question: {previous_question}\nUser's Answer: {user_answer}\nNext Question: {next_question}\n\n"
        # prompt += f"Selected Topic for part1: {selected_topic}\n\n"
        # prompt += f"Based on the user's answer to the previous question, generate a more relevant and context-aware question that is related to the topic and the user's response (these quetions is part of IELTS Speaking part 1 and your quetios shold be simple and not complex to help the ielts candidate asnwer properly). If the next question is on a different topic, include a transitional phrase to smoothly move to the new topic. Provide only the modified question and the number of the quetion without any additional text (in the first quetion you might will not recieve any asnwer or previvous question or any text so just ask the same quetion again also your quetions should be simple and relevent to the topic {selected_topic} when the quetions refers to move to the topic note that first questios are about the IELTS Candidate )."
        # print(pre_questions)
        # print(pre_answers)
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
            model="llama3-70b-8192",
        )
        result = chat_completion.choices[0].message.content
        return result.strip()
    except Exception as e:
        print("Groq error switching to perplexity",e)
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
            model="llama-3-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        # result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        # result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        # print("feadback report generated")
        return result.strip()
# Function to convert text to audio using Deepgram TTS API
async def convert_text_to_audio(text):
    global  examiner_voice
    try:
        global  examiner_voice
        print("convert text to audio")
        if not text.strip():
            raise ValueError("Input text contains no characters.")
        
        if examiner_voice== "":
            examiner_voice = "Liv" 
        unreal_speech_api = random.choice(unreal_speech_API_keys)    
        response = requests.post(
            'https://api.v6.unrealspeech.com/stream',
            headers={
                'Authorization': unreal_speech_API1
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
            raise Exception(f"Failed to convert text to audio. Status code: {response.status_code}")
    except Exception as e:
        
        print("convert text to audio")
        if not text.strip():
            raise ValueError("Input text contains no characters.")
        
        if examiner_voice== "":
            examiner_voice = "Liv" 
        unreal_speech_api = random.choice(unreal_speech_API_keys) 
        response = requests.post(
            'https://api.v6.unrealspeech.com/stream',
            headers={
                'Authorization': unreal_speech_API2
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
            raise Exception(f"Failed to convert text to audio. Status code: {response.status_code}")
# Function to convert audio to text using Deepgram STT API
async def convert_audio_to_text(file_id, update, context):
    try:
        try:
            deppgarm_api = random.choice(deepgram_api_keys)
            deepgram_client = DeepgramClient(deppgarm_api)
            print("convert audio to text")
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
            response = deepgram_client.listen.prerecorded.v("1").transcribe_url(AUDIO_URL, options)
            transcript = response.to_json(indent=4)
            response_data = json.loads(transcript)
            transcript_text = response_data['results']['channels'][0]['alternatives'][0]['transcript']
            return transcript_text
        except Exception as e:
            deppgarm_api = random.choice(deepgram_api_keys)
            deepgram_client = DeepgramClient(deppgarm_api)
            print("convert audio to text")
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
            response = deepgram_client.listen.prerecorded.v("1").transcribe_url(AUDIO_URL, options)
            transcript = response.to_json(indent=4)
            response_data = json.loads(transcript)
            transcript_text = response_data['results']['channels'][0]['alternatives'][0]['transcript']
            return transcript_text
    except Exception as e:
        print("convert audio to text function",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
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
async def show_results_part1(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print("-----------------------FEEDBACK PART 1------------------------")
        global voice_urls, targeted_score
        # print("Voice URLs:", voice_urls)
        print("user_targetes_score: ",targeted_score)
        # Generate typical answers
        typical_answers = await generate_typical_answers(questions_list, answers_list)
        await send_long_message(update, context, f"Typical Answers:\n\n{typical_answers}")
        
        # Send the sticker and waiting message after sending typical answers
        animated_emoji = "⏳"  # Hourglass emoji
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        # await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
        # Send the initial waiting message with 0% progress and an empty progress bar
        progress_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Wait a few minutes until results are ready...\n\n[                             ] 0%")

        # Send a message asking the user to share the bot
        share_message = (
            f"Discover this IELTS Speaking  Bot! It simulates the IELTS speaking test and provides detailed feedback about your speaking skills and estimated IELTS band score. to help you improve. Try it for free now: https://t.me/ielts_speakingAI_bot"
        )
        keyboard = [
            [InlineKeyboardButton("Share the Bot", switch_inline_query=share_message)]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        share_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="while waiting the results. Would you like to share this bot?",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        # Define the steps of the processing
        steps = [
            "Transcribing answers...",
            "Analyzing responses...",
            "Generating feedback...",
            "Compiling results..."
        ]

        total_steps = len(steps) + len(mock_part1_questions) + len(mock_part3_questions) + 1  # +1 for Part 2 assessment
        current_step = 0

        def update_progress():
            nonlocal current_step
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_bar = "[" + "█" * (progress // 5) + " " * (20 - (progress // 5)) + "]"
            return progress, progress_bar
    
        scores_list = []
        assessment_tasks = []
        
        for i in range(len(questions_list)):
            audio_url = voice_urls[i]
            question_prompt = questions_list[i]
            task_type = "ielts_part1"  # Change as needed
            
            assessment_task = asyncio.create_task(assess_speech_async(audio_url, question_prompt, task_type))
            assessment_tasks.append(assessment_task)
            # print(assessment_task)
            # Update progress
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Wait a few minutes until results are ready...\n{progress_bar} {progress}%"
                # text=f"Wait a few minutes until results are ready...\n{progress_bar} {i*10}%"
            )
        # Wait for all assessment tasks to complete
        scores_results = await asyncio.gather(*assessment_tasks)
        
        for i, scores in enumerate(scores_results):
            if scores:
                scores_list.append(scores)
                print(f"Assessment successful for question {i+1}")
            else:
                print(f"Assessment failed for question {i+1}")
        
        if scores_list:
            # Delete the waiting message (sticker)
            # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
            
            overall_avg = sum(score["overall"] for score in scores_list) / len(scores_list)
            pronunciation_avg = sum(score["pronunciation"] for score in scores_list) / len(scores_list)
            fluency_avg = sum(score["fluency"] for score in scores_list) / len(scores_list)
            grammar_avg = sum(score["grammar"] for score in scores_list) / len(scores_list)
            vocabulary_avg = sum(score["vocabulary"] for score in scores_list) / len(scores_list)
            
            # overall_avg = (pronunciation_avg + fluency_avg + grammar_avg + vocabulary_avg) / 4
            # print(overall_avg)
            # Round the scores to the nearest 0.5
            overall_avg = round_to_ielts_score(overall_avg)
            pronunciation_avg = round_to_ielts_score(pronunciation_avg)
            fluency_avg = round_to_ielts_score(fluency_avg)
            grammar_avg = round_to_ielts_score(grammar_avg)
            vocabulary_avg = round_to_ielts_score(vocabulary_avg)
            overall_score = round_to_ielts_score(overall_avg)
            
            overall_scores = {"pronunciation": pronunciation_avg,
                            "fluency": fluency_avg,
                            "grammar": grammar_avg,
                            "vocabulary": vocabulary_avg,
                            "IELTS band score": overall_avg,
                            }
            
            # Generate feedback report
            feedback_report = await generate_feedback(scores_list, questions_list, answers_list, overall_scores)
            context.user_data[f'{userID}overall_feedback'] = feedback_report
            # Update progress
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                # text=f"Wait a few minutes until results are ready...\n{progress_bar} {progress}%"
                text=f"Wait a few minutes until results are ready...\n{progress_bar} 100%"
            )
            # Delete the waiting message and the share message
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=progress_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=share_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id,message_id=waiting_message.message_id)
            
            # Send feedback report
            await send_long_message(update, context, feedback_report)
            
            # audio_file_path = await convert_text_to_audio(feedback_report)
                        
            # with open(audio_file_path, 'rb') as audio:
            #         await update.effective_message.reply_voice(voice=audio)
        
            
            # Display feedback visualization
            await display_feedback(update, context, overall_avg, pronunciation_avg, fluency_avg, grammar_avg, vocabulary_avg)
            
            # Send the band score as text
            band_score = f"Your estimated IELTS band score is: {overall_avg:.1f}"
            await context.bot.send_message(chat_id=update.effective_chat.id, text=band_score)
            
            # Get the CEFR level based on the IELTS score
            cefr_level = get_cefr_level(overall_avg)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Your language level is: {cefr_level}")
            await append_speaking_score(update,context,"part1", overall_avg)
            await increment_practice_count(update, context)
            # Provide user options
            keyboard = [
                [InlineKeyboardButton("Continue to Part 2", callback_data='continue_part_2')],
                [InlineKeyboardButton("See Detailed Results", callback_data='detailed_results')],
                [InlineKeyboardButton("Translate", callback_data='translate_overall_feedback')],
                [InlineKeyboardButton("Retake Part 1", callback_data='retake_part_1')],
                [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            
            # Clear the lists after the feedback is completed
            # voice_urls.clear()
            # questions_list.clear()
            # answers_list.clear()
        else:
            await update.callback_query.message.reply_text("Failed to assess the answers. Please try again.")
    except Exception as e:
        print("show results part 1 function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def show_result2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print("-----------------------FEEDBACK PART 2------------------------")
        global part2_voice_urls
        analysis_data = analysis2_list
        # print("Voice URLs:", part2_voice_urls)
        print("user_targetes_score: ",targeted_score)
        # Send the sticker and waiting message after sending typical answers
        animated_emoji = "⏳"  # Hourglass emoji
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        # await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
        
        progress_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Wait a few minutes until results are ready...\n\n[                             ] 0%")

        # Send a message asking the user to share the bot
        share_message = (
            f"Discover this IELTS Speaking  Bot! It simulates the IELTS speaking test and provides detailed feedback about your speaking skills and estimated IELTS band score. to help you improve. Try it for free now: https://t.me/ielts_speakingAI_bot"
        )
        keyboard = [
            [InlineKeyboardButton("Share the Bot", switch_inline_query=share_message)]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        share_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="while waiting the results. Would you like to share this bot?",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        # Define the steps of the processing
        steps = [
            "Transcribing answers...",
            "Analyzing responses...",
            "Generating feedback...",
            "Compiling results..."
        ]

        total_steps = len(steps) + len(mock_part1_questions) + len(mock_part3_questions) + 1  # +1 for Part 2 assessment
        current_step = 0

        def update_progress():
            nonlocal current_step
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_bar = "[" + "█" * (progress // 5) + " " * (20 - (progress // 5)) + "]"
            return progress, progress_bar
    
    
    
    
    
    
        scores2_list = []
        assessment_tasks = []
        
        for i in range(len(part2_questions)):
            audio_url = part2_voice_urls[i]
            question_prompt = part2_questions[i]
            task_type = "ielts_part2"  # Change as needed
            
            assessment_task = asyncio.create_task(assess_part2_speech_async(audio_url, question_prompt, task_type))
            assessment_tasks.append(assessment_task)
            # print("assessment_task: ",assessment_task)
            # Update progress
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Wait a few minutes until results are ready...\n{progress_bar} {progress}%"
            )
        # Wait for all assessment tasks to complete
        scores_results = await asyncio.gather(*assessment_tasks)
        # print("scores_results ",scores_results)
        for i, scores in enumerate(scores_results):
            if scores:
                scores2_list.append(scores)
                # print("scores ",scores)
                print(f"Assessment successful for question {i+1}")
            else:
                print(f"Assessment failed for question {i+1}")
        
        if scores2_list:
            # print(scores2_list)
            # Delete the waiting message (sticker)
            # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
            
            #Get the scores from the assessment results
            overall_score = scores2_list[0].get("overall", 0)
            pronunciation_score = scores2_list[0].get("pronunciation", 0)
            fluency_score = scores2_list[0].get("fluency", 0)
            grammar_score = scores2_list[0].get("grammar", 0)
            vocabulary_score = scores2_list[0].get("vocabulary", 0)
            
            # Generate feedback report
            feedback2_report = await generate_feedback2(scores2_list, part2_questions, part2_answers, overall_score)
            context.user_data[f'{userID}overall_part2_feedback'] = feedback2_report
            # Send feedback report
            # Update progress
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                # text=f"Wait a few minutes until results are ready...\n{progress_bar} {progress}%"
                text=f"Wait a few minutes until results are ready...\n{progress_bar} 100%"
            )
            # Delete the waiting message and the share message
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=progress_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=share_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id,message_id=waiting_message.message_id)
            await send_long_message(update, context, feedback2_report)
            # audio_file_path = await convert_text_to_audio(feedback2_report)
                        
            # with open(audio_file_path, 'rb') as audio:
            #         await update.effective_message.reply_voice(voice=audio)
            #
            # Round the scores to the nearest 0.5
            overall_score = round_to_ielts_score(overall_score)
            pronunciation_score = round_to_ielts_score(pronunciation_score)
            fluency_score = round_to_ielts_score(fluency_score)
            grammar_score = round_to_ielts_score(grammar_score)
            vocabulary_score = round_to_ielts_score(vocabulary_score)
            
            # Display feedback visualization
            await display_feedback(update, context, overall_score, pronunciation_score, fluency_score, grammar_score, vocabulary_score)
            
            # Send the band score as text
            band_score = f"Your estimated IELTS band score for Part 2 is: {overall_score:.1f}"
            await context.bot.send_message(chat_id=update.effective_chat.id, text=band_score)
            
            # Get the CEFR level based on the IELTS score
            cefr_level = get_cefr_level(overall_score)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Your language level is: {cefr_level}")
            await append_speaking_score(update,context,"part2", overall_score)
            await increment_practice_count(update, context)
            # generate_pronunciation_visualization(analysis_data)
            # with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
            #     await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)
            # Provide user options
            keyboard = [
                [InlineKeyboardButton("Continue to Part 3", callback_data='continue_part3')],
                [InlineKeyboardButton("See Detailed Results", callback_data='detailed2_results')],
                [InlineKeyboardButton("Translate", callback_data='translate_overall2_feedback')],
                [InlineKeyboardButton("Retake Part 2", callback_data='retake_part2')],
                [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            
            # Clear the lists after the feedback is completed
            # voice_urls.clear()
            # questions_list.clear()
            # answers_list.clear()
        else:
            await update.callback_query.message.reply_text("Failed to assess the answers. Please try again.")
    except Exception as e:
        print("show results part 2",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def generate_suggested_answer(question, previous_answer, part_type):
    try:
        # print(previous_answer)
        prompt = f"Question: {question}\nuser Answer: {previous_answer}\nIELTS Speaking {part_type}\n\n"
        prompt += f"Provide a suggested response for the given IELTS speaking question type, ensuring that the answer is appropriate in length and complexity based on the part type [{part_type}] specified. If the part is part 1, the suggested answer should be simple and not too long. If it is part 3, the suggested answer should not be too short or too long. Please only provide the suggested answer without any additional content."
        
        
        # Use the LLM to generate the suggested answer
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
        )
        result = chat_completion.choices[0].message.content
        print("suggetion generated")
        return result
    except Exception as e:
        print("Groq error switching to perplexity",e)
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
            model="llama-3-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("feadback report generated")
        return result
async def send_long_message(update, context, message):
    num_mesages = len(message)
    try:
        max_length = 4096  # Maximum message length allowed by Telegram
        message_chunks = [message[i:i+max_length] for i in range(0, num_mesages, max_length)]
        
        for chunk in message_chunks:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=chunk)
    except Exception as e:
        print("send long messages function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def assess_speech_async(audio_url, question_prompt, task_type):
    # start_time = time.time()
    
    try:
        # Download the voice file from the URL
        response = requests.get(audio_url)
        
        # Generate a unique filename for the downloaded file
        filename = f"voice_{int(time.time())}.oga"
        
        # Save the downloaded file locally
        with open(filename, "wb") as file:
            file.write(response.content)
        # scores,analysis_data = assess_speech(filename, question_prompt, task_type)
        # response_json = scores  # Assuming assess_speech returns the JSON response directly
        # analysis_list.append(analysis_data)
        # # print(analysis_data)
        # if 'result' in response_json:
        #     result = response_json['result']
        #     scores = {
        #         "overall": result.get("overall", "N/A"),
        #         "pronunciation": result.get("pronunciation", "N/A"),
        #         "fluency": result.get("fluency_coherence", "N/A"),
        #         "grammar": result.get("grammar", "N/A"),
        #         "vocabulary": result.get("lexical_resource", "N/A"),
        #         "relevance": result.get("relevance", "N/A"),
        #         "transcription": result.get("transcription", "N/A"),
        #         # "pause_filler": result.get("pause_filler", {}),
        #         "sentences": [
        #             {
        #                 "sentence": sentence.get("sentence", ""),
        #                 "pronunciation": sentence.get("pronunciation", "N/A"),
        #                 "grammar": sentence.get("grammar", {}).get("corrected", "")
        #             }
        #             for sentence in result.get("sentences", [])
        #         ]
        #     }
        #     # analysis_list.append(scores)
        
        # # end_time = time.time()
        # # execution_time = end_time - start_time
        # # print(f"Execution time: {execution_time} seconds")
        # os.remove(filename)
        # # print(scores)
        # return scores
        scores, analysis_data = assess_speech(filename, question_prompt)
        analysis_list.append(analysis_data)
        # print(analysis_data)
        # Process the scores and analysis_data to match the expected output format
        processed_scores = {
            "overall": scores['ielts_score']['overall'],
            "pronunciation": scores['ielts_score']['pronunciation'],
            "fluency": scores['ielts_score']['fluency'],
            "grammar": scores['ielts_score']['grammar'],
            "vocabulary": scores['ielts_score']['vocabulary'],
            "relevance": scores['relevance']['class'],
            "transcription": scores['transcription'],
            # "sentences": []  # SpeechAce doesn't provide sentence-level data in the same format
        }

        # Add word-level details to the processed_scores
        # processed_scores["word_details"] = analysis_data["word_pronunciation_details"]

        # Clean up the temporary file
        os.remove(filename)
        # print(processed_scores)
        return processed_scores
    except Exception as e:
        print(f"Error assessing speech: {str(e)}")
        return None

async def assess_part2_speech_async(audio_url, question_prompt, task_type) :
    try:
        # Download the voice file from the URL
        response = requests.get(audio_url)
        # print(audio_url, question_prompt, task_type)
        # Generate a unique filename for the downloaded file
        filename = f"voice_{int(time.time())}.oga"
        
        # Save the downloaded file locally
        with open(filename, "wb") as file:
            file.write(response.content)
        
        # Assess the speech using the existing assess_speech function
        scores, analysis_data = assess_speech(filename, question_prompt, task_type)
        # response_json = scores  # Assuming assess_speech returns the JSON response directly
        analysis2_list.append(analysis_data)
        # if 'result' in response_json:
        #     result = response_json['result']
        #     scores = {
        #         "overall": result.get("overall", "N/A"),
        #         "pronunciation": result.get("pronunciation", "N/A"),
        #         "fluency": result.get("fluency_coherence", "N/A"),
        #         "grammar": result.get("grammar", "N/A"),
        #         "vocabulary": result.get("lexical_resource", "N/A"),
        #         "relevance": result.get("relevance", "N/A"),
        #         "transcription": result.get("transcription", "N/A"),
        #         "pause_filler": result.get("pause_filler", {}),
        #         "sentences": [
        #             {
        #                 "sentence": sentence.get("sentence", ""),
        #                 "pronunciation": sentence.get("pronunciation", "N/A"),
        #                 "grammar": sentence.get("grammar", {}).get("corrected", "")
        #             }
        #             for sentence in result.get("sentences", [])
        #         ]
        #     }
        
        # os.remove(filename)
        # # print(analysis_data)
        # return scores
        processed_scores = {
            "overall": scores['ielts_score']['overall'],
            "pronunciation": scores['ielts_score']['pronunciation'],
            "fluency": scores['ielts_score']['fluency'],
            "grammar": scores['ielts_score']['grammar'],
            "vocabulary": scores['ielts_score']['vocabulary'],
            "relevance": scores['relevance']['class'],
            "transcription": scores['transcription'],
            # "sentences": []  # SpeechAce doesn't provide sentence-level data in the same format
        }

        # Add word-level details to the processed_scores
        # processed_scores["word_details"] = analysis_data["word_pronunciation_details"]

        # Clean up the temporary file
        os.remove(filename)
        # print(processed_scores)
        return processed_scores
    except Exception as e:
        print(f"Error assessing speech: {str(e)}")
        return None
async def generate_typical_answers(questions, answers):
    try:
        prompt = "Provide typical answers for the following IELTS speaking questions and user answers:\n\n"
        for i in range(len(questions)):
            prompt += f"Question {i+1}: {questions[i]}\nUser Answer: {answers[i]}\n\n"
        
        prompt += "Provide a typical answer for each question based on the user's response. Each typical answer should be presented below the original answer. These responses are from a user who was answering the IELTS Speaking test, and the typical answers should be of high quality to help the user improve for the next time. Please only include the structured format of: 1- Question, 2- Answer, 3- Typical Answer. Do not include any other text except what has been specified."

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
        )
        result = chat_completion.choices[0].message.content
        # Remove Markdown formatting characters using regular expressions
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        return result
    except Exception as e:
        print("Groq error switching to perplexity",e)
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
            model="llama-3-70b-instruct",
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
        print("dispaly feedback function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def generate_feedback(scores_list, questions, answers, overall_avg):
    try:
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
            model="llama3-70b-8192",
        )
        result = chat_completion.choices[0].message.content
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("feadback report generated")
        return result
    except Exception as e:
        print("Groq error switching to perplexity",e)
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
            model="llama-3-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("feadback report generated")
        return result
async def generate_feedback2(scores_list, questions, answers,overall_score):
    try:
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
            model="llama3-70b-8192",
        )
        result = chat_completion.choices[0].message.content
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("feadback report generated")
        return result
    except Exception as e:
        print("Groq error switching to perplexity",e)
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
            model="llama-3-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("feadback report generated")
        return result



async def generate_detailed_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        detailed_feedback = []
        global questions_list, answers_list, analysis_list
        for i in range(len(questions_list)):
            question = questions_list[i]
            user_answer = answers_list[i]
            analysis_data = analysis_list[i]
            user_voice_url = voice_urls[i]
            # print(user_voice_url)
            prompt = f"Question: {question}\nUser Answer: {user_answer}\nAnalysis Data: {analysis_data}\n\n"
            prompt += """Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance. Generate useful feedback that helps the user understand their performance in depth.
            you should orginze it in a clear and good feedback to the IELTS Candidate he expects to kmow his mistakes and improve next time also it is a good idea to write the quetion in the feedback and the answer and then start your analysis
            
            """
            
            feedback = await generate_feedback_with_llm(prompt)
            detailed_feedback.append(feedback)
            detailed_feedback_list.append(feedback)
            # await context.bot.send_message(chat_id=update.effective_chat.id, text=feedback)
            await send_long_message(update, context, feedback)
            # Generate and send the pronunciation visualization image
            generate_pronunciation_visualization(analysis_data)
            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)
            # Send a message to compare pronunciation
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Now, let's compare your pronunciation with a native speaker's pronunciation.")
            
            # Download the voice file from the URL
            response = requests.get(user_voice_url)
            voice_filename = f"user_voice_{i}.oga"
            with open(voice_filename, "wb") as file:
                file.write(response.content)
            
            # Send user's original voice
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Your Voice:")
            with open(voice_filename, "rb") as user_voice_file:
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=user_voice_file)
            
            # Delete the user's voice file from disk
            os.remove(voice_filename)
            await asyncio.sleep(2)
            # Generate native speaker's audio with slow speed
            slow_audio_path = await convert_answer_to_audio(user_answer, speed='-0.56')
            
            # Send native speaker's audio with slow speed
            with open(slow_audio_path, 'rb') as slow_audio_file:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Slow Speed):")
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
            os.remove(slow_audio_path)
            await asyncio.sleep(2)
            # Generate native speaker's audio with normal speed
            normal_audio_path = await convert_answer_to_audio(user_answer, speed='0')
            
            # Send native speaker's audio with normal speed
            with open(normal_audio_path, 'rb') as normal_audio_file:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Normal Speed):")
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
            os.remove(normal_audio_path)
            # Send a message to encourage comparison
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Listen to your voice and compare it with the native speaker's pronunciation at both speeds.")
        return detailed_feedback
    except Exception as e:
        print("generate detailed feedback part 1 function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def generate_detailed2_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        detailed_feedback = []
        global part2_questions, part2_answers, analysis2_list,part2_voice_urls
        for i in range(len(part2_questions)):
            question = part2_questions[i]
            user_answer = part2_answers[i]
            analysis_data = analysis2_list[i]
            user_voice_url = part2_voice_urls[i]
            # print(user_voice_url)
            # print(analysis_data)
            prompt = f"Question: {question}\nUser Answer: {user_answer}\nAnalysis Data: {analysis_data}\n\n"
            prompt += """Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance. Generate useful feedback that helps the user understand their performance in depth and should be in details.
            you should orginze it in a clear and good feedback to the IELTS Candidate he expects to kmow his mistakes and improve next time also it is a good idea to write the quetion in the feedback and the answer and then start your analysis
            
            """
            
            feedback = await generate_feedback_with_llm(prompt)
            detailed_feedback.append(feedback)
            detailed_feedback2_list.append(feedback)
            # await context.bot.send_message(chat_id=update.effective_chat.id, text=feedback)
            await send_long_message(update, context, feedback)
            # Generate and send the pronunciation visualization image
            generate_pronunciation_visualization(analysis_data)
            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)
            # Send a message to compare pronunciation
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Now, let's compare your pronunciation with a native speaker's pronunciation.")
            
            # Download the voice file from the URL
            response = requests.get(user_voice_url)
            voice_filename = f"user_voice_{i+1}.oga"
            with open(voice_filename, "wb") as file:
                file.write(response.content)
            # await asyncio.sleep(2)
            # Send user's original voice
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Your Voice:")
            with open(voice_filename, "rb") as user_voice_file:
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=user_voice_file)
            
            # Delete the user's voice file from disk
            os.remove(voice_filename)
            # print(user_answer)
            # Generate native speaker's audio with slow speed
            # await asyncio.sleep(2)
            slow_audio_path = await convert_answer_to_audio(user_answer, speed='-0.56')

            if slow_audio_path:
                # Send native speaker's audio with slow speed
                with open(slow_audio_path, 'rb') as slow_audio_file:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Slow Speed):")
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
                os.remove(slow_audio_path)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to generate native speaker's voice (slow speed).")

            # Generate native speaker's audio with normal speed
            normal_audio_path = await convert_answer_to_audio(user_answer, speed='0')

            if normal_audio_path:
                # Send native speaker's audio with normal speed
                with open(normal_audio_path, 'rb') as normal_audio_file:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Normal Speed):")
                    await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
                os.remove(normal_audio_path)
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to generate native speaker's voice (normal speed).")
        return detailed_feedback
    except Exception as e:
        print("generate detailed feedback part 2 function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

def generate_pronunciation_visualization(answer_data):
    try:
        # Extract the word pronunciation details from the answer data
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
        
        for word_info in word_pronunciation_details:
            word = word_info['word']
            word_width, word_height = draw.textbbox((0, 0), word, font=text_font)[2:]
            
            # Check if word fits in the current line, if not move to the next line
            if y + word_width > max_line_width:
                y += word_height + 10
            
            y += word_height + 10
        
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
        
        # Draw words with color coding
        y = title_y + title_height + 40
        x = padding
        for word_info in word_pronunciation_details:
            word = word_info['word']
            score = float(word_info['pronunciation'])  # Ensure score is a float
            
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
        print("generate pronunciation visualization function ", e)
        # await update.message.reply_text(issue_message)
async def convert_answer_to_audio(user_answer, speed):
    global  examiner_voice
    if examiner_voice== "":
            examiner_voice = "Liv" 

    try:
        response = requests.post(
            'https://api.v6.unrealspeech.com/stream',
            headers={
                'Authorization': unreal_speech_API1
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
            print(f"Error converting answer to audio. Status code: {response.status_code}")
            return ""  # Return an empty string instead of None

    except Exception as e:
        # global  examiner_voice
        if examiner_voice== "":
                examiner_voice = "Liv" 

        
        response = requests.post(
                'https://api.v6.unrealspeech.com/stream',
                headers={
                    'Authorization': unreal_speech_API2
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
            print(f"Error converting answer to audio. Status code: {response.status_code}")
            return ""  # Return an empty string instead of None


    
async def generate_feedback_with_llm(prompt):
    # Use the LLM to generate the detailed feedback based on the prompt
    # You can integrate with your chosen LLM service here
    # Example:
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt}
            ],
            model="llama3-70b-8192",
        )
        feedback = chat_completion.choices[0].message.content
        feedback = re.sub(r'\*', '', feedback)  # Remove asterisks (*)
        feedback = re.sub(r'#', '', feedback)  # Remove hash symbols (#)
        return feedback
    except Exception as e:
        print("Groq error switching to perplexity",e)
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
            model="llama-3-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        feedback = (response.choices[0].message.content)
        feedback = re.sub(r'\*', '', feedback)  # Remove asterisks (*)
        feedback = re.sub(r'#', '', feedback)  # Remove hash symbols (#)
        return feedback
async def translate_feedback(user_id, feedback, update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Retrieve the user's native language from the database
        user = supabase.table('ielts_speaking_users').select('native_language').eq('user_id', user_id).execute()
        # print(user)
        native_language = user.data[0]['native_language']
        print("language ",native_language)
        
        # Use the Gemini API to translate the feedback into the user's native language
        # genai.configure(api_key=Gemini_API_Key)
        pro =  'gemini-1.5-pro'
        flash = 'gemini-1.5-flash'
        
        model_type =  [pro, flash]
        # prompt = f"Translate the following text from English to {native_language}:\n\n{feedback}"
        prompt = f"""
            Translate the provided IELTS evaluation of speaking test text from english into {native_language}. Ensure that the translation is accurate, contextually appropriate, and adheres to the linguistic standards of {native_language}.
            Instructions:

            1- Content Focus: Only include the evaluation text. Exclude any non-evaluative content to maintain the focus on the assessment aspects of the text.
            2- Rearrange the text to align with the typical format and flow of {native_language}, while preserving the original order and organization of content.
            3- Language Specifics:
            - You should translate based on the required context. If the context requires any word or sentence to remain in English, leave it in English for grammar or pronunciation or any place in the text it always be between two ("") and make them inside the qoutes (""). Be cautious when you encounter this.
            - Adjust the sentence structure and phrasing to fit the grammatical and stylistic norms of {native_language}, ensuring that the translation reads naturally to native speakers.
            4- Accuracy and Contextual Integrity:
            -Carefully maintain the original context and meaning of the evaluation text during translation.
            - Ensure that all translated terms and phrases are appropriate for the context and do not alter the evaluative tone or content. 
            
            the evaluation text that needed to translates is:\n\n
            {feedback}
        """
        supported_languages = [
        "English", "Español", "Français", "Deutsch", "Italiano", "Português", "Русский", "日本語", "한국어", "中文",
        "العربية", "हिन्दी", "বাংলা", "ਪੰਜਾਬੀ", "Tiếng Việt", "Türkçe",
        "Polski", "Українська", "Nederlands", "Ελληνικά", "Svenska", "Norsk", "Dansk", "Suomi", "Čeština", "Română",
        "Magyar", "Српски", "Hrvatski", "Български", "Lietuvių", "Latviešu", "Eesti", "Slovenščina", "Slovenčina",
        "Kiswahili", "Bahasa Indonesia",
        "ภาษาไทย",  "Монгол хэл", "English",
        "Spanish",
        "French",
        "German",
        "Italian",
        "Portuguese",
        "Russian",
        "Japanese",
        "Korean",
        "Chinese",
        "Arabic",
        "Hindi",
        "Bengali",
        "Punjabi",
        "Vietnamese",
        "Turkish",
        "Polish",
        "Ukrainian",
        "Dutch",
        "Greek",
        "Swedish",
        "Norwegian",
        "Danish",
        "Finnish",
        "Czech",
        "Romanian",
        "Hungarian",
        "Serbian",
        "Croatian",
        "Bulgarian",
        "Lithuanian",
        "Latvian",
        "Estonian",
        "Slovenian",
        "Slovak",
        "Swahili",
        "Indonesian",
        "Thai",
        "Mongolian",
        "Hebrew"
        "עברית"  , 
        
    ]
        max_retries = 3
        retry_count = 0
        if native_language in supported_languages:
            # Use the Gemini API to translate the feedback into the user's native language
            while retry_count < max_retries:
                try:
                    used_key = random.choice(keys)
                    used_model = random.choice(model_type)
                    print("used model ",used_model)
                    genai.configure(api_key=used_key)
                    model = genai.GenerativeModel(used_model)
                    # prompt = f"Translate the following text from English to {native_language}:\n\n{feedback}"
                    response = model.generate_content(prompt)
                    response.resolve()
                    translated_feedback = response.text
                    translated_feedback = re.sub(r'\*', '', translated_feedback)  # Remove asterisks (*)
                    translated_feedback = re.sub(r'#', '', translated_feedback)  # Remove hash symbols (#)
                    return translated_feedback
                except Exception as e:  
                    retry_count += 1
                    print(f"Translation error occurred: {str(e)}. Retrying ({retry_count}/{max_retries})...")
                # If the maximum number of retries is reached, send a failure message to the user
            # await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, the translation service is currently unavailable. Please try again later.")
            keyboard = [
            # [InlineKeyboardButton("Continue to Part 2", callback_data='continue_part_2')],
            # [InlineKeyboardButton("Retake Part 1", callback_data='retake_part_1')],
            [InlineKeyboardButton("End the Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, the translation service is currently unavailable. Please try again later.", reply_markup=reply_markup)
        
        else:
            # Return None to indicate that translation is not available
            return None
    except Exception as e:
        print("translate feedback",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def start_part2_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Randomly select a question from the list of Part 2 questions
        # question = random.choice(questions_part2)
        # context.user_data['part2_question'] = question  # Store the question in context.user_data
        # Randomly select a topic from the dictionary
        selected_topic = random.choice(list(ielts_questions.keys()))

        # Extract the Part 2 question and Part 3 questions for the selected topic
        part2_question = ielts_questions[selected_topic]["part_2_question"]
        part_3_questions = ielts_questions[selected_topic]["part_3_questions"]

        # Store the Part 2 question and Part 3 questions in the context
        context.user_data[f'{userID}part2_question'] = part2_question
        context.user_data['part3_questions'] = part3_questions
            # Clear the part3_questions list before adding new questions
        part3_questions.clear()

        # Add the Part 3 questions to the part3_questions list
        part3_questions.extend(part_3_questions)
        # print(part3_questions)
        keyboard = [
            [InlineKeyboardButton("Change Question", callback_data='change_question')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        # Convert the question to audio using the Unrealspeech API
        audio_path = await convert_text_to_audio(part2_question)
        
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

        for remaining in range(59, 0, -1):
            await asyncio.sleep(1)
            try:
                await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=countdown_message.message_id, text=f"{remaining} seconds remaining...")
            except Exception as e:
                print(f"Failed to update countdown message: {e}")

        # Delete the countdown message and hourglass emoji
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=countdown_message.message_id)
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=preparation_message.message_id)
        # Request the user to record their answer
        context.user_data[f'{userID}part2_answering'] = True
        recording_message = await update.effective_message.reply_text("Please record your answer. It should be between 1 to 2 minutes long.")
        # Store the message IDs in the context
        context.user_data[f'{userID}part2_question_message_id'] = question_message.message_id
        context.user_data[f'{userID}part2_audio_message_id'] = audio_message.message_id
        context.user_data[f'{userID}part2_preparation_message_id'] = preparation_message.message_id
        context.user_data[f'{userID}part2_waiting_message_id'] = waiting_message.message_id
        context.user_data[f'{userID}part2_recording_message_id'] = recording_message.message_id
    except Exception as e:
        print("start part 2 test",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)


async def start_part3_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("start part 3 test")
    try:
        context.user_data[f'{userID}part3_answers'] = {}
        context.user_data[f'{userID}part3_current_question_index'] = 0
        await update.effective_message.reply_text("IELTS Speaking Part 3:\n\n"
                f"Now, we will begin Part 3 of the IELTS Speaking test. In this part, I will ask you ({len(part3_questions)}) abstract and complex questions related to the topic from Part 2. Please answer each question in detail, with as much depth as possible. \n\nLet's start")
        time.sleep(3)
        await ask_part3_question(update, context)
    except Exception as e:
        print("start part 3 test function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def ask_part3_question(update: Update, context: ContextTypes.DEFAULT_TYPE, retry=False):
    print("ask part 3 question")
    try:
        global part3_questions, part3_answers
        current_question_index = context.user_data.get(f'{userID}part3_current_question_index', 0)
        # print(len(part3_questions))
        # print(part3_questions)
        if retry:
            # await update.effective_message.reply_text("Please re-answer the question.")
            
            # Provide only the "Suggest Answer" option
            keyboard = [
                [InlineKeyboardButton("Suggest Answer", callback_data='part3_suggest_answer')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.effective_message.reply_text("Please re-answer the question.", reply_markup=reply_markup)
        else:
            if current_question_index < len(part3_questions):
                current_question = part3_questions[current_question_index].strip()
                user_answer = part3_answers[current_question_index] if current_question_index < len(part3_answers) else ""
                context.user_data[f'{userID}current_part3_question'] = current_question
                question_number = current_question_index + 1
                formatted_message = f"{question_number}. {current_question}"
                # print(part3_questions[current_question_index])
                # print(len(part3_questions))
                # Convert question to audio using Deepgram TTS API
                try:
                    await update.effective_message.reply_text(formatted_message)
                    audio_file_path = await convert_text_to_audio(formatted_message)
                    
                    with open(audio_file_path, 'rb') as audio:
                        await update.effective_message.reply_voice(voice=audio)
                    await update.effective_message.reply_text("Please record your answer.")
                    context.user_data[f'{userID}answering_part3_question'] = True
                    print("Set answering_part3_question to True")
                except Exception as e:
                    print(f"Error converting text to audio: {e}")
                    # Retry the conversion without sending an error message
                    await update.effective_message.reply_text("Please record your answer.")
                    context.user_data[f'{userID}answering_part3_question'] = True
                    print("Set answering_part3_question to True")
            else:
                await show_part3_summary(update, context)
    except Exception as e:
        print("ask part 3 questions function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def part3_suggest_answer(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()

        question = context.user_data[f'{userID}current_part3_question']
        previous_answer = context.user_data[f'{userID}current_part3_answer']

        suggested_answer = await generate_suggested_answer(question, previous_answer, "part 3" )
        await query.edit_message_reply_markup(reply_markup=None)
        await update.effective_message.reply_text(f"Suggested Answer:\n\n{suggested_answer}")
        await update.effective_message.reply_text("Please record your answer again.")
    except Exception as e:
        print("part 3 suggested answer function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def show_part3_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        summary_message = "Part 3 Questions and Answers:\n\n"
        for i in range(len(part3_questions)):
            question = part3_questions[i]
            answer = part3_answers[i]
            summary_message += f"Question {i+1}: {question}\nAnswer: {answer}\n\n"

        await update.effective_message.reply_text(summary_message)

        keyboard = [
            [InlineKeyboardButton("Show Results", callback_data='part3_show_results')],
            [InlineKeyboardButton("Retake Part 3", callback_data='part3_retake')],
            [InlineKeyboardButton("End Test", callback_data='end_test')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.effective_message.reply_text("What would you like to do next?", reply_markup=reply_markup)
    except Exception as e:
        print("show part 3 summary function  ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def assess_part3_speech_async(audio_url, question_prompt, task_type):
    try:
        # Download the voice file from the URL
        response = requests.get(audio_url)
        
        # Generate a unique filename for the downloaded file
        filename = f"voice_{int(time.time())}.oga"
        
        # Save the downloaded file locally
        with open(filename, "wb") as file:
            file.write(response.content)
        
        # Assess the speech using the existing assess_speech function
        scores, analysis_data = assess_speech(filename, question_prompt, task_type)
        # response_json = scores  # Assuming assess_speech returns the JSON response directly
        analysis3_list.append(analysis_data)
        # print(analysis_data)
        # print('\n\nanalysis_data added succussfuly')
        # if 'result' in response_json:
        #     result = response_json['result']
        #     scores = {
        #         "overall": result.get("overall", "N/A"),
        #         "pronunciation": result.get("pronunciation", "N/A"),
        #         "fluency": result.get("fluency_coherence", "N/A"),
        #         "grammar": result.get("grammar", "N/A"),
        #         "vocabulary": result.get("lexical_resource", "N/A"),
        #         "relevance": result.get("relevance", "N/A"),
        #         "transcription": result.get("transcription", "N/A"),
        #         "pause_filler": result.get("pause_filler", {}),
        #         "sentences": [
        #             {
        #                 "sentence": sentence.get("sentence", ""),
        #                 "pronunciation": sentence.get("pronunciation", "N/A"),
        #                 "grammar": sentence.get("grammar", {}).get("corrected", "")
        #             }
        #             for sentence in result.get("sentences", [])
        #         ]
        #     }
        
        # os.remove(filename)
        # return scores
        processed_scores = {
            "overall": scores['ielts_score']['overall'],
            "pronunciation": scores['ielts_score']['pronunciation'],
            "fluency": scores['ielts_score']['fluency'],
            "grammar": scores['ielts_score']['grammar'],
            "vocabulary": scores['ielts_score']['vocabulary'],
            "relevance": scores['relevance']['class'],
            "transcription": scores['transcription'],
            # "sentences": []  # SpeechAce doesn't provide sentence-level data in the same format
        }

        # Add word-level details to the processed_scores
        # processed_scores["word_details"] = analysis_data["word_pronunciation_details"]

        # Clean up the temporary file
        os.remove(filename)
        # print(processed_scores)
        return processed_scores
    except Exception as e:
        print(f"Error assessing speech: {str(e)}")
        return None

async def generate_feedback3(scores_list, questions, answers, overall_avg):
    try:
        global targeted_score
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
            model="llama3-70b-8192",
        )
        result = chat_completion.choices[0].message.content
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("Feedback report generated")
        return result
    except Exception as e:
        print("Groq error switching to perplexity",e)
        
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
            model="llama-3-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        result = (response.choices[0].message.content)
        result = re.sub(r'\*', '', result)  # Remove asterisks (*)
        result = re.sub(r'#', '', result)  # Remove hash symbols (#)
        print("Feedback report generated")
        return result
async def generate_detailed3_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        global part3_questions, part3_answers, analysis3_list, part3_voice_urls
        # print(part3_voice_urls)
        # print(len(analysis3_list))
        # print(len(part3_voice_urls))
        detailed_feedback = []
        
        for i in range(len(part3_questions)):
            question = part3_questions[i]
            user_answer = part3_answers[i]
            analysis_data = analysis3_list[i]
            # print(f"analysis_data{i}:\n",analysis_data)
            user_voice_url = part3_voice_urls[i]
            # print( f"url_voice{i}:\n",user_voice_url )
            prompt = f"Question: {question}\nUser Answer: {user_answer}\nAnalysis Data: {analysis_data}\n\n"
            prompt += """Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance. Generate useful feedback that helps the user understand their performance in depth.
            You should organize it in a clear and good feedback to the IELTS Candidate. They expect to know their mistakes and improve next time. Also, it is a good idea to write the question in the feedback and the answer and then start your analysis.
            """
            
            feedback = await generate_feedback_with_llm(prompt)
            detailed_feedback.append(feedback)
            detailed_feedback3_list.append(feedback)
            # await context.bot.send_message(chat_id=update.effective_chat.id, text=feedback)
            await send_long_message(update, context, feedback)
            # Generate and send the pronunciation visualization image
            generate_pronunciation_visualization(analysis_data)
            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)
            
            # Send a message to compare pronunciation
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Now, let's compare your pronunciation with a native speaker's pronunciation.")
            
            # Download the voice file from the URL
            response = requests.get(user_voice_url)
            voice_filename = f"user_voice_{i+1}.oga"
            with open(voice_filename, "wb") as file:
                file.write(response.content)
            await asyncio.sleep(2)
            # Send user's original voice
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Your Voice:")
            with open(voice_filename, "rb") as user_voice_file:
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=user_voice_file)
            
            # Delete the user's voice file from disk
            os.remove(voice_filename)
            await asyncio.sleep(2)
            # Generate native speaker's audio with slow speed
            slow_audio_path = await convert_answer_to_audio(user_answer, speed='-0.56')
            
            # Send native speaker's audio with slow speed
            with open(slow_audio_path, 'rb') as slow_audio_file:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Slow Speed):")
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
            os.remove(slow_audio_path)
            await asyncio.sleep(2)
            # Generate native speaker's audio with normal speed
            normal_audio_path = await convert_answer_to_audio(user_answer, speed='0')
            
            # Send native speaker's audio with normal speed
            with open(normal_audio_path, 'rb') as normal_audio_file:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Normal Speed):")
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
            os.remove(normal_audio_path)
            # Send a message to encourage comparison
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Listen to your voice and compare it with the native speaker's pronunciation at both speeds.")
        
        return detailed_feedback 
    except Exception as e:
        print("generate detailed feedback part 3 function  ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def part3_show_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print("-----------------------FEEDBACK PART 3------------------------")
        global part3_questions, part3_answers, part3_voice_urls, analysis3_list
        query = update.callback_query
        await query.answer()
        print("user_targetes_score: ",targeted_score)
        typical_answers = await generate_typical_answers(part3_questions, part3_answers)
        await send_long_message(update, context, f"Typical Answers for Part 3:\n\n{typical_answers}")

        
        # Send the sticker and waiting message after sending typical answers
        animated_emoji = "⏳"  # Hourglass emoji
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        # await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
        
        progress_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Wait a few minutes until results are ready...\n\n[                             ] 0%")

        # Send a message asking the user to share the bot
        share_message = (
            f"Discover this IELTS Speaking  Bot! It simulates the IELTS speaking test and provides detailed feedback about your speaking skills and estimated IELTS band score. to help you improve. Try it for free now: https://t.me/ielts_speakingAI_bot"
        )
        keyboard = [
            [InlineKeyboardButton("Share the Bot", switch_inline_query=share_message)]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        share_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="while waiting the results. Would you like to share this bot?",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        # Define the steps of the processing
        steps = [
            "Transcribing answers...",
            "Analyzing responses...",
            "Generating feedback...",
            "Compiling results..."
        ]

        total_steps = len(steps) + len(mock_part1_questions) + len(mock_part3_questions) + 1  # +1 for Part 2 assessment
        current_step = 0

        def update_progress():
            nonlocal current_step
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_bar = "[" + "█" * (progress // 5) + " " * (20 - (progress // 5)) + "]"
            return progress, progress_bar
        scores_list = []
        assessment_tasks = []
        for i in range(len(part3_questions)):
            audio_url = part3_voice_urls[i]
            question_prompt = part3_questions[i]
            task_type = "ielts_part3"  # Change as needed
            
            assessment_task = asyncio.create_task(assess_part3_speech_async(audio_url, question_prompt, task_type))
            assessment_tasks.append(assessment_task)
            # print(assessment_task)
            # Update progress
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                # text=f"Wait a few minutes until results are ready...\n{progress_bar} {progress}%"
                text=f"Wait a few minutes until results are ready...\n{progress_bar} {i*10}%"
            )
        # Wait for all assessment tasks to complete
        scores_results = await asyncio.gather(*assessment_tasks)
        
        for i, scores in enumerate(scores_results):
            if scores:
                scores_list.append(scores)
                print(f"Assessment successful for question {i+1}")
            else:
                print(f"Assessment failed for question {i+1}")

        print("All assessments completed.") 
        # print('len(analysis3_list) in show_results\n',len(analysis3_list))
        if scores_list:
            # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
            overall_avg = sum(score["overall"] for score in scores_list) / len(scores_list)
            # overall_score = round_to_ielts_score(overall_avg)
            pronunciation_avg = sum(score["pronunciation"] for score in scores_list) / len(scores_list)
            fluency_avg = sum(score["fluency"] for score in scores_list) / len(scores_list)
            grammar_avg = sum(score["grammar"] for score in scores_list) / len(scores_list)
            vocabulary_avg = sum(score["vocabulary"] for score in scores_list) / len(scores_list)
            # overall_avg = (pronunciation_avg + fluency_avg + grammar_avg + vocabulary_avg) / 4
            # Round the scores to the nearest 0.5
            overall_avg = round_to_ielts_score(overall_avg)
            pronunciation_avg = round_to_ielts_score(pronunciation_avg)
            fluency_avg = round_to_ielts_score(fluency_avg)
            grammar_avg = round_to_ielts_score(grammar_avg)
            vocabulary_avg = round_to_ielts_score(vocabulary_avg)
            overall_scores = {"pronunciation": pronunciation_avg,
                            "fluency": fluency_avg,
                            "grammar": grammar_avg,
                            "vocabulary": vocabulary_avg,
                            "IELTS band score": overall_avg,
                            }
            overall_feedback3 = await generate_feedback3(scores_list, part3_questions, part3_answers, overall_scores)
            context.user_data[f'{userID}overall_part3_feedback'] = overall_feedback3
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                # text=f"Wait a few minutes until results are ready...\n{progress_bar} {progress}%"
                text=f"Wait a few minutes until results are ready...\n{progress_bar} 100%"
            )      
            # Delete the waiting message and the share message
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=progress_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=share_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)      
            await send_long_message(update, context, overall_feedback3)
            # audio_file_path = await convert_text_to_audio(overall_feedback3)
            # Update progress
            
            # with open(audio_file_path, 'rb') as audio:
            #         await update.effective_message.reply_voice(voice=audio)
            
        
            # Display feedback visualization
            await display_feedback(update, context, overall_avg, pronunciation_avg, fluency_avg, grammar_avg, vocabulary_avg)
            
            # Send the band score as text
            band_score = f"Your estimated IELTS band score is: {overall_avg:.1f}"
            await context.bot.send_message(chat_id=update.effective_chat.id, text=band_score)
            
            # Get the CEFR level based on the IELTS score
            cefr_level = get_cefr_level(overall_avg)
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Your language level is: {cefr_level}")
            await append_speaking_score(update,context,"part3", overall_avg)
            await increment_practice_count(update, context)
            keyboard = [
                [InlineKeyboardButton("Show Detailed Results", callback_data='part3_detailed_results')],
                [InlineKeyboardButton("Translate", callback_data='part3_translate_feedback')],
                [InlineKeyboardButton("Retake Part 3", callback_data='part3_retake')],
                [InlineKeyboardButton("End Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Failed to assess the answers. Please try again.")
    except Exception as e:
        print("part 3 show results function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def part3_detailed_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()

        await generate_detailed3_feedback(update, context)
        # detailed_feedback = await generate_detailed3_feedback(update, context)
        # for feedback in detailed_feedback:
        #     await asyncio.sleep(4)
        #     await send_long_message(update, context, feedback)

        # keyboard = [
        #     [InlineKeyboardButton("Retake Part 3", callback_data='part3_retake')],
        #     [InlineKeyboardButton("Translate", callback_data='part3_translate_detailed_feedback')],
        #     [InlineKeyboardButton("End Test", callback_data='end_test')]
        # ]
        # reply_markup = InlineKeyboardMarkup(keyboard)
        # await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
    except Exception as e:
        print("part 3 detailed results function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
#-----------------MOCK TEST FUNCTIONS-----------------

async def start_mock_test(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        #--------------PART1-----------------
        questions_list.clear()
        answers_list.clear()
        detailed_feedback_list.clear()
        voice_urls.clear()  # Replace with actual URLs
        questions.clear()
        analysis_list.clear()
        list_previous_quetions.clear()
        list_previous_answers.clear()
        #--------------PART2-----------------
        part2_voice_urls.clear()  # List to store Part 2 voice URLs
        part2_questions.clear()  # List to store Part 2 questions
        part2_answers.clear()
        analysis2_list.clear()
        detailed_feedback2_list.clear()
        #--------------PART3-----------------
        part3_voice_urls.clear()  # List to store Part 3 voice URLs
        part3_questions.clear()  # List to store Part 3 questions
        part3_answers.clear()
        analysis3_list.clear()
        detailed_feedback3_list.clear()
        # Clear the mock test lists
        mock_part1_questions.clear()
        mock_part1_answers.clear()
        mock_part1_voice_urls.clear()
        list_previous_quetions.clear()
        list_previous_answers.clear()
        mock_part2_questions.clear()
        mock_part2_answers.clear()
        mock_part2_voice_urls.clear()
        
        mock_part3_questions.clear()
        mock_part3_answers.clear()
        mock_part3_voice_urls.clear()
        context.user_data.clear()
        global part_1_topics
        part_1_topics = [
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
        
        # Randomly select a topic for Part 1 from the topics list
        selected_topic = random.choice(part_1_topics)
        context.user_data[f'{userID}selected_topic'] = selected_topic

        # Generate initial questions for the selected topic using the Groq API
        initial_questions = await generate_questions(selected_topic)
        if not initial_questions:
            await update.effective_message.reply_text("Failed to generate questions. Please try again.")
            return

        # Store the initial questions in the mock_part1_questions list
        mock_part1_questions.extend(initial_questions)

        # Send a message to the user indicating the start of the mock test
        
        # await update.effective_message.reply_text("The mock test will now begin. (Here are some things you should consider when you answer quetions to help you  get higher scores:) \n\n in Part 1 your answers should be between 10 - 30 seconds \n\n in Part 2 your answers should be between 1 - 2 minutes \n\n in Part 3 your answers should be between 30 seconds - 1 minute \n\n ")
        await update.effective_message.reply_text("The IELTS Speaking mock test will now begin.\n\n"
                "To help you achieve higher scores, please consider the following guidelines for your answers:\n\n"
                "🔹 In Part 1: Your answers should be between 10 - 30 seconds.\n"
                "🔹 In Part 2: Your answers should be between 1 - 2 minutes.\n"
                "🔹 In Part 3: Your answers should be between 30 seconds - 1 minute.\n\n"
                "Let's start!")
        time.sleep(3)
        await update.effective_message.reply_text("Mock Test - Part 1")
        # await update.effective_message.reply_text(f"Topic: {selected_topic}")
        # Call the mock_part1_process function to start Part 1
        time.sleep(3)
        await mock_part1_process(update, context)
    except Exception as e:
        print("start mock test function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def mock_part1_process(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        global list_previous_quetions
        # Get the current question index from the context
        current_question_index = context.user_data.get(f'{userID}mock_part1_current_question_index', 0)

        # Check if there are more questions to ask
        if current_question_index < len(mock_part1_questions):
            previous_question = mock_part1_questions[current_question_index - 1] if current_question_index > 0 else ""
            current_question = mock_part1_questions[current_question_index]
            user_answer = mock_part1_answers[current_question_index - 1] if current_question_index > 0 else ""
            selected_topic = context.user_data[f'{userID}selected_topic']
            list_previous_quetions.append(previous_question) 
            list_previous_answers.append(user_answer)
            # Generate an interactive question based on the previous question, user's answer, and the current question
            interactive_question = await generate_interactive_question(previous_question, user_answer, current_question, selected_topic,list_previous_quetions, list_previous_answers)
            # print("Current question: ", current_question, "\nInteractive question: ", interactive_question)
            if interactive_question:
                # Replace the current question with the generated interactive question
                mock_part1_questions[current_question_index] = interactive_question
                current_question = interactive_question

            # Convert the current question to audio using text-to-speech
            question_audio_path = await convert_text_to_audio(current_question)

            # Send the current question as voice message
            with open(question_audio_path, 'rb') as audio:
                await update.effective_message.reply_voice(voice=audio)

            # Send a message to the user to record their answer
            await update.effective_message.reply_text("Please record your answer.")

            # Set the flag to indicate that the user is answering a question in the mock test
            context.user_data[f'{userID}mock_part1_answering'] = True

            # Store the current question index in the context for the voice answer handler
            context.user_data[f'{userID}mock_part1_current_question_index'] = current_question_index

        else:
            # All questions have been asked, move to the next part
            await update.effective_message.reply_text("Mock Test - Part 1 completed. Moving to Part 2.")
            time.sleep(5)
            await mock_part2_process(update, context)
    except Exception as e:
        print("mock test part 1 process function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def mock_part2_process(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Randomly select a topic from the ielts_questions dictionary
        selected_topic = random.choice(list(ielts_questions.keys()))

        # Extract the Part 2 question and Part 3 questions for the selected topic
        part2_question = ielts_questions[selected_topic]["part_2_question"]
        part3_questions = ielts_questions[selected_topic]["part_3_questions"]

        # Add the Part 2 question to the mock_part2_questions list
        mock_part2_questions.append(part2_question)

        # Clear the mock_part3_questions list before adding new questions
        mock_part3_questions.clear()

        # Add the Part 3 questions to the mock_part3_questions list
        mock_part3_questions.extend(part3_questions)

        # Convert the Part 2 question to audio using text-to-speech
        question_audio_path = await convert_text_to_audio(part2_question)

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
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=countdown_message.message_id)
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=preparation_message.message_id)

        # Set the flag to indicate that the user is answering a question in the mock test Part 2
        context.user_data[f'{userID}mock_part2_answering'] = True

        # Send a message to the user to record their answer
        await update.effective_message.reply_text("Please record your answer.")
    except Exception as e:
        print("mock test part 2 process function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def mock_part3_process(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Get the current question index from the context
        current_question_index = context.user_data.get(f'{userID}mock_part3_current_question_index', 0)

        # Check if there are more questions to ask
        if current_question_index < len(mock_part3_questions):
            current_question = mock_part3_questions[current_question_index]

            # Convert the current question to audio using text-to-speech
            question_audio_path = await convert_text_to_audio(current_question)

            # Send the current question as voice message
            with open(question_audio_path, 'rb') as audio:
                await update.effective_message.reply_voice(voice=audio)

            # Send a message to the user to record their answer
            await update.effective_message.reply_text("Please record your answer.")

            # Set the flag to indicate that the user is answering a question in the mock test Part 3
            context.user_data[f'{userID}mock_part3_answering'] = True

            # Store the current question index in the context for the voice answer handler
            context.user_data[f'{userID}mock_part3_current_question_index'] = current_question_index

        else:
            await mock_test_completed(update,context)
            # All questions have been asked, show the inline keyboard markup for next steps
            keyboard = [
                [InlineKeyboardButton("Show Results", callback_data='mock_test_show_results')],
                [InlineKeyboardButton("Retake Mock Test", callback_data='mock_test_retake')],
                [InlineKeyboardButton("End Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await update.effective_message.reply_text("Mock Test - Part 3 completed. What would you like to do next?", reply_markup=reply_markup)
    except Exception as e:
        print("mock test part 3 process function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)



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
    #     await error_handling(update, context)

async def mock_test_retake(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        await query.answer()

        # Clear the mock test lists
        mock_part1_questions.clear()
        mock_part1_answers.clear()
        mock_part1_voice_urls.clear()
        
        mock_part2_questions.clear()
        mock_part2_answers.clear()
        mock_part2_voice_urls.clear()
        
        mock_part3_questions.clear()
        mock_part3_answers.clear()
        mock_part3_voice_urls.clear()

        # Reset the current question indices
        context.user_data[f'{userID}mock_part1_current_question_index'] = 0
        context.user_data[f'{userID}mock_part3_current_question_index'] = 0

        # Randomly select a new topic for Part 1 from the topics list
        selected_topic = random.choice(part_1_topics)
        context.user_data[f'{userID}selected_topic'] = selected_topic

        # Generate new initial questions for the selected topic using the Groq API
        initial_questions = await generate_questions(selected_topic)
        if not initial_questions:
            await query.edit_message_text("Failed to generate questions. Please try again.")
            return

        # Store the new initial questions in the mock_part1_questions list
        mock_part1_questions.extend(initial_questions)

        # Edit the message to indicate the start of the new mock test
        await query.edit_message_text("Retaking the Mock Test - Part 1")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Topic: {selected_topic}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="The mock test will now begin again. Please answer the following questions.")

        # Call the mock_part1_process function to start Part 1
        await mock_part1_process(update, context)
    except Exception as e:
        print("mock test retake function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def send_user_answers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Send Part 1 answers
        part1_answers_text = "Mock Test - Part 1 Answers:\n\n"
        for i in range(len(mock_part1_questions)):
            question = mock_part1_questions[i]
            answer = mock_part1_answers[i]
            part1_answers_text += f"Question {i+1}: {question}\n\nAnswer: {answer}\n\n"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=part1_answers_text)

        # Send Part 2 answer
        part2_answer_text = "Mock Test - Part 2 Answer:\n\n"
        question = mock_part2_questions[0]
        answer = mock_part2_answers[0]
        part2_answer_text += f"Question: {question}\n\nAnswer: {answer}\n\n"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=part2_answer_text)

        # Send Part 3 answers
        part3_answers_text = "Mock Test - Part 3 Answers:\n\n"
        for i in range(len(mock_part3_questions)):
            question = mock_part3_questions[i]
            answer = mock_part3_answers[i]
            part3_answers_text += f"Question {i+1}: {question}\n\nAnswer: {answer}\n\n"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=part3_answers_text)
    except Exception as e:
        print("send user answers function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def mock_test_completed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Send a waiting message to the user
        waiting_message = await update.effective_message.reply_text("Please wait while your answers and conversation are being prepared...")
        animated_emoji = "⏳"  # Hourglass emoji
        waiting_message2 = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")

        # Prepare the user's answers in text
        await send_user_answers(update, context)
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message2.message_id)
        # # Prepare the audio recording of the entire conversation
        # animated_emoji = "⏳"  # Hourglass emoji
        # waiting_message3 = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        # await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until result is ready.")
        # try:
        #   await generate_conversation_audio(update, context)
        # except Exception as e:
        #     print("Error in generate_conversation_audio:", e)
        # Delete the waiting message
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
        # await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message3.message_id)            
        # Send a message to the user indicating that the answers and audio are ready
        # await update.effective_message.reply_text("Your answers and conversation are ready!")
    except Exception as e:
        print("mock test completed function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

import asyncio
import time
from telegram import InlineKeyboardButton, InlineKeyboardMarkup

async def show_mock_test_results(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        print("-----------------------FEEDBACK MOCK TEST------------------------")
        global mock_part1_questions, mock_part1_answers, mock_part1_voice_urls
        global mock_part2_questions, mock_part2_answers, mock_part2_voice_urls
        global mock_part3_questions, mock_part3_answers, mock_part3_voice_urls

        # print(len(mock_part1_questions), len(mock_part1_answers), len(mock_part1_voice_urls))
        # print(len(mock_part2_questions), len(mock_part2_answers), len(mock_part2_voice_urls))
        # print(len(mock_part3_questions), len(mock_part3_answers), len(mock_part3_voice_urls))
        print("user_targetes_score: ", targeted_score)
        animated_emoji = "⏳"  # Hourglass emoji
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        # Send the initial waiting message with 0% progress and an empty progress bar
        progress_message = await context.bot.send_message(chat_id=update.effective_chat.id, text="Wait a few minutes until results are ready...\n\n[                             ] 0%")

        # Send a message asking the user to share the bot
        share_message = (
            f"Discover this IELTS Speaking  Bot! It simulates the IELTS speaking test and provides detailed feedback about your speaking skills and estimated IELTS band score. to help you improve. Try it for free now: https://t.me/ielts_speakingAI_bot"
        )
        keyboard = [
            [InlineKeyboardButton("Share the Bot", switch_inline_query=share_message)]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        share_message = await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="while waiting the results. Would you like to share this bot?",
            reply_markup=reply_markup,
            parse_mode='HTML'
        )

        # Define the steps of the processing
        steps = [
            "Transcribing answers...",
            "Analyzing responses...",
            "Generating feedback...",
            "Compiling results..."
        ]

        total_steps = len(steps) + len(mock_part1_questions) + len(mock_part3_questions) + 1  # +1 for Part 2 assessment
        current_step = 0

        def update_progress():
            nonlocal current_step
            current_step += 1
            progress = int((current_step / total_steps) * 100)
            progress_bar = "[" + "█" * (progress // 5) + " " * (20 - (progress // 5)) + "]"
            return progress, progress_bar

        # Assess Part 1
        part1_scores_list = []
        part1_assessment_tasks = []
        for i in range(len(mock_part1_questions)):
            audio_url = mock_part1_voice_urls[i]
            question_prompt = mock_part1_questions[i]
            task_type = "ielts_part1"
            
            assessment_task = asyncio.create_task(assess_part1_mock_async(audio_url, question_prompt, task_type))
            part1_assessment_tasks.append(assessment_task)
            # print("part1 assessment", assessment_task)
            
            # Update progress
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Wait a few minutes until results are ready...\n{progress_bar} {progress}%"
            )

        # Wait for all Part 1 assessment tasks to complete
        part1_scores_results = await asyncio.gather(*part1_assessment_tasks)
        
        for i, scores in enumerate(part1_scores_results):
            if scores:
                part1_scores_list.append(scores)
                print(f"Assessment successful for Part 1 question {i+1}: ")
            else:
                print(f"Assessment failed for Part 1 question {i+1}")

        # Assess Part 2
        part2_scores = await assess_part2_mock_async(mock_part2_voice_urls[0], mock_part2_questions[0], "ielts_part2")
        
        # Update progress
        progress, progress_bar = update_progress()
        await context.bot.edit_message_text(
            chat_id=update.effective_chat.id,
            message_id=progress_message.message_id,
            text=f"Processing your mock test results...\n{progress_bar} {progress}%"
        )

        # Assess Part 3
        part3_scores_list = []
        part3_assessment_tasks = []
        for i in range(len(mock_part3_questions)):
            audio_url = mock_part3_voice_urls[i]
            question_prompt = mock_part3_questions[i]
            task_type = "ielts_part3"
            
            assessment_task = asyncio.create_task(assess_part3_mock_async(audio_url, question_prompt, task_type))
            part3_assessment_tasks.append(assessment_task)
            # print("part3 assessment", assessment_task)
            
            # Update progress
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                text=f"Processing your mock test results...\n{progress_bar} {progress}%"
            )

        # Wait for all Part 3 assessment tasks to complete
        part3_scores_results = await asyncio.gather(*part3_assessment_tasks)
        
        for i, scores in enumerate(part3_scores_results):
            if scores:
                part3_scores_list.append(scores)
                print(f"Assessment successful for Part 3 question {i+1}")
            else:
                print(f"Assessment failed for Part 3 question {i+1}")

        if part1_scores_list and part2_scores and part3_scores_list:
            # Calculate average scores for each part
            part1_avg_scores = calculate_average_scores(part1_scores_list)
            part2_avg_scores = part2_scores  # No need to average for Part 2
            part3_avg_scores = calculate_average_scores(part3_scores_list)

            # Calculate overall average scores
            overall_avg_scores = calculate_overall_average_scores(part1_avg_scores, part2_avg_scores, part3_avg_scores)

            # Round the scores to the nearest 0.5
            overall_avg_scores['overall'] = round_to_ielts_score(overall_avg_scores['overall'])
            overall_avg_scores['pronunciation'] = round_to_ielts_score(overall_avg_scores['pronunciation'])
            overall_avg_scores['fluency'] = round_to_ielts_score(overall_avg_scores['fluency'])
            overall_avg_scores['grammar'] = round_to_ielts_score(overall_avg_scores['grammar'])
            overall_avg_scores['vocabulary'] = round_to_ielts_score(overall_avg_scores['vocabulary'])
            mock_score = overall_avg_scores['overall']

            # Generate overall feedback using LLM
            overall_mock_feedback = generate_overall_feedback(part1_scores_list, part2_scores, part3_scores_list,
                                                            mock_part1_questions, mock_part1_answers,
                                                            mock_part2_questions[0], mock_part2_answers[0],
                                                            mock_part3_questions, mock_part3_answers, mock_score)
            
            context.user_data[f'{userID}overall_mock_test_feedback'] = overall_mock_feedback
            # Update progress
            progress, progress_bar = update_progress()
            await context.bot.edit_message_text(
                chat_id=update.effective_chat.id,
                message_id=progress_message.message_id,
                # text=f"Processing your mock test results...\n{progress_bar} {progress}%"
                text=f"Wait a few minutes until results are ready...\n{progress_bar} 100%"
            )
            # Delete the waiting message and the share message
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=progress_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=share_message.message_id)
            await context.bot.delete_message(chat_id=update.effective_chat.id,message_id=waiting_message.message_id)
            # Send overall feedback to the user
            await send_long_message(update, context, overall_mock_feedback)

            # Display scores using the display_feedback function
            await display_feedback(update, context, overall_avg_scores['overall'], overall_avg_scores['pronunciation'],
                                overall_avg_scores['fluency'], overall_avg_scores['grammar'], overall_avg_scores['vocabulary'])

            # Send the band score as text
            band_score = f"Your estimated IELTS band score for the mock test is: {overall_avg_scores['overall']:.1f}"
            await context.bot.send_message(chat_id=update.effective_chat.id, text=band_score)

            # Get the CEFR level based on the IELTS score
            cefr_level = get_cefr_level(overall_avg_scores['overall'])
            await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Your language level is: {cefr_level}")
            await append_speaking_score(update,context,"mock_test", mock_score)
            await increment_practice_count(update, context)
            # Provide user options
            keyboard = [
                [InlineKeyboardButton("Show Detailed Results", callback_data='mock_test_detailed_results')],
                [InlineKeyboardButton("Translate", callback_data='mock_test_translate_feedback')],
                [InlineKeyboardButton("Retake Mock Test", callback_data='mock_test_retake')],
                [InlineKeyboardButton("End Test", callback_data='end_test')]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Assessment failed for one or more parts. Please try again.")
    except Exception as e:
        print("show mock test result function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def assess_part1_mock_async(audio_urls, question_prompts, task_type):
      
    try:
        # Download the voice file from the URL
        response = requests.get(audio_urls)
        
        # Generate a unique filename for the downloaded file
        filename = f"voice_{int(time.time())}.oga"
        
        # Save the downloaded file locally
        with open(filename, "wb") as file:
            file.write(response.content)
        scores,analysis_data = assess_speech(filename, question_prompts, task_type)
        response_json = scores  # Assuming assess_speech returns the JSON response directly
        mock_part1_analysis_list.append(analysis_data)
        # print(analysis_data)
        # if 'result' in response_json:
        #     result = response_json['result']
        #     scores = {
        #         "overall": result.get("overall", "N/A"),
        #         "pronunciation": result.get("pronunciation", "N/A"),
        #         "fluency": result.get("fluency_coherence", "N/A"),
        #         "grammar": result.get("grammar", "N/A"),
        #         "vocabulary": result.get("lexical_resource", "N/A"),
        #         "relevance": result.get("relevance", "N/A"),
        #         "transcription": result.get("transcription", "N/A"),
        #         "pause_filler": result.get("pause_filler", {}),
        #         "sentences": [
        #             {
        #                 "sentence": sentence.get("sentence", ""),
        #                 "pronunciation": sentence.get("pronunciation", "N/A"),
        #                 "grammar": sentence.get("grammar", {}).get("corrected", "")
        #             }
        #             for sentence in result.get("sentences", [])
        #         ]
        #     }
        #     # analysis_list.append(scores)
        
        # # end_time = time.time()
        # # execution_time = end_time - start_time
        # # print(f"Execution time: {execution_time} seconds")
        # os.remove(filename)
        # # print(scores)
        # return scores
        processed_scores = {
            "overall": scores['ielts_score']['overall'],
            "pronunciation": scores['ielts_score']['pronunciation'],
            "fluency": scores['ielts_score']['fluency'],
            "grammar": scores['ielts_score']['grammar'],
            "vocabulary": scores['ielts_score']['vocabulary'],
            "relevance": scores['relevance']['class'],
            "transcription": scores['transcription'],
            # "sentences": []  # SpeechAce doesn't provide sentence-level data in the same format
        }

        # Add word-level details to the processed_scores
        # processed_scores["word_details"] = analysis_data["word_pronunciation_details"]

        # Clean up the temporary file
        os.remove(filename)
        # print(processed_scores)
        return processed_scores
    except Exception as e:
        print(f"Error assessing speech: {str(e)}")
        return None


async def assess_part2_mock_async(audio_url, question_prompt, task_type):
   
    try:
        # Download the voice file from the URL
        response = requests.get(audio_url)
        
        # Generate a unique filename for the downloaded file
        filename = f"voice_{int(time.time())}.oga"
        
        # Save the downloaded file locally
        with open(filename, "wb") as file:
            file.write(response.content)
        scores,analysis_data = assess_speech(filename, question_prompt, task_type)
        response_json = scores  # Assuming assess_speech returns the JSON response directly
        mock_part2_analysis_list.append(analysis_data)
        # print(analysis_data)
        # if 'result' in response_json:
        #     result = response_json['result']
        #     scores = {
        #         "overall": result.get("overall", "N/A"),
        #         "pronunciation": result.get("pronunciation", "N/A"),
        #         "fluency": result.get("fluency_coherence", "N/A"),
        #         "grammar": result.get("grammar", "N/A"),
        #         "vocabulary": result.get("lexical_resource", "N/A"),
        #         "relevance": result.get("relevance", "N/A"),
        #         "transcription": result.get("transcription", "N/A"),
        #         "pause_filler": result.get("pause_filler", {}),
        #         "sentences": [
        #             {
        #                 "sentence": sentence.get("sentence", ""),
        #                 "pronunciation": sentence.get("pronunciation", "N/A"),
        #                 "grammar": sentence.get("grammar", {}).get("corrected", "")
        #             }
        #             for sentence in result.get("sentences", [])
        #         ]
        #     }
        #     # analysis_list.append(scores)
        
        # # end_time = time.time()
        # # execution_time = end_time - start_time
        # # print(f"Execution time: {execution_time} seconds")
        # os.remove(filename)
        # # print(scores)
        # return scores
        processed_scores = {
            "overall": scores['ielts_score']['overall'],
            "pronunciation": scores['ielts_score']['pronunciation'],
            "fluency": scores['ielts_score']['fluency'],
            "grammar": scores['ielts_score']['grammar'],
            "vocabulary": scores['ielts_score']['vocabulary'],
            "relevance": scores['relevance']['class'],
            "transcription": scores['transcription'],
            # "sentences": []  # SpeechAce doesn't provide sentence-level data in the same format
        }

        # Add word-level details to the processed_scores
        # processed_scores["word_details"] = analysis_data["word_pronunciation_details"]

        # Clean up the temporary file
        os.remove(filename)
        # print(processed_scores)
        return processed_scores
    except Exception as e:
        print(f"Error assessing speech: {str(e)}")
        return None


async def assess_part3_mock_async(audio_urls, question_prompts, task_type):
     
    try:
        # Download the voice file from the URL
        response = requests.get(audio_urls)
        
        # Generate a unique filename for the downloaded file
        filename = f"voice_{int(time.time())}.oga"
        
        # Save the downloaded file locally
        with open(filename, "wb") as file:
            file.write(response.content)
        scores,analysis_data = assess_speech(filename, question_prompts, task_type)
        response_json = scores  # Assuming assess_speech returns the JSON response directly
        mock_part3_analysis_list.append(analysis_data)
        # print(analysis_data)
        # if 'result' in response_json:
        #     result = response_json['result']
        #     scores = {
        #         "overall": result.get("overall", "N/A"),
        #         "pronunciation": result.get("pronunciation", "N/A"),
        #         "fluency": result.get("fluency_coherence", "N/A"),
        #         "grammar": result.get("grammar", "N/A"),
        #         "vocabulary": result.get("lexical_resource", "N/A"),
        #         "relevance": result.get("relevance", "N/A"),
        #         "transcription": result.get("transcription", "N/A"),
        #         "pause_filler": result.get("pause_filler", {}),
        #         "sentences": [
        #             {
        #                 "sentence": sentence.get("sentence", ""),
        #                 "pronunciation": sentence.get("pronunciation", "N/A"),
        #                 "grammar": sentence.get("grammar", {}).get("corrected", "")
        #             }
        #             for sentence in result.get("sentences", [])
        #         ]
        #     }
        #     # analysis_list.append(scores)
        
        # # end_time = time.time()
        # # execution_time = end_time - start_time
        # # print(f"Execution time: {execution_time} seconds")
        # os.remove(filename)
        # # print(scores)
        # return scores
        processed_scores = {
            "overall": scores['ielts_score']['overall'],
            "pronunciation": scores['ielts_score']['pronunciation'],
            "fluency": scores['ielts_score']['fluency'],
            "grammar": scores['ielts_score']['grammar'],
            "vocabulary": scores['ielts_score']['vocabulary'],
            "relevance": scores['relevance']['class'],
            "transcription": scores['transcription'],
            # "sentences": []  # SpeechAce doesn't provide sentence-level data in the same format
        }

        # Add word-level details to the processed_scores
        # processed_scores["word_details"] = analysis_data["word_pronunciation_details"]

        # Clean up the temporary file
        os.remove(filename)
        # print(processed_scores)
        return processed_scores
    except Exception as e:
        print(f"Error assessing speech: {str(e)}")
        return None


async def generate_overall_feedback(part1_scores, part2_scores, part3_scores, part1_questions, part1_answers, part2_question, part2_answer, part3_questions, part3_answers, mock_score):
    try:
        # Prepare the prompt for generating overall feedback
        global targeted_score
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
            model="llama3-70b-8192",
        )
        
        # Extract the generated feedback from the response
        feedback = chat_completion.choices[0].message.content
        
        return feedback
    except Exception as e:
        print("Groq error switching to perplexity",e)
        
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
            model="llama-3-70b-instruct",
            messages=messages
        )
        
        # print(response.choices[0].message.content)
        feedback = (response.choices[0].message.content)
        return feedback

def calculate_average_scores(scores):
    # try:
        avg_scores = {
            'overall': sum(score['overall'] for score in scores) / len(scores),
            'pronunciation': sum(score['pronunciation'] for score in scores) / len(scores),
            'fluency': sum(score['fluency'] for score in scores) / len(scores),
            'grammar': sum(score['grammar'] for score in scores) / len(scores),
            'vocabulary': sum(score['vocabulary'] for score in scores) / len(scores)
        }
        return avg_scores
    # except Exception as e:
    #     print("caculate average scores function ",e)
    #     # await update.message.reply_text(issue_message)
    #     await error_handling(update, context)

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
    #     await error_handling(update, context)
        # await update.message.reply_text(issue_message)
async def generate_detailed_feedback_part1(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        detailed_feedback = []
        global mock_part1_questions, mock_part1_answers, mock_part1_analysis_list, mock_part1_voice_urls
        
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
            
            feedback = await generate_feedback_with_llm(prompt)
            detailed_feedback.append(feedback)
            mock_part1_detailed_feedback_list.append(feedback)
            # await context.bot.send_message(chat_id=update.effective_chat.id, text=feedback)
            await send_long_message(update, context, feedback)
            # Generate and send the pronunciation visualization image
            generate_pronunciation_visualization(analysis_data)
            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)
            
            # Send a message to compare pronunciation
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Now, let's compare your pronunciation with a native speaker's pronunciation.")
            
            # Download the voice file from the URL
            response = requests.get(user_voice_url)
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
            slow_audio_path = await convert_answer_to_audio(user_answer, speed='-0.56')
            
            # Send native speaker's audio with slow speed
            with open(slow_audio_path, 'rb') as slow_audio_file:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Slow Speed):")
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
            
            # Generate native speaker's audio with normal speed
            normal_audio_path = await convert_answer_to_audio(user_answer, speed='0')
            
            # Send native speaker's audio with normal speed
            with open(normal_audio_path, 'rb') as normal_audio_file:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Normal Speed):")
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
            
            # Send a message to encourage comparison
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Listen to your voice and compare it with the native speaker's pronunciation at both speeds.")
            
            # Add a delay to avoid hitting the rate limit
            await asyncio.sleep(3)
        
        return detailed_feedback
    except Exception as e:
        print("generate detailed feedback part 1 function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def generate_detailed_feedback_part2(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        detailed_feedback = []
        global mock_part2_questions, mock_part2_answers, mock_part2_analysis_list, mock_part2_voice_urls
        
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Detailed Feedback for Part 2:")
        
        question = mock_part2_questions[0]
        user_answer = mock_part2_answers[0]
        analysis_data = mock_part2_analysis_list[0]
        user_voice_url = mock_part2_voice_urls[0]
        
        prompt = f"Question: {question}\nUser Answer: {user_answer}\nAnalysis Data: {analysis_data}\n\n"
        prompt += """Please provide a detailed analysis of the user's answer, considering the pronunciation, fluency, grammar, vocabulary, and relevance. Generate useful feedback that helps the user understand their performance in depth.
        You should organize it in a clear and good feedback to the IELTS Candidate. They expect to know their mistakes and improve next time. Also, it is a good idea to write the question in the feedback and the answer and then start your analysis.
        """
        
        feedback = await generate_feedback_with_llm(prompt)
        detailed_feedback.append(feedback)
        mock_part2_detailed_feedback_list.append(feedback)
        # await context.bot.send_message(chat_id=update.effective_chat.id, text=feedback)
        await send_long_message(update, context, feedback)
        # Generate and send the pronunciation visualization image
        generate_pronunciation_visualization(analysis_data)
        with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
            await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)
        
        # Send a message to compare pronunciation
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Now, let's compare your pronunciation with a native speaker's pronunciation.")
        
        # Download the voice file from the URL
        response = requests.get(user_voice_url)
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
        slow_audio_path = await convert_answer_to_audio(user_answer, speed='-0.56')
        
        # Send native speaker's audio with slow speed
        with open(slow_audio_path, 'rb') as slow_audio_file:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Slow Speed):")
            await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
        
        # Generate native speaker's audio with normal speed
        normal_audio_path = await convert_answer_to_audio(user_answer, speed='0')
        
        # Send native speaker's audio with normal speed
        with open(normal_audio_path, 'rb') as normal_audio_file:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Normal Speed):")
            await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
        
        # Send a message to encourage comparison
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Listen to your voice and compare it with the native speaker's pronunciation at both speeds.")
        
        # Add a delay to avoid hitting the rate limit
        await asyncio.sleep(3)
        
        return detailed_feedback
    except Exception as e:
        print("generate detiailed feedback part 2 function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def generate_detailed_feedback_part3(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        detailed_feedback = []
        global mock_part3_questions, mock_part3_answers, mock_part3_analysis_list, mock_part3_voice_urls
        
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
            
            feedback = await generate_feedback_with_llm(prompt)
            detailed_feedback.append(feedback)
            mock_part3_detailed_feedback_list.append(feedback)
            # await context.bot.send_message(chat_id=update.effective_chat.id, text=feedback)
            await send_long_message(update, context, feedback)
            # Generate and send the pronunciation visualization image
            generate_pronunciation_visualization(analysis_data)
            with open('pronunciation_visualization_with_padding.png', 'rb') as image_file:
                await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_file)
            
            # Send a message to compare pronunciation
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Now, let's compare your pronunciation with a native speaker's pronunciation.")
            
            # Download the voice file from the URL
            response = requests.get(user_voice_url)
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
            slow_audio_path = await convert_answer_to_audio(user_answer, speed='-0.56')
            
            # Send native speaker's audio with slow speed
            with open(slow_audio_path, 'rb') as slow_audio_file:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Slow Speed):")
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=slow_audio_file)
            
            # Generate native speaker's audio with normal speed
            normal_audio_path = await convert_answer_to_audio(user_answer, speed='0')
            
            # Send native speaker's audio with normal speed
            with open(normal_audio_path, 'rb') as normal_audio_file:
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Native Speaker's Voice (Normal Speed):")
                await context.bot.send_voice(chat_id=update.effective_chat.id, voice=normal_audio_file)
            
            # Send a message to encourage comparison
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Listen to your voice and compare it with the native speaker's pronunciation at both speeds.")
            
            # Add a delay to avoid hitting the rate limit
            await asyncio.sleep(3)
        
        return detailed_feedback
    except Exception as e:
        print("generate detailed feedback part 3 function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def generate_mock_test_detailed_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        # Notify the user that detailed feedback generation is starting
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Starting detailed feedback generation for each part of the mock test.")

        # Generate detailed feedback for Part 1
        # await context.bot.send_message(chat_id=update.effective_chat.id, text="Detailed Feedback for Part 1:")
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
        detailed_feedback = part1_detailed_feedback + part2_detailed_feedback + part3_detailed_feedback
        
        # Notify the user that detailed feedback has been generated for all parts
        # await context.bot.send_message(chat_id=update.effective_chat.id, text="Detailed feedback for all parts has been generated.")

        # Provide user options after detailed results
        keyboard = [
            [InlineKeyboardButton("Translate Detailed Feedback", callback_data='mock_test_translate_detailed_feedback')],
            [InlineKeyboardButton("Retake Mock Test", callback_data='mock_test_retake')],
            [InlineKeyboardButton("End Test", callback_data='end_test')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)

        return detailed_feedback
    except Exception as e:
        print("generate mock test detailed feedback function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)

async def translate_mock_test_overall_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        user_id = query.from_user.id
        overall_feedback = context.user_data.get(f'{userID}overall_mock_test_feedback') # Make sure this key is correct
        # Provide the user with options
        keyboard = [
            [InlineKeyboardButton("See Detailed Results", callback_data='mock_test_detailed_results')],
            [InlineKeyboardButton("Retake Mock Test", callback_data='mock_test_retake')],
            [InlineKeyboardButton("End Test", callback_data='end_test')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)   
        if overall_feedback:
            # Send a waiting message
            animated_emoji = "⏳"  # Hourglass emoji
            waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes until the translation is ready.")

            # Translate the overall feedback
            translated_feedback = await translate_feedback(user_id, overall_feedback, update, context)
            
            if translated_feedback:
                # Send the translated feedback to the user
                await context.bot.send_message(chat_id=update.effective_chat.id, text=translated_feedback)
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
            else:
                # Send a message indicating that translation is not available
                await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, translation is not available for your language at the moment.")
                await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
                await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
        else:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, there was an issue retrieving the feedback.")
            await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)
            await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
        
        
        
    except Exception as e:
        print("translate mock test overall feedback function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
async def translate_mock_test_detailed_feedback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        query = update.callback_query
        user_id = query.from_user.id

        # Send a waiting message
        animated_emoji = "⏳"  # Hourglass emoji
        waiting_message = await context.bot.send_message(chat_id=update.effective_chat.id, text=animated_emoji)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Please wait a few minutes while the detailed feedback is being translated.")

        # Translate detailed feedback for Part 1
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Translated Detailed Feedback for Part 1:")
        for feedback in mock_part1_detailed_feedback_list:
            translated_msg = await translate_feedback(user_id, feedback, update, context)
            if translated_msg:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=translated_msg)
                await asyncio.sleep(1)  # Add a delay to avoid hitting the rate limit
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f"sorry, Translation failed")

        # Translate detailed feedback for Part 2
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Translated Detailed Feedback for Part 2:")
        for feedback in mock_part2_detailed_feedback_list:
            translated_msg = await translate_feedback(user_id, feedback, update, context)
            if translated_msg:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=translated_msg)
                await asyncio.sleep(1)  # Add a delay to avoid hitting the rate limit
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f"sorry, Translation failed")

        # Translate detailed feedback for Part 3
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Translated Detailed Feedback for Part 3:")
        for feedback in mock_part3_detailed_feedback_list:
            translated_msg = await translate_feedback(user_id, feedback, update, context)
            if translated_msg:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=translated_msg)
                await asyncio.sleep(1)  # Add a delay to avoid hitting the rate limit
            else:
                await context.bot.send_message(chat_id=update.effective_chat.id, text=f"sorry, Translation failed")

        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=waiting_message.message_id)

        # Provide user options after translation
        keyboard = [
            [InlineKeyboardButton("Retake Mock Test", callback_data='mock_test_retake')],
            [InlineKeyboardButton("End Test", callback_data='end_test')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_message(chat_id=update.effective_chat.id, text="What would you like to do next?", reply_markup=reply_markup)
    except Exception as e:
        print("translate mock test detailed feedback function ",e)
        # await update.message.reply_text(issue_message)
        await error_handling(update, context)
def is_valid_gmail(email):
    try:
        print("is valid gmail")
        gmail_regex = re.compile(r'^[a-zA-Z0-9._%+-]+@gmail\.com$')
        return gmail_regex.match(email) is not None
    except Exception as e:
        print("is valid gmail function",e)
        return True
        # await update.message.reply_text(issue_message)

def is_real_gmail(email):
    try:
        print("is real gmail")
        api_keys = [
            "5d64616c34d64b4e98f2647a29648a53",
            "ff1552bbc5dd4dbc87e5c85645db1cb7",
            "e9f36cc57580421184a1bc62fd297fb0",
            "851b30614b984b6588ef91fb6c9b69ab",
           
        ]
        retries = 3
        while retries > 0:
            api_key = random.choice(api_keys)
            try:
                url = "https://emailvalidation.abstractapi.com/v1"
                querystring = {"api_key": api_key, "email": email}
            
                response = requests.get(url, params=querystring)
                data = response.json()
                deliverability = data.get("deliverability")
            
                if deliverability == "DELIVERABLE":
                    return True
                elif deliverability == "UNDELIVERABLE":
                    return False
            except Exception as e:
                print(f"Error verifying email with API key {api_key}: {e}")
                retries -= 1
        
        return True
    except Exception as e:
        print(e)
        return True
        # await update.message.reply_text(issue_message)

def main():
    print("main")
    # application = Application.builder().token(TOKEN).build()
    application = Application.builder().token(BOT_TOKEN).read_timeout(200).build()
    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)
    # application.add_handler(CallbackQueryHandler(topic_selection_handler))
    message_handler_instance = MessageHandler(filters.TEXT & ~filters.COMMAND, message_handler)
    application.add_handler(message_handler_instance)

    voice_handler_instance = MessageHandler(filters.VOICE, voice_handler)
    application.add_handler(voice_handler_instance)
    # # Register a handler to wait for the user's voice answer
    # voice_handler_instance2 = MessageHandler(filters.VOICE & ~filters.COMMAND, handle_voice_answer)
    # application.add_handler(voice_handler_instance2)
    application.add_handler(CommandHandler("language", change_language))
    application.add_handler(CommandHandler("voice", change_voice))
    button_handler_instance = CallbackQueryHandler(button_handler)
    application.add_handler(button_handler_instance)

    application.run_polling()

if __name__ == '__main__':
    main()
