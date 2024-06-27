import time
import hashlib
import requests
import json
import requests
import os
import time
appKey = "1717662733000321"
secretKey = "7e7d1a029ebd4229b4145da5ebba4049"
baseURL = "https://api.speechsuper.com/"

def assess_speech2(audioPath, question_prompt, task_type):
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            timestamp = str(int(time.time()))
            coreType = "speak.eval.pro"
            audioType = "mp3"
            audioSampleRate = 16000
            userId = "guest"

            url = baseURL + coreType
            connectStr = (appKey + timestamp + secretKey).encode("utf-8")
            connectSig = hashlib.sha1(connectStr).hexdigest()
            startStr = (appKey + timestamp + userId + secretKey).encode("utf-8")
            startSig = hashlib.sha1(startStr).hexdigest()

            params = {
                "connect": {
                    "cmd": "connect",
                    "param": {
                        "sdk": {
                            "version": 16777472,
                            "source": 9,
                            "protocol": 2
                        },
                        "app": {
                            "applicationId": appKey,
                            "sig": connectSig,
                            "timestamp": timestamp
                        }
                    }
                },
                "start": {
                    "cmd": "start",
                    "param": {
                        "app": {
                            "userId": userId,
                            "applicationId": appKey,
                            "timestamp": timestamp,
                            "sig": startSig
                        },
                        "audio": {
                            "audioType": audioType,
                            "channel": 1,
                            "sampleBytes": 2,
                            "sampleRate": audioSampleRate
                        },
                        "request": {
                            "coreType": coreType,
                            "test_type": "ielts",
                            "question_prompt": question_prompt,
                            "task_type": task_type,
                            "phoneme_output": 1,
                            "model": "non_native",
                            "penalize_offtopic": 1,
                            "decimal_point": 0,
                            "tokenId": "tokenId"
                        }
                    }
                }
            }

            datas = json.dumps(params)
            data = {'text': datas}
            headers = {"Request-Index": "0"}
            files = {"audio": open(audioPath, 'rb')}
            res = requests.post(url, data=data, headers=headers, files=files)

            response_json = res.json()
            if 'result' in response_json:
                result = response_json['result']
                scores = {
                        "overall": result.get("overall", "N/A"),
                        "pronunciation": result.get("pronunciation", "N/A"),
                        "fluency": result.get("fluency_coherence", "N/A"),
                        "grammar": result.get("grammar", "N/A"),
                        "vocabulary": result.get("lexical_resource", "N/A"),
                        "relevance": result.get("relevance", "N/A"),
                        "transcription": result.get("transcription", "N/A"),
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
            # Create a new variable to store word-level pronunciation details
                word_pronunciation_details = []
                for sentence in result.get("sentences", []):
                    sentence_details = []
                    for word_info in sentence.get("details", []):
                        word = word_info.get("word", "")
                        pronunciation = word_info.get("pronunciation", "N/A")
                        sentence_details.append({"word": word, "pronunciation": pronunciation})
                    word_pronunciation_details.append(sentence_details)
                
                # Combine the scores and word_pronunciation_details into a single dictionary
                analysis_data = {
                    "scores": scores,
                    "word_pronunciation_details": word_pronunciation_details
                }
                
                # print(scores)
                return scores,analysis_data
                # return scores
            else:
                print("Failed to get the assessment result.")
                raise("error while assessing in speech assessment", e)
                return None
        except Exception as e:
            retries += 1
            print("An internal error has occurred: now will use ", e)
            print("Retrying...")
            continue
    else:
        raise("error while assessing in speech assessment", e)

import requests
import json

def assess_speech(audio_path, question_prompt):
    # url = "https://api5.speechace.com/api/scoring/speech/v0.5/json"
    max_retries = 3
    retries = 0
    while retries < max_retries:
        try:
            # Your SpeechAce API key
            api_key = "BJ9VT3IxOwMJJF6iEs5sP%2B880nQZGmGtsK5N2kOXktWbq1n8g3SwsJW1DQO6rkKbqzOV4VK9yvB2TMvFSnBSfO4%2F1NW%2BEHrEegYR%2B8ax%2BGQ6L2cMHYbtxBzcgOV4%2FgEQ"
            url = f"https://api5.speechace.com/api/scoring/speech/v0.7/json?key={api_key}&dialect=en-us&user_id=XYZ-ABC-99001"
            payload = {
                # 'key': api_key,
                # 'dialect': 'en-us',
                # 'user_id': 'XYZ-ABC-99001',
                'include_fluency': '1',
                'include_ielts_subscore': '1',
                'include_unknown_words': '1',
                "pronunciation_score_mode" : "strict",
                'relevance_context': question_prompt
            }
            
            files = [
                ('user_audio_file', ('audio_file', open(audio_path, 'rb'), 'application/octet-stream'))
            ]
            
            headers = {}
            
            try:
                response = requests.post(url, headers=headers, data=payload, files=files)
                response.raise_for_status()
                result = response.json()
                # print(result)
                # Extract the specific information you're interested in
                scores = {
                "ielts_score": {
                    "overall": result.get("speech_score", {}).get("fluency", {}).get("overall_metrics", {}).get("ielts_estimate", "N/A"),
                    "pronunciation": (result.get("speech_score", {}).get("quality_score","N/A") - 15 ) /10,
                    "fluency": (result.get("speech_score", {}).get("fluency", {}).get("overall_metrics", {}).get("fluency_score", "N/A") - 10)/10,
                    "grammar": result.get("speech_score", {}).get("fluency", {}).get("overall_metrics", {}).get("ielts_subscore", {}).get("grammar", "N/A"),
                    "coherence": result.get("speech_score", {}).get("fluency", {}).get("overall_metrics", {}).get("ielts_subscore", {}).get("coherence", "N/A"),
                    "vocabulary": result.get("speech_score", {}).get("fluency", {}).get("overall_metrics", {}).get("ielts_subscore", {}).get("vocab", "N/A"),
                    
                },
                "relevance": {
                    "class": result.get("speech_score", {}).get("relevance", {}).get("class", True)
                },
                "transcription": result.get("speech_score", {}).get("transcript", " "),
                
            
        }
                word_score_list = result.get("speech_score", {}).get("word_score_list", [])

                word_pronunciation_details = [
                    {
                        "word": word_info.get("word", ""),
                        "pronunciation": word_info.get("quality_score", 0),
                        # "phone_score_list": word_info.get("phone_score_list", []),
                        # "syllable_score_list": word_info.get("syllable_score_list", [])
                    }
                    for word_info in word_score_list
                ]


                # Combine all the extracted information into a single dictionary
                analysis_data = {
                    "scores": scores,
                    "word_pronunciation_details": word_pronunciation_details
                }
                # print(analysis_data)
                return scores,analysis_data
            
            except Exception as e:
                print(f"An error occurred: {e}")
                return None
        except Exception as e:
            retries += 1
            print("An internal error has occurred: now will use ", e)       
            print("Retrying...")
            continue
    else:
        raise("error while assessing in speech assessment", e)
# Test the function
# if __name__ == "__main__":
#     audio_path = "file_1056.oga"  # Replace with the actual path to your audio file
#     question_prompt = "Describe the healthy streets program and its impact on the residents of Austin Texas."
    
#     result = assess_speech2(audio_path, question_prompt)
    
#     if result:
#         scores, analysis_data = result
#         print(f"IELTS Pronunciation: {scores['ielts_score']['pronunciation']}")
#         print(f"IELTS Fluency: {scores['ielts_score']['fluency']}")
#         print(f"IELTS Grammar: {scores['ielts_score']['grammar']}")
#         print(f"IELTS Coherence: {scores['ielts_score']['coherence']}")
#         print(f"IELTS Vocabulary: {scores['ielts_score']['vocabulary']}")
#         print(f"IELTS Overall: {scores['ielts_score']['overall']}")
#         print(f"Relevance: {scores['relevance']['class']}")
#         print(f"Transcription: {scores['transcription']}")
        
#         # You can also print word pronunciation details if needed
#         for word_detail in analysis_data["word_pronunciation_details"]:
#             print(f"Word: {word_detail['word']}, pronunciation_score: {word_detail['pronunciation_score']}")
#     else:
#         print("Failed to get the assessment result.")
        
# import requests



# payload={'include_fluency': '1',
# 'include_ielts_subscore': '1',
# 'include_unknown_words': '1',
# 'relevance_context': 'Describe the healthy streets program and its impact on the residents of Austin Texas.'}
# files=[
#   ('user_audio_file',('kevin_j_krizek_tedx_2021_2.m4a',open('/Users/shimiz/speechace_dev/audio/kevin_j_krizek_tedx_2021_2.m4a','rb'),'application/octet-stream'))
# ]
# headers = {}

# response = requests.request("POST", url, headers=headers, data=payload, files=files)

# print(response.text)

# # Test the function
# if __name__ == "__main__":
#     audioPath = "voice_01.oga"  # Replace with the actual path to your audio file
#     question_prompt = "Do you like your voice? Explain why."
#     task_type = "ielts_part2"
#     result = assess_speech(audioPath, question_prompt, task_type)
#     if result:
#         print(f"IELTS Scores:")
#         for key, value in result.items():
#             print(f"{key.capitalize()}: {value}")
# question_prompt = "Describe the healthy streets program and its impact on the residents of Austin Texas."
# assess_speech2("voice_1719213370.oga", question_prompt, "ielts_part2")
