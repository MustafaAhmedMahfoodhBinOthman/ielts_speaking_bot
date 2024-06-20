import time
import hashlib
import requests
import json
import requests
import os
import time
from dotenv import load_dotenv
load_dotenv()
# appKey = os.getenv("appkey")
# secretKey = os.getenv("secretKey")
appKey = "1717662733000321"
secretKey = "7e7d1a029ebd4229b4145da5ebba4049"
baseURL = "https://api.speechsuper.com/"

def assess_speech(audioPath, question_prompt, task_type):
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
                
                return scores,analysis_data
                # return scores
            else:
                print("Failed to get the assessment result.")
                # raise("error while assessing in speech assessment", e)
                return None
        except Exception as e:
            retries += 1
            print("An internal error has occurred: now will use ", e)
            print("Retrying...")
            continue
    else:
        raise("error while assessing in speech assessment", e)

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
