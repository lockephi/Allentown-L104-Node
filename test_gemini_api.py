import os
os.chdir('/workspaces/Allentown-L104-Node')

from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

api_key = os.environ.get('GEMINI_API_KEY')
print(f'API Key loaded: {api_key[:20]}...')

genai.configure(api_key=api_key)

model = genai.GenerativeModel('gemini-2.0-flash')
response = model.generate_content('Say hello and confirm you are Gemini')

print(f'Gemini Response: {response.text}')
