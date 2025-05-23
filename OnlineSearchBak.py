"""
OnlineSearch-
对qwen提问，让其给出评价，然后再提问当地的天气，得到天气信息后，询问其衣着情况
其中查询天气的APPID需要用openweather的注册的id，可以在线申请
Author: wzpym
Date: 2025/5/17
"""

import json
import os
import dashscope
from dashscope.api_entities.dashscope_response  import Role
import requests
from idna import decode

def find_location():
    url = 'https://ipinfo.io/json'
    response = requests.get(url)
    type(response.content)
    data = json.loads(response.content.decode('utf-8'))
    # data = json.dumps(response.content,data)
    city = data['city']
    print(f'The local address is {response}')
    return city
#url ='http://pv.sohu.com/cityjson'

def query_realtime_weather(location = 'Chengdu', unit="fahrenheit"):
    url = f'http://api.openweathermap.org/data/2.5/weather?q={location},cn&APPID=*********&units={unit}'
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200:
        weather = {
            "city": data["name"],
            "temperature": data["main"]["temp"],
            "weather_description": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"]
        }
        return weather
    else:
        return None


#heards ={'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36'}
#response = requests.get( url, heards)

functions = [
{
        "name": "query_realtime_weather",
        "description": "Get the current weather info for a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                    },
                    "description": "The city of the location"
                },
                "unit": {
                    "type": "string",
                    "enum": ["fahrenheit", "celsius"]
                }
            },
            "required": ["location", "unit"]
        }
    },
    {
        "name": "find_location",
        "description": "Get the current location info according the ip information",
        "parameters": {
        }
    }
]

def receive_response(messages,location):
    response = dashscope.Generation.call(
        model='qwen-turbo',
        messages=messages,
        result_format='message',  # 将输出设置为message形式
        functions = functions,
        location=location
    )
    return response

review = '先查询我的地理位置，然后根据我的地理位置查出实时的天气情报，最后给出旅行建议。'
content = '您是天气预报员，请根据我的问题，给出答案，谢谢'
location = 'Chengdu'
messages=[
{"role": "system", "content": content},
{"role": "user", "content": review}
]
response = ''
while True:
    response = receive_response(messages, location)
    print(response)
    message = response.output.choices[0].message
    messages.append(message)
    if response.output.choices[0].finish_reason == 'stop':
        print(response.output.choices[0].message.content)
        break
    if hasattr(message, 'function_call') and message.function_call:
        function_call = message.function_call
        fn_name = function_call['name']
        arguments = json.loads(function_call['arguments'])
        function = locals()[fn_name]
        if not arguments:
           fn_response = function()
           location = fn_response
        else:
            fn_response = function(location,"celsius")
            fn_response = json.dumps(fn_response)
        fn_info = {"role": "function", "name": fn_name, "content": fn_response}
        messages.append(fn_info)
    #response = receive_response(messages,location)
print(response)
