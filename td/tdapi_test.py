import pandas as pd
import requests
from pandas.io.json import json_normalize
from datetime import datetime
import time
from urllib.parse import unquote
from pathlib import Path

class credentials:
    def __init__(self):
        # URL decoded refresh token
        self.code = 'NQgSzR8OA%2Fgex%2B7TtvDEUSJMklmGMJrQ7qzrjwg3pFDHzRf%2ByHIFUSJ4nGHmyc6jvB23Vt8QX6dTx6ZI9u8xUoct5iiaT8oYR1QYoET3P76lmzACvKrY0XmnPvqziI9h8AhPW4hJDJpsuypav374uxNzKPpkP9iy2kTOqaAs1J%2BPNZIz2hXDZd7Dr9DKiFipKHI4NgixeF3G4ii5chI19KKE1mPirww%2BHLmT0NBC%2BG49GtmRq4VCDaHZHdtoNBZGzl%2Bb%2Fq8BZunPIS3sw1p2ubAXLxFE8bZ6Kpn2LY97G%2BbZ6ZWpdAtOJ8ISdOebqTFGKKmzLLSI5JAitFoChYO297EpdgaKTG6UuG3G%2FqBBcU6Z5tQQIVEgX4tVjVkz7UY0LZQsgVJIi360dWezdLjA1bH7FlkeMRKJAQJpdrDY96GHXn2082wWJbEyj%2Fu100MQuG4LYrgoVi%2FJHHvl237se%2F0Bw9OfZkEJwrCYiuyv%2FdXwI%2FvMS8iJN6pa%2BDloLwB1%2BSadqzKONd%2Fybi%2FYs6qVPin3QqcGGRwq01GDXR%2FZAobT0ctoyMoMkLw8Lld3ybFFOPd6FKyUBTpSUNRtfVjJe3aoAUTega%2FegOkTgfsnRiw%2BjIkAJtGOhnRZQ1P4h97JBeJRag%2FfgUHj%2FZEyUVKZreA3RfWjevrkFDUZnuRAuS6sP1r4%2BnQxOvLxYs617cgiQ5jW3Y%2BbZyrFzoOJq2VaNobn3g%2FNylho7aEz5EGRFPPSqOJUjonWn%2BE1MQ%2Fxt4r2ob1ddndm5knWxenvQz%2Bs1C53EXdTJnd1C1nYTlE%2FXDHk20QVd%2BPtVwsIazsFSt1vH8wdGe0HoKI2a23V%2FjXbyiUg0FWR1%2F%2B4vLM4vrwqeNEyF5yzINXGvjGic3tVQErguXq39A5ZoSY%3D212FD3x19z9sWBHDJACbC00B75E'
        self.apikey = 'probability@AMER.OAUTHAP'
        self.client_id = 'probability@AMER.OAUTHAP'


credentials = credentials()
class Td:
    def __init__(self, code=credentials.code, client_id = credentials.client_id, apikey=credentials.apikey):
        # URL decoded refresh token
        self.code = unquote(code)
        self.apikey = apikey
        self.client_id = client_id
        self.token_path ='td/refresh-token.txt'
        self.main()


    # Checks if token file is available
    # token file : refresh-token.txt
    #
    def main(self):
        refresh_token_file = Path(self.token_path)
        if refresh_token_file.is_file():
            self.get_access_token()
        else:
            self.auth_code()

    # Save a refresh token
    # token file : refresh-token.txt
    #
    def auth_code(self):
        headers = {'Content-Type': 'application/x-www-form-urlencoded'}
        data = {
            'grant_type': 'authorization_code',
            'access_type': 'offline',
            'code': self.code,
            'client_id': self.client_id,
            'redirect_uri': 'http://localhost:8080'
        }
        authReply = requests.post(
            'https://api.tdameritrade.com/v1/oauth2/token',
            headers=headers,
            data=data)
        if authReply.status_code == 200:
            refresh_token = authReply.json()
            f = open(self.token_path, "w+")
            f.write(refresh_token['refresh_token'])
            f.close()
        else:
            print('Failed to obtain a refresh token: auth_code(): Status',
                  authReply.status_code)
            print('Obtain a new Code from TDAmeritrade')


    # Gets Access Token
    #
    #


    def get_access_token(self):
        #Post Access Token Request.
        my_file = Path(self.token_path)
        authReply = None
        if my_file.is_file():
            f = open(self.token_path, "r")
            if f.mode == 'r':
                token = f.read()
                f.close()
                # authReply = token
                headers = {'Content-Type': 'application/x-www-form-urlencoded'}
                data = {
                    'grant_type': 'refresh_token',
                    'refresh_token': token,
                    # 'access_type': 'offline',
                    'client_id': self.client_id,
                    'redirect_uri': 'http://localhost:8080'
                }
                authReply = requests.post(
                    'https://api.tdameritrade.com/v1/oauth2/token',
                    headers=headers,
                    data=data)
                # if authReply.status_code == 200:
                #     refresh_token = authReply.json()
                #     f = open(self.token_path, "w+")
                #     f.write(refresh_token['refresh_token'])
                #     f.close()
                # else:
                #     print(authReply.json())
        return authReply


    # Get Qoute
    # Param : Symbol
    #
    def get_quotes(self, symbol):
        access_token = self.get_access_token().json()
        access_token = access_token['access_token']
        # access_token = self.get_access_token()

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Bearer {}'.format(access_token)
        }
        data = {'symbol': symbol, 'apikey': self.apikey}
        authReply = requests.get(
            'https://api.tdameritrade.com/v1/marketdata/quotes',
            headers=headers,
            params=data)
        return (authReply.json())


    # Convert time to Unix Time Stamp.
    # Param : Time.
    #
    def unix_time_millis(self, dt):
        epoch = datetime.utcfromtimestamp(0)
        return int((dt - epoch).total_seconds() * 1000.0)


    # Get price History.
    # Param : Symbol, Start date , End date.
    #
    def get_price_history(self, symbol, startDate=None, endDate=None):
        if self.get_access_token() is None:
            return None

        access_token = self.get_access_token().json()
        access_token = access_token['access_token']
        # access_token = self.get_access_token()

        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Authorization': 'Bearer {}'.format(access_token)
        }
        data = {
            'periodType': 'year',
            'frequencyType': 'daily',
            'startDate': startDate,
            'endDate': endDate
        }
        authReply = requests.get(
            'https://api.tdameritrade.com/v1/marketdata/' + symbol +
            '/pricehistory',
            headers=headers,
            params=data)
        candles = authReply.json()
        df = json_normalize(authReply.json())
        df = pd.DataFrame(candles['candles'])
        return df


# code=''
# apikey ='xxxxxx@AMER.OAUTHAP'
# client_id = 'xxxxxxx@AMER.OAUTHAP'
#
# p = Td(code, client_id, apikey)
#
# start_date = datetime.strptime('04 3 2018  1:33PM', '%m %d %Y %I:%M%p')
# end_date = datetime.strptime('05 3 2018  1:33PM', '%m %d %Y %I:%M%p')
# print(p.get_price_history('SNAP', p.unix_time_millis(start_date),
#                           p.unix_time_millis(end_date)))
