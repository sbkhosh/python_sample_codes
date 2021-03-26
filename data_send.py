#!/usr/bin/python3

import cookie
import pandas as pd
import pdb
import sys
import time
import telepot
from telepot.loop import MessageLoop

pd.options.mode.chained_assignment = None 
pd.set_option('display.max_colwidth', 150)
pd.set_option('display.max_rows', 50)

class DataSend():
    def __init__(self, input_directory, output_directory, input_prm_file):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.input_prm_file = input_prm_file

    def __repr__(self):
        return(f'{self.__class__.__name__}({self.input_directory!r}, {self.output_directory!r}, {self.input_prm_file!r})')

    def __str__(self):
        return('input directory = {}, output directory = {},input parameter file  = {}'.\
               format(self.input_directory, self.output_directory, self.input_prm_file))

    @Helper.timing
    def read_prm(self):
        filename = os.path.join(self.input_directory,self.output_directory,self.input_prm_file)
        with open(filename) as fnm:
            self.conf = yaml.load(fnm, Loader=yaml.FullLoader)
        self.teltoken = self.conf.get('teltoken')
                       
    def handle(self,msg):
        content_type, chat_type, chat_id = telepot.glance(msg)
        print(content_type, chat_type, chat_id)
        if content_type == 'text' and msg["text"].lower() == "news":
            # let the human know that the pdf is on its way        
            bot.sendMessage(chat_id, "preparing pdf of fresh news, pls wait..")
            file2send = '/'.join(self.output_directory,'news.pdf')

            # send the pdf doc
            bot.sendDocument(chat_id=chat_id, document=open(file2send, 'rb'))
        elif content_type == 'text':
            bot.sendMessage(chat_id, "sorry, I can only deliver news")

    def send_file(self):
        TOKEN = self.ttoken
        bot = telepot.Bot(TOKEN)
        bot.sendDocument(@sbkhosh, file)
        MessageLoop(bot, DataSend.handle).run_as_thread()
        print ('Listening ...')
        # Keep the program running.
        while 1:
            time.sleep(10)

        
            
