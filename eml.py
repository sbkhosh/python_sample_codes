#!/usr/bin/python3

import email, smtplib, ssl
import pandas as pd

from io import BytesIO
from reportlab.pdfgen import canvas
from django.http import HttpResponse
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

lst = [ 'chocolate', 'cottage cheese', 'olive oil'
        'lemon juice', 'ceasar sauce', 'panadol', 'alcohol pad']
qty = ['1 kg','2 pcs','3 units','2 units', '1 unit', '1 unit', '4 units', '2 units', '1 unit', '2 units', '2 units', '3 units', '2 units', '2 units', '2 units', '2 units', '1 unit']

df = pd.DataFrame(list(zip(lst,qty)),columns=['product','quantity'])


subject = "list"
body = """\
    <html>
      <head></head>
      <body>
        <p>Hi COKPOBNWE!<br>
           Here is the list:<br>
           {0}

        </p>
      </body>
    </html>

    """.format(df.to_html())

sender_email = "sbkhosh@gmail.com"
receiver_email = "trushkovaantonina@gmail.com"
password = "ewskaukcgicoytij"
email = MIMEMultipart()
email["From"] = sender_email
email["To"] = receiver_email 
email["Subject"] = subject

email.attach(MIMEText(body, "Arial"))
session = smtplib.SMTP('smtp.gmail.com:587')
session.ehlo()
session.starttls()
session.login(sender_email, password)
text = email.as_string()
session.sendmail(sender_email, receiver_email, text)
session.quit()
print('Mail Sent')

# ######################################################################################################
# import email, smtplib, ssl

# from io import BytesIO
# from reportlab.pdfgen import canvas
# from django.http import HttpResponse
# from email import encoders
# from email.mime.base import MIMEBase
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText

# subject = "cv quant position"
# body = "Hello,\n\nCould you please consider my profile for a quant focused position.\nI have attached my resume for reference.\n\nBest regards,\n\nSohrab KHOSH AGHDAM"
# sender_email = "sbkhosh@gmail.com"
# receiver_email = "sbkhosh@gmail.com"
# filenm = "docs/cv_sk.pdf"
# password = "ewskaukcgicoytij"
# email = MIMEMultipart()
# email["From"] = sender_email
# email["To"] = receiver_email 
# email["Subject"] = subject

# email.attach(MIMEText(body, "Arial"))
# attach_file = open(filenm, "rb")
# report = MIMEBase("application", "octate-stream")
# report.set_payload((attach_file).read())
# encoders.encode_base64(report)
# email.attach(report)
# session = smtplib.SMTP('smtp.gmail.com:587')
# session.ehlo()
# session.starttls()
# session.login(sender_email, password)
# text = email.as_string()
# session.sendmail(sender_email, receiver_email, text)
# session.quit()
# print('Mail Sent')

# ######################################################################################################
# import smtplib
# from email.mime.multipart import MIMEMultipart
# from email.mime.text import MIMEText
# from email.mime.image import MIMEImage
# import pandas as pd
# import matplotlib

# def send_mail(cdf):
#     df = cdf  # Make anaother DF; in you case you'll may be pass another Data Frame to this function
#     sender = "xxx@yy.com"
#     receiver = ['xxxx@yy.com']
#     msg = MIMEMultipart('related')

#     msg['Subject'] = "Subject Here"
#     msg['From'] = sender
#     msg['To'] = ", ".join(receiver)
#     html = """\
#     <html>
#       <head></head>
#       <body>
#         <p>Hi!<br>
#            Here is first Data Frame data:<br>
#            {0}
#            <br>Here is second Data Frame data:<br>
#            {1}

#            Regards,
#         </p>
#       </body>
#     </html>

#     """.format(cdf.to_html(), df.to_html())

#     partHTML = MIMEText(html, 'html')
#     msg.attach(partHTML)
#     ser = smtplib.SMTP('gateway_server', port_number)
#     ser.login("username", "password")
#     ser.sendmail(sender, receiver, msg.as_string())

# ######################################################################################################

# def send_email():
#             import smtplib

#             gmail_user = ""
#             gmail_pwd = ""
#             FROM = ''
#             TO = ['']
#             SUBJECT = "Testing sending using gmail"
#             TEXT = "Testing sending mail using gmail servers"

#             # Prepare actual message
#             message = """\From: %s\nTo: %s\nSubject: %s\n\n%s
#             """ % (FROM, ", ".join(TO), SUBJECT, TEXT)
#             try:
#                 #server = smtplib.SMTP(SERVER) 
#                 server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
#                 server.ehlo()
#                 server.starttls()
#                 server.login(gmail_user, gmail_pwd)
#                 server.sendmail(FROM, TO, message)
#                 #server.quit()
#                 server.close()
#                 print 'successfully sent the mail'
#             except:
#                 print "failed to send mail"


# def send_email_test():
#             import smtplib
#             from email.mime.text import MIMEText

#             gmail_user = ""
#             gmail_pwd = ""
#             FROM = ''
#             TO = ['']
#             SUBJECT = "Testing sending using gmail"

#             # Prepare actual message
#             message = """\From: %s\nTo: %s\nSubject: %s\n\n
#             """ % (FROM, ", ".join(TO), SUBJECT)
#             try:
#                 fp = open(textfile, 'rb')
#                 txt = MIMEText(fp.read())
#                 fp.close()
#                 message.attach(txt)
#                 # msg.attach(MIMEImage(file("strats.pdf").read()))
#                 #server = smtplib.SMTP(SERVER) 
#                 server = smtplib.SMTP("smtp.gmail.com", 587) #or port 465 doesn't seem to work!
#                 server.ehlo()
#                 server.starttls()
#                 server.login(gmail_user, gmail_pwd)
#                 server.sendmail(FROM, TO, message)
#                 #server.quit()
#                 server.close()
#                 print 'successfully sent the mail'
#             except:
#                 print "failed to send mail"

# ######################################################################################################
# from smtplib import SMTP
# from smtplib import SMTPException
# from email.MIMEMultipart import MIMEMultipart
# from email.MIMEText import MIMEText
# from email.MIMEImage import MIMEImage
# import sys
 
# EMAIL_SUBJECT = "Email from Python script with attachment."
# EMAIL_FROM = ''
# EMAIL_RECEIVER = ''
# GMAIL_SMTP = "smtp.gmail.com"
# GMAIL_SMTP_PORT = 587
# TEXT_SUBTYPE = "plain"
 
# def listToStr(lst):
#     """This method makes comma separated list item string"""
#     return ','.join(lst)
 
# def send_email(content, pswd):
#     """This method sends an email"""
   
#     #Create the email.
#     msg = MIMEMultipart()
#     msg["Subject"] = EMAIL_SUBJECT
#     msg["From"] = EMAIL_FROM
#     msg["To"] = EMAIL_RECEIVER
#     body = MIMEMultipart('alternative')
#     body.attach(MIMEText(content, TEXT_SUBTYPE ))
#     #Attach the message
#     msg.attach(body)
#     #Attach a text file
#     # msg.attach(MIMEText(file("code4reference.txt").read()))
#     #Attach a picture or pdf.
#     msg.attach(MIMEImage(file("strats.pdf").read()))
     
#     try:
#       smtpObj = SMTP(GMAIL_SMTP, GMAIL_SMTP_PORT)
#       #Identify yourself to GMAIL ESMTP server.
#       smtpObj.ehlo()
#       #Put SMTP connection in TLS mode and call ehlo again.
#       smtpObj.starttls()
#       smtpObj.ehlo()
#       #Login to service
#       smtpObj.login(user=EMAIL_FROM, password=pswd)
#       #Send email
#       smtpObj.sendmail(EMAIL_FROM, EMAIL_RECEIVER, msg.as_string())
#       #close connection and session.
#       smtpObj.quit()
#     except SMTPException as error:
#       print "Error: unable to send email :  {err}".format(err=error)
 
# def main(pswd):
#     """This is a simple main() function which demonstrate sending of email using smtplib."""
#     send_email("Test email", pswd);
 
# if __name__ == "__main__":
#     """If this script is run as stand alone then call main() function."""
#     if len(sys.argv) == 2:
#         main(sys.argv[1]);
#     else:
#         print "Please provide the password"

# ######################################################################################################
# import smtplib
# import time
# import imaplib
# import email

# ORG_EMAIL   = "@gmail.com"
# FROM_EMAIL  = "" + ORG_EMAIL
# FROM_PWD    = ""
# SMTP_SERVER = "imap.gmail.com"
# SMTP_PORT   = 993

# def read_email_from_gmail():
#     try:
#         mail = imaplib.IMAP4_SSL(SMTP_SERVER)
#         mail.login(FROM_EMAIL,FROM_PWD)
#         mail.select('inbox')

#         type, data = mail.search(None, 'ALL')
#         mail_ids = data[0]

#         id_list = mail_ids.split()   
#         first_email_id = int(id_list[0])
#         latest_email_id = int(id_list[-1])


#         for i in range(latest_email_id,first_email_id, -1):
#             typ, data = mail.fetch(i, '(RFC822)' )

#             for response_part in data:
#                 if isinstance(response_part, tuple):
#                     msg = email.message_from_string(response_part[1])
#                     email_subject = msg['subject']
#                     email_from = msg['from']
#                     print 'From : ' + email_from + '\n'
#                     print 'Subject : ' + email_subject + '\n'

#     except Exception, e:
#         print str(e)


# read_email_from_gmail()        
        
